import numpy as np
import tensorflow as tf
import logging

from .camera import Camera
from .transform import Transform, GridTransform
from .serialization import to_dict, from_dict
from lib.tf_ops import shape_list, tf_to_dict

OLD_LIGHTING = True

class Light(object):
	def __init__(self, color=[1.,1.,1.], intensity=1., reflection=1.):
		self.c = np.asarray(color)
		self.i = intensity
		self.r = reflection # portion of the not transmitted light that is reflected
	
	@classmethod
	def from_dict(cls, d):
		return cls(**d)
	
	@property
	def monochrome(self):
		return np.isscalar(self.c) or np.asarray(self.c).shape==(1,)
		
	def grid_lighting(self, denisty_data, density_transforms, renderer=None, scattering_func=None):
	#	if scattering_func is not None:
	#		return scattering_func(density=density_transform.data, light_in=self.c * self.i, light=self)
		if OLD_LIGHTING:
			cell_reflection = denisty_data
		else:
			cell_transmission = tf.exp(-denisty_data)
			cell_reflection = (1.0 - cell_transmission) * self.r
		return cell_reflection * self.c * self.i # * setup.rendering.lighting.scattering_ratio
	
	def to_dict(self):
		return {
				"intensity":tf_to_dict(self.i),
				"color":list(self.c),
			}
		
class PointLight(Light):
	def __init__(self, transform, color=[1.,1.,1.], intensity=1., range_scale=1, range_limit=0):
		super().__init__(color, intensity)
		self.t = transform
		self.r = range_limit
		self.s = range_scale
	
	@classmethod
	def from_dict(cls, d):
		t = d.pop("transform")
		t = from_dict(t)
		return cls(t, **d)
		
	
	def grid_lighting_old(self, density_transform, renderer=None, scattering_func=None):
		raise NotImplementedError("This is extremely broken!")
		#M_M_inv = np.linalg.inv(density_transform.get_transform_matrix())
		# position of light source in grid OS, normalized by size -> [0,1]
	#	# !problem: pos is xyz, size is zyx
		position_grid_OS = (density_transform.get_inverse_transform()@self.t.position_global())[:3]/np.asarray(density_transform.grid_size)
		
		#normalized coordinate grid zyx
		# !problem: +0.5 offset of cell centers
		light_data = tf.meshgrid( #linspace: end is inclusive
			tf.linspace(0.,1.,density_transform.grid_size[0]),
			tf.linspace(0.,1.,density_transform.grid_size[1]),
			tf.linspace(0.,1.,density_transform.grid_size[2]),
		indexing='ij')
		#offset by light position
		light_data = tf.transpose(light_data, (1,2,3,0))- position_grid_OS[::-1]# x,y,z -> z,y,x to match grid
		#falloff in normalized grid distance
		# !problem: no scaling with actual grid size in WS
		light_data = 1/(1+tf.pow(tf.norm(light_data, axis=-1, keepdims=True)*self.s, 2.0))
		light_data = light_data * self.c * self.i #tf.repeat(light_data, 3, axis=-1)
		return density_transform.data * tf.cast(light_data, tf.float32)
	
	def grid_lighting(self, denisty_data, density_transforms, renderer=None, scattering_func=None):
		# !problem: does not respect shear?
		lights_data = []
		for density_transform in density_transforms:
			grid_shape = density_transform.grid_shape
			grid_size = grid_shape.xyz.value
			cell_size_WS = density_transform.cell_size_world()
			grid_size_WS = density_transform.grid_size_world()
			light_pos_WS = self.t.position_global()
			# position of light source in grid OS
			light_pos_grid_OS = (density_transform.get_inverse_transform()@light_pos_WS)[:3]
			light_pos_grid_OS_WSscaled = light_pos_grid_OS * cell_size_WS.value
			
			#sclaled coordinate grid xyz, with +0.5 cell-center offset
			light_data = tf.meshgrid( #linspace: end is inclusive
				tf.linspace(tf.constant(0.5 * cell_size_WS.x, dtype=tf.float32), tf.constant(grid_size_WS.x - 0.5*cell_size_WS.x, dtype=tf.float32), grid_shape.x),
				tf.linspace(tf.constant(0.5 * cell_size_WS.y, dtype=tf.float32), tf.constant(grid_size_WS.y - 0.5*cell_size_WS.y, dtype=tf.float32), grid_shape.y),
				tf.linspace(tf.constant(0.5 * cell_size_WS.z, dtype=tf.float32), tf.constant(grid_size_WS.z - 0.5*cell_size_WS.z, dtype=tf.float32), grid_shape.z),
			indexing='ij') #CWHD
			#offset by light position
			light_data = tf.transpose(light_data, (3,2,1,0))- light_pos_grid_OS_WSscaled #DHWC, origin at light position
			lights_data.append(light_data)
		light_data = tf.stack(lights_data, axis=0)
		del lights_data
		#falloff in WS grid distance
		light_data = 1/(1+tf.pow(tf.norm(light_data, axis=-1, keepdims=True)*self.s, 2.0))
		if OLD_LIGHTING:
			light_data = light_data * self.c * self.i #tf.repeat(light_data, 3, axis=-1)
		#return denisty_data * tf.cast(light_data, tf.float32) # * setup.rendering.lighting.scattering_ratio
		# OR
		return light_data * super().grid_lighting(denisty_data, density_transforms, renderer=renderer, scattering_func=scattering_func)
	
	def to_dict(self):
		d = super().to_dict()
		d.update({
			"transform":to_dict(self.t),
			"range_limit":float(self.r),
			"range_scale":float(self.s),
		})
		return d

class SpotLight(PointLight):
	def __init__(self, transform, color=[1.,1.,1.], intensity=1., range_scale=1, range_limit=0, angle_deg=45, cast_shadows=False, shadow_resolution=[64,64,64], shadow_clip=[1,10], cone_mask=True, static=None):
		super().__init__(transform, color, intensity, range_scale, range_limit)
		self.a = angle_deg
		self.cast_shadows = cast_shadows
		self.shadow_resolution = shadow_resolution
		self.shadow_cam = Camera(GridTransform.from_transform(transform, shadow_resolution), shadow_clip, fov=self.a, static=static)
		self.cone_mask = cone_mask
	
	@classmethod
	def from_dict(cls, d):
		t = d.pop("transform")
		t = Transform.from_dict(t)
		return cls(t, **d)
		
	def _get_shadow_mask(self):
		shadow_mask = tf.meshgrid(tf.linspace(tf.constant(-1., dtype=tf.float32),1.,self.shadow_resolution[1]), tf.linspace(tf.constant(-1., dtype=tf.float32),1.,self.shadow_resolution[2]), indexing='ij')
		shadow_mask = tf.transpose(shadow_mask, (1,2,0))
		shadow_mask = tf.norm(shadow_mask, axis=-1, keepdims=True)
		shadow_mask = tf.cast(shadow_mask<=1, tf.float32)
		return shadow_mask
	
	def _render_shadow_map(self, denisty_data, density_transforms, renderer):
		if self.cast_shadows:
			#print(density_transform.data.get_shape(), tf.reduce_mean(density_transform.data))
			shadow_density = renderer.sample_camera(denisty_data, density_transforms, self.shadow_cam, inverse=False, use_step_channel=[0], squeeze_batch=False)#NVDHWC with V=1
			shadow_density = tf.squeeze(shadow_density, 1)
			#print(shadow_density.get_shape(), tf.reduce_mean(shadow_density))
			shadow_density = renderer.blending.reduce_grid_blend(shadow_density, renderer.blend_mode, keep_dims=True)
			#shadow_density = renderer._blend_grid(shadow_density, renderer.blend_mode, keep_dims=True)
			#shift by one cell into depth to avoid cell-self-shadowing (?)
			if renderer.boundary_mode=='BORDER':
				shadow_shape = shape_list(shadow_density)
				shadow_shape[-4] = 1
				pad = tf.zeros(shadow_shape, dtype=tf.float32)
			elif renderer.boundary_mode=='CLAMP':
				pad = shadow_density[...,:1,:,:,:]
			elif renderer.boundary_mode=='WRAP':
				pad = shadow_density[...,-1:,:,:,:]
			else:
				raise ValueError("Unknow boundary_mode %s"%renderer.boundary_mode)
			shadow_density = tf.concat([pad, shadow_density[...,:-1,:,:,:]], axis=-4)
			
			if renderer.blend_mode=='BEER_LAMBERT':
				#shadow_density = tf.math.cumsum(shadow_density, axis=-4, exclusive=True) #+ remove cell shift
				transmission = tf.exp(-shadow_density)
			elif renderer.blend_mode=='ALPHA':
				#shadow_density = tf.math.cumprod(shadow_density, axis=-4, exclusive=True) #? TODO
				transmission = (1-tf.clip_by_value(shadow_density, 0, 1))
			else:
				raise ValueError('Unknown blend_mode \'{}\''.format(renderer.blend_mode))
		else:
			transmission = tf.ones([1] + list(self.shadow_resolution) + [1], dtype=tf.float32)
		if self.cone_mask:
			transmission*=self._get_shadow_mask()
		
		transmission = tf.squeeze(renderer.sample_camera(transmission, density_transforms, self.shadow_cam, inverse=True, use_step_channel=None, squeeze_batch=False), 1)
		#print(transmission.get_shape(), tf.reduce_mean(transmission))
		return transmission
	
	
	def grid_lighting(self, denisty_data, density_transforms, renderer, scattering_func=None):
		light = self._render_shadow_map(denisty_data, density_transforms, renderer)
		light *= super().grid_lighting(denisty_data, density_transforms, renderer=renderer, scattering_func=scattering_func)
		return light
	
	def to_dict(self):
		d = super().to_dict()
		d.update({
			"angle_deg":float(self.a),
			"cast_shadows":bool(self.cast_shadows),
			"shadow_resolution":list(self.shadow_resolution),
			"shadow_clip":list(self.shadow_cam.clip),
			"cone_mask":bool(self.cone_mask),
		})
		return d