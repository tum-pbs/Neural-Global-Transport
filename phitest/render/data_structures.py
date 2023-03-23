import copy, os, numbers
import tensorflow as tf
import numpy as np
from lib.tf_ops import shape_list, spacial_shape_list, has_shape, has_rank, tf_tensor_stats, tf_norm2, tf_angle_between, tf_image_resize_mip, tf_split_to_size
from lib.util import load_numpy, is_None_or_type
from .renderer import Renderer
from .camera import Camera
from .transform import Transform, GridTransform
from .vector import GridShape, Vector3
import logging #, warnings

LOG = logging.getLogger("Structs")



# --- DATA Structs ---

def get_coord_field(shape, offset=[0,0,0], lod=0.0, concat=True):
	'''
	shape: z,y,x
	offset: x,y,z
	returns: 1,z,y,x,c with c=x,z,y,lod
	'''
	coord_z, coord_y, coord_x = tf.meshgrid(tf.range(shape[0], dtype=tf.float32), tf.range(shape[1], dtype=tf.float32), tf.range(shape[2], dtype=tf.float32), indexing='ij') #z,y,x
	coord_data = [tf.reshape(coord_x + offset[0], [1]+shape+[1]),
		tf.reshape(coord_y + offset[1], [1]+shape+[1]),
		tf.reshape(coord_z + offset[2], [1]+shape+[1])] #3 x 1DHW1
	if lod is not None:
		lod_data = tf.constant(lod, shape=[1]+shape+[1], dtype=tf.float32) #tf.ones([1]+shape+[1])*lod
		coord_data.append(lod_data)#4 x 1DHW1
	if concat:
		coord_data = tf.concat(coord_data, axis=-1)

	#coord_data = tf.meshgrid(tf.range(shape[2], dtype=tf.float32), tf.range(shape[1], dtype=tf.float32), tf.range(shape[0], dtype=tf.float32), indexing='ij') #x,y,z
	#coord_data = tf.transpose(coord_data, (3,2,1,0)) #c,x,y,z -> z,y,x,c
	#coord_data = tf.reshape(coord_data, [1]+shape+[3]) + tf.constant(offset, dtype=tf.float32)
	#if lod is not None:
	#	coord_data = tf.concat([coord_data, lod_data], axis=-1)
	
	return coord_data

class ResourceCacheTF:
	def __init__(self, device, value=None):
		self.__device = device
		self.set_value(value)
	def set_value(self, value):
		if not (isinstance(value, (tf.Tensor, np.ndarray)) or value is None):
			raise TypeError("")
		if value is None:
			self.__value = value
		else:
			with tf.device(self.__device):
				self.__value = tf.identity(value)
	def get_value(self):
		return self.__value
	@property
	def value(self):
		return self.get_value()
	@value.setter
	def value(self, value):
		self.set_value(value)
	def clear(self):
		self.set_value(None)
	@property
	def has_value(self):
		return self.__value is not None

class ResourceCacheDictTF:
	def __init__(self, device):
		self.__device = device
		self.__values = {}
	def set_value(self, key, value):
		if not (isinstance(value, (tf.Tensor, np.ndarray)) or value is None):
			raise TypeError("Value must be ndarray, Tensor or None. is %s"%(type(value).__name__,))
		if value is None:
			self.__values[key] = value
		else:
			with tf.device(self.__device):
				self.__values[key] = tf.identity(value)
	def __contains__(self, key):
		return self.has_value(key)
	def __getitem__(self, key):
		return self.get_value(key)
	def __setitem__(self, key, value):
		return self.set_value(key, value)
	def __delitem__(self, key):
		self.remove_value(key)
	def has_value(self, key):
		return key in self.__values
	def __check_key(self, key):
		if not self.has_value(key):
			raise KeyError("Resource Cache does not contain '{}'".format(keys))
	def keys(self):
		return self.__values.keys()
	def get_value(self, key):
		self.__check_key(key)
		return self.__values[key]
	def get_values(self):
		return copy.copy(self.__values)
	def items(self):
		return self.__values.items()
	def remove_value(self, key):
		self.__check_key(key)
		del self.__values[key]
	def clear(self):
		self.__values = {}
	def __len__(self):
		return len(self.__values)
	def is_empty(self):
		return len(self)==0

class ImageSet:
	def __init__(self, base_images, shape=None, device=None, var_name="image_set", trainable=True, resize_method="LINEAR"):
		assert isinstance(trainable, bool)
		assert isinstance(var_name, str)
		self.device = device
		self.var_name = var_name
		assert resize_method in ["LINEAR", "NEAREST"]
		self.resize_method = resize_method
		
		assert shape is None or has_shape(shape, [2])
		assert isinstance(base_images, (tf.Tensor, tf.Variable, np.ndarray)), "ImageSet: images must be tf.Tensor or np.ndarray."
		if isinstance(base_images, (tf.Tensor, tf.Variable)):
			if not tf.reduce_all(tf.is_finite(base_images)).numpy():
				LOG.warning("Base images of '%s' are not finite.", var_name)
		elif isinstance(base_images, np.ndarray):
			if not np.all(np.isfinite(base_images)):
				LOG.warning("Base images of '%s' are not finite.", var_name)
		
		image_shape = shape_list(base_images)
		image_rank = len(image_shape)
		if image_rank==4:
			base_images = GridShape.from_tensor(base_images).normalize_tensor_shape(base_images)
		elif not image_rank==5: raise ValueError("ImageSet: images must be rank 5, NVHWC.")
		with tf.device(self.device):
			self._base_images = tf.identity(base_images)
		self.resize(self.base_shape.yx.value if shape is None else shape)
		
		self._MS_images = {}
	
	def _create_size(self, shape):
		# resize images to requested shape, using base_images as basis for interpolation/filtering
		if not (has_shape(shape, [2])):
			raise ValueError("ImageSet.resize(): image shape must be (height, width), is %s with shape %s"%(shape,shape_list(shape)))
		if np.all(self.base_shape.yx.value==shape):
			images = self.base_images
		else:
			base_shape = self.base_shape
			images = tf.reshape(self.base_images, (base_shape.n*base_shape.z, base_shape.y, base_shape.x, base_shape.c))
			if self.resize_method=="LINEAR":
				images = tf_image_resize_mip(images, shape, mip_bias=0.5, method=tf.image.ResizeMethod.BILINEAR)
			elif self.resize_method=="NEAREST":
				images = tf_image_resize_mip(images, shape, mip_bias=float("-inf"), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
			images = tf.reshape(images, (base_shape.n, base_shape.z, shape[0], shape[1], base_shape.c))
		
		return images
	
	def resize(self, shape):
		# resize images to requested shape, using base_images as basis for interpolation/filtering
		images = self._create_size(shape)
		with tf.device(self.device):
			self._scaled_images = tf.identity(images)
	
	def create_MS_stack(self, scale_shapes):
		# scale_shapes: dict {scale: [shape]]
		self._MS_images = {}
		for scale, shape in scale_shapes.items():
			images = self._create_size(shape)
			with tf.device(self.device):
				self._MS_images[scale] = tf.identity(images)
	
	@property
	def images(self):
		return self._scaled_images
	
	def get_images_of_views(self, view_indices=None):
		if view_indices is None:
			return self.images
		else:
			imgs = self.images
			#LOG.info("selecting views %s from image set '%s' with shape %s", view_indices, self.var_name, shape_list(imgs))
			#return imgs[:,view_indices,...]
			imgs = tf.unstack(imgs, axis=1)
			return tf.stack([imgs[_] for _ in view_indices], axis=1)
	
	#def __array__(self, dtype=None):
	#	if dtype:
	#		return tf.cast(self.images, dtype)
	#	else:
	#		return self.images
	
	@property
	def shape(self):
		return GridShape.from_tensor(self._scaled_images)
	@property
	def base_images(self):
		return self._base_images
	@property
	def base_shape(self):
		return GridShape.from_tensor(self._base_images)
	
	def shape_MS(self, scale):
		return GridShape.from_tensor(self._MS_images[scale])
	
	def images_MS(self, scale):
		return self._MS_images[scale]
	
	def get_images_of_views_MS(self, scale, view_indices=None):
		if view_indices is None:
			return self.images_MS(scale)
		else:
			imgs = self.images_MS(scale)
			#LOG.info("selecting views %s from image set '%s' with shape %s", view_indices, self.var_name, shape_list(imgs))
			#return imgs[:,view_indices,...]
			imgs = tf.unstack(imgs, axis=1)
			return tf.stack([imgs[_] for _ in view_indices], axis=1)
	
	@property
	def num_channels(self):
		return self.base_shape.c
	@property
	def num_views(self):
		return self.base_shape.z
	@property
	def batch_size(self):
		return self.base_shape.n
		
	
	def save(self, renderer, path, format="PNG", idx=0, name=None):
		if name is None:
			name = self.var_name
		shape = self.shape
		renderer.write_images_batch_views(self.base_images, '%s_b{batch:04d}_v{view:02d}_{idx:04d}'%name, base_path=path, frame_idx=idx, input_format="NVHWC", image_format=format)
	
	def save_scaled(self, renderer, path, format="PNG", idx=0, name=None):
		if name is None:
			name = self.var_name
		shape = self.shape
		renderer.write_images_batch_views(self.images, '%s_b{batch:04d}_v{view:02d}_{idx:04d}'%name, base_path=path, frame_idx=idx, input_format="NVHWC", image_format=format)
	
	def save_MS_stack(self, renderer, path, format="PNG", idx=0, name=None):
		if name is None:
			name = self.var_name
		for scale, image in self._MS_images.items():
			renderer.write_images_batch_views(image, '%s_%02d_b{batch:04d}_v{view:02d}_{idx:04d}'%(name, scale), base_path=path, frame_idx=idx, input_format="NVHWC", image_format=format)


class ImageSetMS(ImageSet):
	def __init__(self, base_images, device=None, var_name="image_set_MS", trainable=True, resize_method="LINEAR"):
		assert isinstance(base_images, dict)
		
		self._scales_MS = sorted(base_images.keys())
		self._scale_shapes_MS = {}
		self._base_images_MS = {}
		for s in self._scales_MS:
			scale_image = base_images[s]
			assert isinstance(scale_image, (tf.Tensor, tf.Variable, np.ndarray)), "ImageSet: images must be tf.Tensor or np.ndarray."
			if isinstance(scale_image, (tf.Tensor, tf.Variable)):
				if not tf.reduce_all(tf.is_finite(scale_image)).numpy():
					LOG.warning("Base images [%d] of '%s' are not finite.", s, var_name)
			elif isinstance(scale_image, np.ndarray):
				if not np.all(np.isfinite(scale_image)):
					LOG.warning("Base images [%d] of '%s' are not finite.", s, var_name)
			
			scale_shape = shape_list(scale_image)
			scale_rank = len(scale_shape)
			if scale_rank==4:
				scale_image = GridShape.from_tensor(scale_image).normalize_tensor_shape(scale_image)
			elif not scale_rank==5: raise ValueError("ImageSet: images [%d] must be rank 4 or 5, NVHWC."%(s))
			
			with tf.device(device):
				self._base_images_MS[s] = tf.identity(scale_image)
			self._scale_shapes_MS[s] = GridShape.from_tensor(scale_image)
			
			if s is not self._scales_MS[0]:
				assert self._scale_shapes_MS[s].n == self._scale_shapes_MS[0].n, "Batch size mismatch"
				assert self._scale_shapes_MS[s].z == self._scale_shapes_MS[0].z, "View size mismatch"
				assert self._scale_shapes_MS[s].c == self._scale_shapes_MS[0].c, "Channel size mismatch"
		
		super().__init__(self._base_images_MS[self._scales_MS[-1]], shape=None, device=device, var_name=var_name, trainable=trainable, resize_method=resize_method)
	
	def has_base_MS_scale(self, scale):
		return scale in self._scales_MS
	def _check_base_MS_scale(self, scale):
		if not self.has_base_MS_scale(scale): raise KeyError("Scale {} not in ImageSetMS '{}', available scales: {}".format(scale, self.var_name, self._scales_MS))
	
	def base_shape_MS(self, scale):
		self._check_base_MS_scale(scale)
		return self._scale_shapes_MS[scale]
	def base_images_MS(self, scale):
		self._check_base_MS_scale(scale)
		return self._base_images_MS[scale]
	def get_base_images_of_views_MS(self, scale, view_indices=None):
		self._check_base_MS_scale(scale)
		if view_indices is None:
			return self.base_images_MS(scale)
		else:
			imgs = self.base_images_MS(scale)
			imgs = tf.unstack(imgs, axis=1)
			return tf.stack([imgs[_] for _ in view_indices], axis=1)
	def save_base_MS_stack(self, renderer, path, format="PNG", idx=0, name=None):
		if name is None:
			name = self.var_name
		for scale, image in self._base_images_MS.items():
			renderer.write_images_batch_views(image, '%s_MS%02d_b{batch:04d}_v{view:02d}_{idx:04d}'%(name, scale), base_path=path, frame_idx=idx, input_format="NVHWC", image_format=format)

class Zeroset:
	def __init__(self, initial_value, shape=None, as_var=True, outer_bounds="OPEN", device=None, var_name="zeroset", trainable=True):
		self.outer_bounds = outer_bounds
		self.is_var = as_var
		self._device = device
		self._name = var_name
		self._is_trainable = trainable
		
		with tf.device(self._device):
			if shape is not None:
				assert isinstance(shape, GridShape)
				initial_value = tf.constant(initial_value, shape=shape.value, dtype=tf.float32)
			if as_var:
				self._levelset = tf.Variable(initial_value=initial_value, name=var_name, trainable=trainable)
			else:
				self._levelset = tf.identity(initial_value)
	
	@property
	def grid_shape(self):
		return GridShape.from_tensor(self._levelset)
	
	def _hull_staggered_lerp_weight(self, a, b):
		a_leq = tf.less_equal(a,0)
		return tf.where( tf.logical_xor(a_leq, tf.less_equal(b,0)), #sign change along iterpolation
				tf.abs( tf.divide( tf.minimum(a,b), tf.subtract(a,b) ) ),
				tf.cast(a_leq, dtype=a.dtype)
			)
	
	def _hull_simple_staggered_component(self, axis):
		assert axis in [1,2,3,-2,-3,-4]
		axis = axis%5
		pad = [(0,0),(0,0),(0,0),(0,0),(0,0)]
		pad[axis]=(1,1)
		shape = self.grid_shape.value
		shape[axis] -= 1
		offset = np.zeros((5,), dtype=np.int32)
		cells_prev = tf.slice(self._levelset, offset, shape) #self._levelset[:,:,:,:-1,:]
		offset[axis] += 1
		cells_next = tf.slice(self._levelset, offset, shape) #self._levelset[:,:,:, 1:,:]
		hull = self._hull_staggered_lerp_weight(cells_prev,cells_next)
		hull = tf.pad(hull, pad, constant_values=1 if self.outer_bounds=="OPEN" else 0)
		return hull
	
	def to_hull_simple_staggered(self):
		return self._hull_simple_staggered_component(-2), self._hull_simple_staggered_component(-3), self._hull_simple_staggered_component(-4)
	
	def to_hull_simple_centered(self):
		raise NotImplementedError()
	
	def to_denstiy_simple_centered(self):
		return tf.where(tf.greater(self._levelset, 0), 250, 0)
	
	def resize(self, shape):
		assert shape_list(shape)==[3]
		new_shape = GridShape(shape)
		if new_shape==self.grid_shape:
			return
		raise NotImplementedError("Zeroset.resize() not implemented.")
	
	def assign(levelset):
		raise NotImplementedError()




class DensityGrid():
	def __init__(self, shape, constant=0.1, as_var=True, d=None, scale_renderer=None, hull=None, inflow=None, inflow_offset=None, inflow_mask=None, device=None, var_name="denstiy", trainable=True, restrict_to_hull=True, is_SDF=False):
		
		#super().__init__(inputs=[], models=[], device=device)
		
		assert isinstance(as_var, bool), "DensityGrid: as_var must be bool, is: %s"%(type(as_var).__name__,)
		assert isinstance(trainable, bool), "DensityGrid: trainable must be bool, is: %s"%(type(trainable).__name__,)
		#if restrict_to_hull is None: restrict_to_hull=False
		assert isinstance(restrict_to_hull, bool), "DensityGrid: restrict_to_hull must be bool, is: %s"%(type(restrict_to_hull).__name__,)
		assert isinstance(var_name, str), "DensityGrid: var_name must be str, is: %s"%(type(var_name).__name__,)
		assert d is None or isinstance(d, (tf.Tensor, tf.Variable, np.ndarray))
		assert scale_renderer is None or isinstance(scale_renderer, Renderer)
		
		self.shape = list(shape)
		if d is not None:
			d_shape = shape_list(d)
			if not len(d_shape)==5 or not d_shape[-1]==1 or not self.shape==spacial_shape_list(d):
				raise ValueError("Invalid shape of density on assignment: %s"%d_shape)
		self.is_var = as_var
		self._device = device
		self._name = var_name
		self._is_trainable = trainable
		self._is_SDF = False #is_SDF
		if as_var:
			rand_init = tf.constant_initializer(constant)
			with tf.device(self._device):
				self._d = tf.Variable(initial_value=d if d is not None else rand_init(shape=[1]+self.shape+[1], dtype=tf.float32), name=var_name+'_dens', trainable=True)
			
		else:
			with tf.device(self._device):
				if d is not None:
					self._d = tf.identity(d)
				else:
					self._d = tf.constant(constant, shape=[1]+self.shape+[1], dtype=tf.float32)
			
		self.scale_renderer = scale_renderer
		with tf.device(self._device):
			self.hull = tf.constant(hull, dtype=tf.float32) if hull is not None else None
		self.restrict_to_hull = restrict_to_hull
		
		if inflow is not None:
			with tf.device(self._device):
				if isinstance(inflow, str) and inflow=='CONST':
					assert isinstance(inflow_mask, (tf.Tensor, tf.Variable, np.ndarray))
					inflow = rand_init(shape=shape_list(inflow_mask), dtype=tf.float32)
				if as_var:
					self._inflow = tf.Variable(initial_value=inflow, name=var_name+'_inflow', trainable=True)
				else:
					self._inflow = tf.constant(inflow, dtype=tf.float32)
				self.inflow_mask = tf.constant(inflow_mask, dtype=tf.float32) if inflow_mask is not None else None
			inflow_shape = spacial_shape_list(self._inflow) #.get_shape().as_list()[-4:-1]
			self._inflow_padding = [[0,0]]+[[inflow_offset[_],self.shape[_]-inflow_offset[_]-inflow_shape[_]] for _ in range(3)]+[[0,0]]
			self.inflow_offset = inflow_offset
		else:
			self._inflow = None
	
	
	@property
	def trainable(self):
		return self._is_trainable and self.is_var
	@property
	def is_SDF(self):
		return False #self._is_SDF
	
	@property
	def d(self):
		if self.restrict_to_hull:
			return self.with_hull()
		else:
			return tf.identity(self._d)
	
	def with_hull(self):
		if self.hull is not None:
			return self._d * self.hull # hull is a (smooth) binary mask
		else:
			return tf.identity(self._d)
	
	@property
	def inflow(self):
		if self._inflow is None:
			return tf.zeros_like(self._d, dtype=tf.float32)
		elif self.inflow_mask is not None: #hasattr(self, 'inflow_mask') and 
			return tf.pad(self._inflow*self.inflow_mask, self._inflow_padding)
		else:
			return tf.pad(self._inflow, self._inflow_padding)
	
	def with_inflow(self):
		density = self.d
		if self._inflow is not None:
			density = tf.maximum(density+self.inflow, 0)
		return density
	
	@classmethod
	def from_file(cls, path, as_var=True, scale_renderer=None, hull=None, inflow=None, inflow_offset=None, inflow_mask=None, device=None, var_name="denstiy", trainable=True, restrict_to_hull=True, is_SDF=False):
		try:
			with np.load(path) as np_data:
				d = np_data['arr_0']
				shape =spacial_shape_list(d)
				if 'hull' in np_data and hull is None:
					hull = np_data['hull']
				if 'inflow' in np_data and inflow is None:
					inflow=np_data['inflow']
					if 'inflow_mask' in np_data and inflow_mask is None:
						inflow_mask=np_data['inflow_mask']
					if 'inflow_offset' in np_data and inflow_offset is None:
						inflow_offset=np_data['inflow_offset'].tolist()
					#grid = cls(shape, d=d, as_var=as_var, scale_renderer=scale_renderer, hull=hull, inflow=np_data['inflow'], inflow_offset=np_data['inflow_offset'].tolist(), device=device)
				grid = cls(shape, d=d, as_var=as_var, scale_renderer=scale_renderer, hull=hull, inflow=inflow, inflow_offset=inflow_offset, inflow_mask=inflow_mask, \
					device=device, var_name=var_name, trainable=trainable, restrict_to_hull=restrict_to_hull, is_SDF=is_SDF)
		except:
			LOG.warning("Failed to load density from '%s':", path, exc_info=True)
			return None
		else:
			return grid
		
	@classmethod
	def from_scalarFlow_file(cls, path, as_var=True, shape=None, scale_renderer=None, hull=None, inflow=None, inflow_offset=None, inflow_mask=None, device=None, var_name="sF_denstiy", trainable=True, restrict_to_hull=True, is_SDF=is_SDF):
		# if shape is set the loaded grid will be reshaped if necessary
	#	with np.load(path) as np_data:
	#		density = np_data['data'].astype(np.float32)[::-1] # DHWC with C=1 and D/z reversed
		density = load_numpy(path).astype(np.float32)[::-1]
		density = density.reshape([1] + list(density.shape)) # 
		density = tf.constant(density, dtype=tf.float32)
		d_shape = spacial_shape_list(density)
		if shape is not None and shape!=d_shape:
			if scale_renderer is None:
				raise ValueError("No renderer provided to scale density.")
			LOG.debug("scaling scalarFlow density from %s to %s", d_shape, shape)
			density = scale_renderer.resample_grid3D_aligned(density, shape)
			d_shape = shape
		else:
			# cut of SF inflow region and set as inflow. or is it already cut off in SF dataset? it is, but not in the synth dataset or my own sF runs.
			# lower 15 cells...
			inflow, density= tf.split(density, [15, d_shape[1]-15], axis=-3)
			inflow_mask = tf.ones_like(inflow, dtype=tf.float32)
			inflow_offset = [0,0,0]
			density = tf.concat([tf.zeros_like(inflow, dtype=tf.float32), density], axis=-3)
		return cls(d_shape, d=density, as_var=as_var, scale_renderer=scale_renderer, hull=hull, inflow=inflow, inflow_offset=inflow_offset, inflow_mask=inflow_mask, \
			device=device, var_name=var_name, trainable=trainable, restrict_to_hull=restrict_to_hull)
	
	def copy(self, as_var=None, device=None, var_name=None, trainable=None, restrict_to_hull=None):
		if as_var is None:
			as_var = self.is_var
		if var_name is None:
			var_name = self._name + '_cpy'
		if trainable is None:
			trainable = self._is_trainable
		if restrict_to_hull is None:
			restrict_to_hull = self.restrict_to_hull
		if self._inflow is not None:
			grid = DensityGrid(self.shape, d=tf.identity(self._d), as_var=as_var, scale_renderer=self.scale_renderer, hull=self.hull, \
				inflow=tf.identity(self._inflow), inflow_offset=self.inflow_offset, inflow_mask=self.inflow_mask, \
				device=device, var_name=var_name, trainable=trainable, restrict_to_hull=restrict_to_hull, is_SDF=self._is_SDF)
		else:
			grid = DensityGrid(self.shape, d=tf.identity(self._d), as_var=as_var, scale_renderer=self.scale_renderer, hull=self.hull, \
				device=device, var_name=var_name, trainable=trainable, restrict_to_hull=restrict_to_hull, is_SDF=self._is_SDF)
		return grid
	
	def copy_empty(self, as_var=None, device=None, var_name=None, trainable=None, restrict_to_hull=None):
		if as_var is None:
			as_var = self.is_var
		if var_name is None:
			var_name = self._name + '_cpy'
		if trainable is None:
			trainable = self._is_trainable
		if restrict_to_hull is None:
			restrict_to_hull = self.restrict_to_hull
		if self._inflow is not None:
			grid = DensityGrid(self.shape, constant=0.0, as_var=as_var, scale_renderer=self.scale_renderer, hull=self.hull, \
				inflow=tf.identity(self._inflow), inflow_offset=self.inflow_offset, inflow_mask=self.inflow_mask, \
				device=device, var_name=var_name, trainable=trainable, restrict_to_hull=restrict_to_hull, is_SDF=self._is_SDF)
		else:
			grid = DensityGrid(self.shape, constant=0.0, as_var=as_var, scale_renderer=self.scale_renderer, hull=self.hull, \
				device=device, var_name=var_name, trainable=trainable, restrict_to_hull=restrict_to_hull, is_SDF=self._is_SDF)
		return grid
	
	def scaled(self, new_shape, with_inflow=False):
		with self.scale_renderer.profiler.sample("get density scaled"):
			if not (isinstance(new_shape, list) and len(new_shape)==3):
				raise ValueError("Invalid shape")
			density = self.d if not with_inflow else self.with_inflow()
			if new_shape!=self.shape:
				LOG.debug("Scaling density from %s to %s", self.shape, new_shape)
				with self.scale_renderer.profiler.sample("scale density"):
					d_scaled = self.scale_renderer.resample_grid3D_aligned(density, new_shape)
					if self._is_SDF:
						with self.scale_renderer.profiler.sample("scale SDF"):
							# scale values to be valid OS distances
							scale_factor = np.mean([o/i for o,i in zip(new_shape, self.shape)])
							d_scaled = d_scaled * scale_factor
			else:
				LOG.debug("No need to scale density to same shape %s", self.shape)
				d_scaled = density #tf.identity(density) # self.d already copies
		return d_scaled
	
	def rescale(self, new_shape, base_scale_fn):
		'''re-scale/re-shape in place'''
		if not isinstance(new_shape, (list,tuple)) or not len(new_shape)==3:
				raise ValueError("Invalid shape for density re-scaling: %s"%new_shape)
		d = self.scaled(new_shape)
		hull = self.scale_renderer.resample_grid3D_aligned(self.base_hull, new_shape) if self.hull is not None else None
		if self._inflow is not None:
			if_off = base_scale_fn(self.base_inflow_offset)
			if_shape = base_scale_fn(self.base_inflow_shape)
			if_scaled = self.scale_renderer.resample_grid3D_aligned(self._inflow, if_shape)
			if_mask = None if self.inflow_mask is None else self.scale_renderer.resample_grid3D_aligned(self.base_inflow_mask, if_shape)
			LOG.info("Frame %04d: inflow to %s, offset to %s", state.frame, if_shape, if_off)
		
		self.shape = list(new_shape)
		with tf.device(self._device):
			if self.is_var:
				self._d = tf.Variable(initial_value=d, name=var_name+'_rs', trainable=True)
			else:
				self._d = tf.identity(d)
			
			if hull is not None:
				self.hull = tf.identity(hull)
			
			if self._inflow is not None:
				if as_var:
					self._inflow = tf.Variable(initial_value=if_scaled, name=var_name+'_inflow', trainable=True)
				else:
					self._inflow = tf.identity(if_scaled)
				self.inflow_mask = tf.identity(if_mask) if if_mask is not None else None
				inflow_shape = spacial_shape_list(self._inflow) #.get_shape().as_list()[-4:-1]
				self._inflow_padding = [[0,0]]+[[if_off[_],self.shape[_]-if_off[_]-inflow_shape[_]] for _ in range(3)]+[[0,0]]
				self.inflow_offset = if_off
	
	def copy_scaled(self, new_shape, as_var=None, device=None, var_name=None, trainable=None, restrict_to_hull=None):
		'''Does not copy inflow and hull, TODO'''
		if as_var is None:
			as_var = self.is_var
		if as_var and var_name is None:
			var_name = self._name + '_scaled'
		if trainable is None:
			trainable = self._is_trainable
		if restrict_to_hull is None:
			restrict_to_hull = self.restrict_to_hull
		d_scaled = self.scaled(new_shape)
		grid = DensityGrid(new_shape, d=d_scaled, as_var=as_var, scale_renderer=self.scale_renderer, device=device, var_name=var_name, trainable=trainable, restrict_to_hull=restrict_to_hull, is_SDF=self._is_SDF)
		return grid
	
	def warped(self, vel_grid, order=1, dt=1.0, clamp="NONE"):
		if not (isinstance(vel_grid, VelocityGrid)):
			raise ValueError("Invalid velocity grid")
		return vel_grid.warp(self.with_inflow(), order=order, dt=dt, clamp=clamp)
	
	def copy_warped(self, vel_grid, as_var=None, order=1, dt=1.0, device=None, var_name=None, clamp="NONE", trainable=None, restrict_to_hull=None):
		'''Does not copy inflow and hull, TODO'''
		if as_var is None:
			as_var = self.is_var
		if var_name is None:
			var_name = self._name + '_warped'
		if trainable is None:
			trainable = self._is_trainable
		if restrict_to_hull is None:
			restrict_to_hull = self.restrict_to_hull
		d_warped = self.warped(vel_grid, order=order, dt=dt, clamp=clamp)
		grid = DensityGrid(self.shape, d=d_warped, as_var=as_var, scale_renderer=self.scale_renderer, device=device, var_name=var_name, trainable=trainable, restrict_to_hull=restrict_to_hull, is_SDF=self._is_SDF)
		return grid
	
	def scale(self, scale):
		self.assign(self._d*scale)
	
	def apply_clamp(self, vmin, vmax):
		if not self._is_SDF:
			vmin = tf.maximum(vmin, 0)
		elif vmin >= 0.0:
			LOG.warning("apply_clamp called on SDF DensityGrid with min >= 0.")
		d = tf.clip_by_value(self._d, vmin, vmax)
		inflow = None
		if self._inflow is not None:
		#	inflow_shape = self._inflow.get_shape().as_list()
		#	density = self._d[:,self.inflow_offset[0]:self.inflow_offset[0]+inflow_shape[-4], \
		#		self.inflow_offset[1]:self.inflow_offset[1]+inflow_shape[-3], \
		#		self.inflow_offset[2]:self.inflow_offset[2]+inflow_shape[-2],:]
			# use already clamped density for consistency
			denstiy_shape = shape_list(d)
			density_cropped = d[self._inflow_padding[0][0] : denstiy_shape[0]-self._inflow_padding[0][1],
				self._inflow_padding[1][0] :  denstiy_shape[1]-self._inflow_padding[1][1],
				self._inflow_padding[2][0] :  denstiy_shape[2]-self._inflow_padding[2][1],
				self._inflow_padding[3][0] :  denstiy_shape[3]-self._inflow_padding[3][1],
				self._inflow_padding[4][0] :  denstiy_shape[4]-self._inflow_padding[4][1]]
			inflow = tf.clip_by_value(self._inflow, vmin - density_cropped, vmax - density_cropped)
		self.assign(d, inflow)
	
	def assign(self, d, inflow=None):
		shape = shape_list(d)
		if not len(shape)==5 or not shape[-1]==1 or not shape[-4:-1]==self.shape:
			raise ValueError("Invalid or incompatible shape of density on assignment: is {}, required: NDHW1 with DHW={}".format(shape, self.shape))
		if self.is_var:
			self._d.assign(d)
			if self._inflow is not None and inflow is not None:
				self._inflow.assign(inflow)
		else:
			with tf.device(self._device):
				self._d = tf.identity(d)
				if self._inflow is not None and inflow is not None:
					self._inflow = tf.identity(inflow)
	
	def assign_scaled(self, d):
		shape = shape_list(d)
		if not len(shape)==5 or not shape[-1]==1:
			raise ValueError("Invalid shape of density on assignment: is {}, required: NDHW1".format(shape))
		d = self.scale_renderer.resample_grid3D_aligned(d, self.shape)
		self.assign(d)
	
	def var_list(self):
		if self.is_var:
			if self._inflow is not None:
				return [self._d, self._inflow]
			return [self._d]
		else:
			raise TypeError("This DensityGrid is not a variable.")
	
	def get_variables(self):
		if self.is_var:
			var_dict = {'density': self._d}
			if self._inflow is not None:
				var_dict['inflow'] = self._inflow
			return var_dict
		else:
			#raise TypeError("This DensityGrid is not a variable.")
			return dict()
	
	def get_output_variables(self, include_MS=False, include_residual=False, only_trainable=False):
		var_dict = {'density': self._d}
		if self._inflow is not None:
			var_dict['inflow'] = self._inflow
		return var_dict
		
	
	def save(self, path):
		density = self._d
		if isinstance(density, (tf.Tensor, tf.Variable)):
			density = density.numpy()
		save = {}
		if self.hull is not None:
			hull = self.hull
			if isinstance(hull, (tf.Tensor, tf.Variable)):
				hull = hull.numpy()
			save['hull']=hull
		if self._inflow is not None:
			inflow = self._inflow
			if isinstance(inflow, (tf.Tensor, tf.Variable)):
				inflow = inflow.numpy()
			save['inflow']=inflow
			if self.inflow_mask is not None:
				inflow_mask = self.inflow_mask
				if isinstance(inflow_mask, (tf.Tensor, tf.Variable)):
					inflow_mask = inflow_mask.numpy()
				save['inflow_mask']=inflow_mask
			save['inflow_offset']=np.asarray(self.inflow_offset)
			#np.savez_compressed(path, density, inflow=inflow, inflow_offset=np.asarray(self.inflow_offset))
		np.savez_compressed(path, density, **save)
	
	def mean(self):
		return tf.reduce_mean(self.d)
	
	def stats(self, mask=None, state=None, **warp_kwargs):
		'''
			mask: optional binary float mask, stats only consider cells>0.5
		'''
		d = self.d
		if mask is not None:
			mask =  mask if mask.dtype==tf.bool else tf.greater(mask, 0.5)
			d = tf.boolean_mask(d, mask)
		
		stats = {
			#'dMean':tf.reduce_mean(d), 'dMax':tf.reduce_max(d), 'dMin':tf.reduce_min(d),'dAbsMean':tf.reduce_mean(tf.abs(d)), 
			'density': tf_tensor_stats(d, as_dict=True),
			'shape':self.shape,
		}
		if state is not None and state.prev is not None and state.prev.density is not None and state.prev.velocity is not None:
			warp_SE = tf.squared_difference(state.prev.density_advected(**warp_kwargs), self.d)
			if mask is not None:
				warp_SE = tf.boolean_mask(warp_SE, mask)
			stats["warp_SE"] = tf_tensor_stats(warp_SE, as_dict=True)
		else:
			stats["warp_SE"] = tf_tensor_stats(tf.zeros([1,1,1,1,1], dtype=tf.float32), as_dict=True)
		return stats
	
	def clear_cache(self):
		pass #super().clear_cache()
	@property
	def is_MS(self):
		return False
	@property
	def has_MS_output(self):
		return self.is_MS and False

class VelocityGrid:
	@staticmethod
	def component_shapes(centered_shape):
		assert len(centered_shape)==3
		x_shape = copy.copy(centered_shape)
		x_shape[2] +=1
		y_shape = copy.copy(centered_shape)
		y_shape[1] +=1
		z_shape = copy.copy(centered_shape)
		z_shape[0] +=1
		return x_shape, y_shape, z_shape
	@staticmethod
	def component_potential_shapes(centered_shape):
		assert len(centered_shape)==3
		x_shape = copy.copy(centered_shape)
		x_shape[0] +=1
		x_shape[1] +=1
		y_shape = copy.copy(centered_shape)
		y_shape[0] +=1
		y_shape[2] +=1
		z_shape = copy.copy(centered_shape)
		z_shape[1] +=1
		z_shape[2] +=1
		return x_shape, y_shape, z_shape
		
	def __init__(self, centered_shape, std=0.1, as_var=True, x=None, y=None, z=None, boundary=None, scale_renderer=None, warp_renderer=None, *, coords=None, lod=None, device=None, var_name="velocity", trainable=True):
		self.centered_shape = centered_shape.tolist() if isinstance(centered_shape, np.ndarray) else centered_shape
		self.x_shape, self.y_shape, self.z_shape = VelocityGrid.component_shapes(self.centered_shape)
		self.set_boundary(boundary)
		self.is_var = as_var
		self._device = device
		self._name = var_name
		self._is_trainable = trainable
		if as_var:
			if x is not None:
				x_shape = shape_list(x)
				if not len(x_shape)==5 or not x_shape[-1]==1 or not x_shape[-4:-1]==self.x_shape:
					raise ValueError("Invalid shape of velocity x component on assignment")
			if y is not None:
				y_shape = shape_list(y)
				if not len(y_shape)==5 or not y_shape[-1]==1 or not y_shape[-4:-1]==self.y_shape:
					raise ValueError("Invalid shape of velocity y component on assignment")
			if z is not None:
				z_shape = shape_list(z)
				if not len(z_shape)==5 or not z_shape[-1]==1 or not z_shape[-4:-1]==self.z_shape:
					raise ValueError("Invalid shape of velocity z component on assignment")
			# in a box
			#rand_init = tf.random_normal_initializer(0.0, std)
			std = tf.abs(std)
			rand_init = tf.random_uniform_initializer(-std, std)
			# maybe even uniformly in space and in a sphere?: http://6degreesoffreedom.co/circle-random-sampling/
			with tf.device(self._device):
				self._x = tf.Variable(initial_value=x if x is not None else rand_init(shape=[1]+self.x_shape+[1], dtype=tf.float32), name=var_name + '_x', trainable=True)
				self._y = tf.Variable(initial_value=y if y is not None else rand_init(shape=[1]+self.y_shape+[1], dtype=tf.float32), name=var_name + '_y', trainable=True)
				self._z = tf.Variable(initial_value=z if z is not None else rand_init(shape=[1]+self.z_shape+[1], dtype=tf.float32), name=var_name + '_z', trainable=True)
		else:
			if x is None:
				x = tf.constant(tf.random.uniform([1]+self.x_shape+[1], -std, std, dtype=tf.float32))
			if y is None:
				y = tf.constant(tf.random.uniform([1]+self.y_shape+[1], -std, std, dtype=tf.float32))
			if z is None:
				z = tf.constant(tf.random.uniform([1]+self.z_shape+[1], -std, std, dtype=tf.float32))
			self.assign(x,y,z)
		
	#	if coords is None:
	#		self.coords = get_coord_field(self.centered_shape, lod=None)
	#	else:
	#		self.coords = coords
		if lod is None:
			lod = tf.zeros([1]+self.centered_shape+[1])
		with tf.device(self._device):
			self.lod_pad = tf.identity(lod)
		
		self.scale_renderer = scale_renderer#Renderer(profiler, filter_mode='LINEAR', mipmapping='NONE', sample_gradients=False)
		if self.scale_renderer is not None:
			if (self.outer_bounds=='CLOSED' and self.scale_renderer.boundary_mode!='BORDER') \
				or (self.outer_bounds=='OPEN' and self.scale_renderer.boundary_mode!='CLAMP'):
				LOG.warning("Velocity outer boundary %s does not match scale renderer boundary mode %s", self.outer_bounds, self.scale_renderer.boundary_mode)
		self.warp_renderer = warp_renderer
		if self.warp_renderer is not None:
			if (self.outer_bounds=='CLOSED' and self.warp_renderer.boundary_mode!='BORDER') \
				or (self.outer_bounds=='OPEN' and self.warp_renderer.boundary_mode!='CLAMP'):
				LOG.warning("Velocity outer boundary %s does not match scale renderer boundary mode %s", self.outer_bounds, self.warp_renderer.boundary_mode)
	
	def set_boundary(self, boundary):
		assert (boundary is None) or isinstance(boundary, Zeroset)
		self.boundary = boundary
		self.outer_bounds = self.boundary.outer_bounds if self.boundary is not None else "OPEN"
	
	@staticmethod
	def shape_centered_to_staggered(shape):
		return [_+1 for _ in shape]
	@property
	def staggered_shape(self):
		return self.shape_centered_to_staggered(self.centered_shape)
	
	@property
	def trainable(self):
		return self._is_trainable and self.is_var
	
	
	def _staggered(self):
		return (self._x, self._y, self._z)
	
	@property
	def x(self):
		v = self._x
		if self.boundary is not None:
			v*= self.boundary._hull_simple_staggered_component(-2)
		return v
	@property
	def y(self):
		v = self._y
		if self.boundary is not None:
			v*= self.boundary._hull_simple_staggered_component(-3)
		return v
	@property
	def z(self):
		v = self._z
		if self.boundary is not None:
			v*= self.boundary._hull_simple_staggered_component(-4)
		return v
	
	@classmethod
	def from_centered(cls, centered_grid, as_var=True, boundary=None, scale_renderer=None, warp_renderer=None, device=None, var_name="velocity", trainable=True):
		centered_shape = shape_list(centered_grid)
		assert len(centered_shape)==5, "input grid must be NDWHC, is %s"%(centered_shape,)
		assert centered_shape[-1]==3
		#assert centered_shape[0]==1
		centered_shape = centered_shape[-4:-1]
		vel_grid = cls(centered_shape, as_var=as_var, boundary=boundary, scale_renderer=scale_renderer, warp_renderer=warp_renderer, device=device, var_name=var_name, trainable=trainable)
		x,y,z = vel_grid._centered_to_staggered(centered_grid)
		vel_grid.assign(x,y,z)
		return vel_grid
	@classmethod
	def from_staggered_combined(cls, staggered_grid, as_var=True, boundary=None, scale_renderer=None, warp_renderer=None, device=None, var_name="velocity", trainable=True):
		centered_shape = shape_list(staggered_grid)
		assert len(centered_shape)==5, "input grid must be NDWHC, is %s"%(centered_shape,)
		assert centered_shape[-1]==3
		#assert centered_shape[0]==1
		centered_shape = [_-1 for _ in centered_shape[-4:-1]]
		vel_grid = cls(centered_shape, as_var=as_var, boundary=boundary, scale_renderer=scale_renderer, warp_renderer=warp_renderer, device=device, var_name=var_name, trainable=trainable)
		vel_grid.assign_staggered_combined(staggered_grid)
		return vel_grid
		
	@classmethod
	def from_file(cls, path, as_var=True, boundary=None, scale_renderer=None, warp_renderer=None, device=None, var_name="velocity", trainable=True):
		try:
			with np.load(path) as vel:
				if 'centered_shape' not in vel:#legacy
					shape = shape_list(vel["vel_x"])
					LOG.debug("%s", shape)
					shape[-2] -=1
					shape = shape[1:-1]
				else:
					shape = vel['centered_shape'].tolist()
				vel_grid = cls(shape, x=vel["vel_x"].astype(np.float32), y=vel["vel_y"].astype(np.float32), z=vel["vel_z"].astype(np.float32), \
					as_var=as_var, boundary=boundary, scale_renderer=scale_renderer, warp_renderer=warp_renderer, device=device, var_name=var_name, trainable=trainable)
		except:
			LOG.warning("Failed to load velocity from '%s':", path, exc_info=True)
			return None
		else:
			return vel_grid
	
	@classmethod
	def from_scalarFlow_file(cls, path, as_var=True, shape=None, boundary=None, scale_renderer=None, warp_renderer=None, device=None, var_name="sF_velocity", trainable=True):
		# sF velocities are stored as combined staggered grid with upper cells missing, DHWC with C=3
	#	with np.load(path) as np_data:
	#		velocity = np_data['data'].astype(np.float32)[::-1] # DHWC with C=3 and D/z reversed
		velocity = load_numpy(path).astype(np.float32)[::-1]
		v_shape = GridShape.from_tensor(velocity)
		velocity = v_shape.normalize_tensor_shape(velocity) #.reshape([1] + list(velocity.shape)) # NDHWC
		velocity = tf.constant(velocity, dtype=tf.float32)
		v_shape = v_shape.zyx.value
		v_x, v_y, v_z = tf.split(velocity, 3, axis=-1)
		p0 = (0,0)
		# extend missing upper cell
		v_x = tf.pad(v_x, [p0,p0,p0,(0,1),p0], "SYMMETRIC")
		v_y = tf.pad(v_y, [p0,p0,(0,1),p0,p0], "SYMMETRIC")
		v_z = tf.pad(-v_z, [p0,(1,0),p0,p0,p0], "SYMMETRIC") #z value/direction reversed, pad lower value as axis is reversed (?)
		#v_shape = spacial_shape_list(velocity)
		if shape is not None and v_shape!=shape:
			assert len(shape)==3
			if scale_renderer is None:
				raise ValueError("No renderer provided to scale velocity.")
		#	shape = GridShape(shape).zyx
		#	vel_scale = shape/v_shape #[o/i for i,o in zip(v_shape, shape)] #z,y,x
			LOG.debug("scaling scalarFlow velocity from %s to %s with magnitude scale %s", v_shape, shape)
			v_tmp = cls(v_shape, x=v_x, y=v_y, z=v_z, as_var=False, boundary=boundary, scale_renderer=scale_renderer, warp_renderer=warp_renderer, device=device, var_name="sF_tmp", trainable=False)
			v_x, v_y, v_z = v_tmp.scaled(shape, scale_magnitude=True)
			# can only scale 1 and 4 channel grids
		#	v_x = scale_renderer.resample_grid3D_aligned(v_x, shape.value)*vel_scale.x#[2]
		#	v_y = scale_renderer.resample_grid3D_aligned(v_y, shape.value)*vel_scale.y#[1]
		#	v_z = scale_renderer.resample_grid3D_aligned(v_z, shape.value)*vel_scale.z#[0]
		#	velocity = tf.concat([v_x, v_y, v_z], axis=-1)
			v_shape = shape
		
		#return cls.from_centered(velocity,as_var=as_var, boundary=boundary, scale_renderer=scale_renderer, warp_renderer=warp_renderer, device=device, var_name=var_name)
		return cls(v_shape, x=v_x, y=v_y, z=v_z,as_var=as_var, boundary=boundary, scale_renderer=scale_renderer, warp_renderer=warp_renderer, device=device, var_name=var_name, trainable=trainable)
	
	def copy(self, as_var=None, device=None, var_name=None, trainable=None):
		if as_var is None:
			as_var = self.is_var
		if as_var and var_name is None:
			var_name = self._name + '_cpy'
		if trainable is None:
			trainable = self._is_trainable
		grid = VelocityGrid(self.centered_shape, x=tf.identity(self._x), y=tf.identity(self._y), z=tf.identity(self._z), as_var=as_var, \
			boundary=self.boundary, scale_renderer=self.scale_renderer, warp_renderer=self.warp_renderer, device=device, var_name=var_name, trainable=trainable)
		return grid
		
	
	def scaled(self, centered_shape, scale_magnitude=True):
		if not (isinstance(centered_shape, list) and len(centered_shape)==3):
			raise ValueError("Invalid shape")
		#resample velocity
		if centered_shape!=self.centered_shape:
			with self.scale_renderer.profiler.sample("scale velocity"):
				x_shape, y_shape, z_shape = VelocityGrid.component_shapes(centered_shape)
				LOG.debug("Scaling velocity from %s to %s", self.centered_shape, centered_shape)
				x_scaled = self.scale_renderer.resample_grid3D_aligned(self.x, x_shape, align_x='center')
				y_scaled = self.scale_renderer.resample_grid3D_aligned(self.y, y_shape, align_y='center')
				z_scaled = self.scale_renderer.resample_grid3D_aligned(self.z, z_shape, align_z='center')
				if scale_magnitude:
					vel_scale = [o/i for i,o in zip(self.centered_shape, centered_shape)] #z,y,x
					LOG.debug("Scaling velocity magnitude with %s", vel_scale)
					x_scaled *= vel_scale[2]
					y_scaled *= vel_scale[1]
					z_scaled *= vel_scale[0]
		else:
			LOG.debug("No need to scale velocity to same shape %s", self.centered_shape)
			x_scaled = tf.identity(self.x)
			y_scaled = tf.identity(self.y)
			z_scaled = tf.identity(self.z)
		return x_scaled, y_scaled, z_scaled
	
	@staticmethod
	def resample_velocity(scale_renderer, vel, scale=None, shape=None, is_staggered=True, scale_magnitude=True):
		if scale is None and shape is None: raise ValueError("You must provide 1 of scale or shape.")
		if scale is not None and shape is not None: raise ValueError("You must provide 1 of scale or shape.")
		
		#vel base shape
		if isinstance(vel, tuple):
			x_shape = shape_list(vel[0])
			assert len(x_shape)==5
			centered_shape = x_shape
			centered_shape[-2] -= 1
		elif isinstance(vel, (np.ndarray, tf.Tensor)):
			centered_shape = shape_list(vel)
			assert len(centered_shape)==5
		else:
			raise ValueError("Unknown velcity format '%s'"%(type(vel).__name__,))
		
		if scale is not None:
			assert isinstance(scale, numbers.Number), "invalid scaling factor"
			shape = [int(round(_*scale)) for _ in centered_shape[1:-1]]
		if shape is not None:
			assert isinstance(shape, (list, tuple)) and len(shape)==3, "invalid shape"
			scale = np.mean([float(t)/float(s) for s,t in zip(centered_shape[1:-1], shape)])
		
		# our sampler can only handle 1,2 and 4 channels, not 3.
		if is_staggered:
			if isinstance(vel, tuple):
				# separate staggered grids with different dimenstions
				x_shape, y_shape, z_shape = VelocityGrid.component_shapes(shape)
				x,y,z = vel
				vel_components = [
					scale_renderer.resample_grid3D_aligned(x, x_shape, align_x="CENTER", align_y="BORDER", align_z="BORDER"),
					scale_renderer.resample_grid3D_aligned(y, y_shape, align_x="BORDER", align_y="CENTER", align_z="BORDER"),
					scale_renderer.resample_grid3D_aligned(z, z_shape, align_x="BORDER", align_y="BORDER", align_z="CENTER")]
				
				if scale_magnitude:
					vel_components = [_*scale for _ in vel_components]
				return tuple(vel_components)
				
			elif isinstance(vel, (np.ndarray, tf.Tensor)):
				# combined staggered grid, like mantaflow
				raise NotImplementedError("TODO: check correct shape")
				x,y,z = tf.split(vel, 3, axis=-1)
				vel_components = [
					scale_renderer.resample_grid3D_aligned(x, shape, align_x="CENTER", align_y="BORDER", align_z="BORDER"),
					scale_renderer.resample_grid3D_aligned(y, shape, align_x="BORDER", align_y="CENTER", align_z="BORDER"),
					scale_renderer.resample_grid3D_aligned(z, shape, align_x="BORDER", align_y="BORDER", align_z="CENTER")]
				
				vel =  tf.concat(vel_components, axis=-1)
				if scale_magnitude:
					vel = vel*scale
				return vel
		else:
			vel_components = tf.split(vel, [2,1], axis=-1)
			vel_components = [scale_renderer.resample_grid3D_aligned(_, shape) for _ in vel_components]
		
			vel =  tf.concat(vel_components, axis=-1)
			if scale_magnitude:
				vel = vel*scale
			return vel
	
	def copy_scaled(self, centered_shape, scale_magnitude=True, as_var=None, device=None, var_name=None, trainable=None):
		if as_var is None:
			as_var = self.is_var
		if as_var and var_name is None:
			var_name = self._name + '_scaled'
		if trainable is None:
			trainable = self._is_trainable
		x_scaled, y_scaled, z_scaled = self.scaled(centered_shape, scale_magnitude)
		grid = VelocityGrid(centered_shape, x=x_scaled, y=y_scaled, z=z_scaled, as_var=as_var, \
			boundary=self.boundary, scale_renderer=self.scale_renderer, warp_renderer=self.warp_renderer, device=device, var_name=var_name, trainable=trainable)
		return grid
	
	def _lut_warp_vel(self, shape, dt=1.0):
		# use to get lookup positions to warp velocity components
		vel = self._sampled_to_shape(shape) #3 x 1DHW1
		#coords = get_coord_field(shape, lod=0.0, concat=False) #4 x 1DHW1
		#vel_lut = [coords[i] - vel[i]*dt for i in range(len(vel))] + [coords[-1]] #4 x 1DHW1
		vel_lut = [- vel[i]*dt for i in range(len(vel))] #3 x 1DHW1
		vel_lut = tf.concat(vel_lut, axis = -1) #1DHW3
		return vel_lut
	
	def _warp_vel_component(self, data, lut, order=1, dt=1.0, clamp="NONE"):
		if order<1 or order>2:
			raise ValueError("Unsupported warp order '{}'".format(order))
		warped = self.warp_renderer._sample_LuT(data, lut, True, relative=True)
		clamp = clamp.upper()
		if order==2: #MacCormack
			warped_back = self.warp_renderer._sample_LuT(warped, -lut, True, relative=True)
			corrected = warped + 0.5*(data-warped_back)
			if clamp=="MC" or clamp=="MC_SMOOTH":
				#raise NotImplementedError("MacCormack clamping has not been implemented.")
				fm = self.warp_renderer.filter_mode
				self.warp_renderer.filter_mode = "MIN"
				data_min = self.warp_renderer._sample_LuT(data, lut, True, relative=True)
				self.warp_renderer.filter_mode = "MAX"
				data_max = self.warp_renderer._sample_LuT(data, lut, True, relative=True)
				self.warp_renderer.filter_mode = fm
				if clamp=='MC':
					#LOG.warning("Experimental clamp for MacCormack velocity advection.")
					raise NotImplementedError("MIM and MAX warp sampling have wrong gradients.")
					corrected = tf.clip_by_value(corrected, data_min, data_max)
				if clamp=='MC_SMOOTH':
					#LOG.warning("Experimental 'revert' clamp for MacCormack velocity advection.")
					clamp_OOB = tf.logical_or(tf.less(corrected, data_min), tf.greater(corrected, data_max))
					corrected = tf.where(clamp_OOB, warped, corrected)
			warped = corrected
		return warped
	
	def warped(self, vel_grid=None, order=1, dt=1.0, clamp="NONE"):
		if vel_grid is None:
			#vel_grid = self
			pass
		elif not isinstance(vel_grid, VelocityGrid):
			raise TypeError("Invalid VelocityGrid")
		with self.warp_renderer.profiler.sample("warp velocity"):
			LOG.debug("Warping velocity grid")
			#TODO will cause errors if grid shapes do not match, resample if necessary?
			if vel_grid is None:
				lut_x = tf.concat([-vel*dt for vel in self._sampled_to_component_shape('X', concat=False)], axis=-1)
			else:
				lut_x = vel_grid._lut_warp_vel(self.x_shape, dt)
			x_warped = self._warp_vel_component(self.x, lut_x, order=order, dt=dt, clamp=clamp)
			del lut_x
			
			if vel_grid is None:
				lut_y = tf.concat([-vel*dt for vel in self._sampled_to_component_shape('Y', concat=False)], axis=-1)
			else:
				lut_y = vel_grid._lut_warp_vel(self.y_shape, dt)
			y_warped = self._warp_vel_component(self.y, lut_y, order=order, dt=dt, clamp=clamp)
			del lut_y
			
			
			if vel_grid is None:
				lut_z = tf.concat([-vel*dt for vel in self._sampled_to_component_shape('Z', concat=False)], axis=-1)
			else:
				lut_z = vel_grid._lut_warp_vel(self.z_shape, dt)
			z_warped = self._warp_vel_component(self.z, lut_z, order=order, dt=dt, clamp=clamp)
			del lut_z
			'''
			lut_warp_x = vel_grid._lut_warp_vel(self.x_shape, dt)
			lut_warp_y = vel_grid._lut_warp_vel(self.y_shape, dt)
			lut_warp_z = vel_grid._lut_warp_vel(self.z_shape, dt)
			x_warped = self.warp_renderer._sample_LuT(self.x, lut_warp_x, True, relative=True)
			y_warped = self.warp_renderer._sample_LuT(self.y, lut_warp_y, True, relative=True)
			z_warped = self.warp_renderer._sample_LuT(self.z, lut_warp_z, True, relative=True)
		#	x_warped = self.warp_renderer._sample_LuT(self.x, vel_grid._lut_warp_vel(self.x_shape, dt), True, relative=False)
		#	y_warped = self.warp_renderer._sample_LuT(self.y, vel_grid._lut_warp_vel(self.y_shape, dt), True, relative=False)
		#	z_warped = self.warp_renderer._sample_LuT(self.z, vel_grid._lut_warp_vel(self.z_shape, dt), True, relative=False)
			clamp = clamp.upper()
			if order==2: #MacCormack
				#raise NotImplementedError
				x_warped_back = self.warp_renderer._sample_LuT(x_warped, -lut_warp_x, True, relative=True)
				x_warped += 0.5*(self.x-x_warped_back)
				y_warped_back = self.warp_renderer._sample_LuT(y_warped, -lut_warp_y, True, relative=True)
				y_warped += 0.5*(self.y-y_warped_back)
				z_warped_back = self.warp_renderer._sample_LuT(z_warped, -lut_warp_z, True, relative=True)
				z_warped += 0.5*(self.z-z_warped_back)
				#clamp?
				if clamp=="MC" or clamp=="MC_SMOOTH":
					raise NotImplementedError()
			elif order>2:
				raise ValueError("Unsupported warp order '{}'".format(order))
			'''
		#VelocityGrid(self.centered_shape, as_var=False, x=x_warped, y=y_warped, z=z_warped, scale_renderer=scale_renderer, warp_renderer=warp_renderer, coords=self.coords, lod=self.lod_pad)
		return x_warped, y_warped, z_warped
	
	def copy_warped(self, vel_grid=None, as_var=None, order=1, dt=1.0, device=None, var_name=None, clamp="NONE", trainable=None):
		if as_var is None:
			as_var = self.is_var
		if as_var and var_name is None:
			var_name = self._name + '_warped'
		if trainable is None:
			trainable = self._is_trainable
		x_warped, y_warped, z_warped = self.warped(vel_grid, order, dt, clamp=clamp)
		grid = VelocityGrid(self.centered_shape, x=x_warped, y=y_warped, z=z_warped, as_var=as_var, \
			boundary=self.boundary, scale_renderer=self.scale_renderer, warp_renderer=self.warp_renderer, device=device, var_name=var_name, trainable=trainable)
		#grid.assign(x_warped, y_warped, z_warped)
		return grid
	
	def divergence_free(self, residual=1e-5):
		raise NotImplementedError
	
	def var_list(self):
		if self.is_var:
			return [self._x, self._y, self._z]
		else:
			raise TypeError("This VelocityGrid is not a variable.")
	
	def get_variables(self):
		if self.is_var:
			return {'velocity_x': self._x, 'velocity_y': self._y, 'velocity_z': self._z}
		else:
			raise TypeError("This VelocityGrid is not a variable.")
	
	def get_output_variables(self, centered=True, staggered=True, include_MS=False, include_residual=False, only_trainable=False):
		return {'velocity_x': self._x, 'velocity_y': self._y, 'velocity_z': self._z}
	
	def save(self, path):
		np.savez_compressed(path, centered_shape=self.centered_shape, vel_x=self.x.numpy(), vel_y=self.y.numpy(), vel_z=self.z.numpy())
	
	#def load(self, path):
	#	vel = np.load(path)
	#	self.x.assign(vel["vel_x"])
	#	self.y.assign(vel["vel_y"])
	#	self.z.assign(vel["vel_z"])
	
	def assign(self, x,y,z):
		x_shape = shape_list(x)
		if not len(x_shape)==5 or not x_shape[-1]==1 or not x_shape[-4:-1]==self.x_shape:
			raise ValueError("Invalid or incompatible shape of velocity x component on assignment: is {}, required: NDHW1 with DHW={}".format(x_shape, self.x_shape))
		y_shape = shape_list(y)
		if not len(y_shape)==5 or not y_shape[-1]==1 or not y_shape[-4:-1]==self.y_shape:
			raise ValueError("Invalid or incompatible shape of velocity y component on assignment: is {}, required: NDHW1 with DHW={}".format(y_shape, self.y_shape))
		z_shape = shape_list(z)
		if not len(z_shape)==5 or not z_shape[-1]==1 or not z_shape[-4:-1]==self.z_shape:
			raise ValueError("Invalid or incompatible shape of velocity z component on assignment: is {}, required: NDHW1 with DHW={}".format(z_shape, self.z_shape))
		if self.is_var:
			self._x.assign(x)
			self._y.assign(y)
			self._z.assign(z)
		else:
			with tf.device(self._device):
				self._x = tf.identity(x)
				self._y = tf.identity(y)
				self._z = tf.identity(z)
	
	def assign_centered(self, centered_grid):
		assert isinstance(centered_grid, (np.ndarray, tf.Tensor))
		centered_shape = shape_list(centered_grid)
		if not len(centered_shape)==5 or not centered_shape[-1]==3 or not centered_shape[-4:-1]==self.centered_shape:
			raise ValueError("Invalid or incompatible shape of centered velocity assignment: is {}, required: NDHW3 with DHW={}".format(centered_shape, self.centered_shape))
		
		x,y,z = self._centered_to_staggered(centered_grid)
		if self.is_var:
			self._x.assign(x)
			self._y.assign(y)
			self._z.assign(z)
		else:
			with tf.device(self._device):
				self._x = tf.identity(x)
				self._y = tf.identity(y)
				self._z = tf.identity(z)
	
	def assign_staggered_combined(self, staggered_grid):
		# from a combined staggered grid where all dimensions are centered+1
		assert isinstance(staggered_grid, (np.ndarray, tf.Tensor))
		x,y,z = tf.split(staggered_grid, 3, axis=-1)
		x = x[...,:-1,:-1,:,:]
		y = y[...,:-1,:,:-1,:]
		z = z[...,:,:-1,:-1,:]
		self.assign(x,y,z)
	
	def assign_centered_scaled(self, centered_grid, scale_magnitude=True):
		assert isinstance(centered_grid, (np.ndarray, tf.Tensor))
		centered_shape = shape_list(centered_grid)
		if not len(centered_shape)==5 or not centered_shape[-1]==3:
			raise ValueError("Invalid or incompatible shape of centered velocity assignment: is {}, required: NDHW3 with DHW={}".format(centered_shape, self.centered_shape))
		
		x,y,z = self._centered_to_staggered(centered_grid, output_centered_shape=self.centered_shape, scale_magnitude=scale_magnitude)
		if self.is_var:
			self._x.assign(x)
			self._y.assign(y)
			self._z.assign(z)
		else:
			with tf.device(self._device):
				self._x = tf.identity(x)
				self._y = tf.identity(y)
				self._z = tf.identity(z)
	
	def assign_staggered_combined_scaled(self, staggered_grid):
		# TODO
		self.assign_staggered_combined(staggered_grid)
	
	def assign_add(self, x,y,z):
		x_shape = shape_list(x)
		if not len(x_shape)==5 or not x_shape[-1]==1 or not x_shape[-4:-1]==self.x_shape:
			raise ValueError("Invalid or incompatible shape of velocity x component on assignment: is {}, required: NDHW1 with DHW={}".format(x_shape, self.x_shape))
		y_shape = shape_list(y)
		if not len(y_shape)==5 or not y_shape[-1]==1 or not y_shape[-4:-1]==self.y_shape:
			raise ValueError("Invalid or incompatible shape of velocity y component on assignment: is {}, required: NDHW1 with DHW={}".format(y_shape, self.y_shape))
		z_shape = shape_list(z)
		if not len(z_shape)==5 or not z_shape[-1]==1 or not z_shape[-4:-1]==self.z_shape:
			raise ValueError("Invalid or incompatible shape of velocity z component on assignment: is {}, required: NDHW1 with DHW={}".format(z_shape, self.z_shape))
		if self.is_var:
			self._x.assign_add(x)
			self._y.assign_add(y)
			self._z.assign_add(z)
		else:
			with tf.device(self._device):
				self._x = tf.identity(self._x+x)
				self._y = tf.identity(self._y+y)
				self._z = tf.identity(self._z+z)
	
	def assign_sub(self, x,y,z):
		x_shape = shape_list(x)
		if not len(x_shape)==5 or not x_shape[-1]==1 or not x_shape[-4:-1]==self.x_shape:
			raise ValueError("Invalid or incompatible shape of velocity x component on assignment: is {}, required: NDHW1 with DHW={}".format(x_shape, self.x_shape))
		y_shape = shape_list(y)
		if not len(y_shape)==5 or not y_shape[-1]==1 or not y_shape[-4:-1]==self.y_shape:
			raise ValueError("Invalid or incompatible shape of velocity y component on assignment: is {}, required: NDHW1 with DHW={}".format(y_shape, self.y_shape))
		z_shape = shape_list(z)
		if not len(z_shape)==5 or not z_shape[-1]==1 or not z_shape[-4:-1]==self.z_shape:
			raise ValueError("Invalid or incompatible shape of velocity z component on assignment: is {}, required: NDHW1 with DHW={}".format(z_shape, self.z_shape))
		if self.is_var:
			self._x.assign_sub(x)
			self._y.assign_sub(y)
			self._z.assign_sub(z)
		else:
			with tf.device(self._device):
				self._x = tf.identity(self._x-x)
				self._y = tf.identity(self._y-y)
				self._z = tf.identity(self._z-z)
	
	def scale_magnitude(self, scale):
		if np.isscalar(scale):
			scale = [scale]*3
		assert len(scale)==3
		self.assign(self.x*scale[0],self.y*scale[1], self.z*scale[2])
	
	def _centered_to_staggered_old(self, centered):
		centered_shape = shape_list(centered)
		assert len(centered_shape)==5
		assert centered_shape[-1]==3
		#assert centered_shape[0]==1
		batch_size = centered_shape[0]
		assert self.centered_shape==centered_shape[-4:-1]
		with self.scale_renderer.profiler.sample("centered velocity to staggered"):
			x,y,z= tf.split(centered, 3, axis=-1)
			# TODO: rework to use Renderer.resample_grid3D_aligned()
			centered_x_transform = GridTransform(self.centered_shape, scale=[2./_ for _ in self.x_shape[::-1]], center=True)
			centered_y_transform = GridTransform(self.centered_shape, scale=[2./_ for _ in self.y_shape[::-1]], center=True)
			centered_z_transform = GridTransform(self.centered_shape, scale=[2./_ for _ in self.z_shape[::-1]], center=True)
			# only shape important here
			staggered_x_transform = GridTransform(self.x_shape)#,translation=[0.5,0,0])
			staggered_y_transform = GridTransform(self.y_shape)#,translation=[0,0.5,0])
			staggered_z_transform = GridTransform(self.z_shape)#,translation=[0,0,0.5])
			x = tf.squeeze(self.scale_renderer._sample_transform(x, [centered_x_transform]*batch_size, [staggered_x_transform]),1)
			y = tf.squeeze(self.scale_renderer._sample_transform(y, [centered_y_transform]*batch_size, [staggered_y_transform]),1)
			z = tf.squeeze(self.scale_renderer._sample_transform(z, [centered_z_transform]*batch_size, [staggered_z_transform]),1)
		return x,y,z
	
	def _centered_to_staggered(self, centered, output_centered_shape=None, scale_magnitude=True):
		centered_shape = shape_list(centered)
		assert len(centered_shape)==5 #NDHWC
		assert centered_shape[-1]==3 #C=3
		#assert centered_shape[0]==1
		batch_size = centered_shape[0]
		if output_centered_shape is None:
			output_centered_shape = centered_shape[-4:-1]
			scale_factors = None
		else:
			assert has_shape(output_centered_shape, [3])
			scale_factors = [o/i for o,i in zip(output_centered_shape, centered_shape[-4:-1])] #shape DHW -> zyx
		#assert self.centered_shape==centered_shape[-4:-1]
		with self.scale_renderer.profiler.sample("centered velocity to staggered"):
			x_shape, y_shape, z_shape = self.component_shapes(output_centered_shape)
			x,y,z= tf.split(centered, 3, axis=-1)
			x = self.scale_renderer.resample_grid3D_aligned(x, x_shape, align_x="CENTER", align_y="BORDER", align_z="BORDER")
			y = self.scale_renderer.resample_grid3D_aligned(y, y_shape, align_x="BORDER", align_y="CENTER", align_z="BORDER")
			z = self.scale_renderer.resample_grid3D_aligned(z, z_shape, align_x="BORDER", align_y="BORDER", align_z="CENTER")
		if scale_magnitude and scale_factors is not None:
			x = x*scale_factors[2]
			y = y*scale_factors[1]
			z = z*scale_factors[0]
		return x,y,z
	
	def _scalar_centered_to_staggered(self, centered, *, allow_split_channels=False):
		centered_shape = shape_list(centered)
		assert len(centered_shape)==5
		assert (allow_split_channels or centered_shape[-1] in [1,2,4]), "resampling only supports 1,2 or 4 channels."
		batch_size = centered_shape[0]
		staggered_shape = self.shape_centered_to_staggered(centered_shape[-4:-1])
		with self.scale_renderer.profiler.sample("centered scalar to staggered"):
			if (allow_split_channels and centered_shape[-1] not in [1,2,4]):
				splits = tf_split_to_size(centered, [1,2,4], axis=-1)
				staggered = []
				for split in splits:
					staggered.append(self.scale_renderer.resample_grid3D_aligned(split, staggered_shape, align_x='STAGGER_OUTPUT', align_y='STAGGER_OUTPUT', align_z='STAGGER_OUTPUT'))
				staggered = tf.concat(staggered, axis=-1)
				assert shape_list(staggered)[-1]==centered_shape[-1]
			else:
				staggered = self.scale_renderer.resample_grid3D_aligned(centered, staggered_shape, align_x='STAGGER_OUTPUT', align_y='STAGGER_OUTPUT', align_z='STAGGER_OUTPUT')
		return staggered
	
	def _staggeredTensor_to_components(self, tensor, reverse=False):
		tensor_shape = GridShape.from_tensor(tensor)
	#	assert len(tensor_shape)==5
		assert tensor_shape.c==3
		#assert tensor_shape.n==1
		#assert np.asarray(self.centered_shape)+np.asarray([1,1,1])== tensor_shape.xyz.as_shape() #tensor_shape[-4:-1]
		tensor = tensor_shape.normalize_tensor_shape(tensor)
		components = tf.split(tensor, 3, axis=-1)
		if reverse:
			components = components[::-1]
		x = components[0][:,:-1,:-1,:]
		y = components[1][:,:-1,:,:-1]
		z = components[2][:,:,:-1,:-1]
		return x,y,z
	
	def _components_to_staggeredTensor(self, comp_x,comp_y,comp_z, reverse=False):
		z = (0,0)
		p = (0,1)
		components = [
			tf.pad(comp_x, [z,p,p,z,z]),
			tf.pad(comp_y, [z,p,z,p,z]),
			tf.pad(comp_z, [z,z,p,p,z]),
		]
		if reverse:
			components = components[::-1]
		return tf.concat(components, axis=-1)
	
	def as_staggeredTensor(self, reverse=False):
		return self._components_to_staggeredTensor(self.x, self.y, self.z, reverse=reverse)
	
	def _sampled_to_shape(self, shape):
		with self.scale_renderer.profiler.sample("velocity to shape"):
			# uniform scaling, centered grids
			#_sample_transform is currently experimental and assumes the output grid to be in a centered [-1,1] cube, so scale input accordingly
			# scale with output shape to get the right 0.5 offset
			scale = [2./_ for _ in shape[::-1]]
			staggered_x_transform = GridTransform(self.x_shape, scale=scale, center=True)
			staggered_y_transform = GridTransform(self.y_shape, scale=scale, center=True)
			staggered_z_transform = GridTransform(self.z_shape, scale=scale, center=True)
			# only shape important here
			sample_transform = GridTransform(shape)
			#check if shape matches component shape to avoid sampling (e.g. for self warping)
			vel_sampled = [
				tf.squeeze(self.scale_renderer._sample_transform(self.x, [staggered_x_transform], [sample_transform]),1) \
					if not shape==self.x_shape else tf.identity(self.x), #1DHW1
				tf.squeeze(self.scale_renderer._sample_transform(self.y, [staggered_y_transform], [sample_transform]),1) \
					if not shape==self.y_shape else tf.identity(self.y),
				tf.squeeze(self.scale_renderer._sample_transform(self.z, [staggered_z_transform], [sample_transform]),1) \
					if not shape==self.z_shape else tf.identity(self.z),
			]
		return vel_sampled
	
	def _staggered_to_centered(self, staggered, concat=True):
		staggered_shape = shape_list(staggered)
		assert len(staggered_shape) == 5
		assert staggered_shape[-1] == 3
		with self.warp_renderer.profiler.sample("staggered_to_centered"):
			h = tf.constant(0.5, dtype=tf.float32)
			x,y,z = tf.split(staggered, 3, axis=-1)
		#	x = components[0][:,:-1,:-1,:]
		#	y = components[1][:,:-1,:,:-1]
		#	z = components[2][:,:,:-1,:-1]
			vel_centered = [
				(x[:,:-1,:-1,1:] + x[:,:-1,:-1,:-1])*h,
				(y[:,:-1,1:,:-1] + y[:,:-1,:-1,:-1])*h,
				(z[:,1:,:-1,:-1] + z[:,:-1,:-1,:-1])*h,
			]
			if concat:
				vel_centered = tf.concat(vel_centered, axis=-1)
		return vel_centered
			
	
	@staticmethod
	def _check_vel_component_shape(x,y,z):
		# staggered velocity: shape(z,y,x)
		#   x: (+0,+0,+1)
		#   y: (+0,+1,+0)
		#   z: (+1,+0,+0)
		x_shape = shape_list(x)
		if len(x_shape)!=5 or x_shape[-1]!=1: return False
		c_shape = copy.copy(x_shape[-4:-1])
		c_shape[2] -=1
		batch = x_shape[-1]
		
		y_shape = shape_list(y)
		if len(y_shape)!=5 or y_shape[-1]!=1: return False
		if y_shape[-4]!=c_shape[0] or y_shape[-3]!=(c_shape[1]+1) or y_shape[-2]!=c_shape[2]: return False
		if y_shape[-1]!=batch: return False
		
		z_shape = shape_list(z)
		if len(z_shape)!=5 or z_shape[-1]!=1: return False
		if z_shape[-4]!=(c_shape[0]+1) or z_shape[-3]!=c_shape[1] or z_shape[-2]!=c_shape[2]: return False
		if z_shape[-1]!=batch: return False
		
		return True
	
	@staticmethod
	def _check_potential_component_shape(x,y,z):
		# staggered potential: shape(z,y,x)
		#   x: (+1,+1,+0)
		#   y: (+1,+0,+1)
		#   z: (+0,+1,+1)
		x_shape = shape_list(x)
		if len(x_shape)!=5 or x_shape[-1]!=1: return False
		c_shape = copy.copy(x_shape[-4:-1]) #z,y,x
		c_shape[0] -=1
		c_shape[1] -=1
		batch = x_shape[-1]
		
		y_shape = shape_list(y)
		if len(y_shape)!=5 or y_shape[-1]!=1: return False
		if y_shape[-4]!=(c_shape[0]+1) or y_shape[-3]!=(c_shape[1]) or y_shape[-2]!=(c_shape[2]+1): return False
		if y_shape[-1]!=batch: return False
		
		z_shape = shape_list(z)
		if len(z_shape)!=5 or z_shape[-1]!=1: return False
		if z_shape[-4]!=(c_shape[0]) or z_shape[-3]!=(c_shape[1]+1) or z_shape[-2]!=(c_shape[2]+1): return False
		if z_shape[-1]!=batch: return False
		
		return True
	
	def _components_to_centered(self, x,y,z, concat=True):
		if not self._check_vel_component_shape(x,y,z): raise ValueError("shapes of components do not fit. x: {}, y: {}, z: {}".format(shape_list(x), shape_list(y), shape_list(z)))
		with self.warp_renderer.profiler.sample("components_to_centered"):
			h = tf.constant(0.5, dtype=tf.float32)
			vel_centered = [
				(x[:,:,:,1:] + x[:,:,:,:-1])*h,
				(y[:,:,1:] + y[:,:,:-1])*h,
				(z[:,1:] + z[:,:-1])*h,
			]
			if concat:
				vel_centered = tf.concat(vel_centered, axis=-1)
		return vel_centered
		
	
	def _staggered_components_potential_to_staggered_components(self, pot_x, pot_y, pot_z):
		if not self._check_potential_component_shape(pot_x, pot_y, pot_z):
			raise ValueError("shapes of potential components do not fit. x: {}, y: {}, z: {}".format(shape_list(pot_x), shape_list(pot_y), shape_list(pot_z)))
		
		vel_components = [
			(pot_z[:,:,:-1,:,:] - pot_z[:,:,1:,:,:]) - (pot_y[:,:-1,:,:,:] - pot_y[:,1:,:,:,:]), # dPz/dy - dPy/dz
			(pot_x[:,:-1,:,:,:] - pot_x[:,1:,:,:,:]) - (pot_z[:,:,:,:-1,:] - pot_z[:,:,:,1:,:]), # dPx/dz - dPz/dx
			(pot_y[:,:,:,:-1,:] - pot_y[:,:,:,1:,:]) - (pot_x[:,:,:-1,:,:] - pot_x[:,:,1:,:,:]), # dPy/dx - dPx/dy
		]
		
		return tuple(vel_components)
		
	def _staggeredTensor_potential_to_staggered_components(self, pot):
		tensor_shape = GridShape.from_tensor(pot)
		assert tensor_shape.c==3
		pot = tensor_shape.normalize_tensor_shape(pot)
		pot_x, pot_y, pot_z = tf.split(pot, 3, axis=-1)
		
		vel_components = [
			(pot_z[:,:-1,:-1,:,:] - pot_z[:,:-1,1:,:,:]) - (pot_y[:,:-1,:-1,:,:] - pot_y[:,1:,:-1,:,:]), # dPz/dy - dPy/dz
			(pot_x[:,:-1,:,:-1,:] - pot_x[:,1:,:,:-1,:]) - (pot_z[:,:-1,:,:-1,:] - pot_z[:,:-1,:,1:,:]), # dPx/dz - dPz/dx
			(pot_y[:,:,:-1,:-1,:] - pot_y[:,:,:-1,1:,:]) - (pot_x[:,:,:-1,:-1,:] - pot_x[:,:,1:,:-1,:]), # dPy/dx - dPx/dy
		]
		
		return tuple(vel_components)
	
	def _staggeredTensor_potential_to_components(self, tensor, reverse=False):
		tensor_shape = GridShape.from_tensor(tensor)
	#	assert len(tensor_shape)==5
		assert tensor_shape.c==3
		#assert tensor_shape.n==1
		#assert np.asarray(self.centered_shape)+np.asarray([1,1,1])== tensor_shape.xyz.as_shape() #tensor_shape[-4:-1]
		tensor = tensor_shape.normalize_tensor_shape(tensor)
		components = tf.split(tensor, 3, axis=-1)
		if reverse:
			components = components[::-1]
		x = components[0][:,:,:,:-1,:]
		y = components[1][:,:,:-1,:,:]
		z = components[2][:,:-1,:,:,:]
		return x,y,z
	
	def _components_potential_to_staggeredTensor_potential(self, comp_x, comp_y, comp_z, reverse=False):
		z = (0,0)
		p = (0,1)
		components = [
			tf.pad(comp_x, [z,z,z,p,z]),
			tf.pad(comp_y, [z,z,p,z,z]),
			tf.pad(comp_z, [z,p,z,z,z]),
		]
		if reverse:
			components = components[::-1]
		return tf.concat(components, axis=-1)
	
	def _staggeredTensor_potential_to_staggeredTensor(self, pot):
		return self._components_to_staggeredTensor(*self._staggeredTensor_potential_to_staggered_components(pot))
	
	def _staggeredTensor_potential_to_centered(self, pot):
		return self._components_to_centered(*self._staggeredTensor_potential_to_staggered_components(pot))
	
	def _centered_to_curl(self, vel):
		raise NotImplementedError("Deprecated")
		assert isinstance(vel, (tf.Tensor, np.ndarray))
		vel_shape = shape_list(vel)
		assert len(vel_shape)==5 and vel_shape[-1]==3 #NDHWC
		
		# https://en.wikipedia.org/wiki/Curl_(mathematics)
		# 
		with self.warp_renderer.profiler.sample("centered_to_curl"):
			#vel = tf.pad(vel, [(0,0),(1,1),(1,1),(1,1),(0,0)], "SYMMETRIC")
			vel_x, vel_y, vel_z = tf.split(vel, 3, axis=-1)
			# curl_x = d vel_z/d y - d vel_y/d z
			#   d vel_z/d y -> finite (central) difference
			
			def central_diff(v, axis):
				pad = [(0,0)]*5
				pad[axis] = (1,1)
				v = tf.pad(v, pad, "SYMMETRIC")
				
				slice_size = shape_list(v)
				slice_size[axis] -= 2
				
				slice_begin = [0]*5
				v_a = tf.slice(v, begin=slice_begin, size=slice_size)
				
				slice_begin[axis] = 2
				v_b = tf.slice(v, begin=slice_begin, size=slice_size)
				
				return v_a - v_b
			
			# vel_x_dy = central_diff(vel_x, axis=-3) #vel_x[:,:,:-2,:,:] - vel_x[:,:,2:,:,:]
			# vel_x_dz = central_diff(vel_x, axis=-4) #vel_x[:,:-2,:,:,:] - vel_x[:,2:,:,:,:]
			# vel_y_dx = central_diff(vel_y, axis=-2) #vel_y[:,:,:,:-2,:] - vel_y[:,:,:,2:,:]
			# vel_y_dz = central_diff(vel_y, axis=-4) #vel_y[:,:-2,:,:,:] - vel_y[:,2:,:,:,:]
			# vel_z_dx = central_diff(vel_z, axis=-2) #vel_z[:,:,:,:-2,:] - vel_z[:,:,:,2:,:]
			# vel_z_dy = central_diff(vel_z, axis=-3) #vel_z[:,:,:-2,:,:] - vel_z[:,:,2:,:,:]
			
			# curl_x = vel_z_dy - vel_y_dz
			# curl_y = vel_x_dz - vel_z_dx
			# curl_z = vel_y_dx - vel_x_dy
			
			curl = [
				central_diff(vel_z, axis=-3) - central_diff(vel_y, axis=-4),
				central_diff(vel_x, axis=-4) - central_diff(vel_z, axis=-2),
				central_diff(vel_y, axis=-2) - central_diff(vel_x, axis=-3),
			]
			
			return tf.concat(curl, axis=-1)
		
		
	def _staggered_to_curl(self, x,y,z):
		if not self._check_vel_component_shape(x,y,z): raise ValueError("shapes of components do not fit. x: {}, y: {}, z: {}".format(shape_list(x), shape_list(y), shape_list(z)))
		with self.warp_renderer.profiler.sample("staggered_to_curl"):
			pass
		raise NotImplementedError
	
	def centered(self, pad_lod=False, concat=True):#, shape=None):
	#	if shape is None:
		shape = self.centered_shape
		with self.warp_renderer.profiler.sample("velocity to centered"):
			#vel_centered = self._sampled_to_shape(shape)#3 x 1DHW1
			h = tf.constant(0.5, dtype=tf.float32)
			vel_centered = [
				(self.x[:,:,:,1:] + self.x[:,:,:,:-1])*h,
				(self.y[:,:,1:] + self.y[:,:,:-1])*h,
				(self.z[:,1:] + self.z[:,:-1])*h,
			]
			if pad_lod:
				vel_centered.append(self.lod_pad)#4 x 1DHW1
			if concat:
				vel_centered = tf.concat(vel_centered, axis=-1) #1DHW[3|4]
	#		vel_centered = [tf.squeeze(_, -1) for _ in vel_centered] #3 x 1DHW
	#		vel_centered = tf.transpose(vel_centered, (1,2,3,4,0))#1DHW3
	#		if pad_lod:
	#			vel_centered = tf.concat([vel_centered, self.lod_pad], axis = -1)
		return vel_centered
	
	def _sampled_to_component_shape(self, component, pad_lod=False, concat=True):
		# grids have the same spacing/resolution, so global/constant offset
		component = component.upper()
		offset_coord_from = 0.5
		offset_coord_to = -0.5
		with self.warp_renderer.profiler.sample("velocity to component shape"):
			vel_sampled = []
			# sample x
			vel_sampled.append(tf.identity(self.x) if component=='X' else \
				tf.squeeze(self.warp_renderer.resample_grid3D_offset(self.x, \
					offsets = [[offset_coord_from,offset_coord_to,0.0] if component=='Y' else [offset_coord_from,0.0,offset_coord_to],], \
					target_shape = self.y_shape if component=='Y' else self.z_shape), 1))
			# sample y
			vel_sampled.append(tf.identity(self.y) if component=='Y' else \
				tf.squeeze(self.warp_renderer.resample_grid3D_offset(self.y, \
					offsets = [[offset_coord_to,offset_coord_from,0.0] if component=='X' else [0.0,offset_coord_from,offset_coord_to],], \
					target_shape = self.x_shape if component=='X' else self.z_shape), 1))
			# sample z
			vel_sampled.append(tf.identity(self.z) if component=='Z' else \
				tf.squeeze(self.warp_renderer.resample_grid3D_offset(self.z, \
					offsets = [[offset_coord_to,0.0,offset_coord_from] if component=='X' else [0.0,offset_coord_to,offset_coord_from],], \
					target_shape = self.x_shape if component=='X' else self.y_shape), 1))
			
			if pad_lod:
				vel_sampled.append(self.lod_pad)#4 x 1DHW1
			if concat:
				vel_sampled = tf.concat(vel_sampled, axis=-1) #1DHW[3|4]
		return vel_sampled
	
	def centered_lut_grid(self, dt=1.0, centered_velocity=None):
		vel_centered = self.centered() if (centered_velocity is None) else centered_velocity
		#vel_lut = tf.concat([self.coords - vel_centered * dt, self.lod_pad], axis = -1)
		vel_lut = vel_centered * (- dt)
		return vel_lut
	
	def warp(self, data, order=1, dt=1.0, clamp="NONE", centered_velocity=None):
		with self.warp_renderer.profiler.sample("warp scalar"):
			v = self.centered_lut_grid(dt=dt, centered_velocity=centered_velocity)
			data_shape = spacial_shape_list(data)
			vel_shape = spacial_shape_list(v)
			if data_shape!=vel_shape: #self.centered_shape:
				raise ValueError("Shape mismatch in centered warp: data {}, velocity {}".format(data_shape, vel_shape))
			#	LOG.debug("Scaling velocity grid from %s to %s for warping", self.centered_shape, data_shape)
				#TODO handle vector mipmapping and filtering...
			#	v = self.scale_renderer.resample_grid3D_aligned(v, data_shape)
			#	vel_scale = [o/i for o,i in zip(data_shape, self.centered_shape)]#z,y,x
			#	LOG.debug("Rescale velocity for warping from %s to %s with vector scale %s", self.centered_shape, data_shape, vel_scale)
			#	vel_scale = tf.constant(vel_scale[::-1] + [0.], dtype=tf.float32) #z,y,x -> x,y,z,lod(=0)
				#TODO this is wrong, v is already a LuT (absolute positions)
			#	v = v * vel_scale
			LOG.debug("Warping density grid")
			data_warped = self.warp_renderer._sample_LuT(data, v, True, relative=True)
			
			clamp = clamp.upper()
			if order==2: #MacCormack
				#raise NotImplementedError
				data_warped_back = self.warp_renderer._sample_LuT(data_warped, -v, True, relative=True)
				#data_warped_back = self.warp(data_warped, dt=-dt) #self.warp_renderer._sample_LuT(data, v, True)
				data_corr = data_warped + 0.5*(data-data_warped_back)
				if clamp=='MC' or clamp=='MC_SMOOTH': #smooth clamp
					#raise NotImplementedError("MacCormack clamping has not been implemented.")
					fm = self.warp_renderer.filter_mode
					self.warp_renderer.filter_mode = "MIN"
					data_min = self.warp_renderer._sample_LuT(data, v, True, relative=True)
					self.warp_renderer.filter_mode = "MAX"
					data_max = self.warp_renderer._sample_LuT(data, v, True, relative=True)
					self.warp_renderer.filter_mode = fm
					if clamp=='MC':
						#LOG.warning("Experimental clamp for MacCormack density advection.")
						raise NotImplementedError("MIM and MAX warp sampling have wrong gradients.")
						data_corr = tf.clip_by_value(data_corr, data_min, data_max)
					if clamp=='MC_SMOOTH':
						#LOG.warning("Experimental 'revert' clamp for MacCormack density advection.")
						clamp_OOB = tf.logical_or(tf.less(data_corr, data_min), tf.greater(data_corr, data_max))
						data_corr = tf.where(clamp_OOB, data_warped, data_corr)
				data_warped = data_corr
			elif order>2:
				raise ValueError("Unsupported warp order '{}'".format(order))
			
			if clamp=='NEGATIVE':
				data_warped = tf.maximum(data_warped, 0)
			
			return data_warped
	
	def with_buoyancy(self, value, scale_grid):
		# value: [x,y,z]
		# scale_grid: density 1DHW1
		if isinstance(scale_grid, DensityGrid):
			scale_grid = scale_grid.with_inflow() #.d
		assert len(shape_list(value))==1
		if not isinstance(value, (tf.Tensor, tf.Variable)):
			value = tf.constant(value, dtype=tf.float32)
		value = tf.reshape(value, [1,1,1,1,shape_list(value)[0]])
		buoyancy = value*scale_grid # 1DHW3
		return self + buoyancy
	
	"""
	def apply_buoyancy(self, value, scale_grid):
		# value: [x,y,z]
		# scale_grid: density 1DHW1
		assert len(shape_list(value))==1
		value = tf.reshape(tf.constant(value, dtype=tf.float32), [1,1,1,1,shape_list(value)[0]])
		buoyancy = value*scale_grid # 1DHW3
		self += buoyancy
	"""
	#centered
	def divergence(self, world_scale=[1,1,1]):
		#out - in per cell, per axis
		x_div = self.x[:,:,:,1:,:] - self.x[:,:,:,:-1,:]
		y_div = self.y[:,:,1:,:,:] - self.y[:,:,:-1,:,:]
		z_div = self.z[:,1:,:,:,:] - self.z[:,:-1,:,:,:]
		# sum to get total divergence per cell
		div = x_div*world_scale[0]+y_div*world_scale[1]+z_div*world_scale[2]
		return div
	#centered
	def magnitude(self, world_scale=[1,1,1]):
		with self.warp_renderer.profiler.sample("magnitude"):
			v = self.centered(pad_lod=False)*tf.constant(world_scale, dtype=tf.float32)
			return tf_norm2(v, axis=-1, keepdims=True) #tf.norm(v, axis=-1, keepdims=True)
	
	def stats(self, world_scale=[1,1,1], mask=None, state=None, **warp_kwargs):
		'''
			mask: optional binary float mask, stats only consider cells>0.5
		'''
		x = self.x
		if mask is not None:
			mask_x = tf.greater(self.scale_renderer.resample_grid3D_aligned(mask, self.x_shape, align_x='stagger_output'), 0.5)
			x = tf.boolean_mask(x, mask_x)
		y = self.y
		if mask is not None:
			mask_y = tf.greater(self.scale_renderer.resample_grid3D_aligned(mask, self.y_shape, align_y='stagger_output'), 0.5)
			y = tf.boolean_mask(y, mask_y)
		z = self.z
		if mask is not None:
			mask_z = tf.greater(self.scale_renderer.resample_grid3D_aligned(mask, self.z_shape, align_z='stagger_output'), 0.5)
			z = tf.boolean_mask(z, mask_z)
		if mask is not None and mask.dtype!=tf.bool:
			mask = tf.greater(mask, 0.5)
		
		divergence = self.divergence(world_scale)
		if mask is not None: divergence = tf.boolean_mask(divergence, mask)
		magnitude = self.magnitude(world_scale)
		if mask is not None: magnitude = tf.boolean_mask(magnitude, mask)
		
		stats = {
		#	'divMean':tf.reduce_mean(divergence), 'divMax':tf.reduce_max(divergence), 'divMin':tf.reduce_min(divergence), 'divAbsMean':tf.reduce_mean(tf.abs(divergence)),
		#	'magMean':tf.reduce_mean(magnitude), 'magMax':tf.reduce_max(magnitude), 'magMin':tf.reduce_min(magnitude),
		#	'xMean':tf.reduce_mean(x), 'xMax':tf.reduce_max(x), 'xMin':tf.reduce_min(x), 'xAbsMean':tf.reduce_mean(tf.abs(x)),
		#	'yMean':tf.reduce_mean(y), 'yMax':tf.reduce_max(y), 'yMin':tf.reduce_min(y), 'yAbsMean':tf.reduce_mean(tf.abs(y)),
		#	'zMean':tf.reduce_mean(z), 'zMax':tf.reduce_max(z), 'zMin':tf.reduce_min(z), 'zAbsMean':tf.reduce_mean(tf.abs(z)),
			'divergence': tf_tensor_stats(divergence, as_dict=True),
			'magnitude': tf_tensor_stats(magnitude, as_dict=True),
			'velocity_x': tf_tensor_stats(x, as_dict=True),
			'velocity_y': tf_tensor_stats(y, as_dict=True),
			'velocity_z': tf_tensor_stats(z, as_dict=True),
			'shape':self.centered_shape, 'bounds':self.outer_bounds,
		}
		
		if state is not None and state.prev is not None and state.prev.velocity is not None:
			prev_warped = state.prev.velocity_advected(**warp_kwargs)
			
			def vel_warp_SE_stats(prev, curr, mask):
				warp_SE = tf.squared_difference(prev, curr)
				if mask is not None:
					warp_SE = tf.boolean_mask(warp_SE, mask)
				return tf_tensor_stats(warp_SE, as_dict=True)
			stats["warp_x_SE"] = vel_warp_SE_stats(prev_warped.x, self.x, mask_x if mask is not None else None)
			stats["warp_y_SE"] = vel_warp_SE_stats(prev_warped.y, self.y, mask_y if mask is not None else None)
			stats["warp_z_SE"] = vel_warp_SE_stats(prev_warped.z, self.z, mask_z if mask is not None else None)
			
			warp_vdiff_mag = (prev_warped-self).magnitude()
			if mask is not None:
				warp_vdiff_mag = tf.boolean_mask(warp_vdiff_mag, mask)
			stats["warp_vdiff_mag"] = tf_tensor_stats(warp_vdiff_mag, as_dict=True)
			del warp_vdiff_mag
			
			vel_CangleRad_mask = tf.greater(state.prev.velocity.magnitude() * self.magnitude(), 1e-8)
			if mask is not None:
				vel_CangleRad_mask = tf.logical_and(mask, vel_CangleRad_mask)
			warp_CangleRad = tf_angle_between(state.prev.velocity.centered(), self.centered(), axis=-1, keepdims=True)
			stats["warp_angleCM_rad"] = tf_tensor_stats(tf.boolean_mask(warp_CangleRad, vel_CangleRad_mask), as_dict=True)
			del warp_CangleRad
			
		else:
			stats["warp_x_SE"] = tf_tensor_stats(tf.zeros([1,1,1,1,1], dtype=tf.float32), as_dict=True)
			stats["warp_y_SE"] = tf_tensor_stats(tf.zeros([1,1,1,1,1], dtype=tf.float32), as_dict=True)
			stats["warp_z_SE"] = tf_tensor_stats(tf.zeros([1,1,1,1,1], dtype=tf.float32), as_dict=True)
			stats["warp_vdiff_mag"] = tf_tensor_stats(tf.zeros([1,1,1,1,1], dtype=tf.float32), as_dict=True)
			stats["warp_angleCM_rad"] = tf_tensor_stats(tf.zeros([1,1,1,1,1], dtype=tf.float32), as_dict=True)
		
		return stats
	
	def clear_cache(self):
		pass
	
	def __add__(self, other):
		if isinstance(other, VelocityGrid):
			if self.centered_shape!=other.centered_shape:
				raise ValueError("VelocityGrids of shape %s and %s are not compatible"%(self.centered_shape, other.centered_shape))
			return VelocityGrid(self.centered_shape, x=self.x+other.x, y=self.y+other.y, z=self.z+other.z, as_var=False, \
				boundary=self.boundary, scale_renderer=self.scale_renderer, warp_renderer=self.warp_renderer, device=None)
		if isinstance(other, (np.ndarray, tf.Tensor, tf.Variable)):
			other_shape = shape_list(other)
			if self.centered_shape!=spacial_shape_list(other) or other_shape[0]!=1 or other_shape[-1]!=3:
				raise ValueError("VelocityGrid of shape %s is not compatible with tensor of shape %s are not compatible"%(self.centered_shape, spacial_shape_list(other)))
			x,y,z = self._centered_to_staggered(other)
			return VelocityGrid(self.centered_shape, x=self.x+x, y=self.y+y, z=self.z+z, as_var=False, \
				boundary=self.boundary, scale_renderer=self.scale_renderer, warp_renderer=self.warp_renderer, device=None)
		else:
			return NotImplemented
	
	def __iadd__(self, other):
		if isinstance(other, VelocityGrid):
			if self.centered_shape!=other.centered_shape:
				raise ValueError("VelocityGrids of shape %s and %s are not compatible"%(self.centered_shape, other.centered_shape))
			self.assign_add(other.x, other.y, other.z)
			return self
		if isinstance(other, (np.ndarray, tf.Tensor, tf.Variable)):
			other_shape = shape_list(other)
			if self.centered_shape!=spacial_shape_list(other) or other_shape[0]!=1 or other_shape[-1]!=3:
				raise ValueError("VelocityGrid of shape %s is not compatible with tensor of shape %s are not compatible"%(self.centered_shape, spacial_shape_list(other)))
			x,y,z = self._centered_to_staggered(other)
			self.assign_add(x, y, z)
			return self
		else:
			return NotImplemented
	
	def __sub__(self, other):
		if isinstance(other, VelocityGrid):
			if self.centered_shape!=other.centered_shape:
				raise ValueError("VelocityGrids of shape %s and %s are not compatible"%(self.centered_shape, other.centered_shape))
			return VelocityGrid(self.centered_shape, x=self.x-other.x, y=self.y-other.y, z=self.z-other.z, as_var=False, \
				boundary=self.boundary, scale_renderer=self.scale_renderer, warp_renderer=self.warp_renderer, device=None)
		if isinstance(other, (np.ndarray, tf.Tensor, tf.Variable)):
			other_shape = shape_list(other)
			if self.centered_shape!=spacial_shape_list(other) or other_shape[0]!=1 or other_shape[-1]!=3:
				raise ValueError("VelocityGrid of shape %s is not compatible with tensor of shape %s are not compatible"%(self.centered_shape, spacial_shape_list(other)))
			x,y,z = self._centered_to_staggered(other)
			return VelocityGrid(self.centered_shape, x=self.x-x, y=self.y-y, z=self.z-z, as_var=False, \
				boundary=self.boundary, scale_renderer=self.scale_renderer, warp_renderer=self.warp_renderer, device=None)
		else:
			return NotImplemented
	
	def __isub__(self, other):
		if isinstance(other, VelocityGrid):
			if self.centered_shape!=other.centered_shape:
				raise ValueError("VelocityGrids of shape %s and %s are not compatible"%(self.centered_shape, other.centered_shape))
			self.assign_sub(other.x, other.y, other.z)
			return self
		if isinstance(other, (np.ndarray, tf.Tensor, tf.Variable)):
			other_shape = shape_list(other)
			if self.centered_shape!=spacial_shape_list(other) or other_shape[0]!=1 or other_shape[-1]!=3:
				raise ValueError("VelocityGrid of shape %s is not compatible with tensor of shape %s are not compatible"%(self.centered_shape, spacial_shape_list(other)))
			x,y,z = self._centered_to_staggered(other)
			self.assign_sub(x, y, z)
			return self
		else:
			return NotImplemented
	
	@property
	def is_centered(self):
		return False
	@property
	def is_staggered(self):
		return True
	@property
	def is_MS(self):
		return False
	@property
	def has_MS_output(self):
		return False

class State:
	def __init__(self, density, velocity, frame, prev=None, next=None, transform=None, targets=None, targets_raw=None, bkgs=None, masks=None):
		self.density = density
		self.velocity = velocity
		
		self.density_target = None
		self.velocity_target = None
		
		self.density_proxy = None
		
		assert isinstance(frame, numbers.Integral)
		self.frame = frame
		assert is_None_or_type(prev, State)
		self.prev = prev
		assert is_None_or_type(next, State)
		self.next = next
		
		assert isinstance(transform, Transform)
		self.transform = transform
		self.base_targets_raw = targets_raw
		self.base_targets = targets
		self.base_bkgs = bkgs
		self.base_masks = masks
		self.base_target_cameras = None
		self.set_base_target_cameras_MS(None)
		self.target_mask = None
		self.images = None
		self.t = None
		self.SDF_positions = None
		self._density_images_MS = {}
		self._density_images_t_MS = {}
		self._SDF_positions_MS = {}
	
	class StateIterator:
		def __init__(self, state):
			self.curr_state = state
		def __next__(self):
			if self.curr_state is not None:
				state = self.curr_state
				self.curr_state = state.next
				return state
			raise StopIteration
	def __iter__(self):
		return self.StateIterator(self)
	
	@property
	def has_density(self):
		return self.__density is not None
	@property
	def has_density_neural(self):
		return self.has_density and False
	@property
	def density(self):
		if self.has_density:
			return self.__density
		else:
			raise AttributeError("State for frame {} does not contain density".format(self.frame))
	@density.setter
	def density(self, value):
		assert is_None_or_type(value, DensityGrid)
		self.__density = value
	@property
	def _density(self):
		raise AttributeError("State._density is deprecated. Use State.density".format(self.frame))
	@density.setter
	def _density(self, value):
		raise AttributeError("State._density is deprecated. Assign to State.density".format(self.frame))
	
	@property
	def has_velocity(self):
		return self.__velocity is not None
	@property
	def velocity(self):
		if self.has_velocity:
			return self.__velocity
		else:
			raise AttributeError("State for frame {} does not contain velocity".format(self.frame))
	@velocity.setter
	def velocity(self, value):
		assert is_None_or_type(value, VelocityGrid)
		self.__velocity = value
	@property
	def _velocity(self):
		raise AttributeError("State._velocity is deprecated. Use State.velocity".format(self.frame))
	@velocity.setter
	def _velocity(self, value):
		raise AttributeError("State._velocity is deprecated. Assign to State.velocity".format(self.frame))
	
	@property
	def has_density_target(self):
		return self.__density_target is not None
	@property
	def density_target(self):
		if self.has_density_target:
			return self.__density_target
		else:
			raise AttributeError("State for frame {} does not contain a density target".format(self.frame))
	@density_target.setter
	def density_target(self, value):
		assert is_None_or_type(value, DensityGrid)
		self.__density_target = value
	
	@property
	def has_density_proxy(self):
		return self.__density_proxy is not None
	@property
	def density_proxy(self):
		if self.has_density_proxy:
			return self.__density_proxy
		else:
			raise AttributeError("State for frame {} does not contain a density proxy".format(self.frame))
	@density_proxy.setter
	def density_proxy(self, value):
		assert is_None_or_type(value, DensityGrid)
		self.__density_proxy = value
	
	@property
	def has_velocity_target(self):
		return self.__velocity_target is not None
	@property
	def velocity_target(self):
		if self.has_velocity_target:
			return self.__velocity_target
		else:
			raise AttributeError("State for frame {} does not contain a velocity target".format(self.frame))
	@velocity_target.setter
	def velocity_target(self, value):
		assert is_None_or_type(value, VelocityGrid)
		self.__velocity_target = value
	
	@property
	def base_target_cameras(self):
		if self.__target_cameras is not None:
			return self.__target_cameras
		else:
			raise AttributeError("State: base_target_cameras not set.")
	@base_target_cameras.setter
	def base_target_cameras(self, value):
		assert value is None or (isinstance(value, list) and all(isinstance(_, Camera) for _ in value))
		self.__target_cameras = value
	@property
	def target_cameras(self):
		if self.target_mask is not None:
			return [self.base_target_cameras[_] for _ in self.target_mask]
		else:
			return copy.copy(self.base_target_cameras)
	@target_cameras.setter
	def target_cameras(self, value):
		raise AttributeError("Can't set target_cameras on State. Set base_target_cameras instead.")
	
	def base_target_cameras_MS(self, scale):
		if self.__target_cameras_MS is not None:
			return self.__target_cameras_MS[scale]
		else:
			raise AttributeError("State: base_target_cameras_MS not set.")
	def set_base_target_cameras_MS(self, cameras):
		assert  cameras is None or (isinstance(cameras, dict) and all(isinstance(k, int) and \
			isinstance(v, list) and all(isinstance(c, Camera) for c in v) for k,v in cameras.items())), \
			"invalid MS cameras setup, must be dict of lists of cameras: {int: [Camera,]}"
		self.__target_cameras_MS = copy.deepcopy(cameras)
	def target_cameras_MS(self, scale):
		if self.target_mask is not None:
			return [self.base_target_cameras_MS(scale)[_] for _ in self.target_mask]
		else:
			return copy.copy(self.base_target_cameras_MS(scale))
	@property
	def target_cameras_fwd_WS(self):
		cams = self.target_cameras
		pos = [cam.transform.forward_global()[:3] for cam in cams]
		pos = - tf.constant(pos, dtype=tf.float32) #cam looks backwards?
		pos = tf.reshape(pos, (1,len(cams),1,1,3)) #NVHWC
		return pos
	@property
	def target_cameras_pos_WS(self):
		cams = self.target_cameras
		pos = [cam.transform.position_global()[:3] for cam in cams]
		pos = tf.constant(pos, dtype=tf.float32) #cam looks backwards?
		pos = tf.reshape(pos, (1,len(cams),1,1,3)) #NVHWC
		return pos
	
	def get_target_camera_MS_scale_shapes(self):
		if self.__target_cameras_MS is not None:
			return {scale: cams[0].transform.grid_size for scale, cams in self.__target_cameras_MS.items()}
		else:
			raise AttributeError("State: base_target_cameras_MS not set.")
	
	def __make_hull(self, image, eps=1e-5):
		return tf.cast(tf.greater_equal(image, eps), dtype=image.dtype)
	
	@property
	def base_targets_raw(self):
		if self.__targets_raw is not None:
			return self.__targets_raw
		else:
			raise AttributeError("State: targets_raw not set.")
	@base_targets_raw.setter
	def base_targets_raw(self, value):
		assert is_None_or_type(value, ImageSet)
		self.__targets_raw = value
	@property
	def targets_raw(self):
		return self.base_targets_raw.get_images_of_views(self.target_mask)
	def targets_raw_MS(self, scale):
		return self.base_targets_raw.get_base_images_of_views_MS(scale, self.target_mask) if isinstance(self.base_targets_raw, ImageSetMS) else self.base_targets_raw.get_images_of_views_MS(scale, self.target_mask)
	@property
	def target_raw_hulls(self):
		raise NotImplemented("Deprecated, use state.masks")
		return self.__make_hull(self.targets_raw)
	def target_raw_hulls_MS(self, scale):
		raise NotImplemented("Deprecated, use state.masks_MS")
		return self.__make_hull(self.targets_raw_MS(scale))
	
	@property
	def base_targets(self):
		if self.__targets is not None:
			return self.__targets
		else:
			raise AttributeError("State: targets not set.")
	@base_targets.setter
	def base_targets(self, value):
		assert is_None_or_type(value, ImageSet)
		self.__targets = value
	@property
	def targets(self):
		return self.base_targets.get_images_of_views(self.target_mask)
	def targets_MS(self, scale):
		return self.base_targets.get_base_images_of_views_MS(scale, self.target_mask) if isinstance(self.base_targets, ImageSetMS) else self.base_targets.get_images_of_views_MS(scale, self.target_mask)
	@property
	def target_hulls(self):
		raise NotImplemented("Deprecated, use state.masks")
		return self.__make_hull(self.targets)
	def target_hulls_MS(self, scale):
		raise NotImplemented("Deprecated, use state.masks_MS")
		return self.__make_hull(self.targets_MS(scale))
	
	@property
	def base_bkgs(self):
		if self.__bkgs is not None:
			return self.__bkgs
		else:
			raise AttributeError("State: backgrounds not set.")
	@base_bkgs.setter
	def base_bkgs(self, value):
		assert is_None_or_type(value, ImageSet)
		self.__bkgs = value
	@property
	def bkgs(self):
		return self.base_bkgs.get_images_of_views(self.target_mask)
	def bkgs_MS(self, scale):
		return self.base_bkgs.get_base_images_of_views_MS(scale, self.target_mask) if isinstance(self.base_bkgs, ImageSetMS) else self.base_bkgs.get_images_of_views_MS(scale, self.target_mask)
	
	@property
	def has_masks(self):
		return self.__masks is not None
	@property
	def base_masks(self):
		if self.__masks is not None:
			return self.__masks
		else:
			raise AttributeError("State: image masks not set.")
	@base_masks.setter
	def base_masks(self, value):
		assert is_None_or_type(value, ImageSet)
		self.__masks = value
	@property
	def masks(self):
		return self.base_masks.get_images_of_views(self.target_mask)
	def masks_MS(self, scale):
		return self.base_masks.get_base_images_of_views_MS(scale, self.target_mask) if isinstance(self.base_masks, ImageSetMS) else self.base_masks.get_images_of_views_MS(scale, self.target_mask)
	
	@classmethod
	def from_file(cls, path, frame, transform=None, as_var=True, boundary=None, scale_renderer=None, warp_renderer=None, device=None, density_filename="density.npz", velocity_filename="velocity.npz"):
		density = DensityGrid.from_file(os.path.join(path, density_filename), as_var=as_var, scale_renderer=scale_renderer, device=device)
		#density = np.load(os.path.join(path, 'density.npz'), allow_pickle=True)['arr_0'].item(0).numpy() #for accidentally pickled tf.Variable
		velocity = VelocityGrid.from_file(os.path.join(path, velocity_filename), as_var=as_var, \
			boundary=boundary, scale_renderer=scale_renderer, warp_renderer=warp_renderer, device=device)
		state = cls(density, velocity, frame, transform=transform)
		return state
	
	@classmethod
	def from_scalarFlow_file(cls, density_path, velocity_path, frame, transform=None, as_var=True, boundary=None, scale_renderer=None, warp_renderer=None, device=None):
		density = DensityGrid.from_scalarFlow_file(density_path, as_var=as_var, scale_renderer=scale_renderer, device=device)
		#density = np.load(os.path.join(path, 'density.npz'), allow_pickle=True)['arr_0'].item(0).numpy() #for accidentally pickled tf.Variable
		velocity = VelocityGrid.from_scalarFlow_file(velocity_path, as_var=as_var, \
			boundary=boundary, scale_renderer=scale_renderer, warp_renderer=warp_renderer, device=device)
		state = cls(density, velocity, frame, transform=transform)
		return state
	
	def copy(self, as_var=None, device=None):
		s = State(self.density.copy(as_var=as_var, device=device), self.velocity.copy(as_var=as_var, device=device), frame=self.frame, \
			transform=self.transform, targets_raw=self.base_targets_raw, targets=self.base_targets, bkgs=self.base_bkgs)
		s.base_target_cameras = self.base_target_cameras
		s.target_mask = self.target_mask
	#	m = copy.copy(self.__dict__)
	#	del m["_velocity"]
	#	del m["_density"]
	#	del m["prev"]
	#	del m["next"]
	#	for k,v in m.items():
	#		setattr(s,k,v)
		return s
	
	def copy_warped(self, order=1, dt=1.0, frame=None, as_var=None, targets=None, targets_raw=None, bkgs=None, device=None, clamp="NONE"):
		d = self.density.copy_warped(order=order, dt=dt, as_var=as_var, device=device, clamp=clamp)
		v = self.velocity.copy_warped(order=order, dt=dt, as_var=as_var, device=device, clamp=clamp)
		return State(d, v, frame, transform=self.transform, targets=targets, targets_raw=targets_raw, bkgs=bkgs)
	
	def get_density_transform(self):
		if isinstance(self.transform, GridTransform):
		#	density_transform = copy.copy(self.transform)
		#	density_transform.set_data(self.density.d)
			density_transform = self.transform.copy_new_data(self.density.d)
			return density_transform
		else:
			raise TypeError("state.transform is not a GridTransform")
	
	def get_density_transform_MS(self, scale):
		if isinstance(self.transform, GridTransform):
		#	density_transform = copy.copy(self.transform)
		#	density_transform.set_data(self.density.d)
			density_transform = self.transform.copy_new_data(self.density.d_MS(scale))
			return density_transform
		else:
			raise TypeError("state.transform is not a GridTransform")
	
	def get_velocity_transform(self):
		if isinstance(self.transform, GridTransform):
		#	velocity_transform = copy.copy(self.transform)
		#	velocity_transform.set_data(self.velocity.lod_pad)
			return self.transform.copy_new_data(self.velocity.lod_pad)
		else:
			raise TypeError("state.transform is not a GridTransform")
	
	def render_density(self, render_ctx, custom_ops=None, keep_SDF_positions=False, super_sampling=1):
	#	imgs = tf.concat(render_ctx.dens_renderer.render_density(self.get_density_transform(), render_ctx.lights, render_ctx.cameras, cut_alpha=False, monochrome=render_ctx.monochrome), axis=0) #, background=bkg
		if render_ctx.render_SDF:
			if keep_SDF_positions:
				imgs, pos = render_ctx.dens_renderer.render_SDF(self.get_density_transform(), light_list=render_ctx.lights, camera_list=self.target_cameras, cut_alpha=False, monochrome=render_ctx.monochrome, custom_ops=custom_ops, output_positions=True, super_sampling=super_sampling)
				imgs = tf.stack(imgs, axis=1)
				pos = tf.stack(pos, axis=1)
				self.SDF_positions = pos
			else:
				imgs = tf.stack(render_ctx.dens_renderer.render_SDF(self.get_density_transform(), light_list=render_ctx.lights, camera_list=self.target_cameras, cut_alpha=False, monochrome=render_ctx.monochrome, custom_ops=custom_ops, super_sampling=super_sampling), axis=1)
			imgs, t = tf.split(imgs, [3,1], axis=-1)
		else:
			if not super_sampling==1: raise NotImplementedError
			imgs = tf.stack(render_ctx.dens_renderer.render_density(self.get_density_transform(), light_list=render_ctx.lights, camera_list=self.target_cameras, cut_alpha=False, monochrome=render_ctx.monochrome, custom_ops=custom_ops), axis=1) #, background=bkg
			imgs, d = tf.split(imgs, [3,1], axis=-1)
			t = tf.exp(-d)
		self.images = imgs
		self.t = t
	
	def render_density_MS_stack(self, render_ctx, scale_shapes=None, custom_ops=None, keep_SDF_positions=False, super_sampling=1):
		if not self.density.is_MS:
			raise RuntimeError("Frame {} has no MS density to render.".format(self.frame))
		if scale_shapes is None:
			scale_shapes = {scale: shape[1:] for scale, shape in self.get_target_camera_MS_scale_shapes().items()}
		self._density_images_MS = {}
		self._density_images_t_MS = {}
		self._SDF_positions_MS = {}
		for scale, shape in scale_shapes.items():
			if render_ctx.render_SDF:
				if keep_SDF_positions:
					imgs, pos = render_ctx.dens_renderer.render_SDF(self.get_density_transform_MS(scale), light_list=render_ctx.lights, camera_list=self.target_cameras_MS(scale), cut_alpha=False, monochrome=render_ctx.monochrome, custom_ops=custom_ops, output_positions=True, super_sampling=super_sampling)
					imgs = tf.stack(imgs, axis=1)
					pos = tf.stack(pos, axis=1)
					self._SDF_positions_MS[scale] = pos
				else:
					imgs = tf.stack(render_ctx.dens_renderer.render_SDF(self.get_density_transform_MS(scale), light_list=render_ctx.lights, camera_list=self.target_cameras_MS(scale), cut_alpha=False, monochrome=render_ctx.monochrome, custom_ops=custom_ops, super_sampling=super_sampling), axis=1) #, background=bkg
				imgs, t = tf.split(imgs, [3,1], axis=-1)
			else:
				if not super_sampling==1: raise NotImplementedError
				imgs = tf.stack(render_ctx.dens_renderer.render_density(self.get_density_transform_MS(scale), light_list=render_ctx.lights, camera_list=self.target_cameras_MS(scale), cut_alpha=False, monochrome=render_ctx.monochrome, custom_ops=custom_ops), axis=1) #, background=bkg
				imgs, d = tf.split(imgs, [3,1], axis=-1)
				t = tf.exp(-d)
			self._density_images_MS[scale] = imgs
			self._density_images_t_MS[scale] = t
		
		
	def images_MS(self, scale):
		return self._density_images_MS[scale]
	def t_MS(self, scale):
		return self._density_images_t_MS[scale]
	def SDF_positions_MS(self, scale):
		return self._SDF_positions_MS[scale]
	@property
	def image_masks(self):
		if not self.has_density or not self.density.is_SDF:
			raise RuntimeError()
		if self.t is None:
			raise RuntimeError("images have not been rendered.")
		return 1.0 - self.t
	def image_masks_MS(self, scale):
		if not self.has_density or not self.density.is_SDF:
			raise RuntimeError()
		if scale not in self._density_images_t_MS:
			raise RuntimeError("Images for scale %s have not been rendered."%(scale,))
		return 1.0 - self._density_images_t_MS[scale]
	
	def density_advected(self, dt=1.0, order=1, clamp="NONE"):
		return self.density.warped(self.velocity, order=order, dt=dt, clamp=clamp)#self.velocity.warp(self.density, scale_renderer)
	def velocity_advected(self, dt=1.0, order=1, clamp="NONE"):
		return self.velocity.copy_warped(order=order, dt=dt, as_var=False, clamp=clamp)
	
	def rescale_density(self, shape, device=None):
		#density = renderer.resample_grid3D_aligned(self.density, dens_shape)
		self.density = self.density.copy_scaled(shape, device=device)
	def rescale_velocity(self, shape, scale_magnitude=True, device=None):
		self.velocity = self.velocity.copy_scaled(shape, scale_magnitude=scale_magnitude, device=device)
	def rescale(self, dens_shape, vel_shape, device=None):
		rescale_density(self, dens_shape, device=device)
		rescale_velocity(self, vel_shape, device=device)
	
	def var_list(self):
		var_list = []
		if self.has_density:
			var_list += self.density.var_list()
		if self.has_velocity:
			var_list += self.velocity.var_list()
		return var_list
	
	def get_variables(self):
		var_dict = {}
		if self.has_density:
			var_dict.update(self.density.get_variables())
		if self.has_velocity:
			var_dict.update(self.velocity.get_variables())
		return var_dict
	
	def get_output_variables(self, centered=True, staggered=True, include_MS=False, include_residual=False):
		var_dict = {}
		if self.has_density:
			var_dict.update(self.density.get_output_variables(include_MS=include_MS, include_residual=include_residual))
		if self.has_velocity:
			var_dict.update(self.velocity.get_output_variables(centered=centered, staggered=staggered, include_MS=include_MS, include_residual=include_residual))
		return var_dict
		
	
	def stats(self, vel_scale=[1,1,1], mask=None, render_ctx=None, **warp_kwargs):
		target_stats = None
		if render_ctx is not None and getattr(self, "target_cameras", None) is not None:
			target_stats = {}
			self.render_density(render_ctx)
			if getattr(self, "targets_raw") is not None and getattr(self, "bkgs") is not None:
				target_stats["SE_raw"] = tf_tensor_stats(tf.math.squared_difference(self.images + self.bkgs*self.t, self.targets_raw), as_dict=True)
			if getattr(self, "targets") is not None:
				target_stats["SE"] = tf_tensor_stats(tf.math.squared_difference(self.images, self.targets), as_dict=True)
		return self.density.stats(mask=mask, state=self, **warp_kwargs), self.velocity.stats(vel_scale, mask=mask, state=self, **warp_kwargs), target_stats
	
	def stats_target(self, vel_scale=[1,1,1], mask=None, **warp_kwargs):
		return self.density_target.stats(mask=mask, state=self, **warp_kwargs) if self.has_density_target else None, \
			self.velocity_target.stats(vel_scale, mask=mask, state=self, **warp_kwargs) if self.has_velocity_target else None
	
	def save(self, path, suffix=None):
		self.density.save(os.path.join(path, 'density.npz' if suffix is None else 'density_'+suffix+'.npz'))
		self.velocity.save(os.path.join(path, 'velocity.npz' if suffix is None else 'velocity_'+suffix+'.npz'))
		if self.has_density_proxy:
			self.density_proxy.save(os.path.join(path, 'density_proxy.npz' if suffix is None else 'density_proxy_'+suffix+'.npz'))
		if self.has_density_target:
			self.density_target.save(os.path.join(path, 'density_target.npz' if suffix is None else 'density_target_'+suffix+'.npz'))
		
	
	def clear_cache(self):
		self.images = None
		self.t = None
		self.SDF_positions = None
		self._density_images_MS = {}
		self._density_images_t_MS = {}
		self._SDF_positions_MS = {}
		if self.has_density: self.density.clear_cache()
		if self.has_density_proxy: self.density_proxy.clear_cache()
		if self.has_velocity: self.velocity.clear_cache()
	
	@property
	def is_MS(self):
		return (((not self.has_density) or self.density.is_MS) \
			and ((not self.has_velocity) or self.velocity.is_MS))

#import phitest.render.neural_data_structures as NDS

class Sequence:
	def __init__(self, states):
		#raise NotImplementedError("TODO")
		#assert len(states)==len(ctxs)
		self.sequence = [state for state in states]
		#self.contexts = [ctx for ctx in ctxs]
	
	class SequenceIterator:
		def __init__(self, sequence):
			self.seq = sequence
			self.idx = 0
		def __next__(self):
			if self.idx<len(self.seq):
				idx = self.idx
				self.idx +=1
				return self.seq[idx]
			raise StopIteration
	def __iter__(self):
		return self.SequenceIterator(self)
	
	def __getitem__(self, idx):
		return self.sequence[idx]#, self.contexts[idx]
	
	def __len__(self):
		return len(self.sequence)
	
	@classmethod
	def from_file(cls, load_path, frames, transform=None, as_var=True, base_path=None, boundary=None, scale_renderer=None, warp_renderer=None, device=None, density_filename="density.npz", velocity_filename="velocity.npz", frame_callback=lambda idx, frame: None):
		#raise NotImplementedError()
		sequence = []
		prev = None
		for idx, frame in enumerate(frames):
			frame_callback(idx, frame)
			sub_dir = 'frame_{:06d}'.format(frame)
			data_path = os.path.join(load_path, sub_dir)
			state = State.from_file(data_path, frame, transform=transform, as_var=as_var, boundary=boundary, scale_renderer=scale_renderer, warp_renderer=warp_renderer, \
				device=device, density_filename=density_filename, velocity_filename=velocity_filename)
			if base_path is not None:
				state.data_path = os.path.join(base_path, sub_dir)
				os.makedirs(state.data_path, exist_ok=True)
			state.prev = prev
			prev = state
			sequence.append(state)
		for i in range(len(sequence)-1):
			sequence[i].next = sequence[i+1]
		return cls(sequence)
	
	@classmethod
	def from_scalarFlow_file(cls, density_path_mask, velocity_path_mask, frames, transform=None, as_var=True, base_path=None, boundary=None, scale_renderer=None, warp_renderer=None, device=None, vel_frame_offset=1, frame_callback=lambda idx, frame: None):
		sequence = []
		prev = None
		for idx, frame in enumerate(frames):
			frame_callback(idx, frame)
			sub_dir = 'frame_{:06d}'.format(frame)
			#data_path = os.path.join(load_path, sub_dir)
			density_path = density_path_mask.format(frame=frame)
			velocity_path = velocity_path_mask.format(frame=frame+vel_frame_offset)
			state = State.from_scalarFlow_file(density_path, velocity_path, frame=frame, transform=transform, as_var=as_var, boundary=boundary, scale_renderer=scale_renderer, warp_renderer=warp_renderer, device=device)
			if base_path is not None:
				state.data_path = os.path.join(base_path, sub_dir)
				os.makedirs(state.data_path, exist_ok=True)
			state.prev = prev
			prev = state
			sequence.append(state)
		for i in range(len(sequence)-1):
			sequence[i].next = sequence[i+1]
		return cls(sequence)
	
	def copy(self, as_var=None, device=None):
		s = [_.copy(as_var=as_var, device=device) for _ in self]
		for i in range(len(s)):
			if i>0:
				s[i].prev = s[i-1]
			if i<(len(s)-1):
				s[i].next = s[i+1]
		return Sequence(s)
	
	# Defined in neural_data_structures.py to avoid circular dependencies.
	# def set_density_for_neural_globt(self, as_var=False, device=None, dt=1.0, order=1, clamp='NONE'):
		# for i, state in enumerate(self):
			# if i>0:
				# #state.density = state.density.copy(as_var=as_var, device=device)
				# #state.density = state.density.copy_empty(as_var=as_var, device=device)
				# state.density = NDS.WarpedDensityGrid(order=order, dt=dt, clamp=clamp, device=device, scale_renderer=state.density.scale_renderer, is_SDF=state.density.is_SDF)
			# state.density.parent_state = state
		
	
	def copy_for_neural_globt(self, as_var=False, device=None):
		#assert all(isinstance(state, NeuralState) for state in self)
		s = [copy.copy(_) for _ in self]
		for i, state in enumerate(s):
			state.velocity = copy.copy(state.velocity)
			state.velocity.parent_state = state
			
			if i==0:
				state.density = copy.copy(state.density)
			else:
				state.density = state.density.copy(as_var=as_var, device=device)
			state.density.parent_state = state
			
			if i>0:
				state.prev = s[i-1]
			if i<(len(s)-1):
				state.next = s[i+1]
		s = Sequence(s)
		s.clear_cache()
		return s
	
	def copy_from_targets(self, as_var=False, device=None):
		#assert all(isinstance(state, NeuralState) for state in self)
		s = [copy.copy(_) for _ in self]
		for i, state in enumerate(s):
			if state.has_velocity_target:
				state.velocity = state.velocity_target.copy(as_var=as_var, device=device)
				state.velocity.parent_state = state
			else:
				state.velocity = None
			
			if state.has_density_target:
				state.density = state.density_target.copy(as_var=as_var, device=device)
				state.density.parent_state = state
			else:
				state.density = None
			
			
			if i>0:
				state.prev = s[i-1]
			if i<(len(s)-1):
				state.next = s[i+1]
		s = Sequence(s)
		s.clear_cache()
		return s
	
	def copy_from_proxy(self, as_var=False, device=None):
		#assert all(isinstance(state, NeuralState) for state in self)
		s = [copy.copy(_) for _ in self]
		for i, state in enumerate(s):
			if state.has_velocity:
				state.velocity = state.velocity.copy(as_var=as_var, device=device)
				state.velocity.parent_state = state
			else:
				state.velocity = None
			
			if state.has_density_proxy:
				state.density = state.density_proxy #.copy(as_var=as_var, device=device)
				state.density.parent_state = state
			else:
				state.density = None
			
			
			if i>0:
				state.prev = s[i-1]
			if i<(len(s)-1):
				state.next = s[i+1]
		s = Sequence(s)
		s.clear_cache()
		return s
	
	def get_sub_sequence(self, length):
		if length>len(self):
			raise ValueError("Sequence has a length of %d"%(len(self),))
		
		s = [self[i] for i in range(length)]
		s = Sequence(s)
		s.restore_connections()
		
		return s
	
	def restore_connections(self):
		for i, state in enumerate(self):
			if state.has_velocity:
				state.velocity.parent_state = state
			
			if state.has_density:
				state.density.parent_state = state
			
			if state.has_density_proxy:
				state.density_proxy.parent_state = state
			
			if i>0:
				state.prev = self[i-1]
			else:
				state.prev = None
			
			if i<(len(self)-1):
				state.next = self[i+1]
			else:
				state.next = None
	
	def insert_state(self, state, idx):
		raise NotImplementedError("need to set state.next and state.prev")
		self.sequence.insert(state, idx)
		#self.contexts.insert(ctx, idx)
	
	def append_state(self, state):
		raise NotImplementedError("need to set state.next and state.prev")
		self.sequence.append(state)
		#self.contexts.append(ctx)
	
	def start_iteration(self, iteration):
		for state in self:
			ctx.start_iteration(iteration)
	
	def stats(self, vel_scale=[1,1,1], mask=None, **warp_kwargs):
		return [_.stats(vel_scale, mask=mask, state=_, **warp_kwargs) for _ in self]
	
	def save(self, path=None, suffix=None):
		for state in self:
			if path is None and hasattr(state, 'data_path'):
				state.save(state.data_path, suffix)
			else:
				state.save(os.path.join(path, 'frame_{:06d}'.format(state.frame)), suffix)
	
	def clear_cache(self):
		for state in self: state.clear_cache()
	@property
	def advect_steps(self):
		return len(self)-1
	
	def copy_densities_advect_fwd(self, dt=1.0, order=1, clamp='NONE'):
		#
		raise NotImplementedError
		LOG.warning("Sequence.copy_densities_advect_fwd() only creates new density objects.")
		s = []
		for i in range(1, len(self)):
			pass
	
	def densities_advect_fwd(self, dt=1.0, order=1, clamp='NONE', print_progress=None, clear_cache=False):
		raise NotImplementedError("Deprecated, use set_density_for_neural_globt()")
		if clear_cache:
			LOG.warning("densities_advect_fwd with cleared cache.")
		#raise NotImplementedError
		if clamp is None or clamp.upper() not in ['LOCAL', 'GLOBAL']:
			for i in range(1, len(self)):
				self[i].density.assign(self[i-1].density_advected(order=order, dt=dt, clamp=clamp))
				#copy_warped(self, vel_grid, as_var=None, order=1, dt=1.0, device=None, var_name=None, clamp="NONE", trainable=None, restrict_to_hull=None)
				# self[i].density = self[i-1].density.copy_warped(self[i-1].velocity, as_var=False, order=order, dt=dt, clamp=clamp, device=self[i-1].density._device, trainable=False, var_name="DensFwdAdv")
				# self[i].density.parent_state = self[i]
				if print_progress is not None: print_progress.update(desc="Advect density {: 3d}/{: 3d}".format(i, len(self)-1))
				if clear_cache and i>1:
					self[i-2].clear_cache()
		elif clamp.upper()=='LOCAL': #clamp after each step, before the next warp
			for i in range(1, len(self)):
				self[i].density.assign(tf.maximum(self[i-1].density_advected(order=order, dt=dt), 0))
				if print_progress is not None: print_progress.update(desc="Advect density {: 3d}/{: 3d}".format(i, len(self)-1))
		elif clamp.upper()=='GLOBAL': #clamp after all warping
			for i in range(1, len(self)):
				self[i].density.assign(self[i-1].density_advected(order=order, dt=dt))
				if print_progress is not None: print_progress.update(desc="Advect density {: 3d}/{: 3d}".format(i, len(self)-1))
			for i in range(1, len(self)):
				self[i].density.assign(tf.maximum(self[i].density._d, 0))
	def velocities_advect_fwd(self, dt=1.0, order=1, clamp='NONE'):
		#raise NotImplementedError
		for i in range(1, len(self)):
			self[i].velocity.assign(*self[i-1].velocity.warped(order=order, dt=dt, clamp=clamp))
	
	def generate_outputs(self, clear_cache=False):
		if clear_cache:
			self.clear_cache()
		for state in self:
			state.density.d
			state.velocity.centered()