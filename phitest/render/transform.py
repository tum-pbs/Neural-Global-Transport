import copy
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import special_ortho_group
from .serialization import to_dict, from_dict

from .vector import *

from lib.tf_ops import shape_list

class MatrixTransform(object):
	def __init__(self, transform_matrix=None, parent=None, grid_size=None, static=False):
		if transform_matrix is not None:
			self._matrix = transform_matrix
		else:
			self._matrix = self.identity_matrix()
		if parent is not None and not isinstance(parent, MatrixTransform):
			raise TypeError("parent must be a Transform object or None.")
		self.parent = parent
		self.grid_size=grid_size
	@classmethod
	def from_dict(cls, d):
		p = d.pop("parent")
		p = from_dict(p)
		return cls(parent=p, **d)
	
	@classmethod
	def from_lookat(cls, eye, lookat, parent=None):
		pass
	
	@classmethod
	def from_fwd_up_right_pos(cls, fwd, up, right, pos, parent=None):
		mat = np.asarray(
		[[right[0],up[0],fwd[0],pos[0]],
		 [right[1],up[1],fwd[1],pos[1]],
		 [right[2],up[2],fwd[2],pos[2]],
		 [0,0,0,1]],
		dtype=np.float32)
		return cls(mat, parent)
	
	@classmethod
	def from_transform(cls, transform, parent=None):
		raise NotImplementedError()
		
	@staticmethod
	def translation_matrix(translation):
		return np.asarray(
		[[1,0,0,translation[0]],
		 [0,1,0,translation[1]],
		 [0,0,1,translation[2]],
		 [0,0,0,1]],
		dtype=np.float32)
	@staticmethod
	def get_random_rotation(random_state=None):
		#return special_ortho_group.rvs(3) #Rotation.from_matrix(special_ortho_group.rvs(3)) needs scipy>=1.4.0
		return Rotation.random(random_state=random_state).as_euler("xyz", degrees=True)
	@staticmethod
	def rotation_matrix(rotation):
		if isinstance(rotation, Rotation):
			rot = rotation.as_dcm()
		elif shape_list(rotation)==[3,3]:
			rot = np.array(rotation)
		elif shape_list(rotation)==[3,]:
			#rot = Rotation.from_euler('zyx', np.flip(rotation), degrees=True).as_dcm()
			rot = Rotation.from_euler('xyz', rotation, degrees=True).as_dcm()
		else:
			raise ValueError("Can't use %s as rotation"%(rotation,))
		rot = np.pad(rot, (0,1), mode='constant')
		rot[-1,-1]=1
		return rot
	@staticmethod
	def scale_matrix(scale):
		return np.asarray(
		[[scale[0],0,0,0],
		 [0,scale[1],0,0],
		 [0,0,scale[2],0],
		 [0,0,0,1]],
		dtype=np.float32)
	@staticmethod
	def identity_matrix():
		return np.asarray(
		[[1,0,0,0],
		 [0,1,0,0],
		 [0,0,1,0],
		 [0,0,0,1]],
		dtype=np.float32)
	
	def set_parent(self, parent_transform):
		if parent_transform is not None and not isinstance(parent_transform, MatrixTransform):
			raise TypeError("parent must be a Transform object or None.")
		self.parent = parent_transform
	
	def copy_no_data(self):
		return copy.deepcopy(self)
	
	def get_local_transform(self):
		return self._matrix
		
	def position_global(self):
		return self.get_transform_matrix()@np.asarray([0,0,0,1])
	def forward_global(self):
		v = self.get_transform_matrix()@np.asarray([0,0,1,0])
		return v/np.linalg.norm(v)
	def up_global(self):
		v = self.get_transform_matrix()@np.asarray([0,1,0,0])
		return v/np.linalg.norm(v)
	def right_global(self):
		v = self.get_transform_matrix()@np.asarray([1,0,0,0])
		return v/np.linalg.norm(v)
	
	def transform(self, vector):
		if isinstance(vector, (Vector2,Vector3)):
			raise TypeError("use Vector4 for transformations")
		elif isinstance(vector, Vector4):
			return Vector4(self.get_transform_matrix() @ vector.value)
		else:
			return self.get_transform_matrix() @ np.asarray(vector)
	
	def transform_AABB(self, corner_min=[0,0,0], corner_max=[1,1,1], expand_corners=False):
		corners = []
		cmin = Vector3(corner_min)
		cmax = Vector3(corner_max)
		corners.append(self.transform([cmin.x, cmin.y, cmin.z, 1]))
		if expand_corners:
			corners.append(self.transform([cmax.x,cmin.y,cmin.z, 1]))
			corners.append(self.transform([cmin.x,cmax.y,cmin.z, 1]))
			corners.append(self.transform([cmin.x,cmin.y,cmax.z, 1]))
			corners.append(self.transform([cmax.x,cmax.y,cmin.z, 1]))
			corners.append(self.transform([cmax.x,cmin.y,cmax.z, 1]))
			corners.append(self.transform([cmin.x,cmax.y,cmax.z, 1]))
		corners.append(self.transform([cmax.x,cmax.y,cmax.z, 1]))
		return corners
		
		
	def get_transform_matrix(self):
		if self.parent is not None:
			return self.parent.get_transform_matrix() @ self.get_local_transform()
		else:
			return self.get_local_transform()
	def get_inverse_transform(self):
		return np.linalg.inv(self.get_transform_matrix())
	def inverse(self):
		return MatrixTransform(self.get_inverse_transform()) #includes parent transform!
	def is_static(self):
		if not self.static:
			return False
		elif self.parent is not None:
			return self.parent.is_static()
		else:
			return True
	#operators
	def __eq__(self, other):
		return self.get_transform_matrix() == other.get_transform_matrix()
	
	def to_dict(self):
		return {
				"transform_matrix":np.asarray(self._matrix).tolist(),
				"parent":to_dict(self.parent),
				"grid_size":self.grid_size,
			}

class Transform(MatrixTransform):
	def __init__(self, translation=[0,0,0], rotation_deg=[0,0,0], scale=[1,1,1], parent=None, static=False, rotation_rotvec=None, rotation_quat=None):
		self.translation = translation
		if rotation_quat is not None:
			self.rotation_quat = rotation_quat
		elif rotation_rotvec is not None:
			self.rotation_rotvec = rotation_rotvec
		else:
			self.rotation_deg = rotation_deg
		self.scale = scale
		self.parent = parent
	@classmethod
	def from_dict(cls, d):
		p = d.pop("parent")
		p = from_dict(p)
		return cls(parent=p, **d)
	
	def set_translation(self, translation):
		if translation is None:
			self.translation = [0,0,0]
		else:
			assert isinstance(translation, (list, np.ndarray)) and len(translation)==3
			self.translation = translation
	
	@property
	def rotation_deg(self):
		return self._rotation.as_euler("xyz", degrees=True)
	@rotation_deg.setter
	def rotation_deg(self, value):
		assert isinstance(value, (list, np.ndarray)) and len(value)==3
		self._rotation = Rotation.from_euler("xyz", value, degrees=True)
	def add_rotation_deg(self, x=0,y=0,z=0):
		r = self.rotation_deg
		r[0] += x
		r[1] += y
		r[2] += z
		self.rotation_deg = r
	def set_rotation_deg(self, x=None,y=None,z=None):
		r = self.rotation_deg
		if x is not None:
			r[0] = x
		if y is not None:
			r[1] = y
		if z is not None:
			r[2] = z
		self.rotation_deg = r
	@property
	def rotation_rotvec(self):
		return self._rotation.as_rotvec()
	@rotation_rotvec.setter
	def rotation_rotvec(self, value):
		assert isinstance(value, (list, np.ndarray)) and len(value)==3
		self._rotation = Rotation.from_rotvec(value)
	@property
	def rotation_quat(self):
		return self._rotation.as_quat()
	@rotation_quat.setter
	def rotation_quat(self, value):
		assert isinstance(value, (list, np.ndarray)) and len(value)==4
		self._rotation = Rotation.from_quat(value)
	
	def set_rotation_angle(self, rotation_deg):
		if rotation_deg is None:
			self.rotation_deg = [0,0,0]
		else:
			assert isinstance(rotation_deg, (list, np.ndarray)) and len(rotation_deg)==3
			self.rotation_deg = rotation_deg
	
	def set_scale(self, scale):
		if scale is None:
			self.scale = [1,1,1]
		else:
			assert isinstance(scale, (list, np.ndarray)) and len(scale)==3
			self.scale = scale
	
	def set_rotation_quaternion(self, rotation):
		raise NotImplementedError
	
	def translate_local(self, translation):
		raise NotImplementedError
	def rotate_around_local(self, axis, angle_deg):
		raise NotImplementedError
	def scale_local(self, scale):
		raise NotImplementedError
	
	def get_local_transform(self):
		M_scale = Transform.scale_matrix(self.scale)
		M_rot = Transform.rotation_matrix(self._rotation)
		M_trans = Transform.translation_matrix(self.translation)
		return M_trans@(M_rot@M_scale)
	#operators
	def __eq__(self, other):
		return self.get_transform_matrix() == other.get_transform_matrix()
	def __str__(self):
		return '{}: t={}, r={}, s={}; p=({})'.format(type(self).__name__, self.translation, self.rotation_deg, self.scale ,self.parent)
	
	def to_dict(self):
		return {
				"translation":list(self.translation),
				"rotation_quat":list(self.rotation_quat),
				"scale":list(self.scale),
				"parent":to_dict(self.parent),
			}

class GridTransform(Transform):
	def __init__(self, grid_size, translation=[0,0,0], rotation_deg=[0,0,0], scale=[1,1,1], center=False, normalize='NONE', parent=None, static=False, rotation_rotvec=None, rotation_quat=None):
		# center: offset grid s.t. its center is at (0,0,0) is OS
		# normalize: normalize size to (1,1,1) with 1/grid-size
		super().__init__(translation, rotation_deg=rotation_deg, scale=scale, parent=parent, rotation_rotvec=rotation_rotvec, rotation_quat=rotation_quat)
		self.__data=None
		self.grid_size=grid_size
		self.center = center
		self.normalize = normalize
	@classmethod
	def from_dict(cls, d):
		p = d.pop("parent")
		p = from_dict(p)
		return cls(parent=p, **d)
	
	@classmethod
	def from_transform(cls, transform, grid_size, center=False, normalize='NONE'):
		return cls(grid_size, translation=transform.translation, rotation_quat=transform.rotation_quat, scale=transform.scale, center=center, normalize=normalize, parent=transform.parent)
	@classmethod
	def from_grid(cls, grid, translation=[0,0,0], rotation_deg=[0,0,0], scale=[1,1,1], center=False, normalize='NONE', parent=None):
		pass
	@classmethod
	def from_grid_transform(cls, grid, transform, center=False, normalize='NONE'):
		pass
	
	@property
	def grid_size(self):
		return self.__grid_shape.spatial_vector.as_shape.tolist()
	@grid_size.setter
	def grid_size(self, value):
		if self.__data is not None:
			raise ValueError("Can't set grid_size if GridTransform that has assoziated data set.")
		else:
			assert isinstance(value, Int3) or shape_list(value)==[3,]
			self.__grid_shape = GridShape(value)
	
	@property
	def has_data(self):
		return (self.__data is not None)
	
	@property
	def data(self):
		return self.__data
	@data.setter
	def data(self, value):
		self.set_data(value)
	
	def set_data(self, data, format='NDHWC'): #TODO rename: set_grid
		if data is None:
			self.__data = None
			return
		assert isinstance(data, (tf.Tensor, np.ndarray))
		#data_shape = data.get_shape().as_list()
		#self.grid_size = [data_shape[format.index(_)] for _ in 'DHW']
		self.__data = data
		self.__grid_shape = GridShape.from_tensor(self.__data)
	def get_grid(self):
		return self.data
	
	def _data_shape(self):
		if self.__data is not None:
			return GridShape.from_tensor(self.__data)
		else:
			raise ValueError("data is not set")
	
	@property
	def grid_shape(self):
		return self.__grid_shape.copy()
	@grid_shape.setter
	def grid_shape(self, value):
		if self.__data is not None:
			raise ValueError("Can't set grid_size if GridTransform that has assoziated data set.")
		else:
			assert isinstance(value, GridShape)
			self.__grid_shape = value.copy()
	
	def get_grid_size(self): #TODO rename: get_grid_shape
		return np.asarray(self.grid_size)
	
	def get_channel(self):
		if self.__data is not None:
			#return self.data.get_shape().as_list()[-1]
			return self.grid_shape.c
		else: raise ValueError("data is not set")
	
	def get_batch_size(self):
		if self.__data is not None:
			#return self.data.get_shape().as_list()[0]
			return self.grid_shape.n
		else: raise ValueError("data is not set")
	
	def copy_no_data(self):
		gt = copy.copy(self)
		gt.data = None
		return copy.deepcopy(gt)
	
	def copy_new_data(self, data):
		gt = self.copy_no_data()
		gt.set_data(data)
		return gt
	
	def copy_same_data(self):
		return self.copy_new_data(self.data)
	
	def get_local_transform(self):
		size = np.flip(self.get_grid_size()) # shape is zyx, but coordinates are xyz
		M_center = Transform.translation_matrix(-size/2.0)
		if self.normalize=='ALL':
			M_norm_scale = Transform.scale_matrix(1.0/size)
		if self.normalize=='MIN':
			M_norm_scale = Transform.scale_matrix(np.asarray([1.0/np.min(size)]*3, dtype=np.float32))
		if self.normalize=='MAX':
			M_norm_scale = Transform.scale_matrix(np.asarray([1.0/np.max(size)]*3, dtype=np.float32))
		M_scale = Transform.scale_matrix(self.scale)
		M_rot = Transform.rotation_matrix(self._rotation)
		M_trans = Transform.translation_matrix(self.translation)
		M = M_scale@M_center if self.center else M_scale
		M = M_norm_scale@M if self.normalize!='NONE' else M
		return M_trans@(M_rot@M)
	
	def grid_corners_world(self, all_corners=False):
		gs = self.grid_shape
		return self.transform_AABB(corner_max=[gs.x,gs.y,gs.z], expand_corners=all_corners)
	
	def grid_size_world(self):
		gs = self.grid_shape
		dir_x = self.transform(Float4(gs.x,0,0,0)).xyz
		dir_y = self.transform(Float4(0,gs.y,0,0)).xyz
		dir_z = self.transform(Float4(0,0,gs.z,0)).xyz
		return Float3(dir_x.magnitude, dir_y.magnitude, dir_z.magnitude)
	
	def grid_min_world(self):
		return self.transform(Float4(0,0,0,1)).xyz
	
	def grid_max_world(self):
		gs = self.grid_shape
		return self.transform(Float4(gs.xyz,1)).xyz
	
	def cell_size_world(self):
		dir_x = self.transform(Float4(1,0,0,0)).xyz
		dir_y = self.transform(Float4(0,1,0,0)).xyz
		dir_z = self.transform(Float4(0,0,1,0)).xyz
		return Float3(dir_x.magnitude, dir_y.magnitude, dir_z.magnitude)
	
	def __eq__(self, other):
		if np.any(np.not_equal(self.get_transform_matrix(), other.get_transform_matrix())): return False
		if np.any(np.not_equal(self.get_grid_size(), other.get_grid_size())): return False
		#if self.get_grid() != other.get_grid(): return False
		return True
	def __str__(self):
		return '{}: {}{}, t={}{}, r={}, s={}{}; p=({})'.format(type(self).__name__, self.grid_size, 'Y' if self.has_data else 'N', self.translation, 'C' if self.center else '', \
			self.rotation_deg, self.scale, self.normalize if self.normalize!='NONE' else '' ,self.parent)
	
	def to_dict(self):
		return {
				"translation":list(self.translation),
				"rotation_quat":list(self.rotation_quat),
				"scale":list(self.scale),
				"center":bool(self.center),
				"normalize":str(self.normalize),
				"grid_size":list(self.grid_size),
				"parent":to_dict(self.parent),
			}