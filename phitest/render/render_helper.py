
import sys, os, logging, re
import numpy as np
import tensorflow as tf

LOG = logging.getLogger("Render Helper")

from .vector import GridShape
import lib.tf_ops

""" helper functions for various rendering


"""

class ImageWriter:
	def __init__(self, renderer, directory, name_proto="img_{id:04d}", format="EXR"):
		self.dir = directory
		os.makedirs(self.dir, exist_ok=True)
		self.name_proto = name_proto
		self.format = format
		self._id = 0
	
	def write_image(images, **kwargs):
		raise NotImplementedError
	
	def write_image_batch(images, **kwargs):
		raise NotImplementedError

def image_volume_slices(data, axis, normalize=True, abs_value=True):
	""" split volume along axis
	
	If a batch dimension (N) is present the returned list
	
	Args:
		data (tf.Tensor): the volume to 'render', with shape DWH, DHWC or NDHWC.
	
	Returns:
		list of tf.Tensor: a list of the volume slices as images, [HWC].
	"""
	data_shape = GridShape.from_tensor(data)
	data = data_shape.normalize_tensor_shape(data)
	
	images = []
	for grid in data:
		if abs_value:
			grid = tf.abs(grid)
		if normalize:
			grid = grid / tf.reduce_max(grid)
		
		images.extend(tf.unstack(grid, axis=axis))
	
	return images

def with_border_planes(data, planes=["Z-"], density=1., width=1, offset=0):
	""" 
	
	Args:
		data (tf.Tensor): the volume to 'render', with shape DWH, DHWC or NDHWC.
		planes (list of str): 
		density (float):
		width (int):
	
	Returns:
		tf.Tensor: data with added border planes with shape NDHWC
	"""
	plane_mask = re.compile("^[XYZxyz][+-]$")
	axes = {'X': 3, 'Y': 2, 'Z': 1}
	zero_pad = (0,0)
	
	grid_shape = GridShape.from_tensor(data)
	data = grid_shape.normalize_tensor_shape(data)
	
	for plane in planes:
		if plane_mask.search(plane) is None:
			raise ValueError
		axis = axes[plane.upper()[0]]
		plane_shape = grid_shape.value
		plane_shape[axis] = width
		padding = [zero_pad] * 5
		padding[axis] = (offset,grid_shape[axis]-width-offset) if plane[1]=="-" else (grid_shape[axis]-width-offset,offset)
		
		data = data + tf.pad(tf.ones(plane_shape, dtype=data.dtype) * density, padding)
		
	return data

def render_back_planes(data_transform, lights, cameras, renderer, plane_transforms):
	""" 
	
	Args:
		data_transform (GridTransform): with tf.tensor in it 'data' attribute
		lights (list of Light): 
		cameras (list of Camera): 
		renderer (Renderer): 
		plane_transforms (list of GridTransform): 
	"""
	
	# use camera with depth 1 to sample plane
	
	

def render_vel_divergence(divergence, transform, cameras, renderer, normalize=True):
	raise NotImplementedError

def render_vel_magnitude_CFL(magnitude, transform, cameras, renderer):
	raise NotImplementedError

def render_vel_abs(vel_centered, transform, cameras, renderer):
	raise NotImplementedError

def render_vel_pos(vel_centered, transform, cameras, renderer):
	raise NotImplementedError

def render_vel_neg(vel_centered, transform, cameras, renderer):
	raise NotImplementedError

