import os, re
import tensorflow as tf
import numpy as np
#import numbers

import logging
log = logging.getLogger('TFops')
log.setLevel(logging.DEBUG)

from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.framework import tensor_shape
import scipy.signal

#https://stackoverflow.com/questions/14267555/find-the-smallest-power-of-2-greater-than-n-in-python
def next_pow_two(x):
	return 1<<(x-1).bit_length()

def next_div_by(x, div):
	return ((x + div - 1) // div) * div

def shape_list(arr):
	if isinstance(arr, (tf.Tensor, tf.Variable)):
		return arr.get_shape().as_list()
	elif isinstance(arr, np.ndarray):
		return list(arr.shape)
	else:
		try:
			return list(np.shape(arr))
		except:
			return None
	return None

def spatial_shape_list(arr):
	shape = shape_list(arr)
	assert len(shape)==5
	return shape[-4:-1]

def has_rank(arr, rank):
	return len(shape_list(arr))==rank
def has_shape(arr, shape):
	arr_shape = shape_list(arr)
	return len(arr_shape)==len(shape) and all(arr_dim==dim or dim==-1 or (dim is None) for arr_dim,dim in zip(arr_shape, shape)) #np.all(np.equal(shape_list(arr), shape))

spacial_shape_list = spatial_shape_list #compatibility...


def splits_of_size(channels, sizes=[1,2,4]):
	sizes.sort(reverse=True)
	splits = []
	rem_channels = channels
	while rem_channels>0:
		sizes_idx = 0 # = sizes[0]
		while sizes[sizes_idx]>rem_channels:
			if sizes_idx>=len(sizes):
				raise RuntimeError("Failed to split dimension of size {} into sizes {}.".format(channels, sizes))
			sizes_idx +=1
		splits.append(sizes[sizes_idx])
		rem_channels -= sizes[sizes_idx]
	return splits

def tf_split_to_size(tensor, sizes=[1,2,4], axis=-1):
	shape = shape_list(tensor)
	dim_size = shape[axis]
	if dim_size in sizes:
		return [tensor]
	splits = tf.split(tensor, splits_of_size(dim_size), axis=axis)
	return splits

def reshape_array_format(arr, in_fmt, out_fmt='NDHWC'):
	shape = shape_list(arr)
	if len(shape)!=len(in_fmt):
		raise ValueError("Array shape {} does not math input format '{}'".format(shape, in_fmt))
	if in_fmt==out_fmt:
		return arr
	squeeze = [in_fmt.index(_) for _ in in_fmt if _ not in out_fmt]
	if squeeze:
		if isinstance(arr, np.ndarray):
			arr = np.squeeze(arr, squeeze)
		elif isinstance(arr, tf.Tensor):
			arr = tf.squeeze(arr, squeeze)
	expand = [out_fmt.index(_) for _ in out_fmt if _ not in in_fmt]
	for axis in expand:
		if isinstance(arr, np.ndarray):
			arr = np.expand_dims(arr, axis)
		elif isinstance(arr, tf.Tensor):
			arr = tf.expand_dims(arr, axis)
	return arr


def grad_log(tensor, msg, print_fn=log.info):
	
	@tf.custom_gradient
	def fwd(x):
		print_fn("FWD: "+msg)
		def grad(y):
			print_fn("GRAD: "+msg)
			return y
		return tf.identity(x), grad
	
	return fwd(tensor)

def tf_to_dict(obj):
	"""Convert a tf.Tensor or np.ndarray or any class to a JSON-serializable type."""
	if obj is None or isinstance(obj, (int, bool, float, str, list, tuple, dict)):
		return obj
	if isinstance(obj, (np.ndarray, np.number)):
		# alternative: save ndarray as .npz and put path here. might need base path
		return obj.tolist()
	if isinstance(obj, (tf.Tensor, tf.Variable)):
		# alternative: save ndarray as .npz and put path here. might need base path
		return obj.numpy().tolist()
	d = {
		"__class__":obj.__class__.__name__,
		"__module__":obj.__module__
	}
	if hasattr(obj, "to_dict"):
		d.update(obj.to_dict())
	else:
		d.update(obj.__dict__)
	return d

def tf_pad_to_shape(tensor, shape, alignment="CENTER", allow_larger_dims=False, **pad_args):
	"""pads 'tensor' to have at least size 'shape'
	
	Args:
		tensor (tf.Tensor): the tensor to pad
		shape (Iterable): the target shape, must have the same rank as the tensor shape. elements may be -1 to have no padding.
		alignment (str):
			"CENTER": equally pad before and after, pad less before in case of odd padding.
			"BEFORE": pad after the data (data comes before the padding).
			"AFTER" : pad before the data (data comes after the padding).
		allow_larger_dims (bool): ignore tensor dimensions that are larger than the target dimension (no padding). otherwise raise a ValueError
		**pad_args: kwargs passed to tf.pad (mode, name, constant_value)
	
	Returns:
		tf.Tensor: The padded input tensor
	
	Raises:
		ValueError:
			If shape is not compatible with the shape of tensor
			Or alignment is not one of 'CENTER', 'BEFORE', 'AFTER'
			Or any dimension of the input tensor is larger than its target (if allow_larger_dims==False)
	"""
	tensor_shape = shape_list(tensor)
	if len(tensor_shape) != len(shape): raise ValueError("Tensor shape %s is not compatible with target shape %s"%(tensor_shape, shape))
	alignment = alignment.upper()
	if not alignment in ["CENTER", "BEFORE", "AFTER"]: raise ValueError("Unknown alignment '%s', use 'CENTER', 'BEFORE', 'AFTER'"%alignment)
	
	diff = []
	for i, (ts, s) in enumerate(zip(tensor_shape, shape)):
		if ts>s and (not allow_larger_dims) and s!=-1:
			raise ValueError("Tensor size %d of dimension %d is larger than target size %d"%(ts, i, s))
		elif ts>=s or s==-1:
			diff.append(0)
		else: #ts<s
			diff.append(s-ts)
	
	paddings = []
	for i, d in enumerate(diff):
		if alignment=="CENTER":
			paddings.append((d//2, d-(d//2)))
		elif alignment=="BEFORE":
			paddings.append((0, d))
		elif alignment=="AFTER":
			paddings.append((d, 0))
			
	return tf.pad(tensor=tensor, paddings=paddings, **pad_args)

def tf_pad_to_next_pow_two(data, pad_axes=(0,1,2)):
	shape = shape_list(data)
	rank = len(shape)
	pad_axes = [_%rank for _ in pad_axes]
	paddings = []
	for axis in range(len(shape)):
		if axis in pad_axes:
			dim=shape[axis]
			dif = next_pow_two(dim) - dim
			if dif%2==0:
				paddings.append([dif//2,dif//2])
			else:
				paddings.append([dif//2 +1,dif//2])
		else:
			paddings.append([0,0])
	return tf.pad(data, paddings)


def tf_pad_to_next_div_by(data, div, pad_axes=(0,1,2), return_paddings=False):
	shape = shape_list(data)
	rank = len(shape)
	pad_axes = [_%rank for _ in pad_axes]
	paddings = []
	for axis in range(len(shape)):
		if axis in pad_axes:
			dim=shape[axis]
			dif = next_div_by(dim, div) - dim
			if dif%2==0:
				paddings.append([dif//2,dif//2])
			else:
				paddings.append([dif//2 +1,dif//2])
		else:
			paddings.append([0,0])
	data = tf.pad(data, paddings)
	if return_paddings:
		return data, paddings
	else:
		return data

#https://stackoverflow.com/questions/45254554/tensorflow-same-padding-calculation
def getSamePadding(shape, kernel, stride):
	#dim = len(shape)
	#kernel = conv_utils.normalize_tupel(kernel, dim, 'kernel_size')
	#stride = onv_utils.normalize_tupel(stride, dim, 'stride')
	pad = []
	for dim, k, s in zip(shape, kernel, stride):
		if dim is None:
			pad.append([0, 0])
		else:
			out_dim = int(np.ceil(dim/s))
			pad_total = max((out_dim - 1)*s + k - dim, 0)
			pad_before = int(pad_total//2)
			pad_after = pad_total - pad_before
			pad.append([pad_before, pad_after])
	return pad

def tf_color_gradient(data, c1, c2, vmin=0, vmax=1):
	data_norm = (data - vmin)/(vmax - vmin) # -> [0,1]
	return c1 + (c2-c1)*data_norm # lerp

def tf_element_transfer_func(data, grads):
	channel = shape_list(grads[0][1])[-1]
	r = tf.zeros(shape_list(data)[:-1] +[channel])
	for g1, g2 in zip(grads[:-1], grads[1:]):
		grad = tf_color_gradient(data, g1[1], g2[1], g1[0], g2[0])
		condition = tf.broadcast_to(tf.logical_and(tf.greater_equal(data, g1[0]), tf.less_equal(data,g2[0])), grad.get_shape())
		r = tf.where(condition, grad, r)
	return r

def tf_shift(tensor, shift, axis):
	tensor_shape = shape_list(tensor)
	tensor_rank = len(tensor_shape)
	if axis<0:
		axis += tensor_rank
	if axis<0 or tensor_rank<=axis:
		raise ValueError("Tensor axis out of bounds.")
	if shift==0:
		return tf.identity(tensor)
	elif shift<=-tensor_shape[axis] or shift>=tensor_shape[axis]:
		return tf.zeros_like(tensor)
	elif shift<0:
		shift = -shift
		pad = [(0,0)]*axis + [(0,shift)] + [(0,0)]*max(tensor_rank-axis-1, 0)
		return tf.pad(tf.split(tensor, [shift, tensor_shape[axis]-shift], axis=axis)[1], pad)
	elif shift>0:
		pad = [(0,0)]*axis + [(shift,0)] + [(0,0)]*max(tensor_rank-axis-1, 0)
		return tf.pad(tf.split(tensor, [tensor_shape[axis]-shift, shift], axis=axis)[0], pad)
		

def tf_reduce_dot(x,y, axis=None, keepdims=False, name=None):
	return tf.reduce_sum(tf.multiply(x,y), axis=axis, keepdims=keepdims, name=name)

def tf_reduce_var(input_tensor, axis=None, keepdims=False, name=None):
	m = tf.reduce_mean(input_tensor, axis=axis, keepdims=keepdims)
	return tf.reduce_mean(tf.square(tf.abs(input_tensor - m)), axis=axis, keepdims=keepdims)

def tf_reduce_std(input_tensor, axis=None, keepdims=False, name=None):
	return tf.sqrt(tf_reduce_var(input_tensor, axis=axis, keepdims=keepdims, name=name))

def tf_None_to_const(input_tensor, constant=0, dtype=tf.float32):
	return tf.constant(0, dtype=dtype) if input_tensor is None else input_tensor

@tf.custom_gradient
def tf_ReLU_lin_grad(input_tensor):
	y = tf.relu(input_tensor)
	def grad(dy):
		return tf.identity(dy)
	return y, grad

@tf.custom_gradient
def tf_grad_NaN_to_num(tensor):
	y = tf.identity(tensor)
	def grad(dy):
		return tf.where(tf.is_nan(dy), tf.zeros_like(dy), dy)
	return y, grad

@tf.custom_gradient
def tf_norm2(x, axis=None, keepdims=False, name=None):
	
	y = tf.norm(x, ord='euclidean', axis=axis, keepdims=keepdims, name=name)
	def grad(dy, variables=None):
		var_grads = []
		if variables is not None:
			log.warning("tf_norm2() called with variables: %s", [(_.name, shape_list(_)) for _ in variables])
			var_grads = [None for _ in variables]
		if keepdims:
			return (tf.div_no_nan(x, y)*dy, var_grads)
		else:
			return (tf.div_no_nan(x, tf.expand_dims(y, axis))*tf.expand_dims(dy, axis), var_grads)
		
	return y, grad

def tf_angle_between(x, y, axis, mode="RAD", keepdims=False, name=None, undefined="ORTHOGONAL", norm_eps=1e-12, constant=1):
	''' Angle between vectors (axis) in radians (mode=RAD), degrees (DEG) or as cosine similarity (COS).
		https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249
		undefined:
			ORTHOGONAL: returns 90 degree if any of the two vectors has norm 0
			CONSTANT: returns constant if any of the two vector norms is smaller than norm_eps
	'''
#	norm_x = tf_norm2(x, axis=axis, keepdims=keepdims)
#	norm_y = tf_norm2(y, axis=axis, keepdims=keepdims)
#	c = tf.div_no_nan(tf_reduce_dot(x,y, axis=axis, keepdims=keepdims), norm_x * norm_y)
	x = tf.math.l2_normalize(x, axis=axis, epsilon=norm_eps)
	y = tf.math.l2_normalize(y, axis=axis, epsilon=norm_eps)
	c = tf_reduce_dot(x,y, axis=axis, keepdims=keepdims)
	
	if mode.upper()=="COS":
		result = c
	else:
		a = tf.acos(tf.clip_by_value(c, -1, 1))
		if mode.upper()=="RAD":
			result = a
		elif mode.upper()=="DEG":
			result = a * (180 / np.pi)
		else:
			raise ValueError("Unknown mode '%s' for angle between vectors."%mode)
	
	if undefined.upper()=="ORTHOGONAL":
		pass
	elif undefined.upper()=="CONSTANT":
		norm_x = tf_norm2(x, axis=axis, keepdims=keepdims)
		norm_y = tf_norm2(y, axis=axis, keepdims=keepdims)
		cond = tf.logical_or(tf.less(norm_x, norm_eps), tf.less(norm_y, norm_eps))
		constant = tf.cast(tf.broadcast_to(constant, result.get_shape()), dtype=result.dtype)
		result = tf.where(cond, constant, result)
	else:
		raise ValueError("Unknown mode '%s' for undefined angles."%undefined)
		
	return result

def tf_cosine_similarity(x, y, axis, keepdims=False):
	return tf_angle_between(x, y, axis, keepdims=keepdims, mode="COS", undefined="ORTHOGONAL")

def tf_PSNR(x, y, max_val=1.0, axes=[-3,-2,-1]):
	''' or use tf.image.psnr for 2D tensors'''
	rrmse = tf.rsqrt(tf.reduce_mean(tf.squared_difference(x,y)), axes)
	return tf.log(max_val*rrmse) * (20./tf.log(10.)) #tf.constant(8.685889638065035, dtype=tf.float32)

def tf_tensor_stats(data, scalar=False, as_dict=False):
	data_abs = tf.abs(data)
	
	d = {
		"min":tf.reduce_min(data),
		"max":tf.reduce_max(data),
		"mean":tf.reduce_mean(data),
		"std":tf_reduce_std(data),
		"abs":{
			"min":tf.reduce_min(data_abs),
			"max":tf.reduce_max(data_abs),
			"mean":tf.reduce_mean(data_abs),
			"std":tf_reduce_std(data_abs),
		},
	}
	
	if scalar:
		def to_scalar(inp):
			if isinstance(inp, tf.Tensor):
				return inp.numpy()
			if isinstance(inp, dict):
				for k in inp:
					inp[k] = to_scalar(inp[k])
			return inp
		d = to_scalar(d)
	
	if as_dict:
		return d
	else:
		return d['max'], d['min'], d['mean'], d['abs']['mean'] #tf.reduce_max(data), tf.reduce_min(data), tf.reduce_mean(data), tf.reduce_mean(tf.abs(data))

def tf_print_stats(data, name, log=None):
	max, min, mean, abs_mean = tf_tensor_stats(data)
	if log is None:
		print('{} stats: min {:.010e}, max {:.010e}, mean {:.010e}, abs-mean {:.010e}'.format(name, min, max, mean, abs_mean))
	else:
		log.info('{} stats: min {:.010e}, max {:.010e}, mean {:.010e}, abs-mean {:.010e}'.format(name, min, max, mean, abs_mean))
	return max, min, mean

def tf_log_barrier_ext(x, t):
	"""https://arxiv.org/abs/1904.04205
	Args:
		x (tf.Tensor): 
		t (scalar): scale or strength
	"""
	t_inv = 1./t
	t_inv_2 = t_inv*t_inv
	v1 = -t_inv*tf.log(-x)
	v2 = t*x - (t_inv*tf.log(t_inv_2) + t_inv)
	cond = (x<=(-t_inv2))
	return tf.where(cond, v1, v2)

def tf_log_barrier_ext_sq(x, t):
	"""https://arxiv.org/abs/1904.04205
	with quadratic extension
	
	Args:
		x (tf.Tensor): 
		t (scalar): scale or strength
	"""
	t_inv = 1./t
	t_inv_2 = t_inv*t_inv
	t_half = t*0.5
	
	v1 = -t_inv*tf.log(-x)
	
	v2_x = (x + (1 + t_inv_2))
	v2 = t_half*(v2_x*v2_x) - (t_inv*tf.log(t_inv_2) + t_half)
	
	cond = (x<=(-t_inv2))
	return tf.where(cond, v1, v2)

def tf_image_resize_mip(images, size, mip_bias=0.5, **resize_kwargs):
	'''Resize the image using nearest mip-mapping (if down-sampliing) and tf.image.resize_images
	
	N.B.: should switch to TF 2 image.resize(antialias=True) eventually
		https://hackernoon.com/how-tensorflows-tf-image-resize-stole-60-days-of-my-life-aba5eb093f35
		https://github.com/tensorflow/tensorflow/issues/6720
	'''
#	images_shape = tf.cast(tf.shape(images), dtype=tf.float32)
#	target_shape = tf.cast(tf.convert_to_tensor(size), dtype=tf.float32)
	#print(images_shape)
#	relative_step = tf.reduce_max(tf.math.divide(images_shape[-3:-1], target_shape))
#	lod_raw = tf.math.divide(tf.log(relative_step), tf.log(tf.convert_to_tensor(2., dtype=tf.float32)))
#	lod = tf.cast(tf.floor(tf.maximum(0., tf.math.add(lod_raw, mip_bias))), tf.int32)
#	def create_mip():
#		window_size = tf.bitwise.left_shift(1, lod) #(1 << lod) ##tf.pow(2., lod)
#		window = (1,window_size,window_size,1)
#		return tf.nn.avg_pool(images, window, window, padding="VALID", data_format="NHWC")
		
#	images = tf.cond(tf.greater(lod, tf.convert_to_tensor(0, dtype=tf.int32)), create_mip, lambda: tf.identity(images))
	
	images_shape = np.asarray(shape_list(images), dtype=np.float32)
	target_shape = np.asarray(size, dtype=np.float32)
	relative_step = np.amax(images_shape[-3:-1]/target_shape)
	lod_raw = np.log2(relative_step)
	lod = np.floor(np.maximum(0., lod_raw + mip_bias)).astype(np.int32)
	if lod > 0.:
		window_size = np.left_shift(1, lod)
		window = (1,window_size,window_size,1)
		images = tf.nn.avg_pool(images, window, window, padding="SAME", data_format="NHWC")
	
	return tf.image.resize_images(images, size, **resize_kwargs)

def gaussian_1dkernel(size=5, sig=1.):
	"""
	Returns a 1D Gaussian kernel array with side length size and a sigma of sig
	"""
	gkern1d = tf.constant(scipy.signal.gaussian(size, std=sig), dtype=tf.float32)
	return (gkern1d/tf.reduce_sum(gkern1d))

def gaussian_2dkernel(size=5, sig=1.):
	"""
	Returns a 2D Gaussian kernel array with side length size and a sigma of sig
	"""
	gkern1d = tf.constant(scipy.signal.gaussian(size, std=sig), dtype=tf.float32)
	gkern2d = tf.einsum('i,j->ij',gkern1d, gkern1d)
	return (gkern2d/tf.reduce_sum(gkern2d))

def gaussian_3dkernel(size=5, sig=1.):
	"""
	Returns a 3D Gaussian kernel array with side length size and a sigma of sig
	"""
	gkern1d = tf.constant(scipy.signal.gaussian(size, std=sig), dtype=tf.float32)
	gkern3d = tf.einsum('i,j,k->ijk',gkern1d, gkern1d, gkern1d)
	return (gkern3d/tf.reduce_sum(gkern3d))
	
def tf_data_gaussDown2D(data, sigma = 1.5, stride=4, channel=3, padding='VALID'):
	"""
	tensorflow version of the 2D down-scaling by 4 with Gaussian blur
	sigma: the sigma used for Gaussian blur
	return: down-scaled data
	"""
	k_w = 1 + 2 * int(sigma * 3.0)
	gau_k = gaussian_2dkernel(k_w, sigma)
	gau_0 = tf.zeros_like(gau_k)
	gau_list = [[gau_k if i==o else gau_0 for i in range(channel)] for o in range(channel)]
	#	[gau_k, gau_0, gau_0],
	#	[gau_0, gau_k, gau_0],
	#	[gau_0, gau_0, gau_k]] # only works for RGB images!
	gau_wei = tf.transpose(gau_list, [2,3,0,1])
	
	fix_gkern = tf.constant( gau_wei, shape = [k_w, k_w, channel, channel], name='gauss_blurWeights', dtype=tf.float32)
	# shape [batch_size, crop_h, crop_w, 3]
	cur_data = tf.nn.conv2d(data, fix_gkern, strides=[1,stride,stride,1], padding=padding, name='gauss_downsample')

	return cur_data
	
def tf_data_gaussDown3D(data, sigma = 1.5, stride=4, channel=3, padding='VALID'):
	"""
	tensorflow version of the 3D down-scaling by 4 with Gaussian blur
	sigma: the sigma used for Gaussian blur
	return: down-scaled data
	"""
	k_w = 1 + 2 * int(sigma * 3.0)
	gau_k = gaussian_3dkernel(k_w, sigma)
	gau_0 = tf.zeros_like(gau_k)
	gau_list = [[gau_k if i==o else gau_0 for i in range(channel)] for o in range(channel)]
	gau_wei = tf.transpose(gau_list, [2,3,4,0,1])
	
	fix_gkern = tf.constant(gau_wei, shape = [k_w, k_w, k_w, channel, channel], name='gauss_blurWeights', dtype=tf.float32)
	# shape [batch_size, crop_h, crop_w, 3]
	cur_data = tf.nn.conv3d(data, fix_gkern, strides=[1,stride,stride,stride,1], padding=padding, name='gauss_downsample')

	return cur_data

def _tf_laplace_kernel_3d(neighbours=1):
	if neighbours==0:
		laplace_kernel = np.zeros((3,3,3), dtype=np.float32)
	elif neighbours==1:
		laplace_kernel = np.asarray(
		[	[[ 0, 0, 0],
			 [ 0,-1, 0],
			 [ 0, 0, 0]],
			[[ 0,-1, 0],
			 [-1, 6,-1],
			 [ 0,-1, 0]],
			[[ 0, 0, 0],
			 [ 0,-1, 0],
			 [ 0, 0, 0]],
		], dtype=np.float32)/6.
	elif neighbours==2:
		laplace_kernel = np.asarray(
		[	[[ 0,-1, 0],
			 [-1,-2,-1],
			 [ 0,-1, 0]],
			[[-1,-2,-1],
			 [-2,24,-2],
			 [-1,-2,-1]],
			[[ 0,-1, 0],
			 [-1,-2,-1],
			 [ 0,-1, 0]],
		], dtype=np.float32)/24.
	elif neighbours==3:
		# https://en.wikipedia.org/wiki/Discrete_Laplace_operator#Image_Processing
		laplace_kernel = np.asarray(
		[	[[-2,-3,-2],
			 [-3,-6,-3],
			 [-2,-3,-2]],
			[[-3,-6,-3],
			 [-6,88,-6],
			 [-3,-6,-3]],
			[[-2,-3,-2],
			 [-3,-6,-3],
			 [-2,-3,-2]],
		], dtype=np.float32)/88.
	return tf.constant(laplace_kernel, dtype=tf.float32)


''' old n=3
laplace_kernel = np.asarray(
[	[[-1,-2,-1],
	 [-2,-4,-2],
	 [-1,-2,-1]],
	[[-2,-4,-2],
	 [-4,56,-4],
	 [-2,-4,-2]],
	[[-1,-2,-1],
	 [-2,-4,-2],
	 [-1,-2,-1]],
], dtype=np.float32)/56.
'''


def tf_laplace_filter_3d(inp, neighbours=1, padding='SAME', name='gauss_filter'):
	with tf.name_scope(name):
		channel = inp.get_shape().as_list()[-1]
		if channel != 1:
			raise ValueError('input channel must be 1')
		laplace_kernel = _tf_laplace_kernel_3d(neighbours)
		laplace_kernel = laplace_kernel[:,:,:,tf.newaxis, tf.newaxis]
		#laplace_kernel = tf.concat([gauss_kernel]*channel, axis=3)
		return tf.nn.conv3d(inp, laplace_kernel, strides=[1,1,1,1,1], padding=padding)

#inspired by: https://openreview.net/pdf?id=HJlnC1rKPB and https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf
# EXPERIMENTAL: TODO test
def tf_channel_attention(Q, K, V, q_shape=None, k_shape=None, v_shape=None, scale=True):
	'''
	Q, K, V: Tensors with shape NHWC
	_shape: 2 elements that must fit Channel in the respective input Tensor. if None: [Channel, 1]
	returns: Tensor with shape NHWC where Channel are q_shape[0]*v_shape[1]
	'''
	def make_shape(T, mat_shape=None):
		t_shape = T.get_shape()
		channel = t_shape[-1]
		if mat_shape is None:
			mat_shape = tf.constant([channel, 1], dtype=tf.int)
		else:
			mat_shape = tf.constant(mat_shape, dtype=tf.int)
			assert tf.get_shape().as_list()==[2]
		assert tf.reduce_prod(mat_shape)==channel
		return tf.concat([t_shape[:-1], mat_shape])
	
	q_shape = make_shape(Q, q_shape) #NHW Q Dk
	k_shape = make_shape(K, k_shape) #NHW K Dk
	v_shape = make_shape(V, v_shape) #NHW V Dv
	assert q_shape[-1]==k_shape[-1] #Dk==DK
	assert k_shape[-2]==v_shape[-2] #K==V
	Q = tf.reshape(Q, q_shape)
	K = tf.reshape(K, k_shape)
	V = tf.reshape(V, v_shape)
	
	A = tf.linalg.matmul(Q, K, transpose_b=True) #NHW Q K
	if scale:
		scale = 1./tf.sqrt(k_shape[-1]) #scale with 1/sqrt(Dk)
		A = A*scale
	A = tf.nn.softmax(A, axis=-1) #norm over K
	
	R = tf.linalg.matmul(A, V) #NHW Q Dv
	r_shape = R.get_shape()
	r_shape = tf.concat([r_shape[:-2], [r_shape[-2]*r_shape[-1]]])
	R = tf.reshape(R, r_shape) #NHWC with C=Q*Dv
	return R

def tf_build_density_grid(sim_transform, density_scale=0.06, cube_thickness=8, sphere_radius=40, sphere_thickness=6):
	#density = 0.06
	#border = 1
	coords = tf.meshgrid(tf.range(sim_transform.grid_size[0], dtype=tf.float32), tf.range(sim_transform.grid_size[1], dtype=tf.float32), tf.range(sim_transform.grid_size[2], dtype=tf.float32), indexing='ij')
	coords = tf.transpose(coords, (1,2,3,0))
	coords_centerd = coords - np.asarray(sim_transform.grid_size)/2.0
	dist_center = tf.norm(coords_centerd, axis=-1, keepdims=True)
	density = tf.zeros(list(sim_transform.grid_size)+[1], dtype=tf.float32)
	ones = tf.ones(list(sim_transform.grid_size)+[1], dtype=tf.float32)
	
	is_in_sphere = tf.logical_and(dist_center<sphere_radius, dist_center>(sphere_radius-sphere_thickness))
	#print(is_in_sphere.get_shape().as_list())
	density = tf.where(is_in_sphere, ones, density)
	#print(tf.reduce_mean(density))
	
	is_in_cube = tf.reduce_sum(tf.cast(tf.logical_or(coords<cube_thickness, coords>(np.asarray(sim_transform.grid_size)-1 -cube_thickness)), dtype=tf.int8), axis=-1, keepdims=True)
	density = tf.where(is_in_cube>1, ones, density)
	#print(tf.reduce_mean(density))
	
	sim_data = np.expand_dims(np.expand_dims(density, 0),-1) #NDHWC
	sim_data *=density_scale
	sim_data = tf.constant(sim_data, dtype=tf.float32)
	sim_transform.set_data(sim_data)
	return sim_data


#https://stackoverflow.com/questions/49189496/can-symmetrically-paddding-be-done-in-convolution-layers-in-keras
# mirror (reflect) padding for convolutions, does not add a conv layer, instead use before valid-padding conv layer.
class MirrorPadND(tf.keras.layers.Layer):
	#def __init__(self, dim, kernel, stride, **kwargs):
	def __init__(self, dim=2, kernel=4, stride=1, **kwargs): #compatibility values, TODO: remove
		self.kernel = conv_utils.normalize_tuple(kernel, dim, 'kernel_size')
		self.stride = conv_utils.normalize_tuple(stride, dim, 'stride')
		self.dim = dim
		super(MirrorPadND, self).__init__(**kwargs)
	
	def build(self, input_shape):
		super(MirrorPadND, self).build(input_shape)
		
	def call(self, inputs):
		inputs_shape = inputs.get_shape().as_list()
		dim = len(inputs_shape)-2
		shape = inputs.get_shape().as_list()[-(dim+1):-1]
		pad = [[0,0]] + getSamePadding(shape, self.kernel, self.stride) + [[0,0]]
		padded = tf.pad(inputs, pad, 'REFLECT')
		return padded
	
	def compute_output_shape(self, input_shape):
		#print("MirrorPadND input shape:", input_shape)
	#	spatial_size = [int(_) for _ in input_shape[1:-1]]
	#	print("MirrorPadND spatial size:", spatial_size)
	#	shape = list(input_shape[:1]) + list(np.asarray(spatial_size) + np.sum(getSamePadding(spatial_size, self.kernel, self.stride), axis=-1)) + list(input_shape[-1:])
	#	print("MirrorPadND output shape:", shape)
		
		spatial_input_shape = input_shape[1:-1]
		#print("MirrorPadND spatial size:", spatial_input_shape)
		shape = [input_shape[1]]
		for dim, k, s in zip(spatial_input_shape, self.kernel, self.stride):
			if dim.__int__() is None:
				shape.append(None)
			else:
				dim = int(dim)
				shape.append(dim + np.sum(getSamePadding([dim], [k], [s])))
		shape.append(input_shape[-1])
		shape = tensor_shape.TensorShape(shape)
		#print("MirrorPadND output shape:", shape)
		return shape
	
	def get_config(self):
		config = {
			'dim':self.dim,
			'kernel':self.kernel,
			'stride':self.stride,
		}
		base_config = super(MirrorPadND, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

class LayerNormalization(tf.keras.layers.Layer):
	# Layer Normalization: https://arxiv.org/abs/1607.06450
	# Understanding and Improving Layer Normalization: https://arxiv.org/abs/1911.07013
	def __init__(self, axes=[-1], eps=1e-5, name="LayerNorm", **layerargs):
		self.cnf = {
			"axes":axes, "eps":eps,
		}
		super().__init__(name=name, **layerargs)
		log.debug("Create LayerNormalization '%s': axes=%s, eps=%f", name, axes, eps)
		self.axes = axes
		self.eps = eps
	
	def call(self, inputs):
		log.debug("LayerNormalization '%s' called with input %s", self.name, shape_list(inputs))
		
		mean, var = tf.nn.moments(inputs, self.axes, keep_dims=True)
		inv_std = tf.math.rsqrt(var + self.eps)
		
		return (inputs - mean ) * inv_std
	
	def compute_output_shape(self, input_shapes):
		if isinstance(input_shapes, list):
			assert len(input_shapes)==1
			input_shapes = input_shapes[0]
		return tf.TensorShape(input_shapes)
	
	def get_config(self):
		config = super().get_config()
		config.update(self.cnf)
		return config

class AdaptiveNormalization(tf.keras.layers.Layer):
	# Understanding and Improving Layer Normalization: https://arxiv.org/abs/1911.07013
	def __init__(self, axes=[-1], eps=1e-5, k=0.1, name="LayerNorm", **layerargs):
		self.cnf = {
			"axes":axes, "eps":eps, "k":k,
		}
		super().__init__(name=name, **layerargs)
		log.debug("Create AdaptiveNormalization '%s': axes=%s, k=%f, eps=%f", name, axes, k, eps)
		self.axes = axes
		self.eps = eps
		self.k = k
	
	def call(self, inputs):
		log.debug("AdaptiveNormalization '%s' called with input %s", self.name, shape_list(inputs))
		# https://github.com/lancopku/AdaNorm/blob/546faea0c3297061d743482d690ccd7d51f1ac38/machine%20translation/fairseq/modules/layer_norm.py l.63
		mean, var = tf.nn.moments(inputs, self.axes, keep_dims=True)
		inv_std = tf.math.rsqrt(var + self.eps)
		inputs = inputs - mean
		mean = tf.math.reduce_mean(inputs, self.axes, keep_dims=True)
		graNorm = tf.stop_gradient(k* (inputs - mean) * inv_std)
		input_norm = (inputs - inputs * graNorm) * inv_std
		
		return input_norm
	
	def compute_output_shape(self, input_shapes):
		if isinstance(input_shapes, list):
			assert len(input_shapes)==1
			input_shapes = input_shapes[0]
		return tf.TensorShape(input_shapes)
	
	def get_config(self):
		config = super().get_config()
		config.update(self.cnf)
		return config

def is_obj_tuple_equal(a, t):
	assert isinstance(t, tuple)
	if not isinstance(a, (list, tuple, np.ndarray)):
		a = [a]*len(t)
	for i in range(len(t)):
		if a[i]!=t[i]: return False
	return True

_int_types = ["dim", "filters", "kernel_size", "stride"]
_float_types = ["alpha"]
_str_types = ["normalization", "padding", "activation"]
def parse_str_config_types(str_dict, default=None):
	out_dict = {}
	for k,v in str_dict.items():
		if v is None:
			continue
		elif k in _int_types:
			out_dict[k] = int(v)
		elif k in _float_types:
			out_dict[k] = float(v)
		elif k in _str_types:
			out_dict[k] = str(v)
		else:
			if default is None:
				raise ValueError("Unknown string config argument '%s'"%(k,))
			else:
				out_dict[k] = default(v)
	return out_dict

c_mask_str = r"^(?:C\:)?(?P<dim>\d+)D_(?P<filters>\d+)(?:-(?P<kernel_size>\d+))?(?:-s(?P<stride>\d+))?(?:_(?P<normalization>LN|LNL))?(?:_(?P<padding>ZERO|MIRROR))?(?:_(?P<activation>relu|lrelu|gelu)(?:-(?P<alpha>\d+\.\d*))?)?$"
c_mask = re.compile(c_mask_str)
class ConvLayerND(tf.keras.layers.Layer):
	def __init__(self, dim, filters, kernel_size=3, stride=1, activation='none', alpha=0.2, padding='ZERO', normalization="NONE", name="convND", **layerargs):
		self.cnf = {
			"dim":dim, "filters":filters, "kernel_size":kernel_size, "stride":stride,
			"activation":activation, "alpha":alpha, "padding":padding, "normalization":normalization,
		}
		self.dim = dim
		self.filters = filters
		self.stride = stride if isinstance(stride, (tuple, list)) else [stride]*dim
		self.padding = padding
		super().__init__(name=name, **layerargs)
		log.debug("Create %dD ConvLayerND '%s': f=%d, ks=%d, s=%d, act=%s (alpha=%f), pad=%s, norm=%s", dim, name, filters, kernel_size, stride, activation, alpha, padding, normalization)
		
		self.up_layer = None
		if stride<0: #negative stride means up-scaling
			stride = abs(stride)
			# these are simple NN interpolations
			if dim==1:
				self.up_layer = tf.keras.layers.UpSampling1D(size=stride, name=name + "_up")
			if dim==2:
				self.up_layer = tf.keras.layers.UpSampling2D(size=stride, data_format="channels_last", name=name + "_up")
			if dim==3:
				self.up_layer = tf.keras.layers.UpSampling3D(size=stride, data_format="channels_last", name=name + "_up")
			stride = 1
		
		self.mpad_layer = None
		if padding.upper()=='MIRROR':
			self.mpad_layer = MirrorPadND(dim, kernel_size, stride, name=name + "_Mpad")
			padding = 'valid'
		elif padding.upper()=='ZERO':
			padding = 'same'
		elif padding.upper()=='NONE':
			padding = 'valid'
		else:
			raise ValueError('Unsupported padding: {}'.format(padding))
		
		self.conv_layer = None
		if dim==1:
			self.conv_layer = tf.keras.layers.Conv1D(filters, kernel_size, stride, padding=padding, name=name + "_conv")
		elif dim==2:
			self.conv_layer = tf.keras.layers.Conv2D(filters, kernel_size, stride, padding=padding, name=name + "_conv")
		elif dim==3:
			self.conv_layer = tf.keras.layers.Conv3D(filters, kernel_size, stride, padding=padding, name=name + "_conv")
		else:
			raise ValueError('Unsupported dimension: {}'.format(self.dim))
		
		self.activation_layer = None
		if activation=='relu':
			self.activation_layer = tf.keras.layers.ReLU()
		elif activation=='lrelu':
			self.activation_layer = tf.keras.layers.LeakyReLU(alpha=alpha)
		elif activation=='gelu':
			self.activation_layer = GELU()
		elif activation=='none':
			self.activation_layer = None
		else:
			raise ValueError
		
		self.late_norm = False
		self.normalization_layer = None
		if normalization in ["LAYER", "LAYER_LATE", "LN", "LNL"] :
			self.normalization_layer = LayerNormalization(axes = [-a for a in range(1, dim+2)], name=name+"_norm")
			if normalization in ["LAYER_LATE", "LNL"]:
				self.late_norm = True
		elif normalization=="NONE":
			pass
		else:
			raise ValueError
	
	@classmethod
	def from_string(cls, s, name="convND", **kwargs):
		match = c_mask.search(s)
		if match is not None:
			#d = {k:((int(v) if k not in ["alpha"] else float(v)) if k not in ["normalization", "padding", "activation"] else v) for k,v in match.groupdict().items() if v is not None}
			d = parse_str_config_types(match.groupdict())
			kwargs.update(d)
			return cls(name=name, **kwargs)
		else:
			raise ValueError("invalid ConvLayerND string config: %s"%(s,))
	
	def call(self, inputs):
		log.debug("ConvLayer%dD '%s' called with input %s", self.cnf["dim"], self.name, shape_list(inputs))
		x = inputs
		
		if self.up_layer is not None:
			x = self.up_layer(x)
			
		if self.mpad_layer is not None:
			x = self.mpad_layer(x)
		
		x = self.conv_layer(x)
		
		if self.normalization_layer is not None and not self.late_norm:
			x = self.normalization_layer(x)
		
		if self.activation_layer is not None:
			x = self.activation_layer(x)
		
		if self.normalization_layer is not None and self.late_norm:
			x = self.normalization_layer(x)
		
		return x
	
	
	def num_output_channels(self):
		return self.filters
	
#	def compute_output_shape(self, input_shapes):
#		if isinstance(input_shapes, list):
#			assert len(input_shapes)==1
#			input_shapes = input_shapes[0]
#		#log.debug("compute_output_shape for '%s' with input shape %s (%s)", self.name, input_shapes, input_shapes.__class__.__name__)
#		output_shape = []
#		output_shape.append(input_shapes[0])
#		for idx in range(1, self.dim+1):
#			if input_shapes[idx] is None:
#				output_shape.append(None)
#			elif self.padding in ["ZERO", "MIRROR"]:
#				output_shape.append(input_shapes[idx]//self.stride[idx-1])
#			else:
#				#raise NotImplementedError("ConvLayerND.compute_output_shape not implemented for current configuration.")
#				return super().compute_output_shape(input_shapes)
#		output_shape.append(self.filters)
#		output_shape = tf.TensorShape(output_shape)
#		#log.debug("compute_output_shape for '%s': %s", self.name, output_shape)
#		log.debug("compute_output_shape for ConvLayerND '%s' with input shape %s: %s", self.name, input_shapes, output_shape)
#		return output_shape
	
	@property
	def variables(self):
		return self.conv_layer.variables
	@property
	def trainable_variables(self):
		return self.conv_layer.trainable_variables
	@property
	def non_trainable_variables(self):
		return self.conv_layer.non_trainable_variables
	
	
	@property
	def trainable_weights(self):
		return self.conv_layer.trainable_weights
	@property
	def non_trainable_weights(self):
		return self.conv_layer.non_trainable_weights
	@property
	def weights(self):
		return self.conv_layer.weights
	
	def get_config(self):
		config = super().get_config()
		config.update(self.cnf)
		return config

def ConvLayer(in_layer, dim, filters, kernel_size, stride=1, activation='none', alpha=0.2, padding='ZERO', **kwargs):
	return_layer = kwargs.get("return_layer", False)
	if return_layer:
		del kwargs["return_layer"]
	
	x = in_layer
	
	if stride<0: #negative stride means up-scaling
		stride = abs(stride)
		# these are simple NN interpolations
		if dim==1:
			x = tf.keras.layers.UpSampling1D(size=stride)(x)
		if dim==2:
			x = tf.keras.layers.UpSampling2D(size=stride, data_format="channels_last")(x)
		if dim==3:
			x = tf.keras.layers.UpSampling3D(size=stride, data_format="channels_last")(x)
		
		stride = 1
	
	if padding.upper()=='MIRROR':
		x = MirrorPadND(dim, kernel_size, stride)(x)
		padding = 'valid'
	elif padding.upper()=='ZERO':
		padding = 'same'
	elif padding.upper()=='NONE':
		padding = 'valid'
	else:
		raise ValueError('Unsupported padding: {}'.format(padding))
	
	if not "conv_layer" in kwargs:
		if dim==1:
			L = tf.keras.layers.Conv1D(filters, kernel_size, stride, padding=padding, **kwargs)
		elif dim==2:
			L = tf.keras.layers.Conv2D(filters, kernel_size, stride, padding=padding, **kwargs)
		elif dim==3:
			L = tf.keras.layers.Conv3D(filters, kernel_size, stride, padding=padding, **kwargs)
		else:
			raise ValueError('Unsupported dimension: {}'.format(self.dim))
	else: #for layer/weight reuse/sharing
		L = kwargs["conv_layer"]
		assert dim==L.rank
		assert L.filters==filters
		assert is_obj_tuple_equal(kernel_size, L.kernel_size)
		assert is_obj_tuple_equal(stride, L.strides)
		assert padding==L.padding
	x = L(x)
	
	if activation=='relu':
		x = tf.keras.layers.ReLU()(x)
	elif activation=='lrelu':
		x = tf.keras.layers.LeakyReLU(alpha=alpha)(x)
	
	if return_layer:
		return x, L
	else:
		return x

# mid_filters[-mid0_kernel_size]_out_filters[-mid1_kernel_size][skip_kernel_size[-stride]], all integer
rb_mask_str = r"^(?:RB\:)?(?P<dim>\d+)D_(?P<mid_filters>\d+)(?:-(?P<mid0_kernel_size>\d+))?_(?P<out_filters>\d+)(?:-(?P<mid1_kernel_size>\d+))?(?:_s(?P<skip_kernel_size>\d+)(?:-(?P<stride>\d+))?)?(?:_(?P<normalization>LN|LNL))?(?:_(?P<padding>ZERO|MIRROR))?(?:_(?P<activation>relu|lrelu|gelu)(?:-(?P<alpha>\d+\.\d*))?)?$" #TODO: 
rb_mask = re.compile(rb_mask_str)

class ResBlock(tf.keras.layers.Layer):
	def __init__(self, dim, mid_filters, out_filters, mid0_kernel_size=3, mid1_kernel_size=3, skip_kernel_size=1, stride=1, \
			mid_activation=None, out_activation=None, mid_alpha=None, out_alpha=None, activation="relu", alpha=0.2, padding="ZERO", normalization="NONE", name="ResBlock", **layerargs):
		self.cnf = {
			"dim":dim, "mid_filters":mid_filters, "out_filters":out_filters,
			"mid0_kernel_size":mid0_kernel_size, "mid1_kernel_size":mid1_kernel_size, "skip_kernel_size":skip_kernel_size,
			"stride":stride, "padding":padding, "normalization":normalization,
			"mid_activation":mid_activation, "out_activation":out_activation, "mid_alpha":mid_alpha, "out_alpha":out_alpha,
			"activation":activation, "alpha":alpha,
		}
		self.dim = dim
		self.stride = stride if isinstance(stride, (list, tuple)) else [stride]*self.dim
		assert padding in ["ZERO", "MIRROR"]
		self.padding = padding
		super().__init__(name=name, **layerargs)
		if mid_activation is None: mid_activation = activation
		if mid_alpha is None: mid_alpha = alpha
		if out_activation is None: out_activation = activation
		if out_alpha is None: out_alpha = alpha
		log.debug("Create %dD ResBlock '%s': f=(%d,%d), ks=(%d,%d,%d), s=%d, act=(%s,%s) (alpha=(%f,%f)), pad=%s, norm=%s", dim, name, mid_filters, out_filters, \
			mid0_kernel_size, mid1_kernel_size, skip_kernel_size, stride, mid_activation, out_activation, mid_alpha, out_alpha, padding, normalization)
		
		self.mid0_layer = ConvLayerND(dim=dim, filters=mid_filters, kernel_size=mid0_kernel_size, stride=stride, activation=mid_activation, alpha=mid_alpha, \
			padding=padding, normalization=normalization, name=name+"_m0conv")
		
		if False:
			log.warning("ResBlock '%s' has no normalization for skip and 2nd conv.", name)
			normalization = "NONE"
		
		self.mid1_layer = ConvLayerND(dim=dim, filters=out_filters, kernel_size=mid1_kernel_size, stride=1, activation="none", normalization=normalization, padding=padding, name=name+"_m1conv")
		
		if False:
			log.warning("ResBlock '%s' has no normalization for skip", name)
			normalization = "NONE"
		
		self.skip_layer = None
		if skip_kernel_size:
			self.skip_layer = ConvLayerND(dim=dim, filters=out_filters, kernel_size=skip_kernel_size, stride=stride, activation="none", normalization=normalization, padding=padding, name=name+"_sconv")
		else:
			assert stride==1, "stride must be 1 when using blank skip"
		
		self.merge_layer = tf.keras.layers.Add()
		self.activation_layer = None
		if out_activation=='relu':
			self.activation_layer = tf.keras.layers.ReLU()
		elif out_activation=='lrelu':
			self.activation_layer = tf.keras.layers.LeakyReLU(alpha=out_alpha)
		elif out_activation=='gelu':
			self.activation_layer = GELU()
		elif activation=='none':
			self.activation_layer = None
		else:
			raise ValueError
	
	@classmethod
	def from_string(cls, s, name="ResBlock", **kwargs):
		match = rb_mask.search(s)
		if match is not None:
			#d = {k:int(v) if k not in ["normalization", "padding"] else v for k,v in match.groupdict().items() if v is not None}
			d = parse_str_config_types(match.groupdict(), default=int)
			#log.info("ResBlock from string %s: %s", s, d)
			kwargs.update(d)
			return cls(name=name, **kwargs)
		else:
			raise ValueError("invalid ResBlock string config: %s"%(s,))
	
	def call(self, inputs):
		log.debug("ResBlock '%s' called with input %s", self.name, shape_list(inputs))
		m = self.mid0_layer(inputs)
		m = self.mid1_layer(m)
		
		s = self.skip_layer(inputs) if self.skip_layer is not None else inputs
		
		x = self.merge_layer([s,m])
		
		if self.activation_layer is not None:
			x = self.activation_layer(x)
		
		return x
	
	def num_output_channels(self):
		return self.cnf["out_filters"]
	
#	def compute_output_shape(self, input_shapes):
#		if isinstance(input_shapes, list):
#			assert len(input_shapes)==1
#			input_shapes = input_shapes[0]
#		#log.debug("compute_output_shape for '%s' with input shape %s (%s)", self.name, input_shapes, input_shapes.__class__.__name__)
#		output_shape = []
#		output_shape.append(input_shapes[0])
#		for idx in range(1, self.dim+1):
#			if input_shapes[idx] is None:
#				output_shape.append(None)
#			elif self.padding in ["ZERO", "MIRROR"]:
#				output_shape.append(input_shapes[idx]//self.stride[idx-1])
#			else:
#				#raise NotImplementedError("ConvLayerND.compute_output_shape not implemented for current configuration.")
#				return super().compute_output_shape(input_shapes)
#		output_shape.append(self.num_output_channels())
#		output_shape = tf.TensorShape(output_shape)
#		#log.debug("compute_output_shape for '%s': %s", self.name, output_shape)
#		log.debug("compute_output_shape for ResBlock '%s' with input shape %s: %s", self.name, input_shapes, output_shape)
#		return output_shape
	
	@property
	def variables(self):
		return self.mid0_layer.variables + self.mid1_layer.variables + \
			(self.skip_layer.variables if self.skip_layer is not None else [])
	@property
	def trainable_variables(self):
		return self.mid0_layer.trainable_variables + self.mid1_layer.trainable_variables + \
			(self.skip_layer.trainable_variables if self.skip_layer is not None else [])
	@property
	def non_trainable_variables(self):
		return self.mid0_layer.non_trainable_variables + self.mid1_layer.non_trainable_variables + \
			(self.skip_layer.non_trainable_variables if self.skip_layer is not None else [])
	
	@property
	def trainable_weights(self):
		return self.mid0_layer.trainable_weights + self.mid1_layer.trainable_weights + \
			(self.skip_layer.trainable_weights if self.skip_layer is not None else [])
	@property
	def non_trainable_weights(self):
		return self.mid0_layer.non_trainable_weights + self.mid1_layer.non_trainable_weights + \
			(self.skip_layer.non_trainable_weights if self.skip_layer is not None else [])
	@property
	def weights(self):
		return self.trainable_weights + self.non_trainable_weights
	
	def get_config(self):
		config = super().get_config()
		config.update(self.cnf)
		return config

dcb_mask_str = r"^(?:DCB\:)?(?P<dim>\d+)D(?:_(?:\d+)(?:-(?:\d+))?)+$" #TODO (?:_(?P<normalization>LN|LNL))?(?:_(?P<padding>ZERO|MIRROR))?
dcb_mask = re.compile(dcb_mask_str)
class DenseConvBlock(tf.keras.layers.Layer):
	default_kernel_size = 3
	def __init__(self, dim, filters=[1], kernel_size=[default_kernel_size], activation='relu', alpha=0.2, padding='ZERO', normalization="NONE", name="DenseConvBlock", merge_mode="CONCAT", **layerargs):
		self.cnf = {
			"dim":dim, "filters":filters, "kernel_size":kernel_size, "padding":padding, "normalization":normalization,
			"activation":activation, "alpha":alpha, "merge_mode":merge_mode,
		}
		super().__init__(name=name, **layerargs)
		log.debug("Create %dD DenseConvBlock '%s': f=%s, ks=%s, act=%s (alpha=%f), merge=%s, pad=%s, norm=%s", dim, name, filters, kernel_size, \
			activation, alpha, merge_mode, padding, normalization)
		
		self.conv_layers = [ConvLayerND(dim=dim, filters=f, kernel_size=f, stride=1, activation=activation, alpha=alpha, padding=padding, normalization=normalization, name=name + "_conv%d"%idx) for idx, (f, k) in enumerate(zip(filters, kernel_size))]
		
		assert merge_mode in ["CONCAT", "SUM"]
		self.merge_layers = [tf.keras.layers.Concatenate(axis=-1) if merge_mode=="CONCAT" else tf.keras.layers.Add() for i in range(len(self.conv_layers))]
		
		self.stride = 1
	
	@classmethod
	def from_string(cls, s, name="DenseConvBlock", **kwargs):
		match = dcb_mask.search(s)
		if match is not None:
			layers = [_.split("-") for _ in s.split("_")[1:]]
			filters, kernels = zip(*[(int(L[0]), int(L[1]) if len(L)>1 else cls.default_kernel_size) for L in layers])
			d = {k:int(v) if k not in ["normalization", "padding"] else v for k,v in match.groupdict().items() if v is not None}
			kwargs.update(d)
			return cls(name=name, filters=filters, kernel_size=kernels, **kwargs)
		else:
			raise ValueError("invalid DenseConvBlock string config: %s"%(s,))
	
	def call(self, inputs):
		log.debug("DenseConvBlock '%s' called with input %s", self.name, shape_list(inputs))
		x = inputs
		for idx, (layer, merge) in enumerate(zip(self.conv_layers, self.merge_layers)):
			x = merge([x, layer(x)])
		return x
	
	def num_output_channels(self):
		raise NotImplementedError
		return self.cnf["out_filters"]
	
	@property
	def variables(self):
		variables = []
		for (layer, merge) in zip(self.conv_layers, self.merge_layers): variables += layer.variables + merge.variables
		return variables
	@property
	def trainable_variables(self):
		variables = []
		for (layer, merge) in zip(self.conv_layers, self.merge_layers): variables += layer.trainable_variables + merge.trainable_variables
		return variables
	@property
	def non_trainable_variables(self):
		variables = []
		for (layer, merge) in zip(self.conv_layers, self.merge_layers): variables += layer.non_trainable_variables + merge.non_trainable_variables
		return variables
	
	@property
	def trainable_weights(self):
		variables = []
		for (layer, merge) in zip(self.conv_layers, self.merge_layers): variables += layer.trainable_weights + merge.trainable_weights
		return variables
	@property
	def non_trainable_weights(self):
		variables = []
		for (layer, merge) in zip(self.conv_layers, self.merge_layers): variables += layer.non_trainable_weights + merge.non_trainable_weights
		return variables
	@property
	def weights(self):
		return self.trainable_weights + self.non_trainable_weights
	
	def get_config(self):
		config = super().get_config()
		config.update(self.cnf)
		return config



class GELU(tf.keras.layers.Layer):
	# Gaussian Error Linear Units (GELUs): https://arxiv.org/abs/1606.08415
	def __init__(self, name="GELU", **layerargs):
		self.cnf = {
		}
		super().__init__(name=name, **layerargs)
		log.debug("Create GELU '%s'", name)
		self.__rsqrt2 = tf.math.rsqrt(tf.constant(2, dtype=tf.float32)) #1/sqrt(2)
	
	def call(self, inputs):
		log.debug("GELU '%s' called with input %s", self.name, shape_list(inputs))
		x = inputs
		
		return x*0.5*(1 + tf.math.erf(x*self.__rsqrt2))
	
	def compute_output_shape(self, input_shapes):
		if isinstance(input_shapes, list):
			assert len(input_shapes)==1
			input_shapes = input_shapes[0]
		return tf.TensorShape(input_shapes)
	
	def get_config(self):
		config = super().get_config()
		config.update(self.cnf)
		return config

# RNBX:<in_filters>[-<in_kernel_size>]_<mid_filters>_<out_filters>[_s<skip_kernel_size>[-<stride>]], all integer
rnxb_mask_str = r"^(?:RNXB\:)?(?P<dim>\d+)D_(?P<in_filters>\d+)(?:-(?P<in_kernel_size>\d+))?_(?P<mid_filters>\d+)_(?P<out_filters>\d+)(?:_s(?P<skip_kernel_size>\d+)(?:-(?P<stride>\d+))?)?(?:_(?P<normalization>LN|LNL))?(?:_(?P<padding>ZERO|MIRROR))?(?:_(?P<activation>relu|lrelu|gelu)(?:-(?P<alpha>\d+\.\d*))?)?$" #TODO: 
rnxb_mask = re.compile(rnxb_mask_str)

class ConvNeXtBlock(tf.keras.layers.Layer):
	def __init__(self, dim, in_filters, mid_filters, out_filters, in_kernel_size=7, skip_kernel_size=1, stride=1, \
			activation="gelu", alpha=0.2, padding="ZERO", normalization="LAYER", name="ConvNeXtBlock", **layerargs):
		self.cnf = {
			"dim":dim, "in_filters":in_filters, "mid_filters":mid_filters, "out_filters":out_filters,
			"in_kernel_size":in_kernel_size, "skip_kernel_size":skip_kernel_size,
			"stride":stride, "padding":padding, "normalization":normalization,
			"activation":activation, "alpha":alpha,
		}
		self.dim = dim
		self.stride = stride if isinstance(stride, (list, tuple)) else [stride]*self.dim
		assert padding in ["ZERO", "MIRROR"]
		self.padding = padding
		super().__init__(name=name, **layerargs)
		#log.debug("Create %dD ConvNeXtBlock '%s': f=(%d,%d), ks=(%d,%d,%d), s=%d, act=(%s,%s) (alpha=(%f,%f)), pad=%s, norm=%s", dim, name, mid_filters, out_filters, \
		#	mid0_kernel_size, mid1_kernel_size, skip_kernel_size, stride, mid_activation, out_activation, mid_alpha, out_alpha, padding, normalization)
		
		#raise NotImplementedError("Depthwise 3D convolution required.")
		self.in_layer  = ConvLayerND(dim=dim, filters=in_filters, kernel_size=in_kernel_size, stride=stride, activation="none", padding=padding, normalization=normalization, name=name+"_m0conv")
		self.mid_layer = ConvLayerND(dim=dim, filters=mid_filters, kernel_size=1, stride=1, activation=activation, alpha=alpha, normalization="NONE", padding=padding, name=name+"_m1conv")
		self.out_layer = ConvLayerND(dim=dim, filters=out_filters, kernel_size=1, stride=1, activation="none", normalization="NONE", padding=padding, name=name+"_m2conv")
		
		self.skip_layer = None
		if skip_kernel_size:
			self.skip_layer = ConvLayerND(dim=dim, filters=out_filters, kernel_size=skip_kernel_size, stride=stride, activation="none", normalization="NONE", padding=padding, name=name+"_sconv")
		else:
			assert stride==1, "stride must be 1 when using blank skip"
		
		self.merge_layer = tf.keras.layers.Add()
		
		self.activation_layer = None
	
	@classmethod
	def from_string(cls, s, name="ConvNeXtBlock", **kwargs):
		match = rnxb_mask.search(s)
		if match is not None:
			#d = {k:int(v) if k not in ["normalization", "padding"] else v for k,v in match.groupdict().items() if v is not None}
			d = parse_str_config_types(match.groupdict(), default=int)
			#log.info("ResBlock from string %s: %s", s, d)
			kwargs.update(d)
			return cls(name=name, **kwargs)
		else:
			raise ValueError("invalid ConvNeXtBlock string config: %s"%(s,))
	
	def call(self, inputs):
		log.debug("ConvNeXtBlock '%s' called with input %s", self.name, shape_list(inputs))
		m = self.in_layer(inputs)
		m = self.mid_layer(m)
		m = self.out_layer(m)
		
		s = self.skip_layer(inputs) if self.skip_layer is not None else inputs
		
		x = self.merge_layer([s,m])
		
		if self.activation_layer is not None:
			x = self.activation_layer(x)
		
		return x
	
	def num_output_channels(self):
		return self.cnf["out_filters"]
	
#	def compute_output_shape(self, input_shapes):
#		if isinstance(input_shapes, list):
#			assert len(input_shapes)==1
#			input_shapes = input_shapes[0]
#		#log.debug("compute_output_shape for '%s' with input shape %s (%s)", self.name, input_shapes, input_shapes.__class__.__name__)
#		output_shape = []
#		output_shape.append(input_shapes[0])
#		for idx in range(1, self.dim+1):
#			if input_shapes[idx] is None:
#				output_shape.append(None)
#			elif self.padding in ["ZERO", "MIRROR"]:
#				output_shape.append(input_shapes[idx]//self.stride[idx-1])
#			else:
#				#raise NotImplementedError("ConvLayerND.compute_output_shape not implemented for current configuration.")
#				return super().compute_output_shape(input_shapes)
#		output_shape.append(self.num_output_channels())
#		output_shape = tf.TensorShape(output_shape)
#		#log.debug("compute_output_shape for '%s': %s", self.name, output_shape)
#		log.debug("compute_output_shape for ResBlock '%s' with input shape %s: %s", self.name, input_shapes, output_shape)
#		return output_shape
	
	@property
	def variables(self):
		return self.in_layer.variables + self.mid_layer.variables + self.out_layer.variables + \
			(self.skip_layer.variables if self.skip_layer is not None else [])
	@property
	def trainable_variables(self):
		return self.in_layer.trainable_variables + self.mid_layer.trainable_variables + self.out_layer.trainable_variables + \
			(self.skip_layer.trainable_variables if self.skip_layer is not None else [])
	@property
	def non_trainable_variables(self):
		return self.in_layer.non_trainable_variables + self.mid_layer.non_trainable_variables + self.out_layer.non_trainable_variables + \
			(self.skip_layer.non_trainable_variables if self.skip_layer is not None else [])
	
	@property
	def trainable_weights(self):
		return self.in_layer.trainable_weights + self.mid_layer.trainable_weights + self.out_layer.trainable_weights + \
			(self.skip_layer.trainable_weights if self.skip_layer is not None else [])
	@property
	def non_trainable_weights(self):
		return self.in_layer.non_trainable_weights + self.mid_layer.non_trainable_weights + self.out_layer.non_trainable_weights + \
			(self.skip_layer.non_trainable_weights if self.skip_layer is not None else [])
	@property
	def weights(self):
		return self.trainable_weights + self.non_trainable_weights
	
	def get_config(self):
		config = super().get_config()
		config.update(self.cnf)
		return config

def layer_from_string(s, **kwargs):
	log.debug("Create layer from string '%s', with args %s", s, kwargs)
	if s.startswith("C:"):
		return ConvLayerND.from_string(s, **kwargs)
	if s.startswith("RB:"):
		return ResBlock.from_string(s, **kwargs)
	if s.startswith("DCB:"):
		return DenseConvBlock.from_string(s, **kwargs)
	if s.startswith("RNXB:"):
		return ConvNeXtBlock.from_string(s, **kwargs)
	raise ValueError("Falied to determine layer type from '%s'"%(s,))


def discriminator(input_shape=[None,None,3], layers=[16]*4, kernel_size=3, strides=1, final_fc=None, activation='relu', alpha=0.2, noise_std=0.0, padding='ZERO'):
	dim = len(input_shape)-1
	num_layers = len(layers)
	if np.isscalar(strides):
		strides = [strides]*num_layers
	x = tf.keras.layers.Input(shape=input_shape, name='disc_input')
	inputs = x
	if noise_std>0:
		x = tf.keras.layers.GaussianNoise(stddev=noise_std)(x)
	x = ConvLayer(x, dim, 8, kernel_size, 1, activation, alpha, padding=padding, name='disc_in_conv')
	for filters, stride in zip(layers, strides):
		x = ConvLayer(x, dim, filters, kernel_size, stride, activation, alpha, padding=padding)
	if final_fc is not None:
		assert isinstance(final_fc, list)
		x = tf.keras.layers.Flatten()(x)
		for i, filters in enumerate(final_fc):
			x = tf.keras.layers.Dense(filters, name='disc_dense%d'%(i,))(x)
		x = tf.keras.layers.Dense(1, name='disc_output')(x)
	else:
		x = ConvLayer(x, dim, 1, kernel_size, padding=padding, name='disc_output')
	outputs = x
	return tf.keras.Model(inputs=[inputs], outputs=[outputs])
	'''
	keras_layers=[]
	if noise_std>0:
		keras_layers.append(tf.keras.layers.GaussianNoise(stddev=noise_std, input_shape=input_shape))
		keras_layers.append(tf.keras.layers.Conv2D(8, kernel_size, padding=padding))
	else:
		keras_layers.append(tf.keras.layers.Conv2D(8, kernel_size, padding=padding, input_shape=input_shape))
	if activation=='relu':
		keras_layers.append(tf.keras.layers.ReLU())
	elif activation=='lrelu':
		keras_layers.append(tf.keras.layers.LeakyReLU(alpha=alpha))
	for filters in layers:
		keras_layers.append(tf.keras.layers.Conv2D(filters, kernel_size, strides, padding=padding))
		if activation=='relu':
			keras_layers.append(tf.keras.layers.ReLU())
		elif activation=='lrelu':
			keras_layers.append(tf.keras.layers.LeakyReLU(alpha=alpha))
	if final_fc:
		keras_layers.append(tf.keras.layers.Flatten())
		keras_layers.append(tf.keras.layers.Dense(1))
	else:
		keras_layers.append(tf.keras.layers.Conv2D(1, kernel_size, padding=padding))
	return tf.keras.Sequential(keras_layers)
	'''

def discriminator_regNet(input_shape=[None,None,3], blocks=[{'d':3,'w':16,'g':4,'b':1,},]*2, kernel_size=3, final_fc=False, activation='relu', alpha=0.2, noise_std=0.0, padding='MIRROR'):
	'''
		loosely after: https://arxiv.org/pdf/2003.13678
	'''
	return None

def generator(input_shape=[None,None,None,3], layers=[16]*4, kernel_size=3, strides=1, activation='relu', alpha=0.2, padding='same'):
	keras_layers = [tf.keras.layers.InputLayer(input_shape=input_shape, name='gen_input')]
	for filters in layers:
		keras_layers.append(tf.keras.layers.Conv3D(filters, kernel_size, strides, padding=padding))
		if activation=='relu':
			keras_layers.append(tf.keras.layers.ReLU())
		elif activation=='lrelu':
			keras_layers.append(tf.keras.layers.LeakyReLU(alpha=alpha))
	keras_layers.append(tf.keras.layers.Conv3D(1, kernel_size, padding=padding, name='gen_output'))
	keras_layers.append(tf.keras.layers.ReLU())
	return tf.keras.Sequential(keras_layers)

def UNet_block_down(in_layer, filters, kernel_size, activation, alpha, down=True, padding='ZERO'):
	x = ConvLayer(in_layer, 3, filters, kernel_size, (2 if down else 1), activation, alpha, padding=padding)
	x = ConvLayer(x, 3, filters, kernel_size, 1, activation, alpha, padding=padding)
	return x

def UNet_block_up(in_layer, skip_layer, filters, kernel_size, activation, alpha, up=True, padding='ZERO'):
	x = in_layer
	if up:
		x = tf.keras.layers.UpSampling3D((2,2,2))(x)
	x = tf.keras.layers.concatenate([x, skip_layer])
	x = UNet_block_down(x, filters, kernel_size, activation, alpha, down=False)
	return x

#https://github.com/karolzak/keras-unet/blob/master/keras_unet/models/custom_unet.py
def generator_UNet(input_shape=[None,None,None,3], layers=[16]*4, kernel_size=3, strides=1, activation='relu', alpha=0.2, padding='ZERO'):
	if padding not in ['ZERO','MIRROR']:
		raise ValueError('UNet only support ZERO and MIRROR padding.')
	inputs = tf.keras.layers.Input(shape=input_shape, name='gen_input')
	
	x = UNet_block_down(inputs, layers[0], kernel_size, activation, alpha, down=False, padding=padding)
	down_layers = [x]
	for filters in layers[1:]:
		x = UNet_block_down(x, filters, kernel_size, activation, alpha, padding=padding)
		down_layers.append(x)
	
	for filters, layer in zip(reversed(layers[:-1]), reversed(down_layers[:-1])):
		x = UNet_block_up(x, layer, filters, kernel_size, activation, alpha, padding=padding)
	
	x = ConvLayer(x, 3, 1, kernel_size, name='gen_output') #tf.keras.layers.Conv3D(1, kernel_size, padding=padding, name='gen_output')(x)
	x = tf.keras.layers.Activation(tf.keras.activations.tanh, name='tanh')(x)
	outputs = tf.keras.layers.ReLU(name='non_negative')(x)
	
	return tf.keras.Model(inputs=[inputs], outputs=[outputs])
	
def generator_UNet_simple(input_shape=[None,None,None,3], layers=[16]*4, kernel_size=3, strides=1, activation='relu', alpha=0.2, padding='same'):
	inputs = tf.keras.layers.Input(shape=input_shape, name='gen_input')
	x = tf.keras.layers.Conv3D(8, kernel_size, 1, padding=padding)(inputs)
	down_layers = [x]
	for filters in layers:
		x = tf.keras.layers.Conv3D(filters, kernel_size, 2, padding=padding)(x)
		if activation=='relu':
			x = tf.keras.layers.ReLU()(x)
		elif activation=='lrelu':
			x = tf.keras.layers.LeakyReLU(alpha=alpha)(x)
		down_layers.append(x)
	
	for filters, layer in reversed(zip(layers[:-1], down_layers[:-1])):
		x = tf.keras.layers.UpSampling3D(x)
		x = tf.keras.layers.concatenate([x, layer])
		x = tf.keras.layers.Conv3D(filters, kernel_size, 1, padding=padding)(x)
		if activation=='relu':
			x = tf.keras.layers.ReLU()(x)
		elif activation=='lrelu':
			x = tf.keras.layers.LeakyReLU(alpha=alpha)(x)
	
	x = tf.keras.layers.Conv3D(1, kernel_size, padding=padding, name='gen_output')(x)
	outputs = tf.keras.layers.ReLU()(x)
	
	return tf.keras.Model(inputs=[inputs], outputs=[outputs])

# save any keras model
def save_NNmodel(model, name, path):
	if isinstance(model, (list, tuple)):
		for i, m in enumerate(model): save_NNmodel(m, name=name+"_{:04d}".format(i), path=path)
		return
	if isinstance(model, dict):
		for k, m in model.items(): save_NNmodel(m, name=name+"_"+str(k), path=path)
		return
	try:
		model.save(os.path.join(path, '{}_model.h5'.format(name)))
		log.info('Saved keras model %s', name)
	except:
		log.error('Failed to save full model. Saving weights instead.', exc_info=True)
		try:
			model.save_weights(os.path.join(path, '{}_weights.h5'.format(name)))
			log.info('Saved keras model weights %s', name)
		except:
			log.error('Failed to save model weights. Saving as npz instead.', exc_info=True)
			try:
				weights = model.get_weights()
				np.savez_compressed(os.path.join(path, '{}_weights.npz'.format(name)), weights)
				log.info('Saved keras model weights as npz %s', name)
			except:
				log.exception('Failed to save model weights.')


def get_NNmodel_names(base_path, allow_dict=False):
	# returns a flat list of model files saved by save_NNmodel() where base_path = path+name
	base_dir, base_name = os.path.split(base_path)
	if not os.path.isdir(base_dir):
		return []
		
	model_names = []
	name_mask = re.compile(r"^" + re.escape(base_name) + (r"(_.+)*_model\.h5$" if allow_dict else r"(_\d{4})*_model\.h5$"))
	#path_mask = re.compile(r"^(?P<basepath>.*" + re.escape(base_name) + (r")(_.+)*_model\.h5$" if allow_dict else r"(_\d{4})*_model\.h5$"))
	for entry in os.scandir(base_dir):
		if entry.is_file():
			match = name_mask.search(entry.name)
			if match is not None:
				model_names.append(entry.path[:-9]) # filename without '_model.h5'
	model_names.sort()
	
	return model_names