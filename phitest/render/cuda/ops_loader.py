import os

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

#import numpy as np
import tensorflow as tf
import logging #, warnings

from phitest.render.profiling import SAMPLE

LOG = logging.getLogger("CudaOpLoader")

kernel_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'build') #'test/laplace_op.so')#

kernel_path = os.path.join(kernel_dir, 'resample_grid.so')
sampling_ops = tf.load_op_library(kernel_path)

@ops.RegisterGradient("SampleGridTransform")
def _sample_grid_transform_grad(op, grad):
	with SAMPLE("SampleGridTransformGRAD"):
		in_grad = sampling_ops.sample_grid_transform_grad( \
				input					= op.inputs[0], \
				output_grad				= grad, \
				matrix_m				= op.inputs[1], \
				matrix_v				= op.inputs[2], \
				matrix_p				= op.inputs[3], \
				frustum_params			= op.inputs[4], \
				interpolation			= op.get_attr('interpolation'), \
				boundary				= op.get_attr('boundary'), \
				mipmapping				= op.get_attr('mipmapping'), \
				num_mipmaps				= op.get_attr('num_mipmaps'), \
				mip_bias				= op.get_attr('mip_bias'), \
				coordinate_mode			= op.get_attr('coordinate_mode'), \
				cell_center_offset		= op.get_attr('cell_center_offset'), \
				separate_camera_batch	= op.get_attr('separate_camera_batch'))
	return [array_ops.reshape(in_grad, array_ops.shape(op.inputs[0]))]

sample_grid_transform = sampling_ops.sample_grid_transform

@ops.RegisterGradient("SampleGridLut")
def _sample_grid_lut_grad(op, grad):
	with SAMPLE("SampleGridLutGRAD"):
		in_grad, lut_grad = sampling_ops.sample_grid_lut_grad( \
				input					= op.inputs[0], \
				output_grad				= grad, \
				lookup					= op.inputs[1], \
				interpolation			= op.get_attr('interpolation'), \
				boundary				= op.get_attr('boundary'), \
				mipmapping				= op.get_attr('mipmapping'), \
				num_mipmaps				= op.get_attr('num_mipmaps'), \
				mip_bias				= op.get_attr('mip_bias'), \
				coordinate_mode			= op.get_attr('coordinate_mode'), \
				cell_center_offset		= op.get_attr('cell_center_offset'), \
				relative_coords			= op.get_attr('relative_coords'), \
				normalized_coords		= op.get_attr('normalized_coords'), \
				separate_camera_batch	= op.get_attr('separate_camera_batch'))
	return [array_ops.reshape(in_grad, array_ops.shape(op.inputs[0])), array_ops.reshape(lut_grad, array_ops.shape(op.inputs[1]))]

sample_grid_lut = sampling_ops.sample_grid_lut

kernel_path = os.path.join(kernel_dir, 'reduce_blend.so')
blending_ops = tf.load_op_library(kernel_path)
@ops.RegisterGradient("ReduceGridBlend")
def _reduce_grid_blend_grad(op, grad):
	with SAMPLE("ReduceGridBlendGRAD"):
		grad = blending_ops.reduce_grid_blend_grad(grad, op.outputs[0], op.inputs[0], op.get_attr('blending_mode'), op.get_attr('keep_dims'))
	return [array_ops.reshape(grad, array_ops.shape(op.inputs[0]))]

reduce_grid_blend = blending_ops.reduce_grid_blend

# Fused Rendering, sample & blend
kernel_path = os.path.join(kernel_dir, 'raymarch_grid.so')
raymarching_ops = tf.load_op_library(kernel_path)
@ops.RegisterGradient("RaymarchGridTransform")
def _raymarch_grid_transform_grad(op, grad):
	with SAMPLE("RaymarchGridTransformGRAD"):
		#LOG.info("_raymarch_grid_transform_grad: in %s, out %s, grads %s", array_ops.shape(op.inputs[0]).numpy(), array_ops.shape(op.outputs[0]).numpy(), array_ops.shape(grad).numpy())
		in_grad = raymarching_ops.raymarch_grid_transform_grad( \
				input					= op.inputs[0], \
				output					= op.outputs[0], \
				output_grad				= grad, \
				matrix_m				= op.inputs[1], \
				matrix_v				= op.inputs[2], \
				matrix_p				= op.inputs[3], \
				frustum_params			= op.inputs[4], \
				output_shape			= op.inputs[5], \
				interpolation			= op.get_attr('interpolation'), \
				boundary				= op.get_attr('boundary'), \
				blending_mode			= op.get_attr('blending_mode'), \
				keep_dims				= op.get_attr('keep_dims'), \
				separate_camera_batch	= op.get_attr('separate_camera_batch'))
	return [array_ops.reshape(in_grad, array_ops.shape(op.inputs[0]))]

raymarch_grid_transform = raymarching_ops.raymarch_grid_transform


@ops.RegisterGradient("RaymarchGridTransformGrad")
def _raymarch_grid_transform(op, grad):
	with SAMPLE("RaymarchGridTransformGradGRAD"):
		blend_mode = op.get_attr('blending_mode')
		if blend_mode not in ["ADDITIVE", b"ADDITIVE"]: raise NotImplementedError("Gradient of raymach op gradient only available for ADDITIVE blending, is %s"%(blend_mode))
		in_grad_grad = raymarching_ops.raymarch_grid_transform( \
				input					= grad, \
			#	output					= op.outputs[0], \
			#	output_grad				= grad, \
				matrix_m				= op.inputs[3], \
				matrix_v				= op.inputs[4], \
				matrix_p				= op.inputs[5], \
				frustum_params			= op.inputs[6], \
				output_shape			= op.inputs[7], \
				interpolation			= op.get_attr('interpolation'), \
				boundary				= op.get_attr('boundary'), \
				blending_mode			= op.get_attr('blending_mode'), \
				keep_dims				= op.get_attr('keep_dims'), \
				separate_camera_batch	= op.get_attr('separate_camera_batch'))
		#LOG.info("_raymarch_grid_transform: input shape %s, output shape %s, target shape %s", array_ops.shape(grad).numpy(), array_ops.shape(in_grad_grad).numpy(), array_ops.shape(op.inputs[2]).numpy())
	return [None, None, array_ops.reshape(in_grad_grad, array_ops.shape(op.inputs[2]))]


