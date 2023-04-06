from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tensorflow as tf
import numpy as np
import copy
import munch

import logging
log = logging.getLogger('discriminator')
log.setLevel(logging.DEBUG)

from tensorflow.python.keras.utils import conv_utils

#https://stackoverflow.com/questions/45254554/tensorflow-same-padding-calculation
def getSamePadding(shape, kernel, stride):
	dim = len(shape)
	#kernel = conv_utils.normalize_tupel(kernel, dim, 'kernel_size')
	#stride = onv_utils.normalize_tupel(stride, dim, 'stride')
	pad = []
	for dim, k, s in zip(shape, kernel, stride):
		out_dim = int(ceil(dim/s))
		pad_total = max((out_dim - 1)*s + k - dim, 0)
		pad_before = int(pad_total//2)
		pad_after = pad_total - pad_before
		pad.append([pad_before, pad_after])
	return pad

#https://stackoverflow.com/questions/49189496/can-symmetrically-paddding-be-done-in-convolution-layers-in-keras
class MirrorPadND(tf.keras.Layer):
	def __init__(self, kernel, stride, **kwargs):
		self.kernel = conv_utils.normalize_tupel(kernel, dim, 'kernel_size')
		self.stride = onv_utils.normalize_tupel(stride, dim, 'stride')
		super(ConvNDPad, self).__init__(**kwargs)
	
	def build(self, input_shape):
		super(ConvNDPad, self).build(input_shape)
		
	def call(self, inputs, kernel, stride):
		inputs_shape = inputs.get_shape().as_list()
		dim = len(inputs_shape)-2
		shape = inputs.get_shape().as_list()[-(dim+1):-1]
		pad = [[0,0]] + getSamePadding(shape, kernel, stride) + [[0,0]]
		padded = tf.pad(inputs, pad, 'REFLECT')
		return padded
	
	def compute_output_shape(self, input_shape):
		shape = list(np.asarray(input_shape)[:-1] + np.sum(getSamePadding(input_shape[:-1], self.kernel, self.stride), axis=-1)) + list(input_shape[-1:])
		return shape

class DiscriminatorInput(tf.keras.Layer):
	def __init__(self):
		super(DiscriminatorInput, self).__init__()
		
	def call(self):
		pass

class Discriminator(tf.keras.Model):
	@staticmethod
	def from_config(cls, config):
		return cls(**config)
	
	def __init__(self, dim, input, layers, noise_stddev=0.0, fully_connected=False):
		super(Discriminator, self).__init__()
		self.dim = dim
		self.layers = [tf.keras.layers.InputLayer(input_shape=input, name='disc_input')]
		if noise_stddev>0.0:
			self.layers.append(tf.keras.layers.GaussianNoise(stddev=noise_stddev, name='input_noise'))
		
		for layer in layers:
			self._add_conv_layer(**layer)
		
		if fully_connected:
			self.layers.append(tf.keras.layers.Flatten())
			self.layers.append(tf.keras.layers.Dense(1, name='disc_output'))
		else:
			keras_layers.append(tf.keras.layers.Conv2D(1, 1, 1, padding='same', name='disc_output'))
	
	def _add_conv_layer(self, filters, kernel=4, stride=1, activation="lrelu", alpha=0.2, padding="same",name=None):
		if self.dim==1:
			self.layers.append(tf.keras.layers.Conv1D(filters, kernel, stride, padding=padding))
		elif self.dim==2:
			self.layers.append(tf.keras.layers.Conv2D(filters, kernel, stride, padding=padding))
		elif self.dim==3:
			self.layers.append(tf.keras.layers.Conv3D(filters, kernel, stride, padding=padding))
		else:
			raise ValueError('Unsupported dimension: {}'.format(self.dim))
		
		if activation=='relu':
			self.layers.append(tf.keras.layers.ReLU(tf.keras.layers.LeakyReLU(alpha=alpha)))
		elif activation=='lrelu':
			self.layers.append()
		else:
			raise ValueError('Unknown activation \'{}\'.'.format(activation))
	
	def call(self, input):
		x = input
		for layer in self.layers:
			x = layer(x)
		return x
	
	def scores(self, input):
		return tf.math.sigmoid(self.call(input))
	
	def cross_entropy(self, input, labels, get_scores=False):
		logits = self.call(input)
		loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
		if get_scores:
			return loss, tf.math.sigmoid(logits)
		else:
			return loss
	
	def make_input(self, **inputs):
		raise NotImplementedError()
		return inputs['input'] #PH
	
	@staticmethod
	def get_layer_config(*,filters=16, kernel=4, stride=1, activation="lrelu", alpha=0.2, padding="same", _munch=None):
		layer = {
			"filters":filters,
			"kernel":kernel,
			"stride":stride,
			"activation":activation,
			"alpha":alpha,
			"padding":padding,
		#	"normalization":"NONE", #NONE, BATCH, PATCH
		#	"channel_inhibition":None,
		#	"name":None,
		#	"input_layer":"PREV", #PREV, <idx>, <name>, IN_<idx>
		}
		if _munch is not None:
			layer = _munch(layer)
		return layer
	
	@staticmethod
	def get_default_config(*,num_layers=4, layer_config=None, _munch=None):
		if layer_config is None:
			layer_config = Discriminator.get_layer_config()
		config = {
			"dim":2,
			"input":[None,None,1],
			"noise_stddev":0.0,
			"layers":[copy.deepcopy(layer_config) for _ in range(num_layers)],
			"fully_connected":False,
		}
		if _munch is not None:
			config = _munch(config)
		return config


