from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tensorflow as tf
import numpy as np
import copy
import munch

import logging
log = logging.getLogger('generator')
log.setLevel(logging.DEBUG)

class UNet(tf.keras.Model):
	@staticmethod
	def from_config(cls, config):
		return cls(**config)
	
	def __init__(self, dim, input, layers, skip=False):
		super(Discriminator, self).__init__()
		self.dim = dim
		self.layers = [tf.keras.layers.InputLayer(input_shape=input, name='gen_input')]
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
		else
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

# circular buffer of fixed size
# most of it still has to be tested...
class HistoryBuffer:
	def __init__(self, size):
		self.size = size
		self.buf = [None]*size
		self.head = 0
		self.elements = 0
	
	#https://en.wikipedia.org/wiki/Modulo_operation
	def _get_buf_index(self, idx):
		return (self.head+index)%self.size
	def _move_head(self, steps=1):
		self.head = _get_buf_index(steps)
	def __getitem__(self, index):
		if index>=self.elements or index<-self.elements:
			raise IndexError('Invalid position in buffer.')
		if index<0:
			index=self.elements+index
		return self.buf[_get_buf_index(index)]
	def __setitem__(self, index, value):
		if index>=self.elements or index<-self.elements:
			raise IndexError('Invalid position in buffer.')
		if index<0:
			index=self.elements+index
		self.buf[_get_buf_index(index)] = value
	def __len__(self):
		return self.elements
	def __call__(self, index=None):
		if index is None:
			return self.get()
		else:
			return self[index]
	@property
	def empty(self):
		return self.elements==0
	@property
	def full(self):
		return self.elements==self.size
	@property
	def list(self)
		return [self[_] for _ in range(len(self))]
	# add a new element, if full the oldest will be overwritten
	def push(self, element):
		self.buf[self.head]=element
		self._move_head()
		if self.elements<self.size:
			self.elements +=1
	def push_samples(self, samples, sample_chance=1.0):
		if sample_chance>=1.0:
			for sample in samples:
				self.push(sample)
		else:
			rands = np.random.random(len(samples))
			for sample, r in zip(samples, rands):
				if r<sample_chance: self.push(sample)
	
	# get a random valid element. does not remove the element
	def get(self):
		if self.elements==0:
			raise IndexError('Cant get element from empty buffer.')
		index = np.random.randint(self.elements)
		return self[index] #self.buf[idx]
	def get_samples(self, num_samples):
		if history.is_empty():
			return []
		return [history.get() for _ in range(num_samples)]
	
	def _pop_first(self):
		elem = self[0]
		self[0] = None
		self._move_head(-1)
		self.elements -=1
	def _pop_last(self):
		elem = self[-1]
		self[-1] = None
		self.elements -=1
	def pop(self, reverse=False):
		if reverse:
			return _pop_last()
		else:
			return _pop_first()
	
	def resize(self, new_size):
		raise NotImplementedError()
	def reset(self):
		del self.buf
		self.buf = [None]*size
		self.head = 0
		self.elements = 0
	
	def __str__(self):
		return 'HistoryBuffer: {}/{} (internal head at {}): {}'.format(self.elements, self.size, self.head, self.list)

