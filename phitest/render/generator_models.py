import tensorflow as tf
import numbers, json, copy, math
from collections.abc import Iterable
from lib.tf_ops import *
import munch
import logging
from .vector import GridShape
from .profiling import SAMPLE
LOG = logging.getLogger('Generators')
LOG.setLevel(logging.DEBUG)

"""
	Density:
target view encoder: 2D target -> 3D latent/features, for single view, shared in multi view
view combination: 3D latent/features of multiple views together, may involve resampling/transformation based on view calibration
volume decoder: combined 3D latent/features to 3D density volume

	Velcoity:

"""
class GeneratorSingleviewEncoder2D:
	"""
	simple conv stack to encode a single target
	"""
	def __init__(self, targets):
		raise NotImplementedError()

class GeneratorSingleviewLifting:
	"""
	transform a 2D (latent) tensor to 3D
	"""
	def __init__(self, model_encoder):
		raise NotImplementedError()

class GeneratorMultiviewMerge3D:
	"""
	merge several (latent) 3D tensors, respecting camera calibration of the corresponding, encoded targets (if available)
	simple number-invariant combination of tensors(e.g. addition) followed by some convolutions
	"""
	def __init__(self, embeddings_v):
		raise NotImplementedError()

class GeneratorDensityDecoder3D:
	"""
	decode a (latent) 3D tensor into a volume of density
	may get additional embeddings of past and future states/targets; or decoded past densities
	"""
	def __init__(self, embeddings_t):
		raise NotImplementedError()



class GeneratorVelocity:
	"""
	decode a (latent) 3D tensor into a volume of velocities
	may get additional embeddings of past and future states/targets + last/past (adevcted) velocity for coherence
	"""
	def __init__(self, density_or_embedding, veloicty_prev):
		raise NotImplementedError()


def get_view_encoder(input_channels=3, layers=[4]*6, kernel_size=3, strides=1, skip_indices=None, skips={}, activation='relu', alpha=0.2, noise_std=0.0, padding='ZERO', skip_mode="CONCAT"):
	input_shape = [None,None,input_channels]
	dim = len(input_shape)-1
	num_layers = len(layers)
	skip_mode = skip_mode.upper()
	if np.isscalar(strides):
		strides = [strides]*num_layers
	if np.isscalar(kernel_size):
		kernel_size = [kernel_size]*num_layers
	if skip_indices is None:
		skip_indices = [0]*num_layers
	x = tf.keras.layers.Input(shape=input_shape, name='view-enc_input')
	inputs = x
	if noise_std>0:
		x = tf.keras.layers.GaussianNoise(stddev=noise_std)(x)
	for filters, stride, ks, skip_idx in zip(layers, strides, kernel_size, skip_indices):
		if skip_idx<0:
			idx = abs(skip_idx)
			if idx not in skips: raise ValueError("Skip connection %d not defined."%idx)
			if skip_mode=="CONCAT":
				x = tf.keras.layers.Concatenate(axis=-1)([x, skips[idx]])
			elif skip_mode=="ADD":
				x = tf.keras.layers.Add()([x, skips[idx]])
			else: raise ValueError("Unknown skip mode '%s'"%(skip_mode,))
		x = ConvLayer(x, dim, filters, ks, stride, activation, alpha, padding=padding)
		if skip_idx>0:
			if skip_idx in skips: raise ValueError("Skip connection %d already defined."%skip_idx)
			skips[skip_idx] = x
	outputs = x
	return tf.keras.Model(inputs=[inputs], outputs=[outputs])


def get_density_decoder(input_channels=4, layers=[4]*6, kernel_size=3, strides=1, skip_indices=None, skips={}, activation='relu', alpha=0.2, noise_std=0.0, padding='ZERO', skip_mode="CONCAT"):
	input_shape = [None,None,None,input_channels]
	dim = len(input_shape)-1
	num_layers = len(layers)
	skip_mode = skip_mode.upper()
	if np.isscalar(strides):
		strides = [strides]*num_layers
	if np.isscalar(kernel_size):
		kernel_size = [kernel_size]*num_layers
	if skip_indices is None:
		skip_indices = [0]*num_layers
	x = tf.keras.layers.Input(shape=input_shape, name='dens-gen_input')
	inputs = x
	if noise_std>0:
		x = tf.keras.layers.GaussianNoise(stddev=noise_std)(x)
	for filters, stride, ks, skip_idx in zip(layers, strides, kernel_size, skip_indices):
		if skip_idx<0:
			idx = abs(skip_idx)
			if idx not in skips: raise ValueError("Skip connection %d not defined."%idx)
			if skip_mode=="CONCAT":
				x = tf.keras.layers.Concatenate(axis=-1)([x, skips[idx]])
			elif skip_mode=="ADD":
				x = tf.keras.layers.Add()([x, skips[idx]])
			else: raise ValueError("Unknown skip mode '%s'"%(skip_mode,))
		x = ConvLayer(x, dim, filters, ks, stride, activation, alpha, padding=padding)
		if skip_idx>0:
			if skip_idx in skips: raise ValueError("Skip connection %d already defined."%skip_idx)
			skips[skip_idx] = x
	x = ConvLayer(x, dim, 1, 3, activation="relu", padding=padding, name='dens-gen_output')
	outputs = x
	return tf.keras.Model(inputs=[inputs], outputs=[outputs])
	
def get_velocity_decoder(input_channels=4, layers=[4]*6, kernel_size=3, strides=1, skip_indices=None, skips={}, activation='relu', alpha=0.2, noise_std=0.0, padding='ZERO'):
	input_shape = [None,None,None,input_channels]
	dim = len(input_shape)-1
	num_layers = len(layers)
	if np.isscalar(strides):
		strides = [strides]*num_layers
	if skip_indices is None:
		skip_indices = [0]*num_layers
	x = tf.keras.layers.Input(shape=input_shape, name='vel-gen_input')
	inputs = x
	if noise_std>0:
		x = tf.keras.layers.GaussianNoise(stddev=noise_std)(x)
	for filters, stride, skip_idx in zip(layers, strides, skip_indices):
		if skip_idx<0:
			idx = abs(skip_idx)
			if idx not in skips: raise ValueError("Skip connection %d not defined."%idx)
			x = tf.keras.layers.Concatenate(axis=-1)([x, skips[idx]])
		x = ConvLayer(x, dim, filters, kernel_size, stride, activation, alpha, padding=padding)
		if skip_idx>0:
			if skip_idx in skips: raise ValueError("Skip connection %d already defined."%skip_idx)
			skips[skip_idx] = x
	x = ConvLayer(x, dim, 3, kernel_size, padding=padding, name='vel-gen_output')
	outputs = x
	return tf.keras.Model(inputs=[inputs], outputs=[outputs])

class ChannelPadding(tf.keras.layers.Layer):
	def __init__(self, padding, constant=0, **kwargs):
		if not isinstance(constant, numbers.Number): raise ValueError("constant must be a scalar.")
		if isinstance(padding, numbers.Integral):
			padding = (padding, padding)
		elif (not isinstance(padding, (list, tuple))) or (not len(padding)==2) or (not isinstance(padding[0], numbers.Integral)) or (not isinstance(padding[1], numbers.Integral)):
			raise  ValueError("padding must be int or tuple of 2 int.")
		super().__init__(**kwargs)
		LOG.debug("Create ChannelPadding '%s': pad=%s, c=%s", self.name, padding, constant)
		self.padding = padding
		self.constant = constant
	
	def _get_paddings(self, tensor_rank):
		return [(0,0)]*(tensor_rank-1) + [self.padding]
	
	def call(self, inputs):
		return tf.pad(inputs, self._get_paddings(tf.rank(inputs).numpy()), mode="CONSTANT", constant_values=self.constant)
	
	def compute_output_shape(self, input_shapes):
		if isinstance(input_shapes, list):
			assert len(input_shapes)==1
			input_shapes = input_shapes[0]
		output_shape = list(input_shapes)[:-1]
		output_shape.append(input_shapes[-1] + self.padding[0] + self.padding[1])
		output_shape = tf.TensorShape(output_shape)
		return output_shape
	
	def get_config(self):
		config = super().get_config()
		config.update({"padding":self.padding, "constant":self.constant})
		return config

class WeightedSum(tf.keras.layers.Add):
	def __init__(self, alpha, **kwargs):
		super().__init__(**kwargs)
		LOG.debug("Create WeightedSum '%s': alpha=%s", self.name, alpha)
		self.alpha = tf.keras.backend.variable(alpha, dtype="float32", name="ws_alpha")
	
	def _merge_function(self, inputs):
		assert (len(inputs)==2), "WeightedSum takes 2 inputs"
		LOG.debug("WeightedSum '%s' merging inputs %s and %s", self.name, shape_list(inputs[0]), shape_list(inputs[1]))
		return ((1.0-self.alpha) * inputs[0]) + (self.alpha * inputs[1])
	
	@property
	def variables(self):
		return [self.alpha]
	@property
	def trainable_variables(self):
		return []
	@property
	def non_trainable_variables(self):
		return [self.alpha]
	
	
	@property
	def trainable_weights(self):
		return []
	@property
	def non_trainable_weights(self):
		return []
	@property
	def weights(self):
		return []
	
	def get_config(self):
		config = super().get_config()
		config.update({"alpha":self.alpha.numpy().tolist()})
		return config

class ScalarMul(tf.keras.layers.Layer):
	def __init__(self, alpha, **kwargs):
		super().__init__(**kwargs)
		LOG.debug("Create ScalarMul '%s': alpha=%s", self.name, alpha)
		self.alpha = tf.keras.backend.variable(alpha, dtype="float32", name="smul_alpha")
	
	def call(self, inputs):
		LOG.debug("WeightedSum '%s' scaling %s with %s", self.name, shape_list(inputs), self.alpha)
		return inputs * self.alpha
		
	@property
	def variables(self):
		return [self.alpha]
	@property
	def trainable_variables(self):
		return []
	@property
	def non_trainable_variables(self):
		return [self.alpha]
	
	
	@property
	def trainable_weights(self):
		return []
	@property
	def non_trainable_weights(self):
		return []
	@property
	def weights(self):
		return []
	
	def get_config(self):
		config = super().get_config()
		config.update({"alpha":self.alpha.numpy().tolist()})
		return config

class UnprojectionLayer(tf.keras.layers.Layer):
	def __init__(self, grid_transform, cameras, renderer, merge_mode="MEANPROD", **layer_kwargs):
		super().__init__(**layer_kwargs)
		self.__renderer = renderer
		self.__cameras = copy.copy(cameras)
		self._set_output_transform(grid_transform)
		assert merge_mode in ["SUM", "MEAN", "PROD", "SUMPROD", "MEANPROD"]
		self.__merge_mode = merge_mode
		if not self.__renderer.blend_mode=="ADDITIVE":
			raise ValueError("target merging requires ADDITIVE blend mode used in target lifting")
		#assert isinstance(spatial_output_shape, (list, tuple)) and len(spatial_output_shape)==3 and all(isinstance(dim, numbers.Integral) for dim in spatial_output_shape)
		#self.__shape = copy.copy(spatial_output_shape)
	
	def get_num_views(self):
		return len(self.__cameras)
	
	def set_output_shape(self, shape):
		assert isinstance(shape, (list, tuple)) and len(shape)==3 and all(isinstance(dim, numbers.Integral) for dim in shape)
		self.__transform.grid_size = shape
	
	def _set_output_transform(self, grid_transform):
		assert grid_transform is not None
		self.__transform = grid_transform.copy_no_data()
	
	def __unproject_cameras(self, tensor, cameras):
		shape = GridShape.from_tensor(tensor)
		with SAMPLE("unproject %s-%d>%s"%([shape.y, shape.x],cameras[0].transform.grid_size[0], self.__transform.grid_size)):
			# image shape for unprojection: NVHWC with C in [1,2,4]
			if shape.c not in [1,2,4]:
				channel_div = shape.c//4 if shape.c%4==0 else shape.c//2 if shape.c%2==0 else shape.c
				tensor = tf.reshape( \
									tf.transpose( \
										tf.reshape(tensor, (shape.n, shape.z, shape.y, shape.x, channel_div, shape.c//channel_div)), \
										(0,4,1,2,3,5)), \
									(shape.n * channel_div, shape.z, shape.y, shape.x, shape.c//channel_div) \
								)
				#raise NotImplementedError("can only unproject with channels 1,2 or 4. is %d"%shape.c)
				# roll channels into batch?
			# camera XY resolution does not matter, raymarch_camera uses the input image resolution
			
			tensor = self.__renderer.raymarch_camera(data=tensor, cameras=cameras, transformations=self.__transform, inverse=True, squeeze_batch=False)
			vol_shape = GridShape.from_tensor(tensor)
			if shape.c not in [1,2,4]:
				tensor = tf.reshape( \
									tf.transpose( \
										tf.reshape(tensor, (shape.n, channel_div, vol_shape.z, vol_shape.y, vol_shape.x, shape.c//channel_div)), \
										(0,2,3,4,1,5)), \
									(shape.n, vol_shape.z, vol_shape.y, vol_shape.x, shape.c) \
								)
		return tensor
	
	def _unproject(self, tensor):
		assert isinstance(tensor, tf.Tensor)
		shape = shape_list(tensor)
		assert len(shape)==5 #NVHWC
		shape = GridShape.from_tensor(tensor)
		num_views = self.get_num_views()
		assert num_views==shape.z, "view dimension does not match camera list"
		
		if self.__merge_mode not in ["SUM", "MEAN"]:
			tensors = tf.split(tensor, shape.z, axis=1) #V-N1WHC
			shape.z = 1
			cameras = [[cam] for cam in self.__cameras]
		else:
			#if not self.lifting_renderer.blend_mode=="ADDITIVE":
			#	raise ValueError("SUM and MEAN unprojection merging requires ADDITIVE blend mode used in target lifting")
			tensors = [tensor] #1-NVWHC
			cameras = [self.__cameras]
		
		data = []
		for tensor, cams in zip(tensors, cameras):
			data.append(self.__unproject_cameras(tensor, cams))
		
		with SAMPLE("merge"):
			if num_views==1 or self.__merge_mode=="SUM":
				return data[0]
			elif self.__merge_mode=="MEAN":
				return data[0] * tf.constant(1.0/num_views, dtype=data[0].dtype)
			elif self.__merge_mode=="PROD":
				return tf.reduce_prod([tf.tanh(_) for _ in data], axis=0)
			elif self.__merge_mode in ["SUMPROD", "MEANPROD"]:
				add_channels = shape.c//2
				
				add_data = tf.reduce_sum([_[...,:add_channels] for _ in data], axis=0)
				mul_data = tf.reduce_prod([tf.tanh(_[...,add_channels:]) for _ in data], axis=0)
				
				if self.__merge_mode=="MEANPROD":
					add_data = add_data * tf.constant(1.0/num_views, dtype=data[0].dtype)
				
				return tf.concat([add_data, mul_data], axis=-1)
			
		
	
	def call(self, inputs):
		with SAMPLE("UnprojectionLayer"):
			inp_shape = shape_list(inputs)
			assert len(inp_shape)==4 #NHWC
			inputs = tf.expand_dims(inputs, axis=1) #->N1HWC
			shape = GridShape.from_tensor(inputs)
		
			tensor = self._unproject(inputs)
		
		return tensor
	
	def compute_output_shape(self, input_shapes):
		if isinstance(input_shapes, list):
			assert len(input_shapes)==1
			input_shapes = input_shapes[0]
		output_shape = [input_shapes[0]] + self.__transform.grid_size + [input_shapes[-1]] #NDHWC
		output_shape = tf.TensorShape(output_shape)
		return output_shape
	
	@property
	def variables(self):
		return []
	@property
	def trainable_variables(self):
		return []
	@property
	def non_trainable_variables(self):
		return []
	
	
	@property
	def trainable_weights(self):
		return []
	@property
	def non_trainable_weights(self):
		return []
	@property
	def weights(self):
		return []
	
	def get_config(self):
		config = super().get_config()
		#config.update(self.cnf)
		return config
	
	@classmethod
	def from_config(cls, config_dict):
		raise NotImplementedError("serialization of camera and transform")
		return cls(**config_dict)

class GridSamplingLayer(tf.keras.layers.Layer):
	def __init__(self, spatial_output_shape, filter_mode="LINEAR", mipmapping="NONE", mip_levels="AUTO", boundary_mode="CLAMP", sample_gradients=False, **kwargs):
		super().__init__(**kwargs)
		self._renderer = Renderer(filter_mode=filter_mode, boundary_mode=boundary_mode, \
			mipmapping=mipmapping, num_mips=0, mip_bias=0, sample_gradients=sample_gradients, \
			name=self.name+"_GSLrenderer")
		self.__filter_mode=filter_mode
		self.__mipmapping=mipmapping
		self.__mip_levels=mip_levels
		self.__boundary_mode=boundary_mode
		self.__sample_gradients=sample_gradients
		assert has_shape(spatial_output_shape, [3])
		self._output_shape = spatial_output_shape
		
	def call(self, inputs):
		if self.mip_levels=="AUTO":
			pass
		LOG.debug("GridSamplingLayer '%s' sampling %s to %s", self.name, shape_list(inputs), self.alpha)
		return self._renderer.resample_grid3D_aligned(inputs, self._output_shape)
	
	def compute_output_shape(self, input_shapes):
		if isinstance(input_shapes, list):
			assert len(input_shapes)==1
			input_shapes = input_shapes[0]
		output_shape = list(self._output_shape)
		output_shape.append(input_shapes[-1])
		output_shape = tf.TensorShape(output_shape)
		return output_shape
	
	@property
	def variables(self):
		return []
	@property
	def trainable_variables(self):
		return []
	@property
	def non_trainable_variables(self):
		return []
	
	
	@property
	def trainable_weights(self):
		return []
	@property
	def non_trainable_weights(self):
		return []
	@property
	def weights(self):
		return []
	
	def get_config(self):
		config = super().get_config()
		config.update({
			"spatial_output_shape":self._output_shape,
			"filter_mode":self.__filter_mode,
			"mipmapping":self.__mipmapping,
			"mip_levels":self.__mip_levels,
			"boundary_mode":self.__boundary_mode,
			"sample_gradients":self.__sample_gradients,
		})
		return config

# get object from dict if it exist, otherwise construct, add and return it
def dict_get_make(d, k, v_fn=lambda: None):
	if k not in d:
		d[k] = v_fn()
	return d[k]

def normalize_block_config(block_config, num_levels, is_shared):
	# list of str if shared
	# list of list of str
	if not isinstance(block_config, (list, tuple)):
		raise TypeError("Invalid Block configuration. must be list(list(str)) or list(str), is: %s"%(block_config,))
	is_single = len(block_config)==0 or isinstance(block_config[0], str)
	if is_shared and (not is_single) and len(block_config)!=1:
		raise ValueError("Block configuration must be a list(str) or [list(str)] for shared layers, is: %s"%(block_config,))
	if (not is_shared) and (not is_single) and (len(block_config)!=num_levels):
		raise ValueError("Block configuration must be specified for %d levels, is: %s"%(num_levels, block_config,))
	for sub in block_config:
		if (isinstance(sub, str) and not is_single) or (isinstance(sub, (list, tuple)) and is_single):
			raise TypeError("Invalid Block configuration. must be list(list(str)) or list(str), is: %s"%(block_config,))
		if not is_single:
			for block in sub:
				if not isinstance(block, str):
					raise TypeError("Invalid Block configuration. must be list(list(str)) or list(str), is: %s"%(block_config,))
	if is_single:
		block_config = [block_config] * (1 if is_shared else num_levels)
	return block_config

# from optimizer.py
def _var_key(var):
	# if hasattr(var, "op"):
		# return (var.op.graph, var.op.name)
	return var._unique_id

class DeferredBackpropNetwork:
	def __init__(self, model, name="DeferredBackpropNetwork"):
		assert isinstance(model, tf.keras.models.Model)
		self._model = model
		assert isinstance(name, str)
		self.name = name
		self._frozen = False
		self._clear_gradient_accumulators()
		self.clear_gradients()
	
	def __call__(self, inputs):
		return self._model(inputs)
	
	@property
	def trainable_variables(self):
		return self._model.trainable_variables
	
	@property
	def is_frozen_weights(self):
		return self._frozen
	def set_frozen_weights(self, freeze):
		if freeze:
			self.freeze_weights()
		else:
			self.unfreeze_weights()
	def freeze_weights(self):
		if self.has_pending_gradients:
			raise RuntimeError("DeferredBackpropNetwork: clear gradients before freezing weights.")
		self._frozen = True
	def unfreeze_weights(self):
		self._frozen = False
	
	def _get_gradient_accumulator(self, var):
		#assert isinstance(var, tf.Variable)
		#if not hasattr(self, "_grad_acc"):
		#	self._grad_acc = {}
		var_key = _var_key(var)
		if var_key not in self._grad_acc:
			self._grad_acc[var_key] = tf.Variable(tf.zeros_like(var), trainable=False)
		return self._grad_acc[var_key]
	
	def _get_gradient_accumulators(self):
		variables = self.trainable_variables
		# if len(self._grad_acc)>len(variables): #can have more if reducing recursions
			# raise RuntimeError("%s built %d gradient accumulators, but has only %d variables."%(self.name, len(self._grad_acc), len(variables)))
		# elif len(self._grad_acc)<len(variables):
			# LOG.warning("%s has %d gradient accumulators for %d variables, some gradients might be empty.",self.name, len(self._grad_acc), len(variables))
		# else:
			# LOG.info("%s has %d gradient accumulators for %d variables.",self.name, len(self._grad_acc), len(variables))
		return [self._get_gradient_accumulator(var) for var in variables]
	
	def _clear_gradient_accumulators(self):
		self._grad_acc = {}
	
	def _build_gradient_accumulators(self):
		raise NotImplementedError("There seems to be a memory leak related to building these to often.")
		self._grad_acc = [tf.Variable(tf.zeros_like(tvar), trainable=False) for tvar in self.trainable_variables]
		self._pending_gradients = False
		self._num_pending_gradients = 0
	
	def add_gradients(self, grads, allow_none=False, variables=None):
		if self.is_frozen_weights: return
		with SAMPLE("{} add grads".format(self.name)):
			if variables is None:
				variables = self.trainable_variables
				if not len(variables)==len(grads):
					raise ValueError("%s got %d gradients for %d variables"%(self.name, len(grads), len(variables)))
			# if not len(self._grad_acc)==len(grads):
				# num_vars = len(self.trainable_variables)
				# if not self.has_pending_gradients and len(grads)==num_vars:
					# LOG.warning("%s gradient accumulators do not match grads/vars, rebuilding.", self.name)
					# self._build_gradient_accumulators()
				# else:
					# raise ValueError("%s vars %d, grads %d, acc %d, has_pending_gradients %s"%(self.name, num_vars, len(grads), len(self._grad_acc), self.has_pending_gradients, ))
			#for i, (acc, grad) in enumerate(zip(self._grad_acc, grads)):
			for i, (var, grad) in enumerate(zip(variables, grads)):
				if grad is not None:
					acc = self._get_gradient_accumulator(var)
					acc.assign_add(grad)
					self._pending_gradients = True
				elif not allow_none:
					raise ValueError("{} gradient #{} is None".format(self.name, i))
			self._num_pending_gradients += 1
			
	
	def apply_gradients(self, optimizer, keep_gradients=False):
		raise NotImplementedError("Collect gradients via get_grads_vars(), call optimizer.apply_gradients() ONCE, then clear_gradients().")
		with SAMPLE("{} apply grads".format(self.name)):
			if not self.has_pending_gradients:
				raise RuntimeError("'%s' does not have any recorded gradients to apply."%(self.name,))
			grads_vars = self.get_grads_vars() #zip(self._grad_acc, self.trainable_variables)
			optimizer.apply_gradients(grads_vars)
		if not keep_gradients:
			self.clear_gradients()
	
	def get_grads_vars(self, keep_gradients=True, normalize=False):
		if self.is_frozen_weights or not self.has_pending_gradients:
			#raise RuntimeError("'%s' does not have any recorded gradients."%(self.name,))
			return tuple()
		
		#DEBUG
		#normalize=True
		
		gradients = self.get_pending_gradients(normalize=normalize) #copy of gradient accumulator
		variables = self.trainable_variables
		grads_vars = tuple(zip(gradients, variables))
		
		# can clear gradients here as we return a copy
		if not keep_gradients:
			self.clear_gradients()
		
		return grads_vars
	
	def clear_gradients(self):
		with SAMPLE("{} clear grads".format(self.name)):
			#https://stackoverflow.com/questions/46772685/how-to-accumulate-gradients-in-tensorflow
			for _, acc in self._grad_acc.items():
				acc.assign(tf.zeros_like(acc))
			self._pending_gradients = False
			self._num_pending_gradients = 0
	
	@property
	def has_pending_gradients(self):
		return self._pending_gradients and self._num_pending_gradients>0
	@property
	def num_pending_gradients(self):
		return self._num_pending_gradients
	
	def _get_gradient_normalization_factor(self):
		return tf.constant((1/self._num_pending_gradients) if self._num_pending_gradients>0 else 1, dtype=tf.float32)
	
	def get_pending_gradients(self, normalize=False):
		if normalize:
			s = self._get_gradient_normalization_factor()
			return [tf.identity(_)*s for _ in self._get_gradient_accumulators()]
		else:
			return [tf.identity(_) for _ in self._get_gradient_accumulators()]
	
	def get_pending_gradients_summary(self, normalize=False):
		s = 1.0/self._num_pending_gradients if normalize and self._num_pending_gradients>0 else 1.0
		return [[tf.reduce_mean(_).numpy()*s, tf.reduce_max(tf.abs(_)).numpy()*s] for _ in self._get_gradient_accumulators()]
	
	def get_weights_summary(self):
		return [[tf.reduce_mean(_).numpy(), tf.reduce_max(tf.abs(_)).numpy()] for _ in self.trainable_variables]
	
	def compute_regularization_gradients(self, weight=1.0, add_gradients=False):
		weights = self.trainable_variables
		with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
			tape.watch(weights)
			#loss = tf.reduce_mean([tf.reduce_mean(tf.nn.l2_loss(var)) for var in weights]) * weight
			loss = tf.reduce_sum([tf.reduce_sum(tf.nn.l2_loss(var)) for var in weights]) * weight
		grads = tape.gradient(loss, weights)
		if add_gradients:
			self.add_gradients(grads)
		return grads
	
	def summary(self, print_fn=print):
		self._model.summary(print_fn=print_fn)
	
	def save(self, path):
		model_path = path
		path = path.replace("_model.h5",".json")
		self._model.save(model_path)
		
		cnfg = munch.Munch()
		cnfg._config = self.get_config()
		
		with open(path, "w") as config_file:
			json.dump(cnfg, config_file, sort_keys=True, indent=2)
		
	
	@classmethod
	def load(cls, path, **override_kwargs):
		with open(path, "r") as config_file:
			cnfg = json.load(config_file)
		cnfg = munch.munchify(cnfg)
		cnfg._config.update(override_kwargs)
		
		net = cls.from_config(cnfg._config)
		net.load_weights(path.replace(".json", "_model.h5"))
		
		return net
	
	def get_weights(self):
		return self._model.get_weights()
	
	def copy_weights_from(self, other):
		assert isinstance(other, DeferredBackpropNetwork), "can only copy weights from other GrowingUNet."
		
		try:
			self._model.set_weights(other._model.get_weights())
		except Exception as e:
			LOG.error("Failed to copy weights from %s to %s", other.name, self.name)
			raise e
	
	def save_weights(self, path):
		self._model.save_weights(path)
	
	def load_weights(self, model_path, by_name=True):
		self._model.load_weights(model_path, by_name=by_name)
	
	def get_config(self):
		return {"name": self.name}
		
	@classmethod
	def from_config(cls, config_dict):
		return cls(**config_dict)

class LiftingNetwork(DeferredBackpropNetwork):
	def __init__(self, input_shape, output_shape, name="LiftingNetwork"):
		assert isinstance(input_shape, (list, tuple)) and len(input_shape)==3 #HWC
		assert isinstance(output_shape, (list, tuple)) and len(output_shape)==4 #DHWC
		self.__input_shape = input_shape
		self.__output_shape = output_shape
		
		conv2d_channels = [32,64] #OUTput channels
		dense_sizes = [1024,1024,512,1024,1024]
		conv3d_channels = [64,32,16] #INput channels
		conv3d_channels.append(output_shape[-1])
		dense_sizes.append(np.prod(output_shape[:-1]).tolist()*conv3d_channels[0])
		
		x = tf.keras.layers.Input(shape=input_shape, name=name+"_input")
		inp = x
		for c in conv2d_channels:
			x = tf.keras.layers.Conv2D(c, 3, padding="same")(x)
			x = tf.keras.layers.ReLU()(x)
		
		x = tf.keras.layers.Flatten()(x)
		for dense_size in dense_sizes:
			x = tf.keras.layers.Dense(dense_size)(x)
			x = tf.keras.layers.ReLU()(x)
		
		x = x = tf.keras.layers.Reshape(output_shape[:-1] + [conv3d_channels[0]])(x)
		for c in conv3d_channels[1:]:
			x = tf.keras.layers.Conv3D(c, 3, padding="same")(x)
			x = tf.keras.layers.ReLU()(x)
		
		model = tf.keras.Model(inputs=[inp], outputs=[x])
		super().__init__(model, name)
	
	def __call__(self, inputs):
		if isinstance(inputs, tf.Tensor):
			inputs = [inputs]
		for i in range(len(inputs)):
			inp = inputs[i]
			rank = tf.rank(inp).numpy()
			if rank==5:
				inp = tf.transpose(inp, (0,2,3,1,4))
				shape = shape_list(inp)
				assert shape[-2]==1 #debug
				inp = tf.reshape(inp, shape[:-2] + [shape[-2]*shape[-1]])
				inputs[i] = inp
			elif not rank==4:
				raise RuntimeError
		
		assert len(inputs)==1
		assert shape_list(inputs[0])[1:]==self.__input_shape, "invaldi input shape. is: %s, required: %s"%(shape_list(inputs[0]), self.__input_shape)
		
		return self._model(inputs)

def _check_make_list(value, length, _type):
	if isinstance(value, (list, tuple)):
		if not len(value)==length: raise ValueError("Expected %s to have length %d."%(value, length))
		if any(not isinstance(v, _type) for v in value): raise TypeError("Expected %s to have type %s, but has types %s."%(value, _type.__name__, [type(_).__name__ for _ in value]))
		return list(value)
	else:
		if not isinstance(value, _type): raise TypeError("Expected %s to have type %s, but has type %s."%(value, _type.__name__, type(value).__name__))
		return [value]*length

class GrowingUNet(DeferredBackpropNetwork):
	@staticmethod
	def get_max_levels(resolution, scale_factor, min_size=1, allow_padded=False):
		# maximum number of levels the GrowingUNet can have with the current resolution
		# that does not cause rounding errors when down- and up-scaling in the network
		if isinstance(resolution, Iterable):
			return min(GrowingUNet.get_max_levels(_, scale_factor=scale_factor, min_size=min_size, allow_padded=allow_padded) for _ in resolution)
		l = 1
		while (resolution%scale_factor == 0) or (allow_padded and resolution>=min_size):
			if allow_padded:
				resolution = int(math.ceil(resolution/scale_factor))
			else:
				resolution //= scale_factor
			if resolution<min_size:
				break
			l +=1
		return l
	"""
		# https://machinelearningmastery.com/how-to-implement-progressive-growing-gan-models-in-keras/
	U-Net model that can fade in upper layers
		down branch, up branch, skip connections
		weight sharing
		new-level - input fade, prev-level - skip fade
	def __init__(self, dimension=2, num_levels=3, level_scale_factor=2, input_channels=3, \
			input_levels=2, create_inputs=True, input_blocks=["C:1-1"], share_input_layer=True, \
			down_mode="CONV", down_blocks=["C:1-4_s2"], share_down_layer=True, \
			encoder_blocks=["RB:1-3_1-3"], share_encoder=True, \
			decoder_blocks=["RB:1-3_1-3"], share_decoder=True, \
			up_mode="NNSAMPLE", up_blocks=["C:1-4"], share_up_layer=True, \
			skip_merge_mode="CONCAT", \
			output_levels=1, output_blocks=["C:1-1"], share_output_layer=True,, output_activation="none", \
			conv_activation="relu", alpha=0.2, conv_padding="ZERO", \
			name="GrowingUNet", normalization="NONE"):
	"""
	def __init__(self, dimension=2, num_levels=3, level_scale_factor=2, input_channels=3, \
			input_levels=2, create_inputs=True, input_blocks=None, share_input_layer=True, \
			input_conv_filters=1, input_conv_kernel_size=1, \
			down_mode="STRIDED", down_conv_filters=None, down_conv_kernel_size=4, share_down_layer=True, \
			encoder_filters=[[1]], encoder_kernel_sizes=[[3]], \
			encoder_resblocks=None, share_encoder=False, \
			decoder_filters=[[1]], decoder_kernel_sizes=[[3]], \
			decoder_resblocks=None, share_decoder=False, \
			up_mode="NNSAMPLE", up_conv_filters=1, up_conv_kernel_size=4, share_up_layer=True, \
			skip_merge_mode="CONCAT", \
			output_levels=1, output_blocks=None, share_output_layer=True, output_activation="none", \
			output_channels=1, output_conv_kernel_size=1, output_mode="SINGLE", \
			conv_activation="relu", alpha=0.2, conv_padding="ZERO", \
			name="GrowingUNet", normalization="NONE", **kwargs):
		
		self.__models_built = False
		
		#[1,inf)
		self.__config = munch.Munch()
		assert num_levels>0
		self.num_levels = num_levels 
		self.__config.num_levels = num_levels 
		if num_levels>1:
			assert isinstance(level_scale_factor, numbers.Integral) and level_scale_factor>0
		self.level_scale_factor = level_scale_factor
		self.__config.level_scale_factor = level_scale_factor
		
		#self.num_levels if 0 #[1,self.num_levels]
		assert (input_levels==-1) or (input_levels>0 and input_levels<=self.num_levels)
		self.max_input_levels = input_levels 
		self.__config.input_levels = input_levels 
		self.create_inputs = create_inputs
		self.__config.create_inputs = create_inputs
		
		#self.num_levels if 0 #[1,self.num_levels]
		assert output_levels>0 and output_levels<=self.num_levels
		self.max_output_levels = output_levels 
		self.__config.output_levels = output_levels 
		self._active_level = 0
		
		#2 or 3
		assert dimension==2 or dimension==3 or dimension=="LIFTING"
		self.dim = dimension
		self.__config.dimension = dimension
		assert input_channels>0
		self.input_channels = input_channels
		self.__config.input_channels = input_channels
		
		assert conv_padding in ["ZERO", "MIRROR"]
		self.padding = conv_padding #SAME MIRROR
		self.__config.conv_padding = conv_padding #SAME MIRROR
		
		
		# - Input -
		assert isinstance(share_input_layer, bool)
		self.share_input_layer = share_input_layer
		self.__config.share_input_layer = share_input_layer
		self.input_blocks = None
		if input_blocks is not None:
			self.input_blocks = normalize_block_config(input_blocks, num_levels=self.num_levels, is_shared=self.share_input_layer)
		else:
			raise NotImplementedError("input_conv_filters and input_conv_kernel_size are deprecated, use input_blocks instead.")
			# if self.share_input_layer:
				# assert isinstance(input_conv_filters, numbers.Integral) and input_conv_filters>0
				# self.input_conv_filters = [input_conv_filters]*self.num_levels
				# assert isinstance(input_conv_kernel_size, (numbers.Integral, tuple))
				# self.input_conv_kernel_size = [input_conv_kernel_size]*self.num_levels
			# else:
				# self.input_conv_filters = _check_make_list(input_conv_filters, self.num_levels, numbers.Integral)
				# self.input_conv_kernel_size = _check_make_list(input_conv_kernel_size, self.num_levels, numbers.Integral)
		self.__config.input_blocks = self.input_blocks
		# self.__config.input_conv_filters = input_conv_filters
		# self.__config.input_conv_kernel_size = input_conv_kernel_size
		self._input_conv_layers = {}
		
		
		# - Encoder downsampling -
		assert isinstance(share_down_layer, bool)
		self.share_down_layer = share_down_layer
		self.__config.share_down_layer = share_down_layer
		assert down_mode in ["NONE", "AVGPOOL", "MAXPOOL", "STRIDED"]
		self.down_mode = down_mode # AVGPOOL, MAXPOOL, STRIDED
		if self.share_down_layer:
			self.down_conv_filters = [down_conv_filters]*self.num_levels if down_conv_filters is not None else None
			assert isinstance(down_conv_kernel_size, (numbers.Integral, tuple))
			self.down_conv_kernel_size = [down_conv_kernel_size]*self.num_levels
		else:
			self.down_conv_filters = _check_make_list(down_conv_filters, self.num_levels, numbers.Integral) if down_conv_filters is not None else None
			self.down_conv_kernel_size = _check_make_list(down_conv_kernel_size, self.num_levels, numbers.Integral)
		self.__config.down_conv_filters = down_conv_filters
		self.__config.down_conv_kernel_size = down_conv_kernel_size
		self._down_conv_layers = {}
		
		self.level_input_weights = {}
		self._down_merge_layers = {}
		
		
		# - Encoder -
		assert isinstance(share_encoder, bool)
		self.share_encoder = share_encoder
		self.__config.share_encoder = share_encoder
		self.encoder_resblocks = None
		if encoder_resblocks is not None:
			self.encoder_resblocks = normalize_block_config(encoder_resblocks, num_levels=self.num_levels, is_shared=self.share_encoder) #[encoder_resblocks]*self.num_levels if self.share_encoder else encoder_resblocks
		else:
			#LOG.warning("encoder_filters and encoder_kernel_sizes are deprecated, use encoder_resblocks instead.")
			raise NotImplementedError("encoder_filters and encoder_kernel_sizes are deprecated, use encoder_resblocks instead.")
			# if self.share_encoder:
				# self.encoder_filters = [encoder_filters]*self.num_levels #list(levels) of list(enc_layers) of int
				# self.encoder_kernel_sizes = [encoder_kernel_sizes]*self.num_levels #list(levels) of list(enc_layers) of int
			# else:
				# assert len(encoder_filters)==self.num_levels
				# self.encoder_filters = encoder_filters #list(levels) of list(enc_layers) of int
				# assert len(encoder_kernel_sizes)==self.num_levels
				# self.encoder_kernel_sizes = encoder_kernel_sizes #list(levels) of list(enc_layers) of int or tuple
		self.__config.encoder_resblocks = self.encoder_resblocks
		# self.__config.encoder_filters = encoder_filters
		# self.__config.encoder_kernel_sizes = encoder_kernel_sizes
		self._encoder_layers = {}
		
		
		# - Lifting -
		if dimension=="LIFTING":
			self.__lifting_renderer = kwargs["lifting_renderer"]
			self._set_lifting_cameras(kwargs["lifting_cameras"])
			self._set_lifting_transform(kwargs["lifting_transform"])
			lifting_shapes, _ = self._compute_lifting_shapes(kwargs["lifting_shape"])
			self.__lifting_shapes = {}
			self.set_lifting_shapes(lifting_shapes)
			self._lifting_layers = {}
		
		
		# - Decoder -
		assert isinstance(share_decoder, bool)
		self.share_decoder = share_decoder
		self.__config.share_decoder = share_decoder
		
		self.decoder_resblocks = None
		if decoder_resblocks is not None:
			self.decoder_resblocks = normalize_block_config(decoder_resblocks, num_levels=self.num_levels, is_shared=self.share_decoder) #[decoder_resblocks]*self.num_levels if self.share_decoder else decoder_resblocks
		else:
			raise NotImplementedError("decoder_filters and decoder_kernel_sizes are deprecated, use decoder_resblocks instead.")
			# if self.share_decoder:
				# self.decoder_filters = [decoder_filters]*self.num_levels #list(levels) of list(dec_layers) of int
				# self.decoder_kernel_sizes = [decoder_kernel_sizes]*self.num_levels #list(levels) of list(dec_layers) of int or tuple
			# else:
				# assert len(decoder_filters)==self.num_levels
				# self.decoder_filters = decoder_filters #list(levels) of list(dec_layers) of int
				# assert len(decoder_kernel_sizes)==self.num_levels
				# self.decoder_kernel_sizes = decoder_kernel_sizes #list(levels) of list(dec_layers) of int or tuple
		self.__config.decoder_resblocks = self.decoder_resblocks
		# self.__config.decoder_filters = decoder_filters
		# self.__config.decoder_kernel_sizes = decoder_kernel_sizes
		self._decoder_layers = {}
		
		self._decoder_residual_layers = {}
		self.decoder_residual_weights = {}
		
		
		# - Decoder upsampling -
		assert isinstance(share_up_layer, bool)
		self.share_up_layer = share_up_layer
		self.__config.share_up_layer = share_up_layer
		assert up_mode in ["NNSAMPLE", "LINSAMPLE", "STRIDED", "NNSAMPLE_CONV", "LINSAMPLE_CONV"]
		self.up_mode = up_mode # NNSAMPLE, LINSAMPLE, STRIDED
		self.__config.up_mode = up_mode
		if self.share_up_layer:
			assert up_conv_filters>0
			self.up_conv_filters = [up_conv_filters]*self.num_levels
			assert isinstance(up_conv_kernel_size, (numbers.Integral, tuple))
			self.up_conv_kernel_size = [up_conv_kernel_size]*self.num_levels
		else:
			self.up_conv_filters = _check_make_list(up_conv_filters, self.num_levels, numbers.Integral)
			self.up_conv_kernel_size = _check_make_list(up_conv_kernel_size, self.num_levels, numbers.Integral)
		self.__config.up_conv_filters = up_conv_filters
		self.__config.up_conv_kernel_size = up_conv_kernel_size
		self._up_conv_layers = {}
		
		assert skip_merge_mode in ["CONCAT", "WSUM", "SUM"]
		self.skip_merge_mode = skip_merge_mode #CONCAT, WSUM
		self.__config.skip_merge_mode = skip_merge_mode
		self.level_skip_weights = {}
		self._up_merge_layers = {}
		
		
		# - Output -
		assert isinstance(share_output_layer, bool)
		self.share_output_layer = share_output_layer
		self.__config.share_output_layer = share_output_layer
		self.output_blocks = None
		if output_blocks is not None:
			self.output_blocks = normalize_block_config(output_blocks, num_levels=self.num_levels, is_shared=self.share_output_layer)
		#else:
		#	LOG.warning("output_channels and output_conv_kernel_size are deprecated, use output_blocks instead.")
		if self.share_output_layer:
			assert isinstance(output_conv_kernel_size, (numbers.Integral, tuple))
			self.output_conv_kernel_size = [output_conv_kernel_size]*self.num_levels
		else:
			#assert len(output_conv_kernel_size)==self.num_levels
			self.output_conv_kernel_size = _check_make_list(output_conv_kernel_size, self.num_levels, numbers.Integral)
		assert isinstance(output_channels, numbers.Integral) and output_channels>0
		self.output_channels = output_channels
		self._output_conv_layers = {}
		self.output_activation = output_activation
		assert output_mode in ["SINGLE", "RESIDUAL", "RESIDUAL_WEIGHTED"]
		self.output_mode = output_mode
		self.__config.output_blocks = self.output_blocks
		self.__config.output_channels = output_channels
		self.__config.output_conv_kernel_size = output_conv_kernel_size
		self.__config.output_activation = output_activation
		self.__config.output_mode = output_mode
		
		self.__output_slice_args = None
		
		self._enc_output = kwargs.get("enc_outputs", False)
		if self._enc_output:
			pass
		
		self.activation = conv_activation
		self.alpha = alpha
		self.__config.conv_activation = conv_activation
		self.__config.alpha = alpha
		
		assert normalization in ["NONE", "LAYER", "LAYER_LATE", "LN", "LNL"]
		self.normalization = normalization
		self.__config.normalization = normalization
		
		assert isinstance(name, str)
		self.name = name
		self.__config.name = name
		
		self._check_config()
		self.last_iteration = -1
		self.set_grow_intervals([])
		self.can_grow = False
		self.input_merge_weight_schedule = None #lambda it: 0
		self.skip_merge_weight_schedule = None #lambda it: 0
		self.train_top_level_only_schedule = None #lambda it: False
		self.train_mode = "ALL"
		self._current_train_mode = self.train_mode
		
		self._build_models()
		
		#self._grad_acc = [tf.Variable(tf.zeros_like(tvar), trainable=False) for tvar in self.trainable_variables] persistent causes issues with growth. TODO: maybe in self.set_active_level ?
		#self._build_gradient_accumulators()
		self._clear_gradient_accumulators()
		self.clear_gradients()
	
	def set_input_merge_weight(self, weight, level):
		#assert 0<=level and level<(self.num_levels-1)
		if not level in self.level_input_weights: raise IndexError("Level %d has no input merging stage with weight."%(level,))
		if not (0<=weight and weight<=1): raise ValueError("Input merge weight has to be in [0,1].")
		# 0: only direct input from same level
		# 1: only downscaled input from upper level
		self.level_input_weights[level].assign(weight)
	def get_input_merge_weight(self, level):
		if not level in self.level_input_weights: raise IndexError("Level %d has no input merging stage with weight."%(level,))
		return self.level_input_weights[level].numpy()
	
	def set_skip_merge_weight(self, weight, level):
		#assert 0<level and level<self.num_levels
		if self.skip_merge_mode=="SUM": raise ValueError("SUM skip merging has no weights")
		if not level in self.level_skip_weights: raise IndexError("Level %d has no skip merging stage with weight."%(level,))
		if self.skip_merge_mode=="WSUM":
			if not (0<=weight and weight<=1): raise ValueError("Skip merge weight for WSUM has to be in [0,1].")
			# 0: only upscaled input from lower level
			# 1: only input from skip from same level
		self.level_skip_weights[level].assign(weight)
	def get_skip_merge_weight(self, level):
		if self.skip_merge_mode=="SUM": raise ValueError("SUM skip merging has no weights")
		if not level in self.level_skip_weights: raise IndexError("Level %d has no skip merging stage with weight."%(level,))
		return self.level_skip_weights[level].numpy()
	
	def set_output_residual_weight(self, weight, level):
		#assert 0<level and level<self.num_levels
		if self.output_mode in ["SINGLE", "RESIDUAL"]: raise ValueError("SINGLE and RESIDUAL output modes have no weights")
		if not level in self.decoder_residual_weights: raise IndexError("Level %d has no residual stage with weight."%(level,))
		if self.output_mode=="RESIDUAL_WEIGHTED":
			if not (0<=weight and weight<=1): raise ValueError("Output residual weight for RESIDUAL_WEIGHTED has to be in [0,1].")
			# 0: only output from current level
			# 1: only upscaled output from lower level
		self.decoder_residual_weights[level].assign(weight)
	def get_output_residual_weight(self, level):
		if self.output_mode in ["SINGLE", "RESIDUAL"]: raise ValueError("SINGLE and RESIDUAL output modes have no weights")
		if not level in self.decoder_residual_weights: raise IndexError("Level %d has no residual stage with weight."%(level,))
		return self.decoder_residual_weights[level].numpy()
	
	def set_train_mode(self, mode, schedule=None):
		assert mode in ["ALL", "TOP", "TOP_DEC"]
		self.train_mode = mode
		self.train_top_level_only_schedule = schedule
		if self.train_top_level_only_schedule is None:
			self._current_train_mode = self.train_mode
		#self._build_gradient_accumulators()
	
	def set_level_lr(self, learning_rate, level):
		raise NotImplementedError
		assert 0<level and level<self.num_levels
		self._learning_rates[level].assing(learning_rate)
	
	# --- Lifting ---
	
	@property
	def is_lifting(self):
		return self.dim=="LIFTING"
	@property
	def input_dim(self):
		return 2 if self.is_lifting else self.dim
	@property
	def output_dim(self):
		return 3 if self.is_lifting else self.dim
	
	def set_lifting_shapes(self, shapes):
		#if self.__models_built:
			#raise RuntimeError("Networks are already set up.")
			# TODO: allow to change shapes of built networks?
		
		assert isinstance(shapes, (list,tuple))
		assert len(shapes) == self.num_levels
		assert all(isinstance(shape, (list,tuple)) and len(shape)==3 for shape in shapes)
		assert all(isinstance(dim, numbers.Integral) for shape in shapes for dim in shape)
		
		for level in range(self.num_levels-1):
			assert shapes[level+1]==[dim*self.level_scale_factor for dim in shapes[level]], "shapes do not match leve scale factor"
		
		lifting_shapes = {}
		for level in range(self.num_levels):
			lifting_shapes[level] = copy.copy(shapes[level])
		
		updated_lifting_shape = False
		for level in range(self.num_levels):
			if level not in self.__lifting_shapes or not self.__lifting_shapes[level]==lifting_shapes[level]:
				updated_lifting_shape = True
		
		self.__lifting_shapes = lifting_shapes
		if self.__models_built:
			self.__update_lifting_layers()
		
		if updated_lifting_shape:
			LOG.debug("Updated lifting shapes to %s, updated layers: %s", self.__lifting_shapes, self.__models_built)
		else:
			LOG.debug("Set lifting shapes to %s, updated layers: %s", self.__lifting_shapes, self.__models_built)
	
	def _get_lifting_shape(self, level):
		if not level in self.__lifting_shapes:
			raise KeyError
		return self.__lifting_shapes[level]
	
	def _compute_lifting_shapes(self, shape, min_size=2, max_levels=None):
		assert isinstance(shape, (list,tuple)) and len(shape)==3 and all(isinstance(_, numbers.Integral) for _ in shape), str(shape)
		if max_levels is None:
			max_levels = self.num_levels
		level = min(max_levels, GrowingUNet.get_max_levels(shape, scale_factor=self.level_scale_factor, min_size=min_size, allow_padded=True)) - 1
		
		div = self.level_scale_factor**level
		padded_lifting_size = [next_div_by(_, div) for _ in shape]
		output_slice_args = {"size":tuple(shape), "begin":tuple((p-s)//2 for p, s in zip(padded_lifting_size, shape))}
		
		lifting_shapes = []
		for l in range(self.num_levels):
			factor = self.level_scale_factor**(l - level)
			lifting_shapes.append([int(_*factor) for _ in padded_lifting_size])
		
		return lifting_shapes, output_slice_args
	
	def _set_lifting_transform(self, grid_transform):
		self.__lifting_transform = grid_transform.copy_no_data()
	
	def _get_lifting_transform(self, level):
		shape = self._get_lifting_shape(level)
		t = self.__lifting_transform.copy_no_data()
		t.grid_size = shape
		return t
	
	def _set_lifting_cameras(self, lifting_cameras):
		self.__lifting_cameras = lifting_cameras
	
	def _get_lifting_cameras(self, level):
		return self.__lifting_cameras
	
	# --- Level and Growing ---
	
	def get_active_level(self):
		return self._active_level
	
	def set_active_level(self, level):
		if not (0<=level and level<self.num_levels): raise IndexError("Can't set level %d active: GrowingUNet has only %d levels"%(level, self.num_levels,))
		old_level = self._active_level
		self._active_level = level
		#if not old_level==level:
		#	self._build_gradient_accumulators()
	
	def set_active_level_from_grid_size(self, grid_size, min_size, lifting_size=None):
		assert isinstance(grid_size, (list, tuple)) and len(grid_size)==self.input_dim and all(isinstance(_, numbers.Integral) for _ in grid_size)
		assert isinstance(min_size, numbers.Integral)
		assert self.share_input_layer
		input_factor = self._get_input_blocks_scale_factor(0)
		u_grid_size = [int(math.ceil(_/input_factor)) for _ in grid_size]
		level = min(self.num_levels, GrowingUNet.get_max_levels(u_grid_size, scale_factor=self.level_scale_factor, min_size=min_size, allow_padded=True)) - 1
		
		self.__output_slice_args = None
		if self.is_lifting:
			# assert isinstance(lifting_size, (list, tuple)) and len(lifting_size)==self.output_dim and all(isinstance(_, numbers.Integral) for _ in lifting_size)
			# level_out = min(self.num_levels, GrowingUNet.get_max_levels(lifting_size, scale_factor=self.level_scale_factor, min_size=min_size, allow_padded=True)) - 1
			# level = min(level, level_out)
			
			# raise NotImplementedError("compute new output shapes")
			# div = self.level_scale_factor**level
			# padded_lifting_size = [next_div_by(_, div) for _ in lifting_size]
			# self.__output_slice_args = {"size":tuple(lifting_size), "begin":tuple((p-s)//2 for p, s in zip(padded_lifting_size, lifting_size))}
			
			# lifting_shapes = []
			# for l in range(self.num_levels):
				# factor = self.level_scale_factor**(l - level)
				# lifting_shapes.append([int(_*factor) for _ in padded_lifting_size])
			
			lifting_shapes, self.__output_slice_args = self._compute_lifting_shapes(lifting_size, min_size, level+1)
			
			self.set_lifting_shapes(lifting_shapes)
		
		if not level == self.get_active_level():
			LOG.debug("Upated active level of '%s' to %d for shape %s (%s / %d) (min=%d, lifting=%s)", self.name, level, u_grid_size, grid_size, input_factor, min_size, lifting_size)
		else:
			LOG.debug("Set active level of '%s' to %d for shape %s (%s / %d) (min=%d, lifting=%s)", self.name, level, u_grid_size, grid_size, input_factor, min_size, lifting_size)
		
		self.set_active_level(level)
	
	def __len__(self):
		return self.num_levels
	
	def grow(self):
		if self._active_level>=(self.num_levels-1):
			raise RuntimeError("GrowingUNet is already at max size.")
		self.set_active_level(self._active_level+1)
	
	def set_grow_intervals(self, intervals):
		assert isinstance(intervals, (list, tuple))
		#assert len(intervals)==(self.num_levels-1)
		self.grow_intervals = intervals
		if False: #always start from smallest
			self.grow_iterations = [0] + np.cumsum(self.grow_intervals, dtype=np.int32).tolist()
		else: #always end with largest
			self.grow_iterations = [0]*(self.num_levels - len(self.grow_intervals)) + np.cumsum(self.grow_intervals, dtype=np.int32).tolist()
		self.can_grow = True
		
	
	def level_start_iteration(self, level):
		if level<len(self.grow_iterations):
			return self.grow_iterations[level]
		else:
			return self.grow_iterations[-1]
			
		
	def _get_grow_level(self, iteration):
		level = 0
		while level<min(self.num_levels, len(self.grow_iterations)) and self.level_start_iteration(level)<=iteration: level +=1
		return level -1
	
	def step(self, iteration):
		#raise NotImplementedError("WIP")
		rebuild_grad_acc = False
		current_level = self._get_grow_level(iteration) #TODO
		
		if self.last_iteration == iteration or not self.can_grow:
			# could call step() multiple times (e.g. from different grid handler objects) without adverse effects
			return
		
		if (current_level>0) and (self.train_top_level_only_schedule is not None):
			train_mode = self.train_mode if bool(self.train_top_level_only_schedule(iteration - self.level_start_iteration(self.current_level))) else "ALL"
			if self._current_train_mode != train_mode:
				LOG.debug("GrowingUNet: train mode changed from %s to %s", self._current_train_mode, train_mode)
				self._current_train_mode = train_mode
				rebuild_grad_acc = True
		
		# check growth schedule, grow if necessary
		if current_level!=self.current_level:
			LOG.info("GrowingUNet: set active level to %d in interation %d.", current_level, iteration)
			self.set_active_level(current_level)
			rebuild_grad_acc = False #already done in self.set_active_level
		
		# set weights by schedule
		for level in range(current_level+1):
			level_start_iteration = self.level_start_iteration(level)
			if (level in self.level_input_weights) and (self.input_merge_weight_schedule is not None):
				self.set_input_merge_weight(self.input_merge_weight_schedule(iteration - level_start_iteration), level)
			if (self.skip_merge_mode!="SUM") and (level>0) and (self.skip_merge_weight_schedule is not None):
				self.set_skip_merge_weight(self.skip_merge_weight_schedule(iteration - level_start_iteration), level)
			#self.set_level_lr(self.level_lr_schedule(iteration - level_start_iteration), level)
		
		self.last_iteration = iteration
		
		#if rebuild_grad_acc:
		#	self._build_gradient_accumulators()
	
	def _check_config(self):
		#if (self.skip_merge_mode=="CONCAT" and self.share_decoder):
		#	LOG.info("padding lowest level decoder input")
		pass
	
	# --- Setup ---
	
	def _build_models(self):
		LOG.debug("GrowingUNet: building %d models...", self.num_levels)
		self.models = {}
		for level in range(self.num_levels):
			try:
				self.models[level] = self._build_model(level)
			except Exception as e:
				LOG.error("Failed to build GrowingUNet level %d", level)
				raise e
		self.__models_built = True
	
	def _build_model(self, level):
		LOG.debug("GrowingUNet: building level %d model", level)
		inputs = []#{l: self._get_input_block(l) for l in range(level - self.max_input_levels + 1, level+1)}
		inp_convs = {}
		num_input_levels = (level + 1) if self.max_input_levels==-1 else self.max_input_levels
		for l in range(max(level - num_input_levels + 1, 0), level+1):
			inp_convs[l], inp = self._get_input_block(l)
			inputs.append(inp)
		inputs.reverse()
		
		enc_outputs = {}
		# build uppermost encoder
		enc_outputs[level] = self._add_encoder_block(inp_convs[level], level)
		
		#build lower encoders
		# with inputs
		for l in range(level-1, -1, -1):
			if self.down_mode=="NONE":
				x = inp_convs[l]
			else:
				#downscale
				LOG.debug("Add down block level %d %s in model level %d", l, shape_list(enc_outputs[level]), level)
				x = self._add_down_block(enc_outputs[l+1], l)
				
				#add input
				if l>(level - num_input_levels):
					x = self._add_encoder_input_merge_block(inp_convs[l], x, l)
			
			#add encoder stack
			enc_outputs[l] = self._add_encoder_block(x, l)
		
		if self.is_lifting:# lifting
			lift_outputs = {}
			for l in range(level+1):
				lift_outputs[l] = self._add_lifting_block(enc_outputs[l], l)
			enc_outputs = lift_outputs
		
		dec_outputs = {}
		#build lowermost decoder
		if self.skip_merge_mode=="CONCAT" and self.share_decoder:
			dec_outputs[0] = self._add_decoder_block(ChannelPadding((self.up_conv_filters[0],0))(enc_outputs[0]), 0)
		else:
			dec_outputs[0] = self._add_decoder_block(enc_outputs[0], 0)
		
		up_outputs = {}
		#build higher decoders
		for l in range(1, level+1):
			#upscale
			x = self._add_up_block(dec_outputs[l-1], l)
			up_outputs[l] = x
			
			#add skip connection
			x = self._add_decoder_input_merge_block(x, enc_outputs[l], l)
			
			#add decoder stack
			dec_outputs[l] = self._add_decoder_block(x, l, up_output=up_outputs[l])
			# if self.output_mode=="RESIDUAL":
				# dec_outputs[l] = tf.keras.layers.Add()([self._add_decoder_block(x, l), up_outputs[l]])
			# elif self.output_mode=="SINGLE":
				# dec_outputs[l] = self._add_decoder_block(x, l)
			# else: raise ValueError("Unknown output_mode '{}'".format(output_mode))
		
		
		#build outputs
		#outputs = {l: self._add_output_block(dec_outputs[l],l) for l in range(level - self.max_output_levels + 1, level+1)}
		output_levels = list(range(level - self.max_output_levels + 1, level+1))
	#	if self.output_mode == "RESIDUAL":
	#		outputs = []
	#		current_output = dec_outputs[0]
	#		for l in range(level+1):
	#			if l > 0:
	#				current_output = tf.keras.layers.Add()([dec_outputs[l], (tf.keras.layers.UpSampling2D if self.dim==2 else tf.keras.layers.UpSampling3D)(self.level_scale_factor)(current_output)])
	#			if l in output_levels:
	#				outputs.append(self._add_output_block(current_output,l))
	#	elif self.output_mode == "SINGLE":
		outputs = [self._add_output_block(dec_outputs[l],l) for l in output_levels]
	#	else: raise ValueError("Unknown output_mode '{}'".format(output_mode))
		
		#outputs = [self._add_output_block(dec_outputs[l],l) for l in range(level - self.max_output_levels + 1, level+1)]
		outputs.reverse()
		
		if self._enc_output:
			outputs.extend(enc_outputs[l] for l in range(level+1))
		
		return tf.keras.Model(inputs=inputs, outputs=outputs)
	
	def __get_layer_name(self, level=0, block_name="", layer_type="", layer_idx=0):
		return "{name}_l{level:d}_{block}_{type}{idx:d}".format(name=self.name, block=block_name, level=level, idx=layer_idx, type=layer_type)
	
	def _get_input_layers(self, level):
		level = 0 if self.share_input_layer else level
		return self._input_conv_layers.get(level, None)
	
	def _get_input_block(self, level):
		input_shape = tuple([None]*self.input_dim + [self.input_channels])
		x = tf.keras.layers.Input(shape=input_shape, name=self.__get_layer_name(level, "input", "input"))
		inp = x
		conv_args = {}
		l = 0 if self.share_input_layer else level
		if l not in self._input_conv_layers:
			if self.input_blocks is not None:
				inp_layers = []
				for i, l_str in enumerate(self.input_blocks[l]):
					idx = l_str.find(":") + 1
					inp_layers.append(layer_from_string(l_str[:idx] + "%dD_"%self.input_dim + l_str[idx:], name=self.__get_layer_name(l, "inp", "block", i), \
						activation=self.activation, alpha=self.alpha, normalization=self.normalization, padding=self.padding))
				self._input_conv_layers[l] = inp_layers
			else:
				self._input_conv_layers[l] = [ConvLayerND(self.input_dim, self.input_conv_filters[level], self.input_conv_kernel_size[level], 1, activation=self.activation, alpha=self.alpha, \
					padding=self.padding, normalization=self.normalization, name=self.__get_layer_name(l, "input", "conv"))]
		for L in self._input_conv_layers[l]:
			x = L(x)
		return x, inp
	
	def _get_input_blocks_scale_factor(self, level):
		layers = self._get_input_layers(level)
		if layers is None:
			raise RuntimeError("'%s' has no input layers for level %d"%(self.name, level))
		strides = []
		for layer in layers:
			stride = layer.stride
			if isinstance(stride, (list,tuple)):
				if not all(stride[0]==_ for _ in stride):
					raise ValueError("Uniform strides required for input scale factor.")
				stride = stride[0]
			strides.append(stride)
		return np.prod(strides).tolist()
	
	def _get_down_layers(self, level):
		level = 0 if self.share_down_layer else level
		return self._down_conv_layers.get(level, None)
	
	def _add_down_block(self, upper_input, level):
		if self.down_mode=="STRIDED":
			l = 0 if self.share_down_layer else level
			if l not in self._down_conv_layers:
				self._down_conv_layers[l] = ConvLayerND(self.input_dim, filters=self._get_encoder_input_channels(level), kernel_size=self.down_conv_kernel_size[level], stride=self.level_scale_factor, \
					activation=self.activation, alpha=self.alpha, padding=self.padding, normalization=self.normalization, name=self.__get_layer_name(l, "down", "conv"))
			x = self._down_conv_layers[l](upper_input)
		elif self.down_mode=="AVGPOOL":
			x = (tf.keras.layers.AveragePooling2D if self.input_dim==2 else tf.keras.layers.AveragePooling3D)(self.level_scale_factor)(upper_input)
		elif self.down_mode=="MAXPOOL":
			x = (tf.keras.layers.MaxPooling2D if self.input_dim==2 else tf.keras.layers.MaxPooling3D)(self.level_scale_factor)(upper_input)
		else:
			raise ValueError("Unknown downscale mode '%s'"%(self.down_mode,))
		return x
	
	def _add_encoder_input_merge_block(self, inp, down_inp, level):
		if level not in self._down_merge_layers:
			L = WeightedSum(0.0)
			self._down_merge_layers[level] = L
			self.level_input_weights[level] = L.alpha
		x = self._down_merge_layers[level]([inp,down_inp])
		return x
	
	def _get_lifting_layer(self, level):
		return self._lifting_layers.get(level, None)
		
	def _add_lifting_block(self, encoder_output, level):
		x = encoder_output
		l = level
		if l not in self._lifting_layers:
			self._lifting_layers[l] = UnprojectionLayer(self._get_lifting_transform(l), self._get_lifting_cameras(l), self.__lifting_renderer, name=self.__get_layer_name(l, "lifting", "unprojection"))
		x = self._lifting_layers[l](x)
		return x
	
	def __update_lifting_layers(self):
		for level in range(self.num_levels):
			L = self._get_lifting_layer(level)
			assert isinstance(L, UnprojectionLayer)
			shape = self._get_lifting_shape(level)
			L.set_output_shape(shape)
	
	def _get_encoder_layers(self, level):
		level = 0 if self.share_encoder else level
		return self._encoder_layers.get(level, None)
	
	def _add_encoder_block(self, encoder_input, level):
		x = encoder_input
		l = level
		if self.share_encoder:
			l = 0
		if l not in self._encoder_layers:
			enc_layers = []
			if self.encoder_resblocks is not None:
				for i, l_str in enumerate(self.encoder_resblocks[l]):
					#enc_layers.append(ResBlock.from_string("%dD_"%self.input_dim + rb_str, name=self.__get_layer_name(l, "enc", "rb", i)))
					idx = l_str.find(":") + 1
					enc_layers.append(layer_from_string(l_str[:idx] + "%dD_"%self.input_dim + l_str[idx:], name=self.__get_layer_name(l, "enc", "block", i), \
						activation=self.activation, alpha=self.alpha, normalization=self.normalization, padding=self.padding))
			else:
				for i, (filters, kernel_size) in enumerate(zip(self.encoder_filters[level], self.encoder_kernel_sizes[level])):
					L = ConvLayerND(self.input_dim, filters, kernel_size, 1, activation=self.activation, alpha=self.alpha, padding=self.padding, normalization=self.normalization, \
						name=self.__get_layer_name(l, "enc", "conv", i))
					enc_layers.append(L)
			self._encoder_layers[l] = enc_layers
		
		enc_layers = self._encoder_layers[l]
		for L in enc_layers:
			x = L(x)
		return x
	
	def _get_decoder_layers(self, level):
		level = 0 if self.share_decoder else level
		return self._decoder_layers.get(level, None)
	
	def _add_decoder_block(self, encoder_output, level, up_output=None):
		x = encoder_output
		l = 0 if self.share_decoder else level
		if l not in self._decoder_layers:
			dec_layers = []
			if self.decoder_resblocks is not None:
				for i, l_str in enumerate(self.decoder_resblocks[l]):
					#dec_layers.append(ResBlock.from_string("%dD_"%self.output_dim + rb_str, name=self.__get_layer_name(l, "dec", "rb", i)))
					idx = l_str.find(":") + 1
					dec_layers.append(layer_from_string(l_str[:idx] + "%dD_"%self.output_dim + l_str[idx:], name=self.__get_layer_name(l, "dec", "block", i), \
						activation=self.activation, alpha=self.alpha, normalization=self.normalization, padding=self.padding))
			else:
				for i, (filters, kernel_size) in enumerate(zip(self.decoder_filters[level], self.decoder_kernel_sizes[level])):
					L = ConvLayerND(self.output_dim, filters, kernel_size, 1, activation=self.activation, alpha=self.alpha, padding=self.padding, normalization=self.normalization, \
						name=self.__get_layer_name(l, "dec", "conv", i))
					dec_layers.append(L)
			self._decoder_layers[l] = dec_layers
			
			
			if self.output_mode=="SINGLE" or up_output is None:
				self._decoder_residual_layers[l] = None
			elif self.output_mode=="RESIDUAL":
				self._decoder_residual_layers[l] = tf.keras.layers.Add()
			elif self.output_mode=="RESIDUAL_WEIGHTED":
				L = WeightedSum(0.5)
				self._decoder_residual_layers[l] = L
				self.decoder_residual_weights[l] = L.alpha
			else: raise ValueError("Unknown output_mode '{}'".format(output_mode))
			
		dec_layers = self._decoder_layers[l]
		for L in dec_layers:
			x = L(x)
		if self._decoder_residual_layers[l] is not None:
			x = self._decoder_residual_layers[l]([x, up_output])
		return x
	
	def _get_up_layers(self, level):
		level = 0 if self.share_decoder else level
		return self._up_conv_layers.get(level, None)
	
	def _add_up_block(self, lower_input, level):
		assert level>0
		conv_args = {}
		l = 0 if self.share_up_layer else level
		
		if self.up_mode=="STRIDED":
			raise NotImplementedError
		else:
			if self.up_mode in ["NNSAMPLE", "NNSAMPLE_CONV"]:
				x = (tf.keras.layers.UpSampling2D if self.output_dim==2 else tf.keras.layers.UpSampling3D)(self.level_scale_factor)(lower_input)
			elif self.up_mode in ["LINSAMPLE", "LINSAMPLE_CONV"]:
				if self.output_dim==2:
					raise NotImplementedError("Linear upsampling only implemented for 3D.")
				output_shape = [int(_*self.level_scale_factor) for _ in shape_list(lower_input)[-4:-1]]
				x = GridSamplingLayer(output_shape)(lower_input)
			else:
				raise ValueError("Unknown upscale mode '%s'"%(self.up_mode,))
			if self.up_mode in ["NNSAMPLE_CONV", "LINSAMPLE_CONV"]:
				if l not in self._up_conv_layers:
					self._up_conv_layers[l] = ConvLayerND(self.output_dim, self.up_conv_filters[level], self.up_conv_kernel_size[level], 1, activation=self.activation, alpha=self.alpha, \
						padding=self.padding, normalization=self.normalization, name=self.__get_layer_name(l, "up", "conv"))
				x = self._up_conv_layers[l](x)
		return x
	
	def _add_decoder_input_merge_block(self, up_inp, skip_inp, level):
		if self.skip_merge_mode=="CONCAT":
			if level not in self.level_skip_weights:
				#self.level_skip_weights[level] = tf.keras.backend.variable(0.0, dtype="float32", name="skip_weight")
				#self._up_merge_layers[level] = tf.keras.layers.Lambda(lambda t: t * self.level_skip_weights[level], name=self.__get_layer_name(level, "skip", "weighting"))
				self._up_merge_layers[level] = ScalarMul(0.0, name=self.__get_layer_name(level, "skip", "weighting"))
				self.level_skip_weights[level] = self._up_merge_layers[level].alpha
				#self.level_skip_weights[level] = skip_weight
			#skip_weight = self.level_skip_weights[level]
			y = self._up_merge_layers[level](skip_inp) #tf.keras.layers.Multiply()([skip_inp, [skip_weight]])
			x = tf.keras.layers.Concatenate(axis=-1)([up_inp,y])
		elif self.skip_merge_mode=="WSUM":
			if level not in self._up_merge_layers:
				L = WeightedSum(0.0)
				self._up_merge_layers[level] = L
				self.level_skip_weights[level] = L.alpha
			L = self._up_merge_layers[level]
			x = L([up_inp, skip_inp])
		elif self.skip_merge_mode=="SUM":
			if level not in self._up_merge_layers:
				L = tf.keras.layers.Add()
				self._up_merge_layers[level] = L
			L = self._up_merge_layers[level]
			x = L([up_inp, skip_inp])
		else:
			raise ValueError("Unknown skip connection merge mode '%s'"%(self.skip_merge_mode,))
		return x
	
	def _get_output_layers(self, level):
		level = 0 if self.share_output_layer else level
		return self._output_conv_layers.get(level, None)
	
	def _add_output_block(self, decoder_output, level):
		x = decoder_output
		l = 0 if self.share_output_layer else level
		if l not in self._output_conv_layers:
			outp_layers = []
			if self.output_blocks is not None:
				for i, l_str in enumerate(self.output_blocks[l]):
					idx = l_str.find(":") + 1
					outp_layers.append(layer_from_string(l_str[:idx] + "%dD_"%self.output_dim + l_str[idx:], name=self.__get_layer_name(l, "outp", "block", i), \
						activation=self.activation, alpha=self.alpha, normalization=self.normalization, padding=self.padding))
			
			if self.output_conv_kernel_size[level]>0:
				outp_layers.append(ConvLayerND(self.output_dim, self.output_channels, self.output_conv_kernel_size[level], 1, activation=self.output_activation, \
					alpha=self.alpha, padding=self.padding, name=self.__get_layer_name(l, "output", "conv")))
			self._output_conv_layers[l] = outp_layers
		for L in self._output_conv_layers[l]:
			x = L(x)
		return x
	
	def _get_encoder_input_channels(self, level):
		if self.down_conv_filters is None:
			level = 0 if self.share_input_layer else level
			if self.input_blocks is not None:
				return self._input_conv_layers[level][-1].num_output_channels()
			else:
				return self.input_conv_filters[level]
		else:
			return self.down_conv_filters[level]
		#return self.input_conv_filters[level] if self.down_conv_filters is None else self.down_conv_filters[level]
	def _get_encoder_output_channels(self, level):
		if self.share_encoder:
			level = 0
		return self._encoder_layers[level][-1].num_output_channels()
		#return self.encoder_filters[level][-1]
	def _get_decoder_input_channels(self, level):
		if self.skip_merge_mode in ["WSUM", "SUM"] or level==0:
			return self._get_encoder_output_channels(level)
		else:
			return self._get_encoder_output_channels(level) + (self.up_conv_filters[level] if self.up_mode=="STRIDED" else self._get_decoder_output_channels(level))
	def _get_decoder_output_channels(self, level):
		return self.decoder_filters[level][-1]
	
	def _create_scaled_inputs(self, base_input):
		inputs = [base_input]
		for i in range(1, self.num_inputs):
			inputs.append(self._scale_tensor_down(inputs[i-1], self.level_scale_factor))
		return inputs
	
	def _scale_tensor_down(self, tensor, scale):
		if self.input_dim==2:
			return tf.nn.avg_pool(tensor, (1,scale,scale,1), (1,scale,scale,1), "SAME")
		elif self.input_dim==3:
			return tf.nn.avg_pool3d(tensor, (1,scale,scale,scale,1), (1,scale,scale,scale,1), "SAME")
		else:
			raise ValueError
		#return tf_image_resize_mip(tensor, size, mip_bias=0.5, method=tf.image.ResizeMethod.BILINEAR)
	
	@property
	def num_inputs(self):
		return (self._active_level+1) if self.max_input_levels==-1 else min(self.max_input_levels, self._active_level+1)
	@property
	def num_outputs(self):
		return (self._active_level+1) if self.max_output_levels==-1 else min(self.max_output_levels, self._active_level+1)
	
	def get_scale(self, level=None):
		if level is None:
			level = self.current_level
		return self.level_scale_factor ** level
	
	def check_inputs(self, inputs):
		if not isinstance(inputs, (list, tuple)) or not all(isinstance(inp, tf.Tensor) for inp in inputs):
			raise TypeError("Inputs must be list of tf.Tensor")
		if not len(inputs)==self.num_inputs:
			raise ValueError("Expected %d inputs for '%s', got %d"%(self.num_inputs, self.name, len(inputs)))
		last_shape = None
		for i, inp in enumerate(inputs):
			shape = shape_list(inp)
			if not len(shape)==(self.input_dim+2):
				raise ValueError("Input %d of '%s' has wrong rank. expected %d, is %d %s"%(i, self.name, self.input_dim+2, len(shape), shape))
			c = shape[-1]
			if c != self.input_channels:
				raise ValueError("Input %d of '%s' has %d channels, but %d are required."%(i, self.name, c, self.input_channels))
			if last_shape is not None and not last_shape==[_*self.level_scale_factor for _ in shape[-(self.input_dim+1):-1]]:
				raise ValueError("Input %d of '%s' has shape %s, but %s is required for scale factor %d."%(i, self.name, shape[-(self.input_dim+1):-1], [_//self.level_scale_factor for _ in last_shape]))
			last_shape = shape[-(self.input_dim+1):-1]
	
	def _get_current_padding_scale_factor(self, level=None):
		if level is None: level = self.get_active_level()
		input_factor = self._get_input_blocks_scale_factor(level)
		unet_factor = self.level_scale_factor**(level)
		return input_factor * unet_factor
	
	def _get_padded_input_shape(self, input_shape, level=None):
		assert isinstance(input_shape, (list, tuple)) and len(input_shape)==3 and all(isinstance(_, numbers.Integral) for _ in input_shape)
		level = level if level is not None else self.get_active_level()
		scale_factor = self._get_current_padding_scale_factor(level)
		if scale_factor==1:
			return list(input_shape)
		return [next_div_by(_, scale_factor) for _ in input_shape]
	
	def _pad_inputs(self, inputs):
		assert len(inputs)==1
		level = self.get_active_level()
		scale_factor = self._get_current_padding_scale_factor(level)
		if scale_factor==1:
			return inputs
		inp, pad = tf_pad_to_next_div_by(inputs[0], div=scale_factor, pad_axes=list(range(1,self.input_dim+1)), return_paddings=True)
		slice_args = {"begin": [_[0] for _ in pad], "size": shape_list(inputs[0])}
		return [inp], slice_args
	
	def _cut_outputs(self, outputs, begin, size):
		# need to adjust for potentially different amount of channels
		
		if isinstance(outputs, (list, tuple)):
			output_list = True
			outputs = outputs[0]
		else:
			output_list = False
		shape = shape_list(outputs)
		assert len(shape)==5
		
		size = list(size)
		if len(size)==5:
			size[-1] = shape[-1]
		elif len(size)==3:
			size = [shape[0]] + size + [shape[-1]]
		else:
			raise ValueError
		
		begin = list(begin)
		if len(begin)==5:
			pass
		elif len(begin)==3:
			begin = [0] + begin + [0]
		else:
			raise ValueError
		
		outputs = tf.slice(outputs, begin=begin, size=size)
		if not has_shape(outputs, size):
			raise ValueError("Expected cut output of active level %d to have shape %s, got %s"%(self.get_active_level(), size, shape_list(outputs),))
		if output_list:
			outputs = [outputs]
		
		return outputs
	
	def check_outputs(self, outputs):
		if isinstance(outputs, tf.Tensor):
			outputs = [outputs]
		if not isinstance(outputs, (list, tuple)) or not all(isinstance(outp, tf.Tensor) for outp in outputs):
			raise TypeError("Internal: Outputs must be list of tf.Tensor")
		if not len(outputs)==self.num_outputs:
			raise ValueError("Internal: Expected %d outputs for '%s', got %d"%(self.num_outputs, self.name, len(outputs)))
		
		if self.is_lifting:
			level = self.get_active_level()
			lifting_shape = self._get_lifting_shape(level)
			if not has_shape(outputs[-1], [None]+lifting_shape+[None]):
				raise ValueError("Expected output of active level %d to have shape %s, got %s"%(level, [None]+lifting_shape+[None], shape_list(outputs[-1]),))
	
	def __call__(self, inputs):
		with SAMPLE("call %s lvl %d"%(self.name[:10], self._active_level)):
			if self.create_inputs:
				assert isinstance(inputs, tf.Tensor)
				inputs = self._create_scaled_inputs(inputs)
			elif isinstance(inputs, tf.Tensor):
				inputs = [inputs]
			LOG.debug("GrowingUNet: called '%s' with inputs: %s", self.name, [shape_list(_) for _ in inputs])
			
			padded_input = False
			if not self.is_lifting:
				self.__output_slice_args = None
			if self.num_inputs==1 and self.num_levels>1 and self.num_outputs==1:
				padded_input = True
				shape1 = shape_list(inputs[0])
				inputs, output_slice_args = self._pad_inputs(inputs)
				level = self.get_active_level()
				if not self.is_lifting and self._get_input_blocks_scale_factor(level)==1:
					self.__output_slice_args = output_slice_args
				shape2 = shape_list(inputs[0])
				LOG.debug("GrowingUNet: padded input of '%s' from %s to %s for %d levels with scale factor %d", self.name, shape1, shape2, self.get_active_level()+1, self.level_scale_factor)
			
			self.check_inputs(inputs)
			#inputs = {l: inputs[self._active_level - l] for l in range(self._active_level, self._active_level-self.num_inputs, -1)}
			with SAMPLE("model"):
				outputs = self.models[self._active_level](inputs)
			
			if self._enc_output:
				enc_outputs = outputs[self.num_outputs:]
				outputs = outputs[:self.num_outputs]
				if len(outputs)==1: outputs = outputs[0]
			#outputs = [v for k,v in sorted(outputs.items(), key=lambda k,v:k, reverse=True)]
			#LOG.info("%s output shape: %s", self.name, shape_list(outputs))
			self.check_outputs(outputs)
			if self.__output_slice_args is not None:
				outputs = self._cut_outputs(outputs, **self.__output_slice_args)
		
		if self._enc_output:
			return outputs, enc_outputs
		else:
			return outputs
	
	def get_layers(self, level):
		layers = []
		
		L = self._get_input_layers(level)
		if L is not None: layers.extend(L)
		L = self._get_down_layers(level)
		if L is not None: layers.append(L)
		
		L = self._get_encoder_layers(level)
		if L is not None: layers.extend(L)
		L = self._get_decoder_layers(level)
		if L is not None: layers.extend(L)
		
		L = self._get_up_layers(level)
		if L is not None: layers.append(L)
		L = self._get_output_layers(level)
		if L is not None: layers.extend(L)
		
		return layers
	
	def get_trainable_variables(self, level):
		layers = self.get_layers(level)
		layer_vars = []
		for L in layers: layer_vars += L.trainable_variables
		return layer_vars
	
	def get_layer_decoder(self, level):
		layers = []
		
		L = self._get_decoder_layers(level)
		if L is not None: layers.extend(L)
		
		L = self._get_up_layers(level)
		if L is not None: layers.append(L)
		L = self._get_output_layers(level)
		if L is not None: layers.extend(L)
		
		return layers
		
	
	def get_trainable_variables_decoder(self, level):
		layers = self.get_layer_decoder(level)
		layer_vars = []
		for L in layers: layer_vars += L.trainable_variables
		return layer_vars
	
	@property
	def current_level(self):
		return self._active_level
	
	@property
	def trainable_variables(self):
		# TODO: return 
		# how to handle weight sharing with different learning rates?
		if self._current_train_mode == "TOP": #top_level_only:
			return self.get_trainable_variables(self._active_level)
		elif self._current_train_mode == "TOP_DEC": #top_level_only:
			return self.get_trainable_variables_decoder(self._active_level)
		elif self._current_train_mode == "ALL":
			return self.models[self._active_level].trainable_variables
		else:
			raise ValueError("Unknown train mode: %s"(self._current_train_mode,))
	
	
	def get_grads_vars_by_level(self, keep_gradients=True, normalize=False):
		active_levels = tuple(range(self._active_level+1))
		if self.is_frozen_weights or not self.has_pending_gradients:
			#raise RuntimeError("'%s' does not have any recorded gradients."%(self.name,))
			return [tuple() for _ in active_levels]
		
		#DEBUG
		#normalize=True
		grads_vars = []
		for level in active_levels:
			vars_level = self.get_trainable_variables(level)
			if normalize:
				s = self._get_gradient_normalization_factor()
				grads_level = [tf.identity(self._get_gradient_accumulator(var))*s for var in vars_level]
			else:
				grads_level = [tf.identity(self._get_gradient_accumulator(var)) for var in vars_level]
				
			grads_vars_level = tuple(zip(grads_level, vars_level))
			grads_vars.append(grads_vars_level)
		
		
		# can clear gradients here as we return a copy
		if not keep_gradients:
			self.clear_gradients()
		
		return grads_vars
	
	# def add_gradients(self, grads, allow_none=False):
		# with SAMPLE("{} add grads".format(self.name)):
			# for i, (acc, grad) in enumerate(zip(self._grad_acc, grads)):
				# if grad is not None:
					# acc.assign_add(grad)
					# self._pending_gradients = True
				# elif not allow_none:
					# raise ValueError("{} gradient #{} is None.".format(self.name, i))
	
	# def apply_gradients(self, optimizer, keep_gradients=False):
		# with SAMPLE("{} apply grads".format(self.name)):
			# if not self.has_pending_gradients:
				# raise RuntimeError("GrowingUNet '%s' does not have any recorded gradients to apply."%(self.name,))
			# grads_vars = zip(self._grad_acc, self.trainable_variables)
			# optimizer.apply_gradients(grads_vars)
		# if not keep_gradients:
			# self.clear_gradients()
	
	# def _build_gradient_accumulators(self):
		# self._grad_acc = [tf.Variable(tf.zeros_like(tvar), trainable=False) for tvar in self.trainable_variables]
		# self._pending_gradients = False
		
	
	# def clear_gradients(self):
		# with SAMPLE("{} clear grads".format(self.name)):
			# #https://stackoverflow.com/questions/46772685/how-to-accumulate-gradients-in-tensorflow
			# for acc in self._grad_acc:
				# acc.assign(tf.zeros_like(acc))
			# self._pending_gradients = False
	
	# @property
	# def has_pending_gradients(self):
		# return self._pending_gradients
	
	# def get_pending_gradients(self):
		# return [tf.identity(_) for _ in self._grad_acc]
	
	# def get_pending_gradients_summary(self):
		# return [[tf.reduce_mean(_).numpy(), tf.reduce_max(tf.abs(_)).numpy()] for _ in self._grad_acc]
	
	# def get_weights_summary(self):
		# return [[tf.reduce_mean(_).numpy(), tf.reduce_max(tf.abs(_)).numpy()] for _ in self.trainable_variables]
	
	# def compute_regularization_gradients(self, weight=1.0, add_gradients=False):
		# weights = self.trainable_variables
		# with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
			# tape.watch(weights)
			# #loss = tf.reduce_mean([tf.reduce_mean(tf.nn.l2_loss(var)) for var in weights]) * weight
			# loss = tf.reduce_sum([tf.reduce_sum(tf.nn.l2_loss(var)) for var in weights]) * weight
		# grads = tape.gradient(loss, weights)
		# if add_gradients:
			# self.add_gradients(grads)
		# return grads
	
	def summary(self, print_fn=print):
		self.models[self.num_levels-1].summary(print_fn=print_fn)
	
	def save(self, path):
		model_path = path
		path = path.replace("_model.h5",".json")
		self.models[self.num_levels-1].save(model_path)
		
		cnfg = munch.Munch()
		cnfg._config = self.get_config()
		cnfg._state = munch.Munch()
		cnfg._state.level_skip_weights = {l:v.numpy().tolist() for l,v in self.level_skip_weights.items()}
		cnfg._state.level_input_weights = {l:v.numpy().tolist() for l,v in self.level_input_weights.items()}
		cnfg._state.active_level = self._active_level
		#cnfg._state.model_path = model_path
		
		with open(path, "w") as config_file:
			json.dump(cnfg, config_file, sort_keys=True, indent=2)
		
	
	@classmethod
	def load(cls, path, **override_kwargs):
		with open(path, "r") as config_file:
			cnfg = json.load(config_file)
		cnfg = munch.munchify(cnfg)
		cnfg._config.update(override_kwargs)
		net = cls.from_config(cnfg._config)
		
		net_max_level = net.num_levels-1
		net.load_weights(path.replace(".json", "_model.h5"))
		net.set_active_level(min(cnfg._state.active_level, net_max_level))
		for l, v in cnfg._state.level_input_weights.items():
			if int(l)<=net_max_level:
				net.set_input_merge_weight(v,int(l))
		for l, v in cnfg._state.level_skip_weights.items():
			if int(l)<=net_max_level:
				net.set_skip_merge_weight(v,int(l))
		
		return net
	
	def get_weights(self, level=None):
		if level is None:
			level = self.num_levels-1
		return self.models[level].get_weights()
	
	def copy_weights_from(self, other):
		assert isinstance(other, GrowingUNet), "can only copy weights from other GrowingUNet."
		# check compatibility
		assert self.num_levels==other.num_levels
		
		# set weights of models
		for level in range(self.num_levels):
			try:
				self.models[level].set_weights(other.models[level].get_weights())
			except Exception as e:
				LOG.error("Failed to copy weights from level %d of %s to %s", level, other.name, self.name)
				raise e
	
	def save_weights(self, path):
		self.models[self.num_levels-1].save_weights(path)
	
	def load_weights(self, model_path, by_name=True):
		for level, model in self.models.items():
			model.load_weights(model_path, by_name=by_name)
	
	def get_config(self):
		return copy.deepcopy(self.__config)
		
	@classmethod
	def from_config(cls, config_dict):
		return cls(**config_dict)
	
	@staticmethod
	def config_is_level_variable(config):
		if not isinstance(config, dict):
			#assume path
			with open(config, "r") as config_file:
				config = json.load(config_file)
			config = config["_config"]
		config = munch.munchify(config)
		
		return config.share_decoder \
			and config.share_down_layer \
			and config.share_encoder \
			and config.share_input_layer \
			and config.share_output_layer \
			and config.share_up_layer


class SDFDiffAENetwork(DeferredBackpropNetwork):
	def __init__(self, input_channels, name="SDFDiffAENetwork"):
		input_shape = [64,64,input_channels]
		
		x = tf.keras.layers.Input(shape=input_shape, name=name+"_input")
		inp = x
		
		# Encoder
		
		# res=64
		x = tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=2, padding="same")(x)
		x = tf.keras.layers.BatchNormalization(axis=-1)(x)
		x = tf.keras.layers.ReLU()(x)
		# res=32
		x = tf.keras.layers.Conv2D(filters=128, kernel_size=5, strides=2, padding="same")(x)
		x = tf.keras.layers.BatchNormalization(axis=-1)(x)
		x = tf.keras.layers.ReLU()(x)
		# res=16
		x = tf.keras.layers.Conv2D(filters=256, kernel_size=5, strides=2, padding="same")(x)
		x = tf.keras.layers.BatchNormalization(axis=-1)(x)
		x = tf.keras.layers.ReLU()(x)
		# res=8
		x = tf.keras.layers.Flatten()(x)
		x = tf.keras.layers.Dense(1024)(x)
		x = tf.keras.layers.ReLU()(x)
		
		x = tf.keras.layers.Dense(1024)(x)
		x = tf.keras.layers.ReLU()(x)
		
		x = tf.keras.layers.Dense(512)(x)
		x = tf.keras.layers.ReLU()(x)
		
		# Decoder
		
		x = tf.keras.layers.Dense(1024)(x)
		x = tf.keras.layers.ReLU()(x)
		
		x = tf.keras.layers.Dense(1024)(x)
		x = tf.keras.layers.ReLU()(x)
		
		x = tf.keras.layers.Dense(32*4*4*4)(x)
		x = tf.keras.layers.ReLU()(x)
		
		x = tf.keras.layers.Dense(64*8*8*8)(x)
		x = tf.keras.layers.ReLU()(x)
		x = tf.keras.layers.Reshape((8,8,8,64))(x)
		# res=8
		
		# Dense block 1
		d1_0 = x
		x = tf.keras.layers.Conv3DTranspose(filters=32, kernel_size=3, strides=1, padding="same")(x)
		x = tf.keras.layers.BatchNormalization(axis=-1)(x)
		x = tf.keras.layers.ReLU()(x)
		d1_1 = x
		
		x = tf.keras.layers.Concatenate(axis=-1)([d1_0, d1_1])
		x = tf.keras.layers.Conv3DTranspose(filters=32, kernel_size=3, strides=1, padding="same")(x)
		x = tf.keras.layers.BatchNormalization(axis=-1)(x)
		x = tf.keras.layers.ReLU()(x)
		d1_2 = x
		
		x = tf.keras.layers.Concatenate(axis=-1)([d1_0, d1_1, d1_2])
		x = tf.keras.layers.Conv3DTranspose(filters=32, kernel_size=3, strides=1, padding="same")(x)
		x = tf.keras.layers.BatchNormalization(axis=-1)(x)
		x = tf.keras.layers.ReLU()(x)
		d1_3 = x
		
		x = tf.keras.layers.Concatenate(axis=-1)([d1_0, d1_1, d1_2, d1_3])
		x = tf.keras.layers.Conv3DTranspose(filters=64, kernel_size=1, strides=1, padding="same")(x)
		x = tf.keras.layers.BatchNormalization(axis=-1)(x)
		x = tf.keras.layers.ReLU()(x)
		
		# upscale
		x = tf.keras.layers.Conv3DTranspose(filters=32, kernel_size=4, strides=2, padding="same")(x)
		x = tf.keras.layers.BatchNormalization(axis=-1)(x)
		x = tf.keras.layers.ReLU()(x)
		# res=16
		
		# Dense block 2
		d2_0 = x
		x = tf.keras.layers.Conv3DTranspose(filters=16, kernel_size=3, strides=1, padding="same")(x)
		x = tf.keras.layers.BatchNormalization(axis=-1)(x)
		x = tf.keras.layers.ReLU()(x)
		d2_1 = x
		
		x = tf.keras.layers.Concatenate(axis=-1)([d2_0, d2_1])
		x = tf.keras.layers.Conv3DTranspose(filters=16, kernel_size=3, strides=1, padding="same")(x)
		x = tf.keras.layers.BatchNormalization(axis=-1)(x)
		x = tf.keras.layers.ReLU()(x)
		d2_2 = x
		
		x = tf.keras.layers.Concatenate(axis=-1)([d2_0, d2_1, d2_2])
		x = tf.keras.layers.Conv3DTranspose(filters=16, kernel_size=3, strides=1, padding="same")(x)
		x = tf.keras.layers.BatchNormalization(axis=-1)(x)
		x = tf.keras.layers.ReLU()(x)
		d2_3 = x
		
		x = tf.keras.layers.Concatenate(axis=-1)([d2_0, d2_1, d2_2, d2_3])
		x = tf.keras.layers.Conv3DTranspose(filters=32, kernel_size=1, strides=1, padding="same")(x)
		x = tf.keras.layers.BatchNormalization(axis=-1)(x)
		x = tf.keras.layers.ReLU()(x)
		
		# upscale
		x = tf.keras.layers.Conv3DTranspose(filters=16, kernel_size=4, strides=2, padding="same")(x)
		x = tf.keras.layers.BatchNormalization(axis=-1)(x)
		x = tf.keras.layers.ReLU()(x)
		# res=32
		
		# Dense block 3
		d3_0 = x
		x = tf.keras.layers.Conv3DTranspose(filters=8, kernel_size=3, strides=1, padding="same")(x)
		x = tf.keras.layers.BatchNormalization(axis=-1)(x)
		x = tf.keras.layers.ReLU()(x)
		d3_1 = x
		
		x = tf.keras.layers.Concatenate(axis=-1)([d3_0, d3_1])
		x = tf.keras.layers.Conv3DTranspose(filters=8, kernel_size=3, strides=1, padding="same")(x)
		x = tf.keras.layers.BatchNormalization(axis=-1)(x)
		x = tf.keras.layers.ReLU()(x)
		d3_2 = x
		
		x = tf.keras.layers.Concatenate(axis=-1)([d3_0, d3_1, d3_2])
		x = tf.keras.layers.Conv3DTranspose(filters=8, kernel_size=3, strides=1, padding="same")(x)
		x = tf.keras.layers.BatchNormalization(axis=-1)(x)
		x = tf.keras.layers.ReLU()(x)
		d3_3 = x
		
		x = tf.keras.layers.Concatenate(axis=-1)([d3_0, d3_1, d3_2, d3_3])
		x = tf.keras.layers.Conv3DTranspose(filters=16, kernel_size=1, strides=1, padding="same")(x)
		x = tf.keras.layers.BatchNormalization(axis=-1)(x)
		x = tf.keras.layers.ReLU()(x)
		
		x = tf.keras.layers.Conv3DTranspose(filters=8, kernel_size=3, strides=1, padding="same")(x)
		x = tf.keras.layers.BatchNormalization(axis=-1)(x)
		x = tf.keras.layers.ReLU()(x)
		
		# upscale
		x = tf.keras.layers.Conv3DTranspose(filters=1, kernel_size=4, strides=2, padding="same")(x)
		# res=64
		self.output_channels = 1
		
		model = tf.keras.Model(inputs=[inp], outputs=[x])
		super().__init__(model, name)
	
	def __call__(self, inputs):
		
		x = self._model(inputs)
		x = tf.math.sigmoid(x) * 4 - 2
		return x


class SDFDiffRefinerNetwork(DeferredBackpropNetwork):
	def __init__(self, name="SDFDiffRefinerNetwork"):
		input_shape = [64,64,64,1]
		
		x = tf.keras.layers.Input(shape=input_shape, name=name+"_input")
		inp = x
		
		# MaxPool3D gradients are broken in my version...
		pool_fn = tf.keras.layers.AvgPool3D #tf.keras.layers.MaxPool3D
		
		# Down
		# res=64
		l0 = x
		x = tf.keras.layers.Conv3D(filters=32, kernel_size=4, strides=1, padding="same")(x)
		x = tf.keras.layers.BatchNormalization(axis=-1)(x)
		x = tf.keras.layers.LeakyReLU(0.2)(x)
		x = pool_fn((2,2,2))(x)
		# res=32
		l1 = x
		x = tf.keras.layers.Conv3D(filters=64, kernel_size=4, strides=1, padding="same")(x)
		x = tf.keras.layers.BatchNormalization(axis=-1)(x)
		x = tf.keras.layers.LeakyReLU(0.2)(x)
		x = pool_fn((2,2,2))(x)
		# res=16
		l2 = x
		x = tf.keras.layers.Conv3D(filters=128, kernel_size=4, strides=1, padding="same")(x)
		x = tf.keras.layers.BatchNormalization(axis=-1)(x)
		x = tf.keras.layers.LeakyReLU(0.2)(x)
		x = pool_fn((2,2,2))(x)
		# res=8
		l3 = x
		x = tf.keras.layers.Conv3D(filters=256, kernel_size=4, strides=1, padding="same")(x)
		x = tf.keras.layers.BatchNormalization(axis=-1)(x)
		x = tf.keras.layers.LeakyReLU(0.2)(x)
		x = pool_fn((2,2,2))(x)
		# res=4
		l4 = x
		
		#Latent
		x = tf.keras.layers.Flatten()(x)
		x = tf.keras.layers.Dense(2048)(x)
		x = tf.keras.layers.ReLU()(x)
		
		x = tf.keras.layers.Dense(8192*2)(x)
		x = tf.keras.layers.ReLU()(x)
		x = tf.keras.layers.Reshape((4,4,4,256))(x)
		
		# Up
		x = tf.keras.layers.Add()([l4, x])
		x = tf.keras.layers.Conv3DTranspose(filters=128, kernel_size=4, strides=2, padding="same")(x)
		x = tf.keras.layers.BatchNormalization(axis=-1)(x)
		x = tf.keras.layers.ReLU()(x)
		# res=8
		x = tf.keras.layers.Add()([l3, x])
		x = tf.keras.layers.Conv3DTranspose(filters=64, kernel_size=4, strides=2, padding="same")(x)
		x = tf.keras.layers.BatchNormalization(axis=-1)(x)
		x = tf.keras.layers.ReLU()(x)
		# res=16
		x = tf.keras.layers.Add()([l2, x])
		x = tf.keras.layers.Conv3DTranspose(filters=32, kernel_size=4, strides=2, padding="same")(x)
		x = tf.keras.layers.BatchNormalization(axis=-1)(x)
		x = tf.keras.layers.ReLU()(x)
		# res=32
		x = tf.keras.layers.Add()([l1, x])
		x = tf.keras.layers.Conv3DTranspose(filters=1, kernel_size=4, strides=2, padding="same")(x)
		# res=64
		#x = tf.keras.layers.Add()([l1, x])
		
		model = tf.keras.Model(inputs=[inp], outputs=[x])
		super().__init__(model, name)
	
	def __call__(self, inputs):
		
		x = self._model(inputs)
		x = tf.math.sigmoid(x)
		x = (x + inputs) * 0.5
		return x


class RWDensityGeneratorNetwork(DeferredBackpropNetwork):
	def __init__(self, input_channels, w1=0.5, w2=0.5, name="RWDensityGeneratorNetwork"):
		input_shape = [64,64,64,input_channels]
		self._single_view = w2==0
		
		shared_RB0 = ResBlock(dim=3, mid_filters= 8, out_filters= 8, mid0_kernel_size=3, mid1_kernel_size=3, skip_kernel_size=1, stride=1, activation="lrelu", alpha=0.2, name=name+"_shared_RB0")
		shared_RB1 = ResBlock(dim=3, mid_filters= 8, out_filters= 8, mid0_kernel_size=3, mid1_kernel_size=3, skip_kernel_size=0, stride=1, activation="lrelu", alpha=0.2, name=name+"_shared_RB1")
		shared_RB2 = ResBlock(dim=3, mid_filters=16, out_filters=16, mid0_kernel_size=3, mid1_kernel_size=3, skip_kernel_size=1, stride=1, activation="lrelu", alpha=0.2, name=name+"_shared_RB2")
		shared_RB3 = ResBlock(dim=3, mid_filters=16, out_filters=16, mid0_kernel_size=3, mid1_kernel_size=3, skip_kernel_size=0, stride=1, activation="lrelu", alpha=0.2, name=name+"_shared_RB3")
		shared_RB4 = ResBlock(dim=3, mid_filters=32, out_filters=32, mid0_kernel_size=3, mid1_kernel_size=3, skip_kernel_size=1, stride=1, activation="lrelu", alpha=0.2, name=name+"_shared_RB4")
		shared_RB5 = ResBlock(dim=3, mid_filters=32, out_filters=32, mid0_kernel_size=3, mid1_kernel_size=3, skip_kernel_size=0, stride=1, activation="lrelu", alpha=0.2, name=name+"_shared_RB5")
		
		inp1 = tf.keras.layers.Input(shape=input_shape, name=name+"_input1")
		x1 = inp1
		x1 = shared_RB0(x1)
		l0_1 = x1
		x1 = shared_RB1(x1)
		x1 = shared_RB2(x1)
		x1 = shared_RB3(x1)
		x1 = shared_RB4(x1)
		x1 = shared_RB5(x1)
		x1 = ScalarMul(w1, name=name+"_weightEnc1")(x1) #tf.keras.layers.Multiply()([w1, x1])
		
		if not self._single_view:
			inp2 = tf.keras.layers.Input(shape=input_shape, name=name+"_input2")
			inp = [inp1,inp2]
			x2 = inp2
			x2 = shared_RB0(x2)
			l0_2 = x2
			x2 = shared_RB1(x2)
			x2 = shared_RB2(x2)
			x2 = shared_RB3(x2)
			x2 = shared_RB4(x2)
			x2 = shared_RB5(x2)
			x2 = ScalarMul(w2, name=name+"_weightEnc2")(x2) #tf.keras.layers.Multiply()([w2, x2])
		
			x = tf.keras.layers.Concatenate(axis=-1, name=name+"_concatEnc")([x1, x2])
		else:
			inp = [inp1]
			x = x1
		
		x = ResBlock(dim=3, mid_filters=16, out_filters=16, mid0_kernel_size=3, mid1_kernel_size=3, skip_kernel_size=1, stride=1, activation="lrelu", alpha=0.2, name=name+"_RB6")(x)
		x = ResBlock(dim=3, mid_filters=8, out_filters=8, mid0_kernel_size=3, mid1_kernel_size=3, skip_kernel_size=1, stride=1, activation="lrelu", alpha=0.2, name=name+"_RB7")(x)
		
		x = tf.keras.layers.Concatenate(axis=-1, name=name+"_concatSkip")([l0_1, x] if self._single_view else [l0_1, l0_2, x])
		
		x = ResBlock(dim=3, mid_filters=1, out_filters=1, mid0_kernel_size=3, mid1_kernel_size=3, skip_kernel_size=1, stride=1, activation="lrelu", alpha=0.2, name=name+"_RB8")(x)
		
		x = tf.keras.layers.Add(name=name+"_addSkip")([x, ScalarMul(w1, name=name+"_weightInp1")(inp1)] if self._single_view else [x, ScalarMul(w1, name=name+"_weightInp1")(inp1), ScalarMul(w2, name=name+"_weightInp2")(inp2)])
		
		model = tf.keras.Model(inputs=inp, outputs=[x])
		super().__init__(model, name)
	
	def __call__(self, inputs):
		# split concatenated inputs
		if not self._single_view:
			inputs = tf.split(inputs, 2, axis=-1)
		
		x = self._model(inputs)
		return x

class RWVelocityGeneratorNetwork(DeferredBackpropNetwork):
	def __init__(self, dens_channels, unp_channels, use_proxy=False, name="RWVelocityGeneratorNetwork"):
		dens_shape = [64,64,64,dens_channels]
		unp_shape = [64,64,64,unp_channels]
		
		self._use_proxy = use_proxy
		
		if self._use_proxy:
			inp = [
				tf.keras.layers.Input(shape=dens_shape, name=name+"_inputDens0"),
				tf.keras.layers.Input(shape=dens_shape, name=name+"_inputDens1"),
			]
		else:
			inp = [
				tf.keras.layers.Input(shape=dens_shape, name=name+"_inputDens0"),
				tf.keras.layers.Input(shape=unp_shape, name=name+"_inputUnp0"),
				tf.keras.layers.Input(shape=unp_shape, name=name+"_inputUnp1"),
			]
		
		x = tf.keras.layers.Concatenate(axis=-1)(inp)
		
		x = ResBlock(dim=3, mid_filters=16, out_filters=16, mid0_kernel_size=3, mid1_kernel_size=3, skip_kernel_size=1, stride=1, activation="lrelu", alpha=0.2, name=name+"_RB0")(x)
		x = ResBlock(dim=3, mid_filters=32, out_filters=32, mid0_kernel_size=3, mid1_kernel_size=3, skip_kernel_size=1, stride=1, activation="lrelu", alpha=0.2, name=name+"_RB1")(x)
		x = ResBlock(dim=3, mid_filters=48, out_filters=48, mid0_kernel_size=3, mid1_kernel_size=3, skip_kernel_size=1, stride=1, activation="lrelu", alpha=0.2, name=name+"_RB2")(x)
		x = ResBlock(dim=3, mid_filters=16, out_filters=16, mid0_kernel_size=3, mid1_kernel_size=3, skip_kernel_size=1, stride=1, activation="lrelu", alpha=0.2, name=name+"_RB3")(x)
		x = ResBlock(dim=3, mid_filters= 3, out_filters= 3, mid0_kernel_size=3, mid1_kernel_size=3, skip_kernel_size=1, stride=1, activation="lrelu", alpha=0.2, name=name+"_RB4")(x)
		
		model = tf.keras.Model(inputs=inp, outputs=[x])
		super().__init__(model, name)
	
	def __call__(self, inputs):
		# split concatenated inputs
		inputs = tf.split(inputs, 2 if self._use_proxy else 3, axis=-1)
		
		x = self._model(inputs)
		return x

