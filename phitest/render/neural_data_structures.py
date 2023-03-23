
import logging, copy, warnings #, itertools #
from numbers import Number, Integral

import numpy as np
import tensorflow as tf
from .data_structures import DensityGrid, VelocityGrid, State, ResourceCacheDictTF, Sequence
from .vector import GridShape, Vector
from .generator_models import GrowingUNet, DeferredBackpropNetwork
from lib.tf_ops import shape_list, has_shape, has_rank, grad_log, tf_pad_to_next_div_by, tf_norm2
from lib.data import make_sphere_SDF, make_cube_SDF
from .profiling import SAMPLE

LOG = logging.getLogger("NeuralStructs")

def is_None_or_type(value, types):
	return (value is None) or isinstance(value, types)

def _handle_wrong_output_users(output_users, expected_types, name):
	users_type_names = {out_id: [type(_).__name__ for _ in users] for out_id, users in output_users} if output_users is not None else None
	type_names = {out_id: [_.__name__ for _ in types] for out_id, types in expected_types}
	LOG.error("Expected output users of %s to be %s, is %s", name, type_names, users_type_names)
	# assert False, "Expected output users of %s to be %s, is %s"%(name, [_.__name__ for _ in expected_types], [type(_).__name__ for _ in output_users] if output_users is not None else None)

def check_output_users(output_users, expected_types, name):
	output_users = output_users or []
	num_output_ids = len(output_users)
	exp_num_ids = len(expected_types)
	if not (num_output_ids==exp_num_ids and set(output_users.keys())==set(expected_types.keys())):
		_handle_wrong_output_users(output_users, expected_types, name)
		return
	
	for out_id, users in output_users.items():
		types = expected_types[out_id]
		num_output_users = len(users)
		exp_num_users = len(types)
		if not num_output_users==exp_num_users:
			_handle_wrong_output_users(output_users, expected_types, name)
			return
	
		output_users_types = [type(_) for _ in users]
		for t in set(types):
			if not output_users_types.count(t)==types.count(t):
				_handle_wrong_output_users(output_users, expected_types, name)
				return
	
	#LOG.info("Output users of %s are correct: expected %s, is %s", name, [_.__name__ for _ in expected_types], [type(_).__name__ for _ in output_users] if output_users is not None else None)

def input_key(other, output_id):
	return (other, output_id)

from abc import ABC, abstractmethod

class BackpropInterface(ABC):
	@abstractmethod
	def outputs(self):
		raise NotImplementedError
	
	@abstractmethod
	def output(self, output_id):
		raise NotImplementedError
	
	@abstractmethod
	def _register_output_user(self, other, output_id):
		raise NotImplementedError
	
	@abstractmethod
	def _compute_input_grads(self):
		raise NotImplementedError
	
	@abstractmethod
	def _get_input_grad(self, other, output_id):
		raise NotImplementedError
	
	@abstractmethod
	def has_gradients_for(self, other, output_id):
		raise NotImplementedError
	
	@abstractmethod
	def can_backprop(self):
		raise NotImplementedError
	
	@abstractmethod
	def requires_backprop(self):
		raise NotImplementedError

class NeuralGrid(BackpropInterface):
	def __init__(self, inputs, models, device=None):
		assert isinstance(inputs, (list, tuple))
		assert all(isinstance(inp, BackpropInterface) or callable(inp) for inp in inputs)
		assert isinstance(models, (list, tuple))
		assert all(isinstance(m, DeferredBackpropNetwork) for m in models)
		
		self._inputs = copy.copy(inputs)
		self._models = copy.copy(models)
		
		self.__output_users = {"OUTPUT":[]}
		self.__cache = ResourceCacheDictTF(device)
		self.__input_ref_cache = None
	
	def __register_inputs(self):
		for inp, out_id in self._inputs:
			if isinstance(inp, BackpropInterface):
				inp._register_output_user(self, out_id)
	
	def _register_output_user(self, other, output_id):
		if isinstance(other, BackpropInterface) and other not in self.__output_users[output_id]:
			self.__output_users[output_id].append(other)
	
	def __gather_inputs(self):
		return [inp(out_id) if callable(inp) else inp.output(out_id) for inp, out_id in self._inputs]
	
	def _get_input_values(self):
		if self.__input_ref_cache is None:
			self.__register_inputs()
			self.__input_ref_cache = self.__gather_inputs()
		return copy.copy(self.__input_ref_cache)
	
	def _check_inputs(self):
		inputs = self.__gather_inputs()
		assert self.__input_ref_cache is not None
		assert len(self.__input_ref_cache)==len(inputs)
		
		inputs_shape = [shape_list(_) for _ in inputs]
		cache_shape = [shape_list(_) for _ in self.__input_ref_cache]
		assert all(a==b for a,b in zip(inputs_shape, cache_shape)), "%s, %s"%(inputs_shape, cache_shape)
		assert all(tf.reduce_all(tf.equal(i1, i2)).numpy().tolist() for i1,i2 in zip(self.__input_ref_cache, inputs))
		
		#assert all(i1 is i2 for i1,i2 in zip(self.__input_ref_cache, inputs))
	
	def _compute_output(self):
		raise NotImplementedError
		# outp = tf.concat(self._get_input_values(), axis=-1)
		# for model in self._models:
			# outp = model(outp)
		# return outp
	
	@property
	def has_output(self):
		return "output" in self.__cache
	
	def outputs(self):
		if "output" not in self.__cache:
			# outputs = self._compute_output()
			# for out_id, output in outputs.items():
				# self.__cache["output:"+out_id] = output
			self.__cache["output"] = self._compute_output()["OUTPUT"]
		else:
			self._check_inputs()
		return {"OUTPUT": self.__cache["output"]}
	
	def output(self, output_id):
		return self.outputs()[output_id]
	
	def input_index(self, other, output_id):
		t = input_key(other, output_id)
		if t in self._inputs:
			return self._inputs.index(t)
		else:
			raise KeyError
	
	def _get_input_grad_scales(self):
		return {}
	
	def _compute_input_grads(self):
		if all("input_grads_%d"%(i,) not in self.__cache for i in range(len(self._inputs))):
			input_grads = self.__backprop()
			scales = self._get_input_grad_scales()
			for i, grad in enumerate(input_grads):
				if i in scales:
					grad = grad * scales[i]
				self.__cache["input_grads_%d"%(i,)] = grad
		else:
			# check if the input is still valid
			self._check_inputs()
	
	def _get_input_grad(self, other, output_id):
		# get gradients for a specific input
		# check if it is one of the inputs
		#assert other in self._inputs
		idx = self.input_index(other, output_id)
		
		self._compute_input_grads()
		return self.__cache["input_grads_%d"%(idx,)]
	
	def has_gradients_for(self, other, output_id):
		return (input_key(other, output_id) in self._inputs) and (("input_grads_%d"%(self.input_index(other, output_id),) in self.__cache) or self.can_backprop)
	
	@property
	def can_backprop(self):
		# has output gradients available
		#LOG.info("%s can backprop: out grad %s; output users %d, provides grads %s", type(self).__name__, "output_grad" in self.__cache, len(self.__output_users), [_.has_gradients_for(self) for _ in self.__output_users])
		return ("output_grad" in self.__cache) or (len(self.__output_users)>0 and any(len(users)>0 and any(_.has_gradients_for(self, out_id) for _ in users) for out_id, users in self.__output_users.items()))
	
	@property
	def requires_backprop(self):
		return len(self._models)>0 and any(not _.is_frozen_weights for _ in self._models) and any("input_grads_%d"%(i,) in self.__cache for i in range(len(self._inputs))) and self.can_backprop
	
	def _has_output_shape(self, tensor):
		assert self.has_output
		return has_shape(tensor, shape_list(self.output("OUTPUT")))
	
	def add_output_grad(self, grad):
		assert self._has_output_shape(grad)
		if "output_grad" not in self.__cache:
			self.__cache["output_grad"] = grad
		else:
			self.__cache["output_grad"] = self.__cache["output_grad"] + grad
	
	def __gather_output_grads(self):
		
		if self.parent_state.next is not None:
			check_output_users(self.__output_users, {"OUTPUT": [WarpedDensityGrid, NeuralVelocityGrid]}, "WarpedDensityGrid")
		else:
			check_output_users(self.__output_users, {"OUTPUT": []}, "last frame WarpedDensityGrid")
		#assert self.parent_state.next is None or (self.__output_users is not None and len(self.__output_users)==2), "Expected velocity output users of WarpedDensityGrid to be one WarpedDensityGrid and one NeuralVelocityGrid, is %s"%([type(_).__name__ for _ in self.__output_users] if self.__output_users is not None else None)
		
		for out_id, users in self.__output_users.items():
			for other in users:
				other._compute_input_grads()
		
		grads = {}
		for out_id, users in self.__output_users.items():
			grads[out_id] = tf.zeros_like(self.output(out_id))
			for other in users:
				other_grad = other._get_input_grad(self, out_id)
				assert has_shape(other_grad, shape_list(grads[out_id]))
				grads[out_id] = grads[out_id] + other_grad
		
		assert len(self.__output_users)==1 and "OUTPUT" in self.__output_users
		if "output_grad" in self.__cache:
			assert has_shape(self.__cache["output_grad"], shape_list(grads["OUTPUT"]))
			grads["OUTPUT"] = grads["OUTPUT"] + self.__cache["output_grad"]
			del self.__cache["output_grad"]
		
		return grads
	
	def get_variables(self):
		return [model.trainable_variables for model in self._models]
	
	def get_output_variables(self):
		return {"output": self.output("OUTPUT")}
	
	def __backprop(self):
		# should only be called once per iteration
		assert "output" in self.__cache
		self._check_inputs()
		
		output_grads = self.__gather_output_grads()
		
		LOG.debug("Backprop %s", type(self).__name__)
		variables = {"inputs": self._get_input_values(), "model_params": self.get_variables()}
		
		with tf.GradientTape(watch_accessed_variables=False) as tape:
			tape.watch(variables)
			output = self._compute_output()
		
		gradients = tape.gradient(output, variables, output_gradients=output_grads)
		for model, model_grads in zip(self._models, gradients["model_params"]):
			model.add_gradients(model_grads)
		
		return gradients["inputs"]
	
	def clear_cache(self):
		self.__cache.clear()
		self.__output_users = {"OUTPUT":[]}
		self.__input_ref_cache = None

class WarpedDensityGrid(DensityGrid, NeuralGrid):
	def __init__(self, order=1, dt=1.0, clamp="NONE", device=None, scale_renderer=None, var_name="denstiy", is_SDF=False):
		#self.__density = density_input
		#self.__velocity = velocity_input
		
		self.__input_index_density = 0
		self.__input_index_velocity = 1
		NeuralGrid.__init__(self, inputs=[], models=[], device=device)
		# self.__order = order
		# self.__dt = dt
		# self.__clamp = clamp
		self.set_warp_params(order, dt, clamp)
		
		# init DensityGrid
		#super().__init__(self, )
		self.is_var = False
		self._device = device
		self._name = var_name
		self._is_trainable = False
		self._is_SDF = is_SDF
			
		self.scale_renderer = scale_renderer
		self.hull = None
		self.restrict_to_hull = False
		
		self._inflow = None
		
		self.set_density_grad_scale(1)
	
	@property
	def shape(self):
		try:
			shape = self.__density_grid_input.shape
		except IndexError: # inputs not yet set
			shape = self.parent_state.prev.density.shape
		return shape
	
	@property
	def _d(self):
		return self.output("OUTPUT")
	
	@property
	def d(self):
		# the density grid with modifications (inflow) and constraints (hull, non-negativity)
		d = self._d
		if not self._is_SDF:
			d = tf.maximum(d, 0)
		return d
	
	def set_warp_params(self, order, dt=1.0, clamp="NONE"):
		LOG.info("Set WarpedDensityGrid warp: order=%d, dt=%f, clamp=%s", order, dt, clamp)
		self.__order = order
		self.__dt = dt
		self.__clamp = clamp
	
	def __set_inputs(self):
		assert hasattr(self, "parent_state") and self.parent_state is not None and self.parent_state.prev is not None
		ps = self.parent_state.prev
		density_input = ps.density
		velocity_input = ps.velocity
		assert isinstance(density_input, (DensityGrid)) and isinstance(density_input, BackpropInterface)
		assert isinstance(velocity_input, (VelocityGrid)) and isinstance(velocity_input, BackpropInterface)
		self._inputs = [(density_input, "OUTPUT"), (velocity_input, "CENTERED")]
	
	def set_density_grad_scale(self, value):
		self.__density_grad_scale = tf.constant(value, dtype=tf.float32)
	
	def _get_input_grad_scales(self):
		return {self.__input_index_density: self.__density_grad_scale}
	
	@property
	def __density_grid_input(self):
		return self._inputs[self.__input_index_density][0]
	@property
	def __velocity_grid_input(self):
		return self._inputs[self.__input_index_velocity][0]
	
	def _compute_output(self):
		self.__set_inputs()
		with SAMPLE("Warp Denisty Grid"):
			with SAMPLE("get inputs"):
				inputs = self._get_input_values()
				density = inputs[self.__input_index_density]
				velocity = inputs[self.__input_index_velocity]
				velocity_grid = self.__velocity_grid_input
			dens_shape = shape_list(density)[1:-1]
			if not dens_shape == shape_list(velocity)[1:-1]:
				with SAMPLE("scale velocity"):
					velocity = NeuralVelocityGrid.resample_velocity(velocity_grid.scale_renderer, velocity, shape=dens_shape, is_staggered=False, scale_magnitude=True)
			with SAMPLE("warp"):
				density_warped = velocity_grid.warp(density, centered_velocity=velocity, order=self.__order, dt=self.__dt, clamp=self.__clamp)
		return {"OUTPUT": density_warped}
	
	# inherited from DensityGrid
	# def get_output_variables(self):
		# return {"output": self.output()}
	
	def set_output_gradients_for_backprop_accumulate(self, output_gradients, **kwargs):
		self.add_output_grad(output_gradients["density"])
	
	def apply_clamp(self, vmin, vmax):
		pass
	
	def assign(self, d, inflow=None):
		raise TypeError("Can't assign to WarpedDensityGrid")
	
	def clear_cache(self):
		self._inputs = []
		super().clear_cache()
		NeuralGrid.clear_cache(self)

class NeuralDensityGrid(DensityGrid, BackpropInterface):
	def __init__(self, volume_decoder, parent_state, scale_renderer=None, hull=None, inflow=None, inflow_offset=None, inflow_mask=None, device=None, var_name="denstiy", trainable=True, restrict_to_hull=True, \
		step_input_density=[], step_input_density_target=[], step_input_features=[0,1], type_input_features=["TARGET_UNPROJECTION"], base_input="ZERO", is_SDF=False, base_SDF_mode="NONE"):
		self._device = device
		self.cache_output = True
		self.__input_grad_cache = ResourceCacheDictTF(self._device)
		self.clear_cache()
		self.parent_state = parent_state
		self.use_raw_images = True
		self.volume_decoder = volume_decoder
		if hull is not None:
			raise NotImplementedError("NeuralDensityGrid does not support hull")
		self.hull = None
		self.restrict_to_hull = restrict_to_hull
		if inflow is not None:
			raise NotImplementedError("NeuralDensityGrid does not support inflow")
		self._inflow = None
		self.scale_renderer = scale_renderer
		
		# needed for copy to fixed DensityGrid
		self.is_var = False
		self._name = var_name
		self._is_trainable = trainable
		self._is_SDF = is_SDF
		#raise NotImplementedError()
		
		self.recursive_MS = False
		
		assert isinstance(step_input_density, (list, tuple, np.ndarray)) and all(isinstance(_, Integral) for _ in step_input_density)
		if 0 in step_input_density: raise ValueError("Can't use own output as input")
		self.step_input_density = step_input_density
		assert isinstance(step_input_density_target, (list, tuple, np.ndarray)) and all(isinstance(_, Integral) for _ in step_input_density_target)
		self.step_input_density_target = step_input_density_target
		assert isinstance(step_input_features, (list, tuple, np.ndarray)) and all(isinstance(_, Integral) for _ in step_input_features)
		self.step_input_features = step_input_features
		
		# also check requires_parent_state_variables if adding new feature inputs
		assert isinstance(type_input_features, (list, tuple, np.ndarray)) and all(isinstance(_, str) for _ in type_input_features)
		assert all(_ in ["INPUT_IMAGES_UNPROJECTION","INPUT_IMAGES_RAW_UNPROJECTION","INPUT_IMAGES_HULL","ENC3D"] for _ in type_input_features)
		# DEBUG
		assert len(type_input_features)==1 and type_input_features[0]=="ENC3D"
		self.type_input_features = type_input_features
		
		assert base_input in ["ZERO", "INPUT_IMAGES_HULL"] #TARGET_HULL
		self.base_input_features = base_input #previous density input of the lowest resolution
		
		self.norm_input_mode = "NONE" # debug, was GROUP. NONE, SINGLE, GROUP, ALL
		
		LOG.info("Setup NeuralDensityGrid: SDF=%s, use raw=%s, step_input_density=%s, step_input_density_target=%s, step_input_features=%s, type_input_features=%s, base_input_features=%s, norm_input_mode=%s", self._is_SDF, self.use_raw_images, self.step_input_density, self.step_input_density_target, self.step_input_features, self.type_input_features, self.base_input_features, self.norm_input_mode)
		
		
		assert base_SDF_mode in ["NONE", "RESIDUAL", "INPUT_RESIDUAL"]
		self.__use_base_SDF = (base_SDF_mode in ["RESIDUAL", "INPUT_RESIDUAL"]) and self._is_SDF
		self.__input_base_SDF = (base_SDF_mode=="INPUT_RESIDUAL") and self.__use_base_SDF
		self.__base_SDF = None
		self.__base_SDF_strength = 4 #number of SDF border cells
	
	# backprop interface
	def __add_used_input(self, other, output_id):
		self.__inputs_for_backprop.append(input_key(other, output_id))
	
	def __register_inputs(self):
		for inp, out_id in self.__inputs_for_backprop:
			assert isinstance(inp, BackpropInterface)
			inp._register_output_user(self, out_id)
	
	def __gather_inputs(self):
		return [inp.output(out_id) for inp, out_id in self.__inputs_for_backprop]
	
	def outputs(self):
		return {"OUTPUT": self._d}
	
	def output(self, output_id):
		return self.outputs()["OUTPUT"]
	
	def _register_output_user(self, other, output_id):
		assert isinstance(other, BackpropInterface)
		if other not in self.__output_users[output_id]:
			self.__output_users[output_id].append(other)
	
	def _compute_input_grads(self):
		if self.__input_grad_cache.is_empty():
			assert self.__output_gradient_cache is not None
			self.backprop_accumulate(**self.__output_gradient_cache)
			self.__output_gradient_cache = None
	
	def _get_input_grad(self, other, output_id):
		assert not self.__input_grad_cache.is_empty()
		t = input_key(other, output_id)
		idx = self.__inputs_for_backprop.index(t)
		return self.__input_grad_cache["input_grad_%d"%(idx,)]
	
	def has_gradients_for(self, other, output_id):
		return (input_key(other, output_id) in self.__inputs_for_backprop) and (not self.__input_grad_cache.is_empty() or self.can_backprop)
	
	@property
	def can_backprop(self):
		# has output gradients available
		return (self.__output_gradient_cache is not None) or (len(self.__output_users)>0 and any(len(users)>0 and any(_.has_gradients_for(self, out_id) for _ in users) for out_id, users in self.__output_users.items()))
	
	@property
	def requires_backprop(self):
		return (not self.volume_decoder.is_frozen_weights) and self.__input_grad_cache.is_empty() and self.can_backprop
	
	# own methods
	
	@property
	def shape(self):
		if self.__state is None: raise ValueError("Parent state is not set")
		return self.__state.transform.grid_size
	@property
	def parent_state(self):
		return self.__state
	@parent_state.setter
	def parent_state(self, value):
		assert is_None_or_type(value, NeuralState)
		self.__state = value
		
	def _get_base_SDF(self):
		if self.__base_SDF is None or shape_list(self.__base_SDF)[1:-1]!= self.shape:
			shape = np.asarray(self.shape, dtype=np.float32)
			self.__base_SDF = tf.maximum(1, make_cube_SDF(self.shape, np.maximum(0, shape/2 - self.__base_SDF_strength)))
		return self.__base_SDF
	
	def _get_batch_size(self):
		if self.__d is not None:
			return shape_list(self.__d)[0]
		elif self.parent_state is not None:
			return self.parent_state._get_batch_size()
		else:
			raise RuntimeError("NeuralVelocityGrid: Can't determine batch size without cached velocity or parent state.")
			#return self._batch_size
	
	def rescale(self, new_shape):
		raise NotImplementedError
		self.parent_state.transform.grid_size = new_shape
	
	def _get_generator_input(self):
		if "BASE" not in self.__generator_inputs:
			shape = self.shape #self.centered_shape
			with SAMPLE("density base input"):
				inp = []
				for step in self.step_input_density:
					raise RuntimeError("DEBUG")
					state = self._get_state_by_step(step)
					if state is not None:
						if state.density is self: raise RuntimeError("Can't use own output as input")
						inp.append(state.density.scaled(shape, with_inflow=True))
						#inp.append(tf.zeros_like(state.density.scaled(self.centered_shape, with_inflow=True)))
						feature_shape = shape_list(inp[-1])
					else:
						LOG.debug("NeuralDensityGrid of frame %d is missing density input of state %d (step %d)", self.parent_state.frame, self.parent_state.frame+step, step)
						inp.append(tf.zeros(feature_shape))
				
				for step in self.step_input_density_target:
					raise RuntimeError("DEBUG")
					state = self._get_state_by_step(step)
					if state is not None:
						inp.append(state.density_target.scaled(shape, with_inflow=True))
						#inp.append(tf.zeros_like(state.density.scaled(self.centered_shape, with_inflow=True)))
						feature_shape = shape_list(inp[-1])
					else:
						LOG.debug("NeuralDensityGrid of frame %d is missing density target input of state %d (step %d)", self.parent_state.frame, self.parent_state.frame+step, step)
						inp.append(tf.zeros(feature_shape))
				
				for step in self.step_input_features:
					state = self._get_state_by_step(step)
					if state is not None:
						assert self.type_input_features==["ENC3D"]
						inp.append(state.output("OUTPUT"))
						assert has_shape(inp[-1], [None]+shape+[None]), "%s, %s"%(shape_list(inp[-1]), [None]+shape+[None])
						#self.__inputs_for_backprop.append(state)
						self.__add_used_input(state, "OUTPUT")
						#assert feature_shape[-1] == 78, "debug: %s"%(feature_shape,)
					else:
						LOG.debug("NeuralDensityGrid of frame %d is missing feature input of state %d (step %d)", self.parent_state.frame, self.parent_state.frame+step, step)
						inp.append(tf.zeros(feature_shape))
			
			self.__register_inputs()
			
			self.__generator_inputs["BASE"] = inp
		
		return self.__generator_inputs["BASE"]
	
	
	def _get_state_by_step(self, step):
		assert isinstance(step, Integral) and step>=0
		state = self.parent_state
		while step>0:
			if state.next is None:
				#raise AttributeError("Can't get next state")
				return None
			state = state.next
			step -=1
		return state
	
	def _get_generator_input_MS(self, scale):
		#raise NotImplementedError("TODO")
		recurrent_input_dens = False
		warp_recurrent_dens = False #try this?
		# TODO: recursive_MS input generation; possibly multi-scale input/features provided by parent state
		if scale not in self.__generator_inputs:
			if self._is_top_scale(scale) or self._parent_provides_input_scale(scale):
				scale_shape = self.shape_of_scale(scale) #self.centered_shape
				with SAMPLE("density top MS input"):
					inp = []
					for step in self.step_input_density:
						raise RuntimeError("DEBUG")
						state = self._get_state_by_step(step)
						if state is not None:
							if state.density is self: raise RuntimeError("Can't use own output as input")
							inp.append(state.density.scaled(scale_shape, with_inflow=True))
							#inp.append(tf.zeros_like(state.density.scaled(self.centered_shape, with_inflow=True)))
							feature_shape = shape_list(inp[-1])
						else:
							LOG.debug("NeuralDensityGrid of frame %d is missing density input of state %d (step %d)", self.parent_state.frame, self.parent_state.frame+step, step)
							inp.append(tf.zeros(feature_shape))
					
					for step in self.step_input_density_target:
						raise RuntimeError("DEBUG")
						state = self._get_state_by_step(step)
						if state is not None:
							inp.append(state.density_target.scaled(scale_shape, with_inflow=True))
							#inp.append(tf.zeros_like(state.density.scaled(self.centered_shape, with_inflow=True)))
							feature_shape = shape_list(inp[-1])
						else:
							LOG.debug("NeuralDensityGrid of frame %d is missing density target input of state %d (step %d)", self.parent_state.frame, self.parent_state.frame+step, step)
							inp.append(tf.zeros(feature_shape))
					
					for step in self.step_input_features:
						state = self._get_state_by_step(step)
						if state is not None:
							#if self.use_raw_images:
							# inp.append(state.get_volume_features(types=self.type_input_features, shape=scale_shape)) #shape=scale_shape
							# feature_shape = shape_list(inp[-1])
							# raise NotImplementedError("TODO:")
							assert self.type_input_features==["ENC3D"]
							inp.append(state.output("OUTPUT"))
							assert has_shape(inp[-1], [None]+scale_shape+[None]), "%s, %s"%(shape_list(inp[-1]), [None]+scale_shape+[None])
							#self.__inputs_for_backprop.append(state)
							self.__add_used_input(state, "OUTPUT")
							#assert feature_shape[-1] == 78, "debug: %s"%(feature_shape,)
						else:
							LOG.debug("NeuralDensityGrid of frame %d is missing feature input of state %d (step %d)", self.parent_state.frame, self.parent_state.frame+step, step)
							inp.append(tf.zeros(feature_shape))
						
					
					if recurrent_input_dens and not self.parent_state.prev.density.has_MS_output:
						if self.parent_state.prev is not None:
							inp.append(self.parent_state.prev.density.d_MS(scale))
				
				self.__register_inputs()
				
				#if self.is_staggered:
				#	inp = [self._scalar_centered_to_staggered(_) for _ in inp]
			else:
				inp = self._get_generator_input_MS(self._get_larger_scale(scale))
				inp = [self._scale_input_down(_, scale) for _ in inp]
			
			if recurrent_input_dens:
				if self.parent_state.prev is None:
					inp.append(tf.zeros(self.shape_of_scale(scale)))
				elif self.parent_state.prev.density.has_MS_output:
					inp.append(self.parent_state.prev.density.d_MS(scale))
				else:
					raise RuntimeError
			
				
				
			
			self.__generator_inputs[scale] = inp
		
		return self.__generator_inputs[scale]
	
	def _get_top_scale(self):
		return self.recursive_MS_scales[self.recursive_MS_max_level]
	def _get_top_active_scale(self):
		return self.recursive_MS_scales[self.recursive_MS_current_level]
	def _is_top_scale(self, scale):
		return scale==self.recursive_MS_scales[self.recursive_MS_current_level]
	def _parent_provides_input_scale(self, scale):
		return self.recursive_MS_direct_input
		#return False
	def _get_larger_scale(self, scale):
		return self.recursive_MS_scales[self.recursive_MS_scales.index(scale)+1]
	
	
	def set_recursive_MS(self, num_scales: Integral, scale_factor: Number, shared_decoder=True, train_mode="ALL", shapes=None, shape_cast_fn=round, as_residual=True, direct_input=False):
		self.recursive_MS = True
		self.recursive_MS_direct_input = direct_input
		self.recursive_MS_residual = as_residual #True
		self.recursive_MS_max_level = num_scales-1
		self.recursive_MS_current_level = self.recursive_MS_max_level
		assert isinstance(scale_factor, Number)
		if scale_factor<=0: LOG.warning("recursive MS scale factor should be positive. is: %s", scale_factor)
		self.recursive_MS_scale_factor = scale_factor
		self.recursive_MS_scale_magnitude = True
		self.recursive_MS_scales = list(range(num_scales))
		self.recursive_MS_shape_cast_fn = shape_cast_fn
		if shapes is None:
			self._recursive_MS_shapes = list(reversed([[int(shape_cast_fn(_/(self.recursive_MS_scale_factor**scale))) for _ in self.shape] for scale in self.recursive_MS_scales]))
		else:
			assert isinstance(shapes, list) and len(shapes)==num_scales
			assert all(isinstance(shape, list) for shape in shapes)
			assert all(has_shape(_, shape_list(self.shape)) for _ in shapes)
			if not all(shapes[-1]==self.shape): LOG.warning("Maximum resolution of shapes provided for recursive MS {} does not match main resolution of this NeuralDensityGrid {}".format(shapes[-1], self.shape))
			if not all(all(shapes[i]<=shapes[+1]) for i in range(len(shapes)-1)): LOG.warning("Shapes provided for recursive MS are not growing in all dimensions: {}".format(shapes))
			self._recursive_MS_shapes = copy.deepcopy(shapes)
		#self.set_recursive_MS_level(num_scales, scale_factor, shapes, shape_cast_fn)
		
		self.recursive_MS_input_dens = True
		if shared_decoder:
			assert isinstance(self.volume_decoder, (DeferredBackpropNetwork, tf.keras.Model))
		else:
			assert isinstance(self.volume_decoder, list) \
				and len(self.volume_decoder)==(self.recursive_MS_max_level+1) \
				and all(isinstance(_, (DeferredBackpropNetwork, tf.keras.Model)) for _ in self.volume_decoder)
		self.recursive_MS_shared_decoder = shared_decoder
		assert train_mode in ["ALL", "TOP"]
		self.recursive_MS_train_mode = train_mode #"ALL" if self.recursive_MS_shared_decoder else "TOP" #ALL, TOP; no effect with shared decoder
		self.cache_MS = True
		
		LOG.info("Setup %srecursive multi-scale for NeuralDensityGrid of frame %s with %d scales (%s), factor %f, %s decoders, training '%s'. The output scales are%s cached.", \
			"residual " if self.recursive_MS_residual else "", \
			"?" if self.parent_state is None else self.parent_state.frame, self.num_recursive_MS_scales, self.recursive_MS_shapes, self.recursive_MS_scale_factor, \
			"shared" if self.recursive_MS_shared_decoder else "separate", self.recursive_MS_train_mode, "" if self.has_MS_output else " not")
		
		if self.recursive_MS_shared_decoder and self.recursive_MS_train_mode!="ALL":
			LOG.warning("recursive_MS_train_mode 'TOP' has no effect on shared decoder.")
	
	@property
	def recursive_MS_shapes(self):
		if self.recursive_MS_shared_decoder:
			return list(reversed([[int(self.recursive_MS_shape_cast_fn(_/(self.recursive_MS_scale_factor**scale))) for _ in self.shape] for scale in self.gen_current_MS_scales()]))
		else:
			return self._recursive_MS_shapes
	
	def set_recursive_MS_level(self, level: Integral, copy_weights: bool = False):
		if not self.recursive_MS: raise RuntimeError("recursive_MS was not set up.")
		if level<0 or self.recursive_MS_max_level<level: raise ValueError("Level must be between 0 and {}, is: {}".format(self.recursive_MS_max_level, level))
		if copy_weights:
			if not self.recursive_MS_current_level==(level-1): raise RuntimeError("copy_weights require to grow recursion by exactly 1.")
			if self.recursive_MS_shared_decoder: raise ValueError("copy_weights requires separate decoders.")
			self.volume_decoder[level].copy_weights_from(self.volume_decoder[self.recursive_MS_current_level])
		self.recursive_MS_current_level = level
		#if scale_factor is None: scale_factor=self.recursive_MS_scale_factor
	
	def gen_current_MS_shapes(self):
		for level in range(self.recursive_MS_current_level+1):
			yield self.recursive_MS_shapes[level]
	def gen_current_MS_scales(self):
		for level in range(self.recursive_MS_current_level+1):
			yield self.recursive_MS_scales[level]
	def gen_current_trainable_MS_scales(self):
		for level in range(0 if self.recursive_MS_train_mode=="ALL" else self.recursive_MS_current_level, self.recursive_MS_current_level+1):
			yield self.recursive_MS_scales[level]
	
	def has_same_MS_shapes(self, other):
		if not self.is_MS or not isinstance(other, DensityGrid) or not other.is_MS:
			return True
		self_shapes = list(self.gen_current_MS_shapes())
		other_shapes = list(other.gen_current_MS_shapes())
		if len(self_shapes)!=len(other_shapes):
			return False
		return all( all(self_dim==other_dim for self_dim, other_dim in zip(self_shape, other_shape)) for self_shape, other_shape in zip(self_shapes, other_shapes))
	
	@property
	def num_recursive_MS_scales(self):
		if self.is_MS:
			assert len(self.recursive_MS_scales)==(self.recursive_MS_max_level+1)
			return self.recursive_MS_max_level+1
		else: return 0
	
	@property
	def is_MS(self):
		return self.recursive_MS
	@property
	def has_MS_output(self):
		return self.is_MS and (self.cache_output) and (self.cache_MS)
	@property
	def has_multiple_decoders(self):
		multi_dec = self.recursive_MS and (not self.recursive_MS_shared_decoder)
		if multi_dec:
			assert isinstance(self.volume_decoder, list) and all(isinstance(_, (GrowingUNet, DeferredBackpropNetwork, tf.keras.Model)) for _ in self.volume_decoder), "velocity_decoder setup is invalid"
		else:
			assert isinstance(self.volume_decoder, (GrowingUNet, DeferredBackpropNetwork, tf.keras.Model)), "density volume_decoder setup is invalid"
		return multi_dec
	
	@property
	def active_decoders(self):
		if self.has_multiple_decoders:
			return [self.volume_decoder[s] for s in self.gen_current_MS_scales()]
		else:
			return [self.volume_decoder]
	
	def shape_of_scale(self, scale):
		return self.recursive_MS_shapes[scale]
	
	def _scale_input_down(self, data, scale):
		with SAMPLE("density scale input"):
			shape = self.shape_of_scale(scale)
			data = self.scale_renderer.resample_grid3D_aligned(data, shape, allow_split_channels=True)
		return data
	
	def _upscale_density(self, dens, scale):
		with SAMPLE("upscale_density"):
			old_shape = shape_list(dens)[-4:-1]
			shape = self.shape_of_scale(scale)
			dens = self.scale_renderer.resample_grid3D_aligned(dens, shape)
			if self._is_SDF:
				scale_factor = np.mean([o/i for o,i in zip(shape, old_shape)])
				dens = dens * scale_factor
		return dens
	
	def generate_denstiy(self):
		with SAMPLE("generate density"):
			def normalize_tensor(tensor, axes=[-1,-2,-3,-4], eps=1e-5): #3D, norm all except the batch dimension
				with SAMPLE("normalize_tensor"):
					mean, var = tf.nn.moments(tensor, axes, keep_dims=True)
					inv_std = tf.math.rsqrt(var + eps)
					#LOG.info("norm: %s, %s, %s", mean, var, inv_std)
					ret = (tensor - mean ) * inv_std
				return ret
			if self.recursive_MS:
				if self.__input_base_SDF: raise NotImplementedError
				#LOG.info("centered MS scales: %s", self.recursive_MS_shapes)
				active_MS_scales = list(self.gen_current_MS_scales())
				if self.base_input_features=="ZERO":
					dens = tf.zeros([self._get_batch_size()] + self.shape_of_scale(active_MS_scales[0]) + [1]) #vel initialized as zero
				elif self.base_input_features=="INPUT_IMAGES_HULL":
					if self._is_SDF:
						# use inverted scaled binary hull, so outside is 1 and inside is -1 (from 0 outside, 1 inside). simple SDF/levelset approximation using a hull.
						#dens = self._scale_input_down(1.0 - 2.0*self.parent_state.get_volume_features(types=["TARGET_HULL_BINARY"]), scale=active_MS_scales[0])
						#LOG.warning("Using smooth hull as input.")
						if self.parent_state.input_view_mask==[0]: # frontal single view
							dens = 1.0-2.0*self.parent_state._unproject_2D(self.parent_state.input_masks, accumulate="MIN", grid_shape=self.shape_of_scale(active_MS_scales[0]))
							shape = GridShape.from_tensor(dens)
							border = shape.z//4
							inner = shape.z-2*border
							mask = tf.constant([1]*border+[0]*inner+[1]*border, dtype=tf.float32)
							mask = tf.reshape(mask, (1,shape.z,1,1,1))
							dens = dens*(1-mask) + mask
						else:
							dens = 1.0-2.0*self.parent_state._unproject_2D(self.parent_state.input_masks, accumulate="MIN", grid_shape=self.shape_of_scale(active_MS_scales[0])) #, binary=True, binary_eps_3d=0.5
					else:
						if self.recursive_MS_direct_input:
							dens = self.parent_state.get_volume_features(types=["INPUT_IMAGES_HULL"], scale=self.shape_of_scale(active_MS_scales[0]))
						else:
							dens = self._scale_input_down(self.parent_state.get_volume_features(types=["INPUT_IMAGES_HULL"]), scale=active_MS_scales[0])
					assert has_shape(dens, [self._get_batch_size()] + self.shape_of_scale(active_MS_scales[0]) + [1])
				else:
					raise ValueError("Unknown density base input '%s'"%(self.base_input_features,))
				
				for s in active_MS_scales:
					inputs_frames = self._get_generator_input_MS(s) #list of inputs for current level/scale
					
					# no need to scale zero dens
					if s is not active_MS_scales[0]:
						#upscale previous velocity or potential. create centered vel to warp inputs
						dens = self._upscale_density(dens, s)
						# TODO: warped prev state as input?
						#for idx in self.warp_input_indices:
						#	inputs_frames[idx] = self._warp_centered_inputs(inputs_frames[idx], centered_vel) #warp the current frame to the target, expected to be first input
					
					
					num_density_inputs = len(self.step_input_density) + len(self.step_input_density_target)
					num_feature_inputs = len(self.step_input_features)
					if self.norm_input_mode=="SINGLE":
						#inputs_frames = map(normalize_tensor, inputs_frames)
						for idx in range(num_density_inputs+num_feature_inputs):
							# we don't want to normalize velocities
							inputs_frames[idx] = normalize_tensor(inputs_frames[idx])
					
					elif self.norm_input_mode=="GROUP":
						#inputs_frames at this point: densities, features, recurrent velocity
						dens_inputs = inputs_frames[:num_density_inputs]
						features_inputs = inputs_frames[num_density_inputs:num_density_inputs+num_feature_inputs]
						inputs_frames = inputs_frames[num_density_inputs+num_feature_inputs:]
						
						if num_density_inputs>0:
							inputs_frames.insert(0, normalize_tensor(tf.concat(dens_inputs, axis=-1)))
						del dens_inputs
						
						if num_feature_inputs>0:
							inputs_frames.insert(1, normalize_tensor(tf.concat(features_inputs, axis=-1)))
						del features_inputs
					
					elif self.norm_input_mode=="ALL" and (num_density_inputs+num_feature_inputs)>0:
						inputs_frames = [normalize_tensor(tf.concat(inputs_frames[:num_density_inputs+num_feature_inputs], axis=-1))] + inputs_frames[num_density_inputs+num_feature_inputs:]
					
					# dens is also input
					if self.recursive_MS_input_dens:
						# LOG.warning("TEST: NO RECURSIVE DENSITY INPUT!")
						# inputs_frames.append(tf.zeros_like(dens))
						inputs_frames.append(dens)
					
					#LOG.info("input stats: %s", [(tf.reduce_mean(_).numpy().tolist(), tf.reduce_min(_).numpy().tolist(), tf.reduce_max(_).numpy().tolist()) for _ in inputs_frames])
					inputs_frames = tf.concat(inputs_frames, axis=-1)
					
					with SAMPLE("volume_decoder"):
						if self.recursive_MS_shared_decoder:
							dens_residual = self.volume_decoder(inputs_frames)
						else:
							dens_residual = self.volume_decoder[s](inputs_frames)
					
					if False:
						warnings.warn("Sigmoid on density residual output.")
						dens_residual = tf.math.sigmoid(dens_residual) * 4 - 2
					
					if self.recursive_MS_residual:
						#LOG.warning("TEST: NO DENSITY GENERATOR!")
						#dens = dens + 0.0*dens_residual
						#dens = (dens + dens_residual)*0.5
						dens = dens + dens_residual
					else:
						dens = dens_residual
					
					if self.cache_MS:
						with tf.device(self._device):
							self.__d_MS[s] = tf.identity(dens)
							self.__d_MS_residual[s] = tf.identity(dens_residual)
					
						
				self.clear_input_cache() #?
				d = dens
			else:
				inputs_frames = self._get_generator_input() #list of inputs for current level/scale
				
				num_density_inputs = len(self.step_input_density) + len(self.step_input_density_target)
				num_feature_inputs = len(self.step_input_features)
				if self.norm_input_mode=="SINGLE":
					#inputs_frames = map(normalize_tensor, inputs_frames)
					for idx in range(num_density_inputs+num_feature_inputs):
						# we don't want to normalize velocities
						inputs_frames[idx] = normalize_tensor(inputs_frames[idx])
				
				elif self.norm_input_mode=="GROUP":
					#inputs_frames at this point: densities, features, recurrent velocity
					dens_inputs = inputs_frames[:num_density_inputs]
					features_inputs = inputs_frames[num_density_inputs:num_density_inputs+num_feature_inputs]
					inputs_frames = inputs_frames[num_density_inputs+num_feature_inputs:]
					
					if num_density_inputs>0:
						inputs_frames.insert(0, normalize_tensor(tf.concat(dens_inputs, axis=-1)))
					del dens_inputs
					
					if num_feature_inputs>0:
						inputs_frames.insert(1, normalize_tensor(tf.concat(features_inputs, axis=-1)))
					del features_inputs
				
				elif self.norm_input_mode=="ALL" and (num_density_inputs+num_feature_inputs)>0:
					inputs_frames = [normalize_tensor(tf.concat(inputs_frames[:num_density_inputs+num_feature_inputs], axis=-1))] + inputs_frames[num_density_inputs+num_feature_inputs:]
				
				if self.__input_base_SDF:
					inputs_frames.append(tf.tile(self._get_base_SDF(), [self._get_batch_size(),1,1,1,1]))
				
				#LOG.info("input stats: %s", [(tf.reduce_mean(_).numpy().tolist(), tf.reduce_min(_).numpy().tolist(), tf.reduce_max(_).numpy().tolist()) for _ in inputs_frames])
				inputs_frames = tf.concat(inputs_frames, axis=-1)
				
				with SAMPLE("volume_decoder"):
					dens = self.volume_decoder(inputs_frames)
				
				if False:
					warnings.warn("Sigmoid on density residual output.")
					dens = tf.math.sigmoid(dens) * 4 - 2
					
				self.clear_input_cache() #?
				d = dens
			
			if self.__use_base_SDF:
				d = d + self._get_base_SDF()
		return d
	
	@property
	def _d(self):
		# the raw density grid
		if self.__d is not None:
			d = self.__d
		else:
			d = self.generate_denstiy()
			if self.cache_output:
				with tf.device(self._device):
					self.__d = tf.identity(d)
				d = self.__d
		return d
	@_d.setter
	def _d(self, value):
		raise AttributeError("Can't set density of NeuralDensityGrid")
	
	@property
	def d(self):
		# the density grid with modifications (inflow) and constraints (hull, non-negativity)
		d = self._d
		if not self._is_SDF:
			d = tf.maximum(d, 0)
		return d
	
	def _d_MS(self, scale):
		if not self.has_MS_output: raise ValueError("Mutiscale density is not available.")
		# check if base vel has been generated. if not, do so to also generate the MS scales.
		if self.__d is not None:
			if scale not in self.__d_MS: #density has been generated, but the scale is (still) not available
				raise ValueError("Mutiscale density of scale \"{}\" is not available. Available scales: {}".format(scale, list(self.__d_MS.keys())))
		else:
			if scale not in self.recursive_MS_scales:
				raise ValueError("Mutiscale density of scale \"{}\" is not available. Available scales: {}".format(scale, self.recursive_MS_scales))
			self._d #generate density
		d = self.__d_MS[scale]
		return d
		
	
	def d_MS(self, scale):
		d = self._d_MS(scale)
		if not self._is_SDF:
			d = tf.maximum(d, 0)
		return d
	
	def _d_MS_r(self, scale):
		if not self.has_MS_output: raise ValueError("Mutiscale density is not available.")
		# check if base vel has been generated. if not, do so to also generate the MS scales.
		if self.__d is not None:
			if scale not in self.__d_MS_residual: #density has been generated, but the scale is (still) not available
				raise ValueError("Mutiscale density of scale \"{}\" is not available. Available scales: {}".format(scale, list(self.__d_MS_residual.keys())))
		else:
			if scale not in self.recursive_MS_scales:
				raise ValueError("Mutiscale density of scale \"{}\" is not available. Available scales: {}".format(scale, self.recursive_MS_scales))
			self._d #generate density
		d = self.__d_MS_residual[scale]
		return d
	
	def d_MS_r(self, scale):
		return self._d_MS_r(scale)
		
	@property
	def requires_parent_state_variables(self):
		return False #"ENC3D" in self.type_input_features
	
	def get_variables(self, scale=None):
		if self.has_multiple_decoders: #recursive_MS and (not self.recursive_MS_shared_decoder):
			if scale is None:
				if self.recursive_MS_train_mode=="ALL":
					var_dict = {'density_decoder': [var for dec in self.active_decoders for var in dec.trainable_variables]}
				elif self.recursive_MS_train_mode=="TOP":
					var_dict = {'density_decoder': self.volume_decoder[self.recursive_MS_scales[self.recursive_MS_current_level]].trainable_variables}
				else: raise ValueError
			else:
				var_dict = {'density_decoder': self.volume_decoder[scale].trainable_variables}
		else:
			var_dict = {'density_decoder': self.volume_decoder.trainable_variables}
		
		if self.requires_parent_state_variables and self.parent_state is not None:
			var_dict.update(self.parent_state.get_variables())
		if self._inflow is not None:
			var_dict['inflow'] = self._inflow
		return var_dict
	
	def get_output_variables(self, include_MS=False, include_residual=False, only_trainable=False):
		if not self.cache_output:
			raise RuntimeError("Output caching must be enabled for output variables to have meaning.")
		var_dict = {}
		if include_MS and self.has_MS_output:
			scales = list(self.gen_current_trainable_MS_scales()) if only_trainable else list(self.gen_current_MS_scales())
			for scale in scales:
				var_dict['density_%s'%(scale,)] = self._d_MS(scale)
				if include_residual:
					var_dict['density_r%s'%(scale,)] = self._d_MS_r(scale)
		# else:
			# var_dict['density'] = self._d
		var_dict['density'] = self._d
		
		return var_dict
	
	def __get_MS_variable_keys_scale(self, scale, include_residual=False):
		key_list = []
		key_list.append('density_%s'%(scale,))
		if include_residual:
			key_list.append('density_r%s'%(scale,))
		
		return key_list
	
	def _get_output_variable_keys(self, include_MS=False, include_residual=False):
		key_list = []
		if include_MS:
			for scale in self.gen_current_MS_scales():
				key_list.extend(self.__get_MS_variable_keys_scale(scale, include_residual=include_residual))
		# else:
			# key_list.append('density')
		key_list.append('density')
		
		return key_list
		
	def map_gradient_output_to_MS(self, grad_dict):
		assert isinstance(grad_dict, dict)
		output_keys = self._get_output_variable_keys(include_MS=False)
		MS_keys = self.__get_MS_variable_keys_scale(scale=self._get_top_active_scale(), include_residual=False)
		assert len(output_keys)==len(MS_keys)
		
		out_dict = {}
		for key, grad in grad_dict.items():
			if key not in output_keys: raise KeyError("Invalid output gradient key '{}'".format(key))
			out_dict[MS_keys[output_keys.index(key)]] = grad
		return out_dict
	
	def apply_clamp(self, vmin, vmax):
		pass
	
	def assign(self, d, inflow=None):
		raise TypeError("Can't assign to NeuralDensityGrid")
	
	def clear_cache(self):
		self.__d = None
		self.__d_MS = {}
		self.__d_MS_residual = {}
		self.clear_input_cache()
		
		self.__output_users = {"OUTPUT": []}
		self.__output_gradient_cache = None
		self.__inputs_for_backprop = []
		self.__input_grad_cache.clear()
	
	def clear_input_cache(self):
		self.__generator_inputs = {}
	
	def clear_cache_for_backprop(self):
		# self.clear_cache()
		# if self.requires_parent_state_variables:
			# self.parent_state.clear_cache_for_backprop()
		self.__d = None
		self.__d_MS = {}
		self.__d_MS_residual = {}
		self.clear_input_cache()
	
	def _get_output_users_grads(self):
		for out_id, users in self.__output_users.items():
			for other in users:
				other._compute_input_grads()
		extra_output_gradients = {}
		for out_id, users in self.__output_users.items():
			extra_output_gradients[out_id] = tf.zeros_like(self.output(out_id))
			for other in users:
				extra_output_gradients[out_id] = extra_output_gradients[out_id] + other._get_input_grad(self, out_id)
		return extra_output_gradients
	
	def backprop(self, output_gradients, include_MS=False, include_residual=False, only_trainable=False):
		
		if self.parent_state.next is not None:
			check_output_users(self.__output_users, {"OUTPUT": [WarpedDensityGrid, NeuralVelocityGrid]}, "NeuralDensityGrid")
		else:
			check_output_users(self.__output_users, {"OUTPUT": []}, "single frame NeuralDensityGrid")
		#assert self.__output_users is not None and len(self.__output_users)==2, "Expected velocity output users of NeuralDensityGrid to be one WarpedDensityGrid and one NeuralVelocityGrid, is %s"%([type(_).__name__ for _ in self.__output_users] if self.__output_users is not None else None)
		
		extra_output_gradients = self._get_output_users_grads()
		
		extra_output_gradients = {"density":extra_output_gradients["OUTPUT"]}
		if include_MS:
			extra_output_gradients = self.map_gradient_output_to_MS(extra_output_gradients)
		
		for k in output_gradients:
			if k in extra_output_gradients:
				if output_gradients[k] is None:
					output_gradients[k] = extra_output_gradients[k]
				else:
					output_gradients[k] += extra_output_gradients[k]
		
		LOG.debug("Backprop density frame %d", self.parent_state.frame)
		
		with SAMPLE("NDG backprop"):
			if not include_MS and not ('density' in output_gradients):
				raise ValueError("No gradients to backprop.")
			if include_MS:
				for scale in (self.gen_current_trainable_MS_scales() if only_trainable else self.gen_current_MS_scales()):
					if not ('density_%s'%(scale,) in output_gradients) or (include_residual and not ('density_r%s'%(scale,) in output_gradients)):
						raise ValueError("No gradients to backprop for MS scale %s."%(scale,))
			self.clear_cache_for_backprop()
			var_dict = self.get_variables()
			var_dict["inputs_for_backprop"] = self.__gather_inputs()
			
			with SAMPLE("forward"), tf.GradientTape(watch_accessed_variables=False) as tape:
				tape.watch(var_dict)
				output = self.get_output_variables(include_MS=include_MS, include_residual=include_residual)
			
			output = {k: output[k] for k in output if (k in output_gradients)}
			output_gradients = {
				k: (output_gradients[k] if output_gradients[k] is not None else tf.zeros_like(output[k])) \
				for k in output_gradients \
				if k in output \
			}
			
		#	for k in output:
		#		LOG.info("vel output '%s': out type=%s, grad type=%s", k, output[k].__class__.__name__, output_gradients[k].__class__.__name__)
			with SAMPLE("gradients"):
				gradients = tape.gradient(output, var_dict, output_gradients=output_gradients)
		
		for i, grad in enumerate(gradients["inputs_for_backprop"]):
			self.__input_grad_cache["input_grad_%d"%(i,)] = grad
		del gradients["inputs_for_backprop"]
		
		return gradients
	
	def __split_gradients_for_decoders(self, grads):
		assert self.has_multiple_decoders and self.recursive_MS_train_mode=="ALL"
		num_var_per_decoder = [len(dec.trainable_variables) for dec in self.active_decoders]
		if sum(num_var_per_decoder)!=len(grads): raise RuntimeError("gradients do not match variables: got {} gradients for {} variables of {} decoders {}".\
			format(len(grads), sum(num_var_per_decoder), len(self.active_decoders), num_var_per_decoder))
		split_grads = []
		pos = 0
		for size in num_var_per_decoder:
			split_grads.append(grads[pos:pos+size])
			pos +=size
		assert all(size==len(grad) for size, grad in zip(num_var_per_decoder, split_grads))
		return split_grads
	
	def set_output_gradients_for_backprop_accumulate(self, output_gradients, **kwargs):
		kwargs["output_gradients"] = output_gradients
		self.__output_gradient_cache = kwargs
	
	def backprop_accumulate(self, output_gradients, include_MS=False, include_residual=False, only_trainable=False):
		gradients = self.backprop(output_gradients, include_MS=include_MS, include_residual=include_residual, only_trainable=only_trainable)
		
		with SAMPLE("NDG acc grads"):
			dens_net_gradients = gradients['density_decoder']
			if self.has_multiple_decoders:
				if self.recursive_MS_train_mode=="ALL":
					for dec, grads in zip(self.volume_decoder, self.__split_gradients_for_decoders(dens_net_gradients)):
						dec.add_gradients(grads)
				elif self.recursive_MS_train_mode=="TOP":
					self.volume_decoder[self.recursive_MS_scales[self.recursive_MS_current_level]].add_gradients(dens_net_gradients)
				else: raise ValueError
			else:
				self.volume_decoder.add_gradients(dens_net_gradients)
			
			if self.requires_parent_state_variables:
				raise NotImplementedError("Should be handled via BackpropInterface now.")
				state_gradients = {k: gradients[k] for k in self.parent_state.get_variables()}
				self.parent_state.accumulate_gradients(state_gradients)
	
	@property
	def has_pending_gradients(self):
		if self.has_multiple_decoders:
			return any(_.has_pending_gradients for _ in self.volume_decoder)
		else:
			return self.volume_decoder.has_pending_gradients
	
	def apply_gradients(self, optimizer, keep_gradients=False):
		raise NotImplementedError
		LOG.debug("Applying density gradients of frame %d", self.parent_state.frame)
		if self.has_multiple_decoders:
			if not self.has_pending_gradients: raise RuntimeError("No decoder has any recorded gradients to apply."%(self.name,))
			for dec in self.volume_decoder:
				if dec.has_pending_gradients:
					dec.apply_gradients(optimizer)
					if not keep_gradients:
						dec.clear_gradients()
		else:
			self.volume_decoder.apply_gradients(optimizer)
			if not keep_gradients:
				self.volume_decoder.clear_gradients()
	
	def get_grads_vars(self, keep_gradients=True):
		grads_vars = []
		if self.has_multiple_decoders:
			for dec in self.volume_decoder:
				if dec.has_pending_gradients:
					grads_vars.extend(dec.get_grads_vars(keep_gradients=keep_gradients))
					if not keep_gradients:
						dec.clear_gradients()
		else:
			grads_vars.extend(self.volume_decoder.get_grads_vars(keep_gradients=keep_gradients))
			if not keep_gradients:
				self.volume_decoder.clear_gradients()
		return grads_vars
	
	def clear_gradients(self):
		if self.has_multiple_decoders:
			for dec in self.volume_decoder:
				dec.clear_gradients()
		else:
			self.volume_decoder.clear_gradients()
		


# generate velocity via network with input from 2 consecutive frames
# input: lifting (density decoder input), generated density
# ouput: centered (or staggered?) velocity
# cache: both centered and staggered velocity, generate one from the other.
# rest of functionality inherited from normal/super VelocityGrid
class NeuralVelocityGrid(VelocityGrid, BackpropInterface):
	def __init__(self, volume_decoder, parent_state, boundary=None, scale_renderer=None, warp_renderer=None, device=None, var_name="velocity", trainable=True, velocity_format="CENTERED", \
			step_input_density=[], step_input_density_target=[], step_input_density_proxy=[], step_input_features=[0,1], type_input_features=["TARGET_UNPROJECTION"], warp_input_indices=[0], \
			downscale_input_modes=["RESAMPLE"]):
		self.cache_output = True
		self._device = device
		self.__output_cache = ResourceCacheDictTF(self._device)
		self.__output_gradient_cache = ResourceCacheDictTF(self._device)
		self.__input_cache = ResourceCacheDictTF(self._device)
		self.__input_grad_cache = ResourceCacheDictTF(self._device)
		self.clear_cache()
		self.parent_state = parent_state
		self.use_raw_images = True
		self.volume_decoder = volume_decoder
		
		self.set_input_encoder(None)
		self.set_downscale_encoder(None)
		
		assert velocity_format in ["CENTERED", "STAGGERED", "CURL_CENTERED", "CURL_STAGGERED"]
		self.velocity_format = "CENTERED" if velocity_format in ["CENTERED", "CURL_CENTERED"] else "STAGGERED" #CENTERED, STAGGERED
		
		self.use_curl_potential = velocity_format in ["CURL_CENTERED", "CURL_STAGGERED"] #True
		#if self.use_curl_potential and not velocity_format=="CENTERED":
		#	raise ValueError("When generating velocity via curl the velocity_format must be CENTERED.")
		
		# use potential for MS residual instead of velocity
		self.residual_potential = True # interpolation when up-scaling vel causes noticeable divergence. so up-scale the potential instead to keep the velocity div-free
		
		if self.use_curl_potential:
			# TODO expose curl settings in setup and remove this
			LOG.info("Generating %s velocity via curl.", self.velocity_format)
		
		self.set_boundary(boundary)
		self.scale_renderer = scale_renderer
		self.warp_renderer = warp_renderer
		
		self.is_var = False
		self._name = var_name
		self._is_trainable = trainable
		#raise NotImplementedError()
		
		self.recursive_MS = False
		#self.input_current_features = True
		
		assert step_input_density==[0], "DEBUG"
		self.step_input_density = step_input_density
		assert step_input_density_target==[], "DEBUG"
		self.step_input_density_target = step_input_density_target
		#assert step_input_density_proxy==[], "DEBUG"
		self.step_input_density_proxy = step_input_density_proxy
		#assert step_input_features==[0,1], "DEBUG"
		self.step_input_features = step_input_features
		
		# also check requires_parent_state_variables() if adding new feature inputs
		assert all(_ in ["INPUT_IMAGES_UNPROJECTION","INPUT_IMAGES_UNPROJECTION_CONCAT","INPUT_IMAGES_RAW_UNPROJECTION","INPUT_IMAGES_HULL", "ENC3D", "ENCLIFT"] for _ in type_input_features)
		#assert type_input_features==["ENC3D"], "DEBUG"
		self.type_input_features = type_input_features
		#assert warp_input_indices==[0,1], "DEBUG"
		self.warp_input_indices = warp_input_indices 
		self.downscale_input_modes = downscale_input_modes
		
		self.norm_input_mode = "GROUP" #NONE, SINGLE, GROUP, ALL
		
		self.__centered_shape = None
		
		# self.__output_users = []
		# self.__output_gradient_cache = None
		# self.__inputs_for_backprop = []
	
	# backprop interface
	def __add_used_input(self, other, output_id):
		self.__inputs_for_backprop.append(input_key(other, output_id))
	
	def __register_inputs(self):
		for inp, out_id in self.__inputs_for_backprop:
			assert isinstance(inp, BackpropInterface)
			inp._register_output_user(self, out_id)
	
	def __gather_inputs(self):
		return [inp.output(out_id) for inp, out_id in self.__inputs_for_backprop]
	
	def outputs(self):
		return {"CENTERED": self.centered()}
		#"STAGGERED_X": self._x
	
	def output(self, output_id="CENTERED"):
		assert output_id=="CENTERED"
		return self.outputs()[output_id]
	
	def _register_output_user(self, other, output_id):
		assert isinstance(other, BackpropInterface)
		if other not in self.__output_users[output_id]:
			self.__output_users[output_id].append(other)
	
	def _compute_input_grads(self):
		if self.__input_grad_cache.is_empty():
			assert self.__output_gradient_cache is not None
			self.backprop_accumulate(**self.__output_gradient_cache) #**self.__get_output_grads()) #
			self.__output_gradient_cache = None
	
	def _get_input_grad(self, other, output_id):
		assert not self.__input_grad_cache.is_empty()
		t = input_key(other, output_id)
		idx = self.__inputs_for_backprop.index(t)
		return self.__input_grad_cache["input_grad_%d"%(idx,)]
	
	def has_gradients_for(self, other, output_id):
		return (input_key(other, output_id) in self.__inputs_for_backprop) and (not self.__input_grad_cache.is_empty() or self.can_backprop)
	
	@property
	def can_backprop(self):
		# has output gradients available
		#LOG.info("'%s' can backprop: out grad %s; output users %d, provides grads %s", self._name, self.__output_gradient_cache is not None, len(self.__output_users), [_.has_gradients_for(self) for _ in self.__output_users])
		return (self.__output_gradient_cache is not None) or (len(self.__output_users)>0 and any(len(users)>0 and any(_.has_gradients_for(self, out_id) for _ in users) for out_id, users in self.__output_users.items()))
	
	@property
	def requires_backprop(self):
		is_frozen = all(_.is_frozen_weights for _ in self.volume_decoder) if self.has_multiple_decoders else self.volume_decoder.is_frozen_weights
		return (not is_frozen) and self.__input_grad_cache.is_empty() and self.can_backprop
	
	# own
	
	@property
	def parent_state(self):
		return self.__state
	@parent_state.setter
	def parent_state(self, value):
		assert is_None_or_type(value, NeuralState)
		self.__state = value
	
	def _get_batch_size(self):
		if self.__centered is not None:
			return shape_list(self.__centered)[0]
		elif self.parent_state is not None:
			return self.parent_state._get_batch_size()
		else:
			raise RuntimeError("NeuralVelocityGrid: Can't determine batch size without cached velocity or parent state.")
			#return self._batch_size
	
	@property
	def lod_pad(self):
		return tf.zeros([self._get_batch_size()]+self.centered_shape+[1])
	def set_centered_shape(self, centered_shape):
		assert is_None_or_type(centered_shape, (list, tuple))
		if not centered_shape is None:
			assert len(centered_shape)==3
			assert all(isinstance(_, Integral) for _ in centered_shape)
		self.__centered_shape = centered_shape
	@property
	def centered_shape(self):
		if self.__centered_shape is None:
			if self.__state is None: raise ValueError("Parent state is not set")
			return self.__state.transform.grid_size
		else:
			return self.__centered_shape
	@property
	def x_shape(self):
		return self.component_shapes(self.centered_shape)[0]
	@property
	def y_shape(self):
		return self.component_shapes(self.centered_shape)[1]
	@property
	def z_shape(self):
		return self.component_shapes(self.centered_shape)[2]
	
	def set_input_encoder(self, input_encoder):
		if isinstance(input_encoder, list):
			assert all(isinstance(_, DeferredBackpropNetwork) for _ in input_encoder)
			self.__shared_input_encoder = False
		else:
			assert is_None_or_type(input_encoder, DeferredBackpropNetwork)
			self.__shared_input_encoder = True
		self.__input_encoder = input_encoder
	
	@property
	def _has_input_encoder(self):
		return self.__input_encoder is not None
	
	def _get_input_encoder(self, scale):
		if self.__shared_input_encoder:
			return self.__input_encoder
		else:
			return self.__input_encoder[scale]
	
	@property
	def active_input_encoders(self):
		if not self.__shared_input_encoder:
			return [self.__input_encoder[s] for s in self.gen_current_MS_scales()]
		else:
			return [self.__input_encoder]
	
	def set_downscale_encoder(self, encoder):
		if isinstance(encoder, list):
			assert all(isinstance(_, DeferredBackpropNetwork) for _ in encoder)
			self.__shared_downscale_encoder = False
		else:
			assert is_None_or_type(encoder, DeferredBackpropNetwork)
			self.__shared_downscale_encoder = True
		self.__downscale_encoder = encoder
	@property
	def _has_downscale_encoder(self):
		return self.__downscale_encoder is not None
	def _get_downscale_encoder(self, scale):
		if self.__shared_downscale_encoder:
			return self.__downscale_encoder
		else:
			return self.__downscale_encoder[scale]
	@property
	def active_downscale_encoders(self):
		if not self.__shared_downscale_encoder:
			scales = list(self.gen_max_MS_scales() if self.recursive_MS_use_max_level_input else self.gen_current_MS_scales())[:-1]
			return [self.__downscale_encoder[s] for s in scales]
		else:
			return [self.__downscale_encoder]
	
	def _get_generator_input(self):
		if "BASE" not in self.__generator_inputs:
			centered_shape = self.centered_shape #self.centered_shape
			with SAMPLE("velocity input"):
				inp = []
				feature_shape = None
				for step in self.step_input_density:
					state = self._get_state_by_step(step)
					if state is not None:
						#inp.append(state.density.output())
						inp.append(state.density.scaled(centered_shape, with_inflow=True))
						feature_shape = shape_list(inp[-1])
						assert has_shape(inp[-1], [None]+centered_shape+[None]), "%s, %s"%(shape_list(inp[-1]), [None]+centered_shape+[None])
						#self.__inputs_for_backprop.append(state.density)
						self.__add_used_input(state.density, "OUTPUT")
					else:
						LOG.debug("NeuralVelocityGrid of frame %d is missing density input of state %d (step %d)", self.parent_state.frame, self.parent_state.frame+step, step)
						inp.append(tf.zeros(feature_shape))
				
				#feature_shape = None
				for step in self.step_input_density_target:
					state = self._get_state_by_step(step)
					if state is not None:
						raise NotImplementedError("Debug")
						temp_inp = state.density_target.scaled(centered_shape, with_inflow=True)
						inp.append(temp_inp)
						##feature_shape = shape_list(temp_inp)
						del temp_inp
					else:
						LOG.debug("NeuralVelocityGrid of frame %d is missing density target input of state %d (step %d)", self.parent_state.frame, self.parent_state.frame+step, step)
						inp.append(tf.zeros(feature_shape))
				
				#feature_shape = None
				for step in self.step_input_density_proxy:
					state = self._get_state_by_step(step)
					if state is not None and state.has_density_proxy:
						inp.append(state.density_proxy.output("OUTPUT"))
						feature_shape = shape_list(inp[-1])
						assert has_shape(inp[-1], [None]+centered_shape+[None]), "%s, %s"%(shape_list(inp[-1]), [None]+centered_shape+[None])
						#self.__inputs_for_backprop.append(state.density_proxy)
						self.__add_used_input(state.density_proxy, "OUTPUT")
					else:
						# FOR TESTING
						if state is not None:
							raise RuntimeError("State %d is missing denstiy proxy as input for frame %d.", state.frame, self.parent_state.frame)
						
						LOG.debug("NeuralVelocityGrid of frame %d is missing density proxy input of state %d (step %d)", self.parent_state.frame, self.parent_state.frame+step, step)
						inp.append(tf.zeros(feature_shape))
				
				feature_shape = None
				for step in self.step_input_features:
					state = self._get_state_by_step(step)
					if state is not None:
						# assert not self.use_raw_images and self.type_input_features==["ENC3D"]
						# inp.append(state.output())
						# feature_shape = shape_list(inp[-1])
						# assert has_shape(inp[-1], [None]+centered_shape+[None]), "%s, %s"%(shape_list(inp[-1]), [None]+centered_shape+[None])
						# self.__inputs_for_backprop.append(state)
						
						assert not self.use_raw_images 
						temp_inp = state.get_volume_features(types=self.type_input_features, shape=centered_shape, concat=False)
						
						if "ENC3D" in self.type_input_features:
							temp_inp.append(state.output("OUTPUT"))
							assert has_shape(temp_inp[-1], [None]+centered_shape+[None]), "%s, %s"%(shape_list(temp_inp[-1]), [None]+centered_shape+[None])
							#self.__inputs_for_backprop.append(state)
							self.__add_used_input(state, "OUTPUT")
						
						inp.append(tf.concat(temp_inp, axis=-1))
						del temp_inp
						feature_shape = shape_list(inp[-1])
					else:
						LOG.debug("NeuralVelocityGrid of frame %d is missing feature input of state %d (step %d)", self.parent_state.frame, self.parent_state.frame+step, step)
						inp.append(tf.zeros(feature_shape))
				del feature_shape
				
				self.__register_inputs()
				
				inp_stag = []
			
			self.__generator_inputs["BASE"] = (inp, inp_stag)
		
		return self.__generator_inputs["BASE"]
	
	def _get_state_by_step(self, step):
		assert isinstance(step, Integral) and step>=0
		state = self.parent_state
		while step>0:
			if state.next is None:
				#raise AttributeError("Can't get next state")
				return None
			state = state.next
			step -=1
		return state
	
	def get_advected(self, scale=None, allow_potential=True):
		# return own self-advected velocity
		raise NotImplementedError
		if self.is_MS and scale is not None:
			pass
		else:
			pass
	
	def _get_generator_input_MS(self, scale, cache_inputs=False):
		recurrent_input_vel = False
		recurrent_input_vel_MS = False #use all scales or just top?
		recurrent_input_vel_potential = True #use generated potential instead of velocity if generating potentials
		recurrent_input_vel_warp = False #try this?
		# TODO: recursive_MS input generation; possibly multi-scale input/features provided by parent state
		
		
		centered_scale_shape = self.centered_shape_of_scale(scale) #self.centered_shape
		if scale not in self.__generator_inputs:
			if (self.__is_top_scale(scale) if self.recursive_MS_use_max_level_input else self.__is_top_active_scale(scale)) or self._parent_provides_input_scale(scale):
				
				#LOG.info("Gather input for scale %d %s", scale, centered_scale_shape)
				
				with SAMPLE("velocity top MS input"):
					inp = []
					feature_shape = None
					for step in self.step_input_density:
						state = self._get_state_by_step(step)
						if state is not None:
							# temp_inp = state.density.scaled(centered_scale_shape, with_inflow=True)
							# temp_inp = tf.stop_gradient(temp_inp)
							# inp.append(temp_inp)
							# if cache_inputs:
								# self.__input_cache[("density", state.frame, scale)] = temp_inp
							# feature_shape = shape_list(temp_inp)
							# del temp_inp
							
							#inp.append(state.density.output())
							inp.append(state.density.scaled(centered_scale_shape, with_inflow=True))
							feature_shape = shape_list(inp[-1])
							assert has_shape(inp[-1], [None]+centered_scale_shape+[None]), "%s, %s"%(shape_list(inp[-1]), [None]+centered_scale_shape+[None])
							#self.__inputs_for_backprop.append(state.density)
							self.__add_used_input(state.density, "OUTPUT")
						else:
							LOG.debug("NeuralVelocityGrid of frame %d is missing density input of state %d (step %d)", self.parent_state.frame, self.parent_state.frame+step, step)
							inp.append(tf.zeros(feature_shape))
					
					#feature_shape = None
					for step in self.step_input_density_target:
						state = self._get_state_by_step(step)
						if state is not None:
							raise NotImplementedError("Debug")
							temp_inp = state.density_target.scaled(centered_scale_shape, with_inflow=True)
							inp.append(temp_inp)
							if cache_inputs:
								self.__input_cache[("density_target", state.frame, scale)] = temp_inp
							##feature_shape = shape_list(temp_inp)
							del temp_inp
						else:
							LOG.debug("NeuralVelocityGrid of frame %d is missing density target input of state %d (step %d)", self.parent_state.frame, self.parent_state.frame+step, step)
							inp.append(tf.zeros(feature_shape))
					
					#feature_shape = None
					for step in self.step_input_density_proxy:
						state = self._get_state_by_step(step)
						if state is not None and state.has_density_proxy:
							# temp_inp = state.density_proxy.scaled(centered_scale_shape, with_inflow=True)
							# temp_inp = tf.stop_gradient(temp_inp)
							# inp.append(temp_inp)
							# if cache_inputs:
								# self.__input_cache[("density_proxy", state.frame, scale)] = temp_inp
							# feature_shape = shape_list(temp_inp)
							# del temp_inp
							#inp.append(state.density_proxy.output())
							inp.append(state.density_proxy.scaled(centered_scale_shape, with_inflow=True))
							feature_shape = shape_list(inp[-1])
							assert has_shape(inp[-1], [None]+centered_scale_shape+[None]), "%s, %s"%(shape_list(inp[-1]), [None]+centered_scale_shape+[None])
							#self.__inputs_for_backprop.append(state.density_proxy)
							self.__add_used_input(state.density_proxy, "OUTPUT")
						else:
							# FOR TESTING
							if state is not None:
								raise RuntimeError("State %d is missing denstiy proxy as input for frame %d.", state.frame, self.parent_state.frame)
							
							LOG.debug("NeuralVelocityGrid of frame %d is missing density proxy input of state %d (step %d)", self.parent_state.frame, self.parent_state.frame+step, step)
							inp.append(tf.zeros(feature_shape))
					
					feature_shape = None
					for step in self.step_input_features:
						state = self._get_state_by_step(step)
						if state is not None:
							# if self.use_raw_images:
								# #inp.append(state.targets_raw_feature_volume)
								# raise NotImplementedError
								# #inp.append(state.get_volume_features(types=self.type_input_features))
								# #inp.append(tf.zeros_like(state.targets_raw_feature_volume))
							# else:
								# temp_inp = state.get_volume_features(types=self.type_input_features, shape=centered_scale_shape)
								# inp.append(temp_inp)
								# if cache_inputs:
									# self.__input_cache[("density_proxy", state.frame, scale)] = temp_inp
							# feature_shape = shape_list(temp_inp)
							# del temp_inp
							# raise NotImplementedError("TODO:")
							assert not self.use_raw_images 
							temp_inp = state.get_volume_features(types=self.type_input_features, shape=centered_scale_shape, concat=False)
							
							if "ENC3D" in self.type_input_features:
								temp_inp.append(state.output("OUTPUT"))
								assert has_shape(temp_inp[-1], [None]+centered_scale_shape+[None]), "%s, %s"%(shape_list(temp_inp[-1]), [None]+centered_scale_shape+[None])
								#self.__inputs_for_backprop.append(state)
								self.__add_used_input(state, "OUTPUT")
							
							if len(temp_inp)>0: #could be empty due to external handling of ENCLIFT
								inp.append(tf.concat(temp_inp, axis=-1))
								feature_shape = shape_list(inp[-1])
							else:
								break
							del temp_inp
						else:
							LOG.debug("NeuralVelocityGrid of frame %d is missing feature input of state %d (step %d)", self.parent_state.frame, self.parent_state.frame+step, step)
							inp.append(tf.zeros(feature_shape))
					del feature_shape
						
					
					self.__register_inputs()
					
					inp_stag = []
					if recurrent_input_vel and not (self.parent_state.prev.velocity.has_MS_output and recurrent_input_vel_MS):
						if self.parent_state.prev is not None and self.parent_state.prev.has_velocity:
							if recurrent_input_vel_warp: raise NotImplementedError
							if self.parent_state.prev.velocity.use_curl_potential and recurrent_input_vel_potential:
								temp_inp = self.parent_state.prev.velocity.__curl_potential_MS[scale]
								inp_stag.append(temp_inp)
								if cache_inputs:
									self.__input_cache[("velocity_prev_curl_potential", state.frame, scale)] = temp_inp
							elif self.is_staggered:
								temp_inp = self.parent_state.prev.velocity._staggered(scale)
								inp_stag.append(self._components_to_staggeredTensor(*temp_inp))
								if cache_inputs:
									self.__input_cache[("velocity_prev_staggered_components", state.frame, scale)] = temp_inp
							elif self.is_centered:
								temp_inp = self.parent_state.prev.velocity.centered(scale)
								if cache_inputs:
									self.__input_cache[("velocity_prev_centered", state.frame, scale)] = temp_inp
								inp.append(temp_inp)
							del temp_inp
				#if self.is_staggered:
				#	inp = [self._scalar_centered_to_staggered(_) for _ in inp]
			else:
				inp, inp_stag = self._get_generator_input_MS(self._get_larger_scale(scale), cache_inputs)
				
				#LOG.info("Scale input to scale %d %s", scale, centered_scale_shape)
				
				if "ENCLIFT" in self.type_input_features:
					# cut features that are created per-level
					inp = inp[:-len(self.step_input_features)]
				
				if self._has_downscale_encoder:
					assert len(inp)==len(self.downscale_input_modes), "inputs do not match downscale modes."
					inp = [self._scale_input_down(data, scale, mode) for data, mode in zip(inp, self.downscale_input_modes)]
				else:
					inp = [self._scale_input_down(data, scale, "RESAMPLE") for data in inp]
			
			# temp, TODO: handle on-scale creation vs. downsampling per-input
			if "ENCLIFT" in self.type_input_features:
				for step in self.step_input_features:
					state = self._get_state_by_step(step)
					if state is not None:
						output_id = "LIFTING_"+("-".join(str(_) for _ in centered_scale_shape))
						temp_inp = state.output(output_id)
						if self._has_input_encoder:
							temp_inp = self._get_input_encoder(scale)(temp_inp)
						inp.append(temp_inp)
						self.__add_used_input(state, output_id)
						feature_shape = shape_list(inp[-1])
					else:
						inp.append(tf.zeros(feature_shape))
				del feature_shape
			
			if recurrent_input_vel and (self.parent_state.prev.velocity.has_MS_output and recurrent_input_vel_MS):
				if self.parent_state.prev is not None:
					raise NotImplementedError
					if self.parent_state.prev.velocity.use_curl_potential and recurrent_input_vel_potential:
						inp_stag.append(self.parent_state.prev.velocity.__curl_potential_MS[scale])
					elif self.is_staggered:
						inp_stag.append(self.parent_state.prev.velocity._staggered_MS(scale))
					elif self.is_centered:
						inp.append(self.parent_state.prev.velocity.centered_MS(scale))
			
			if recurrent_input_vel and self.parent_state.prev is None:
				raise NotImplementedError
				if self.is_staggered:
					inp_stag.append(tf.zeros(self.shape_of_scale(scale)))
				elif self.is_centered:
					inp.append(tf.zeros(self.shape_of_scale(scale)))
				
				
			
			self.__generator_inputs[scale] = (inp, inp_stag)
		
		return self.__generator_inputs[scale]
	
	def _get_top_scale(self):
		return self.recursive_MS_scales[self.recursive_MS_max_level]
	def _get_top_active_scale(self):
		return self.recursive_MS_scales[self.recursive_MS_current_level]
	def __is_top_scale(self, scale):
		return scale==self.recursive_MS_scales[self.recursive_MS_max_level]
	def __is_top_active_scale(self, scale):
		return scale==self.recursive_MS_scales[self.recursive_MS_current_level]
	def _parent_provides_input_scale(self, scale):
		return self.recursive_MS_direct_input #False
	def _get_larger_scale(self, scale):
		return self.recursive_MS_scales[self.recursive_MS_scales.index(scale)+1]
	
	def set_recursive_MS(self, num_scales: Integral, scale_factor: Number, shared_decoder=True, train_mode="ALL", shapes=None, shape_cast_fn=round, direct_input=False, max_level_input=False):
		self.recursive_MS = True
		self.recursive_MS_direct_input = direct_input
		self.recursive_MS_max_level = num_scales-1
		self.recursive_MS_current_level = self.recursive_MS_max_level
		self.recursive_MS_use_max_level_input = max_level_input
		assert isinstance(scale_factor, Number)
		if scale_factor<=0: LOG.warning("recursive MS scale factor should be positive. is: %s", scale_factor)
		self.recursive_MS_scale_factor = scale_factor
		self.recursive_MS_scale_magnitude = True
		self.recursive_MS_scales = list(range(num_scales))
		self.recursive_MS_shape_cast_fn = shape_cast_fn
		if shapes is None:
			self._recursive_MS_shapes = list(reversed([[int(shape_cast_fn(_/(self.recursive_MS_scale_factor**scale))) for _ in self.centered_shape] for scale in self.recursive_MS_scales]))
		else:
			assert isinstance(shapes, list) and len(shapes)==num_scales
			assert all(isinstance(shape, list) for shape in shapes)
			assert all(has_shape(_, shape_list(self.centered_shape)) for _ in shapes)
			if not all(shapes[-1]==self.centered_shape): LOG.warning("Maximum resolution of shapes provided for recursive MS {} does not match main resolution of this NeuralVelocityGrid {}".format(shapes[-1], self.centered_shape))
			if not all(all(shapes[i]<=shapes[+1]) for i in range(len(shapes)-1)): LOG.warning("Shapes provided for recursive MS are not growing in all dimensions: {}".format(shapes))
			self._recursive_MS_shapes = copy.deepcopy(shapes)
		#self.set_recursive_MS_level(num_scales, scale_factor, shapes, shape_cast_fn)
		
		self.recursive_MS_input_vel = True
		if shared_decoder:
			assert isinstance(self.volume_decoder, (DeferredBackpropNetwork, tf.keras.Model))
		else:
			assert isinstance(self.volume_decoder, list) \
				and len(self.volume_decoder)==(self.recursive_MS_max_level+1) \
				and all(isinstance(_, (DeferredBackpropNetwork, tf.keras.Model)) for _ in self.volume_decoder)
		self.recursive_MS_shared_decoder = shared_decoder
		assert train_mode in ["ALL", "TOP"]
		self.recursive_MS_train_mode = train_mode #"ALL" if self.recursive_MS_shared_decoder else "TOP" #ALL, TOP; no effect with shared decoder
		self.cache_MS = True
		
		LOG.info("Setup recursive multi-scale for NeuralVelocityGrid of frame %s with %d scales (%s), factor %f, %s decoders, training '%s'. The output scales are%s cached.", \
			"?" if self.parent_state is None else self.parent_state.frame, self.num_recursive_MS_scales, self.recursive_MS_shapes, self.recursive_MS_scale_factor, \
			"shared" if self.recursive_MS_shared_decoder else "separate", self.recursive_MS_train_mode, "" if self.has_MS_output else " not")
		
		if self.recursive_MS_shared_decoder and self.recursive_MS_train_mode!="ALL":
			LOG.warning("recursive_MS_train_mode 'TOP' has no effect on shared decoder.")
		
		self.recursive_MS_residual_weights = [1.0 for _ in range(num_scales)]
	
	@property
	def recursive_MS_shapes(self):
		#if self.recursive_MS_shared_decoder:
		#	return list(reversed([[int(self.recursive_MS_shape_cast_fn(_/(self.recursive_MS_scale_factor**scale))) for _ in self.centered_shape] for scale in self.gen_max_MS_scales()]))
		#else:
			return self._recursive_MS_shapes
	
	def set_recursive_MS_level(self, level: Integral, copy_weights: bool = False):
		if not self.recursive_MS: raise RuntimeError("recursive_MS was not set up.")
		if level<0 or self.recursive_MS_max_level<level: raise ValueError("Level must be between 0 and {}, is: {}".format(self.recursive_MS_max_level, level))
		if copy_weights:
			if not self.recursive_MS_current_level==(level-1): raise RuntimeError("copy_weights require to grow recursion by exactly 1.")
			if self.recursive_MS_shared_decoder: raise ValueError("copy_weights requires separate decoders.")
			self.volume_decoder[level].copy_weights_from(self.volume_decoder[self.recursive_MS_current_level])
		self.recursive_MS_current_level = level
		#if scale_factor is None: scale_factor=self.recursive_MS_scale_factor
	
	def gen_current_MS_shapes(self):
		for level in range(self.recursive_MS_current_level+1):
			yield self.recursive_MS_shapes[level]
	def gen_current_MS_scales(self):
		for level in range(self.recursive_MS_current_level+1):
			yield self.recursive_MS_scales[level]
	def gen_current_trainable_MS_scales(self):
		for level in range(0 if self.recursive_MS_train_mode=="ALL" else self.recursive_MS_current_level, self.recursive_MS_current_level+1):
			yield self.recursive_MS_scales[level]
	def gen_max_MS_scales(self):
		for level in range(self.recursive_MS_max_level+1):
			yield self.recursive_MS_scales[level]
	
	def has_same_MS_shapes(self, other):
		if not self.is_MS or not isinstance(other, VelocityGrid) or not other.is_MS:
			return True
		self_shapes = list(self.gen_current_MS_shapes())
		other_shapes = list(other.gen_current_MS_shapes())
		if len(self_shapes)!=len(other_shapes):
			return False
		return all( all(self_dim==other_dim for self_dim, other_dim in zip(self_shape, other_shape)) for self_shape, other_shape in zip(self_shapes, other_shapes))
	
	@property
	def num_recursive_MS_scales(self):
		if self.is_MS:
			assert len(self.recursive_MS_scales)==(self.recursive_MS_max_level+1)
			return self.recursive_MS_max_level+1
		else: return 0
	
	def set_residual_weight(self, level, weight):
		if level<0 or self.recursive_MS_max_level<level:
			raise ValueError("Invalid level for residual weight.")
		self.recursive_MS_residual_weights[level] = weight
	
	def get_residual_weight(self, level):
		if level<0 or self.recursive_MS_max_level<level:
			raise ValueError("Invalid level for residual weight.")
		return self.recursive_MS_residual_weights[level]
		
	
	@property
	def is_centered(self):
		return self.velocity_format=="CENTERED"
	@property
	def is_staggered(self):
		return self.velocity_format=="STAGGERED"
	@property
	def is_MS(self):
		return self.recursive_MS
	@property
	def has_MS_output(self):
		return self.is_MS and (self.cache_output) and (self.cache_MS)
	@property
	def has_multiple_decoders(self):
		multi_dec = self.recursive_MS and (not self.recursive_MS_shared_decoder)
		if multi_dec:
			assert isinstance(self.volume_decoder, list) and all(isinstance(_, (DeferredBackpropNetwork, tf.keras.Model)) for _ in self.volume_decoder), "velocity_decoder setup is invalid"
		else:
			assert isinstance(self.volume_decoder, (DeferredBackpropNetwork, tf.keras.Model)), "velocity_decoder setup is invalid"
		return multi_dec
	
	@property
	def active_decoders(self):
		if self.has_multiple_decoders:
			return [self.volume_decoder[s] for s in self.gen_current_MS_scales()]
		else:
			return [self.volume_decoder]
	
	def centered_shape_of_scale(self, scale):
		#return [int(_/(self.recursive_MS_scale_factor**scale) for _ in self.centered_shape]
		return self.recursive_MS_shapes[scale]
	
	def staggered_shape_of_scale(self, scale):
		return self.shape_centered_to_staggered(self.centered_shape_of_scale(scale))
	
	def shape_of_scale(self, scale):
		return self.staggered_shape_of_scale(scale) if self.is_staggered else self.centered_shape_of_scale(scale)
	#def scale_of_shape(self, shape, is_staggered=None):
	#	assert has_shape(shape, [3])
	#	if is_staggered is None: is_staggered = self.is_staggered
	#	if is_staggered:
	#		
	#	else:
	#	
	
	def _scale_input_down(self, data, scale, mode="RESAMPLE"):
		with SAMPLE("velocity scale input"):
			shape = self.centered_shape_of_scale(scale)
			if self._has_downscale_encoder and mode=="ENCODER":
				data = self._get_downscale_encoder(scale)(data)
				assert has_shape(data, [None]+list(shape)+[None])
			else:
				data = self.scale_renderer.resample_grid3D_aligned(data, shape, allow_split_channels=True)
		return data
	
	
	#def __scale_factor_from_shapes(from_shape, to_shape):
	#	rank = len(shape_list(from_shape))
	#	assert shape_list(from_shape)==shape_list(to_shape)
	#	assert rank<4, "shape should only include spatial dimensions"
	#	return sum(t/f for f,t in zip(from_shape, to_shape))/rank
	def _upscale_velocity(self, vel, scale):
		with SAMPLE("upscale_velocity"):
			#shape = self.shape_of_scale(scale)
			shape = self.centered_shape_of_scale(scale)
			# our sampler can only handle 1,2 and 4 channels, not 3.
			if self.is_staggered:
				#x,y,z = tf.split(vel, 3, axis=-1)
				# TODO: fix alignment in sampling setup to avoid cut and pad?
				x_shape, y_shape, z_shape = VelocityGrid.component_shapes(shape)
				x,y,z = self._staggeredTensor_to_components(vel)
				vel_components = [
					self.scale_renderer.resample_grid3D_aligned(x, x_shape, align_x="CENTER", align_y="BORDER", align_z="BORDER"),
					self.scale_renderer.resample_grid3D_aligned(y, y_shape, align_x="BORDER", align_y="CENTER", align_z="BORDER"),
					self.scale_renderer.resample_grid3D_aligned(z, z_shape, align_x="BORDER", align_y="BORDER", align_z="CENTER")]
				vel = self._components_to_staggeredTensor(*vel_components)
			elif self.is_centered:
				vel_components = tf.split(vel, [2,1], axis=-1)
				vel_components = [self.scale_renderer.resample_grid3D_aligned(_, shape) for _ in vel_components]
				vel =  tf.concat(vel_components, axis=-1)
			if self.recursive_MS_scale_magnitude:
				vel = vel*self.recursive_MS_scale_factor
		return vel
	
	def _upscale_potential(self, vel, scale):
		with SAMPLE("upscale_potential"):
			vel_shape = GridShape.from_tensor(vel)
			#shape = self.shape_of_scale(scale)
			shape = self.centered_shape_of_scale(scale)
			
			# DEBUG
			filter_mode = self.scale_renderer.filter_mode
			self.scale_renderer.filter_mode = "QUADRATIC"
			mip_mode = self.scale_renderer.mip_mode
			self.scale_renderer.mip_mode = "NONE"
			
			# our sampler can only handle 1,2 and 4 channels, not 3.
			if self.is_staggered:
				#x,y,z = tf.split(vel, 3, axis=-1)
				# TODO: fix alignment in sampling setup to avoid cut and pad?
				x_shape, y_shape, z_shape = VelocityGrid.component_potential_shapes(shape)
				x,y,z = self._staggeredTensor_potential_to_components(vel)
				vel_components = [
					self.scale_renderer.resample_grid3D_aligned(x, x_shape, align_x="BORDER", align_y="CENTER", align_z="CENTER"),
					self.scale_renderer.resample_grid3D_aligned(y, y_shape, align_x="CENTER", align_y="BORDER", align_z="CENTER"),
					self.scale_renderer.resample_grid3D_aligned(z, z_shape, align_x="CENTER", align_y="CENTER", align_z="BORDER")]
				vel = self._components_potential_to_staggeredTensor_potential(*vel_components)
			elif self.is_centered:
				vel_components = tf.split(vel, [2,1], axis=-1)
				vel_components = [self.scale_renderer.resample_grid3D_aligned(_, shape) for _ in vel_components]
				vel =  tf.concat(vel_components, axis=-1)
			if self.recursive_MS_scale_magnitude:
				#LOG.info("Scale potental from %s to %s, mag %s", vel_shape, shape, self.recursive_MS_scale_factor)
				vel = vel * (self.recursive_MS_scale_factor**2)
				# Why square factor?:
				# curl operator is linear, so s*curl(v) = curl(s*v)
				# if the potential is interpolated (scale factor 2), the difference between the midpoint and the original values is half that of the original difference
				# thus, the velocity (in world space) is half. so first multiplication is for curl interpolation
				# second multiplication is because we represent velocity in grid space.
			#else:
			#	raise NotImplementedError("DEBUG")
			
			# DEBUG
			self.scale_renderer.filter_mode = filter_mode
			self.scale_renderer.mip_mode = mip_mode
			
		return vel
	
	
	def _warp_centered_inputs(self, tensor, centered_vel):
		#cache = self.__centered
		#self.__centered = centered_vel
		#warped = self.warp(tensor, centered_velocity=centered_vel)
		#self.__centered = cache
		in_shape = GridShape.from_tensor(tensor)
		
		# channel into batch
		tensors = tf.split(tensor, in_shape.c, axis=-1)
		# sample with single channel
		warped = [self.warp(tensor, centered_velocity=centered_vel) for tensor in tensors]
		warped = tf.concat(warped, axis=-1)
		
		return warped
	
	def __normalize_tensor(self, tensor, axes=[-1,-2,-3,-4], eps=1e-5): #3D, norm all except the batch dimension
		with SAMPLE("normalize_tensor"):
			mean, var = tf.nn.moments(tensor, axes, keep_dims=True)
			inv_std = tf.math.rsqrt(var + eps)
			#LOG.info("norm: %s, %s, %s", mean, var, inv_std)
			ret = (tensor - mean ) * inv_std
		return ret
	
	def generate_velocity(self):
		#LOG.debug("Generating density for frame %d", self.parent_state.frame)
		#feature_volume = grad_log(feature_volume, "generate_denstiy start", LOG.info)
		ignore_zero_residual = True
		with SAMPLE("generate_velocity"):
			if self.recursive_MS:
				#if self.use_curl_potential and not self.is_centered:
				#	raise RuntimeError
				#warp_indices = [0,1] if self.input_current_features else [0]
				#self._build_inputs(scales) #create the input pyramid, subject to caching?
				#LOG.info("centered MS scales: %s", self.recursive_MS_shapes)
				active_MS_scales = list(self.gen_current_MS_scales())
				vel = tf.zeros([self._get_batch_size()] + self.shape_of_scale(active_MS_scales[0]) + [3]) #vel initialized as zero
				for s in active_MS_scales:
					with SAMPLE("level %s"%(s,)):
						residual_weight = self.recursive_MS_residual_weights[s]
						
						# no need to warp with or scale zero vel
						if s is not active_MS_scales[0]:
							with SAMPLE("upscale prev"):
								#upscale previous velocity or potential. create centered vel to warp inputs
								if self.use_curl_potential and self.residual_potential:
									vel = self._upscale_potential(vel, s)
									centered_vel = self._centered_to_curl(vel) if self.is_centered else self._staggeredTensor_potential_to_centered(vel)
								else:
									vel = self._upscale_velocity(vel, s)
									centered_vel = vel if self.is_centered else self._staggered_to_centered(vel)
						
						if (not (residual_weight==0 and ignore_zero_residual)) or (s is active_MS_scales[0]):
							inputs_frames, inputs_frames_stag = self._get_generator_input_MS(s) #list of inputs for current level/scale
							
							# no need to warp with or scale zero vel
							if s is not active_MS_scales[0]:
								with SAMPLE("warp inputs"):
									#warp inputs
									for idx in self.warp_input_indices:
										inputs_frames[idx] = self._warp_centered_inputs(inputs_frames[idx], centered_vel) #warp the current frame to the target, expected to be first input
							
							#transform to staggered after warp
							with SAMPLE("transform format"):
								if self.is_staggered:
									inputs_frames = [self._scalar_centered_to_staggered(_, allow_split_channels=True) for _ in inputs_frames] #resample to .5 shifted grid with dim+1
									inputs_frames += inputs_frames_stag
								if self.is_centered:
									inputs_frames += [self._staggered_to_centered(_) for _ in inputs_frames_stag]
							
							with SAMPLE("normalize"):
								num_density_inputs = len(self.step_input_density) + len(self.step_input_density_target) + len(self.step_input_density_proxy)
								num_feature_inputs = len(self.step_input_features)
								if self.norm_input_mode=="SINGLE":
									#inputs_frames = map(normalize_tensor, inputs_frames)
									for idx in range(num_density_inputs+num_feature_inputs):
										# we don't want to normalize velocities
										inputs_frames[idx] = self.__normalize_tensor(inputs_frames[idx])
								
								elif self.norm_input_mode=="GROUP":
									#inputs_frames at this point: densities, features, recurrent velocity
									dens_inputs = inputs_frames[:num_density_inputs]
									features_inputs = inputs_frames[num_density_inputs:num_density_inputs+num_feature_inputs]
									inputs_frames = inputs_frames[num_density_inputs+num_feature_inputs:]
									
									if num_density_inputs>0:
										inputs_frames.insert(0, self.__normalize_tensor(tf.concat(dens_inputs, axis=-1)))
									del dens_inputs
									
									if num_feature_inputs>0:
										inputs_frames.insert(1, self.__normalize_tensor(tf.concat(features_inputs, axis=-1)))
									del features_inputs
								
								elif self.norm_input_mode=="ALL" and (num_density_inputs+num_feature_inputs)>0:
									inputs_frames = [self.__normalize_tensor(tf.concat(inputs_frames[:num_density_inputs+num_feature_inputs], axis=-1))] + inputs_frames[num_density_inputs+num_feature_inputs:]
							
							with SAMPLE("make input"):
								# vel is also input
								if self.recursive_MS_input_vel:
									inputs_frames.append(vel)
								
								#LOG.info("input stats: %s", [(tf.reduce_mean(_).numpy().tolist(), tf.reduce_min(_).numpy().tolist(), tf.reduce_max(_).numpy().tolist()) for _ in inputs_frames])
								inputs_frames = tf.concat(inputs_frames, axis=-1)
							
							with SAMPLE("volume_decoder"):
								if self.recursive_MS_shared_decoder:
									vel_residual = self.volume_decoder(inputs_frames)
								else:
									vel_residual = self.volume_decoder[s](inputs_frames)
							
							with SAMPLE("make output"):
								# https://de.wikipedia.org/wiki/Rotation_eines_Vektorfeldes#Rechenregeln
								# curl(c*F + G) = c*curl(F) + curl(G)
								# => could use residual potential instead of (or in addition to) residual velocity
								if self.use_curl_potential:
									if not self.residual_potential:
										vel_residual = self._centered_to_curl(vel_residual) if self.is_centered else self._staggeredTensor_potential_to_staggeredTensor(vel_residual) #turn potential into divergence-free velocity
										vel = vel + self.recursive_MS_residual_weights[s] * vel_residual
									else:
										#warnings.warn("Only using scaled min scale vel!")
										#if s is active_MS_scales[0]:
										vel = vel + self.recursive_MS_residual_weights[s] * vel_residual
										vel_potential = vel
										#vel_residual_potential = vel_residual
										
										vel = self._centered_to_curl(vel) if self.is_centered else self._staggeredTensor_potential_to_staggeredTensor(vel)
										vel_residual = self._centered_to_curl(vel_residual) if self.is_centered else self._staggeredTensor_potential_to_staggeredTensor(vel_residual)
								else:
									#vel += vel_residual
									vel = vel + self.recursive_MS_residual_weights[s] * vel_residual
						
						else:
							vel_residual = tf.zeros_like(vel)
							if self.use_curl_potential and self.residual_potential:
								vel_potential = vel
								vel = self._centered_to_curl(vel) if self.is_centered else self._staggeredTensor_potential_to_staggeredTensor(vel)

						if self.cache_MS:
							with SAMPLE("cache"):
								if self.is_centered:
									with tf.device(self._device):
										self.__centered_MS[s] = tf.identity(vel)
										self.__centered_MS_residual[s] = tf.identity(vel_residual)
								if self.is_staggered:
									tmp = self._staggeredTensor_to_components(vel)
									with tf.device(self._device):
										self.__staggered_MS[s] = [tf.identity(_) for _ in tmp]
									tmp = self._staggeredTensor_to_components(vel_residual)
									with tf.device(self._device):
										self.__staggered_MS_residual[s] = [tf.identity(_) for _ in tmp]
									del tmp
								# if self.use_curl_potential:
									# with tf.device(self._device):
										# self.__curl_potential_MS[s] = vel_potential
									# self.__output_cache[("curl_potential_MS",s)] = vel_potential
						
						if self.use_curl_potential and self.residual_potential and s is not active_MS_scales[-1]:
							vel = vel_potential
					
					# END scale profilie
				
				# END scale loop
				
				self.clear_input_cache() #?
				#with tf.device(self._device):
				#	self.__curl_potential = tf.identity(vel_potential) if self.is_centered else [tf.identity(_) for _ in vel_potential]
				v = vel #if self.centered else self._staggeredTensor_to_components(vel)
			else:
				inputs_frames, inputs_frames_stag = self._get_generator_input() #list of inputs for current level/scale
				
				with SAMPLE("transform format"):
					#transform to staggered after warp
					if self.is_staggered:
						inputs_frames = [self._scalar_centered_to_staggered(_, allow_split_channels=True) for _ in inputs_frames] #resample to .5 shifted grid with dim+1
						inputs_frames += inputs_frames_stag
					if self.is_centered:
						inputs_frames += [self._staggered_to_centered(_) for _ in inputs_frames_stag]
				
				with SAMPLE("normalize"):
					num_density_inputs = len(self.step_input_density) + len(self.step_input_density_target) + len(self.step_input_density_proxy)
					num_feature_inputs = len(self.step_input_features)
					if self.norm_input_mode=="SINGLE":
						#inputs_frames = map(normalize_tensor, inputs_frames)
						for idx in range(num_density_inputs+num_feature_inputs):
							# we don't want to normalize velocities
							inputs_frames[idx] = self.__normalize_tensor(inputs_frames[idx])
					
					elif self.norm_input_mode=="GROUP":
						#inputs_frames at this point: densities, features, recurrent velocity
						dens_inputs = inputs_frames[:num_density_inputs]
						features_inputs = inputs_frames[num_density_inputs:num_density_inputs+num_feature_inputs]
						inputs_frames = inputs_frames[num_density_inputs+num_feature_inputs:]
						
						if num_density_inputs>0:
							inputs_frames.insert(0, self.__normalize_tensor(tf.concat(dens_inputs, axis=-1)))
						del dens_inputs
						
						if num_feature_inputs>0:
							inputs_frames.insert(1, self.__normalize_tensor(tf.concat(features_inputs, axis=-1)))
						del features_inputs
					
					elif self.norm_input_mode=="ALL" and (num_density_inputs+num_feature_inputs)>0:
						inputs_frames = [self.__normalize_tensor(tf.concat(inputs_frames[:num_density_inputs+num_feature_inputs], axis=-1))] + inputs_frames[num_density_inputs+num_feature_inputs:]
				
				with SAMPLE("make input"):
					#LOG.info("input stats: %s", [(tf.reduce_mean(_).numpy().tolist(), tf.reduce_min(_).numpy().tolist(), tf.reduce_max(_).numpy().tolist()) for _ in inputs_frames])
					inputs_frames = tf.concat(inputs_frames, axis=-1)
				
				with SAMPLE("volume_decoder"):
					vel = self.volume_decoder(inputs_frames)
				
				with SAMPLE("make output"):
					# https://de.wikipedia.org/wiki/Rotation_eines_Vektorfeldes#Rechenregeln
					# curl(c*F + G) = c*curl(F) + curl(G)
					# => could use residual potential instead of (or in addition to) residual velocity
					if self.use_curl_potential:
						vel = self._centered_to_curl(vel) if self.is_centered else self._staggeredTensor_potential_to_staggeredTensor(vel)
						
				self.clear_input_cache() #?
				v = vel #if self.centered else self._staggeredTensor_to_components(vel)
			
			if self.is_staggered:
				with SAMPLE("make components"):
					v = self._staggeredTensor_to_components(v)
			#d = grad_log(d, "generate_denstiy end", LOG.info)
		return v
	
	
	def centered(self, pad_lod=False, concat=True):
		if self.__centered is not None:
			v = self.__centered
		else:
			if self.is_centered:
				v = self.generate_velocity()
			elif self.is_staggered:
				v = super().centered(pad_lod=pad_lod, concat=True)
			
			if self.cache_output:
				with tf.device(self._device):
					self.__centered = tf.identity(v)
				v = self.__centered
		if pad_lod:
			raise NotImplementedError("pad_lod is not supported, it would conflict with caching and gradient calculations.")
		if not concat:
			#raise NotImplementedError("not concat is not supported, it would conflict with caching and gradient calculations.")
			return tf.split(v, 3, axis=-1)
		return v
	
	def centered_MS(self, scale, pad_lod=False, concat=True):
		if not self.has_MS_output: raise ValueError("Mutiscale velocity is not available.")
		#if not self.is_centered: raise NotImplementedError("accessor staggered MS not implemented.")
		if self.is_centered:
			if self.__centered is not None:
				if scale not in self.__centered_MS:
					raise ValueError("Mutiscale velocity of scale \"{}\" is not available. Available scales: {}".format(scale, list(self.__centered_MS.keys())))
			else:
				if scale not in self.recursive_MS_scales:
					raise ValueError("Mutiscale velocity of scale \"{}\" is not available. Available scales: {}".format(scale, self.recursive_MS_scales))
				self.centered()
			v = self.__centered_MS[scale]
		elif self.is_staggered:
			if scale not in self.__centered_MS:
				v = self._components_to_centered(*self._staggered_MS(scale))
				with tf.device(self._device):
					self.__centered_MS[scale] = tf.identity(v)
			v = self.__centered_MS[scale]
		if pad_lod:
			raise NotImplementedError("pad_lod is not supported, it would conflict with caching and gradient calculations.")
		if not concat:
			#raise NotImplementedError("not concat is not supported, it would conflict with caching and gradient calculations.")
			return tf.split(v, 3, axis=-1)
		return v
	def centered_MS_residual(self, scale, pad_lod=False, concat=True):
		if not self.has_MS_output: raise ValueError("Mutiscale velocity is not available.")
		#if not self.is_centered: raise NotImplementedError("accessor staggered MS not implemented.")
		if self.is_centered:
			if self.__centered is not None:
				if scale not in self.__centered_MS_residual:
					raise ValueError("Mutiscale velocity of scale \"{}\" is not available. Available scales: {}".format(scale, list(self.__centered_MS_residual.keys())))
			else:
				if scale not in self.recursive_MS_scales:
					raise ValueError("Mutiscale velocity of scale \"{}\" is not available. Available scales: {}".format(scale, self.recursive_MS_scales))
				self.centered()
			v = self.__centered_MS_residual[scale]
		elif self.is_staggered:
			if scale not in self.__centered_MS_residual:
				v = self._components_to_centered(*self._staggered_MS_residual(scale))
				with tf.device(self._device):
					self.__centered_MS_residual[scale] = tf.identity(v)
			v = self.__centered_MS_residual[scale]
		if pad_lod:
			raise NotImplementedError("pad_lod is not supported, it would conflict with caching and gradient calculations.")
		if not concat:
			#raise NotImplementedError("not concat is not supported, it would conflict with caching and gradient calculations.")
			return tf.split(v, 3, axis=-1)
		return v
	
	def _staggered(self):
		if self.__staggered is not None:
			v = self.__staggered
		else:
			if self.is_staggered:
				v = self.generate_velocity()
			elif self.is_centered:
				v = self._centered_to_staggered(self.centered())
			
			if self.cache_output:
				with tf.device(self._device):
					self.__staggered = [tf.identity(_) for _ in v]
				v = self.__staggered
		return v
	@property
	def _x(self):
		return self._staggered()[0]
	@property
	def _y(self):
		return self._staggered()[1]
	@property
	def _z(self):
		return self._staggered()[2]
	
	def _staggered_MS(self, scale):
		if not self.has_MS_output: raise ValueError("Mutiscale velocity is not available.")
		#if not self.is_centered: raise NotImplementedError("accessor staggered MS not implemented.")
		if self.is_staggered:
			# check if base vel has been generated. if not, do so to also generate the MS scales.
			if self.__staggered is not None:
				if scale not in self.__staggered_MS: #velocity has been generated, but the scale is (still) not available
					raise ValueError("Mutiscale velocity of scale \"{}\" is not available. Available scales: {}".format(scale, list(self.__staggered_MS.keys())))
			else:
				if scale not in self.recursive_MS_scales:
					raise ValueError("Mutiscale velocity of scale \"{}\" is not available. Available scales: {}".format(scale, self.recursive_MS_scales))
				self._staggered()
			v = self.__staggered_MS[scale]
		elif self.is_centered:
			if scale not in self.__staggered_MS:
				v = self._centered_to_staggered(self.centered_MS(scale))
				with tf.device(self._device):
					self.__staggered_MS[scale] = [tf.identity(_) for _ in v]
			v = self.__staggered_MS[scale]
		return v
	def _staggered_MS_residual(self, scale):
		if not self.has_MS_output: raise ValueError("Mutiscale velocity is not available.")
		#if not self.is_centered: raise NotImplementedError("accessor staggered MS not implemented.")
		if self.is_staggered:
			if self.__staggered is not None:
				if scale not in self.__staggered_MS_residual:
					raise ValueError("Mutiscale velocity of scale \"{}\" is not available. Available scales: {}".format(scale, list(self.__staggered_MS_residual.keys())))
			else:
				if scale not in self.recursive_MS_scales:
					raise ValueError("Mutiscale velocity of scale \"{}\" is not available. Available scales: {}".format(scale, self.recursive_MS_scales))
				self._staggered()
			v = self.__staggered_MS_residual[scale]
		elif self.is_centered:
			if scale not in self.__staggered_MS_residual:
				v = self._centered_to_staggered(self.centered_MS_residual(scale))
				with tf.device(self._device):
					self.__staggered_MS_residual[scale] = [tf.identity(_) for _ in v]
			v = self.__staggered_MS_residual[scale]
		return v
	
	def _x_MS(self, scale):
		return self._staggered_MS(scale)[0]
	def _y_MS(self, scale):
		return self._staggered_MS(scale)[1]
	def _z_MS(self, scale):
		return self._staggered_MS(scale)[2]
	def x_MS(self, scale):
		if self.boundary is not None: raise NotImplementedError("Boundaries not available for multi-scale velocity.")
		return self._x_MS(scale)
	def y_MS(self, scale):
		if self.boundary is not None: raise NotImplementedError("Boundaries not available for multi-scale velocity.")
		return self._y_MS(scale)
	def z_MS(self, scale):
		if self.boundary is not None: raise NotImplementedError("Boundaries not available for multi-scale velocity.")
		return self._z_MS(scale)
	
	def _x_MS_r(self, scale):
		return self._staggered_MS_residual(scale)[0]
	def _y_MS_r(self, scale):
		return self._staggered_MS_residual(scale)[1]
	def _z_MS_r(self, scale):
		return self._staggered_MS_residual(scale)[2]
	def x_MS_r(self, scale):
		if self.boundary is not None: raise NotImplementedError("Boundaries not available for multi-scale velocity.")
		return self._x_MS_r(scale)
	def y_MS_r(self, scale):
		if self.boundary is not None: raise NotImplementedError("Boundaries not available for multi-scale velocity.")
		return self._y_MS_r(scale)
	def z_MS_r(self, scale):
		if self.boundary is not None: raise NotImplementedError("Boundaries not available for multi-scale velocity.")
		return self._z_MS_r(scale)
	
	
	def divergence_MS(self, scale, world_scale=[1,1,1]):
		#out - in per cell, per axis
		x_div = self.x_MS(scale)[:,:,:,1:,:] - self.x_MS(scale)[:,:,:,:-1,:]
		y_div = self.y_MS(scale)[:,:,1:,:,:] - self.y_MS(scale)[:,:,:-1,:,:]
		z_div = self.z_MS(scale)[:,1:,:,:,:] - self.z_MS(scale)[:,:-1,:,:,:]
		# sum to get total divergence per cell
		div = x_div*world_scale[0]+y_div*world_scale[1]+z_div*world_scale[2]
		return div
	
	def divergence_MS_residual(self, scale, world_scale=[1,1,1]):
		#out - in per cell, per axis
		x_div = self.x_MS_r(scale)[:,:,:,1:,:] - self.x_MS_r(scale)[:,:,:,:-1,:]
		y_div = self.y_MS_r(scale)[:,:,1:,:,:] - self.y_MS_r(scale)[:,:,:-1,:,:]
		z_div = self.z_MS_r(scale)[:,1:,:,:,:] - self.z_MS_r(scale)[:,:-1,:,:,:]
		# sum to get total divergence per cell
		div = x_div*world_scale[0]+y_div*world_scale[1]+z_div*world_scale[2]
		return div
	
	def magnitude_MS(self, scale, world_scale=[1,1,1]):
		with self.warp_renderer.profiler.sample("magnitude"):
			v = self.centered_MS(scale, pad_lod=False)*tf.constant(world_scale, dtype=tf.float32)
			return tf_norm2(v, axis=-1, keepdims=True) #tf.norm(v, axis=-1, keepdims=True)
	
	def magnitude_MS_residual(self, scale, world_scale=[1,1,1]):
		with self.warp_renderer.profiler.sample("magnitude"):
			v = self.centered_MS_residual(scale, pad_lod=False)*tf.constant(world_scale, dtype=tf.float32)
			return tf_norm2(v, axis=-1, keepdims=True) #tf.norm(v, axis=-1, keepdims=True)
	
	
	def var_list(self):
		raise AttributeError("use .get_variables().")
	
	@property
	def requires_parent_state_variables(self):
		return False #"ENC3D" in self.type_input_features
	
	#def _get_variables_per_decoder(self):
	#	pass
	def get_variables(self, scale=None):
		if self.has_multiple_decoders: #recursive_MS and (not self.recursive_MS_shared_decoder):
			if scale is None:
				if self.recursive_MS_train_mode=="ALL":
					var_dict = {'velocity_decoder': [var for dec in self.active_decoders for var in dec.trainable_variables]}
				elif self.recursive_MS_train_mode=="TOP":
					var_dict = {'velocity_decoder': self.volume_decoder[self.recursive_MS_scales[self.recursive_MS_current_level]].trainable_variables}
				else: raise ValueError
			else:
				var_dict = {'velocity_decoder': self.volume_decoder[scale].trainable_variables}
		else:
			var_dict = {'velocity_decoder': self.volume_decoder.trainable_variables}
		if self._has_input_encoder:
			var_dict["input_encoder"] = [var for enc in self.active_input_encoders for var in enc.trainable_variables]
		if self._has_downscale_encoder:
			var_dict["downscale_encoder"] = [var for enc in self.active_downscale_encoders for var in enc.trainable_variables]
		#if self.requires_parent_state_variables and self.parent_state is not None:
		#	var_dict.update(self.parent_state.get_variables())
		return var_dict
	
	# def get_input_variables(self):
		# # output tensors of other Grid objects that are used as input to this network
		# # density output of state.density, state.density_proxy
		# # velocity and curl potential of state.velocity
		# # volume feature tensor of state
		# # of all used MS scales that are not generated by _get_generator_input_MS via downscaling
		# raise NotImplementedError
		# var_inp = {}
		# if scale not in self.__generator_inputs:
			# if self._is_top_scale(scale) or self._parent_provides_input_scale(scale):
				# # input may not be generated and cached yet
				# self._get_generator_input_MS(scale)
	
	
	def _scatter_generator_input_gradients(self, input_gradients):
		raise NotImplementedError
		recurrent_input_vel = False
		recurrent_input_vel_MS = False #use all scales or just top?
		recurrent_input_vel_potential = True #use generated potential instead of velocity if generating potentials
		warp_recurrent_vel = False #try this?
		# TODO: recursive_MS input generation; possibly multi-scale input/features provided by parent state
		if scale not in self.__generator_inputs:
			if self._is_top_scale(scale) or self._parent_provides_input_scale(scale):
				with SAMPLE("velocity top MS input"):
					i = 0
					for step in self.step_input_density:
						state = self._get_state_by_step(step)
						if state is not None:
							#inp.append(state.density.scaled(self.centered_shape, with_inflow=True))
							state.density.add_output_gradients(input_gradients[scale][i])
						i+=1
					
					for step in self.step_input_density_target:
						state = self._get_state_by_step(step)
						if state is not None:
							# inp.append(state.density_target.scaled(self.centered_shape, with_inflow=True))
							state.density_target.add_output_gradients(input_gradients[scale][i])
						i+=1
					
					for step in self.step_input_density_proxy:
						state = self._get_state_by_step(step)
						if state is not None and state.has_density_proxy:
							# inp.append(state.density_proxy.scaled(self.centered_shape, with_inflow=True))
							state.density_proxy.add_output_gradients(input_gradients[scale][i])
						i+=1
					
					for step in self.step_input_features:
						state = self._get_state_by_step(step)
						if state is not None:
							if self.use_raw_images:
								raise NotImplementedError
							else:
								#inp.append(state.get_volume_features(types=self.type_input_features))
								state.add_volume_feature_gradients(input_gradients[scale][i], types=self.type_input_features)
						i+=1
						
					
					inp_stag = []
					if recurrent_input_vel and not (self.parent_state.prev.velocity.has_MS_output and recurrent_input_vel_MS):
						if self.parent_state.prev is not None:
							self.parent_state.prev.velocity.add_output_gradients(input_gradients[scale][i])
							# if self.parent_state.prev.velocity.use_curl_potential and recurrent_input_vel_potential:
								# inp_stag.append(self.parent_state.prev.velocity.__curl_potential_MS[scale])
							# elif self.is_staggered:
								# inp_stag.append(self._components_to_staggeredTensor(*self.parent_state.prev.velocity._staggered(scale)))
							# elif self.is_centered:
								# inp.append(self.parent_state.prev.velocity.centered(scale))
			
			# if recurrent_input_vel and (self.parent_state.prev.velocity.has_MS_output and recurrent_input_vel_MS):
				# if self.parent_state.prev is not None:
					# if self.parent_state.prev.velocity.use_curl_potential and recurrent_input_vel_potential:
						# inp_stag.append(self.parent_state.prev.velocity.__curl_potential_MS[scale])
					# elif self.is_staggered:
						# inp_stag.append(self.parent_state.prev.velocity._staggered_MS(scale))
					# elif self.is_centered:
						# inp.append(self.parent_state.prev.velocity.centered_MS(scale))
			
			# if recurrent_input_vel and self.parent_state.prev is None:
				# if self.is_staggered:
					# inp_stag.append(tf.zeros(self.shape_of_scale(scale)))
				# elif self.is_centered:
					# inp.append(tf.zeros(self.shape_of_scale(scale)))
				
	
	def __get_MS_variable_keys_scale(self, scale, centered=True, staggered=True, include_residual=False):
		key_list = []
		if centered:
			key_list.append('velocity_%s_c'%(scale,))
			if include_residual:
				key_list.append('velocity_r%s_c'%(scale,))
		if staggered:
			key_list.append('velocity_%s_x'%(scale,))
			key_list.append('velocity_%s_y'%(scale,))
			key_list.append('velocity_%s_z'%(scale,))
			if include_residual:
				key_list.append('velocity_r%s_x'%(scale,))
				key_list.append('velocity_r%s_y'%(scale,))
				key_list.append('velocity_r%s_z'%(scale,))
		
		return key_list
	
	def _get_output_variable_keys(self, centered=True, staggered=True, include_MS=False, include_residual=False):
		key_list = []
		if include_MS:
			for scale in self.gen_current_MS_scales():
				key_list.extend(self.__get_MS_variable_keys_scale(scale, centered=centered, staggered=staggered, include_residual=include_residual))
		else:
			if centered:
				key_list.append('velocity_c')
			if staggered:
				key_list.append('velocity_x')
				key_list.append('velocity_y')
				key_list.append('velocity_z')
		
		return key_list
	
	def get_output_variables(self, centered=True, staggered=True, include_MS=False, include_residual=False, only_trainable=False):
		if not self.cache_output:
			raise RuntimeError("Output caching must be enabled for output variables to have meaning.")
		var_dict = {}
		if include_MS and self.has_MS_output:
			scales = list(self.gen_current_trainable_MS_scales()) if only_trainable else list(self.gen_current_MS_scales())
			for scale in scales:
				if centered:
					var_dict['velocity_%s_c'%(scale,)] = self.centered_MS(scale)
					if include_residual:
						var_dict['velocity_r%s_c'%(scale,)] = self.centered_MS_residual(scale)
				if staggered:
					var_dict['velocity_%s_x'%(scale,)] = self._x_MS(scale)
					var_dict['velocity_%s_y'%(scale,)] = self._y_MS(scale)
					var_dict['velocity_%s_z'%(scale,)] = self._z_MS(scale)
					if include_residual:
						var_dict['velocity_r%s_x'%(scale,)] = self._x_MS_r(scale)
						var_dict['velocity_r%s_y'%(scale,)] = self._y_MS_r(scale)
						var_dict['velocity_r%s_z'%(scale,)] = self._z_MS_r(scale)
		else:
			if centered:
				var_dict['velocity_c'] = self.centered()
			if staggered:
				var_dict['velocity_x'] = self._x
				var_dict['velocity_y'] = self._y
				var_dict['velocity_z'] = self._z
		
		return var_dict
		
	def map_gradient_output_to_MS(self, grad_dict):
		assert isinstance(grad_dict, dict)
		output_keys = self._get_output_variable_keys(centered=True, staggered=True, include_MS=False)
		MS_keys = self.__get_MS_variable_keys_scale(scale=self._get_top_active_scale(), centered=True, staggered=True, include_residual=False)
		assert len(output_keys)==len(MS_keys)
		#also assume same order: c, x, y, z
		assert all(a.endswith(b[-2:]) for a,b in zip(output_keys, MS_keys))
		out_dict = {}
		for key, grad in grad_dict.items():
			if key not in output_keys: raise KeyError("Invalid output gradient key '{}'".format(key))
			out_dict[MS_keys[output_keys.index(key)]] = grad
		return out_dict
	
	def assign(self, x,y,z):
		raise TypeError("Can't assign to NeuralVelocityGrid")
	
	def assign_add(self, x,y,z):
		raise TypeError("Can't assign_add to NeuralVelocityGrid")
	
	def assign_sub(self, x,y,z):
		raise TypeError("Can't assign_sub to NeuralVelocityGrid")
	
	# def _cache(self, key, value):
		# self.__output_cache[key] = value
	
	def clear_cache(self):
		self.__output_cache.clear()
		self.__centered = None
		self.__centered_MS = {}
		self.__centered_MS_residual = {}
		self.__staggered = None
		self.__staggered_MS = {}
		self.__staggered_MS_residual = {}
		self.clear_input_cache()
		
		self.__output_users = {"CENTERED": []}
		self.__output_gradient_cache = None
		self.__inputs_for_backprop = []
		self.__input_grad_cache.clear()
	
	def clear_input_cache(self):
		self.__generator_inputs = {}
		self.__input_cache.clear()
	
	def clear_cache_for_backprop(self):
		# self.clear_cache()
		# if self.requires_parent_state_variables:
			# self.parent_state.clear_cache_for_backprop()
		self.__output_cache.clear()
		#self.__curl_potential = None
		#self.__curl_potential_MS = {}
		self.__centered = None
		self.__centered_MS = {}
		self.__centered_MS_residual = {}
		self.__staggered = None
		self.__staggered_MS = {}
		self.__staggered_MS_residual = {}
		self.clear_input_cache()
	
	def inspect_output_gradient_stats(self, opt_ctx):
		if self.__output_gradient_cache is not None and "output_gradients" in self.__output_gradient_cache:
			for name, grad in self.__output_gradient_cache["output_gradients"].tiems():
				opt_ctx.inspect_gradients_func(opt_ctx=opt_ctx, gradients=grad, name="vel_out/"+name)
		opt_ctx.inspect_gradients_func(opt_ctx=opt_ctx, gradients=self._get_output_users_grads()["CENTERED"], name="velc_out/users")
	
	def add_output_gradients(self, output_gradients):
		raise NotImplementedError
		if not isinstance(output_gradients, dict): raise TypeError
		for key, grad in output_gradients.items():
			if key in self.__output_gradient_cache:
				grad = grad + self.__output_gradient_cache[key]
			self.__output_gradient_cache[key] = grad
	
	# def __get_output_grads(self):
		# if self.__output_gradient_cache is not None and "output_gradients" in self.__output_gradient_cache:
			# return self.__output_gradient_cache
		# else:
			# out_vars = self.get_output_variables(True, True, self.is_MS, True, True)
			# out_grads = {
				# "output_gradients": {name: tf.zeros_like(var) for name, var in out_vars.items()}
				# "include_MS": self.is_MS,
				# "include_residual": True,
				# "only_trainable": True,
			# }
			# return out_grads
	
	def _get_output_users_grads(self):
		for out_id, users in self.__output_users.items():
			for other in users:
				other._compute_input_grads()
		extra_output_gradients = {}
		for out_id, users in self.__output_users.items():
			extra_output_gradients[out_id] = tf.zeros_like(self.output(out_id))
			for other in users:
				extra_output_gradients[out_id] = extra_output_gradients[out_id] + other._get_input_grad(self, out_id)
		
		# extra_output_gradients = tf.zeros_like(self.output())
		# if self.__output_users is not None and len(self.__output_users)>0:
			# assert len(self.__output_users)==1
			# assert isinstance(self.__output_users[0], WarpedDensityGrid)
			# for other in self.__output_users:
				# other._compute_input_grads()
			# for other in self.__output_users:
				# extra_output_gradients = extra_output_gradients + other._get_input_grad(self)
		return extra_output_gradients
	
	def backprop(self, output_gradients, include_MS=False, include_residual=False, only_trainable=False, provide_input_gradients=False, keep_output_gradients=False):
		
		#assert output_gradients is None, "DEBUG"
		
		check_output_users(self.__output_users, {"CENTERED": [WarpedDensityGrid]}, "NeuralVelocityGrid")
		
		extra_output_gradients = {"velocity_c":self._get_output_users_grads()["CENTERED"]}
		if include_MS:
			extra_output_gradients = self.map_gradient_output_to_MS(extra_output_gradients)
		
		for k in output_gradients:
			if k in extra_output_gradients:
				if output_gradients[k] is None:
					output_gradients[k] = extra_output_gradients[k]
				else:
					output_gradients[k] += extra_output_gradients[k]
		
		for k in extra_output_gradients:
			if k not in output_gradients:
				raise KeyError("could not map extra output gradient '%s'."%(k,))
		
		LOG.debug("Backprop velocity frame %d", self.parent_state.frame)
		with SAMPLE("NVG backprop"):
			if not include_MS and not (('velocity_c' in output_gradients) or \
					('velocity_x' in output_gradients and 'velocity_y' in output_gradients and 'velocity_z' in output_gradients) \
				):
				raise ValueError("No gradients to backprop.")
			if include_MS:
				for scale in (self.gen_current_trainable_MS_scales() if only_trainable else self.gen_current_MS_scales()):
					if not (('velocity_%s_c'%(scale,) in output_gradients) or \
							('velocity_%s_x'%(scale,) in output_gradients and 'velocity_%s_y'%(scale,) in output_gradients and 'velocity_%s_z'%(scale,) in output_gradients)) \
						or ( \
							include_residual and not (('velocity_r%s_c'%(scale,) in output_gradients) or \
							('velocity_r%s_x'%(scale,) in output_gradients and 'velocity_r%s_y'%(scale,) in output_gradients and 'velocity_r%s_z'%(scale,) in output_gradients)) \
						):
						raise ValueError("No gradients to backprop for MS scale %s."%(scale,))
			self.clear_cache_for_backprop()
			var_dict = self.get_variables()
			var_dict["inputs_for_backprop"] = self.__gather_inputs()
			
			if provide_input_gradients:
				var_dict.update(self.get_input_variables())
			
			with SAMPLE("forward"), tf.GradientTape(watch_accessed_variables=False) as tape:
				tape.watch(var_dict)
				output = self.get_output_variables(include_MS=include_MS, include_residual=include_residual)
			
			output = {k: output[k] for k in output if (k in output_gradients)}
			output_gradients = {
				k: (output_gradients[k] if output_gradients[k] is not None else tf.zeros_like(output[k])) \
				for k in output_gradients \
				if k in output \
			}
			
		#	for k in output:
		#		LOG.info("vel output '%s': out type=%s, grad type=%s", k, output[k].__class__.__name__, output_gradients[k].__class__.__name__)
			with SAMPLE("gradients"):
				gradients = tape.gradient(output, var_dict, output_gradients=output_gradients)
		
		for i, grad in enumerate(gradients["inputs_for_backprop"]):
			self.__input_grad_cache["input_grad_%d"%(i,)] = grad
		del gradients["inputs_for_backprop"]
		
		return gradients
	
	def __split_gradients_for_decoders(self, grads, decoders):
		#assert self.has_multiple_decoders and self.recursive_MS_train_mode=="ALL"
		num_var_per_decoder = [len(dec.trainable_variables) for dec in decoders]
		if sum(num_var_per_decoder)!=len(grads): raise RuntimeError("gradients do not match variables: got {} gradients for {} variables of {} decoders {}".\
			format(len(grads), sum(num_var_per_decoder), len(decoders), num_var_per_decoder))
		split_grads = []
		pos = 0
		for size in num_var_per_decoder:
			split_grads.append(grads[pos:pos+size])
			pos +=size
		assert all(size==len(grad) for size, grad in zip(num_var_per_decoder, split_grads))
		return split_grads
	
	def set_output_gradients_for_backprop_accumulate(self, output_gradients, **kwargs):
		kwargs["output_gradients"] = output_gradients
		self.__output_gradient_cache = kwargs
	
	def backprop_accumulate(self, output_gradients, include_MS=False, include_residual=False, only_trainable=False, provide_input_gradients=False, keep_output_gradients=False):
		
		gradients = self.backprop(output_gradients, include_MS=include_MS, include_residual=include_residual, only_trainable=only_trainable, \
			provide_input_gradients=provide_input_gradients, keep_output_gradients=keep_output_gradients)
		
		with SAMPLE("NVG acc grads"):
			if provide_input_gradients:
				raise NotImplementedError
				# can just add density variables and backprop through the whole thing, if memory allows
				# currently grads there are blocked in input assembler to prevent them flowing to the volume encoder in state via the density.
				input_gradients = gradients['inputs']
			
			vel_net_gradients = gradients['velocity_decoder']
			if self.has_multiple_decoders:
				if self.recursive_MS_train_mode=="ALL":
					#var_dict = {'velocity_decoder': [var for var in dec.trainable_variables for dec in self.volume_decoder]}
					# definitely not the best way of doing this...
					decoders = self.active_decoders
					for dec, grads in zip(decoders, self.__split_gradients_for_decoders(vel_net_gradients, decoders)):
						dec.add_gradients(grads)
				elif self.recursive_MS_train_mode=="TOP":
					#var_dict = {'velocity_decoder': self.volume_decoder[self.recursive_MS_scales[self.recursive_MS_current_level]].trainable_variables}
					self.volume_decoder[self.recursive_MS_scales[self.recursive_MS_current_level]].add_gradients(vel_net_gradients)
				else: raise ValueError
			else:
				self.volume_decoder.add_gradients(vel_net_gradients)
			
			if self._has_input_encoder:
				enc_net_grads = gradients["input_encoder"]
				encoders = self.active_input_encoders
				for enc, grads in zip(encoders, self.__split_gradients_for_decoders(enc_net_grads, encoders)):
					enc.add_gradients(grads)
			
			if self._has_downscale_encoder:
				enc_net_grads = gradients["downscale_encoder"]
				encoders = self.active_downscale_encoders
				for enc, grads in zip(encoders, self.__split_gradients_for_decoders(enc_net_grads, encoders)):
					enc.add_gradients(grads)
			
			if self.requires_parent_state_variables:
				state_gradients = {k: gradients[k] for k in self.parent_state.get_variables()}
				#LOG.info("state grads from vel: %s"%([[tf.reduce_mean(_).numpy(), tf.reduce_max(tf.abs(_)).numpy()] if _ is not None else "None" for k,v in state_gradients.items() for _ in v]))
				self.parent_state.accumulate_gradients(state_gradients)
	
	@property
	def has_pending_gradients(self):
		if self.has_multiple_decoders:
			return any(_.has_pending_gradients for _ in self.volume_decoder)
		else:
			return self.volume_decoder.has_pending_gradients
	
	def apply_gradients(self, optimizer, keep_gradients=False):
		raise NotImplementedError
		LOG.debug("Applying velocity gradients of frame %d", self.parent_state.frame)
		if self.has_multiple_decoders:
			if not self.has_pending_gradients: raise RuntimeError("No decoder has any recorded gradients to apply."%(self.name,))
			for dec in self.volume_decoder:
				if dec.has_pending_gradients:
					dec.apply_gradients(optimizer)
					if not keep_gradients:
						dec.clear_gradients()
		else:
			self.volume_decoder.apply_gradients(optimizer)
			if not keep_gradients:
				self.volume_decoder.clear_gradients()
	
	def get_grads_vars(self, keep_gradients=True, normalize=False):
		grads_vars = []
		if self.has_multiple_decoders:
			for dec in self.volume_decoder:
				if dec.has_pending_gradients:
					grads_vars.extend(dec.get_grads_vars(keep_gradients=keep_gradients, normalize=normalize))
					if not keep_gradients:
						dec.clear_gradients()
		else:
			grads_vars.extend(self.volume_decoder.get_grads_vars(keep_gradients=keep_gradients, normalize=normalize))
			if not keep_gradients:
				self.volume_decoder.clear_gradients()
		
		if self._has_input_encoder:
			for enc in self.active_input_encoders:
				grads_vars.extend(enc.get_grads_vars(keep_gradients=keep_gradients, normalize=normalize))
				if not keep_gradients:
					enc.clear_gradients()
		
		if self._has_downscale_encoder:
			for enc in self.active_downscale_encoders:
				grads_vars.extend(enc.get_grads_vars(keep_gradients=keep_gradients, normalize=normalize))
				if not keep_gradients:
					enc.clear_gradients()
		
		return grads_vars
	
	def clear_gradients(self):
		if self.has_multiple_decoders:
			for dec in self.volume_decoder:
				dec.clear_gradients()
		else:
			self.volume_decoder.clear_gradients()
		
		if self._has_input_encoder:
			for enc in self.active_input_encoders:
				enc.clear_gradients()
		
		if self._has_downscale_encoder:
			for enc in self.active_downscale_encoders:
				enc.clear_gradients()
	
	
	def save(self, path):
		if self.is_staggered:
			np.savez_compressed(path, centered_shape=self.centered_shape, vel_x=self.x.numpy(), vel_y=self.y.numpy(), vel_z=self.z.numpy())
		elif self.is_centered:
			np.savez_compressed(path, centered_shape=self.centered_shape, vel=self.centered().numpy())

def _tf_tensors_equal(a,b):
	assert isinstance(a, (tf.Tensor, tf.Variable)), "_tf_tensors_equal input 0 is not a tf.Tensor"
	assert isinstance(b, (tf.Tensor, tf.Variable)), "_tf_tensors_equal input 1 is not a tf.Tensor"
	return (shape_list(a)==shape_list(b)) and tf.reduce_all(tf.equal(a,b)).numpy().tolist()

class NeuralState(State, BackpropInterface):
	def __init__(self, density, velocity, target_encoder, encoder_output_types, target_lifting, lifting_renderer, target_merging, volume_encoder, frame, prev=None, next=None, transform=None, targets=None, targets_raw=None, bkgs=None, lifting_network=None, frame_merge_network=None):
		super().__init__(density=density, velocity=velocity, frame=frame, prev=prev, next=next, transform=transform, targets=targets, targets_raw=targets_raw, bkgs=bkgs)
		
		self.__feature_cache = ResourceCacheDictTF(self._device)
		self.__input_grad_cache = ResourceCacheDictTF(device=self._device)
		assert isinstance(target_lifting, str)
		self.target_lifting = target_lifting
		
		assert is_None_or_type(target_encoder, DeferredBackpropNetwork)
		self.target_encoder = target_encoder
		if self.target_lifting.upper()=="UNPROJECT":
			assert is_None_or_type(volume_encoder, DeferredBackpropNetwork)
			self.volume_encoder = volume_encoder
		else:
			self.volume_encoder = None
		self.encoder_output_types = encoder_output_types
		if self.target_lifting.upper()=="NETWORK":
			assert is_None_or_type(lifting_network, DeferredBackpropNetwork)
			self.lifting_network = lifting_network
		else:
			self.lifting_network = None
		assert is_None_or_type(frame_merge_network, DeferredBackpropNetwork)
		self.frame_merge_network = frame_merge_network
		
		self.lifting_renderer = lifting_renderer
		assert isinstance(target_merging, str)
		self.target_merging = target_merging
		self.clear_cache()
		self.disable_cache = False #True
		self.cache_warn_level = 1 #0: no warning, 1: warning, 2: error
		self.input_view_mask = None
		
		self.__warned_shape = False
		
		# self.__output_users = []
		# self.__output_gradient_cache = None
		# self.__inputs_for_backprop = []
	
	# backprop interface
	def __add_used_input(self, other, output_id):
		self.__inputs_for_backprop.append(input_key(other, output_id))
	
	def __register_inputs(self):
		for inp, out_id in self.__inputs_for_backprop:
			assert isinstance(inp, BackpropInterface)
			inp._register_output_user(self, out_id)
	
	def __gather_inputs(self):
		return [inp.output(out_id) for inp, out_id in self.__inputs_for_backprop]
	
	def outputs(self):
		outputs = {"OUTPUT": self.targets_feature_volume()}
		for cache_name, value in self.__feature_cache.items():
			if cache_name.startswith("enc_lift_volume_"):
				shape = cache_name[16:]
				outputs["LIFTING_"+shape] = value
		
		for key in outputs.keys():
			if key not in self.__output_users:
				self.__output_users[key] = []
		
		return outputs
		
	
	def output(self, output_id):
		outputs = self.outputs()
		if output_id not in outputs:
			raise KeyError("%s not in outputs. Available outputs: %s"%(output_id, tuple(outputs.keys())))
		return outputs[output_id]
	
	def _register_output_user(self, other, output_id):
		assert isinstance(other, BackpropInterface)
		if other not in self.__output_users[output_id]:
			self.__output_users[output_id].append(other)
	
	def _compute_input_grads(self):
		if self.__input_grad_cache.is_empty():
			input_grads = self.__backprop()
			for i, grad in enumerate(input_grads):
				self.__input_grad_cache["input_grads_%d"%(i,)] = grad
	
	def _get_input_grad(self, other, output_id):
		# get gradients for a specific input
		# check if it is one of the inputs
		assert other in self.__inputs_for_backprop
		t = input_key(other, output_id)
		idx = self.__inputs_for_backprop.index(t)
		#assert not self.__input_grad_cache.is_empty()
		
		self._compute_input_grads()
		return self.__input_grad_cache["input_grads_%d"%(idx,)]
	
	def has_gradients_for(self, other, output_id):
		return (input_key(other, output_id) in self.__inputs_for_backprop) and (not self.__input_grad_cache.is_empty() or self.can_backprop)
	
	@property
	def can_backprop(self):
		# has output gradients available
		#LOG.info("'%s' can backprop: out grad %s; output users %d, provides grads %s", self._name, self.__output_gradient_cache is not None, len(self.__output_users), [_.has_gradients_for(self) for _ in self.__output_users])
		return (self.__output_gradient_cache is not None) or (len(self.__output_users)>0 and any(len(users)>0 and any(_.has_gradients_for(self, out_id) for _ in users) for out_id, users in self.__output_users.items()))
	
	@property
	def requires_backprop(self):
		return (not self.is_all_frame_encoders_frozen) and self.__input_grad_cache.is_empty() and self.can_backprop
	
	# own methods
	
	@property
	def _device(self):
		return self.density._device
	
	@property
	def has_density_neural(self):
		return self.has_density and type(self.density)==NeuralDensityGrid
	
	@property
	def inputs_raw(self):
		return self.base_targets_raw.get_images_of_views(self.input_view_mask)
	@property
	def inputs(self):
		return self.base_targets.get_images_of_views(self.input_view_mask)
	@property
	def input_bkgs(self):
		return self.base_bkgs.get_images_of_views(self.input_view_mask)
	@property
	def input_masks(self):
		return self.base_masks.get_images_of_views(self.input_view_mask)
	@property
	def input_cameras(self):
		if self.input_view_mask is not None:
			return [self.base_target_cameras[_] for _ in self.input_view_mask]
		else:
			return copy.copy(self.base_target_cameras)
	
	def _get_batch_size(self):
		return self.base_targets_raw.batch_size
	
	def _make_3D_cache_name(self, base_name, *, is_raw=False, is_binary=False, shape=None):
		assert shape is None or (isinstance(shape, (list, tuple, Vector)) and len(shape) in [2,3])
		if shape is None: shape = self.transform.grid_size
		return "{}{}{}_{}".format(base_name, "_raw" if is_raw else "", "_bin" if is_binary else "", "-".join(str(_) for _ in shape))
	
	@property
	def targets_raw_feature_volume(self):
		if self.disable_cache or "target_raw_feature_volume" not in self.__feature_cache: #self.__target_raw_feature_volume is None:
			# tmp = self._generate_volume_encoding(raw_targets=True)
			# with tf.device(self._device):
				# self.__target_raw_feature_volume = tmp
			self.__feature_cache["target_raw_feature_volume"] = self._generate_volume_encoding(raw_targets=True)
		else: self.__check_input_cache(raw=True)
		
		return self.__feature_cache["target_raw_feature_volume"] #self.__target_raw_feature_volume
	#@property
	def targets_feature_volume(self, grid_shape=None):
		cache_name = self._make_3D_cache_name("target_feature_volume", shape=grid_shape)
		if self.disable_cache or cache_name not in self.__feature_cache: #self.__target_feature_volume is None:
			# tmp = self._generate_volume_encoding(raw_targets=False)
			# with tf.device(self._device):
				# self.__target_feature_volume = tmp
			self.__feature_cache[cache_name] = self._generate_volume_encoding(raw_targets=False, grid_shape=grid_shape)
		else: self.__check_input_cache(raw=False)
		
		return self.__feature_cache[cache_name]
	
	@property
	def targets_raw_feature_images(self):
		if self.disable_cache or "target_raw_feature_images" not in self.__feature_cache: #self.__target_raw_feature_images is None:
			# self.__target_raw_feature_images = self._generate_image_encoding(raw_targets=True)
			# self.__target_raw_inputs = self.inputs_raw
			self.__feature_cache["target_raw_feature_images"] = self._generate_image_encoding(raw_targets=True)
			self.__feature_cache["target_raw_inputs"] = self.inputs_raw
		elif self.cache_warn_level>0 and "target_raw_inputs" in self.__feature_cache: #self.__target_raw_inputs is not None:
			with SAMPLE("check_input_eq"):
				#if not _tf_tensors_equal(self.__target_raw_inputs, self.inputs_raw): self.__cache_warning("NeuralState (%d): image encoding has not been generated or cleared for current raw targets.", self.frame)
				if not _tf_tensors_equal(self.__feature_cache["target_raw_inputs"], self.inputs_raw): 
					self.__cache_warning("NeuralState (%d): image encoding has not been generated or cleared for current raw targets.", self.frame)
		return self.__feature_cache["target_raw_feature_images"]
	@property
	def targets_feature_images(self):
		if self.disable_cache or "target_feature_images" not in self.__feature_cache: #self.__target_feature_images is None:
			# self.__target_feature_images = self._generate_image_encoding(raw_targets=False)
			# self.__target_inputs = self.inputs
			self.__feature_cache["target_feature_images"] = self._generate_image_encoding(raw_targets=False)
			self.__feature_cache["target_inputs"] = self.inputs
		elif self.cache_warn_level>0 and "target_inputs" in self.__feature_cache: #self.__target_inputs is not None:
			with SAMPLE("check_input_eq"):
				#if not _tf_tensors_equal(self.__target_inputs, self.inputs): self.__cache_warning("NeuralState (%d): image encoding has not been generated or cleared for current targets.", self.frame)
				if not _tf_tensors_equal(self.__feature_cache["target_inputs"], self.inputs):
					self.__cache_warning("NeuralState (%d): image encoding has not been generated or cleared for current targets.", self.frame)
		return self.__feature_cache["target_feature_images"] #elf.__target_feature_images
	
	def __check_input_cache(self, raw):
		assert isinstance(raw, bool)
		#cache = (self.__target_raw_inputs if raw else self.__target_inputs)
		cache = self.__feature_cache["target_raw_inputs" if raw else "target_inputs"]
		if self.cache_warn_level>0 and cache is not None:
			with SAMPLE("check_input_cache"):
				if not isinstance(cache, tf.Tensor): raise TypeError("NeuralState cache is not a tf.Tensor. is: {}".format(cache.__class__.__name__))
				inputs = (self.inputs_raw if raw else self.inputs)
				if not isinstance(inputs, tf.Tensor): raise TypeError("NeuralState input is not a tf.Tensor. is: {}".format(cache.__class__.__name__))
				inputs_shape = shape_list(inputs)
				cache_shape = shape_list(cache)
				if not len(inputs_shape)==len(cache_shape) or not all(a==b for a,b in zip(inputs_shape, cache_shape)):
					self.__cache_warning("NeuralState (%d): targets and cache do not have the same shape: %s - %s. \n\ttargets: %f, %f, %f\n\tcache: %f, %f, %f", self.frame, inputs_shape, cache_shape, \
						tf.reduce_min(inputs).numpy(), tf.reduce_max(inputs).numpy(), tf.reduce_mean(inputs).numpy(), \
						tf.reduce_min(cache).numpy(), tf.reduce_max(cache).numpy(), tf.reduce_mean(cache).numpy())
				elif not tf.reduce_all(tf.equal(inputs,cache)).numpy().tolist():
					self.__cache_warning("NeuralState (%d): targets and cache do not have the same value (or contain NaN): \n\ttargets: %f, %f, %f\n\tcache: %f, %f, %f", self.frame, \
						tf.reduce_min(inputs).numpy(), tf.reduce_max(inputs).numpy(), tf.reduce_mean(inputs).numpy(), \
						tf.reduce_min(cache).numpy(), tf.reduce_max(cache).numpy(), tf.reduce_mean(cache).numpy())
	
	def __cache_warning(self, *msg):
		if self.cache_warn_level == 1:
			LOG.warning(*msg)
		elif self.cache_warn_level == 2:
			LOG.error(*msg)
			raise ValueError("NeuralState cache consistency error.")
	
	def _image_luminance(self, images):
		assert has_rank(images, 5)
		shape = GridShape.from_tensor(images)
		if shape.c==1: return images #L
		elif shape.c==2: return images[...,:1] #LA
		elif shape.c==3: return tf.reduce_sum(images*self.lifting_renderer.luma, axis=-1, keepdims=True) #RGB
		elif shape.c==4: return tf.reduce_sum(images[...,:3]*self.lifting_renderer.luma, axis=-1, keepdims=True) #RGBA
		
	
	def _encode_images(self, images):
		with SAMPLE("encode Images"):
			assert has_rank(images, 5)
			shape = GridShape.from_tensor(images)
			#images = grad_log(images, "_encode_images start", LOG.info)
			encoder_output = []
			
			if "NETWORK" in self.encoder_output_types:
					# make sure the image resolution fits the encoders strides, might get issues from distortions otherwise
					# inp_div = 1
					# if isinstance(self.target_encoder, GrowingUNet):
						# inp_div = self.target_encoder.get_scale()
					# images = tf_pad_to_next_div_by(images, inp_div, pad_axes=(-3,-2))
					
					# DEBUG
					#assert self.target_encoder.current_level==0
					
					shape = GridShape.from_tensor(images)
					
					conv_shape = [shape.n*shape.z,shape.y,shape.x,shape.c]
					enc = self.target_encoder(tf.reshape(images, conv_shape))
					
					enc_shape = GridShape.from_tensor(enc)
					enc_shape.n = shape.n # batch
					enc_shape.z = shape.z # views
					enc = tf.reshape(enc, enc_shape.as_shape)
					
					#images = grad_log(images, "_encode_images end", LOG.info)
					if not self.__warned_shape and (shape.x!=enc_shape.x or shape.y!=enc_shape.y): #check xy (spatial) match
						LOG.warning("Image shape %s does not match encoded shape %s, lifting might be distorted", shape, enc_shape)
						self.__warned_shape = True
					
					encoder_output.append(enc)
				
			if "L" in self.encoder_output_types:
				if shape.c==1: encoder_output.append(images) #L
				elif shape.c==2: encoder_output.append(images[...,:1]) #LA
				elif shape.c==3: encoder_output.append(tf.reduce_sum(images*self.lifting_renderer.luma, axis=-1, keepdims=True)) #RGB
				elif shape.c==4: encoder_output.append(tf.reduce_sum(images[...,:3]*self.lifting_renderer.luma, axis=-1, keepdims=True)) #RGBA
			
			if "IDENTITY" in self.encoder_output_types:
				encoder_output.append(images)
			
			return tf.concat(encoder_output, axis=-1)
	
	def _unproject_2D(self, images_encoding, accumulate="SUM", binary=False, binary_eps_2d=1e-5, binary_eps_3d=0.5, grid_shape=None):
		"""
		Args:
			images_encoding tf.Tensor: shape: NVHWC
		returns:
			tf.Tensor: features lifted to 3D, NDHWC if sum_views, VNDHWC else
		"""
		assert has_rank(images_encoding, 5)
		assert isinstance(accumulate, str)
		accumulate = accumulate.upper()
		assert accumulate in ["SUM", "MEAN", "MIN", "MAX", "NONE", "CHANNELS"]
		shape = GridShape.from_tensor(images_encoding) #NVHWC
		
		if binary:
			images_encoding = tf.cast(tf.greater_equal(images_encoding, binary_eps_2d), dtype=images_encoding.dtype)
		
		if accumulate not in ["SUM", "MEAN"]:
			images_encodings = tf.split(images_encoding, shape.z, axis=1) #V-N1WHC
			shape.z = 1
			cams = [[cam] for cam in self.input_cameras]
		else:
			if not self.lifting_renderer.blend_mode=="ADDITIVE":
				raise ValueError("SUM and MEAN unprojection merging requires ADDITIVE blend mode used in target lifting")
			images_encodings = [images_encoding] #1-NVWHC
			cams = [self.input_cameras]
		
		enc_out = []
		
		transformation = self.transform.copy_no_data()
		if grid_shape is not None:
			transformation.grid_size = grid_shape
		
		for images_encoding, cameras in zip(images_encodings, cams): # for each view
			with SAMPLE("unproject"):
				# image shape for unprojection: NVHWC with C in [1,2,4]
				if shape.c not in [1,2,4]:
					channel_div = shape.c//4 if shape.c%4==0 else shape.c//2 if shape.c%2==0 else shape.c
					images_encoding = tf.reshape( \
										tf.transpose( \
											tf.reshape(images_encoding, (shape.n, shape.z, shape.y, shape.x, channel_div, shape.c//channel_div)), \
											(0,4,1,2,3,5)), \
										(shape.n * channel_div, shape.z, shape.y, shape.x, shape.c//channel_div) \
									)
					#raise NotImplementedError("can only unproject with channels 1,2 or 4. is %d"%shape.c)
					# roll channels into batch?
				# camera XY resolution does not matter, raymarch_camera uses the input image resolution
				
				volume_encoding = self.lifting_renderer.raymarch_camera(data=images_encoding, cameras=cameras, transformations=transformation, inverse=True, squeeze_batch=False)
				vol_shape = GridShape.from_tensor(volume_encoding)
				if shape.c not in [1,2,4]:
					volume_encoding = tf.reshape( \
										tf.transpose( \
											tf.reshape(volume_encoding, (shape.n, channel_div, vol_shape.z, vol_shape.y, vol_shape.x, shape.c//channel_div)), \
											(0,2,3,4,1,5)), \
										(shape.n, vol_shape.z, vol_shape.y, vol_shape.x, shape.c) \
									)
				enc_out.append(volume_encoding)
		# enc_out now: V-NDHWC
		
		
		if accumulate=="SUM":
			ret = enc_out[0]
		elif accumulate=="MEAN":
			ret = enc_out[0] * (1.0/float(shape.z))
		elif accumulate=="MIN":
			if binary:
				enc_out = tf.cast(tf.greater_equal(enc_out, binary_eps_3d), dtype=images_encoding.dtype)
			ret = tf.reduce_min(enc_out, axis=0)
		elif accumulate=="MAX":
			if binary:
				enc_out = tf.cast(tf.greater_equal(enc_out, binary_eps_3d), dtype=images_encoding.dtype)
			ret = tf.reduce_max(enc_out, axis=0)
		elif accumulate=="CHANNELS":
			ret = tf.concat(enc_out, axis=-1)
			ret_shape = GridShape.from_tensor(ret)
			assert ret_shape.c == shape.c*len(cams)
			assert ret_shape.n == shape.n
			#LOG.info("DEBUG: #target cams: %d, target mask: %s, input mask: %s."%(len(self.base_target_cameras), self.target_mask, self.input_view_mask))
			#assert ret_shape.c == 78, "debug: %s, %d"%(str(ret_shape), len(cams))
		else:
			ret = enc_out
		
		#ret_shape = GridShape.from_tensor(ret)
		#assert ret_shape.n==shape.n, "batch size mismatch after unprojection"
		#assert ret_shape.c==shape.c, "channel size mismatch after unprojection"
		
		return ret
	
	def _lift_merge_encoding(self, images_encoding, grid_shape=None):
		"""
		Args:
			images_encoding tf.Tensor: shape: NVHWC
		returns:
			tf.Tensor: features lifted to 3D, NDHWC
		"""
		assert has_rank(images_encoding, 5)
		shape = GridShape.from_tensor(images_encoding) #NVHWC
		#images_encoding = grad_log(images_encoding, "_lift_merge_encoding start", LOG.info)
		
		with SAMPLE("lift encoding"):
			if self.target_lifting=="UNPROJECT":
				
				if self.target_merging in ["SUM","SUM_NETWORK","MEAN","MEAN_NETWORK"]:
					if not self.lifting_renderer.blend_mode=="ADDITIVE":
						raise ValueError("SUM and MEAN target merging requires ADDITIVE blend mode used in target lifting")
					volume_encoding = self._unproject_2D(images_encoding,accumulate="SUM", grid_shape=grid_shape)
					
					if self.target_merging in ["MEAN","MEAN_NETWORK"]:
						volume_encoding *= (1./float(shape.z)) #normalize with number of views
					
					if self.target_merging in ["SUM_NETWORK","MEAN_NETWORK"]:
						volume_encoding = self.volume_encoder(volume_encoding)
				
				elif self.target_merging in ["CONCAT", "CONCAT_NETWORK"]:
					# concat unprojections
					volume_encoding = self._unproject_2D(images_encoding,accumulate="NONE", grid_shape=grid_shape)
					with SAMPLE("Merge views"):
						volume_encoding = tf.concat(volume_encoding, axis=-1)
					if self.target_merging=="CONCAT_NETWORK":
						with SAMPLE("Encode volume"):
							volume_encoding = self.volume_encoder(volume_encoding)
				elif self.target_merging=="NETWORK_CONCAT":
					volume_encoding = self._unproject_2D(images_encoding,accumulate="NONE", grid_shape=grid_shape)
					encodings = []
					for v in volume_encoding:
						with SAMPLE("Encode volume"):
							encodings.append(self.volume_encoder(v))
							#LOG.info("Volume encoder level: %d, shape %s", self.volume_encoder.get_active_level(), shape_list(encodings[-1]))
					with SAMPLE("Merge views"):
						volume_encoding = tf.concat(encodings, axis=-1)
				elif self.target_merging=="NETWORK_SUMPROD":
					# individually unproject and encode, then sum some channels and multiply others.
					volume_encoding = self._unproject_2D(images_encoding,accumulate="NONE", grid_shape=grid_shape)
					encodings = []
					for v in volume_encoding:
						with SAMPLE("Encode volume"):
							encodings.append(self.volume_encoder(v))
					
					with SAMPLE("Merge views"):
						#encoding_shape = shape_list(encodings[0])
						channels = self.volume_encoder.output_channels #encoding_shape[-1]
						add_channels = channels//2
						#mul_channels = channels - add_channels
						
						add_encoding = tf.reduce_sum([_[...,:add_channels] for _ in encodings], axis=0)
						mul_encoding = tf.reduce_prod([tf.tanh(_[...,add_channels:]) for _ in encodings], axis=0)
						
						volume_encoding = tf.concat([add_encoding, mul_encoding], axis=-1)
				else:
					raise ValueError("Unknown merging mode '%s'"%(self.target_merging,))
			elif self.target_lifting=="NETWORK":
				assert shape.z==1, "lifting network only support single view"
				# To supprt multi-view the network needs to support multi-view unprojection (and view merging)
				images_encoding = tf.reshape(images_encoding, (shape.n*shape.z,shape.y,shape.x,shape.c))
				volume_encoding = self.lifting_network(images_encoding)
				if self.lifting_network._enc_output:
					volume_encoding, lift_encoding = volume_encoding
					for level, volume in enumerate(lift_encoding):
						cache_name = self._make_3D_cache_name("enc_lift_volume", shape=shape_list(volume)[1:-1])
						self.__feature_cache[cache_name] = volume
			else:
				raise ValueError("Unknown lifting method '%s'"%(self.target_lifting,))
		
		if self.frame_merge_network is not None:
			with SAMPLE("merge next frame"):
				if self.next is not None:
					assert isinstance(self.next, NeuralState)
					#self.__inputs_for_backprop.append(self.next)
					next_volume_encoding = self.next.output("OUTPUT")
					self.__add_used_input(self.next, "OUTPUT")
				else:
					next_volume_encoding = tf.zeros_like(volume_encoding)
				#DEBUG
				#LOG.warning("DEBUG: merge network next frame input is zeroed!")
				#next_volume_encoding = tf.zeros_like(next_volume_encoding)
				volume_encoding = self.frame_merge_network(tf.concat([volume_encoding, next_volume_encoding], axis=-1))
		
		self.__register_inputs()
		
		#volume_encoding = grad_log(volume_encoding, "_lift_merge_encoding end", LOG.info)
		return volume_encoding
		#return self.target_lifting(images_encoding)
	
	#def _merge_3D_encoding(self, lifted_encoding):
	#	if self.target_merging=="SUM":
	#		return lifted_encoding #tf.reduce_sum(lifted_encoding, axis=1) sum already done by batched unprojection
	#	else:
	#		raise ValueError("Unknown lifting method '%s'"%self.lifting)
	
	def _generate_image_encoding(self, raw_targets=False):
		if raw_targets: raise ValueError("DEBUGGING")
		images = self.inputs_raw if raw_targets else self.inputs
		return self._encode_images(images)
	
	def _generate_volume_encoding(self, raw_targets=False, grid_shape=None):
		if raw_targets: raise ValueError("DEBUGGING")
		image_encoding = self.targets_raw_feature_images if raw_targets else self.targets_feature_images
		
		# multi-frame input
		# 1. simple image concatenation
		# 2. per-frame lifting, 3D concatenation
		# - recursive merge instead of concat
		# - alignment/warping before concat/merge
		
		return self._lift_merge_encoding(image_encoding, grid_shape=grid_shape)
	
	def _get_target_unprojection(self, is_raw, grid_shape=None):
		#return self.targets_raw_feature_volume if is_raw else self.targets_feature_volume
		#cache_name = "target_raw_unprojection" if is_raw else "target_unprojection"
		#cache_name += "_"+str(grid_shape)
		cache_name = self._make_3D_cache_name("target_unprojection", is_raw=is_raw, shape=grid_shape)
		images = self.targets_raw if is_raw else self.targets
		if self.disable_cache or cache_name not in self.__feature_cache:
			self.__feature_cache[cache_name] = self._unproject_2D(images, accumulate="MEAN", grid_shape=grid_shape)
		return self.__feature_cache[cache_name]
	
	def _get_input_images_unprojection(self, is_raw, grid_shape=None):
		#cache_name = "input_images_raw_unprojection" if is_raw else "input_images_unprojection"
		#cache_name += "_"+str(grid_shape)
		cache_name = self._make_3D_cache_name("input_images_unprojection", is_raw=is_raw, shape=grid_shape)
		images = self.inputs_raw if is_raw else self.inputs
		if self.disable_cache or cache_name not in self.__feature_cache:
			self.__feature_cache[cache_name] = self._unproject_2D(images, accumulate="MEAN", grid_shape=grid_shape) #MEAN
		return self.__feature_cache[cache_name]
	
	def _get_input_images_unprojection_concat(self, is_raw, grid_shape=None):
		#cache_name = "input_images_raw_unprojection" if is_raw else "input_images_unprojection"
		#cache_name += "_"+str(grid_shape)
		cache_name = self._make_3D_cache_name("input_images_unprojection_concat", is_raw=is_raw, shape=grid_shape)
		images = self.inputs_raw if is_raw else self.inputs
		if self.disable_cache or cache_name not in self.__feature_cache:
			self.__feature_cache[cache_name] = self._unproject_2D(images, accumulate="CHANNELS", grid_shape=grid_shape) #MEAN
		return self.__feature_cache[cache_name]
	
	def _get_target_hull(self, is_raw, binary=False, grid_shape=None):
		# data = self.targets_raw_feature_images if is_raw else self.targets_feature_images
		# return self._unproject_2D(data, accumulate="MIN") # NDHWC
		cache_name = self._make_3D_cache_name("target_hull_volume", is_raw=is_raw, is_binary=binary, shape=grid_shape)
		if self.disable_cache or cache_name not in self.__feature_cache:
			if binary:
				if not self.has_masks:
					raise RuntimeError("No masks available to create binary target hull.")
				self.__feature_cache[cache_name] = self._unproject_2D(self.masks, accumulate="MIN", binary=True, binary_eps_2d=1e-5, binary_eps_3d=0.5, grid_shape=grid_shape)
			else:
				data = self.targets_raw_feature_images if is_raw else self.targets_feature_images
				self.__feature_cache[cache_name] = self._unproject_2D(data, accumulate="MIN", grid_shape=grid_shape)
		data = self.__feature_cache[cache_name]
		return data
	
	def _get_input_images_hull(self, is_raw, binary=False, grid_shape=None):
		# data = self.targets_raw_feature_images if is_raw else self.targets_feature_images
		# return self._unproject_2D(data, accumulate="MIN") # NDHWC
		cache_name = self._make_3D_cache_name("input_images_hull_volume", is_raw=is_raw, is_binary=binary, shape=grid_shape)
		if self.disable_cache or cache_name not in self.__feature_cache:
			if binary:
				if not self.has_masks:
					raise RuntimeError("No masks available to create binary target hull.")
				self.__feature_cache[cache_name] = self._unproject_2D(self.input_masks, accumulate="MIN", binary=True, binary_eps_2d=1e-5, binary_eps_3d=0.5, grid_shape=grid_shape)
			else:
				data = self.input_masks #self._image_luminance(self.inputs if is_raw else self.inputs_raw)
				self.__feature_cache[cache_name] = self._unproject_2D(data, accumulate="MIN", grid_shape=grid_shape)
		data = self.__feature_cache[cache_name]
		return data
	
	def get_volume_features(self, types=[], shape=None, concat=True):
		#assert "L" in self.encoder_output_types and len(self.encoder_output_types)==1, "enc types: {}".format(self.encoder_output_types) #currently using the target encoding pipeline for testing, so make sure it provides only images
		data = []
		
		if "INPUT_IMAGES_UNPROJECTION" in types:
			data.append(self._get_input_images_unprojection(is_raw=False, grid_shape=shape))
		if "INPUT_IMAGES_UNPROJECTION_CONCAT" in types:
			data.append(self._get_input_images_unprojection_concat(is_raw=False, grid_shape=shape))
		if "INPUT_IMAGES_RAW_UNPROJECTION" in types:
			data.append(self._get_input_images_unprojection(is_raw=True, grid_shape=shape))
		if "INPUT_IMAGES_HULL" in types:
			#if shape is not None: raise NotImplementedError
			data.append(self._get_input_images_hull(is_raw=False, grid_shape=shape))
		
		if "TARGET_UNPROJECTION" in types:
			data.append(self._get_target_unprojection(is_raw=False, grid_shape=shape))
		if "TARGET_RAW_UNPROJECTION" in types:
			data.append(self._get_target_unprojection(is_raw=True, grid_shape=shape))
		if "TARGET_HULL" in types:
			data.append(self._get_target_hull(is_raw=False, grid_shape=shape))
		if "TARGET_HULL_BINARY" in types:
			data.append(self._get_target_hull(is_raw=False, binary=True, grid_shape=shape))
		if "TARGET_RAW_HULL" in types:
			data.append(self._get_target_hull(is_raw=True, grid_shape=shape))
		
		if "ENC2D_UNPROJECTION" in types:
			raise NotImplementedError
			assert self.has_2D_encoder
			#data.append(self.targets_feature_volume) #targets_raw_feature_volume
		# if "ENC3D" in types:
			# raise NotImplementedError("use state.output()")
			# #if shape is not None: raise NotImplementedError
			# #assert self.has_3D_encoder
			# data.append(self.targets_feature_volume(grid_shape=shape))
		
		# if len(data)==0: raise ValueError("No input from types: {}".format(types))
		
		return tf.concat(data, axis=-1) if concat else data
	
	@staticmethod
	def get_base_feature_channels(types=[], *, image_encoder_channels=None, volume_encoder_channels=None, color_channels=3, num_views=None):
		channels = 0
		
		if "INPUT_IMAGES_UNPROJECTION" in types:
			channels += color_channels #1
		if "INPUT_IMAGES_UNPROJECTION_CONCAT" in types:
			channels += color_channels * num_views #1
		if "INPUT_IMAGES_RAW_UNPROJECTION" in types:
			channels += color_channels #1
		if "INPUT_IMAGES_HULL" in types:
			channels +=1
		
		if "TARGET_UNPROJECTION" in types:
			channels += color_channels #1
		if "TARGET_RAW_UNPROJECTION" in types:
			channels += color_channels #1
		if "TARGET_HULL" in types:
			channels +=1
		if "TARGET_RAW_HULL" in types:
			channels +=1
		
		if "ENC2D_UNPROJECTION" in types:
			channels += image_encoder_channels
		if "ENC3D" in types:
			channels += volume_encoder_channels
		
		#if channels==0: raise ValueError("No input from types: {}".format(types))
		
		return channels
		
	
	def get_variables(self):
		var_dict = {}
		if isinstance(self.target_encoder, (tf.keras.models.Model, DeferredBackpropNetwork)):
			#LOG.debug("add encoder to NeuralState variables")
			var_dict["target_encoder"] = self.target_encoder.trainable_variables
		if isinstance(self.lifting_network, (tf.keras.models.Model, DeferredBackpropNetwork)):
			var_dict["lifting_network"] = self.lifting_network.trainable_variables
		if isinstance(self.volume_encoder, (tf.keras.models.Model, DeferredBackpropNetwork)):
			#LOG.debug("add encoder to NeuralState variables")
			var_dict["volume_encoder"] = self.volume_encoder.trainable_variables
		if isinstance(self.frame_merge_network, (tf.keras.models.Model, DeferredBackpropNetwork)):
			var_dict["frame_merge_network"] = self.frame_merge_network.trainable_variables
		
		return var_dict
	
	# def get_output_variables(self):
		# pass
	
	def clear_cache(self, volume_only=False):
		if not volume_only:
			self._clear_image_cache()
		self._clear_volume_cache()
		
		self.__output_users = {"OUTPUT": []}
		self.__output_gradient_cache = None
		self.__inputs_for_backprop = []
		self.__input_grad_cache.clear()
		
		super().clear_cache() #density, velocity and rendered images
	
	def _clear_image_cache(self):
		self.__target_raw_inputs = None
		self.__target_inputs = None
		self.__target_raw_feature_images = None
		self.__target_feature_images = None
		
	def _clear_volume_cache(self):
		self.__target_raw_feature_volume = None
		self.__target_feature_volume = None
		self.__feature_cache.clear()
		
	
	def clear_cache_for_backprop(self):
		if "NETWORK" in self.encoder_output_types:
			LOG.debug("State.clear_cache_for_backprop: clearing full cache.")
			self._clear_image_cache()
			self._clear_volume_cache()
		elif self.target_lifting!="UNPROJECT" or ("NETWORK" in self.target_merging):
			LOG.debug("State.clear_cache_for_backprop: clearing volume cache.")
			self._clear_volume_cache()
	
	def __gather_output_grads(self):
		
		if self.prev is None and self.next is None:
			check_output_users(self.__output_users, {"OUTPUT": [NeuralDensityGrid]}, "single frame NeuralState")
		elif self.prev is None and self.next is not None:
			check_output_users(self.__output_users, {"OUTPUT": [NeuralDensityGrid, NeuralVelocityGrid]}, "first frame NeuralState")
		elif self.prev is not None and self.next is not None:
			check_output_users(self.__output_users, {"OUTPUT": [NeuralState, NeuralVelocityGrid, NeuralVelocityGrid]}, "mid frame NeuralState")
		elif self.prev is not None and self.next is None:
			check_output_users(self.__output_users, {"OUTPUT": [NeuralState, NeuralVelocityGrid]}, "last frame NeuralState")
		
		for out_id, users in self.__output_users.items():
			for other in users:
				other._compute_input_grads()
		grads = {}
		for out_id, users in self.__output_users.items():
			grads[out_id] = tf.zeros_like(self.output(out_id))
			for other in users:
				grads[out_id] = grads[out_id] + other._get_input_grad(self, out_id)
		
		return grads
	
	def __backprop(self):
		with SAMPLE("state backprop"):
			output_gradients = self.__gather_output_grads()
			LOG.debug("Backprop state frame %d", self.frame)
			
			var_dict = self.get_variables()
			var_dict["inputs_for_backprop"] = self.__gather_inputs()
			
			self.clear_cache_for_backprop()
			
			with SAMPLE("forward"), tf.GradientTape(watch_accessed_variables=False) as tape:
				tape.watch(var_dict)
				output = self.outputs()
			
			
			with SAMPLE("gradients"):
				gradients = tape.gradient(output, var_dict, output_gradients=output_gradients)
			
			#if self.prev is not None:
			self.accumulate_gradients(gradients)
			#else:
			#	warnings.warn("no gradients for State of first frame.")
		
		return gradients["inputs_for_backprop"]
	
	def accumulate_gradients(self, grad_dict):
		#gradients dict with keys from get_variables
		if 'target_encoder' in grad_dict:
			self.target_encoder.add_gradients(grad_dict["target_encoder"])
		if 'lifting_network' in grad_dict:
			self.lifting_network.add_gradients(grad_dict["lifting_network"])
		if 'volume_encoder' in grad_dict:
			self.volume_encoder.add_gradients(grad_dict["volume_encoder"])
		if 'frame_merge_network' in grad_dict:
			self.frame_merge_network.add_gradients(grad_dict["frame_merge_network"])
	
	@property
	def has_pending_gradients(self):
		return (isinstance(self.target_encoder, DeferredBackpropNetwork) and self.target_encoder.has_pending_gradients) \
			or (isinstance(self.volume_encoder, DeferredBackpropNetwork) and self.volume_encoder.has_pending_gradients) \
			or (isinstance(self.lifting_network, DeferredBackpropNetwork) and self.lifting_network.has_pending_gradients) \
			or (isinstance(self.frame_merge_network, DeferredBackpropNetwork) and self.frame_merge_network.has_pending_gradients)
		#(isinstance(self.target_lifting, GrowingUNet) and self.target_lifting.has_pending_gradients) or (isinstance(self.target_merging, GrowingUNet) and self.target_merging.has_pending_gradients) or 
	
	@property
	def is_all_frame_encoders_frozen(self):
		return not self.is_any_frame_encoders_unfrozen
	@property
	def is_any_frame_encoders_unfrozen(self):
		return (isinstance(self.target_encoder, DeferredBackpropNetwork) and not self.target_encoder.is_frozen_weights) \
			or (isinstance(self.volume_encoder, DeferredBackpropNetwork) and not self.volume_encoder.is_frozen_weights) \
			or (isinstance(self.lifting_network, DeferredBackpropNetwork) and not self.lifting_network.is_frozen_weights) \
			or (isinstance(self.frame_merge_network, DeferredBackpropNetwork) and not self.frame_merge_network.is_frozen_weights)
	
	def apply_gradients(self, optimizer, apply_density_gradients=True, apply_velocity_gradients=True, keep_gradients=False):
		raise NotImplementedError("Collect gradients via get_grads_vars(), call optimizer.apply_gradients() ONCE, then clear_gradients().")
		if isinstance(self.target_encoder, DeferredBackpropNetwork) and self.target_encoder.has_pending_gradients:
			self.target_encoder.apply_gradients(optimizer)
			if not keep_gradients:
				self.target_encoder.clear_gradients()
		if isinstance(self.lifting_network, DeferredBackpropNetwork) and self.lifting_network.has_pending_gradients:
			LOG.debug("NeuralState.apply_gradients to lifting_network.")
			self.lifting_network.apply_gradients(optimizer)
			if not keep_gradients:
				self.lifting_network.clear_gradients()
		if isinstance(self.volume_encoder, DeferredBackpropNetwork) and self.volume_encoder.has_pending_gradients:
			LOG.debug("NeuralState.apply_gradients to volume_encoder.")
			self.volume_encoder.apply_gradients(optimizer)
			if not keep_gradients:
				self.volume_encoder.clear_gradients()
		if isinstance(self.frame_merge_network, DeferredBackpropNetwork) and self.frame_merge_network.has_pending_gradients:
			LOG.debug("NeuralState.apply_gradients to frame_merge_network.")
			self.frame_merge_network.apply_gradients(optimizer)
			if not keep_gradients:
				self.frame_merge_network.clear_gradients()
		
		if apply_velocity_gradients and isinstance(self.velocity, NeuralVelocityGrid):
			self.velocity.apply_gradients(optimizer, keep_gradients=keep_gradients)
		if apply_density_gradients and isinstance(self.density, NeuralDensityGrid):
			self.density.apply_gradients(optimizer, keep_gradients=keep_gradients)
	
	def get_grads_vars(self, get_density_gradients=True, get_velocity_gradients=True, keep_gradients=True):
		grads_vars = []
		if isinstance(self.target_encoder, DeferredBackpropNetwork): #and self.target_encoder.has_pending_gradients:
			grads_vars.extend(self.target_encoder.get_grads_vars(keep_gradients=keep_gradients))
		if isinstance(self.lifting_network, DeferredBackpropNetwork): # and self.lifting_network.has_pending_gradients:
			grads_vars.extend(self.lifting_network.get_grads_vars(keep_gradients=keep_gradients))
		if isinstance(self.volume_encoder, DeferredBackpropNetwork): # and self.volume_encoder.has_pending_gradients:
			grads_vars.extend(self.volume_encoder.get_grads_vars(keep_gradients=keep_gradients))
		if isinstance(self.frame_merge_network, DeferredBackpropNetwork): # and self.frame_merge_network.has_pending_gradients:
			grads_vars.extend(self.frame_merge_network.get_grads_vars(keep_gradients=keep_gradients))
		
		if get_velocity_gradients and isinstance(self.velocity, NeuralVelocityGrid):
			grads_vars.extend(self.velocity.get_grads_vars(keep_gradients=keep_gradients))
		if get_density_gradients and isinstance(self.density, NeuralDensityGrid):
			grads_vars.extend(self.density.get_grads_vars(keep_gradients=keep_gradients))
		
		return grads_vars
	
	def clear_gradients(self, clear_density_gradients=True, clear_velocity_gradients=True):
		if isinstance(self.target_encoder, DeferredBackpropNetwork):
			self.target_encoder.clear_gradients()
		if isinstance(self.lifting_network, DeferredBackpropNetwork):
			self.lifting_network.clear_gradients()
		if isinstance(self.volume_encoder, DeferredBackpropNetwork):
			self.volume_encoder.clear_gradients()
		if isinstance(self.frame_merge_network, DeferredBackpropNetwork):
			self.frame_merge_network.clear_gradients()
		
		if clear_velocity_gradients and isinstance(self.velocity, NeuralVelocityGrid):
			self.velocity.clear_gradients()
		if clear_density_gradients and isinstance(self.density, NeuralDensityGrid):
			self.density.clear_gradients()
		


def Sequence_set_density_for_neural_globt(self, as_var=False, device=None, dt=1.0, order=1, clamp='NONE'):
	for i, state in enumerate(self):
		if i>0:
			#state.density = state.density.copy(as_var=as_var, device=device)
			#state.density = state.density.copy_empty(as_var=as_var, device=device)
			state.density = WarpedDensityGrid(order=order, dt=dt, clamp=clamp, device=device, scale_renderer=state.density.scale_renderer, is_SDF=state.density.is_SDF)
		state.density.parent_state = state

Sequence.set_density_for_neural_globt = Sequence_set_density_for_neural_globt