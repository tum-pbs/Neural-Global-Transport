import sys, os
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
from lib.util import HistoryBuffer, NO_OP, NO_CONTEXT, lerp, lerp_fast, copy_nested_structure
from lib.tf_ops import tf_None_to_const, tf_cosine_similarity, tf_pad_to_shape, shape_list, tf_laplace_filter_3d, tf_norm2
from lib.scalar_schedule import scalar_schedule
from .renderer import RenderingContext
from .vector import GridShape
import logging, warnings

LOG = logging.getLogger("Optimization")

# --- LOSSES ---
class LossSchedules:
	def __init__(self, *, \
			density_target=lambda i: 0.0, density_target_raw=lambda i: 0.0, \
			density_target_vol=lambda i: 0.0, density_proxy_vol=lambda i: 0.0, \
			density_target_depth_smoothness=lambda i: 0.0, \
			density_hull=lambda i: 0.0, \
			density_negative=lambda i: 0.0, density_smoothness=lambda i: 0.0, density_smoothness_2=lambda i: 0.0, \
			density_smoothness_temporal=lambda i: 0.0, density_warp=lambda i: 0.0, density_disc=lambda i: 0.0, \
			density_center=lambda i: 0.0, \
			
			SDF_target_pos=lambda i: 0.0, \
			
			velocity_target_vol=lambda i: 0.0, \
			velocity_warp_dens=lambda i: 0.0, velocity_warp_dens_proxy=lambda i: 0.0, velocity_warp_dens_target=lambda i: 0.0, velocity_warp_vel=lambda i: 0.0, velocity_divergence=lambda i: 0.0, \
			velocity_smoothness=lambda i: 0.0, velocity_cossim=lambda i: 0.0, velocity_magnitude=lambda i: 0.0, \
			velocity_CFLcond=lambda i: 0.0, velocity_MS_coherence=lambda i: 0.0, \
			
			density_lr=lambda i: 0.0, light_lr=lambda i: 0.0, velocity_lr=lambda i: 0.0, discriminator_lr=lambda i: 0.0, \
			density_decoder_train=lambda i: True, velocity_decoder_train=lambda i: True, frame_encoders_train=lambda i: True, \
			
			view_encoder_regularization=lambda i: 0.0, density_decoder_regularization=lambda i: 0.0, velocity_decoder_regularization=lambda i: 0.0, \
			discriminator_regularization=lambda i: 0.0, \
			
			velocity_warp_dens_MS_weighting=lambda i: 1.0, velocity_warp_dens_proxy_MS_weighting=lambda i: 1.0, velocity_warp_dens_tar_MS_weighting=lambda i: 1.0, velocity_warp_vel_MS_weighting=lambda i: 1.0, velocity_divergence_MS_weighting=lambda i: 1.0, \
			velocity_magnitude_MS_weighting=lambda i: 1.0, \
			velocity_CFLcond_MS_weighting=lambda i: 1.0, velocity_MS_coherence_MS_weighting=lambda i: 1.0, \
			
			sequence_length=lambda i: -1 \
		):
		self.density_target = density_target
		self.density_target_raw = density_target_raw
		self.density_target_vol = density_target_vol
		self.density_proxy_vol = density_proxy_vol
		self.density_target_depth_smoothness = density_target_depth_smoothness
		self.density_hull = density_hull
		self.density_negative = density_negative
		self.density_smoothness = density_smoothness
		self.density_smoothness_2 = density_smoothness_2
		self.density_smoothness_temporal = density_smoothness_temporal
		self.density_warp = density_warp
		self.density_disc = density_disc
		self.density_center = density_center
		
		self.SDF_target_pos = SDF_target_pos
		
		self.velocity_target_vol = velocity_target_vol
		self.velocity_warp_dens = velocity_warp_dens
		self.velocity_warp_dens_proxy = velocity_warp_dens_proxy
		self.velocity_warp_dens_target = velocity_warp_dens_target
		self.velocity_warp_vel = velocity_warp_vel
		self.velocity_divergence = velocity_divergence
		self.velocity_smoothness = velocity_smoothness
		self.velocity_cossim = velocity_cossim
		self.velocity_magnitude = velocity_magnitude
		self.velocity_CFLcond = velocity_CFLcond
		self.velocity_MS_coherence = velocity_MS_coherence
		
		self.density_lr = density_lr
		self.light_lr = light_lr
		self.velocity_lr = velocity_lr
		self.discriminator_lr = discriminator_lr
		
		self.density_decoder_train=density_decoder_train
		self.velocity_decoder_train=velocity_decoder_train
		self.frame_encoders_train=frame_encoders_train
		
		self.view_encoder_regularization = view_encoder_regularization
		self.density_decoder_regularization = density_decoder_regularization
		self.velocity_decoder_regularization = velocity_decoder_regularization
		self.discriminator_regularization = discriminator_regularization
		
		self.velocity_warp_dens_MS_weighting = velocity_warp_dens_MS_weighting
		self.velocity_warp_dens_proxy_MS_weighting = velocity_warp_dens_proxy_MS_weighting
		self.velocity_warp_dens_tar_MS_weighting = velocity_warp_dens_tar_MS_weighting
		self.velocity_warp_vel_MS_weighting = velocity_warp_vel_MS_weighting
		self.velocity_divergence_MS_weighting = velocity_divergence_MS_weighting
		self.velocity_magnitude_MS_weighting = velocity_magnitude_MS_weighting
		self.velocity_CFLcond_MS_weighting = velocity_CFLcond_MS_weighting
		self.velocity_MS_coherence_MS_weighting = velocity_MS_coherence_MS_weighting
		
		self.sequence_length = sequence_length
	
	def set_schedules(self, **kwargs):
		for name, value in kwargs.items():
			if not hasattr(self, name):
		#	try:
		#		item = getattr(self, name)
		#	except (KeyError, TypeError):
				raise AttributeError('Loss schedule {} does not exist.'.format(name))
		#	item.set(self, value)
			setattr(self, name, value)
		return self

def scale_losses(losses, scale):
	if isinstance(losses, (list, tuple)):
		return [_ * scale for _ in losses]
	elif isinstance(losses, tf.Tensor):
		return losses * scale
	else:
		raise TypeError

def reduce_losses(losses):
	if isinstance(losses, (list, tuple)):
		return tf.reduce_sum([tf.reduce_mean(_) for _ in losses])
	elif isinstance(losses, tf.Tensor):
		return tf.reduce_mean(losses)
	else:
		raise TypeError

class OptimizationContext:
	def __init__(self, setup, iteration, loss_schedules, \
			rendering_context, vel_scale=[1,1,1], warp_order=1, dt=1.0, buoyancy=None, \
			dens_warp_clamp="NONE", vel_warp_clamp="NONE", \
			density_optimizer=None, density_lr=1.0, light_optimizer=None, light_lr=1.0, \
			velocity_optimizer=None, velocity_lr=1.0, \
			frame=None, tf_summary=None, summary_interval=1, summary_pre=None, profiler=None, light_var_list=[], \
			allow_MS_losses=False, norm_spatial_dims=False):
		self.setup = setup
		self.profiler = profiler if profiler is not None else Profiler(active=False)
		self.iteration = iteration
		self.frame = frame
		#self.loss = 0
		self._losses = {}
		#self.loss_func = loss_func
		self.l1_loss = lambda i, t: tf.abs(i-t)
		self.l2_loss = tf.math.squared_difference #lambda i, t: (i-t)**2
		def huber_loss(i, t, delta=1.0):
			#tf.losses.huber_loss does not support broadcasting...
			abs_error = tf.abs(t-i)
			sqr = tf.minimum(abs_error, delta)
			lin = abs_error - sqr
			return (0.5*(sqr*sqr)) + (lin*delta)
		
		self.base_loss_functions = {
		#	'L0.5': lambda i, t: tf.sqrt(tf.abs(i-t)),
		#	'L1': self.l1_loss,
		#	'L2': self.l2_loss,
		#	'L3': lambda i, t: tf.pow(tf.abs(i-t), 3),
			'RAE': lambda i, t: tf.sqrt(tf.abs(i-t)),
			'MRAE': lambda i, t: tf.reduce_mean(tf.sqrt(tf.abs(i-t))),
			'AE': lambda i, t: tf.abs(i-t),
			'SAE': lambda i, t: tf.reduce_sum(tf.abs(i-t)),
			'MAE': lambda i, t: tf.reduce_mean(tf.abs(i-t)),
			'SE': tf.math.squared_difference,
			'SSE': lambda i, t: tf.reduce_sum(tf.math.squared_difference(i,t)),
			'MSE': lambda i, t: tf.reduce_mean(tf.math.squared_difference(i,t)),
			'RMSE': lambda i, t: tf.sqrt(tf.reduce_mean(tf.math.squared_difference(i,t))),
			'CAE': lambda i, t: tf.pow(tf.abs(i-t), 3),
			
			'HUBER': huber_loss, #lambda i,t: tf.losses.huber_loss(predictions=i, labels=t, reduction=tf.losses.Reduction.NONE),
			#'LBE': tf_log_barrier_ext,
		}
		self.default_loss_function = None #self.l2_loss
		self.loss_functions = {
			"density/target":		self.base_loss_functions["AE"], #self.l1_loss,
			"density/target_raw":	self.base_loss_functions["AE"], #self.l1_loss,
			"density/target_pos":	self.base_loss_functions["AE"], #self.l1_loss,
			"density/target_vol":	self.base_loss_functions["AE"], #self.l1_loss,
			"density/proxy_vol":	self.base_loss_functions["AE"], #self.l1_loss,
			"density/target_depth_smooth":	self.base_loss_functions["SE"], #self.l2_loss,
			"density/hull":			self.base_loss_functions["SE"], #self.l2_loss,
			"density/negative":		self.base_loss_functions["SE"], #self.l2_loss,
			"density/edge":			self.base_loss_functions["SE"], #self.l2_loss,
			"density/smooth":		self.base_loss_functions["SE"], #self.l2_loss,
			"density/smooth-temp":		self.base_loss_functions["SE"], #self.l2_loss,
			"density/warp":			self.base_loss_functions["AE"], #self.l1_loss,
			"density/center":		self.base_loss_functions["SE"], #self.l1_loss,
			
			"velocity/target_vol":		self.base_loss_functions["AE"], #self.l1_loss,
			"velocity/density_warp":	self.base_loss_functions["AE"], #self.l1_loss,
			"velocity/densProxy_warp":	self.base_loss_functions["SE"], #self.l1_loss,
			"velocity/densTar_warp":	self.base_loss_functions["AE"], #self.l1_loss,
			"velocity/velocity_warp":	self.base_loss_functions["AE"], #self.l1_loss,
			"velocity/divergence":		self.base_loss_functions["SE"], #self.l2_loss,
			"velocity/magnitude":		self.base_loss_functions["SE"], #self.l2_loss,
			"velocity/CFL":				self.base_loss_functions["SE"], #self.l2_loss,
			"velocity/smooth":			self.base_loss_functions["SE"],
			"velocity/cossim":			self.base_loss_functions["SE"],
		}
		self.loss_schedules = loss_schedules
		self.density_optimizer = density_optimizer
		self.density_lr = density_lr
		self.light_optimizer = light_optimizer
		self.light_lr = light_lr
		self.velocity_optimizer = velocity_optimizer
		self.velocity_lr = velocity_lr
		self._loss_summary = {}
		self._tf_summary = tf_summary
		self._summary_interval = summary_interval
		self.summary_pre = summary_pre
		self._compute_loss_summary = False
		
		self._target_weights = None
		self._target_weights_norm = 1.0
		
		self.render_ctx = rendering_context
		self.render_ops = {}
		self.vel_scale = vel_scale
		self.buoyancy = buoyancy
		self.warp_order = warp_order
		self.dens_warp_clamp = dens_warp_clamp
		self.vel_warp_clamp = vel_warp_clamp
		self.dt = dt
		self.light_var_list = light_var_list
		
		self.allow_MS_losses = allow_MS_losses
		self.norm_spatial_dims = norm_spatial_dims
		
		self.warp_dens_grads = False
		self.warp_dens_grads_decay = 0.9
		self.warp_vel_grads = False
		self.warp_vel_grads_decay = 0.9
		self.custom_dens_grads_weight = 1.0
		self.custom_vel_grads_weight = 1.0
		
		self._gradient_tape = None
		
		self.inspect_gradients = False
		self.inspect_gradients_func = NO_OP
		self.inspect_gradients_images_func = NO_OP
		self.inspect_gradients_images = {}
	
	def start_iteration(self, iteration, force=False, compute_loss_summary=False):
		'''Reset losses and set iteration
			will do nothing if iteration is already set
		'''
		self._compute_loss_summary = compute_loss_summary
		self.set_gradient_tape()
		if self.iteration==iteration and not force:
			return
		LOG.debug("Start iteration %d, update optimization context", iteration)
		self.iteration = iteration
		self._loss_summary = {}
	#	self.loss = 0
		self._losses = {}
		
		self.density_lr.assign(self.loss_schedules.density_lr(self.iteration))
		self.light_lr.assign(self.loss_schedules.light_lr(self.iteration))
		self.velocity_lr.assign(self.loss_schedules.velocity_lr(self.iteration))
		if self.record_summary:
			summary_names = self.make_summary_names('density/learning_rate')
			self._tf_summary.scalar(summary_names[0], self.density_lr.numpy(), step=self.iteration)
			summary_names = self.make_summary_names('velocity/learning_rate')
			self._tf_summary.scalar(summary_names[0], self.velocity_lr.numpy(), step=self.iteration)
	
	@property
	def target_weights(self):
		return self._target_weights
	@target_weights.setter
	def target_weights(self, weights):
		if weights is None:
			self._target_weights = None
			self._target_weights_norm = 1.0
		else:
			self._target_weights = tf.constant(weights, dtype=tf.float32)[:, np.newaxis, np.newaxis, np.newaxis]
			self._target_weights_norm = tf.constant(1./tf.reduce_sum(self._target_weights), dtype=tf.float32)
	@property
	def target_weights_norm(self):
		return self._target_weights_norm
	
	@property
	def tape(self):
		return self._gradient_tape
	
	def set_gradient_tape(self, tape=None):
		self._gradient_tape = tape
	
	def set_loss_func(self, loss_name, loss_function):
		if callable(loss_function):
			self.loss_functions[loss_name] = loss_function
		elif isinstance(loss_function, str):
			loss_function = loss_function.upper()
			if loss_function in self.base_loss_functions:
				self.loss_functions[loss_name] = self.base_loss_functions[loss_function]
			else:
				raise ValueError("Unknown loss function {} for loss {}".format(loss_function, loss_name))
		else:
			raise TypeError("Invalid loss function for loss {}".format(loss_name))
	
	def get_loss_func(self, loss_name):
		return self.loss_functions.get(loss_name, self.default_loss_function)
	
	def get_loss_func_name(self, loss_name):
		func = self.get_loss_func(loss_name)
		name = "UNKNOWN"
		for n,f in self.base_loss_functions.items():
			if f==func:
				name = n
				break
		return name
		
	
	def get_losses(self):
		loss_list = []
		for loss_tensors in self._losses.values():
			loss_list.extend(loss_tensors)
		return loss_list
	
	def pop_losses(self):
		'''Return current loss value and reset it to 0'''
		loss = self.get_losses()
	#	self.loss = 0
		self._losses = {}
		return loss
	
	def pop_loss_summary(self):
		loss_summary = self._loss_summary
		self._loss_summary = {}
		return loss_summary
	
	@property
	def record_summary(self):
		return self._tf_summary is not None and ((self.iteration+1)%self._summary_interval)==0
	
	def compute_loss_summary(self):
		return (self.record_summary or self._compute_loss_summary)
	
	@property
	def scale_density_target(self):
		return self.loss_schedules.density_target(self.iteration)
	
	def CV(self, schedule, it=None):
		'''Current Value of a scalar schedule'''
		if callable(schedule):
			return schedule(self.iteration if it is None else it)
		else: #isinstance(schedule, collection.abs.Mapping) and ('type' in schedule):
			return scalar_schedule(schedule, self.iteration if it is None else it)
	
	def LA(self, loss_scale):
		'''Loss Active'''
		if isinstance(loss_scale, bool):
			return loss_scale
		else:
			return not np.isclose(loss_scale, 0, atol=self.setup.training.loss_active_eps)
	
	#def add_loss(self, loss, loss_raw=None, loss_scale=None, loss_name=None):
	def add_loss(self, loss_tensors, loss_value=None, loss_value_scaled=None, loss_scale=None, loss_name=None):
		'''add a loss to the accumulator and write summaries
		change to:
			loss_tensor, for optimization
			loss_value, reduced loss_tensor, for value output
			loss_scale, 
		'''
		#self.loss +=loss
		if isinstance(loss_tensors, (list, tuple)):
			self._losses[loss_name] = loss_tensors
		elif isinstance(loss_tensors, tf.Tensor):
			self._losses[loss_name] = [loss_tensors]
		else:
			raise TypeError
		
		if loss_name is not None:
			self._loss_summary[loss_name] = (loss_value_scaled, loss_value, loss_scale)
			if self.record_summary:
				summary_names = self.make_summary_names(loss_name)
				#self._tf_summary.scalar(summary_names[0], loss, step=self.iteration)
				if loss_value_scaled is not None:
					self._tf_summary.scalar(summary_names[0], loss_value_scaled, step=self.iteration)
				if loss_value is not None:
					self._tf_summary.scalar(summary_names[1], loss_value, step=self.iteration)
				if loss_scale is not None:
					self._tf_summary.scalar(summary_names[2], loss_scale, step=self.iteration)
	
	def get_loss(self, loss_name):
		if loss_name in self._losses:
			return self._losses[loss_name]
		else:
			raise KeyError("Loss '%s' not recorded, available losses: %s"%(loss_name, list(self._losses.keys())))
	
	#def get_total_loss(self):
	#	return self.loss
	
	def add_render_op(self, name, func):
		if not name in self.render_ops: self.render_ops[name] = []
		self.render_ops[name].append(func)
	def remove_render_op(self, name, func):
		if name in self.render_ops:
			try:
				i = self.render_ops[name].index(func)
			except ValueError:
				pass
			else:
				del self.render_ops[name][i]
	def remove_render_ops(self, name):
		if name in self.render_ops:
			del self.render_ops[name]
	
	def RO_grid_dens_grad_scale(self, weight=1.0, sharpness=1.0, eps=1e-5):
	#	LOG.info("SCALE GRAD: init dens grid with weight %s", weight)
		@tf.custom_gradient
		def op(x):
			# input: combined light-density grid NDHWC with C=4 (3 light, 1 dens)
			gs = GridShape.from_tensor(x)
			channel = gs.c
			d = x[...,-1:]
		#	LOG.info("SCALE GRAD: dens grid fwd with shape %s, dens_shape %s", gs, GridShape.from_tensor(d))
			y = tf.identity(x)
			def grad(dy):
				# scale density gradient with exisiting density distribution
				#lerp: (1-w)*dy + w*(dy*(dens/mean(dens)))
				d_s = tf.pow(tf.abs(d), sharpness)
				m = tf.maximum(tf.reduce_max(d_s, axis=[-4,-3,-2], keepdims=True), eps)
				c = (1 - weight) + weight*(d_s/m)
				if channel>1:
					c = tf.pad(c, [(0,0),(0,0),(0,0),(0,0),(channel-1,0)], constant_values=1)
				gs_c = GridShape.from_tensor(c)
				dx = dy * c
				gs_dx = GridShape.from_tensor(dx)
		#		LOG.info("SCALE GRAD: dens grid bwd with shape %s, c-shape %s, weight %s", gs_dx, gs_c, weight)
				return dx
			return y, grad
		return op
	def RO_frustum_dens_grad_scale(self, weight=1.0, sharpness=1.0, eps=1e-5):
		#inspired by 'Single-image Tomography: 3D Volumes from 2D Cranial X-Rays' https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.13369
		@tf.custom_gradient
		def op(x):
			# input: sampled frustum grid NDHWC with C=4 (3 light, 1 dens)
			gs = GridShape.from_tensor(x)
			channel = gs.c
			d = x[...,-1:]
			y = tf.identity(x)
			def grad(dy):
				# scale density gradient with exisiting density distribution along view ray (z-axis)
				#lerp: (1-w)*dy + w*(dy*(dens/mean(dens)))
				d_s = tf.pow(tf.abs(d), sharpness)
				m = tf.maximum(tf.reduce_max(d_s, axis=-4, keepdims=True), eps)
			#	dx_c = dy * tf.pad((d/m), [(0,0),(0,0),(0,0),(0,0),(3,0)], constant_values=1)
			#	dx = dy + weight*(dx_c - dy) 
				c = (1 - weight) + weight*(d_s/m)
				if channel>1:
					c = tf.pad(c, [(0,0),(0,0),(0,0),(0,0),(channel-1,0)], constant_values=1)
				return dy * c #tf.pad((1 - weight) + weight*(d_s/m), [(0,0),(0,0),(0,0),(0,0),(3,0)], constant_values=1)
			return y, grad
		return op
	
	def set_inspect_gradient(self, active, func=None, img_func=None):
		self.inspect_gradients_images = {}
		if active:
			self.inspect_gradients = True
			self.inspect_gradients_func = func if func is not None else NO_OP
			self.inspect_gradients_images_func = img_func if img_func is not None else NO_OP
		else:
			self.inspect_gradients = False
			self.inspect_gradients_func = NO_OP
			self.inspect_gradients_images_func = NO_OP
	
	def make_summary_names(self, loss_name):
		summary_name = []
		if self.summary_pre is not None:
			summary_name.append(self.summary_pre)
		summary_name.append(loss_name)
		summary_name.append("{type}")
		if self.frame is not None:
			summary_name.append('f{:04d}'.format(self.frame))
		summary_name = "/".join(summary_name)
		return summary_name.format(type="final"), summary_name.format(type="raw"), summary_name.format(type="scale")
	
	def frame_pre(self, name):
		if self.frame is not None:
			return 'f{:04d}_{}'.format(self.frame, name)
		return name
### Density

def loss_dens_target(ctx, state, loss_func=None):
	# Render loss for density against targets without background
	loss_scale = ctx.scale_density_target 
	if ctx.LA(loss_scale):
		use_target_hulls = False #state.density.is_SDF 
		if use_target_hulls:
			warnings.warn("Using experimental hulls for the target loss.")
		if loss_func is None: loss_func = ctx.get_loss_func("density/target")
		if ctx.target_weights is None:
			norm_axis = [0,1] if state.density.is_SDF else [0]
			if 1 in norm_axis: warnings.warn("View norm in target loss.")
			if ctx.norm_spatial_dims: norm_axis += [2,3]
		elif ctx.norm_spatial_dims:
			raise NotImplementedError
		
		tmp_loss = []
		if not (ctx.allow_MS_losses and state.density.has_MS_output):
			with ctx.profiler.sample("target loss"):
				if ctx.target_weights is not None:
					if use_target_hulls:
						tmp_loss.append(tf.reduce_sum(loss_func(state.images, state.targets)*state.masks * ctx.target_weights, axis=0) * ctx.target_weights_norm) #*state.target_hulls
					else:
						tmp_loss.append(tf.reduce_sum(loss_func(state.images, state.targets)*ctx.target_weights, axis=0) * ctx.target_weights_norm) #weighted mean
				else:
					if use_target_hulls:
						tmp_loss.append(tf.reduce_mean(loss_func(state.images, state.targets)*state.masks, axis=norm_axis))
					else:
						tmp_loss.append(tf.reduce_mean(loss_func(state.images, state.targets), axis=norm_axis)) #mean over batch/cameras to be independent of it
		else:
			with ctx.profiler.sample("target loss MS"):
				for scale in state.density.gen_current_trainable_MS_scales():
					if ctx.target_weights is not None:
						if use_target_hulls:
							tmp_loss.append(tf.reduce_sum(loss_func(state.images_MS(scale), state.targets_MS(scale))*state.masks_MS(scale) * ctx.target_weights, axis=0) * ctx.target_weights_norm)
						else:
							tmp_loss.append(tf.reduce_sum(loss_func(state.images_MS(scale), state.targets_MS(scale))*ctx.target_weights, axis=0) * ctx.target_weights_norm) #weighted mean
					else:
						if use_target_hulls:
							tmp_loss.append(tf.reduce_mean(loss_func(state.images_MS(scale), state.targets_MS(scale))*state.masks_MS(scale), axis=norm_axis))
						else:
							tmp_loss.append(tf.reduce_mean(loss_func(state.images_MS(scale), state.targets_MS(scale)), axis=norm_axis)) #mean over batch/cameras to be independent of it
		
		tmp_loss_scaled = [_ * loss_scale for _ in tmp_loss]
		if ctx.compute_loss_summary():
			ctx.add_loss(tmp_loss_scaled, tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss]), tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss_scaled]), loss_scale, 'density/target')
		else:
			ctx.add_loss(tmp_loss_scaled, None, None, loss_scale, 'density/target')
		return True
	return False


def loss_dens_target_raw(ctx, state, loss_func=None):
	# Render with blended bkg loss for density against raw targets (with background)
	#loss_scale = ctx.CV(ctx.setup.training.density.raw_target_loss)
	loss_scale = ctx.loss_schedules.density_target_raw(ctx.iteration)
	if ctx.LA(loss_scale): #and (state.targets_raw is not None or not opt_ctx.allow_free_frames)
		use_target_hulls = state.density.is_SDF and not ctx.render_ctx.lights=="CAMLIGHT" #True
		if use_target_hulls:
			warnings.warn("Using hulls for the target_raw loss.")
		if loss_func is None: loss_func = ctx.get_loss_func("density/target_raw")
		loss_func_name = ctx.get_loss_func_name("density/target_raw")
		warnings.warn("target_raw loss func %s"%(loss_func_name))
		if not loss_func_name=="SE":
			raise ValueError("Target raw loss should be using SE!")
		
		if ctx.target_weights is None:
			norm_axis = [0,1] if state.density.is_SDF else [0]
			if 1 in norm_axis: warnings.warn("View norm in target_raw loss.")
			if ctx.norm_spatial_dims: norm_axis += [2,3]
		elif ctx.norm_spatial_dims:
			raise NotImplementedError
		
		tmp_loss = []
		if not (ctx.allow_MS_losses and state.density.has_MS_output):
			with ctx.profiler.sample("raw target loss"):
				if ctx.target_weights is not None:
					if use_target_hulls:
						tmp_loss.append(tf.reduce_sum(loss_func(state.images + state.bkgs*state.t, state.targets_raw)*state.masks*ctx.target_weights, axis=0) * ctx.target_weights_norm)#*state.target_raw_hulls
					else:
						tmp_loss.append(tf.reduce_sum(loss_func(state.images + state.bkgs*state.t, state.targets_raw)*ctx.target_weights, axis=0) * ctx.target_weights_norm)
				else:
					if use_target_hulls:
						tmp_loss.append(tf.reduce_mean(loss_func(state.images + state.bkgs*state.t, state.targets_raw)*state.masks, axis=norm_axis))
					else:
						tmp_loss.append(tf.reduce_mean(loss_func(state.images + state.bkgs*state.t, state.targets_raw), axis=norm_axis))
		else:
			with ctx.profiler.sample("raw target loss MS"):
				for scale in state.density.gen_current_trainable_MS_scales():
					if ctx.target_weights is not None:
						if use_target_hulls:
							tmp_loss.append(tf.reduce_sum(loss_func(state.images_MS(scale) + state.bkgs_MS(scale)*state.t_MS(scale), state.targets_raw_MS(scale))*state.masks_MS(scale)*ctx.target_weights, axis=0) * ctx.target_weights_norm)
						else:
							tmp_loss.append(tf.reduce_sum(loss_func(state.images_MS(scale) + state.bkgs_MS(scale)*state.t_MS(scale), state.targets_raw_MS(scale))*ctx.target_weights, axis=0) * ctx.target_weights_norm)
					else:
						img = state.images_MS(scale)
						img = img + state.bkgs_MS(scale)*state.t_MS(scale)
						tar = state.targets_raw_MS(scale)
						loss = loss_func(img, tar)
						if use_target_hulls:
							loss = loss*state.masks_MS(scale)
							tmp_loss.append(tf.reduce_mean(loss, axis=norm_axis))
						else:
							tmp_loss.append(tf.reduce_mean(loss, axis=norm_axis))
			
		tmp_loss_scaled = [_ * loss_scale for _ in tmp_loss]
		if ctx.compute_loss_summary():
			
			ctx.add_loss(tmp_loss_scaled, tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss]).numpy(), tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss_scaled]).numpy(), loss_scale, 'density/target_raw')
		else:
			ctx.add_loss(tmp_loss_scaled, None, None, loss_scale, 'density/target_raw')
		return True
	return False


def loss_dens_target_vol(ctx, state, loss_func=None):
	# Render with blended bkg loss for density against raw targets (with background)
	loss_scale = ctx.loss_schedules.density_target_vol(ctx.iteration)
	if ctx.LA(loss_scale): 
		if loss_func is None: loss_func = ctx.get_loss_func("density/target_vol")
		
		# shape: NDHWC
		if ctx.target_weights is None:
			norm_axis = [0]
			if ctx.norm_spatial_dims: norm_axis += [1,2,3]
		elif ctx.norm_spatial_dims:
			raise NotImplementedError
		
		if not (ctx.allow_MS_losses and state.density.has_MS_output):
			with ctx.profiler.sample("volume density target loss"):
				tmp_loss = []
				d_curr = state.density.d
				d_tar = state.density_target.d
				
				tmp_loss.append(tf.reduce_mean(loss_func(d_curr, d_tar), axis=norm_axis))
		else:
			with ctx.profiler.sample('volume density target loss MS'):
				tmp_loss = []
				for scale in state.density.gen_current_trainable_MS_scales(): #gen_current_MS_scales():
					c_shape = state.density.shape_of_scale(scale)
					d_curr = state.density.d_MS(scale)
					d_tar = state.density_target.scaled(c_shape)
					tmp_loss.append(tf.reduce_mean(loss_func(d_curr, d_tar), axis=norm_axis))
		tmp_loss_scaled = [_ * loss_scale for _ in tmp_loss]
		if ctx.compute_loss_summary():
			ctx.add_loss(tmp_loss_scaled, tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss]), tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss_scaled]), loss_scale, 'density/target_vol')
		else:
			ctx.add_loss(tmp_loss_scaled, None, None, loss_scale, 'density/target_vol')
		return True
	return False

def loss_dens_proxy_vol(ctx, state, loss_func=None):
	if state.has_density_neural:
		return False
	#loss_scale = ctx.CV(ctx.setup.training.density.raw_target_loss)
	loss_scale = ctx.loss_schedules.density_proxy_vol(ctx.iteration)
	if ctx.LA(loss_scale): #and (state.targets_raw is not None or not opt_ctx.allow_free_frames)
		if loss_func is None: loss_func = ctx.get_loss_func("density/proxy_vol")
		
		# shape: NDHWC
		if ctx.target_weights is None:
			norm_axis = [0]
			if ctx.norm_spatial_dims: norm_axis += [1,2,3]
		elif ctx.norm_spatial_dims:
			raise NotImplementedError
		
		if not (ctx.allow_MS_losses and state.density.has_MS_output):
			with ctx.profiler.sample("volume density proxy loss"):
				tmp_loss = []
				d_curr = state.density.d
				with ctx.tape.stop_recording():
					d_tar = state.density_proxy.d
				
				tmp_loss.append(tf.reduce_mean(loss_func(d_curr, d_tar), axis=norm_axis))
		else:
			with ctx.profiler.sample('volume density proxy loss MS'):
				tmp_loss = []
				for scale in state.density.gen_current_trainable_MS_scales(): 
					c_shape = state.density.shape_of_scale(scale)
					d_curr = state.density.d_MS(scale)
					with ctx.tape.stop_recording():
						if state.density_proxy.is_MS:
							d_tar = state.density_proxy.d_MS(scale)
						else:
							d_tar = state.density_proxy.scaled(c_shape)
					tmp_loss.append(tf.reduce_mean(loss_func(d_curr, d_tar), axis=norm_axis))
		tmp_loss_scaled = [_ * loss_scale for _ in tmp_loss]
		if ctx.compute_loss_summary():
			ctx.add_loss(tmp_loss_scaled, tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss]), tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss_scaled]), loss_scale, 'density/proxy_vol')
		else:
			ctx.add_loss(tmp_loss_scaled, None, None, loss_scale, 'density/proxy_vol')
		return True
	return False

def loss_dens_MS_coherence(ctx, state, loss_func=None):
	#use highest vel output to constrain lower vel scales
	loss_scale = ctx.loss_schedules.density_MS_coherence(ctx.iteration)
	top_as_label = True
	
	if ctx.LA(loss_scale):
		if loss_func is None: loss_func = ctx.get_loss_func("density/MS_coherence")
		# shape: NDHWC
		if ctx.target_weights is None:
			norm_axis = [0]
			if ctx.norm_spatial_dims: norm_axis += [1,2,3]
		elif ctx.norm_spatial_dims:
			raise NotImplementedError
		
		if not (ctx.allow_MS_losses and state.density.has_MS_output):
			LOG.warning("Loss 'density/MS_coherence' is active, but density has no MS or MS losses are not allowed.")
			return False
		else:
			LOG.debug("Run multi-scale coherence loss for density.")
			with ctx.profiler.sample('density MS coherence loss'):
				tmp_loss = []
				dens_MS_scales = list(reversed(list(state.density.gen_current_MS_scales())[:-1])) #fine to coarse, starting at 2nd highest
				if len(dens_MS_scales)<1:
					LOG.debug("Insuficient scales for density multi-scale coherence loss.")
					return False
				
				last_scale_dens = state.density.d_MS(state.density._get_top_active_scale())
				
				for scale in dens_MS_scales: #fine to coarse, starting at 2nd highest
					MS_weight = ctx.loss_schedules.density_MS_coherence_MS_weighting(scale)
					# what is better, sampling always from the finest scale or only from the next finer? (regarding performance and gradient quality)
					last_scale_dens = state.density.resample_density(state.density.scale_renderer, last_scale_dens, shape=state.density.shape_of_scale(scale))
					if top_as_label:
						last_scale_dens = tf.stop_gradient(last_scale_dens)
					
					tmp_loss.append( MS_weight * tf.reduce_mean(loss_func(last_scale_dens, state.density.d_MS(scale)), axis=norm_axis))
				
		tmp_loss_scaled = [_*loss_scale for _ in tmp_loss]
		if ctx.compute_loss_summary():
			ctx.add_loss(tmp_loss_scaled, tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss]), tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss_scaled]), loss_scale, 'velocity/MS_coherence')
		else:
			ctx.add_loss(tmp_loss_scaled, None, None, loss_scale, 'velocity/MS_coherence')
		return True
	return False

def loss_dens_target_depth_smooth(ctx, state, frustum_density, loss_func=None):
	# Smoothness loss for density using forward differences (gradient computation of the 3D laplace filter convolution is so slow...)
	#loss_scale = ctx.CV(ctx.setup.training.density.smoothness_loss)
	loss_scale = ctx.loss_schedules.density_target_depth_smoothness(ctx.iteration)
	if ctx.LA(loss_scale):
		if loss_func is None: loss_func = ctx.get_loss_func("density/target_depth_smooth")
		
		with ctx.profiler.sample('density target depth gradient loss'):
			tmp_loss = [
				loss_func(frustum_density[:,1:,:,:,:], frustum_density[:,:-1,:,:,:]), #z_grad = d[:,1:,:,:,:] - d[:,:-1,:,:,:]
			]
			#tmp_loss = loss_func( (1.0/3.0)*(tf.abs(x_grad) + tf.abs(y_grad) + tf.abs(z_grad)), 0)
		tmp_loss_scaled = [_ * loss_scale for _ in tmp_loss]
		if ctx.compute_loss_summary():
			ctx.add_loss(tmp_loss_scaled, tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss]), tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss_scaled]), loss_scale, loss_name='density/target_depth_smooth')
		else:
			ctx.add_loss(tmp_loss_scaled, None, None, loss_scale, loss_name='density/target_depth_smooth')
		return True
	return False

def loss_dens_hull(ctx, state, loss_func=None):
	""" Loss to reduce density outside the hull: density*(1-hull)
		This loss considers the raw density without the hull applied, even if density.restrict_to_hull is True
	"""
	loss_scale = ctx.loss_schedules.density_hull(ctx.iteration)
	if ctx.LA(loss_scale): 
		raise NotImplementedError("Implement normalization")
		if loss_func is None: loss_func = ctx.get_loss_func("density/hull")
		with ctx.profiler.sample("density hull loss"):
			tmp_loss = loss_func(state.density._d * (1.-state.density.hull), 0)
		tmp_loss_scaled = tmp_loss * loss_scale
		if ctx.compute_loss_summary():
			ctx.add_loss([tmp_loss_scaled], tf.reduce_mean(tmp_loss), tf.reduce_mean(tmp_loss_scaled), loss_scale, 'density/hull')
		else:
			ctx.add_loss([tmp_loss_scaled], None, None, loss_scale, 'density/hull')
		return True
	return False

def loss_dens_negative(ctx, state, loss_func=None):
	# loss for negative density
	loss_scale = ctx.loss_schedules.density_negative(ctx.iteration)
	if ctx.LA(loss_scale): 
		raise NotImplementedError("Implement normalization")
		if loss_func is None: loss_func = ctx.get_loss_func("density/negative")
		with ctx.profiler.sample("negative density loss"):
			tmp_loss = tf.reduce_mean(loss_func(tf.maximum(-state.density._d, 0), 0), axis=0)
		tmp_loss_scaled = tmp_loss * loss_scale
		if ctx.compute_loss_summary():
			ctx.add_loss([tmp_loss_scaled], tf.reduce_mean(tmp_loss), tf.reduce_mean(tmp_loss_scaled), loss_scale, 'density/negative')
		else:
			ctx.add_loss([tmp_loss_scaled], None, None, loss_scale, 'density/negative')
		return True
	return False

def get_narrow_band(data, value):
	return tf.cast(tf.less_equal(tf.abs(data), value), tf.float32)

def loss_dens_smooth(ctx, state, loss_func=None):
	# Smoothness (laplace edge filter) loss for density
	#loss_scale = ctx.CV(ctx.setup.training.density.smoothness_loss)
	loss_scale = ctx.loss_schedules.density_smoothness(ctx.iteration)
	narrow_band = 0
	if ctx.LA(loss_scale):
		# shape: NDHWC
		if ctx.target_weights is None:
			norm_axis = [0]
			if ctx.norm_spatial_dims: norm_axis += [1,2,3]
		elif ctx.norm_spatial_dims:
			raise NotImplementedError
		
		if loss_func is None: loss_func = ctx.get_loss_func("density/edge")
		if not (ctx.allow_MS_losses and state.density.has_MS_output):
			with ctx.profiler.sample('density edge loss'):
				#tmp_loss = tf.reduce_mean(tf.abs(tf_laplace_filter_3d(state.density.d, neighbours=ctx.setup.training.density.smoothness_neighbours)))
				if narrow_band>0:
					d = state.density.d
					with ctx.tape.stop_recording():
						mask = get_narrow_band(d, narrow_band)
					tmp_loss = [tf.reduce_mean(loss_func(tf_laplace_filter_3d(d, neighbours=1, padding='VALID'), 0)*mask, axis=norm_axis)]
				else:
					tmp_loss = [tf.reduce_mean(loss_func(tf_laplace_filter_3d(state.density.d, neighbours=1, padding='VALID'), 0), axis=norm_axis)] #VALID=ignore borders
		else:
			with ctx.profiler.sample('density edge loss MS'):
				tmp_loss = []
				for scale in state.density.gen_current_trainable_MS_scales():
					if narrow_band>0:
						d = state.density._d_MS(scale)
						with ctx.tape.stop_recording():
							mask = get_narrow_band(d, narrow_band)
						tmp_loss.append(tf.reduce_mean(loss_func(tf_laplace_filter_3d(d, neighbours=1, padding='VALID'), 0)*mask, axis=norm_axis))
					else:
						tmp_loss.append(tf.reduce_mean(loss_func(tf_laplace_filter_3d(state.density._d_MS(scale), neighbours=1, padding='VALID'), 0), axis=norm_axis)) #VALID=ignore borders
		tmp_loss_scaled = [_ * loss_scale for _ in tmp_loss]
		if ctx.compute_loss_summary():
			ctx.add_loss(tmp_loss_scaled, tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss]), tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss_scaled]), loss_scale, loss_name='density/edge')
		else:
			ctx.add_loss(tmp_loss_scaled, None, None, loss_scale, loss_name='density/edge')
		return True
	return False

def loss_dens_smooth_2(ctx, state, loss_func=None):
	# Smoothness loss for density using forward differences (gradient computation of the 3D laplace filter convolution is so slow...)
	#loss_scale = ctx.CV(ctx.setup.training.density.smoothness_loss)
	loss_scale = ctx.loss_schedules.density_smoothness_2(ctx.iteration)
	if ctx.LA(loss_scale):
		# shape: NDHWC
		if ctx.target_weights is None:
			norm_axis = [0]
			if ctx.norm_spatial_dims: norm_axis += [1,2,3]
		elif ctx.norm_spatial_dims:
			raise NotImplementedError
		
		if loss_func is None: loss_func = ctx.get_loss_func("density/smooth")
		if not (ctx.allow_MS_losses and state.density.has_MS_output):
			with ctx.profiler.sample('density gradient loss'):
				d = state.density.d
				tmp_loss = [
					tf.reduce_mean(loss_func(d[:,:,:,1:,:] - d[:,:,:,:-1,:], 0), axis=norm_axis), #x_grad = d[:,:,:,1:,:] - d[:,:,:,:-1,:]
					tf.reduce_mean(loss_func(d[:,:,1:,:,:] - d[:,:,:-1,:,:], 0), axis=norm_axis), #y_grad = d[:,:,1:,:,:] - d[:,:,:-1,:,:]
					tf.reduce_mean(loss_func(d[:,1:,:,:,:] - d[:,:-1,:,:,:], 0), axis=norm_axis), #z_grad = d[:,1:,:,:,:] - d[:,:-1,:,:,:]
				]
				#tmp_loss = loss_func( (1.0/3.0)*(tf.abs(x_grad) + tf.abs(y_grad) + tf.abs(z_grad)), 0)
		else:
			raise NotImplementedError
		tmp_loss_scaled = [_ * loss_scale for _ in tmp_loss]
		if ctx.compute_loss_summary():
			ctx.add_loss(tmp_loss_scaled, tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss]), tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss_scaled]), loss_scale, loss_name='density/smooth')
		else:
			ctx.add_loss(tmp_loss_scaled, None, None, loss_scale, loss_name='density/smooth')
		return True
	return False


def loss_dens_smooth_temporal(ctx, state, loss_func=None):
	# Temoral smoothness loss for density, for tomofluid tests
	#loss_scale = ctx.CV(ctx.setup.training.density.smoothness_loss)
	loss_scale = ctx.loss_schedules.density_smoothness_temporal(ctx.iteration)
	if ctx.LA(loss_scale) and (state.prev is not None or state.next is not None):
		raise NotImplementedError("Implement normalization")
		if loss_func is None: loss_func = ctx.get_loss_func("density/smooth-temp")
		with ctx.profiler.sample('density temporal gradient loss'):
			d = state.density.d
			tmp_loss = 0
			if state.prev is not None:
				tmp_loss += loss_func(state.prev.density.d, d)
			if state.next is not None:
				tmp_loss += loss_func(d, state.next.density.d)
		tmp_loss_scaled = tmp_loss * loss_scale
		if ctx.compute_loss_summary():
			ctx.add_loss([tmp_loss_scaled], tf.reduce_mean(tmp_loss), tf.reduce_mean(tmp_loss_scaled), loss_scale, 'density/smooth-temp')
		else:
			ctx.add_loss([tmp_loss_scaled], None, None, loss_scale, 'density/smooth-temp')
		return True
	return False

def loss_dens_warp(ctx, state, loss_func=None):
	#warp loss "loss(A(dt, vt), dt+1)" between prev and current and current and next state for density.
	# will scale the velocity to match the density shape/resolution
	#loss_scale = ctx.CV(ctx.setup.training.density.warp_loss)
	loss_scale = ctx.loss_schedules.density_warp(ctx.iteration)
	if ctx.LA(loss_scale) and (state.prev is not None or state.next is not None):
		raise NotImplementedError("Implement normalization")
		if loss_func is None: loss_func = ctx.get_loss_func("density/warp")
		tmp_loss = 0
		with ctx.profiler.sample('density warp loss'):
			if state.prev is not None:
				tmp_loss += loss_func(state.prev.density_advected(order=ctx.warp_order, dt=ctx.dt, clamp=ctx.dens_warp_clamp), state.density.d)
			if state.next is not None:
				tmp_loss += loss_func(state.density_advected(order=ctx.warp_order, dt=ctx.dt, clamp=ctx.dens_warp_clamp), state.next.density.d)
		tmp_loss_scaled = tmp_loss * loss_scale
		if ctx.compute_loss_summary():
			ctx.add_loss([tmp_loss_scaled], tf.reduce_mean(tmp_loss), tf.reduce_mean(tmp_loss_scaled), loss_scale, 'density/warp')
		else:
			ctx.add_loss([tmp_loss_scaled], None, None, loss_scale, 'density/warp')
		return True
	return False

#randomize_rot_cams(disc_cameras, [-30,30], [0,360])
def loss_dens_disc(ctx, state, disc, img_list=None):
	# loss from the discriminator for the density
	if ctx.setup.training.discriminator.active and ctx.setup.training.discriminator.start_delay<=ctx.iteration:
		#loss_scale = ctx.CV(ctx.setup.training.discriminator.loss_scale, ctx.iteration-ctx.setup.training.discriminator.start_delay)
		loss_scale = ctx.loss_schedules.density_disc(ctx.iteration-ctx.setup.training.discriminator.start_delay)
		loss_active = ctx.LA(loss_scale)
		#randomize_rot_cams(disc_cameras, [-30,30], [0,360])
		if loss_active or disc.record_history: #only render if needed for history or loss
			LOG.debug('Render discriminator input for density loss')
			disc_in = disc.fake_samples(state, history_samples=False, concat=False, spatial_augment=False, name="dens_disc_samples")
		#		
			if img_list is not None and isinstance(img_list, list):
				img_list.extend(disc_in)
			if loss_active:
				LOG.debug('Run discriminator loss for density')
				with ctx.profiler.sample("dens disc loss"):
					disc_in = tf.concat(disc_in, axis=0)
					if ctx.inspect_gradients:
						ctx.inspect_gradients_images['density/discriminator'] = disc_in
					#disc_in = disc.check_input(disc_in, "dens_disc")
					disc_in = (disc_in,)if (disc.loss_type in ["SGAN"]) else (disc.real_samples(spatial_augment=True, intensity_augment=True),disc_in)
					#disc_out = disc.model(disc_in, training=False)
					tmp_loss, disc_scores = disc.loss(disc_in, flip_target=not (disc.loss_type in ["SGAN"]), training=False) #tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_out, labels=disc.real_labels(disc_out))
					#disc.check_output(disc_out, disc_loss, disc_in, "dens_disc")
					#tmp_loss = tf.math.reduce_mean(disc_loss)
				tmp_loss_scaled = scale_losses(tmp_loss, loss_scale)
				if ctx.compute_loss_summary():
					ctx.add_loss(tmp_loss_scaled, reduce_losses(tmp_loss), reduce_losses(tmp_loss_scaled), loss_scale, 'density/discriminator')
				else:
					ctx.add_loss(tmp_loss_scaled, None, None, loss_scale, 'density/discriminator')
				return True
		#END disc render or density loss
	return False

def get_dens_center_mask(shape, scale=1.0, squared=False):
	mask = tf.abs(tf.cast(tf.linspace(1.0,-1.0, shape[0]), dtype=tf.float32)) #D
	mask = mask * scale
	if squared:
		mask = mask*mask
	mask = tf.expand_dims(mask, axis=-1) #DH
	mask = tf.expand_dims(mask, axis=-1) #DHW
	mask = tf.tile(mask, (1,shape[1], shape[2])) #DHW
	return tf.expand_dims(mask, axis=-1) #DHWC

def loss_dens_center(ctx, state, loss_func=None):
	# push mass towards the center
	loss_scale = ctx.loss_schedules.density_center(ctx.iteration)
	if ctx.LA(loss_scale):
		
		norm_axis = [0]
		if ctx.norm_spatial_dims: norm_axis += [1,2,3]
		
		if loss_func is None: loss_func = ctx.get_loss_func("density/center")
		if not (ctx.allow_MS_losses and state.density.has_MS_output):
			with ctx.profiler.sample('density center loss'):
				d = state.density.d
				with ctx.tape.stop_recording():
					weight = get_dens_center_mask(state.density.shape, scale=1.0, squared=True)
				tmp_loss = [tf.reduce_mean(loss_func(tf.abs(d), 0)*weight, axis=norm_axis) ]
		else:
			with ctx.profiler.sample('density center loss MS'):
				tmp_loss = []
				for scale in state.density.gen_current_trainable_MS_scales():
					d = state.density._d_MS(scale)
					with ctx.tape.stop_recording():
						weight = get_dens_center_mask(state.density.shape_of_scale(scale), scale=1.0, squared=True)
					tmp_loss.append(tf.reduce_mean(loss_func(tf.abs(d), 0)*weight, axis=norm_axis) )
		tmp_loss_scaled = [_ * loss_scale for _ in tmp_loss]
		if ctx.compute_loss_summary():
			ctx.add_loss(tmp_loss_scaled, tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss]), tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss_scaled]), loss_scale, loss_name='density/center')
		else:
			ctx.add_loss(tmp_loss_scaled, None, None, loss_scale, loss_name='density/center')
		return True
	return False

def loss_view_enc_weights(ctx, state):
	loss_scale = ctx.loss_schedules.view_encoder_regularization(ctx.iteration)
	if ctx.LA(loss_scale):
		with ctx.profiler.sample('view encoder regularization'):
			weights = state.get_variables()["encoder"]
			tmp_loss = tf.reduce_mean([tf.reduce_mean(tf.nn.l2_loss(var)) for var in weights])
		tmp_loss_scaled = tmp_loss * loss_scale
		ctx.add_loss([tmp_loss_scaled], tmp_loss, tmp_loss_scaled, loss_scale, 'viewenc/regularization')
		return True
	return False

def loss_dens_dec_weights(ctx, state):
	loss_scale = ctx.loss_schedules.density_decoder_regularization(ctx.iteration)
	if ctx.LA(loss_scale):
		with ctx.profiler.sample('density decoder regularization'):
			weights = state.density.get_variables()["density_decoder"]
			tmp_loss = tf.reduce_mean([tf.reduce_mean(tf.nn.l2_loss(var)) for var in weights])
		tmp_loss_scaled = tmp_loss * loss_scale
		ctx.add_loss([tmp_loss_scaled], tmp_loss, tmp_loss_scaled, loss_scale, 'density/regularization')
		return True
	return False

def warp_dens_grads(opt_ctx, state, grads, order='FWD'):
	if order.upper()=='FWD': #propagate density gradients to next state, simple forward warp. do not clamp negative gradients
		raise NotImplementedError
		return state.prev.velocity.warp(grads, order=opt_ctx.warp_order, dt=opt_ctx.dt, clamp="NONE" if opt_ctx.dens_warp_clamp=="NEGATIVE" else opt_ctx.dens_warp_clamp), tf.constant(0, dtype=tf.float32), tf.constant(0, dtype=tf.float32), tf.constant(0, dtype=tf.float32)
	elif order.upper()=='BWD': #propagate density gradients to previous state, backprop through prev->warp
		#d = state.density.with_inflow()
		var_list = state.get_output_variables() #state.density.var_list() + state.velocity.var_list() #state.var_list()
		with tf.GradientTape(watch_accessed_variables=False) as tape:
			tape.watch(var_list)
			d_warp = state.density_advected(order=opt_ctx.warp_order, dt=opt_ctx.dt, clamp=opt_ctx.dens_warp_clamp)#state.velocity.warp(d, order=opt_ctx.warp_order, dt=opt_ctx.dt)
		return tape.gradient([d_warp], var_list, output_gradients=[grads])
	else:
		raise ValueError

def apply_custom_grads_with_check(grad, custom_grad, custom_grad_scale=1.):
	if grad is None and custom_grad is None:
		LOG.warning("apply_custom_grads: base and custom gradient is None.")
	
	if grad is None:
		#LOG.warning("apply_custom_grads: base gradient is None.")
		grad = 0.
	if custom_grad is None:
		#LOG.warning("apply_custom_grads: custom gradient is None.")
		custom_grad = 0.
	
	return grad + custom_grad * custom_grad_scale

def optStep_density(opt_ctx, state, use_vel=False, disc_ctx=None, disc_samples_list=None, custom_dens_grads=None, apply_dens_grads=False):
	
	#dens_var_list = state.density.var_list()
	dens_vars = state.density.get_output_variables(include_MS=opt_ctx.allow_MS_losses, include_residual=True, only_trainable=True) # get_variables()
	with opt_ctx.profiler.sample('optStep_density'):
		#with opt_ctx.profiler.sample('loss'), tf.GradientTape(watch_accessed_variables=False, persistent=opt_ctx.inspect_gradients) as dens_tape:
		with opt_ctx.profiler.sample('loss'), tf.GradientTape(watch_accessed_variables=False, persistent=True) as dens_tape:
			dens_tape.watch(dens_vars)
			if opt_ctx.light_var_list:
				dens_tape.watch(opt_ctx.light_var_list)
			if opt_ctx.inspect_gradients:
				dens_inspect_vars = {}
				dens_inspect_vars.update(dens_vars)
				if opt_ctx.light_var_list: dens_inspect_vars['lights'] = opt_ctx.light_var_list
			opt_ctx.set_gradient_tape(dens_tape)
			
			catch_frustum_grid = opt_ctx.LA(opt_ctx.loss_schedules.density_target_depth_smoothness(opt_ctx.iteration))
			fg_container = []
			if catch_frustum_grid:
				# use the custom render op hooks to catch the reference to the frutum grid tensor
				def _catch_FG(fg):
					fg_container.append(fg)
					return fg
				opt_ctx.add_render_op("FRUSTUM", _catch_FG)
			else:
				fg_container.append(None)
			
			
			state.render_density(opt_ctx.render_ctx, custom_ops=opt_ctx.render_ops)
			if opt_ctx.allow_MS_losses and state.density.is_MS:
				state.render_density_MS_stack(opt_ctx.render_ctx, custom_ops=opt_ctx.render_ops)
			
			if catch_frustum_grid:
				opt_ctx.remove_render_op("FRUSTUM", _catch_FG)
			
			LOG.debug("Density losses")
			active_density_loss = False
			# Render loss for density against targets without background
			active_density_loss = loss_dens_target(opt_ctx, state) or active_density_loss
			active_density_loss = loss_dens_target_raw(opt_ctx, state) or active_density_loss
			active_density_loss = loss_dens_target_vol(opt_ctx, state) or active_density_loss
			active_density_loss = loss_dens_proxy_vol(opt_ctx, state) or active_density_loss
			active_density_loss = loss_dens_target_depth_smooth(opt_ctx, state, fg_container[0]) or active_density_loss
			active_density_loss = loss_dens_hull(opt_ctx, state) or active_density_loss
			active_density_loss = loss_dens_negative(opt_ctx, state) or active_density_loss
			active_density_loss = loss_dens_smooth(opt_ctx, state) or active_density_loss
			active_density_loss = loss_dens_smooth_2(opt_ctx, state) or active_density_loss
			active_density_loss = loss_dens_smooth_temporal(opt_ctx, state) or active_density_loss
			
			if use_vel and state.velocity is not None:
				active_density_loss = loss_dens_warp(opt_ctx, state) or active_density_loss
			
			if disc_ctx is not None:
				active_density_loss = loss_dens_disc(opt_ctx, state, disc_ctx, disc_samples_list) or active_density_loss
			
			#density_loss = opt_ctx.loss #opt_ctx.pop_loss()
		#END gradient tape
		if active_density_loss:
			if custom_dens_grads is not None:
				cdg_scale = opt_ctx.CV(opt_ctx.custom_dens_grads_weight)
			with opt_ctx.profiler.sample('gradient'):
				if opt_ctx.inspect_gradients:
					for loss_name in opt_ctx._losses:
						dens_grads = dens_tape.gradient(opt_ctx.get_loss(loss_name), dens_inspect_vars)
						for k, g in dens_grads.items():
							#LOG.info("inspect gradient %s", k)
							if k.startswith('density'):
								if g is not None:
									opt_ctx.inspect_gradients_func(opt_ctx=opt_ctx, gradients=g, name=loss_name+k[7:])
								#else:
									#LOG.info("gradient %s is None", k)
						del dens_grads
						
						if loss_name in opt_ctx.inspect_gradients_images:
							img_grads = dens_tape.gradient(opt_ctx.get_loss(loss_name), [opt_ctx.inspect_gradients_images[loss_name]])
							opt_ctx.inspect_gradients_images_func(opt_ctx=opt_ctx, gradients=img_grads[0], name=loss_name)
							del img_grads
							del opt_ctx.inspect_gradients_images[loss_name]
					
					if custom_dens_grads is not None and opt_ctx.LA(cdg_scale):
						
						has_valid_dens_grads = False
						for k, g in custom_dens_grads.items():
							if k.startswith('density'):
								if g is not None:
									has_valid_dens_grads = True
									opt_ctx.inspect_gradients_func(opt_ctx=opt_ctx, gradients=g, name="density/custom_grad"+k[7:])
								else:
									LOG.debug("Custom density gradient of frame %d for '%s' is None.", state.frame, k)
						if not has_valid_dens_grads:
							LOG.warning("All custom density gradients of frame %d are None.", state.frame)
						
						if custom_dens_grads.get('inflow') is not None:
							opt_ctx.inspect_gradients_func(opt_ctx=opt_ctx, gradients=custom_dens_grads['inflow']*cdg_scale, name="custom_inflow_grad")
					opt_ctx.inspect_gradients_images = {}
				#
				#if opt_ctx.inspect_gradients:
					dens_grads = dens_tape.gradient(opt_ctx.get_losses(), dens_inspect_vars)
					has_valid_dens_grads = False
					for k, g in dens_grads.items():
						if k.startswith('density'):
							if g is not None:
								has_valid_dens_grads = True
								opt_ctx.inspect_gradients_func(opt_ctx=opt_ctx, gradients=g, name="density/_local"+k[7:])
							else:
								LOG.debug("Total density of frame %d for '%s' is None.", state.frame, k)
					if not has_valid_dens_grads:
						LOG.warning("All local density gradients of frame %d are None.", state.frame)
					
					if dens_grads.get('inflow') is not None: #len(dens_grads)>1 and dens_grads[1] is not None:
						opt_ctx.inspect_gradients_func(opt_ctx=opt_ctx, gradients=dens_grads['inflow'], name="density/inflow")
					if dens_grads.get('lights') is not None:
						for i, lg in enumerate(dens_grads['lights']):
							opt_ctx.inspect_gradients_func(opt_ctx=opt_ctx, gradients=lg, name="density/light_{}".format(i))
					
					del dens_grads
				
				LOG.debug('Compute and apply density gradients')
				if opt_ctx.light_var_list:
					dens_vars['lights'] = opt_ctx.light_var_list
				dens_grads = dens_tape.gradient(opt_ctx.get_losses(), dens_vars)
				del dens_tape
				
				curr_dens_grads = copy_nested_structure(dens_grads)
				if custom_dens_grads is not None and opt_ctx.LA(cdg_scale):
					if opt_ctx.allow_MS_losses and state.density.has_MS_output:
						# back-warping give gradients w.r.t the the output variables, not the MS-output variables, so map to the highest MS-output.
						custom_dens_grads = state.density.map_gradient_output_to_MS(custom_dens_grads)
					#dens_grads = nest.map_structure(lambda d, c: apply_custom_grads_with_check(d, c, cdg_scale), dens_grads, custom_dens_grads)
					
					for k in dens_grads: #depth 1 sufficient for now...
						if k in custom_dens_grads: 
							if custom_dens_grads[k] is not None:
								LOG.debug("Update density gradient '%s' with custom gradient.", k)
								if dens_grads[k] is None:
									#LOG.warning("This should crash now...")
									dens_grads[k] = custom_dens_grads[k]*cdg_scale
								else:
									dens_grads[k] += custom_dens_grads[k]*cdg_scale
							#backprop_accumulate hadles this already
						#	elif dens_grads[k] is None:
						#		dens_grads[k] = 0.0
							#else:
							#	LOG.debug("Custom density gradient '%s' of frame %d is None", k, state.frame)
					
					for k in custom_dens_grads:
						if k not in dens_grads:
							LOG.warning("Custom density gradient '%s' can't be mapped and will be ignored.", k)
				
				if opt_ctx.light_var_list:
					opt_ctx.light_optimizer.apply_gradients(zip(dens_grads['lights'], opt_ctx.light_var_list))
					del dens_grads['lights']
					del dens_vars['lights']
				
				if "inflow" in dens_grads:
					opt_ctx.density_optimizer.apply_gradients(zip(dens_grads["inflow"], dens_vars["inflow"]))
					del dens_grads['inflow']
					del dens_vars['inflow']
				
				#LOG.debug("Apply dens grads of frame %d in iteration %d: %s", state.frame, opt_ctx.iteration, apply_dens_grads)
				state.density.set_output_gradients_for_backprop_accumulate(dens_grads, include_MS=opt_ctx.allow_MS_losses, include_residual=True, only_trainable=True)
				if apply_dens_grads:
					#state.density.backprop_accumulate(dens_grads, include_MS=opt_ctx.allow_MS_losses, include_residual=True, only_trainable=True)
					state.density._compute_input_grads() # calls backprop_accumulate with grads set before.
					
		else:
			curr_dens_grads = nest.map_structure(lambda v: tf.constant(0, dtype=tf.float32), dens_vars) #[tf.constant(0, dtype=tf.float32) for _ in range(len(dens_var_list))]
		
		opt_ctx.set_gradient_tape()
		#del dens_tape
		
		with opt_ctx.profiler.sample('clamp density'): #necessary as negative density really breaks the rendering
			d = state.density.d
		#	if opt_ctx.setup.training.density.use_hull:
		#		d = d*state.hull # hull is a binary mask
			#d = tf.clip_by_value(d, opt_ctx.CV(opt_ctx.setup.data.density.min), opt_ctx.CV(opt_ctx.setup.data.density.max))
			#state.density.assign(d)
			state.density.apply_clamp(opt_ctx.CV(opt_ctx.setup.data.density.min), opt_ctx.CV(opt_ctx.setup.data.density.max))
				
	return active_density_loss, curr_dens_grads

### Velocity

def loss_vel_target_vol(ctx, state, loss_func=None):
	# Render with blended bkg loss for density against raw targets (with background)
	#loss_scale = ctx.CV(ctx.setup.training.density.raw_target_loss)
	loss_scale = ctx.loss_schedules.velocity_target_vol(ctx.iteration)
	if ctx.LA(loss_scale): #and (state.targets_raw is not None or not opt_ctx.allow_free_frames)
		if loss_func is None: loss_func = ctx.get_loss_func("velocity/target_vol")
		# shape: NDHWC
		if ctx.target_weights is None:
			norm_axis = [0]
			if ctx.norm_spatial_dims: norm_axis += [1,2,3]
		elif ctx.norm_spatial_dims:
			raise NotImplementedError
		
		if not (ctx.allow_MS_losses and state.velocity.has_MS_output):
			with ctx.profiler.sample("volume velocity target loss"):
				if state.velocity.is_centered:
					tmp_loss = [
						tf.reduce_mean(loss_func(state.velocity.centered(), state.velocity_target.centered()), axis=norm_axis),
					]
				elif state.velocity.is_staggered:
					tmp_loss = [
						tf.reduce_mean(loss_func(state.velocity.x, state.velocity_target.x), axis=norm_axis),
						tf.reduce_mean(loss_func(state.velocity.y, state.velocity_target.y), axis=norm_axis),
						tf.reduce_mean(loss_func(state.velocity.z, state.velocity_target.z), axis=norm_axis),
					]
				else:
					raise ValueError("Unknown velocity type.")
		else:
			with ctx.profiler.sample("volume velocity target MS loss"):
				tmp_loss=[]
				if state.velocity.is_centered:
					
					for scale in state.velocity.gen_current_trainable_MS_scales():
						#MS_weight = ctx.loss_schedules.velocity_target_vol_MS_weighting(scale)
						# what is better, sampling always from the finest scale or only from the next finer? (regarding performance and gradient quality)
						vel_target = state.velocity.resample_velocity(state.velocity.scale_renderer, state.velocity_target.centered(), shape=state.velocity.centered_shape_of_scale(scale), \
							is_staggered=state.velocity.is_staggered, scale_magnitude=True)
						
						tmp_loss.append( tf.reduce_mean(loss_func(vel_target, state.velocity.centered_MS(scale)), axis=norm_axis))
				
				elif state.velocity.is_staggered:
					#last_scale_vel = state.velocity._staggered_MS(state.velocity._get_top_active_scale()) # (x,y,z)
					
					for scale in state.velocity.gen_current_trainable_MS_scales():
						#MS_weight = ctx.loss_schedules.velocity_target_vol_MS_weighting(scale)
						# what is better, sampling always from the finest scale or only from the next finer? (regarding performance and gradient quality)
						vel_target = state.velocity.resample_velocity(state.velocity.scale_renderer, state.velocity_target._staggered(), shape=state.velocity.centered_shape_of_scale(scale), \
							is_staggered=state.velocity.is_staggered, scale_magnitude=True)
						
						tmp_loss.append( tf.reduce_mean(loss_func(vel_target[0], state.velocity._staggered_MS(scale)[0]), axis=norm_axis)) #x
						tmp_loss.append( tf.reduce_mean(loss_func(vel_target[1], state.velocity._staggered_MS(scale)[1]), axis=norm_axis)) #y
						tmp_loss.append( tf.reduce_mean(loss_func(vel_target[2], state.velocity._staggered_MS(scale)[2]), axis=norm_axis)) #z
				else:
					raise ValueError("Unknown velocity type.")
		tmp_loss_scaled = [_ * loss_scale for _ in tmp_loss]
		if ctx.compute_loss_summary():
			ctx.add_loss(tmp_loss_scaled, tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss]), tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss_scaled]), loss_scale, 'velocity/target_vol')
		else:
			ctx.add_loss(tmp_loss_scaled, None, None, loss_scale, 'velocity/target_vol')
		return True
	return False

def loss_vel_warp_dens(ctx, state, loss_func=None):
	#warp loss "loss(A(dt, vt), dt+1)" between current and next state for velocity.
	#loss_scale = ctx.CV(ctx.setup.training.velocity.density_warp_loss)
	loss_scale = ctx.loss_schedules.velocity_warp_dens(ctx.iteration)
	if ctx.LA(loss_scale) and state.next is not None:
		if loss_func is None: loss_func = ctx.get_loss_func("velocity/density_warp")
		# shape: NDHWC
		if ctx.target_weights is None:
			norm_axis = [0]
			if ctx.norm_spatial_dims: norm_axis += [1,2,3]
		elif ctx.norm_spatial_dims:
			raise NotImplementedError
		
		if not (ctx.allow_MS_losses and state.velocity.has_MS_output):
			LOG.debug("Run dens-warp loss for velocity, curr to next")
			with ctx.profiler.sample('velocity dens warp loss'):
				curr_dens = state.density.scaled(state.velocity.centered_shape, with_inflow=True)
				next_dens = state.next.density.scaled(state.velocity.centered_shape)
				tmp_loss = [tf.reduce_mean(loss_func(state.velocity.warp(curr_dens, order=ctx.warp_order, dt=ctx.dt, clamp=ctx.dens_warp_clamp), next_dens), axis=norm_axis)]
		else:
			LOG.debug("Run multi-scale dens-warp loss for velocity, curr to next")
			with ctx.profiler.sample('velocity dens warp loss MS'):
				tmp_loss = []
				for scale in state.velocity.gen_current_trainable_MS_scales(): #gen_current_MS_scales():
					c_shape = state.velocity.centered_shape_of_scale(scale)
					curr_dens = state.density.scaled(c_shape, with_inflow=True)
					next_dens = state.next.density.scaled(c_shape)
					c_vel = state.velocity.centered_MS(scale)
					MS_weight = ctx.loss_schedules.velocity_warp_dens_MS_weighting(scale)
					tmp_loss.append( tf.reduce_mean(loss_func(state.velocity.warp(curr_dens, order=ctx.warp_order, dt=ctx.dt, clamp=ctx.dens_warp_clamp, centered_velocity=c_vel), next_dens), axis=norm_axis) * MS_weight)
		tmp_loss_scaled = [_*loss_scale for _ in tmp_loss]
		if ctx.compute_loss_summary():
			ctx.add_loss(tmp_loss_scaled, tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss]), tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss_scaled]), loss_scale, 'velocity/density_warp')
		else:
			ctx.add_loss(tmp_loss_scaled, None, None, loss_scale, 'velocity/density_warp')
		return True
	return False

def loss_vel_warp_dens_proxy(ctx, state, loss_func=None):
	#warp loss "loss(A(dt, vt), dt+1)" between current and next state for velocity.
	#loss_scale = ctx.CV(ctx.setup.training.velocity.density_warp_loss)
	loss_scale = ctx.loss_schedules.velocity_warp_dens_proxy(ctx.iteration)
	if ctx.LA(loss_scale) and state.next is not None:
		if not state.next.has_density_proxy:
			raise RuntimeError("proxy warp loss target state has no density proxy.")
		if loss_func is None: loss_func = ctx.get_loss_func("velocity/densProxy_warp")
		# shape: NDHWC
		if ctx.target_weights is None:
			norm_axis = [0]
			if ctx.norm_spatial_dims: norm_axis += [1,2,3]
		elif ctx.norm_spatial_dims:
			raise NotImplementedError
		
		if not (ctx.allow_MS_losses and state.velocity.has_MS_output):
			LOG.debug("Run dens-proxy-warp loss for velocity, curr to next")
			with ctx.profiler.sample('velocity dens-proxy warp loss'):
				curr_dens = state.density.scaled(state.velocity.centered_shape, with_inflow=True)
				with ctx.tape.stop_recording():
					next_dens = state.next.density_proxy.scaled(state.velocity.centered_shape)
				#next_dens = tf.stop_gradient(next_dens)
				tmp_loss = [tf.reduce_mean(loss_func(state.velocity.warp(curr_dens, order=ctx.warp_order, dt=ctx.dt, clamp=ctx.dens_warp_clamp), next_dens), axis=norm_axis)]
		else:
			LOG.debug("Run multi-scale dens-proxy-warp loss for velocity, curr to next")
			with ctx.profiler.sample('velocity dens-proxy warp loss MS'):
				tmp_loss = []
				for scale in state.velocity.gen_current_trainable_MS_scales(): #gen_current_MS_scales():
					c_shape = state.velocity.centered_shape_of_scale(scale)
					curr_dens = state.density.scaled(c_shape, with_inflow=True)
					with ctx.tape.stop_recording():
						next_dens = state.next.density_proxy.scaled(c_shape)
					#next_dens = tf.stop_gradient(next_dens)
					c_vel = state.velocity.centered_MS(scale)
					MS_weight = ctx.loss_schedules.velocity_warp_dens_proxy_MS_weighting(scale)
					tmp_loss.append( tf.reduce_mean(loss_func(state.velocity.warp(curr_dens, order=ctx.warp_order, dt=ctx.dt, clamp=ctx.dens_warp_clamp, centered_velocity=c_vel), next_dens), axis=norm_axis) * MS_weight)
		tmp_loss_scaled = [_*loss_scale for _ in tmp_loss]
		if ctx.compute_loss_summary():
			ctx.add_loss(tmp_loss_scaled, tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss]), tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss_scaled]), loss_scale, 'velocity/densProxy_warp')
		else:
			ctx.add_loss(tmp_loss_scaled, None, None, loss_scale, 'velocity/densProxy_warp')
		return True
	return False

def loss_vel_warp_dens_target(ctx, state, loss_func=None):
	#warp loss "loss(A(dt, vt), dt+1)" between current and next state for velocity.
	#loss_scale = ctx.CV(ctx.setup.training.velocity.density_warp_loss)
	loss_scale = ctx.loss_schedules.velocity_warp_dens_target(ctx.iteration)
	if ctx.LA(loss_scale) and state.next is not None:
		if loss_func is None: loss_func = ctx.get_loss_func("velocity/densTar_warp")
		# shape: NDHWC
		if ctx.target_weights is None:
			norm_axis = [0]
			if ctx.norm_spatial_dims: norm_axis += [1,2,3]
		elif ctx.norm_spatial_dims:
			raise NotImplementedError
		
		if not (ctx.allow_MS_losses and state.velocity.has_MS_output):
			LOG.debug("Run dens-tar-warp loss for velocity, curr to next")
			with ctx.profiler.sample('velocity dens-target warp loss'):
				curr_dens = state.density.scaled(state.velocity.centered_shape, with_inflow=True)
				next_dens = state.next.density_target.scaled(state.velocity.centered_shape)
				tmp_loss = [tf.reduce_mean(loss_func(state.velocity.warp(curr_dens, order=ctx.warp_order, dt=ctx.dt, clamp=ctx.dens_warp_clamp), next_dens), axis=norm_axis)]
		else:
			LOG.debug("Run multi-scale dens-tar-warp loss for velocity, curr to next")
			with ctx.profiler.sample('velocity dens-target warp loss MS'):
				tmp_loss = []
				for scale in state.velocity.gen_current_trainable_MS_scales(): #gen_current_MS_scales():
					c_shape = state.velocity.centered_shape_of_scale(scale)
					curr_dens = state.density.scaled(c_shape, with_inflow=True)
					next_dens = state.next.density_target.scaled(c_shape)
					c_vel = state.velocity.centered_MS(scale)
					MS_weight = ctx.loss_schedules.velocity_warp_dens_tar_MS_weighting(scale)
					tmp_loss.append( tf.reduce_mean(loss_func(state.velocity.warp(curr_dens, order=ctx.warp_order, dt=ctx.dt, clamp=ctx.dens_warp_clamp, centered_velocity=c_vel), next_dens), axis=norm_axis) * MS_weight)
		tmp_loss_scaled = [_*loss_scale for _ in tmp_loss]
		if ctx.compute_loss_summary():
			ctx.add_loss(tmp_loss_scaled, tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss]), tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss_scaled]), loss_scale, 'velocity/densTar_warp')
		else:
			ctx.add_loss(tmp_loss_scaled, None, None, loss_scale, 'velocity/densTar_warp')
		return True
	return False

def loss_vel_warp_vel(ctx, state, loss_func=None):
	#warp loss "loss(A(vt, vt), vt+1)" between prev and current and current and next state for velocity.
	#loss_scale = ctx.CV(ctx.setup.training.velocity.velocity_warp_loss)
	loss_scale = ctx.loss_schedules.velocity_warp_vel(ctx.iteration)
	if ctx.LA(loss_scale) and (state.prev is not None or state.next is not None):
		raise NotImplementedError("Implement normalization")
		if loss_func is None: loss_func = ctx.get_loss_func("velocity/velocity_warp")
		LOG.debug("Run vel-warp loss for velocity")
		tmp_loss = [0,0,0]
		with ctx.profiler.sample('velocity vel warp loss'):
			if state.prev is not None:# and state.next is not None:
				if (ctx.allow_MS_losses and state.velocity.has_MS_output and state.prev.velocity.has_MS_output):
					if not state.velocity.has_same_MS_shapes(state.prev.velocity):
						raise RuntimeError("multi-scale shapes of velocities of frames %d and %d are not compatible."%(state.frame, state.next.frame))
					raise NotImplementedError("MS loss for vel warp vel loss not implemented.")
				LOG.debug("Warp loss prev to curr")
				prev_warped = state.prev.velocity_advected(order=ctx.warp_order, dt=ctx.dt, clamp=ctx.vel_warp_clamp)
				
				# buoyancy
				if ctx.setup.training.optimize_buoyancy or tf.reduce_any(tf.not_equal(ctx.buoyancy, 0.0)):
					prev_warped = prev_warped.with_buoyancy(ctx.buoyancy, state.density)
				
				tmp_loss[0] += loss_func(prev_warped.x, state.velocity.x)
				tmp_loss[1] += loss_func(prev_warped.y, state.velocity.y)
				tmp_loss[2] += loss_func(prev_warped.z, state.velocity.z)
			if state.next is not None:# and state.next.next is not None:
				if (ctx.allow_MS_losses and state.velocity.has_MS_output and state.next.velocity.has_MS_output):
					if not state.velocity.has_same_MS_shapes(state.next.velocity):
						raise RuntimeError("multi-scale shapes of velocities of frames %d and %d are not compatible."%(state.frame, state.next.frame))
					raise NotImplementedError("MS loss for vel warp vel loss not implemented.")
				LOG.debug("Warp loss curr to next")
				curr_warped = state.velocity_advected(order=ctx.warp_order, dt=ctx.dt, clamp=ctx.vel_warp_clamp)
				
				# buoyancy
				if ctx.setup.training.optimize_buoyancy or tf.reduce_any(tf.not_equal(ctx.buoyancy, 0.0)):
					curr_warped = curr_warped.with_buoyancy(ctx.buoyancy, state.next.density)
				
				tmp_loss[0] += loss_func(curr_warped.x, state.next.velocity.x)
				tmp_loss[1] += loss_func(curr_warped.y, state.next.velocity.y)
				tmp_loss[2] += loss_func(curr_warped.z, state.next.velocity.z)
		tmp_loss_scaled = [_ * loss_scale for _ in tmp_loss]
		if ctx.compute_loss_summary():
			ctx.add_loss(tmp_loss_scaled, tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss]), tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss_scaled]), loss_scale, 'velocity/velocity_warp')
		else:
			ctx.add_loss(tmp_loss_scaled, None, None, loss_scale, 'velocity/velocity_warp')
		return True
	return False

'''
def loss_vel_smooth(ctx, state):
	''Smoothness (laplace edge filter) loss for velocity''
	#loss_scale = ctx.CV(ctx.setup.training.velocity.smoothness_loss)
	loss_scale = ctx.loss_schedules.velocity_smoothness(ctx.iteration)
	if ctx.LA(loss_scale):
		raise NotImplementedError
		LOG.debug("Run smoothness loss for velocity")
		with ctx.profiler.sample('velocity edge loss'):
			vel_components = state.velocity.var_list()
			#normalize to unit width to prevent loss scaling issues with growing optimization
			vel_components = [component*scale for component, scale in zip(vel_components, ctx.vel_scale[::-1])]
			tmp_loss = tf.reduce_mean([tf.reduce_mean(tf.abs(tf_laplace_filter_3d(vel_cmp, neighbours=ctx.setup.training.velocity.smoothness_neighbours))) for vel_cmp in vel_components])
		tmp_loss_scaled = tmp_loss * loss_scale
		ctx.add_loss(tmp_loss_scaled, tmp_loss, loss_scale, 'velocity/edge')
		return True
	return False
'''

def loss_vel_smooth(ctx, state, loss_func=None):
	'''Smoothness (forward differences) loss for velocity'''
	#loss_scale = ctx.CV(ctx.setup.training.velocity.smoothness_loss)
	MS_residual_loss = False
	loss_scale = ctx.loss_schedules.velocity_smoothness(ctx.iteration)
	if ctx.LA(loss_scale):
		if loss_func is None: loss_func = ctx.get_loss_func("velocity/smooth")
		
		norm_MS_total_scales = True and state.velocity.is_MS and state.velocity.recursive_MS_shared_decoder
		norm_MS_affected_scales = True and state.velocity.is_MS and state.velocity.recursive_MS_shared_decoder
		norm_vel_scale = True
		norm_vel_scale_by_world_size = False
		
		# shape: NDHWC
		if ctx.target_weights is None:
			norm_axis = [0]
			if ctx.norm_spatial_dims: norm_axis += [1,2,3]
		elif ctx.norm_spatial_dims:
			raise NotImplementedError
		
		if not (ctx.allow_MS_losses and state.velocity.has_MS_output):
			if MS_residual_loss and state.velocity.has_MS_output:
				raise NotImplementedError("TODO: MS_residual_loss might not work if backprop does not use MS variables, wich is tied to ctx.allow_MS_losses.")
			with ctx.profiler.sample('%s velocity gradient loss'%("residual" if MS_residual_loss else "full")):
				tmp_loss = []
				
				if norm_vel_scale:
					if norm_vel_scale_by_world_size:
						raise NotImplementedError
						t = state.velocity_tranform
						t.grid_size = self.shape_of_scale(scale)
						vel_scale = t.cell_size_world()
					else:
						if state.velocity.is_MS:
							scale_factor = state.velocity.recursive_MS_scale_factor
							scale = state.velocity._get_top_active_scale()
						else:
							#raise NotImplementedError
							# r64 starting at r4 with scale 2
							scale_factor = 2
							scale = 4
						vel_scale = [1/(scale_factor**scale)]*3
				else:
					vel_scale = [1,1,1]
					
				if state.velocity.is_centered:
					vel_components = state.velocity.centered(pad_lod=False, concat=False)
				elif state.velocity.is_staggered:
					vel_components = state.velocity._staggered()
				else:
					raise ValueError("Unknown velocity type.")
				
				for c, s in zip(vel_components, vel_scale):
					c = c*s
					tmp_loss.append(tf.reduce_mean(loss_func(tf.abs(c[:,:,:,1:,:] - c[:,:,:,:-1,:]), 0), axis=norm_axis)) #x_grad = d[:,:,:,1:,:] - d[:,:,:,:-1,:]
					tmp_loss.append(tf.reduce_mean(loss_func(tf.abs(c[:,:,1:,:,:] - c[:,:,:-1,:,:]), 0), axis=norm_axis)) #y_grad = d[:,:,1:,:,:] - d[:,:,:-1,:,:]
					tmp_loss.append(tf.reduce_mean(loss_func(tf.abs(c[:,1:,:,:,:] - c[:,:-1,:,:,:]), 0), axis=norm_axis)) #z_grad = d[:,1:,:,:,:] - d[:,:-1,:,:,:]
		else:
			with ctx.profiler.sample('velocity gradient loss'):
				tmp_loss = []
				scales = tuple(state.velocity.gen_current_trainable_MS_scales())
				scale_factor = state.velocity.recursive_MS_scale_factor
				for scale in scales: #gen_current_MS_scales():
					
					if norm_vel_scale:
						if norm_vel_scale_by_world_size:
							raise NotImplementedError
							t = state.velocity_tranform
							t.grid_size = self.shape_of_scale(scale)
							vel_scale = t.cell_size_world()
						else:
							vel_scale = [1/(scale_factor**scale)]*3
					else:
						vel_scale = [1,1,1]
					
					MS_weight = 1.0
					if norm_MS_affected_scales:
						MS_weight /= scale + 1
					if norm_MS_total_scales:
						MS_weight /= len(scales)
					
					if state.velocity.is_centered:
						if MS_residual_loss:
							vel_components = state.velocity.centered_MS_residual(scale, pad_lod=False, concat=False)
						else:
							vel_components = state.velocity.centered_MS(scale, pad_lod=False, concat=False)
					elif state.velocity.is_staggered:
						if MS_residual_loss:
							vel_components = state.velocity._staggered_MS_residual(scale)
						else:
							vel_components = state.velocity._staggered_MS(scale)
					else:
						raise ValueError("Unknown velocity type.")
					
					for c, s in zip(vel_components, vel_scale):
						c = c*s
						tmp_loss.append(tf.reduce_mean(loss_func(tf.abs(c[:,:,:,1:,:] - c[:,:,:,:-1,:]), 0), axis=norm_axis)*MS_weight) #x_grad = d[:,:,:,1:,:] - d[:,:,:,:-1,:]
						tmp_loss.append(tf.reduce_mean(loss_func(tf.abs(c[:,:,1:,:,:] - c[:,:,:-1,:,:]), 0), axis=norm_axis)*MS_weight) #y_grad = d[:,:,1:,:,:] - d[:,:,:-1,:,:]
						tmp_loss.append(tf.reduce_mean(loss_func(tf.abs(c[:,1:,:,:,:] - c[:,:-1,:,:,:]), 0), axis=norm_axis)*MS_weight) #z_grad = d[:,1:,:,:,:] - d[:,:-1,:,:,:]
		
		# with ctx.profiler.sample('velocity gradient loss'):
			# vel_components = state.velocity.var_list()
			# tmp_loss = []
			# for c in vel_components:
				# tmp_loss.append(tf.reduce_mean(loss_func(c[:,:,:,1:,:] - c[:,:,:,:-1,:], 0), axis=norm_axis)) #x_grad = d[:,:,:,1:,:] - d[:,:,:,:-1,:]
				# tmp_loss.append(tf.reduce_mean(loss_func(c[:,:,1:,:,:] - c[:,:,:-1,:,:], 0), axis=norm_axis)) #y_grad = d[:,:,1:,:,:] - d[:,:,:-1,:,:]
				# tmp_loss.append(tf.reduce_mean(loss_func(c[:,1:,:,:,:] - c[:,:-1,:,:,:], 0), axis=norm_axis)) #z_grad = d[:,1:,:,:,:] - d[:,:-1,:,:,:]
			
		tmp_loss_scaled = [_ * loss_scale for _ in tmp_loss]
		if ctx.compute_loss_summary():
			ctx.add_loss(tmp_loss_scaled, tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss]), tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss_scaled]), loss_scale, loss_name='velocity/smooth')
		else:
			ctx.add_loss(tmp_loss_scaled, None, None, loss_scale, loss_name='velocity/smooth')
		return True
	return False

def loss_vel_cossim(ctx, state, loss_func=None):
	'''Smoothness (forward differences) loss for velocity'''
	#loss_scale = ctx.CV(ctx.setup.training.velocity.smoothness_loss)
	loss_scale = ctx.loss_schedules.velocity_cossim(ctx.iteration)
	if ctx.LA(loss_scale):
		raise NotImplementedError("Implement normalization")
		if loss_func is None: loss_func = ctx.get_loss_func("velocity/cossim")
		with ctx.profiler.sample('velocity cosine loss'):
			v = state.velocity.centered()
			tmp_loss = [
				loss_func(tf_cosine_similarity(v[:,:,:,1:,:], v[:,:,:,:-1,:], axis=-1)*(-0.5)+0.5, 0),
				loss_func(tf_cosine_similarity(v[:,:,1:,:,:], v[:,:,:-1,:,:], axis=-1)*(-0.5)+0.5, 0),
				loss_func(tf_cosine_similarity(v[:,1:,:,:,:], v[:,:-1,:,:,:], axis=-1)*(-0.5)+0.5, 0),
			]
			
		tmp_loss_scaled = [_ * loss_scale for _ in tmp_loss]
		if ctx.compute_loss_summary():
			ctx.add_loss(tmp_loss_scaled, tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss]), tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss_scaled]), loss_scale, loss_name='velocity/cossim')
		else:
			ctx.add_loss(tmp_loss_scaled, None, None, loss_scale, loss_name='velocity/cossim')
		return True
	return False

def loss_vel_divergence(ctx, state, loss_func=None):
	'''divergence loss'''
	#loss_scale = ctx.CV(ctx.setup.training.velocity.divergence_loss)
	loss_scale = ctx.loss_schedules.velocity_divergence(ctx.iteration)
	if ctx.LA(loss_scale):
		if ctx.target_weights is None:
			norm_axis = [0]
			if ctx.norm_spatial_dims: norm_axis += [1,2,3]
		elif ctx.norm_spatial_dims:
			raise NotImplementedError
		
		if loss_func is None: loss_func = ctx.get_loss_func("velocity/divergence")
		LOG.debug("Run divergence loss for velocity")
		if not (ctx.allow_MS_losses and state.velocity.has_MS_output):
			with ctx.profiler.sample('divergence loss'):
				tmp_loss = [tf.reduce_mean(loss_func(state.velocity.divergence(ctx.vel_scale), 0), axis=norm_axis)]
		else:
			with ctx.profiler.sample('divergence loss MS'):
				tmp_loss = []
				for scale in state.velocity.gen_current_MS_scales():
					MS_weight = ctx.loss_schedules.velocity_divergence_MS_weighting(scale)
					tmp_loss.append( tf.reduce_mean(MS_weight * loss_func(state.velocity.divergence_MS(scale, ctx.vel_scale), 0), axis=norm_axis) )
		tmp_loss_scaled = [_ * loss_scale for _ in tmp_loss]
		#	if setup.training.velocity.divergence_normalize>0:
		#		raise NotImplementedError()
		if ctx.compute_loss_summary():
			ctx.add_loss(tmp_loss_scaled, tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss]), tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss_scaled]), loss_scale, 'velocity/divergence')
		else:
			ctx.add_loss(tmp_loss_scaled, None, None, loss_scale, 'velocity/divergence')
		return True
	return False

def loss_vel_magnitude(ctx, state, loss_func=None):
	'''
		loss to minimize velocities
		tf.norm can cause issues (NaN gradients at 0 magnitude): https://github.com/tensorflow/tensorflow/issues/12071
	'''
	loss_scale = ctx.loss_schedules.velocity_magnitude(ctx.iteration)
	if ctx.LA(loss_scale):
		if loss_func is None: loss_func = ctx.get_loss_func("velocity/magnitude")
		
		norm_MS_total_scales = True and state.velocity.is_MS and state.velocity.recursive_MS_shared_decoder
		norm_MS_affected_scales = True and state.velocity.is_MS and state.velocity.recursive_MS_shared_decoder
		norm_vel_scale = True
		norm_vel_scale_by_world_size = False
		
		
		# shape: NDHWC
		if ctx.target_weights is None:
			norm_axis = [0]
			if ctx.norm_spatial_dims: norm_axis += [1,2,3]
		elif ctx.norm_spatial_dims:
			raise NotImplementedError
		
		LOG.debug("Run vector magnitude loss for velocity")
		if not (ctx.allow_MS_losses and state.velocity.has_MS_output):
			with ctx.profiler.sample('magnitude loss'):
				if norm_vel_scale:
					if norm_vel_scale_by_world_size:
						raise NotImplementedError
						t = state.velocity_tranform
						t.grid_size = self.shape_of_scale(scale)
						vel_scale = t.cell_size_world()
					else:
						if state.velocity.is_MS:
							scale_factor = state.velocity.recursive_MS_scale_factor
							scale = state.velocity._get_top_active_scale()
						else:
							#raise NotImplementedError
							# r64 starting at r4 with scale 2
							scale_factor = 2
							scale = 4
						vel_scale = [1/(scale_factor**scale)]*3
				else:
					vel_scale = [1,1,1]
				tmp_loss = [tf.reduce_mean(loss_func(state.velocity.magnitude(vel_scale), 0), axis=norm_axis)]
		else:
			with ctx.profiler.sample('magnitude loss MS'):
				tmp_loss = []
				scales = tuple(state.velocity.gen_current_trainable_MS_scales())
				scale_factor = state.velocity.recursive_MS_scale_factor
				for scale in scales:
					
					if norm_vel_scale:
						if norm_vel_scale_by_world_size:
							raise NotImplementedError
							t = state.velocity_tranform
							t.grid_size = self.shape_of_scale(scale)
							vel_scale = t.cell_size_world()
						else:
							vel_scale = [1/(scale_factor**scale)]*3
					else:
						vel_scale = [1,1,1]
					
					MS_weight = ctx.loss_schedules.velocity_magnitude_MS_weighting(scale)
					if norm_MS_affected_scales:
						MS_weight /= scale + 1
					if norm_MS_total_scales:
						MS_weight /= len(scales)
					
					tmp_loss.append(tf.reduce_mean(loss_func(state.velocity.magnitude_MS(scale, vel_scale), 0), axis=norm_axis) * MS_weight )
		tmp_loss_scaled = [_ * loss_scale for _ in tmp_loss]
		if ctx.compute_loss_summary():
			ctx.add_loss(tmp_loss_scaled, tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss]), tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss_scaled]), loss_scale, 'velocity/magnitude')
		else:
			ctx.add_loss(tmp_loss_scaled, None, None, loss_scale, 'velocity/magnitude')
		return True
	return False

def loss_vel_CFLcond(ctx, state, loss_func=None):
	'''loss to minimize velocities'''
	loss_scale = ctx.loss_schedules.velocity_CFLcond(ctx.iteration)
	if ctx.LA(loss_scale):
		if loss_func is None: loss_func = ctx.get_loss_func("velocity/CFL")
		loss_before_cutoff = True
		MS_residual_loss = True
		distance_L2 = False
		Cmax = 1.0
		LOG.debug("Run vector magnitude loss (CFL) for velocity")
		# with ctx.profiler.sample('CFL condition loss'):
			# vel_x, vel_y, vel_z = state.velocity.centered(pad_lod=False, concat=False)
			# #https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition ; cell-size=1, dt=1, Cmax=1
			# tmp_loss = loss_func(tf.maximum((vel_x + vel_y + vel_z) - 1.0, 0.0), 0.0)
		# tmp_loss_scaled = tmp_loss * loss_scale
		# ctx.add_loss(tmp_loss_scaled, tmp_loss, loss_scale, 'velocity/CFL')
		
		if loss_before_cutoff:
			loss_fn = lambda mag: tf.maximum(loss_func(mag, 0.0) - Cmax, 0.0)
		else:
			loss_fn = lambda mag: loss_func(tf.maximum(mag - Cmax, 0.0), 0.0)
		
		# shape: NDHWC
		if ctx.target_weights is None:
			norm_axis = [0]
			if ctx.norm_spatial_dims: norm_axis += [1,2,3]
		elif ctx.norm_spatial_dims:
			raise NotImplementedError
		
		if not (ctx.allow_MS_losses and state.velocity.has_MS_output):
			if MS_residual_loss and state.velocity.has_MS_output:
				raise NotImplementedError("TODO: MS_residual_loss might not work if backprop does not use MS variables, wich is tied to ctx.allow_MS_losses.")
			with ctx.profiler.sample('CFL loss%s%s'%(" residual" if MS_residual_loss else " full", " centered" if state.velocity.is_centered else " staggered")):
				# if distance_L2:
					# mag = state.velocity.magnitude()
				# else:
					# vel_x, vel_y, vel_z = state.velocity.centered(pad_lod=False, concat=False)
					# mag = tf.abs(vel_x) + tf.abs(vel_y) + tf.abs(vel_z)
				# tmp_loss = [loss_fn(mag)]
				tmp_loss = []
				if distance_L2:
					if MS_residual_loss and state.velocity.has_MS_output:
						mag = state.velocity.magnitude_MS_residual(state.velocity._get_top_active_scale())
					else:
						mag = state.velocity.magnitude()
					tmp_loss.append(tf.reduce_mean(loss_fn(mag), axis=norm_axis))
				else:
					if state.velocity.is_centered:
						if MS_residual_loss and state.velocity.has_MS_output:
							vel_components = state.velocity.centered_MS_residual(state.velocity._get_top_active_scale(), pad_lod=False, concat=False)
						else:
							vel_components = state.velocity.centered(pad_lod=False, concat=False)
					elif state.velocity.is_staggered:
						if MS_residual_loss and state.velocity.has_MS_output:
							vel_components = state.velocity._staggered_MS_residual(state.velocity._get_top_active_scale())
						else:
							vel_components = state.velocity._staggered()
					else:
						raise ValueError("Unknown velocity type.")
					for vel_comp in vel_components:
						tmp_loss.append(tf.reduce_mean(loss_fn(tf.abs(vel_comp)), axis=norm_axis))
		else:
			with ctx.profiler.sample('CFL loss MS%s%s'%(" residual" if MS_residual_loss else " full", " centered" if state.velocity.is_centered else " staggered")):
				tmp_loss = []
				for scale in state.velocity.gen_current_trainable_MS_scales(): #gen_current_MS_scales():
					MS_weight = ctx.loss_schedules.velocity_CFLcond_MS_weighting(scale)
					if distance_L2:
						if MS_residual_loss:
							mag = state.velocity.magnitude_MS_residual(scale)
						else:
							mag = state.velocity.magnitude_MS(scale)
						tmp_loss.append(MS_weight * tf.reduce_mean(loss_fn(mag), axis=norm_axis))
					else:
						if state.velocity.is_centered:
							if MS_residual_loss:
								vel_components = state.velocity.centered_MS_residual(scale, pad_lod=False, concat=False)
							else:
								vel_components = state.velocity.centered_MS(scale, pad_lod=False, concat=False)
						elif state.velocity.is_staggered:
							if MS_residual_loss:
								vel_components = state.velocity._staggered_MS_residual(scale)
							else:
								vel_components = state.velocity._staggered_MS(scale)
						else:
							raise ValueError("Unknown velocity type.")
						for vel_comp in vel_components:
							tmp_loss.append(MS_weight * tf.reduce_mean(loss_fn(tf.abs(vel_comp)), axis=norm_axis))
		tmp_loss_scaled = [_ * loss_scale for _ in tmp_loss]
		if ctx.compute_loss_summary():
			ctx.add_loss(tmp_loss_scaled, tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss]), tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss_scaled]), loss_scale, 'velocity/CFL')
		else:
			ctx.add_loss(tmp_loss_scaled, None, None, loss_scale, 'velocity/CFL')
		return True
	return False

def loss_vel_MS_coherence(ctx, state, loss_func=None):
	#use highest vel output to constrain lower vel scales
	#loss_scale = ctx.CV(ctx.setup.training.velocity.density_warp_loss)
	loss_scale = ctx.loss_schedules.velocity_MS_coherence(ctx.iteration)
	top_as_label = True
	
	if ctx.LA(loss_scale):
		if loss_func is None: loss_func = ctx.get_loss_func("velocity/MS_coherence")
		# shape: NDHWC
		if ctx.target_weights is None:
			norm_axis = [0]
			if ctx.norm_spatial_dims: norm_axis += [1,2,3]
		elif ctx.norm_spatial_dims:
			raise NotImplementedError
		
		if not (ctx.allow_MS_losses and state.velocity.has_MS_output):
			LOG.warning("Loss 'velocity/MS_coherence' is active, but velocity has no MS or MS losses are not allowed.")
			return False
		else:
			LOG.debug("Run multi-scale coherence loss for velocity.")
			with ctx.profiler.sample('velocity MS coherence loss'):
				tmp_loss = []
				vel_MS_scales = list(reversed(list(state.velocity.gen_current_MS_scales())[:-1])) #fine to coarse, starting at 2nd highest
				if len(vel_MS_scales)<1:
					LOG.debug("Insuficient scales for velocity multi-scale coherence loss.")
					return False
				
				if state.velocity.is_centered:
					last_scale_vel = state.velocity.centered_MS(state.velocity._get_top_active_scale())
					
					for scale in vel_MS_scales: #fine to coarse, starting at 2nd highest
						MS_weight = ctx.loss_schedules.velocity_MS_coherence_MS_weighting(scale)
						# what is better, sampling always from the finest scale or only from the next finer? (regarding performance and gradient quality)
						last_scale_vel = state.velocity.resample_velocity(state.velocity.scale_renderer, last_scale_vel, shape=state.velocity.centered_shape_of_scale(scale), \
							is_staggered=state.velocity.is_staggered, scale_magnitude=True)
						if top_as_label:
							last_scale_vel = tf.stop_gradient(last_scale_vel)
						
						tmp_loss.append( MS_weight * tf.reduce_mean(loss_func(last_scale_vel, state.velocity.centered_MS(scale)), axis=norm_axis))
				
				elif state.velocity.is_staggered:
					last_scale_vel = state.velocity._staggered_MS(state.velocity._get_top_active_scale()) # (x,y,z)
					last_scale_vel = tuple(last_scale_vel)
					
					for scale in vel_MS_scales: #fine to coarse, starting at 2nd highest
						MS_weight = ctx.loss_schedules.velocity_MS_coherence_MS_weighting(scale)
						# what is better, sampling always from the finest scale or only from the next finer? (regarding performance and gradient quality)
						last_scale_vel = state.velocity.resample_velocity(state.velocity.scale_renderer, last_scale_vel, shape=state.velocity.centered_shape_of_scale(scale), \
							is_staggered=state.velocity.is_staggered, scale_magnitude=True)
						if top_as_label:
							last_scale_vel = tuple(tf.stop_gradient(_) for _ in last_scale_vel)
						
						tmp_loss.append( MS_weight * tf.reduce_mean(loss_func(last_scale_vel[0], state.velocity._staggered_MS(scale)[0]), axis=norm_axis)) #x
						tmp_loss.append( MS_weight * tf.reduce_mean(loss_func(last_scale_vel[1], state.velocity._staggered_MS(scale)[1]), axis=norm_axis)) #y
						tmp_loss.append( MS_weight * tf.reduce_mean(loss_func(last_scale_vel[2], state.velocity._staggered_MS(scale)[2]), axis=norm_axis)) #z
				else:
					raise ValueError("Unknown velocity type.")
					
				
		tmp_loss_scaled = [_*loss_scale for _ in tmp_loss]
		if ctx.compute_loss_summary():
			ctx.add_loss(tmp_loss_scaled, tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss]), tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss_scaled]), loss_scale, 'velocity/MS_coherence')
		else:
			ctx.add_loss(tmp_loss_scaled, None, None, loss_scale, 'velocity/MS_coherence')
		return True
	return False

def warp_vel_grads(opt_ctx, state, grads, order='FWD'):
	raise NotImplementedError
	if order.upper()=='FWD': #propagate velocity gradients to next state, simple forward warp
		v = state.veloctiy
		grads = VelocityGrid(v.centered_shape, x=grads[0], y=grads[1], z=grads[2], as_var=False, boundary=v.boundary, \
			warp_renderer=v.warp_renderer, scale_renderer=v.scale_renderer)
		return grads.warped(vel_grid=v, order=opt_ctx.warp_order, dt=opt_ctx.dt, clamp=opt_ctx.vel_warp_clamp)
	elif order.upper()=='BWD': #propagate velocity gradients to previous state, backprop through prev->warp
		var_list = state.velocity.var_list()
		with tf.GradientTape(watch_accessed_variables=False) as tape:
			tape.watch(var_list)
			v_warp = state.velocity.warped(order=opt_ctx.warp_order, dt=opt_ctx.dt, clamp=opt_ctx.vel_warp_clamp)
		return tape.gradient(v_warp, var_list, grads)
	else:
		raise ValueError

def optStep_velocity(opt_ctx, state, custom_vel_grads=None, optimize_inflow=False, apply_vel_grads=False):
	with opt_ctx.profiler.sample('optStep_velocity'):
		with opt_ctx.profiler.sample('get variables'):
			#vel_var_list = state.velocity.var_list()
			#vel_vars = state.velocity.get_variables()
			vel_vars = state.velocity.get_output_variables(include_MS=opt_ctx.allow_MS_losses, include_residual=True, only_trainable=True) #this generates velocity if using a network
			if optimize_inflow:
				dens_vars = state.density.get_variables()
				if 'inflow' in dens_vars: # inflow variable available
					vel_vars['inflow'] = dens_vars['inflow']
			if opt_ctx.setup.training.optimize_buoyancy:
				#vel_var_list.append(opt_ctx.buoyancy)
				vel_vars['buoyancy'] = opt_ctx.buoyancy
			#LOG.info("Velocity output variables: %s", list(vel_vars.keys()))
		with opt_ctx.profiler.sample('loss'), tf.GradientTape(watch_accessed_variables=False, persistent=opt_ctx.inspect_gradients) as vel_tape:
			vel_tape.watch(vel_vars)
			if opt_ctx.inspect_gradients:
				vel_inspect_vars = {}
				vel_inspect_vars.update(vel_vars)
			#	if "velocity_decoder" in vel_vars:
			#		vel_inspect_vars["velocity_c"] = state.velocity.centered()
			#		vel_inspect_vars["velocity_x"] = state.velocity._x
			#		vel_inspect_vars["velocity_y"] = state.velocity._y
			#		vel_inspect_vars["velocity_z"] = state.velocity._z
			#	else:
			#		vel_inspect_vars["velocity_x"] = vel_vars["velocity_x"]
			#		vel_inspect_vars["velocity_y"] = vel_vars["velocity_y"]
			#		vel_inspect_vars["velocity_z"] = vel_vars["velocity_z"]
				if "inflow" in vel_vars: vel_inspect_vars["inflow"] = vel_vars["inflow"]
				if "buoyancy" in vel_vars: vel_inspect_vars["buoyancy"] = vel_vars["buoyancy"]
				#vel_tape.watch(vel_inspect_vars)
			opt_ctx.set_gradient_tape(vel_tape)
		#	velocity_loss = 0
			active_velocity_loss = False
			LOG.debug("velocity losses")
			
			#warp losses
			active_velocity_loss = loss_vel_target_vol(opt_ctx, state) or active_velocity_loss
			active_velocity_loss = loss_vel_warp_dens(opt_ctx, state) or active_velocity_loss
			active_velocity_loss = loss_vel_warp_dens_proxy(opt_ctx, state) or active_velocity_loss
			active_velocity_loss = loss_vel_warp_dens_target(opt_ctx, state) or active_velocity_loss
			active_velocity_loss = loss_vel_warp_vel(opt_ctx, state) or active_velocity_loss
			
			#direct losses
			active_velocity_loss = loss_vel_smooth(opt_ctx, state) or active_velocity_loss
			active_velocity_loss = loss_vel_cossim(opt_ctx, state) or active_velocity_loss
			active_velocity_loss = loss_vel_divergence(opt_ctx, state) or active_velocity_loss
			active_velocity_loss = loss_vel_magnitude(opt_ctx, state) or active_velocity_loss
			active_velocity_loss = loss_vel_CFLcond(opt_ctx, state) or active_velocity_loss
			active_velocity_loss = loss_vel_MS_coherence(opt_ctx, state) or active_velocity_loss
			
		#	velocity_loss = opt_ctx.loss #opt_ctx.pop_loss()
		#END gradient tape
		if active_velocity_loss:
			with opt_ctx.profiler.sample('gradient'):
				if custom_vel_grads is not None:
					cvg_scale = opt_ctx.CV(opt_ctx.custom_vel_grads_weight)
				if opt_ctx.inspect_gradients:
					for loss_name in opt_ctx._losses:
						vel_grads = vel_tape.gradient(opt_ctx.get_loss(loss_name), vel_inspect_vars)
						#LOG.debug("vel grads: %s", [_ for _ in vel_grads])
						for k, g in vel_grads.items():
							if k.startswith('velocity_') and k[-2:] in ["_c", "_x", "_y", "_z"]:
								if g is not None:
									opt_ctx.inspect_gradients_func(opt_ctx=opt_ctx, gradients=g, name=loss_name+k[8:])
						del vel_grads
					
					if custom_vel_grads is not None and opt_ctx.LA(cvg_scale):
						has_valid_vel_grads = False
						for k, g in custom_vel_grads.items():
							if k.startswith('velocity_') and k[-2:] in ["_c", "_x", "_y", "_z"]:
								if g is not None:
									has_valid_vel_grads = True
									opt_ctx.inspect_gradients_func(opt_ctx=opt_ctx, gradients=g, name="velocity/custom_grad"+k[8:])
								else:
									LOG.debug("Custom velocity gradient of frame %d for '%s' is None.", state.frame, k)
						if not has_valid_vel_grads:
							LOG.warning("All custom velocity gradients of frame %d are None.", state.frame)
					opt_ctx.inspect_gradients_images = {}
				
					vel_grads = vel_tape.gradient(opt_ctx.get_losses(), vel_inspect_vars)
				
				#if opt_ctx.inspect_gradients:
					has_valid_vel_grads = False
					for k, g in vel_grads.items():
						if k.startswith('velocity_') and k[-2:] in ["_c", "_x", "_y", "_z"]:
							if g is not None:
								has_valid_vel_grads = True
								opt_ctx.inspect_gradients_func(opt_ctx=opt_ctx, gradients=g, name="velocity/_local"+k[8:])
							else:
								LOG.debug("Total gradient of frame %d for '%s' is None.", state.frame, k)
					if not has_valid_vel_grads:
						LOG.warning("All local velocity gradients of frame %d are None.", state.frame)
					
					if vel_grads.get('buoyancy') is not None:
						opt_ctx.inspect_gradients_func(opt_ctx=opt_ctx, gradients=vel_grads['buoyancy'][0], name="velocity/buoyancy_x")
						opt_ctx.inspect_gradients_func(opt_ctx=opt_ctx, gradients=vel_grads['buoyancy'][1], name="velocity/buoyancy_y")
						opt_ctx.inspect_gradients_func(opt_ctx=opt_ctx, gradients=vel_grads['buoyancy'][2], name="velocity/buoyancy_z")
					if vel_grads.get('inflow') is not None:
						opt_ctx.inspect_gradients_func(opt_ctx=opt_ctx, gradients=vel_grads['inflow'], name="velocity/inflow")
					
					if custom_vel_grads is not None and opt_ctx.allow_MS_losses:
						# back-warping give gradients w.r.t the the output variables, not the MS-output variables, so map to the highest MS-output.
						custom_vel_grads_inspect = state.velocity.map_gradient_output_to_MS(custom_vel_grads)
					else:
						custom_vel_grads_inspect = custom_vel_grads
					
					has_valid_vel_grads = False
					for k in vel_grads: #depth 1 sufficient for now...
						if custom_vel_grads is not None and opt_ctx.LA(cvg_scale):
							if k in custom_vel_grads_inspect: 
								if vel_grads[k] is not None and custom_vel_grads_inspect[k] is not None:
									LOG.info("Update velocity gradient '%s' with custom gradient with scale %f.", k, cvg_scale)
									vel_grads[k] += custom_vel_grads_inspect[k]*cvg_scale
								elif vel_grads[k] is None and custom_vel_grads_inspect[k] is not None:
									LOG.info("Set velocity gradient '%s' to custom gradient with scale %f.", k, cvg_scale)
									vel_grads[k] = custom_vel_grads_inspect[k]*cvg_scale
								else:
									LOG.debug("Custom velocity gradient '%s' of frame %d is None", k, state.frame)
						if vel_grads[k] is not None:
							has_valid_vel_grads = True
						#	opt_ctx.inspect_gradients_func(opt_ctx=opt_ctx, gradients=vel_grads[k], name="velocity/_total"+k[8:])
					if not has_valid_vel_grads:
						LOG.error("All velocity gradients of frame %d are None.", state.frame)
				
				LOG.debug('Compute and apply velocity gradients')
				vel_grads = vel_tape.gradient(opt_ctx.get_losses(), vel_vars)
					
				curr_vel_grads = copy_nested_structure(vel_grads)
				
				if custom_vel_grads is not None and opt_ctx.LA(cvg_scale):
					
					if opt_ctx.allow_MS_losses:
						# back-warping give gradients w.r.t the the output variables, not the MS-output variables, so map to the highest MS-output.
						custom_vel_grads = state.velocity.map_gradient_output_to_MS(custom_vel_grads)
					
					#vel_grads = [v + c*cvg_scale for v,c in zip(vel_grads[:len(custom_vel_grads)], custom_vel_grads)] + vel_grads[len(custom_vel_grads):]
					for k in vel_grads: #depth 1 sufficient for now...
						if k in custom_vel_grads: 
							if custom_vel_grads[k] is not None:
								LOG.debug("Update velocity gradient '%s' with custom gradient.", k)
								if vel_grads[k] is None:
									#LOG.warning("This should crash now...")
									vel_grads[k] = custom_vel_grads[k]*cvg_scale
								else:
									vel_grads[k] += custom_vel_grads[k]*cvg_scale
							#backprop_accumulate hadles this already
						#	elif vel_grads[k] is None:
						#		vel_grads[k] = 0.0
							#else:
							#	LOG.debug("Custom velocity gradient '%s' of frame %d is None", k, state.frame)
					
					for k in custom_vel_grads:
						if k not in vel_grads:
							LOG.warning("Custom velocity gradient '%s' can't be mapped and will be ignored.", k)
					
				if apply_vel_grads:
					if vel_grads.get('inflow') is not None:
						inflow_grads_vars = ((vel_grads['inflow'], vel_vars['inflow']),)
						opt_ctx.density_optimizer.apply_gradients(inflow_grads_vars)
						del vel_grads['inflow']
						del vel_vars['inflow']
				#	vel_grads_vars = zip(nest.flatten(vel_grads), nest.flatten(vel_vars))
				#	opt_ctx.velocity_optimizer.apply_gradients(vel_grads_vars)
				state.velocity.set_output_gradients_for_backprop_accumulate(vel_grads, include_MS=opt_ctx.allow_MS_losses, include_residual=True, only_trainable=True)
				
				if apply_vel_grads:
					#state.velocity.backprop_accumulate(vel_grads, include_MS=opt_ctx.allow_MS_losses, include_residual=True, only_trainable=True)
					state.velocity._compute_input_grads()
		else:
			#curr_vel_grads = nest.map_structure(lambda v: tf.constant(0, dtype=tf.float32), vel_vars) #[tf.constant(0, dtype=tf.float32) for _ in vel_var_list]
			curr_vel_grads = nest.map_structure(lambda v: tf.zeros_like(v), vel_vars)
			state.velocity.set_output_gradients_for_backprop_accumulate(curr_vel_grads, include_MS=opt_ctx.allow_MS_losses, include_residual=True, only_trainable=True)
		del vel_tape
	return active_velocity_loss, curr_vel_grads

def optStep_state(opt_ctx, state, disc_ctx=None, disc_samples_list=None, custom_dens_grads=None, custom_vel_grads=None, apply_dens_grads=False, apply_vel_grads=False):
	LOG.debug("Optimization step for state: frame %d", state.frame)
	with  opt_ctx.profiler.sample('optStep_state'):
		prev_losses = opt_ctx._losses
		opt_ctx.pop_losses()
		opt_ctx.frame = state.frame
		
		# density_proxy optimization here, the velocity might need the gradients
		# TODO?
		
		# velocity optimization, uses the density, so the density needs the gradients
		if (state.next is not None): #the last frame has no target for a velocity
			vel_active, vel_grads = optStep_velocity(opt_ctx, state, custom_vel_grads=custom_vel_grads, apply_vel_grads=apply_vel_grads)
			#velocity_loss = opt_ctx.pop_losses()
			opt_ctx.pop_losses()
		else:
			vel_active = False
			vel_grads = None
		
		dens_active, dens_grads = optStep_density(opt_ctx, state, use_vel=True, disc_ctx=disc_ctx, disc_samples_list=disc_samples_list, custom_dens_grads=custom_dens_grads, apply_dens_grads=apply_dens_grads)
		
		opt_ctx.pop_losses()
		opt_ctx._losses = prev_losses
	
	#return dens_grads, vel_grads

def optStep_sequence(opt_ctx, sequence, disc_ctx=None, disc_samples_list=None, order='FWD'):
	LOG.debug("Optimization step for sequence with %d states", len(sequence))
	total_losses = []
	loss_summaries = {}
	if order.upper()=='FWD':
		optim_order = list(range(len(sequence)))
	elif order.upper()=='BWD':
		optim_order = list(reversed(range(len(sequence))))
	elif order.upper()=='RAND':
		optim_order = np.random.permutation(len(sequence)).tolist()
	else:
		raise ValueError
	
	#warnings.warn("Optimizing only first frame of sequence.")
	for i in optim_order: #[0]: # 
		state = sequence[i]
		
		with opt_ctx.profiler.sample("Frame"):
			optStep_state(opt_ctx, state, disc_ctx, disc_samples_list)
			loss_summaries[state.frame]=opt_ctx.pop_loss_summary()
		
	return loss_summaries

### Discriminator
class DiscriminatorContext:
	CHECK_INPUT_NONE = 0x0
	CHECK_INPUT_CLAMP = 0x1
	CHECK_INPUT_DUMP  = 0x2
	CHECK_INPUT_CHECK = 0x10
	CHECK_INPUT_CHECK_NOTFINITE = 0x20
	CHECK_INPUT_SIZE = 0x40
	CHECK_INPUT_RAISE = 0x100
	CHECK_INPUT_RAISE_NOTFINITE = 0x200
	def __init__(self, ctx, model, rendering_context, real_data, loss_type, optimizer, learning_rate, crop_size=None, scale_range=[1,1], rotation_mode="NONE", check_input=0, check_info_path=None, \
			resource_device=None, scale_samples_to_input_resolution=False, \
			use_temporal_input=False, temporal_input_steps=None, cam_x_range=[-30,30]):
		assert isinstance(ctx, OptimizationContext)
		assert isinstance(rendering_context, RenderingContext)
		assert isinstance(learning_rate, tf.Variable)
		#raise NotImplementedError()
		self.model = model
		self.real_data = real_data
		self.opt_ctx = ctx
		self.render_ctx = rendering_context
		self.cam_x_range = cam_x_range
		self.history = None
		self._train_base = True
		self._train = True
		if ctx.setup.training.discriminator.history.samples>0:
			self.history = HistoryBuffer(ctx.setup.training.discriminator.history.size)
		#self._label_fake = 0.0
		self._label_real = ctx.setup.training.discriminator.target_label
		self._conditional_hull = ctx.setup.training.discriminator.conditional_hull
		self.optimizer = optimizer
		self.lr = learning_rate
		self._last_it = self.opt_ctx.iteration
		self.center_crop = True # center crop on density center of mass + random offset
		self.crop_size = crop_size
		self.scale_range = scale_range
		self.rotation_mode = rotation_mode
		self.dump_path = None
		self._check_input = check_input #bit-mask
		self._check_info_path = check_info_path
		self._input_range = [0.0, 10.0]
		self.resource_device = resource_device
		loss_types = ["SGAN", "RpSGAN", "RpLSGAN", "RaSGAN", "RaLSGAN"]
		if loss_type not in loss_types:
			raise ValueError("Unknown Discriminator loss_type {}. Available losses: {}".format(loss_type, loss_types))
		self.loss_type = loss_type #SGAN, RpSGAN, RaSGAN, RcSGAN
		
		self.scale_to_input_res = scale_samples_to_input_resolution
		#self.input_res = model.input_shape[-3:-1] #input_base_resolution
		
		self._temporal_input = use_temporal_input
		self._temporal_input_steps = temporal_input_steps
	
	@property
	def train(self):
		return self._train and self._train_base
	
	@train.setter
	def train(self, train):
		self._train_base = train
	
	def start_iteration(self, iteration, force=False, compute_loss_summary=False):
		self.opt_ctx.start_iteration(iteration, force, compute_loss_summary)
		if iteration==self._last_it and not force:
			return
		
		curr_lr = self.opt_ctx.loss_schedules.discriminator_lr(self.opt_ctx.iteration - self.opt_ctx.setup.training.discriminator.start_delay)
		self._train = self.opt_ctx.LA(curr_lr)
		self.lr.assign(curr_lr)
		
		if self.opt_ctx.record_summary:
			summary_names = self.opt_ctx.make_summary_names('discriminator/learning_rate')
			self.opt_ctx._tf_summary.scalar(summary_names[0], self.lr.numpy(), step=self.opt_ctx.iteration)
		self._last_it = self.opt_ctx.iteration
	
	@property
	def input_res(self):
		return self.model.input_shape[-3:-1]
	
	@property
	def record_history(self):
		return self.train and self.history is not None
	
	def var_list(self):
		return self.model.trainable_variables
		
	def real_labels(self, logits):
		return tf.ones_like(logits)*self._label_real
	def fake_labels(self, logits):
		return tf.zeros_like(logits)
	
	def _scale_range(self, shape, target_shape, max_scale_range):
		scale = np.amax([t/i for t,i in zip(target_shape, shape)])
		if scale>max_scale_range[1]: #scale<max_scale_range[0] or 
			raise ValueError("Scaling impossible with shape {}, target shape {}, resulting scale {} and scale_range {}".format(shape, target_shape, scale, max_scale_range))
		return [max(scale, max_scale_range[0]), max_scale_range[1]]
	
	def dump_samples(self, sample_batch, is_real):
		if self.dump_path is not None:
			with self.opt_ctx.profiler.sample("dump disc samples"):
				if is_real: name = 'real_s{:06d}_i{:06d}'
				else: name = 'fake_s{:06d}_i{:06d}'
				if self._temporal_input:
					for i, batch in enumerate(tf.split(sample_batch, 3, axis=-1)):
						self.render_ctx.dens_renderer.write_images([batch], [name + '_t%02d'%i], base_path=self.dump_path, use_batch_id=True, frame_id=self.opt_ctx.iteration, format='EXR')
				else:
					self.render_ctx.dens_renderer.write_images([sample_batch], [name], base_path=self.dump_path, use_batch_id=True, frame_id=self.opt_ctx.iteration, format='EXR')
	
	def check_input(self, input, name="input"):
		# debug discriminator failure
		dump_input = False
		nan_input = False
		if (self._check_input & (self.CHECK_INPUT_CHECK_NOTFINITE | self.CHECK_INPUT_CHECK))>0:
			nan_in = tf.reduce_any(tf.is_nan(input), axis=[1,2,3])
			if tf.reduce_any(nan_in).numpy():
				LOG.warning("NaN in samples %s of discriminator input '%s' in iteration %d.", np.where(nan_in.numpy())[0], name, self.opt_ctx.iteration)
				dump_input = True
				nan_input = True
		if (self._check_input & self.CHECK_INPUT_SIZE)>0:
			input_shape = shape_list(input)
			if tf.reduce_any(tf.not_equal(input.get_shape()[-3:], self.model.input_shape[-3:])).numpy():
				LOG.warning("shape %s of input '%s' does not match discriminator input shape %s", shape_list(input), name, self.model.input_shape)
		if (self._check_input & self.CHECK_INPUT_CHECK)>0:
			if tf.reduce_any(tf.less(input, self._input_range[0])).numpy():
				in_min = tf.reduce_min(input).numpy()
				LOG.warning("Minimum value %f of discriminator input '%s' exceeds minimum %f in iteration %d.", in_min, name, self._input_range[0], self.opt_ctx.iteration)
				dump_input = True
			if tf.reduce_any(tf.greater(input, self._input_range[1])).numpy():
				in_max = tf.reduce_max(input).numpy()
				LOG.warning("Maximum value %f of discriminator input '%s' exceeds maximum %f in iteration %d.", in_max, name, self._input_range[1], self.opt_ctx.iteration)
				dump_input = True
		if dump_input and (self._check_input & self.CHECK_INPUT_DUMP)>0 and self._check_info_path is not None:
			name = "{}_{:06d}".format(name, self.opt_ctx.iteration) + "_{:04d}"
			self.render_ctx.dens_renderer.write_images([input], [name], base_path=self._check_info_path, use_batch_id=True, format='EXR')
		if (dump_input and (self._check_input & self.CHECK_INPUT_RAISE)>0) or (nan_input and (self._check_input & self.CHECK_INPUT_RAISE_NOTFINITE)>0):
			raise ValueError("Discriminator input {} error.".format(name))
			
		
		if (self._check_input & self.CHECK_INPUT_CLAMP)>0:
			return tf.minimum(tf.maximum(input, self._input_range[0]), self._input_range[1]) #also makes nan and -inf to min and inf to max (TF 1.12 on GPU)
		else:
			return input
	
	def check_output(self, output, loss, input, name='output'):
		if (self._check_input & (self.CHECK_INPUT_CHECK_NOTFINITE | self.CHECK_INPUT_CHECK))>0:
			dump_input = False
			if not tf.reduce_all(tf.is_finite(output)):
				LOG.warning("Discriminator output '%s' in iteration %d is not finite: %s", name, self.opt_ctx.iteration, output.numpy())
				dump_input = True
			if not tf.reduce_all(tf.is_finite(loss)):
				LOG.warning("Discriminator loss '%s' in iteration %d is not finite: %s", name, self.opt_ctx.iteration, loss.numpy())
				dump_input = True
			if dump_input and (self._check_input & self.CHECK_INPUT_DUMP)>0 and self._check_info_path is not None:
				file_name = "{}_{:06d}".format(name, self.opt_ctx.iteration) + "_{:04d}"
				self.render_ctx.dens_renderer.write_images([input], [file_name], base_path=self._check_info_path, use_batch_id=True, format='EXR')
			if dump_input and (self._check_input & (self.CHECK_INPUT_RAISE_NOTFINITE |self.CHECK_INPUT_RAISE))>0:
				raise ValueError("Discriminator output {} error.".format(name))
	
	def _scale_samples_to_input_res(self, *samples_raw):
		if self.scale_to_input_res:
			#LOG.info("Scaling disc input before augmentation from %s to %s.", [shape_list(_) for _ in samples_raw], self.input_res)
			return [tf.image.resize_bilinear(sample_raw, self.input_res) for sample_raw in samples_raw]
		else:
			return samples_raw
	
	def _pad_samples_to_input_res(self, *samples_raw):
		return [tf_pad_to_shape(sample_raw, [-1]+ list(self.input_res) +[-1], allow_larger_dims=True) for sample_raw in samples_raw]
	
	def image_center_of_mass(self, img):
		sample_mean_y = tf.reduce_mean(sample, axis=[-3,-1]) #NW
	#	LOG.info("mean y shape: %s", sample_mean_y.get_shape().as_list())
		coords_x = tf.reshape(tf.range(0, scale_shape[-1], 1,dtype=tf.float32), (1,scale_shape[-1])) #1W
		center_x = tf.reduce_sum(coords_x*sample_mean_y, axis=-1)/tf.reduce_sum(sample_mean_y, axis=-1) #N
		sample_mean_x = tf.reduce_mean(sample, axis=[-2,-1]) #NH
		coords_y = tf.reshape(tf.range(0, scale_shape[-2], 1,dtype=tf.float32), (1,scale_shape[-2])) #1H
		center_y = tf.reduce_sum(coords_y*sample_mean_x, axis=-1)/tf.reduce_sum(sample_mean_x, axis=-1) #N
		return Float2(center_x, center_y)
	
	#def augment_spatial(self, *samples, scale_range=(1.,1.), rotation_mode="90", out_shape="SAME"):
	#	# use renderer to do scaling, rotation, crop and CoM shift in one step
	#	# renderer only support 1,2,4 channels, we have 3 or 9. mipmapping for 2D is also not working.
	#	out_samples = []
	#	for sample in samples:
	#		sample_shape = GridShape(
	#		with self.opt_ctx.profiler.sample('augment_spatial'):
	#			if out_shape=="INPUT":
	#				_out_shape = [1] + list(self.input_res)
	#			elif out_shape=="SAME":
	#				_out_shape = 
	
	def _prepare_samples(self, *samples_raw, scale_range=(1.,1.), rotation_mode="90", crop_shape="INPUT"):
		""" Data augmentation for discriminator input.
		
		1. scale image resolution with random scaling factor from scale_range using bilinear interpolation.
		2. appy random rotation
		3. pad the image to be at have least size crop_shape
		4. apply random crop, focusing on the center of mass, if possible
		
		"""
		samples = []
		with self.opt_ctx.profiler.sample('prepare_crop_flip'):
			for sample_raw in samples_raw:
				sample_shape = shape_list(sample_raw)
				#raw shape & target/crop shape -> scale range
				#check allowed scale range
				#now allow any scale range and pad later if necessary
				#scale_range = self._scale_range(sample_shape[-3:-1], crop_shape, scale_range)
				if not (scale_range==None or scale_range==(1.,1.)):
					scale = np.random.uniform(*scale_range)
					if scale!=1.:
						scale_shape = [int(np.ceil(_*scale)) for _ in sample_shape[-3:-1]]
						sample = tf.image.resize_bilinear(sample_raw, scale_shape)
					else:
						sample = sample_raw
				
				#random 90deg rotation and mirroring
				if rotation_mode==90 or rotation_mode=="90":
					r = np.random.randint(2, size=3)
					if r[0]==1:
						sample = tf.transpose(sample, (0,2,1,3)) #swap x and y of NHWC tensor
					flip_axes = []
					if r[1]==1:
						flip_axes.append(-2) #flip x
					if r[2]==1:
						flip_axes.append(-3) #flip y
					if flip_axes:
						sample = tf.reverse(sample, flip_axes)
				elif rotation_mode.upper()=="CONTINUOUS":
					raise NotImplementedError
				#	angle = np.random.uniform(0,360)
				#	sample_shape = shape_list(sample)
				#	sample_shape = [sample_shape[0], 1] + sample_shape[1:]
				#	sample = tf.reshape(sample, sample_shape)
				#	t_from = GridTransform(sample_shape[-4:-1], rotation_deg=(0.,0.,angle), center=True)
				#	t_to = GridTransform(sample_shape[-4:-1], center=True)
				#	
				#	sample = tf.squeeze(self.render_ctx.dens_renderer._sample_transform(sample, [t_from], [t_to], fix_scale_center=True), [1,2])
				elif not (rotation_mode is None or rotation_mode.upper()=="NONE"):
					raise ValueError("Unknown rotation_mode %s"%rotation_mode)
				
				if crop_shape is not None:
					if crop_shape=="INPUT":
						crop_shape = self.input_res
					sample_shape = shape_list(sample) #tf.shape(sample).numpy()
					
					if np.any(np.less(sample_shape[-3:-1], crop_shape)):
						sample = tf_pad_to_shape(sample, [-1]+ list(crop_shape) +[-1], allow_larger_dims=True) #, mode="REFLECT")
						sample_shape = shape_list(sample)
					
					# don't crop if shape already matches
					if not np.all(np.equal(sample_shape[-3:-1], crop_shape)):
						# -> find a "center of mass" and crop around that, with some random offset
						# TODO what if sample is empty/all 0?
						crop_eps = 1e-4
						if self.center_crop and tf.reduce_mean(sample).numpy()>crop_eps:
						#	LOG.info("scale shape: %s", scale_shape)
							sample_mean_y = tf.reduce_mean(sample, axis=[-3,-1]) #NW
						#	LOG.info("mean y shape: %s", sample_mean_y.get_shape().as_list())
							coords_x = tf.reshape(tf.range(0, sample_shape[-2], 1,dtype=tf.float32), (1,sample_shape[-2])) #1W
							center_x = tf.reduce_sum(coords_x*sample_mean_y, axis=-1)/tf.reduce_sum(sample_mean_y, axis=-1) #N
							sample_mean_x = tf.reduce_mean(sample, axis=[-2,-1]) #NH
							coords_y = tf.reshape(tf.range(0, sample_shape[-3], 1,dtype=tf.float32), (1,sample_shape[-3])) #1H
							center_y = tf.reduce_sum(coords_y*sample_mean_x, axis=-1)/tf.reduce_sum(sample_mean_x, axis=-1) #N
							
							# get offset s.t. crop is in bounds, centered on center of mass (+ noise)
							crop_shape = tf.constant(crop_shape, dtype=tf.int32) #HW
							offset_bounds = sample.get_shape()[-3:-1] - crop_shape #2
							offset = tf.stack([center_y, center_x], axis=-1) + tf.random.uniform([sample.get_shape()[0], 2], -20,21, dtype=tf.float32) - tf.cast(crop_shape/2, dtype=tf.float32) #N2
							offset = tf.clip_by_value(tf.cast(offset, dtype=tf.int32), [0,0], offset_bounds)
							sample = tf.stack([tf.image.crop_to_bounding_box(s, *o, *crop_shape) for s, o in zip(sample, offset)], axis=0)
						else:
							sample = tf.random_crop(sample, [sample_shape[0]]+list(crop_shape)+[sample_shape[-1]])
				
				samples.append(sample)
		return samples if len(samples)>1 else samples[0]
	
	def augment_intensity(self, samples, scale_range, gamma_range):
		with self.opt_ctx.profiler.sample('augment_intensity'):
			scale_shape = (shape_list(samples)[0],1,1,1)
			scale = tf.random.uniform(scale_shape, *scale_range, dtype=samples.dtype)
			gamma = tf.random.uniform(scale_shape, *gamma_range, dtype=samples.dtype)
			scale = [scale,scale,scale]
			gamma = [gamma,gamma,gamma]
			if self._conditional_hull: #do not scale the intensity of the hull, the disc should be invariant to intensities
				scale.append(tf.ones(scale_shape, dtype=samples.dtype))
				gamma.append(tf.ones(scale_shape, dtype=samples.dtype))
			if self._temporal_input:
				scale *=3
				gamma *=3
			samples = tf.pow(tf.multiply(samples, tf.concat(scale, axis=-1)), tf.concat(gamma, axis=-1))
		return samples
	
	def real_samples(self, spatial_augment=True, intensity_augment=True):
		with self.opt_ctx.profiler.sample('real_samples'):
			with self.opt_ctx.profiler.sample('get data'):
				samples = self.real_data.get_next()
			samples = self._scale_samples_to_input_res(samples)[0]
			if spatial_augment:
				samples = self._prepare_samples(*tf.split(samples, samples.get_shape()[0], axis=0), \
					crop_shape="INPUT" if self.scale_to_input_res else self.crop_size, scale_range=self.scale_range, rotation_mode=self.rotation_mode)
			else:
				samples = self._pad_samples_to_input_res(*tf.split(samples, samples.get_shape()[0], axis=0))
			if intensity_augment:
				samples = self.augment_intensity(tf.concat(samples, axis=0), self.opt_ctx.setup.data.discriminator.scale_real, self.opt_ctx.setup.data.discriminator.gamma_real)
			#	scale_shape = (self.opt_ctx.setup.training.discriminator.num_real,1,1,1)
			#	scale = tf.random.uniform(scale_shape, *self.opt_ctx.setup.data.discriminator.scale_real, dtype=tf.float32)
			#	gamma = tf.random.uniform(scale_shape, *self.opt_ctx.setup.data.discriminator.gamma_real, dtype=tf.float32)
			#	scale = [scale,scale,scale]
			#	gamma = [gamma,gamma,gamma]
			#	if self._conditional_hull: #do not scale the intensity of the hull, the disc should be invariant to intensities
			#		scale.append(tf.ones(scale_shape, dtype=tf.float32))
			#		gamma.append(tf.ones(scale_shape, dtype=tf.float32))
			#	if self._temporal_input:
			#		scale *=3
			#		gamma *=3
			#	samples = tf.pow(tf.multiply(samples, tf.concat(scale, axis=-1)), tf.concat(gamma, axis=-1))
		return samples
	
	def _render_fake_samples(self, state, name="render_fake_samples"):
		dens_transform = state.get_density_transform()
		#LOG.debug("Render fake samples '%s' with jitter %s", name, [_.jitter for _ in self.render_ctx.cameras])
		imgs_fake = self.render_ctx.dens_renderer.render_density(dens_transform, self.render_ctx.lights, self.render_ctx.cameras, monochrome=self.render_ctx.monochrome, custom_ops=self.opt_ctx.render_ops) # [NHWC]*V
		#imgs_fake = [self.check_input(_, name="%s_render_%04d"%(name, i)) for _, i in zip(imgs_fake, range(len(imgs_fake)))]
		if self._conditional_hull:
			imgs_hull = self.render_ctx.dens_renderer.project_hull(state.hull, dens_transform, self.render_ctx.cameras) #NDWC
			imgs_hull = tf.split(imgs_hull, len(self.render_ctx.cameras), axis=0) #[1DWC]*N
			imgs_fake = [tf.concat([f,h], axis=-1) for f,h in zip(imgs_fake, imgs_hull)]
		return imgs_fake
		
	def fake_samples(self, state, history_samples=True, concat=True, spatial_augment=True, intensity_augment=False, name="fake_samples"):
		with self.opt_ctx.profiler.sample('fake_samples'):
			#prepare fake samples
			in_fake = []
			if state is not None:
				self.render_ctx.randomize_camera_rotation(x_range=self.cam_x_range, z_range=[0,0])
			
			#		
				
				with self.opt_ctx.profiler.sample('render fake'):
					# TODO temporal disc input:
					if self._temporal_input:
						cur_idx = 1
						tmp_fake = [None]*3
						with NO_CONTEXT() if self.opt_ctx.tape is None else self.opt_ctx.tape.stop_recording(): # don't need gradients for cmp images (and probably don't have memory for it...)
							#TODO random step. consider data/reconstuction step (i.e. frame skipping) vs dataset steps?
							#	for testing, use fixed prev/next step
							#TODO border handling. use border image in disc triplet? also randomly for other frames/states?
							#	curr: needs at least 3 frame sequence or will break
							#	or black inputs. TODO needs black prev/next in real data.
							# use fixed random camera transform from current disc input
							if state.prev is None:
								tmp_fake[1] = self._render_fake_samples(state.next, name=name + "_next")
								tmp_fake[2] = self._render_fake_samples(state.next.next, name=name + "_nnext")
								cur_idx = 0
							elif state.next is None:
								tmp_fake[0] = self._render_fake_samples(state.prev.prev, name=name + "_pprev")
								tmp_fake[1] = self._render_fake_samples(state.prev, name=name + "_prev")
								cur_idx = 2
							else:
								tmp_fake[0] = self._render_fake_samples(state.prev, name=name + "_prev")
								tmp_fake[2] = self._render_fake_samples(state.next, name=name + "_next")
								cur_idx = 1
							LOG.debug("Render temporal fake disc input '%s', current idx %d. tape available: %s", name, cur_idx, self.opt_ctx.tape is not None)
					
					imgs_fake = self._render_fake_samples(state, name=name)
					
					if self._temporal_input:
						tmp_fake[cur_idx] = imgs_fake
						imgs_fake = [tf.concat(_, axis=-1) for _ in zip(*tmp_fake)]
					in_fake += imgs_fake
			#		dens_transform = state.get_density_transform()
			#		#LOG.debug("Render fake samples '%s' with jitter %s", name, [_.jitter for _ in self.render_ctx.cameras])
			#		imgs_fake = self.render_ctx.dens_renderer.render_density(dens_transform, self.render_ctx.lights, self.render_ctx.cameras, monochrome=self.render_ctx.monochrome, custom_ops=self.opt_ctx.render_ops) #[1DWC]*N
			#		imgs_fake = [self.check_input(_, name="%s_render_%04d"%(name, i)) for _, i in zip(imgs_fake, range(len(imgs_fake)))]
			#		if self._conditional_hull:
			#			imgs_hull = self.render_ctx.dens_renderer.project_hull(state.hull, dens_transform, self.render_ctx.cameras) #NDWC
			#			imgs_hull = tf.split(imgs_hull, len(self.render_ctx.cameras), axis=0) #[1DWC]*N
			#			in_fake += [tf.concat([f,h], axis=-1) for f,h in zip(imgs_fake, imgs_hull)]
			#		else:
			#			in_fake += imgs_fake
				#if disc_dump_samples: self.render_ctx.dens_renderer.write_images([tf.concat(disc_in_fake, axis=0)], ['zz_disc_{1:04d}_fake_render_cam{0}'], base_path=setup.paths.data, use_batch_id=True, frame_id=it, format='PNG')
			with NO_CONTEXT() if self.opt_ctx.tape is None else self.opt_ctx.tape.stop_recording():
				if self.record_history:
					if history_samples:
						in_history = self.history.get_samples(self.opt_ctx.setup.training.discriminator.history.samples, replace=False, allow_partial=True)
					with tf.device(self.resource_device): #copy to resource device
						hist_samples = [tf.identity(_) for batch in in_fake for _ in tf.split(batch, shape_list(batch)[0], axis=0)]
						self.history.push_samples(hist_samples, self.opt_ctx.setup.training.discriminator.history.keep_chance, 'RAND')
					if history_samples:
						in_fake += in_history
					#if disc_dump_samples and len(disc_in_history)>0: self.render_ctx.dens_renderer.write_images([tf.concat(disc_in_history, axis=0)], ['zz_disc_{1:04d}_fake_history{0}'], base_path=setup.paths.data, use_batch_id=True, frame_id=it, format='PNG')
					if self.opt_ctx.record_summary:
						summary_names = self.opt_ctx.make_summary_names('discriminator/history_size')
						self.opt_ctx._tf_summary.scalar(summary_names[0], len(self.history), step=self.opt_ctx.iteration)
			
			in_fake = self._scale_samples_to_input_res(*in_fake)
			if spatial_augment:
				in_fake = self._prepare_samples(*in_fake, crop_shape="INPUT" if self.scale_to_input_res else self.crop_size, scale_range=self.scale_range, rotation_mode=self.rotation_mode)
			else:
				in_fake = self._pad_samples_to_input_res(*in_fake)
			
			if intensity_augment:
				raise NotImplementedError
		#		samples = self.augment_intensity(samples, self.opt_ctx.setup.data.discriminator.scale_fake, self.opt_ctx.setup.data.discriminator.gamma_fake)
		return tf.concat(in_fake, axis=0) if concat else in_fake
	
	def postprocess_loss(self, loss, out, name="loss"):
		self.opt_ctx.add_loss(tf.math.reduce_mean(loss), loss_name='discriminator/'+name)
		if self.opt_ctx.setup.training.discriminator.use_fc:
			scores = tf.math.sigmoid(out)
		else:
			scores = tf.reduce_mean(tf.math.sigmoid(out), axis=[1,2,3])
		return scores
	
	def loss(self, input, flip_target=False, training=True):
		'''
			Relativistic discriminator: https://github.com/AlexiaJM/relativistic-f-divergences
		input: (real, fake), (input) for SGAN
		flip_target: 
		'''
		name = "fake_loss" if flip_target else "real_loss"
		if self.loss_type=="SGAN":
			if flip_target:
				out = self.model(self.check_input(input[0], "fake"), training=training)
				loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out, labels=self.fake_labels(out)))
				#return loss_disc_fake(self, in_fake, training)
			else:
				out = self.model(self.check_input(input[0], "real"), training=training)
				loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out, labels=self.real_labels(out)))
				#return loss_disc_real(self, in_real, training)
			
			if self.opt_ctx.setup.training.discriminator.use_fc:
				scores = tf.math.sigmoid(out)
			else:
				scores = tf.reduce_mean(tf.math.sigmoid(out), axis=[1,2,3])
		else:
			out_real = self.model(self.check_input(input[0], "real"), training=training)
			out_fake = self.model(self.check_input(input[1], "fake"), training=training)
			if self.loss_type in ["RpSGAN", "RpLSGAN"]:
				#relativistic paired
				#batch and (disc out) resolution of fake and real must match here
				if flip_target:
					out_rel = out_fake-out_real
				else:
					out_rel = out_real-out_fake
				
				if self.loss_type=="RpSGAN":
					loss = 2*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out_rel, labels=self.real_labels(out_rel)))
					if self.opt_ctx.setup.training.discriminator.use_fc:
						scores = tf.math.sigmoid(out)
					else:
						scores = tf.reduce_mean(tf.math.sigmoid(out), axis=[1,2,3])
				elif self.loss_type=="RpLSGAN":
					loss = 2*tf.reduce_mean(tf.math.squared_difference(out_rel, self._label_real))
					if self.opt_ctx.setup.training.discriminator.use_fc:
						scores = out_rel
					else:
						scores = tf.reduce_mean(out_rel, axis=[1,2,3])
				out = out_rel
				
				
			elif self.loss_type in ["RaSGAN", "RaLSGAN"]:
				# relativistic average. patch gan/disc: cmp to average value of every patch
				if flip_target:
					out_rel_real = out_fake-tf.reduce_mean(out_real)#, axis=0)
					out_rel_fake = out_real-tf.reduce_mean(out_fake)#, axis=0)
				else:
					out_rel_real = out_real-tf.reduce_mean(out_fake)#, axis=0)
					out_rel_fake = out_fake-tf.reduce_mean(out_real)#, axis=0)
				
				if self.loss_type=="RaSGAN":
					loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out_rel_real, labels=self.real_labels(out_rel_real))) \
						+ tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out_rel_fake, labels=self.fake_labels(out_rel_fake)))
				elif self.loss_type=="RaLSGAN":
					loss = tf.reduce_mean(tf.math.squared_difference(out_rel_real, self._label_real)) \
						+ tf.reduce_mean(tf.math.squared_difference(out_rel_fake, -self._label_real))
				
				out = (out_rel_real, out_rel_fake)
				scores = tf.zeros([1], dtype=tf.float32)
		
		return loss, scores#, name

def loss_disc_real(disc, in_real, training=True):
	in_real = disc.check_input(in_real, "real")
	out_real = disc.model(in_real, training=training)
	#labels_real = tf.concat([tf.ones([setup.training.discriminator.num_real] + disc_out_real.get_shape().as_list()[1:])*setup.training.discriminator.target_label], axis=0)
	loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=out_real, labels=disc.real_labels(out_real))
	disc.check_output(out_real, loss_real, in_real, "real")
	loss_real = tf.math.reduce_mean(loss_real)
	disc.opt_ctx.add_loss([loss_real], loss_value_scaled=loss_real, loss_name='discriminator/real_loss')
	if disc.opt_ctx.setup.training.discriminator.use_fc:
		scores_real = tf.math.sigmoid(out_real)
	else:
		scores_real = tf.reduce_mean(tf.math.sigmoid(out_real), axis=[1,2,3])
	return scores_real

def loss_disc_fake(disc, in_fake, training=True):
	in_fake = disc.check_input(in_fake, "fake")
	out_fake = disc.model(in_fake, training=training)
	#labels_fake = tf.concat([tf.zeros([len(disc_in_fake)] + disc_out_fake.get_shape().as_list()[1:])], axis=0)
	loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=out_fake, labels=disc.fake_labels(out_fake))
	disc.check_output(out_fake, loss_fake, in_fake, "fake")
	loss_fake = tf.math.reduce_mean(loss_fake)
	disc.opt_ctx.add_loss([loss_fake], loss_value_scaled=loss_fake, loss_name='discriminator/fake_loss')
	if disc.opt_ctx.setup.training.discriminator.use_fc:
		scores_fake = tf.math.sigmoid(out_fake)
	else:
		scores_fake = tf.reduce_mean(tf.math.sigmoid(out_fake), axis=[1,2,3])
	return scores_fake

def loss_disc_weights(disc):
	loss_scale = disc.opt_ctx.loss_schedules.discriminator_regularization(disc.opt_ctx.iteration)
	if disc.opt_ctx.LA(loss_scale):
		with disc.opt_ctx.profiler.sample('discriminator regularization'):
			disc_weights = disc.var_list()
			tmp_loss = tf.reduce_mean([tf.reduce_mean(tf.nn.l2_loss(var)) for var in disc_weights])
		tmp_loss_scaled = tmp_loss * loss_scale
		disc.opt_ctx.add_loss([tmp_loss_scaled], tmp_loss, tmp_loss_scaled, loss_scale, 'discriminator/regularization')
		return True
	return False

def optStep_discriminator(disc_ctx, state=None, additional_fake_samples=None):
	if disc_ctx.train and disc_ctx.opt_ctx.setup.training.discriminator.start_delay<=disc_ctx.opt_ctx.iteration:
		LOG.debug("Optimization step for discriminator")
		with disc_ctx.opt_ctx.profiler.sample('optStep_discriminator'):
			#prepare real samples
			disc_in_real = disc_ctx.real_samples(spatial_augment=True, intensity_augment=True)
			disc_ctx.dump_samples(disc_in_real, True)
			
			if disc_ctx.loss_type in ["SGAN"]:
				with disc_ctx.opt_ctx.profiler.sample('real'):
					with tf.GradientTape() as disc_tape:
						disc_loss_real, disc_scores_real = disc_ctx.loss((disc_in_real,), flip_target=False, training=True) #disc_scores_real = loss_disc_real(disc_ctx, disc_in_real)
						disc_ctx.opt_ctx.add_loss(disc_loss_real, loss_value_scaled=reduce_losses(disc_loss_real), loss_name='discriminator/loss_real')
						loss_disc_weights(disc_ctx)
						disc_loss_real = disc_ctx.opt_ctx.pop_losses()
					grads = disc_tape.gradient(disc_loss_real, disc_ctx.var_list())
					disc_ctx.optimizer.apply_gradients(zip(grads, disc_ctx.var_list()))
			
			#prepare fake samples
			disc_in_fake = []
			if additional_fake_samples is not None:
				r = np.random.choice(len(additional_fake_samples), len(disc_ctx.render_ctx.cameras), replace=False)
				disc_in_fake.extend([additional_fake_samples[_] for _ in r])
			disc_in_fake.extend(disc_ctx.fake_samples(state, concat=False, spatial_augment=False, name="disc_fake_samples"))
			if disc_ctx.crop_size is not None or disc_ctx.scale_to_input_res:
				disc_in_fake = disc_ctx._prepare_samples(*disc_in_fake, crop_shape="INPUT" if disc_ctx.scale_to_input_res else disc_ctx.crop_size, scale_range=disc_ctx.scale_range, rotation_mode=disc_ctx.rotation_mode)
			
			with disc_ctx.opt_ctx.profiler.sample('fake'):
				disc_in_fake = disc_ctx.augment_intensity(tf.concat(disc_in_fake, axis=0), disc_ctx.opt_ctx.setup.data.discriminator.scale_fake, disc_ctx.opt_ctx.setup.data.discriminator.gamma_fake)
			#	scale_shape = (len(disc_in_fake),1,1,1)
			#	scale = tf.random.uniform(scale_shape, *disc_ctx.opt_ctx.setup.data.discriminator.scale_fake, dtype=tf.float32)
			#	gamma = tf.random.uniform(scale_shape, *disc_ctx.opt_ctx.setup.data.discriminator.gamma_fake, dtype=tf.float32)
			#	scale = [scale,scale,scale]
			#	gamma = [gamma,gamma,gamma]
			#	if disc_ctx._conditional_hull: #do not scale the intensity of the hull, the disc should be invariant to intensities
			#		scale.append(tf.ones(scale_shape, dtype=tf.float32))
			#		gamma.append(tf.ones(scale_shape, dtype=tf.float32))
			#	if disc_ctx._temporal_input:
			#		scale *=3
			#		gamma *=3
			#	disc_in_fake = tf.pow(tf.multiply(tf.concat(disc_in_fake, axis=0), tf.concat(scale, axis=-1)), tf.concat(gamma, axis=-1))
				disc_ctx.dump_samples(disc_in_fake, False)
				if disc_ctx.loss_type in ["SGAN"]:
					with tf.GradientTape() as disc_tape:
						disc_loss_fake, disc_scores_fake = disc_ctx.loss((disc_in_fake,), flip_target=True, training=True) #disc_scores_fake = loss_disc_fake(disc_ctx, disc_in_fake)
						disc_ctx.opt_ctx.add_loss(disc_loss_fake, loss_value_scaled=reduce_losses(disc_loss_fake), loss_name='discriminator/loss_fake')
						loss_disc_weights(disc_ctx)
						disc_loss_fake = disc_ctx.opt_ctx.pop_losses()
					grads = disc_tape.gradient(disc_loss_fake, disc_ctx.var_list())
					disc_ctx.optimizer.apply_gradients(zip(grads, disc_ctx.var_list()))
			
			if not (disc_ctx.loss_type in ["SGAN"]):
				with disc_ctx.opt_ctx.profiler.sample(disc_ctx.loss_type):
					with tf.GradientTape() as disc_tape:
						disc_loss, disc_scores = disc_ctx.loss((disc_in_real, disc_in_fake), False, True) #disc_scores_fake = loss_disc_fake(disc_ctx, disc_in_fake)
						disc_ctx.opt_ctx.add_loss(disc_loss, loss_value_scaled=reduce_losses(disc_loss), loss_name='discriminator/'+disc_ctx.loss_type)
						loss_disc_weights(disc_ctx)
						disc_loss = disc_ctx.opt_ctx.pop_losses()
					grads = disc_tape.gradient(disc_loss, disc_ctx.var_list())
					disc_ctx.optimizer.apply_gradients(zip(grads, disc_ctx.var_list()))
		
		if (disc_ctx.loss_type in ["SGAN"]):
			if disc_ctx.opt_ctx.record_summary:
				summary_names = disc_ctx.opt_ctx.make_summary_names('discriminator/real_score')
				disc_ctx.opt_ctx._tf_summary.scalar(summary_names[0], tf.reduce_mean(disc_scores_real), step=disc_ctx.opt_ctx.iteration)
				summary_names = disc_ctx.opt_ctx.make_summary_names('discriminator/fake_score')
				disc_ctx.opt_ctx._tf_summary.scalar(summary_names[0], tf.reduce_mean(disc_scores_fake), step=disc_ctx.opt_ctx.iteration)
			return disc_loss_real, disc_loss_fake, disc_scores_real, disc_scores_fake
		else:
			if disc_ctx.opt_ctx.record_summary:
				summary_names = disc_ctx.opt_ctx.make_summary_names('discriminator/score')
				disc_ctx.opt_ctx._tf_summary.scalar(summary_names[0], tf.reduce_mean(disc_scores), step=disc_ctx.opt_ctx.iteration)
			return disc_loss[0], disc_scores
	else:
		LOG.debug("Optimization discriminator inactive")
		return 0,0,0,0
