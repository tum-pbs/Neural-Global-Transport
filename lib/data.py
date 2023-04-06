import os, copy, re
import tensorflow as tf
import numpy as np
import numbers, collections.abc
from lib.tf_ops import shape_list, has_shape
from phitest.render.transform import GridTransform, Transform
from phitest.render.camera import Camera
from phitest.render.profiling import SAMPLE
from scipy.spatial.transform import Rotation

import warnings
import logging
log = logging.getLogger('data')
log.setLevel(logging.DEBUG)

def load_density_grid(sim_transform, path, density=0.06, frame=0, mask='density_{:06d}.npz', input_format='DHW', reverse_dims='', array_name='arr_0'):
	out_format = 'NDHWC'
	if mask is not None and len(mask)>0:
		filename = mask.format(frame)
		p = os.path.join(path, filename)
	else:
		filename = path
		p = path
	if p.endswith('.npy'):
		d = np.load(p)
	else:
		with np.load(p) as np_data:
			try:
				d = np_data[array_name].astype(np.float32)
			except KeyError:
				raise KeyError('key \'{}\' not in archive. Available keys: {}'.format(array_name, list(np_data.keys())))
	log.info('loaded density grid \'{}\' from \'{}\' with shape {}.'.format(filename, path, d.shape))
	if len(d.shape)!=len(input_format):
		raise ValueError('Given input format {} does not match loaded data shape {}'.format(input_format, d.shape))
#	print_stats(d, filename, log)
	#d = d[:,:128]
	reverse = [input_format.index(_) for _ in reverse_dims]
	d = tf.reverse(d, reverse)
	for dim in out_format:
		if dim not in input_format:
			d = tf.expand_dims(d, out_format.index(dim))
	d = tf.constant(d*density, dtype=tf.float32)
	#d = tf_pad_to_next_pow_two(d)
	sim_transform.set_data(d)
	return d

def load_targets(pack_path_mask, simulations=[0], frames=[140], cam_ids=[0,1,2,3,4], threshold=0.0, bkg_subtract=True, flip_y=False):
	bkgs = []
	targets = []
	targets_raw = []
	for sim in simulations:
		log.debug('loading sim {}'.format(sim))
		with np.load(pack_path_mask.format(sim=sim, frame=0)) as np_data:
			bkg = np_data['data'][cam_ids].astype(np.float32)
			if flip_y:
				bkg = np.flip(bkg, axis=-3)
		bkgs.append(bkg)
		for frame in frames:
			log.debug('loading frame {}'.format(frame))
			with np.load(pack_path_mask.format(sim=sim, frame=frame)) as np_data:
				target_raw = np_data['data'][cam_ids].astype(np.float32)
				if flip_y:
					target_raw = np.flip(target_raw, axis=-3)
			if bkg_subtract:
				target = tf.maximum(target_raw-bkg, 0.0)
				if threshold>0.0: #cut off noise in background 
					condition = tf.greater_equal(target, threshold)
					target_raw = tf.where(condition, target_raw, bkg)
					target = tf.where(condition, target, tf.zeros_like(target))
				targets.append(target)
			targets_raw.append(target_raw)
	if bkg_subtract:
		return tf.concat(targets_raw, axis=0), tf.concat(targets, axis=0), tf.concat(bkgs, axis=0)
	else:
		return tf.concat(targets_raw, axis=0), tf.concat(bkgs, axis=0)

'''
def background_substract(img_bkg, bkg, threshold=1e-2, hull_blur=0.0):
	img_bsub = tf.maximum(img_bkg-bkg, 0)
	condition = tf.greater_equal(img_bsub, threshold)
	mask = 
'''

class ScalarFlowDataset(tf.data.Dataset):
	def _generator(sim_range, frame_range, cam_ids=[0,1,2,3,4]):
		#path_mask = '/mnt/netdisk1/eckert/scalarFlow/sim_{:06d}/input/cam/imgsUnproc_{:06d}.npz'
		for sim in range(*sim_range):
			with np.load(path_mask.format(sim=sim, frame=0)) as np_data:
				bkgs = np_data['data'][cam_ids].astype(np.float32)
			for frame in range(*frame_range):
				with np.load(path_mask.format(sim=sim, frame=frame)) as np_data:
					views = np_data['data'][cam_ids].astype(np.float32)
				log.debug('loaded frame {} of sim {}'.format(frame, sim))
				for view, bkg in zip(views, bkgs):
					yield (view, bkg)
				#yield (views, bkgs)
	
	def __new__(cls, sim_range, frame_range, cam_ids=[0,1,2,3,4]):
		return tf.data.Dataset.from_generator(
			cls._generator,
			output_types=(tf.float32, tf.float32),
			output_shapes=(tf.TensorShape([1920,1080,1]),tf.TensorShape([1920,1080,1])),
			args=(sim_range, frame_range, cam_ids)
		)

class ScalarFlowIndexDataset(tf.data.Dataset):
	def _generator(sim_range, frame_range):
		for sim in range(*sim_range):
			for frame in range(*frame_range):
				yield (sim, frame)
	
	def __new__(cls, sim_range, frame_range):
		return tf.data.Dataset.from_generator(
			cls._generator,
			output_types=(tf.int32, tf.int32),
			output_shapes=([],[]),
			args=(sim_range, frame_range)
		)

def get_scalarflow_dataset(sim_range, frame_range, path_mask, cam_ids=[0,1,2,3,4], down_scale=1, threshold=0.0, flat=True, cache=True, raw=True, preproc=True, bkg=True, hull=False, path_preproc=None, temporal_input_steps=None):
	'''
		create a dataset from scalarFlow target images.
		sim_range: the simulation ids to use. parameters for the range() object (start[, stop[, step]]).
		frame_range: the frame numbers to use from each simulation. parameters for the range() object (start[, stop[, step]]).
		path_mask: string to use with .format(sim, frame) to give the file-paths to the images.
		cam_ids: indices of the image packs to use.
		down_scale: integer. window for average pooling.
		threshold: threshodl for background and noise substraction.
		flat: bool. whether to flatten the dataset, camera id dimension.
		cache: cache the loaded dataset in RAM.
		raw, preproc, bkg, hull: bool. the data types to provide.
		path_preproc: load scalarFlow preprocessed images instad of raw images from this path if not None.
		temporal_input_steps: create prev-curr-next inputs with this frame step if not None. trncate frame_range accordingly to avoid out-of-bounds accesses.
	'''
	if not (raw or preproc or bkg or hull):
		raise ValueError("empty dataset")
	from_preproc = (path_preproc is not None)
	mask = path_preproc if from_preproc else path_mask
	
	
	make_temporal_input = False
	if temporal_input_steps is not None:
		make_temporal_input = True
		if isinstance(temporal_input_steps, numbers.Integral):
			temporal_input_steps = [temporal_input_steps]
		elif not isinstance(temporal_input_steps, collections.abc.Iterable):
			raise ValueError("Invalid temporal_input_steps.")
	
	def load(sim, frame):
		#path_mask = '/mnt/netdisk1/eckert/scalarFlow/sim_{:06d}/input/cam/imgsUnproc_{:06d}.npz'
		bkgs = None
		if bkg or preproc or hull:
			with np.load(path_mask.format(sim=sim, frame=0)) as np_data:
				bkgs = np_data['data'][cam_ids].astype(np.float32)
		with np.load(mask.format(sim=sim, frame=frame)) as np_data:
			views = np_data['data'][cam_ids].astype(np.float32)
		
		if make_temporal_input:
			t_step = np.random.choice(temporal_input_steps)
			with np.load(mask.format(sim=sim, frame=frame-t_step)) as np_data:
				views_prev = np_data['data'][cam_ids].astype(np.float32)
			with np.load(mask.format(sim=sim, frame=frame+t_step)) as np_data:
				views_next = np_data['data'][cam_ids].astype(np.float32)
			views = [views_prev, views, views_next]
		
		log.debug('loaded frame {} of sim {}'.format(frame, sim))
		return views, bkgs
	#load_func = lambda s,f: tf.py_func(load, [s,f], (tf.float32,tf.float32))
	
	def preprocess(img_view, img_bkg):
	#	img_raw = tf.reshape(img_raw, [1]+list(img_raw.shape))
	#	img_bkg = tf.reshape(img_bkg, [1]+list(img_bkg.shape))
		img_view = tf.nn.avg_pool(img_view, (1,down_scale, down_scale, 1), (1,down_scale, down_scale, 1), 'VALID')
		img_bkg = tf.nn.avg_pool(img_bkg, (1,down_scale, down_scale, 1), (1,down_scale, down_scale, 1), 'VALID')
	#	img_raw = tf.reshape(img_raw, img_raw.shape[1:])
	#	img_bkg = tf.reshape(img_bkg, img_bkg.shape[1:])
		#print(img_raw.shape)
		if from_preproc:
			img_raw = img_view + img_bkg
			view_preproc = img_view
		else:
			img_raw = img_view
			view_preproc = tf.maximum(img_raw-img_bkg, 0)
		
		if threshold>0.0: #cut off noise in background 
			condition = tf.greater_equal(view_preproc, threshold)
			if not from_preproc:
				img_raw = tf.where(condition, img_raw, img_bkg)
				if preproc: view_preproc = tf.where(condition, view_preproc, tf.zeros_like(view_preproc))
			if hull: img_hull = tf.cast(condition, tf.float32) #tf.where(condition, tf.ones_like(view_preproc), tf.zeros_like(view_preproc))
		
		ret = []
		if raw: ret.append(img_raw)
		if preproc: ret.append(view_preproc)
		if bkg: ret.append(img_bkg)
		if hull: ret.append(img_hull)
		return tuple(ret)
	#map_func = lambda v,b: tf.py_func(preprocess, [v,b], (tf.float32,tf.float32,tf.float32))
	
	def fused_load(sim, frame):
		views, bkgs = load(sim, frame)
		if make_temporal_input:
			return ( \
				*preprocess(views[0], bkgs), \
				*preprocess(views[1], bkgs), \
				*preprocess(views[2], bkgs), \
			)
		else:
			return preprocess(views, bkgs) #views_raw, views_preproc, bkgs
	ret_type = []
	types = []
	if raw:
		ret_type.append(tf.float32)
		types.append("img_raw")
	if preproc:
		ret_type.append(tf.float32)
		types.append("img_preproc")
	if bkg:
		ret_type.append(tf.float32)
		types.append("img_bkg")
	if hull:
		ret_type.append(tf.float32)
		types.append("hull")
	if make_temporal_input:
		ret_type *=3
	f_func = lambda s,f: tf.py_func(fused_load, [s,f], tuple(ret_type))
	#
	num_sims = len(range(*sim_range))
	num_frames = len(range(*frame_range))
	log.info("Initialize scalarFlow dataset:\n\ttypes: %s\n\t%d frames from %d simulations each%s", types, num_frames, num_sims, ("\n\ttemporal steps %s"%temporal_input_steps) if make_temporal_input else "")
	#,num_parallel_calls=4
	loaded_data = ScalarFlowIndexDataset(sim_range, frame_range).shuffle(num_sims*num_frames,reshuffle_each_iteration=False).map(f_func,num_parallel_calls=4).prefetch(8)
	if flat:
		#loaded_data = loaded_data.flat_map(lambda x,y,z: tf.data.Dataset.from_tensor_slices((x,y,z)))
		loaded_data = loaded_data.flat_map(lambda *d: tf.data.Dataset.from_tensor_slices(d))
#	if preproc_only:
#		loaded_data = loaded_data.map(lambda x,y,z: y)
	if cache:
		loaded_data = loaded_data.cache()
	return loaded_data.repeat().shuffle(64)

USE_ALL_SEQUENCE_STEPS = False

class TargetsIndexDataset(tf.data.Dataset):
	def __get_rng(self, seed):
		try:
			rng = self.__rng
		except AttributeError:
			self.__rng = np.random.default_rng(seed)
		return self.__rng
	def _generator(sim_indices, frame_start, frame_stop, frame_strides, sequence_step, sequence_length, view_indices, num_views, seed):
		"""
		
		Args:
			sim_indices, view_indices: list(int)
			frame_start, frame_stop, sequence_strides, frame_step, sequence_length, num_views: int
		
		Yields:
			sim_idx
			list(frame_idx)
			list(view_idx)
		"""
		is_multistep = isinstance(sequence_step, collections.abc.Iterable)
		fix_views = True
		
		rng = np.random.default_rng(seed)
		
		if not USE_ALL_SEQUENCE_STEPS or not is_multistep:
			if is_multistep:
				assert all(_>0 for _ in sequence_step), "only logical forward steps are supported."
			max_step = max(sequence_step) if is_multistep else sequence_step
			log.info("max step of %s: %s", sequence_step, max_step)
			start_frames = list(range(frame_start, frame_stop - sequence_length*max_step +1, frame_strides))
			
			for sim_idx in sim_indices:
				for frame_idx in start_frames:
					step = np.random.choice(sequence_step) if is_multistep else sequence_step
					yield ( \
						sim_idx, \
						list(range(frame_idx, frame_idx+sequence_length*step, step)), \
						view_indices if fix_views else np.random.choice(view_indices, size=num_views, replace=False), \
						rng.integers(np.iinfo(np.int32).max) \
					)
		
		else:
			min_step = min(sequence_step)
			max_step = max(sequence_step)
			start_frames = list(range(frame_start, frame_stop, frame_strides))
			
			for sim_idx in sim_indices:
				for frame_idx in start_frames:
					for step in sequence_step:
						#frames = list(range(frame_idx, frame_idx+sequence_length*step, step))
						#all(frame_start<=f and f<frame_stop for f in frames)
						#not any(f<frame_start or frame_stop<=f for f in frames)
						if step>0 and (frame_idx+sequence_length*step<frame_stop):
							yield ( \
								sim_idx, \
								list(range(frame_idx, frame_idx+sequence_length*step, step)), \
								view_indices if fix_views else np.random.choice(view_indices, size=num_views, replace=False), \
								rng.integers(np.iinfo(np.int32).max) \
							)
						
						elif step<0 and (frame_idx+sequence_length*step>=frame_start):
							yield ( \
								sim_idx, \
								list(range(frame_idx, frame_idx+sequence_length*step, step)), \
								view_indices if fix_views else np.random.choice(view_indices, size=num_views, replace=False), \
								rng.integers(np.iinfo(np.int32).max) \
							)
						
						else: #step==0
							yield ( \
								sim_idx, \
								[frame_idx for _ in range(sequence_length)], \
								view_indices if fix_views else np.random.choice(view_indices, size=num_views, replace=False), \
								rng.integers(np.iinfo(np.int32).max) \
							)
	
	def __new__(cls, sim_indices, frame_start, frame_stop, frame_strides, sequence_step, sequence_length, view_indices, num_views, seed):
		assert isinstance(num_views, numbers.Integral)
		assert isinstance(sequence_length, numbers.Integral)
		return tf.data.Dataset.from_generator(
			cls._generator,
			output_types=(tf.int32, tf.int32, tf.int32, tf.int32),
			output_shapes=([],[sequence_length], [num_views], []),
			args=(sim_indices, frame_start, frame_stop, frame_strides, sequence_step, sequence_length, view_indices, num_views, seed)
		)
		
# --- data loading and handling for SF dataset ---

# Try to load all data only once and store only a single copy, with multiple aliases if necessary.
# could maybe use Grids from data_structures.py here...

class ImageSample:
	def __init__(self, sim, frame, cam_ids, path_mask, resample_fn=None, device=None):
		self.__device = device
		self.__sim = sim
		self.__frame = frame
		assert cam_ids is not None
		self.__cam_ids = cam_ids
		self.__path_mask = path_mask
		self.__resample_fn = resample_fn
		self.__data = None #VHWC
	@property
	def is_loaded(self):
		return self.__data is not None
	def _set(self, data):
		if not isinstance(data, (tf.Tensor, np.ndarray)): raise TypeError
		if not len(shape_list(data))==4: raise ValueError("ImageSample data must have shape VHWC, is {}".format(shape_list(data)))
		with tf.device(self.__device):
			self.__data = tf.identity(data)
	def get(self):
		if not self.is_loaded:
			self.load()
		return self.__data
	def load(self):
		with SAMPLE("Load ImageSample"):
			path = self.__path_mask.format(sim=self.__sim, frame=self.__frame)
			
			log.debug('Load images for frame {} of sim {} from {}'.format(self.__frame, self.__sim, path))
			
			with SAMPLE("load"):
				with np.load(path) as np_data:
					views = np_data['data'][self.__cam_ids].astype(np.float32)
			
			if self.__resample_fn is not None:
				with SAMPLE("resample_fn"):
					views = self.__resample_fn(views)
			
			with SAMPLE("set"):
				self._set(views)

class VolumeSample:
	def __init__(self, sim, frame, path_mask, load_fn=None, device=None):
		self.__device = device
		self.__sim = sim
		self.__frame = frame
		self.__path_mask = path_mask
		self.__load_fn = load_fn
		self.__data = None #DHWC
		self.__metadata = {}
	@property
	def is_loaded(self):
		return self.__data is not None
	def _set(self, data):
		if isinstance(data, tuple) and len(data)==2:
			if not isinstance(data[1], dict): raise TypeError("VolumeSample metadata must be dict.")
			self.__metadata = copy.copy(data[1])
			data = data[0]
		if not isinstance(data, (tf.Tensor, np.ndarray)): raise TypeError
		if not len(shape_list(data))==4: raise ValueError("VolumeSample data must have shape DHWC, is {}".format(shape_list(data)))
		with tf.device(self.__device):
			self.__data = tf.identity(data)
	def get(self):
		if not self.is_loaded:
			self.load()
		return self.__data
	def get_meta(self, key):
		if not self.is_loaded:
			raise RuntimeError("Data not yet loaded.")
		if not key in self.__metadata:
			raise KeyError(str(key))
		return self.__metadata[key]
	def load(self):
		#assuming SF data...
		with SAMPLE("Load VolumeSample"):
			#log.info("%s", self.__path_mask)
			path = self.__path_mask.format(sim=self.__sim, frame=self.__frame)
			with np.load(path) as np_data:
				if self.__load_fn is not None:
					data = self.__load_fn(np_data)
				else:
					with SAMPLE("load"):
						is_SF = "data" in np_data
						data = np_data['data' if is_SF else "arr_0"].astype(np.float32) # DHWC with C=1 and D/z reversed
			# if density_flipZ: #do this in resample_fn
				# data = data[::-1]
			# if self.__dtype=="velocitySF": raise NotImplementedError("flip vel z")
			# if self.__resample_fn is not None:
				# with SAMPLE("resample_fn"):
					# data = self.__resample_fn(data)
				
			log.debug("Loaded grid {} for frame {} of sim {} from {}".format(shape_list(data), self.__frame, self.__sim, path))
			with SAMPLE("set"):
				self._set(data)

class ScalarFlowDatasetCache:
	def __init__(self, image_channels=1, image_down_scale=1, noise_threshold=0.0, from_SFpreproc=False, \
			image_raw_path_mask=None, image_preproc_path_mask=None, SF_frame_offset=11, \
			density_path_mask=None, density_t_src=None, velocity_path_mask=None, velocity_t_src=None, domain_t_dst=None, grid_sampler=None, \
			density_type="SF", velocity_type="SF", velocity_staggered=True, device=None):
		
		self.__device = device
		#self.__batch_size = batch_size
		#self.__index_dataset = index_dataset.make_one_shot_iterator()
		#self.__dtypes = dtypes
		
		self.__image_channels = image_channels
		self.__image_down_scale = image_down_scale
		self.__noise_threshold = noise_threshold
		self.__from_SFpreproc = from_SFpreproc
		self.__SF_frame_offset = SF_frame_offset
		
		self.__image_raw_path_mask = image_raw_path_mask
		self.__image_preproc_path_mask = image_preproc_path_mask
		
		self.__density_path_mask = density_path_mask
		self.__density_t_src = density_t_src.copy_no_data()
		self.__density_t_dst = domain_t_dst.copy_no_data()
		self.__density_sampler = grid_sampler
		assert density_type in ["SF", "OWN", "MANTA"]
		self.__density_type = density_type
		self.__velocity_path_mask = velocity_path_mask
		self.__velocity_t_src = velocity_t_src.copy_no_data()
		self.__velocity_staggered = velocity_staggered
		self.__velocity_scale_magnitude_by_resolution = True
		assert velocity_type in ["SF", "OWN", "MANTA"]
		self.__velocity_type = velocity_type
		
		self.__base_sample_cache = {} #(type,sim,frame):ImageSample/VolumeSample
		#self.__frame_sample_cache = {} #(sim, frame):FrameSample
		# self.__samples = {} #sample_id:SequenceSample
		
		#self.__current_batch = {}
		#self.step()
	
	def is_compatible(self, image_channels=1, image_down_scale=1, noise_threshold=0.0, from_SFpreproc=False, \
			image_raw_path_mask=None, image_preproc_path_mask=None, SF_frame_offset=-11, \
			density_path_mask=None, density_t_src=None, velocity_path_mask=None, velocity_t_src=None, domain_t_dst=None, grid_sampler=None, \
			density_type="SF", velocity_type="SF", velocity_staggered=True, device=None):
		
		return self.__device==device \
			and self.__image_channels==image_channels \
			and self.__image_down_scale==image_down_scale \
			and self.__noise_threshold==noise_threshold \
			and self.__from_SFpreproc==from_SFpreproc \
			and (self.__image_raw_path_mask==image_raw_path_mask) \
			and self.__SF_frame_offset==SF_frame_offset \
			and ((self.__image_preproc_path_mask==image_preproc_path_mask) if self.__from_SFpreproc else True) \
			and (self.__density_path_mask==density_path_mask or density_path_mask is None) \
			and (self.__density_type==density_type or (density_path_mask is None))\
			and (self.__velocity_type==velocity_type or (velocity_path_mask is None))\
			and (self.__velocity_path_mask==velocity_path_mask or velocity_path_mask is None) \
			and (self.__velocity_staggered==velocity_staggered or velocity_path_mask is None)
			#and self.__density_t_src==density_t_src
			#and self.__density_t_dst==domain_t_dst
			#and self.__velocity_t_src==velocity_t_src
			#and self.__density_sampler==grid_sampler
	
	def _adjust_images_fn(self, imgs):
		
		if False: #flip y
			imgs = imgs[...,::-1,:,:]
		
		shape = shape_list(imgs)
		c = shape[-1]
		if self.__image_channels==c:
			pass
		elif self.__image_channels>1 and c==1:
			imgs = tf.tile(imgs, [1]*(len(shape)-1) + [self.__image_channels])
		elif self.__image_channels==1 and c>1:
			imgs = tf.reduce_mean(imgs, axis=-1, keepdims=True)
		else:
			raise ValueError("Can't adjust channels from %d to %d"%(c, self.__image_channels))
		
		return tf.nn.avg_pool(imgs, (1,self.__image_down_scale, self.__image_down_scale, 1), (1,self.__image_down_scale, self.__image_down_scale, 1), 'VALID') #NHWC
	
	def _resample_SFdensity_fn(self, np_data):
		with SAMPLE("load"):
			if self.__density_type == "SF":
				density = np_data["data"].astype(np.float32)
				density = density[::-1] #invert z-axis
			elif self.__density_type == "OWN":
				density = np_data["arr_0"].astype(np.float32)
				density = np.squeeze(density, axis=0)
			elif self.__density_type == "MANTA":
				density = np_data["arr_0"].astype(np.float32)
				density = density[::-1] #invert z-axis
				#warnings.warn("scaling loaded density with 2.0.")
				#density *= 2.0
			centered_shape = shape_list(density)[:-1]
			
		
		if self.__density_t_src is not None and self.__density_t_dst is not None:
			with SAMPLE("resample"):
				density = tf.expand_dims(tf.identity(density), 0) # NDHWC
				t_src = self.__density_t_src.copy_no_data() #to set grid size
				t_src.grid_size = centered_shape
				density = tf.squeeze(self.__density_sampler._sample_transform(density, [t_src], [self.__density_t_dst], fix_scale_center=True), (0,1)) #NVDHWC -> DHWC
		
		return density
	
	def _resample_SFvelocity_fn(self, np_data):
		with SAMPLE("load"):
			if self.__velocity_type=="SF":
				# SF velocity data: combined staggered grid. a single grid with 3 channels, same spatial dimensions as density. Therefor the top x-y-z cells are missing.
				velocity = np_data["data"].astype(np.float32)
				velocity = velocity[::-1] #invert z-axis
				#velocity *= np.asarray([1,1,-1]) #invert z component
				x,y,z = tf.split(velocity, 3, axis=-1)
				z = z * np.float32(-1)
				centered_shape = shape_list(velocity)[:-1]
			elif self.__velocity_type=="OWN":
				x = np_data["vel_x"].astype(np.float32)
				x = np.squeeze(x, axis=0)
				y = np_data["vel_y"].astype(np.float32)
				y = np.squeeze(y, axis=0)
				z = np_data["vel_z"].astype(np.float32)
				z = np.squeeze(z, axis=0)
				centered_shape = np_data["centered_shape"]
			elif self.__density_type == "MANTA":
				# Mantaflow velocity data: combined staggered grid. a single grid with 3 channels, same spatial dimensions as density. Therefor the top x-y-z cells are missing.
				velocity = np_data["arr_0"].astype(np.float32)
				velocity = velocity[::-1] #invert z-axis
				x,y,z = tf.split(velocity, 3, axis=-1)
				z = z * np.float32(-1)
				centered_shape = shape_list(velocity)[:-1]
		
		# ideally sample only once to requested centered or staggered grid, using the sampler for necessary extrapolations.
		
		# these are specific to the transformation setup in the main script.
		# src is basic SFtransform
		# dst
		if self.__velocity_t_src is not None and self.__density_t_dst is not None:
			with SAMPLE("resample"):
				t_src = self.__velocity_t_src.copy_no_data()
				t_src.grid_size = centered_shape
				if self.__velocity_type in ["SF", "MANTA"]:
					cell_size_world_src = t_src.cell_size_world()
					x_t_src = t_src.copy_no_data()
					x_t_src.translation[0] -= cell_size_world_src.x*0.5 #same dimension and spatial size as centered grid, so only offset for staggered x
					y_t_src = t_src.copy_no_data()
					y_t_src.translation[1] -= cell_size_world_src.y*0.5
					z_t_src = t_src.copy_no_data()
					z_t_src.translation[2] -= cell_size_world_src.z*0.5
				elif self.__velocity_type=="OWN":
					cell_size_world_src = t_src.cell_size_world()
					x_t_src = t_src.copy_no_data()
					gs = x_t_src.grid_shape
					if centered_shape[2]!=gs.x or centered_shape[1]!=gs.y or centered_shape[0]!=gs.z: raise ValueError("Centered shape %s does not match transform shape %s"%(centered_shape, gs))
					if (gs.x==gs.z and gs.x<=gs.y and x_t_src.normalize=="MIN") or x_t_src.normalize=="ALL":
						gs.x = gs.x+1
						x_t_src.grid_shape = gs
						y_t_src = self.__density_t_src.copy_no_data()
						gs = y_t_src.grid_shape
						gs.y = gs.y+1
						y_t_src.grid_shape = gs
						z_t_src = self.__density_t_src.copy_no_data()
						gs = z_t_src.grid_shape
						gs.z = gs.z+1
						z_t_src.grid_shape = gs
						if x_t_src.normalize=="ALL":
							#adjust scaling
							# log.info("x-pre: %s, %s", cell_size_world_src, x_t_src.cell_size_world())
							x_t_src.scale[0] *= x_t_src.grid_shape.x/t_src.grid_shape.x
							y_t_src.scale[1] *= y_t_src.grid_shape.y/t_src.grid_shape.y
							z_t_src.scale[2] *= z_t_src.grid_shape.z/t_src.grid_shape.z
							# log.info("x: %s, %s", cell_size_world_src, x_t_src.cell_size_world())
							# log.info("y: %s, %s", cell_size_world_src, y_t_src.cell_size_world())
							# log.info("z: %s, %s", cell_size_world_src, z_t_src.cell_size_world())
							# assert cell_size_world_src == x_t_src.cell_size_world(), "%s != %s"%(cell_size_world_src, x_t_src.cell_size_world())
							# assert cell_size_world_src == y_t_src.cell_size_world(), "%s != %s"%(cell_size_world_src, y_t_src.cell_size_world())
							# assert cell_size_world_src == z_t_src.cell_size_world(), "%s != %s"%(cell_size_world_src, z_t_src.cell_size_world())
							
					else:
						 raise NotImplementedError("relying on automatic normalization for correct scaling of staggered grid.")
				
				if self.__velocity_staggered:
					# target: own separate staggered grid format with dimension+1 for the respective components. resampled in domain given by transform (with matching offsets).
					cell_size_world_dst = self.__density_t_dst.cell_size_world()
					x_t_dst = self.__density_t_dst.copy_no_data()
					# dimension +1, spatial size +1 cell, adjust scale to keep same cell size, no offset needed as center stay the same.
					# only need to adjust scale if the changed grid size changes anything in combination with scale normalizations
					gs = x_t_dst.grid_shape
					if not (gs.x==gs.z and gs.x<=gs.y and x_t_dst.normalize=="MIN"): raise NotImplementedError("relying on automatic normalization for correct scaling of staggered grid.")
					gs.x = gs.x+1
					x_t_dst.grid_shape = gs
					#x_t_dst.parent.parent.parent = Transform(scale=[gs.x/self.__density_t_dst.grid_shape.x,1,1])
					#log.info("vel transform test:\nbase dst: %s\nnew dst: %s\nbase cell size world: %s\nnew cell size world: %s", self.__density_t_dst, x_t_dst, cell_size_world_dst, x_t_dst.cell_size_world())
					#raise NotImplementedError
					y_t_dst = self.__density_t_dst.copy_no_data()
					gs = y_t_dst.grid_shape
					gs.y = gs.y+1
					y_t_dst.grid_shape = gs
					z_t_dst = self.__density_t_dst.copy_no_data()
					gs = z_t_dst.grid_shape
					gs.z = gs.z+1
					z_t_dst.grid_shape = gs
				else:
					raise NotImplementedError
				
				x = tf.expand_dims(x, 0) # NDHWC
				y = tf.expand_dims(y, 0) # NDHWC
				z = tf.expand_dims(z, 0) # NDHWC
				x = tf.squeeze(self.__density_sampler._sample_transform(x, [x_t_src], [x_t_dst], fix_scale_center=True), (0,1)) #NVDHWC -> DHWC
				y = tf.squeeze(self.__density_sampler._sample_transform(y, [y_t_src], [y_t_dst], fix_scale_center=True), (0,1)) #NVDHWC -> DHWC
				z = tf.squeeze(self.__density_sampler._sample_transform(z, [z_t_src], [z_t_dst], fix_scale_center=True), (0,1)) #NVDHWC -> DHWC
				
				if self.__velocity_scale_magnitude_by_resolution:
					x = x * (cell_size_world_src.x / cell_size_world_dst.x)
					y = y * (cell_size_world_src.y / cell_size_world_dst.y)
					z = z * (cell_size_world_src.z / cell_size_world_dst.z)
				
				# combine for transport
				velocity = tf.concat([
					tf.pad(x, ((0,1),(0,1),(0,0),(0,0))),
					tf.pad(y, ((0,1),(0,0),(0,1),(0,0))),
					tf.pad(z, ((0,0),(0,1),(0,1),(0,0))),
					], axis=-1)
				#log.info("velocity sampled to shape: %s, %s, %s -> %s", shape_list(x), shape_list(y), shape_list(z), shape_list(velocity))
		else:
			if self.__velocity_staggered:
				raise NotImplementedError
			else:
				raise NotImplementedError
		
		return velocity
	
	
	def _get_sample(self, dtype, sim, frame, cam_ids=None, cache_sample=True):
		key = (dtype, sim, frame)
		if key not in self.__base_sample_cache:
			log.debug("Key '%s' not found in cache.", key)
			if dtype=="RAW":
				if self.__from_SFpreproc:
					sample = ImageSample(sim, frame, cam_ids, path_mask=None, device=self.__device)
					#img_raw = img_view + img_bkg
					sample._set(self._get_sample("PREPROC",sim,frame,cam_ids,cache_sample).get() + self._get_sample("BKG",sim,0,cam_ids,cache_sample).get())
				else:
					sample = ImageSample(sim, frame, cam_ids, path_mask=self.__image_raw_path_mask, resample_fn=self._adjust_images_fn, device=self.__device)
					if self.__noise_threshold>0.0:
						tmp_preproc = tf.maximum(sample.get() - self._get_sample("BKG",sim,0,cam_ids,cache_sample).get(), 0)
						sample._set( \
							tf.where( tf.greater_equal(tmp_preproc, self.__noise_threshold), \
								sample.get(), \
								self._get_sample("BKG",sim,0,cam_ids,cache_sample).get()))
			elif dtype=="PREPROC":
				if self.__from_SFpreproc:
					sample = ImageSample(sim, frame+self.__SF_frame_offset, cam_ids, path_mask=self.__image_preproc_path_mask, resample_fn=self._adjust_images_fn, device=self.__device)
				else:
					sample = ImageSample(sim, frame, cam_ids, path_mask=None, device=self.__device)
					#view_preproc = tf.maximum(img_raw-img_bkg, 0)
					sample._set(tf.maximum(self._get_sample("RAW",sim,frame,cam_ids,cache_sample).get() - self._get_sample("BKG",sim,0,cam_ids,cache_sample).get(), 0))
					if self.__noise_threshold>0.0:
						sample._set( \
							tf.where( tf.greater_equal(sample.get(), self.__noise_threshold), \
								sample.get(), \
								tf.zeros_like(sample.get())))
			elif dtype=="BKG":
				assert frame==0
				sample = ImageSample(sim, frame, cam_ids, path_mask=self.__image_raw_path_mask, resample_fn=self._adjust_images_fn, device=self.__device)
			elif dtype=="HULL":
				sample = ImageSample(sim, frame, cam_ids, path_mask=None, device=self.__device)
				if self.__from_SFpreproc:
					tmp_preproc = self._get_sample("PREPROC",sim,frame,cam_ids,cache_sample).get()
				else:
					tmp_preproc = tf.maximum(self._get_sample("RAW",sim,frame,cam_ids,cache_sample).get() - self._get_sample("BKG",sim,0,cam_ids,cache_sample).get(), 0)
				sample._set(tf.cast(tf.greater_equal(tmp_preproc, self.__noise_threshold), tf.float32))
			
			elif dtype=="DENSITY":
				sample = VolumeSample(sim, frame+self.__SF_frame_offset, path_mask=self.__density_path_mask, load_fn=self._resample_SFdensity_fn, device=self.__device)
			elif dtype=="VELOCITY":
				sample = VolumeSample(sim, frame+self.__SF_frame_offset, path_mask=self.__velocity_path_mask, load_fn=self._resample_SFvelocity_fn, device=self.__device)
			
			else:
				raise ValueError("Unknown data type.")
			
			if cache_sample:
				self.__base_sample_cache[key] = sample
		else:
			log.debug("Key '%s' found in cache.", key)
			sample = self.__base_sample_cache[key]
		return sample
	
	def _set_sample(self, data, dtype, sim, frame, cam_ids=None):
		key = (dtype, sim, frame)
		
		if dtype in ["RAW", "PREPROC", "BKG", "HULL"]:
			sample = ImageSample(sim, frame, cam_ids, path_mask=None, device=self.__device)
			sample._set(data)
		elif dtype in ["DENSITY", "VELOCITY"]:
			sample = VolumeSample(sim, frame, path_mask=None, device=self.__device)
			sample._set(data)
		else:
			raise ValueError("Unknown data type.")
		
		log.debug("Set '%s' in cache.", key)
		self.__base_sample_cache[key] = sample
	
	def _contains_sample_for(self, dtype, sim, frame):
		key = (dtype, sim, frame)
		return key in self.__base_sample_cache
	
	def clear(self):
		self.__base_sample_cache = {}

class DatasetIndex:
	def __init__(self, base_path, sub_path_mask):
		self.__base_path = os.path.abspath(base_path)
		assert "?P<sim>" in sub_path_mask and "?P<frame>" in sub_path_mask, "sub_path_mask needs named groups 'sim' and 'frame'"
		self.__path_mask = re.compile(re.escape(base_path + "/") + sub_path_mask)
		# assert isinstance(keys, (tuple, list)) and all(isinstance(key, str) for key in keys)
		# self.__keys = tuple(keys)
		
		self._build_index()
					
	def _build_index(self):
		paths = {}
		num_paths = 0
		for root, dirs, files in os.walk(self.__base_path):
			for file in files:
				path = os.path.abspath(os.path.join(root, file))
				match = self.__path_mask.search(path)
				if match:
					sim_name, frame_name = self._get_key(match)
					if sim_name not in paths:
						paths[sim_name] = {}
					paths[sim_name][frame_name] = path
					num_paths += 1
		
		
		self.__paths = paths
		self.__num_paths = num_paths
		self.__sim_names = sorted(paths.keys())
		self.__frame_names = [sorted(paths[sim_name].keys()) for sim_name in self.__sim_names]
		
		log.info("Built DatasetIndex with %d samples from %d simulations from '%s'.", num_paths, len(self.__paths), self.__base_path)
	
	# def __check_make_key(self, key):
		# if not isinstance(key, tuple):
			# raise TypeError
		# if not len(key)==len(self.__keys):
			# raise ValueError
		# if not allisinstance(k, str) for k in key):
			# key = tuple(str(k) for k in key)
		# return key
	
	# def __add_path(self, sim, frame, path):
		# #key = self.__check_make_key(key)
		# self.__paths[key] = path
	
	def _get_key(self, match):
		return match["sim"], match["frame"] #tuple(str(match[key]) for key in self.__keys)
	
	# def __getitem__(self, key):
		# key = self.__check_make_key(key)
		# if key not in self.__paths:
			# raise KeyError(str(key))
		# return self.__paths[key]
	def get_num_sims(self):
		return len(self.__sim_names)
	def get_num_frames(self, sim_idx):
		return len(self.__frame_names[sim_idx])
	def get_path_by_index(self, sim, frame):
		return self.__paths[self.__sim_names[sim]][self.__frame_names[sim][frame]]
	def get_path_by_key(self, sim, frame):
		return self.__paths[sim][frame]
	
	# def get_by_named_keys(self, **keys):
		# key = self._get_key(keys)
		# return self[key]

class SDFDatasetCache:
	def __init__(self, path_mask=None, device=None):
		self.__device = device
		if isinstance(path_mask, (list, tuple)):
			assert len(path_mask)==2
			path_mask = DatasetIndex(*path_mask)
		self.__path_mask = path_mask
		self.__base_sample_cache = {}
	
	def _path_mask(self, sim, frame):
		if isinstance(self.__path_mask, DatasetIndex):
			return self.__path_mask.get_path_by_index(sim, frame)
		return self.__path_mask
	
	def _load_SDF_fn(self, np_data):
		data = np_data["SDF"]
		data = data * (1/np_data["voxel_size"]) # distance values from world-space to grid-space
		
		shape = np.asarray(data.shape, dtype=np.float32) #zyx
		metadata = {
			"position": np_data["bounding_box_min"], #xyz
			"size": np.flip(shape) * np_data["voxel_size"], #xyz
		}
		
		data = np.expand_dims(data, -1)
		return data, metadata
	
	def _get_sample(self, sim, frame):
		key = (sim, frame)
		if key not in self.__base_sample_cache:
			log.debug("Key '%s' not found in cache.", key)
			self.__base_sample_cache[key] = VolumeSample(sim, frame, path_mask=self._path_mask(sim, frame), load_fn=self._load_SDF_fn, device=self.__device)
		else:
			log.debug("Key '%s' found in cache.", key)
		return self.__base_sample_cache[key]
	
	def clear(self):
		self.__base_sample_cache = {}

class FrameSample:
	def __init__(self, data_cache, sim, frame, cam_ids, dtypes, *args, **kwargs):
		self.__sim = sim
		self.__frame = frame
		self.__cam_ids = cam_ids
		self.__dtypes = dtypes
		if "RAW" in dtypes:
			self.__image_raw = data_cache._get_sample("RAW",sim,frame,cam_ids)
		if "PREPROC" in dtypes:
			self.__image_preproc = data_cache._get_sample("PREPROC",sim,frame,cam_ids)
		if "BKG" in dtypes:
			self.__image_bkg = data_cache._get_sample("BKG",sim,0,cam_ids)
		if "HULL" in dtypes:
			self.__image_hull = data_cache._get_sample("HULL",sim,frame,cam_ids)
		if "DENSITY" in dtypes:
			self.__density = data_cache._get_sample("DENSITY",sim,frame)
		if "VELOCITY" in dtypes:
			self.__velocity = data_cache._get_sample("VELOCITY",sim,frame)
	@property
	def sim(self):
		return self.__sim
	@property
	def frame(self):
		return self.__frame
	
	# def _set_dtype(self, dtype, sample):
		# if dtype=="RAW":
			# self.__image_raw = sample
		# elif dtype=="PREPROC":
			# self.__image_preproc = sample
		# elif dtype=="BKG":
			# self.__image_bkg = sample
		# elif dtype=="HULL":
			# self.__image_hull = sample
		# elif dtype=="DENSITY":
			# raise NotImplementedError
		# elif dtype=="VELOCITY":
			# raise NotImplementedError
		# else:
			# raise ValueError("Unknown dtype %s"%(dtype,))
		
	
	def get_dtype(self, dtype):
		if dtype=="RAW":
			return self.__image_raw.get()
		elif dtype=="PREPROC":
			return self.__image_preproc.get()
		elif dtype=="BKG":
			return self.__image_bkg.get()
		elif dtype=="HULL":
			return self.__image_hull.get()
		elif dtype=="DENSITY":
			return self.__density.get()
		elif dtype=="VELOCITY":
			return self.__velocity.get()
		else:
			raise ValueError("Unknown dtype %s"%(dtype,))

class SequenceSample:
	def __init__(self, dataset, sim, frames, cam_ids, *args, **kwargs):
		if len(frames)<1:
			raise ValueError("Empty sequence sample")
		elif len(frames)>1:
			self.__frame_strides = [n-c for c,n in zip(frames[:-1], frames[1:])]
			self.__frame_strides += [self.__frame_strides[-1]]
		else:
			self.__frame_strides = [1]
		self.__frame_samples = [dataset._get_frame_sample(sim, frame, cam_ids) for frame in frames]
	def __len__(self):
		return len(self.__frame_samples)
	def __getitem__(self, index):
		return self.__frame_samples[index]
	def get_frame_stride(self, index):
		#return self[index+1].frame - self[index].frame
		return self.__frame_strides[index]

class ScalaFlowDataset:
	def __init__(self, data_cache, index_dataset, batch_size, dtypes, device, base_transform=None, \
			render_targets=False, density_renderer=None, cameras=None, lights=None):
		
		self.__data_cache = data_cache
		#self.__batch_size = batch_size
		self.set_batch_size(batch_size)
		self.__index_dataset = index_dataset.make_one_shot_iterator()
		self.__base_transform = None if base_transform is None else base_transform.copy_no_data()
		self.__dtypes = dtypes
		self.__device = device
		
		if self.__base_transform is not None and self.__base_transform.parent is None and "TRANSFORM" in self.__dtypes:
			raise ValueError("base transform needs parent for randomization.")
		
		#self.__image_channels = image_channels
		#self.__image_down_scale = image_down_scale
		#self.__noise_threshold = noise_threshold
		#self.__from_SFpreproc = from_SFpreproc
		
		#self.__image_raw_path_mask = image_raw_path_mask
		#self.__image_preproc_path_mask = image_preproc_path_mask
		
		#self.__density_path_mask = density_path_mask
		#self.__density_t_src = density_t_src
		#self.__density_t_dst = density_t_dst
		#self.__density_sampler = density_sampler
		#self.__density_flipZ = density_flipZ
		#self.__velocity_path_mask = velocity_path_mask
		#self.__velocity_staggered = True
		self.__velocity_scale_magnitude_by_frame_stride = True
		
		self.__frame_sample_cache = {} #(sim, frame):FrameSample
		
		self.__is__render_targets = render_targets
		if self.__is__render_targets:
			if self.__base_transform is None: raise ValueError(".")
			if density_renderer is None: raise ValueError("")
			self.__density_renderer = density_renderer
			if cameras is None: raise ValueError("")
			self.__cameras = cameras
			if lights is None: raise ValueError("")
			self.__lights = lights
		
		#self.__current_batch = {}
		self.step()
	
	def set_batch_size(self, batch_size):
		assert isinstance(batch_size, numbers.Integral) and batch_size>0
		self.__batch_size = batch_size
	
	@property
	def RAW(self):
		return "RAW" in self.__dtypes
	@property
	def PREPROC(self):
		return "PREPROC" in self.__dtypes
	@property
	def BKG(self):
		return "BKG" in self.__dtypes
	@property
	def HULL(self):
		return "HULL" in self.__dtypes
	@property
	def MASK(self):
		return "MASK" in self.__dtypes
	@property
	def DENSITY(self):
		return "DENSITY" in self.__dtypes
	@property
	def VELOCITY(self):
		return "VELOCITY" in self.__dtypes
	@property
	def TRANSFORM(self):
		return "TRANSFORM" in self.__dtypes
	
	def _get_randomized_transform(self, base_transform, seed=None, transform_allow_scale=True, transform_allow_scale_non_uniform=False, transform_allow_mirror=True, transform_allow_rotY=True, transform_allow_translate=False):
		rng = np.random.default_rng(seed)
		randomize_transform = True
		t = base_transform.copy_no_data()
		
		if transform_allow_scale:
			scale_min, scale_max = 0.94, 1.06
			if transform_allow_scale_non_uniform:
				scale = rng.uniform(scale_min, scale_max, 3).tolist()
			else:
				scale = [rng.uniform(scale_min, scale_max)]*3
		else:
			scale = [1,1,1]
		
		if transform_allow_mirror:
			for dim in range(len(scale)):
				flip = rng.random()<0.5
				if flip: scale[dim] *= -1
		
		if transform_allow_rotY:
			#rotation = Transform.get_random_rotation()
			rotation = [0,rng.uniform(0.,360.), 0]
		else:
			rotation = [0,0,0]
		
		if transform_allow_translate:
			translation_min, translation_max = t.cell_size_world()*-6, t.cell_size_world()*6
			translation = rng.uniform(translation_min, translation_max).tolist()
		else:
			translation = [0,0,0]
		
		#log.debug("Randomized grid transform: s=%s, r=%s, t=%s", scale, rotation, translation)
		
		t.parent.set_scale(scale if randomize_transform else None)
		t.parent.set_rotation_angle(rotation if randomize_transform else None)
		t.parent.set_translation(translation if randomize_transform else None)
		
		#log.info("generated transform: %s", t)
		
		return t
	
	def _transform_samples(self, frame_samples, dtype):
		t = self.__current_batch["TRANSFORM"][0]
		if dtype=="DENSITY":
			frame_samples = tf.stack(tf.identity(frame_samples)) # NDHWC
			frame_samples = tf.squeeze(self.__density_sampler._sample_transform(frame_samples, [self.__base_transform], [t], fix_scale_center=True), (1)) #NVDHWC -> DHWC
			frame_samples = tf.unstack(frame_samples)
		elif dtype=="VELOCITY":
			raise NotImplementedError("randomized transform resampling for staggered velocity.")
		else:
			pass
		return frame_samples
	
	def __render_targets(self, density, super_sampling=1):
		cameras = self.__cameras
		transform = self.__base_transform
		with SAMPLE("render sequence"):
			image_shape = list(cameras[0].transform.grid_size)
			if super_sampling>1:
				warnings.warn("synth data rendering supersampling set to %d"%super_sampling)
				image_shape[1] *= super_sampling
				image_shape[2] *= super_sampling
				cameras = copy.deepcopy(cameras)
				for cam in cameras: cam.transform.grid_size = image_shape
				def downsample(imgs):
					return tf.nn.avg_pool(img, (1,super_sampling, super_sampling, 1), (1,super_sampling, super_sampling, 1), 'VALID') #VWHC
			else:
				def downsample(imgs):
					return imgs
			
			if self.RAW or self.BKG:
				img_bkg = tf.zeros((len(cameras), image_shape[1], image_shape[2], 3))
				
				#img_bkg = tf.identity(img_bkg)
			mask = None
			if self.RAW or self.PREPROC or self.HULL or self.MASK:
				#img = [] #tf.zeros((num_views, image_shape[1], image_shape[2], 3))]*sequence_length
				if not tf.reduce_all(tf.is_finite(density)).numpy():
					log.warning("Density generated from %d is not finite.", seed)
				if tf.reduce_any(tf.less(density, 0.0)).numpy():
					log.warning("Density generated from %d is negative.", seed)
				
				tmp_transform = transform.copy_new_data(tf.expand_dims(density, 0))
				tmp = self.__density_renderer.render_density(tmp_transform, self.__lights, cameras, cut_alpha=not self.MASK) #V-NHWC, here N=1
				#img = tf.stack(tmp, axis=1) #NVHWC
				img = tf.concat(tmp, axis=0) #VHWC
				
				if self.MASK:
					img, mask = tf.split(img, [3,1], axis=-1)
					mask = tf.cast(tf.greater(mask, 0), dtype=tf.float32)
				if not tf.reduce_all(tf.is_finite(img)).numpy():
					log.warning("Images rendered for %d are not finite.", seed)
			
			img = downsample(img)
			
			if self.RAW:
				img_raw = img#downsample(img)
			else:
				img_raw = None
			
			if self.BKG:
				img_bkg = downsample(img_bkg)
			else:
				img_bkg = None
			
			if self.HULL:
				img_hull = img#downsample(img)
			else:
				img_hull = None
			
			if self.PREPROC:
				pass #img = downsample(img)
			else:
				img = None
			
			
		return img_raw, img, img_bkg, img_hull, mask
	
	# def __set_target_to_frame_sample(self, sim, frame, cam_ids, img, dtype, frame_sample):
		# image_sample = ImageSample(sim, frame, cam_ids, path_mask=None, device=self.__device)
		# image_sample._set(img)
		# frame_sample._set_dtype(dtype, image_sample)
		
	
	def _get_frame_sample(self, sim, frame, cam_ids):
		key = (sim, frame)
		if key not in self.__frame_sample_cache:
			if self.__is__render_targets:
				
				log.debug("Render targets %s for %d:%d", self.__dtypes, sim, frame)
				
				# get density to render via cache. only cache if needed later
				density = self.__data_cache._get_sample("DENSITY",sim,frame,cache_sample="DENSITY" in self.__dtypes).get()
				
				# render target images
				img_raw, img, img_bkg, img_hull, mask = self.__render_targets(density) # VHWC
				
				# set images in cache
				if self.RAW:
					self.__data_cache._set_sample(img_raw, "RAW", sim, frame, cam_ids)
					#self.__set_target_to_frame_sample(sim, frame, cam_ids, img_raw, "RAW", frame_sample)
				if self.PREPROC:
					self.__data_cache._set_sample(img, "PREPROC", sim, frame, cam_ids)
				if self.BKG and not self.__data_cache._contains_sample_for("BKG", sim, 0):
					self.__data_cache._set_sample(img_bkg, "BKG", sim, 0, cam_ids)
				if self.HULL:
					self.__data_cache._set_sample(img_hull, "HULL", sim, frame, cam_ids)
				#if self.MASK:
				#	self.__data_cache._set_sample(mask, "MASK", sim, frame, cam_ids)
				
				# proceed as with loaded targets.
			self.__frame_sample_cache[key] = FrameSample(self.__data_cache, sim, frame, cam_ids, dtypes=self.__dtypes)
		return self.__frame_sample_cache[key]
	
	def _get_next_index_sample(self):
		# sim, frames, cam_ids
		return self.__index_dataset.next()
	
	def get_next_sample(self):
		# sample_id = self._get_next_id()
		# return self.__samples[sample_id]
		sim, frames, cam_ids, rand_seed = self._get_next_index_sample()
		self.__rand_seed = rand_seed.numpy()
		sim = sim.numpy()
		frames = frames.numpy()
		cam_ids = cam_ids.numpy()
		#assert all(cam_ids==[2,1,0,4,3]), "%s"%(cam_ids,)
		log.debug("Next sample: sim %d, frames %s, cams %s.", sim, frames, cam_ids)
		return SequenceSample(dataset=self, sim=sim, frames=frames, cam_ids=cam_ids)
	
	def get_next_batch(self, batch_size=None):
		if batch_size is None: batch_size = self.__batch_size
		samples = [self.get_next_sample() for _ in range(batch_size)]
		return samples
		# NFVHWC or NFDHWC?
	
	def step(self):
		with SAMPLE("generate batch"):
			self.__current_batch = {} #{dtype:[frames[samples]]}
			# get data from next batch
			batch = self.get_next_batch()
			if "TRANSFORM" in self.__dtypes:
				# one transform per batch, same for all frames
				self.__current_batch["TRANSFORM"] = [self._get_randomized_transform(base_transform=self.__base_transform, seed=self.__rand_seed)]*len(batch[0])
			# stack data on batch dimension
		with SAMPLE("process batch"):
			for dtype in self.__dtypes:
				if dtype=="TRANSFORM":
					pass
				else:
					self.__current_batch[dtype] = []
					for frame_idx in range(len(batch[0])):
						if dtype=="VELOCITY" and self.__velocity_scale_magnitude_by_frame_stride:
							frame_samples = [seq_sample[frame_idx].get_dtype(dtype) * np.float32(seq_sample.get_frame_stride(frame_idx)) for seq_sample in batch]
						else:
							frame_samples = [seq_sample[frame_idx].get_dtype(dtype) for seq_sample in batch]
						if dtype=="TRANSFORM":
							frame_samples = self._transform_samples(frame_samples, dtype)
						frame_samples = tf.stack(frame_samples)
						with tf.device(self.__device):
							frame_samples = tf.identity(frame_samples)
						self.__current_batch[dtype].append(frame_samples) #NVHWC
		
	def frame_targets(self, idx, as_dict=False):
		return {dtype:self.__current_batch[dtype][idx] for dtype in self.__dtypes} if as_dict else [self.__current_batch[dtype][idx] for dtype in self.__dtypes]

# ---

class TargetDataset:
	def __init__(self, loader, resource_device=None):
		self.resource_device = resource_device
		self.loader = loader.make_one_shot_iterator()
		self.step()
	def step(self):
		self.batch = self.loader.next() # NFVHWC or NFDHWC
		self.frames = [tf.unstack(ttype, axis=1) for ttype in self.batch] # per type: split frames: F-NVHWC
	def frame_targets(self, idx, as_dict=False):
		return [ttype[idx] for ttype in self.frames] # list of types: T-NVHWC; selected frame

def get_targets_dataset_size(sim_indices, frame_start, frame_stop, frame_strides, sequence_step, sequence_length):
	if not USE_ALL_SEQUENCE_STEPS:
		num_sims = len(sim_indices)
		max_step = max(sequence_step) if isinstance(sequence_step, list) else sequence_step
		num_frames = len(list(range(frame_start, frame_stop - sequence_length*max_step +1, frame_strides)))
		return num_sims * num_frames
	else:
		raise NotImplementedError

def get_targets_dataset_v2(sim_indices, frame_start, frame_stop, frame_strides, sequence_step, sequence_length, view_indices, num_views, \
		batch_size, path_raw, path_preproc=None, SF_frame_offset=-11, \
		raw=True, preproc=True, bkg=True, hull=False, down_scale=1, channels=1, threshold=0.0, shuffle_frames=True, \
		density=False, path_density=None, density_t_src=None, density_t_dst=None, density_sampler=None, density_type="SF", \
		velocity=False, path_velocity=None, velocity_t_src=None, velocity_type="SF", \
		randomize_transform=None, \
		cache_device=None, data_cache=None, seed=0, \
		render_targets=False, density_renderer=None, cameras=None, lights=None):
	
	# SF_frame_offset: SF raw input images are shifted by 11 frames compared to preproc and reconstruction.
	
	if not (raw or preproc or bkg or hull):
		raise ValueError("empty dataset")
	from_preproc = (path_preproc is not None)
	
	if render_targets:
		assert path_density is not None, "No density to render."
	
	create_data_cache = True
	if data_cache is not None:
		if not isinstance(data_cache, ScalarFlowDatasetCache): raise TypeError
		if data_cache.is_compatible(image_channels=channels, image_down_scale=down_scale, noise_threshold=threshold, from_SFpreproc=from_preproc, \
				image_raw_path_mask=path_raw, image_preproc_path_mask=path_preproc, SF_frame_offset=SF_frame_offset, \
				density_path_mask=path_density, density_t_src=density_t_src, velocity_path_mask=path_velocity, velocity_t_src=velocity_t_src, domain_t_dst=density_t_dst, grid_sampler=density_sampler, \
				density_type=density_type, velocity_type=velocity_type, velocity_staggered=True, device=cache_device):
			create_data_cache = False
	if create_data_cache:
		data_cache = ScalarFlowDatasetCache(image_channels=channels, image_down_scale=down_scale, noise_threshold=threshold, from_SFpreproc=from_preproc, \
			image_raw_path_mask=path_raw, image_preproc_path_mask=path_preproc, SF_frame_offset=SF_frame_offset, \
			density_path_mask=path_density, density_t_src=density_t_src, velocity_path_mask=path_velocity, velocity_t_src=velocity_t_src, domain_t_dst=density_t_dst, grid_sampler=density_sampler, \
			density_type=density_type, velocity_type=velocity_type, velocity_staggered=True, device=cache_device)
	
	if density and (path_density is None or density_t_src is None or density_t_dst is None or density_sampler is None): raise ValueError("")
	if velocity and (path_velocity is None or velocity_t_src is None or density_t_dst is None or density_sampler is None): raise ValueError("")
	
	
	dtypes = []
	if raw: dtypes.append("RAW")
	if preproc: dtypes.append("PREPROC")
	if bkg: dtypes.append("BKG")
	if hull: dtypes.append("HULL")
	if density: dtypes.append("DENSITY")
	if velocity: dtypes.append("VELOCITY")
	if randomize_transform: dtypes.append("TRANSFORM")
	
	num_sims = len(sim_indices)
	max_step = max(sequence_step) if isinstance(sequence_step, list) else sequence_step
	num_frames = len(list(range(frame_start, frame_stop - sequence_length*max_step +1, frame_strides)))
	log.info("Initialize target images dataset V2:\n\ttypes: %s\n\tfrom SF preproc: %s, render: %s\n\tSF prerpoc/reconstruction frame offset: %d\n\t%d frames from %d simulations with %s steps each.%s", dtypes, from_preproc, render_targets, SF_frame_offset, num_frames, num_sims, sequence_step, \
		" Created new data cache." if create_data_cache else " Reused provided data cache.")
	
	log.info("Load image targets from:\n\traw:%s\n\tpreproc: %s",path_raw,path_preproc)
	if density:
		log.info("Load density from:\n\t%s\n\t%s",path_density,density_t_src)
	if velocity:
		log.info("Load velocity from:\n\t%s\n\t%s",path_velocity,velocity_t_src)
	if density or velocity:
		log.info("Target transform:\n\t%s", density_t_dst)
		
	
	if len(view_indices)!=num_views:
		raise NotImplementedError
	
	index_data = TargetsIndexDataset(sim_indices, frame_start, frame_stop, frame_strides, sequence_step, sequence_length, view_indices, num_views, seed)
	if shuffle_frames:
		index_data = index_data.shuffle(num_sims * num_frames, reshuffle_each_iteration=True)
	index_data = index_data.repeat()
	
	return ScalaFlowDataset(data_cache=data_cache, index_dataset=index_data, batch_size=batch_size, dtypes=dtypes, device=cache_device, base_transform=density_t_dst, render_targets=render_targets, density_renderer=density_renderer, cameras=cameras, lights=lights), data_cache

def get_targets_dataset(sim_indices, frame_start, frame_stop, frame_strides, sequence_step, sequence_length, view_indices, num_views, path_raw, path_preproc=None, \
		raw=True, preproc=True, bkg=True, hull=False, down_scale=1, channels=1, threshold=0.0, cache=True, shuffle_size=64, shuffle_frames=True, \
		density=False, path_density=None, density_t_src=None, density_t_dst=None, density_sampler=None, density_flipZ=True):
	"""
	
	
	returns:
		Dataloader for target images, shape: NSVHWC (batch, sequence, views, height, width, channels)
		raw, preproc, bkg, hull (if active)
	"""
	
	if not (raw or preproc or bkg or hull):
		raise ValueError("empty dataset")
	from_preproc = (path_preproc is not None)
	mask = path_preproc if from_preproc else path_raw
	
	load_bkg = bkg or preproc or hull
	
	if density and (path_density is None or density_t_src is None or density_t_dst is None or density_sampler is None): raise ValueError("")
	
	make_temporal_input = False
	bkg_frame=0
	
	def load(sim, frame, cam_ids):
		#path_mask = '/mnt/netdisk1/eckert/scalarFlow/sim_{:06d}/input/cam/imgsUnproc_{:06d}.npz'
		# expand here for different file formats/types
		with np.load(mask.format(sim=sim, frame=frame)) as np_data:
			views = np_data['data'][cam_ids].astype(np.float32)
		log.debug('loaded frame {} of sim {}'.format(frame, sim))
		return views #VHWC
	def load_grid(sim, frame):
		#assuming SF data...
		with np.load(path_density.format(sim=sim, frame=frame)) as np_data:
			g = np_data['data'].astype(np.float32) # DHWC with C=1 and D/z reversed
		if density_flipZ:
			g = g[::-1]
		#g = g.reshape([1] + list(g.shape)) # 
		log.debug('loaded grid frame {} of sim {}'.format(frame, sim))
		return g # DHWC
	
	def scale_image_batch(imgs, down_scale):
		return tf.nn.avg_pool(imgs, (1,down_scale, down_scale, 1), (1,down_scale, down_scale, 1), 'VALID') #NHWC
	
	def adjust_channels(imgs):
		shape = shape_list(imgs)
		c = shape[-1]
		if channels==c:
			return imgs
		elif channels>1 and c==1:
			return tf.tile(imgs, [1]*(len(shape)-1) + [channels])
		elif channels==1 and c>1:
			return tf.reduce_mean(imgs, axis=-1, keepdims=True)
		else:
			raise ValueError("Can't adjust channels from %d to %d"%(c, channels))
	
	def preprocess(img_view, img_bkg):
		# np: FVHWC
		img_view = adjust_channels(img_view)
		img_bkg = adjust_channels(img_bkg)
		img_bkg = tf.broadcast_to(img_bkg, tf.shape(img_view))
		
		if from_preproc:
			img_raw = img_view + img_bkg
			view_preproc = img_view
		else:
			img_raw = img_view
			view_preproc = tf.maximum(img_raw-img_bkg, 0)
		
		if threshold>0.0: #cut off noise in background 
			condition = tf.greater_equal(view_preproc, threshold)
			if not from_preproc:
				img_raw = tf.where(condition, img_raw, img_bkg)
				if preproc: view_preproc = tf.where(condition, view_preproc, tf.zeros_like(view_preproc))
			if hull: img_hull = tf.cast(condition, tf.float32) #tf.where(condition, tf.ones_like(view_preproc), tf.zeros_like(view_preproc))
		ret = []
		if raw: ret.append(img_raw)
		if preproc: ret.append(view_preproc)
		if bkg: ret.append(img_bkg)
		if hull: ret.append(img_hull)
		return tuple(ret) # per type: FVHWC
	#map_func = lambda v,b: tf.py_func(preprocess, [v,b], (tf.float32,tf.float32,tf.float32))
	
	def fused_load(sim, frames, views, rand_seed):
		views = view_indices #fix for now
		bgks = None
		if load_bkg:
			bkgs = load(sim, bkg_frame, views)
			bkgs = [scale_image_batch(bkgs, down_scale)]
			bkgs = tf.identity(bkgs) #tf.constant(bkgs, dtype=tf.float32) #FVHWC
			
		imgs = [scale_image_batch(load(sim, frame, views), down_scale) for frame in frames]
		#imgs = [scale_image_batch(img for img in imgs]
		imgs = tf.identity(imgs) #tf.constant(imgs, dtype=tf.float32) #FVHWC
		
		targets = preprocess(imgs, bkgs) # per type: FVHWC
		
		if density:
			dens = []
			for frame in frames:
				d = load_grid(sim, frame) # DHWC
				# TODO: target alignment, aspect mismatch
				# e.g. SF calib -> current used. need calibration, transforms and a sampler(renderer). can include downscale (+mipmapping).
				##tf.squeeze(scale_renderer._sample_transform(sF_d.d, [sF_transform], [dens_transform], fix_scale_center=True), (0,))
				# this might not work if force to run on CPU
				#with tf.device("gpu"):
				d = tf.expand_dims(tf.identity(d), 0) # NDHWC
				d = tf.squeeze(density_sampler._sample_transform(d, [density_t_src], [density_t_dst], fix_scale_center=True), (0,1)) #NVDHWC -> DHWC
				#raise NotImplementedError
				#dens = avg_pool(dens, density_scale)
				dens.append(d) # -> FDHWC
			
			targets = (*targets, tf.stack(dens),)
		
		
		return targets #views_raw, views_preproc, bkgs  # per type: FVHWC
	
	ret_type = []
	types = []
	if raw:
		ret_type.append(tf.float32)
		types.append("img_raw")
	if preproc:
		ret_type.append(tf.float32)
		types.append("img_preproc")
	if bkg:
		ret_type.append(tf.float32)
		types.append("img_bkg")
	if hull:
		ret_type.append(tf.float32)
		types.append("hull")
	if density:
		ret_type.append(tf.float32)
		types.append("density")
	if make_temporal_input:
		ret_type *=3
	f_func = lambda s,f,v,r: tf.py_func(fused_load, [s,f,v,r], tuple(ret_type))
	#
	num_sims = len(sim_indices)
	max_step = max(sequence_step) if isinstance(sequence_step, list) else sequence_step
	num_frames = len(list(range(frame_start, frame_stop - sequence_length*max_step +1, frame_strides)))
	log.info("Initialize target images dataset:\n\ttypes: %s\n\t%d frames from %d simulations with %s steps each%s", types, num_frames, num_sims, sequence_step, ("\n\ttemporal steps %s"%temporal_input_steps) if make_temporal_input else "")
		
	index_data = TargetsIndexDataset(sim_indices, frame_start, frame_stop, frame_strides, sequence_step, sequence_length, view_indices, num_views)
	if shuffle_frames:
		index_data = index_data.shuffle(num_sims * num_frames, reshuffle_each_iteration=not cache)
	loaded_data = index_data.map(f_func)
	#loaded_data = index_data.map(f_func,num_parallel_calls=4).prefetch(8)
	if cache:
		loaded_data = loaded_data.cache()
	loaded_data = loaded_data.repeat()
	if shuffle_size>0:
		loaded_data = loaded_data.shuffle(shuffle_size)
	return loaded_data

def make_cube(shape, inner, outer=None, cut_type=1):
	#shape, inner, outer: zyx-order
	# cut_type: 1=faces, 2=edges, 3=corners
	dim = 3
	assert isinstance(shape, list) and has_shape(shape, [dim])
	assert isinstance(inner, list) and has_shape(inner, [dim])
	if outer is None:
		outer = [1.0 for s in shape]
	else:
		assert isinstance(outer, list) and has_shape(outer, [dim])
	assert all((1>=o and o>i and i>=0) for i,o in zip(inner, outer))
	
	inner = [s*i*0.5 for s,i in zip(shape, inner)]
	outer = [s*o*0.5 for s,o in zip(shape, outer)]
	assert all((s>=o*2 and o>i and i>=0) for s,i,o in zip(shape, inner, outer))
	assert cut_type>0 and cut_type<=dim
	
	start = [(_-1)/2.0 for _ in shape]
	coord_grid = tf.meshgrid( #linspace: end is inclusive
			*(tf.linspace(-s,s,S) for S,s in zip(shape,start)),
		indexing='ij')
	coord_grid = abs(tf.transpose(coord_grid, list(range(dim+1))[::-1]))
	
	cond = tf.logical_and( \
		tf.greater_equal( \
			tf.reduce_sum(tf.cast(tf.less_equal(coord_grid,outer), tf.int32), axis=-1, keepdims=True), \
			cut_type), \
		tf.greater_equal( \
			tf.reduce_sum(tf.cast(tf.greater_equal(coord_grid,inner), tf.int32), axis=-1, keepdims=True), \
			cut_type) \
		)
	return tf.cast(cond, tf.float32)

def make_sphere(shape, inner, outer=None):
	dim = 3
	assert isinstance(shape, list) and has_shape(shape, [dim])
	min_shape = min(shape)
	assert isinstance(inner, numbers.Number)
	if outer is None:
		outer = 1.0
	else:
		assert isinstance(outer, numbers.Number)
	assert (1>=outer and outer>inner and inner>=0)
	
	inner = min_shape*inner*0.5
	outer = min_shape*outer*0.5
	assert (min_shape>=outer*2 and outer>inner and inner>=0)
	
	start = [(_-1)/2.0 for _ in shape]
	coord_grid = tf.meshgrid( #linspace: end is inclusive
			*(tf.linspace(-s,s,S) for S,s in zip(shape,start)),
		indexing='ij')
	
	coord_grid = tf.transpose(coord_grid, list(range(dim+1))[::-1])#(3,2,1,0))
	coord_grid = tf.norm(coord_grid, axis=-1, keepdims=True)
	
	cond = tf.logical_and( \
		tf.less_equal(coord_grid,outer), \
		tf.greater_equal(coord_grid,inner) \
		)
	return tf.cast(cond, tf.float32)

def make_sphere_smooth(shape, inner, outer=None):
	dim = 3
	assert isinstance(shape, list) and has_shape(shape, [dim])
	min_shape = min(shape)
	assert isinstance(inner, numbers.Number)
	if outer is None:
		outer = 1.0
	else:
		assert isinstance(outer, numbers.Number)
	assert (1>=outer and outer>inner and inner>=0)
	
	#inner = min_shape*inner*0.5
	#outer = min_shape*outer*0.5
	#assert (min_shape>=outer*2 and outer>inner and inner>=0)
	
	start = [(_-1)/2.0 for _ in shape]
	coord_grid = tf.meshgrid( #linspace: end is inclusive
			*(tf.linspace(np.float32(-1.0),np.float32(1.0),S) for S,s in zip(shape,start)),
		indexing='ij')
	
	coord_grid = tf.transpose(coord_grid, list(range(dim+1))[::-1])#(3,2,1,0))
	coord_grid = tf.norm(coord_grid, axis=-1, keepdims=True)
	
	a = -1.0/(outer-inner)
	b = a * (- outer)
	
	coord_grid = tf.maximum(0, tf.minimum(1, coord_grid*a + b))
	
	return tf.cast(coord_grid, tf.float32)

def make_torus(shape, inner, outer=None):
	if outer is None:
		outer = 1.0
	if inner < 0.1: inner = 0.1
	if inner > 0.6: inner = 0.6
	outer = outer - inner - 0.05
	
	scale = min(shape) * 0.5
	
	torus_SDF = tf.squeeze(make_torus_SDF(shape, outer*scale, inner*scale, "HALF"), axis=0)
	
	return tf.cast(tf.less_equal(torus_SDF, 0), tf.float32)

# SDF shapes
#https://www.ronja-tutorials.com/post/035-2d-sdf-combination/
def union_SDF(shape1, shape2):
	return tf.minimum(shape1, shape2)
def intersection_SDF(shape1, shape2):
	return tf.maximum(shape1, shape2)
def subtraction_SDF(shape1, shape2):
	return intersection_SDF(shape1, -shape2)
def interpolation_SDF(shape1, shape2, amount):
	# lerp
	return shape1 * (1-amount) + shape2 * amount

def _get_SDF_base_grid(size, center_offset="HALF"):
	data_SDF = tf.meshgrid( #linspace: end is inclusive
			tf.range(size[2], dtype=tf.float32),
			tf.range(size[1], dtype=tf.float32),
			tf.range(size[0], dtype=tf.float32),
		indexing='ij') #CWHD
	#offset by center position
	if isinstance(center_offset, str) and center_offset.upper()=="HALF":
		center_offset = [(_-1)/2 for _ in size[::-1]]
	assert isinstance(center_offset, list) and len(center_offset)==3
	for i in range(3):
		if isinstance(center_offset[i], str) and center_offset[i].upper()=="HALF":
			center_offset[i] = (size[-(i+1)]-1)/2
		assert isinstance(center_offset[i], numbers.Number)
	
	return tf.transpose(data_SDF, (3,2,1,0))- tf.constant(center_offset, dtype=tf.float32)

def make_sphere_SDF(size, radius, center_offset="HALF", surface_offset=0, binary=False):
	# surface offset has no effect here
	data_SDF = _get_SDF_base_grid(size, center_offset)
	
	data_SDF = tf.norm(data_SDF, axis=-1, keepdims=True)
	data_SDF = data_SDF - radius
	
	if binary: # only -1 and 1 on grid
		log.info("Using binary 1/-1 grid.")
		data_SDF = tf.where(tf.less(data_SDF, 0), -tf.ones_like(data_SDF), tf.ones_like(data_SDF))
	
	data_SDF = tf.expand_dims(data_SDF, axis=0) #NDHWC
	return data_SDF

def make_cube_SDF(size, radius, center_offset="HALF", surface_offset=0, binary=False):
	# surface_offset for smooth edges
	data_SDF = _get_SDF_base_grid(size, center_offset)
	radius = np.asarray(radius, dtype=np.float32) - surface_offset
	
	# distance to edge
	data_SDF = tf.abs(data_SDF) - radius
	data_SDF_out = tf.norm(tf.maximum(data_SDF,0), axis=-1, keepdims=True)
	data_SDF_in = tf.minimum(tf.reduce_max(data_SDF, axis=-1, keepdims=True), 0)
	data_SDF = data_SDF_out + data_SDF_in
	data_SDF = data_SDF - surface_offset #move zero-level outwards
	
	if binary: # only -1 and 1 on grid
		log.info("Using binary 1/-1 grid.")
		data_SDF = tf.where(tf.less(data_SDF, 0), -tf.ones_like(data_SDF), tf.ones_like(data_SDF))
	data_SDF = tf.expand_dims(data_SDF, axis=0)
	return data_SDF

def make_octahedron_SDF(size, radius, center_offset="HALF", surface_offset=0, binary=False):
	# surface_offset for smooth edges
	data_SDF = _get_SDF_base_grid(size, center_offset)
	radius = radius - surface_offset
	
	# https://www.shadertoy.com/view/wsSGDG
	pos_abs = tf.abs(data_SDF)
	# simple manhatten distance to surface
	m = (tf.reduce_sum(pos_abs, axis=-1, keepdims=True) - radius) * (1.0/3.0)
	# offset position (per-component) with distance to surface
	o = pos_abs-m
	# 
	k = tf.minimum(o, 0)
	o = o + tf.reduce_sum(k, axis=-1, keepdims=True)*0.5 - k*1.5
	o = tf.clip_by_value(o, 0.0, radius)
	data_SDF = tf.norm(pos_abs - o, axis=-1, keepdims=True) * tf.where(tf.less(m,0),-tf.ones_like(m),tf.ones_like(m))
	
	data_SDF = data_SDF - surface_offset #move zero-level outwards
	
	##data_SDF = tf.reduce_sum(tf.abs(data_SDF), axis=-1, keepdims=True)
	#data_SDF = tf.abs(data_SDF - 30)
	##data_SDF = data_SDF - radius
	if binary: # only -1 and 1 on grid
		log.info("Using binary 1/-1 grid.")
		data_SDF = tf.where(tf.less(data_SDF, 0), -tf.ones_like(data_SDF), tf.ones_like(data_SDF))
	data_SDF = tf.expand_dims(data_SDF, axis=0)
	return data_SDF

#https://github.com/marklundin/glsl-sdf-primitives/blob/master/sdTorus.glsl
def make_torus_SDF(size, radius1, radius2, center_offset="HALF", binary=False):
	data_SDF = _get_SDF_base_grid(size, center_offset)
	
	p_xy, p_z = tf.split(data_SDF, [2,1], axis=-1)
	data_SDF = tf.concat([tf.norm(p_xy, axis=-1, keepdims=True) - radius1, p_z], axis=-1)
	data_SDF = tf.norm(data_SDF, axis=-1, keepdims=True) - radius2
	
	data_SDF = tf.expand_dims(data_SDF, axis=0)
	return data_SDF

def make_2spheres_SDF(size, binary=False):
	# WIP
	min_size = min(size)
	r = min_size/3 -1
	half_size = [(_-1)/2 for _ in size[::-1]]
	offset = min_size/6
	offset1 = [half_size[0]-offset, half_size[1]+offset, half_size[2]]
	offset2 = [half_size[0]+offset, half_size[1]-offset, half_size[2]]
	shape1 = make_sphere_SDF(size, r, center_offset=offset1, binary=binary)
	shape2 = make_sphere_SDF(size, r, center_offset=offset2, binary=binary)
	return union_SDF(shape1, shape2)

# def make_hollow_sphere_SDF(size, inner, outer, offset):
	# data_SDF = tf.meshgrid( #linspace: end is inclusive
			# tf.range(size[2], dtype=tf.float32),
			# tf.range(size[1], dtype=tf.float32),
			# tf.range(size[0], dtype=tf.float32),
		# indexing='ij') #CWHD
	# data_SDF = tf.transpose(data_SDF, (3,2,1,0))
	# #offset by center position
	# r_half = [(_-1)/2 for _ in size[::-1]]
	# sphere_outer = data_SDF - tf.constant(r_half, dtype=tf.float32)
	# sphere_outer = tf.norm(sphere_outer, axis=-1, keepdims=True) - outer
	# r_half[0] += offset
	# sphere_inner = data_SDF - tf.constant(r_half, dtype=tf.float32)
	# sphere_inner = tf.norm(sphere_inner, axis=-1, keepdims=True) - inner
	# sphere_inner = sphere_inner
	
	# # sphere = tf.where(tf.less_equal(sphere_inner, 0), -sphere_inner,
		# # tf.where(tf.less_equal(sphere_outer, 0), tf.minimum(-sphere_inner, sphere_outer),
		# # sphere_outer))
	# sphere = tf.where(tf.greater(sphere_outer, 1), sphere_outer,
		# tf.where(tf.less(sphere_inner, -1), -sphere_inner,
		# tf.maximum(-sphere_inner, sphere_outer)))
	
	# return tf.expand_dims(sphere, axis=0)

# def make_hollow_cube_binarySDF(size, inner, outer, cut_x, cut_y, cut_z):
	# data_SDF = tf.meshgrid( #linspace: end is inclusive
			# tf.range(size[2], dtype=tf.float32),
			# tf.range(size[1], dtype=tf.float32),
			# tf.range(size[0], dtype=tf.float32),
		# indexing='ij') #CWHD
	# #offset by center position
	# r_half = tf.constant([(_-1)/2 for _ in size[::-1]], dtype=tf.float32)
	# radius = (inner+outer)*0.5
	# t = (outer-inner)*0.5
	# data_SDF = tf.transpose(data_SDF, (3,2,1,0)) - r_half
	# x,y,z = tf.split(data_SDF, 3, axis=-1)
	# # z = tf.abs(z)
	# # y = tf.abs(y)
	# # x = tf.where(tf.less_equal(x, radius), tf.minimum(tf.minimum(x,y),z), x)
	# # x = tf.abs(x)
	# # data_SDF = tf.concat([x,y,z], axis=-1)
	# # data_SDF = tf.reduce_max(data_SDF, axis=-1, keepdims=True)
	# # data_SDF = tf.abs(data_SDF - radius) - t
	# x_abs = tf.abs(x)
	# y_abs = tf.abs(y)
	# z_abs = tf.abs(z)
	# x_outer = tf.cast(tf.less_equal(x_abs, outer), dtype=tf.float32)
	# y_outer = tf.cast(tf.less_equal(y_abs, outer), dtype=tf.float32)
	# z_outer = tf.cast(tf.less_equal(z_abs, outer), dtype=tf.float32)
	# cube_outer = x_outer*y_outer*z_outer
	
	# x_inner = tf.cast(tf.less_equal(x if cut_x else x_abs, inner), dtype=tf.float32)
	# y_inner = tf.cast(tf.less_equal(y if cut_y else y_abs, inner), dtype=tf.float32)
	# z_inner = tf.cast(tf.less_equal(z if cut_z else z_abs, inner), dtype=tf.float32)
	# cube_inner = x_inner*y_inner*z_inner
	
	# cube = cube_outer*(1-cube_inner)
	# cube = cube * -2 + 1
	
	# return tf.expand_dims(cube, axis=0)

def make_cube_edges_SDF(size, inner, outer):
	data_SDF = tf.meshgrid( #linspace: end is inclusive
			tf.range(size[2], dtype=tf.float32),
			tf.range(size[1], dtype=tf.float32),
			tf.range(size[0], dtype=tf.float32),
		indexing='ij') #CWHD
	#offset by center position
	r_half = tf.constant([(_-1)/2 for _ in size[::-1]], dtype=tf.float32)
	data_SDF = tf.transpose(data_SDF, (3,2,1,0)) - r_half
	x,y,z = tf.split(data_SDF, 3, axis=-1)
	
	x_abs = tf.abs(x)
	y_abs = tf.abs(y)
	z_abs = tf.abs(z)
	x_outer = tf.cast(tf.less_equal(x_abs, outer), dtype=tf.float32)
	y_outer = tf.cast(tf.less_equal(y_abs, outer), dtype=tf.float32)
	z_outer = tf.cast(tf.less_equal(z_abs, outer), dtype=tf.float32)
	cube_outer = x_outer*y_outer*z_outer
	
	x_inner = tf.cast(tf.less_equal(x_abs, inner), dtype=tf.float32)
	z_inner = tf.cast(tf.less_equal(z_abs, inner), dtype=tf.float32)
	y_inner = tf.cast(tf.less_equal(y_abs, inner), dtype=tf.float32)
	
	x_tube = z_inner*y_inner
	y_tube = x_inner*z_inner
	z_tube = x_inner*y_inner
	
	cross_inner_inv = (1-x_tube)*(1-y_tube)*(1-z_tube)
	
	cube = cube_outer*cross_inner_inv
	cube = cube * -2 + 1
	
	return tf.expand_dims(cube, axis=0)


class CameraGenerator:
	def __init__(self, resolution, pivot, pivot_range, rotY_range, rotX_range, distance_range, frustum_depth, horizontal_fov_range, batch_range, seed=0):
		# all values in world space, angles in deg
		self.__rng = np.random.default_rng(seed)
		self.__resolution = np.asarray(resolution)
		self.__pivot = np.asarray(pivot)
		self.__pivot_range = copy.copy(pivot_range)
		self.__rotY_range = copy.copy(rotY_range)
		self.__rotX_range = copy.copy(rotX_range)
		self.__distance_range = copy.copy(distance_range)
		self.__depth_half = frustum_depth * 0.5
		self.__fov_range = copy.copy(horizontal_fov_range)
		self.__aspect = resolution[2]/resolution[1]
		self.__batch_range = copy.copy(batch_range)
	
	def __sample_float(self, vmin, vmax, **rngargs):
		return (vmax-vmin)*self.__rng.random(**rngargs)+vmin
	
	def _get_random_camera(self):
		distance = self.__sample_float(*self.__distance_range)
		cam = Camera(
				GridTransform(self.__resolution.tolist(), translation=[0,0,distance], parent=
					Transform(rotation_deg=[self.__sample_float(*self.__rotX_range),0,0], parent=
						Transform(rotation_deg=[0,self.__sample_float(*self.__rotY_range),0]), translation=self.__pivot + self.__sample_float(*self.__pivot_range, size=3))),
				nearFar=[distance-self.__depth_half,distance+self.__depth_half], fov=self.__sample_float(*self.__fov_range), aspect=self.__aspect)
		return cam
	
	def get_camera_batch(self, batch_size=None):
		if batch_size is None:
			batch_size = self.__rng.integers(*self.__batch_range, endpoint=True)
		return [self._get_random_camera() for _ in range(batch_size)]

class CameraSelector:
	def __init__(self, SF_cameras, batch_range, seed=0):
		raise NotImplementedError
		self.__cameras = copy.copy(SF_cameras)
		self.__rng = np.random.default_rng(seed)
		self.__batch_range = copy.copy(batch_range)
	
	def _get_random_camera(self):
		return self.__cameras[self.__rng.choice(len(self.__cameras))]
	
	def get_camera_batch(self, batch_size=None):
		if batch_size is None:
			batch_size = self.__rng.integers(*self.__batch_range, endpoint=True)
		if batch_size>len(self.__cameras):
			raise ValueError("Requested %d cameras from CameraSelector with %d."%(batch_size, len(self.__cameras)))
		return [self.__cameras[_] for _ in self.__rng.choice(len(self.__cameras), size=batch_size, replace=False)]

class LightGenerator:
	def __init__(self):
		raise NotImplementedError

class SyntheticFrameSample:
	def __init__(self, image_raw, image_preproc, image_bkg, image_hull, image_mask, density, velocity,  sim, frame, cam_ids, device):
		self.__sim = sim
		self.__frame = frame
		self.__cam_ids = cam_ids
		if image_raw is not None:
			self.__image_raw = ImageSample(sim=self.__sim, frame=self.__frame, cam_ids=self.__cam_ids, path_mask=None, device=device)
			self.__image_raw._set(image_raw)
		if image_preproc is not None:
			self.__image_preproc = ImageSample(sim=self.__sim, frame=self.__frame, cam_ids=self.__cam_ids, path_mask=None, device=device)
			self.__image_preproc._set(image_preproc)
		if image_bkg is not None:
			self.__image_bkg = ImageSample(sim=self.__sim, frame=self.__frame, cam_ids=self.__cam_ids, path_mask=None, device=device)
			self.__image_bkg._set(image_bkg)
		if image_hull is not None:
			self.__image_hull = ImageSample(sim=self.__sim, frame=self.__frame, cam_ids=self.__cam_ids, path_mask=None, device=device)
			self.__image_hull._set(image_hull)
		if image_mask is not None:
			self.__image_mask = ImageSample(sim=self.__sim, frame=self.__frame, cam_ids=self.__cam_ids, path_mask=None, device=device)
			self.__image_mask._set(image_mask)
		if density is not None:
			self.__density = VolumeSample(sim=self.__sim, frame=self.__frame, path_mask=None, device=device)
			self.__density._set(density)
		if velocity is not None:
			self.__velocity = VolumeSample(sim=self.__sim, frame=self.__frame, path_mask=None, device=device)
			self.__velocity._set(velocity)
	
	def get_dtype(self, dtype):
		if dtype=="RAW":
			return self.__image_raw.get()
		elif dtype=="PREPROC":
			return self.__image_preproc.get()
		elif dtype=="BKG":
			return self.__image_bkg.get()
		elif dtype=="HULL":
			return self.__image_hull.get()
		elif dtype=="MASK":
			return self.__image_mask.get()
		elif dtype=="DENSITY":
			return self.__density.get()
		elif dtype=="VELOCITY":
			return self.__velocity.get()
		else:
			raise ValueError("Unknown dtype %s"%(dtype,))

class SyntheticSequenceSample:
	def __init__(self, frame_samples):
		# generate a new sample
		if len(frame_samples)<1:
			raise ValueError("Empty sequence sample")
		self.__frame_samples = frame_samples
	def __len__(self):
		return len(self.__frame_samples)
	def __getitem__(self, index):
		length = len(self)
		if (-length)>index or index>=length: raise IndexError("SyntheticSequenceSample with %d frames: index %s out of bounds"%(length, index))
		return self.__frame_samples[index]
	def get_frame_stride(self, index):
		return 1

class SyntheticSequenceSampleMS:
	def __init__(self, frame_samples):
		# generate a new sample
		if len(frame_samples)<1:
			raise ValueError("Empty scales sample")
		self.__frame_samples = frame_samples
	def __len__(self):
		return len(self.__frame_samples)
	def __getitem__(self, index):
		length = len(self)
		if (-length)>index or index>=length: raise IndexError("SyntheticSequenceSampleMS with %d scales: index %s out of bounds"%(length, index))
		return self.__frame_samples[index]
	def get_scale(self, index):
		return self[index]

class SyntheticShapesDataset(ScalaFlowDataset): #
	def __init__(self, index_dataset, batch_size, dtypes, device, \
			base_grid_transform, sequence_length, cameras, lights, \
			density_range, inner_range, scale_range, translation_range, rotation_range, \
			channels=1, SDF=False, advect_density=False, density_sampler=None, density_renderer=None, sample_overrides={}, \
			SDF_cache=None, generate_shape=True, generate_sequence=True, sims=[], frames=[], steps=[1]):
		self.__sequence_length = sequence_length
		self.__index_dataset = index_dataset.make_one_shot_iterator()
		#self.__batch_size = batch_size
		self.set_batch_size(batch_size)
		self.__dtypes = dtypes
		self.__device = device
		self.__sample_overrides = sample_overrides
		self.__lights = lights
		self.__advect_density = advect_density
		self.__density_sampler = density_sampler
		assert self.__density_sampler.boundary_mode=="CLAMP" or not SDF
		self.__density_renderer = density_renderer
		self.__SDF = SDF
		
		self.__SDF_cache = SDF_cache
		self.__generate_shape = generate_shape
		self.__generate_sequence = generate_sequence
		self.__sims = sims
		self.__frames = frames
		self.__steps = steps
		# similar interface as ScalarFlowDataset
		# sequence must be generated as one.
		# no caching needed.
		
		if isinstance(base_grid_transform, collections.abc.Iterable):
			if not all(isinstance(_, GridTransform) for _ in base_grid_transform): 
				raise TypeError("base_grid_transform must be GridTransform or list of GridTransform.")
			assert all(isinstance(_.parent, Transform) and isinstance(_.parent.parent, Transform) for _ in base_grid_transform), "Invalid Transform setup"
			self.__volume_MS = True
			self.__base_grid_transform = [_.copy_no_data() for _ in base_grid_transform]
			self.__base_shape = [_.grid_size for _ in base_grid_transform]
		elif isinstance(base_grid_transform, GridTransform):
			assert isinstance(base_grid_transform.parent, Transform) and isinstance(base_grid_transform.parent.parent, Transform), "Invalid Transform setup"
			self.__volume_MS = False
			self.__base_grid_transform = base_grid_transform.copy_no_data()
			self.__base_shape = base_grid_transform.grid_size
		else:
			raise TypeError("base_grid_transform must be GridTransform or list of GridTransform.")
		
		assert isinstance(cameras, collections.abc.Iterable)
		if all(isinstance(_, collections.abc.Iterable) for _ in cameras):
			if not all(isinstance(c, Camera) for _ in cameras for c in _):
				raise TypeError("")
			if self.__volume_MS and not len(cameras)==len(self.__base_grid_transform):
				raise ValueError("Number of MS scales for volume and images must match.")
			self.__images_MS = True
			self.__cameras = copy.deepcopy(cameras)
		elif self.__volume_MS:
			raise ValueError("MS camera setup needed when using MS volume setup.")
		elif all(isinstance(_, Camera) for _ in cameras):
			self.__images_MS = False
			self.__cameras = copy.deepcopy(cameras)
		else:
			raise TypeError("")
		
		#assert isinstance(base_shape, list) and has_shape(base_shape, [3])
		#min_shape = min(base_shape)
		assert isinstance(density_range, (list, tuple)) and has_shape(density_range, [2]) and 0<density_range[0] and density_range[0]<=density_range[1]
		self.__density_range = density_range
		assert isinstance(inner_range, (list, tuple)) and has_shape(inner_range, [2]) and 0<=inner_range[0] and inner_range[0]<=inner_range[1] and inner_range[1]<=1.0
		self.__inner_range = inner_range
		assert isinstance(scale_range, (list, tuple)) and has_shape(scale_range, [2]) and 0<scale_range[0] and scale_range[0]<=scale_range[1]
		self.__scale_range = scale_range
		self.__scale_uniform = True
		log.warning("uniform scaling: %s.", self.__scale_uniform)
		self.__rotation_range = rotation_range
		self.__base_grid_size_world = self.base_grid_transform.grid_size_world().value[::-1]
		max_translation = self.__base_grid_size_world
		max_max_translation = max(max_translation)
		log.info("max_translation: %s", max_translation)
		assert isinstance(translation_range, (list, tuple)) and has_shape(translation_range, [2]) and translation_range[0]<=translation_range[1]
		self.__translation_range = translation_range
		if translation_range[0]<(-max_max_translation) or max_max_translation<translation_range[1]:
			log.warning("translation_range %s exceeds max safe translation %s, clamping.", translation_range, max_translation)
			self.__translation_range = [np.clip(_, -max_max_translation, max_max_translation) for _ in translation_range]
		
		self._no_depth_translation = False
		if self._no_depth_translation:
			log.info("Depth (z) translation disabled.")
		
		self._no_screen_translation = False
		if self._no_screen_translation:
			log.info("Screen (xy) translation disabled.")
		
		self.__object_shape = [32]*3 #[int(min_shape * max(scale_range))]*3
		#generate shape on grid with sufficient resolution
		#scale_range = [] #scale!=1 causes issues with divergence? or sampling? would be diffusion
		#base_density_cube = make_cube(object_shape) #the shape does not change, so we can use a global template
		#self.__image_shape = list(cameras[0].transform.grid_size)
		calibration_center = np.asarray(copy.copy(self.base_grid_transform.parent.parent.translation), dtype=np.float32)
		self.__calibration_scale = np.asarray(copy.copy(base_grid_transform.parent.parent.scale), dtype=np.float32)
		log.info("calibration center: %s", calibration_center)
		self.__base_transform = Transform(translation=calibration_center.tolist())#[0.32726666666666665, 0.3898192272727273, -0.240541])
		self.__max_tries = 20
		
		# can generate with randomized transform?
		self.step()
	
	def set_batch_size(self, batch_size):
		assert isinstance(batch_size, numbers.Integral) and batch_size>0
		self.__batch_size = batch_size
	
	def _transform_samples(self, frame_samples, dtype):
		raise NotImplementedError
	
	def _get_frame_sample(self, sim, frame, cam_ids):
		raise NotImplementedError
	
	def __step_index(self):
		self.__current_index = self.__index_dataset.next()[0].numpy()
		self.__rng = np.random.default_rng(self.__current_index)
	
	def _get_next_index_sample(self):
		raise NotImplementedError
	
	def get_next_sample(self):
		raise NotImplementedError
	
	def get_next_batch(self, batch_size=None):
		raise NotImplementedError
	
	@property
	def RAW(self):
		return "RAW" in self.__dtypes
	@property
	def PREPROC(self):
		return "PREPROC" in self.__dtypes
	@property
	def BKG(self):
		return "BKG" in self.__dtypes
	@property
	def HULL(self):
		return "HULL" in self.__dtypes
	@property
	def MASK(self):
		return "MASK" in self.__dtypes
	@property
	def DENSITY(self):
		return "DENSITY" in self.__dtypes
	@property
	def VELOCITY(self):
		return "VELOCITY" in self.__dtypes
	@property
	def TRANSFORM(self):
		return "TRANSFORM" in self.__dtypes
	
	@property
	def is_MS(self):
		return self.__volume_MS or self.__images_MS
	@property
	def num_MS_scales(self):
		if not self.is_MS: raise RuntimeError("MS not set.")
		return len(self.__cameras)
	@property
	def base_grid_transform(self):
		return self.__base_grid_transform[0] if self.__volume_MS else self.__base_grid_transform
	
	@property
	def __shape_types(self):
		shape_types = self.__sample_overrides.get('shape_type', [0,1,2,3,4])
		if not isinstance(shape_types, (list, tuple)):
			shape_types = [shape_types]
		return shape_types
	@property
	def __shape_inner(self):
		return self.__sample_overrides.get('shape_inner', [self.__rng.uniform(*self.__inner_range) for _ in range(3)])
	@property
	def __base_scale(self):
		scale = [self.__rng.uniform(*self.__scale_range) for _ in range(3)]
		if self.__scale_uniform:
			scale = [scale[0]]*3
		return self.__sample_overrides.get('base_scale', scale)
	
	@property
	def __initial_translation(self):
		#return self.__sample_overrides.get("initial_translation", self.__rng.uniform(low=-self.__max_translation, high=self.__max_translation))
		return self.__sample_overrides.get("initial_translation", self.__rng.uniform(low=self.__translation_range[0], high=self.__translation_range[1], size=3))
	@property
	def __initial_rotation_rotvec(self):
		return self.__sample_overrides.get("initial_rotation_rotvec", self.__get_rotation_rotvec([0,2*np.pi]))
	@property
	def __density_scale(self):
		return self.__sample_overrides.get('density_scale', self.__rng.uniform(*self.__density_range))
	@property
	def __rotvec(self):
		return self.__sample_overrides.get("rotvec", self.__rng.normal(size=3))
	
	def __max_translation(self, scale):
		return (self.__base_grid_size_world - scale)*0.5 
	def __clamp_translation(self, translation, scale):
		max_translation = self.__max_translation(scale)
		return np.clip(translation, -max_translation, max_translation)
	
	def __SDF_to_density(self, density):
		raise NotImplementedError
	
	def __get_base_denstiy(self):
		if self.__generate_shape:
			with SAMPLE("generate volume"):
				shape_types = self.__shape_types
				shape_type = self.__rng.choice(shape_types)
				shape_inner = self.__shape_inner
				#shape_outer
				if self.__SDF:
					base_radius = min(self.__object_shape)//2
					if shape_type==0:
						base_density = make_sphere_SDF(self.__object_shape, radius=base_radius - 3) #shape_inner[0])
					elif shape_type==1:
						base_density = make_cube_SDF(self.__object_shape, radius=base_radius - 2) #shape_inner[0])
					elif shape_type==2:
						base_density = make_octahedron_SDF(self.__object_shape, radius=base_radius - 2) #shape_inner[0])
					elif shape_type==3:
						base_density = make_hollow_cube_binarySDF(self.__object_shape, outer=base_radius - 3, inner=base_radius - 7, cut_x=True, cut_y=False, cut_z=False)
					elif shape_type==4:
						base_density = make_hollow_cube_binarySDF(self.__object_shape, outer=base_radius - 3, inner=base_radius - 7, cut_x=True, cut_y=True, cut_z=False)
					elif shape_type==5:
						base_density = make_hollow_cube_binarySDF(self.__object_shape, outer=base_radius - 3, inner=base_radius - 7, cut_x=True, cut_y=True, cut_z=True)
					elif shape_type==6:
						base_density = make_cube_edges_SDF(self.__object_shape, outer=base_radius - 3, inner=base_radius - 8)
					elif shape_type==7:
						base_density = make_torus_SDF(self.__object_shape, radius1=base_radius - 4, radius2=3)
					elif shape_type==8:
						base_density = make_2spheres_SDF(self.__object_shape)
					else:
						raise RuntimeError("Unknown shape_type")
					base_density = tf.clip_by_value(base_density, -1, 1)
					base_density = tf.squeeze(base_density, axis=0) #NDHWC -> DHWC
				else:
					if shape_type==0:
						#base_density = make_sphere(object_shape, inner=shape_inner[0])
						base_density = make_sphere_smooth(self.__object_shape, inner=shape_inner[0])
					elif shape_type<4:
						base_density = make_cube(self.__object_shape, inner=shape_inner, cut_type=shape_type)
					elif shape_type==4:
						base_density = make_sphere(self.__object_shape, inner=shape_inner[0])
					elif shape_type==5: #torus
						base_density = make_torus(self.__object_shape, inner=shape_inner[0])
					else:
						raise RuntimeError("Unknown shape_type")
				den_shape = shape_list(base_density)
				#log.info("Generated object %d with shape %s and inner shape %s", shape_type, shape_list(base_density), shape_inner)
				assert len(den_shape)==4 and all(d==o for d,o in zip(den_shape[:3], self.__object_shape)) and den_shape[-1]==1, "Invalid density shape %s, must be DHWC %s, C=1"%(den_shape,self.__object_shape)
		else:
			with SAMPLE("load volume"):
				sim = self.__rng.choice(self.__sims)
				frame = self.__rng.choice(self.__frames)
				base_density = self.__SDF_cache._get_sample(sim=sim, frame=frame).get()
				if not self.__SDF:
					base_density = self.__SDF_to_density(base_density)
		
		return tf.expand_dims(base_density, 0)#DHWC -> NDHWC
	
	def __get_transforms(self, base_transform, object_shape):
		base_scale = self.__base_scale
		#translation: make sure to stay (mostly) in domain, don't always start in the center
		translation = self.__clamp_translation(np.asarray(self.__initial_translation, dtype=np.float32), base_scale) # np.asarray(sample_overrides.get("initial_translation", rng.uniform(low=-max_translation, high=max_translation)), dtype=np.float32) #np.zeros([3])
		# start with random rotation
		rotation = Rotation.from_rotvec(self.__initial_rotation_rotvec) # Rotation.from_rotvec(sample_overrides.get("initial_rotation_rotvec", get_rotation_rotvec(rng,  [0,2*np.pi], sample_overrides=sample_overrides)))
		transforms = []
		if self.__advect_density:
			for step in range(self.__sequence_length):
				#new_rotation = Rotation.from_rotvec(get_rotation_rotvec(rng, sample_overrides=sample_overrides))
				if (step==0):
					t = GridTransform(object_shape, translation=translation.tolist(), rotation_quat=rotation.as_quat(), scale=base_scale, normalize='MAX', center=True, parent = base_transform)
				else:
					# translate grid center to shape center (its translation), rotate, translate back, add new translation
					t = self.base_grid_transform.copy_no_data()
					#log.info("Base grid T: %s", t)
					#raise RuntimeError
					if t.parent is None:
						t.parent=Transform()
					t.translation = (-translation).tolist()
					t.scale = self.__calibration_scale.tolist()
					t.parent.parent.scale = [1,1,1]
					
					t.parent.rotation_quat = rotation.as_quat()
					translation = self.__clamp_translation(translation + new_translation, base_scale) #, -self.__max_translation, self.__max_translation)
					t.parent.translation = translation.tolist()
					#log.info("%s", t)
				transforms.append(t)
				
				new_translation = self.__rng.uniform(low=self.__translation_range[0], high=self.__translation_range[1], size=3)
				if self._no_depth_translation:
					new_translation[2] = 0
				if self._no_screen_translation:
					new_translation[0] = 0
					new_translation[1] = 0
				rotation = Rotation.from_rotvec(self.__get_rotation_rotvec())
				
		else:
			for step in range(self.__sequence_length):
				t = GridTransform(object_shape, translation=translation.tolist(), rotation_quat=rotation.as_quat(), scale=base_scale, normalize='MAX', center=True, parent = base_transform)
				transforms.append(t)
				
				new_translation = self.__rng.uniform(low=self.__translation_range[0], high=self.__translation_range[1], size=3)
				if self._no_depth_translation:
					new_translation[2] = 0
				if self._no_screen_translation:
					new_translation[0] = 0
					new_translation[1] = 0
				translation = self.__clamp_translation(translation + new_translation, base_scale)
				rotation = Rotation.from_rotvec(self.__get_rotation_rotvec())*rotation
		return transforms
	
	def __set_transforms_scale_MS(self, base_transform, transforms, scale):
		assert self.__volume_MS
		assert 0<=scale and scale<self.num_MS_scales
		scale_shape = self.__base_grid_transform[scale].grid_size #ZYX
		
		base_transform.grid_size = scale_shape
		
		if self.__advect_density:
			for idx in range(1, len(transforms)):
				transforms[idx].grid_size = scale_shape
		else:
			pass
	
	def __get_cell_scale_between(self, t_from, t_to):
		return np.mean(t_to.cell_size_world() / t_from.cell_size_world()).tolist()
	
	def __get_sequence(self, base_density, base_transform, transforms):
		density_scale = self.__density_scale
		#sample to main grid with transforms
		densities = []
		velocities = []
		if self.__advect_density:
			# sample first density to base_grid
			d = tf.squeeze(self.__density_sampler._sample_transform(base_density, [transforms[0]], [base_transform], fix_scale_center=True), (0,1))
			
			if not self.__SDF:
				# scale/normalize total density
				total_densities = tf.reduce_mean(d)
				if total_densities<=0:
					log.warning("Densities generated empty: %s", total_densities.numpy())
					return True, [], []
				scale = (density_scale/total_densities)
			else:
				scale = 1 / self.__get_cell_scale_between(transforms[0], base_transform)
			
			d = d*scale
			
			densities.append(d)
			
			for t_idx in range(len(transforms)-1):
				if True:
					#generate vel - transform FROM base_grid+translation+rot TO base_grid
					v = self.__density_sampler.get_transform_LuT([transforms[t_idx+1]], [base_transform], relative=True, fix_scale_center=True)
					v = v[...,:-1]
					#log.info("lut_shape: %s, lut mean: %s, abs: %s", shape_list(v), tf.reduce_mean(v).numpy(), tf.reduce_mean(tf.abs(v)).numpy())
					if self.VELOCITY:
						velocities.append(-tf.squeeze(v, (0,)))
					
					#advect density to next
					#TODO: use full warping method with MC support
					d = tf.squeeze(self.__density_sampler._sample_LuT(tf.expand_dims(densities[t_idx], 0), v, True, relative=True, cell_center_offset=0.5), (0,))
					assert len(shape_list(d))==4, "dens shape: %s"%(shape_list(d),)
				else:
					d = tf.squeeze(self.__density_sampler._sample_transform(tf.expand_dims(densities[t_idx], 0), [transforms[t_idx+1]], [base_transform], fix_scale_center=True), (0,1))
					
				if (not self.__SDF) and tf.reduce_mean(d)<=0:
					log.warning("Advected densities are 0 in step %d", t_idx)
				densities.append(d)
			
			velocities.append(tf.zeros(self.__base_shape + [3]))
			
		else:
			for t_idx, t in enumerate(transforms):
				if self.VELOCITY:
					raise NotImplementedError
					#this is wrong-> v = self.__density_sampler.get_transform_LuT([t], [base_transform], relative=True, fix_scale_center=True)
					velocities.append(-tf.squeeze(v[...,:-1], (0,))) #vel is negative relative lookup position without LoD (assuming SL advection)
				
				renderargs = {}
				if self.__SDF: # disable mipmapping when sampling SDF to prevent border/extrapolation artifacts
					renderargs["mipmapping"] = "NONE"
					# TODO: scale SDF values to match new grid??
				d = tf.squeeze(self.__density_sampler._sample_transform(base_density, [t], [base_transform], fix_scale_center=True, **renderargs), (0,1))
				if self.__SDF:
					d = d * (1/self.__get_cell_scale_between(t, base_transform))
				densities.append(d)
			
			if not self.__SDF:
				total_densities = [tf.reduce_mean(d) for d in densities]
				total_densities_np = [d.numpy() for d in total_densities]
				if any(d<=0 for d in total_densities_np):
					log.warning("Densities generated are empty: %s", total_densities_np)
					return True, [], []
				# total density is prop. to avg when the grid resolution stays the same
				#  total density in all should be the same to respect divergence (norm total density and scale to same rand)
				densities = [d*(density_scale/t) for d, t in zip(densities, total_densities)]
		
		densities = tf.stack(densities)
		log.debug("densities shape: %s", shape_list(densities))
		if self.VELOCITY:
			velocities = tf.stack(velocities)
			log.debug("velocities shape: %s", shape_list(velocities))
		
		return False, densities, velocities
	
	def __get_rotation_rotvec(self, rad_range=None):
		if rad_range is None: rad_range = np.deg2rad(self.__rotation_range)
		rotation = np.asarray(self.__rotvec, dtype=np.float32)
		rotation_rad = self.__rng.uniform(low=rad_range[0], high=rad_range[1])
		if np.sum(np.abs(rotation)) > 1e-5:
			rotation *= rotation_rad/np.linalg.norm(rotation)
		return rotation
	
	def __get_cameras(self):
		if isinstance(self.__cameras, (CameraGenerator, CameraSelector)):
			raise NotImplementedError("needs some more work in renderer")
			return self.__cameras.get_camera_batch()
		else:
			return self.__cameras
	
	def __render_sequence(self, transform, densities, cameras, super_sampling=1):
		with SAMPLE("render sequence"):
			image_shape = list(cameras[0].transform.grid_size)
			if super_sampling>1:
				warnings.warn("synth data rendering supersampling set to %d"%super_sampling)
				image_shape[1] *= super_sampling
				image_shape[2] *= super_sampling
				cameras = copy.deepcopy(cameras)
				for cam in cameras: cam.transform.grid_size = image_shape
				def downsample(imgs):
					return [tf.nn.avg_pool(img, (1,super_sampling, super_sampling, 1), (1,super_sampling, super_sampling, 1), 'VALID') for img in imgs] #NWHC
			else:
				def downsample(imgs):
					return imgs
			
			if self.RAW or self.BKG:
				img_bkg = [tf.zeros((len(cameras), image_shape[1], image_shape[2], 3))]*self.__sequence_length
				
				#img_bkg = tf.identity(img_bkg)
			mask = [None]*self.__sequence_length
			if self.RAW or self.PREPROC or self.HULL or self.MASK:
				#img = [] #tf.zeros((num_views, image_shape[1], image_shape[2], 3))]*sequence_length
				if not tf.reduce_all(tf.is_finite(densities)).numpy():
					log.warning("Density generated from %d is not finite.", self.__current_index)
				if (not self.__SDF) and tf.reduce_any(tf.less(densities, 0.0)).numpy():
					log.warning("Density generated from %d is negative.", self.__current_index)
				
				img = []
				for dens in tf.unstack(densities, axis=0):
					tmp_transform = transform.copy_new_data(tf.expand_dims(dens, 0))
					if self.__SDF:
						tmp = self.__density_renderer.render_SDF(tmp_transform, self.__lights, cameras, cut_alpha=not self.MASK) #V-NHWC
						img.append(tf.stack(tmp, axis=1))
					else:
						tmp = self.__density_renderer.render_density(tmp_transform, self.__lights, cameras, cut_alpha=not self.MASK) #V-NHWC
						img.append(tf.stack(tmp, axis=1))
				img = tf.concat(img, axis=0) #NVHWC
				
				if self.MASK:
					img, mask = tf.split(img, [3,1], axis=-1)
					mask = 1-mask if self.__SDF else tf.cast(tf.greater(mask, 0), dtype=tf.float32)
					mask = tf.unstack(mask, axis=0) #N - VHWC
				#log.info("Image sample shape: %s", shape_list(img))
				if not tf.reduce_all(tf.is_finite(img)).numpy():
					log.warning("Images rendered for %d are not finite.", self.__current_index)
				#log.info("Rendered sample shape: %s", shape_list(img))
				img = tf.unstack(img, axis=0) #N - VHWC
			
			img = downsample(img)
			
			if self.RAW:
				img_raw = img#downsample(img)
			else:
				img_raw = [None]*self.__sequence_length
			
			if self.BKG:
				img_bkg = downsample(img_bkg)
			else:
				img_bkg = [None]*self.__sequence_length
			
			if self.HULL:
				img_hull = img#downsample(img)
			else:
				img_hull = [None]*self.__sequence_length
			
			if self.PREPROC:
				pass #img = downsample(img)
			else:
				img = [None]*self.__sequence_length
			
			
		return img_raw, img, img_bkg, img_hull, mask
	
	# @property
	# def __image_shape(self):
		# if isinstance(self.__cameras, (CameraGenerator, CameraSelector)):
			# raise NotImplementedError
		# else:
			# return list(self.__cameras[0].transform.grid_size)
	
	def __get_synthetic_sequence(self, base_transform):
		with SAMPLE("generate sequence"):
		
			gen_sample=True
			i=0
			while gen_sample and i<self.__max_tries:
				gen_sample = False
				i+=1
				
				base_density = self.__get_base_denstiy()
				
				transforms = self.__get_transforms(self.__base_transform, shape_list(base_density)[1:-1])
				
				gen_sample, densities, velocities = self.__get_sequence(base_density, base_transform, transforms)
			if gen_sample:
				raise RuntimeError("Failed to generate a valid sample from %d after %d attempts."%(self.__current_index, self.__max_tries))
			
			if self.__volume_MS:
				densities = [densities]
				velocities = [velocities]
				for scale in range(1, self.num_MS_scales):
					self.__set_transforms_scale_MS(base_transform, transforms, scale)
					_, d, v = self.__get_sequence(base_density, base_transform, transforms)
					densities.append(d)
					velocities.append(v)
		
		return densities, velocities
	
	def __load_sequence(self, base_transform):
		step = self.__rng.choice(self.__steps)
		sim = self.__rng.choice(self.__sims)
		start_frame_idx = self.__rng.choice(len(self.__frames) - self.__sequence_length*step)
		frames = [self.__frames[idx] for idx in range(start_frame_idx, start_frame_idx + self.__sequence_length, step)]
		
		# get sub-sequence from data cache #self.__sequence_length
		# every frame is cropped to it's own bounding box (+padding)
		# world size and offset are in the metadata
		samples = [self.__SDF_cache._get_sample(sim=sim, frame=frame) for frame in frames]
		densities = [tf.expand_dims(sample.get(), 0) for sample in samples] #S-NDHWC
		positions = [sample.get_meta("position") for sample in samples] #S-xyz
		sizes = [sample.get_meta("size") for sample in samples] #S-xyz
		
		# calculate transforms to sample to common domain
		# augment with random 90deg rotation and flip? before sampling
		transforms = [GridTransform(shape_list(d)[1:-1], translation=position.tolist(), scale=size.tolist(), normalize="ALL") for d, position, size in zip(densities, positions, sizes)] #set each sample to it's own origin and size
		# common domain
		bottom_position = np.amin(positions, axis=0)
		top_position = np.amax([position + size for position, size in zip(positions, sizes)], axis=0)
		size = top_position - bottom_position
		center = bottom_position + size*0.5
		# fit to base shape to avoid distortions
		scale = size / np.asarray(self.__base_shape, dtype=np.float32)
		scale = [np.amax(scale).tolist()]*3
		common_transform = GridTransform(self.__base_shape, translation=center, scale=scale, center=True) #set target transfrom to BB of all samples
		
		# resample to common domain
		# disable mipmapping when sampling SDF to prevent border/extrapolation artifacts
		densities = [tf.squeeze(self.__density_sampler._sample_transform(d, [t], [common_transform], fix_scale_center=True, mipmapping="NONE"), (0,1)) for d, t in zip(densities, transforms)]
		densities = [d * (1/self.__get_cell_scale_between(t, common_transform)) for d, t in zip(densities, transforms)]
		
		velocities = [None]*len(densities)
		
		densities = tf.stack(densities) # SDHWC
		if self.__volume_MS:
			raise NotImplementedError
			for scale in range(1, self.num_MS_scales):
				# sample down
				pass
			
			velocities = [velocities]*self.num_MS_scales
		
		return densities, velocities
	
	def __get_volumes(self, base_transform):
		with SAMPLE("get volumes"):
			if self.__generate_sequence:
				return self.__get_synthetic_sequence(base_transform)
			else:
				if self.VELOCITY: raise NotImplementedError
				return self.__load_sequence(base_transform)
	
	def _generate_next_sample(self):
		with SAMPLE("generate sample"):
			# sim, frames, cam_ids = self._get_next_index_sample()
			# seed = self.__current_index
			# self.__rng = np.random.default_rng(seed)
			base_transform = self.__current_batch["TRANSFORM"][0]
			# main part of get_synthTargets_dataset() here:
			densities, velocities = self.__get_volumes(base_transform)
			
			
			if not self.__images_MS:
				cameras = self.__get_cameras()
				img_raw, img, img_bkg, img_hull, mask = self.__render_sequence(base_transform, densities, cameras, super_sampling=1)
				
				if self.DENSITY:
					densities = tf.unstack(densities, axis=0) #N - DHWC
				else:
					densities = [None]*self.__sequence_length
				if self.VELOCITY:
					velocities = tf.unstack(velocities, axis=0) #N - DHWC
				else:
					velocities = [None]*self.__sequence_length
				
				frame_samples = []
				zipped_data = list(zip(img_raw, img, img_bkg, img_hull, mask, densities, velocities))
				#log.info("Frame data: %d, from: %s", len(zipped_data), [len(_) for _ in [img_raw, img, img_bkg, img_hull, densities, velocities]])
				for frame, data in enumerate(zipped_data):
					frame_samples.append(SyntheticFrameSample(*data, sim=0, frame=frame, cam_ids=list(range(len(cameras))), device=self.__device))
				#log.info("Frame samples: %d", len(frame_samples))
				return SyntheticSequenceSample(frame_samples=frame_samples)
			else:
				scale_samples = []
				for scale in range(self.num_MS_scales):
					cameras = self.__get_cameras()[scale]
					if self.__volume_MS:
						d = densities[scale]
						v = velocities[scale]
					else:
						d = densities
						v = velocities
					#self.__set_transforms_scale_MS(base_transform, [], scale) #not necessary as __render_sequence sets data to the transform
					img_raw, img, img_bkg, img_hull, mask = self.__render_sequence(base_transform, d, cameras, super_sampling=1)
					
					if self.DENSITY:
						d = tf.unstack(d, axis=0) #N - DHWC
					else:
						d = [None]*self.__sequence_length
					if self.VELOCITY:
						v = tf.unstack(v, axis=0) #N - DHWC
					else:
						v = [None]*self.__sequence_length
					
					frame_samples = []
					zipped_data = list(zip(img_raw, img, img_bkg, img_hull, mask, d, v))
					for frame, data in enumerate(zipped_data):
						frame_samples.append(SyntheticFrameSample(*data, sim=0, frame=frame, cam_ids=list(range(len(cameras))), device=self.__device))
					#log.info("Frame samples: %d", len(frame_samples))
					scale_samples.append(SyntheticSequenceSample(frame_samples))
			
				return SyntheticSequenceSampleMS(scale_samples)
	
	def _generate_next_batch(self, batch_size=None):
		if batch_size is None: batch_size = self.__batch_size
		samples = [self._generate_next_sample() for _ in range(batch_size)]
		return samples
		# NFVHWC or NFDHWC?
	
	def step(self):
		with SAMPLE("generate batch"):
			self.__step_index()
			self.__current_batch = {} #{dtype:[frames[samples]]}
			# get data from next batch
			if self.TRANSFORM:
				# one transform per batch, same for all frames
				self.__current_batch["TRANSFORM"] = [self._get_randomized_transform(base_transform=self.base_grid_transform, seed=self.__current_index)]*self.__sequence_length
			else:
				self.__current_batch["TRANSFORM"] = [self.base_grid_transform.copy_no_data()]*self.__batch_size
			batch = self._generate_next_batch(self.__batch_size)
		with SAMPLE("process batch"):
			# stack data on batch dimension
			for dtype in self.__dtypes:
				if dtype=="TRANSFORM":
					pass
				else:
					self.__current_batch[dtype] = []
					for frame_idx in range(self.__sequence_length): #range(len(batch[0])):
						if self.is_MS:
							scales = []
							for scale in range(self.num_MS_scales):
								if dtype=="VELOCITY" and self.__velocity_scale_magnitude_by_frame_stride:
									frame_samples = [seq_sample[scale][frame_idx].get_dtype(dtype) * np.float32(seq_sample.get_frame_stride(frame_idx)) for seq_sample in batch]
								else:
									frame_samples = [seq_sample[scale][frame_idx].get_dtype(dtype) for seq_sample in batch]
								# if dtype=="TRANSFORM":
									# frame_samples = self._transform_samples(frame_samples, dtype)
								frame_samples = tf.stack(frame_samples)
								with tf.device(self.__device):
									frame_samples = tf.identity(frame_samples)
								scales.append(frame_samples)
							self.__current_batch[dtype].append(scales) #NVHWC
						else:
							if dtype=="VELOCITY" and self.__velocity_scale_magnitude_by_frame_stride:
								frame_samples = [seq_sample[frame_idx].get_dtype(dtype) * np.float32(seq_sample.get_frame_stride(frame_idx)) for seq_sample in batch]
							else:
								frame_samples = [seq_sample[frame_idx].get_dtype(dtype) for seq_sample in batch]
							# if dtype=="TRANSFORM":
								# frame_samples = self._transform_samples(frame_samples, dtype)
							frame_samples = tf.stack(frame_samples)
							with tf.device(self.__device):
								frame_samples = tf.identity(frame_samples)
							self.__current_batch[dtype].append(frame_samples) #NVHWC
		
	def frame_targets(self, idx, as_dict=False):
		# current_batch: dtype-frame(-MSscale)-NVHWC
		# returns: dtype(-MSscale)-NVHWC
		return {dtype:self.__current_batch[dtype][idx] for dtype in self.__dtypes} if as_dict else [self.__current_batch[dtype][idx] for dtype in self.__dtypes]

#combined synthetic shapes and SF smoke data dataset
class MultiDataset():
	def __init__(self, datasets, weights=None, seed=None):
		self.__rng = np.random.default_rng(seed)
		assert isinstance(datasets, (list, tuple)) and len(datasets)>0
		self.__datasets = datasets
		assert weights is None or isinstance(weights, (list, tuple)) and len(weights)==len(datasets)
		self.__weights = weights
		self.__current_dataset = self._get_dataset()
	@property
	def num_datasets(self):
		return len(self.__datasets)
	def _get_dataset(self):
		return self.__datasets[self.__rng.choice(self.num_datasets, p=self.__weights)]
	def step(self):
		self.__current_dataset = self._get_dataset()
		self.__current_dataset.step()
	def frame_targets(self, idx, as_dict=False):
		return self.__current_dataset.frame_targets(idx=idx, as_dict=as_dict)
	def set_batch_size(self, batch_size):
		for dataset in self.__datasets:
			dataset.set_batch_size(batch_size)


def get_synthTargets_dataset_v2(batch_size, device, base_grid_transform, sequence_length, view_indices, num_views, cameras, lights, \
		density_range, inner_range, scale_range, translation_range, rotation_range, \
		raw=True, preproc=True, bkg=True, hull=False, mask=False, channels=1, SDF=False, \
		density=False, velocity=False, advect_density=False, density_sampler=None, density_renderer=None, randomize_transform=False, seed=0, sample_overrides={}, cache_device=None, \
		data_cache=None, generate_shape=True, generate_sequence=True, sims=[], frames=[], steps=[1]):
	
	dtypes = []
	if raw:
		dtypes.append("RAW")
	if preproc:
		dtypes.append("PREPROC")
	if bkg:
		dtypes.append("BKG")
	if mask:
		dtypes.append("MASK")
	if hull:
		dtypes.append("HULL")
	if density:
		dtypes.append("DENSITY")
	if velocity:
		raise NotImplementedError("change to staggered velocity output")
		dtypes.append("VELOCITY")
	if randomize_transform:
		dtypes.append("TRANSFORM")
	
	grid_shapes = []
	if isinstance(base_grid_transform, Transform):
		grid_shapes.append(base_grid_transform.grid_size)
	elif isinstance(base_grid_transform, collections.abc.Iterable) and all(isinstance(_, Transform) for _ in base_grid_transform):
		grid_shapes.extend(_.grid_size for _ in base_grid_transform)
	
	cam_shapes = []
	if isinstance(cameras, collections.abc.Iterable) and all(isinstance(_, Camera) for _ in cameras):
		cam_shapes.append(cameras[0].transform.grid_size)
	elif isinstance(cameras, collections.abc.Iterable) and all(isinstance(_, collections.abc.Iterable) for _ in cameras) and all(isinstance(c, Camera) for _ in cameras for c in _):
		cam_shapes.extend(_[0].transform.grid_size for _ in cameras)
		
		
	#
	log.info("Initialize synthetic %s shapes dataset V2:\n\trequested data types: %s\n\t%s steps\n\tseed: %d\n\tdensity_range %s, inner_range %s, scale_range %s, translation_range %s, rotation_range %s\n\tvolume shapes: %s\n\timage shapes: %s\n\tSDF threashold: %f\n\toverrides: %s","SDF" if SDF else "Density", dtypes, sequence_length, seed, density_range, inner_range, scale_range, translation_range, rotation_range, grid_shapes, cam_shapes, density_renderer.SDF_threshold, sample_overrides)
	
	
	#get a rng
	def gen_seed(seed):
		rng = np.random.RandomState(seed)
		while True:
			yield (rng.randint(np.iinfo(np.int32).max),)
	rand_dataset = tf.data.Dataset.from_generator(gen_seed, output_types=(tf.int32,), output_shapes=([],), args=(seed,))
	
	#if data_cache is None:
		#data_cache = SDFDatasetCache(path_mask="/home/erik/data/softbody/test_{sim:03d}/SDF_np/{frame:04d}.npz", device=device)
		#data_cache = SDFDatasetCache(path_mask=("/home/erik/data/ShapeNetCore.v2/03001627", r"(?P<sim>[0-9A-Fa-f]+)/(?P<frame>models)/model_normalized\.npz"), device=device)
	
	return SyntheticShapesDataset(index_dataset=rand_dataset, batch_size=batch_size, dtypes=dtypes, device=device, \
			base_grid_transform=base_grid_transform, sequence_length=sequence_length, cameras=cameras, lights=lights, \
			density_range=density_range, inner_range=inner_range, scale_range=scale_range, translation_range=translation_range, rotation_range=rotation_range, \
			channels=channels, SDF=SDF, advect_density=advect_density, density_sampler=density_sampler, density_renderer=density_renderer, sample_overrides=sample_overrides, \
			SDF_cache=data_cache, generate_shape=generate_shape, generate_sequence=generate_sequence, sims=sims, frames=frames, steps=steps)

def get_synthTargets_dataset(base_grid_transform, sequence_length, view_indices, num_views, cameras, lights, \
		density_range, inner_range, scale_range, translation_range, rotation_range, \
		raw=True, preproc=True, bkg=True, hull=False, channels=1, \
		density=False, velocity=False, advect_density=False, density_sampler=None, seed=0, sample_overrides={}):
	"""
	simple shapes with simple motions, generated from seed
	"""
	assert isinstance(base_grid_transform, GridTransform)
	assert isinstance(base_grid_transform.parent, Transform) and isinstance(base_grid_transform.parent.parent, Transform)
	base_shape = base_grid_transform.grid_size
	#assert isinstance(base_shape, list) and has_shape(base_shape, [3])
	min_shape = min(base_shape)
	assert isinstance(density_range, (list, tuple)) and has_shape(density_range, [2]) and 0<density_range[0] and density_range[0]<=density_range[1]
	assert isinstance(inner_range, (list, tuple)) and has_shape(inner_range, [2]) and 0<=inner_range[0] and inner_range[0]<=inner_range[1] and inner_range[1]<=1.0
	assert isinstance(scale_range, (list, tuple)) and has_shape(scale_range, [2]) and 0<scale_range[0] and scale_range[0]<=scale_range[1]
	max_translation = (base_grid_transform.grid_size_world().value[::-1] - scale_range[1])*0.5 #1
	max_max_translation = max(max_translation)
	log.info("max_translation: %s", max_translation)
	assert isinstance(translation_range, (list, tuple)) and has_shape(translation_range, [2]) and translation_range[0]<=translation_range[1]
	if translation_range[0]<(-max_max_translation) or max_max_translation<translation_range[1]:
		log.warning("translation_range %s exceeds max safe translation %s, clamping.", translation_range, max_translation)
		translation_range = [np.clip(_, -max_max_translation, max_max_translation) for _ in translation_range]
	
	object_shape = [32]*3 #[int(min_shape * max(scale_range))]*3
	make_temporal_input = False
	#get a rng
	def gen_seed(seed):
		rng = np.random.RandomState(seed)
		while True:
			yield (rng.randint(np.iinfo(np.int32).max),)
	#generate shape on grid with sufficient resolution
	#scale_range = [] #scale!=1 causes issues with divergence? or sampling? would be diffusion
	#base_density_cube = make_cube(object_shape) #the shape does not change, so we can use a global template
	image_shape = list(cameras[0].transform.grid_size)
	calibration_center = np.asarray(copy.copy(base_grid_transform.parent.parent.translation), dtype=np.float32)
	calibration_scale = np.asarray(copy.copy(base_grid_transform.parent.parent.scale), dtype=np.float32)
	log.info("calibration center: %s", calibration_center)
	base_transform = Transform(translation=calibration_center.tolist())#[0.32726666666666665, 0.3898192272727273, -0.240541])
	max_tries = 20
	def get_rotation_rotvec(rng, rad_range=None, sample_overrides={}):
		#https://math.stackexchange.com/a/1585996
		# if "rotvec" in sample_overrides:
			# rotation = sample_overrides["rotvec"]
		# else:
			# while True:
				# rotation = rng.normal(size=3)
				# if np.sum(np.abs(rotation)) > 1e-5:
					# break
		if rad_range is None: rad_range = np.deg2rad(rotation_range)
		rotation = np.asarray(sample_overrides.get("rotvec", rng.normal(size=3)), dtype=np.float32)
		rotation_rad = rng.uniform(low=rad_range[0], high=rad_range[1])
		if np.sum(np.abs(rotation)) > 1e-5:
			rotation *= rotation_rad/np.linalg.norm(rotation)
		return rotation
	
	def make_sample(seed):
		rng = np.random.RandomState(seed)
		gen_sample=True
		i=0
		while gen_sample and i<max_tries:
			gen_sample = False
			i+=1
			shape_types = sample_overrides.get('shape_type', [0,1,2,3,4])
			if not isinstance(shape_types, (list, tuple)):
				shape_types = [shape_types]
			shape_type = rng.choice(shape_types) #sample_overrides.get('shape_type', rng.randint(2))
			shape_inner = sample_overrides.get('shape_inner', [rng.uniform(*inner_range) for _ in range(3)])
			#shape_outer
			if shape_type==0:
				#base_density = make_sphere(object_shape, inner=shape_inner[0])
				base_density = make_sphere_smooth(object_shape, inner=shape_inner[0])
			elif shape_type<4:
				base_density = make_cube(object_shape, inner=shape_inner, cut_type=shape_type)
			elif shape_type==4:
				base_density = make_sphere(object_shape, inner=shape_inner[0])
			else:
				raise RuntimeError
			den_shape = shape_list(base_density)
			#log.info("Generated object %d with shape %s and inner shape %s", shape_type, shape_list(base_density), shape_inner)
			assert len(den_shape)==4 and all(d==o for d,o in zip(den_shape[:3], object_shape)) and den_shape[-1]==1, "Invalid density shape %s, must be DHWC %s, C=1"%(den_shape,object_shape)
			
			# if not tf.reduce_all(tf.is_finite(base_density)).numpy():
				# log.warning("Base Density %s from %d is not finite. type: %d, inner: %s", den_shape, seed, shape_type, shape_inner)
			# if tf.reduce_any(tf.less(base_density, 0.0)).numpy():
				# log.warning("Base Density %s from %d is negative. type: %d, inner: %s", den_shape, seed, shape_type, shape_inner)
			
			base_density = tf.expand_dims(base_density, 0)
			
			#base_transform = GridTransform(base_shape, normalize='MIN', center=True) #non-uniform base scale should be ok
			base_scale = sample_overrides.get('base_scale', [rng.uniform(*scale_range) for _ in range(3)])
			#generate source and target transforms. use normalizations to start from centered objects?
			#  for sequences: each transform is an evolution of the previous state, so chain them (parenting)
			
			#translation: make sure to stay (mostly) in domain, don't always start in the center
			translation = np.asarray(sample_overrides.get("initial_translation", rng.uniform(low=-max_translation, high=max_translation)), dtype=np.float32) #np.zeros([3])
			# start with random rotation
			rotation = Rotation.from_rotvec(sample_overrides.get("initial_rotation_rotvec", get_rotation_rotvec(rng,  [0,2*np.pi], sample_overrides=sample_overrides)))
			last_transform = base_transform
			transforms = []
			if advect_density:
				for step in range(sequence_length):
					#new_rotation = Rotation.from_rotvec(get_rotation_rotvec(rng, sample_overrides=sample_overrides))
					if (step==0):
						t = GridTransform(object_shape, translation=translation.tolist(), rotation_quat=rotation.as_quat(), scale=base_scale, normalize='MIN', center=True, parent = base_transform)
					else:
						# translate grid center to shape center (its translation), rotate, translate back, add new translation
						t = base_grid_transform.copy_no_data()
						#log.info("Base grid T: %s", t)
						#raise RuntimeError
						if t.parent is None:
							t.parent=Transform()
						t.translation = (-translation).tolist()
						t.scale = calibration_scale.tolist()
						t.parent.parent.scale = [1,1,1]
						
						t.parent.rotation_quat = rotation.as_quat()
						translation = np.clip(translation + new_translation, -max_translation, max_translation)
						t.parent.translation = translation.tolist()
						#log.info("%s", t)
					transforms.append(t)
					
					new_translation = rng.uniform(low=translation_range[0], high=translation_range[1], size=3)
					rotation = Rotation.from_rotvec(get_rotation_rotvec(rng, sample_overrides=sample_overrides))
					
			else:
				for step in range(sequence_length):
					t = GridTransform(object_shape, translation=translation.tolist(), rotation_quat=rotation.as_quat(), scale=base_scale, normalize='MIN', center=True, parent = base_transform)
					transforms.append(t)
					
					translation = np.clip(translation + rng.uniform(low=translation_range[0], high=translation_range[1], size=3), -max_translation, max_translation)
					rotation = Rotation.from_rotvec(get_rotation_rotvec(rng, sample_overrides=sample_overrides))*rotation
				
				
			
			
			#log.info("Density target: %s (%s, %s)", base_grid_transform.position_global(), base_grid_transform.grid_size_world(), base_grid_transform.cell_size_world())
			
			density_scale = sample_overrides.get('density_scale', rng.uniform(*density_range))
			#sample to main grid with transforms
			densities = []
			velocities = []
			if advect_density:
				# sample first density to base_grid
				d = tf.squeeze(density_sampler._sample_transform(base_density, [transforms[0]], [base_grid_transform], fix_scale_center=True), (0,1))
				
				# scale/normalize total density
				total_densities = tf.reduce_mean(d)
				if total_densities<=0:
					log.warning("Densities generated from %d (%d) are empty: %s", seed, i, total_densities_np)
					gen_sample = True
					continue
					
				d = d*(density_scale/total_densities)
				
				densities.append(d)
				
				for t_idx in range(len(transforms)-1):
					if True:
						#generate vel - transform FROM base_grid+translation+rot TO base_grid
						v = density_sampler.get_transform_LuT([transforms[t_idx+1]], [base_grid_transform], relative=True, fix_scale_center=True)
						v = v[...,:-1]
						#log.info("lut_shape: %s, lut mean: %s, abs: %s", shape_list(v), tf.reduce_mean(v).numpy(), tf.reduce_mean(tf.abs(v)).numpy())
						if velocity:
							velocities.append(-tf.squeeze(v, (0,)))
						
						#advect density to next
						#TODO: use full warping method with MC support
						d = tf.squeeze(density_sampler._sample_LuT(tf.expand_dims(densities[t_idx], 0), v, True, relative=True, cell_center_offset=0.5), (0,))
						assert len(shape_list(d))==4, "dens shape: %s"%(shape_list(d),)
					else:
						d = tf.squeeze(density_sampler._sample_transform(tf.expand_dims(densities[t_idx], 0), [transforms[t_idx+1]], [base_grid_transform], fix_scale_center=True), (0,1))
						
					if tf.reduce_mean(d)<=0:
						log.warning("Advected densities are 0 in step %d", t_idx)
					densities.append(d)
				
				velocities.append(tf.zeros(base_shape + [3]))
				
			else:
				for t_idx, t in enumerate(transforms):
					if velocity:
						raise NotImplementedError
						#this is wrong-> v = density_sampler.get_transform_LuT([t], [base_grid_transform], relative=True, fix_scale_center=True)
						velocities.append(-tf.squeeze(v[...,:-1], (0,))) #vel is negative relative lookup position without LoD (assuming SL advection)
					
					d = tf.squeeze(density_sampler._sample_transform(base_density, [t], [base_grid_transform], fix_scale_center=True), (0,1))
					# if not tf.reduce_all(tf.is_finite(base_density)).numpy():
						# log.warning("Resampled Density %d %s from %d is not finite. type: %d, inner: %s, T: %s", t_idx, den_shape, seed, shape_type, shape_inner, t)
						# gen_sample = True
					# if tf.reduce_any(tf.less(base_density, 0.0)).numpy():
						# log.warning("Resampled Density %d %s from %d is negative. type: %d, inner: %s, T: %s", t_idx, den_shape, seed, shape_type, shape_inner, t)
						# gen_sample = True
					densities.append(d)
				
				#densities = [tf.squeeze(base_density, (0)) for t in transforms]
				total_densities = [tf.reduce_mean(d) for d in densities]
				total_densities_np = [d.numpy() for d in total_densities]
				if any(d<=0 for d in total_densities_np):
					log.warning("Densities generated from %d (%d) are empty: %s", seed, i, total_densities_np)
					gen_sample = True
					continue
				# total density is prop. to avg when the grid resolution stays the same
				#  total density in all should be the same to respect divergence (norm total density and scale to same rand)
				densities = [d*(density_scale/t) for d, t in zip(densities, total_densities)]
			
			if False: # full density test
				factor = 1 #density_scale/(min_shape**3)
				densities = [tf.ones_like(_)*factor for _ in densities]
			
			densities = tf.stack(densities)
			log.debug("densities shape: %s", shape_list(densities))
			if velocity:
				velocities = tf.stack(velocities)
				log.debug("velocities shape: %s", shape_list(velocities))
		#log.info("Density sample shape: %s; with total densities %s", shape_list(densities), total_densities)
		#  multi-object? clipping, divergence, more ambigous solutions. 2nd step maybe...
		if gen_sample:
			raise RuntimeError("Failed to generate a valid sample from %d after %d attempts."%(seed, max_tries))
		#generate reference motion from transforms on main grid
		#  can use transform-to-lut func?
		#velocities = [density_sampler.get_transform_LuT(_) for _ in transforms]
		
		if raw or bkg:
			img_bkg = [tf.zeros((num_views, image_shape[1], image_shape[2], 3))]*sequence_length
			
			img_bkg = tf.identity(img_bkg)
		
		if raw or preproc or img_hull:
			#img = [] #tf.zeros((num_views, image_shape[1], image_shape[2], 3))]*sequence_length
			if not tf.reduce_all(tf.is_finite(densities)).numpy():
				log.warning("Density generated from %d is not finite.", seed)
			if tf.reduce_any(tf.less(densities, 0.0)).numpy():
				log.warning("Density generated from %d is negative.", seed)
			tmp_transform = base_grid_transform.copy_new_data(densities)
			img = density_sampler.render_density(tmp_transform, lights, cameras, cut_alpha=True) #V-NHWC
			img = tf.stack(img, axis=1) #NVHWC
			if not tf.reduce_all(tf.is_finite(img)).numpy():
				log.warning("Images rendered for %d are not finite.", seed)
			#log.info("Rendered sample shape: %s", shape_list(img))
			img = tf.identity(img)
		
		if raw:
			img_raw = img
			
			img_raw = tf.identity(img_raw)
		
		if hull:
			img_hull = img
			
			img_hull = tf.identity(img_hull)
		
		ret = []
		if raw: ret.append(img_raw)
		if preproc: ret.append(img)
		if bkg: ret.append(img_bkg)
		if hull: ret.append(img_hull)
		
		if density:
			ret.append(tf.identity(densities))
		if velocity:
			ret.append(tf.identity(velocities))
		
		return tuple(ret)
	
	
	ret_type = []
	types = []
	if raw:
		ret_type.append(tf.float32)
		types.append("img_raw")
	if preproc:
		ret_type.append(tf.float32)
		types.append("img_preproc")
	if bkg:
		ret_type.append(tf.float32)
		types.append("img_bkg")
	if hull:
		ret_type.append(tf.float32)
		types.append("hull")
	if density:
		ret_type.append(tf.float32)
		types.append("density")
	if velocity:
		raise NotImplementedError("change to staggered velocity output")
		ret_type.append(tf.float32)
		types.append("velocity")
	if make_temporal_input:
		ret_type *=3
	f_func = lambda s: tf.py_func(make_sample, [s], tuple(ret_type))
	
	#
	log.info("Initialize synthetic shapes dataset:\n\trequested data types: %s\n\t%s steps\n\tseed: %d\n\tdensity_range %s, inner_range %s, scale_range %s, translation_range %s, rotation_range %s\n\toverrides: %s",types, sequence_length, seed, density_range, inner_range, scale_range, translation_range, rotation_range, sample_overrides)
	
	rand_dataset = tf.data.Dataset.from_generator(gen_seed, output_types=(tf.int32,), output_shapes=([],), args=(seed,))
	
	#,num_parallel_calls=4
	loaded_data = rand_dataset.map(f_func)
	
	#loaded_data = loaded_data.repeat()
	#if shuffle_size>0:
	#	loaded_data = loaded_data.shuffle(shuffle_size)
	
	#raise NotImplementedError
	return loaded_data
