import copy, munch, numbers, collections.abc

import lib.scalar_schedule as sSchedule

#https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
def _CSETUPS_update_dict_recursive(d, u, deepcopy=False, new_key='KEEP'):
	if deepcopy:
		d = copy.deepcopy(d)
	for k, v in u.items():
		if not k in d:
			if new_key.upper()=='ERROR':
				raise KeyError("Update key {} does not exisit in base dictionary.".format(k))
			elif new_key.upper()=='DISCARD':
				continue
			elif new_key.upper()=='KEEP':
				pass
			else:
				raise ValueError("Unknown policy for new key: {}".format(new_key))
		if isinstance(v, collections.abc.Mapping):
			if k in d and not isinstance(d[k], collections.abc.Mapping):
				# if something that is not a Mapping is updated with a Mapping
				#e.g. a default constant schedule (int, float, list) with a complex one (dict {"type":<type>, ...})
				if isinstance(d[k], (numbers.Number, list)) and "type" in v and isinstance(v["type"], str) and v["type"].upper() in ['SCHEDULE', 'CONST','LINEAR','EXPONENTIAL','ROOT_DECAY'] and "start" in v:
					d[k] = sSchedule._get_base_schedule()
				else:
					d[k] = {}
			d[k] = _CSETUPS_update_dict_recursive(d.get(k, {}), v, deepcopy=deepcopy, new_key=new_key)
		else:
			if deepcopy:
				d[k] = copy.deepcopy(v)
			else:
				d[k] = v
	return d


#######################
# --- Setup Modules ---
#######################

#crop and main-opt growth with new loss balaning (after scale fixes)
RECONSTRUCT_SEQUENCE_SETUP_BASE = {
	'title':'seq_test',
	'desc':'sequence reconstruction run description',
	'paths':{
		'base':"./runs",
		'group':"sequence_recon_test",
	},
	'rendering':{
		'monochrome':False,
		'luma':[0.2126,0.7152,0.0722], #[0.299,0.587,0.144] #https://en.wikipedia.org/wiki/Luma_(video)
		'filter_mode':'LINEAR', #NEAREST, LINEAR
		'mip':{
			'mode':'LINEAR', #NONE, NEAREST, LINEAR
			'level':4,
			'bias':0.0,
		},
		'blend_mode':'BEER_LAMBERT', #BEER_LAMBERT, ALPHA, ADDITIVE
		'sample_gradients':True,
		'boundary': None,
		'SDF_threshold': 0.02,
		
		'steps_per_cycle':24,
		'num_images': 24,
		
		'main_camera':{
			'base_resolution':[256,1920,1080], #z(depth), y(height), x(width)
			'resolution_scale':1./3., # only for xy
			'fov':40,
			'near':0.3,
			'distance':0.8,
			'far':1.3,
		},
		'target_cameras':{
			'calibration_file':"scalaFlow_cameras.json",
			'camera_ids':[2,1,0,4,3],
			'crop_frustum':False, # crop frustum grid to AABB of vidual hull. for performance
			'crop_frustum_pad':2, # view space cells
		},
		
		'allow_static_cameras':False,
		'allow_fused_rendering':True,
		
		'background':{
			'type':'COLOR', #'CAM', 'COLOR', 'NONE'; only for vis-rendering, not used in optimization
			'color': [0,0.5,1.0], #[0,0.5,0],
		},
		
		'lighting':{
			#'scattering_ratio':1.,
			'ambient_intensity':0.64,
			'initial_intensity':0.85,
			'shadow_resolution':[256,196,196], #DHW
		},
		"velocity_scale":1024,
		"synthetic_target":{
			'filter_mode':'LINEAR',
			'blend_mode':'BEER_LAMBERT',
			'ambient_intensity':0.64,
			'initial_intensity':0.85,
		}
	},#rendering
	'data':{
		"rand_seed_global": 460585320,
		'run_dirs':['./runs/test-and-debug','./runs/sequence_recon_test'],
		'grid_size':128, #x and z resolution/grid size, y is ceil(this*scale_y) from callibration (scale_y ~ 1.77)
		'y_scale': 2, #"SF"
		'clip_grid':False,
		'clip_grid_pad':6,
		'crop_grid':True,
		'hull':'TARGETS', #ALL, TARGETS, ROT, [<ids>, ], empty==ALL
		
		'simulation':0,
		'start':140,
		'stop':142, #exclusive
		'step':1,
		'scalarFlow_frame_offset':-11,
		"SDF": False,
		'MS_volume':False,
		'MS_images':False,
		'density':{
			'scale': 1,
			#'initial_value':'HULL_TIGHT',#"CONST", "ESTIMATE", "HULL" or path to an npz file '[RUNID:000000-000000]frame_{frame:06d}/density_pre-opt.npz'
			'initial_value':'data/scalarFlow/sim_{sim:06d}/reconstruction/density_{frame:06d}.npz',
			'density_type': "SF", #OWN SF MANTA
			'min':0.0,
			'max':256.0,
			'target_type': "RAW", #PREPROC, SYNTHETIC
			'render_targets': False,
			'target': 'data/scalarFlow/sim_{sim:06d}/input/cam/imgsUnproc_{frame:06d}.npz', # 'data/scalarFlow/sim_{:06d}/input/cam/imgsUnproc_{:06d}.npz',
			'target_preproc': 'data/scalarFlow/sim_{sim:06d}/input/postprocessed/imgs_{frame:06d}.npz',
			'target_flip_y': False,
			'target_cam_ids':"ALL",#[0,1,2,3,4], #which ScalarFlow camera targets to use. 0:center, 1:center-left, 2:left, 3:right, 4:center-right (CW starting from center)
			'target_threshold':4e-2, #only used for disc dataset
			'target_scale': 1.5,
			'hull_image_blur_std':1.0,
			'hull_volume_blur_std':0.5,
			'hull_smooth_blur_std':0.0,
			'hull_threshold':4e-2,
			'inflow':{
				'active':True,
				'hull_height':10,
				'height':'MAX',
			},
			'scalarFlow_reconstruction':'data/scalarFlow/sim_{sim:06d}/reconstruction/density_{frame:06d}.npz',
			'synthetic_target_density_scale':1.,
		},
		'velocity':{
			'initial_value':'data/scalarFlow/sim_{sim:06d}/reconstruction/velocity_{frame:06d}.npz', #'RAND',
			'load_step':1,
			'init_std':0.1,
			'init_mask':'HULL_TIGHT_NEXT', #NONE HULL, HULL_TIGHT
			'boundary':'CLAMP', #BORDER (closed 0 bounds), CLAMP (open bounds)
			'scalarFlow_reconstruction':'data/scalarFlow/sim_{sim:06d}/reconstruction/velocity_{frame:06d}.npz',
		},
		'initial_buoyancy':[0.,0.,0.],
		'discriminator':{
			'simulations':[0,6],
			'frames':[45,145, 1],
			'target_type': "RAW", #PREPROC, (SYNTHETIC not supported)
			'initial_value':'data/scalarFlow/sim_{sim:06d}/reconstruction/density_{frame:06d}.npz', #path to density. needed for target rendering
			'density_type': "SF", #OWN SF MANTA
			'render_targets': False,
			'density_type': "SF", #OWN SF MANTA
			'target': 'data/scalarFlow/sim_{sim:06d}/input/cam/imgsUnproc_{frame:06d}.npz', # 'data/scalarFlow/sim_{:06d}/input/cam/imgsUnproc_{:06d}.npz',
			'target_preproc': 'data/scalarFlow/sim_{sim:06d}/input/postprocessed/imgs_{frame:06d}.npz',
			#augmentation
			'crop_size':[96,96], #HW, input size to disc
			'scale_input_to_crop':False, #resize all discriminator input to its base input resolution (crop_size) before augmentation
			'real_res_down': 4,
			'scale_real_to_cam':True, #scale real data resolution to current discriminator fake-sample camera (to match resolution growth)
			'scale_range':[0.85,1.15],
			'rotation_mode': "90",
		#	'resolution_fake':[256,1920/4,1080/4],
		#	'resolution_scales_fake':[1, 1/2],
		#	'resolution_loss':[256,1920/8,1080/8],
			'scale_real':[0.8, 1.8], #range for random intensity scale on real samples
			'scale_fake':[0.7, 1.4],
		#	'scale_loss':[0.8, 1.2],
			'gamma_real':[0.5,2], #range for random gamma correction on real samples (value here is inverse gamma)
			'gamma_fake':[0.5,2], #range for random gamma correction on fake samples (value here is inverse gamma)
		#	'gamma_loss':[0.5,2], #range for random gamma correction applied to input when evaluating the disc as loss (for density) (value here is inverse gamma)
			
		},
		'load_sequence':None, #only for rendering without optimization
		'load_sequence_pre_opt':False,
		'synth_shapes':{ #dataset of randomly generated shapes
			'active':True, #overrides other data loader
			'max_translation': 0.4, #
			'shape_types': 0, # single type-id or list of type-ids. 0=smooth sphere, 1=cube, 2=cube edges, 3=cube vertices, 4=sphere
			'init_center': True,
		},
		'resource_device': '/cpu:0',#'/cpu:0'
	},#data
	'training':{
		'iterations':5000,
		'start_iteration': 0,
		'frame_order':'FWD', #FWD, BWD, RAND
		'sequence_length': sSchedule.setup_constant_schedule(start=-1), #shedule for sequence length. cast to int, -1 to use full length
		
		
		'resource_device':'/cpu:0', #'/cpu:0', '/gpu:0'
		
		#'loss':'L2',
		'train_res_down':6,
		'loss_active_eps':1e-18, #minimum absolute value of a loss scale for the loss to be considered active (to prevent losses scaled to (almost) 0 from being evaluated)
		
		'randomization':{
			'transform':False,
			'sequence_length':False,
			'grid_size_relative': 1.0, # minimum relative grid size, 1 to diable grid size randomizaton
			'grid_size_min': 3, # minimum absolute grid size
			
			'grow_mode': None, # None, "RAND", "ITERATE", "RANDINTERVAL"
			
			# range of random intensity scale for images (inputs and targets)
			'scale_images_min':1,
			'scale_images_max':1,
			# range of random density scale for velocity training
			'scale_density_min':1,
			'scale_density_max':1,
			
			'inputs':True,
			'input_weights':None, #relative weights for targets, none for uniform
			'min_inputs':1,
			'max_inputs':5,
			'num_inputs_weights':None, #relative weights for number of targets, none for uniform
			
			'targets':False,
			'target_weights':None, #relative weights for targets, none for uniform
			'min_targets':1,
			'max_targets':5,
			'num_targets_weights':None, #relative weights for number of targets, none for uniform
		},
		'density':{
			'optim_beta':0.9,
			'use_hull':True,
			'warp_clamp':"MC_SMOOTH",
			'camera_jitter':False,
			'scale_render_grads_sharpness':0.0,
			'error_functions':{
				'raw_target_loss':'SE',
				'preprocessed_target_loss':'SE',
				'volume_target_loss':'SE',
				'volume_proxy_loss':'SE',
				'target_depth_smoothness_loss':'SE',
				'hull':'SE',
				'negative':'SE',
				'smoothness_loss':'SE',
				'smoothness_loss_2':'SE',
				'temporal_smoothness_loss':'SE',
				'warp_loss':'SE',
				'center_loss':'SE',
				'SDF_pos_loss':'AE',
			},
			'pre_optimization':True, #whether pre-optim will run for density, affects forward propagation/advection of state and optimization
			# to only have fwd advection without optimization set iterations to 0
			'pre_opt':{
				'first':{ #settings for first frame
					'iterations':0,#30000,
					'learning_rate':sSchedule.setup_exponential_schedule(start=3.0, base=0.5, scale=2/30000), #{'type':'exponential', 'start':3.0, 'base':0.5, 'scale':2/30000},
					
					'raw_target_loss':8.7e-7 *20,
					'preprocessed_target_loss':0.,
					'volume_target_loss':0.,
					'target_depth_smoothness_loss':0.,
					'hull':0.,
					'negative':0.,
					'smoothness_loss':0.0,#6.8e-14,
					'smoothness_neighbours':3,
					'smoothness_loss_2':0.0,#8.2e-14,
					'temporal_smoothness_loss':0.0,#0.2
					'discriminator_loss':0.0,
					'warp_loss':0.0,
					'center_loss':0.0,
					'SDF_pos_loss':0.0,
					'regularization':0.0001,
				},
				#settings for remaining frames
				'iterations':2400,#5000,
				'seq_init':"WARP", #WARP, COPY, BASE
				'learning_rate':sSchedule.setup_exponential_schedule(start=3.0, base=0.5, scale=1/3000), #{'type':'exponential', 'start':0.00005, 'base':0.5, 'scale':2/30000},
				
				'raw_target_loss':8.7e-7 *20,
				'preprocessed_target_loss':0.,
				'volume_target_loss':0.,
				'target_depth_smoothness_loss':0.,
				'hull':0.,
				'negative':0.,
				'smoothness_loss':0.0,#6.8e-14,
				'smoothness_neighbours':3,
				'smoothness_loss_2':0.0,#8.2e-14,
				'temporal_smoothness_loss':0.0,#0.2
				'discriminator_loss':0.0,
				'warp_loss':0.0,
				'center_loss':0.0,
				'SDF_pos_loss':0.0,
				'regularization':0.0001,
				
				'inspect_gradients':1,
				'grow':{
					"factor":1.2,#2. factor per interval, max down-scaling is factor^len(intervals)
					'intervals':[],
				},
			},
			
			'grow_lifting_skip': None,
			'grow_lifting_train': None,
			'grow_lifting_lr': None,
			
			'grow_lifting_residual': None,
			'grow_volenc_residual': None,
			
			'learning_rate':sSchedule.setup_exponential_schedule(start=2.45, base=0.5, scale=2/30000),#0.00015, #[0.00001,0.00001,0.0001, 0.00009/20000, 4000],
			
			'raw_target_loss':8.7e-7 *20,#for AE; for SE *40; for Huber *80
			'preprocessed_target_loss':0.,
			'volume_target_loss':0.,
			'volume_proxy_loss':0.,
			'target_depth_smoothness_loss':0.,
			'hull':0.,#1e-12,
			'negative':0.,#1e-12,
			
			'smoothness_loss':0.0, 
			'smoothness_neighbours':3, # the kind of neighbourhood to consider in the edge filter (e.g. wether to use diagonals), NOT the kernel size.
			'smoothness_loss_2':0.0,
			'temporal_smoothness_loss':0.0,#0.2
			
			'discriminator_loss':1.5e-5,
			'warp_loss':[6.7e-11 *4,6.7e-11 *4,13.4e-11 *4, 6.7e-11 *4/2000, 2000],#for AE; for SE *8, for Huber *24
			'center_loss':0.0,
			'SDF_pos_loss':0.0,#for AE; for SE *8, for Huber *24
			'regularization':0.0001,
			
			'main_warp_fwd':False,
			'warp_gradients':{
				'weight':sSchedule.setup_constant_schedule(start=1.0),
				'active':False,
				'decay':sSchedule.setup_constant_schedule(start=0.9), #[0,1], lower is faster decay
				'update_first_only':False,
			},
			"view_interpolation":{
				"steps":0,
			},
			
			'grow':{ 
				"factor":1.2,#2. factor per interval, max down-scaling is factor^len(intervals)
				"pre_grow_actions":[],# "WARP", list, unused
				"post_grow_actions":[],# "WARP", list
				#iterations for each grow step, empty to disable
				'intervals':[],
			},
		},
		'velocity':{
			'optim_beta':0.9,
			'warp_order':2,
			'warp_clamp':"MC_SMOOTH",
			'error_functions':{
				'volume_target_loss':'SE',
				'density_warp_loss':'SE',
				'density_proxy_warp_loss':'SE',
				'density_target_warp_loss':'SE',
				'velocity_warp_loss':'SE',
				'divergence_loss':'SE',
				'magnitude_loss':'SE',
				'CFL_loss':'SE',
				'MS_coherence_loss':'SE',
			},
			'pre_optimization':True, #whether pre-optim will run for velocity, affects forward propagation/advection of state and optimization
			'pre_opt':{
				'first':{ #settings for first frame
					'iterations':0,
					'learning_rate':0.04,
					
					'volume_target_loss':0.,
					'density_warp_loss':8.2e-11 *5,
					'density_proxy_warp_loss':0,
					'density_target_warp_loss':8.2e-11 *5,
					'velocity_warp_loss':0.0,
					'smoothness_loss':0.0,
					'smoothness_neighbours':3,
					'cossim_loss':0.0,
					'divergence_loss':sSchedule.setup_exponential_schedule(start=4.3e-10 *2, base=1.2, scale=1/10000), #sSchedule.setup_linear_schedule_2(start=0, end=4.3e-10, steps=7000),#
					#'divergence_normalize':0.0,
					#adjust according to data.step, large values can lead to NaN.
					'magnitude_loss':0.0,#{'type':'exponential', 'start':4e-11, 'base':0.10, 'scale':1/5000},
					'CFL_loss':0.0,
					'MS_coherence_loss':0.0,
					'regularization':0.0001,
					'grow':{
						"factor":1.2,
						"scale_magnitude":True,
						'intervals':[200, 260, 330, 400, 460, 530, 600, 660, 800, 930, 1060, 1260], #7490
					},
				
				},
				#settings for remaining frames
				'iterations':1200,
				'seq_init':"WARP", #WARP, COPY, BASE
				'learning_rate':0.02,
				
				'volume_target_loss':0.,
				'density_warp_loss':8.2e-11 *5,
				'density_proxy_warp_loss':0,
				'density_target_warp_loss':8.2e-11 *5,
				'velocity_warp_loss':0.0,
				'smoothness_loss':0.0,
				'smoothness_neighbours':3,
				'cossim_loss':0.0,
				'divergence_loss':4.3e-10 *6,
				#'divergence_normalize':0.0,
				'magnitude_loss':0.0,#4e-12,
				'CFL_loss':0.0,
				'MS_coherence_loss':0.0,
				'regularization':0.0001,
				
				'grow':{
					"factor":1.2,#2. factor per interval, max down-scaling is factor^len(intervals)
					"scale_magnitude":True,
					'intervals':[],
				},
			},
			
			'noise_std':sSchedule.setup_constant_schedule(start=0.0),
			'learning_rate':sSchedule.setup_exponential_schedule(start=0.02, base=0.5, scale=2/30000),#{'type':'exponential', 'max':0.02, 'start':0.02, 'base':0.5, 'scale':2/30000, 'offset':0},
					
		#	'lr_decay':0.00,
			
		#	'loss':'L2',
			'volume_target_loss':0.,
			'density_warp_loss':8.2e-11 *5,#for AE; for SE *10; for Huber *25 #influence of loss(A(dt, vt), dt+1) on velocity, can be a schedule
			'density_warp_loss_MS_weighting':sSchedule.setup_constant_schedule(start=1.0),
			'density_proxy_warp_loss':0,
			'density_target_warp_loss':8.2e-11 *5,#for AE; for SE *10; for Huber *25 #influence of loss(A(dt, vt), dt+1) on velocity, can be a schedule
			'density_target_warp_loss_MS_weighting':sSchedule.setup_constant_schedule(start=1.0),
			'velocity_warp_loss':sSchedule.setup_linear_schedule_2(start=1.35e-11 *3, end=1.35e-11 *6, steps=5000), #2.7e-12 *5,#for AE; for SE *10; for Huber *20 #influence of loss(A(vt, vt), vt+1) on velocity, can be a schedule
			# 'velocity_warp_loss_MS_weighting':sSchedule.setup_constant_schedule(start=1.0),
			
			'smoothness_loss':0.0,
			'smoothness_neighbours':3, # the kind of neighbourhood to consider in the edge filter (e.g. wether to use diagonals), NOT the kernel size.
			'cossim_loss':0.0,
			
			'divergence_loss':sSchedule.setup_exponential_schedule(start=4.3e-10 *6, base=1.2, scale=1/500),
			'divergence_loss_MS_weighting':sSchedule.setup_constant_schedule(start=1.0),
			'divergence_normalize':0.0,
			
			'magnitude_loss':0.0,#1e-12,
			'CFL_loss':0.0,
			'CFL_loss_MS_weighting':sSchedule.setup_constant_schedule(start=1.0),
			'MS_coherence_loss':0.0,
			'MS_coherence_loss_MS_weighting':sSchedule.setup_constant_schedule(start=1.0),
			'regularization':0.0001,
			
			'warp_gradients':{
				'weight':sSchedule.setup_constant_schedule(start=1.0), #affects warp gradients for velocity from backward dens warp, even if vel- warp gradients are inactive
				'active':False,
				'decay':sSchedule.setup_constant_schedule(start=0.9), #[0,1], lower is faster decay
			},
			
			'grow':{
				"factor":1.2,#2. factor per interval, max down-scaling is factor^len(intervals)
				"scale_magnitude":True,
				'intervals':[],
			},
		},
		"MS_weighting": sSchedule.setup_constant_schedule(start=1.0), #level-relative loss weighting for multi-scale losses. here the "iteration" is the level index, which starts with 0 at the COARSEST resolution.
		"allow_MS_losses": True,
		
		'optimize_buoyancy':False,
		"light":{
			"optimize":False,
			'optim_beta':0.9,
			"min":0.01,
			"max":6.0,
			"learning_rate":{'type':'exponential', 'start':0.001, 'base':1, 'scale':0},
		},
	#	"scattering":{
	#		"optimize":False,
	#		'optim_beta':0.9,
	#		"min":0.01,
	#		"max":1.0,
	#		"learning_rate":{'type':'exponential', 'start':0.001, 'base':1, 'scale':0},
	#	},
		
		
		'discriminator':{
			'active':False,
			'model':None,#'[RUNID:200227-162722]disc_model.h5',#
			'loss_type':"RaLSGAN", #"SGAN", "RpSGAN", "RpLSGAN", "RaSGAN", "RaLSGAN"
			'target_label':1.0,#0.9, #0.9 for label smoothing, 1.0 for LS-GAN 
			# l4s
			'layers':[16,16,24,24,32,32,32,64,64,64,16, 4],
			'stride':[ 2, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1],
			'kernel_size':4,
			'padding':'MIRROR', #NONE(=valid), ZERO(=same), MIRROR(=tf.pad(REFLECT))
			'activation':'lrelu', # relu or lrelu
			'activation_alpha':0.2, # leak for lrelu
			'use_fc': None, # list of filters, usually makes disc stronger
			'noise_std':0.0,
			'start_delay':0, #50,
			'pre_opt':{
				'first':{
					'train':False,
					'learning_rate':{'type':'exponential', 'start':4e-4, 'base':0.5, 'scale':4/30000},
			#		'steps':1,
					'regularization':0.002,
				},
				'train':False,
				'learning_rate':1.6e-4,
			#	'steps':1,
				'regularization':0.002,
			},
			'train':True,
			'learning_rate':2e-4,
			'steps':1,
			'regularization':0.002,
			'optim_beta':0.5,
			
			'grow':{ # not yet used
				"factor":2.,#2. factor per interval, max down-scaling is factor^len(intervals)
				#iterations for each grow step, empty to disable
				'intervals':[]
			},
			
			'conditional_hull':False,
			'temporal_input':{
				'active':False,
				'step_range':(-3,4,1), #-3 to 3 inclusive, 0 will be removed
			},
			'num_real':4,
			'cam_res_down':6,
			'num_fake':3,
			'fake_camera_jitter':False,
			"history":{
				'load':None,
				'samples':4, #use older samples as fake samples as well. 0 to disable
				'size':800, #
				'keep_chance':0.01, # chance to put a rendered sample in the history buffer
				'save':False,
				'sequence_reuse':True,
				'reset_on_density_grow':True,
			#	'reset_on_discriminator_grow':False,
			},
			
		#	'sequence_reuse':True,
		},#discriminator
		'summary_interval':100,
		'checkpoint_interval':500,
	},#training
	'validation':{
		'output_interval':100,
		'stats':True,
		'cmp_vol_targets':False,
		'cmp_scalarFlow':False,
		'cmp_scalarFlow_render':False,
		'warp_test':["BASE", "NO_INFLOW", "ONLY_INFLOW"],
		'warp_test_render':False,
		'render_cycle':False,
		'render_cycle_steps':8,
		'render_density':True,
		'render_shadow':True,
		'render_target':True,
		'render_velocity':True,
		'render_MS':False,
		# if data.synth_shapes.active
		'synth_data_seed': 1802168824,
		'synth_data_shape_types':0, #for training mode
		'synth_data_eval_setup':"SPHERE", #pre-made test-cases for evaluation mode # SF, SPHERE, CUBE, ROTCUBE
	},
	'debug':{
		'print_weight_grad_stats':False,
		'target_dump_samples':False,
		'disc_dump_samples':False,
	},
}

# RECONSTRUCT_SEQUENCE_SETUP_GROW = {
	# "training":{
		# 'iterations':7000,
		# "density":{
			# 'pre_opt':{
				# 'first':{
					# 'iterations':400,
				# },
				# 'iterations':400,
			# },
			# 'grow':{ 
				# "factor":1.2,
				# "pre_grow_actions":[],
				# "post_grow_actions":[],
				# 'intervals':[200,300,400,500,500,600,700,800], #4000
				# #'intervals':[400,400,400,400,400,400,400,400], #3200
			# },
		# },
		# 'velocity':{
			# 'pre_opt':{
				# 'first':{
					# 'iterations':5000,
					# 'divergence_loss':sSchedule.setup_exponential_schedule(start=4.3e-10 *2, base=1.2, scale=1/1000),
					# 'grow':{
						# "factor":1.2,
						# "scale_magnitude":True,
						# 'intervals':[660, 860, 1100, 1300], #3920
					# },
				# },
				# 'iterations':400,
				# 'divergence_loss':4.3e-10 *6,
			# },
			# 'divergence_loss':sSchedule.setup_exponential_schedule(start=4.3e-10 *6, base=1.2, scale=1/700),
			# 'grow':{
				# "factor":1.2,
				# "scale_magnitude":True,
				# 'intervals':[200,300,400,500,500,600,700,800], #4000
			# },
		# },
	# },
# }
# RECONSTRUCT_SEQUENCE_SETUP_GROW = _CSETUPS_update_dict_recursive(RECONSTRUCT_SEQUENCE_SETUP_BASE, RECONSTRUCT_SEQUENCE_SETUP_GROW, deepcopy=True, new_key='ERROR')

# RECONSTRUCT_SEQUENCE_SETUP_GROW_DGRADWARP = {
	# "training":{
		# 'frame_order':'BWD',
		# "density":{
			# 'main_warp_fwd':True,
			# 'warp_gradients':{
				# 'weight':1.0,
				# 'active':True,
				# 'decay':0.9,
				# 'update_first_only':True,
			# },
		# },
		# 'velocity':{
			# 'warp_gradients':{
				# 'weight':1.0,
				# 'active':False,
				# 'decay':0.9,
			# },
		# },
	# },
# }

# RECONSTRUCT_SEQUENCE_SETUP_GROW_DGRADWARP = _CSETUPS_update_dict_recursive(RECONSTRUCT_SEQUENCE_SETUP_GROW, RECONSTRUCT_SEQUENCE_SETUP_GROW_DGRADWARP, deepcopy=True, new_key='ERROR')

# RECONSTRUCT_SEQUENCE_SETUP_GROW_DGRADWARP_FRONT = {
	# "desc":"front-loaded training. 50% longer pre-training, linear growth intervals (longer low res, shorter high res)",
	# "training":{
		# 'iterations':4200,
		# "density":{
			# 'pre_opt':{
				# 'first':{
					# 'iterations':600,
				# },
				# 'iterations':600,
			# },
			# 'grow':{ 
				# "factor":1.2,
				# "pre_grow_actions":[],
				# "post_grow_actions":[],
				# #'intervals':[200,300,400,500,500,600,700,800], #4000
				# 'intervals':[400,400,400,400,400,400,400,400], #3200
			# },
		# },
		# 'velocity':{
			# 'pre_opt':{
				# 'first':{
					# 'iterations':6000,
					# 'divergence_loss':sSchedule.setup_exponential_schedule(start=4.3e-10 *2, base=1.2, scale=1/1000),
					# 'grow':{
						# "factor":1.2,
						# "scale_magnitude":True,
						# 'intervals':[1000, 1000, 1000, 1000], #4000
					# },
				# },
				# 'iterations':600,
				# 'divergence_loss':4.3e-10 *6,
			# },
			# 'divergence_loss':sSchedule.setup_exponential_schedule(start=4.3e-10 *6, base=1.2, scale=1/400),
			# 'grow':{
				# "factor":1.2,
				# "scale_magnitude":True,
				# 'intervals':[400,400,400,400,400,400,400,400], #3200
			# },
		# },
	# },
# }

# RECONSTRUCT_SEQUENCE_SETUP_GROW_DGRADWARP_FRONT = _CSETUPS_update_dict_recursive(RECONSTRUCT_SEQUENCE_SETUP_GROW_DGRADWARP, RECONSTRUCT_SEQUENCE_SETUP_GROW_DGRADWARP_FRONT, deepcopy=True, new_key='ERROR')

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
			name="GrowingUNet", normalization="NONE", **kwargs)
"""
GROWINGUNET_CONFIG = {
	#"dimension": 2,
	#"num_levels": 3,
	"level_scale_factor": 2,
	
	#"input_channels": 3,
	"input_levels": 1,
	"create_inputs": True,
	"input_blocks": ["C:1-1"],
	#"input_conv_filters": 1,
	#"input_conv_kernel_size": 1,
	"share_input_layer": False,
	
	"down_mode": "STRIDED", # STRIDED, NONE
	"down_conv_filters": None,
	"down_conv_kernel_size": 4,
	"share_down_layer": True,
	
	"encoder_resblocks": ["RB:1-1_1-1"],
	"share_encoder": False,
	
	"decoder_resblocks": ["RB:1-1_1-1"],
	"share_decoder": False,
	
	"up_mode": "NNSAMPLE_CONV",
	"up_conv_filters": 1,
	"up_conv_kernel_size": 4,
	"share_up_layer": True,
	
	"skip_merge_mode": "CONCAT", # CONCAT, WSUM, SUM
	
	#"output_levels": 1,
	"output_blocks": None,
	#"output_channels": 1,
	"share_output_layer": True,
	"output_activation": "none", # none, relu, lrelu
	"output_conv_kernel_size": 1,
	"output_mode": "SINGLE", # SINGLE, RESIDUAL
	# global settings
	"conv_activation": "relu", # none, relu, lrelu
	"alpha": 0.2, # leak for lrelu
	"conv_padding": "ZERO", # ZERO, MIRRIR
	"normalization": "LAYER", # NONE, LAYER
}

RECONSTRUCT_SEQUENCE_SETUP_NEURAL_DENSITY = {
	"desc":"train neural networks to generate density directly from the targets.",
	"data":{
		'grid_size':48, #128, #resolution
		'y_scale': 2, # "SF" ~1.77 (16/9)
		'clip_grid':False,
		'clip_grid_pad':4,
		'crop_grid':True,
		'res_down_factor':128*5, #target data down scale factor is 6 (i.e. 1/6 of raw image resolution) at 128 base grid resolution
		
		'start':20,#56,#40,
		'stop':141,#68,#52, #exclusive
		'step':2,#2,
		# 
		"randomize":64,
		"batch_size":4,
		"sims":list(range(20)),
		"sequence_step":1,
		"sequence_length":1,
		
		"density":{
			'target_type': "PREPROC", #RAW, PREPROC, SYNTHETIC
			'scale':1,
			'inflow':{
				'active':False,
				'hull_height':4,
				'height':'MAX',
			},
		},
	},
	"training":{
		'iterations':0,
		"view_encoder":{
			#"active":True, active= density.decoder.active or velocity.decoder.active
			'encoder': ['L'], #NETWORK, NONE
			# 'layers':[4,8,16,16,16,4],
			# 'stride':1,
			# 'skip_connections':None,
			# 'skip_mode':"CONCAT",
			# 'kernel_size':4,
			# 'padding':'MIRROR', #NONE(=valid), ZERO(=same), MIRROR(=tf.pad(REFLECT))
			# 'activation':'lrelu', # relu or lrelu
			# 'activation_alpha':0.2, # leak for lrelu
			# "load_encoder":None,
			"lifting":"UNPROJECT", #UNPROJECT ;2D->3D method
			#"load_lifting":None, #lifting could be NN
			"merge":"MEAN", # SUM, MEAN ;method to merge 3D embeddings of multiple views
			#"load_merge":None, #merging could be NN
			
			"model":{
				"num_levels": "VARIABLE",
				"level_scale_factor": 2,
				
				"input_levels": 1,
				"create_inputs": True,
				"input_blocks": ["C:1-1"],
				#"input_conv_filters": 1,
				#"input_conv_kernel_size": 1,
				"share_input_layer": False,
				
				"down_mode": "STRIDED", # STRIDED, NONE
				"down_conv_filters": None,
				"down_conv_kernel_size": 4,
				"share_down_layer": True,
				
				"encoder_resblocks": ["RB:1-1_1-1"],
				"share_encoder": False,
				
				"decoder_resblocks": ["RB:1-1_1-1"],
				"share_decoder": False,
				
				"up_mode": "NNSAMPLE_CONV",
				"up_conv_filters": 1,
				"up_conv_kernel_size": 4,
				"share_up_layer": True,
				
				"skip_merge_mode": "CONCAT", # CONCAT, WSUM, SUM
				
				#"output_levels": 1,
				"output_blocks": None,
				"output_channels": 1,
				"share_output_layer": True,
				"output_activation": "none", # none, relu, lrelu
				"output_conv_kernel_size": 1,
				"output_mode": "SINGLE", # SINGLE, RESIDUAL
				# global settings
				"conv_activation": "relu", # none, relu, lrelu
				"alpha": 0.2, # leak for lrelu
				"conv_padding": "ZERO", # ZERO, MIRRIR
				"normalization": "LAYER", # NONE, LAYER
			}, # or path to model file
			"start_level": 0,
			"min_grid_res": 8,
			"train_mode": "ALL", #"ALL", "TOP", "TOP_DEC"
			"train_mode_schedule": sSchedule.setup_boolean_schedule(start=True, offset=0),
			"skip_merge_weight_schedule": sSchedule.setup_linear_schedule_2(start=1., end=1.0, steps=650, offset=250),
			"grow_intervals": [],
		},
		"volume_encoder":{
			"active":False,
			#"model":"decoder_model",
			"model":{
				"num_levels": "VARIABLE",
				"level_scale_factor": 2,
				
				"input_levels": 1,
				"create_inputs": True,
				"input_blocks": ["C:1-1"],
				#"input_conv_filters": 1,
				#"input_conv_kernel_size": 1,
				"share_input_layer": False,
				
				"down_mode": "STRIDED", # STRIDED, NONE
				"down_conv_filters": None,
				"down_conv_kernel_size": 4,
				"share_down_layer": True,
				
				"encoder_resblocks": ["RB:1-1_1-1"],
				"share_encoder": False,
				
				"decoder_resblocks": ["RB:1-1_1-1"],
				"share_decoder": False,
				
				"up_mode": "NNSAMPLE_CONV",
				"up_conv_filters": 1,
				"up_conv_kernel_size": 4,
				"share_up_layer": True,
				
				"skip_merge_mode": "CONCAT", # CONCAT, WSUM, SUM
				
				#"output_levels": 1,
				"output_blocks": None,
				"output_channels": 1,
				"share_output_layer": True,
				"output_activation": "none", # none, relu, lrelu
				"output_conv_kernel_size": 1,
				"output_mode": "SINGLE", # SINGLE, RESIDUAL
				# global settings
				"conv_activation": "relu", # none, relu, lrelu
				"alpha": 0.2, # leak for lrelu
				"conv_padding": "ZERO", # ZERO, MIRRIR
				"normalization": "LAYER", # NONE, LAYER
			}, # or path to model file
			"start_level": 0,
			"min_grid_res": 8,
			"train_mode": "ALL", #"ALL", "TOP", "TOP_DEC"
			"train_mode_schedule": sSchedule.setup_boolean_schedule(start=True, offset=0),
			"skip_merge_weight_schedule": sSchedule.setup_linear_schedule_2(start=1., end=1.0, steps=650, offset=250),
			"grow_intervals": [],
		},
		"lifting_network":{
			"active":False,
			#"model":"decoder_model",
			"model":{
				"num_levels": "VARIABLE",
				"level_scale_factor": 2,
				
				"input_levels": 1,
				"create_inputs": True,
				"input_blocks": ["C:1-1"],
				#"input_conv_filters": 1,
				#"input_conv_kernel_size": 1,
				"share_input_layer": False,
				
				"down_mode": "STRIDED", # STRIDED, NONE
				"down_conv_filters": None,
				"down_conv_kernel_size": 4,
				"share_down_layer": True,
				
				"encoder_resblocks": ["RB:1-1_1-1"],
				"share_encoder": False,
				
				"decoder_resblocks": ["RB:1-1_1-1"],
				"share_decoder": False,
				
				"up_mode": "NNSAMPLE_CONV",
				"up_conv_filters": 1,
				"up_conv_kernel_size": 4,
				"share_up_layer": True,
				
				"skip_merge_mode": "CONCAT", # CONCAT, WSUM, SUM
				
				#"output_levels": 1,
				"output_blocks": None,
				"output_channels": 1,
				"share_output_layer": True,
				"output_activation": "none", # none, relu, lrelu
				"output_conv_kernel_size": 1,
				"output_mode": "SINGLE", # SINGLE, RESIDUAL
				# global settings
				"conv_activation": "relu", # none, relu, lrelu
				"alpha": 0.2, # leak for lrelu
				"conv_padding": "ZERO", # ZERO, MIRRIR
				"normalization": "LAYER", # NONE, LAYER
			}, # or path to model file
			"start_level": 0,
			"min_grid_res": 8,
			"train_mode": "ALL", #"ALL", "TOP", "TOP_DEC"
			"train_mode_schedule": sSchedule.setup_boolean_schedule(start=True, offset=0),
			"skip_merge_weight_schedule": sSchedule.setup_linear_schedule_2(start=1., end=1.0, steps=650, offset=250),
			"grow_intervals": [],
		},
		"frame_merge_network":{
			"active":False,
			#"model":"decoder_model",
			"model":{
				"num_levels": "VARIABLE",
				"level_scale_factor": 2,
				
				"input_levels": 1,
				"create_inputs": True,
				"input_blocks": ["C:1-1"],
				#"input_conv_filters": 1,
				#"input_conv_kernel_size": 1,
				"share_input_layer": False,
				
				"down_mode": "STRIDED", # STRIDED, NONE
				"down_conv_filters": None,
				"down_conv_kernel_size": 4,
				"share_down_layer": True,
				
				"encoder_resblocks": ["RB:1-1_1-1"],
				"share_encoder": False,
				
				"decoder_resblocks": ["RB:1-1_1-1"],
				"share_decoder": False,
				
				"up_mode": "NNSAMPLE_CONV",
				"up_conv_filters": 1,
				"up_conv_kernel_size": 4,
				"share_up_layer": True,
				
				"skip_merge_mode": "CONCAT", # CONCAT, WSUM, SUM
				
				#"output_levels": 1,
				"output_blocks": None,
				"share_output_layer": True,
				"output_activation": "none", # none, relu, lrelu
				"output_conv_kernel_size": 1,
				"output_mode": "SINGLE", # SINGLE, RESIDUAL
				# global settings
				"conv_activation": "relu", # none, relu, lrelu
				"alpha": 0.2, # leak for lrelu
				"conv_padding": "ZERO", # ZERO, MIRRIR
				"normalization": "LAYER", # NONE, LAYER
			}, # or path to model file
			"start_level": 0,
			"min_grid_res": 8,
			"train_mode": "ALL", #"ALL", "TOP", "TOP_DEC"
			"train_mode_schedule": sSchedule.setup_boolean_schedule(start=True, offset=0),
			"skip_merge_weight_schedule": sSchedule.setup_linear_schedule_2(start=1., end=1.0, steps=650, offset=250),
			"grow_intervals": [],
		},
		'train_frame_encoders': sSchedule.setup_constant_schedule(start=True),
		"density":{
			"decoder":{
				"active":True,
			#	'layers':[4,8,16,8,4],
			#	'stride':1,
			#	'skip_connections':None,
			#	'skip_mode':"CONCAT",
			#	'kernel_size':4,
			#	'padding':'MIRROR', #NONE(=valid), ZERO(=same), MIRROR(=tf.pad(REFLECT))
			#	'activation':'lrelu', # relu or lrelu
			#	'activation_alpha':0.2, # leak for lrelu
				'input_type':"PREPROC", #use raw or preproc images
				#"model":"decoder_model", #3D features to density
				"model":{
					#"dimension": 2,
					"num_levels": "VARIABLE",
					"level_scale_factor": 2,
					
					#"input_channels": 3,
					"input_levels": 1,
					"create_inputs": True,
					"input_blocks": ["C:1-1"],
					#"input_conv_filters": 1,
					#"input_conv_kernel_size": 1,
					"share_input_layer": False,
					
					"down_mode": "STRIDED", # STRIDED, NONE
					"down_conv_filters": None,
					"down_conv_kernel_size": 4,
					"share_down_layer": True,
					
					"encoder_resblocks": ["RB:1-1_1-1"],
					"share_encoder": False,
					
					"decoder_resblocks": ["RB:1-1_1-1"],
					"share_decoder": False,
					
					"up_mode": "NNSAMPLE_CONV",
					"up_conv_filters": 1,
					"up_conv_kernel_size": 4,
					"share_up_layer": True,
					
					"skip_merge_mode": "CONCAT", # CONCAT, WSUM, SUM
					
					#"output_levels": 1,
					"output_blocks": None,
					#"output_channels": 1,
					"share_output_layer": True,
					"output_activation": "none", # none, relu, lrelu
					"output_conv_kernel_size": 1,
					"output_mode": "SINGLE", # SINGLE, RESIDUAL
					# global settings
					"conv_activation": "relu", # none, relu, lrelu
					"alpha": 0.2, # leak for lrelu
					"conv_padding": "ZERO", # ZERO, MIRRIR
					"normalization": "LAYER", # NONE, LAYER
				}, # or path to model file
				
				"start_level": 0,
				"min_grid_res": 10,
				"train_mode": "ALL", #"ALL", "TOP", "TOP_DEC"
				"train_mode_schedule": sSchedule.setup_boolean_schedule(start=True, offset=0),
				# network inputs
				"base_input": "ZERO", #ZERO TARGET_HULL
				"step_input_density": [], #0 is the current frame, 1 the next, etc. can be negative
				"step_input_density_target": [], #0 is the current frame, 1 the next, etc. can be negative
				"step_input_features": [0,1],
				"type_input_features": ["TARGET_UNPROJECTION"], # case-sensitive, order-invariant. TARGET_UNPROJECTION, TARGET_HULL
				"warp_input_indices": [0], #indices of network inputs to be warped. first the the density inputs, then the elements of step_input_features
				
				"recursive_MS": False, #do recursion and growing in NeuralGrid instead of UNet/Model. evey level uses the full specified model, potetially with multiple levels.
				"recursive_MS_levels": "VARIABLE",
				"recursive_MS_residual": True,
				"recursive_MS_direct_input": False, #if false: generate density decoder input at highest scale/resolution and sample down. if true: generate input at resolution required for current scale.
				"recursive_MS_scale_factor": 2,
				"recursive_MS_shared_model": True,
				"recursive_MS_train_mode": "ALL", #ALL, TOP
				"skip_merge_weight_schedule": sSchedule.setup_linear_schedule_2(start=1., end=1.0, steps=650, offset=250),
				"grow_intervals": [],
				"recursive_MS_copy_on_grow": True,
				
				"base_SDF_mode": "NONE", # "NONE", "RESIDUAL", "INPUT_RESIDUAL"
			},
			'pre_optimization': True,
			'pre_opt':{
				'first':{
					'iterations':0,
				},
				'iterations':2000,
				'learning_rate':{'type':'exponential', 'max':4e-5, 'min': 1e-7, 'start':4e-5, 'base':0.5, 'scale':1/2000, 'offset':2000},
				'raw_target_loss':1.0,
				'discriminator_loss': sSchedule.setup_linear_schedule_2(start=1e-6, end=5e-5, steps=6500) ,
				'grow':{
					"factor":2,
					'intervals':[], #6000
				},
			},
			'train_decoder': sSchedule.setup_constant_schedule(start=True),
		},
		"velocity":{
			"decoder":{
				"active":False,
			#	'layers':[4,8,16,8,4],
			#	'stride':1,
			#	'skip_connections':None,
			#	'skip_mode':"CONCAT",
			#	'kernel_size':4,
			#	'padding':'MIRROR', #NONE(=valid), ZERO(=same), MIRROR(=tf.pad(REFLECT))
			#	'activation':'lrelu', # relu or lrelu
			#	'activation_alpha':0.2, # leak for lrelu
				'input_type':"PREPROC", #use raw or preproc images
				'velocity_format': "CENTERED", #CENTERED, STAGGERED
				"model": {
					#"dimension": 2,
					"num_levels": "VARIABLE",
					"level_scale_factor": 2,
					
					#"input_channels": 3,
					"input_levels": 1,
					"create_inputs": True,
					"input_blocks": ["C:1-1"],
					#"input_conv_filters": 1,
					#"input_conv_kernel_size": 1,
					"share_input_layer": False,
					
					"down_mode": "STRIDED", # STRIDED, NONE
					"down_conv_filters": None,
					"down_conv_kernel_size": 4,
					"share_down_layer": True,
					
					"encoder_resblocks": ["RB:1-1_1-1"],
					"share_encoder": False,
					
					"decoder_resblocks": ["RB:1-1_1-1"],
					"share_decoder": False,
					
					"up_mode": "NNSAMPLE_CONV",
					"up_conv_filters": 1,
					"up_conv_kernel_size": 4,
					"share_up_layer": True,
					
					"skip_merge_mode": "CONCAT", # CONCAT, WSUM, SUM
					
					#"output_levels": 1,
					"output_blocks": None,
					#"output_channels": 1,
					"share_output_layer": True,
					"output_activation": "none", # none, relu, lrelu
					"output_conv_kernel_size": 1,
					"output_mode": "SINGLE", # SINGLE, RESIDUAL
					# global settings
					"conv_activation": "relu", # none, relu, lrelu
					"alpha": 0.2, # leak for lrelu
					"conv_padding": "ZERO", # ZERO, MIRRIR
					"normalization": "LAYER", # NONE, LAYER
				}, #3D features to velocity
				"start_level": 0,
				"min_grid_res": 10,
				"train_mode": "ALL", #"ALL", "TOP", "TOP_DEC"
				"train_mode_schedule": sSchedule.setup_boolean_schedule(start=True, offset=0),
				# network inputs
				"step_input_density": [], #0 is the current frame, 1 the next, etc. can be negative
				"step_input_density_target": [], #0 is the current frame, 1 the next, etc. can be negative
				"step_input_density_proxy": [], #0 is the current frame, 1 the next, etc. can be negative
				"step_input_features": [0,1],
				"type_input_features": ["TARGET_UNPROJECTION"], # case-sensitive, order-invariant. TARGET_UNPROJECTION, TARGET_HULL
				"downscale_input_modes": ["RESAMPLE", "RESAMPLE"], #
				"warp_input_indices": [0], #indices of network inputs to be warped. first the the density inputs, then the elements of step_input_features
				
				"share_downscale_encoder": False,
				
				"recursive_MS": False, #do recursion and growing in NeuralGrid instead of UNet/Model. evey level uses the full specified model, potetially with multiple levels.
				"recursive_MS_levels": "VARIABLE",
				"recursive_MS_direct_input": False, #if false: generate density decoder input at highest scale/resolution and sample down. if true: generate input at resolution required for current scale.
				"recursive_MS_use_max_level_input": False, 
				"recursive_MS_scale_factor": 2,
				"recursive_MS_shared_model": True,
				"recursive_MS_train_mode": "ALL", #ALL, TOP
				"recursive_MS_residual_weight": None, 
				"skip_merge_weight_schedule": sSchedule.setup_linear_schedule_2(start=1., end=1.0, steps=650, offset=250),
				"grow_intervals": [],
				"recursive_MS_copy_on_grow": True,
			#	"load":None, #path to model file
			},
			'pre_optimization': False,
			'train_decoder': sSchedule.setup_constant_schedule(start=True),
		},
	},
	'validation':{
		'input_view_mask':[1,3],
		'simulation':80,
		'start':80,
		'stop':141,
		'step':50,
		'batch_size':2,
		'warp_test':[],
		'warp_test_render':False,
		'render_cycle':False,
		'render_cycle_steps':8,
		'render_density':True,
		'render_shadow':True,
		'render_target':True,
		'render_velocity':False,
	#	'sequence_step':1,
	#	'sequence_length':1,
	},
}
RECONSTRUCT_SEQUENCE_SETUP_NEURAL_DENSITY = _CSETUPS_update_dict_recursive(RECONSTRUCT_SEQUENCE_SETUP_BASE, RECONSTRUCT_SEQUENCE_SETUP_NEURAL_DENSITY, deepcopy=True, new_key='KEEP')

RECONSTRUCT_SEQUENCE_SETUP_NEURAL_VELOCITY = {
	"desc":"train neural networks to generate velocity directly from consecutive densities.",
	"data":{
		'grid_size':48, #128, #resolution
		'y_scale': 2, # "SF" ~1.77 (16/9)
		'clip_grid':False,
		'clip_grid_pad':4,
		'crop_grid':True,
		'res_down_factor':128*5, #target data down scale factor is 6 (i.e. 1/6 of raw image resolution) at 128 base grid resolution
		
		'start':20,#56,#40,
		'stop':141,#68,#52, #exclusive
		'step':2,#2,
		# 
		"randomize":64,
		"batch_size":4,
		"sims":list(range(20)),
		"sequence_step":1,
		"sequence_length":2,
		"density":{
			'target_type': "PREPROC", #RAW, PREPROC, SYNTHETIC
			'scale':1,
			'inflow':{
				'active':False,
				'hull_height':4,
				'height':'MAX',
			},
		},
	},
	"training":{
		'iterations':0,
		"view_encoder":{
			#"active":True, active= density.decoder.active or velocity.decoder.active
			'encoder': ['L'], #NETWORK, NONE
			# 'layers':[4,8,16,16,16,4],
			# 'stride':1,
			# 'skip_connections':None,
			# 'skip_mode':"CONCAT",
			# 'kernel_size':4,
			# 'padding':'MIRROR', #NONE(=valid), ZERO(=same), MIRROR(=tf.pad(REFLECT))
			# 'activation':'lrelu', # relu or lrelu
			# 'activation_alpha':0.2, # leak for lrelu
			# "load_encoder":None,
			"lifting":"UNPROJECT", #UNPROJECT ;2D->3D method
			#"load_lifting":None, #lifting could be NN
			"merge":"MEAN", # SUM, MEAN ;method to merge 3D embeddings of multiple views
			#"load_merge":None, #merging could be NN
			
			"model":{
				"num_levels": "VARIABLE",
				"level_scale_factor": 2,
				
				"input_levels": 1,
				"create_inputs": True,
				"input_blocks": ["C:1-1"],
				#"input_conv_filters": 1,
				#"input_conv_kernel_size": 1,
				"share_input_layer": False,
				
				"down_mode": "STRIDED", # STRIDED, NONE
				"down_conv_filters": None,
				"down_conv_kernel_size": 4,
				"share_down_layer": True,
				
				"encoder_resblocks": ["RB:1-1_1-1"],
				"share_encoder": False,
				
				"decoder_resblocks": ["RB:1-1_1-1"],
				"share_decoder": False,
				
				"up_mode": "NNSAMPLE_CONV",
				"up_conv_filters": 1,
				"up_conv_kernel_size": 4,
				"share_up_layer": True,
				
				"skip_merge_mode": "CONCAT", # CONCAT, WSUM, SUM
				
				#"output_levels": 1,
				"output_blocks": None,
				"output_channels": 1,
				"share_output_layer": True,
				"output_activation": "none", # none, relu, lrelu
				"output_conv_kernel_size": 1,
				"output_mode": "SINGLE", # SINGLE, RESIDUAL
				# global settings
				"conv_activation": "relu", # none, relu, lrelu
				"alpha": 0.2, # leak for lrelu
				"conv_padding": "ZERO", # ZERO, MIRRIR
				"normalization": "LAYER", # NONE, LAYER
			}, # or path to model file
			"start_level": 0,
			"train_mode": "ALL", #"ALL", "TOP", "TOP_DEC"
			"train_mode_schedule": sSchedule.setup_boolean_schedule(start=True, offset=0),
			"skip_merge_weight_schedule": sSchedule.setup_linear_schedule_2(start=1., end=1.0, steps=650, offset=250),
			"grow_intervals": [],
		},
		"volume_encoder":{
			"active":False,
			#"model":"decoder_model",
			"model":{
				"num_levels": "VARIABLE",
				"level_scale_factor": 2,
				
				"input_levels": 1,
				"create_inputs": True,
				"input_blocks": ["C:1-1"],
				#"input_conv_filters": 1,
				#"input_conv_kernel_size": 1,
				"share_input_layer": False,
				
				"down_mode": "STRIDED", # STRIDED, NONE
				"down_conv_filters": None,
				"down_conv_kernel_size": 4,
				"share_down_layer": True,
				
				"encoder_resblocks": ["RB:1-1_1-1"],
				"share_encoder": False,
				
				"decoder_resblocks": ["RB:1-1_1-1"],
				"share_decoder": False,
				
				"up_mode": "NNSAMPLE_CONV",
				"up_conv_filters": 1,
				"up_conv_kernel_size": 4,
				"share_up_layer": True,
				
				"skip_merge_mode": "CONCAT", # CONCAT, WSUM, SUM
				
				#"output_levels": 1,
				"output_blocks": None,
				"output_channels": 1,
				"share_output_layer": True,
				"output_activation": "none", # none, relu, lrelu
				"output_conv_kernel_size": 1,
				"output_mode": "SINGLE", # SINGLE, RESIDUAL
				# global settings
				"conv_activation": "relu", # none, relu, lrelu
				"alpha": 0.2, # leak for lrelu
				"conv_padding": "ZERO", # ZERO, MIRRIR
				"normalization": "LAYER", # NONE, LAYER
			}, # or path to model file
		},
		"lifting_network":{
			"active":False,
			#"model":"decoder_model",
			"model":{
				"num_levels": "VARIABLE",
				"level_scale_factor": 2,
				
				"input_levels": 1,
				"create_inputs": True,
				"input_blocks": ["C:1-1"],
				#"input_conv_filters": 1,
				#"input_conv_kernel_size": 1,
				"share_input_layer": False,
				
				"down_mode": "STRIDED", # STRIDED, NONE
				"down_conv_filters": None,
				"down_conv_kernel_size": 4,
				"share_down_layer": True,
				
				"encoder_resblocks": ["RB:1-1_1-1"],
				"share_encoder": False,
				
				"decoder_resblocks": ["RB:1-1_1-1"],
				"share_decoder": False,
				
				"up_mode": "NNSAMPLE_CONV",
				"up_conv_filters": 1,
				"up_conv_kernel_size": 4,
				"share_up_layer": True,
				
				"skip_merge_mode": "CONCAT", # CONCAT, WSUM, SUM
				
				#"output_levels": 1,
				"output_blocks": None,
				"output_channels": 1,
				"share_output_layer": True,
				"output_activation": "none", # none, relu, lrelu
				"output_conv_kernel_size": 1,
				"output_mode": "SINGLE", # SINGLE, RESIDUAL
				# global settings
				"conv_activation": "relu", # none, relu, lrelu
				"alpha": 0.2, # leak for lrelu
				"conv_padding": "ZERO", # ZERO, MIRRIR
				"normalization": "LAYER", # NONE, LAYER
			}, # or path to model file
			"start_level": 0,
			"min_grid_res": 8,
			"train_mode": "ALL", #"ALL", "TOP", "TOP_DEC"
			"train_mode_schedule": sSchedule.setup_boolean_schedule(start=True, offset=0),
			"skip_merge_weight_schedule": sSchedule.setup_linear_schedule_2(start=1., end=1.0, steps=650, offset=250),
			"grow_intervals": [],
		},
		"frame_merge_network":{
			"active":False,
			#"model":"decoder_model",
			"model":{
				"num_levels": "VARIABLE",
				"level_scale_factor": 2,
				
				"input_levels": 1,
				"create_inputs": True,
				"input_blocks": ["C:1-1"],
				#"input_conv_filters": 1,
				#"input_conv_kernel_size": 1,
				"share_input_layer": False,
				
				"down_mode": "STRIDED", # STRIDED, NONE
				"down_conv_filters": None,
				"down_conv_kernel_size": 4,
				"share_down_layer": True,
				
				"encoder_resblocks": ["RB:1-1_1-1"],
				"share_encoder": False,
				
				"decoder_resblocks": ["RB:1-1_1-1"],
				"share_decoder": False,
				
				"up_mode": "NNSAMPLE_CONV",
				"up_conv_filters": 1,
				"up_conv_kernel_size": 4,
				"share_up_layer": True,
				
				"skip_merge_mode": "CONCAT", # CONCAT, WSUM, SUM
				
				#"output_levels": 1,
				"output_blocks": None,
				"share_output_layer": True,
				"output_activation": "none", # none, relu, lrelu
				"output_conv_kernel_size": 1,
				"output_mode": "SINGLE", # SINGLE, RESIDUAL
				# global settings
				"conv_activation": "relu", # none, relu, lrelu
				"alpha": 0.2, # leak for lrelu
				"conv_padding": "ZERO", # ZERO, MIRRIR
				"normalization": "LAYER", # NONE, LAYER
			}, # or path to model file
			"start_level": 0,
			"train_mode": "ALL", #"ALL", "TOP", "TOP_DEC"
			"train_mode_schedule": sSchedule.setup_boolean_schedule(start=True, offset=0),
			"skip_merge_weight_schedule": sSchedule.setup_linear_schedule_2(start=1., end=1.0, steps=650, offset=250),
			"grow_intervals": [],
		},
		"density":{
			"decoder":{
				"active":True,
				'input_type':"PREPROC", #use raw or preproc images
				"model": "[RUNID:210305-172218]density_decoder", # or path to model file
				"start_level": 0,
				"min_grid_res": 3,
				"skip_merge_weight_schedule": sSchedule.setup_linear_schedule_2(start=1., end=1.0, steps=650, offset=250),
				"grow_intervals": [],
			},
			'pre_optimization': False,
		},
		"velocity":{
			"decoder":{
				"active":True,
				'input_type':"PREPROC", #use raw or preproc images
				'velocity_format': "CENTERED", #CENTERED, STAGGERED
				"model": {
					#"dimension": 2,
					"num_levels": "VARIABLE",
					"level_scale_factor": 2,
					
					#"input_channels": 3,
					"input_levels": -1,
					"create_inputs": True,
					"input_blocks": ["C:16-1"],
					#"input_conv_filters": 1,
					#"input_conv_kernel_size": 1,
					"share_input_layer": True,
					
					"down_mode": "NONE", # STRIDED, NONE
					"down_conv_filters": None,
					"down_conv_kernel_size": 4,
					"share_down_layer": True,
					
					"encoder_resblocks": ["RB:8_16_s0","RB:8_16_s0","RB:8_16_s0","RB:8_16_s0"],
					"share_encoder": True,
					
					"decoder_resblocks": ["RB:8_16","RB:8_16_s0","RB:8_16_s0","RB:8_16_s0"],
					"share_decoder": True,
					
					"up_mode": "NNSAMPLE_CONV",
					"up_conv_filters": 16,
					"up_conv_kernel_size": 4,
					"share_up_layer": True,
					
					"skip_merge_mode": "CONCAT", # CONCAT, WSUM, SUM
					
					#"output_levels": 1,
					"output_blocks": ["RB:8_16_s0","RB:8_16_s0","RB:8_16_s0","RB:8_16_s0","RB:8_16_s0"],
					#"output_channels": 1,
					"share_output_layer": True,
					"output_activation": "none", # none, relu, lrelu
					"output_conv_kernel_size": 1,
					"output_mode": "SINGLE", # SINGLE, RESIDUAL
					# global settings
					"conv_activation": "relu", # none, relu, lrelu
					"alpha": 0.2, # leak for lrelu
					"conv_padding": "ZERO", # ZERO, MIRRIR
					"normalization": "LAYER", # NONE, LAYER
				}, #3D features to velocity
				"start_level": 0,
				"min_grid_res": 3,
				"skip_merge_weight_schedule": sSchedule.setup_linear_schedule_2(start=1., end=1.0, steps=650, offset=250),
				# network inputs
				"step_input_density": [], #0 is the current frame, 1 the next, etc. can be negative
				"step_input_density_target": [], #0 is the current frame, 1 the next, etc. can be negative
				"step_input_density_proxy": [], #0 is the current frame, 1 the next, etc. can be negative
				"step_input_features": [0,1],
				"type_input_features": ["TARGET_UNPROJECTION"],
				"downscale_input_modes": ["RESAMPLE", "RESAMPLE"], #
				"warp_input_indices": [0], #indices of network inputs to be warped. first the the density inputs, then the elements of step_input_features
				
				
				
				"recursive_MS": False, #do recursion and growing in NeuralGrid instead of UNet/Model. evey level uses the full specified model, potetially with multiple levels.
				"recursive_MS_levels": "VARIABLE",
				"recursive_MS_direct_input": False, #if false: generate density decoder input at highest scale/resolution and sample down. if true: generate input at resolution required for current scale.
				"recursive_MS_use_max_level_input": False, 
				"recursive_MS_scale_factor": 2,
				"recursive_MS_shared_model": True,
				"recursive_MS_train_mode": "ALL", #ALL, TOP
				"recursive_MS_residual_weight": None, 
				"grow_intervals": [1200,1400,1600,1800],
				"recursive_MS_copy_on_grow": True,
				"train_mode": "ALL", #"ALL", "TOP", "TOP_DEC"
				"train_mode_schedule": sSchedule.setup_boolean_schedule(start=True, offset=0),
			},
			'pre_optimization': True,
			'pre_opt':{
				'first':{ #settings for first frame
					'iterations':16000,#10000, #30000,
					'learning_rate':{'type':'exponential', 'max':1e-5, 'start':1e-5, 'base':0.5, 'scale':1/2000, 'offset':0},
					
					'density_warp_loss':1, 
					'velocity_warp_loss':0.0,
					'smoothness_loss':0.0,
					'smoothness_neighbours':3,
					'cossim_loss':0.0,
					'divergence_loss': sSchedule.setup_linear_schedule_2(start=0.0,end=2.0,steps=5000,offset=1000),
					'magnitude_loss':{'type':'exponential', 'start':2e-2, 'base':0.5, 'scale':1/2000},
					'CFL_loss':0.0,
					'MS_coherence_loss':0.0,
					'regularization':0.0001,
					'grow':{
						"factor":2,
						"scale_magnitude":False,
						'intervals':[1200,1400,1600,1800], #6000
					},
				
				},
			},
		},
	},
	'validation':{
		'input_view_mask':[1,3],
		'simulation':80,
		'start':80,
		'stop':142,
		'step':50,
		'batch_size':2,
		'warp_test':[],
		'warp_test_render':False,
		'render_cycle':True,
		'render_cycle_steps':8,
		'render_density':True,
		'render_shadow':False,
		'render_target':False,
		'render_velocity':True,
	#	'sequence_step':1,
	#	'sequence_length':1,
	},
}
RECONSTRUCT_SEQUENCE_SETUP_NEURAL_VELOCITY = _CSETUPS_update_dict_recursive(RECONSTRUCT_SEQUENCE_SETUP_BASE, RECONSTRUCT_SEQUENCE_SETUP_NEURAL_VELOCITY, deepcopy=True, new_key='KEEP')

#RECONSTRUCT_SEQUENCE_SETUP_NEURAL = TODO


def reconstruct_sequence_setup_compatibility(setup, log_func=lambda s, *p: None):
	'''
		backwards compat for changes in the configuration
	'''
	setup = copy.deepcopy(setup)
	setup = munch.munchify(setup)
	# changed/replaced keys
	def log_key_update(k1, v1, k2, v2):
		log_func("Update old key '%s' with value '%s' to '%s' with value '%s'", k1, v1, k2, v2)
	#	target type
	if "synthetic_target" in setup.data.density and not "target_type" in setup.data.density:
		setup.data.density.target_type = "SYNTHETIC" if setup.data.density.synthetic_target else "RAW"
		log_key_update('setup.data.density.synthetic_target', setup.data.density.synthetic_target, 'setup.data.density.target_type', setup.data.density.target_type)
		del setup.data.density.synthetic_target
	
	# adjustments for mechanical changes
	#	if using data from before the sampling step correction: multiply density with 256, warn about loss and shadow/light scaling

# old setups