import subprocess, argparse, json, os, sys, copy
import munch
from lib.util import RunIndex, PartialFormatter

SDC = 256 #step density correction
setup_warp_test = {
	'title':'eval{stats}{note}_gs{gridsize}_R{runid}_{title}', 
	'desc':'evaluation of run {runid:} "{title}".',
	'paths':{
		'group':"eval",
	},
	'rendering':{
		'monochrome':False,
	#	'luma':[],
		'filter_mode':'LINEAR', #NEAREST, LINEAR
		#'boundary':'BORDER', #BORDER, CLAMP
		'mip':{
			'mode':'LINEAR', #NONE, NEAREST, LINEAR
			'level':4,
			'bias':0.0,
		},
		'blend_mode':'BEER_LAMBERT', #BEER_LAMBERT, ALPHA, ADDITIVE
		'sample_gradients':True,
		
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
		
		'allow_static_cameras':False,
		'allow_fused_rendering':True,
		
		'target_cameras':{
		},
		
		'background':{
			'type':'COLOR', #'CAM', 'COLOR', 'NONE'; only for vis-rendering, not used in optimization
			'color': [0,0.5,1.0], #[0,0.5,0],
		},
		
		'lighting':{
			'ambient_intensity':0.64,
			'initial_intensity':0.85,#2.4,#1.8, corrected up due to increased shadow falloff after step scaling fix
			'shadow_resolution':[256,196,196],#[64,64,64], #DHW
		},
		"velocity_scale":2.0*SDC,
	},#rendering
	'data':{
		'run_dirs':['runs/sequence_reconstruction'],
		'grid_size':128, #resolution
		
		'simulation':0,
		'start':82,#56,#40,
		'stop':90,#68,#52, #exclusive
		'step':2,#3,
		"sims":[0], #,2,3,4],
		'scalarFlow_frame_offset': 0, #-11,
		'density':{
			'scale': 1., #0.01*SDC,
			'initial_value':'data/scalarFlow/sim_{sim:06d}/reconstruction/density_{frame:06d}.npz',
			'min':0.0,
			'max':1.0 *SDC,
			'target_type': "RAW", #RAW, PREPROC, SYNTHETIC
			'target_cam_ids': "ALL",
			'target': 'data/scalarFlow/sim_{sim:06d}/input/cam/imgsUnproc_{frame:06d}.npz',
			'target_preproc': 'data/scalarFlow/sim_{sim:06d}/input/postprocessed/imgs_{frame:06d}.npz',
			'target_threshold':4e-2,
			'target_scale': 1.5,
			'hull_image_blur_std':1.0,
			'hull_volume_blur_std':0.5,
			'hull_smooth_blur_std':0.0,#2.5,
			'hull_threshold':4e-2,
			'inflow':{
				'active':True,
				'hull_height':4,
				'height':'MAX',
			},
			'scalarFlow_reconstruction':'data/scalarFlow/sim_{sim:06d}/reconstruction/density_{frame:06d}.npz',
		},
		'velocity':{
			'initial_value':'data/scalarFlow/sim_{sim:06d}/reconstruction/velocity_{frame:06d}.npz',
			'load_step':2,
			'init_std':0.1,
			'boundary':'CLAMP', #BORDER (closed 0 bounds), CLAMP (open bounds)
			'scalarFlow_reconstruction':'data/scalarFlow/sim_{sim:06d}/reconstruction/velocity_{frame:06d}.npz',
		},
		'initial_buoyancy':[0.,0.,0.],
		'discriminator':{
			'simulations':[0,6],
			'frames':[45,145, 1],
			'target_type': "PREPROC", #RAW, PREPROC
			#augmentation
			'crop_size':[96,96],
			'real_res_down': 4,
			'scale_range':[0.52,1.0],
			'scale_real':[0.8, 1.8], #range for random intensity scale on real samples, due to background substraction these tend to be darker than the images generated when using raw targets, so ~ *1.5 to correct
			'scale_fake':[0.7, 1.4],
			'gamma_real':[0.5,2], #range for random gamma correction on real samples (value here is inverse gamma)
			'gamma_fake':[0.5,2], #range for random gamma correction on fake samples (value here is inverse gamma)
			
		},
		'load_sequence':'[RUNID:{runid:}]', #None, #only for rendering without optimization
		'load_sequence_pre_opt':False,
	},#data
	'training':{
		'iterations':0,
		
		'resource_device':'/cpu:0', #'/cpu:0', '/gpu:0'
		
		#'loss':'L2',
		'train_res_down':6,
		'loss_active_eps':1e-08, #minimum absolute value of a loss scale for the loss to be considered active (to prevent losses scaled to (almost) 0 from being evaluated)
		"view_encoder":{
			#"active":True, active= density.decoder.active or velocity.decoder.active
			"model": "[RUNID:{runid:}]target_encoder{suffix}",
		},
		"volume_encoder":{
			"active":False,
			#"model":"decoder_model",
			"model": "[RUNID:{runid:}]volume_encoder{suffix}",
		},
		"lifting_network":{
			"active":False,
			#"model":"decoder_model",
			"model": "[RUNID:{runid:}]lifting_network{suffix}",
		},
		"frame_merge_network":{
			"active":False,
			#"model":"decoder_model",
			"model": "[RUNID:{runid:}]frame_merge_network{suffix}",
		},
		'density':{
			'use_hull':False, #[],
			'warp_clamp':"MC_SMOOTH",
			'pre_optimization':True, #whether pre-optim will run for density, affects forward propagation/advection of state and optimization
			# to only have fwd advection without optimization set iterations to 0
			# to have pre-opt-like optimization without advection disable pre-opt and use loss schedules
			"decoder":{
				"active":False,
				"model":"[RUNID:{runid:}]density_decoder{suffix}", #path to model file
			},
			'pre_opt':{
				'first':{ #settings for first frame
					'iterations':0,
				},
				#settings for remaining frames
				'iterations':0,
				'inspect_gradients':False,
			},
			'grow':{ # not yet used
				"factor":1.2,#2. factor per interval, max down-scaling is factor^len(intervals)
				#iterations for each grow step, empty to disable
				'intervals':[],
			},
		},
		'velocity':{
			'warp_order':2,
			'warp_clamp':"MC_SMOOTH",
		#	'use_hull':False,
			'pre_optimization':False, 
			"decoder":{
				"active":True,
				"model":"[RUNID:{runid:}]velocity_decoder{suffix}", #path to model file
				"min_grid_res": 10,
				"step_input_density": [0], #0 is the current frame, 1 the next, etc. can be negative
				"step_input_density_target": [], #0 is the current frame, 1 the next, etc. can be negative
				"step_input_density_proxy": [], #0 is the current frame, 1 the next, etc. can be negative
				"step_input_features": [0,1],
				"warp_input_indices": [0,1], #indices of network inputs to be warped. first the the density inputs, then the elements of step_input_features
				"recursive_MS_levels": "VARIABLE",
				"recursive_MS_scale_factor": 1.4,
			},
			'pre_opt':{
				'first':{ #settings for first frame
					'iterations':0,
					'grow':{
						"factor":1.2,
						"scale_magnitude":False,
						'intervals':[],
					},
				
				},
				#settings for remaining frames
				'iterations':0,
			},
			
			'grow':{
				"factor":1.2,#2. factor per interval, max down-scaling is factor^len(intervals)
				"scale_magnitude":False,
				'intervals':[],
			},
		},
		'optimize_buoyancy':False,
		"light":{
			"optimize":False,
			"min":0.01,
			"max":6.0,
			"learning_rate":{'type':'exponential', 'start':0.001, 'base':0.5, 'scale':2/3000},#0.0002,
		},
		
		'discriminator':{
			'active':False,
		},#discriminator
		'summary_interval':100,
		#'test_interval':250,
	},#training
	'validation':{
		'output_interval':100,
		'input_view_mask': None, #[0,1,2,3,4],
		'stats':False,
		'cmp_vol_targets':False,
		'cmp_scalarFlow':False,
		'cmp_scalarFlow_render':False,
		'warp_test':[],#False,
		'warp_test_render':False,
		'render_cycle':False,
		'render_target':False,
		'render_MS':False,
	}
}

def target_scale_from_grid_size(grid_size):
	# down-scale factor for target/input images: 6 at 128, 12 at 64
	return int(6 * 128/grid_size)

if __name__=='__main__':
	base_dirs = ['./runs/test-and-debug/'] #'./runs/', './runs/test-and-debug/', '/data/erik/voldifrender/runs/sequence_recon_test/'
	
	parser = argparse.ArgumentParser(description='Load and evaluate or render runs.')
	parser.add_argument('runIDs', metavar='R', type=str, nargs='*', default=[], help='ids/timestamps of the runs to test')
	parser.add_argument('-d', '--baseDirs', dest='base_dirs', metavar='D', type=str, nargs='*', default=base_dirs, help='directories to search for runs')
	parser.add_argument('-o', '--out', dest='out_path', type=str, default='/data/erik/voldifrender/runs_eval', help='path to result directory')
	parser.add_argument('--noRecursive', dest='recursive', action='store_false', help='search base directories recursive')
	parser.add_argument('-w','--warp', dest='warp', type=int, nargs='*', default=None, help='run warp test')
	parser.add_argument('-W','--warpRender', dest='warp_render', nargs='*', default=None, help='run warp test with render, overrides -w')
	parser.add_argument('--warpOrder', dest='warp_order', type=int, default=0, help='order of integration scheme used in the warp test, requires -w or -W')
	parser.add_argument('-s','--scalarFlow', dest='sF', action='store_true', help='run scalarFlow comparison')
	parser.add_argument('-S','--scalarFlowRender', dest='sF_render', action='store_true', help='run scalarFlow comparison with render, overrides -s')
	parser.add_argument('-v','--volTar', dest='volTar', action='store_true', help='run volume target comparison. requires --stats')
	#parser.add_argument('-V','--scalarFlowRender', dest='sF_render', action='store_true', help='run scalarFlow comparison with render, overrides -s')
	parser.add_argument('-r','--render', dest='render', action='store_true', help='render sequence')
	parser.add_argument('-g','--gridSize', dest='grid_size', type=int, default=64, help='resolution of grid')
	parser.add_argument('--simulation', dest='simulation', type=int, default=0, help='id of SF sim  to use.')
	parser.add_argument('--frameRange', dest='frame_range', type=int, nargs=3, default=None, help='range (start, stop, step) of frames to use.')
	parser.add_argument('--camIDs', dest='cam_ids', type=int, nargs='+', default=None, help='targets/views to use as inputs. defaults to all.')
	#parser.add_argument('--viewMask', dest='view_mask', type=int, nargs='+', default=None, help='targets/views to use as targets. defaults to all.')
	parser.add_argument('--stats', dest='stats', action='store_true', help='compute statistics.')
	#parser.add_argument('--renderCycle', dest='render_cycle', action='store_true', help='render still cycle views, requires -r')
	parser.add_argument('--renderCycle', dest='render_cycle', type=int, nargs='?', const=12, default=None, help='render still cycle views, requires -r')
	parser.add_argument('--renderDensity', dest='render_density', action='store_true', help='render density, requires -r')
	parser.add_argument('--renderShadow', dest='render_shadow', action='store_true', help='render render high density with shadow, requires --renderDensity')
	parser.add_argument('--renderTarget', dest='render_target', action='store_true', help='render target views, requires -r')
	parser.add_argument('--renderVelocity', dest='render_velocity', action='store_true', help='render velocity, requires -r')
	parser.add_argument('--renderMS', dest='render_MS', action='store_true', help='render multi-scale grids, if available. requires -r')
	parser.add_argument('--testCams', dest='test_cams', action='store_true', help='render test cams, requires --renderTarget')
	parser.add_argument('--preOpt', dest='pre_opt', action='store_true', help='load pre-optimization result')
	parser.add_argument('--densScale', dest='dens_scale', type=float, default=1, help='scaling factor for density')
	parser.add_argument('--lightIntensity', dest='light_intensity', type=float, default=None, help='intensity for the shadow point light')
	parser.add_argument('--ambientIntensity', dest='ambient_intensity', type=float, default=None, help='intensity for the ambient light')
	parser.add_argument('--bkgColor', dest='bkg_color', type=float, nargs=3, default=None, help='RGB background color')
	parser.add_argument('-t', '--title', dest='title', type=str, default=None, help='title, overrides original run title')
	parser.add_argument('-n', '--note', dest='note', type=str, default=None, help='prepends a note to the title')
	parser.add_argument('--device', dest='device', type=str, default=None, help='GPU to use')
	parser.add_argument('--debug', dest='debug', action='store_true', help='run eval script in debug mode')
	parser.add_argument('--modelName', dest='modelName', type=str, default='', help='model name suffix to use. e.g.: "", "_ckp", "_part". Defaults to the final model "".')
	parser.add_argument('--evalDataSetup', dest='evalDataSetup', type=str, default='SPHERE', help='data-setup for evaluation: SF, SPHERE, CUBE, ROTCUBE.')
	parser.add_argument('--evalDataRunID', dest='evalDataRunID', type=str, default=None, help='. requires --evalDataSetup SF.')
	parser.add_argument('--synthMaxT', dest='synthMaxT', type=float, default=None, help='maximum translation for SPHERE, CUBE.')
	parser.add_argument('--saveVol', dest='save_volume', action='store_true', help='save final volumes.')
	
	args = parser.parse_args()
	
	setup_file = os.path.join(args.out_path, "warp_test_setup.json")
	
	run_index = RunIndex(args.base_dirs, recursive=args.recursive)
	if not args.runIDs:
		runs = run_index.query_runs()
	else:
		runs = [run_index.runs[_] for _ in args.runIDs]
	
	f = PartialFormatter()
	
	os.makedirs(args.out_path, exist_ok=True)
	c = r'-\|/'
	interval = 0.5
	i=0
	for run_entry in runs:
		runid = run_entry.runid
		run_config = munch.munchify(run_entry.config)
		setup = munch.munchify(copy.deepcopy(setup_warp_test))
		stats_title = ""
		
		setup.paths.base = args.out_path
		setup.paths.group = ""
		setup.data.run_dirs += [run_entry.parentdir]
		setup.data.load_sequence = f.format(setup.data.load_sequence, runid=runid)
		if args.pre_opt:
			stats_title += '_pO'
			setup.data.load_sequence_pre_opt = args.pre_opt
		if args.stats:
			stats_title += '_stats'
			setup.validation.stats = True
		if args.sF_render:
			stats_title += '_sFcmpR'
			setup.validation.cmp_scalarFlow = True
			setup.validation.cmp_scalarFlow_render = True
		elif args.sF:
			stats_title += '_sFcmp'
			setup.validation.cmp_scalarFlow = True
		if args.volTar:
			stats_title += '_vTcmp'
			setup.validation.cmp_vol_targets = True
		warp_args = args.warp
		warp_title = '_warp'
		if args.warp_render is not None:
			warp_args = args.warp_render
			warp_title = '_warpR'
			setup.validation.warp_test_render = True
		if warp_args is not None:
			stats_title += warp_title
			setup.validation.warp_test = warp_args if len(warp_args)>0 else ["BASE"] #, "FIXED_INFLOW"
		if args.warp_order>0:
			setup.training.velocity.warp_order = args.warp_order
			stats_title += '-%d' % args.warp_order
		else:
			setup.training.velocity.warp_order = run_config.training.velocity.warp_order
		if "warp_clamp" in run_config.training.density:
			setup.training.density.warp_clamp = run_config.training.density.warp_clamp
		if "warp_clamp" in run_config.training.velocity:
			setup.training.velocity.warp_clamp = run_config.training.velocity.warp_clamp
		
		if args.light_intensity is not None:
			setup.rendering.lighting.initial_intensity = args.light_intensity
		elif run_entry.scalars.get("light_intensity") is not None:
			setup.rendering.lighting.initial_intensity = run_entry.scalars["light_intensity"][0]
		else:
			setup.rendering.lighting.initial_intensity = run_config.rendering.lighting.initial_intensity
		if args.ambient_intensity is not None:
			setup.rendering.lighting.ambient_intensity = args.ambient_intensity
		elif run_entry.scalars.get("light_intensity") is not None:
			setup.rendering.lighting.ambient_intensity = run_entry.scalars["light_intensity"][1]
		else:
			setup.rendering.lighting.ambient_intensity = run_config.rendering.lighting.ambient_intensity
		if args.test_cams:
			setup.rendering.target_cameras.calibration_file = "test_cameras.json"
			setup.rendering.target_cameras.camera_ids = [_ for _ in range(32)]
		elif "target_cameras" in run_config.rendering:
			setup.rendering.target_cameras = run_config.rendering.target_cameras
		if args.bkg_color is not None:
			setup.rendering.background.color = args.bkg_color
		
		setup.rendering.boundary = run_config.rendering.get("boundary", None)
		#test override
		#setup.training.density.warp_clamp = "MC_SMOOTH"
		#setup.training.velocity.warp_clamp = "MC_SMOOTH"
		
		setup.training.view_encoder.lifting = run_config.training.view_encoder.lifting
		setup.training.view_encoder.encoder = run_config.training.view_encoder.encoder
		if "NETWORK" in run_config.training.view_encoder.encoder:
			setup.training.view_encoder.model = f.format(setup.training.view_encoder.model, runid=runid, suffix=args.modelName)
		
		if run_config.training.volume_encoder.active:
			setup.training.volume_encoder.active = True
			setup.training.volume_encoder.model = f.format(setup.training.volume_encoder.model, runid=runid, suffix=args.modelName)
		
		if "lifting_network" in run_config.training and run_config.training.lifting_network.active:
			setup.training.lifting_network.active = True
			setup.training.lifting_network.model = f.format(setup.training.lifting_network.model, runid=runid, suffix=args.modelName)
		
		if run_config.training.frame_merge_network.active:
			setup.training.frame_merge_network.active = True
			setup.training.frame_merge_network.model = f.format(setup.training.frame_merge_network.model, runid=runid, suffix=args.modelName)
		
		if run_config.training.density.decoder.active:
			setup.training.density.decoder.model = f.format(setup.training.density.decoder.model, runid=runid, suffix=args.modelName)
			setup.training.density.decoder.recursive_MS = run_config.training.density.decoder.recursive_MS
			setup.training.density.decoder.recursive_MS_scale_factor = run_config.training.density.decoder.model.level_scale_factor if not hasattr(run_config.training.density.decoder, "recursive_MS_scale_factor") else run_config.training.density.decoder.recursive_MS_scale_factor
			setup.training.density.decoder.recursive_MS_direct_input = run_config.training.density.decoder.get("recursive_MS_direct_input", False)
			setup.training.density.decoder.recursive_MS_levels = "VARIABLE" #1 #run_config.training.density.decoder.recursive_MS_levels #
			setup.training.density.decoder.recursive_MS_residual = run_config.training.density.decoder.get("recursive_MS_residual", True)
			setup.training.density.decoder.min_grid_res = run_config.training.density.decoder.min_grid_res #5 #
			setup.training.density.decoder.base_input = run_config.training.density.decoder.get("base_input", "ZERO")
			setup.training.density.decoder.step_input_density = run_config.training.density.decoder.step_input_density
			setup.training.density.decoder.step_input_density_target = run_config.training.density.decoder.step_input_density_target
			setup.training.density.decoder.step_input_features = run_config.training.density.decoder.step_input_features
			#setup.training.density.decoder.warp_input_indices = run_config.training.density.decoder.warp_input_indices
		else:
			setup.training.density.decoder.model = None
		
		#print(run_config.training.velocity.decoder.active)
		setup.training.velocity.decoder.model = f.format(setup.training.velocity.decoder.model, runid=runid, suffix=args.modelName) if run_config.training.velocity.decoder.active else None
		setup.training.velocity.decoder.recursive_MS = run_config.training.velocity.decoder.recursive_MS
		setup.training.velocity.decoder.recursive_MS_scale_factor = run_config.training.velocity.decoder.model.level_scale_factor if not hasattr(run_config.training.velocity.decoder, "recursive_MS_scale_factor") else run_config.training.velocity.decoder.recursive_MS_scale_factor
		setup.training.velocity.decoder.recursive_MS_levels = "VARIABLE" #1 #run_config.training.velocity.decoder.recursive_MS_levels #
		setup.training.velocity.decoder.min_grid_res = run_config.training.velocity.decoder.min_grid_res #5 #
		setup.training.velocity.decoder.step_input_density = run_config.training.velocity.decoder.step_input_density
		setup.training.velocity.decoder.step_input_density_target = run_config.training.velocity.decoder.step_input_density_target
		setup.training.velocity.decoder.step_input_density_proxy = run_config.training.velocity.decoder.get("step_input_density_proxy", [])
		setup.training.velocity.decoder.step_input_features = run_config.training.velocity.decoder.step_input_features
		setup.training.velocity.decoder.warp_input_indices = run_config.training.velocity.decoder.warp_input_indices
		setup.training.velocity.decoder.velocity_format = run_config.training.velocity.decoder.velocity_format
		
		#setup.training.view_encoder.load_encoder = f.format(setup.training.view_encoder.load_encoder, runid=runid)
		
		setup.desc = f.format(setup.desc, runid=runid, title=run_entry.title)
		
		setup.data.res_down_factor = run_config.data.res_down_factor
		setup.data.grid_size = args.grid_size
		setup.data.y_scale = run_config.data.y_scale
		#setup.data.y_scale = 1.5
		setup.training.train_res_down = target_scale_from_grid_size(args.grid_size)
		setup.data.simulation = args.simulation #run_config.data.simulation
		setup.data.sims = [setup.data.simulation]
		if args.frame_range is not None:
			setup.data.start, setup.data.stop, setup.data.step = args.frame_range
		else:
			setup.data.start = run_config.data.start
			setup.data.stop = run_config.data.stop
			setup.data.step = run_config.data.step
		setup.data.velocity.load_step = run_config.data.step
		#setup.data.density.initial_value = f.format(setup.data.density.initial_value, runid=runid)
		setup.data.density.scale = args.dens_scale
		#setup.data.velocity.initial_value = f.format(setup.data.velocity.initial_value, runid=runid)
		if args.synthMaxT is not None:
			setup.data.synth_shapes = munch.Munch(max_translation = args.synthMaxT)
		
		#setup.validation.input_view_mask = args.view_mask if args.view_mask is not None else run_config.validation.input_view_mask
		setup.validation.render_cycle = args.render_cycle is not None
		setup.validation.render_cycle_steps = args.render_cycle if args.render_cycle is not None else 12
		setup.validation.render_density = args.render_density
		setup.validation.render_shadow = args.render_shadow
		setup.validation.render_target = args.render_target
		setup.validation.render_velocity = args.render_velocity
		setup.validation.render_MS = args.render_MS
		
		setup.validation.synth_data_eval_setup = args.evalDataSetup.upper()
		if args.evalDataSetup.upper()=="SF":
			if args.evalDataRunID is not None:
				setup.data.density.initial_value  = f.format('[RUNID:{runid:}]frame_{frame:06d}/density.npz',  runid=args.evalDataRunID)
				setup.data.velocity.initial_value = f.format('[RUNID:{runid:}]frame_{frame:06d}/velocity.npz', runid=args.evalDataRunID)
				setup.data.scalarFlow_frame_offset = 0
			else:
				setup.data.density.initial_value  = run_config.data.density.initial_value
				setup.data.density.target  = run_config.data.density.target
				setup.data.density.target_preproc  = run_config.data.density.target_preproc
				setup.data.velocity.initial_value = run_config.data.velocity.initial_value
				setup.data.scalarFlow_frame_offset = run_config.data.scalarFlow_frame_offset
		
		
		setup.data.SDF = run_config.data.SDF
		
		if args.cam_ids is not None:
			setup.data.density.target_cam_ids = args.cam_ids
			#setup.validation.input_view_mask = args.cam_ids
			stats_title = "_c%s"%(len(args.cam_ids),) + stats_title
		if isinstance(run_config.training.randomization.inputs, list):
			setup.validation.input_view_mask = run_config.training.randomization.inputs
		
		setup.title = f.format(setup.title, note=("_"+args.note if args.note is not None else ""), stats=stats_title, gridsize=args.grid_size, runid=runid, title=(run_entry.title if args.title is None else args.title))
	#	setup.training.velocity.pre_optimization = args.warp_vel
		#if "target_cam_ids" in run_config.data.density:
		#	setup.data.density.target_cam_ids = run_config.data.density.target_cam_ids
		with open(setup_file, 'w') as file:
			json.dump(setup, file)
			file.flush()
		cmd = ['python', "reconstruct_sequence.py",
			'--setup', setup_file,
	#		'--fit', #run optimization
	#		'--noRender',
	#		'--noConsole',
		]
		if not args.render:
			cmd.append('--noRender')
		if args.device is not None:
			cmd.append('--device')
			cmd.append(args.device)
		if args.debug:
			cmd.append("--debug")
		if args.save_volume:
			cmd.append("--saveVol")
		p = subprocess.Popen(cmd)
		p.communicate()
