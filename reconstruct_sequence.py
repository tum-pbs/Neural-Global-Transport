import os, sys, shutil, socket, faulthandler, signal, math, copy, random, psutil
import datetime, time
import logging, warnings, argparse
import json
import munch, collections.abc
import imageio


parser = argparse.ArgumentParser(description='Reconstruct volumetric smoke densities from 2D views.')
parser.add_argument('-s', '--setup', dest='setup_file', default=None, help='setup from JSON file to use')
parser.add_argument('-m', '--model', dest='model', default=None, help='setup from JSON file to use')
parser.add_argument('-d', '--deviceID', dest="cudaID", default="0", help='id of cuda device to use')
parser.add_argument('-r', '--noRender', dest='render', action='store_false', help='turn off final rendering.')
parser.add_argument('--saveVol', dest='save_volume', action='store_true', help='save final volumes.')
parser.add_argument('-f', '--fit', dest='fit', action='store_true', help='run density volume optimization.')
parser.add_argument('-c', '--noConsole', dest='console', action='store_false', help='turn off console output')
parser.add_argument('--debug', dest='debug', action='store_true', help='enable debug output.')
parser.add_argument('--maxMem', dest='max_memory', type=float, default=0.25, help='MiB or fraction of total RAM.')
args = parser.parse_args()

cudaID = args.cudaID


os.environ["CUDA_VISIBLE_DEVICES"]=cudaID
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
tf.enable_resource_variables()

from phitest.render import *
import phitest.render.render_helper as render_helper
#from phitest.render.profiling import Profiler
from phitest.render.profiling import DEFAULT_PROFILER #so other modules can import and use SAMPLE without passing on a Profiler() object.
from phitest.render.serialization import to_dict, from_dict
from lib.logger import StreamCapture
from lib.progress_bar import ProgressBar


from lib.util import *
from lib.scalar_schedule import *
from lib.tf_ops import *
from lib.data import *
from lib.tf_colormap import *
	


def get_clip_nearFar(position, focus, depth):
	cam_dh = depth*0.5 #depth half
	dist = np.linalg.norm(focus-position)
	return [dist-cam_dh,dist+cam_dh]

def build_camera_from_sFcallibration(position, forward, up, right, resolution, fov_horizontal, fov_vertical, focus, focus_depth_clip=1.0, **kwargs):
	flip_z = lambda v: np.asarray(v)*np.asarray([1,1,-1])
	invert_v = lambda v: np.asarray(v)*(-1)
	pos = flip_z(position)
	fwd = invert_v(flip_z(forward))
	up = flip_z(up)
	right = flip_z(right)
	cam_focus = flip_z(focus)
	aspect = fov_horizontal/fov_vertical #resolution[2]/resolution[1] #W/H
	cam_dh = focus_depth_clip*0.5 #depth half
	
	dist = np.linalg.norm(cam_focus-pos)
	cam = Camera(MatrixTransform.from_fwd_up_right_pos(fwd, up, right, pos), nearFar=[dist-cam_dh,dist+cam_dh], fov=fov_horizontal, aspect=aspect, static=None)
	cam.transform.grid_size = copy.copy(resolution)
	
	return cam

def build_scalarFlow_cameras(setup, ids=[2,1,0,4,3], focus_depth_clip=1.0, interpolation_weights=[]):
	scalarFlow_cameras = []
	cam_resolution_scale = 1./setup.training.train_res_down #0.125#0.3
	train_cam_resolution = copy.copy(setup.rendering.main_camera.base_resolution)
	train_cam_resolution[1] = int(train_cam_resolution[1]*cam_resolution_scale)
	train_cam_resolution[2] = int(train_cam_resolution[2]*cam_resolution_scale)
	log.info('scalarFlow train camera resolution: %s', str(train_cam_resolution))
#	cam_dh = focus_depth_clip*0.5 #depth half
	
	aspect = train_cam_resolution[2]/train_cam_resolution[1]
	
	for cam_id in ids:
		cam_calib = setup.calibration[str(cam_id)]
		if cam_calib.fov_horizontal is None:
			cam_calib.fov_horizontal = setup.calibration.fov_horizontal_average
		if setup.data.synth_shapes.active or setup.data.SDF:
			cam_calib.fov_vertical = cam_calib.fov_horizontal/aspect
		else:
			if cam_calib.fov_vertical is None:
				cam_calib.fov_vertical = setup.calibration.fov_vertical_average
	
	for i in range(len(ids)):
		cam_calib = setup.calibration[str(ids[i])]
		cam = build_camera_from_sFcallibration(**cam_calib, **setup.calibration, resolution=train_cam_resolution, focus_depth_clip=focus_depth_clip)
		scalarFlow_cameras.append(cam)
		
		if interpolation_weights and i<(len(ids)-1):
			for w in interpolation_weights:
				cam_calib = interpolate_camera_callibration(setup.calibration[str(ids[i])], setup.calibration[str(ids[i+1])], w, setup.calibration)
				cam = build_camera_from_sFcallibration(**cam_calib, **setup.calibration, resolution=train_cam_resolution, focus_depth_clip=focus_depth_clip)
				scalarFlow_cameras.append(cam)
				
	return scalarFlow_cameras

#from view_interpolation_test import get_dense_optical_flow, lerp_image, lerp_image_2, lerp_vector, slerp_vector

def interpolate_camera_callibration(cal1, cal2, interpolation_weight, calib_base):
	calib = munch.Munch()
	t = interpolation_weight
	calib["forward"] = slerp_vector(cal1["forward"], cal2["forward"], t, normalized=True)
	calib["up"] = slerp_vector(cal1["up"], cal2["up"], t, normalized=True)
	calib["right"] = slerp_vector(cal1["right"], cal2["right"], t, normalized=True)
	if True: #focus_slerp is not None:
		p1 = np.subtract(cal1["position"], calib_base["focus"])
		p2 = np.subtract(cal2["position"], calib_base["focus"])
		calib["position"] = np.add(slerp_vector(p1, p2, t, normalized=False), calib_base["focus"])
	else:
		calib["position"] = lerp_vector(cal1["position"], cal2["position"], t)
	calib["fov_horizontal"] = lerp(cal1["fov_horizontal"], cal2["fov_horizontal"], t)
	calib["fov_vertical"] = lerp(cal1["fov_vertical"], cal2["fov_vertical"], t)
	
	return calib

def interpolate_image(target1, target2, interpolation_weights, use_backwards_flow=True):
	single = False
	if np.isscalar(interpolation_weights):
		single = True
		interpolation_weights = [interpolation_weights]
	
	is_tf = False
	if isinstance(target1, tf.Tensor):
		target1 = target1.numpy()
		is_tf = True
	if isinstance(target2, tf.Tensor):
		target2 = target2.numpy()
		is_tf = True
	
	flow = get_dense_optical_flow(target1, target2)
	if use_backwards_flow:
		flow_back = get_dense_optical_flow(target2, target1)
	targets = [lerp_image_2(target1, target2, w, flow, flow_back) if use_backwards_flow else lerp_image(target1, target2, w, flow) for w in interpolation_weights]
	
	if is_tf:
		targets = [tf.constant(_) if len(_.shape)==3 else tf.constant(_[...,np.newaxis]) for _ in targets]
	
	if single:
		return targets[0]
	else:
		return targets

def interpolate_images(images, interpolation_weights, use_backwards_flow=True):
	ret = []
	for img1, img2 in zip(images[:-1], images[1:]):
		ret.append(img1)
		ret.extend(interpolate_image(img1, img2, interpolation_weights, use_backwards_flow))
	ret.append(images[-1])
	return ret

def setup_target_cameras(base_cameras, frustum_resolution, crop_coordinates=None, crop_pad=0, normalize_resolution=False, jitter=False):
	cams = copy.deepcopy(base_cameras)
	for cam in cams:
		cam.transform.grid_size = frustum_resolution
	if crop_coordinates is not None:
		cams = [cam.copy_with_frustum_crop(crop_coordinates, crop_pad) for cam in cams]
		#normalize cams to same grid size to allow sampling batching
		if normalize_resolution:
			resolutions = [cam.transform.grid_size for cam in cams]
			resolution_hull = np.amax(resolutions, axis=0)
			for cam in cams:
				pass
	if jitter:
		raise NotImplementedError("TODO: fix too large uv jitter.")
		for cam in cams:
			cam.jitter = cam.depth_step
	return cams


def preestimate_volume(grid_transform, targets, cameras):
#--- Volume Estimation ---
	unprojections = []
	for i in range(len(cameras)):
		cam = cameras[i]
		#expand target to frustum volume (tile along z)
		tar = tf.reshape(targets[i], [1,1] + list(targets[i].shape))
		tar = tf.tile(tar, (1,cam.transform.grid_size[0],1,1,1))
		#sample target to shared volume
		unprojections.append(renderer.sample_camera(tar, grid_transform, cam, inverse=True))
	unprojection = tf.reduce_min(unprojections, axis=0)
	return unprojection

def generate_volume(grid_transform, targets, cameras, gen_model, setup, render_cameras=None, cut_alpha=True, random_rotation_pivot=None):
	# set a random rotation of the (shared) volume to make the generator rotationally invariant
	if random_rotation_pivot is not None:
		random_rotation_pivot.rotation_deg = np.random.uniform(0,360, 3).tolist()
	# get initial estimate from unprojected targets
	with profiler.sample('pre-estimate volume'):
		volume_estimate = preestimate_volume(sim_transform, targets, cameras)
	# let the generator refine the volume
	with profiler.sample('generate volume'):
		volume = volume_estimate
		for rec in range(setup.training.generator.recursion):
			volume = tf.clip_by_value(gen_model(volume)*setup.training.generator.out_scale, setup.training.generator.out_min, setup.training.generator.out_max)
	sim_transform.set_data(volume)
	# render images from refined volume for loss
	if render_cameras is not None:
		imgs = renderer.render_density_SDF_switch(sim_transform, lights, render_cameras, cut_alpha=cut_alpha)
		return volume, imgs
	return volume

def hull_AABB_OS(hull, hull_threshold = 0.1):
	'''min and max coord in object-space for each axis'''
	assert len(hull.get_shape().as_list())==3, "hull must be 3D DHW"
	def min_max_coords(flat_hull):
		coords = tf.cast(tf.where(tf.greater_equal(flat_hull, hull_threshold)), tf.float32)
		min_coord = tf.minimum(tf.reduce_min(coords), tf.cast(tf.shape(flat_hull)[0], tf.float32))
		max_coord = tf.maximum(tf.reduce_max(coords), 0.)
		return min_coord, max_coord 
	x_min, x_max = min_max_coords(tf.reduce_max(hull, axis=(-2,-3))) #W
	y_min, y_max = min_max_coords(tf.reduce_max(hull, axis=(-1,-3))) #H
	z_min, z_max = min_max_coords(tf.reduce_max(hull, axis=(-1,-2))) #D
	return ([x_min, y_min, z_min],[x_max, y_max, z_max])

def create_inflow(hull, hull_height, height):
	hull_threshold = 0.1
	assert len(hull.get_shape().as_list())==3, "hull must be 3D DHW"
	if tf.reduce_max(hull)<1.0:
		log.warning("Empty hull -> no inflow.")#raise ValueError('Empty hull')
		return None, None, None
	#find lowest part of hull, https://stackoverflow.com/questions/42184663/how-to-find-an-index-of-the-first-matching-element-in-tensorflow
	y_hull = tf.reduce_max(hull, axis=(-1,-3)) #H
	y_idx_min = tf.reduce_min(tf.where(tf.greater_equal(y_hull, hull_threshold)))
	y_idx_max = y_idx_min + hull_height
	
	#take max xz extend of hull from hull_min to hull_min+hull_height
	hull_slice = hull[:,y_idx_min:y_idx_max,:]
	flat_hull_slice = tf.reduce_max(hull_slice, axis=(-2), keepdims=True)
	x_hull_slice_idx = tf.where(tf.greater_equal(tf.reduce_max(flat_hull_slice, axis=(-2,-3)), hull_threshold))
	x_hull_slice_idx_min = tf.reduce_min(x_hull_slice_idx)
	x_hull_slice_idx_max = tf.reduce_max(x_hull_slice_idx) +1
	z_hull_slice_idx = tf.where(tf.greater_equal(tf.reduce_max(flat_hull_slice, axis=(-2,-1)), hull_threshold))
	z_hull_slice_idx_min = tf.reduce_min(z_hull_slice_idx)
	z_hull_slice_idx_max = tf.reduce_max(z_hull_slice_idx) +1
	
	flat_hull_slice = flat_hull_slice[z_hull_slice_idx_min:z_hull_slice_idx_max, :, x_hull_slice_idx_min:x_hull_slice_idx_max]
	
	if height=='MAX': #extend inflow all the way to the lower end of the grid
		height = y_idx_max.numpy().tolist()
	else:
		height = max(y_idx_max.numpy().tolist(), height)
	inflow_mask = tf.tile(flat_hull_slice, (1,height,1))
	inflow_shape = [(z_hull_slice_idx_max-z_hull_slice_idx_min).numpy().tolist(), height, (x_hull_slice_idx_max-x_hull_slice_idx_min).numpy().tolist()]
	inflow_offset = [z_hull_slice_idx_min.numpy().tolist(), (y_idx_max-height).numpy().tolist(), x_hull_slice_idx_min.numpy().tolist()]
	
	#return size and (corner)position
	return inflow_mask, inflow_shape, inflow_offset

# --- RENDERING ---

def render_cameras(grid_transform, cameras, lights, renderer, img_path, name_pre='img', bkg=None, \
		format='EXR', img_transfer=None, cut_alpha=True, img_normalize=False):
	
	imgs = renderer.render_density_SDF_switch(grid_transform, lights, cameras, background=bkg, cut_alpha=cut_alpha)
	imgs = tf.stack(imgs, axis=1)
	if not cut_alpha:
		imgs, imgs_d = tf.split(imgs, [3,1], axis=-1)
		imgs = tf.concat([imgs, tf.exp(-imgs_d)], axis=-1)
	if img_normalize:
		imgs /= tf.reduce_max(imgs, axis=(-3,-2,-1), keepdims=True)
	if img_transfer:
		imgs = tf_element_transfer_func(imgs, img_transfer)
	with renderer.profiler.sample("save image"):
		renderer.write_images_batch_views(imgs, name_pre+'_b{batch:04d}_cam{view:02d}', input_format="NVHWC", base_path=img_path, image_format=format)

def render_cycle(grid_transform, cameras, lights, renderer, img_path, name_pre='img', steps=12, steps_per_cycle=12, bkg=None, \
		format='EXR', img_transfer=None, img_stats=True, rotate_cameras=False, cut_alpha=True, img_normalize=False):
	
	r_step = 360.0/steps_per_cycle
	
	if renderer.can_render_fused and rotate_cameras:
		cams = []
		for camera in cameras:
			for i in range(steps):
				cam = copy.deepcopy(camera)
				cam.transform.parent.add_rotation_deg(y=-r_step*i)
				cams.append(cam)
		if bkg is not None:
			bkg = [_ for _ in bkg for i in range(steps)]
		render_cameras(grid_transform, cams, lights, renderer, img_path, name_pre, bkg, format, img_transfer, cut_alpha, img_normalize)
		return
	
	if rotate_cameras:
		cameras = copy.deepcopy(cameras)
	else:
		rot = grid_transform.rotation_deg
		grid_transform.rotation_deg = [0,0,0]
	with renderer.profiler.sample("render cycle "+name_pre):
		for i in range(steps):
			if not rotate_cameras:
				grid_transform.rotation_deg = [0,i*r_step,0]
			with renderer.profiler.sample("render step"):
				imgs = renderer.render_density_SDF_switch(grid_transform, lights, cameras, background=bkg, cut_alpha=cut_alpha)#, background=bkg
				imgs = tf.concat(imgs, axis=0)
				if not cut_alpha:
					imgs, imgs_d = tf.split(imgs, [3,1], axis=-1)
					imgs = tf.concat([imgs, tf.exp(-imgs_d)], axis=-1)
				if img_normalize:
					imgs /= tf.reduce_max(imgs, axis=(-3,-2,-1), keepdims=True)
				if img_transfer:
					if isinstance(img_transfer, tuple):
						imgs = tf_cmap_nearest(imgs, *img_transfer)
					else:
						imgs = tf_element_transfer_func(imgs, img_transfer)
			if args.console and img_stats:
				print_stats(imgs, 'frame '+str(i))
			with renderer.profiler.sample("save image"):
				renderer.write_images([imgs], [name_pre+'_cam{}_{:04d}'], base_path=img_path, use_batch_id=True, frame_id=i, format=format)
			if rotate_cameras:
				for camera in cameras:
					camera.transform.parent.add_rotation_deg(y=-r_step)#counter-rotate cam to match same object-view as object rotation
	if not rotate_cameras: grid_transform.rotation_deg = rot

def _slice_single_channel_color_transfer(data):
	assert isinstance(data, tf.Tensor)
	data_shape = shape_list(data)
	assert len(data_shape)>=1
	assert data_shape[-1]==1
	
	#return tf.concat([tf.maximum(data,0), tf.abs(data), tf.maximum(-data, 0)], axis=-1)
	# with narrow band
	return tf.concat([tf.maximum(data,0), tf.abs(data), tf.maximum(-data, 0), tf.cast(tf.less_equal(tf.abs(data), 1.6), tf.float32)], axis=-1)

def render_slices(data, slices, img_path, name_pre='slc', format='EXR', normalize=False, slice_indices=None):
	
	assert isinstance(slices, list)
	assert isinstance(data, tf.Tensor)
	assert len(shape_list(data))==5
	
	data_shape = GridShape.from_tensor(data)
	
	if normalize:
		data = data * (1.0/tf.reduce_max(tf.abs(data), axis=(-4,-3,-2,-1), keepdims=True))
	
	if data_shape.c==1:
		data = _slice_single_channel_color_transfer(data)
	
	if "X" in slices:
		data_slices = tf.unstack(data, axis=-2) #V-NHWC
		if slice_indices is not None:
			data_slices = [data_slices[_] for _ in slice_indices]
		renderer.write_images_batch_views(data_slices, name_pre + '_b{batch:04d}_camX{view:02d}', base_path=img_path, frame_idx=None, image_format=format)
	
	if "Y" in slices:
		data_slices = tf.unstack(data, axis=-3) #V-NHWC
		if slice_indices is not None:
			data_slices = [data_slices[_] for _ in slice_indices]
		renderer.write_images_batch_views(data_slices, name_pre + '_b{batch:04d}_camY{view:02d}', base_path=img_path, frame_idx=None, image_format=format)
	
	if "Z" in slices:
		data_slices = tf.unstack(data, axis=-4) #V-NHWC
		if slice_indices is not None:
			data_slices = [data_slices[_] for _ in slice_indices]
		renderer.write_images_batch_views(data_slices, name_pre + '_b{batch:04d}_camZ{view:02d}', base_path=img_path, frame_idx=None, image_format=format)

def render_gradients(gradients, grid_transform, cameras, renderer, path, image_mask, steps=12, steps_per_cycle=12, format='EXR', img_stats=True, name="gradients", log=None):
	tf_print_stats(gradients, "gradients " + name, log=log)
	os.makedirs(path, exist_ok=True)
	grad_shape = GridShape.from_tensor(gradients)
	if grad_shape.c==1: #density gradients
		grad_light = tf.concat([tf.maximum(gradients,0), tf.zeros_like(gradients), tf.maximum(-gradients, 0)], axis=-1)
	elif grad_shape.c==3: #velocity gradients
		grad_light = tf.abs(gradients)
	grid_transform = grid_transform.copy_new_data(tf.zeros_like(gradients))
	grid_transform.rotation_deg = [0,0,0]
	r_step = 360.0/steps_per_cycle
	with renderer.profiler.sample("render gradients cycle"):
		for i in range(steps):
			grid_transform.rotation_deg = [0,i*r_step,0]
			with renderer.profiler.sample("render step"):
				imgs = renderer.render_density(grid_transform, [grad_light], cameras, cut_alpha=True)
				imgs = tf.stack(imgs, axis=0) #VNHWC
			imgs /=tf.reduce_max(imgs)
			with renderer.profiler.sample("save image"):
				renderer.write_images_batch_views(imgs, image_mask, input_format="VNHWC", base_path=path, frame_idx=i, image_format=format)

def write_image_gradients(gradient_images, renderer, path, image_mask, image_neg_mask, format='EXR', img_stats=True):
	os.makedirs(path, exist_ok=True)
	if args.console and img_stats:
		print_stats(gradient_images, 'gradients frame '+str(i))
	imgs = gradient_images / tf.reduce_max(tf.abs(gradient_images))
	imgs_neg = tf.maximum(-imgs, 0)
	imgs = tf.maximum(imgs, 0)
	with renderer.profiler.sample("save image"):
		renderer.write_images([imgs, imgs_neg], [image_mask, image_neg_mask], base_path=path, use_batch_id=True, format=format)

'''
def advect_step(density, velocity):
	density = velocity.warp(density)
	velocity = velocity.copy_warped()
	return density, velocity
'''
def world_scale(shape, size=None, width=None, as_np=True):
	'''
	shape and size are z,y,x
	width corresponds to x and keeps aspect/cubic cells
	'''
	assert len(shape)==3
	if size is not None and width is not None:
		raise ValueError("Specify only one of size or width.")
	if size is not None:
		assert len(size)==3
		scale = np.asarray(size, dtype=np.float32)/np.asarray(shape, dtype=np.float32)
	elif width is not None:
		scale = np.asarray([width/shape[-1]]*3, dtype=np.float32)
	else:
		raise ValueError("Specify one of size or width.")
	if as_np:
		return scale
	else:
		return scale.tolist()

SDC = 256 #step density correction
# loss weight corrections after mean/sum reduction changes
tar_c = 1/(320 * 180)
grid_c = 1/(128*227*128)
train_dens = False

del SDC
if __name__=='__main__':
	from common_setups import RECONSTRUCT_SEQUENCE_SETUP_NEURAL_DENSITY
	if args.setup_file is not None:
		try:
			with open(args.setup_file, 'r') as setup_json:
				setup = json.load(setup_json)
		except:
			raise
	else:
		raise RuntimeError("No setup specified.")
	setup = update_dict_recursive(RECONSTRUCT_SEQUENCE_SETUP_NEURAL_DENSITY, setup, deepcopy=True, new_key='DISCARD_WARN') # new_key: DISCARD_WARN, ERROR
	
	with open(setup["rendering"]["target_cameras"]["calibration_file"], 'r') as calibration_file:
		cam_setup = json.load(calibration_file)
	setup['calibration']=cam_setup
	def flip_z(v):
		return v*np.asarray([1,1,-1])
	
	setup = munch.munchify(setup)
	cam_setup = setup.calibration
	
	hostname = socket.gethostname()
	now = datetime.datetime.now()
	now_str = now.strftime("%y%m%d-%H%M%S")
	try:
		paths = setup.paths
	except AttributeError:
		setup.paths = munch.Munch()
	prefix = 'seq'
	try:
		base_path = setup.paths.base
	except AttributeError:
		setup.paths.base = "./"
		base_path = setup.paths.base
	try:
		run_path = setup.paths.run
	except AttributeError:
		if args.fit:
			setup.paths.run = 'recon_{}_{}_{}'.format(prefix, now_str, setup.title)
		else:
			setup.paths.run = 'render_{}_{}_{}'.format(prefix, now_str, setup.title)
	if hasattr(setup.paths, 'group'):
		setup.paths.path = os.path.join(setup.paths.base, setup.paths.group, setup.paths.run)
	else:
		setup.paths.path = os.path.join(setup.paths.base, setup.paths.run)
	
	
	if os.path.isdir(setup.paths.path):
		setup.paths.path, _ = makeNextGenericPath(setup.paths.path)
	else:
		os.makedirs(setup.paths.path)
	
	setup.paths.log = os.path.join(setup.paths.path, 'log')
	os.makedirs(setup.paths.log)
	setup.paths.config = os.path.join(setup.paths.path, 'config')
	os.makedirs(setup.paths.config)
	setup.paths.data = setup.paths.path
	if setup.validation.warp_test:
		setup.paths.warp_test = os.path.join(setup.paths.path, 'warp_test')
		os.makedirs(setup.paths.warp_test)
	
	sys.stderr = StreamCapture(os.path.join(setup.paths.log, 'stderr.log'), sys.stderr)
	
	#setup logging
	log_format = '[%(asctime)s][%(name)s:%(levelname)s] %(message)s'
	log_formatter = logging.Formatter(log_format)
	root_logger = logging.getLogger()
	root_logger.setLevel(logging.INFO)
	logfile = logging.FileHandler(os.path.join(setup.paths.log, 'logfile.log'))
	logfile.setLevel(logging.INFO)
	logfile.setFormatter(log_formatter)
	root_logger.addHandler(logfile)
	errlog = logging.FileHandler(os.path.join(setup.paths.log, 'error.log'))
	errlog.setLevel(logging.WARNING)
	errlog.setFormatter(log_formatter)
	root_logger.addHandler(errlog)
	if args.debug:
		debuglog = logging.FileHandler(os.path.join(setup.paths.log, 'debug.log'))
		debuglog.setLevel(logging.DEBUG)
		debuglog.setFormatter(log_formatter)
		root_logger.addHandler(debuglog)
	if args.console:
		console = logging.StreamHandler(sys.stdout)
		console.setLevel(logging.INFO)
		console_format = logging.Formatter('[%(name)s:%(levelname)s] %(message)s')
		console.setFormatter(console_format)
		root_logger.addHandler(console)
	log = logging.getLogger('train')
	log.setLevel(logging.DEBUG)
	
	logging.captureWarnings(True)
	
	if args.debug:
		root_logger.setLevel(logging.DEBUG)
		log.info("Debug output active")
	
	
	
	with open(os.path.join(setup.paths.config, 'setup.json'), 'w') as config:
		json.dump(setup, config, sort_keys=True, indent=2)
	
	sources = [sys.argv[0], "common_setups.py"]
	sources.extend(os.path.join("lib", _) for _ in os.listdir("lib") if _.endswith(".py"))
	sources.extend(os.path.join("phitest/render", _) for _ in os.listdir("phitest/render") if _.endswith(".py"))
	archive_files(os.path.join(setup.paths.config,'sources.zip'), *sources)
	
	log.info('--- Running test: %s ---', setup.title)
	log.info('Test description: %s', setup.desc)
	log.info('Test directory: %s', setup.paths.path)
	log.info('Python: %s', sys.version)
	log.info('TensorFlow version: %s', tf.__version__)
	log.info('host: %s, device: %s, pid: %d', hostname, cudaID, os.getpid())
	
	
	max_memory = args.max_memory
	if max_memory<=0:
		log.info("No memory limit.")
		max_memory = -1
	if max_memory<1:
		max_memory = int(psutil.virtual_memory().total * max_memory)
	else:
		max_memory = int(min(max_memory * 1024 * 1024, psutil.virtual_memory().total))
	if max_memory>0:
		log.info("Memory limit: %d MiB (%f%%).", max_memory/(1024*1024), max_memory/psutil.virtual_memory().total * 100)
	
	if setup.data.rand_seed_global is not None:
		os.environ['PYTHONHASHSEED']=str(setup.data.rand_seed_global)
		random.seed(setup.data.rand_seed_global)
		np.random.seed(setup.data.rand_seed_global)
		tf.set_random_seed(setup.data.rand_seed_global)
	log.info("global random seed: %s", setup.data.rand_seed_global)
	
	profiler = DEFAULT_PROFILER #Profiler()
	renderer = Renderer(profiler,
		filter_mode=setup.rendering.filter_mode,
		boundary_mode= setup.rendering.boundary if setup.rendering.boundary is not None else ("CLAMP" if setup.data.SDF else "BORDER"),
		mipmapping=setup.rendering.mip.mode,
		num_mips=setup.rendering.mip.level,
		mip_bias=setup.rendering.mip.bias,
		blend_mode=setup.rendering.blend_mode,
		SDF_threshold=setup.rendering.SDF_threshold,
		sample_gradients=setup.rendering.sample_gradients,
		fast_gradient_mip_bias_add=0.0,
		luma = setup.rendering.luma,
		fused=setup.rendering.allow_fused_rendering,
		render_as_SDF=setup.data.SDF)
	vel_renderer = Renderer(profiler,
		filter_mode=setup.rendering.filter_mode,
		mipmapping=setup.rendering.mip.mode,
		num_mips=setup.rendering.mip.level,
		mip_bias=setup.rendering.mip.bias,
		blend_mode='ADDITIVE',
		SDF_threshold=setup.rendering.SDF_threshold,
		sample_gradients=setup.rendering.sample_gradients,
		luma = setup.rendering.luma,
		fused=setup.rendering.allow_fused_rendering)
	scale_renderer = Renderer(profiler,
		filter_mode='LINEAR',
		boundary_mode=setup.data.velocity.boundary.upper(),
		mipmapping=setup.rendering.mip.mode,
		num_mips=setup.rendering.mip.level,
		mip_bias=setup.rendering.mip.bias,
		blend_mode='ADDITIVE',
		SDF_threshold=setup.rendering.SDF_threshold,
		sample_gradients=setup.rendering.sample_gradients,
		render_as_SDF=setup.data.SDF)
	density_sampler = Renderer(profiler,
		filter_mode=setup.rendering.filter_mode,
		boundary_mode="CLAMP" if setup.data.SDF else "BORDER",
		mipmapping=setup.rendering.mip.mode,
		num_mips=setup.rendering.mip.level,
		mip_bias=setup.rendering.mip.bias,
		blend_mode='ADDITIVE',
		SDF_threshold=setup.rendering.SDF_threshold,
		sample_gradients=setup.rendering.sample_gradients,
		render_as_SDF=setup.data.SDF)
	warp_renderer = Renderer(profiler,
		filter_mode='LINEAR',
		boundary_mode=setup.data.velocity.boundary.upper(),
		mipmapping='NONE',
		blend_mode='ADDITIVE',
		SDF_threshold=setup.rendering.SDF_threshold,
		sample_gradients=False,
		render_as_SDF=setup.data.SDF)
	
	synth_target_renderer = Renderer(profiler,
		filter_mode=setup.rendering.synthetic_target.filter_mode,
		boundary_mode= setup.rendering.boundary if setup.rendering.boundary is not None else ("CLAMP" if setup.data.SDF else "BORDER"),
		mipmapping=setup.rendering.mip.mode,
		num_mips=setup.rendering.mip.level,
		mip_bias=setup.rendering.mip.bias,
		blend_mode=setup.rendering.synthetic_target.blend_mode,
		SDF_threshold=setup.rendering.SDF_threshold if setup.data.synth_shapes.active else 0.5, # 0.5 for ShapeNet SDF
		sample_gradients=setup.rendering.sample_gradients,
		fast_gradient_mip_bias_add=0.0,
		luma = setup.rendering.luma,
		fused=setup.rendering.allow_fused_rendering,
		render_as_SDF=setup.data.SDF)
		
	upscale_renderer = Renderer(profiler,
		filter_mode='LINEAR',
		boundary_mode=setup.data.velocity.boundary.upper(),
		mipmapping='NONE',
		blend_mode='ADDITIVE',
		SDF_threshold=setup.rendering.SDF_threshold,
		sample_gradients=setup.rendering.sample_gradients,
		luma = setup.rendering.luma)
	lifting_renderer = Renderer(profiler,
		filter_mode='LINEAR',
		boundary_mode="BORDER",
		mipmapping='NONE',
		blend_mode='ADDITIVE',
		SDF_threshold=setup.rendering.SDF_threshold,
		sample_gradients=setup.rendering.sample_gradients,
		luma = setup.rendering.luma)
	max_renderer = Renderer(profiler,
		filter_mode=setup.rendering.filter_mode,
		mipmapping=setup.rendering.mip.mode,
		num_mips=setup.rendering.mip.level,
		mip_bias=setup.rendering.mip.bias,
		blend_mode='MAX',
		SDF_threshold=setup.rendering.SDF_threshold,
		sample_gradients=setup.rendering.sample_gradients,
		fast_gradient_mip_bias_add=0.0,
		fused=setup.rendering.allow_fused_rendering)
	
	grad_renderer = vel_renderer
	
	pFmt = PartialFormatter()
	run_index = RunIndex(setup.data.run_dirs, ['recon_seq',])
	
	
	def load_model(path, *, num_levels, input_merge_weight=None, skip_merge_weight=None, output_residual_weight=None, **load_kwargs):
		model_path = run_index[path] 
		if model_path is None:
			model_path = path
		config_path = model_path + ".json"
		if os.path.isfile(config_path):
			with open(config_path, "r") as config_file:
				model_config = json.load(config_file)
			model_config = munch.munchify(model_config)
			
			if model_config._config.name=="RWDensityGeneratorNetwork":
				single_view = True 
				return RWDensityGeneratorNetwork.load(config_path, input_channels=1, w1=(1.0 if single_view else 0.5), w2=(0 if single_view else 0.5))
			
			if model_config._config.name=="RWVelocityGeneratorNetwork":
				return RWVelocityGeneratorNetwork.load(config_path, dens_channels=1, unp_channels=1, use_proxy=True)
			
			variable_level_model = GrowingUNet.config_is_level_variable(model_path + ".json")
			if variable_level_model:
				max_levels = num_levels
				log.info("Loading variable level model with %d levels.", max_levels)
				model = GrowingUNet.load(model_path+".json", num_levels=max_levels, **load_kwargs)
				model.set_active_level(max_levels-1)
			else:
				model = GrowingUNet.load(model_path+".json", **load_kwargs)
				log.info("Loaded fixed level model with %d levels for resolution %s", model.num_levels, sim_transform.grid_size)
			
			max_levels = model.num_levels
			
			if model.down_mode != "NONE" and input_merge_weight is not None:
				log.info("Setting input merge weights to %f.", input_merge_weight)
				for l in range(max(0, max_levels - model.max_input_levels), max_levels-1):
					model.set_input_merge_weight(input_merge_weight,l)
			if model.skip_merge_mode != "SUM" and skip_merge_weight is not None:
				log.info("Setting skip merge weights to %f.", skip_merge_weight)
				for l in range(1, max_levels):
					model.set_skip_merge_weight(skip_merge_weight,l)
			if model.output_mode=="RESIDUAL_WEIGHTED" and output_residual_weight is not None:
				log.info("Setting output residual weights to %f.", output_residual_weight)
				for l in range(1, max_levels):
					model.set_output_residual_weight(output_residual_weight, l)
		
		elif os.path.isfile(model_path + "_model.h5"):
			log.warning("No UNet spec found, loading plain keras model.")
			model = tf.keras.models.load_model(model_path + "_model.h5", custom_objects=custom_keras_objects)
		else:
			raise IOError("Can't load UNet or keras model from '%s'"%(model_path,))
		if False: #copy weights from broken serialization
			log.warning("Replacing model weights")
			model.load_weights(run_index[setup.training.density.decoder.model.replace("init", "model_nonorm")], by_name=True)
			save_NNmodel(model, 'density_decoder', setup.paths.data)
		
		return model

	def load_velocity(mask, fmt=None, boundary=None, scale_renderer=None, warp_renderer=None, device=None, var_name="velocity"):
		sf = RunIndex.parse_scalarFlow(mask)
		load_mask = run_index[mask]
		if load_mask is not None:
			load_mask = pFmt.format(load_mask, **fmt) if fmt is not None else load_mask
			log.info("load velocity grid from run %s", load_mask)
			vel_grid = VelocityGrid.from_file(load_mask, boundary=boundary, scale_renderer=scale_renderer, warp_renderer=warp_renderer, device=device, var_name=var_name)
		elif sf is not None: #mask.startswith('[SF:') and id_end!=-1: # explicit scalarFlow
			if sf["frame"] is None:
				raise ValueError("Missing frame from scalarFlow specifier.")
			fmt['sim'] += sf["sim"]
			fmt['frame'] += sf["frame"]
			run_path = os.path.normpath(os.path.join(setup.data.velocity.scalarFlow_reconstruction, sf["relpath"])).format(**fmt)
			log.info("load velocity grid from ScalarFlow %s", run_path)
			vel_grid = VelocityGrid.from_scalarFlow_file(run_path, boundary=boundary, scale_renderer=scale_renderer, warp_renderer=warp_renderer, device=device, var_name=var_name)
		else: # not in runs, assume centered vel
			load_mask = mask.format(**fmt) if fmt is not None else mask
			log.info("load centered velocity grid from file %s", load_mask)
			with np.load(load_mask) as np_data:
				vel_centered = reshape_array_format(np_data['data'], 'DHWC')
			vel_grid = VelocityGrid.from_centered(tf.constant(vel_centered, dtype=tf.float32), boundary=boundary, scale_renderer=scale_renderer, warp_renderer=warp_renderer, device=device, var_name=var_name)
		return vel_grid
	
	custom_keras_objects = {
		'MirrorPadND':MirrorPadND,
		'LayerNormalization':LayerNormalization,
		'AdaptiveNormalization':AdaptiveNormalization,
		'ConvLayerND':ConvLayerND,
		'ResBlock':ResBlock,
		'DenseConvBlock':DenseConvBlock,
		'WeightedSum':WeightedSum,
		'ScalarMul':ScalarMul,
	}
	
	color_channel = 1 if setup.rendering.monochrome else 3
	
	# automatic target scaling to grid resolution
	if True:
		def get_res_down(base_grid_size, base_factor=128 * 6):
			return int(base_factor / base_grid_size)
		res_down = get_res_down(setup.data.grid_size, base_factor=setup.data.res_down_factor)
		log.info("Setting automatic target/image resolution down-scale to %d for grid resolution %d", res_down, setup.data.grid_size)
		setup.training.train_res_down = res_down
		setup.training.discriminator.cam_res_down = res_down
		setup.data.discriminator.real_res_down = res_down
		
	
	train_res_down=setup.training.train_res_down
	
	density_size = [setup.data.grid_size]*3
	if setup.data.y_scale=="SF":
		setup.data.y_scale = cam_setup.scale_y
	density_size[1] = int(math.ceil(density_size[1] * setup.data.y_scale))
	volume_size = [cam_setup.marker_width, cam_setup.marker_width * cam_setup.scale_y, cam_setup.marker_width]
	sim_center = flip_z(cam_setup.volume_offset + np.asarray(volume_size)/2.0)
	def make_sim_transform():
		calibration_transform = Transform(translation=sim_center, scale=[cam_setup.marker_width]*3)
		randomization_transform = Transform(parent=calibration_transform)
		normalization_transform = GridTransform(density_size, center=True, normalize='MIN', parent=randomization_transform)
		return normalization_transform
	sim_transform = make_sim_transform()
	log.info("Base sim transform: %s", sim_transform)
	if True:
		sim_transform.parent.parent.translation[1]=sim_transform.grid_size_world().y/2 + (0.05 if setup.data.density.render_targets else 0.005)
		log.info("Set domain Y-center to %f for SF data inflow handling", sim_transform.parent.parent.translation[1])
	sF_transform = GridTransform([100,178,100], translation=flip_z(cam_setup.volume_offset + np.asarray([0,0,cam_setup.marker_width])), scale=[cam_setup.marker_width]*3, normalize='MIN')
	density_size = Int3(density_size[::-1])
	
	cam_resolution = copy.copy(setup.rendering.main_camera.base_resolution)
	aspect = cam_resolution[2]/cam_resolution[1]
	cam_resolution[1] *= setup.rendering.main_camera.resolution_scale
	cam_resolution[2] *= setup.rendering.main_camera.resolution_scale
	cam_dist = setup.rendering.main_camera.distance
	main_camera = Camera(GridTransform(cam_resolution, translation=[0,0,cam_dist], parent=Transform(translation=[0.33913451, 0.38741691, -0.25786148], rotation_deg=[0,0,0.])), nearFar=[setup.rendering.main_camera.near,setup.rendering.main_camera.far],
	fov = setup.rendering.main_camera.fov, aspect=aspect)
	cameras = [
		main_camera,
	]
	if not args.fit:
		tmp_cam = copy.deepcopy(main_camera)
		tmp_cam.transform.parent.rotation_deg = [0,90,0]
		cameras.append(tmp_cam)
		tmp_cam = copy.deepcopy(main_camera)
		tmp_cam.transform.parent.rotation_deg = [0,225,0]
		cameras.append(tmp_cam)
		del tmp_cam
	else:
		tmp_cam = copy.deepcopy(main_camera)
		tmp_cam.transform.parent.rotation_deg = [0,90,0]
		cameras.append(tmp_cam)
		del tmp_cam
	
	for _cam in cameras:
		renderer.check_LoD(sim_transform, _cam, check_inverse=True, name="main camera")
	
	scalarFlow_cam_ids = setup.rendering.target_cameras.camera_ids #[2,1,0,4,3] #[0,2,3] #
	
	if setup.data.density.target_cam_ids =="ALL":
		setup.data.density.target_cam_ids = list(range(len(scalarFlow_cam_ids)))
	
	view_interpolation_weights = [(_+1)/(setup.training.density.view_interpolation.steps+1) for _ in range(setup.training.density.view_interpolation.steps)]
	scalarFlow_cameras = build_scalarFlow_cameras(setup, scalarFlow_cam_ids, interpolation_weights=view_interpolation_weights)
	scalarFlow_cameras_base = [scalarFlow_cameras[_*(setup.training.density.view_interpolation.steps+1)] for _ in range(len(scalarFlow_cam_ids))]
	
	scalarFlow_cam_focus = flip_z(setup.calibration.focus)
	cam_resolution_scale = 1./setup.training.train_res_down 
	train_cam_resolution = copy.copy(setup.rendering.main_camera.base_resolution)
	train_cam_resolution[1] = int(train_cam_resolution[1]*cam_resolution_scale)
	train_cam_resolution[2] = int(train_cam_resolution[2]*cam_resolution_scale)
	log.info('scalarFlow train camera resolution: %s', str(train_cam_resolution))
	for sF_cam in scalarFlow_cameras:
		renderer.check_LoD(sim_transform, sF_cam, check_inverse=True, name="scalarFlow camera")
	cam_dh = 0.5 #depth half
	if setup.training.discriminator.active:
		disc_cam_resolution_scale = 1./setup.training.discriminator.cam_res_down 
		disc_cam_resolution = copy.copy(setup.rendering.main_camera.base_resolution)
		disc_cam_resolution[1] = int(disc_cam_resolution[1]*disc_cam_resolution_scale)
		disc_cam_resolution[2] = int(disc_cam_resolution[2]*disc_cam_resolution_scale)
		disc_cam_dist = 1.3
		disc_camera = Camera(GridTransform(disc_cam_resolution, translation=[0,0,disc_cam_dist], parent=Transform(translation=scalarFlow_cam_focus, rotation_deg=[0,0,0])), nearFar=[disc_cam_dist-cam_dh,disc_cam_dist+cam_dh], fov=cam_setup.fov_horizontal_average, aspect=aspect)
		disc_cameras = [copy.deepcopy(disc_camera) for _ in range(setup.training.discriminator.num_fake)]
		if setup.training.discriminator.fake_camera_jitter:
			raise NotImplementedError("TODO: fix too large uv jitter.")
			for cam in disc_cameras:
				cam.jitter = cam.depth_step
		log.info('discriminator camera resolution: %s, jitter: %s', str(disc_cam_resolution), setup.training.discriminator.fake_camera_jitter)
		renderer.check_LoD(sim_transform, disc_camera, check_inverse=True, name="discriminator camera")
	log.debug('Main camera transform: %s', main_camera.transform)
	
	lights = []
	
	if setup.rendering.lighting.initial_intensity=="CAMLIGHT":
		log.info("Using camlight lighting for rendering.")
		if not setup.data.SDF:
			raise ValueError("Camlight requires SDF mode.")
		lights = "CAMLIGHT"
	else:
		if setup.rendering.lighting.initial_intensity>=0:
			if setup.data.SDF:
				lights.append( # left red light
					PointLight(Transform(translation=[0,0,4], parent= \
						Transform(rotation_deg=[0,0,0], parent= \
							Transform(translation=scalarFlow_cam_focus, rotation_deg=[0,-120,0]))), \
					range_scale=0.5,
					intensity=setup.rendering.lighting.initial_intensity, \
					color=[1,0,0], \
					)
				)
				lights.append( # center top green light
					PointLight(Transform(translation=[0,0,4], parent= \
						Transform(rotation_deg=[0,0,0], parent= \
							Transform(translation=scalarFlow_cam_focus, rotation_deg=[0,0,0]))), \
					range_scale=0.5,
					intensity=setup.rendering.lighting.initial_intensity, \
					color=[0,1,0], \
					)
				)
				lights.append( # right blue light
					PointLight(Transform(translation=[0,0,4], parent= \
						Transform(rotation_deg=[0,0,0], parent= \
							Transform(translation=scalarFlow_cam_focus, rotation_deg=[0,120,0]))), \
					range_scale=0.5,
					intensity=setup.rendering.lighting.initial_intensity, \
					color=[0,0,1], \
					)
				)
			
			else:
				lights.append(
					SpotLight(Transform(translation=[0,0,2], parent=Transform(translation=scalarFlow_cam_focus, rotation_deg=[-40,0,0])), intensity=setup.rendering.lighting.initial_intensity, \
					cast_shadows=True, shadow_clip=[1.35, 2.65], range_scale=0.825, angle_deg=25., shadow_resolution=setup.rendering.lighting.shadow_resolution, cone_mask=False, \
					static=sim_transform if setup.rendering.allow_static_cameras else None)
				)
		
		if setup.rendering.lighting.ambient_intensity>=0:
			lights.append(Light(intensity=setup.rendering.lighting.ambient_intensity)) #some simple constant/ambient light as scattering approximation
		
		for light in lights:
			if isinstance(light, SpotLight) and light.cast_shadows:
				renderer.check_LoD(sim_transform, light.shadow_cam, check_inverse=True, name="shadow camera")
	
	
		shadow_lights = [
			SpotLight(Transform(translation=[0,0,2], parent=Transform(translation=scalarFlow_cam_focus, rotation_deg=[-40,0,0])), intensity=2.0, \
			cast_shadows=True, shadow_clip=[1.35, 2.65], range_scale=0.825, angle_deg=25., shadow_resolution=setup.rendering.lighting.shadow_resolution, cone_mask=False, \
			static=sim_transform if setup.rendering.allow_static_cameras else None),
			Light(intensity=0.08),
		]
		
		synth_target_lights = []
		if setup.rendering.synthetic_target.initial_intensity>=0:
			synth_target_lights.append(
				SpotLight(Transform(translation=[0,0,2], parent=Transform(translation=scalarFlow_cam_focus, rotation_deg=[-40,0,0])), intensity=setup.rendering.synthetic_target.initial_intensity, \
				cast_shadows=True, shadow_clip=[1.35, 2.65], range_scale=0.825, angle_deg=25., shadow_resolution=setup.rendering.lighting.shadow_resolution, cone_mask=False, \
				static=sim_transform if setup.rendering.allow_static_cameras else None)
			)
		
		if setup.rendering.synthetic_target.ambient_intensity>=0:
			synth_target_lights.append(Light(intensity=setup.rendering.synthetic_target.ambient_intensity))
	
	
	if not args.fit:
		# scene serialization
		scene = {
			"cameras":cameras,
			"sFcameras":scalarFlow_cameras,
			"lighting":lights,
			"objects":[sim_transform],
		}
		scene_file = os.path.join(setup.paths.config, "scene.json")
		#log.debug("Serializing scene to %s ...", scene_file)
		with open(scene_file, "w") as file:
			try:
				json.dump(scene, file, default=to_dict, sort_keys=True)#, indent=2)
			except:
				log.exception("Scene serialization failed.")
	
	main_render_ctx = RenderingContext([main_camera], lights, renderer, vel_renderer, setup.rendering.monochrome, render_SDF=setup.data.SDF)
	
	def get_validation_sequence_step(setup):
		return max(setup.data.sequence_step) if isinstance(setup.data.sequence_step, collections.abc.Iterable) else setup.data.sequence_step
	def get_frame_step(setup):
		return get_validation_sequence_step(setup) if (setup.data.randomize>0 and args.fit) else setup.data.step
	def get_vel_render_scale(setup):
		return 1.0 / float(get_frame_step(setup))*setup.rendering.velocity_scale
	def get_vel_scale_for_render(setup, transform):
		return transform.cell_size_world().value * get_vel_render_scale(setup)
	
	def render_sequence(sequence, vel_pad, cycle=True, cycle_steps=12, sF_cam=False, render_density=True, render_shadow=True, render_velocity=True, render_MS=True, slices=None):
		log.debug("Render images for sequence")
		clip_cams = False #True
		with profiler.sample('render sequence'):
			if cycle:
				cycle_cams = [main_camera]
			
			if render_shadow:
				shadow_cams = [copy.deepcopy(main_camera) for _ in range(1)]
				shadow_cams[0].transform.parent.add_rotation_deg(y=-60)
				shadow_cams_cycle = [main_camera]
				shadow_dens_scale = 4.
			
			if clip_cams:
				AABB_corners_WS = []
				AABB_corners_WS_cycle = []
				GRID_corners_WS_cycle = []
				for state in sequence:
					dens_transform = state.get_density_transform()
					dens_hull = state.density.hull 
					if dens_hull is None:
						continue
					corners_OS = hull_AABB_OS(tf.squeeze(dens_hull, (0,-1)))
					AABB_corners_WS += dens_transform.transform_AABB(*corners_OS, True)
					
					dens_shape = dens_transform.grid_shape
					grid_OS = (np.asarray([0,0,0], dtype=np.float32), np.asarray(dens_shape.xyz, dtype=np.float32))
					cycle_transform = dens_transform.copy_no_data()
					AABB_corners_WS_cycle.extend(cycle_transform.transform_AABB(*corners_OS, True))
					GRID_corners_WS_cycle.extend(cycle_transform.transform_AABB(*grid_OS, True))
					for i in range(1, cycle_steps):
						cycle_transform.add_rotation_deg(y=i * 360/cycle_steps) #rotation_deg[1] += i * 360/cycle_steps
						AABB_corners_WS_cycle.extend(cycle_transform.transform_AABB(*corners_OS, True))
						GRID_corners_WS_cycle.extend(cycle_transform.transform_AABB(*grid_OS, True))
					
					del dens_hull
				if AABB_corners_WS:
					seq_cams = [cam.copy_clipped_to_world_coords(AABB_corners_WS)[0] for cam in cameras]
				else:
					seq_cams = cameras
				
				if cycle and AABB_corners_WS_cycle:
					cycle_cams = [cam.copy_clipped_to_world_coords(AABB_corners_WS_cycle)[0] for cam in cycle_cams]
				
				if render_shadow and GRID_corners_WS_cycle:
					shadow_cams = [cam.copy_clipped_to_world_coords(GRID_corners_WS_cycle)[0] for cam in shadow_cams]
					if cycle:
						shadow_cams_cycle = [cam.copy_clipped_to_world_coords(GRID_corners_WS_cycle)[0] for cam in shadow_cams_cycle]
				
				split_cams = True
			else:
				seq_cams = cameras
				split_cams = False
			
			i=0
			if args.console:
				substeps = 0
				if render_density: 
					substeps += 3 if cycle else 1
					if sF_cam: substeps += 1
					if slices is not None: substeps += 1
				if render_velocity: substeps += 3 if cycle else 1
				cycle_pbar = ProgressBar(len(sequence)*substeps, name="Render Sequence: ")
				substep = 0
				def update_pbar(frame, desc):
					nonlocal substep
					cycle_pbar.update(i*substeps + substep, desc="Frame {:03d} ({:03d}/{:03d}): {:30}".format(frame, i+1, len(sequence), desc))
					substep +=1
			
			for state in sequence:
				if render_density:
					log.debug("Render density frame %d (%d)", state.frame, i)
					if args.console: update_pbar(state.frame, "Density, main cameras")
					bkg_render = None
					bkg_render_alpha = None
					if setup.rendering.background.type=='CAM':
						bkg_render = state.bkgs
					elif setup.rendering.background.type=='COLOR':
						bkg_render = [tf.constant(setup.rendering.background.color, dtype=tf.float32)]*len(seq_cams)
						bkg_render_alpha = [tf.constant(list(setup.rendering.background.color) + [0.], dtype=tf.float32)]*len(seq_cams)
					dens_transform = state.get_density_transform()
					val_imgs = renderer.render_density_SDF_switch(dens_transform, lights, seq_cams, background=bkg_render, split_cameras=split_cams)
					renderer.write_images_batch_views(val_imgs, 'seq_img_b{batch:04d}_cam{view:02d}_{idx:04d}', base_path=setup.paths.data, frame_idx=i, image_format='PNG')
					if state.has_density_proxy:
						dens_proxy_transform = dens_transform.copy_new_data(state.density_proxy.d)
						val_imgs = renderer.render_density_SDF_switch(dens_proxy_transform, lights, seq_cams, background=bkg_render, split_cameras=split_cams)
						renderer.write_images_batch_views(val_imgs, 'seq_imgP_b{batch:04d}_cam{view:02d}_{idx:04d}', base_path=setup.paths.data, frame_idx=i, image_format='PNG')
					if render_shadow:
						shadow_dens = dens_transform.copy_new_data(render_helper.with_border_planes(dens_transform.data *shadow_dens_scale, planes=["Z-","Y-"], density=100., width=3, offset=2))
						shadow_imgs = renderer.render_density_SDF_switch(shadow_dens, shadow_lights, shadow_cams, background=bkg_render, split_cameras=split_cams)
						renderer.write_images_batch_views(shadow_imgs, 'seq_sdw_b{batch:04d}_cam{view:02d}_{idx:04d}', base_path=setup.paths.data, frame_idx=i, image_format='PNG')
					if cycle or sF_cam:
						tmp_transform = state.get_density_transform()
						tmp_transform.set_data(tf.zeros_like(state.density.d))
						dens_grads = ("viridis", 0., 2.5)
					if cycle:
						if args.console: update_pbar(state.frame, "Density, cycle")
						render_cycle(dens_transform, cycle_cams, lights, renderer, state.data_path, steps=cycle_steps, steps_per_cycle=cycle_steps, name_pre='img', bkg=bkg_render_alpha, img_stats=False, rotate_cameras=True, cut_alpha=False, format='PNG')
						if state.has_density_proxy:
							render_cycle(dens_proxy_transform, cycle_cams, lights, renderer, state.data_path, steps=cycle_steps, steps_per_cycle=cycle_steps, name_pre='imgP', bkg=bkg_render_alpha, img_stats=False, rotate_cameras=True, cut_alpha=False, format='PNG')
						if render_shadow:
							if True:
								del shadow_dens
								shadow_dens = dens_transform.copy_new_data(dens_transform.data *shadow_dens_scale)
							render_cycle(shadow_dens, shadow_cams_cycle, shadow_lights, renderer, \
								state.data_path, steps=cycle_steps, steps_per_cycle=cycle_steps, name_pre='img_sdw', bkg=bkg_render, img_stats=False, rotate_cameras=True, format="PNG")
							del shadow_dens
						render_cycle(tmp_transform, [main_camera], [tf.concat([state.density.d]*3, axis=-1)], vel_renderer, state.data_path, steps=cycle_steps, steps_per_cycle=cycle_steps, name_pre='dens', img_transfer=dens_grads, img_stats=False, format="PNG")
						if args.console: update_pbar(state.frame, "Density inflow, cycle")
					if sF_cam and getattr(state, "target_cameras", None) is not None:
						if args.console: update_pbar(state.frame, "Density, target cameras")
						imgs = tf.stack(renderer.render_density_SDF_switch(dens_transform, lights, state.target_cameras, cut_alpha=False), axis=1)#, background=bkg
						imgs, imgs_d = tf.split(imgs, [3,1], axis=-1)
						renderer.write_images_batch_views(imgs, 'train_b{batch:04d}_cam{view:02d}', input_format="NVHWC", base_path=state.data_path, image_format='PNG')
						renderer.write_images_batch_views(tf.exp(-imgs_d), 'train_trans_b{batch:04d}_cam{view:02d}', input_format="NVHWC", base_path=state.data_path, image_format='PNG')
						render_cycle(tmp_transform, state.target_cameras, [tf.concat([state.density.d]*3, axis=-1)], vel_renderer, state.data_path, steps=1, steps_per_cycle=1, name_pre="train_dens", \
							img_transfer=dens_grads, img_stats=False, format="PNG")
						if getattr(state, 'targets_raw', None) is not None and (len(state.target_cameras)==shape_list(state.targets_raw)[1]):
							imgs_bkg = imgs +state.bkgs*tf.exp(-imgs_d)
							renderer.write_images_batch_views(imgs_bkg, 'train_bkg_b{batch:04d}_cam{view:02d}', input_format="NVHWC", base_path=state.data_path, image_format='PNG')
							renderer.write_images_batch_views(tf.abs(imgs_bkg - state.targets_raw), 'train_err_b{batch:04d}_cam{view:02d}', input_format="NVHWC", base_path=state.data_path, image_format='EXR')
							renderer.write_images_batch_views(tf.abs(imgs - state.targets_raw), 'train_err_bkg_b{batch:04d}_cam{view:02d}', input_format="NVHWC", base_path=state.data_path, image_format='PNG')
							renderer.write_images_batch_views(state.targets_raw, 'target_raw_b{batch:04d}_cam{view:02d}', input_format="NVHWC", base_path=state.data_path, image_format='PNG')
							if getattr(state, 'targets', None) is not None:
								renderer.write_images_batch_views(state.targets, 'target_b{batch:04d}_cam{view:02d}', input_format="NVHWC", base_path=state.data_path, image_format='PNG')
					
					if slices is not None:
						if args.console: update_pbar(state.frame, "Density, slices")
						slice_path = os.path.join(state.data_path, "slices")
						os.makedirs(slice_path, exist_ok=True)
						render_slices(dens_transform.data, slices, slice_path, name_pre="slc", format="EXR", normalize=False)
					
				if render_velocity and (state.next is not None):
					vel_transform = state.get_velocity_transform()
					vel_scale = vel_transform.cell_size_world().value
					log.debug("Render velocity frame %d (%d) with cell size %s", state.frame, i, vel_scale)
					if args.console: update_pbar(state.frame, "Velocity, main cameras")
					vel_centered = state.velocity.centered() * get_vel_scale_for_render(setup, vel_transform)
					val_imgs = vel_renderer.render_density(vel_transform, [tf.abs(vel_centered)], cameras, split_cameras=split_cams)
					vel_renderer.write_images_batch_views(val_imgs, 'seq_velA_{batch:04d}_cam{view:02d}_{idx:04d}', base_path=setup.paths.data, frame_idx=i, image_format='PNG')
					if cycle:
						if args.console: update_pbar(state.frame, "Velocity, cycle")
						render_cycle(vel_transform, [main_camera], [tf.abs(vel_centered)], vel_renderer, state.data_path, steps=cycle_steps, steps_per_cycle=cycle_steps, name_pre='velA', img_stats=False, format='PNG')
						
					#	vel_mag = state.velocity.magnitude()
					#	max_mag = tf.reduce_max(vel_mag)
					#	vel_mag_grads = [(0.0, tf.constant([0,0,0], dtype=tf.float32)),
					#		(1.0, tf.constant([1.0,1.0,1.0], dtype=tf.float32)),
					#		(1.0, tf.constant([0.5,0.5,1.0], dtype=tf.float32)),
					#		(max_mag, tf.constant([1.0,0.0,0.0], dtype=tf.float32))]
						#vel_mag = tf_element_transfer_func(vel_mag, vel_mag_grads)
						if args.console: update_pbar(state.frame, "Velocity magnitude, cycle")
						
						vel_div = state.velocity.divergence()
						vel_div = tf.concat((tf.maximum(vel_div, 0), tf.abs(vel_div), tf.maximum(-vel_div, 0)), axis=-1)
						render_cycle(vel_transform, [main_camera], [vel_div], vel_renderer, state.data_path, steps=cycle_steps, steps_per_cycle=cycle_steps, name_pre='velDiv', img_stats=False, img_normalize=True, format="PNG")
						del vel_div
					if render_MS and state.velocity.is_MS and state.velocity.has_MS_output:
						for vel_MS_scale in state.velocity.gen_current_MS_scales():
							vel_MS_shape = state.velocity.centered_shape_of_scale(vel_MS_scale)
							vel_MS_centered = state.velocity.centered_MS_residual(vel_MS_scale)
							vel_MS_transform = vel_transform.copy_new_data(tf.zeros([state.velocity._get_batch_size()] + vel_MS_shape + [1]))
							vel_MS_centered = vel_MS_centered * get_vel_scale_for_render(setup, vel_MS_transform)
							vel_MS_imgs = vel_renderer.render_density(vel_MS_transform, [tf.abs(vel_MS_centered)], cameras, split_cameras=split_cams)
							vel_renderer.write_images_batch_views(vel_MS_imgs, 'seq_velAr-%03d_{batch:04d}_cam{view:02d}_{idx:04d}'%(vel_MS_shape[0],), base_path=setup.paths.data, frame_idx=i, image_format='PNG')
					if False:
						vel_shape = GridShape.from_tensor(vel_centered)
						vel_slices_z = render_helper.image_volume_slices(vel_centered, axis=-4, abs_value=False)
						vel_renderer.write_images([tf.stack(vel_slices_z, axis=0)], ['vel_slice_{:04d}'], base_path=os.path.join(state.data_path, "vel_xy"), use_batch_id=True, format='EXR')
						vel_slices_x = render_helper.image_volume_slices(vel_centered, axis=-2, abs_value=False)
						vel_slices_x = (tf.transpose(_, (1,0,2)) for _ in vel_slices_x)
						vel_slices_x = list(tf.concat((tf.abs(_), tf.maximum(_,0), tf.maximum(-_,0)), axis=-2) for _ in vel_slices_x)
						vel_renderer.write_images([tf.stack(vel_slices_x, axis=0)], ['vel_slice_{:04d}'], base_path=os.path.join(state.data_path, "vel_zy"), use_batch_id=True, format='EXR')
				
				i +=1
				substep = 0
			if args.console:
				cycle_pbar.update(cycle_pbar._max_steps, "Done")
				cycle_pbar.close()
				#progress_bar(i*7,len(sequence)*7, "Frame {:03d} ({:03d}/{:03d}): {:30}".format(state.frame, i,len(sequence), "Done"), length=30)
	
	stop_training = False
	def handle_train_interrupt(sig, frame):
		global stop_training
		if stop_training:
			log.info('Training still stopping...')
		else:
			log.warning('Training interrupted, stopping...')
		stop_training = True
	
	data_device = setup.data.resource_device #'/cpu:0'
	resource_device = setup.training.resource_device #'/cpu:0'
	compute_device = '/gpu:0'
	log.debug("dataset device (volumes and images): %s", data_device)
	log.debug("resource device (volumes and images): %s", resource_device)
	log.debug("compute device: %s", compute_device)
	
	def wrap_resize_images(images, size):
		return tf_image_resize_mip(images, size, mip_bias=0.5, method=tf.image.ResizeMethod.BILINEAR)
		#return tf.image.resize_bilinear(images, size)
	
	# rebuild the interpolation of the active cameras here to match the target interpolation
	target_cameras = build_scalarFlow_cameras(setup, [scalarFlow_cam_ids[_] for _ in setup.data.density.target_cam_ids], interpolation_weights=view_interpolation_weights)
	target_cameras_base = [target_cameras[_] for _ in range(0, len(target_cameras), setup.training.density.view_interpolation.steps +1)]
	
	if setup.data.randomize>0:
		frames = list(range(setup.data.sequence_length))
	else:
		raise NotImplementedError("Direct reconstruction no longer supported.")
		frames = list(range(setup.data.start, setup.data.stop, setup.data.step))
	
	def _make_ImageSet(data, name, is_MS, resize_method):
		if is_MS:
			return ImageSetMS({scale:images for scale,images in enumerate(data)}, device=resource_device, var_name=name, resize_method=resize_method)
		else:
			return ImageSet(data, device=resource_device, var_name=name, resize_method=resize_method)
	
	def frame_loadTargets(setup, frame, sim_transform, target_dataset=None):
		'''
		load or render targets
		preprocessing: background subtraction, masking, view interpolation
		choose targets used for visual hull
		'''
		# setup targets and hulls at base_res
		if not setup.data.randomize>0:
			raise NotImplementedError("Direct reconstruction no longer supported.")
		
		loaded_data = target_dataset.frame_targets(frame, as_dict=True)
		if isinstance(loaded_data, dict):
			targets_raw=loaded_data["RAW"]
			targets=loaded_data["PREPROC"]
			bkgs=loaded_data["BKG"]
			masks=loaded_data.get("MASK", None)
			targets_for_hull=loaded_data["HULL"]
			density=loaded_data.get("DENSITY", None)
			velocity=loaded_data.get("VELOCITY", None)
			transform=loaded_data.get("TRANSFORM", None)
		else:
			#log.info("loaded %d types of data: %s", len(loaded_data), [shape_list(_) for _ in loaded_data])
			targets_raw, targets, bkgs, targets_for_hull = loaded_data[:4]
			if len(loaded_data)==5:
				if shape_list(loaded_data[4])[-1]==1:
					density = loaded_data[4]
				else:
					velocity = loaded_data[4]
			elif len(loaded_data)==6:
				density = loaded_data[4]
				velocity = loaded_data[5]
				#log.info("loaded density: %s", shape_list(density))
			transform = None
		
		aux = munch.Munch()
		
		is_MS = hasattr(target_dataset, "is_MS") and target_dataset.is_MS
		
		aux.targets_raw = _make_ImageSet(targets_raw, "targets_raw_f%06d"%frame, is_MS, "LINEAR")
		aux.targets = _make_ImageSet(targets, "targets_f%06d"%frame, is_MS, "LINEAR")
		aux.bkgs = _make_ImageSet(bkgs, "bkgs_f%06d"%frame, is_MS, "LINEAR")
		if masks is not None:
			aux.masks = _make_ImageSet(masks, "masks_f%06d"%frame, is_MS, "NEAREST")
		else:
			aux.masks = None
		aux.targets_for_hull = _make_ImageSet(targets_raw, "targets_for_hull_f%06d"%frame, is_MS, "LINEAR")
		aux.density = density[-1] if is_MS and density is not None else density
		aux.velocity = velocity[-1] if is_MS and velocity is not None else velocity
		aux.transform = transform
		
		return aux
	
	
	
	def get_random_target_mask(available_target_ids, *, target_weights=None, min_targets=1, max_targets=None, target_num_weights=None):
		
		assert min_targets>0
		if max_targets is None: max_targets=len(available_target_ids)
		else: assert max_targets<=len(available_target_ids)
		
		#select number of targets used, based on min/max targets and the chances for an amount of targets
		target_nums = list(range(min_targets, max_targets+1))
		if target_num_weights is not None:
			assert len(target_num_weights)==len(target_nums)
			target_num_chances_sum = sum(target_num_weights)
			target_num_weights = [_/target_num_chances_sum for _ in target_num_weights] #normalize for np.choice
		num_targets = np.random.choice(target_nums, p=target_num_weights)
		
		#select n out of m targets based on their weights
		if target_weights is not None:
			assert len(available_target_ids)==len(target_weights)
			target_weight_sum = sum(target_weights)
			target_weights = [_/target_weight_sum for _ in target_weights] #normalize for np.choice
		target_ids = np.random.choice(available_target_ids, num_targets, replace=False, p=target_weights)
		
		return target_ids
	
	def state_randomize(state, randomize_input_views=False, randomize_target_views=False, randomize_transform=False, \
			transform_allow_scale=True, transform_allow_scale_non_uniform=False, transform_allow_mirror=True, transform_allow_rotY=True, transform_allow_translate=False, disable_transform_reset=False):
		sequence_randomize([state], randomize_input_views, randomize_target_views, randomize_transform, \
			transform_allow_scale, transform_allow_scale_non_uniform, transform_allow_mirror, transform_allow_rotY, transform_allow_translate, disable_transform_reset=disable_transform_reset)
	
	def sequence_randomize(sequence, randomize_input_views=False, randomize_target_views=False, randomize_transform=False, \
			transform_allow_scale=True, transform_allow_scale_non_uniform=False, transform_allow_mirror=True, transform_allow_rotY=True, transform_allow_translate=False, disable_transform_reset=False):
		#raise NotImplementedError
		
		if isinstance(randomize_target_views, (list, tuple)):
			assert all(_ < len(setup.data.density.target_cam_ids) for _ in randomize_target_views)
			target_mask = randomize_target_views
		elif randomize_target_views==True:
			target_mask = get_random_target_mask(list(range(len(setup.data.density.target_cam_ids))), target_weights=setup.training.randomization.target_weights, \
				min_targets=setup.training.randomization.min_targets, max_targets=setup.training.randomization.max_targets, target_num_weights=setup.training.randomization.num_targets_weights)
			log.debug("Randomized target mask: %s.", target_mask)
		else:
			target_mask = None
		for state in sequence:
			state.target_mask = target_mask
		
		if randomize_input_views=="TARGET":
			input_view_mask = target_mask
		elif isinstance(randomize_input_views, (list, tuple)):
			assert all(_ < len(setup.data.density.target_cam_ids) for _ in randomize_input_views)
			warnings.warn("Using fixed input views {}".format(randomize_input_views))
			input_view_mask = randomize_input_views
		elif randomize_input_views==True:
			input_view_mask = get_random_target_mask(list(range(len(setup.data.density.target_cam_ids))), target_weights=setup.training.randomization.input_weights, \
				min_targets=setup.training.randomization.min_inputs, max_targets=setup.training.randomization.max_inputs, target_num_weights=setup.training.randomization.num_inputs_weights)
			log.debug("Randomized input mask: %s.", input_view_mask)
		else:
			input_view_mask = None #setup.data.density.input_cam_ids ?
		for state in sequence:
			state.input_view_mask = input_view_mask
		
		if randomize_transform:
			raise NotImplementedError("Use randomized transform provided by dataloader.")
			if transform_allow_scale:
				scale_min, scale_max = 0.94, 1.06
				if transform_allow_scale_non_uniform:
					scale = np.random.uniform(scale_min, scale_max, 3).tolist()
				else:
					scale = [np.random.uniform(scale_min, scale_max)]*3
			else:
				scale = [1,1,1]
			
			if transform_allow_mirror:
				for dim in range(len(scale)):
					flip = np.random.random()<0.5
					if flip: scale[dim] *= -1
			
			if transform_allow_rotY:
				#rotation = Transform.get_random_rotation()
				rotation = [0,np.random.uniform(0.,360.), 0]
			else:
				rotation = [0,0,0]
			
			if transform_allow_translate:
				translation_min, translation_max = state.transform.cell_size_world()*-6, state.transform.cell_size_world()*6
				translation = np.random.uniform(translation_min, translation_max).tolist()
			else:
				translation = [0,0,0]
			
			log.debug("Randomized grid transform: s=%s, r=%s, t=%s", scale, rotation, translation)
		
		if not disable_transform_reset or randomize_transform:
			for state in sequence:
				state.transform.parent.set_scale(scale if randomize_transform else None)
				state.transform.parent.set_rotation_angle(rotation if randomize_transform else None)
				state.transform.parent.set_translation(translation if randomize_transform else None)
	
	def state_set_targets(state, aux_sequence, set_size=True):
		state.clear_cache()
		state.base_targets_raw = aux_sequence[state.frame].targets_raw
		state.base_targets = aux_sequence[state.frame].targets
		state.base_bkgs = aux_sequence[state.frame].bkgs
		state.base_masks = aux_sequence[state.frame].masks
		
		transform = aux_sequence[state.frame].get("transform", None) or sim_transform
		state.transform.parent.set_scale(transform.parent.scale)
		state.transform.parent.set_rotation_angle(transform.parent.rotation_deg)
		state.transform.parent.set_translation(transform.parent.translation)
		
		set_density = (("density" in aux_sequence[state.frame]) and (aux_sequence[state.frame].density is not None) and (state.has_density) and (type(state.density)==DensityGrid)) #not isinstance(state.density, NeuralDensityGrid):
		set_density_target = (("density" in aux_sequence[state.frame]) and (aux_sequence[state.frame].density is not None) and (state.has_density_target) and (type(state.density_target)==DensityGrid))
		set_velocity_target = (("velocity" in aux_sequence[state.frame]) and (aux_sequence[state.frame].velocity is not None) and (state.has_velocity_target) and (type(state.velocity_target)==VelocityGrid))
		
		if set_size:
			try:
				res = curr_cam_res[1:]
			except NameError:
				log.error("Failed to get current camera resolution, using base resolution")
			else:
				state.base_targets_raw.resize(res)
				state.base_targets.resize(res)
				state.base_bkgs.resize(res)
				if state.has_masks:
					state.base_masks.resize(res)
			
			try:
				res_MS = grow_handler.get_image_MS_scale_shapes()
				state.base_targets_raw.create_MS_stack(res_MS)
				state.base_targets.create_MS_stack(res_MS)
				state.base_bkgs.create_MS_stack(res_MS)
				if state.has_masks:
					state.base_masks.create_MS_stack(res_MS)
			except NameError:
				log.exception("NameError when creating target MS stacks:")
				pass
			
			if set_density:
				#log.info("Frame %d: assinged new scaled density %s", state.frame, shape_list(aux_sequence[state.frame].density))
				state.density.assign_scaled(aux_sequence[state.frame].density)
			if set_density_target:
				#log.info("Frame %d: assinged new scaled density target %s", state.frame, shape_list(aux_sequence[state.frame].density))
				state.density_target.assign_scaled(aux_sequence[state.frame].density)
			if set_velocity_target: #TODO
				#log.info("Frame %d: assinged new scaled velocity target %s", state.frame, shape_list(aux_sequence[state.frame].density))
				state.velocity_target.assign_staggered_combined_scaled(aux_sequence[state.frame].velocity)
		else:
			if set_density:
				#log.info("Frame %d: assinged new density %s", state.frame, shape_list(aux_sequence[state.frame].density))
				state.density.assign(aux_sequence[state.frame].density)
			if set_density_target:
				#log.info("Frame %d: assinged new density target %s", state.frame, shape_list(aux_sequence[state.frame].density))
				state.density_target.assign(aux_sequence[state.frame].density)
			if set_velocity_target:
				#log.info("Frame %d: assinged new velocity target %s", state.frame, shape_list(aux_sequence[state.frame].density))
				state.velocity_target.assign_staggered_combined(aux_sequence[state.frame].velocity)
		
		#log.info("state transform for frame %d: %s\n\tfrom %s", state.frame, state.transform, transform)
	
	def sequence_set_targets(sequence, aux_sequence, set_size=True):
		for state in sequence:
			state_set_targets(state, aux_sequence, set_size=set_size)
	
	if args.fit:
		log.info("Reconstructing sequence for frames %s", frames)
		setup.paths.data = setup.paths.path
		os.makedirs(setup.paths.data, exist_ok=True)
		try:
			faultlog = open(os.path.join(setup.paths.log, 'fault.log'), 'a')
			faulthandler.enable(file=faultlog)
			summary = tf.contrib.summary
			summary_writer = summary.create_file_writer(setup.paths.log)
			
			
			if True:
				plot_schedule(setup.training.density.learning_rate, setup.training.iterations, os.path.join(setup.paths.config, 'dens_lr.png'), 'Density LR')
				plot_schedule(setup.training.velocity.learning_rate, setup.training.iterations, os.path.join(setup.paths.config, 'vel_lr.png'), 'Velocity LR')
				
				plot_schedule(setup.training.density.warp_loss, setup.training.iterations, os.path.join(setup.paths.config, 'dens_warp_loss.png'), 'Density Warp Loss Scale')
				plot_schedule(setup.training.velocity.density_warp_loss, setup.training.iterations, os.path.join(setup.paths.config, 'vel_dens_warp_loss.png'), 'Velocity Density Warp Loss Scale')
				plot_schedule(setup.training.velocity.velocity_warp_loss, setup.training.iterations, os.path.join(setup.paths.config, 'vel_vel_warp_loss.png'), 'Velocity Velocity Warp Loss Scale')
				plot_schedule(setup.training.velocity.divergence_loss, setup.training.iterations, os.path.join(setup.paths.config, 'vel_div_loss.png'), 'Velocity Divergence Loss Scale')
				
				labels = ['Dens warp', 'Vel dens-warp', 'Vel vel-warp']#, 'Vel div']
				schedules = [setup.training.density.warp_loss, setup.training.velocity.density_warp_loss, setup.training.velocity.velocity_warp_loss]#, setup.training.velocity.divergence_loss]
				plot_schedules(schedules, setup.training.iterations, os.path.join(setup.paths.config, 'warp_loss_cmp.png'), labels=labels, title='Warp Loss Comparison')
				
				if setup.training.discriminator.active:
					plot_schedule(setup.training.discriminator.learning_rate, setup.training.iterations, os.path.join(setup.paths.config, 'disc_lr.png'), 'Discriminator LR')
			
			frustum_half = 0.75
			dist = 4.
			log.debug("Setup validation")
			val_cameras = [
				Camera(GridTransform(cam_resolution, translation=[0,0,0.8], parent=Transform(rotation_deg=[-30,0,0], parent=Transform(translation=scalarFlow_cam_focus, rotation_deg=[0,-85,0]))), nearFar=[0.3,1.3], fov=40, aspect=aspect, static=sim_transform if setup.rendering.allow_static_cameras else None),
			]
			
			
			if len(frames)<1:
				log.error("Not enough frames for sequence reconstruction: %s", frames)
				sys.exit(1)
			if len(frames)==1:
				log.warning("Single frame reconstruction can not provide meaningfull velocity.")
			
			base_shape = density_size.as_shape #copy.copy(density_size)
			sim_transform.grid_size = base_shape #curr_dens_shape
		#	print(density_size, base_shape)
			
			def get_max_recursive_MS_grow_levels(decoder_config, shape_cast_fn=round):
				if decoder_config.recursive_MS_levels=="VARIABLE":
					#return GrowingUNet.get_max_levels(density_size, scale_factor=setup.training.velocity.decoder.recursive_MS_scale_factor, min_size=setup.training.velocity.decoder.min_grid_res)
					i = 0
					while (min(shape_cast_fn(_/(decoder_config.recursive_MS_scale_factor**i)) for _ in sim_transform.grid_size)>=decoder_config.min_grid_res):
						i +=1
					return i
				else:
					return decoder_config.recursive_MS_levels
			
			log.info("Set up GrowHandler ...")
			grow_handler = GrowHandler(base_shape=list(base_shape), base_cam_shape=train_cam_resolution, max_dens_level=get_max_recursive_MS_grow_levels(setup.training.density.decoder)-1, max_vel_level=get_max_recursive_MS_grow_levels(setup.training.velocity.decoder)-1, setup=setup) # , base_cam_factor=1
			log.warning("Image grow factor set to 1!")
			log.info("GrowHandler test:\n%s", str(grow_handler))
			
					
			log.info("--- Pre-setup ---")
			
					
			# TomoFluid: interpolated targets have less weight, based on the angle to a 'real' target
			def get_target_weights(cameras, real_cameras, focus, mode="COS", **kwargs):
				focus = Float3(focus)
				# get angles from cameras to next base/real camera
				angles = []
				for camera in cameras:
					dir_from_focus = (Float3(camera.transform.transform(Float4(0,0,0,1)))-focus).normalized
					angle = np.pi
					for cam in real_cameras:
						dff = (Float3(cam.transform.transform(Float4(0,0,0,1)))-focus).normalized
						angle = np.minimum(np.arccos(np.dot(dir_from_focus, dff)), angle)
					angles.append(angle)
				
				if mode=="COS":
					weights = [np.cos(_*2)*0.5+0.5 for _ in angles]
			#	elif mode=="POW":
			#		min_weight = kwargs.get("min_weight", 1e-3)
			#		base = min_weight ** (-2./np.pi)
			#		weights = [base**(-_) for _ in angles]
				elif mode=="EXP":
					min_weight = kwargs.get("min_weight", 1e-4)
					scale = np.log(min_weight) * (-2./np.pi)
					weights = [np.exp(-_*scale) for _ in angles]
				else:
					raise ValueError("Unknown target weight mode %s"%mode)
				return weights
			view_interpolation_target_weights = get_target_weights(target_cameras, target_cameras_base, cam_setup.focus, mode="EXP") if setup.training.density.view_interpolation.steps>0 else None
			
			if isinstance(setup.data.batch_size, (list,tuple)):
				assert len(setup.data.batch_size)==2
				batch_size, batch_group_size = setup.data.batch_size
				assert not batch_group_size==0
			else:
				batch_size = setup.data.batch_size
				batch_group_size = 1
			assert batch_size>0
			log.info("Using %s batch group size %d, batch size %d.", "adaptive" if batch_group_size>0 else "fixed", abs(batch_group_size), batch_size)
			
			train_disc = setup.training.discriminator.active and (setup.training.discriminator.train or setup.training.discriminator.pre_opt.train or setup.training.discriminator.pre_opt.first.train)
			make_disc_dataset = train_disc or setup.training.discriminator.loss_type not in ["SGAN"]
			
			load_density_dataset = False
			load_velocity_dataset = False
			SF_data_cache = None
			if setup.data.randomize>0:
				load_density_dataset = (not setup.training.density.decoder.active) or (setup.training.density.volume_target_loss!=0.0) #True
				load_velocity_dataset = (setup.training.velocity.volume_target_loss!=0.0) #False
				
				if setup.data.synth_shapes.active in [False, "BOTH"] and not setup.data.SDF:
					log.info("Using new SF data loader.")
					target_dataset, SF_data_cache = get_targets_dataset_v2(sim_indices=setup.data.sims, frame_start=setup.data.start, frame_stop=setup.data.stop, frame_strides=setup.data.step, \
						raw=True, preproc=True, bkg=True, hull=True, batch_size=batch_size, \
						sequence_step=setup.data.sequence_step, sequence_length=setup.data.sequence_length, \
						view_indices=[scalarFlow_cam_ids[_] for _ in setup.data.density.target_cam_ids], num_views=len(setup.data.density.target_cam_ids), \
						path_raw=setup.data.density.target, path_preproc=None, SF_frame_offset=setup.data.scalarFlow_frame_offset, \
						down_scale=setup.training.train_res_down, channels=color_channel, threshold=setup.data.density.hull_threshold, shuffle_frames=True,\
						density=load_density_dataset, path_density=setup.data.density.initial_value, \
						density_t_src=sF_transform, density_t_dst=sim_transform, density_sampler=scale_renderer, \
						velocity=load_velocity_dataset, path_velocity=setup.data.velocity.initial_value, velocity_t_src=sF_transform, \
						randomize_transform=setup.training.randomization.transform, \
						cache_device=data_device, data_cache=SF_data_cache, \
						render_targets=setup.data.density.render_targets, density_renderer=synth_target_renderer, cameras=target_cameras, lights=lights, \
						density_type=setup.data.density.density_type, velocity_type=setup.data.density.density_type)
					
					validation_dataset, _ = get_targets_dataset_v2(sim_indices=[setup.validation.simulation], frame_start=setup.validation.start, \
						frame_stop=setup.validation.stop, frame_strides=setup.validation.step, \
						raw=True, preproc=True, bkg=True, hull=True, batch_size=setup.validation.batch_size, \
						sequence_step=get_validation_sequence_step(setup), sequence_length=setup.data.sequence_length, \
						view_indices=[scalarFlow_cam_ids[_] for _ in setup.data.density.target_cam_ids], num_views=len(setup.data.density.target_cam_ids), \
						path_raw=setup.data.density.target, path_preproc=None, SF_frame_offset=setup.data.scalarFlow_frame_offset, \
						down_scale=setup.training.train_res_down, channels=color_channel, threshold=setup.data.density.hull_threshold, shuffle_frames=True,\
						density=load_density_dataset, path_density=setup.data.density.initial_value, density_t_src=sF_transform, density_t_dst=sim_transform, density_sampler=scale_renderer, \
						velocity=load_velocity_dataset, path_velocity=setup.data.velocity.initial_value, velocity_t_src=sF_transform, \
						randomize_transform=False, \
						cache_device=data_device, data_cache=SF_data_cache, \
						render_targets=setup.data.density.render_targets, density_renderer=synth_target_renderer, cameras=target_cameras, lights=lights, \
						density_type=setup.data.density.density_type, velocity_type=setup.data.density.density_type)
					
					if make_disc_dataset:
						log.info("Using smoke dataset for disc.")
						disc_dataset, _ = get_targets_dataset_v2(sim_indices=list(range(*setup.data.discriminator.simulations)), frame_start=setup.data.discriminator.frames[0], \
							frame_stop=setup.data.discriminator.frames[1], frame_strides=setup.data.discriminator.frames[2], \
							raw=False, preproc=True, bkg=False, hull=setup.training.discriminator.conditional_hull, batch_size=setup.training.discriminator.num_real, \
							sequence_step=setup.data.sequence_step if setup.training.discriminator.temporal_input.active else 1, \
							sequence_length=3 if setup.training.discriminator.temporal_input.active else 1, \
							view_indices=[scalarFlow_cam_ids[_] for _ in setup.data.density.target_cam_ids], num_views=len(setup.data.density.target_cam_ids), \
							path_raw=setup.data.density.target, path_preproc=None, SF_frame_offset=setup.data.scalarFlow_frame_offset, \
							down_scale=setup.data.discriminator.real_res_down, channels=color_channel, threshold=setup.data.density.hull_threshold, shuffle_frames=True,\
							density=False, path_density=setup.data.discriminator.initial_value, density_t_src=sF_transform, density_t_dst=sim_transform, density_sampler=scale_renderer, \
							velocity=False, path_velocity=None, velocity_t_src=sF_transform, \
							randomize_transform=False, \
							cache_device=data_device, data_cache=SF_data_cache if setup.data.discriminator.render_targets==setup.data.density.render_targets else None, \
							render_targets=setup.data.discriminator.render_targets, density_renderer=synth_target_renderer, cameras=target_cameras, lights=lights, \
							density_type=setup.data.discriminator.density_type, velocity_type=setup.data.discriminator.density_type)
						
					if setup.data.synth_shapes.active=="BOTH":
						SF_target_dataset = target_dataset
						del target_dataset
						if make_disc_dataset:
							SF_disc_dataset = disc_dataset
							del disc_dataset
					
				if setup.data.synth_shapes.active in [True,"BOTH"] or setup.data.SDF:
					def get_max_cell_size():
						min_grid_res = [setup.training.velocity.decoder.min_grid_res, int(math.ceil(setup.training.velocity.decoder.min_grid_res * setup.data.y_scale)), setup.training.velocity.decoder.min_grid_res]
						tmp_T = sim_transform.copy_no_data()
						tmp_T.grid_size = min_grid_res
						max_cell_size = tmp_T.cell_size_world()
						return min(max_cell_size)
					synth_max_cell_size = get_max_cell_size() #cell size at coarsest resolution, as defined by min_grid_res
					synth_max_translation = synth_max_cell_size * setup.data.synth_shapes.max_translation
					is_dens_preopt = True #setup.training.density.pre_optimization # larger objects/shapes
					log.info("Using synthetic shape dataset. Max cell size: %f", synth_max_cell_size)
					if is_dens_preopt: log.info("Using larger shapes.")
					def _get_dataset_transform():
						if setup.data.MS_volume:
							transforms = []
							for interval in grow_handler.main_dens_schedule.intervals:
								t = copy.deepcopy(sim_transform)
								t.grid_size = interval.value
								transforms.append(t)
							return transforms
						else:
							return sim_transform
					def _get_dataset_cameras():
						if setup.data.MS_images:
							scales_cameras = []
							for interval in grow_handler.main_cam_schedule.intervals:
								cameras = copy.deepcopy(target_cameras)
								for camera in cameras:
									camera.transform.grid_size = interval.value
								scales_cameras.append(cameras)
							return scales_cameras
						else:
							return target_cameras
					log.info("Using new Synth data loader/generator.")
					#sample_overrides = {'density_scale': 0.2, "shape_type":setup.data.synth_shapes.shape_types, "initial_translation":[0,0,0]} #, "base_scale":[0.35]*3,"initial_rotation_rotvec":[0,0,0], "rotvec":[0,0,0], 'density_scale': 0.2 
					sample_overrides = {"shape_type":setup.data.synth_shapes.shape_types,} #, "base_scale":[0.35]*3,"initial_rotation_rotvec":[0,0,0], "rotvec":[0,0,0], 'density_scale': 0.2 
					if setup.data.synth_shapes.init_center:
						log.info("Centered initial position .")
						sample_overrides["initial_translation"]=[0,0,0]
					
					SDF_dataset_cache = None
					SDF_frames = []
					if setup.data.synth_shapes.active==False:
						SDF_dataset_cache = SDFDatasetCache(path_mask=setup.data.density.initial_value, device=data_device)
						SDF_frames = list(range(setup.data.start, setup.data.stop, setup.data.step))
					
					target_dataset = get_synthTargets_dataset_v2(batch_size=batch_size, base_grid_transform=_get_dataset_transform(), sequence_length=setup.data.sequence_length, \
						view_indices=[scalarFlow_cam_ids[_] for _ in setup.data.density.target_cam_ids], num_views=len(setup.data.density.target_cam_ids), \
						cameras=_get_dataset_cameras(), lights=lights, device=resource_device, \
						density_range=[0.2,0.4] if is_dens_preopt else [0.05,0.14], \
						inner_range=[0.0,0.4], \
						scale_range=[0.25,0.35] if setup.data.SDF else [synth_max_cell_size,0.5], #if is_dens_preopt else [synth_max_cell_size,synth_max_cell_size*2.0],
						translation_range=[-synth_max_translation,synth_max_translation], \
						rotation_range=[10,30], #[0,20],
						raw=True, preproc=True, bkg=True, hull=True, mask=setup.data.SDF, channels=1, SDF=setup.data.SDF, \
						density=load_density_dataset, velocity=load_velocity_dataset, advect_density=load_velocity_dataset, \
						density_sampler=density_sampler, density_renderer=synth_target_renderer, randomize_transform=setup.training.randomization.transform, \
						seed=np.random.randint(np.iinfo(np.int32).max), sample_overrides=sample_overrides, \
						data_cache=SDF_dataset_cache, generate_shape=setup.data.synth_shapes.active, \
						generate_sequence=setup.data.synth_shapes.active or len(SDF_frames)<2 , \
						sims=setup.data.sims, frames=SDF_frames, steps=setup.data.sequence_step)
					
					if not setup.data.synth_shapes.active=="BOTH" or setup.data.SDF:
						SDF_val_frames = list(range(setup.validation.start, setup.validation.stop, setup.validation.step))
						val_sample_overrides = copy.copy(sample_overrides)
						val_sample_overrides.update({"shape_type":setup.validation.synth_data_shape_types})
						validation_dataset = get_synthTargets_dataset_v2(batch_size=setup.validation.batch_size, base_grid_transform=_get_dataset_transform(), sequence_length=setup.data.sequence_length, \
							view_indices=[scalarFlow_cam_ids[_] for _ in setup.data.density.target_cam_ids], num_views=len(setup.data.density.target_cam_ids), \
							cameras=_get_dataset_cameras(), lights=lights, device=resource_device, \
							density_range=[0.15,0.25] if is_dens_preopt else [0.05,0.14], \
							inner_range=[0.1,0.4], \
							scale_range=[0.25,0.35] if setup.data.SDF else [0.3,0.5], # if is_dens_preopt else [synth_max_cell_size,synth_max_cell_size*2.0],
							translation_range=[-synth_max_translation,synth_max_translation], \
							rotation_range=[10,30], #[0,20],
							raw=True, preproc=True, bkg=True, hull=True, mask=setup.data.SDF, channels=1, SDF=setup.data.SDF, \
							density=load_density_dataset, velocity=load_velocity_dataset, advect_density=load_velocity_dataset, \
							density_sampler=density_sampler, density_renderer=synth_target_renderer, randomize_transform=False, \
							seed=np.random.randint(np.iinfo(np.int32).max) if setup.validation.synth_data_seed is None else setup.validation.synth_data_seed, \
							sample_overrides=val_sample_overrides, # if is_dens_preopt else 0.1,, "base_scale":[0.35]*3 if is_dens_preopt else [0.18]*3, "initial_translation":[0,0,0]
							data_cache=SDF_dataset_cache, generate_shape=setup.data.synth_shapes.active, \
							generate_sequence=setup.data.synth_shapes.active or len(SDF_val_frames)<2 , \
							sims=[setup.validation.simulation], frames=SDF_val_frames, steps=setup.data.sequence_step)
					else:
						synth_target_dataset = target_dataset
						del target_dataset
					
					if make_disc_dataset:
						SDF_disc_frames = list(range(*setup.data.discriminator.frames))
						disc_dataset = get_synthTargets_dataset_v2(batch_size=setup.training.discriminator.num_real, base_grid_transform=_get_dataset_transform(), \
							sequence_length=3 if setup.training.discriminator.temporal_input.active else 1, \
							view_indices=[scalarFlow_cam_ids[_] for _ in setup.data.density.target_cam_ids], num_views=len(setup.data.density.target_cam_ids), \
							cameras=_get_dataset_cameras(), lights=lights, device=resource_device, \
							density_range=[0.12,0.35] if is_dens_preopt else [0.05,0.14], \
							inner_range=[0.1,0.4], \
							scale_range=[0.25,0.35] if setup.data.SDF else [synth_max_cell_size,0.5], #if is_dens_preopt else [synth_max_cell_size,synth_max_cell_size*2.0],
							translation_range=[-synth_max_translation,synth_max_translation], \
							rotation_range=[10,30], #[0,20],
							raw=False, preproc=True, bkg=False, hull=setup.training.discriminator.conditional_hull, mask=False, channels=1, SDF=setup.data.SDF, \
							density=False, velocity=False, advect_density=load_velocity_dataset, \
							density_sampler=density_sampler, density_renderer=synth_target_renderer, randomize_transform=False, \
							seed=np.random.randint(np.iinfo(np.int32).max), sample_overrides=sample_overrides, \
							data_cache=SDF_dataset_cache, generate_shape=setup.data.synth_shapes.active, \
							generate_sequence=setup.data.synth_shapes.active or len(SDF_disc_frames)<2 , \
							sims=list(range(*setup.data.discriminator.simulations)), frames=SDF_disc_frames, \
							steps=setup.data.sequence_step if setup.training.discriminator.temporal_input.active else [1])
				
				if setup.data.synth_shapes.active=="BOTH" and not setup.data.SDF:
					target_dataset = MultiDataset((SF_target_dataset, synth_target_dataset), weights=(0.5,0.5), seed=np.random.randint(np.iinfo(np.int32).max))
					if make_disc_dataset:
						disc_dataset = MultiDataset((SF_disc_dataset, disc_dataset), weights=(0.5,0.5), seed=np.random.randint(np.iinfo(np.int32).max))
				
				# returns: [NSVHWC for type]
				if False and setup.data.synth_shapes.active:
					target_dataset = TargetDataset(target_data, resource_device=resource_device)
					validation_dataset = TargetDataset(validation_data, resource_device=resource_device)
			
			
			aux_sequence = {}
			val_sequence = {}
		#	if setup.data.clip_grid:
			log.info("Load targets")
			for frame in frames:
				aux_sequence[frame] = frame_loadTargets(setup, frame, sim_transform, target_dataset)
				val_sequence[frame] = frame_loadTargets(setup, frame, sim_transform, validation_dataset)
			
			
			if not setup.data.randomize>0:
				raise NotImplementedError
			else:
				log.info("%s, cell size %s, grid size %s from %s to %s", sim_transform, sim_transform.cell_size_world(), sim_transform.grid_size_world(), sim_transform.grid_min_world(), sim_transform.grid_max_world())
				
			
			curr_cam_res = current_grow_shape(train_cam_resolution, 0, setup.training.density.grow.factor, setup.training.density.grow.intervals)
			main_opt_start_dens_shape = current_grow_shape(base_shape, 0, setup.training.density.grow.factor, setup.training.density.grow.intervals)
			main_opt_start_vel_shape = current_grow_shape(base_shape, 0, setup.training.velocity.grow.factor, setup.training.velocity.grow.intervals)
			pre_opt_start_dens_shape = copy.deepcopy(main_opt_start_dens_shape)
			pre_opt_start_vel_shape = current_grow_shape(main_opt_start_vel_shape, 0, setup.training.velocity.pre_opt.grow.factor, setup.training.velocity.pre_opt.grow.intervals)
			curr_dens_shape = pre_opt_start_dens_shape if setup.training.density.pre_optimization else main_opt_start_dens_shape
			curr_vel_shape = pre_opt_start_vel_shape if setup.training.velocity.pre_optimization else main_opt_start_vel_shape
			z = tf.zeros([1] + curr_vel_shape + [1])
			log.info("Inital setup for sequence reconstruction:\n\tbase shape %s,\n\tinitial density shape %s,\n\tinitial render shape %s,\n\tpre-opt velocity shape %s,\n\tinitial velocity shape %s", \
				base_shape, main_opt_start_dens_shape, curr_cam_res, pre_opt_start_vel_shape, main_opt_start_vel_shape)
			
			vel_bounds = None if setup.data.velocity.boundary.upper()=='CLAMP' else \
				Zeroset(-1, shape=density_size, outer_bounds="OPEN" if setup.data.velocity.boundary.upper()=='CLAMP' else 'CLOSED', as_var=False, device=resource_device)
			
			### NETWORK SETUP ###
			
			
			def setup_decoder(decoder_config, in_channels, out_channels, shape, name="Decoder", **kwargs):
				dim = len(shape) #"LIFTING" if "lifting_shape" in kwargs else 
				if "lifting_shape" in kwargs:
					dim = "LIFTING"
					lifting_shape = kwargs["lifting_shape"]
					shape = shape if min(lifting_shape)>min(shape) else lifting_shape
				if (not hasattr(decoder_config.model, "num_levels")) or decoder_config.model.num_levels=="VARIABLE":
					num_levels = GrowingUNet.get_max_levels(shape, scale_factor=decoder_config.model.level_scale_factor, min_size=decoder_config.min_grid_res, allow_padded=dim in [2, "LIFTING"])
				else:
					num_levels = decoder_config.model.num_levels
				log.info("%s with %d levels for %s", name, num_levels, shape)
				net_args = copy.deepcopy(decoder_config.model)
				if hasattr(net_args, "num_levels"): del net_args["num_levels"]
				decoder_model = GrowingUNet( dimension=dim, num_levels=num_levels, input_channels=in_channels, \
					output_levels=1, output_channels=out_channels, name=name, \
					**net_args, **kwargs)
				decoder_model.set_active_level(decoder_config.start_level)
				decoder_model.set_train_mode(decoder_config.train_mode, schedule=make_schedule(decoder_config.train_mode_schedule))
				decoder_model.skip_merge_weight_schedule = make_schedule(decoder_config.skip_merge_weight_schedule)
				if not decoder_config.recursive_MS:
					decoder_model.set_grow_intervals(decoder_config.grow_intervals)
				
				string_buffer = StringWriter()
				string_buffer.write_line(name)
				string_buffer.write_line(" Model:")
				decoder_model.summary(print_fn=string_buffer.write_line)
				log.info(string_buffer.get_string())
				string_buffer.reset()
				return decoder_model
			
			def load_decoder(path, decoder_config, in_channels, shape, name="Decoder", **kwargs):
				assert isinstance(path, str)
				dim = len(shape)
				if "lifting_shape" in kwargs:
					dim = "LIFTING"
					lifting_shape = kwargs["lifting_shape"]
					shape = shape if min(lifting_shape)>min(shape) else lifting_shape
				scale_factor = decoder_config.recursive_MS_scale_factor if "recursive_MS_scale_factor" in decoder_config else 2.0
				num_levels = GrowingUNet.get_max_levels(shape, scale_factor=scale_factor, min_size=decoder_config.min_grid_res, allow_padded=dim in [2, "LIFTING"])
				# load model from checkpoint
				decoder_model = load_model(path, num_levels=num_levels, input_merge_weight=0.5, skip_merge_weight=1.0, **kwargs)
				
				if isinstance(decoder_model, RWDensityGeneratorNetwork):
					log.info("Loaded RWDensityGeneratorNetwork from '%s'", path)
				elif isinstance(decoder_model, RWVelocityGeneratorNetwork):
					log.info("Loaded RWVelocityGeneratorNetwork from '%s'", path)
				else:
					log.info("Loaded %s from '%s' with %d levels for %s", name, path, decoder_model.num_levels, shape)
					if decoder_model.input_channels!=in_channels:
						log.error("Loaded model input channels (%d) not compatible with current setup (%d).", decoder_model.input_channels, in_channels)
						sys.exit(1)
				
				string_buffer = StringWriter()
				string_buffer.write_line(name)
				string_buffer.write_line(" Model:")
				decoder_model.summary(print_fn=string_buffer.write_line)
				log.info(string_buffer.get_string())
				string_buffer.reset()
				return decoder_model
				
			
			
			def get_gen_num_inputs(decoder_config, feature_channels, out_channels):
				num_inputs = 0
				num_inputs += len(decoder_config.step_input_density)
				num_inputs += len(decoder_config.step_input_density_target)
				if hasattr(decoder_config, "step_input_density_proxy"):
					num_inputs += len(decoder_config.step_input_density_proxy)
				volume_encoder_channels = color_channel
				if setup.training.view_encoder.lifting=="UNPROJECT" and setup.training.view_encoder.merge in ["NETWORK_CONCAT", "CONCAT_NETWORK"]:
					volume_encoder_channels = volume_encoder_model.output_channels
					if setup.training.randomization.inputs==True:
						raise ValueError("View concatentaion does not work with randomized input views.")
					elif isinstance(setup.training.randomization.inputs, (list,tuple)) and setup.training.view_encoder.merge=="NETWORK_CONCAT":
						volume_encoder_channels *= len(setup.training.randomization.inputs)
					elif not setup.training.view_encoder.merge=="CONCAT_NETWORK":
						volume_encoder_channels *= len(target_cameras)
				elif setup.training.view_encoder.lifting=="NETWORK":
					volume_encoder_channels = lifting_network_model.output_channels
				
				num_views = None
				if isinstance(setup.training.randomization.inputs, (list,tuple)):
					num_views = len(setup.training.randomization.inputs)
				elif setup.training.randomization.inputs==False:
					num_views = len(target_cameras)
				
				num_inputs += len(decoder_config.step_input_features) * NeuralState.get_base_feature_channels(decoder_config.type_input_features, color_channels=color_channel, volume_encoder_channels=volume_encoder_channels, num_views=num_views)
				if "ENCLIFT" in decoder_config.type_input_features:
					num_inputs += len(decoder_config.step_input_features) * vel_input_encoder_channels
				if setup.data.SDF and decoder_config.get("base_SDF_mode", "NONE")=="INPUT_RESIDUAL":
					num_inputs += 1
				if decoder_config.recursive_MS: num_inputs +=out_channels
				return num_inputs
			
			def get_decoder_model(decoder_config, out_channels, shape, name="Decoder", in_channels=None, **kwargs):
				is_load_model = False
				if isinstance(decoder_config.model, str):
					# load model from file
					model_path = run_index[decoder_config.model] #setup.training.density.decoder.model]
					if model_path is None:
						model_path = decoder_config.model
					model_paths = get_NNmodel_names(model_path)
					is_load_model = True
				
				if in_channels is None:
					input_channels = get_gen_num_inputs(decoder_config, view_encoder_channels, out_channels)
				else:
					input_channels = in_channels
				log.info("%s input channels: %d", name, input_channels)
				if decoder_config.recursive_MS and not decoder_config.recursive_MS_shared_model:
					# setup separate model for each level
					max_levels = get_max_recursive_MS_grow_levels(decoder_config)
					decoder_model = []
					if is_load_model:
						if len(model_paths)==1: #loading a shared model
							model_paths = model_paths*max_levels
							log.info("Loading single decoder for recursive multi-scale with %d levels.", max_levels)
						elif len(model_paths)!=max_levels:
							log.error("Trying to load a non-shared model from file, but %d models are available for mask '%s', %d models are needed.", len(model_paths), decoder_config.model, max_levels)
							sys.exit(1)
						else:
							log.info("Loading %d decoders for recursive multi-scale.", max_levels)
						for level in range(max_levels):
							decoder_model.append(load_decoder(model_paths[level], decoder_config, input_channels, name="{}_L{:03d}".format(name,level), shape=shape, **kwargs))
					else:
						log.info("Setup %d decoders for recursive multi-scale.", max_levels)
						for level in range(max_levels):
							decoder_model.append(setup_decoder(decoder_config, input_channels, out_channels, name="{}_L{:03d}".format(name, level), shape=shape, **kwargs))
				else:
					if is_load_model:
						if len(model_paths)>1:
							log.error("Trying to load a shared model from file, but %d models are available for mask '%s': %s", len(model_paths), decoder_config.model, model_paths)
							sys.exit(1)
						decoder_model = load_decoder(model_paths[0], decoder_config, input_channels, name=name, shape=shape, **kwargs)
					else:
						decoder_model = setup_decoder(decoder_config, input_channels, out_channels, name=name, shape=shape, **kwargs)
				
				return decoder_model
			
			
			setup.training.view_encoder.encoder = set(setup.training.view_encoder.encoder)
			def setup_target_encoder():
				view_encoder_channels = 0
				target_encoder_model = None
				if "NETWORK" in setup.training.view_encoder.encoder:
					view_encoder_config = copy.deepcopy(setup.training.view_encoder)
					view_encoder_config.recursive_MS = False
					del view_encoder_config.model["output_channels"]
					target_encoder_model = get_decoder_model(view_encoder_config, in_channels=color_channel, \
					out_channels=setup.training.view_encoder.model.output_channels, name="ViewEncoder", shape=train_cam_resolution[1:])
					
					if target_encoder_model.num_levels>1:
						raise NotImplementedError("handle skip merge weights, default to 0.0.")
					
					view_encoder_channels += setup.training.view_encoder.model.output_channels
				
				if "L" in setup.training.view_encoder.encoder:
					view_encoder_channels += 1
				
				if "IDENTITY" in setup.training.view_encoder.encoder:
					view_encoder_channels += color_channel #1 if setup.rendering.monochrome else 3
				
				if view_encoder_channels<1:
					raise ValueError("Empty input encoder.")
				return target_encoder_model, view_encoder_channels
			
			def setup_lifting_network():
				if setup.training.view_encoder.lifting.upper()=="NETWORK":
					if setup.training.lifting_network.active:
						if isinstance(setup.training.lifting_network.model, str) and setup.training.lifting_network.model.startswith("SDFDiff"):
							log.info("Using SDFDiff lifting network.")
							if not (setup.data.grid_size==64 and len(setup.training.view_encoder.encoder)==1 and 'IDENTITY' in setup.training.view_encoder.encoder and isinstance(setup.training.randomization.inputs, list) and len(setup.training.randomization.inputs) in [1,2]):
								raise RuntimeError("Wrong configuration for SDFDiff lifting network")
							lifting_network_model = SDFDiffAENetwork(input_channels=color_channel)
							
							if setup.training.lifting_network.model.startswith("SDFDiff=[RUNID:"):
								# my tf/keras version seems broken regarding loading full models, so load weights instead
								model_path = setup.training.lifting_network.model[8:]
								model_path = run_index[model_path] or model_path
								model_paths = get_NNmodel_names(model_path)
								assert len(model_paths)==1
								model_path = model_paths[0] + "_model.h5"
								log.info("Loading weights from: %s", model_path)
								lifting_network_model.load_weights(model_path)
							
							string_buffer = StringWriter()
							string_buffer.write_line("SDFDiff Lifting Network Model:")
							lifting_network_model.summary(print_fn=string_buffer.write_line)
							log.info(string_buffer.get_string())
							string_buffer.reset()
							
						else: 
							log.warning("Using lifting UNet.")
							assert len(setup.training.randomization.inputs)==1
							lifting_cameras = [target_cameras[_] for _ in setup.training.randomization.inputs]
							if not setup.training.randomization.inputs==setup.validation.input_view_mask:
								raise NotImplementedError("TODO: Set lifting_network's cameras during training and validation")
							lifting_transform = sim_transform.copy_no_data()
							
							lifting_network_config = copy.deepcopy(setup.training.lifting_network)
							lifting_network_config.recursive_MS = False
							out_channels = 0
							if isinstance(lifting_network_config.model, dict):
								out_channels = setup.training.lifting_network.model.output_channels
								del lifting_network_config.model["output_channels"]
							lifting_network_model = get_decoder_model(lifting_network_config, in_channels=view_encoder_channels, \
								out_channels=out_channels, name="LiftingNetwork", \
								shape=train_cam_resolution[1:], lifting_renderer=lifting_renderer, lifting_cameras=lifting_cameras, lifting_transform=lifting_transform, lifting_shape=density_size.as_shape.tolist(), \
								enc_outputs="ENCLIFT" in setup.training.velocity.decoder.type_input_features)
							
							skip_merge_weight = 1.0
							log.info("Setting skip merge weights to %f.", skip_merge_weight)
							for l in range(1, lifting_network_model.num_levels):
								lifting_network_model.set_skip_merge_weight(skip_merge_weight,l)
					else:
						raise RuntimeError
						# old fully connection version
						log.warning("Experimental lifting network active!")
						# fixed shapes for testing
						image_shape = [33,18,3]
						volume_shape = [11,22,11,setup.training.volume_encoder.model.output_channels]
						lifting_network_model = LiftingNetwork(input_shape=image_shape, output_shape=volume_shape)
					
						string_buffer = StringWriter()
						string_buffer.write_line("Lifting Network Model:")
						lifting_network_model.summary(print_fn=string_buffer.write_line)
						log.info(string_buffer.get_string())
						string_buffer.reset()
					
					return lifting_network_model
				else:
					return None
			
			
			volume_encoder_model = None
			lifting_network_model = None
			if (setup.training.density.decoder.active or setup.training.velocity.decoder.active):
				target_encoder_model, view_encoder_channels = setup_target_encoder()
				
				if setup.training.view_encoder.lifting.upper()=="UNPROJECT" and setup.training.volume_encoder.active:
					def get_volume_encoder_in_channels():
						c = view_encoder_channels
						if setup.training.view_encoder.lifting=="UNPROJECT" and setup.training.view_encoder.merge=="CONCAT_NETWORK":
							if isinstance(setup.training.randomization.inputs, list):
								c *= len(setup.training.randomization.inputs)
							elif setup.training.randomization.inputs in [True, "TARGET"]:
								raise ValueError("View concatentaion does not work with randomized input views.")
							else:
								c *= len(target_cameras)
						return c
					def get_volume_encoder_out_channels():
						if isinstance(setup.training.volume_encoder.model, str):
							# support for loading models
							model_path = run_index[setup.training.volume_encoder.model] #setup.training.density.decoder.model]
							if model_path is None:
								model_path = setup.training.volume_encoder.model
							
							model_path += ".json"
							if not os.path.exists(model_path):
								raise IOError("Can't read config for volume encoder model: file '%s' not found."%(model_path,))
							
							with open(model_path, "r") as config_file:
								config = json.load(config_file)
							return config["_config"]["output_channels"]
							
						else:
							return setup.training.volume_encoder.model.output_channels
					volume_encoder_config = copy.deepcopy(setup.training.volume_encoder)
					volume_encoder_config.recursive_MS = False
					if isinstance(volume_encoder_config.model, dict):
						del volume_encoder_config.model["output_channels"]
					
					volume_encoder_model = get_decoder_model(volume_encoder_config, in_channels=get_volume_encoder_in_channels(), \
						out_channels=get_volume_encoder_out_channels(), shape=density_size, name="VolumeEncoder")
					
					vol_enc_skip_merge_weight = 1.0
					log.info("Setting volume encoder skip merge weights to %f.", vol_enc_skip_merge_weight)
					for l in range(1, volume_encoder_model.num_levels):
						volume_encoder_model.set_skip_merge_weight(vol_enc_skip_merge_weight,l)
				elif setup.training.view_encoder.lifting.upper()=="NETWORK":
					lifting_network_model = setup_lifting_network()
			
			def get_vel_input_encoder():
				vel_input_encoder = None
				out_channels = 0
				if "ENCLIFT" in setup.training.velocity.decoder.type_input_features:
					# take skip lifting output of state lifting encoder
					# make sure it exisits
					if lifting_network_model is None: raise RuntimeError("Can't use ENCLIFT input without lifting network.")
					if not setup.training.velocity.decoder.recursive_MS_shared_model: raise NotImplementedError
					# check if shared vel input encoders are possibel - if lifting network is shared.
					if not lifting_network_model.share_encoder:
						is_load_model = False
						if isinstance(setup.training.velocity.decoder.model, str):
							raise NotImplementedError
							is_load_model = True
						decoder_config = {
								"model": {
									"num_levels": 1,
									"level_scale_factor": 2.0,
									
									"input_levels": 1,
									"create_inputs": False,
									"input_blocks": [],
									"share_input_layer": True,
									
									"down_mode": "NONE", # STRIDED, NONE
									"down_conv_filters": None,
									"down_conv_kernel_size": 4,
									"share_down_layer": True,
									"encoder_resblocks": [],
									"share_encoder": True,
									
									"decoder_resblocks": [],
									"share_decoder": False,
									"up_mode": "NNSAMPLE_CONV",
									"up_conv_filters": 16,
									"up_conv_kernel_size": 4,
									"share_up_layer": True,
									
									"skip_merge_mode": "CONCAT", # CONCAT, WSUM, SUM
									
									#"output_blocks": ["C:16-1"],
									"output_blocks": [],
									"share_output_layer": True,
									"output_conv_kernel_size": 1, #0 to disable additional output convolution
									"output_activation": "relu",
									"output_mode": "SINGLE", # SINGLE, RESIDUAL
									"conv_padding": "ZERO", # ZERO, MIRROR
								},
								"recursive_MS": True,
								#"min_grid_res": 4,#8,
								"start_level": 0,
								"train_mode": "ALL", #ALL, TOP
								"train_mode_schedule": False, #ALL, TOP
								"skip_merge_weight_schedule": 1.0,
								
								"recursive_MS_levels": lifting_network_model.num_levels,
								#"recursive_MS_direct_input": False,
								#"recursive_MS_scale_factor": dec_scale_factor,
								"recursive_MS_shared_model": False,
								"recursive_MS_train_mode": "ALL", #ALL, TOP
								#"recursive_MS_copy_on_grow": False,
								#"grow_intervals": [],
							}
						decoder_config = munch.munchify(decoder_config)
						out_channels = 8
						
						vel_input_encoder = []
						if is_load_model:
							raise NotImplementedError
						else:
							log.info("Setup %d velocity input encoders.", lifting_network_model.num_levels)
							for level in range(lifting_network_model.num_levels):
								vel_input_encoder.append(setup_decoder(decoder_config, lifting_network_model._get_encoder_output_channels(level), out_channels, name="VelocityInpEnc_L{:03d}".format(level), shape=density_size))
					else:
						vel_input_encoder = None
						out_channels = lifting_network_model._get_encoder_output_channels(0)
					# check if required scales are available
				
				return vel_input_encoder, out_channels
			
			def get_vel_downscale_encoder():
				vel_downscale_encoder = None
				channels = 0
				
				if "ENCODER" in setup.training.velocity.decoder.downscale_input_modes:
					is_load_model = False
					if isinstance(setup.training.velocity.decoder.model, str):
						raise NotImplementedError
						is_load_model = True
					
					assert setup.training.velocity.decoder.recursive_MS_scale_factor==2
					
					decoder_config = {
							"model": {
								"num_levels": 1,
								"level_scale_factor": 2.0,
								
								"input_levels": 1,
								"create_inputs": False,
								"input_blocks": ["C:16-4-s2", "C:16-3"], #2x down
								"share_input_layer": True,
								
								"down_mode": "NONE", # STRIDED, NONE
								"down_conv_filters": None,
								"down_conv_kernel_size": 4,
								"share_down_layer": True,
								"encoder_resblocks": [],
								"share_encoder": True,
								
								"decoder_resblocks": [],
								"share_decoder": False,
								"up_mode": "NNSAMPLE_CONV",
								"up_conv_filters": 16,
								"up_conv_kernel_size": 4,
								"share_up_layer": True,
								
								"skip_merge_mode": "CONCAT", # CONCAT, WSUM, SUM
								
								#"output_blocks": ["C:16-1"],
								"output_blocks": [],
								"share_output_layer": True,
								"output_conv_kernel_size": 3, #0 to disable additional output convolution
								"output_activation": "relu",
								"output_mode": "SINGLE", # SINGLE, RESIDUAL
								"conv_padding": "ZERO", # ZERO, MIRROR
							},
							"recursive_MS": True,
							#"min_grid_res": 4,#8,
							"start_level": 0,
							"train_mode": "ALL", #ALL, TOP
							"train_mode_schedule": False, #ALL, TOP
							"skip_merge_weight_schedule": 1.0,
							
							"recursive_MS_levels": lifting_network_model.num_levels,
							#"recursive_MS_direct_input": False,
							#"recursive_MS_scale_factor": dec_scale_factor,
							"recursive_MS_shared_model": False,
							"recursive_MS_train_mode": "ALL", #ALL, TOP
							#"recursive_MS_copy_on_grow": False,
							#"grow_intervals": [],
						}
					decoder_config = munch.munchify(decoder_config)
					
					vel_levels = None
					channels = view_encoder_channels
					
					if setup.training.velocity.decoder.share_downscale_encoder:
						if is_load_model:
							raise NotImplementedError
						else:
							log.info("Setup shared velocity input downscale encoder.")
							vel_downscale_encoder = setup_decoder(decoder_config, channels, channels, name="VelocityDownEnc", shape=density_size)
					else:
						if is_load_model:
							raise NotImplementedError
						else:
							log.info("Setup %d velocity input downscale encoders.", lifting_network_model.num_levels)
							vel_downscale_encoder = []
							for level in range(lifting_network_model.num_levels-1): # no network needed for highest level
								vel_downscale_encoder.append(setup_decoder(decoder_config, channels, channels, name="VelocityDownEnc_L{:03d}".format(level), shape=density_size))
					
				return vel_downscale_encoder, channels
			
			if setup.training.velocity.decoder.active:
				vel_input_encoder_model, vel_input_encoder_channels = get_vel_input_encoder()
				
				# actually a vel input downscale network
				vel_downscale_encoder_model, vel_downscale_encoder_channels = get_vel_downscale_encoder()
				
				if setup.training.velocity.decoder.model=="RW":
					log.warning("Using RW velocity network.")
					if not (setup.data.grid_size==64 and setup.training.velocity.decoder.step_input_density==[0] and setup.training.velocity.decoder.step_input_density_proxy==[1] and setup.training.velocity.decoder.step_input_features==[]):
						raise RuntimeError("Wrong configuration for RW velocity network")
					velocity_decoder_model = RWVelocityGeneratorNetwork(dens_channels=1, unp_channels=1, use_proxy=True)
					
					string_buffer = StringWriter()
					string_buffer.write_line("RW velocity Network Model:")
					velocity_decoder_model.summary(print_fn=string_buffer.write_line)
					log.info(string_buffer.get_string())
					string_buffer.reset()
				
				else: 
					velocity_decoder_model = get_decoder_model(setup.training.velocity.decoder, 3, shape=density_size, name="VelocityDecoder")
					if velocity_decoder_model.num_levels>1:
						skip_merge_weight = 1.0
						log.info("Setting skip merge weights to %f.", skip_merge_weight)
						for l in range(1, velocity_decoder_model.num_levels):
							velocity_decoder_model.set_skip_merge_weight(skip_merge_weight,l)
			
			
			if setup.training.density.decoder.active:
				if setup.training.density.decoder.model=="SDFDiff":
					log.warning("Using SDFDiff refinement network.")
					if not (setup.data.grid_size==64 and isinstance(lifting_network_model, SDFDiffAENetwork)):
						raise RuntimeError("Wrong configuration for SDFDiff refinement network")
					density_decoder_model = SDFDiffRefinerNetwork()
					
					string_buffer = StringWriter()
					string_buffer.write_line("SDFDiff Refinement Network Model:")
					density_decoder_model.summary(print_fn=string_buffer.write_line)
					log.info(string_buffer.get_string())
					string_buffer.reset()
				
				elif setup.training.density.decoder.model=="RW":
					log.warning("Using RW density network.")
					if not (setup.data.grid_size==64 and len(setup.training.view_encoder.encoder)==1 and 'L' in setup.training.view_encoder.encoder and setup.training.view_encoder.lifting=="UNPROJECT" and setup.training.view_encoder.merge=="CONCAT" and isinstance(setup.training.randomization.inputs, list) and len(setup.training.randomization.inputs) in [1,2]):
						raise RuntimeError("Wrong configuration for RW density network")
					single_view = len(setup.training.randomization.inputs)==1
					density_decoder_model = RWDensityGeneratorNetwork(input_channels=1, w1=(1.0 if single_view else 0.5), w2=(0 if single_view else 0.5))
					del single_view
					string_buffer = StringWriter()
					string_buffer.write_line("SDFDiff Refinement Network Model:")
					density_decoder_model.summary(print_fn=string_buffer.write_line)
					log.info(string_buffer.get_string())
					string_buffer.reset()
				
				else:
					density_decoder_model = get_decoder_model(setup.training.density.decoder, 1, shape=density_size, name="DensityDecoder")#, in_channels=9)
					if isinstance(density_decoder_model, GrowingUNet) and density_decoder_model.num_levels>1:
						raise NotImplementedError("handle skip merge weights, default to 0.0.")
			
			# new network, create after the others to keep initialization consistent with previous tests
			def setup_frame_merge_network():
				if (setup.training.density.decoder.active or setup.training.velocity.decoder.active) and setup.training.frame_merge_network.active:
					volume_encoder_channels = volume_encoder_model.output_channels if volume_encoder_model is not None else lifting_network_model.output_channels
					frame_merge_network_config = copy.deepcopy(setup.training.frame_merge_network)
					frame_merge_network_config.recursive_MS = False
					frame_merge_network_model = get_decoder_model(frame_merge_network_config, in_channels=volume_encoder_channels*2, \
						out_channels=volume_encoder_channels, shape=density_size, name="FrameMerge")
					if frame_merge_network_model.num_levels>1:
						raise NotImplementedError("handle skip merge weights, default to 0.0.")
					return frame_merge_network_model
				else:
					return None
			frame_merge_network_model = setup_frame_merge_network()
			
			
			log.info("--- Sequence setup ---")
			def frame_velSetup(aux_sequence, frame, first_frame, vel_init=None):
				#setup velocity
				#with tf.device(resource_device):
				vel_var_name = "velocity_f{:06d}".format(frame)
				if first_frame and setup.training.velocity.pre_optimization and setup.training.velocity.pre_opt.first.grow.intervals:
					vel_var_name = "velocity_f{:06d}_g000".format(frame)
				elif setup.training.velocity.grow.intervals:
					vel_var_name = "velocity_f{:06d}_g000".format(frame)
				
				vel_shape = (pre_opt_first_start_vel_shape if first_frame else pre_opt_start_vel_shape) if setup.training.velocity.pre_optimization else main_opt_start_vel_shape
				
				
				if setup.training.velocity.decoder.active:
					log.debug("using NN for velocity of frame %d", frame)
					velocity = NeuralVelocityGrid(volume_decoder=velocity_decoder_model, boundary=vel_bounds, scale_renderer=scale_renderer, warp_renderer=warp_renderer, \
						device=resource_device, var_name=vel_var_name, parent_state=None, velocity_format=setup.training.velocity.decoder.velocity_format, \
						step_input_density=setup.training.velocity.decoder.step_input_density, \
						step_input_density_target=setup.training.velocity.decoder.step_input_density_target, \
						step_input_density_proxy=setup.training.velocity.decoder.step_input_density_proxy, \
						step_input_features=setup.training.velocity.decoder.step_input_features, \
						type_input_features=setup.training.velocity.decoder.type_input_features, \
						warp_input_indices=setup.training.velocity.decoder.warp_input_indices, \
						downscale_input_modes=setup.training.velocity.decoder.downscale_input_modes)
					velocity.use_raw_images = setup.training.velocity.decoder.input_type=='RAW'
					velocity.set_input_encoder(vel_input_encoder_model)
					velocity.set_downscale_encoder(vel_downscale_encoder_model)
				elif ("velocity" in aux_sequence[frame]) and (aux_sequence[frame].velocity is not None):
					log.warning("Using dummy velocity for frame %d. TODO", frame)
					velocity = VelocityGrid(main_opt_start_vel_shape, setup.data.velocity.init_std * setup.data.step, boundary=vel_bounds, scale_renderer=scale_renderer, warp_renderer=warp_renderer, device=resource_device, var_name='vel_dummy_f{:06d}'.format(frame))
				else:
					if not setup.data.velocity.initial_value.upper().startswith('RAND'):
						if first_frame or not setup.training.velocity.pre_optimization:
							velocity = load_velocity(setup.data.velocity.initial_value, {'sim':setup.data.simulation,'frame':frame}, boundary=vel_bounds, scale_renderer=scale_renderer, warp_renderer=warp_renderer, device=resource_device, var_name=vel_var_name)
							rel_vel_scale = float(setup.data.step/setup.data.velocity.load_step)
							if rel_vel_scale!=1.:
								log.debug("scale loaded velocity with %f", rel_vel_scale)
								velocity.scale_magnitude(rel_vel_scale)
							if velocity.centered_shape != vel_shape:
								log.error("Shape %s of loaded velocity does not match required shape %s.", velocity.centered_shape, \
									(pre_opt_start_vel_shape if setup.training.velocity.pre_optimization else main_opt_start_vel_shape))
								sys.exit(1)
						else: #not first and pre-opt. will be overwritten, put dummy data as file might not exist
							velocity = VelocityGrid(main_opt_start_vel_shape, setup.data.velocity.init_std * setup.data.step, boundary=vel_bounds, scale_renderer=scale_renderer, warp_renderer=warp_renderer, device=resource_device, var_name='vel_dummy_f{:06d}'.format(frame))
					else:
						velocity = VelocityGrid(vel_shape, setup.data.velocity.init_std * setup.data.step, boundary=vel_bounds, scale_renderer=scale_renderer, warp_renderer=warp_renderer, device=resource_device, var_name=vel_var_name)
						if vel_init is not None:
							velocity.assign(vel_init.x, vel_init.y, vel_init.z)
					
					if setup.data.velocity.init_mask != 'NONE':
						if setup.data.velocity.init_mask == 'HULL':
							vel_mask = aux_sequence[frame].hull
						if setup.data.velocity.init_mask == 'HULL_NEXT':
							frame_next = frame+setup.data.step
							if frame_next in aux_sequence:
								vel_mask = aux_sequence[frame_next].hull
							else:
								vel_mask = aux_sequence[frame].hull
						elif setup.data.velocity.init_mask == 'HULL_TIGHT':
							vel_mask = aux_sequence[frame].hull_tight
						elif setup.data.velocity.init_mask == 'HULL_TIGHT_NEXT':
							frame_next = frame+setup.data.step
							if frame_next in aux_sequence:
								vel_mask = aux_sequence[frame_next].hull_tight
							else:
								vel_mask = aux_sequence[frame].hull_tight
						else:
							raise ValueError("Unknown velocity mask %s"%setup.data.velocity.init_mask)
						hull_x = scale_renderer.resample_grid3D_aligned(vel_mask, velocity.x_shape, align_x='STAGGER_OUTPUT')
						hull_y = scale_renderer.resample_grid3D_aligned(vel_mask, velocity.y_shape, align_y='STAGGER_OUTPUT')
						hull_z = scale_renderer.resample_grid3D_aligned(vel_mask, velocity.z_shape, align_z='STAGGER_OUTPUT')
						velocity.assign(x=velocity.x*hull_x, y=velocity.y*hull_y, z=velocity.z*hull_z)
				return velocity
			
			def frame_velTargetSetup(aux_sequence, frame):
				if not ("velocity" in aux_sequence[frame]) and (aux_sequence[frame].velocity is not None):
					raise ValueError("")
				vel_var_name = "velocityTarget_f{:06d}".format(frame)
				velocity = VelocityGrid.from_staggered_combined(aux_sequence[frame].velocity, as_var=False, boundary=vel_bounds, \
					scale_renderer=scale_renderer, warp_renderer=warp_renderer, device=resource_device, var_name=vel_var_name, trainable=False)
				return velocity
			
			def frame_densSetup(aux_sequence, frame, first_frame):
				inflow_init = None
				inflow_mask = None
				inflow_offset = None
				if setup.data.density.inflow.active:
					base_inflow_mask, base_inflow_shape, base_inflow_offset = create_inflow(tf.squeeze(aux_sequence[frame].hull, (0,-1)), setup.data.density.inflow.hull_height, setup.data.density.inflow.height)
					if base_inflow_mask is not None: #setup.data.density.inflow.active:
						base_inflow_mask = tf.reshape(base_inflow_mask, [1]+base_inflow_shape+[1])
						log.info("Base Inflow: %s at %s", base_inflow_shape, base_inflow_offset)
						inflow_init = 'CONST'
						inflow_offset = current_grow_shape(base_inflow_offset, 0, setup.training.density.grow.factor, setup.training.density.grow.intervals)
						inflow_shape = current_grow_shape(base_inflow_shape, 0, setup.training.density.grow.factor, setup.training.density.grow.intervals, cast_fn=lambda x: max(round(x),1))
						inflow_mask = scale_renderer.resample_grid3D_aligned(base_inflow_mask, inflow_shape)
					else:
						log.error("Failed to build inflow.")
				
				if setup.training.density.decoder.active:
					log.debug("using NN for density of frame %d", frame)
					density = NeuralDensityGrid(volume_decoder=density_decoder_model, scale_renderer=scale_renderer, parent_state=None, \
						base_input=setup.training.density.decoder.base_input, \
						step_input_density=setup.training.density.decoder.step_input_density, \
						step_input_density_target=setup.training.density.decoder.step_input_density_target, \
						step_input_features=setup.training.density.decoder.step_input_features, \
						type_input_features=setup.training.density.decoder.type_input_features, \
						device=resource_device, is_SDF=setup.data.SDF, base_SDF_mode=setup.training.density.decoder.base_SDF_mode)
					density.use_raw_images = setup.training.density.decoder.input_type=='RAW'
				elif ("density" in aux_sequence[frame]) and (aux_sequence[frame].density is not None):
					dens_hull = None
					density_grid_shape = GridShape.from_tensor(aux_sequence[frame].density).spatial_vector.as_shape
					density = DensityGrid(density_grid_shape, d=aux_sequence[frame].density, as_var=False, \
						scale_renderer=scale_renderer, hull=dens_hull, inflow=inflow_init, inflow_offset=inflow_offset, inflow_mask=inflow_mask, \
						device=resource_device, restrict_to_hull=setup.training.density.use_hull, is_SDF=setup.data.SDF)
					log.info("Initialized density of frame %d for random loaded data with shape %s", frame, density_grid_shape)
					# global curr_dens_shape
					# curr_dens_shape = density_grid_shape
				else:
					#setup density with start scale
					dens_hull = scale_renderer.resample_grid3D_aligned(aux_sequence[frame].hull, curr_dens_shape) # if setup.training.density.use_hull else None
					#with tf.device(resource_device):
					dens_var_name = "density_f{:06d}".format(frame)
					if setup.data.density.initial_value.upper()=="CONST":
						density = DensityGrid(curr_dens_shape, setup.data.density.scale, scale_renderer=scale_renderer, hull=dens_hull, inflow=inflow_init, inflow_offset=inflow_offset, inflow_mask=inflow_mask, \
							device=resource_device, var_name=dens_var_name, restrict_to_hull=setup.training.density.use_hull, is_SDF=setup.data.SDF)
						log.debug("Initialized density with constant value")
					elif setup.data.density.initial_value.upper()=="ESTIMATE":
						density_estimate = renderer.unproject(sim_transform, aux_sequence[frame].targets.images, target_cameras)#*setup.data.density.scale
						density = DensityGrid(curr_dens_shape, d=density_estimate, scale_renderer=scale_renderer, hull=dens_hull, inflow=inflow_init, inflow_offset=inflow_offset, inflow_mask=inflow_mask, \
							device=resource_device, var_name=dens_var_name, restrict_to_hull=setup.training.density.use_hull, is_SDF=setup.data.SDF)
						log.debug("Initialized density with estimate from targets")
					elif setup.data.density.initial_value.upper()=="HULL":
						density = DensityGrid(curr_dens_shape, d=dens_hull*setup.data.density.scale, scale_renderer=scale_renderer, hull=dens_hull, inflow=inflow_init, inflow_offset=inflow_offset, inflow_mask=inflow_mask, \
							device=resource_device, var_name=dens_var_name, restrict_to_hull=setup.training.density.use_hull, is_SDF=setup.data.SDF)
						log.debug("Initialized density with visual hull from targets")
					elif setup.data.density.initial_value.upper()=="HULL_TIGHT":
						h = scale_renderer.resample_grid3D_aligned(aux_sequence[frame].hull_tight, curr_dens_shape)
						density = DensityGrid(curr_dens_shape, d=h*setup.data.density.scale, scale_renderer=scale_renderer, hull=dens_hull, inflow=inflow_init, inflow_offset=inflow_offset, inflow_mask=inflow_mask, \
							device=resource_device, var_name=dens_var_name, restrict_to_hull=setup.training.density.use_hull, is_SDF=setup.data.SDF)
						del h
						log.debug("Initialized density with visual hull from targets")
					else: #load
						if first_frame or not setup.training.density.pre_optimization:
							try:
								path = run_index[setup.data.density.initial_value]
								if path is None:
									path = setup.data.density.initial_value
								path = path.format(sim=setup.data.simulation, frame=frame)
								# TODO: remove hull and inflow params to load original. they may be non-existent or incompatible from older saves
								density = DensityGrid.from_file(path, scale_renderer=scale_renderer, device=resource_device, var_name=dens_var_name, restrict_to_hull=setup.training.density.use_hull, is_SDF=setup.data.SDF) #, hull=dens_hull, inflow=inflow_mask, inflow_offset=inflow_offset
								if density.hull is None: #no hull loaded, but a hull should be used, so use the newly build one
									density.hull = dens_hull
							except:
								log.exception("Falied to load density for frame %d from '%s'", frame, path)
								sys.exit(1)
							else:
								log.debug("Initialized density for frame %d with value loaded from %s", frame, setup.data.density.initial_value)
							if density.shape != curr_dens_shape:
								log.error("Shape %s of density loaded from '%s' does not match required shape %s.", velocity.centered_shape, path, curr_dens_shape)
								sys.exit(1)
						else: #not first and pre-opt. will be overwritten, put dummy data as file might not exist
							density = DensityGrid(curr_dens_shape, setup.data.density.scale, scale_renderer=scale_renderer, hull=dens_hull, inflow=inflow_init, inflow_offset=inflow_offset, inflow_mask=inflow_mask, \
								device=resource_device, var_name=dens_var_name, restrict_to_hull=setup.training.density.use_hull, is_SDF=setup.data.SDF)
				
					with tf.device(resource_device):
						density.base_hull = tf.identity(aux_sequence[frame].hull) #if setup.training.density.use_hull else None
						if inflow_mask is not None: #setup.data.density.inflow.active:
							density.base_inflow_mask = tf.identity(base_inflow_mask)
							density.base_inflow_shape = base_inflow_shape
							density.base_inflow_offset = base_inflow_offset
				return density
			
			def frame_densTargetSetup(aux_sequence, frame):
				if not ("density" in aux_sequence[frame]) and (aux_sequence[frame].density is not None):
					raise ValueError("")
				density_grid_shape = GridShape.from_tensor(aux_sequence[frame].density).spatial_vector.as_shape
				density = DensityGrid(density_grid_shape, d=aux_sequence[frame].density, as_var=False, \
					scale_renderer=scale_renderer, hull=None, inflow=None, inflow_offset=None, inflow_mask=None, \
					device=resource_device, restrict_to_hull=setup.training.density.use_hull, is_SDF=setup.data.SDF)
				return density
			
			def frame_densProxySetup(aux_sequence, frame):
				if not setup.training.density.decoder.active:
					raise RuntimeError("no density network available for density proxy")
				density = NeuralDensityGrid(volume_decoder=density_decoder_model, scale_renderer=scale_renderer, parent_state=None, \
					base_input=setup.training.density.decoder.base_input, \
					step_input_density=setup.training.density.decoder.step_input_density, \
					step_input_density_target=setup.training.density.decoder.step_input_density_target, \
					step_input_features=setup.training.density.decoder.step_input_features, \
					type_input_features=setup.training.density.decoder.type_input_features, \
					device=resource_device, is_SDF=setup.data.SDF, base_SDF_mode=setup.training.density.decoder.base_SDF_mode)
				density.use_raw_images = setup.training.density.decoder.input_type=='RAW'
				return density
			
			def frame_setup(aux_sequence, frame, first_frame, prev=None, vel_init=None):
				log.info("--- Setup frame %d ---", frame)
				velocity = frame_velSetup(aux_sequence, frame, first_frame, vel_init)
				density = frame_densSetup(aux_sequence, frame, first_frame)
				
				if setup.training.density.decoder.active or setup.training.velocity.decoder.active:
					state = NeuralState(density, velocity, target_encoder=target_encoder_model, encoder_output_types=setup.training.view_encoder.encoder, \
						target_lifting=setup.training.view_encoder.lifting, lifting_renderer=lifting_renderer, target_merging=setup.training.view_encoder.merge, \
						volume_encoder=volume_encoder_model, frame=frame, prev=prev, transform=sim_transform.copy_no_data(), lifting_network=lifting_network_model, frame_merge_network=frame_merge_network_model)#
					density.parent_state = state
					velocity.parent_state = state
					
					if setup.training.density.decoder.active:
						state.density_proxy = frame_densProxySetup(aux_sequence, frame)
						state.density_proxy.parent_state = state
					
					if setup.training.velocity.decoder.active and setup.training.velocity.decoder.recursive_MS:
						num_levels = get_max_recursive_MS_grow_levels(setup.training.velocity.decoder)
						scale_factor = setup.training.velocity.decoder.recursive_MS_scale_factor
						velocity.set_recursive_MS(num_levels, scale_factor, shared_decoder=setup.training.velocity.decoder.recursive_MS_shared_model, \
							train_mode=setup.training.velocity.decoder.recursive_MS_train_mode, direct_input=setup.training.velocity.decoder.recursive_MS_direct_input, \
							max_level_input=setup.training.velocity.decoder.recursive_MS_use_max_level_input)
						log.info("Set recursive-MS velocity for frame %d with %d levels.", frame, num_levels)
					
					if setup.training.density.decoder.active and setup.training.density.decoder.recursive_MS:
						num_levels = get_max_recursive_MS_grow_levels(setup.training.density.decoder)
						scale_factor = setup.training.density.decoder.recursive_MS_scale_factor
						density.set_recursive_MS(num_levels, scale_factor, shared_decoder=setup.training.density.decoder.recursive_MS_shared_model, train_mode=setup.training.density.decoder.recursive_MS_train_mode, as_residual=setup.training.density.decoder.recursive_MS_residual, direct_input=setup.training.density.decoder.recursive_MS_direct_input)
						if state.has_density_proxy:
							state.density_proxy.set_recursive_MS(num_levels, scale_factor, shared_decoder=setup.training.density.decoder.recursive_MS_shared_model, train_mode=setup.training.density.decoder.recursive_MS_train_mode, as_residual=setup.training.density.decoder.recursive_MS_residual, direct_input=setup.training.density.decoder.recursive_MS_direct_input)
						log.info("Set recursive-MS density for frame %d with %d levels.", frame, num_levels)
				else:
					state = State(density, velocity, frame=frame, prev=prev, transform=sim_transform.copy_no_data())
					with tf.device(resource_device):
						state.hull = tf.identity(aux_sequence[frame].hull_tight)
				state.data_path = os.path.join(setup.paths.data, 'frame_{:06d}'.format(frame))
				os.makedirs(state.data_path, exist_ok=True)
				#setup targets
				
				if ("velocity" in aux_sequence[frame]) and (aux_sequence[frame].velocity is not None):
					state.velocity_target = frame_velTargetSetup(aux_sequence, frame)
				if ("density" in aux_sequence[frame]) and (aux_sequence[frame].density is not None):
					state.density_target = frame_densTargetSetup(aux_sequence, frame)
				
				if setup.training.velocity.pre_optimization: raise NotImplementedError
				grow_handler.start_iteration(0, is_pre_opt=setup.training.density.pre_optimization)
				state_set_targets(state, aux_sequence)
				return state
			
			def sequence_setup(aux_sequence):
				sequence = []
				prev = None
				vel_init = None
				first_frame = True
				for frame in frames:
					state = frame_setup(aux_sequence, frame, first_frame, prev, vel_init)
					curr_cam_res_MS = grow_handler.get_camera_MS_scale_shapes()
					
					
					state.base_target_cameras = setup_target_cameras(target_cameras, curr_cam_res, jitter=setup.training.density.camera_jitter)
					state.set_base_target_cameras_MS({scale: setup_target_cameras(target_cameras, shape, jitter=setup.training.density.camera_jitter) for scale, shape in curr_cam_res_MS.items()})
					
					log.debug('Write target images')
					state.base_targets_raw.save(renderer, state.data_path, "PNG")
					state.base_targets_raw.save_MS_stack(renderer, state.data_path, "PNG")
					if isinstance(state.base_targets_raw, ImageSetMS):
						state.base_targets_raw.save_base_MS_stack(renderer, state.data_path, "PNG")
					state.base_targets.save(renderer, state.data_path, "PNG")
					state.base_bkgs.save(renderer, state.data_path, "PNG")
					if state.has_masks:
						state.base_masks.save(renderer, state.data_path, "PNG")
					
					# random velocity initialization, but same for each frame
					if setup.data.velocity.initial_value.upper()=='RAND_CONST':
						vel_init = state.velocity
					prev = state
					sequence.append(state)
					first_frame = False
				return sequence
			
			sequence = sequence_setup(val_sequence) #aux_sequence)
			del sequence_setup
			del frame_setup
			del frame_densSetup
			del frame_velSetup
			del aux_sequence
			
			for i in range(len(sequence)-1):
				sequence[i].next = sequence[i+1]
			sequence = Sequence(sequence)
			
			with tf.device(resource_device):
				if setup.training.optimize_buoyancy:
					buoyancy = tf.Variable(initial_value=setup.data.initial_buoyancy, dtype=tf.float32, name='buoyancy', trainable=True)
				else:
					buoyancy = tf.constant(setup.data.initial_buoyancy, dtype=tf.float32)
			
			if setup.training.velocity.pre_opt.first.iterations>0 and setup.training.velocity.pre_optimization: #and setup.training.velocity.pre_optimization ?
				curr_vel_shape = pre_opt_start_vel_shape
			else:
				curr_vel_shape = main_opt_start_vel_shape
				
			s = "Sequence setup:"
			i=0
			for state in sequence:
				p = -1 if state.prev is None else state.prev.frame
				n = -1 if state.next is None else state.next.frame
				s += "\n{:4d}: frame {:06d}: p={:06d},  n={:06d}".format(i, state.frame, p, n)
				i +=1
			log.info(s)
			
			light_var_list = []
			if setup.training.light.optimize:
				log.debug('Initialize variables (density %f and light intensity %f)', setup.data.density.scale, lights[0].i)
				var_intensity = tf.get_variable(name='intensity', initializer=lights[0].i, constraint=lambda var: tf.clip_by_value(var, setup.training.light.min, setup.training.light.max), dtype=tf.float32, trainable=True)
				lights[0].i=var_intensity
				var_ambient_intensity = tf.get_variable(name='ambient_intensity', initializer=lights[1].i, constraint=lambda var: tf.clip_by_value(var, setup.training.light.min, setup.training.light.max), dtype=tf.float32, trainable=True)
				lights[1].i=var_ambient_intensity
				light_var_list = [var_intensity, var_ambient_intensity]
			
			disc_dump_samples = setup.debug.disc_dump_samples
			if setup.training.discriminator.active:
				log.info('Setup discriminator')
				disc_real_data = None
				disc_input_steps = None
				if setup.training.discriminator.temporal_input.active:
					disc_input_steps = list(range(*setup.training.discriminator.temporal_input.step_range))
					if 0 in disc_input_steps:
						disc_input_steps.remove(0)
				if make_disc_dataset:
					log.debug('Setup discriminator training data.')
					disc_real_res = tf.Variable(initial_value=disc_cam_resolution[1:], name='disc_real_res', dtype=tf.int32, trainable=False)
					
					if setup.training.discriminator.temporal_input.active: raise NotImplementedError
					if setup.training.discriminator.conditional_hull: raise NotImplementedError
					
					class DiscDataMapper:
						def __init__(self, dataset, res_var=None):
							self.__dataset = dataset
							self.__res_var = res_var
						def get_next(self):
							batch = self.__dataset.frame_targets(0)[0]
							shape = shape_list(batch)
							#log.info("Raw batch shape %s, required NHWC.", shape)
							samples = tf.unstack(batch, axis=0)
							batch = tf.stack([sample[np.random.choice(shape[1])] for sample in samples], axis=0)
							if self.__res_var is not None:
								batch = wrap_resize_images(batch, self.__res_var.numpy())
							#log.info("Processed batch shape %s, required NHWC.", shape_list(batch))
							self.__dataset.step()
							return batch
					#if setup.data.SDF: raise NotImplementedError("Discriminator not supported in SDF mode.")
					
					
					disc_real_data = DiscDataMapper(disc_dataset, disc_real_res if setup.data.discriminator.scale_real_to_cam else None)
					
				disc_in_channel = color_channel
				if setup.training.discriminator.conditional_hull:
					disc_in_channel += 1
				if setup.training.discriminator.temporal_input.active:
					disc_in_channel *= 3
				if setup.data.discriminator.crop_size=="CAMERA":
					disc_in_shape = disc_cam_resolution[1:] + [disc_in_channel]
				else:
					disc_in_shape = list(setup.data.discriminator.crop_size) + [disc_in_channel] #disc_targets_shape[-3:]
				if train_disc and setup.training.discriminator.history.samples>0:
					log.debug("Initializing fake samples history buffer for discriminator experience replay.")
					if setup.training.discriminator.history.load is not None:
						history_path = run_index[setup.training.discriminator.history.load]
						if history_path is None:
							history_path = setup.training.discriminator.history.load
						raise NotImplementedError()
				log.debug('Setup discriminator model')
				
				# compatibility
				if setup.training.discriminator.use_fc==True:
					setup.training.discriminator.use_fc=[]
				elif setup.training.discriminator.use_fc==False:
					setup.training.discriminator.use_fc=None
				
				if setup.training.discriminator.model is not None:
					model_path = run_index[setup.training.discriminator.model]
					if model_path is None:
						model_path = setup.training.discriminator.model
					disc_model=tf.keras.models.load_model(model_path, custom_objects=custom_keras_objects)
					log.info('Restored model from %s', model_path)
				else:
					disc_model=discriminator(disc_in_shape, layers=setup.training.discriminator.layers, strides=setup.training.discriminator.stride, kernel_size=setup.training.discriminator.kernel_size, final_fc=setup.training.discriminator.use_fc, activation=setup.training.discriminator.activation, alpha=setup.training.discriminator.activation_alpha, noise_std=setup.training.discriminator.noise_std, padding=setup.training.discriminator.padding)
					log.debug('Built discriminator keras model')
				#disc_model(disc_targets, training=False)
				disc_weights = disc_model.get_weights()
				disc_model.summary(print_fn= lambda s: log.info(s))
				
				if np.any(np.less(disc_in_shape[:-1], disc_cam_resolution[1:])) and setup.training.discriminator.use_fc is not None and (not setup.data.discriminator.scale_input_to_crop):
					log.error("Base fake sample camera resolution exeeds rigid discriminator input resolution. Use the patch discriminator or enable input scaling.")
				curr_disc_cam_res = current_grow_shape(disc_cam_resolution, 0, setup.training.density.grow.factor, setup.training.density.grow.intervals)
				for camera in disc_cameras:
					camera.transform.grid_size = curr_disc_cam_res
					if setup.training.discriminator.fake_camera_jitter:
						camera.jitter = camera.depth_step
				disc_real_res.assign(curr_disc_cam_res[1:])
			#END if setup.training.discriminator.active
			
			def grow_networks(sequence, iteration, preopt_iterations_density=0, preopt_iterations_velocity=0):
				for state in sequence:
					if isinstance(state.density, NeuralDensityGrid) and isinstance(state.density.volume_decoder, GrowingUNet):
						state.density.volume_decoder.step(iteration + preopt_iterations_density)
						
					if isinstance(state.velocity, NeuralVelocityGrid) and isinstance(state.velocity.volume_decoder, GrowingUNet):
						state.velocity.volume_decoder.step(iteration + preopt_iterations_velocity)
			
			def grow_density_networks(sequence, iteration, preopt_iterations_density=0, preopt_iterations_targets=0):
				for state in sequence:
					if isinstance(state.density, NeuralDensityGrid) and isinstance(state.density.volume_decoder, GrowingUNet):
						state.density.volume_decoder.step(iteration + preopt_iterations_density)
			
			def grow_velocity_networks(sequence, iteration, preopt_iterations_velocity=0, preopt_iterations_targets=0):
				for state in sequence:
					if isinstance(state.velocity, NeuralVelocityGrid) and isinstance(state.velocity.volume_decoder, GrowingUNet):
							state.velocity.volume_decoder.step(iteration + preopt_iterations_velocity)
			
			def set_density_network_level(sequence, grid_size=None):
				for state in sequence:
					if isinstance(state.density, NeuralDensityGrid) and isinstance(state.density.volume_decoder, GrowingUNet):
						net = state.density.volume_decoder
						gs = grid_size if grid_size is not None else state.transform.grid_size
						level = min(net.num_levels, GrowingUNet.get_max_levels(gs, scale_factor=net.level_scale_factor, min_size=3)) - 1
						net.set_active_level(level)
						log.debug("Set density generator level to %d for grid size %s (frame %d)", level, gs, state.frame)
			
			def set_velocity_network_level(sequence, grid_size=None):
				for state in sequence:
					if isinstance(state.velocity, NeuralVelocityGrid) and isinstance(state.velocity.volume_decoder, GrowingUNet):
						net = state.velocity.volume_decoder
						level = min(net.num_levels, GrowingUNet.get_max_levels(grid_size if grid_size is not None else state.transform.grid_size, scale_factor=net.level_scale_factor, min_size=3)) - 1
						net.set_active_level(level)
						log.debug("Set velocity generator level to %d (frame %d)", level, state.frame)
			
			def set_growing_network_level(model, grid_size, min_size):
				assert isinstance(model, GrowingUNet)
				level = min(model.num_levels, GrowingUNet.get_max_levels(grid_size, scale_factor=model.level_scale_factor, min_size=min_size, allow_padded=True)) - 1
				if not level==model.get_active_level():
					log.info("Set level of %s from %d to %d for grid size %s.", model.name, model.get_active_level(), level, grid_size)
					model.set_active_level(level)
			
			def scale_state_networks(sequence):
				# set recursive MS scales for active state networks
				# TODO: use some density scale settings for now
				cam_grid_size = grow_handler.get_camera_shape()[1:]
				grid_size = grow_handler.get_density_shape() #state.transform.grid_size
				for state in sequence:
					if isinstance(state.target_encoder, GrowingUNet):
						min_size = setup.training.view_encoder.min_grid_res #min(grow_handler.get_image_MS_scale_shapes()[0])
						set_growing_network_level(model=state.target_encoder, grid_size=cam_grid_size, min_size=min_size)
					
					if isinstance(state.volume_encoder, GrowingUNet):
						min_size = setup.training.volume_encoder.min_grid_res
						set_growing_network_level(model=state.volume_encoder, grid_size=grid_size, min_size=min_size)
					if isinstance(state.lifting_network, GrowingUNet):
						min_size = setup.training.lifting_network.min_grid_res
						state.lifting_network.set_active_level_from_grid_size(grid_size=cam_grid_size, min_size=min_size, lifting_size=grid_size)
					if isinstance(state.frame_merge_network, GrowingUNet):
						min_size = setup.training.frame_merge_network.min_grid_res
						set_growing_network_level(model=state.frame_merge_network, grid_size=grid_size, min_size=min_size)
			
			def scale_density(sequence, iteration, factor, intervals, base_shape, actions=[], save=True, scale_all=False, verbose=True):
				global curr_dens_shape, curr_cam_res
				dens_shape = grow_handler.get_density_shape()
				def check_scale_density(state):
					recursive_MS_grow_level = grow_handler.get_density_MS_scale()
					return state.density.shape!=dens_shape \
						or (state.has_density_neural and state.density.recursive_MS and state.density.recursive_MS_current_level!=recursive_MS_grow_level) \
						or (state.has_density_proxy and state.density_proxy.recursive_MS and state.density_proxy.recursive_MS_current_level!=recursive_MS_grow_level)
				scale_sequence = sequence if scale_all else [state for state in sequence if check_scale_density(state)] #.density.shape!=dens_shape]
				if scale_sequence:
					curr_cam_res = grow_handler.get_camera_shape()
					curr_cam_res_MS = grow_handler.get_camera_MS_scale_shapes()
					curr_tar_res_MS = grow_handler.get_image_MS_scale_shapes()
					(log.info if verbose else log.debug)("Rescaling density of frames %s from %s to %s in iteration %d; cameras to %s", [_.frame for _ in scale_sequence], \
						[_.density.shape for _ in scale_sequence], dens_shape, iteration, curr_cam_res)
					log.debug("Saving sequence")
					with profiler.sample("Save sequence"):
						try:
							for state in scale_sequence:
								if save and type(state.density)==DensityGrid: #not isinstance(state.density, (NeuralDensityGrid, WarpedDensityGrid)):
									state.density.save(os.path.join(state.data_path, \
										"density_{}-{}-{}_{}.npz".format(*state.density.shape, iteration)))
						except:
							log.warning("Failed to save density before scaling from %s to %s in iteration %d:", \
								curr_dens_shape, dens_shape, iteration, exc_info=True)
					log.debug("Rescaling density sequence")
					with profiler.sample('Rescale densities'):
						for state in scale_sequence:
							#scale hull and inflow to new shape based on base values
							hull = state.density.scale_renderer.resample_grid3D_aligned(state.density.base_hull, dens_shape)if state.density.hull is not None else None
							
							if type(state.density)==DensityGrid and state.density.shape!=dens_shape: #not isinstance(state.density, NeuralDensityGrid):
								d = state.density.scaled(dens_shape)
								if state.density._inflow is not None:
									raise NotImplementedError("TODO: randomized growing for inflow shape")
									if_off = current_grow_shape(state.density.base_inflow_offset, iteration, factor, intervals)
									if_shape = current_grow_shape(state.density.base_inflow_shape, iteration, factor, intervals, cast_fn=lambda x: max(round(x),1)) #cast_fn=math.ceil
									if_scaled = upscale_renderer.resample_grid3D_aligned(state.density._inflow, if_shape)
									if_mask = None if state.density.inflow_mask is None else state.density.scale_renderer.resample_grid3D_aligned(state.density.base_inflow_mask, if_shape)
									log.info("Frame %04d: inflow to %s, offset to %s", state.frame, if_shape, if_off)
									density = DensityGrid(shape=dens_shape, d=d, as_var=state.density.is_var, hull=hull, inflow=if_scaled, inflow_offset=if_off, inflow_mask=if_mask, \
										scale_renderer=state.density.scale_renderer, device=state.density._device, var_name=state.density._name+"_scaled", restrict_to_hull=state.density.restrict_to_hull, is_SDF=state.density.is_SDF)
								else:
									density = DensityGrid(shape=dens_shape, d=d, as_var=state.density.is_var, hull=hull, \
										scale_renderer=state.density.scale_renderer, device=state.density._device, var_name=state.density._name+"_scaled", restrict_to_hull=state.density.restrict_to_hull, is_SDF=state.density.is_SDF)
								if hull is not None:
									density.base_hull = state._density.base_hull
								if density._inflow is not None:
									density.base_inflow_mask = state._density.base_inflow_mask
									density.base_inflow_shape = state._density.base_inflow_shape
									density.base_inflow_offset = state._density.base_inflow_offset
								state.density = density
							
							if state.has_density_target:
								state.density_target.rescale(dens_shape, None)
							
							state.transform.grid_size = dens_shape
							
							if type(state.density)==NeuralDensityGrid and state.density.recursive_MS:
								copied_weights = False
								recursive_MS_grow_level = grow_handler.get_density_MS_scale()
								# for rand: get_grow_level(vel_scale)
								if state.density.recursive_MS_current_level!=recursive_MS_grow_level:
									copy_weights=((not setup.training.density.decoder.recursive_MS_shared_model) \
										and setup.training.density.decoder.recursive_MS_copy_on_grow \
										and (state.density.recursive_MS_current_level < recursive_MS_grow_level) \
										and not copied_weights) 
									if copy_weights:
										save_NNmodel(velocity_decoder_model, 'velocity_decoder_grow%02d'%(state.density.recursive_MS_current_level, ), setup.paths.data)
									state.density.set_recursive_MS_level(recursive_MS_grow_level, \
										copy_weights=copy_weights) #and state is sequence[0]
									log.debug("Set recursive MS density level of frame %d to %d%s", state.frame, recursive_MS_grow_level,", copy weights from previous level." if copy_weights else ".")
									copied_weights = copy_weights or copied_weights
							
							if state.has_density_proxy and state.density_proxy.recursive_MS:
								recursive_MS_grow_level = grow_handler.get_density_MS_scale()
								if state.density_proxy.recursive_MS_current_level!=recursive_MS_grow_level:
									if setup.training.density.decoder.recursive_MS_copy_on_grow: raise NotImplementedError
								state.density_proxy.set_recursive_MS_level(recursive_MS_grow_level)
							
							AABB_corners_WS = dens_transform.transform_AABB(*hull_AABB_OS(tf.squeeze(state.density.hull, (0,-1))), True) if setup.rendering.target_cameras.crop_frustum else None
							state.base_target_cameras = setup_target_cameras(target_cameras, curr_cam_res, AABB_corners_WS,setup.rendering.target_cameras.crop_frustum_pad, jitter=setup.training.density.camera_jitter)
							target_cameras_MS = {scale: setup_target_cameras(target_cameras, shape, AABB_corners_WS, setup.rendering.target_cameras.crop_frustum_pad, jitter=setup.training.density.camera_jitter) for scale, shape in curr_cam_res_MS.items()}
							state.set_base_target_cameras_MS(target_cameras_MS)
						curr_dens_shape = dens_shape
						#target and cams scale
					log.debug("Rescaling cameras")
					if setup.training.discriminator.active:
						global curr_disc_cam_res
						if grow_handler.is_randomize_shape or grow_handler.is_iterate_shapes:
							raise NotImplementedError("TODO: randomized growing for disc cam shape")
						curr_disc_cam_res = current_grow_shape(disc_cam_resolution, iteration, factor, intervals)
						log.info("Scaling discriminator camera resolution to %s", curr_disc_cam_res)
						for camera in disc_cameras:
							camera.transform.grid_size = curr_disc_cam_res
							if setup.training.discriminator.fake_camera_jitter:
								camera.jitter = camera.depth_step
						disc_real_res.assign(curr_disc_cam_res[1:])
					if train_disc and setup.training.discriminator.history.samples>0:
						if setup.training.discriminator.history.reset_on_density_grow:
							log.info("Reset disc history after rescale")
							disc_ctx.history.reset()
					# targets scaled from base
					if not setup.data.randomize>0: #randomized data is loaded after calling scale_density (so no reason to resize here)
						raise NotImplementedError
						log.debug("Rescaling targets from base")
						with profiler.sample('Rescale targets'):
							for state in scale_sequence:
								
								state.base_targets_raw.resize(curr_cam_res[1:])
								state.base_targets.resize(curr_cam_res[1:])
								state.base_bkgs.resize(curr_cam_res[1:])
								if state.has_masks:
									state.base_masks.resize(curr_cam_res[1:])
								
								state.base_targets_raw.create_MS_stack(curr_tar_res_MS)
								state.base_targets.create_MS_stack(curr_tar_res_MS)
								state.base_bkgs.create_MS_stack(curr_tar_res_MS)
								if state.has_masks:
									state.base_masks.create_MS_stack(curr_tar_res_MS)
					return True
				else:
					return False
				#END if rescale density
			
			def randomize_scale(sequence, *, min_size_abs, min_size_rel, max_shape, train_cam_res, disc_cam_res=None):
				# choose random grid size between min and max (weighted?)
				# set transform of states to this grid size
				# set network levels
				assert setup.data.randomize>0
				assert all(type(state.density)==NeuralDensityGrid for state in sequence)
				
				def lerp_shape(a,b,t):
					return [int(round(_)) for _ in lerp_vector(a, b, t)]
				
				if min_size_rel==1:
					rand_shape = max_shape
					min_shape = max_shape
					t = 1
				elif min_size_rel<0:
					max_size = min(max_shape)
					dim_factors = [_/max_size for _ in max_shape]
					min_shape = [int(_*min_size_abs) for _ in dim_factors]
					net_factor = 2 #TODO
					sizes = [min_size_abs]
					while sizes[-1]*net_factor <= max_size:
						sizes.append(sizes[-1]*net_factor)
					rand_size = np.random.choice(sizes)
					rand_shape = [int(_*rand_size) for _ in dim_factors]
					t = np.mean(rand_shape) / np.mean(max_shape)
				else:
					max_size = min(max_shape)
					#clamp
					min_size_abs = max(min(max_size, min_size_abs), 1)
					min_size_rel = min(min_size_rel, 1.0)
					
					min_size = max(min_size_abs, int(max_size*min_size_rel))
					min_factor = min_size/max_size
					min_shape = [int(np.ceil(_*min_factor)) for _ in max_shape]
					
					t = np.random.random()
					rand_shape = lerp_shape(min_shape, max_shape, t)
					t = np.mean(rand_shape) / np.mean(max_shape)
				rand_train_cam_res = lerp_shape([0,0,0], train_cam_res, t)
				
				if min_size_rel==1:
					log.debug("set max resolution: grid=%s, target=%s", rand_shape, rand_train_cam_res)
				else:
					log.debug("randomize resolution: grid=%s (%s - %s), target=%s (%s)", rand_shape, min_shape, max_shape, rand_train_cam_res, train_cam_res)
				
				for state in sequence:
					assert state.density.hull is None
					state.transform.grid_size = rand_shape
					if setup.rendering.target_cameras.crop_frustum: raise NotImplementedError
					state.base_target_cameras = setup_target_cameras(target_cameras, rand_train_cam_res, None, setup.rendering.target_cameras.crop_frustum_pad, jitter=setup.training.density.camera_jitter)
					state.base_targets_raw.resize(rand_train_cam_res[1:])
					state.base_targets.resize(rand_train_cam_res[1:])
					state.base_bkgs.resize(rand_train_cam_res[1:])
					if state.has_masks:
						state.base_masks.resize(rand_train_cam_res[1:])
					
				set_density_network_level(sequence)
				set_velocity_network_level(sequence)
				
				if setup.training.discriminator.active:
					rand_disc_cam_res = lerp_shape([0,0,0], disc_cam_res, t)
					for camera in disc_cameras:
						camera.transform.grid_size = rand_disc_cam_res
						if setup.training.discriminator.fake_camera_jitter:
							camera.jitter = camera.depth_step
					disc_real_res.assign(rand_disc_cam_res[1:])
			
			def scale_velocity(sequence, iteration, factor, scale_magnitude, intervals, base_shape, verbose=True):
				global curr_vel_shape, z, curr_cam_res
				#vel_shape = current_grow_shape(base_shape, iteration, factor, intervals)
				vel_shape = grow_handler.get_velocity_shape()
				# for rand: rand_vel_shape(min_shape, vel_shape)
				scale_sequence = []
				for state in sequence:
					if state.velocity.centered_shape!=vel_shape:
						scale_sequence.append(state)
				if scale_sequence:
					#curr_cam_res = current_grow_shape(train_cam_resolution, iteration, factor, intervals)
					curr_cam_res = grow_handler.get_camera_shape()
					(log.info if verbose else log.debug)("Rescaling velocity of frames %s from %s to %s in iteration %d, magnitude: %s; cameras to %s", [_.frame for _ in scale_sequence], \
						[_.velocity.centered_shape for _ in scale_sequence], vel_shape, iteration, scale_magnitude, curr_cam_res)
					log.debug("Saving sequence")
					with profiler.sample("Save sequence"):
							try:
								for state in scale_sequence:
									if type(state.velocity)==VelocityGrid:
										state.velocity.save(os.path.join(state.data_path, \
											"velocity_{}-{}-{}_{}.npz".format(*state.velocity.centered_shape, iteration)))
							except:
								log.warning("Failed to save velocity before scaling from %s to %s in iteration %d:", \
									state.velocity.centered_shape, vel_shape, iteration, exc_info=True)
					log.debug("Rescaling velocity sequence")
					with profiler.sample('Rescale velocities'):
						copied_weights = False
						for state in scale_sequence:
							log.debug("Rescaling velocity frame %d", state.frame)
							state.velocity.set_centered_shape(vel_shape) #also affects NeuralDensityGrid grid size
							if type(state.velocity)==VelocityGrid:
								state.rescale_velocity(vel_shape, scale_magnitude=scale_magnitude, device=resource_device)
							if type(state.velocity)==NeuralVelocityGrid and state.velocity.recursive_MS:
								recursive_MS_grow_level = grow_handler.get_velocity_MS_scale()
								log.debug("NVG target level: %d, from %d", recursive_MS_grow_level, state.velocity.recursive_MS_current_level)
								if state.velocity.recursive_MS_current_level!=recursive_MS_grow_level:
									copy_weights=((not setup.training.velocity.decoder.recursive_MS_shared_model) \
										and setup.training.velocity.decoder.recursive_MS_copy_on_grow \
										and (state.velocity.recursive_MS_current_level < recursive_MS_grow_level) \
										and not copied_weights) 
									if copy_weights:
										save_NNmodel(velocity_decoder_model, 'velocity_decoder_grow%02d'%(state.velocity.recursive_MS_current_level, ), setup.paths.data)
									state.velocity.set_recursive_MS_level(recursive_MS_grow_level, \
										copy_weights=copy_weights) #and state is sequence[0]
									log.info("Set recursive MS velocity level of frame %d to %d%s", state.frame, recursive_MS_grow_level,", copy weights from previous level." if copy_weights else ".")
									copied_weights = copy_weights or copied_weights
					curr_vel_shape = vel_shape
					
					if not setup.data.randomize>0: #randomized data is loaded after calling scale_density
						raise NotImplementedError
						log.debug("Rescaling targets from base")
						with profiler.sample('Rescale targets'):
							for state in scale_sequence:
								state.base_targets_raw.resize(curr_cam_res[1:])
								state.base_targets.resize(curr_cam_res[1:])
								state.base_bkgs.resize(curr_cam_res[1:])
								if state.has_masks:
									state.base_masks.resize(curr_cam_res[1:])
					return True
				else:
					return False
				#END if rescale velocity
			
			
			def render_sequence_val(sequence, vel_pad, it):
				log.debug("Render validation images for sequence, iteration %d", it)
				with profiler.sample('render validation'):
					for state in sequence:
						log.debug("Render density validation frame %d", state.frame)
						dens_transform = state.get_density_transform()
						val_imgs = renderer.render_density_SDF_switch(dens_transform, lights, val_cameras)
						renderer.write_images_batch_views(val_imgs, 'val_img_b{batch:04d}_cam{view:02d}_{idx:04d}', base_path=state.data_path, frame_idx=it, image_format='PNG')
						
						slc = dens_transform.data[...,dens_transform.grid_size[-1]//2,:]
						slc = _slice_single_channel_color_transfer(slc)
						renderer.write_images_batch_views([slc], 'val_slc_b{batch:04d}_camX{view:02d}_{idx:04d}', base_path=state.data_path, frame_idx=it, image_format="EXR")
						
						#render_slices(dens_transform.data, ["X"], state.data_path, name_pre="val_slc", format="EXR", normalize=False, slice_indices=[dens_transform.grid_size[-1]//2])
						
						vel_transform = state.get_velocity_transform()
						vel_scale = vel_transform.cell_size_world().value
						log.debug("Render velocity validation frame %d with scale %s", state.frame, vel_scale)
						#sim_transform.set_data(vel_pad)
						vel_centered = state.velocity.centered() * get_vel_scale_for_render(setup, vel_transform)#vel_scale/float(setup.data.step)*setup.rendering.velocity_scale #
						val_imgs = vel_renderer.render_density(vel_transform, [tf.abs(vel_centered)], val_cameras)
						renderer.write_images_batch_views(val_imgs, 'val_velA_b{batch:04d}_cam{view:02d}_{idx:04d}', base_path=state.data_path, frame_idx=it, image_format='EXR')
					#	val_imgs = vel_renderer.render_density(vel_transform, [tf.maximum(vel_centered, 0)], val_cameras)
					#	vel_renderer.write_images([tf.concat(val_imgs, axis=0)], ['val_velP_cam{}_{:04d}'], base_path=state.data_path, use_batch_id=True, frame_id=it, format='PNG')
					#	val_imgs = vel_renderer.render_density(vel_transform, [tf.maximum(-vel_centered, 0)], val_cameras)
					#	vel_renderer.write_images([tf.concat(val_imgs, axis=0)], ['val_velN_cam{}_{:04d}'], base_path=state.data_path, use_batch_id=True, frame_id=it, format='PNG')
						
						if True: #Debug
							# render hull
							pass
			
			def render_sequence_val_DEBUG(sequence, vel_pad, it):
				log.debug("Render validation images for sequence, iteration %d", it)
				with profiler.sample('render validation'):
					for state in sequence:
						log.debug("Render density validation frame %d", state.frame)
						dens_transform = state.get_density_transform()
						val_imgs = renderer.render_density_SDF_switch(dens_transform, lights, val_cameras)
						val_imgs = [_.numpy() for _ in val_imgs]
						#renderer.write_images_batch_views(val_imgs, 'val_img_b{batch:04d}_cam{view:02d}_{idx:04d}', base_path=state.data_path, frame_idx=it, image_format='PNG')
						
						vel_transform = state.get_velocity_transform()
						vel_scale = vel_transform.cell_size_world().value
						log.debug("Render velocity validation frame %d with scale %s", state.frame, vel_scale)
						vel_centered = state.velocity.centered() * get_vel_scale_for_render(setup, vel_transform)
						val_imgs = vel_renderer.render_density(vel_transform, [tf.abs(vel_centered)], val_cameras)
						val_imgs = [_.numpy() for _ in val_imgs]
						#renderer.write_images_batch_views(val_imgs, 'val_velA_b{batch:04d}_cam{view:02d}_{idx:04d}', base_path=state.data_path, frame_idx=it, image_format='PNG')
			
			
			# print growing stats
			def log_growth(tar_shape, intervals, factor, max_iter, name):
				s = "Growing {}: {:d} steps with factor {:f}".format(name, len(intervals)+1, factor)
				abs_intervals = abs_grow_intervals(intervals, max_iter)
				if abs_intervals[-1][0]>abs_intervals[-1][1]:
					log.warning("Insufficient iterations for all grow intervals")
				for interval in abs_intervals:
					shape = current_grow_shape(tar_shape, interval[0], factor, intervals)
					s += "\n\t[{:d},{:d}] {}".format(interval[0], interval[1]-1, shape)
				log.info(s)
			if setup.training.density.pre_opt.first.iterations>0:
				if setup.training.density.pre_opt.first.grow.intervals:
					log_growth(main_opt_start_vel_shape, setup.training.density.pre_opt.first.grow.intervals, setup.training.density.pre_opt.first.grow.factor, setup.training.density.pre_opt.first.iterations, 'pre-opt density')
			if setup.training.velocity.pre_opt.first.iterations>0:
				if setup.training.velocity.pre_opt.first.grow.intervals:
					log_growth(main_opt_start_vel_shape, setup.training.velocity.pre_opt.first.grow.intervals, setup.training.velocity.pre_opt.first.grow.factor, setup.training.velocity.pre_opt.first.iterations, 'pre-opt velocity')
			if setup.training.density.grow.intervals:
				log_growth(base_shape, setup.training.density.grow.intervals, setup.training.density.grow.factor, setup.training.iterations, 'density')
			if setup.training.velocity.grow.intervals:
				log_growth(base_shape, setup.training.velocity.grow.intervals, setup.training.velocity.grow.factor, setup.training.iterations, 'velocity')
			
			loss_schedules = LossSchedules( \
				density_target =		make_schedule(setup.training.density.preprocessed_target_loss), 
				density_target_raw =	make_schedule(setup.training.density.raw_target_loss), 
				density_target_vol =	make_schedule(setup.training.density.volume_target_loss), 
				density_proxy_vol =		make_schedule(setup.training.density.volume_proxy_loss), 
				density_target_depth_smoothness = make_schedule(setup.training.density.target_depth_smoothness_loss), 
				density_negative =		make_schedule(setup.training.density.negative), 
				density_hull =			make_schedule(setup.training.density.hull), 
				density_smoothness =	make_schedule(setup.training.density.smoothness_loss), 
				density_smoothness_2 =	make_schedule(setup.training.density.smoothness_loss_2), 
				density_smoothness_temporal = make_schedule(setup.training.density.temporal_smoothness_loss), 
				density_warp =			make_schedule(setup.training.density.warp_loss), 
				density_disc =			make_schedule(setup.training.density.discriminator_loss), 
				density_center =		make_schedule(setup.training.density.center_loss), 
				SDF_target_pos =		make_schedule(setup.training.density.SDF_pos_loss), 
				
				velocity_target_vol =	make_schedule(setup.training.velocity.volume_target_loss), 
				velocity_warp_dens =	make_schedule(setup.training.velocity.density_warp_loss), 
				velocity_warp_dens_proxy =	make_schedule(setup.training.velocity.density_proxy_warp_loss), 
				velocity_warp_dens_target =	make_schedule(setup.training.velocity.density_target_warp_loss), 
				velocity_warp_vel =		make_schedule(setup.training.velocity.velocity_warp_loss), 
				velocity_divergence =	make_schedule(setup.training.velocity.divergence_loss), 
				velocity_smoothness =	make_schedule(setup.training.velocity.smoothness_loss), 
				velocity_cossim =		make_schedule(setup.training.velocity.cossim_loss), 
				velocity_magnitude =	make_schedule(setup.training.velocity.magnitude_loss), 
				velocity_CFLcond =		make_schedule(setup.training.velocity.CFL_loss), 
				velocity_MS_coherence =	make_schedule(setup.training.velocity.MS_coherence_loss), 
				
				density_lr =			make_schedule(setup.training.density.learning_rate), 
				light_lr =				make_schedule(setup.training.light.learning_rate), 
				velocity_lr =			make_schedule(setup.training.velocity.learning_rate), 
				discriminator_lr =		make_schedule(setup.training.discriminator.learning_rate), 
				view_encoder_regularization = make_schedule(setup.training.density.regularization), 
				density_decoder_regularization = make_schedule(setup.training.density.regularization), 
				velocity_decoder_regularization = make_schedule(setup.training.velocity.regularization), 
				discriminator_regularization = make_schedule(setup.training.discriminator.regularization), 
				
				velocity_warp_dens_MS_weighting =		make_schedule(setup.training.velocity.density_warp_loss_MS_weighting), 
				velocity_warp_dens_tar_MS_weighting =	make_schedule(setup.training.velocity.density_target_warp_loss_MS_weighting), 
				velocity_divergence_MS_weighting =		make_schedule(setup.training.velocity.divergence_loss_MS_weighting), 
				velocity_CFLcond_MS_weighting =			make_schedule(setup.training.velocity.CFL_loss_MS_weighting), 
				velocity_MS_coherence_MS_weighting =	make_schedule(setup.training.velocity.MS_coherence_loss_MS_weighting)
				
				)
			
			
			light_lr = tf.Variable(initial_value=scalar_schedule(setup.training.light.learning_rate, 0), dtype=tf.float32, name='light_lr', trainable=False)
			light_optimizer = tf.train.AdamOptimizer(light_lr, beta1=setup.training.light.optim_beta)
			dens_lr = tf.Variable(initial_value=scalar_schedule(setup.training.density.learning_rate, 0), dtype=tf.float32, name='density_lr', trainable=False)
			dens_optimizer = tf.train.AdamOptimizer(dens_lr, beta1=setup.training.density.optim_beta) #, epsilon=1e-3
			vel_lr = tf.Variable(initial_value=scalar_schedule(setup.training.velocity.learning_rate, 0), dtype=tf.float32, name='velocity_lr', trainable=False)
			vel_optimizer = tf.train.AdamOptimizer(vel_lr, beta1=setup.training.velocity.optim_beta)
			disc_lr = tf.Variable(initial_value=scalar_schedule(setup.training.discriminator.learning_rate, 0), dtype=tf.float32, name='discriminator_lr', trainable=False)
			disc_optimizer = tf.train.AdamOptimizer(disc_lr, beta1=setup.training.discriminator.optim_beta)
			
			# growing unet
			grow_lifting_fade_layers = setup.training.density.grow_lifting_skip is not None
			if grow_lifting_fade_layers:
				num_lifting_layers = len(setup.training.density.grow_lifting_skip)
				assert setup.training.density.grow_lifting_train is not None
				assert len(setup.training.density.grow_lifting_train)==num_lifting_layers
				assert setup.training.density.grow_lifting_lr is not None
				assert len(setup.training.density.grow_lifting_lr)==num_lifting_layers
				assert lifting_network_model.num_levels==num_lifting_layers
				
				grow_lifting_skip_schedules = [make_schedule(setup.training.density.grow_lifting_skip[level]) for level in range(num_lifting_layers)]
				grow_lifting_train_schedules = [make_schedule(setup.training.density.grow_lifting_train[level]) for level in range(num_lifting_layers)]
				grow_lifting_lr_schedules = [make_schedule(setup.training.density.grow_lifting_lr[level]) for level in range(num_lifting_layers)]
				
				for level in range(num_lifting_layers):
					plot_schedule(setup.training.density.grow_lifting_lr[level], setup.training.iterations, os.path.join(setup.paths.config, 'lift_lr_%d.png'%level), 'Lifting %d LR'%level)
					plot_schedule(setup.training.density.grow_lifting_skip[level], setup.training.iterations, os.path.join(setup.paths.config, 'lift_skip_%d.png'%level), 'Lifting %d skip'%level)
				
				grow_lifting_lr = [tf.Variable(initial_value=s(0), dtype=tf.float32, name='light_lr', trainable=False) for s in grow_lifting_lr_schedules]
				grow_lifting_optimizers = [tf.train.AdamOptimizer(lr, beta1=setup.training.density.optim_beta) for level, lr in enumerate(grow_lifting_lr)]
			
			grow_lifting_fade_residual = setup.training.density.grow_lifting_residual is not None
			if grow_lifting_fade_residual:
				num_lifting_layers = len(setup.training.density.grow_lifting_residual)
				assert lifting_network_model is not None
				assert lifting_network_model.num_levels==num_lifting_layers
				assert lifting_network_model.output_mode=="RESIDUAL_WEIGHTED"
				
				grow_lifting_residual_schedules = [make_schedule(setup.training.density.grow_lifting_residual[level]) for level in range(num_lifting_layers)]
				for level in range(num_lifting_layers):
					plot_schedule(setup.training.density.grow_lifting_residual[level], setup.training.iterations, os.path.join(setup.paths.config, 'lift_residual_%d.png'%level), 'Lifting %d residual'%level)
			
			grow_volenc_fade_residual = setup.training.density.grow_volenc_residual is not None
			if grow_volenc_fade_residual:
				num_lifting_layers = len(setup.training.density.grow_volenc_residual)
				assert volume_encoder_model is not None
				assert volume_encoder_model.num_levels==num_lifting_layers
				assert volume_encoder_model.output_mode=="RESIDUAL_WEIGHTED"
				
				grow_volenc_residual_schedules = [make_schedule(setup.training.density.grow_volenc_residual[level]) for level in range(num_lifting_layers)]
				for level in range(num_lifting_layers):
					plot_schedule(setup.training.density.grow_volenc_residual[level], setup.training.iterations, os.path.join(setup.paths.config, 'volenc_residual_%d.png'%level), 'VolEnc %d residual'%level)
			
			grow_vel_MS_residual = setup.training.velocity.decoder.recursive_MS_residual_weight is not None
			if grow_vel_MS_residual:
				assert get_max_recursive_MS_grow_levels(setup.training.velocity.decoder)==len(setup.training.velocity.decoder.recursive_MS_residual_weight)
				
				grow_vel_MS_residual_schedules = [make_schedule(setup.training.velocity.decoder.recursive_MS_residual_weight[level]) for level in range(num_lifting_layers)]
				for level in range(num_lifting_layers):
					plot_schedule(setup.training.velocity.decoder.recursive_MS_residual_weight[level], setup.training.iterations, os.path.join(setup.paths.config, 'vel_MS_residual_%d.png'%level), 'Vel MS %d residual'%level)
			
			opt_ckpt = tf.train.Checkpoint(dens_optimizer=dens_optimizer, vel_optimizer=vel_optimizer, disc_optimizer=disc_optimizer)
			
			main_ctx = OptimizationContext(setup=setup, iteration=0, loss_schedules=loss_schedules, \
				rendering_context=main_render_ctx, vel_scale=[1,1,1], warp_order=setup.training.velocity.warp_order, dt=1.0, buoyancy=buoyancy, \
				dens_warp_clamp=setup.training.density.warp_clamp, vel_warp_clamp=setup.training.velocity.warp_clamp, \
				density_optimizer=dens_optimizer, density_lr=dens_lr, light_optimizer=light_optimizer, light_lr=light_lr, \
				velocity_optimizer=vel_optimizer, velocity_lr=vel_lr, \
				frame=None, tf_summary=summary, summary_interval=25, summary_pre=None, profiler=profiler,
				light_var_list=light_var_list, allow_MS_losses=setup.training.allow_MS_losses, norm_spatial_dims=True)
			
			main_ctx.set_loss_func("density/target", setup.training.density.error_functions.preprocessed_target_loss)
			main_ctx.set_loss_func("density/target_raw", setup.training.density.error_functions.raw_target_loss)
			main_ctx.set_loss_func("density/target_vol", setup.training.density.error_functions.volume_target_loss)
			main_ctx.set_loss_func("density/proxy_vol", setup.training.density.error_functions.volume_proxy_loss)
			main_ctx.set_loss_func("density/target_depth_smooth", setup.training.density.error_functions.target_depth_smoothness_loss)
			main_ctx.set_loss_func("density/hull", setup.training.density.error_functions.hull)
			main_ctx.set_loss_func("density/negative", setup.training.density.error_functions.negative)
			main_ctx.set_loss_func("density/edge", setup.training.density.error_functions.smoothness_loss)
			main_ctx.set_loss_func("density/smooth", setup.training.density.error_functions.smoothness_loss_2)
			main_ctx.set_loss_func("density/smooth-temp", setup.training.density.error_functions.temporal_smoothness_loss)
			main_ctx.set_loss_func("density/warp", setup.training.density.error_functions.warp_loss)
			main_ctx.set_loss_func("density/center", setup.training.density.error_functions.center_loss)
			main_ctx.set_loss_func("density/target_pos", setup.training.density.error_functions.SDF_pos_loss)
			
			main_ctx.set_loss_func("velocity/target_vol", setup.training.velocity.error_functions.volume_target_loss)
			main_ctx.set_loss_func("velocity/density_warp", setup.training.velocity.error_functions.density_warp_loss)
			main_ctx.set_loss_func("velocity/densProxy_warp", setup.training.velocity.error_functions.density_proxy_warp_loss)
			main_ctx.set_loss_func("velocity/densTar_warp", setup.training.velocity.error_functions.density_target_warp_loss)
			main_ctx.set_loss_func("velocity/velocity_warp", setup.training.velocity.error_functions.velocity_warp_loss)
			main_ctx.set_loss_func("velocity/divergence", setup.training.velocity.error_functions.divergence_loss)
			main_ctx.set_loss_func("velocity/magnitude", setup.training.velocity.error_functions.magnitude_loss)
			main_ctx.set_loss_func("velocity/CFL", setup.training.velocity.error_functions.CFL_loss)
			main_ctx.set_loss_func("velocity/MS_coherence", setup.training.velocity.error_functions.MS_coherence_loss)
			
			#gradient warping:
			main_ctx.update_first_dens_only =	make_schedule(setup.training.density.warp_gradients.update_first_only)
			main_ctx.warp_dens_grads =			make_schedule(setup.training.density.warp_gradients.active)
			main_ctx.warp_dens_grads_decay =	make_schedule(setup.training.density.warp_gradients.decay)
			main_ctx.warp_vel_grads =			make_schedule(setup.training.velocity.warp_gradients.active)
			main_ctx.warp_vel_grads_decay =		make_schedule(setup.training.velocity.warp_gradients.decay)
			main_ctx.custom_dens_grads_weight =	make_schedule(setup.training.density.warp_gradients.weight)
			main_ctx.custom_vel_grads_weight =	make_schedule(setup.training.velocity.warp_gradients.weight)
			
			main_ctx.target_weights = view_interpolation_target_weights
			log.info("Target weights: %s", view_interpolation_target_weights)
			
			sF_render_ctx = copy.copy(main_render_ctx)
			sF_render_ctx.cameras = None #scalarFlow_cameras
			opt_ctx = copy.copy(main_ctx)
			opt_ctx.render_ctx = sF_render_ctx
			
			if setup.training.density.scale_render_grads_sharpness>0.0:
				log.info("Scaling density render gradients with exisiting density distribution.")
				opt_ctx.add_render_op('DENSITY', opt_ctx.RO_grid_dens_grad_scale(weight=1, sharpness=setup.training.density.scale_render_grads_sharpness, eps=1e-5))
			
			if setup.training.discriminator.active:
				disc_render_ctx = copy.copy(main_render_ctx)
				disc_render_ctx.cameras = disc_cameras
				#log.info("Disc cam jitter: %s", [_.jitter for _ in disc_render_ctx.cameras])
				#log.warning("Full discriminator in/out debugging enabled!")
				disc_debug_path = os.path.join(setup.paths.data, 'disc_debug')
				os.makedirs(disc_debug_path)
				disc_ctx = DiscriminatorContext(ctx=opt_ctx, model=disc_model, rendering_context=disc_render_ctx, real_data=disc_real_data, \
					loss_type=setup.training.discriminator.loss_type, optimizer=disc_optimizer, learning_rate=disc_lr, \
					crop_size=disc_in_shape[:-1], scale_range=setup.data.discriminator.scale_range, rotation_mode=setup.data.discriminator.rotation_mode, \
					check_input=DiscriminatorContext.CHECK_INPUT_RAISE_NOTFINITE | DiscriminatorContext.CHECK_INPUT_CHECK_NOTFINITE | DiscriminatorContext.CHECK_INPUT_CLAMP | \
					(DiscriminatorContext.CHECK_INPUT_SIZE if setup.training.discriminator.use_fc is not None else 0x0), \
					check_info_path=disc_debug_path, resource_device=data_device, \
					scale_samples_to_input_resolution=setup.data.discriminator.scale_input_to_crop, \
					use_temporal_input=setup.training.discriminator.temporal_input.active, temporal_input_steps=disc_input_steps, \
					cam_x_range=[-27,-7] if setup.data.discriminator.density_type=="SF" else [-10,10]) #SF cams look up about 17 deg
				#disc_ctx.train = train_disc
				if make_disc_dataset and disc_dump_samples:
					disc_ctx.dump_path = os.path.join(setup.paths.data, 'disc_samples')
					log.warning("Dumping ALL discriminator samples to %s.", disc_ctx.dump_path)
					os.makedirs(disc_ctx.dump_path)
				log.info("Discriminator input shape: %s, res: %s", disc_ctx.model.input_shape, disc_ctx.input_res)
			else:
				disc_ctx = DiscriminatorContext(opt_ctx, None, main_render_ctx, None, "SGAN", None, disc_lr)
				disc_ctx.train = False
			
			dump_samples = False
			val_out_step = setup.validation.output_interval
			out_step = setup.training.summary_interval
			loss_summary = []
			
			class StopTraining(Exception):
				pass
			
			def check_loss_summary(loss_summaries, total_losses, it, gradients=None, grad_max=None):
				# check losses and gradients for NaN/Inf
				for f, f_item in loss_summaries.items():
					for k, k_item in f_item.items():
						if not np.all(np.isfinite(k_item)):
							raise ValueError("Loss summary {} of frame {} is not finite.".format(k,f))
				if total_losses and not np.all(np.isfinite(total_losses)):
					raise ValueError("Combined losses are not finite.".format(k,f))
				if gradients is not None:
					for f, f_item in gradients.items():
						for k, k_item in f_item.items():
							if not np.all(np.isfinite(k_item)):
								raise ValueError("Gradient summary {} of frame {} is not finite.".format(k,f))
							if grad_max is not None:
								if "density/light" in k or "velocity/buoyancy" in k or "-v" in k:
									continue # gradients of gloabl scalars are higher
								if np.any(np.greater(k_item, grad_max)):
									#raise ValueError("Gradient summary {} of frame {} is greater than {}.".format(k,f, grad_max))
									log.warning("Gradient summary {} of frame {} is greater than {}.".format(k,f, grad_max))
			
			def get_total_scaled_loss(loss_summaries):
				return sum(loss[-3] for f in loss_summaries for n, loss in loss_summaries[f].items())
			
			def get_loss_scale(loss_summaries, loss_frames, loss_name):
				for idx in range(len(loss_frames)):
					if loss_name in loss_summaries[loss_frames[idx]]:
						return loss_summaries[loss_frames[idx]][loss_name][-1]
				return 0.0
				
			
			def print_loss_summary(loss_summaries, total_losses, start_time, last_time, it, iterations, gradients=None, regularization_stats=None):
				'''
				numpy scalars:
					loss_summaries: {<frame>: {<name/key>: (scaled, raw, scale), ...}, ...}
					gradients: {<frame>: {<name/key>: [grad, ...], ...}, ...}
				'''
				log.info('RAM: current: %d MiB', psutil.Process(os.getpid()).memory_info().rss/(1024*1024))
				log.info('GPU mem: current: %d MiB, max: %d MiB, limit: %d MiB', \
					tf.contrib.memory_stats.BytesInUse().numpy().tolist()/(1024*1024), \
					tf.contrib.memory_stats.MaxBytesInUse().numpy().tolist()/(1024*1024), \
					tf.contrib.memory_stats.BytesLimit().numpy().tolist()/(1024*1024))
				s = ["--- Loss Summary ---\n"]
				now = time.time()
				avg = (now-start_time)/max(1,it)
				avg_last = (now-last_time)/out_step
				s.append('Timing: elapsed {}, avg/step: total {}, last {}, remaining: total {}, last {}\n'.format(format_time(now-start_time), format_time(avg), format_time(avg_last), format_time(avg*(iterations-it)), format_time(avg_last*(iterations-it))))
				s.append('{:26}(x{:>11}): {:>11}({:>11})| ...\n'.format('Last losses', 'Scale', 'Scaled', 'Raw'))
				loss_names = sorted({n for f in loss_summaries for n in loss_summaries[f]})
				loss_frames = sorted((f for f in loss_summaries))
				#loss_scales = [loss_summaries[loss_frames[0]][k][-1] for k in loss_names]
				loss_scales = [get_loss_scale(loss_summaries, loss_frames, k) for k in loss_names]
				for key, scale in zip(loss_names, loss_scales):
					s.append('{:<26}(x{: 10.04e}):'.format(key, scale))
				#	loss_values = active_losses[key]['frames']
					for f in loss_frames:
						if key in loss_summaries[f]:
							s.append(' {: 10.04e}({: 10.04e})'.format(loss_summaries[f][key][-3], loss_summaries[f][key][-2]))
						else:
							s.append(' {:>11}({:>11})'.format('N/A', 'N/A'))
						s.append('|')
					s.append('\n')
				if gradients is not None:
					s.append('{:26}: {:>11}| ...\n'.format("Per-loss volume gradients", "mean-abs"))
					grad_names = sorted({n for f in gradients for n in gradients[f]})
					grad_frames = sorted((f for f in gradients))
					for key in grad_names:
						s.append('{:<26}:'.format(key))
						#grad_values = active_losses[key]['frames_grad']
						for f in grad_frames:
							if key in gradients[f]:
								s.append(','.join([' {: 10.04e}']*len(gradients[f][key])).format(*gradients[f][key]))
							else:
								s.append(' {:>11}'.format('N/A'))
							s.append('|')
						s.append('\n')
				s.append('total scaled loss (dens, vel):')
				for total_loss in total_losses:
					s.append(' ({: 10.04e},{: 10.04e}),'.format(*total_loss))
				s.append('\n')
				if regularization_stats is not None:
					s.append('-- Regularization Stats --\n')
					for network_name, stats in regularization_stats.items():
						s.append('\t- {} (count: {}, weight: {}, applied: {}): mean, max -\n'.format(network_name, stats["Gradient count"], stats["weight"], stats["applied"]))
						s.append('Weights:   ')
						for vmean, vmax in stats["Weights"]:
							s.append('{: 10.04e},{: 10.04e}|'.format(vmean,vmax))
						s.append('\n')
						s.append('Loss grad: '.format())
						for vmean, vmax in stats["Loss gradients"]:
							s.append('{: 10.04e},{: 10.04e}|'.format(vmean,vmax))
						s.append('\n')
						s.append('Reg grad:  '.format())
						for vmean, vmax in stats["Regularization gradients"]:
							s.append('{: 10.04e},{: 10.04e}|'.format(vmean,vmax))
						s.append('\n')
				s.append('lr: dens {: 10.04e}, vel {: 10.04e}'.format(dens_lr.numpy(), vel_lr.numpy()))
				log.info(''.join(s))
			
			def print_disc_summary(disc_ctx, disc_loss):#_real, disc_scores_real, disc_loss_fake, disc_scores_fake):
				loss_summaries = [disc_ctx.opt_ctx.pop_loss_summary()]
				active_losses = {}
				f = 0
				for summ in loss_summaries:
					for k, e in summ.items():
						if k not in active_losses:
							active_losses[k] = {'frames':{}, 'scale':e[-1] if e[-1] is not None else 1.0}
						active_losses[k]['frames'][f] = (e[-3], e[-2] if e[-2] is not None else e[-3])
					f +=1
				s = ["--- Disc Summary ---\n"]
				s.append('{:26}(x{:>9}): {:>11}({:>11}), ...\n'.format('Last losses', 'Scale', 'Scaled', 'Raw'))
				for key in sorted(active_losses.keys()):
					s.append('{:<26}(x{: 9.06f}):'.format(key, active_losses[key]['scale']))
					loss_values = active_losses[key]['frames']
					for i in range(f):
						if i in loss_values:
							s.append(' {: 10.04e}({: 10.04e}),'.format(*loss_values[i]))
						else:
							s.append(' {:>11}({:>11}),'.format('N/A', 'N/A'))
					s.append('\n')
				if len(disc_loss)==4:
					s.append('Total loss (scores): real {:.06f} ({}), fake {:.06f} ({})\n'.format(disc_loss[0], disc_loss[2], disc_loss[1], disc_loss[3]))
				elif len(disc_loss)==2:
					s.append('Total loss (scores): {:.06f} ({})\n'.format(*disc_loss))
				s.append('lr: {:.08f}'.format(disc_ctx.lr.numpy()))
				if setup.training.discriminator.history.samples>0:
					s.append(', history size: {}'.format(len(disc_ctx.history)))
				log.info(''.join(s))
			
			max_ckp = 4
			next_ckp = 0
			def save_checkpoint(name=None):
				if name is None:
					global next_ckp
					name = str(next_ckp)
					next_ckp = (next_ckp+1)%max_ckp
				log.info("Save checkpoint '%s'", name)
				sequence.save()
				if setup.training.discriminator.active:
					save_NNmodel(disc_ctx.model, 'disc_ckp', setup.paths.data)
					#if setup.training.discriminator.history.save:
					#	disc_ctx.history.serialize(setup.paths.data)
				if (setup.training.density.decoder.active or setup.training.velocity.decoder.active) and target_encoder_model is not None:
					save_NNmodel(target_encoder_model, 'target_encoder_ckp-%s'%name, setup.paths.data)
				if lifting_network_model is not None:
					save_NNmodel(lifting_network_model, 'lifting_network_ckp-%s'%name, setup.paths.data)
				if setup.training.volume_encoder.active and volume_encoder_model is not None:
					save_NNmodel(volume_encoder_model, 'volume_encoder_ckp-%s'%name, setup.paths.data)
				if setup.training.frame_merge_network.active and frame_merge_network_model is not None:
					save_NNmodel(frame_merge_network_model, 'frame_merge_network_ckp-%s'%name, setup.paths.data)
				if setup.training.density.decoder.active and density_decoder_model is not None:
					save_NNmodel(density_decoder_model, 'density_decoder_ckp-%s'%name, setup.paths.data)
				if setup.training.velocity.decoder.active and velocity_decoder_model is not None:
					save_NNmodel(velocity_decoder_model, 'velocity_decoder_ckp-%s'%name, setup.paths.data)
				if setup.training.velocity.decoder.active and vel_input_encoder_model is not None:
					save_NNmodel(vel_input_encoder_model, 'velocity_input_encoder_ckp-%s'%name, setup.paths.data)
				if setup.training.velocity.decoder.active and vel_downscale_encoder_model is not None:
					save_NNmodel(vel_downscale_encoder_model, 'velocity_downscale_encoder_ckp-%s'%name, setup.paths.data)
				
			
			
			def dump_inputs(sequence, idx=0):
				for state in sequence:
					state.base_targets_raw.save_scaled(renderer, state.data_path, "PNG", name="dump_targetraw_%d"%(idx,))
					state.base_targets.save_scaled(renderer, state.data_path, "PNG", name="dump_target_%d"%(idx,))
					state.base_bkgs.save_scaled(renderer, state.data_path, "PNG", name="dump_bkg_%d"%(idx,))
					if state.has_masks:
						state.base_masks.save_scaled(renderer, state.data_path, "PNG", name="dump_mask_%d"%(idx,))
			
			def dump_vel_input_features(sequence, idx=0):
				for state in sequence:
					vel_transform = state.get_velocity_transform()
					vel_inp = state.get_volume_features(setup.training.velocity.decoder.type_input_features)
					shape = GridShape.from_tensor(vel_inp)
					for vel_inp_channel in tf.split(vel_inp, shape.c, axis=-1):
						val_imgs = vel_renderer.render_density(vel_transform, [tf.maximum(vel_inp_channel, 0)], val_cameras)
						#vel_renderer.write_images([tf.concat(val_imgs, axis=0)], ['P_inp_velP_cam{}'], base_path=state.data_path, use_batch_id=True, format='PNG')
						vel_renderer.write_images_batch_views(val_imgs, 'P_trainP_velInpP_{batch:04d}_cam{view:02d}_{idx:04d}', base_path=state.data_path, frame_idx=idx, image_format='PNG')
			
			def dump_dens_input_features(sequence, idx=0):
				for state in sequence:
					dens_transform = state.get_density_transform()
					dens_inp = state.get_volume_features(setup.training.density.decoder.type_input_features)
					shape = GridShape.from_tensor(dens_inp)
					for i, dens_inp_channel in enumerate(tf.split(dens_inp, shape.c, axis=-1)):
						val_imgs = vel_renderer.render_density(dens_transform, [tf.maximum(dens_inp_channel, 0)], val_cameras)
						#vel_renderer.write_images([tf.concat(val_imgs, axis=0)], ['P_inp_velP_cam{}'], base_path=state.data_path, use_batch_id=True, format='PNG')
						vel_renderer.write_images_batch_views(val_imgs, 'P_trainP_velInpP{:02d}'.format(i)+'_{batch:04d}_cam{view:02d}_{idx:04d}', base_path=state.data_path, frame_idx=idx, image_format='PNG')
			
			
			# scene serialization
			scene = {
				"cameras":cameras,
				"sFcameras":scalarFlow_cameras,
				"lighting":lights,
				"objects":[sim_transform],
				"pre_opt_vel_shape": main_opt_start_vel_shape,
				"pre_opt_dens_shape": main_opt_start_dens_shape,
				"vel_shape": base_shape,
				"vel_shape": base_shape,
			}
			scene_file = os.path.join(setup.paths.config, "scene.json")
			#log.debug("Serializing scene to %s ...", scene_file)
			with open(scene_file, "w") as file:
				try:
					json.dump(scene, file, default=tf_to_dict, sort_keys=True)#, indent=2)
				except:
					log.exception("Scene serialization failed.")
		
		except KeyboardInterrupt:
			log.warning("Interrupt during setup.")
			sys.exit(0)
		except:
			log.exception('Exception during setup:')
			sys.exit(1)
		
# --- Optimization ---
		signal.signal(signal.SIGINT, handle_train_interrupt)
		optim_start = time.time()
		try:
			with summary_writer.as_default(), summary.always_record_summaries():
				
				opt_ctx.summary_pre = "Main-Optim"
				#run full-sequence optimization
				log.info('--- Sequence optimization (order: %s) start (%d - %d iterations) ---', setup.training.frame_order, setup.training.start_iteration, setup.training.iterations)
				loss_schedules.set_schedules( \
					density_target =		make_schedule(setup.training.density.preprocessed_target_loss), 
					density_target_raw =	make_schedule(setup.training.density.raw_target_loss), 
					density_target_vol =	make_schedule(setup.training.density.volume_target_loss), 
					density_proxy_vol =		make_schedule(setup.training.density.volume_proxy_loss), 
					density_target_depth_smoothness = make_schedule(setup.training.density.target_depth_smoothness_loss), 
					density_hull =			make_schedule(setup.training.density.hull), 
					density_negative =		make_schedule(setup.training.density.negative), 
					density_smoothness =	make_schedule(setup.training.density.smoothness_loss), 
					density_smoothness_2 =	make_schedule(setup.training.density.smoothness_loss_2), 
					density_smoothness_temporal = make_schedule(setup.training.density.temporal_smoothness_loss), 
					density_warp =			make_schedule(setup.training.density.warp_loss), 
					density_disc =			make_schedule(setup.training.density.discriminator_loss), 
					density_center =		make_schedule(setup.training.density.center_loss), 
					SDF_target_pos =		make_schedule(setup.training.density.SDF_pos_loss), 
					
					velocity_target_vol =	make_schedule(setup.training.velocity.volume_target_loss), 
					velocity_warp_dens =	make_schedule(setup.training.velocity.density_warp_loss), 
					velocity_warp_dens_proxy =	make_schedule(setup.training.velocity.density_proxy_warp_loss), 
					velocity_warp_dens_target =	make_schedule(setup.training.velocity.density_target_warp_loss), 
					velocity_warp_vel =		make_schedule(setup.training.velocity.velocity_warp_loss), 
					velocity_divergence =	make_schedule(setup.training.velocity.divergence_loss), 
					velocity_smoothness =	make_schedule(setup.training.velocity.smoothness_loss), 
					velocity_cossim =		make_schedule(setup.training.velocity.cossim_loss), 
					velocity_magnitude =	make_schedule(setup.training.velocity.magnitude_loss), 
					velocity_CFLcond =		make_schedule(setup.training.velocity.CFL_loss), 
					velocity_MS_coherence =	make_schedule(setup.training.velocity.MS_coherence_loss), 
					
					density_lr =			make_schedule(setup.training.density.learning_rate), 
					velocity_lr =			make_schedule(setup.training.velocity.learning_rate), 
					discriminator_lr =		make_schedule(setup.training.discriminator.learning_rate), 
					density_decoder_train =		make_schedule(setup.training.density.train_decoder), 
					velocity_decoder_train =	make_schedule(setup.training.velocity.train_decoder), 
					frame_encoders_train =		make_schedule(setup.training.train_frame_encoders), 
					
					view_encoder_regularization = make_schedule(setup.training.density.regularization), 
					density_decoder_regularization = make_schedule(setup.training.density.regularization), 
					velocity_decoder_regularization = make_schedule(setup.training.velocity.regularization), 
					discriminator_regularization = make_schedule(setup.training.discriminator.regularization),
					
					velocity_warp_dens_MS_weighting =		make_schedule(setup.training.velocity.density_warp_loss_MS_weighting), 
					velocity_warp_dens_tar_MS_weighting =	make_schedule(setup.training.velocity.density_target_warp_loss_MS_weighting), 
					velocity_divergence_MS_weighting =		make_schedule(setup.training.velocity.divergence_loss_MS_weighting), 
					velocity_CFLcond_MS_weighting =			make_schedule(setup.training.velocity.CFL_loss_MS_weighting), 
					velocity_MS_coherence_MS_weighting =	make_schedule(setup.training.velocity.MS_coherence_loss_MS_weighting),
					
					sequence_length = 		make_schedule(setup.training.sequence_length)
					
					)
				
				velocity_noise_schedule = make_schedule(setup.training.velocity.noise_std)
				def seq_vel_add_noise(opt_ctx, seq):
					vel_noise_std = velocity_noise_schedule(opt_ctx.iteration)
					if opt_ctx.LA(vel_noise_std):
						log.debug("Add noise to sequence velocity: std: %f, it: %d.", vel_noise_std, opt_ctx.iteration) #TODO debug
						for state in seq:
							v = state.velocity
							v.assign_add( \
								x = tf.random.normal([1]+v.x_shape+[1], stddev=vel_noise_std, dtype=tf.float32), \
								y = tf.random.normal([1]+v.y_shape+[1], stddev=vel_noise_std, dtype=tf.float32), \
								z = tf.random.normal([1]+v.z_shape+[1], stddev=vel_noise_std, dtype=tf.float32))
				#def optimize_sequence(opt_ctx, state, iterations, use_vel=False, disc_ctx=None, dens_factor=1.0, dens_intervals=[], vel_factor=1.0, vel_intervals=[], start_iteration=0):
				with profiler.sample('Main Optimization'), tf.device(compute_device):
					def regualrization_gradient_summary(model, weight=1.0, apply=True, stats_dict=None, name="Network"):
						if model is not None:
							if isinstance(model, list):
								for i, m in enumerate(model):
									regualrization_gradient_summary(m, weight, apply, stats_dict, name=name+"_L%d"%(i,))
							elif stats_dict is not None:
								stats_dict[name] = {}
								stats_dict[name]["weight"] = weight
								stats_dict[name]["applied"] = apply
								stats_dict[name]["Weights"] = model.get_weights_summary()
								stats_dict[name]["Gradient count"] = model.num_pending_gradients
								stats_dict[name]["Loss gradients"] = model.get_pending_gradients_summary()
								reg_grads = model.compute_regularization_gradients(weight=weight, add_gradients=apply)
								stats_dict[name]["Regularization gradients"] = [[tf.reduce_mean(_).numpy(), tf.reduce_max(tf.abs(_)).numpy()] for _ in reg_grads]
							elif apply:
								model.compute_regularization_gradients(weight=weight, add_gradients=apply)
					
					def scale_grads_for_smoothing(grads_vars, batch_group_size):
						sf = tf.constant(1/batch_group_size, dtype=tf.float32) #smoothing factor
						scaled_grads_vars = [(g*sf, v) for g, v in grads_vars]
						return scaled_grads_vars
					
					def DEBUG_check_duplicate_vars(grads_vars):
						seen_vars = set()
						duplicates = set()
						for g, v in grads_vars:
							if v in seen_vars:
								duplicates.add(v)
							seen_vars.add(v)
						if len(duplicates)>0:
							raise RuntimeError("%d duplicate variables in grads_vars."%(len(duplicates),))
						
					disc_ctx.train = setup.training.discriminator.train and setup.training.discriminator.active
					inspect_gradients_list = {state.frame:{} for state in sequence}
					if setup.training.density.pre_opt.inspect_gradients==1:
						def ig_func(opt_ctx, gradients, name):
							if name.endswith('_x') or name.endswith('_y') or name.endswith('_z'):
								#velocity gradients are given individually per component with name=loss_name + _(x|y|z)
								c = ['x','y','z'].index(name[-1:])
								name = name[:-2]
								if name not in inspect_gradients_list[opt_ctx.frame]: inspect_gradients_list[opt_ctx.frame][name] = np.asarray([0,0,0], dtype=np.float32)
								inspect_gradients_list[opt_ctx.frame][name][c] = tf.reduce_mean(tf.abs(gradients)).numpy()
							else:
								abs_grad = tf.abs(gradients)
								inspect_gradients_list[opt_ctx.frame][name] = [tf.reduce_mean(abs_grad).numpy(), tf.reduce_max(abs_grad).numpy()]
						iig_func = None
					if setup.training.density.pre_opt.inspect_gradients==2:# or True:
						AABB_corners_WS = []
						for state in sequence:
							dens_transform = state.get_density_transform()
							if state.density.hull is not None:
								AABB_corners_WS += dens_transform.transform_AABB(*hull_AABB_OS(tf.squeeze(state.density.hull, (0,-1))), True)
						if state.density.hull is not None:
							grad_cams = [main_camera.copy_clipped_to_world_coords(AABB_corners_WS)[0]]
						else:
							grad_cams = [main_camera]
						ig_func = lambda opt_ctx, gradients, name: render_gradients(gradients, dens_transform, grad_cams, grad_renderer, \
							path=os.path.join(setup.paths.data, "gradients", "d_f{:04d}_it{:08d}".format(opt_ctx.frame, opt_ctx.iteration)), \
							image_mask=name.replace("/", ".") + "_b{batch:04d}_cam{view:02d}_{idx:04d}", name=name, log=log)
						iig_func = lambda opt_ctx, gradients, name: write_image_gradients(gradients, max_renderer, \
							path=os.path.join(setup.paths.data, "gradients", "d_f{:04d}_it{:08d}".format(opt_ctx.frame, opt_ctx.iteration)), \
							image_mask=name.replace("/", ".") + "_img_cam{:04}", image_neg_mask=name.replace("/", ".") + "_img-neg_cam{:04}")
					last_time = time.time()
					start_time = last_time
					
					warp_sequence = False
					fwd_warp_dens_clamp = setup.training.density.warp_clamp#'NONE'
					warp_sequence_schedule = make_schedule(setup.training.density.main_warp_fwd)
					
					### main opt loop ###
					randomize_sequence_length = setup.training.randomization.sequence_length
					if randomize_sequence_length:
						log.info("Randomizing sequence length during training.")
					last_max_sequence_length = len(sequence)
					def get_train_sequence(sequence, iteration):
						global last_max_sequence_length
						min_length = 2
						max_length = int(loss_schedules.sequence_length(iteration))
						if max_length<1 or max_length>len(sequence):
							max_length = len(sequence)
						if not max_length==last_max_sequence_length:
							log.info("Setting max sequence length from %d to %d in iteration %d", last_max_sequence_length, max_length, iteration)
						last_max_sequence_length = max_length
						
						if randomize_sequence_length and max_length>min_length and not iteration==(setup.training.iterations-1):
							length = np.random.randint(min_length, max_length+1)
							log.debug("Iteration %d: sequence length = %d [%d,%d]", iteration, length, min_length, max_length)
						else:
							length = max_length
						s = sequence.get_sub_sequence(length)
						
						return s
					
					# batch size limits
					class BatchSizeHandler:
						def __init__(self, base_bs, base_grp, max_grp=None, max_res=60, verbose=False):
							self._base_bs = base_bs
							self._curr_bs = base_bs
							self._base_grp = base_grp
							self._max_grp = max_grp
							self._curr_grp = abs(base_grp)
							self._max_res = max_res
							self._verbose = verbose
						def get(self):
							return self._curr_bs, self._curr_grp
						def _get_max_batch_size(self, res):
							#return max(1, int(math.floor((self._max_res/res)**3)))
							# 10GB GPU-mem, deep resblock models, MS sf 1.4, min res 10
							if res>48: return 1 #48 if sf 2.0, maybe 40 if sf is 1.4
							if res>34: return 2
							if res>24: return 4
							if res>18: return 8
							return 16
						def scale_to(self, res):
							if self._base_grp<0: # override adaptive batch size by using a negative group size
								new_bs = self._base_bs
								new_grp = abs(self._base_grp)
							else:
								max_bs = self._get_max_batch_size(res)
								if max_bs<self._base_bs:
									new_bs = max_bs
									new_grp = int(math.ceil(self._base_grp * (self._base_bs/new_bs)))
									if self._max_grp is not None:
										new_grp = min(new_grp, self._max_grp)
								else:
									new_bs = self._base_bs
									new_grp = self._base_grp
								
							if self._verbose and (not new_bs==self._curr_bs or not new_grp==self._curr_grp):
								log.info("Adjusting batch size to %d, group %d for resolution %d.", new_bs, new_grp, res)
							self._curr_bs = new_bs
							self._curr_grp = new_grp
							return self
					
					batch_size_handler = BatchSizeHandler(batch_size, batch_group_size, verbose=True)
					batch_variable_resolution = grow_handler.is_randomize_shape
					
					total_batch_idx = 0
					log.info("Main loop start.")
					for it in range(setup.training.start_iteration, setup.training.iterations):
						log.debug('Start iteration %d', it)
						#exclude first iteration from measurement as it includes some tf setup time
						if it==1: start_time = time.time()
						with profiler.sample('Optim Step', verbose = False):
							
							density_decoder_model.set_frozen_weights(not loss_schedules.density_decoder_train(it))
							
							freeze_vel_encoders = not loss_schedules.velocity_decoder_train(it)
							velocity_decoder_model.set_frozen_weights(freeze_vel_encoders)
							if isinstance(vel_input_encoder_model, list):
								for m in vel_input_encoder_model: m.set_frozen_weights(freeze_vel_encoders)
							elif vel_input_encoder_model is not None:
								vel_input_encoder_model.set_frozen_weights(freeze_vel_encoders)
							if isinstance(vel_downscale_encoder_model, list):
								for m in vel_downscale_encoder_model: m.set_frozen_weights(freeze_vel_encoders)
							elif vel_downscale_encoder_model is not None:
								vel_downscale_encoder_model.set_frozen_weights(freeze_vel_encoders)
							
							freeze_frame_encoders = not loss_schedules.frame_encoders_train(it)
							if target_encoder_model is not None:
								target_encoder_model.set_frozen_weights(freeze_frame_encoders)
							if lifting_network_model is not None:
								lifting_network_model.set_frozen_weights(freeze_frame_encoders)
							if volume_encoder_model is not None:
								volume_encoder_model.set_frozen_weights(freeze_frame_encoders)
							if frame_merge_network_model is not None:
								frame_merge_network_model.set_frozen_weights(freeze_frame_encoders)
							
							if not batch_variable_resolution and it==(setup.training.iterations-1): #last iteration
								grow_handler.is_randomize_shape = False
								grow_handler.is_iterate_shapes = False
							grow_handler.start_iteration(iteration=it)
							log.debug("GrowHandler shapes: density:%s, camera:%s, velocity:%s, vel-level:%s", \
								grow_handler.get_density_shape(), grow_handler.get_camera_shape(), grow_handler.get_velocity_shape(), grow_handler.get_velocity_MS_scale())
								
							
							curr_batch_size, curr_batch_group_size = batch_size_handler.scale_to(res=min(grow_handler.get_current_max_shape())).get() #sequence[0].transform.grid_size
							target_dataset.set_batch_size(curr_batch_size)
							
							summary_iteration = (it+1)%out_step==0 or (it+1)==setup.training.iterations or stop_training
							
							if grow_lifting_fade_layers:
								# set unet skip connection weights
								for level in range(lifting_network_model.num_levels):
									if level>0: # level 0 has no skip weight
										lifting_network_model.set_skip_merge_weight(grow_lifting_skip_schedules[level](it), level)
									grow_lifting_lr[level].assign(grow_lifting_lr_schedules[level](it))
									if summary_iteration:
										log.info("Grow lifting, it %d, level %d: skip %f, lr %f (%f), train %s, active %s", it, level, grow_lifting_skip_schedules[level](it), grow_lifting_lr_schedules[level](it), grow_lifting_lr[level].numpy(), grow_lifting_train_schedules[level](it), level<=lifting_network_model.get_active_level())
							
							if grow_lifting_fade_residual:
								# set output residual weights
								for level in range(lifting_network_model.num_levels):
									if level>0: # level 0 has no skip weight
										lifting_network_model.set_output_residual_weight(grow_lifting_residual_schedules[level](it), level)
										if summary_iteration:
											log.info("Grow lifting, it %d, level %d: residual %f (%f), active %s", it, level, grow_lifting_residual_schedules[level](it), lifting_network_model.get_output_residual_weight(level), level<=lifting_network_model.get_active_level())
							
							if grow_volenc_fade_residual:
								# set output residual weights
								for level in range(volume_encoder_model.num_levels):
									if level>0: # level 0 has no skip weight
										volume_encoder_model.set_output_residual_weight(grow_volenc_residual_schedules[level](it), level)
										if summary_iteration:
											log.info("Grow volenc, it %d, level %d: residual %f (%f), active %s", it, level, grow_volenc_residual_schedules[level](it), volume_encoder_model.get_output_residual_weight(level), level<=volume_encoder_model.get_active_level())
							
							if grow_vel_MS_residual:
								# set output residual weights
								for level in range(len(grow_vel_MS_residual_schedules)):
									# here level 0 can use residual weight
									grow_vel_MS_residual_weight = grow_vel_MS_residual_schedules[level](it)
									for state in sequence:
										state.velocity.set_residual_weight(level, grow_vel_MS_residual_weight)
									if summary_iteration:
										log.info("Grow vel MS, it %d, level %d: residual %f %s, active %s", it, level, grow_vel_MS_residual_weight, \
											[state.velocity.get_residual_weight(level) for state in sequence], [level<=state.velocity.recursive_MS_current_level for state in sequence])
							
							disc_imgs = []
							for batch_group in range(curr_batch_group_size):
								with profiler.sample('Grad Step'):
									summary_iteration = (batch_group==(curr_batch_group_size-1)) and ((it+1)%out_step==0 or (it+1)==setup.training.iterations or stop_training)
									
									if batch_variable_resolution:
										if it==(setup.training.iterations-1) and (batch_group==(curr_batch_group_size-1)): #last iteration
											grow_handler.is_randomize_shape = False
											grow_handler.is_iterate_shapes = False
										grow_handler.start_iteration(iteration=it)
										log.debug("GrowHandler shapes: density:%s, camera:%s, velocity:%s, vel-level:%s", \
											grow_handler.get_density_shape(), grow_handler.get_camera_shape(), grow_handler.get_velocity_shape(), grow_handler.get_velocity_MS_scale())
										
									if batch_variable_resolution or batch_group==0:
										with profiler.sample('Set resolutions'):
											v_scaled = scale_velocity(sequence, it, setup.training.velocity.grow.factor, setup.training.velocity.grow.scale_magnitude, setup.training.velocity.grow.intervals, base_shape=base_shape, verbose=not batch_variable_resolution)
											d_scaled = scale_density(sequence, it, setup.training.density.grow.factor, setup.training.density.grow.intervals, base_shape=base_shape, save=False, \
												scale_all=((it==0) or (setup.training.randomization.grow_mode=="RAND") or v_scaled), verbose=not batch_variable_resolution)
											grow_velocity_networks(sequence, it)
											set_density_network_level(sequence)
											scale_state_networks(sequence)
									
									sequence.clear_cache()
									if setup.data.randomize>0:
										with profiler.sample('Load targets'):
											target_dataset.step()
											sequence_set_targets(sequence, {state.frame: frame_loadTargets(setup, state.frame, sim_transform, target_dataset) for state in sequence}, \
												set_size=True) # also clears the cache #(setup.training.randomization.grid_size_relative==1)
											sequence_randomize(sequence, randomize_input_views=setup.training.randomization.inputs, \
												randomize_target_views=setup.training.randomization.targets, \
												disable_transform_reset=True)
												#randomize_transform=setup.training.randomization.transform)
										
									
									if setup.training.randomization.grid_size_relative!=1:
										with profiler.sample('Randomize scale'):
											randomize_scale(sequence, min_size_abs=setup.training.randomization.grid_size_min, min_size_rel=setup.training.randomization.grid_size_relative, \
												max_shape=curr_vel_shape, train_cam_res=curr_cam_res, disc_cam_res=None)
									
									
									
									if warp_sequence_schedule(it)!=warp_sequence:
										warp_sequence = warp_sequence_schedule(it)
										if warp_sequence:
											log.info("Density set to forward warped with clamp '%s' in iteration %d.", fwd_warp_dens_clamp, it) #
											log.info("dens shapes %s, vel shapes %s", [_.density.shape for _ in sequence], [_.velocity.centered_shape for _ in sequence])
											if setup.training.density.decoder.active:
												log.info("Set sequence densities for neural globt.")
												sequence.set_density_for_neural_globt(order=opt_ctx.warp_order, dt=opt_ctx.dt, clamp=fwd_warp_dens_clamp, as_var=False, device=resource_device)
											##sequence.densities_advect_fwd(order=opt_ctx.warp_order, dt=opt_ctx.dt, clamp=fwd_warp_dens_clamp)
										else:
											log.info("Density set to per-frame in iteration %d.", it) #
									elif warp_sequence and setup.data.randomize>0:
										log.debug("Warp density forward with clamp '%s' in iteration %d.", fwd_warp_dens_clamp, it) #for testing, remove
										#sequence.densities_advect_fwd(order=opt_ctx.warp_order, dt=opt_ctx.dt, clamp=fwd_warp_dens_clamp)
										
									#sequence.clear_cache()
											
									opt_ctx.start_iteration(it, compute_loss_summary=summary_iteration)
									
									seq_vel_add_noise(opt_ctx, sequence)
									
									train_sequence = get_train_sequence(sequence, it)
									
									if summary_iteration and setup.training.density.pre_opt.inspect_gradients:# or it==99 or it==200 or it==500:
										for g in inspect_gradients_list: inspect_gradients_list[g].clear()
										opt_ctx.set_inspect_gradient(True, ig_func, iig_func)
									if setup.training.frame_order=='FWD-BWD':
										loss_summaries = optStep_sequence(opt_ctx, train_sequence, disc_ctx, disc_samples_list=disc_imgs, order='FWD' if (it%2)==0 else 'BWD')
									elif setup.training.frame_order=='BWD-FWD':
										loss_summaries = optStep_sequence(opt_ctx, train_sequence, disc_ctx, disc_samples_list=disc_imgs, order='BWD' if (it%2)==0 else 'FWD')
									else:
										loss_summaries = optStep_sequence(opt_ctx, train_sequence, disc_ctx, disc_samples_list=disc_imgs, order=setup.training.frame_order)
									
									# Backprop
									for state in train_sequence:
										# only backprop if gradients are needed.
										if state.requires_backprop:
											#log.info("backprop state %d in it %d.", state.frame, it)
											state._compute_input_grads()
										if state.has_density_neural and state.density.requires_backprop:
											#log.info("backprop neural density %d in it %d.", state.frame, it)
											state.density._compute_input_grads()
										if state.has_density_proxy and state.density_proxy.requires_backprop:
											#log.info("backprop density proxy %d in it %d.", state.frame, it)
											state.density_proxy._compute_input_grads()
										if state.has_velocity and state.velocity.requires_backprop:
											#log.info("backprop velocity %d in it %d.", state.frame, it)
											state.velocity._compute_input_grads()
									
									with profiler.sample("regularization"):
										print_regularization_stats = setup.debug.print_weight_grad_stats #True #True
										add_reg_gradients = True
										regularization_stats = {} if summary_iteration and print_regularization_stats else None
										
										regualrization_gradient_summary(model=target_encoder_model, weight=loss_schedules.density_decoder_regularization(it), apply=add_reg_gradients, \
											stats_dict=regularization_stats, name="View Encoder")
										regualrization_gradient_summary(model=lifting_network_model, weight=loss_schedules.density_decoder_regularization(it), apply=add_reg_gradients, \
											stats_dict=regularization_stats, name="Lifting Network")
										regualrization_gradient_summary(model=volume_encoder_model, weight=loss_schedules.density_decoder_regularization(it), apply=add_reg_gradients, \
											stats_dict=regularization_stats, name="Volume Encoder")
										regualrization_gradient_summary(model=frame_merge_network_model, weight=loss_schedules.density_decoder_regularization(it), apply=add_reg_gradients, \
											stats_dict=regularization_stats, name="Frame Merge")
										regualrization_gradient_summary(model=density_decoder_model, weight=loss_schedules.density_decoder_regularization(it), apply=add_reg_gradients, \
											stats_dict=regularization_stats, name="Density")
										regualrization_gradient_summary(model=velocity_decoder_model, weight=loss_schedules.velocity_decoder_regularization(it), apply=add_reg_gradients, \
											stats_dict=regularization_stats, name="Velocity")
										regualrization_gradient_summary(model=vel_input_encoder_model, weight=loss_schedules.velocity_decoder_regularization(it), apply=add_reg_gradients, \
											stats_dict=regularization_stats, name="VelocityInEnc")
										regualrization_gradient_summary(model=vel_downscale_encoder_model, weight=loss_schedules.velocity_decoder_regularization(it), apply=add_reg_gradients, \
											stats_dict=regularization_stats, name="VelocityDownEnc")
									
									
									if setup.debug.target_dump_samples:
										log.warning("Dumping input/target images of iteration %d, batch %d.", it, batch_group)
										dump_inputs(sequence, total_batch_idx)
									total_batch_idx +=1
								# END profiling
							# END batch group
							
							if summary_iteration:
								for state in sequence:
									opt_ctx.frame = state.frame
									if state.has_velocity:
										state.velocity.inspect_output_gradient_stats(opt_ctx)
							opt_ctx.set_inspect_gradient(False)
							
							#if (it+1)%gradient_smoothing_window==0:
							with profiler.sample('Apply gradients'):
								
								grads_vars_growing = []
								if grow_lifting_fade_layers:
									grads_vars_growing = lifting_network_model.get_grads_vars_by_level(keep_gradients=False)
								
								grads_vars = []
								grads_vars_vel = []
								for state in sequence:
									grads_vars.extend(state.get_grads_vars( \
										get_density_gradients=setup.training.density.decoder.active and isinstance(state.density, NeuralDensityGrid), \
										get_velocity_gradients=False, #setup.training.velocity.decoder.active and isinstance(state.velocity, NeuralVelocityGrid),
										keep_gradients=False))
									
									if isinstance(state.velocity, NeuralVelocityGrid):
										grads_vars_vel.extend(state.velocity.get_grads_vars(keep_gradients=False, normalize=randomize_sequence_length))
								
								DEBUG_check_duplicate_vars(grads_vars + grads_vars_vel)
								
								# need to normalize with smoothing window to be consistent with batch size normalization
								# might not be needed with Adam, if batch_group_size stays constant
								if not curr_batch_group_size==1:
									grads_vars_growing = [scale_grads_for_smoothing(gvs, curr_batch_group_size) for gvs in grads_vars_growing]
									grads_vars = scale_grads_for_smoothing(grads_vars, curr_batch_group_size)
									grads_vars_vel = scale_grads_for_smoothing(grads_vars_vel, curr_batch_group_size)
								
								if grads_vars_growing:
									for level, gvs in enumerate(grads_vars_growing):
										if grow_lifting_train_schedules[level](it):
											grow_lifting_optimizers[level].apply_gradients(gvs)
									del grads_vars_growing
								if grads_vars:
									opt_ctx.density_optimizer.apply_gradients(grads_vars)
									del grads_vars
								if grads_vars_vel:
									opt_ctx.velocity_optimizer.apply_gradients(grads_vars_vel)
									del grads_vars_vel
								
								for state in sequence:
									# just to be sure
									# clears e.g. single-frame merge network regualrization gradients that never get applied.
									state.clear_gradients(clear_density_gradients=True, clear_velocity_gradients=True)
							
							
							del train_sequence
							#if randomize_sequence_length:
							sequence.restore_connections()
							
							# DISCRIMINATOR
							disc_ctx.start_iteration(it, compute_loss_summary=summary_iteration)
							disc_ctx.opt_ctx.frame = None
							for disc_step in range(setup.training.discriminator.steps):
								disc_loss = optStep_discriminator(disc_ctx, state=None, additional_fake_samples=disc_imgs) #_real, disc_loss_fake, disc_scores_real, disc_scores_fake
							#END disc training
							del disc_imgs
							
							
						if args.console: # and not args.debug:
							progress = it%out_step+1
							progress_bar(progress,out_step, "{:04d}/{:04d}".format(progress, out_step), length=50)
						
						if summary_iteration:
							log.info('--- Step {:04d}/{:04d} ---'.format(it, setup.training.iterations-1))
							print_loss_summary(loss_summaries, [], start_time, last_time, it, setup.training.iterations, \
								inspect_gradients_list if setup.training.density.pre_opt.inspect_gradients==1 else None, regularization_stats=regularization_stats)
							regularization_stats = None
							log.info("buoyancy: %s", opt_ctx.buoyancy.numpy())
							#d_max, d_min, d_mean = tf_print_stats(state.density.d, 'Density', log=log)
							if setup.training.light.optimize:
								log.info("Light intensities: %s", [_.numpy() for _ in light_var_list])
							if disc_ctx is not None and disc_ctx.train and setup.training.discriminator.start_delay<=it:
								print_disc_summary(disc_ctx, disc_loss)#_real, disc_scores_real, disc_loss_fake, disc_scores_fake)
							last_time = time.time()
							summary_writer.flush()
							try:
								check_loss_summary(loss_summaries, [], it, inspect_gradients_list if setup.training.density.pre_opt.inspect_gradients==1 else None, grad_max=5e-2)
							except ValueError as e:
								log.exception("Invalid loss summary in iteration %d:", it)
								dump_inputs(sequence, it)
								dump_vel_input_features(sequence, it)
								stop_training = True
						
						if val_cameras is not None and (it+1)%val_out_step==0:
							sequence.clear_cache()
							if setup.data.randomize>0:
								sequence_set_targets(sequence, val_sequence)
								sequence_randomize(sequence, randomize_input_views=setup.validation.input_view_mask, disable_transform_reset=True) #reset randomization
							log.info("Render validation views %d in iteration %d", int((it+1)//val_out_step), it)
							#sequence.clear_cache()
							try:
								render_sequence_val(sequence, z, int((it+1)//val_out_step))
							except:
								log.exception('Exception when rendering validation views %d for sequence in iteration %d:', int((it+1)//val_out_step), it)
							#sequence.clear_cache()
							
						
						if setup.training.checkpoint_interval>0 and (it+1)%setup.training.checkpoint_interval==0:
							try:
								save_checkpoint()
							except:
								log.exception("Exception when saving checkpoint in iteration %d:", it+1)
						
						if psutil.Process(os.getpid()).memory_info().rss>max_memory:
							log.error("Current memory exceeds limit, stopping.")
							stop_training = True
						
						if stop_training:
							log.warning('Training stopped after %d iterations, saving state...', it+1)
							raise StopTraining
							break
						#iteration profiler
					#END for it in iterations (training loop)
				#optimization profiler
			#tf summary
			log.debug('Save sequence')
			sequence.save()
			if setup.training.discriminator.active:
				save_NNmodel(disc_ctx.model, 'disc', setup.paths.data)
				if setup.training.discriminator.history.save:
					disc_ctx.history.serialize(setup.paths.data)
			if (setup.training.density.decoder.active or setup.training.velocity.decoder.active) and target_encoder_model is not None:
				save_NNmodel(target_encoder_model, 'target_encoder', setup.paths.data)
			if lifting_network_model is not None:
				save_NNmodel(lifting_network_model, 'lifting_network', setup.paths.data)
			if setup.training.volume_encoder.active and volume_encoder_model is not None:
				save_NNmodel(volume_encoder_model, 'volume_encoder', setup.paths.data)
			if setup.training.frame_merge_network.active and frame_merge_network_model is not None:
				save_NNmodel(frame_merge_network_model, 'frame_merge_network', setup.paths.data)
			if setup.training.density.decoder.active and density_decoder_model is not None:
				save_NNmodel(density_decoder_model, 'density_decoder', setup.paths.data)
			if setup.training.velocity.decoder.active and velocity_decoder_model is not None:
				save_NNmodel(velocity_decoder_model, 'velocity_decoder', setup.paths.data)
			if setup.training.velocity.decoder.active and vel_input_encoder_model is not None:
				save_NNmodel(vel_input_encoder_model, 'velocity_input_encoder', setup.paths.data)
			if setup.training.velocity.decoder.active and vel_downscale_encoder_model is not None:
				save_NNmodel(vel_downscale_encoder_model, 'velocity_downscale_encoder', setup.paths.data)
			
			
		except StopTraining:
			log.warning('Optimization stopped after %s, saving state...', format_time(time.time() - optim_start))
			log.debug('Save sequence')
			sequence.save(suffix="part")
			if setup.training.discriminator.active:
				save_NNmodel(disc_ctx.model, 'disc_part', setup.paths.data)
				if setup.training.discriminator.history.save:
					disc_ctx.history.serialize(setup.paths.data, 'part')
			if (setup.training.density.decoder.active or setup.training.velocity.decoder.active) and target_encoder_model is not None:
				save_NNmodel(target_encoder_model, 'target_encoder_part', setup.paths.data)
			if lifting_network_model is not None:
				save_NNmodel(lifting_network_model, 'lifting_network_part', setup.paths.data)
			if setup.training.volume_encoder.active and volume_encoder_model is not None:
				save_NNmodel(volume_encoder_model, 'volume_encoder_part', setup.paths.data)
			if setup.training.frame_merge_network.active and frame_merge_network_model is not None:
				save_NNmodel(frame_merge_network_model, 'frame_merge_network_part', setup.paths.data)
			if setup.training.density.decoder.active and density_decoder_model is not None:
				save_NNmodel(density_decoder_model, 'density_decoder_part', setup.paths.data)
			if setup.training.velocity.decoder.active and velocity_decoder_model is not None:
				save_NNmodel(velocity_decoder_model, 'velocity_decoder_part', setup.paths.data)
			if setup.training.velocity.decoder.active and vel_input_encoder_model is not None:
				save_NNmodel(vel_input_encoder_model, 'velocity_input_encoder_part', setup.paths.data)
			if setup.training.velocity.decoder.active and vel_downscale_encoder_model is not None:
				save_NNmodel(vel_downscale_encoder_model, 'velocity_downscale_encoder_part', setup.paths.data)
			
		# something unexpected happended. save state if possible and exit.
		except:
			log.exception('Exception during training. Attempting to save state...')
			try:
				summary_writer.close()
			except:
				log.error('Could not close summary writer', exc_info=True)
			if 'sequence' in locals():
				try:
					sequence.save(suffix="exc")
				except:
					log.error('Could not save sequence', exc_info=True)
			
			if 'disc_model' in locals():
				try:
					save_NNmodel(disc_ctx.model, 'disc_exc', setup.paths.data)
					if setup.training.discriminator.history.save:
						disc_ctx.history.serialize(setup.paths.data, 'exc')
				except:
					log.exception('Could not save discriminator')
			if 'target_encoder_model' in locals() and target_encoder_model is not None:
				try:
					save_NNmodel(target_encoder_model, 'target_encoder_exc', setup.paths.data)
				except:
					log.exception('Could not save target encoder')
			if 'lifting_network_model' in locals() and lifting_network_model is not None:
				try:
					save_NNmodel(lifting_network_model, 'lifting_network_exc', setup.paths.data)
				except:
					log.exception('Could not save lifting network')
			if 'volume_encoder_model' in locals() and volume_encoder_model is not None:
				try:
					save_NNmodel(volume_encoder_model, 'volume_encoder_exc', setup.paths.data)
				except:
					log.exception('Could not save volume encoder')
			if 'frame_merge_network_model' in locals() and frame_merge_network_model is not None:
				try:
					save_NNmodel(frame_merge_network_model, 'frame_merge_network_exc', setup.paths.data)
				except:
					log.exception('Could not save frame merge network')
			if 'density_decoder_model' in locals():
				try:
					save_NNmodel(density_decoder_model, 'density_decoder_exc', setup.paths.data)
				except:
					log.exception('Could not save density decoder')
			if 'velocity_decoder_model' in locals():
				try:
					save_NNmodel(velocity_decoder_model, 'velocity_decoder_exc', setup.paths.data)
				except:
					log.exception('Could not save velocity decoder')
			if 'vel_input_encoder_model' in locals():
				try:
					save_NNmodel(vel_input_encoder_model, 'velocity_input_encoder_exc', setup.paths.data)
				except:
					log.exception('Could not save velocity input encoder')
			if 'vel_downscale_encoder_model' in locals():
				try:
					save_NNmodel(vel_downscale_encoder_model, 'velocity_downscale_encoder_exc', setup.paths.data)
				except:
					log.exception('Could not save velocity downscale encoder')
			try:
				with open(os.path.join(setup.paths.log, 'profiling.txt'), 'w') as f:
					profiler.stats(f)
			except:
				log.exception('Could not save profiling')
			faulthandler.disable()
			faultlog.close()
			sys.exit(1)
		else:
			log.info('Optimization finished after %s', format_time(time.time() - optim_start))
		finally:
			# reset signal handling
			signal.signal(signal.SIGINT, signal.SIG_DFL)
		
		with open(os.path.join(setup.paths.log, 'profiling.txt'), 'w') as f:
			profiler.stats(f)
		faulthandler.disable()
		faultlog.close()
		
		scalar_results = munch.Munch()
		scalar_results.buoyancy = buoyancy.numpy().tolist() if setup.training.optimize_buoyancy else None
		scalar_results.light_intensity = [_.numpy().tolist() for _ in light_var_list] if setup.training.light.optimize else None
		final_transform = sim_transform.copy_no_data()
		final_transform.grid_size = sequence[0].transform.grid_size if setup.training.density.decoder.active else sequence[0].density.shape
		scalar_results.sim_transform = final_transform
		
		with open(os.path.join(setup.paths.data, "scalar_results.json"), "w") as f:
			try:
				json.dump(scalar_results, f, default=tf_to_dict, sort_keys=True, indent=2)
			except:
				log.exception("Failed to write scalar_results:")
			
		
	if not args.fit:
		log.debug('Load data')
		if setup.data.load_sequence is None:
			raise ValueError("No sequence specified (setup.data.load_sequence)")
		sf = RunIndex.parse_scalarFlow(setup.data.load_sequence)
		if sf is not None:
			log.info("Load scalarFlow sequence for evaluation, sim offset %d, frame offset %d", sf["sim"], sf["frame"])
			frames = list(range(setup.data.start+sf["frame"], setup.data.stop+sf["frame"], setup.data.step))
			vel_bounds = None if setup.data.velocity.boundary.upper()=='CLAMP' else Zeroset(-1, shape=GridShape(), outer_bounds="CLOSED", as_var=False, device=resource_device)
			with profiler.sample("load sF sequence"):
				if args.console:
					load_bar = ProgressBar(len(frames), name="Load Sequence: ")
					def update_pbar(step, frame):
						load_bar.update(step, desc="Frame {:03d} ({:03d}/{:03d})".format(frame, step+1, len(frames)))
				else: update_pbar = lambda i, f: None
				sequence = Sequence.from_scalarFlow_file(pFmt.format(setup.data.density.scalarFlow_reconstruction, sim=setup.data.simulation+sf["sim"]), \
					pFmt.format(setup.data.velocity.scalarFlow_reconstruction, sim=setup.data.simulation+sf["sim"]), \
					frames, transform=sim_transform, #sF_transform,
					as_var=False, base_path=setup.paths.data, boundary=vel_bounds, scale_renderer=scale_renderer, warp_renderer=warp_renderer, device=resource_device, frame_callback=update_pbar)
				if args.console: load_bar.finish(desc="Done")
			frames = list(range(setup.data.start, setup.data.stop, setup.data.step))
			for s,f in zip(sequence, frames):
				s.base_target_cameras = setup_target_cameras(target_cameras, train_cam_resolution, None, setup.rendering.target_cameras.crop_frustum_pad)
				s.velocity.scale_magnitude(setup.data.step)
				s.density.scale(setup.data.density.scale)
				s.frame = f
		else:
			load_entry = run_index.get_run_entry(setup.data.load_sequence)
			log.info("Load sequence from '%s' for evaluation", load_entry.path)
			try:
				load_setup = munch.munchify(load_entry.setup)
				vel_bounds = None if setup.data.velocity.boundary.upper()=='CLAMP' else Zeroset(-1, shape=GridShape(), outer_bounds="CLOSED", as_var=False, device=resource_device)
				try:
					vel_bounds = None if load_setup.data.velocity.boundary.upper()=='CLAMP' else Zeroset(-1, shape=GridShape(), outer_bounds="CLOSED", as_var=False, device=resource_device)
				except:
					log.info("Using default boundaries: %s", vel_bounds)
			except:
				log.exception("failed to load config from %s:", load_entry.path)
				sys.exit(1)
			frames = list(range(setup.data.start, setup.data.stop, setup.data.step))
			
			if setup.data.step!=load_setup.data.step:
				log.info("Loaded frame step does not match data, scaling velocity with %f", setup.data.step/load_setup.data.step)
			
			try:
				load_scalars = load_entry.scalars
				t = from_dict(load_scalars["sim_transform"])
			except:
				log.exception("Failed to load transformation, using default.")
			else:
				sim_transform = t
			sim_transform.grid_size = density_size.as_shape
			log.info("current grid transformation: %s", sim_transform)
			if setup.validation.synth_data_eval_setup.upper()=="SF": # and False:
				sim_transform.parent.parent.translation[1]=sim_transform.grid_size_world().y/2 + 0.005
				log.info("modified grid transformation: %s", sim_transform)
			
			
			
			log.info("-> cell size world: %s", sim_transform.cell_size_world())
			### EVALUATION SETUP
			
			# -- Load Dataset --
			
			load_density_dataset = ("TARGET" in setup.validation.warp_test) or (setup.validation.stats and setup.validation.cmp_vol_targets) or args.save_volume #or (not setup.training.density.decoder.active) #True
			load_velocity_dataset = False #("TARGET" in setup.validation.warp_test) or (setup.validation.stats and setup.validation.cmp_vol_targets) #or (not setup.training.velocity.decoder.active) #True
			sequence_length = len(frames) #setup.data.sequence_length
			sequence_step = setup.data.step #setup.data.sequence_step
			
			def get_max_cell_size():
				min_grid_res = [setup.training.velocity.decoder.min_grid_res, int(math.ceil(setup.training.velocity.decoder.min_grid_res * setup.data.y_scale)), setup.training.velocity.decoder.min_grid_res]
				tmp_T = sim_transform.copy_no_data()
				tmp_T.grid_size = min_grid_res
				max_cell_size = tmp_T.cell_size_world()
				return min(max_cell_size)
			synth_max_cell_size = get_max_cell_size() #cell size at coarsest resolution, as defined by min_grid_res
			synth_max_translation = synth_max_cell_size * setup.data.synth_shapes.max_translation #0.08
			
			eval_data = setup.validation.synth_data_eval_setup.upper() #"CUBE" # SF, SPHERE, CUBE, ROTCUBE, STATICCUBE
			eval_data_is_dataset = []
			if eval_data in ["SF", "SF_RENDER"]:
				sequence_length_frames = sequence_length*sequence_step
				
				eval_data_is_dataset += ["SF", "SF_RENDER"]
				def resolve_paths():
					path_raw = run_index[setup.data.density.target]
					if path_raw is None:
						path_raw = setup.data.density.target
					else:
						pass #raise NotImplementedError
					path_preproc = None
					
					path_density = run_index[setup.data.density.initial_value]
					if path_density is None: #SF data
						path_density = setup.data.density.initial_value
						dens_src_transform = sF_transform
						dens_type = "SF"
					else: # GlobTrans reconstruction
						dens_entry = run_index.get_run_entry(setup.data.density.initial_value)
						dens_src_transform = from_dict(dens_entry.scalars["sim_transform"])
						dens_type = "OWN"
					
					path_velocity = run_index[setup.data.velocity.initial_value]
					if path_velocity is None: #SF data
						path_velocity = setup.data.velocity.initial_value
						vel_src_transform = sF_transform
						vel_type = "SF"
					else: # GlobTrans reconstruction
						vel_entry = run_index.get_run_entry(setup.data.velocity.initial_value)
						vel_src_transform = from_dict(vel_entry.scalars["sim_transform"])
						vel_type = "OWN"
					
					return {"path_raw":path_raw, "path_preproc":path_preproc, 
						"path_density":path_density, "density_t_src":dens_src_transform, "density_type":dens_type, 
						"path_velocity":path_velocity, "velocity_t_src":vel_src_transform, "velocity_type":vel_type, }
				
				kwargs = {}
				if eval_data=="SF_RENDER":
					load_density_dataset = True
					kwargs["render_targets"] = True
					kwargs["density_renderer"] = synth_target_renderer
					kwargs["cameras"] = target_cameras
					kwargs["lights"] = lights
				
				target_dataset, target_data_cache = get_targets_dataset_v2(sim_indices=setup.data.sims, frame_start=setup.data.start, frame_stop=setup.data.start+sequence_length_frames, frame_strides=sequence_length_frames, \
					raw=True, preproc=True, bkg=True, hull=True, batch_size=1, \
					sequence_step=sequence_step, sequence_length=sequence_length, \
					view_indices=[scalarFlow_cam_ids[_] for _ in setup.data.density.target_cam_ids], num_views=len(setup.data.density.target_cam_ids), 
					SF_frame_offset=setup.data.scalarFlow_frame_offset, \
					down_scale=setup.training.train_res_down, channels=color_channel, threshold=setup.data.density.hull_threshold, shuffle_frames=False,\
					density=load_density_dataset, density_t_dst=sim_transform, density_sampler=scale_renderer, \
					velocity=load_velocity_dataset, \
					cache_device=data_device, randomize_transform=False, \
					**resolve_paths(), **kwargs)
					#path_raw=setup.data.density.target, path_preproc=None, path_density=setup.data.density.initial_value, density_t_src=sF_transform, path_velocity=setup.data.velocity.initial_value)
				#dataset_size = len(setup.data.sims) * int(np.ceil((setup.data.stop - setup.data.start)/setup.data.step)) #//batch_size
			elif eval_data == "INFLOW_TEST":
				eval_data_is_dataset += ["INFLOW_TEST"]
				sequence_length_frames = sequence_length*sequence_step
				
				target_dataset, target_data_cache = get_targets_dataset_v2(sim_indices=setup.data.sims, frame_start=setup.data.start, frame_stop=setup.data.start+sequence_length_frames, frame_strides=sequence_length_frames, \
					raw=True, preproc=True, bkg=True, hull=True, batch_size=1, \
					sequence_step=sequence_step, sequence_length=sequence_length, \
					view_indices=[scalarFlow_cam_ids[_] for _ in setup.data.density.target_cam_ids], num_views=len(setup.data.density.target_cam_ids), 
					SF_frame_offset=setup.data.scalarFlow_frame_offset, \
					down_scale=setup.training.train_res_down, channels=color_channel, threshold=setup.data.density.hull_threshold, shuffle_frames=False,\
					density=load_density_dataset, density_t_dst=sim_transform, density_sampler=scale_renderer, \
					velocity=load_velocity_dataset, \
					cache_device=data_device, randomize_transform=False, \
					render_targets=True, density_renderer=synth_target_renderer, cameras=target_cameras, lights=lights, \
					path_density="/home/franz/data/fluid_sim/manta/synth_plume_128_{sim:06d}/density_{frame:06d}.npz", \
					density_t_src=sF_transform, velocity_t_src=sF_transform, density_type="MANTA", \
					path_raw=None)
			elif eval_data in ["SPHERE", "CUBE", "TORUS"]:
				log.info("Using synthetic dataset %s for evaluation", eval_data)
				
				eval_data_is_dataset += ["SPHERE", "CUBE", "TORUS"]
				if not setup.data.SDF:
					target_dataset = get_synthTargets_dataset_v2(batch_size=1, base_grid_transform=sim_transform, sequence_length=sequence_length, \
						view_indices=[scalarFlow_cam_ids[_] for _ in setup.data.density.target_cam_ids], num_views=len(setup.data.density.target_cam_ids), \
						cameras=target_cameras, lights=lights, device=resource_device, \
						density_range=[0.15,0.25], inner_range=[0.1,0.4], scale_range=[0.3,0.5], \
						translation_range=[-synth_max_translation,synth_max_translation], \
						rotation_range=[10,30], \
						raw=True, preproc=True, bkg=True, hull=True, mask=setup.data.SDF, channels=1, SDF=setup.data.SDF, \
						density=load_density_dataset, velocity=load_velocity_dataset, advect_density=False, density_sampler=density_sampler, density_renderer=synth_target_renderer, \
						seed=np.random.randint(np.iinfo(np.int32).max) if setup.validation.synth_data_seed is None else setup.validation.synth_data_seed, \
						sample_overrides={"shape_type":(5 if eval_data=="TORUS" else (0 if eval_data=="SPHERE" else 1)), }) #'density_scale':0.1, "base_scale":[0.18]*3, "initial_translation":[0,0,0], 
				else:
					target_dataset = get_synthTargets_dataset_v2(batch_size=1, base_grid_transform=sim_transform, sequence_length=sequence_length, \
						view_indices=[scalarFlow_cam_ids[_] for _ in setup.data.density.target_cam_ids], num_views=len(setup.data.density.target_cam_ids), \
						cameras=target_cameras, lights=lights, device=resource_device, \
						density_range=[0.05,0.14], inner_range=[0.1,0.4], scale_range=[0.3,0.5], translation_range=[-synth_max_translation,synth_max_translation], rotation_range=[0,20], \
						raw=True, preproc=True, bkg=True, hull=True, mask=setup.data.SDF, channels=1, SDF=setup.data.SDF, \
						density=load_density_dataset, velocity=load_velocity_dataset, advect_density=not setup.data.SDF, density_sampler=density_sampler, density_renderer=synth_target_renderer, \
						seed=np.random.randint(np.iinfo(np.int32).max) if setup.validation.synth_data_seed is None else setup.validation.synth_data_seed, \
						sample_overrides={'density_scale':0.1, "base_scale":[0.30]*3, "shape_type":(0 if eval_data=="SPHERE" else 1), "initial_translation":[0,0,0], })
			elif eval_data=="ROTCUBE":
				log.info("Using synthetic dataset %s for evaluation", eval_data)
				
				eval_data_is_dataset += ["ROTCUBE"]
				target_dataset = get_synthTargets_dataset_v2(batch_size=1, base_grid_transform=sim_transform, sequence_length=sequence_length, \
					view_indices=[scalarFlow_cam_ids[_] for _ in setup.data.density.target_cam_ids], num_views=len(setup.data.density.target_cam_ids), \
					cameras=target_cameras, lights=lights, device=resource_device, \
					density_range=[0.05,0.14], inner_range=[0.1,0.4], scale_range=[0.14,0.22], translation_range=[0,0], rotation_range=[0,20], \
					raw=True, preproc=True, bkg=True, hull=True, channels=1, mask=setup.data.SDF, SDF=setup.data.SDF, \
					density=load_density_dataset, velocity=load_velocity_dataset, density_sampler=density_sampler, density_renderer=synth_target_renderer, randomize_transform=False, \
					seed=np.random.randint(np.iinfo(np.int32).max) if setup.validation.synth_data_seed is None else setup.validation.synth_data_seed, \
					sample_overrides={'density_scale':0.1, "base_scale":[0.16,0.4,0.24], "shape_type":1, "initial_translation":[0,0,0], "initial_rotation_rotvec":[0,0,0], "rotvec":[0,0,0.17]})
					#
			elif eval_data in ["STATICCUBE"]:
				log.info("Using synthetic dataset %s for evaluation", eval_data)
				
				eval_data_is_dataset += ["STATICCUBE"]
				target_dataset = get_synthTargets_dataset_v2(batch_size=1, base_grid_transform=sim_transform, sequence_length=sequence_length, \
					view_indices=[scalarFlow_cam_ids[_] for _ in setup.data.density.target_cam_ids], num_views=len(setup.data.density.target_cam_ids), \
					cameras=target_cameras, lights=lights, device=resource_device, \
					density_range=[0.05,0.14], inner_range=[0.1,0.4], scale_range=[0.14,0.22], translation_range=[0,0], rotation_range=[0,0], \
					raw=True, preproc=True, bkg=True, hull=True, channels=1, mask=setup.data.SDF, SDF=setup.data.SDF, \
					density=load_density_dataset, velocity=load_velocity_dataset, density_sampler=density_sampler, density_renderer=synth_target_renderer, randomize_transform=False, \
					seed=np.random.randint(np.iinfo(np.int32).max) if setup.validation.synth_data_seed is None else setup.validation.synth_data_seed, \
					sample_overrides={'density_scale':0.1, "base_scale":[0.18]*3, "shape_type":1, "initial_translation":[0,0,0], "initial_rotation_rotvec":[0,0,0]})
			# returns: [NSVHWC for type]
			else:
				raise ValueError("Unknown eval data '%s'."%(eval_data,))
			if not eval_data in eval_data_is_dataset:
				target_dataset = TargetDataset(target_data, resource_device=resource_device)
			
			
			
			def get_max_recursive_MS_grow_levels(decoder_config, cast_fn=round):
				if decoder_config.recursive_MS_levels=="VARIABLE":
					#return GrowingUNet.get_max_levels(sim_transform.grid_size, scale_factor=load_setup.training.velocity.decoder.model.level_scale_factor, min_size=setup.training.velocity.decoder.min_grid_res)
					i = 0
					while (min(_/(decoder_config.recursive_MS_scale_factor**i) for _ in sim_transform.grid_size)>=decoder_config.min_grid_res):
						i +=1
					return i
				else:
					return decoder_config.recursive_MS_levels
			
			with profiler.sample("load sequence"):
				
				# -- Setup and Load Networks --
				
				target_encoder_model = None
				if "NETWORK" in load_setup.training.view_encoder.encoder: # and setup.training.view_encoder.load_encoder is not None:
					assert isinstance(setup.training.view_encoder.model, str)
					model_path = run_index[setup.training.view_encoder.model]
					if model_path is None:
						model_path = setup.training.view_encoder.model
					#target_encoder_model = tf.keras.models.load_model(model_path, custom_objects=custom_keras_objects)
					target_encoder_model = load_model(setup.training.view_encoder.model, num_levels=None, input_merge_weight=0.5, skip_merge_weight=1.0)
					string_buffer = StringWriter()
					string_buffer.write_line("View Encoder Model:")
					target_encoder_model.summary(print_fn=string_buffer.write_line)
					log.info(string_buffer.get_string())
					string_buffer.reset()
				
				velocity_decoder_model = None
				vel_input_encoder_model = None
				vel_downscale_encoder_model = None
				# TODO: recMS non-shared decoder loading
				#velocity_grid = None
				if isinstance(setup.training.velocity.decoder.model, str):
					if load_setup.training.velocity.decoder.recursive_MS and not load_setup.training.velocity.decoder.recursive_MS_shared_model:
						model_path = run_index[setup.training.velocity.decoder.model] #setup.training.density.decoder.model]
						if model_path is None:
							model_path = setup.training.velocity.decoder.model
						model_paths = get_NNmodel_names(model_path)
						max_levels = get_max_recursive_MS_grow_levels(setup.training.velocity.decoder)
						if max_levels>len(model_paths):
							log.error("%s only has %d levels, but %d were requested.", setup.training.velocity.decoder.model, len(model_paths), max_levels)
							raise SystemExit(1)
						log.info("load %d/%d decoders for non-shared recursive multi-scale velocity.", max_levels, len(model_paths))
						velocity_decoder_model = []
						if (not hasattr(load_setup.training.velocity.decoder.model, "num_levels")) or load_setup.training.velocity.decoder.model.num_levels=="VARIABLE":
							num_levels = GrowingUNet.get_max_levels(sim_transform.grid_size, scale_factor=load_setup.training.velocity.decoder.recursive_MS_scale_factor, min_size=load_setup.training.velocity.decoder.min_grid_res)
						else:
							num_levels = load_setup.training.velocity.decoder.model.num_levels
						for level in range(max_levels):
							velocity_decoder_model.append(load_model(model_paths[level], num_levels=num_levels, input_merge_weight=0.5, skip_merge_weight=1.0))
							string_buffer = StringWriter()
							string_buffer.write_line("Velocity Decoder Model %d:"%(level,))
							velocity_decoder_model[-1].summary(print_fn=string_buffer.write_line)
							log.info(string_buffer.get_string())
							string_buffer.reset()
							
							#velocity_decoder_model.append(setup_velocity_decoder(velocity_decoder_input_channels, name="VelocityDecoder_L{:03d}".format(level)))
						#raise NotImplementedError("recusive MS with multiple decoders not yet implemented.")
					else:
						max_levels = GrowingUNet.get_max_levels(sim_transform.grid_size, load_setup.training.velocity.decoder.recursive_MS_scale_factor, min_size=load_setup.training.density.decoder.min_grid_res)
						velocity_decoder_model = load_model(setup.training.velocity.decoder.model, num_levels=max_levels, input_merge_weight=0.5, skip_merge_weight=1.0)
						string_buffer = StringWriter()
						string_buffer.write_line("Velocity Decoder Model:")
						velocity_decoder_model.summary(print_fn=string_buffer.write_line)
						log.info(string_buffer.get_string())
						string_buffer.reset()
				elif load_setup.training.velocity.decoder.active:
					log.warning("velocity decoder was active in training")
					
				#	velocity_grid = NeuralVelocityGrid(volume_decoder=velocity_decoder_model, boundary=vel_bounds, scale_renderer=scale_renderer, warp_renderer=warp_renderer, device=resource_device, parent_state=None)
				#	velocity_grid.use_raw_images = load_setup.training.velocity.decoder.input_type=='RAW'
				
				density_decoder_model = None
				#density_grid = None
				if isinstance(setup.training.density.decoder.model, str):
					if load_setup.training.density.decoder.recursive_MS and not load_setup.training.density.decoder.recursive_MS_shared_model:
						model_path = run_index[setup.training.density.decoder.model] #setup.training.density.decoder.model]
						if model_path is None:
							model_path = setup.training.density.decoder.model
						model_paths = get_NNmodel_names(model_path)
						max_levels = get_max_recursive_MS_grow_levels(setup.training.density.decoder)
						if max_levels>len(model_paths):
							log.error("%s only has %d levels, but %d were requested.", setup.training.density.decoder.model, len(model_paths), max_levels)
							raise SystemExit(1)
						log.info("load %d/%d decoders for non-shared recursive multi-scale density.", max_levels, len(model_paths))
						density_decoder_model = []
						if (not hasattr(load_setup.training.density.decoder.model, "num_levels")) or load_setup.training.density.decoder.model.num_levels=="VARIABLE":
							num_levels = GrowingUNet.get_max_levels(sim_transform.grid_size, scale_factor=load_setup.training.density.decoder.recursive_MS_scale_factor, min_size=load_setup.training.density.decoder.min_grid_res)
						else:
							num_levels = load_setup.training.density.decoder.model.num_levels
						for level in range(max_levels):
							density_decoder_model.append(load_model(model_paths[level], num_levels=num_levels, input_merge_weight=0.5, skip_merge_weight=1.0))
							string_buffer = StringWriter()
							string_buffer.write_line("Density Decoder Model %d:"%(level,))
							density_decoder_model[-1].summary(print_fn=string_buffer.write_line)
							log.info(string_buffer.get_string())
							string_buffer.reset()
							
							#density_decoder_model.append(setup_velocity_decoder(velocity_decoder_input_channels, name="VelocityDecoder_L{:03d}".format(level)))
						#raise NotImplementedError("recusive MS with multiple decoders not yet implemented.")
					else:
						max_levels = GrowingUNet.get_max_levels(sim_transform.grid_size, load_setup.training.density.decoder.recursive_MS_scale_factor, min_size=load_setup.training.density.decoder.min_grid_res)
						density_decoder_model = load_model(setup.training.density.decoder.model, num_levels=max_levels, input_merge_weight=0.5, skip_merge_weight=1.0)
						string_buffer = StringWriter()
						string_buffer.write_line("Density Decoder Model:")
						density_decoder_model.summary(print_fn=string_buffer.write_line)
						log.info(string_buffer.get_string())
						string_buffer.reset()
				elif load_setup.training.density.decoder.active:
					log.warning("density decoder was active in training")
				
				
				volume_encoder_model = None
				lifting_network_model = None
				if setup.training.view_encoder.lifting.upper()=="UNPROJECT" and setup.training.volume_encoder.active and isinstance(setup.training.volume_encoder.model, str):
					volume_encoder_model = load_model(setup.training.volume_encoder.model, num_levels=None, input_merge_weight=0.5, skip_merge_weight=1.0)
					string_buffer = StringWriter()
					string_buffer.write_line("Volume Encoder Model:")
					volume_encoder_model.summary(print_fn=string_buffer.write_line)
					log.info(string_buffer.get_string())
					string_buffer.reset()
				elif load_setup.training.view_encoder.lifting.upper()=="UNPROJECT" and "volume_encoder" in load_setup.training and load_setup.training.volume_encoder.active:
					log.warning("volume_encoder was active in training")
				
				if setup.training.view_encoder.lifting.upper()=="NETWORK" and setup.training.lifting_network.active and isinstance(setup.training.lifting_network.model, str):
					#assert len(target_cameras)==26
					assert len(setup.validation.input_view_mask)==1 # and setup.validation.input_view_mask[0]==0
					max_levels = GrowingUNet.get_max_levels(sim_transform.grid_size, 2 if isinstance(load_setup.training.lifting_network.model, str) else load_setup.training.lifting_network.model.level_scale_factor, min_size=load_setup.training.lifting_network.min_grid_res)
					lifting_cameras = [target_cameras[_] for _ in setup.validation.input_view_mask]
					log.info("Number of levels for lifting network: %d", max_levels)
					lifting_network_model = load_model(setup.training.lifting_network.model, num_levels=max_levels, skip_merge_weight=1.0, output_residual_weight=0.0, \
						lifting_renderer=lifting_renderer, lifting_cameras=lifting_cameras, lifting_transform=sim_transform.copy_no_data(), lifting_shape=sim_transform.grid_size)
					#, input_merge_weight=0.5, skip_merge_weight=1.0
					string_buffer = StringWriter()
					string_buffer.write_line("Lifting Network Model:")
					lifting_network_model.summary(print_fn=string_buffer.write_line)
					log.info(string_buffer.get_string())
					string_buffer.reset()
					
					# log.info("active level: %d", lifting_network_model.get_active_level())
					# lifting_network_model.set_active_level_from_grid_size(grid_size=lifting_cameras[0].transform.grid_size[1:], min_size=load_setup.training.lifting_network.min_grid_res, lifting_size=sim_transform.grid_size)
					log.info("active levels: %d, output mode: %s", lifting_network_model.get_active_level()+1, lifting_network_model.output_mode)
					if lifting_network_model.output_mode=="RESIDUAL_WEIGHTED":
						log.info("residual output weights: %s", [lifting_network_model.get_output_residual_weight(l) for l in range(1, lifting_network_model.get_active_level()+1)])
					
				elif load_setup.training.view_encoder.lifting.upper()=="NETWORK" and "lifting_network" in load_setup.training and load_setup.training.lifting_network.active:
					log.warning("lifting_network was active in training")
				
				frame_merge_network_model = None
				if setup.training.frame_merge_network.active and isinstance(setup.training.frame_merge_network.model, str):
					frame_merge_network_model = load_model(setup.training.frame_merge_network.model, num_levels=None, input_merge_weight=0.5, skip_merge_weight=1.0)
					string_buffer = StringWriter()
					string_buffer.write_line("Frame Merge Model:")
					frame_merge_network_model.summary(print_fn=string_buffer.write_line)
					log.info(string_buffer.get_string())
					string_buffer.reset()
				elif "frame_merge_network" in load_setup.training and load_setup.training.frame_merge_network.active:
					log.warning("frame_merge_network was active in training")
			
				def frame_velTargetSetup(aux_sequence, frame):
					if not ("velocity" in aux_sequence[frame]) and (aux_sequence[frame].velocity is not None):
						raise ValueError("")
					vel_var_name = "velocityTarget_f{:06d}".format(frame)
					velocity = VelocityGrid.from_staggered_combined(aux_sequence[frame].velocity, as_var=False, boundary=vel_bounds, \
						scale_renderer=scale_renderer, warp_renderer=warp_renderer, device=resource_device, var_name=vel_var_name, trainable=False)
					return velocity
				def frame_densTargetSetup(aux_sequence, frame):
					if not ("density" in aux_sequence[frame]) and (aux_sequence[frame].density is not None):
						raise ValueError("")
					density_grid_shape = GridShape.from_tensor(aux_sequence[frame].density).spatial_vector.as_shape
					density = DensityGrid(density_grid_shape, d=aux_sequence[frame].density, as_var=False, \
						scale_renderer=scale_renderer, hull=None, inflow=None, inflow_offset=None, inflow_mask=None, \
						device=resource_device, restrict_to_hull=setup.training.density.use_hull, is_SDF=setup.data.SDF)
					return density
				
				# -- Setup Data Modules --
				
				def build_sequence(frames, dataset):
					sequence = []
					aux_sequence = {}
					last_state = None
					for idx, frame in enumerate(frames):
						
						aux_sequence[frame] = frame_loadTargets(setup, idx, sim_transform, dataset)
						
						# compatibility
						vel_type_input_features = load_setup.training.velocity.decoder.get("type_input_features", \
							["TARGET_RAW_UNPROJECTION"] if load_setup.training.velocity.decoder.input_type=='RAW' else ["TARGET_UNPROJECTION"])
						
						velocity_grid = None
						if velocity_decoder_model is not None:
							velocity_grid = NeuralVelocityGrid(volume_decoder=velocity_decoder_model, boundary=vel_bounds, \
								scale_renderer=scale_renderer, warp_renderer=warp_renderer, device=resource_device, parent_state=None, \
								velocity_format=load_setup.training.velocity.decoder.velocity_format, \
								step_input_density=load_setup.training.velocity.decoder.step_input_density, \
								step_input_density_target=load_setup.training.velocity.decoder.get("step_input_density_target", []), \
								step_input_density_proxy=load_setup.training.velocity.decoder.get("step_input_density_proxy", []), \
								step_input_features=load_setup.training.velocity.decoder.step_input_features, \
								type_input_features=vel_type_input_features, \
								warp_input_indices=load_setup.training.velocity.decoder.warp_input_indices, \
								downscale_input_modes=load_setup.training.velocity.decoder.get("downscale_input_modes", ["RESAMPLE"]))
							velocity_grid.use_raw_images = load_setup.training.velocity.decoder.input_type=='RAW'
							velocity_grid.set_input_encoder(vel_input_encoder_model)
							velocity_grid.set_downscale_encoder(vel_downscale_encoder_model)
						else:
							velocity_grid = VelocityGrid(GridShape.from_tensor(aux_sequence[frame].density).spatial_vector.as_shape, 0.0, boundary=vel_bounds, scale_renderer=scale_renderer, warp_renderer=warp_renderer, device=resource_device, var_name='vel_dummy_f{:06d}'.format(frame))
						
						density_grid = None
						density_proxy = None
						if density_decoder_model is not None:
							density_grid = NeuralDensityGrid(volume_decoder=density_decoder_model, scale_renderer=scale_renderer, parent_state=None, \
								base_input=load_setup.training.density.decoder.base_input, \
								step_input_density=load_setup.training.density.decoder.step_input_density, \
								step_input_density_target=load_setup.training.density.decoder.step_input_density_target, \
								step_input_features=load_setup.training.density.decoder.step_input_features, \
								type_input_features=load_setup.training.density.decoder.type_input_features, \
								device=resource_device, is_SDF=setup.data.SDF, base_SDF_mode=setup.training.density.decoder.base_SDF_mode)
							density_grid.use_raw_images = load_setup.training.density.decoder.input_type=='RAW'
							density_proxy = NeuralDensityGrid(volume_decoder=density_decoder_model, scale_renderer=scale_renderer, parent_state=None, \
								base_input=load_setup.training.density.decoder.base_input, \
								step_input_density=load_setup.training.density.decoder.step_input_density, \
								step_input_density_target=load_setup.training.density.decoder.step_input_density_target, \
								step_input_features=load_setup.training.density.decoder.step_input_features, \
								type_input_features=load_setup.training.density.decoder.type_input_features, \
								device=resource_device, is_SDF=setup.data.SDF, base_SDF_mode=setup.training.density.decoder.base_SDF_mode)
							density_proxy.use_raw_images = load_setup.training.density.decoder.input_type=='RAW'
						else:
							dens_hull = None
							density_grid = DensityGrid(GridShape.from_tensor(aux_sequence[frame].density).spatial_vector.as_shape, d=aux_sequence[frame].density, as_var=False, \
								scale_renderer=scale_renderer, hull=dens_hull, inflow=None, inflow_offset=None, inflow_mask=None, \
								device=resource_device, restrict_to_hull=setup.training.density.use_hull, is_SDF=setup.data.SDF)
							log.debug("Initialized density with loaded data")
						
						state = NeuralState(density_grid, velocity_grid, target_encoder=target_encoder_model, encoder_output_types=load_setup.training.view_encoder.encoder, \
							target_lifting=load_setup.training.view_encoder.lifting, lifting_renderer=lifting_renderer, target_merging=load_setup.training.view_encoder.merge, \
							volume_encoder=volume_encoder_model, frame=0, transform=sim_transform.copy_no_data(), lifting_network=lifting_network_model, frame_merge_network=frame_merge_network_model)
						
						state.frame = frame
						state_set_targets(state, aux_sequence)
						#dataset.step()
						state.data_path = os.path.join(setup.paths.data, 'frame_{:06d}'.format(state.frame))
						
						state.density_proxy = density_proxy
						
						#set volume target data if needed/available
						if ("density" in aux_sequence[frame]) and (aux_sequence[frame].density is not None):
							state.density_target = frame_densTargetSetup(aux_sequence, frame)
						if ("velocity" in aux_sequence[frame]) and (aux_sequence[frame].velocity is not None):
							state.velocity_target = frame_velTargetSetup(aux_sequence, frame)
						
						state.base_target_cameras = setup_target_cameras(target_cameras, train_cam_resolution, None, setup.rendering.target_cameras.crop_frustum_pad)
						#state_set_targets()
						if False:
							state_randomize(state, randomize_transform=True)
							log.info("state.transform: %s", state.transform)
						
						state.prev = last_state
						if last_state is not None: last_state.next = state
						last_state = state
						
						if density_grid is not None:
							density_grid.parent_state = state
							if density_decoder_model is not None and load_setup.training.density.decoder.recursive_MS: #setup.training.density.decoder.active
								#setup recusive MS
								num_levels = get_max_recursive_MS_grow_levels(setup.training.density.decoder)
								#log.debug("%s", setup.training.velocity.decoder.recursive_MS_levels)
								# TODO: check trained levels and shared decoder
								if not load_setup.training.density.decoder.recursive_MS_shared_model:
									if num_levels!=len(density_decoder_model):
										log.error("recursive multi-scale density is fixed to %d recursions, %d are requested.", len(density_decoder_model), num_levels)
										raise SystemExit(1)
								scale_factor = setup.training.density.decoder.recursive_MS_scale_factor
								if load_setup.training.density.decoder.recursive_MS_scale_factor!=scale_factor:
									log.warning("network was trained with a resolution scale-factor of %f, evaluation uses %f", load_setup.training.density.decoder.recursive_MS_scale_factor, scale_factor)
								density_grid.set_recursive_MS(num_levels, scale_factor, shared_decoder=load_setup.training.density.decoder.recursive_MS_shared_model, train_mode=load_setup.training.density.decoder.recursive_MS_train_mode, as_residual=setup.training.density.decoder.recursive_MS_residual, direct_input=setup.training.density.decoder.get("recursive_MS_direct_input", False))
						if density_proxy is not None:
							density_proxy.parent_state = state
							if density_decoder_model is not None and load_setup.training.density.decoder.recursive_MS:
								num_levels = get_max_recursive_MS_grow_levels(setup.training.density.decoder)
								scale_factor = setup.training.density.decoder.recursive_MS_scale_factor
								density_proxy.set_recursive_MS(num_levels, scale_factor, shared_decoder=load_setup.training.density.decoder.recursive_MS_shared_model, train_mode=load_setup.training.density.decoder.recursive_MS_train_mode, as_residual=setup.training.density.decoder.recursive_MS_residual, direct_input=setup.training.density.decoder.get("recursive_MS_direct_input", False))
						if velocity_grid is not None:
							velocity_grid.parent_state = state
							if setup.training.velocity.decoder.active and load_setup.training.velocity.decoder.recursive_MS:
								#setup recusive MS
								num_levels = get_max_recursive_MS_grow_levels(setup.training.velocity.decoder)
								#log.debug("%s", setup.training.velocity.decoder.recursive_MS_levels)
								# TODO: check trained levels and shared decoder
								if not load_setup.training.velocity.decoder.recursive_MS_shared_model:
									if num_levels!=len(velocity_decoder_model):
										log.error("recursive multi-scale velocity is fixed to %d recursions, %d are requested.", len(velocity_decoder_model), num_levels)
										raise SystemExit(1)
								scale_factor = setup.training.velocity.decoder.recursive_MS_scale_factor #load_setup.training.velocity.decoder.model.level_scale_factor #allow changing this for evaluation
								if load_setup.training.velocity.decoder.recursive_MS_scale_factor!=scale_factor:
									log.warning("network was trained with a resolution scale-factor of %f, evaluation uses %f", load_setup.training.velocity.decoder.recursive_MS_scale_factor, scale_factor)
								velocity_grid.set_recursive_MS(num_levels, scale_factor, shared_decoder=load_setup.training.velocity.decoder.recursive_MS_shared_model, train_mode=load_setup.training.velocity.decoder.recursive_MS_train_mode, direct_input=setup.training.velocity.decoder.get("recursive_MS_direct_input", False), max_level_input=setup.training.velocity.decoder.get("recursive_MS_use_max_level_input", False))
								#log.info("Set recursive-MS velocity for frame %d with %d levels.", frame, num_levels)
						
						
						sequence.append(state)
					return Sequence(sequence), aux_sequence
			
			
			sequence, eval_sequence = build_sequence(frames, target_dataset)
			log.debug('Load run targets')
			
			
			scalar_results = munch.Munch()
			final_transform = sim_transform.copy_no_data()
			final_transform.grid_size = sequence[0].transform.grid_size if setup.training.density.decoder.active else sequence[0].density.shape
			scalar_results.sim_transform = final_transform
			with open(os.path.join(setup.paths.data, "scalar_results.json"), "w") as f:
				try:
					json.dump(scalar_results, f, default=tf_to_dict, sort_keys=True, indent=2)
				except:
					log.exception("Failed to write scalar_results:")
			
			if False:
				log.warning("LR vel eval: [8,8,8], level 1")
				for state in sequence:
					state.velocity.set_centered_shape([8,8,8])
					state.velocity.set_recursive_MS_level(1)
		
		z = None #tf.zeros([1] + vel_shape + [1])
		
	#	vel_scale = world_scale(vel_shape, width=1.)
	
	if setup.validation.stats or setup.validation.warp_test or args.render:
		
		def print_stats_dict(stats, name, print_fn):
			s = '{}:\n'.format(name)
			for name in sorted(stats.keys()):
				value = stats[name]
				if isinstance(value, tf.Tensor):
					value = value.numpy().tolist()
				if not isinstance(value, float):
					s += '{:<16}: {}\n'.format(name, value)
				else:
					s += '{:<16}: {: 13.06e}\n'.format(name, value)
			print_fn(s)
			
		def render_sequence_cmp(*sequences, cameras, path, name_pre='seq', image_format='PNG', render_velocity=True, background="COLOR", crop_cameras=True):
			#sequences: iterables of states to render or lists with images for every camera.
			assert len(sequences)>1, "need at least 2 sequences to compare"
			length = len(sequences[0])
			for sequence in sequences:
				assert len(sequence)==length, "All sequences must have equal length"
			log.debug("Render comparison of %d sequences:", len(sequences))
			# render image cmp
			AABB_corners_WS = []
			for states in zip(*sequences):
				for state in states:
					if isinstance(state, State):
						dens_hull = state.density.hull #state.hull if hasattr(state, "hull") else 
						if dens_hull is None:
							continue
						dens_transform = state.get_density_transform()
						AABB_corners_WS += dens_transform.transform_AABB(*hull_AABB_OS(tf.squeeze(dens_hull, (0,-1))), True)
			if AABB_corners_WS and crop_cameras:
				seq_cams = [cam.copy_clipped_to_world_coords(AABB_corners_WS)[0] for cam in cameras]
			else:
				seq_cams = cameras
			split_cams = True
			i=0
			for states in zip(*sequences):
				log.debug("Render sequence cmp frame %d", i)
				# density: [orig, dens_warp, veldens_warp]
				if args.console: progress_bar(i*2,len(sequences[0])*2, "Step {:03d}/{:03d}: {:30}".format(i+1,len(sequence), "Sequence cmp Density"), length=30)
				bkg_render = None
				if background=='COLOR':
					bkg_render = [tf.constant(setup.rendering.background.color, dtype=tf.float32)]*len(seq_cams)
				if isinstance(background, list):
					bkg_render = background[i]
				if isinstance(background, (np.ndarray, tf.Tensor)):
					bkg_render = background
				#sim_transform.set_data(state.density)
				dens_imgs = []
				for state in states:
					if isinstance(state, State):
						dens_imgs.append(tf.concat(renderer.render_density_SDF_switch(state.get_density_transform(), lights, seq_cams, background=bkg_render, split_cameras=split_cams), axis=0))
					elif isinstance(state, (list, tuple)):
						state = tf.concat(state, axis=0)
					if isinstance(state, (np.ndarray, tf.Tensor)):
						state_shape = shape_list(state)
						if len(state_shape)!=4 or state_shape[0]!=len(seq_cams):
							raise ValueError
						if state_shape[-1]==1:
							state = tf.tile(state, (1,1,1,3))
						dens_imgs.append(tf.identity(state))
					
				renderer.write_images([tf.concat(dens_imgs, axis=-2)], [name_pre + '_cmp_dens_cam{}_{:04d}'], base_path=path, use_batch_id=True, frame_id=i, format=image_format)
				
				# velocity: [orig, veldens_warp]
				if render_velocity:
					vel_imgs = []
					if args.console: progress_bar(i*2+1,len(sequence)*2, "Step {:03d}/{:03d}: {:30}".format(i+1,len(sequence), "Sequence cmp Velocity"), length=30)
					for state in states:
						if isinstance(state, State):
							vel_transform = state.get_velocity_transform()
							vel_scale = vel_transform.cell_size_world().value 
							log.debug("Render velocity frame %d with cell size %s", i, vel_scale)
							vel_centered = state.velocity.centered() * get_vel_scale_for_render(setup, vel_transform)#vel_scale/float(setup.data.step)*setup.rendering.velocity_scale
							vel_imgs.append(tf.concat(vel_renderer.render_density(vel_transform, [tf.abs(vel_centered)], cameras, split_cameras=split_cams), axis=0))
					vel_renderer.write_images([tf.concat(vel_imgs, axis=-2)], [name_pre + '_cmp_velA_cam{}_{:04d}'], base_path=path, use_batch_id=True, frame_id=i, format=image_format)
				
				i+=1
		
		def get_frame_stats(state, mask=None, cmp_vol_targets=False):
			stats = {}
			try:
				with profiler.sample("stats"):
					dens_stats, vel_stats, tar_stats = state.stats(render_ctx=main_render_ctx, dt=1.0, order=setup.training.velocity.warp_order, clamp=setup.training.density.warp_clamp)#vel_scale)
					vel_stats['scale'] = world_scale(state.velocity.centered_shape, width=1.)
					if mask is not None:
						dens_hull_stats, vel_hull_stats, _ = state.stats(mask=mask, dt=1.0, order=setup.training.velocity.warp_order, clamp=setup.training.density.warp_clamp)
						vel_hull_stats['scale'] = world_scale(state.velocity.centered_shape, width=1.)
					
				
				if cmp_vol_targets:
					vTar_dens_stats, vTar_vel_stats = state.stats_target(dt=1.0, order=setup.training.velocity.warp_order, clamp=setup.training.density.warp_clamp)
					if mask is not None:
						if mask.dtype!=tf.bool:
							mask = tf.not_equal(mask, 0)
						vTar_dens_hull_stats, vTar_vel_hull_stats = state.stats_target(mask=mask, dt=1.0, order=setup.training.velocity.warp_order, clamp=setup.training.density.warp_clamp)
					
					
					if state.has_density_target:
						stats["vTar_density"]=vTar_dens_stats
						dens_SE = (state.density_target.d - state.density.d)**2
						dens_stats['_vTar_SE'] = tf_tensor_stats(dens_SE, as_dict=True)
						if mask is not None:
							stats["vTar_density_hull"]=vTar_dens_hull_stats
							dens_hull_stats['_vTar_SE'] = tf_tensor_stats(tf.boolean_mask(dens_SE, mask), as_dict=True)
					if state.has_velocity_target:
						stats["vTar_velocity"]=vTar_vel_stats
						vel_diffMag = (state.velocity_target - state.velocity).magnitude()
						vel_CangleRad_mask = tf.greater(state.velocity_target.magnitude() * state.velocity.magnitude(), 1e-8)
						vel_CangleRad = tf_angle_between(state.velocity_target.centered(), state.velocity.centered(), axis=-1, keepdims=True)
						vel_stats['_vTar_vdiff_mag'] = tf_tensor_stats(vel_diffMag, as_dict=True)
						vel_stats['_vTar_angleCM_rad'] = tf_tensor_stats(tf.boolean_mask(vel_CangleRad, vel_CangleRad_mask), as_dict=True)
						if mask is not None:
							stats["vTar_velocity_hull"]=vTar_vel_hull_stats
							vel_hull_stats['_vTar_vdiff_mag'] = tf_tensor_stats(tf.boolean_mask(vel_diffMag, mask), as_dict=True)
							vel_hull_stats['_vTar_angleCM_rad'] = tf_tensor_stats(tf.boolean_mask(vel_CangleRad, tf.logical_and(mask, vel_CangleRad_mask)), as_dict=True)
				
				stats["density"]=dens_stats
				stats["velocity"]=vel_stats
				stats["target"]=tar_stats
				if mask is not None:
					stats["density_hull"]=dens_hull_stats
					stats["velocity_hull"]=vel_hull_stats
			except:
				log.exception("Exception during reconstruction stats of frame %d", state.frame)
			return stats
			
		if setup.data.randomize>0: #args.fit and 
			log.info('Setting validation data for evaluation.')
			for state in sequence:
				state.clear_cache()
				state_set_targets(state, val_sequence if args.fit else eval_sequence)
				state_randomize(state, disable_transform_reset=True) #reset randomization
				state.input_view_mask = setup.validation.input_view_mask
			
		if velocity_decoder_model is not None:
			if density_decoder_model is not None:
				log.info("Set sequence densities for neural globt.")
				sequence.set_density_for_neural_globt(order=setup.training.velocity.warp_order, clamp=('NONE' if setup.training.density.warp_clamp=='NEGATIVE' else setup.training.density.warp_clamp), device=resource_device)
				sequence.clear_cache()
		
		if setup.validation.stats:
			log.info("Data Statistics")
			stats_file = os.path.join(setup.paths.log, "stats.json")
			stats_dict = {}
			frame_keys = []
			for state in sequence:
				frame_key = "{:04d}".format(state.frame)
				frame_keys.append(frame_key)
				stats_mask = state.density.hull #tf.greater(state.density.hull, 0.5)
				stats_dict[frame_key] = get_frame_stats(state=state, mask=state.density.hull, cmp_vol_targets=setup.validation.cmp_vol_targets)
			
			try:
				json_dump(stats_file, stats_dict, compressed=True, default=tf_to_dict, sort_keys=True)
			except:
				log.exception("Failed to write stats:")
			del stats_dict
		
		
	if args.render:
		try:
			log.info('Render final output.')
			
			
			render_sequence(sequence, z, cycle=setup.validation.render_cycle, cycle_steps=setup.validation.render_cycle_steps, \
				sF_cam=setup.validation.render_target, \
				render_density=setup.validation.render_density, render_shadow=setup.validation.render_shadow, \
				render_velocity=setup.validation.render_velocity, render_MS=setup.validation.render_MS, \
				slices = ['X','Y','Z'])
		except KeyboardInterrupt:
			log.warning("Interrupted final output rendering.")
		except:
			log.exception("Error during final output rendering:")
			
		
	if args.save_volume:
		try:
			log.info('Saving volumes.')
			sequence.save()
		except KeyboardInterrupt:
			log.warning("Interrupted saving volumes.")
		except:
			log.exception("Error during saving volumes:")
		
		#render_sequence(sequence, vel_scale, z, cycle=True, cycle_steps=12, sF_cam=True)
	
	used_mem = tf.contrib.memory_stats.MaxBytesInUse().numpy().tolist()
	max_mem = tf.contrib.memory_stats.BytesLimit().numpy().tolist()
	log.info('GPU memory usage: max: %d MiB (%.02f%%), limit: %d MiB', \
		used_mem/(1024*1024), (used_mem/max_mem)*100.0, max_mem/(1024*1024))
		
	with open(os.path.join(setup.paths.log, 'profiling.txt'), 'w') as f:
		profiler.stats(f)
	#profiler.stats()
	log.info("DONE")
	logging.shutdown()
	sys.exit(0)
