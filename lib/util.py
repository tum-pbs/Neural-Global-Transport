import os, sys, copy, datetime, math, string ,io
import re, json
import numpy as np
import collections.abc, numbers
import logging, zipfile
from contextlib import contextmanager
from lib.scalar_schedule import _get_base_schedule
from .archiving import json_dump, json_load, archive_files

LOG = logging.getLogger("Util")

def NO_OP(*args, **kwargs):
	pass

# e.g. for https://stackoverflow.com/questions/27803059/conditional-with-statement-in-python
class NO_CONTEXT():
	def __enter__(self):
		return None
	def __exit__(self, exc_type, exc_value, traceback):
		return None


def is_None_or_type(value, types):
	return (value is None) or isinstance(value, types)

#https://stackoverflow.com/questions/27779677/how-to-format-elapsed-time-from-seconds-to-hours-minutes-seconds-and-milliseco
def format_time(t):
	'''t (float): time in seconds'''
	h, r = divmod(t, 3600)
	m, s = divmod(r, 60)
	return '{:02d}:{:02d}:{:06.3f}'.format(int(h), int(m), s)

def time_unit(t, m=1000.):
	'''t (float): time in seconds'''
	units = ['us', 'ms', 's', 'm', 'h', 'd']
	x = [1e-6, 1e-3, 1.0, 60.0, 3600.0, 3600.0*24.0]
	for d, u in zip(x, units):
		if t*d < m or u==units[-1]:
			return '{:.3f}{}'.format(t*d, u)

def byte_unit(b, m=1024.):
	'''b: size in byte'''
	units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
	x = [1024**_ for _ in range(len(units))]
	for d, u in zip(x, units):
		if b/d < m or u==units[-1]:
			return '{:.3f}{}'.format(b/d, u)

class StringWriter:
	def __init__(self):
		self.__s = []
	
	def write(self, s):
		self.__s.append(s)
	
	def write_line(self, s=''):
		if s:
			self.__s.append(s)
		self.__s.append('\n')
	
	def flush(self):
		pass
	
	def reset(self):
		self.__s = []
	
	def get_string(self):
		return ''.join(self.__s)

class PartialFormatter(string.Formatter):
	#https://stackoverflow.com/questions/3536303/python-string-format-suppress-silent-keyerror-indexerror/21754294
	class Unformatted:
		def __init__(self, key):
			self.key = key
		def format(self, format_spec):
			return "{{{}{}}}".format(self.key, ":" + format_spec if format_spec else "")
	def vformat(self, format_string, args, kwargs):
		return super().vformat(format_string, args, kwargs)
	def get_value(self, key, args, kwargs):
		if isinstance(key, int):
			try:
				return args[key]
			except IndexError:
				return PartialFormatter.Unformatted(key)
		else:
			try:
				return kwargs[key]
			except KeyError:
				return PartialFormatter.Unformatted(key)
	def format_field(self, value, format_spec):
		if isinstance(value, PartialFormatter.Unformatted):
			return value.format(format_spec)
		else:
			return format(value, format_spec)

def get_timestamp():
	now = datetime.datetime.now()
	return now, now.strftime("%y%m%d-%H%M%S")

def check_prefix_any(path, prefix):
	if isinstance(prefix, str):
		return path.name.startswith(prefix)
	for pre in prefix:
		if path.name.startswith(pre):
			return True
	return False

def query_strings(s_list, r_list, name="directories"):
	for i, s in zip(range(len(s_list)), s_list):
		print('{: 4d}: {}'.format(i,s))
	print('space-seprated list: \'ID\', \'START-END\', \'START-END-STEP\'. (END is exclusive)')
	ids = input('Choose {}: '.format(name)).split(' ')
	r = []
	for id in ids:
		if '-' in id:
			id = [int(_) for _ in id.split('-')]
			r += list(range(*id))
		else:
			r.append(int(id))
	return [r_list[_] for _ in r]

#https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def progress_bar(curr, total, desc='', length=100, decimals=2):
	percent = ('{:.'+str(int(decimals))+'f}%').format(curr/total *100.0)
	filled = int(length * curr/total)
	bar = '='*filled + '-'*(length-filled)
	sys.stdout.write('[{}] ({}) {}\r'.format(bar, percent, desc))
	if curr>=total:
		sys.stdout.write('\n')
	sys.stdout.flush()

#prefix_voldifrender = ['recon_vol_', 'recon_den_', 'recon_seq_', 'recon_vel_', 'render_vol_', 'render_seq_', 'cmp_vol_', 'gen_vol_']
prefix_voldifrender = ['recon_vol', 'recon_den', 'recon_seq', 'recon_vel', 'render_vol', 'render_seq', 'cmp_vol', 'gen_vol']
class RunIndex:
	class RunEntry:
		def __init__(self, prefix, runid, title, path):
			self.runid = runid
			self.title = title
			self.prefix = prefix
			self.path = os.path.abspath(path)
			self.dirname = os.path.basename(self.path)
			self.parentdir = os.path.dirname(self.path)
			self.exists = os.path.exists(self.path)
			self.is_archive = os.path.isfile(self.path) and self.path.endswith(".zip")
			self._frame_paths = None
			self._frames = None
			self._config = None
			self._scene = None
			self._scalars = None
			self._warp_errors = None
			self._stats = None
		def rel_path(self, *subpaths):
			return os.path.join(self.path, *subpaths)
		@property
		def frames(self):
			if self._frames is None:
				self._frames = [k for k in sorted(self.frame_paths().keys())]
			return copy.copy(self._frames)
		def frame_path(self, frame):
			self.frame_paths() #build frame paths index
			if not frame in self._frame_paths:
				raise KeyError("Frame {} not in run {}".format(frame, self.runid))
			else:
				return self._frame_paths[frame]
		def frame_paths(self):
			if self._frame_paths is None:
				frame_dir_mask = re.compile(r"^frame_(?P<frame>\d+)")
				self._frame_paths = {}
				for e in os.scandir(self.path):
					m = frame_dir_mask.search(e.name)
					if m is not None:
						self._frame_paths[int(m.group("frame"))] = e
			return self._frame_paths
		def frame_paths_list(self):
			fp = self.frame_paths()
			return [fp[k] for k in sorted(fp.keys())]
		@contextmanager
		def _open_file(self, rel_path, mode="r"):
			# run can be archive (self.is_archive)
			# json can be zipped
			#https://stackoverflow.com/questions/12025469/how-to-read-from-a-zip-file-within-zip-file-in-python
			if not mode in ["r", "rb"]:
				raise ValueError("Only reading is supported (r, rb).")
			if self.is_archive:
				run_archive = zipfile.ZipFile(self.path, "r")
				def __open(path, mode="r"):
					if mode=="r":
						return run_archive.open(path)
					elif mode=="rb":
						return io.BytesIO(run_archive.read(path))
				def __isfile(path):
					#return zipfile.Path(run_archive, path).is_file()
					try:
						run_archive.getinfo(path)
					except KeyError:
						return False
					else:
						return True
			else:
				def __open(path, mode="r"):
					return open(os.path.join(self.path,path), mode=mode)
				def __isfile(path):
					return os.path.isfile(os.path.join(self.path,path))
					
			is_archive_file = False
			if not __isfile(rel_path):
				if __isfile(rel_path+".zip"):
					is_archive_file = True
					archive_file = __open(rel_path+".zip", mode="rb")
					file_archive = zipfile.ZipFile(archive_file, "r")
					if mode=="r":
						file = file_archive.open(os.path.basename(rel_path))
					elif mode=="rb":
						file = io.BytesIO(file_archive.read(os.path.basename(rel_path)))
				else:
					if self.is_archive:
						run_archive.close()
					raise IOError("'%s' is not a File."%rel_path)
			else:
				file = __open(rel_path, mode)
			
			try:
				yield file
			finally:
				file.close()
				if is_archive_file:
					file_archive.close()
					archive_file.close()
				if self.is_archive:
					run_archive.close()
		@property
		def config(self):
			if self._config is None:
				try:
					with self._open_file(os.path.join("config", "setup.json")) as f: #open(self.rel_path("config", "setup.json"), "r")
						self._config = json.load(f)
				except:
					LOG.exception("Failed to read setup from run %s", self.runid)
					return None
			return copy.deepcopy(self._config)
		@property
		def setup(self):
			return self.config
		@property
		def log_file(self):
			return open(self.rel_path("log", "logfile.log"), "r")
		@property
		def log_file_relpath(self):
			return os.path.join("log", "logfile.log")
		@property
		def scene(self):
			if self._scene is None:
				try:
					with self._open_file(os.path.join("config", "scene.json")) as f:
						self._scene = json.load(f)
				#	self._warp_errors = json_load(self.rel_path("config", "scene.json"))
				except:
					LOG.exception("Failed to read scene from run %s", self.runid)
					return None
			return copy.deepcopy(self._scene)
		@property
		def scalars(self):
			if self._scalars is None:
				try:
					with self._open_file("scalar_results.json") as f:
						self._scalars = json.load(f)
				except:
					LOG.exception("Failed to read scalar results from run %s", self.runid)
					return dict()
			return copy.deepcopy(self._scalars)
		@property
		def warp_errors(self):
			if self._warp_errors is None:
				try:
					with self._open_file(os.path.join("warp_test","warp_error.json")) as f:
						self._warp_errors = json.load(f)
				#	self._warp_errors = json_load(self.rel_path("warp_test", "warp_error.json"))
				except:
					LOG.exception("Failed to read warp errors from run %s", self.runid)
					return dict()
			return copy.deepcopy(self._warp_errors)
		@property
		def stats(self):
			if self._stats is None:
				try:
					with self._open_file(os.path.join("log","stats.json")) as f:
						self._stats = json.load(f)
				#	self._stats = json_load(self.rel_path("log", "stats.json"))
				except:
					LOG.exception("Failed to read stats from run %s", self.runid)
					return dict()
			return copy.deepcopy(self._stats)
	@staticmethod
	def get_new_run_id():
		now = datetime.datetime.now()
		now_str = now.strftime("%y%m%d-%H%M%S")
		return now_str
	@staticmethod
	def is_run_id(run_id):
		if not isinstance(run_id, str):
			return False
		# format: %y%m%d-%H%M%S
		runid_mask = re.compile(r"\d{6}-\d{6}")
		if len(run_id)==13 and runid_mask.search(run_id) is not None:
			try:
				datetime.datetime.strptime(run_id, "%y%m%d-%H%M%S")
			except ValueError:
				pass
			else:
				return True
		# legacy format: %m%d-%H%M
	#	if len(run_id)==9 and run_id[4]=='-':
	#		try:
	#			datetime.datetime.strptime(run_id, "%m%d-%H%M")
	#		except:
	#			pass
	#		else:
	#			return True
		return False
	
	@staticmethod
	def parse_scalarFlow(s):
		m = re.compile(r"^\[SF:(?P<sim>-?\d+)(?::(?P<frame>-?\d+))?\](?P<relpath>.*)") # [SF:<sim>] or [SF:<sim>:<frame>], can also be negative offsets
		f = m.search(s)
		if f is None:
			return None
		return {
			"sim":int(f.group("sim")),
			"frame":int(f.group("frame")) if f.group("frame") is not None else None,
			"relpath":f.group("relpath"),
		} #frame may be unspecified
	
	def __init__(self, base_path_list, prefix=prefix_voldifrender, recursive=False):
		self.path_mask = re.compile(r"^(?P<prefix>" + "|".join(prefix)+ ")_(?P<runid>\d{6}-\d{6})_(?P<title>.*)")
		self.rel_path_mask = re.compile(r"^\[RUNID:(?P<runid>\d{6}-\d{6})\]/*(?P<relpath>.*)")
		self.run_paths = self._get_run_paths(base_path_list, prefix, recursive)
		self.run_entries = [RunIndex.RunEntry(*self.is_run_dir(os.path.basename(_))[1:], _) for _ in self.run_paths]
		self.run_entries.sort(key = lambda e: e.runid)
		self.runs = {_.runid:_ for _ in self.run_entries}
		if len(self.runs)!=len(self.run_entries):
			runids = [_.runid for _ in self.run_entries]
			duplicates = []
			for runid in runids:
				if runids.count(runid)>1 and runid not in duplicates:
					duplicates.append(runid)
			raise KeyError("Duplicate runids found in base paths {}: {}".format(base_path_list, duplicates))
	
	def _get_run_paths(self, base_path_list, prefix, recursive=False):
		run_paths = []
		for base_path in base_path_list:
			try:
				files = os.scandir(base_path)
			except OSError:
				LOG.warning("Failed to scan '%s' for runs.", base_path)
			else:
				dirs = [_ for _ in files if _.is_dir() or (_.is_file() and _.name.endswith(".zip"))]
				#run_paths += [_.path for _ in dirs if check_prefix_any(_, prefix) and _.path not in run_paths]
				run_paths += [_.path for _ in dirs if self.is_run_dir(_.name)[0] and _.path not in run_paths]
				#run_paths += [_.path for _ in dirs if self.is_run_dir(_.name)[0] and _.path not in run_paths]
				if recursive:
					run_paths.extend(self._get_run_paths([_.path for _ in dirs if _.path not in run_paths and _.is_dir()], prefix, recursive=recursive))
				
		return run_paths
	
	def __getitem__(self, run_id):
		if self.is_run_relative_path(run_id):
			return self.get_run_relative_path(run_id)
		elif RunIndex.is_run_id(run_id):
			return self.get_run_path(run_id)
		return None
		
	@property
	def size(self):
		return len(self.run_paths)
	def __len__(self):
		return self.size
	
	def is_run_dir(self, dir):
		f = self.path_mask.search(dir)
		if f is not None:
			return True, f.group(1), f.group(2), f.group(3)
		else:
			return False, "", "000000-000000", ""
	
	def get_run_id(self, arg):
		if self.is_run_id(arg):
			return arg
		if self.is_run_relative_path(arg):
			return self.rel_path_mask.search(arg).group("runid")
		if self.is_run_dir(arg)[0]:
			return self.is_run_dir(arg)[2]
		raise ValueError("Can't get runid from '{}'".format(arg))
	
	def can_get_run_id(self, arg):
		try:
			self.get_run_id(arg)
		except ValueError:
			return False
		else:
			return True
	
	def get_run_entry(self, arg):
		runid = self.get_run_id(arg)
		if runid not in self.runs:
			raise KeyError("Runid {} not found.".format(runid))
		return self.runs[runid]
	
	def is_run_path(self, path):
		return os.path.isdir(path) and self.is_run_dir(os.path.basename(path))[0]
		
	def get_run_path(self, run_id):
		'''
		run_id: timestamp of the run, e.g. '0714-1042'
		run_paths: output of get_run_paths()
		'''
		if not self.is_run_id(run_id):
			raise ValueError("'{}' is not runid.".format(run_id))
		if run_id not in self.runs:
			raise KeyError("Runid {} not found.".format(run_id))
		return copy.deepcopy(self.runs[run_id].path)
	#	mask1 = '_{}_'.format(run_id)
	#	mask2 = '_{}'.format(run_id)
	#	paths = [_ for _ in self.run_paths if mask1 in _ or _.endswith(mask2)]
	#	if len(paths)==0:
	#		raise ValueError('Could not find path for run ID {}'.format(run_id))
	#	if len(paths)>1:
	#		raise ValueError("Run ID '{}' is not unique: {}", run_id, paths)
	#	return paths[0]
	
	def is_run_relative_path(self, rel_path):
		rel_match = self.rel_path_mask.search(rel_path)
		return rel_match is not None
	#	id_end = rel_path.find(']')
	#	return rel_path.startswith('[RUNID:') and not id_end==-1
		
	def get_run_relative_path(self, rel_path):
		'''
		rel_path = [RUNID:<runid>]<relpath>
		'''
		rel_match = self.rel_path_mask.search(rel_path)
		if rel_match is None:
			raise ValueError("'{}' is not a run relative path.".format(rel_path))
		runid = rel_match.group("runid")
		if runid not in self.runs:
			raise KeyError("Runid {} not found.".format(runid))
		return self.runs[runid].rel_path(rel_match.group("relpath"))
	#	id_end = rel_path.find(']')
	#	if not rel_path.startswith('[RUNID:') or id_end==-1:
	#		raise ValueError("Malformed rel_path: {}".format(rel_path))
	#	run_id = rel_path[7:id_end]
	#	sub_path = rel_path[id_end+1:]
	#	return os.path.join(self.get_run_path(run_id), sub_path)
	
	def query_runs(self):
		return query_strings(["[" + _.runid + "] " + _.title for _ in self.run_entries], self.run_entries, name="runs")
	
	def query_paths(self):
		sorted_paths = sorted(self.run_paths, key=lambda p: os.path.basename(p))
		return query_strings([os.path.basename(_) for _ in sorted_paths], sorted_paths)

# load ndarray from different formats (.npy, .npz)
# if .npy (single ndarray):
#	return loaded array
# if .npz (dict of ndarray):
#	if name!=None: return element indicated by the name
#	if single element and name==None: return that element
#	if multiple elements and name==None: return dict of elements as loaded
def load_numpy(path, name=None):
	data = np.load(path)
	if isinstance(data, np.ndarray):
		#if name is not None:
		#	raise KeyError("Key '{}' was provided but load did not return a dictionary.".format(name))
		return data
	if isinstance(data, collections.abc.Mapping):
		if name is None:
			if len(data)<2: r = data[list(data.keys())[0]]
			else: r = dict(data)
		else:
			try:
				r = data[name]
			except KeyError:
				k = list(data.keys())
				data.close()
				raise KeyError("Key '{}' not in loaded dict. Available keys: {}".format(name, k))
		data.close()
		return r
	else:
		raise TypeError("np.load did not return an expected type.")

def lerp(a,b,t):
	return (1-t)*a + t*b
def lerp_fast(a,b,t):
	return a + t*(b-a)

def lerp_list(A,B,t):
	return [(1-t)*a + t*b for a,b in zip(A,B)]

def print_stats(data, name, log=None):
	max, min, mean, abs_mean = np.max(data), np.min(data), np.mean(data), np.mean(np.abs(data))
	if log is None:
		print('{} stats: min {:.010e}, max {:.010e}, mean {:.010e}, abs-mean {:.010e}'.format(name, min, max, mean, abs_mean))
	else:
		log.info('{} stats: min {:.010e}, max {:.010e}, mean {:.010e}, abs-mean {:.010e}'.format(name, min, max, mean, abs_mean))
	return max, min, mean
	

def makeNextGenericPath(dir, folder_no = 0, path_mask = '%s_%04d'):
	dir = os.path.normpath(dir)
	test_path = path_mask % (dir, folder_no)
	while os.path.exists(test_path):
		folder_no += 1
		test_path = path_mask % (dir, folder_no)
	#print("Using %s dir '%s'" % (dir, test_path) )
	os.makedirs(test_path)
	return test_path, folder_no
	
def get_nested_keys(d, sep="/"):
	ks = []
	for k, v in d.items():
		if isinstance(v, collections.abc.Mapping):
			ks.extend(k+sep+_ for _ in get_nested_keys(v))
		else:
			ks.append(k)
	return ks

#https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
def update_dict_recursive(d, u, deepcopy=False, new_key='KEEP', **kwargs):
	if 'allow_new' in kwargs:
		new_key = 'KEEP' if kwargs['allow_new'] else 'ERROR'
	if deepcopy:
		d = copy.deepcopy(d)
	parent = kwargs.get("_parent", "")
	for k, v in u.items():
		if not k in d:
			if new_key.upper()=='ERROR':
				raise KeyError("Update key {} of {} does not exisit in base dictionary.".format(k, parent))
			elif new_key.upper()=='DISCARD':
				continue
			elif new_key.upper()=='DISCARD_WARN':
				LOG.warning("update_dict_recursive: Discard new key %s of %s", k, parent)
				continue
			elif new_key.upper()=='KEEP':
				pass
			elif new_key.upper()=='KEEP_WARN':
				LOG.warning("update_dict_recursive: Keep new key %s of %s", k, parent)
			else:
				raise ValueError("Unknown policy for new key: {}".format(new_key))
		if isinstance(v, collections.abc.Mapping):
			if k in d and not isinstance(d[k], collections.abc.Mapping):
				# if something that is not a Mapping is updated with a Mapping
				#e.g. a default constant schedule (int, float, list) with a complex one (dict {"type":<type>, ...})
				if isinstance(d[k], (numbers.Number, list)) and "type" in v and isinstance(v["type"], str) and v["type"].upper() in ['SCHEDULE', 'CONST','LINEAR','EXPONENTIAL','ROOT_DECAY'] and "start" in v:
					LOG.info("update_dict_recursive: Replaced %s with base schedule for further updates.", k)
					d[k] = _get_base_schedule()
				else:
					d[k] = {}
			d[k] = update_dict_recursive(d.get(k, {}), v, deepcopy=deepcopy, new_key=new_key, _parent=parent+"."+k)
		else:
			if deepcopy:
				d[k] = copy.deepcopy(v)
			else:
				d[k] = v
	return d

def copy_nested_structure(structure): #, include_paths=None, exclude_paths=None):
	'''
		copy only the (nested) structure (list, tuple, dict)
		the items/contents are not copied
	'''
	if isinstance(structure, list):
		return list(copy_nested_structure(_) for _ in structure)
	if isinstance(structure, tuple):
		return tuple(copy_nested_structure(_) for _ in structure)
	if isinstance(structure, dict):
		return {k: copy_nested_structure(_) for k, _ in structure.items()}
	return structure

def setup_compatibility(setup):
	assert isinstance(setup, dict)
	if 'discriminator' in setup['training']:
		disc_setup = setup['training']['discriminator']
		if 'lr_scale' in disc_setup and 'learning_rate' not in disc_setup:
			disc_setup['learning_rate'] = disc_setup['lr_scale'] * setup['training']['learning_rate']
			disc_setup['lr_decay'] = setup['training']['lr_decay']
			disc_setup['lr_schedule'] = [float('-inf'),float('+inf'), 0.0, 0]
			del disc_setup['lr_scale']

def abs_grow_intervals(intervals, max_iterations):
	abs_intervals = []
	last = 0
	for interval in intervals:
		abs_intervals.append((last, last + interval))
		last += interval
	abs_intervals.append((last, max_iterations))
	return abs_intervals

def current_grow_shape(base_shape, iteration, factor, grow_intervals, cast_fn=round):
	interval = 0
	grow_steps = len(grow_intervals)
	idx = 0
	for i in range(grow_steps):
		interval += grow_intervals[i]
		if iteration<interval:
			break
		idx +=1
	scale = 1./(factor**(grow_steps-idx))
	return [int(cast_fn(_*scale)) for _ in base_shape]

def all_grow_shapes(base_shape, factor, intervals, cast_fn=round):
	abs_intervals = abs_grow_intervals(intervals, max_iter)
	return [current_grow_shape(base_shape, interval[0], factor, intervals, cast_fn=cast_fn) for interval in abs_intervals]

def num_grow_levels(intervals):
	return len(intervals) + 1

def start_grow_level(num_levels, intervals):
	return max(num_grow_levels(intervals) - num_levels, 0)

def current_grow_step(iteration, grow_intervals):
	interval = 0
	grow_steps = len(grow_intervals)
	idx = 0
	for i in range(grow_steps):
		interval += grow_intervals[i]
		if iteration<interval:
			break
		idx +=1
	return idx

def current_grow_level(num_levels, iteration, grow_intervals):
	return current_grow_step(iteration, grow_intervals) + start_grow_level(num_levels, grow_intervals)
	

# TODO
class GrowHandler:
	class GrowSchedule:
		class GrowInterval:
			def __init__(self, start, end, value, end_exclusive=True):
				# end is exclusive
				assert isinstance(start, numbers.Integral)
				assert isinstance(end, numbers.Integral)
				assert (start<end and end_exclusive) or (start<=end and not end_exclusive) or (start==0 and end==0)
				self.start = start
				self.end = end
				self._value = copy.copy(value)
				self.end_exclusive = end_exclusive
			@property
			def value(self):
				return copy.copy(self._value)
			def __contains__(self, key):
				assert isinstance(key, numbers.Integral)
				return self.start<=key and (key<self.end if self.end_exclusive else key<=self.end)
			def interval_equal(self, other):
				if not isinstance(other, GrowHandler.GrowSchedule.GrowInterval): return False
				return self.start==other.start and self.end==other.end and self.end_exclusive==other.end_exclusive
			def __eq__(self, other):
				return self.interval_equal(other) and self.value==other.value
			def __repr__(self):
				return "[{start:d},{end:d}] {value}".format(start=self.start, end=(self.end-1 if self.end_exclusive else self.end),value=self.value)
		
		def __init__(self, grow_intervals=[]):
			
			self.intervals = grow_intervals
		
		
		@classmethod
		def make_shape_schedule(cls, base_shape, factor, intervals, max_iterations, cast_fn=round):
			grow_intervals = []
			last = 0
			grow_steps = len(intervals)
			for i, interval in enumerate(intervals):
				
				scale = 1./(factor**(grow_steps-i))
				shape = [int(cast_fn(_*scale)) for _ in base_shape]
				
				grow_intervals.append(GrowHandler.GrowSchedule.GrowInterval(start=last, end=last+interval, value=shape))
				last += interval
				
			grow_intervals.append(GrowHandler.GrowSchedule.GrowInterval(start=last, end=max(max_iterations, last+1), value=base_shape))
			
			return cls(grow_intervals)
		
		@classmethod
		def make_level_schedule(cls, max_level, step, intervals, max_iterations):
			grow_intervals = []
			last = 0
			grow_steps = len(intervals)
			for i, interval in enumerate(intervals):
				
				level = max_level - (grow_steps-i)*step
				
				grow_intervals.append(GrowHandler.GrowSchedule.GrowInterval(start=last, end=last+interval, value=level))
				last += interval
				
			grow_intervals.append(GrowHandler.GrowSchedule.GrowInterval(start=last, end=max(max_iterations, last+1), value=max_level))
			
			return cls(grow_intervals)
		
		def __getitem__(self, key):
			if not isinstance(key, numbers.Integral): raise KeyError
			for interval in self.intervals:
				if key in interval: return interval
			if key<self.intervals[0].start:
				return self.intervals[0]
			if key>=self.intervals[-1].end:
				return self.intervals[-1]
			raise KeyError
		
		def index(self, key):
			if not isinstance(key, numbers.Integral): raise KeyError
			for i, interval in enumerate(self.intervals):
				if key in interval: return i
			if key<self.intervals[0].start:
				return 0
			if key>=self.intervals[-1].end:
				return len(self.intervals)-1
			raise KeyError
		
		def get_intervals_to(self, key):
			max_interval = self.index(key)
			return self.intervals[:max_interval+1]
		
		def intervals_equal(self, other):
			if not isinstance(other, GrowHandler.GrowSchedule): return False
			if len(self.intervals)!=len(other.intervals): return False
			return all(si.interval_equal(oi) for si,oi in zip(self.intervals, other.intervals))
		
		@property
		def min_value(self):
			return self.intervals[0].value
		@property
		def max_value(self):
			return self.intervals[-1].value
			
		
		def __eq__(self, other):
			if not isinstance(other, GrowHandler.GrowSchedule): return False
			if len(self.intervals)!=len(other.intervals): return False
			return all(si==oi for si,oi in zip(self.intervals, other.intervals))
		
		def __repr__(self):
			return "\n".join("\t"+str(_) for _ in self.intervals)
	
	def __init__(self, base_shape, base_cam_shape, max_dens_level, max_vel_level, setup, shape_cast_fn=round, **kwargs):
		self.seed=0
		self._rng = np.random.RandomState(self.seed)
		self._shape_cast_fn = shape_cast_fn
		#self.min_scale = 6
		assert isinstance(base_shape, (list, tuple)) and len(base_shape)==3
		self.base_shape = list(base_shape)
		self.current_dens_shape = list(base_shape)
		self.current_vel_shape = list(base_shape)
		self.current_max_shape = list(base_shape)
		self.min_shape = [6,12,6]
		self.MS_scale_factor = setup.training.velocity.decoder.recursive_MS_scale_factor
		self.MS_min_grid_res = setup.training.velocity.decoder.min_grid_res
		
		self.is_randomize_shape = False
		self.is_random_shape_from_intervals = False
		self.is_iterate_shapes = False
		if setup.training.randomization.grow_mode=="RAND":
			self.is_randomize_shape = True
		if setup.training.randomization.grow_mode=="RANDINTERVAL":
			self.is_randomize_shape = True
			self.is_random_shape_from_intervals = True
		if setup.training.randomization.grow_mode=="ITERATE":
			self.is_iterate_shapes = True
		self.is_velocity_MS_variable = setup.training.velocity.decoder.recursive_MS_shared_model
		
		self.main_dens_schedule = GrowHandler.GrowSchedule.make_shape_schedule(base_shape=base_shape, \
			factor=setup.training.density.grow.factor, intervals=setup.training.density.grow.intervals, max_iterations=setup.training.iterations)
		self.main_dens_level_schedule = GrowHandler.GrowSchedule.make_level_schedule(max_level=max_dens_level, step=1, \
			intervals=setup.training.density.grow.intervals, max_iterations=setup.training.iterations)
		
		self.main_cam_schedule = GrowHandler.GrowSchedule.make_shape_schedule(base_shape=base_cam_shape, \
			factor=kwargs.get("base_cam_factor",setup.training.density.grow.factor), \
			intervals=setup.training.density.grow.intervals, max_iterations=setup.training.iterations)
		
		self.main_vel_schedule = GrowHandler.GrowSchedule.make_shape_schedule(base_shape=base_shape, \
			factor=setup.training.velocity.grow.factor, intervals=setup.training.velocity.grow.intervals, max_iterations=setup.training.iterations)
		self.main_vel_level_schedule = GrowHandler.GrowSchedule.make_level_schedule(max_level=max_vel_level, step=1, \
			intervals=setup.training.velocity.grow.intervals, max_iterations=setup.training.iterations)
		
		# assert: dens and vel shapes must match in main
		#assert self.main_dens_schedule==self.main_vel_schedule, "main opt density and velocity grow schedules do not match"
		assert self.main_dens_schedule.intervals_equal(self.main_cam_schedule), "main opt density and camera grow schedules do not match"
		assert self.main_vel_schedule.intervals_equal(self.main_vel_level_schedule), "main opt velocity shape and level grow intervals do not match"
		
		if setup.training.density.pre_optimization:
			raise NotImplementedError
			self.pre_dens_schedule = GrowHandler.GrowSchedule.make_shape_schedule(base_shape=self.main_dens_schedule.min_value, \
				factor=setup.training.density.pre_opt.grow.factor, intervals=setup.training.density.pre_opt.grow.intervals, max_iterations=setup.training.density.pre_opt.iterations)
			self.pre_dens_level_schedule = GrowHandler.GrowSchedule.make_level_schedule(max_level=self.main_dens_level_schedule.min_value, step=1, \
				intervals=setup.training.density.pre_opt.grow.intervals, max_iterations=setup.training.density.pre_opt.iterations)
			self.pre_cam_schedule = GrowHandler.GrowSchedule.make_shape_schedule(base_shape=self.main_cam_schedule.min_value, \
				factor=setup.training.density.pre_opt.grow.factor, intervals=setup.training.density.pre_opt.grow.intervals, max_iterations=setup.training.density.pre_opt.iterations)
		
		if setup.training.velocity.pre_optimization:
			raise NotImplementedError
			self.pre_vel_schedule = GrowHandler.GrowSchedule.make_shape_schedule()
			self.pre_vel_level_schedule = GrowHandler.GrowSchedule.make_level_schedule()
			assert self.pre_vel_schedule.intervals_equal(self.pre_vel_level_schedule), "pre opt velocity shape and level grow intervals do not match"
	
	def _get_MS_scale_from_shape(self, shape, min_shape):
		i = 0
		while (min(self._shape_cast_fn(_/(self.MS_scale_factor**i)) for _ in shape)>=min_shape):
			i +=1
		return i-1
	
	def get_velocity_MS_scale_from_shape(self, shape):
		return self._get_MS_scale_from_shape(shape, self.MS_min_grid_res)
	
	def _lerp_shape(self, A,B,t):
		return [int(self._shape_cast_fn((1-t)*a + t*b)) for a,b in zip(A,B)]
	
	@property
	def current_cam_schedule(self):
		return self.pre_cam_schedule if self._is_pre_opt else self.main_cam_schedule
	
	def start_iteration(self, iteration, is_pre_opt=False, is_velocity_opt=False): #, is_first_opt=False
		#if is_pre_opt: raise NotImplementedError
		
		self._iteration = iteration
		self._is_pre_opt = is_pre_opt
		current_dens_schedule = None
		current_vel_schedule = None
		if is_pre_opt:
			raise NotImplementedError
			if is_velocity_opt:
				current_schedule = self.pre_vel_schedule
				current_level_schedule = self.pre_vel_level_schedule
			else:
				current_schedule = self.pre_dens_schedule
				current_level_schedule = self.pre_dens_level_schedule
			current_cam_schedule = self.pre_cam_schedule
		else:
			current_dens_schedule = self.main_dens_schedule
			current_dens_level_schedule = self.main_dens_level_schedule
			current_cam_schedule = self.main_cam_schedule
			current_vel_schedule = self.main_vel_schedule
			current_vel_level_schedule = self.main_vel_level_schedule
		
		self.current_max_shape = current_dens_schedule[self._iteration].value
		
		if self.is_randomize_shape:
			raise NotImplementedError("separate dens/vel schedules")
			if self.is_random_shape_from_intervals:
				available_intervals = current_schedule.get_intervals_to(self._iteration)
				current_interval_idx = self._rng.randint(len(available_intervals))
				self._active_iteration = current_interval_idx
				self.current_shape = available_intervals[current_interval_idx].value
				
				available_intervals = current_cam_schedule.get_intervals_to(self._iteration)
				self.current_cam_shape = available_intervals[current_interval_idx].value
				available_intervals = current_level_schedule.get_intervals_to(self._iteration)
				self.current_MS_scale = available_intervals[current_interval_idx].value
			else:
				self._active_iteration = None
				rng_scale = self._rng.random()
				min_shape = current_schedule.min_value
				self.current_shape = self._lerp_shape(min_shape, current_schedule[self._iteration].value, rng_scale) #uniformly random between min and current shape
				
				min_shape = current_cam_schedule.min_value
				self.current_cam_shape = self._lerp_shape(min_shape, current_cam_schedule[self._iteration].value, rng_scale) #uniformly random between min and current shape
				
				if self.is_velocity_MS_variable:
					self.current_MS_scale = self.get_velocity_MS_scale_from_shape(self.current_shape)
				else:
					raise NotImplementedError
		elif self.is_iterate_shapes:
			raise NotImplementedError("separate dens/vel schedules")
			available_intervals = current_schedule.get_intervals_to(self._iteration)
			current_interval_idx = self._iteration % len(available_intervals)
			self._active_iteration = current_interval_idx
			self.current_shape = available_intervals[current_interval_idx].value
			
			available_intervals = current_cam_schedule.get_intervals_to(self._iteration)
			self.current_cam_shape = available_intervals[current_interval_idx].value
			
			available_intervals = current_level_schedule.get_intervals_to(self._iteration)
			self.current_MS_scale = available_intervals[current_interval_idx].value
		else:
			self._active_iteration = self._iteration
			self.current_dens_shape = current_dens_schedule[self._iteration].value
			self.current_dens_MS_scale = current_dens_level_schedule[self._iteration].value
			self.current_cam_shape = current_cam_schedule[self._iteration].value
			self.current_vel_shape = current_vel_schedule[self._iteration].value
			self.current_vel_MS_scale = current_vel_level_schedule[self._iteration].value
		
	
	def get_density_shape(self):
		return self.current_dens_shape
	
	def get_current_max_shape(self):
		return self.current_max_shape
	
	def get_velocity_shape(self):
		return self.current_vel_shape
	
	def get_camera_shape(self):
		return self.current_cam_shape
		
	def get_camera_MS_shapes(self):
		if self.is_randomize_shape:
			if not self.is_random_shape_from_intervals:
				curr_shape = self.get_camera_shape()
				curr_scale = self.current_MS_scale
				intervals = GrowHandler.GrowSchedule\
					.make_shape_schedule(base_shape=curr_shape, factor=self.MS_scale_factor, intervals=[1]*curr_scale, max_iterations=curr_scale+1)\
					.get_intervals_to(curr_scale+1)
			else:
				intervals = self.current_cam_schedule.intervals[:self.current_MS_scale+1]
		else:
			intervals = self.current_cam_schedule.get_intervals_to(self._iteration)
		return [_.value for _ in intervals]
		
	def get_camera_MS_scale_shapes(self):
		shapes = self.get_camera_MS_shapes()
		return {i: shape for i, shape in enumerate(shapes)}
		
	def get_image_MS_scale_shapes(self):
		shapes = self.get_camera_MS_shapes()
		return {i: shape[1:] for i, shape in enumerate(shapes)}
	
	def get_velocity_MS_scale(self):
		return self.current_vel_MS_scale
	
	def get_density_MS_scale(self):
		return self.current_dens_MS_scale
	
	def __repr__(self):
		return "Main-opt:\n- density:\n{}\n- cameras:\n{}\n- velocity:\n{}\n- vel-levels:\n{}".format( \
			self.main_dens_schedule, self.main_cam_schedule, self.main_vel_schedule, self.main_vel_level_schedule)

# circular buffer of fixed size
# most of it still has to be tested...
class HistoryBuffer:
	def __init__(self, size):
		self.size = size
		self.buf = [None]*size
		self.head = 0
		self.elements = 0
	
	class HistoryBufferIterator:
		def __init__(self, buffer):
			self._buffer = buffer
			self._idx = 0
		def __next__(self):
			if self._idx<self._buffer.elements:
				r = self._buffer[idx]
				idx+=1
				return r
			else:
				raise StopIteration
	def __iter__(self):
		return HistoryBufferIterator(self)
	
	#https://en.wikipedia.org/wiki/Modulo_operation
	def _get_buf_index(self, idx):
		if idx<0:
			idx=self.elements+idx
		return (self.head-idx-1)%self.size
	def _move_head(self, steps=1):
		self.head = (self.head+steps)%self.size #_get_buf_index(steps)
	def __getitem__(self, index):
		if index>=self.elements or index<-self.elements:
			raise IndexError('Invalid position in buffer.')
		return self.buf[self._get_buf_index(index)]
	def __setitem__(self, index, value):
		if index>=self.elements or index<-self.elements:
			raise IndexError('Invalid position in buffer.')
		self.buf[self._get_buf_index(index)] = value
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
	def list(self):
		return [self[_] for _ in range(len(self))]
	# add a new element, if full the oldest (tail) or random (rand) will be overwritten 
	def push(self, element, replacement='tail'):
		if replacement.lower()=='tail' or self.elements<self.size:
			self.buf[self.head]=element
			self._move_head()
			if self.elements<self.size:
				self.elements +=1
		elif replacement.lower()=='rand':
			index = np.random.randint(self.elements)
			#index = int(round(np.random.triangular(0,0,self.elements-1)))
			self[index] = element
		else:
			raise ValueError('Unknown replacement strategy \'{}\''.format(replacement))
	def push_samples(self, samples, sample_chance=1.0, replacement='tail'):
		if sample_chance>=1.0:
			for sample in samples:
				self.push(sample, replacement)
		else:
			rands = np.random.random(len(samples))
			for sample, r in zip(samples, rands):
				if r<sample_chance: self.push(sample, replacement)
	
	# get a random valid element. does not remove the element
	def get(self):
		if self.empty:
			raise IndexError('Cant get element from empty buffer.')
		index = np.random.randint(self.elements)
		return self[index] #self.buf[idx]
	def get_samples(self, num_samples, replace=True, allow_partial=True):
		if num_samples<1:
			raise ValueError("Must request at least 1 sample.")
		if self.empty:
			if allow_partial:
				return []
			else:
				raise ValueError("Insufficient samples in buffer. size: {}, requested {}".format(self.elements, num_samples))
		if num_samples>self.elements and (not allow_partial) and (not replace):
			raise ValueError("Insufficient samples in buffer. size: {}, requested {}".format(self.elements, num_samples))
		if allow_partial and (not replace):
			num_samples = np.minimum(num_samples, self.elements)
		indices = np.random.choice(self.elements, num_samples, replace=replace)
		return [self[_] for _ in indices]
	
	def _pop_first(self):
		elem = self[0]
		self[0] = None
		self._move_head(-1)
		self.elements -=1
		return elem
	def _pop_last(self):
		elem = self[-1]
		self[-1] = None
		self.elements -=1
		return elem
	def pop(self, reverse=False):
		if reverse:
			return self._pop_last()
		else:
			return self._pop_first()
	
	def resize(self, new_size):
		raise NotImplementedError()
	def reset(self):
		del self.buf
		self.buf = [None]*self.size
		self.head = 0
		self.elements = 0
	
	def __str__(self):
		return 'HistoryBuffer: {}/{} (internal head at {}): {}'.format(self.elements, self.size, self.head, self.list)
		
	def all_ndarray(self):
		for elem in self:
			if not isinstance(elem, np.ndarray):
				return False
		return True
		
	def serialize(self, path, suffix=None):
		d = {
			"__class__":self.__class__.__name__,
			"__module__":self.__module__,
			"size":self.size,
			"head":self.head,
			"elements":self.elements,
			"ndarray_path":None,
			"buf":self.buf
		}
		name = "HBuffer"
		if suffix is not None:
			name = name + '_' + suffix
		nd_arrays = {}
		for i in range(self.size):
			elem = self.buf[i]
			if elem.__class__.__name__=='EagerTensor': #isinstance(elem, tf.EagerTensor):
				elem = elem.numpy()
			if isinstance(elem, np.ndarray):
				nd_name = "ndarr_{}".format(i)
				d["buf"][i] = nd_name
				nd_arrays[nd_name] = elem
		#	else:
		#		d["buf"].append(elem)
		if len(nd_arrays)>0:
			ndarray_path = os.path.join(path, name +".npz")
			np.savez_compressed(ndarray_path, **nd_arrays)
			d["ndarray_path"] = ndarray_path
		with open(os.path.join(path, name +".json"), "w") as file:
			json.dump(d, file)
		
	@classmethod
	def deserialize(cls, path, ndarray_to_tf_constant=False, suffix=None):
		name = "HBuffer"
		if suffix is not None:
			name = name + '_' + suffix
		with open(os.path.join(path, name +".json"), "r") as file:
			d = json.load(file)
		hbuf = cls(d["size"])
		hbuf.head = d["head"]
		hbuf.elements = d["elements"]
		
		nd_arrays = {}
		if d["ndarray_path"] is not None:
			with np.load(d["ndarray_path"]) as np_data:
				nd_arrays = dict(np_data)
		
		hbuf.buf = d["buf"]
		for i in range(hbuf.size):
			if isinstance(hbuf.buf[i], str) and hbuf.buf[i].startswith("ndarr_"):
				hbuf.buf[i] = nd_arrays[hbuf.buf[i]]
				if ndarray_to_tf_constant:
					hbuf.buf[i] = tf.constant(hbuf.buf[i])
		return hbuf
