import time

class StepTimer:
	@staticmethod
	def fmt_time(t):
		'''t (float): time in seconds'''
		h, r = divmod(t, 3600)
		m, s = divmod(r, 60)
		return '{:02d}:{:02d}:{:06.3f}'.format(int(h), int(m), s)
	
	def __init__(self):
		self._start = None
		self._last = None
		self._last_step = None
		self._num_steps = 0
	
	def start(self):
		self._start = time.time()
	
	def step(self):
		t = time.time()
		self._last_step = t - self._last
		self._last = t
		self._num_steps +=1
	
	@property
	def elapsed(self):
		return self._last - self._start
	@property
	def avg_step(self):
		return self.elapsed / self._num_steps
	
	def print(self, remaining_steps=None):
		s = "[Timer] elapsed: {}, avg/step: {}, last: {}".format( \
			self.fmt_time(self.elapsed), \
			self.fmt_time(self.avg_step), \
			self.fmt_time((self._last_step)), \
		)
		if remaining_steps is not  None:
			s += ", remaining: {}".format(self.fmt_time(self.avg_step * remaining_steps))
		return s
	
	def __str__(self):
		return self.print()