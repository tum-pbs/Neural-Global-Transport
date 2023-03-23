import numpy as np
import imageio
#import cv2 as cv

class ImageSequenceWriter():
	def __init__(self):
		pass

try:
	import ffmpeg
except ModuleNotFoundError:
	pass
else:
	class ImageMovieWriter(ImageSequenceWriter):
		def __init__(self, file_name, height, width, crf=23, framerate=24):
			self.height = height
			self.width = width
			self.out_stream = (ffmpeg
				.input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
				.output(file_name, framerate=framerate, pix_fmt='yuv420p', crf=crf)#, vcodec='libx264'
				.overwrite_output()
				.run_async(pipe_stdin=True)
			)
		
		def imwrite(self, image):
			if not (isinstance(image, np.ndarray)):
				raise TypeError("image must be a np.ndarray, is: {}".format(image.__class__.__name__))
			if not (len(image.shape)==3):
				raise TypeError("image must have shape HWC, is: {}".format(image.shape))
			if not (image.shape[0]==self.height and image.shape[1]==self.width):
				raise ValueError("image shape is {}, expected shape is {}".format(image.shape, (self.height, self.width, None)))
			
			if image.shape[2]==1:
				image = np.repeat(image, 3, axis=-1)
			elif image.shape[2]==2:
				image = np.pad(image, ((0,0),(0,0),(0,1)))
			elif image.shape[2]>3:
				raise ValueError("input image must have at most 3 channel")
			
			if not (image.dtype==np.uint8):
				image = (image*225.).astype(np.uint8)
			
			image = np.array(image)
			
			self.out_stream.stdin.write(image.tobytes())
		
		def mimwrite(self, images):
			for image in images:
				self.imwrite(image)
		
		def close(self):
			self.out_stream.stdin.close()
			self.out_stream.wait()
		
		def __enter__(self):
			return self
		
		def __exit__(self, exc_type, exc_value, exc_traceback):
			self.close()