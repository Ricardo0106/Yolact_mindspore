import mindspore.nn as nn
class InterpolateModule(nn.Cell):
	"""
	This is a module version of F.interpolate (rip nn.Upsampling).
	Any arguments you give it just get passed along for the ride.
	"""

	def __init__(self, *args, **kwdargs):
		# scale_factor=-kernel_size, align_corners=False, **layer_cfg[2]
		super().__init__()

		self.args = args
		self.kwdargs = kwdargs

	def construct(self, x):
		resize_bili = nn.ResizeBilinear()
		return resize_bili(x, *self.args, **self.kwdargs)