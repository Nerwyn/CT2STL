import matplotlib.pyplot as plt


def slice_viewer(X):
	"""
	=================================
	sliceViewer - Image Slices Viewer
	=================================

	Scroll through 2D image slices of a 3D array.
	Original code from: https://matplotlib.org/3.5.3/gallery/event_handling/image_slices_viewer.html
	"""

	class IndexTracker(object):
		def __init__(self, ax, X):
			self.ax = ax
			self.ax.set_title('use scroll wheel to navigate images')

			self.X = X
			self.slices = X.shape[0]
			self.ind = 0

			self.vmin = X.min()
			self.vmax = X.max()
			self.im = ax.imshow(
				self.X[self.ind, :, :], cmap='gray', vmin=self.vmin, vmax=self.vmax
			)
			self.update()

		def onscroll(self, event):
			match event.button:
				case 'up':
					self.ind = (self.ind + 1) % self.slices
				case 'down':
					self.ind = (self.ind - 1) % self.slices
			self.update()

		def onpress(self, event):
			match event.key:
				case 'pageup':
					self.ind = (self.ind + 10) % self.slices
				case 'pagedown':
					self.ind = (self.ind - 10) % self.slices
				case 'up':
					self.ind = (self.ind + 1) % self.slices
				case 'down':
					self.ind = (self.ind - 1) % self.slices
			self.update()

		def update(self):
			self.im.set_data(self.X[self.ind, :, :])
			self.ax.set_ylabel('slice %s' % self.ind)
			self.im.axes.figure.canvas.draw_idle()
			self.im.axes.figure.canvas.flush_events()

	fig, ax = plt.subplots(1, 1)
	tracker = IndexTracker(ax, X)
	fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
	fig.canvas.mpl_connect('key_press_event', tracker.onpress)
	plt.show()
