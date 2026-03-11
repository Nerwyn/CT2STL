from args import args, np, sp
from skimage.segmentation import flood_fill

sphere_3 = sp.ndimage.generate_binary_structure(3, 1)
sphere_5 = sp.ndimage.iterate_structure(sphere_3, 2)
sphere_9 = sp.ndimage.iterate_structure(sphere_5, 2)


def generate_lung_mask(volume: np.ndarray):
	z, y, x = volume.shape

	# Binarize
	volume: np.ndarray = volume > 3000

	# Morphological open to remove outer thin areas which may be touching body
	# volume = sp.ndimage.binary_opening(volume, sphere_3)

	# Remove area outside of lungs
	volume = volume.get() if args.gpu else volume  # ty: ignore
	for k in [0, z - 1]:
		for j in [0, y - 1]:
			for i in [0, x - 1]:
				volume = flood_fill(volume, (k, j, i), 1)
	volume = np.array(volume)
	volume = np.invert(volume)

	# Remove small areas outside of lungs, assuming lungs are largets region
	labeled, num_regions = sp.ndimage.label(volume)
	region_sizes = np.bincount(labeled.ravel())[1:]
	lung_region_label = np.argmax(region_sizes) + 1
	volume = (labeled == lung_region_label).astype(bool)

	# Fill holes in lung layer by layer
	fill_holes(volume)

	return np.invert(volume)


def fill_holes(volume: np.ndarray):
	"""Fill holes using labels as cupy binary_fill_holes is broken"""
	for k in range(volume.shape[0]):
		labeled, num_regions = sp.ndimage.label(np.invert(volume[k]))
		# region_sizes = np.bincount(labeled.ravel())
		volume[k] = labeled == 1
