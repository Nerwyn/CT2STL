import os
import sys
import math

from skimage import measure
from stl import mesh
from pydicom import dcmread, FileDataset

from args import args, np, sp, to_np

# from slice_viewer import slice_viewer
from lung_mask import generate_lung_mask

SAGITTAL_VECT = np.array([1, 0, 0])
CORONAL_VECT = np.array([0, 1, 0])
TRANSVERSE_VECT = np.array([0, 0, 1])


def main():
	# Read all data
	ds: dict[str, dict[str, list[FileDataset]]] = {}
	for root, _dirs, files in os.walk(args.in_dir):
		for f in files:
			fp = os.path.join(root, f)

			try:
				ds0: FileDataset = dcmread(fp)
				study: str = str(ds0.StudyInstanceUID)
				series: str = str(ds0.SeriesInstanceUID)

				ds.setdefault(study, {})
				ds[study].setdefault(series, [])
				ds[study][series].append(ds0)
			except Exception as _e:
				pass

	# Volumize data
	for study in ds:
		for series in ds[study]:
			try:
				slices = ds[study][series]
				match slices[0].Modality:
					case 'SEG':
						raw = np.array(slices[0].pixel_array)
					case 'CT':
						# Get dimensions
						c, r = slices[0].pixel_array.shape
						n = len(slices)

						# Sort slices by image position and orientation
						orientation = slices[0].ImageOrientationPatient
						row_vect = orientation[:3]
						col_vect = orientation[3:]
						norm_vect = np.cross(row_vect, col_vect)

						positions = [
							np.dot(np.array(dsn.ImagePositionPatient), norm_vect).item()
							for dsn in slices
						]
						e_slices = list(enumerate(slices))
						e_slices.sort(
							key=lambda x: positions[x[0]],
							reverse=True,
						)
						slices = [x[1] for x in e_slices]
						positions.sort(reverse=True)

						# Create volume and load data into it
						raw = np.zeros((n, c, r), dtype=np.int16)
						for dsn in slices:
							raw[dsn.InstanceNumber - 1] = np.array(dsn.pixel_array)

						# Normalize to 1mm3
						distances = []
						for i in range(len(positions) - 2):
							distances.append(positions[i] - positions[i + 1])
						scale_z = sum(distances) / len(distances)
						scale_y, scale_x = slices[0].PixelSpacing
						raw = sp.ndimage.zoom(raw, (scale_z, scale_y, scale_x))

						# Window data to display lungs
						width = 1800
						center = -585
						raw = window_level(raw, width, center)

						# Rotate to transverse plane if coronal or sagittal
						norm_vect = np.abs(norm_vect)
						if not np.array_equal(norm_vect, TRANSVERSE_VECT):
							if np.array_equal(norm_vect, SAGITTAL_VECT):
								raw = np.rot90(raw, axes=(0, 1))
								raw = np.flip(raw, axis=0)
								raw = np.rot90(raw, k=3, axes=(1, 2))
							elif np.array_equal(norm_vect, CORONAL_VECT):
								raw = np.rot90(raw, axes=(0, 1))
								raw = np.flip(raw, axis=0)
							else:
								print(
									'Unsupported orientation: ' + str(norm_vect),
									file=sys.stderr,
								)

						# Convert to 8 bit
						raw = to_8bit(raw)

						# Trim empty slices
						raw = trim_volume(raw)

						# slice_viewer(to_np(raw))

						# Mask lungs
						mask = to_np(generate_lung_mask(raw))
						# lungs = to_np(mask * raw)

						# slice_viewer(mask)
						# slice_viewer(lungs)
					case _:
						continue

				# Export data as stl
				date = slices[0].AcquisitionDate
				export_stl(mask, f'./output/{study}-{series}-{date}.stl')

			except Exception as e:
				print('Failed to process study', study, series, e)


def window_level(a: np.ndarray, w: int, c: int) -> np.ndarray:
	min = c - 0.5 - (w - 1) / 2
	max = c - 0.5 + (w - 1) / 2
	a = np.clip(a, min, max)
	a = (((a - (c - 0.5)) / (w - 1)) + 0.5) * (max - min) + min
	return a.astype(np.int16)


def to_8bit(volume: np.ndarray) -> np.ndarray:
	f_min = volume.min()
	f_max = volume.max()
	volume = ((volume - f_min) / (f_max - f_min)) * 255
	volume = np.clip(
		volume,
		0,
		255,
	).astype(np.uint8)
	return volume


def trim_volume(volume: np.ndarray) -> np.ndarray:
	for i in range(len(volume) - 1):
		if volume[i, :, :].any():
			volume = volume[i:, :, :]
			break
	for i in range(len(volume) - 1, -1, -1):
		if volume[i, :, :].any():
			volume = volume[: i - 1, :, :]
			break
	return volume


def export_stl(volume: np.ndarray, outfile: str):
	"""Export a 3D numpy volume as a stl file."""
	vertices, faces, normals, values = measure.marching_cubes(volume)
	mesh_data = to_np(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
	out_mesh = mesh.Mesh(mesh_data)
	for i, f in enumerate(faces):
		for j in range(3):
			out_mesh.vectors[i][j] = vertices[f[j], :]
	out_mesh.rotate(to_np(np.array([0, 1, 0])), math.radians(-90))
	out_mesh.save(outfile)


if __name__ == '__main__':
	main()
