from slice_viewer import slice_viewer
import os
import math
import traceback

from skimage import measure
from stl import mesh
from pydicom import dcmread, FileDataset

from args import args, np, sp, to_np

# from slice_viewer import slice_viewer
from lung_mask import generate_lung_mask


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
						raw = np.zeros((n, c, r), dtype=np.int32)
						for dsn in slices:
							raw[dsn.InstanceNumber - 1] = np.array(dsn.pixel_array)

						# Normalize to 1mm3
						distances = []
						for i in range(len(positions) - 2):
							distances.append(positions[i] - positions[i + 1])
						scale_z = sum(distances) / len(distances)
						scale_y, scale_x = slices[0].PixelSpacing
						raw = sp.ndimage.zoom(raw, (scale_z, scale_y, scale_x))

						# Rotate to transverse plane
						transverse_vect = np.array([0, 0, 1])
						angle = np.arccos(np.dot(norm_vect, transverse_vect))
						print(angle)
						if angle != 0:
							v = to_np(np.cross(norm_vect, transverse_vect))
							c = np.dot(norm_vect, transverse_vect)
							v_cross = np.array(
								[
									[0, -1 * v[2], v[1]],
									[v[2], 0, -1 * v[0]],
									[-1 * v[1], v[0], 0],
								],
							)
							r = v_cross + np.dot(v_cross, v_cross) * (1 / (1 + c))

							raw = sp.spatial.transform.Rotation.from_matrix(r).apply(
								raw
							)

							# ip0 = slices[0].ImagePositionPatient
							# ipn = slices[n - 1].ImagePositionPatient
							# matrix = np.array(
							# 	[
							# 		[
							# 			col_vect[0] * scale_y,
							# 			row_vect[0] * scale_x,
							# 			(ipn[0] - ip0[0]) / (n - 1),
							# 			ip0[0],
							# 		],
							# 		[
							# 			col_vect[1] * scale_y,
							# 			row_vect[1] * scale_x,
							# 			(ipn[1] - ip0[1]) / (n - 1),
							# 			ip0[1],
							# 		],
							# 		[
							# 			col_vect[2] * scale_y,
							# 			row_vect[2] * scale_x,
							# 			(ipn[2] - ip0[2]) / (n - 1),
							# 			ip0[2],
							# 		],
							# 		[0, 0, 0, 1],
							# 	]
							# )
							# raw = sp.ndimage.affine_transform(raw, matrix)

						# Window data to display lungs
						width = 1800
						center = -585
						mx = np.max(raw)
						mn = np.min(raw)
						c1 = (mx - mn) / width
						c2 = (mx + mn) / 2
						raw = np.clip(c1 * (raw - center) + c2, mn, mx)

						slice_viewer(to_np(raw))

						# Mask lungs
						mask = to_np(generate_lung_mask(raw))
						# lungs = to_np(mask * raw)
					case _:
						continue

				# Export data as stl
				date = slices[0].AcquisitionDate
				export_stl(mask, f'./output/{study}-{series}-{date}.stl')

				# slice_viewer(lungs)

			except Exception as e:
				# print(e)
				traceback.print_exc()


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
