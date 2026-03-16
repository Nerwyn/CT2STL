import os
import math

from skimage import measure
from stl import mesh
from pydicom import dcmread, FileDataset

from args import args, np, sp
from slice_viewer import slice_viewer
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
				match ds[study][series][0].Modality:
					case 'SEG':
						raw = np.array(ds[study][series][0].pixel_array)
					case 'CT':
						# Get dimensions
						c, r = ds[study][series][0].pixel_array.shape
						n = 0
						for dsn in ds[study][series]:
							if dsn.InstanceNumber > n:
								n = dsn.InstanceNumber

						# Create volume and load data into it
						raw = np.zeros((n, c, r), dtype=np.int32)
						for dsn in ds[study][series]:
							raw[dsn.InstanceNumber - 1] = np.array(dsn.pixel_array)

						# Normalize to 1mm3
						scale_y, scale_x = ds[study][series][0].PixelSpacing
						scale_z = ds[study][series][0].SliceThickness
						raw = sp.ndimage.zoom(raw, (scale_z, scale_y, scale_x))

						# Window data to display lungs
						width = 1800
						center = -585
						mx = np.max(raw)
						mn = np.min(raw)
						c1 = (mx - mn) / width
						c2 = (mx + mn) / 2
						raw = np.clip(c1 * (raw - center) + c2, mn, mx)

						# Mask lungs
						mask = generate_lung_mask(raw)
						lungs = mask * raw

						mask = mask.get() if args.gpu else mask
						lungs = lungs.get() if args.gpu else lungs
					case _:
						continue

				# Export data as stl
				date = ds[study][series][0].AcquisitionDate
				export_stl(mask, f'./output/{study}-{series}-{date}.stl')

				# Display data
				# slice_viewer(lungs)

			except Exception as e:
				print(e)


def export_stl(volume: np.ndarray, outfile: str):
	"""Export a 3D numpy volume as a stl file."""
	vertices, faces, normals, values = measure.marching_cubes(volume)
	mesh_data = np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype).get()
	out_mesh = mesh.Mesh(mesh_data)
	for i, f in enumerate(faces):
		for j in range(3):
			out_mesh.vectors[i][j] = vertices[f[j], :]
	out_mesh.rotate(np.array([0, 1, 0]).get(), math.radians(-90))  # ty: ignore
	out_mesh.save(outfile)


if __name__ == '__main__':
	main()
