import argparse
import os
from skimage import measure
from stl import mesh
from pydicom import dcmread, FileDataset

# from slice_viewer import slice_viewer


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('in_dir', type=str)
	parser.add_argument('--gpu', action='store_true', default=False)

	args = parser.parse_args()
	if args.gpu:
		import cupy as np
		import cupyx.scipy as sp
	else:
		import numpy as np
		import scipy as sp

	# Read all data
	ds: dict[str, dict[str, list[FileDataset]]] = {}
	for f in os.listdir(args.in_dir):
		fp = os.path.join(args.in_dir, f)

		ds0: FileDataset = dcmread(fp)
		study: str = str(ds0.StudyInstanceUID)
		series: str = str(ds0.SeriesInstanceUID)

		ds.setdefault(study, {})
		ds[study].setdefault(series, [])
		ds[study][series].append(ds0)

	# Volumize data
	volumes = []
	for study in ds:
		for series in ds[study]:
			# Short circuit SEG data, already a volume
			if ds[study][series][0].Modality == 'SEG':
				raw = np.array(ds[study][series][0].pixel_array)
				continue

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

			volumes.append(raw.get() if args.gpu else raw)

			# Export data as stl
			vertices, faces, normals, values = measure.marching_cubes(raw.get())
			mesh_data = np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype).get()
			raw_mesh = mesh.Mesh(mesh_data)
			for i, f in enumerate(faces):
				for j in range(3):
					raw_mesh.vectors[i][j] = vertices[f[j], :]
			raw_mesh.save(f'./output/{study}|{series}.stl')

	# Display data
	# slice_viewer(volumes[0])


if __name__ == '__main__':
	main()
