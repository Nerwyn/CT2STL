import argparse

parser = argparse.ArgumentParser()
parser.add_argument('in_dir', type=str)
parser.add_argument('--gpu', action='store_true', default=False)

args = parser.parse_args()

if args.gpu:
	import cupy as np
	import cupyx.scipy as sp
else:
	import numpy as np
	import scipy as sp  # noqa


def to_np(x: np.ndarray) -> np.ndarray:
	return x.get() if args.gpu else x  # ty: ignore
