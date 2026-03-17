# CT2STL

Convert DICOM Lung CT files into STLs of the lungs. Supports nvidia and amd gpu acceleration.

![Lung STL](/output.png)

# Setup

Install dependencies from pyproject.toml. This project was created using `uv`.

```
uv sync
```

## GPU Accleration

This project uses `cupy` for gpu acceleration. See their [installation documentation](https://docs.cupy.dev/en/stable/install.html) for more information on how to install prerequisites.

To use gpu acceleration, pass either `gpu-cuda` or `gpu-rocm` to `uv sync` using the `--extra` parameter.

```
uv sync --extra gpu-cuda
```

```
uv sync --extra gpu-rocm
```

# Run

Run `main.py`, passing the path to a direcotory containing DICOM CT files in its subfolders as an argument. Pass the `--gpu` flag to use GPU acceleration. All CT studies/series found will be processed.
