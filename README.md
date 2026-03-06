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

Run `main.py`, passing the path to the DICOM CT series files as an argument. Pass the `--gpu` flag to use GPU acceleration.