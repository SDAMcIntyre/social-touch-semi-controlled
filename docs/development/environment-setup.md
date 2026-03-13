# Environment Setup

## Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- *(Windows only)* [Azure Kinect SDK 1.4.x](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/docs/usage.md) — required by `pyk4a`
- *(optional)* NVIDIA GPU with CUDA 12.x — for CuPy acceleration

---

## 1. Create the environment

```bash
conda env create -f environment.yml
conda activate social-touch
```

This installs all conda and pip packages, including the project itself in editable mode (`-e .`).

---

## 2. Manual installs (GPU / ML)

These are excluded from `environment.yml` because they require system-specific wheels.

**PyTorch (CUDA 12.1 example):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**CuPy (CUDA 12.x):**
```bash
pip install cupy-cuda12x
```

> CuPy must be imported **before** any `preprocessing` package imports.
> See [`docs/development/knowledge-base/note-cupy-import-order.md`](knowledge-base/note-cupy-import-order.md).

---

## 3. Verify

```bash
python -c "import preprocessing; print('OK')"
```

---

## Updating the environment

After pulling changes that modify `environment.yml`:

```bash
conda env update -f environment.yml --prune
```
