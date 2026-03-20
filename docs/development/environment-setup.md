# Environment Setup

## Prerequisites (Windows)

Install these **before** creating the conda environment, in order:

### 1. Microsoft C++ Build Tools

Required to compile the `pyk4a` native extension.

1. Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Run the installer → select **"Desktop development with C++"** workload
3. Complete the install, then **restart your terminal**

### 2. Azure Kinect SDK 1.4.2

Required at both **compile time** (headers for building `pyk4a`) and **runtime**.

1. Download the installer from: https://github.com/microsoft/azure-kinect-sensor-sdk/blob/develop/docs/usage.md
2. Run the installer (default path: `C:\Program Files\Azure Kinect SDK v1.4.2\`)
3. Set the `K4A_INSTALLATION_DIR` environment variable so `pyk4a` can find the SDK headers during `pip install`:
   ```
   set K4A_INSTALLATION_DIR=C:\Program Files\Azure Kinect SDK v1.4.2
   ```
   > Without this variable, `pip install pyk4a` will fail with `Cannot open include file: 'k4a/k4a.h'`.

### 3. Miniconda or Anaconda

Download from: https://docs.conda.io/en/latest/miniconda.html

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
