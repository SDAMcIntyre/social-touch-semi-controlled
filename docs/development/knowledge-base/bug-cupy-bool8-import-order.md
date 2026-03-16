# BUG: CuPy `bool8` TypeError caused by import order

| Field       | Value |
|-------------|-------|
| Status      | **Resolved** (workaround in place) |
| Affected    | CuPy ≤ 14.0.0 + NumPy ≥ 2.0 + Open3D / pyk4a C-extensions |
| Root cause  | NumPy internal dtype state mutation by compiled C-extensions |
| Fix         | Import CuPy **before** any preprocessing package |

---

## Symptom

```
TypeError: Alias 'bool8' was removed in NumPy 2.0.
  Use a name without a digit at the end.
```

Raised inside CuPy's compiled Cython init (`cupy/_core/_dtype.pyx`,
`_init_dtype_dict`) when `import cupy` runs **after** the project's
preprocessing packages have been imported.

## Environment

- Python 3.10, Windows (Anaconda `py31018`)
- NumPy 2.2.6
- CuPy-CUDA12x 14.0.0
- Open3D, pyk4a, PyVista, and other compiled C-extensions

## Investigation summary

1. `import cupy` in a fresh Python session: **works**.
2. `import cupy` after `import pyvista; from PyQt5 import ...`: **works**.
3. `import cupy` after importing `preprocessing.forearm_extraction`,
   `preprocessing.motion_analysis`, etc.: **fails** with `bool8` TypeError.
4. `import cupy` **before** those preprocessing modules, then importing the
   preprocessing modules: **works**, and CuPy remains functional.

The preprocessing packages transitively load compiled C-extensions (Open3D,
pyk4a) that modify NumPy's internal dtype registry at DLL-load time.  CuPy's
Cython `_init_dtype_dict` function references dtype aliases that exist in a
pristine NumPy session but are invalidated after the mutation.

When CuPy is already initialised (cached in `sys.modules`), later imports
return the cached module and skip the Cython init, so the conflict never
triggers.

## Fix applied

### 1. Entry-point script (`code/scripts/view_merged_neural_kinect.py`)

CuPy is imported at the top of the file, **before** any `preprocessing.*`
import:

```python
try:
    import cupy  # noqa: F401  — early-init only
except Exception:
    pass
```

### 2. Viewer module (`code/src/preprocessing/common/gui/neural_kinect_scene_viewer.py`)

CuPy detection moved from a lazy `_detect_cupy()` function to the module's
third-party import section, **above** the internal `preprocessing.*` imports:

```python
_CUPY_AVAILABLE: bool = False
try:
    import cupy as _cp
    _cp.array([1.0])
    _CUPY_AVAILABLE = True
    del _cp
except Exception as _exc:
    print(f"CuPy not available ... ({type(_exc).__name__}: {_exc})")
    del _exc
```

## Guidance for future CuPy usage

Any new script or module that uses CuPy **and** imports from the
`preprocessing` package tree must ensure CuPy is imported first.  Two
approaches:

1. **Entry-point early-init (preferred):** add `import cupy` near the top of
   the script, before any `preprocessing.*` import.  The result is cached in
   `sys.modules`; subsequent `import cupy` calls in downstream modules are
   free.

2. **Module-level import:** in the module that uses CuPy, place the
   `import cupy` line **above** any `from preprocessing...` import.

See also [note-cupy-import-order.md](note-cupy-import-order.md) for the reusable pattern and checklist.
