# Plan: Clean Conda environment.yml

**Date:** 2026-03-13
**Author:** Basil Duvernoy
**Status:** Completed
**Branch:** `chore/clean-conda-environment-yml`

---

## Overview

Replace the current frozen `environment.yml` (320-line `conda env export` dump with
every transitive dependency pinned) with a clean, human-readable file that lists only
direct dependencies, grouped by category. This makes the environment reproducible from
intent rather than from a snapshot, and easier to maintain as packages evolve.

## Problem Statement

The current `environment.yml` was generated with `conda env export`. It lists ~160 conda
packages and ~155 pip packages — including transitive build deps, compiler runtimes, and
platform-specific binaries. This creates several problems:

- **Hard to read** — developers cannot tell which packages are intentional vs transitive.
- **Brittle** — exact pins on transitive deps cause solver conflicts when any direct dep
  updates.
- **Redundant** — pip packages duplicate what is already in `requirements.txt`.
- **Not cross-platform** — frozen Windows pins fail on macOS/Linux.

## Goals

### In Scope
1. Replace `environment.yml` with a clean, categorised, commented file
2. List only direct conda-channel dependencies (native/C++ packages best installed via conda)
3. Delegate all pure-Python packages to `requirements.txt` via `- -r requirements.txt`
4. Keep `environment_mac.yml` unchanged (it already follows the clean pattern)

### Out of Scope
- Updating `requirements.txt` versions
- Adding or removing project dependencies
- Changing the conda environment name (`pcl_env`)
- PyTorch installation (remains a manual step due to CUDA-specific wheels)

## Success Criteria

- [ ] `environment.yml` is ≤ 40 lines
- [ ] `conda env create -f environment.yml` succeeds on a clean Windows machine
- [ ] After activation + `pip install -r requirements.txt`, `python -c "import numpy; import vtk; import pcl; print('OK')"` works
- [ ] No direct dependency is missing (verified against codebase import scan)
- [ ] The file has section comments matching the categories below

---

## Technical Design

### Approach

Write a minimal `environment.yml` that installs only the packages that **must** come from
conda-forge (native C/C++ libraries, complex builds, system-level deps). Everything else
is delegated to `requirements.txt` via pip, which is already maintained and version-pinned.

This mirrors the pattern already used by `environment_mac.yml`.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Clean minimal env.yml (chosen) | Readable, maintainable, cross-platform intent | Loses exact reproducibility of frozen export | **Chosen** — frozen export can be regenerated anytime |
| Keep frozen export, add comments | Fully reproducible | Still 320 lines, still brittle, comments get lost on re-export | Rejected |
| Single env.yml with all pip deps inline | One file for everything | Duplicates requirements.txt, double maintenance | Rejected |
| Use conda-lock for reproducibility | Best of both worlds | Adds tooling dependency, overkill for this project | Rejected |

### Architecture Changes

No code changes. Single config file replacement.

### Knowledge Base Constraints

- **CuPy import order** (`note-cupy-import-order.md`): CuPy ≤14.0 + NumPy ≥2.0 can
  conflict at runtime. This is a code-level import-order constraint, not a packaging one.
  The environment.yml should not pin CuPy (it's already in requirements.txt with
  `>=13.0` and platform guard). A comment will note the import-order requirement.

---

## Implementation Plan

### Phase 1: Write clean environment.yml
**Goal:** Replace the frozen export with a clean, categorised file

- [ ] Task 1.1 — Write new `environment.yml` with the structure below
- [ ] Task 1.2 — Verify the conda-channel package list covers all native deps

**New file content:**

```yaml
name: pcl_env
channels:
  - conda-forge
  - defaults
dependencies:
  # --- Base ---
  - python=3.10.18
  - pip
  - setuptools

  # --- Native C/C++ libs (prefer conda over pip for these) ---
  - numpy                # MKL-linked build
  - vtk=9.4.2            # 3D visualization toolkit
  - hdf5                 # HDF5 C library (h5py backend)
  - pillow               # native image codec support
  - matplotlib-base      # conda-optimised build
  - ffmpeg               # video codec support
  - sdl2                 # pygame backend
  - pyqt                 # Qt runtime for PyQt5

  # --- Point Cloud Library (Windows only; remove on macOS if build fails) ---
  - pcl=1.15.0
  - python-pcl

  # --- All Python packages via pip (see requirements.txt) ---
  - pip:
      - -r requirements.txt
```

**Files Modified:**
- `environment.yml` — Full rewrite (320 lines → ~25 lines)

**Dependencies:** None

### Phase 2: Verify
**Goal:** Confirm the new file works end-to-end

- [ ] Task 2.1 — `conda env create -f environment.yml` on clean system
- [ ] Task 2.2 — `pip install torch --index-url https://download.pytorch.org/whl/cu121`
- [ ] Task 2.3 — `cd code && pip install -e .`
- [ ] Task 2.4 — Import smoke test: numpy, vtk, pcl, scipy, prefect, PyQt5

**Dependencies:** Phase 1

---

## Testing Plan

### Manual Verification
- [ ] `conda env create -f environment.yml` completes without solver errors
- [ ] `conda activate pcl_env && python -c "import numpy; import vtk; print('OK')"`
- [ ] `pip install -r requirements.txt` installs all packages (no conflicts with conda numpy/vtk/pillow)
- [ ] `python code/scripts/launch_pipeline_gui.py` launches the GUI

### Edge Cases
- [ ] Creating env when `pcl_env` already exists (`conda env create --force`)
- [ ] pip/conda version conflicts on numpy (conda installs numpy, requirements.txt also lists it — pip should detect it's already satisfied)

---

## Documentation Plan

- [ ] No README/CLAUDE.md changes needed (environment setup instructions in README already reference `environment.yml`)

---

## Rollback Plan

1. `git checkout HEAD -- environment.yml` restores the frozen export
2. No data changes, no migrations, no breaking changes

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Conda/pip numpy conflict (conda installs numpy, pip requirements.txt also lists it) | Medium | Low | pip detects existing install and skips; if version mismatch, pin numpy version in env.yml to match requirements.txt |
| PCL fails to resolve on some platforms | Low | Low | Already conditional — comment says to remove on macOS |
| Missing a native dep that was in the frozen export | Low | Medium | Verified against codebase import scan; any missing dep surfaces immediately at import time |

---

## References

- Existing frozen export: `environment.yml` (current, to be replaced)
- macOS variant (already clean): `environment_mac.yml`
- Pip dependencies: `requirements.txt`
- Knowledge base: `docs/development/knowledge-base/note-cupy-import-order.md`
