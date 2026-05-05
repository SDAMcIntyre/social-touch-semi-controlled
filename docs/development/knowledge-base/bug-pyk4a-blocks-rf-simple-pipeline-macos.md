# Bug: map_receptive_fields_simple fails on macOS due to transitive pyk4a import

**Date:** 2026-05-05  
**Status:** Known, not fixed  
**Observed:** `map_receptive_fields_simple` flow raises `ModuleNotFoundError: No module named 'pyk4a'` on macOS and exits without writing any output. The batch runner reports SUCCESS regardless (separate issue — it catches the Prefect flow exception but still marks the task complete).

---

## Symptom

```
conda run -n pcl_env python code/scripts/analysis_workflow.py \
  --dag-config configs/analyse_workflow_dag.yaml \
  --tasks map_receptive_fields_simple
```

Batch runner reports `map_receptive_fields_simple: SUCCESS`. No output written to
`4_analysed/receptive_field_maps_simple/`. Prefect log shows:

```
ModuleNotFoundError: No module named 'pyk4a'
Flow run finished in state Failed("Flow run encountered an exception: ModuleNotFoundError: No module named 'pyk4a'")
```

---

## Root cause — full import chain

The simple RF pipeline (`rf_simple_pipeline.py`) has no `pyk4a` dependency.
The failure comes from the package `__init__.py` eagerly importing a task that
is only needed by *clustered* RF flows:

```
map_receptive_fields_simple_flow
  → _rf_mapping()                              # analysis_workflow.py:39
    → from analysis.receptive_field_mapping import ...
      → receptive_field_mapping/__init__.py:14
        from .rf_camera_angle_task import pick_rf_camera_angle_batch, SessionSceneData
          → rf_camera_angle_task.py:38
            from preprocessing.forearm_extraction.registration.csv_spatial_transformer import ...
              → preprocessing/forearm_extraction/__init__.py:20  (module-level)
                from .gui.multivideo_frames_selector import MultiVideoFramesSelector
                  → multivideo_frames_selector.py:9
                    from preprocessing.common import VideoMP4Manager
                      → preprocessing/common/__init__.py:8  (module-level)
                        from .data_access.kinect_mkv_manager import KinectFrame, KinectMKV
                          → kinect_mkv_manager.py:9
                            from pyk4a import K4AException, PyK4APlayback, PyK4ACapture
                              ModuleNotFoundError: No module named 'pyk4a'
```

`pyk4a` is a Windows-only hardware library for the Azure Kinect SDK.
It is excluded from the macOS `pcl_env` install.

---

## Why the deferred-import pattern in `_rf_mapping()` doesn't help

`analysis_workflow.py` already wraps the whole `analysis.receptive_field_mapping`
import in a deferred helper (`_rf_mapping()`, line 39) specifically to avoid
importing this chain at module load. The comment says:

> Deferred import: analysis.receptive_field_mapping chains through
> preprocessing.forearm_extraction → pyk4a (Windows-only hardware lib).

This correctly prevents the import from running when other (non-RF) tasks are
executed. However, it does not help when an RF task *is* executed, because
`_rf_mapping()` imports the entire package, which triggers the `__init__.py`
chain above before `run_simple_rf_mapping` is even reached.

---

## What the simple pipeline actually needs

`rf_simple_pipeline.py` imports only:

```python
import json, logging
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial import cKDTree
```

No `pyk4a`, no `preprocessing.forearm_extraction`. Completely macOS-safe.

`pick_rf_camera_angle_batch` and `SessionSceneData` (the only exports from
`rf_camera_angle_task`) are used exclusively in clustered RF flows:

| Flow | Line in analysis_workflow.py |
|---|---|
| `map_receptive_fields_clustered_flow` | 335, 358 |
| `visualize_receptive_fields_clustered_flow` | 426, 448 |

They are never referenced in `map_receptive_fields_simple_flow`.

---

## Fix approach

Remove `rf_camera_angle_task` from `receptive_field_mapping/__init__.py` exports.
Move it into a separate deferred import used only by the clustered flows.

**Step 1 — `receptive_field_mapping/__init__.py`:**  
Remove lines:
```python
from .rf_camera_angle_task import pick_rf_camera_angle_batch, SessionSceneData
```
and their entries in `__all__`.

**Step 2 — `analysis_workflow.py`:**  
Add a second deferred-import helper for the camera-angle task, used only in
clustered flows:
```python
def _rf_camera_angle():
    from analysis.receptive_field_mapping.rf_camera_angle_task import (
        pick_rf_camera_angle_batch,
        SessionSceneData,
    )
    return pick_rf_camera_angle_batch, SessionSceneData
```
Replace the two call sites in `map_receptive_fields_clustered_flow` and
`visualize_receptive_fields_clustered_flow` accordingly.

This unblocks the simple pipeline on macOS without touching the clustering path.

---

## Secondary issue: batch runner masks flow failures

The `TaskExecutor` marks `map_receptive_fields_simple` as `SUCCESS` even when the
Prefect flow raises an unhandled exception. The exit code is still 0. This means
failures in this stage are invisible unless you inspect the Prefect log or check
whether the output directory was created.

---

## Affected files

- `code/src/analysis/receptive_field_mapping/__init__.py` — eager `rf_camera_angle_task` import
- `code/src/analysis/receptive_field_mapping/rf_camera_angle_task.py` — triggers chain via `preprocessing.forearm_extraction`
- `code/src/preprocessing/forearm_extraction/__init__.py` — module-level `MultiVideoFramesSelector` import
- `code/src/preprocessing/common/__init__.py` — module-level `KinectMKV` import
- `code/scripts/analysis_workflow.py` — `_rf_mapping()` bundles camera-angle task with macOS-safe simple pipeline
