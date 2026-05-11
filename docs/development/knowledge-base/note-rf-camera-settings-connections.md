# RF Camera Settings — Connection Map

`4_analysed/rf_camera_settings/rf_camera_settings.json` is the single source of
truth for per-session forearm orientation. This note documents every component that
reads or writes it.

---

## Producer

| Component | What it does |
|-----------|-------------|
| `RFCameraSettingsViewer` (`gui/rf_camera_settings_viewer.py`) | Interactive PyQt5/PyVista GUI. Researcher orbits the forearm point-cloud heatmap and clicks "Save Camera Settings". Writes position / focal_point / up_vector / view_angle per session. |
| DAG task: `set_rf_camera_settings` | Flow wrapper in `analysis_workflow.py`. Skips if all sessions already have settings and `force_processing=false`. Depends on `touch_series_transforms`. Category: `viewer_required` (runs under Analysis [Processing]). |

---

## Consumers — fail-fast

All consumers call `load_rf_camera_rotation(camera_settings_dir, session_id)`, which
raises `ValueError` if the file is absent or the session key is missing.
The rotation matrix R is derived via `camera_settings_to_rotation()` in
`tangent_plane_alignment.py` (cross-product of view direction and up vector).

| Component | File | Uses R for |
|-----------|------|-----------|
| `run_cluster_rf_visualization` | `rf_cluster_pipeline.py` | RF metrics projection + `render_forearm_heatmap` per session |
| `run_simple_rf_mapping` | `rf_simple_pipeline.py` | `render_forearm_heatmap` per session |
| `run_population_rf_grid_metrics` | `rf_population_grid_metrics_pipeline.py` | `compute_grid_cell_metrics` projection per session |
| `load_gallery_data` | `rf_gallery_data.py` | `GalleryCell.tangent_rotation` for gallery heatmap rendering |
| `load_explorer_data` | `rf_explorer_data.py` | Forearm vertex + contact rotation for 2D display in RF Feature-Space Explorer |

`project_tangent_plane` in `rf_projection.py` raises `ValueError` if
`rotation_matrix=None`, so the failure surface is also at the lowest projection level.

---

## Staleness tracking

Saving new camera settings must invalidate downstream cached outputs. Three
idempotency mechanisms are patched:

| Pipeline | Mechanism | How camera settings invalidate it |
|----------|-----------|----------------------------------|
| `rf_simple_pipeline` | `should_process_task` | `rf_camera_settings.json` added to `input_paths`; newer mtime → re-run |
| `rf_population_grid_metrics_pipeline` | `should_process_task` | same |
| `rf_cluster_pipeline` (visualization) | `visualization_is_up_to_date` / `rf_visualization_summary.json` | `camera_settings_mtime` stored in sentinel on save; mismatch on next run → re-render |

`rf_explorer_data.py` has its own `.npz` sidecar cache. The cache freshness check
includes `rf_camera_settings.json` mtime, so re-saving orientation invalidates the
explorer cache and forces recomputation.

---

## Data flow summary

```
set_rf_camera_settings (GUI)
    └── writes rf_camera_settings.json
            ├── run_cluster_rf_visualization   → RF heatmap PNGs
            ├── run_simple_rf_mapping          → simple RF heatmap PNGs
            ├── run_population_rf_grid_metrics → metric CSVs + heatmaps
            ├── load_gallery_data              → gallery viewer thumbnails
            └── load_explorer_data             → feature-space explorer 2D view
```

---

## File location

```
<database>/
  4_analysed/
    rf_camera_settings/
      rf_camera_settings.json      ← keyed by session_id
```

Format:
```json
{
  "ST13-01_semicontrolled": {
    "camera_position": [152.3, -45.2, 450.8],
    "focal_point":     [152.3, -45.2,  50.8],
    "up_vector":       [0.0, 1.0, 0.0],
    "view_angle":      30.0
  }
}
```
