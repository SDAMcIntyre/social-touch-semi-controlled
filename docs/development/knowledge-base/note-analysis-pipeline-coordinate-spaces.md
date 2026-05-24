# Note: Analysis Pipeline Coordinate Spaces

**Date:** 2026-05-19
**Context:** Receptive field mapping pipeline

## Summary

The analysis pipeline operates in two distinct coordinate spaces separated by the `project_to_2d()` boundary in `rf_projection.py`.

## 3D world space (upstream of projection)

- **Units:** millimetres, origin at Kinect sensor
- **Tasks:** extraction (`run_cluster_rf_extraction`), metrics (`run_cluster_rf_metrics_computation`), population RF maps, population RF grid
- **Data types:** `neuron_contacts_xyz.npy`, `cluster_contacts_xyz.npy`, `forearm_vertices.npy` — all `(N, 3)` float64 arrays in mm
- **Metric computation:** `compute_rf_metrics()` in `rf_metrics.py` accepts 3D spike-count DataFrames (columns `x, y, z, spike_count`) and calls `project_to_2d()` internally to compute 2D hull and centroid metrics

## 2D surface space (downstream of projection)

- **Units:** millimetres in the projected plane
- **Tasks:** visualization (`run_cluster_rf_visualization`), heatmap rendering, population RF map rendering
- **Projection methods** (all in `rf_projection.py`):
  - `tangent_plane` — local tangent plane at the neuron-wide centroid, camera-rotation-aligned
  - `cylindrical_unwrap` — cylindrical projection around the forearm axis
  - `slim` — spectral conformal map; UV coordinates pre-cached in `<session>_slim_uv.npz`

## The neuron-centroid invariant

`project_to_2d()` must be called with the **neuron-wide contact centroid** as origin, not the per-cluster centroid. This preserves the hull containment invariant:

```
heatmap_points ⊆ cluster_contacts ⊆ neuron_contacts
```

in 2D space. Using a per-cluster centroid shifts the origin and can cause cluster points to fall outside the neuron hull, breaking the heatmap mask. See `note-rf-cluster-visualization-overview.md`.

## Cross-references

- `note-3d-to-2d-surface-projection-algorithms.md` — detailed algorithm documentation
- `note-rf-cluster-visualization-overview.md` — hull invariant and sentinel discipline
- `note-somatosensory-units-and-calculations.md` — mm unit origin (Kinect SDK)
