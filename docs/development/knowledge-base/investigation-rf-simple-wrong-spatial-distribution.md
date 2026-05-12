# Investigation: RF Simple Pipeline — Wrong Spatial Distribution After PopulationData Refactor

**Date started:** 2026-05-11
**Status:** Paused — diagnostics in place, root cause not yet identified
**Branch:** `feature/rf-simple-step-diagnostics`

---

## Problem Statement

After the `PopulationData` refactor (commit `e2d2ae7`, "refactor(rf-simple-pipeline): use PopulationData instead of inline parsing"), the 2D scatter plot produced by `run_simple_rf_mapping()` shows a spatially wrong distribution. The spike positions do not match what is expected from the raw contact data.

The final output PNG (`<session_id>_rf_simple_cylindrical_unwrap.png`) shows the heatmap using `render_forearm_heatmap()`. The scatter panel within that figure (or the UV projection) is where the distortion is visible.

---

## What Was Done Before Pausing

### Diagnostic figures implemented

A new module `rf_simple_diagnostics.py` was created (branch `feature/rf-simple-step-diagnostics`) with 4 step-by-step PNG figures saved to `<session_out>/diagnostics/`:

| File | Content |
|------|---------|
| `step1_population_data.png` | Camera-view scatter of all contact points (blue) + spike contacts (red), histogram of contacts/touch, summary counts |
| `step2_spike_extraction.png` | Camera-view scatter of spike vertices sized by occurrence count, vertex reuse histogram |
| `step3_aggregation.png` | Camera-view scatter colored by `spike_count` (LogNorm), centroid comparison (proj vs spike vs mesh), count distribution |
| `step4_projection.png` | Camera-view reference (forearm + spikes, panel 1), UV scatter spike vertices only — centred on spike centroid (panel 2), projection metadata text (panel 3) |

Enable with `save_diagnostics: true` in `configs/analyse_workflow_processing_dag.yaml` under `map_receptive_fields_simple.options`. Already set to `true`.

### Known implementation issues fixed along the way

1. **Agg backend / `plt.show()` silent no-op**: `rf_2d_renderer.py` sets the matplotlib backend to `Agg`. `plt.show()` does nothing. Fix: `os.startfile(saved_path)` fallback in `run_diagnostics()` when backend is Agg and `show=True`.
2. **Large array allocation hang**: `cp_idx` can have millions of entries (contact-point-frames at 1 kHz). Must subsample `cp_idx[::step]` *before* `fv[cp_idx]`, not after, to avoid allocating a (C, 3) array of hundreds of MB. Fixed in `diagnose_population_data`.
3. **Histogram with 30k bins**: `bins = np.arange(0, bin_max + 2)` where `bin_max` could be 30k+ → matplotlib drew 30k bars. Fixed to `bins = 50`.
4. **Unsubsampled forearm vertex scatter**: `_fv_step = max(1, len(forearm_vertices) // 20_000)` applied before plotting forearm background points.

---

## Suspected Root Causes (Not Yet Verified)

These were the hypotheses at the time of pausing. The diagnostic figures should reveal which is responsible.

### 1. Centroid mismatch for projection
- The projection centroid passed to `project_to_2d()` is `neuron_contacts_xyz.mean(axis=0)` — the mean of *all* unique contact vertices across all touches.
- The old (pre-refactor) code may have used a different centroid (e.g., spike-only vertices, or mesh centroid).
- **Step 3 figure shows**: red star = `neuron_contacts_xyz.mean` (proj centroid), green star = `spike_xyz.mean` (spike centroid), blue star = `forearm_vertices.mean` (mesh centroid). If these are far apart, centroid choice is likely the issue.

### 2. KDTree / nearest-vertex threshold change
- Old code used a 2 mm threshold to match contact points to forearm vertices; `load_population_data()` may use 15 mm.
- A looser threshold maps contacts to more distant vertices, shifting the spatial distribution.
- **Where to check**: `touch_population_data.py::load_population_data()` — look for the KDTree query radius.

### 3. Contact point parsing differences
- Old code parsed contact XYZ directly from the merged CSV columns; new code uses `PopulationData.cp_vertex_idx` which is pre-mapped to vertices.
- If the vertex mapping introduces quantisation (many contacts → same vertex), the distribution could appear distorted or smeared.
- **Step 1 figure shows**: if `unique spike vertices` is very small relative to `spike contacts`, quantisation is heavy.

### 4. `cp_spike` vs `spike_elicited` column semantics
- `pop_data.cp_spike` is a boolean mask over contact-point-frames (length C).
- `pop_data.spike_elicited` is a boolean mask over touches (length T).
- If these refer to different things (IFF > 0 vs binary spike flag), the set of "spike contacts" could differ from what the old pipeline extracted.
- **Where to check**: `touch_population_data.py` — how `cp_spike` is derived vs how the old code filtered rows.

---

## How to Resume

1. Run the pipeline with `save_diagnostics: true` (already set) and `force_processing: true` to regenerate all outputs.
2. Open `step1_population_data.png`: check `Unique spike vertices` vs `Total contact pts` — heavy quantisation suggests hypothesis 3.
3. Open `step3_aggregation.png`: check centroid triangle — if proj centroid ≠ spike centroid by many mm, investigate hypothesis 1.
4. Open `step4_projection.png`: panel 1 shows the camera-view reference (same orientation as steps 1–3). Compare panel 1 with step 3's left panel — if the spike positions differ, there is a data-assembly bug between steps 3 and 4. Panels 2–3 show the cylindrical UV scatter — compare with the final `_rf_simple_cylindrical_unwrap.png` to check projection geometry.
5. If step 4 panel 1 matches step 3 but panels 2–3 look wrong compared to the final PNG, the issue is inside `project_to_2d()` / `render_forearm_heatmap()` (cylinder axis, seam reference direction, colormap, or normalisation).
6. If step 4 panel 1 also looks wrong (spike positions differ from step 3), trace back to step 3 → step 2 → step 1 to find the first figure that diverges from expectations.

---

## Relevant Files

| File | Role |
|------|------|
| `rf_simple_pipeline.py` | Pipeline entry point — `run_simple_rf_mapping()` |
| `rf_simple_diagnostics.py` | All 4 diagnostic figure functions + `run_diagnostics()` orchestrator |
| `touch_population_data.py` | `load_population_data()` — KDTree matching, `cp_vertex_idx`, `cp_spike` |
| `rf_cluster_visualizer.py` | `render_forearm_heatmap()` — final PNG renderer |
| `rf_projection.py` | `project_to_2d()`, `cylindrical_unwrap`, `fit_cylinder_axis` |
| `tangent_plane_alignment.py` | `camera_settings_to_rotation()` |

---

## Related

- Plan: `docs/development/plans/active/rf-simple-step-diagnostics.md` (diagnostics implementation — all phases complete)
- Commit introducing the problem: `e2d2ae7` — "refactor(rf-simple-pipeline): use PopulationData instead of inline parsing"
- Knowledge base: `note-3d-to-2d-surface-projection-algorithms.md`, `note-rf-cluster-visualization-overview.md`
