# Analysis Pipeline Outcomes

Complete inventory of every analysis task, the artifacts it produces, and
how those artifacts feed into downstream tasks. Organized by the four DAG
categories: **Foundation**, **Spatial Sensitivity**, **Stimulus Sensitivity**,
and **Cross-Domain**.

Output directories listed below are subdirectories of `4_analysed/` inside
each session's database root. Canonical directory-name constants live in
`code/src/analysis/pipeline/output_dirs.py`.

---

## Dependency graph

```
touch_summarize_blocks ─────────────────────────────────────────────────────────────────────────────────────
                                                                                                            │
touch_prepare_sessions ─────────────────────────────────────────────────────────────────────────────────────┐│
   │                                                                                                       ││
   ├──► spatial_map_single_touch ──┬──► spatial_configure_slim_uv                                          ││
   │                               │        │                                                              ││
   │                               │        └──► spatial_precompute_slim_uv                                ││
   │                               │                 │                                                     ││
   │                               │                 └──► spatial_extract_boundaries ──┬──► spatial_compare_boundaries
   │                               │                                                  │                    ││
   │                               │                                                  └──► spatial_compare_rf_centers
   │                               │                                                                       ││
   │                               ├──► cross_map_feature_grid* ──► cross_extract_grid_metrics*            ││
   │                               │         (also needs stimulus_extract_features)       │                ││
   │                               │                                                      └──► cross_render_grid_metrics*
   │                               │                                                                       ││
   │                               └──► cross_render_sessions                                              ││
   │                                        (also needs stimulus_extract_features + spatial_set_camera)     ││
   │                                                                                                       ││
   └──► touch_compute_series ──┬──► spatial_set_camera ──► spatial_map_baseline                            ││
                               │        (also feeds cross_extract_grid_metrics, cross_render_cluster_rf,    ││
                               │         cross_render_sessions)                                            ││
                               │                                                                           ││
                               └──► stimulus_extract_features ──┬──► stimulus_render_radar                 ││
                                                                ├──► stimulus_compare_sessions             ││
                                                                ├──► stimulus_iff_tuning_curves            ││
                                                                ├──► stimulus_iff_instruction_tuning       ││
                                                                ├──► stimulus_analyse_efficacy*            ││
                                                                │                                          ││
                                                                └──► stimulus_cluster_touches* ─┬──► stimulus_compare_clusters*
                                                                                                │          ││
                                                                                                └──► cross_extract_cluster_rf*
                                                                                                      │   ││
                                                                                      ┌───────────────┘   ││
                                                                                      ├──► cross_compute_cluster_metrics*
                                                                                      └──► cross_render_cluster_rf*

Tasks marked with * are currently disabled in the DAG config.
```

---

## Foundation

### touch_summarize_blocks

| Field | Value |
|-------|-------|
| DAG key | `touch_summarize_blocks` |
| Category | `foundation` |
| Depends on | (none) |
| Output dir | `touch_summarize_blocks/` |

**Outcomes:**

| Artifact | Format | Description |
|----------|--------|-------------|
| Block structure summary | CSV | Per-session block/trial structure metadata |

---

### touch_prepare_sessions

| Field | Value |
|-------|-------|
| DAG key | `touch_prepare_sessions` |
| Category | `foundation` |
| Depends on | (none) |
| Output dir | `touch_prepare_sessions/` |

**Outcomes:**

| Artifact | Format | Description |
|----------|--------|-------------|
| `{session_id}_prepared.csv` | CSV | Cleaned per-session data: NaN-interpolated touch columns, synthesized `block_order_id`, assigned `gesture_type` (`tap`, `stroke_proximal`, `stroke_distal`), invalid touches filtered out |
| `gesture_type_summary.csv` | CSV | Aggregate gesture-type counts across all sessions |

**Consumed by:** `touch_compute_series`, `spatial_map_single_touch`

---

### touch_compute_series

| Field | Value |
|-------|-------|
| DAG key | `touch_compute_series` |
| Category | `foundation` |
| Depends on | `touch_prepare_sessions` |
| Output dir | `touch_compute_series/` |

**Outcomes:**

| Artifact | Format | Description |
|----------|--------|-------------|
| `{session_id}_series_augmented.csv` | CSV | Prepared CSV augmented with per-frame kinematics and derived columns |

**Added columns** (each toggled independently in config):

| Transform | Columns added |
|-----------|---------------|
| Hand position | Resolved hand center x/y/z |
| Hand velocity | `hand_velocity_x`, `hand_velocity_y`, `hand_velocity_z` |
| Hand acceleration | `hand_acceleration_x`, `hand_acceleration_y`, `hand_acceleration_z` |
| Pressure | `pressure` |
| Mechanics of solids | Strain, stress, strain rate, elastic energy, impulse |
| Velocity amplitude | `hand_velocity_amplitude` (scalar magnitude) |
| Velocity signed | Directional velocity component |

**Consumed by:** `stimulus_extract_features`, `spatial_set_camera`

---

## Spatial Sensitivity

### spatial_map_single_touch

| Field | Value |
|-------|-------|
| DAG key | `spatial_map_single_touch` |
| Category | `spatial_sensitivity` |
| Depends on | `touch_prepare_sessions` |
| Output dir | `spatial_map_single_touch/` |

**Outcomes:**

| Artifact | Format | Description |
|----------|--------|-------------|
| `{session_id}/single_touch_rf_maps.npz` | NumPy NPZ (sparse CSR) | Per-touch RF maps: vertex index to mean IFF and max IFF. Sparse format for memory efficiency. |

**Consumed by:** `spatial_configure_slim_uv`, `spatial_precompute_slim_uv`, `spatial_extract_boundaries`, `cross_map_feature_grid`, `cross_render_sessions`

---

### spatial_set_camera

| Field | Value |
|-------|-------|
| DAG key | `spatial_set_camera` |
| Category | `viewer_required` |
| Depends on | `touch_compute_series` |
| Output dir | `spatial_set_camera/` |

**Outcomes:**

| Artifact | Format | Description |
|----------|--------|-------------|
| `rf_camera_settings.json` | JSON | Per-session rotation matrix (4x4), camera angle (degrees). Interactive PyQt5/PyVista GUI. |

All downstream RF rendering and projection tasks load this file; `ValueError`
is raised if absent.

**Consumed by:** `spatial_map_baseline`, `cross_extract_grid_metrics`, `cross_render_cluster_rf`, `cross_render_sessions`

---

### spatial_map_baseline

| Field | Value |
|-------|-------|
| DAG key | `spatial_map_baseline` |
| Category | `spatial_sensitivity` |
| Depends on | `spatial_set_camera` |
| Output dir | `spatial_map_baseline/` |

**Outcomes:**

| Artifact | Format | Description |
|----------|--------|-------------|
| `{session_id}/spike_positions.csv` | CSV | One row per spike contact vertex (x, y, z) |
| `{session_id}/{session_id}_rf_simple.png` | PNG | Forearm heatmap colored by spike density |

---

### spatial_configure_slim_uv

| Field | Value |
|-------|-------|
| DAG key | `spatial_configure_slim_uv` |
| Category | `viewer_required` |
| Depends on | `spatial_map_single_touch` |
| Output dir | (saves config YAML alongside SLIM UV output) |

**Outcomes:**

| Artifact | Format | Description |
|----------|--------|-------------|
| `slim_uv_config.yaml` | YAML | Per-session mesh method (`bpa`/`delaunay`), clean steps, `n_iter`, `max_edge_mm`. Interactive PyQt5/PyVista GUI. |

**Consumed by:** `spatial_precompute_slim_uv`

---

### spatial_precompute_slim_uv

| Field | Value |
|-------|-------|
| DAG key | `spatial_precompute_slim_uv` |
| Category | `spatial_sensitivity` |
| Depends on | `spatial_map_single_touch`, `spatial_configure_slim_uv` |
| Output dir | `spatial_slim_uv/` |

**Outcomes:**

| Artifact | Format | Description |
|----------|--------|-------------|
| `{session_id}/{session_id}_slim_uv.npz` | NumPy NPZ | Cached UV coordinates: vertex index to (u, v) |
| `slim_uv_centroid_xyz.json` | JSON | IFF-weighted centroid used as UV origin |
| `slim_uv_diagnostic_*.png` | PNG (optional) | Mesh cleaning step visualizations (when `save_diagnostics: true`) |

**Consumed by:** `spatial_extract_boundaries`

---

### spatial_extract_boundaries

| Field | Value |
|-------|-------|
| DAG key | `spatial_extract_boundaries` |
| Category | `spatial_sensitivity` |
| Depends on | `spatial_map_single_touch`, `spatial_precompute_slim_uv` |
| Output dir | `spatial_extract_boundaries/` |

**Outcomes (per session):**

| Artifact | Format | Description |
|----------|--------|-------------|
| `{gtype}_population_response_field.png` (x5) | PNG | Per-gesture heatmap: `all`, `tap`, `stroke`, `stroke_proximal`, `stroke_distal` |
| `{session_id}_population_response_fields_composite_scatter.png` | PNG | Five-panel grid layout (scatter mode) |
| `{session_id}_population_response_fields_composite_interpolated.png` | PNG | Five-panel grid layout (interpolated heatmap mode) |
| `{session_id}_population_response_fields_colorbar.png` | PNG | Shared colorbar for the composites |
| `{session_id}_population_response_fields.npz` | NumPy NPZ | Data archive (see key table below) |
| `iff_tuning_sentinel.json` | JSON | Idempotency sentinel (computation timestamp) |

**NPZ key families:**

| Key pattern | Content |
|-------------|---------|
| Per-vertex heatmap arrays | Density values in UV space per gesture type |
| Interpolated grids | 2D arrays in UV space per gesture type |
| `boundary_centroid_uv_{gtype}` | Geometric center of inflection contour (UV) |
| `boundary_centroid_xyz_{gtype}` | Geometric center of inflection contour (3D mm) |
| `boundary_peak_uv_{gtype}` | Hotspot: UV of heatmap intensity maximum |
| `boundary_peak_xyz_{gtype}` | Hotspot: back-projected XYZ (mm) of intensity max |
| Boundary perimeter / area | Inflection boundary size metrics per gesture type |
| Mesh metadata | Vertices, faces, forearm vertex colors (SLIM RGB) |

**Gesture subsets:**

| Subset | Definition |
|--------|------------|
| `all` | Every touch regardless of type |
| `tap` | `gesture_type == 'tap'` |
| `stroke_proximal` | `gesture_type == 'stroke_proximal'` |
| `stroke_distal` | `gesture_type == 'stroke_distal'` |
| `stroke` | Virtual: `stroke_proximal` + `stroke_distal` combined (excludes tap) |

**Consumed by:** `spatial_compare_boundaries`, `spatial_compare_rf_centers`

---

### spatial_compare_boundaries

| Field | Value |
|-------|-------|
| DAG key | `spatial_compare_boundaries` |
| Category | `spatial_sensitivity` |
| Depends on | `spatial_extract_boundaries` |
| Output dir | `spatial_compare_boundaries/` |

**Outcomes:**

| Artifact | Format | Description |
|----------|--------|-------------|
| `session_rf_boundary_summary.csv` | CSV | One row per session x gesture type; columns = boundary metrics (perimeter, area, centroid distance, ...) |
| Contour overlay PNGs | PNG | All sessions' inflection boundaries superimposed, one image per gesture type |
| Metric panel PNGs | PNG | Bar charts: sessions on X-axis, boundary metric on Y-axis, one panel per metric |
| Session x gesture heatmap PNG | PNG | Sessions on Y-axis, gesture types on X-axis, metric as color; optional hierarchical clustering dendrogram |

---

### spatial_compare_rf_centers

| Field | Value |
|-------|-------|
| DAG key | `spatial_compare_rf_centers` |
| Category | `spatial_sensitivity` |
| Depends on | `spatial_extract_boundaries` |
| Output dir | `spatial_compare_rf_centers/` |

**Outcomes:**

| Artifact | Format | Description |
|----------|--------|-------------|
| Per-session heatmap PNGs (centroid) | PNG | Heatmap with violet `+` marker at RF centroid, one per gesture type per session |
| Per-session heatmap PNGs (hotspot) | PNG | Heatmap with red `*` marker at intensity maximum, one per gesture type per session |
| Aggregate centroid scatter plot | PNG | Cross-session scatter: delta-U / delta-V from stroke centroid (proximal vs distal offsets) |
| Aggregate hotspot scatter plot | PNG | Cross-session scatter: delta-U / delta-V from stroke hotspot |
| `rf_center_comparison_summary.csv` | CSV | Per-session centroid and hotspot offsets relative to the `stroke` aggregate baseline |

---

## Stimulus Sensitivity

### stimulus_extract_features

| Field | Value |
|-------|-------|
| DAG key | `stimulus_extract_features` |
| Category | `stimulus_sensitivity` |
| Depends on | `touch_compute_series` |
| Output dir | `stimulus_extract_features/` |

**Outcomes:**

| Artifact | Format | Description |
|----------|--------|-------------|
| `{feature}/{session_id}_touch_summary.csv` | CSV | One row per touch event, one subfolder per enabled aggregation family |

**Enabled aggregation families** (each produces its own subfolder):

| Feature key | Aggregation | Description |
|-------------|-------------|-------------|
| `max` | Per-touch maximum | Max of every numeric column across the touch's frames |
| `mean` | Per-touch mean | Mean of every numeric column |
| `min` | Per-touch minimum | Min of every numeric column |
| `median` | Per-touch median | Median of every numeric column |
| `std` | Per-touch std dev | Standard deviation of every numeric column |
| `range` | Per-touch range | Max minus min of every numeric column |
| `skewness` | Per-touch skewness | Skewness of every numeric column |
| `mean_during_iff` | IFF-active mean | Mean over frames where `Nerve_freq > 0` |
| `mean_before_iff` | Pre-IFF window mean | Mean over a configurable window (default 250 ms) before first spike-active frame |
| `spike_count` | Spike count | Count of non-NaN values in spike column |

**Common columns across all feature CSVs:**
`block_order_id`, `trial_id`, `single_touch_id`, `gesture_type`, `session_id`,
`mean_contact_x/y/z`, `spike_elicited`, `type_metadata`

**Consumed by:** `stimulus_cluster_touches`, `stimulus_render_radar`, `stimulus_compare_sessions`, `stimulus_iff_tuning_curves`, `stimulus_iff_instruction_tuning`, `stimulus_analyse_efficacy`, `cross_map_feature_grid`, `cross_render_sessions`

---

### stimulus_render_radar

| Field | Value |
|-------|-------|
| DAG key | `stimulus_render_radar` |
| Category | `stimulus_sensitivity` |
| Depends on | `stimulus_extract_features` |
| Output dir | `stimulus_render_radar/` |

**Outcomes:**

| Artifact | Format | Description |
|----------|--------|-------------|
| Radar PNGs | PNG | Spider plots comparing touch features across gesture types; one per session per configured `radar_group` |

---

### stimulus_compare_sessions

| Field | Value |
|-------|-------|
| DAG key | `stimulus_compare_sessions` |
| Category | `stimulus_sensitivity` |
| Depends on | `stimulus_extract_features` |
| Output dir | `stimulus_compare_sessions/` |

**Outcomes:**

| Artifact | Format | Description |
|----------|--------|-------------|
| Feature comparison PNGs | PNG | Box-and-strip plots comparing feature distributions across sessions; one per configured feature x aggregation |

---

### stimulus_iff_tuning_curves

| Field | Value |
|-------|-------|
| DAG key | `stimulus_iff_tuning_curves` |
| Category | `stimulus_sensitivity` |
| Depends on | `stimulus_extract_features` |
| Output dir | `stimulus_iff_tuning_curves/` |

**Outcomes (per tuning feature):**

| Artifact | Format | Description |
|----------|--------|-------------|
| `{feature}/{gesture_subset}/{metric}/{session_id}_tuning.png` | PNG | Per-session tuning curve: touch feature on X-axis, IFF response metric on Y-axis, with touch-count bars |
| `{feature}/{gesture_subset}/{metric}/overlay_tuning.png` | PNG | All sessions' tuning curves overlaid on shared axes |

**Configured tuning features:** `contact_area_mean`, `pressure_mean`, `hand_velocity_amplitude_mean`, `contact_depth_mean`

**Binning strategy:** `raw_dots` (scatter with polynomial fit + CI band) or `sliding_window` (windowed bin averages)

---

### stimulus_iff_instruction_tuning

| Field | Value |
|-------|-------|
| DAG key | `stimulus_iff_instruction_tuning` |
| Category | `stimulus_sensitivity` |
| Depends on | `stimulus_extract_features` |
| Output dir | `stimulus_iff_instruction_tuning/` |

**Outcomes:**

| Artifact | Format | Description |
|----------|--------|-------------|
| Instruction heatmap PNGs | PNG | Heatmaps of IFF response by designed instruction category (speed, force, contact area); one per tuning category |

**Configured tuning categories:** `contact_area_metadata`, `speed_metadata`, `force_metadata`

---

### stimulus_analyse_efficacy (disabled)

| Field | Value |
|-------|-------|
| DAG key | `stimulus_analyse_efficacy` |
| Category | `stimulus_sensitivity` |
| Depends on | `stimulus_extract_features` |
| Output dir | `stimulus_analyse_efficacy/` |

**Outcomes:**

| Artifact | Format | Description |
|----------|--------|-------------|
| Efficacy metrics | JSON | Statistical tests relating touch features to spike elicitation (spike_elicited column) |

---

### stimulus_cluster_touches (disabled)

| Field | Value |
|-------|-------|
| DAG key | `stimulus_cluster_touches` |
| Category | `stimulus_sensitivity` |
| Depends on | `stimulus_extract_features` |
| Output dir | `stimulus_cluster_touches/` |

**Outcomes (per combination x clusterer):**

| Artifact | Format | Description |
|----------|--------|-------------|
| `{combo}/{clusterer}/pooled_touch_summary_clustered.csv` | CSV | All touches with added `cluster_label` column (-1 = outlier) |
| `{combo}/{clusterer}/cluster_metadata.json` | JSON | Method params, dispersion per cluster, n_clusters |
| `{combo}/{clusterer}/heatmaps/{session_id}_touch_density.png` | PNG | Per-session cluster density visualization |

When `per_type_clustering: true`, the structure is replicated under
`{combo}/{clusterer}/{gesture_type}/`.

**Available clustering methods:** binning, k-means, hierarchical, DBSCAN, GMM, type-stratified, cartesian binning

**Reduction step** (applied before clustering): configurable scaler (`standard`/`minmax`), optional variance filter, optional PCA

**Evaluation metrics:** silhouette, Davies-Bouldin, Calinski-Harabasz, bootstrap stability (adjusted Rand index)

**Consumed by:** `stimulus_compare_clusters`, `cross_extract_cluster_rf`

---

### stimulus_compare_clusters (disabled)

| Field | Value |
|-------|-------|
| DAG key | `stimulus_compare_clusters` |
| Category | `stimulus_sensitivity` |
| Depends on | `stimulus_cluster_touches` |
| Output dir | `stimulus_compare_clusters/` |

**Outcomes (per combination x clusterer):**

| Artifact | Format | Description |
|----------|--------|-------------|
| `{strategy}_results.json` | JSON | Per-stratum test results (p-value, effect size, etc.) |
| `synthesis_report.json` | JSON | Weighted global summary across strata |

**Statistical test strategies:**

| Strategy | Test | Purpose |
|----------|------|---------|
| `bias` | One-way ANOVA-style | Measurement differences across sensors (sessions) |
| `precision` | Levene's test | Inter-sensor variance homogeneity |
| `distribution` | KS test | Distribution shape differences across sensors |

---

## Cross-Domain

### cross_map_feature_grid (disabled)

| Field | Value |
|-------|-------|
| DAG key | `cross_map_feature_grid` |
| Category | `cross_domain` |
| Depends on | `spatial_map_single_touch`, `stimulus_extract_features` |
| Output dir | `cross_map_feature_grid/` |

**Outcomes (per session):**

| Artifact | Format | Description |
|----------|--------|-------------|
| `{grid_group}_{gesture_type}_grid.npz` | NumPy NPZ | Per-cell population RF: vertex indices, RF values, bin boundaries, touch count |

Sweeps an N-D feature space (e.g. velocity x pressure 2D grid), partitions
touches into grid cells by feature value, then averages single-touch RFs per
cell.

**Consumed by:** `cross_extract_grid_metrics`

---

### cross_extract_grid_metrics (disabled)

| Field | Value |
|-------|-------|
| DAG key | `cross_extract_grid_metrics` |
| Category | `cross_domain` |
| Depends on | `cross_map_feature_grid`, `spatial_set_camera` |
| Output dir | `cross_extract_grid_metrics/` |

**Outcomes (per session):**

| Artifact | Format | Description |
|----------|--------|-------------|
| `{grid_group}_{gesture_type}_metrics.csv` | CSV | One row per grid cell; columns = grid cell coordinates + RF metrics (centroid_u/v, hull_area_mm2, threshold_area_mm2, etc.) |

**Consumed by:** `cross_render_grid_metrics`

---

### cross_render_grid_metrics

| Field | Value |
|-------|-------|
| DAG key | `cross_render_grid_metrics` |
| Category | `cross_domain` |
| Depends on | `cross_extract_grid_metrics` |
| Output dir | `cross_render_grid_metrics/` |

**Outcomes:**

| Artifact | Format | Description |
|----------|--------|-------------|
| `{metric}/{session_id}_{gesture_type}_heatmap.png` | PNG | 2D heatmap: grid cell coordinates as axes, RF metric as color |

**Rendered metrics:** `touch_count`, `mean_iff`, `perimeter_mm`, `hotspot_area_mm2`, `threshold_area_mm2`, `convex_hull_area_mm2`

---

### cross_render_sessions

| Field | Value |
|-------|-------|
| DAG key | `cross_render_sessions` |
| Category | `cross_domain` |
| Depends on | `spatial_map_single_touch`, `stimulus_extract_features`, `spatial_set_camera` |
| Output dir | `cross_render_sessions/` |

**Outcomes:**

| Artifact | Format | Description |
|----------|--------|-------------|
| `session_comparison_heatmaps/{metric}/{gesture_type}_heatmap.png` | PNG | Sessions on Y-axis, velocity bins on X-axis, color = mean IFF. One per gesture type. |

Uses a 1D velocity-amplitude feature grid to partition touches, then renders
a cross-session comparison heatmap.

---

### cross_extract_cluster_rf (disabled)

| Field | Value |
|-------|-------|
| DAG key | `cross_extract_cluster_rf` |
| Category | `cross_domain` |
| Depends on | `stimulus_cluster_touches` |
| Output dir | `cross_cluster_rf/` |

**Outcomes (per combination x clusterer):**

| Artifact | Format | Description |
|----------|--------|-------------|
| `extraction_summary.json` | JSON | Cluster metadata, n_clusters, per-cluster touch/session counts |
| `forearm_vertices.npz` | NumPy NPZ | Shared forearm mesh vertices |
| `cluster_{label}/spike_counts.csv` | CSV | Vertex index to spike count per cluster |
| `neuron_contacts/{cluster_label}.csv` | CSV | Contact point listing per cluster |
| `session_data/{session_id}/neuron_{neuron_id}.csv` | CSV | Per-neuron per-session touches |

**Consumed by:** `cross_compute_cluster_metrics`, `cross_render_cluster_rf`

---

### cross_compute_cluster_metrics (disabled)

| Field | Value |
|-------|-------|
| DAG key | `cross_compute_cluster_metrics` |
| Category | `cross_domain` |
| Depends on | `cross_extract_cluster_rf`, `spatial_set_camera` |
| Output dir | `cross_cluster_rf/` (writes into same tree) |

**Outcomes (per cluster):**

| Artifact | Format | Description |
|----------|--------|-------------|
| `cluster_{label}/rf_metrics.json` | JSON | Centroid (3D xyz + 2D uv), convex hull area, Gaussian fit params, threshold area |

---

### cross_render_cluster_rf (disabled)

| Field | Value |
|-------|-------|
| DAG key | `cross_render_cluster_rf` |
| Category | `cross_domain` |
| Depends on | `cross_extract_cluster_rf`, `spatial_set_camera` |
| Output dir | `cross_cluster_rf/` (writes into same tree) |

**Outcomes:**

| Artifact | Format | Description |
|----------|--------|-------------|
| `cluster_{label}/heatmap.png` | PNG | 2D projected density heatmap per cluster |
| `{session_id}_combined_heatmap.png` | PNG | Multi-cluster overlay per session |

---

## Interactive Viewers (no file output)

These tasks launch interactive PyQt5/PyVista GUIs. They consume processing
outputs but do not write persistent artifacts (except viewer-support caches).

| DAG key | Description | Primary input |
|---------|-------------|---------------|
| `explore_preparation` | Per-touch time-series inspector + 3D forearm view | Prepared CSVs |
| `explore_feature_space` | Feature-space sweep with live RF heatmap | Precomputed explorer caches |
| `explore_touch_playback` | Frame-by-frame touch animation on 3D forearm | Series-augmented CSVs |
| `explore_single_touch_rf` | Per-touch RF heatmap on 3D forearm (session/block/trial/touch dropdowns) | `single_touch_rf_maps.npz` |
| `explore_touch_population` | Per-touch scatter + 3D forearm heatmap with rectangle selection | Population data |
| `explore_rf_gallery` | Thumbnail gallery of cluster RF heatmaps with 3D view | Cluster RF extraction artifacts |
| `explore_rf_surface` | 3D surface where Z = mean IFF intensity | `_population_response_fields.npz` |

### Viewer-support tasks

| DAG key | Description | Output |
|---------|-------------|--------|
| `explore_precompute_caches` | Pre-builds `.npz` sidecar caches for the Feature-Space Explorer | Cache files alongside session data |

---

## Summary: data flow between outcomes

The table below shows, for each outcome type, which tasks produce it and
which tasks consume it.

| Data artifact | Produced by | Consumed by |
|---------------|-------------|-------------|
| Prepared session CSV | `touch_prepare_sessions` | `touch_compute_series`, `spatial_map_single_touch` |
| Series-augmented CSV | `touch_compute_series` | `stimulus_extract_features`, `spatial_set_camera`, viewers |
| Single-touch RF NPZ | `spatial_map_single_touch` | `spatial_configure_slim_uv`, `spatial_precompute_slim_uv`, `spatial_extract_boundaries`, `cross_map_feature_grid`, `cross_render_sessions` |
| Camera settings JSON | `spatial_set_camera` | `spatial_map_baseline`, `cross_extract_grid_metrics`, `cross_render_cluster_rf`, `cross_render_sessions` |
| SLIM UV config YAML | `spatial_configure_slim_uv` | `spatial_precompute_slim_uv` |
| SLIM UV cache NPZ | `spatial_precompute_slim_uv` | `spatial_extract_boundaries` |
| Population RF NPZ (boundaries) | `spatial_extract_boundaries` | `spatial_compare_boundaries`, `spatial_compare_rf_centers`, `explore_rf_surface` |
| Feature CSVs | `stimulus_extract_features` | `stimulus_cluster_touches`, `stimulus_render_radar`, `stimulus_compare_sessions`, `stimulus_iff_tuning_curves`, `stimulus_iff_instruction_tuning`, `stimulus_analyse_efficacy`, `cross_map_feature_grid`, `cross_render_sessions` |
| Clustered CSV | `stimulus_cluster_touches` | `stimulus_compare_clusters`, `cross_extract_cluster_rf` |
| Cluster RF extraction artifacts | `cross_extract_cluster_rf` | `cross_compute_cluster_metrics`, `cross_render_cluster_rf`, `explore_rf_gallery` |
| Feature grid NPZ | `cross_map_feature_grid` | `cross_extract_grid_metrics` |
| Grid metrics CSV | `cross_extract_grid_metrics` | `cross_render_grid_metrics` |
