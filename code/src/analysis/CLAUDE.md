# Analysis subsystem — reference notes

Companion to the root `CLAUDE.md`. Covers touch feature clustering, RF mapping, and
data aggregation. See root file for pipeline run commands and stage status.

---

## Touch Feature Clustering

### Key source files
- `touch_analytics/clustering_pipeline.py` — orchestrator
- `touch_analytics/extraction_pipeline.py` — per-touch feature aggregation
- `touch_analytics/series_pipeline.py` — kinematics / pressure / MoS transforms
- `touch_analytics/pipeline_shared.py` — shared constants (SHARED_COLUMNS, session_id_from_path)
- `touch_analytics/clustering/` — one file per algorithm

### Session ID format
```
YYYY-MM-DD_STNNN-BB   →   e.g. 2022-06-17_ST16-05
```
Extracted from filename via `session_id_from_path()`: splits on `_semicontrolled_`, takes prefix.

### Touch identity columns (present on every per-touch row)
```python
['block_order_id', 'trial_id', 'single_touch_id', 'session_id']
```
These are the join keys when merging across feature folders or linking back to raw CSVs.

### Shared metadata columns (always in pooled output)
```python
SHARED_COLUMNS = [
    'block_order_id', 'trial_id', 'single_touch_id',
    'type_metadata',        # neuronal response type (SAI, RA, CIII …)
    'gesture_type',         # tap | stroke_proximal | stroke_distal
    'mean_contact_x/y/z',   # per-touch contact centroid
    'spike_elicited',       # 1 if any Nerve_spike==1 within touch frames
    'session_id',
]
```

### Available statistical aggregations (auto-applied to every numeric column)
`mean`, `median`, `std`, `min`, `max`, `range`, `skewness`

Outputs land in `4_analysed/touch_features/<agg>/<session>_touch_summary.csv`.
Column naming: `<original_col>_<agg>` e.g. `pressure_mean`, `hand_velocity_amplitude_std`.

### Available features (DATA_TYPE_TO_COLUMNS keys in clustering_pipeline.py)
`contact_area`, `contact_depth`, `hand_velocity` (x/y/z), `hand_velocity_amplitude`,
`hand_acceleration` (x/y/z), `pressure`, `hand_position` (x/y/z),
`mos_strain`, `mos_stress_kpa`, `mos_strain_rate`, `mos_elastic_energy_mj`, `mos_impulse_mns`,
`mechanics_of_solids` (all MoS combined), `location` (uses mean_contact_x/y/z)

### Currently active cluster group
`pressure_velocity_mean_cartesian_binning` (in `configs/analyse_workflow_dag.yaml`):
- Features: `hand_velocity_amplitude[mean]`, `pressure[mean]`
- Method: `cartesian_binning` — 7 bins, equal_frequency, IQR outlier handling
- `per_type_clustering: true` — runs separately for tap / stroke_proximal / stroke_distal

### Available clusterers (registered in clustering/__init__.py)
| Key | Notes |
|---|---|
| `kmeans` | Adaptive K, min_touches_per_cluster constraint |
| `dbscan` | Auto-eps or manual epsilon |
| `binning` | Independent per feature, reports primary |
| `hierarchical` | Ward linkage, adaptive dendrogram cut |
| `type_stratified` | Wraps any base_method, runs per gesture_type |
| `gmm` | BIC-based K selection, full/diag/tied covariance |
| `cartesian_binning` | One cluster per occupied bin combo (currently active) |

### Pooled output structure
`4_analysed/touch_clusters/<group>/<clusterer>/pooled_touch_summary_clustered.csv`

Contains one row per touch, all sessions pooled. Key columns:
- All SHARED_COLUMNS
- Feature columns with aggregation suffix
- `cluster_label` (integer; -1 = DBSCAN noise)
- `bin_<feature>` (cartesian_binning only)
- `cluster_metadata.json` alongside with algorithm params, internal metrics, stability scores

### Pooling logic
All sessions present in **all** requested feature folders are merged (inner join on touch ID cols).
Sessions missing from any folder are skipped with a warning.
Scaling (StandardScaler by default) is applied globally across all sessions together — not per-session.

### Known issues
- `CLUSTERER_REGISTRY` and `get_clusterer` fail to import in the current branch (`sarah_sandbox`).
  Breaks `test_gmm_clusterer.py`, `test_parallax_correction.py`, `test_rf_extraction_io.py`.
  107 other tests pass. Do not treat this as an environment issue.

---

## RF Mapping

### Key source files
- `receptive_field_mapping/rf_cluster_pipeline.py` — cluster-based extraction + viz
- `receptive_field_mapping/rf_simple_pipeline.py` — raw spike-position mapping
- `receptive_field_mapping/rf_extraction_io.py` — all artifact I/O (load/save functions)
- `receptive_field_mapping/rf_metrics.py` — 30+ quantitative RF metrics
- `receptive_field_mapping/rf_mapping_engine.py` — selectivity scoring + DBSCAN RF detection
- `receptive_field_mapping/rf_projection.py` — 3D→2D surface projection
- `receptive_field_mapping/gui/rf_feature_space_explorer.py` — interactive explorer

### Output directory structure
```
4_analysed/
├── receptive_field_maps_simple/<session_id>/
│   ├── spike_positions.csv          # (x, y, z) raw spike contact points
│   └── <session_id>_rf_simple.png
│
└── receptive_field_extraction/<combo_name>/<clusterer_name>[/<gesture_type>]/
    ├── extraction_summary.json      ← sentinel: check this first on rerun
    ├── neuron_touches.json          # {session_id: touch_count}
    ├── sessions_metadata.json       # {session_id: {forearm_ply: path}}
    ├── rf_metrics_summary.csv       # all clusters, all metrics, flat rows
    ├── sessions/<session_id>/
    │   ├── neuron_contacts_xyz.npy  # (N, 3) all contact points for session
    │   └── forearm_vertices.npy     # (M, 3) PLY vertex cache
    └── cluster_<i>/
        ├── spike_counts.csv         # pooled: x, y, z, spike_count, unique_touch_spike_count
        ├── cluster_description.json
        ├── rf_metrics.json
        ├── sessions/<session_id>/
        │   ├── session_spike_counts.csv
        │   └── cluster_contacts_xyz.npy
        └── <session_id>_rf_heatmap_*.png
```

### Idempotency / staleness
- `extraction_summary.json` — written **last** by extraction, checked **first** on rerun
- `rf_visualization_summary.json` — tracks projection_method + disjoint_mask_distance_mm + extraction mtime
- Change projection params (e.g. `cylindrical_unwrap` → `tangent_plane`) → visualization re-runs automatically, extraction does not

### Core I/O functions (rf_extraction_io.py)
```python
# Session-level
save_neuron_contacts(output_dir, session_id, xyz: np.ndarray)   # (N, 3)
load_neuron_contacts(output_dir, session_id) -> np.ndarray

# Cluster × session
save_cluster_session_data(cluster_dir, session_id, spike_df, contacts_xyz)
load_cluster_session_data(cluster_dir, session_id) -> (DataFrame, np.ndarray)
# spike_df columns: x, y, z, spike_count, unique_touch_spike_count

# Pooled spike counts
# spike_counts.csv columns: x, y, z, spike_count, unique_touch_spike_count
# Loaded directly via pd.read_csv(cluster_dir / 'spike_counts.csv')

# Sentinels
save_extraction_summary(output_dir, combo, clusterer, summary_dict)
load_extraction_summary(output_dir) -> dict
```

### RF metrics (rf_metrics.py — RFMetrics dataclass)
Key fields for SAI aggregation work:
```
convex_hull_area_mm2    — RF size (surface area of response zone)
hotspot_area_mm2        — top-20% spike density area
gaussian_sigma_major/minor_mm — Gaussian fit half-widths
weighted_centroid_3d    — (x, y, z) spike-count-weighted center
peak_spike_count        — highest spike count at any contact point
mean_spike_count        — average across all responsive points
total_spikes            — sum of all spikes in cluster
sparsity_index          — how concentrated the response is
gaussian_converged      — bool; check before using gaussian_* fields
r_squared               — Gaussian fit quality
```

Aggregated to `rf_metrics_summary.csv` (flat rows, centroid tuples expanded to `_x/_y/_z`).

### Projection methods
- `cylindrical_unwrap` — recommended for forearm (PCA cylinder axis, arc_length × height)
- `tangent_plane` — local surface normal via KDTree neighbors

### Contact point deduplication
Points are rounded to a 1 mm grid when building spike counters to merge serialization artifacts.
Spike counts represent number of 1kHz frames where Nerve_spike==1 at that rounded location.
`unique_touch_spike_count` = number of distinct touches that elicited a spike at that point.

### Feature-space explorer (interactive)
Run via `explore_rf_feature_space` task or directly:
```python
from analysis.receptive_field_mapping.gui.rf_feature_space_explorer import launch_feature_space_explorer
```
- Left panel: 2D scatter (pressure × velocity_signed, coloured by gesture_type)
- Right panel: 3D forearm mesh with live spike-density heatmap
- Draggable rectangle in scatter → filters which touches contribute to heatmap
- Caches data in `.npz` sidecar alongside the series-augmented CSV (mtime-validated)

---

## Data Aggregation — Current Capabilities and Opportunities

### What is already aggregated

**Within a session, within a cluster** (`session_spike_counts.csv`):
- All 1kHz frames where Nerve_spike==1 are counted per (x, y, z) contact point
- `spike_count` = total spike frames at that point
- `unique_touch_spike_count` = number of distinct touches firing there

**Across sessions, within a cluster** (`spike_counts.csv`):
- `session_spike_counts.csv` pooled across all sessions via Counter union
- Contact points from all sessions land in one CSV

**Across clusters** (`rf_metrics_summary.csv`):
- Per-cluster RF metrics flattened to one row each, all clusters in one file

**Touch-level features** (`pooled_touch_summary_clustered.csv`):
- Every touch from every session in one CSV
- `session_id` column preserved → filter to single session or compare across sessions

### What is NOT aggregated (opportunities)

1. **Per-session RF metrics** — `rf_metrics.json` exists per cluster but there is no file that
   aggregates metrics across sessions for the same cluster. To compare how RF size/location
   varies across neurons (sessions), you need to load `rf_metrics_summary.csv` and filter by
   `session_id` or build a new aggregator that reads per-session `session_spike_counts.csv`.

2. **Cross-session spike-count distributions** — `spike_counts.csv` pools sessions without
   preserving which session contributed each point. The per-session breakdown lives in
   `cluster_<i>/sessions/<session_id>/session_spike_counts.csv` but is never joined across
   sessions into a single multi-session view.

3. **Spike rate per touch type per session** — `spike_elicited` is binary per touch. No
   existing aggregation computes mean firing rate (spikes/sec) or total spike count per
   touch grouped by gesture_type for a given session.

4. **Feature × response coupling** — the Feature-Space Explorer visualises this interactively
   but produces no saved summary. A static version would compute, per feature-space bin:
   mean(spike_elicited), mean(spike_count), per session and across sessions.

5. **Cross-neuron (cross-session) RF comparison table** — `rf_metrics_summary.csv` is per
   pipeline run (one neuron). To compare RF properties across multiple SAI recordings you
   would need to load multiple `rf_metrics_summary.csv` files (one per session's pipeline run)
   and concatenate them, adding a `session_id` column. The data is all there; no loader exists.

### Key columns for aggregation scripts

| Purpose | CSV to load | Key columns |
|---|---|---|
| Per-touch response | `pooled_touch_summary_clustered.csv` | `session_id`, `gesture_type`, `spike_elicited`, `cluster_label`, feature cols |
| Per-contact-point spikes (pooled) | `cluster_<i>/spike_counts.csv` | `x`, `y`, `z`, `spike_count`, `unique_touch_spike_count` |
| Per-contact-point spikes (per session) | `cluster_<i>/sessions/<session>/session_spike_counts.csv` | same |
| RF geometry per cluster | `rf_metrics_summary.csv` | `cluster_label`, `convex_hull_area_mm2`, `weighted_centroid_3d_x/y/z`, etc. |
| Raw kinematics (per frame) | `series_transforms/<session>_series_augmented.csv` | `Nerve_spike`, `pressure`, `hand_velocity_amplitude`, `gesture_type`, touch ID cols |
