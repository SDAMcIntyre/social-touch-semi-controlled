# CLAUDE.md — social-touch-semi-controlled

## Environment

```bash
conda activate pcl_env
```

Python 3.10. macOS install: `pyk4a` and `cupy` are excluded (hardware-only on Windows).
VTK is conda-managed (9.6.x); do not reinstall via pip. pynput is pinned to 1.7.6 due to
a macOS conflict between psychopy and pynput ≥1.8 over pyobjc-framework-Quartz.

Install/update the project editable install from the repo root:
```bash
pip install -e .
```

---

## Active Session

**Session ID:** `2022-06-17_ST16-05`
**Unit type:** SAI (slowly adapting type I mechanoreceptor)

Session is listed in `configs/analyse_workflow_dag.yaml` under `parameters.kinect_configs`
as `valid_configs_ST16-05`. That YAML is the primary settings file for all analysis tasks.

SAI units have sustained firing throughout touch contact (vs RA/CIII which fire at onset/offset).
This matters when interpreting spike_elicited and the spike-count heatmaps in RF mapping — expect
spike activity spread across the full contact duration, not just at contact edges.

---

## Analysis Pipeline Overview

All tasks are configured in `configs/analyse_workflow_dag.yaml` and run via:
```bash
# Full pipeline (all enabled tasks):
conda run -n pcl_env python code/scripts/analysis_workflow.py --dag-config configs/analyse_workflow_dag.yaml
# Single stage (added --tasks flag):
conda run -n pcl_env python code/scripts/analysis_workflow.py --dag-config configs/analyse_workflow_dag.yaml --tasks touch_preparation
# or via GUI:
python code/scripts/launch_pipeline_gui.py
```

### Infrastructure notes
- **Data root**: `semi_controlled_project_data_config.json` → `project_data_root` → OneDrive path ending in `.../Kinect_MNG_data/semi-controlled`
- **Session data**: `<data_root>/3_merged/<session_id>/` — contains `*_semicontrolled_aggregated_session.csv`
- **Analysis outputs**: `<data_root>/4_analysed/` — created on first pipeline run
- **Kinect configs**: `configs/kinect_configs/` does not exist (preprocessing-stage artefact). Pipeline falls back to scanning `3_merged/` directly.
- **Prefect**: version 3.4.x, profile `ephemeral`. Startup timeout set to 120 s (`PREFECT_SERVER_EPHEMERAL_STARTUP_TIMEOUT_SECONDS`). Default 20 s was too short on this machine.

### Key source files for `analysis_workflow.py`
- `code/scripts/analysis_workflow.py` — main entry point; defines one `@flow` per stage; dispatches via `run_batch_analysis()`
- `code/src/utils/pipeline/session_config_resolver.py` — resolves kinect config entries to file paths
- `code/src/primary_processing/data_access/kinect_config_filehandler.py` — loads/resolves YAML with `{variable}` substitution
- `code/src/primary_processing/models/kinect_config.py` — Pydantic model for per-session config; `session_merged_output_dir` is the key field for analysis
- `code/src/analysis/touch_analytics/preparation_pipeline.py` — Stage 0 implementation (`run_preparation`, `_prepare_session`)
  - Inputs: `*_semicontrolled_aggregated_session.csv` (1 kHz, NaN gaps at 30 Hz boundaries)
  - Steps: block-ID synthesis → cubic interpolation of NaN gaps → gesture type assignment (tap/stroke_proximal/stroke_distal) → drop unused columns
  - Outputs: `<session>_prepared.csv` + `gesture_type_summary.csv` → `4_analysed/preparation/`

### Stage order and outputs

Last checked: 2026-05-04. Status applies to session `2022-06-17_ST16-05`.

| Stage | Task key | Output path | Status |
|---|---|---|---|
| 0 | `touch_preparation` | `4_analysed/preparation/<session>_prepared.csv` | ✅ complete |
| 1 | `touch_series_transforms` | `4_analysed/series_transforms/<session>_series_augmented.csv` | not started |
| 2a | `touch_feature_extraction` | `4_analysed/touch_features/<agg>/<session>_touch_summary.csv` | not started |
| 2b | `map_receptive_fields_simple` | `4_analysed/receptive_field_maps_simple/<session>/` | not started |
| 3 | `touch_clustering` | `4_analysed/touch_clusters/<group>/<clusterer>/` | not started |
| 4 | `extract_receptive_fields_clustered` | `4_analysed/receptive_field_extraction/<combo>/<clusterer>/` | not started |
| 5 | `visualize_receptive_fields_clustered` | same dir, adds heatmaps + rf_metrics.json | not started |
| 6 | `explore_rf_feature_space` | interactive GUI only (no file output) | not started |

`map_receptive_fields_simple` and `touch_preparation` have no `depends_on` — they can run first.
`extract_receptive_fields_clustered` depends on `touch_clustering`; it is split from visualization
intentionally so projection params can be changed without re-extracting spike counts.

---

## Touch Feature Clustering

### Key source files
- `code/src/analysis/touch_analytics/clustering_pipeline.py` — orchestrator
- `code/src/analysis/touch_analytics/extraction_pipeline.py` — per-touch feature aggregation
- `code/src/analysis/touch_analytics/series_pipeline.py` — kinematics / pressure / MoS transforms
- `code/src/analysis/touch_analytics/pipeline_shared.py` — shared constants (SHARED_COLUMNS, session_id_from_path)
- `code/src/analysis/touch_analytics/clustering/` — one file per algorithm

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
`pressure_velocity_mean_cartesian_binning` (in `analyse_workflow_dag.yaml`):
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

---

## RF Mapping

### Key source files
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` — cluster-based extraction + viz
- `code/src/analysis/receptive_field_mapping/rf_simple_pipeline.py` — raw spike-position mapping
- `code/src/analysis/receptive_field_mapping/rf_extraction_io.py` — all artifact I/O (load/save functions)
- `code/src/analysis/receptive_field_mapping/rf_metrics.py` — 30+ quantitative RF metrics
- `code/src/analysis/receptive_field_mapping/rf_mapping_engine.py` — selectivity scoring + DBSCAN RF detection
- `code/src/analysis/receptive_field_mapping/rf_projection.py` — 3D→2D surface projection
- `code/src/analysis/receptive_field_mapping/gui/rf_feature_space_explorer.py` — interactive explorer

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

---

## Common Gotchas

- **`configs/kinect_configs/` missing**: Per-session preprocessing config YAMLs are not in the
  repo. `analysis_workflow.py` falls back to scanning `3_merged/` directly when they are absent,
  so the analysis pipeline still runs. The `--tasks <name>` flag was added to allow running a
  single stage (e.g. `--tasks touch_preparation`).

- **macOS: `pyk4a` import error**: RF mapping imports chain through `preprocessing.forearm_extraction`
  → `pyk4a` (Windows-only). Fixed in `analysis_workflow.py` by deferring those imports into
  `_rf_mapping()`, called only inside RF flow functions.

- **`clustering` import error**: `CLUSTERER_REGISTRY` and `get_clusterer` fail to import in
  the current branch. This breaks `test_gmm_clusterer.py`, `test_parallax_correction.py`, and
  `test_rf_extraction_io.py`. 107 other tests pass. Do not treat this as an environment issue.

- **vtk version**: conda has vtk 9.6.1; requirements.txt pins 9.4.2. The conda version is
  correct to keep. Never `pip install vtk` in this environment.

- **NaNs survive cubic interpolation at touch group boundaries**: cubic interpolation only fills
  interior NaNs (requires valid samples on both sides). The Kinect often drops out before the
  touch group boundary ends, so the last rows of a touch group commonly remain NaN in the prepared
  CSV. Confirmed: 100% of stroke groups had a NaN at their final row (pre-interpolation), and these
  survive into `_prepared.csv`. `classify_gesture_type` handles this correctly via `dropna()`, but
  any downstream code reading `contact_location_x.iloc[-1]` will get NaN for most touches.

- **Contact point parsing**: `contact_points` column in raw CSVs is a string
  `"[[x y z] [x y z] ...]"` (space-separated, not comma). Parsed by regex `r'\[([^\]]+)\]'`.
  Forward-filled from 30Hz to 1kHz at load time — raw CSV has NaN between frames.

- **`unique_touch_spike_count` vs `spike_count`**: spike_count counts all 1kHz frames with
  a spike (an SAI touching for 500ms at 10 spikes/s = 5 counts at that location).
  unique_touch_spike_count counts distinct touches — better for RF size estimation.

- **Sentinel check order**: extraction always writes `extraction_summary.json` last.
  If a run was interrupted, delete this file (not the data files) to force a clean rerun.

- **`per_type_clustering: true`** creates a subdirectory per gesture type inside the
  clusterer folder: `touch_clusters/<group>/cartesian_binning/tap/`, `.../stroke_proximal/`,
  `.../stroke_distal/`. RF extraction then looks for clustered CSVs in these subdirectories.

- **Deduplication rounding**: contact points are snapped to a 1 mm grid when building
  spike counters. Two contacts 0.4 mm apart merge into one bin. This is intentional but
  means `spike_counts.csv` point count < raw contact frame count.

- **ReductionPipeline scaling is global**: StandardScaler is fit on the pooled cross-session
  data. Adding a new session changes all scaler parameters → cluster labels shift. Rerun
  clustering when adding sessions.
