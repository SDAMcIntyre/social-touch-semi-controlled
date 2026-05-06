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

## Analysis Pipeline

All tasks are configured in `configs/analyse_workflow_dag.yaml` and run via:
```bash
# Full pipeline (all enabled tasks):
conda run -n pcl_env python code/scripts/analysis_workflow.py --dag-config configs/analyse_workflow_dag.yaml
# Single stage:
conda run -n pcl_env python code/scripts/analysis_workflow.py --dag-config configs/analyse_workflow_dag.yaml --tasks touch_preparation
# or via GUI:
python code/scripts/launch_pipeline_gui.py
```

### Infrastructure
- **Data root**: `semi_controlled_project_data_config.json` → `project_data_root` → OneDrive path ending in `.../Kinect_MNG_data/semi-controlled`
- **Session data**: `<data_root>/3_merged/<session_id>/` — contains `*_semicontrolled_aggregated_session.csv`
- **Analysis outputs**: `<data_root>/4_analysed/` — created on first pipeline run
- **Kinect configs**: `configs/kinect_configs/` does not exist (preprocessing-stage artefact). Pipeline falls back to scanning `3_merged/` directly.
- **Prefect**: version 3.4.x, profile `ephemeral`. Startup timeout set to 120 s (`PREFECT_SERVER_EPHEMERAL_STARTUP_TIMEOUT_SECONDS`). Default 20 s was too short on this machine.

### Key source files
- `code/scripts/analysis_workflow.py` — main entry point; one `@flow` per stage; dispatches via `run_batch_analysis()`
- `code/src/analysis/touch_analytics/preparation_pipeline.py` — Stage 0: load → block-ID synthesis → cubic interpolation → gesture type → drop columns
- `code/src/utils/pipeline/session_config_resolver.py` — resolves kinect config entries to file paths
- `code/src/primary_processing/models/kinect_config.py` — Pydantic model; `session_merged_output_dir` is the key field for analysis

### Stage status

Last checked: 2026-05-05. Status applies to session `2022-06-17_ST16-05`.

**Next session priority:** Before running Stage 2a, spend time understanding the Stage 1
output (`_series_augmented.csv`) and the Stage 2b heatmap. Review what each derived column
means scientifically and what the RF map tells us about this SAI unit before proceeding.

| Stage | Task key | Output path | Status |
|---|---|---|---|
| 0 | `touch_preparation` | `4_analysed/preparation/<session>_prepared.csv` | ✅ complete |
| 1 | `touch_series_transforms` | `4_analysed/series_transforms/<session>_series_augmented.csv` | ✅ complete |
| 2a | `touch_feature_extraction` | `4_analysed/touch_features/<agg>/<session>_touch_summary.csv` | not started |
| 2b | `map_receptive_fields_simple` | `4_analysed/receptive_field_maps_simple/<session>/` | ✅ complete |
| 3 | `touch_clustering` | `4_analysed/touch_clusters/<group>/<clusterer>/` | not started |
| 4 | `extract_receptive_fields_clustered` | `4_analysed/receptive_field_extraction/<combo>/<clusterer>/` | not started |
| 5 | `visualize_receptive_fields_clustered` | same dir, adds heatmaps + rf_metrics.json | not started |
| 6 | `explore_rf_feature_space` | interactive GUI only (no file output) | not started |

`map_receptive_fields_simple` and `touch_preparation` have no `depends_on` — they can run first.
`extract_receptive_fields_clustered` depends on `touch_clustering`; split from visualization so
projection params can be changed without re-extracting spike counts.

### Prepared CSV columns (Stage 0 output)
`time`, `led_on`, `contact_detected`, `contact_location_x/y/z`, `contact_area`, `contact_depth`,
`contact_points`, `sticker_blue/green/yellow_position_x/y/z`, `trial_id`, `single_touch_id`,
`type_metadata`, `speed_metadata`, `contact_area_metadata`, `force_metadata`,
`Nerve_spike`, `Nerve_freq`, `Nerve_TTL`, `source_block_file`, `block_order_id`, `gesture_type`

Dropped by Stage 0: `frame_index`, `green_levels`, `time_nerve`, `time_kinect`, `trial_on`

For clustering, RF mapping, and data aggregation details see [`code/src/analysis/CLAUDE.md`](code/src/analysis/CLAUDE.md).

---

## Working with GitHub Issues

This branch (`sarah_sandbox`) is a working/exploration branch and is not pushed or merged into `dev`. Code changes made here are discarded at the end of the current exploration. 

When asked to post or update a GitHub issue:
- Write it as a **self-contained description** that someone working only on `dev` can understand and act on — no references to this branch, no "we found" or "I fixed", no session-specific context.
- Present fixes as **suggested fixes**, not as completed work.
- The issue body should be a **single, coherent description**: one summary, one root cause, one suggested fix. Do not append update comments; instead, edit the issue body in place to incorporate new information while preserving the format.
- As more is learned (e.g. after testing a fix or discovering a downstream issue), update the issue body to reflect the current full understanding — replace stale details, extend the description, do not layer comments on top.

---

## Common Gotchas

- **`configs/kinect_configs/` missing**: Per-session preprocessing config YAMLs are not in the
  repo. `analysis_workflow.py` falls back to scanning `3_merged/` directly when they are absent,
  so the analysis pipeline still runs.

- **macOS: `pyk4a` import error — `map_receptive_fields_simple` broken**: `pyk4a` is
  Windows-only. `_rf_mapping()` in `analysis_workflow.py` defers the import, which protects
  non-RF tasks. But `receptive_field_mapping/__init__.py` eagerly imports `rf_camera_angle_task`,
  which chains through `preprocessing.forearm_extraction` → `pyk4a`. `rf_simple_pipeline.py`
  itself has no `pyk4a` dependency; the fix is to move `rf_camera_angle_task` out of the package
  `__init__.py` into a separate deferred import used only by clustered RF flows. The batch runner
  also masks this failure — it reports SUCCESS even when the Prefect flow raises an exception.
  See `docs/development/knowledge-base/bug-pyk4a-blocks-rf-simple-pipeline-macos.md`.

- **`render_forearm_heatmap` requires `render_context` — fixed on this branch, not in dev**:
  The function had an unconditional guard requiring `RFRenderContext`. `rf_simple_pipeline.py`
  does not supply one. Fixed in `rf_cluster_visualizer.py`: centroid now falls back to
  `spike_counts_df` mean; 2D projection hull accesses guarded with `render_context is not None`.
  These changes are on `sarah_sandbox` only — need implementing in `dev`. See GitHub issue #70.

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
