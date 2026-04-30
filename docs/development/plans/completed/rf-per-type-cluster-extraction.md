# Plan: RF Per-Type Cluster Extraction

**Date:** 2026-04-30
**Author:** Basil Duvernoy
**Status:** Completed
**Base Branch:** `feature/per-type-clustering-outputs`
**Branch:** `feature/rf-per-type-cluster-extraction`

---

## Overview

The clustering pipeline now writes per-type subdirectories (`tap/`, `stroke_proximal/`,
`stroke_distal/`) instead of a flat `pooled_touch_summary_clustered.csv` when
`per_type_clustering: true` is set on a cluster group. The RF cluster pipeline still
constructs flat paths, so RF extraction silently skips all per-type groups. This plan
updates `rf_cluster_pipeline.py` and `rf_gallery_data.py` to detect the flag and route
each gesture type through its own extraction + visualization pass, producing separate
per-type RF artifact trees and gallery views.

## Problem Statement

`run_cluster_rf_extraction` constructs the clustered CSV path as:
```
<clustering_dir>/<combo>/<clusterer>/pooled_touch_summary_clustered.csv
```
For `pressure_velocity_mean_cartesian_binning` (the active group), this file no longer
exists — the data is at `<clusterer>/tap/pooled_…`, `<clusterer>/stroke_proximal/…`,
etc. The existing guard `if not clustered_csv.exists(): continue` silently skips the
group. Both RF DAG tasks (`extract_receptive_fields_clustered` and
`visualize_receptive_fields_clustered`) are currently `enabled: false` precisely to
avoid this breakage. The gallery viewer (`load_gallery_data`) also hard-codes the flat
`output_dir / combo / clusterer` base path.

Without this fix, per-type clustering produces no RF maps — the entire downstream half
of the analysis pipeline is inoperative for the new cluster group.

## Goals

### In Scope

1. Detect `per_type_clustering: true` from `cluster_group_defs` inside both
   `run_cluster_rf_extraction` and `run_cluster_rf_visualization`
2. When True: iterate over `GESTURE_TYPES`, routing each type through the existing
   extraction / visualization logic with type-scoped paths under
   `<combo>/<clusterer>/<type>/`
3. When False: unchanged flat behavior
4. Update `load_gallery_data` and `launch_gallery_viewer` to accept an optional
   `gesture_type` parameter so per-type artifact trees are loadable
5. Add `gesture_type` to `GalleryData` for display in the gallery viewer title bar
6. Export `GESTURE_TYPES` as a public constant from `clustering_pipeline.py` and
   import it in `rf_cluster_pipeline.py`
7. Enable `extract_receptive_fields_clustered` and `visualize_receptive_fields_clustered`
   in the DAG config once the implementation is verified

### Out of Scope

- Changing the gallery viewer UI to show all three types in one window (type
  selector tab / dropdown) — a single viewer per type is sufficient for now
- Changing how `_build_pairs` works — per-type branching stays inside the pair loop
- Modifying feature extraction, preparation, or series transform stages
- Deprecating `TypeStratifiedClusterer` (different use case)

## Success Criteria

- [ ] Running `extract_receptive_fields_clustered` with `force_processing: true` produces
      `tap/`, `stroke_proximal/`, `stroke_distal/` subdirectories under
      `receptive_field_maps_clustered/pressure_velocity_mean_cartesian_binning/cartesian_binning/`
- [ ] Each type subdirectory contains `extraction_summary.json`, `cluster_*` folders,
      `neuron_touches.json`, and `sessions_metadata.json`
- [ ] Running `visualize_receptive_fields_clustered` renders `*_rf_heatmap_*.png` files
      inside each type's `cluster_*` dirs
- [ ] `gallery_viewer: true` opens one interactive viewer per gesture type, with the
      type name visible in the window title
- [ ] Cluster groups without `per_type_clustering` (or with it `false`) produce
      unchanged flat RF artifact trees
- [ ] Re-running without `force_processing: true` skips up-to-date type extractions

---

## Technical Design

### Approach

Inside the existing `for combo_name, clusterer_name in pairs:` loop in both RF
functions, read the flag from `cluster_group_defs` and build a list of
`(gesture_type_or_None, clustered_csv, base_output)` tuples. When `per_type_clustering`
is True, this list has three entries (one per type); when False, it has one entry (the
current flat paths). The inner logic — session loading, spike extraction, artifact
writing — runs identically for each entry.

```python
group_spec = (cluster_group_defs or {}).get(combo_name, {})
per_type = group_spec.get('per_type_clustering', False)

if per_type:
    runs = [
        (gt,
         clustering_dir / combo_name / clusterer_name / gt / 'pooled_touch_summary_clustered.csv',
         output_dir    / combo_name / clusterer_name / gt)
        for gt in GESTURE_TYPES
    ]
else:
    runs = [
        (None,
         clustering_dir / combo_name / clusterer_name / 'pooled_touch_summary_clustered.csv',
         output_dir    / combo_name / clusterer_name)
    ]

for gesture_type, clustered_csv, base_output in runs:
    ...  # existing inner block, unchanged
```

This requires no changes to `_build_pairs`, the inner extraction logic,
`rf_extraction_io.py`, or `analysis_workflow.py`.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Branch inside existing pair loop (chosen) | Zero impact on inner logic; backward compat; no new public API on `_build_pairs` | Slightly longer outer loop body | **Chosen** |
| Extend `_build_pairs` to return 3-tuples `(combo, clusterer, type\|None)` | Cleaner caller; type visible in pair | Changes function signature used by `analysis_workflow.py`; more refactoring surface | Rejected |
| Filesystem discovery (glob for type subdirs) | No config change needed | Reads stale disk state; diverges from config-driven convention | Rejected |

### Architecture Changes

**`clustering_pipeline.py`** — rename `_GESTURE_TYPES` → `GESTURE_TYPES` (module-level
public constant). No algorithmic change.

**`rf_cluster_pipeline.py`** — two localized changes:
- Import `GESTURE_TYPES` from `clustering_pipeline`
- In `run_cluster_rf_extraction` and `run_cluster_rf_visualization`: add per-type
  branching via the `runs` list pattern above. The inner blocks are unchanged.
- In `launch_gallery_viewer`: add `gesture_type: str | None = None` parameter;
  pass it to `load_gallery_data`.

**`rf_gallery_data.py`** — two localized changes:
- `GalleryData` dataclass: add `gesture_type: Optional[str] = None` field
- `load_gallery_data`: add `gesture_type: str | None = None` parameter; append
  `/ gesture_type` to `base_output` when not None; populate `GalleryData.gesture_type`

**`configs/analyse_workflow_dag.yaml`** — flip both RF tasks to `enabled: true`
after verification.

**`code/src/analysis/CLAUDE.md`** — note per-type RF extraction in the RF pipeline
description.

---

## Implementation Plan

### Phase 1: Export `GESTURE_TYPES` constant

**Goal:** Make the gesture type list importable from `clustering_pipeline.py` so the
RF pipeline has a single authoritative source.

- [x] Rename `_GESTURE_TYPES` → `GESTURE_TYPES` at module level in
      `clustering_pipeline.py` — update the one internal reference in
      `_cluster_combination()`
- [x] Add `from analysis.touch_analytics.clustering_pipeline import GESTURE_TYPES`
      to `rf_cluster_pipeline.py`

**Files Modified:**
- `code/src/analysis/touch_analytics/clustering_pipeline.py` — rename constant
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` — add import

**Dependencies:** None

### Phase 2: Per-type path handling in RF extraction

**Goal:** `run_cluster_rf_extraction` correctly finds and processes per-type CSVs.

- [x] Read `per_type_clustering` from `cluster_group_defs[combo_name]` (with safe
      fallback when `cluster_group_defs` is None or the key is absent)
- [x] Build the `runs` list (per-type or flat) and iterate over it, replacing the
      current hard-coded `clustered_csv` / `base_output` assignments
- [x] For per-type runs: emit a log warning (not an error) when a type's CSV is
      absent — a session set may genuinely lack one type
- [x] Print prefix now includes gesture type when applicable:
      `[RF Extraction] combo/clusterer/tap...`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` — per-type
  branching in `run_cluster_rf_extraction`

**Dependencies:** Phase 1

### Phase 3: Per-type path handling in RF visualization and gallery

**Goal:** `run_cluster_rf_visualization`, `launch_gallery_viewer`, and
`load_gallery_data` all route correctly to per-type artifact trees.

- [x] Apply the same `runs` list pattern to `run_cluster_rf_visualization`;
      update `launch_gallery_viewer` calls to pass `gesture_type`
- [x] Add `gesture_type: str | None = None` to `launch_gallery_viewer` signature;
      pass it through to `load_gallery_data`
- [x] Add `gesture_type: str | None = None` to `load_gallery_data` signature;
      when set, append `/ gesture_type` to `base_output` before all artifact lookups
- [x] Add `gesture_type: Optional[str] = None` field to `GalleryData` dataclass;
      populate it in `load_gallery_data`
- [x] The gallery viewer window title already uses `combo_name` / `clusterer_name`;
      append `gesture_type` when present (e.g. `"… — tap"`)

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` — per-type
  branching in `run_cluster_rf_visualization`; update `launch_gallery_viewer`
- `code/src/analysis/receptive_field_mapping/rf_gallery_data.py` — `GalleryData`
  field; `load_gallery_data` signature + routing

**Dependencies:** Phase 2

### Phase 4: DAG config, CLAUDE.md, and verification

**Goal:** Enable and verify the RF tasks end-to-end.

- [x] In `configs/analyse_workflow_dag.yaml`: set
      `extract_receptive_fields_clustered.enabled: true` and
      `visualize_receptive_fields_clustered.enabled: true`
- [x] Update `code/src/analysis/CLAUDE.md` — add a note in the RF pipeline section
      that `run_cluster_rf_extraction` / `run_cluster_rf_visualization` respect the
      `per_type_clustering` flag from `cluster_group_defs` and produce per-type
      subdirectory trees when enabled
- [ ] Run manual verification (see Testing Plan)

**Files Modified:**
- `configs/analyse_workflow_dag.yaml` — enable RF tasks
- `code/src/analysis/CLAUDE.md` — documentation update

**Dependencies:** Phase 3

---

## Testing Plan

### Unit Tests

No new unit tests required — the per-type branching is structural (path construction
only); all inner logic paths are covered by existing tests.

### Manual Verification

- [ ] Run `extract_receptive_fields_clustered` with `force_processing: true`; confirm
      `receptive_field_maps_clustered/pressure_velocity_mean_cartesian_binning/cartesian_binning/`
      contains `tap/`, `stroke_proximal/`, `stroke_distal/` subdirectories each with
      `extraction_summary.json` and `cluster_*` folders
- [ ] Run `visualize_receptive_fields_clustered` with `force_processing: true`; confirm
      `*_rf_heatmap_*.png` files appear inside each type's `cluster_*` dirs
- [ ] Enable `gallery_viewer: true`; confirm three separate viewer windows open (one
      per type) with the gesture type name in the title bar
- [ ] Disable `per_type_clustering` on another cluster group (or use the legacy
      `pressure_velocity_mean` group); confirm flat RF output is unchanged
- [ ] Re-run without `force_processing: true`; confirm per-type extractions are
      skipped as up-to-date

### Edge Cases

- [ ] A gesture type with no touches in the pooled data: guard logs warning and skips
      (does not raise)
- [ ] `cluster_group_defs` is None (deprecated `feature_combinations` path):
      `per_type_clustering` defaults to False; flat behavior unchanged

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` — note per-type RF extraction in the
      RF cluster pipeline section
- [ ] No user guide needed (config-driven, no new user-facing commands)

---

## Rollback Plan

1. Set `extract_receptive_fields_clustered.enabled: false` and
   `visualize_receptive_fields_clustered.enabled: false` in the DAG config — RF tasks
   stop running
2. Per-type RF artifact trees on disk (`<combo>/<clusterer>/tap/` etc.) do not
   interfere with the flat code path; delete manually if disk space is a concern

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `cluster_group_defs` is None (deprecated `feature_combinations` path used) | Low | Med | Guard: `(cluster_group_defs or {}).get(combo_name, {})` — falls back to `per_type=False` |
| Stale flat RF artifacts on disk from a previous run conflict with per-type output | Med | Low | Different output paths; no conflict. Force re-run clears old sentinel JSON |
| Gallery viewer doesn't surface gesture type visually (confusing when 3 windows open) | Low | Low | Append type to window title in `GalleryData.gesture_type` |

---

## References

- Parent plan: `docs/development/plans/active/per-type-clustering-outputs.md`
- Key files:
  - `code/src/analysis/touch_analytics/clustering_pipeline.py` — `GESTURE_TYPES` constant
  - `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` — `run_cluster_rf_extraction` (line 536), `run_cluster_rf_visualization` (line 822), `launch_gallery_viewer` (line 795), `_build_pairs` (line 501)
  - `code/src/analysis/receptive_field_mapping/rf_gallery_data.py` — `load_gallery_data` (line 122), `GalleryData` (line 65)
  - `configs/analyse_workflow_dag.yaml` — `extract_receptive_fields_clustered` and `visualize_receptive_fields_clustered` tasks
