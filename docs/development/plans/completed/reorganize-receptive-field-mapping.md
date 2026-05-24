# Plan: Reorganize `receptive_field_mapping/` and unify shared constants

**Date:** 2026-05-20
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-05-24 18:03
**Base Branch:** `feature/split-analysis-workflow-processing-viewers`
**Branch:** `feature/reorganize-receptive-field-mapping`

---

## Overview

**What:** Reorganize the flat 33-file `receptive_field_mapping/` directory
into cohesive sub-packages, consolidate duplicated domain constants into a
single-source-of-truth module, and split the 1,662-line
`rf_cluster_pipeline.py` monolith into focused modules.

**Why:** A newcomer opening `receptive_field_mapping/` sees pipelines,
renderers, data loaders, metrics, surface code, and SLIM UV code jumbled
together with no grouping. Domain constants (`GESTURE_TYPES`, touch ID
columns, `'Nerve_spike'`/`'contact_points'` strings) are duplicated in 3-4
places — one copy even diverges (adds `'stroke_unknown'`). Cross-package
imports from RF mapping into `touch_analytics` create hidden coupling.

**How:** Create 5 sub-packages inside `receptive_field_mapping/`, extract
shared constants to `analysis/pipeline/shared_constants.py`, split the
monolith into 4 files, and update all imports while keeping the public
`__init__.py` API unchanged.

## Problem Statement

The `code/src/analysis/receptive_field_mapping/` package has 33 Python files
in a flat directory. Files serving fundamentally different roles (pipeline
orchestration, matplotlib rendering, data I/O, metric computation, mesh
projection) sit side-by-side with no sub-package structure.

Additionally, domain constants are scattered and duplicated:
- `GESTURE_TYPES` defined 3 times (one with an extra value)
- Touch identity columns defined 4 times (`TOUCH_KEYS`, `_TOUCH_ID_COLS`,
  `_KEY_COLS`, `_GROUP_COLS`)
- `'Nerve_spike'` and `'contact_points'` hardcoded as string literals in
  20+ locations
- `_VALID_NEURON_MODES` defined identically in 3 files
- 7 cross-package imports from RF mapping into `touch_analytics` internals

This makes the code hard to navigate, fragile to refactor, and risky for
newcomers who cannot tell what belongs to which concern.

## Goals

### In Scope
1. Group 33 flat RF mapping files into 5 cohesive sub-packages
2. Create `analysis/pipeline/shared_constants.py` as single source of truth
   for domain constants used across both sub-systems
3. Split `rf_cluster_pipeline.py` (1,662 lines) into 4 focused modules
4. Eliminate cross-package imports from RF mapping into `touch_analytics`
5. Replace all hardcoded `'Nerve_spike'`/`'contact_points'` strings with
   named constants
6. Update all internal imports, tests, and workflow scripts

### Out of Scope
- Structural changes to `touch_analytics/` (already well-organized)
- Renaming files for clarity (e.g., `rf_simple_pipeline.py`) — avoid churn
- Moving `preparation_viewer_data.py`'s reverse import of `rf_data_loader` —
  single call site, low priority
- Changing any algorithmic behavior or data flow

## Success Criteria

- [ ] All 33 flat files moved to appropriate sub-packages
- [ ] `analysis/pipeline/shared_constants.py` exists and is the sole
      definition site for `GESTURE_TYPES`, `TOUCH_ID_COLS`, `NERVE_SPIKE_COL`,
      `CONTACT_POINTS_COL`, `NEURON_MODES`, `session_id_from_path`,
      `filter_enabled_profiles`
- [ ] No duplicate constant definitions remain (verified by grep audit)
- [ ] `rf_cluster_pipeline.py` split into 4 files, none exceeding ~660 lines
- [ ] Zero cross-package imports from RF mapping into `touch_analytics`
      (except `DISCRETIZATION_CONFIG` and `DATA_TYPE_TO_COLUMNS` which stay)
- [ ] `receptive_field_mapping/__init__.py` public API unchanged
- [ ] `pytest code/tests/` passes
- [ ] Workflow scripts load without import errors

---

## Technical Design

### Approach

Reorganize by concern into sub-packages with an acyclic dependency graph.
Centralize shared constants in the existing `analysis.pipeline` package,
which was purpose-built as the shared layer between `touch_analytics/` and
`receptive_field_mapping/`.

The `__init__.py` re-export pattern ensures all external consumers
(workflow scripts, tests) continue importing from
`analysis.receptive_field_mapping` without path changes. Only internal
imports need updating.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Sub-packages (chosen) | Clear concern separation, navigable for newcomers, acyclic deps | Import path churn inside RF mapping | **Chosen** — one-time cost, permanent benefit |
| Flat with naming convention | Zero import changes | Still 33 files, no IDE navigation benefit, doesn't fix constants | Rejected |
| Merge RF mapping into touch_analytics | Single package tree | RF mapping is an independent sub-system, forced coupling | Rejected |
| Constants in `analysis/__init__.py` | Simple | Pollutes top-level namespace, mixes concerns | Rejected |
| Constants in `analysis/constants.py` | Dedicated module | `analysis.pipeline` already exists as shared layer | Rejected — use existing infrastructure |

### Architecture Changes

**New module:** `analysis/pipeline/shared_constants.py`

```python
GESTURE_TYPES = ('tap', 'stroke_proximal', 'stroke_distal')
TOUCH_ID_COLS = ('block_order_id', 'trial_id', 'single_touch_id')
TOUCH_ID_COLS_WITH_SESSION = (*TOUCH_ID_COLS, 'session_id')
NERVE_SPIKE_COL = 'Nerve_spike'
CONTACT_POINTS_COL = 'contact_points'
NERVE_FREQ_COL = 'Nerve_freq'
LOCATION_SHARED_COLS = ('mean_contact_x', 'mean_contact_y', 'mean_contact_z')
LOCATION_BASE_COLS = ('contact_location_x', 'contact_location_y', 'contact_location_z')
NEURON_MODES = ('iff', 'spike')
def session_id_from_path(p: Path) -> str: ...
def filter_enabled_profiles(profiles: dict) -> dict: ...
```

**New directory structure:**

```
receptive_field_mapping/
    __init__.py              # updated re-exports (public API unchanged)
    config.py                # was rf_mapping_config.py
    engine.py                # was rf_mapping_engine.py
    pipelines/               # DAG-invoked orchestrators
    rendering/               # matplotlib/pyvista image output
    data/                    # loading, I/O, data models
    metrics/                 # pure RF metric computation
    surface/                 # 3D-to-2D projection, mesh, SLIM UV
    gui/                     # existing, unchanged
```

**Dependency layering** (acyclic by construction):
```
surface/ <-- metrics/ <-- data/ <-- rendering/ <-- pipelines/ --> gui/
```

**Key design decisions:**

- **GUI data loaders stay in `data/`, not `gui/`:** `touch_playback_data.py`,
  `rf_explorer_data.py`, `rf_gallery_data.py`, and `touch_population_data.py`
  are imported by both pipeline files and GUI viewers. Moving them to `gui/`
  would create pipelines importing from GUI.

- **`config.py` and `engine.py` stay at root:** Tiny leaf-node modules
  (~200 lines total) with no internal imports. The `rf_mapping_` prefix is
  redundant inside `receptive_field_mapping/`.

- **Protect the neuron-centroid invariant** (from knowledge base
  `note-rf-cluster-coordinate-spaces.md`): when splitting
  `rf_cluster_pipeline.py`, the 3D-to-2D boundary at
  `rf_projection.py::project_to_2d()` must remain intact. The split is a
  structural refactor only — no calling convention changes.

- **Constants allowlist + grep audit** (from knowledge base
  `note-kinect-depth-access-single-path.md` pattern): after migration,
  verify no duplicate definitions remain by grepping for the old constant
  names.

### `rf_cluster_pipeline.py` split plan

| Lines | Responsibility | New file |
|-------|---------------|----------|
| 76–735 | Extraction orchestration + backward-compat `run_cluster_rf_mapping` | `pipelines/rf_cluster_pipeline.py` (~660 lines) |
| 1073–1235 | `run_cluster_rf_metrics_computation` | `pipelines/rf_cluster_metrics_pipeline.py` (~163 lines) |
| 1237–1541 | `run_cluster_rf_visualization` | `pipelines/rf_cluster_visualization_pipeline.py` (~305 lines) |
| 736–1071, 1594–1662 | `precompute_explorer_caches`, all `launch_*` functions | `pipelines/rf_cluster_gui_launchers.py` (~530 lines) |

Shared private helpers (`_build_pairs`, `_format_cluster_folder`,
`_build_cluster_description`) stay in the extraction file and get imported
by metrics/visualization.

### What stays where it is (not centralized)

- `SHARED_COLUMNS` — stays in `pipeline_shared.py` (describes touch_analytics
  output schema, not a shared domain concept)
- `DATA_TYPE_TO_COLUMNS` — stays in `clustering_pipeline.py` (only used by
  RF to produce human-readable cluster descriptions)
- `DISCRETIZATION_CONFIG` — stays in `touch_config.py` (used by
  `rf_data_loader.py` for legacy selectivity pipeline)
- `_TqdmLineWrapper` — stays in `pipeline_shared.py` (internal plumbing)

---

## Implementation Plan

### Phase 1: Shared constants and migration
**Started:** 2026-05-20
**Completed:** 2026-05-20
**Goal:** Establish single source of truth for all cross-package constants
and eliminate duplicate definitions.

- [x] Task 1.1 — Create `analysis/pipeline/shared_constants.py` with all
      unified constants and shared utility functions
- [x] Task 1.2 — Update `analysis/pipeline/__init__.py` to re-export new symbols
- [x] Task 1.3 — Migrate touch_analytics consumers: replace definitions in
      `clustering_pipeline.py`, `extraction_pipeline.py`,
      `preparation_pipeline.py`, `preparation/gesture_type.py`,
      `pipeline_shared.py` with imports from `analysis.pipeline`; add
      backward-compat re-exports where needed
- [x] Task 1.4 — Replace hardcoded `'Nerve_spike'`/`'contact_points'` strings
      with `NERVE_SPIKE_COL`/`CONTACT_POINTS_COL` in touch_analytics files:
      `extraction_pipeline.py`, `touch_analysis.py`, `interpolation.py`,
      `statistical.py`, `preparation_viewer_data.py`, `touch_preparation_viewer.py`
- [x] Task 1.5 — Migrate RF mapping consumers: update imports in
      `rf_cluster_pipeline.py`, `rf_population_heatmap.py`,
      `rf_population_map_pipeline.py`, `rf_simple_pipeline.py`,
      `rf_single_touch_pipeline.py` to import from `analysis.pipeline`
- [x] Task 1.6 — Replace `_VALID_NEURON_MODES` in `rf_single_touch_pipeline.py`,
      `rf_population_grid_pipeline.py`, `gui/single_touch_rf_explorer.py` with
      import of `NEURON_MODES`
- [x] Task 1.7 — Replace hardcoded `'Nerve_spike'`/`'contact_points'` strings
      in RF mapping files: `rf_cluster_pipeline.py`, `rf_explorer_data.py`,
      `rf_mapping_config.py`, `touch_playback_data.py`, `touch_population_data.py`
- [x] Task 1.8 — Fix `gui/rf_feature_space_explorer.py`: replace local
      `_GESTURE_TYPES` with `_GESTURE_TYPES_WITH_UNKNOWN = (*GESTURE_TYPES, 'stroke_unknown')`
      importing canonical `GESTURE_TYPES`
- [x] Task 1.9 — Run grep audit to confirm no duplicate constant definitions remain

**Files Modified:**
- `code/src/analysis/pipeline/shared_constants.py` — NEW
- `code/src/analysis/pipeline/__init__.py` — add re-exports
- `code/src/analysis/touch_analytics/clustering_pipeline.py` — import + re-export
- `code/src/analysis/touch_analytics/extraction_pipeline.py` — replace constants
- `code/src/analysis/touch_analytics/preparation_pipeline.py` — replace `_GESTURE_TYPES`
- `code/src/analysis/touch_analytics/preparation/gesture_type.py` — replace `_GROUP_COLS`
- `code/src/analysis/touch_analytics/pipeline_shared.py` — import + re-export
- `code/src/analysis/touch_analytics/touch_analysis.py` — replace string literals
- `code/src/analysis/touch_analytics/preparation/interpolation.py` — replace string literals
- `code/src/analysis/touch_analytics/representation/feature_characterization/statistical.py` — replace string literal
- `code/src/analysis/touch_analytics/gui/preparation_viewer_data.py` — replace string literals
- `code/src/analysis/touch_analytics/gui/touch_preparation_viewer.py` — replace string literals
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` — replace imports + constants
- `code/src/analysis/receptive_field_mapping/rf_population_heatmap.py` — replace import
- `code/src/analysis/receptive_field_mapping/rf_population_map_pipeline.py` — replace import
- `code/src/analysis/receptive_field_mapping/rf_simple_pipeline.py` — replace import
- `code/src/analysis/receptive_field_mapping/rf_single_touch_pipeline.py` — replace import + constant
- `code/src/analysis/receptive_field_mapping/rf_population_grid_pipeline.py` — replace constant
- `code/src/analysis/receptive_field_mapping/gui/single_touch_rf_explorer.py` — replace constant
- `code/src/analysis/receptive_field_mapping/gui/rf_feature_space_explorer.py` — fix gesture types
- `code/src/analysis/receptive_field_mapping/rf_explorer_data.py` — replace string literals
- `code/src/analysis/receptive_field_mapping/rf_mapping_config.py` — replace string literals
- `code/src/analysis/receptive_field_mapping/touch_playback_data.py` — replace string literals
- `code/src/analysis/receptive_field_mapping/touch_population_data.py` — replace string literals

**Dependencies:** None

### Phase 2: Sub-package reorganization
**Started:** 2026-05-20
**Completed:** 2026-05-20
**Goal:** Move 33 flat files into 5 cohesive sub-packages, updating all
imports atomically with `__init__.py` re-exports.

- [x] Task 2.1 — Create empty sub-packages: `pipelines/`, `rendering/`,
      `data/`, `metrics/`, `surface/` with `__init__.py` files
- [x] Task 2.2 — Move `surface/` files: `rf_projection.py`,
      `rf_surface_utils.py`, `tangent_plane_alignment.py`,
      `forearm_slim_uv.py`, `_slim_helpers.py` (rename to `slim_helpers.py`),
      `_slim_qc_figures.py` (rename to `slim_qc_figures.py`)
- [x] Task 2.3 — Move `metrics/` files: `rf_metrics.py`,
      `rf_grid_cell_metrics.py`, `rf_baseline_deviation.py`,
      `rf_inflection_boundary.py`
- [x] Task 2.4 — Rename root files: `rf_mapping_config.py` to `config.py`,
      `rf_mapping_engine.py` to `engine.py`
- [x] Task 2.5 — Move `data/` files: `rf_data_loader.py`,
      `rf_extraction_io.py`, `touch_population_data.py`,
      `touch_playback_data.py`, `rf_explorer_data.py`, `rf_gallery_data.py`,
      `rf_population_heatmap.py`
- [x] Task 2.6 — Move `rendering/` files: `rf_cluster_visualizer.py`,
      `rf_2d_renderer.py`, `rf_population_map_renderer.py`,
      `rf_population_grid_metrics_renderer.py`,
      `rf_session_comparison_renderer.py`, `rf_simple_diagnostics.py`,
      `rf_visualizer.py`
- [x] Task 2.7 — Move pipeline files (except `rf_cluster_pipeline.py`):
      `rf_simple_pipeline.py`, `rf_single_touch_pipeline.py`,
      `rf_population_map_pipeline.py`, `rf_population_grid_pipeline.py`,
      `rf_population_grid_metrics_pipeline.py`
- [x] Task 2.8 — Update `receptive_field_mapping/__init__.py` re-exports to
      point to new sub-package paths
- [x] Task 2.9 — Update external consumers:
      `analysis_workflow_processing.py`, `preparation_viewer_data.py`,
      test files importing RF internals directly (backward-compat shims at
      flat-root paths handle all existing consumers without modification)

**Files Modified:**
- All 33 `.py` files in `receptive_field_mapping/` — moved to sub-packages
- `code/src/analysis/receptive_field_mapping/__init__.py` — updated re-exports
- `code/scripts/analysis_workflow_processing.py` — updated direct imports
- `code/src/analysis/touch_analytics/gui/preparation_viewer_data.py` — updated import path
- Test files under `code/tests/test_rf_*` — updated import paths

**Dependencies:** Phase 1

### Phase 3: Split monolith and documentation
**Started:** 2026-05-20
**Completed:** 2026-05-20
**Goal:** Split `rf_cluster_pipeline.py` into 4 focused modules and update
documentation.

- [x] Task 3.1 — Extract `pipelines/rf_cluster_metrics_pipeline.py`
      (lines 1073–1235: `run_cluster_rf_metrics_computation`)
- [x] Task 3.2 — Extract `pipelines/rf_cluster_visualization_pipeline.py`
      (lines 1237–1541: `run_cluster_rf_visualization`)
- [x] Task 3.3 — Extract `pipelines/rf_cluster_gui_launchers.py`
      (lines 736–1071, 1594–1662: `precompute_explorer_caches`, all
      `launch_*` functions)
- [x] Task 3.4 — Move remaining `rf_cluster_pipeline.py` to `pipelines/`
- [x] Task 3.5 — Update `receptive_field_mapping/__init__.py` for split files
- [x] Task 3.6 — Update `code/src/analysis/CLAUDE.md` to reflect new structure
- [x] Task 3.7 — Final grep audit: verify no orphaned imports, no duplicate
      constant definitions, no broken cross-references

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` — split into 4 files
- `code/src/analysis/receptive_field_mapping/pipelines/rf_cluster_metrics_pipeline.py` — NEW
- `code/src/analysis/receptive_field_mapping/pipelines/rf_cluster_visualization_pipeline.py` — NEW
- `code/src/analysis/receptive_field_mapping/pipelines/rf_cluster_gui_launchers.py` — NEW
- `code/src/analysis/receptive_field_mapping/__init__.py` — updated for split
- `code/src/analysis/CLAUDE.md` — updated

**Dependencies:** Phase 2

---

## Testing Plan

### Unit Tests
- [ ] `pytest code/tests/` — full suite passes after each phase
- [ ] Existing RF mapping tests continue to pass with no test code changes
      (other than import paths)

### Integration Tests
- [ ] `python -c "from analysis.receptive_field_mapping import *"` — public
      API intact after each phase
- [ ] `python -c "from analysis.pipeline import shared_constants"` — shared
      constants accessible
- [ ] `python -c "from analysis.pipeline import GESTURE_TYPES, TOUCH_ID_COLS, NERVE_SPIKE_COL"` —
      re-exports work

### Manual Verification
- [ ] `python code/scripts/analysis_workflow_processing.py --help` — loads
      without import errors
- [ ] `python code/scripts/analysis_workflow_viewers.py --help` — loads
      without import errors
- [ ] Spot-check one DAG task (e.g., `map_receptive_fields_simple`) end-to-end

### Edge Cases
- [ ] Backward-compat re-exports: verify `from analysis.touch_analytics.clustering_pipeline import GESTURE_TYPES` still works
- [ ] Backward-compat re-exports: verify `from analysis.touch_analytics.pipeline_shared import session_id_from_path` still works
- [ ] Grep audit: no files define `GESTURE_TYPES =` outside `shared_constants.py`
      (except backward-compat re-exports)
- [ ] Grep audit: no files define `_VALID_NEURON_MODES` anywhere

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` — reflect new sub-package structure,
      shared constants module, updated file paths
- [ ] Update `code/src/analysis/receptive_field_mapping/__init__.py` docstring

---

## Rollback Plan

1. **Before merging:** Each phase is a separate commit. Revert individual
   commits with `git revert` if a phase introduces regressions.

2. **Data considerations:** No data migrations. No file format changes.
   This is a pure code reorganization.

3. **Rollback procedure:**
   - `git revert <commit-hash>` for the offending phase
   - `__init__.py` re-exports ensure public API is stable regardless of
     internal organization

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Broken imports in scripts/tests | Medium | High | Grep for all direct module imports before each move; update atomically with `__init__.py` |
| Circular imports after reorganization | Low | High | Dependency graph is acyclic by construction (`surface` <- `metrics` <- `data` <- `rendering` <- `pipelines`) |
| Constant value mismatch after migration | Low | High | Verify each replaced definition has identical value; tests catch regressions |
| Merge conflicts with current branch | Low | Medium | Separate feature branch; merge to dev after current work lands |
| `rf_feature_space_explorer.py` `stroke_unknown` divergence | Known | Low | Explicitly handled: local extension of canonical tuple |
| Neuron-centroid invariant broken during split | Low | High | Split is structural only — no calling convention changes; `project_to_2d()` boundary stays in `surface/rf_projection.py` |
| Constant-definition drift over time | Medium | Medium | Add grep audit checklist to CLAUDE.md; backward-compat re-exports warn developers to use canonical location |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Shared constants | 2–3 hours | None |
| Phase 2: Sub-package reorganization | 3–4 hours | Phase 1 |
| Phase 3: Split monolith + docs | 1–2 hours | Phase 2 |
| **Total** | **6–9 hours** | |

---

## References

- Related Plan: `docs/development/plans/active/split-analysis-workflow-processing-viewers.md`
- Knowledge Base: `docs/development/knowledge-base/note-rf-cluster-coordinate-spaces.md`
- Knowledge Base: `docs/development/knowledge-base/note-3d-to-2d-surface-projection-algorithms.md`
- Knowledge Base: `docs/development/knowledge-base/note-kinect-depth-access-single-path.md`
