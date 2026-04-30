# Plan: Commit All Uncommitted Changes on feature/per-type-clustering-outputs

**Date:** 2026-04-30
**Author:** Basil Duvernoy
**Status:** Approved
**Base Branch:** `feature/per-type-clustering-outputs`
**Branch:** `feature/per-type-clustering-outputs` _(commits land on this branch directly)_

---

## Overview

28 modified tracked files and 13 new untracked files have accumulated on
`feature/per-type-clustering-outputs`, spanning 15 fully-implemented plans.
This plan defines the commit grouping and ordering so the work lands in a
clean, reviewable git history before the branch is merged.

## Problem Statement

All 15 active plans are implemented but none of their final changes are
committed. A single "everything" commit would obscure what changed and why.
Several files (`rf_cluster_pipeline.py`, `extraction_pipeline.py`,
`clustering_pipeline.py`) carry changes from more than one plan, making
precise per-plan isolation impractical without `git add -p` surgery.

## Goals

### In Scope
1. Seven commits, each covering a coherent functional unit
2. Commit messages follow the project's imperative-present-tense convention
3. All tests pass before the first commit; no regression between commits
4. Plan `.md` files are committed alongside the feature they document

### Out of Scope
- Squashing or rebasing prior commits on this branch
- Merging into `dev` or `main` (separate step)
- Closing out (moving to `completed/`) the active plan files — that is a
  follow-up task tracked in `docs/development/plans/pending/close-out-shipped-active-plans.md`

## Success Criteria

- [ ] `git status` shows a clean working tree after all 7 commits
- [ ] `pytest code/tests/` passes on the final commit
- [ ] Each commit message accurately describes its contents
- [ ] No file is left untracked or unstaged

---

## Technical Design

### Approach

Group files by primary ownership (which plan "introduced" each file or made
the largest change to it). Shared files are assigned to the earliest commit
in dependency order that can own them cleanly. Where a single file genuinely
spans two commits, use `git add -p` to stage only the relevant hunks.

### Commit Dependency Order

```
[1] gesture-type-preparation
        ↓
[2] per-type clustering
        ↓
[3] cartesian-binning renderer + outlier detection (independent)
[4] RF extraction/visualization split
        ↓
[5] RF gallery viewer + hull perimeters + GMM gallery
[6] forearm stage inspector (independent of RF commits)
        ↓
[7] console widget + pending plan docs (housekeeping)
```

---

## Implementation Plan

### Commit 1 — Gesture-type classification pipeline
**Message:** `feat(touch-analytics): complete gesture-type classification pipeline`

**Files to stage:**
```
code/src/analysis/touch_analytics/preparation/gesture_type.py
code/src/analysis/touch_analytics/preparation/__init__.py
code/src/analysis/touch_analytics/preparation/test_gesture_type.py
code/src/analysis/touch_analytics/preparation_pipeline.py
code/src/analysis/touch_analytics/pipeline_shared.py
code/src/analysis/touch_analytics/extraction_pipeline.py
code/src/analysis/touch_analytics/clustering/base.py
code/src/analysis/touch_analytics/clustering/type_stratified_clusterer.py
code/src/analysis/touch_analytics/clustering/test_type_stratified_clusterer.py
code/src/analysis/touch_analytics/reporting.py
code/src/analysis/touch_analytics/representation/feature_characterization/statistical.py
code/src/analysis/touch_analytics/representation/feature_characterization/touch_category.py
code/src/analysis/touch_analytics/representation/series_level/direction.py
code/src/analysis/touch_analytics/touch_analysis.py
code/scripts/diagnose_stroke_direction.py
docs/development/plans/active/gesture-type-preparation.md
```

**Plans covered:** gesture-type-preparation (primary), fast-skip-check-extraction-pipeline,
stage1-touch-data-preparation-interpolation (downstream fps/column changes)

**Note:** `extraction_pipeline.py` carries changes from three plans
(gesture-type consumer update, fast-skip pre-check, stage1 integration).
All are committed here since gesture-type is the most recent and dominant change.

---

### Commit 2 — Per-type clustering refinements
**Message:** `feat(touch-analytics): per-type clustering dialog and pipeline refinements`

**Files to stage:**
```
code/src/analysis/touch_analytics/clustering_pipeline.py
code/src/utils/gui/dag_launcher/cluster_group_dialog.py
```

**Plans covered:** per-type-clustering-outputs

**Note:** `clustering_pipeline.py` also contains gesture-type consumer updates,
but the `per_type_clustering` flag and helper refactor are the dominant changes.

---

### Commit 3 — Cartesian-binning renderer and outlier detection
**Message:** `feat(clustering): add cartesian-binning renderer and outlier detection`

**Files to stage:**
```
code/src/analysis/touch_analytics/clustering/cartesian_binning_clusterer.py
code/src/analysis/touch_analytics/clustering/test_cartesian_binning_clusterer.py
code/src/analysis/touch_analytics/clustering/cartesian_binning_renderer.py
code/src/analysis/touch_analytics/clustering/test_cartesian_binning_renderer.py
code/src/analysis/touch_analytics/clustering/outlier_detection.py
code/src/analysis/touch_analytics/clustering/test_outlier_detection.py
```

**Plans covered:** hand-velocity-scalars-and-cartesian-binning (renderer + outlier extras
beyond original plan scope)

---

### Commit 4 — RF extraction/visualization split
**Message:** `feat(rf-mapping): extraction/visualization split with artifact I/O`

**Files to stage:**
```
code/src/analysis/receptive_field_mapping/rf_extraction_io.py
code/src/analysis/receptive_field_mapping/rf_data_loader.py
code/src/analysis/receptive_field_mapping/rf_surface_utils.py
code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py
```

**Plans covered:** rf-cluster-extraction-visualization-split (primary),
rf-cluster-gallery-viewer (pipeline launch call),
rf-visualization-gui-responsiveness (gallery-launch-on-up-to-date fix),
rf-per-type-cluster-extraction (per-type branching in pipeline),
gesture-type-preparation (gesture_type column consumer)

**Note:** `rf_cluster_pipeline.py` is the most cross-cutting file in the
branch — it carries changes from 5+ plans. It is assigned here because the
extraction/visualization split is its largest structural change.
If a cleaner separation is desired, `git add -p rf_cluster_pipeline.py`
can split the gallery-launch hunk into commit 5.

---

### Commit 5 — RF cluster gallery viewer, hull perimeters, deferred processing, GMM gallery
**Message:** `feat(rf-mapping): cluster gallery viewer with hull perimeters, deferred processing, and GMM ranges`

**Files to stage:**
```
code/src/analysis/receptive_field_mapping/gui/rf_cluster_gallery_viewer.py
code/src/analysis/receptive_field_mapping/rf_gallery_data.py
docs/development/plans/active/rf-hull-perimeter-visualization.md
docs/development/plans/active/rf-per-type-cluster-extraction.md
docs/development/plans/active/gmm-clusterer.md
```

**Plans covered:** rf-cluster-gallery-viewer, rf-gallery-global-process,
rf-visualization-gui-responsiveness (thumbnail non-blocking), rf-cluster-visualization-improvements,
rf-hull-perimeter-visualization, gmm-gallery-feature-ranges, rf-per-type-cluster-extraction

---

### Commit 6 — Forearm stage inspector
**Message:** `feat(postprocessing): add forearm stage inspector for pointcloud diagnosis`

**Files to stage:**
```
code/src/postprocessing/gui/forearm_stage_inspector.py
code/src/postprocessing/gui/__init__.py
code/scripts/postprocess_visualization.py
configs/postprocess_visualization_dag.yaml
docs/development/plans/active/forearm-stage-inspector.md
```

**Plans covered:** forearm-stage-inspector

---

### Commit 7 — Console widget and pending plan docs
**Message:** `chore: console widget improvements and new pending plans`

**Files to stage:**
```
code/src/utils/gui/dag_launcher/console_widget.py
docs/development/plans/pending/close-out-shipped-active-plans.md
docs/development/plans/pending/rf-gallery-persistent-camera-settings.md
```

**Plans covered:** unattributed console improvement + housekeeping plan documents

---

## Testing Plan

### Before Commit 1
- [ ] Run `pytest code/tests/` — all tests must pass on the current working tree

### Between Commits (spot-check)
- [ ] After commit 1: run `pytest code/tests/` to confirm no regression
- [ ] After commit 3: run `pytest code/src/analysis/touch_analytics/clustering/` directly
- [ ] After commit 5: no automated tests for RF GUI — verify plan doc was included

### Final State
- [ ] `git status` shows empty working tree
- [ ] `git log --oneline -10` shows the 7 new commits with correct messages
- [ ] `pytest code/tests/` passes on the final commit

---

## Rollback Plan

Each commit is local-only until pushed. To undo any number of commits:

```bash
git reset --soft HEAD~N   # Unstage N commits, keep changes staged
git reset HEAD            # Unstage everything back to working tree
```

This is non-destructive — no work is lost, only the commit objects are removed.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `rf_cluster_pipeline.py` changes are tangled across commits 4 and 5 | Medium | Low | Accept the grouping; add a note in commit 4 message listing secondary plans covered |
| A plan's changes are split across two commits | Low | Low | Functional behaviour is identical; only history aesthetics affected |
| Tests fail mid-sequence | Low | Medium | Run `pytest` before commit 1; stop and fix before proceeding |
| Untracked file accidentally omitted | Low | Low | Verify `git status` shows clean tree after commit 7 |
