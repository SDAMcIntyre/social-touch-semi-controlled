# Plan: Close Out All Shipped Active Plans

**Date:** 2026-04-29
**Author:** Basil Duvernoy
**Status:** Pending
**Base Branch:** `feature/hand-velocity-scalars-and-cartesian-binning`
**Branch:** `feature/close-out-shipped-active-plans`

---

## Overview

**What:** Commit all uncommitted working-tree changes that belong to shipped
features, then move all 11 plans in `docs/development/plans/active/` to
`completed/`.

**Why:** All 11 plans are code-complete on the dev branch — their presence
in `active/` is misleading and obscures genuinely in-progress work. Four
plans also have uncommitted code on the working tree that must be committed
before the plans can be closed.

**How:** Group the uncommitted changes by feature, create one commit per
group, then batch-move all plan files.

## Problem Statement

A full audit of `docs/development/plans/active/` (11 plans) against the
`dev` branch shows that every plan's implementation phases are 100% checked
off and the corresponding code exists on `dev`. The plans were never moved
to `completed/` after merging. Additionally, 4 plans have code written but
not yet committed:

| Uncommitted change | Belongs to plan |
|--------------------|-----------------|
| `code/src/postprocessing/gui/forearm_stage_inspector.py` (new) | forearm-stage-inspector |
| `code/src/postprocessing/gui/__init__.py` (modified) | forearm-stage-inspector |
| `code/scripts/postprocess_visualization.py` (modified) | forearm-stage-inspector |
| `configs/postprocess_visualization_dag.yaml` (modified) | forearm-stage-inspector |
| `code/src/analysis/receptive_field_mapping/gui/rf_cluster_gallery_viewer.py` (modified) | rf-hull-perimeter-visualization, rf-gallery-global-process |
| `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` (modified) | gmm-gallery-feature-ranges |
| `code/src/analysis/receptive_field_mapping/rf_gallery_data.py` (modified) | rf-cluster-gallery-viewer |
| `code/src/analysis/receptive_field_mapping/rf_extraction_io.py` (modified) | rf-cluster-extraction-visualization-split |
| `code/src/analysis/receptive_field_mapping/rf_surface_utils.py` (modified) | rf-cluster-visualization-improvements |
| `code/src/analysis/touch_analytics/clustering/__init__.py` (modified) | gmm-clusterer |

## Goals

### In Scope
1. Commit the uncommitted code belonging to the 4 open plans (forearm-stage-inspector, rf-hull-perimeter-visualization, gmm-gallery-feature-ranges, rf-gallery-global-process)
2. Move all 11 active plan files to `docs/development/plans/completed/`
3. Mark each plan as `Status: Completed` and add today's completion date

### Out of Scope
- Writing any unit tests that remain unchecked in the plans
- Running manual verification steps listed in testing plans
- Documentation updates referenced in the plans (CLAUDE.md, knowledge-base notes)
- Any new code changes

## Success Criteria

- [ ] Working tree is clean after commits (no staged or unstaged changes to existing feature files)
- [ ] All 11 plan files are in `docs/development/plans/completed/` and no longer in `active/`
- [ ] Each moved plan has `Status: Completed` and a `Completed:` date in its header

---

## Technical Design

### Approach

Group uncommitted changes by the feature they belong to and commit each
group independently with a descriptive commit message. Then move all plan
files in a single follow-up commit.

### Commit grouping

| Commit | Files | Plan closed |
|--------|-------|-------------|
| `feat(forearm-stage-inspector)` | `forearm_stage_inspector.py`, `postprocessing/gui/__init__.py`, `postprocess_visualization.py`, `postprocess_visualization_dag.yaml` | forearm-stage-inspector |
| `feat(rf-hull-perimeter-visualization)` | `rf_cluster_gallery_viewer.py` (alpha-shape / display-mode changes) | rf-hull-perimeter-visualization, rf-gallery-global-process (same file) |
| `feat(gmm-gallery-feature-ranges)` | `rf_cluster_pipeline.py` (gmm branches) | gmm-gallery-feature-ranges |
| Remaining modified files | `rf_gallery_data.py`, `rf_extraction_io.py`, `rf_surface_utils.py`, `clustering/__init__.py` | Other plans |

> Note: `rf_cluster_gallery_viewer.py` contains changes from both
> `rf-hull-perimeter-visualization` AND `rf-gallery-global-process`. Both
> plans go into a single commit for that file.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Single mega-commit for all uncommitted changes | Simple | Hard to bisect; mixes concerns | Rejected |
| One commit per plan | Clean history | Some files belong to 2 plans (gallery viewer) | Rejected — impractical |
| Group by file/feature area | Readable history; keeps related changes together | Slightly larger commits | **Chosen** |

### Architecture Changes

None — this plan involves no code changes.

---

## Implementation Plan

### Phase 1: Commit uncommitted feature changes
**Goal:** Clean working tree by committing each group of changes.

- [ ] Task 1.1 — `git diff --stat` to confirm which files are modified
- [ ] Task 1.2 — Commit forearm-stage-inspector changes (new file + 3 modified files)
- [ ] Task 1.3 — Commit rf-hull-perimeter-visualization + rf-gallery-global-process changes (`rf_cluster_gallery_viewer.py`)
- [ ] Task 1.4 — Commit gmm-gallery-feature-ranges changes (`rf_cluster_pipeline.py`)
- [ ] Task 1.5 — Commit remaining modified files (`rf_gallery_data.py`, `rf_extraction_io.py`, `rf_surface_utils.py`, `clustering/__init__.py`) — attribute to the plans they belong to

**Files Modified:** Working tree only — no source changes
**Dependencies:** None

### Phase 2: Move plan files to completed/ and update headers
**Goal:** All 11 plan files reflect shipped status.

Plans to move (all from `docs/development/plans/active/` → `docs/development/plans/completed/`):

1. `stage1-touch-data-preparation-interpolation.md`
2. `fast-skip-check-extraction-pipeline.md`
3. `rf-cluster-visualization-improvements.md`
4. `rf-cluster-extraction-visualization-split.md`
5. `rf-cluster-gallery-viewer.md`
6. `rf-visualization-gui-responsiveness.md`
7. `gmm-clusterer.md`
8. `forearm-stage-inspector.md`
9. `rf-hull-perimeter-visualization.md`
10. `gmm-gallery-feature-ranges.md`
11. `rf-gallery-global-process.md`

- [ ] Task 2.1 — For each plan file, update header: set `Status: Completed` and add `Completed: 2026-04-29`
- [ ] Task 2.2 — Move all 11 files with `git mv` (preserves history)
- [ ] Task 2.3 — Commit the plan-file moves in a single commit

**Files Modified:** 11 plan markdown files (moved + header edit)
**Dependencies:** Phase 1

---

## Testing Plan

### Manual Verification
- [ ] `git status` shows clean working tree after Phase 1
- [ ] `ls docs/development/plans/active/` returns empty (or only this plan's file if still pending)
- [ ] `ls docs/development/plans/completed/` contains all 11 files
- [ ] Each moved plan has `Status: Completed` in its header

---

## Rollback Plan

1. **Rollback Phase 2 (plan moves):** `git revert` the move commit — git mv is tracked, so reverting restores files to `active/`
2. **Rollback Phase 1 (feature commits):** Revert individual commits. No data or schema changes involved.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| A modified file has changes from two different plans, making atomic commits impossible | Low | Low | Commit together under the plan with the most changes; note the other plan in the message |
| Plan file currently open in editor during `git mv` | Low | Low | Close editors before running Phase 2 |

---

## References

- Analysis: `C:\Users\basil\.claude\plans\analyse-all-active-plans-eager-pizza.md`
