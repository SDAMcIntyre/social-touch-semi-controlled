# Plan: Rename IFF tuning tasks to response tuning + enrich output filenames

**Date:** 2026-06-09
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `dev`
**Branch:** `feature/rename-iff-to-response-tuning`

---

## Overview

The `stimulus_iff_tuning_curves` and `stimulus_iff_instruction_tuning` tasks
have "iff" in their names, but they handle multiple response metrics
(`iff_mean`, `iff_max`, `spike_count_mean`). Rename to use the generic term
"response". Additionally, embed the touch feature, gesture type, and response
metric into every output filename so files are self-describing outside their
directory tree. Also connect same-session dots in the instruction tuning
overlay plots for better visual traceability.

## Problem Statement

1. **Misleading names** — "IFF" (instantaneous firing frequency) is only one
   of three response metrics these tasks produce. The task keys, output
   directories, Python module names, and function names all use "iff",
   creating confusion when spike-count results sit in an "iff" folder.

2. **Opaque filenames** — overlay output files (`overlay_tuning_by_type.png`,
   `overlay_instruction_tuning_by_session.png`) carry no context about which
   feature, gesture, or metric they represent. Moving or sharing them loses
   all provenance.

3. **Poor visual traceability** — the instruction tuning overlay plots show
   jittered dots per session but don't connect them, making it hard to follow
   a single session's trend across instruction levels.

## Goals

### In Scope
1. Rename task keys, output dirs, constants, functions, flows, sentinels, and
   Python source files from "iff" to "response"
2. Embed `{feature}_{gesture}_{metric}` in all per-session and overlay
   output filenames (both tuning curves and instruction tuning)
3. Connect same-session dots with a line in
   `render_overlay_instruction_tuning()`
4. Register old directory names in `RENAME_MAPPING` for migration

### Out of Scope
- Renaming internal option keys (`iff_metric`, `response_metric`, etc.) —
  these describe data content correctly
- Renaming the `RESPONSE_METRICS` dict or its keys — already generic
- Modifying the directory tree structure (feature/gesture/binning/metric
  nesting stays the same)
- Renaming completed plan documents in `docs/development/plans/completed/`

## Success Criteria

- [ ] All `stimulus_iff_tuning` / `stimulus_iff_instruction` references
      replaced (verified by grep, excluding completed plans)
- [ ] All four Python source files renamed via `git mv`
- [ ] Output filenames contain feature, gesture, and metric dimensions
- [ ] Instruction tuning overlay dots connected per session
- [ ] `pytest` passes with no import errors
- [ ] `python -c "from analysis.receptive_field_mapping import run_response_tuning, run_response_instruction_tuning"` succeeds
- [ ] DAG config task keys match new names

---

## Technical Design

### Approach

Atomic rename: change every reference in one pass (constants, files, imports,
functions, flows, config, filenames, renderer) to avoid any broken
intermediate state. Register old output directory names in `RENAME_MAPPING`
so the existing migration script can rename on-disk folders.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Atomic rename (all at once) | No broken state, clean history | Larger single diff | **Chosen** |
| Incremental (rename files first, then functions, then filenames) | Smaller diffs | Broken imports between steps | Rejected |
| Backward-compat shims for renamed files | Old imports keep working | Unnecessary — files are internal modules, not public API | Rejected |

### Rename Mapping

| Current | New |
|---------|-----|
| Task `stimulus_iff_tuning_curves` | `stimulus_response_tuning` |
| Task `stimulus_iff_instruction_tuning` | `stimulus_response_instruction_tuning` |
| Dir `stimulus_iff_tuning_curves/` | `stimulus_response_tuning/` |
| Dir `stimulus_iff_instruction_tuning/` | `stimulus_response_instruction_tuning/` |
| Constant `STIMULUS_IFF_TUNING_CURVES` | `STIMULUS_RESPONSE_TUNING` |
| Constant `STIMULUS_IFF_INSTRUCTION_TUNING` | `STIMULUS_RESPONSE_INSTRUCTION_TUNING` |
| Function `run_iff_tuning_curves()` | `run_response_tuning()` |
| Function `run_iff_instruction_tuning()` | `run_response_instruction_tuning()` |
| Flow `stimulus_iff_tuning_curves_flow()` | `stimulus_response_tuning_flow()` |
| Flow `stimulus_iff_instruction_tuning_flow()` | `stimulus_response_instruction_tuning_flow()` |
| Sentinel `iff_tuning_sentinel.json` | `response_tuning_sentinel.json` |
| Sentinel `iff_instruction_tuning_sentinel.json` | `response_instruction_tuning_sentinel.json` |
| File `rf_iff_tuning_pipeline.py` | `rf_response_tuning_pipeline.py` |
| File `rf_iff_instruction_tuning_pipeline.py` | `rf_response_instruction_tuning_pipeline.py` |
| File `rf_iff_tuning_renderer.py` | `rf_response_tuning_renderer.py` |
| File `rf_iff_instruction_tuning_renderer.py` | `rf_response_instruction_tuning_renderer.py` |

### Filename Enrichment

**Tuning curves** — variables `feature`, `gesture_subset`, `metric_subdir`
are already in scope at every filename construction point:

```
{session_id}_tuning.png                →  {session_id}_{feature}_{gesture}_{metric}_tuning.png
overlay_tuning_by_type.png             →  overlay_{feature}_{gesture}_{metric}_tuning_by_type.png
overlay_tuning_by_session.png          →  overlay_{feature}_{gesture}_{metric}_tuning_by_session.png
overlay_tuning.csv                     →  overlay_{feature}_{gesture}_{metric}_tuning.csv
```

**Instruction tuning** — `category_col` replaces `feature`:

```
{session_id}_instruction_tuning.png    →  {session_id}_{category}_{gesture}_{metric}_instruction_tuning.png
overlay_instruction_tuning_by_type.png →  overlay_{category}_{gesture}_{metric}_instruction_tuning_by_type.png
overlay_instruction_tuning_by_session.png → overlay_{category}_{gesture}_{metric}_instruction_tuning_by_session.png
overlay_instruction_tuning.csv         →  overlay_{category}_{gesture}_{metric}_instruction_tuning.csv
```

Per-session CSVs are derived via `.with_suffix(".csv")` and inherit
the new pattern automatically.

---

## Implementation Plan

### Phase 1: Infrastructure — output directory constants
**Goal:** Rename canonical constants and register old names for migration.

- [x] 1.1 — Rename `STIMULUS_IFF_TUNING_CURVES` to `STIMULUS_RESPONSE_TUNING` (value: `"stimulus_response_tuning"`)
- [x] 1.2 — Rename `STIMULUS_IFF_INSTRUCTION_TUNING` to `STIMULUS_RESPONSE_INSTRUCTION_TUNING` (value: `"stimulus_response_instruction_tuning"`)
- [x] 1.3 — Add old names to `RENAME_MAPPING`

**Files Modified:**
- `code/src/analysis/pipeline/output_dirs.py` — rename constants, add migration entries

**Dependencies:** None

### Phase 2: Rename Python source files
**Goal:** Rename the four modules via `git mv` to preserve history.

- [x] 2.1 — `git mv pipelines/rf_iff_tuning_pipeline.py → rf_response_tuning_pipeline.py`
- [x] 2.2 — `git mv pipelines/rf_iff_instruction_tuning_pipeline.py → rf_response_instruction_tuning_pipeline.py`
- [x] 2.3 — `git mv rendering/rf_iff_tuning_renderer.py → rf_response_tuning_renderer.py`
- [x] 2.4 — `git mv rendering/rf_iff_instruction_tuning_renderer.py → rf_response_instruction_tuning_renderer.py`

**Dependencies:** None (independent of Phase 1)

### Phase 3: Fix all import paths
**Goal:** Update every import statement referencing old module paths or old constant names.

- [x] 3.1 — `receptive_field_mapping/__init__.py` — import paths + `__all__` entries
- [x] 3.2 — `rf_response_tuning_pipeline.py` — `output_dirs` constant import, renderer import
- [x] 3.3 — `rf_response_instruction_tuning_pipeline.py` — `output_dirs` constant import, tuning pipeline import, renderer import
- [x] 3.4 — `rf_response_instruction_tuning_renderer.py` — tuning renderer import
- [x] 3.5 — `code/scripts/analysis_workflow_processing.py` — function imports (lines 44-45), `_resolve_response_metric` import (line 49), constant imports (lines 80-81)

**Dependencies:** Phase 1 + Phase 2

### Phase 4: Rename functions, flows, log messages
**Goal:** Rename entry-point functions, Prefect flows, sentinels, and all log/print messages.

- [x] 4.1 — `rf_response_tuning_pipeline.py`: rename `run_iff_tuning_curves` → `run_response_tuning`, update error prefixes (~4), log/print messages (~15), sentinel filename
- [x] 4.2 — `rf_response_instruction_tuning_pipeline.py`: rename `run_iff_instruction_tuning` → `run_response_instruction_tuning`, update error prefixes (~4), log/print messages (~12), sentinel filename
- [x] 4.3 — `analysis_workflow_processing.py`: rename flow functions, `@flow(name=...)` decorators, docstrings, print messages, dispatch table entries (task name strings + `get_task_options()` keys)

**Dependencies:** Phase 3

### Phase 5: DAG config rename
**Goal:** Rename YAML task keys.

- [x] 5.1 — Rename `stimulus_iff_tuning_curves:` → `stimulus_response_tuning:`
- [x] 5.2 — Rename `stimulus_iff_instruction_tuning:` → `stimulus_response_instruction_tuning:`
- [x] 5.3 — Update line 14 comment referencing both task names

**Files Modified:**
- `configs/analyse_workflow_processing_dag.yaml`

**Dependencies:** Phase 4

### Phase 6: Filename enrichment — tuning curves
**Goal:** Embed feature, gesture, and metric into every output filename.

- [x] 6.1 — Per-session PNG: `f"{session_id}_{feature}_{gesture_subset}_{metric_subdir}_tuning.png"` (both sliding_window and raw_dots paths)
- [x] 6.2 — Overlay by-type: `f"overlay_{feature}_{gesture_subset}_{metric_subdir}_tuning_by_type.png"`
- [x] 6.3 — Overlay by-session: `f"overlay_{feature}_{gesture_subset}_{metric_subdir}_tuning_by_session.png"`
- [x] 6.4 — Overlay CSV: `f"overlay_{feature}_{gesture_subset}_{metric_subdir}_tuning.csv"`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_response_tuning_pipeline.py`

**Dependencies:** Phase 4

### Phase 7: Filename enrichment — instruction tuning
**Goal:** Same enrichment for instruction tuning, using `category_col` as the feature dimension.

- [x] 7.1 — Per-session PNG: `f"{session_id}_{category_col}_{gesture_subset}_{metric_subdir}_instruction_tuning.png"`
- [x] 7.2 — Overlay by-type: `f"overlay_{category_col}_{gesture_subset}_{metric_subdir}_instruction_tuning_by_type.png"`
- [x] 7.3 — Overlay by-session: `f"overlay_{category_col}_{gesture_subset}_{metric_subdir}_instruction_tuning_by_session.png"`
- [x] 7.4 — Overlay CSV: `f"overlay_{category_col}_{gesture_subset}_{metric_subdir}_instruction_tuning.csv"`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_response_instruction_tuning_pipeline.py`

**Dependencies:** Phase 4

### Phase 8: Connect dots per session in instruction tuning overlay
**Goal:** Add connecting lines between same-session dots for visual traceability.

- [x] 8.1 — In `render_overlay_instruction_tuning()`, add `ax.plot()` after the scatter call, inside the per-session loop, connecting `x_jittered[valid_mask]` / `mean_iff[valid_mask]` with `alpha=0.4`, `linewidth=1.0`, `zorder=2`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rendering/rf_response_instruction_tuning_renderer.py`

**Dependencies:** Phase 2 (file must be renamed first)

### Phase 9: Update docstrings and documentation
**Goal:** Update output layout diagrams and documentation references.

- [x] 9.1 — Module docstrings at top of both pipeline files
- [x] 9.2 — `code/src/analysis/CLAUDE.md` — task names in `stimulus_sensitivity` listing
- [x] 9.3 — `configs/analyse_workflow_processing_dag.yaml` line 14 comment

**Dependencies:** Phase 5

---

## Testing Plan

### Unit Tests
- [ ] `pytest` — full test suite passes with no import errors

### Integration Tests
- [ ] `python -c "from analysis.receptive_field_mapping import run_response_tuning, run_response_instruction_tuning"` succeeds

### Manual Verification
- [ ] Grep for remaining `stimulus_iff_tuning` / `stimulus_iff_instruction` references (excluding `docs/development/plans/completed/`)
- [ ] Grep for remaining `rf_iff_tuning` / `rf_iff_instruction` import paths
- [ ] Run pipeline GUI — both tasks appear under `stimulus_sensitivity` and execute

### Edge Cases
- [ ] Longest realistic filename (`ST13-01_hand_velocity_amplitude_mean_stroke_proximal_spike_count_mean_tuning.png`) is ~85 chars — well within filesystem limits

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` with new task names
- [ ] Update module docstrings (output layout diagrams)

---

## Rollback Plan

1. `git revert <commit>` — single atomic commit makes clean revert
2. Old sentinel files won't match new names, so pipeline will re-process
   on next run — no data corruption risk
3. On-disk output directories: if already migrated via `RENAME_MAPPING`,
   manually rename back or re-run migration with reversed mapping

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Missed import reference causes runtime crash | Low | Med | Grep verification + pytest |
| Existing output dirs on disk not renamed | Med | Low | `RENAME_MAPPING` handles migration; old sentinels trigger re-processing |
| Stale `.pyc` files confuse Python | Low | Low | Python ignores stale pyc; clean with `find -name __pycache__` if needed |

---

## References

- Approved approach in `.claude/plans/analyse-the-generation-of-spicy-pebble.md`
- Output dir migration: `code/src/analysis/pipeline/output_dirs.py::RENAME_MAPPING`

## Modified Files

<!-- auto-generated by /plan-implement — do not edit manually -->
- code/scripts/analysis_workflow_processing.py
- code/src/analysis/CLAUDE.md
- code/src/analysis/pipeline/output_dirs.py
- code/src/analysis/receptive_field_mapping/__init__.py
- code/src/analysis/receptive_field_mapping/pipelines/rf_response_instruction_tuning_pipeline.py
- code/src/analysis/receptive_field_mapping/pipelines/rf_response_tuning_pipeline.py
- code/src/analysis/receptive_field_mapping/rendering/rf_response_instruction_tuning_renderer.py
- code/src/analysis/receptive_field_mapping/rendering/rf_response_tuning_renderer.py
- configs/analyse_workflow_processing_dag.yaml
- docs/development/plans/active/rename-iff-to-response-tuning.md
