# Plan: Block Folder Rename and Aggregation Move

**Date:** 2026-03-13
**Author:** Basil Duvernoy
**Status:** Draft
**Branch:** `feature/block-folder-rename-and-aggregation-move`

---

## Overview

The merging pipeline outputs block-level CSVs to a subfolder called `sessions/`, which is misleading since it contains per-block data, not session-level data. Additionally, the aggregation step (concatenating all blocks into one session CSV) runs at the end of merging, before any postprocessing — so the aggregated file contains spatially unprocessed data. This plan renames `sessions/` to `blocks_merged/` (and all downstream `sessions_*` to `blocks_*`), and moves aggregation to be the last stage of the postprocessing pipeline.

## Problem Statement

1. **Misleading folder name:** The `sessions/` subfolder contains individual block-level merged CSVs, not session-level data. Every downstream folder (`sessions_registered/`, `sessions_pca_calibrated/`, `sessions_contact_projected/`) inherits this confusing naming.

2. **Premature aggregation:** The aggregation step runs after merging but before postprocessing (ICP registration, PCA calibration, contact projection). The resulting `*_aggregated_session.csv` contains raw merged data without spatial transforms applied. Any analysis using the aggregated file works on unprocessed coordinates.

## Goals

### In Scope
1. Rename `sessions/` to `blocks_merged/` and `sessions_filtered/` to `blocks_filtered/` in the merging pipeline
2. Rename all `sessions_*` postprocessing output folders to `blocks_*` for consistency
3. Remove aggregation logic from the merging pipeline entirely
4. Add aggregation as the final stage of the postprocessing pipeline, operating on fully processed block CSVs
5. Update all downstream consumers (visualization, analysis) to use the new names

### Out of Scope
- Running postprocessing on filtered blocks separately (postprocessing operates only on the merged blocks path)
- Backward-compatible fallback logic for old folder names (users re-run pipelines)
- Migration scripts for existing data on disk

## Success Criteria

- [ ] Merging pipeline outputs to `blocks_merged/` and `blocks_filtered/` — no `sessions/` or `sessions_filtered/` created
- [ ] Merging pipeline does NOT produce any aggregated session files
- [ ] Postprocessing pipeline outputs to `blocks_registered/`, `blocks_pca_calibrated/`, `blocks_contact_projected/`
- [ ] Postprocessing pipeline produces `*_aggregated_session.csv` as its final step, from fully postprocessed blocks
- [ ] Visualization pipeline finds block CSVs in `blocks_merged/`
- [ ] Analysis pipeline finds the new aggregated file

---

## Technical Design

### Approach

Mechanical rename of all hardcoded `"sessions"` / `"sessions_*"` folder strings to `"blocks_merged"` / `"blocks_*"` across merging, postprocessing, and visualization scripts. Then extract the `aggregate_blocks` flow from the merging pipeline and add it as a new DAG-controlled stage in postprocessing. The underlying `aggregate_session_blocks()` function is pipeline-agnostic and requires no changes — only the orchestration layer changes.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Rename to `blocks_merged/` + move aggregation | Clear naming, aggregated data is fully processed | Requires re-running pipelines on existing data | **Chosen** |
| Rename to `blocks/` (shorter) | Simpler | Less explicit about what "blocks" contains | Rejected |
| Keep `sessions/` and only move aggregation | Fewer file changes | Naming stays confusing | Rejected |
| Add backward-compatible fallback for old names | Existing data still works | Adds complexity for a research pipeline | Rejected |

### Architecture Changes

**Folder structure after changes:**
```
{session_merged_output_dir}/
├── blocks_merged/                     # Merging output (block-level)
├── blocks_filtered/                   # Merging output (filtered blocks)
├── blocks_registered/                 # Postprocessing stage 1
├── blocks_pca_calibrated/             # Postprocessing stage 2
│   └── pca-xyz_transformation-matrices.json
├── forearm_pca_calibrated/            # Postprocessing stage 3 (unchanged name — session-level)
├── blocks_contact_projected/          # Postprocessing stage 4
│   └── projection_stats.csv
└── {session_id}_semicontrolled_aggregated_session.csv  # Postprocessing stage 5 (NEW location)
```

**Key reuse:** `aggregate_session_blocks()` in `code/scripts/_4_merging/aggregate_blocks_session.py` is imported directly by postprocessing — no code duplication.

---

## Implementation Plan

### Phase 1: Folder Renames
**Goal:** Replace all `sessions` / `sessions_*` folder references with `blocks_merged` / `blocks_*`

- [ ] Task 1.1 — Rename in merging pipeline orchestration
- [ ] Task 1.2 — Rename in postprocessing pipeline orchestration
- [ ] Task 1.3 — Rename in visualization pipeline

**Files Modified:**
- `code/scripts/merging_pipeline_neuron_to_kinect_auto.py` — `"sessions"` → `"blocks_merged"` (line 65, 104), `"sessions_filtered"` → `"blocks_filtered"` (lines 219, 338, 347)
- `code/scripts/postprocess_workflow_kinect_auto.py` — `"sessions"` → `"blocks_merged"` (line 127), `"sessions_registered"` → `"blocks_registered"` (line 154), `"sessions_pca_calibrated"` → `"blocks_pca_calibrated"` (line 164), `"sessions_contact_projected"` → `"blocks_contact_projected"` (lines 186-187)
- `code/scripts/merging_pipeline_neuron_to_kinect_visualisation.py` — `"sessions"` → `"blocks_merged"` (line 158)

**Dependencies:** None

### Phase 2: Remove Aggregation from Merging
**Goal:** The merging pipeline no longer produces aggregated session files

- [ ] Task 2.1 — Remove `aggregate_session_blocks` import from merging orchestration
- [ ] Task 2.2 — Remove `aggregate_blocks` flow definition (lines 98-132)
- [ ] Task 2.3 — Remove aggregation invocation in `run_batch_processing` (lines 326-347)
- [ ] Task 2.4 — Remove `aggregate_session_blocks` export from `_4_merging/__init__.py`

**Files Modified:**
- `code/scripts/merging_pipeline_neuron_to_kinect_auto.py` — Remove import (line 21), flow (lines 98-132), invocation (lines 326-347)
- `code/scripts/_4_merging/__init__.py` — Remove `aggregate_session_blocks` from exports

**Dependencies:** Phase 1

### Phase 3: Add Aggregation to Postprocessing
**Goal:** Aggregation becomes the final DAG-controlled stage of postprocessing

- [ ] Task 3.1 — Add `aggregate_session_blocks` import (from `_4_merging.aggregate_blocks_session`)
- [ ] Task 3.2 — Add `aggregate_session_blocks_flow` Prefect flow wrapper
- [ ] Task 3.3 — Add 5th pipeline stage to `pipeline_stages` list, reading from `context["projected_files"]`
- [ ] Task 3.4 — Add `aggregate_session` task entry to postprocessing DAG config

**Files Modified:**
- `code/scripts/postprocess_workflow_kinect_auto.py` — Add import, flow wrapper, pipeline stage entry
- `configs/postprocess_workflow_kinect_auto_dag.yaml` — Add `aggregate_session` task with `depends_on: [project_contacts_onto_forearm]`

**Dependencies:** Phase 2

### Phase 4: Update Downstream Consumers
**Goal:** Analysis pipeline finds the new aggregated file

- [ ] Task 4.1 — Update glob pattern in analysis workflow (two locations)

**Files Modified:**
- `code/scripts/analysis_workflow.py` — Line 252 and line 457: `"*_semicontrolled_aggregated_session_filtered.csv"` → `"*_semicontrolled_aggregated_session.csv"`

**Dependencies:** Phase 3

---

## Testing Plan

### Manual Verification
- [ ] Run merging pipeline on a single session → confirm `blocks_merged/` and `blocks_filtered/` created, no `sessions/` folder, no aggregated CSV
- [ ] Run postprocessing pipeline on same session → confirm `blocks_registered/`, `blocks_pca_calibrated/`, `forearm_pca_calibrated/`, `blocks_contact_projected/` created, and `*_aggregated_session.csv` at session root
- [ ] Run visualization pipeline → confirm it loads block CSVs from `blocks_merged/`
- [ ] Run analysis pipeline → confirm it finds `*_aggregated_session.csv`

### Edge Cases
- [ ] Session with no filtered blocks — merging should not create `blocks_filtered/`, postprocessing aggregation should still work on the main path
- [ ] Session with only one block — aggregation should produce a valid single-block aggregated file

---

## Documentation Plan

- [ ] Update CLAUDE.md if any architecture documentation references the old folder names
- [ ] No new user guides needed — this is an internal pipeline change

---

## Rollback Plan

1. Revert the feature branch commits
2. Re-run pipelines to regenerate data under old folder names
3. No data migration needed — pipeline regenerates all outputs from upstream sources

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Existing data on disk uses old folder names | High | Low | Users re-run pipelines; old folders can be manually deleted |
| Analysis scripts run before postprocessing re-run | Med | Med | Glob pattern update ensures they look for the correct new filename; no match = clear error |
| Other undiscovered consumers of `sessions/` path | Low | Med | Grep for `"sessions"` across codebase during implementation to catch any missed references |

---

## References

- Scratch plan: `.claude/plans/dazzling-painting-narwhal.md`
- Aggregation function: `code/scripts/_4_merging/aggregate_blocks_session.py`
- Merging orchestration: `code/scripts/merging_pipeline_neuron_to_kinect_auto.py`
- Postprocessing orchestration: `code/scripts/postprocess_workflow_kinect_auto.py`
