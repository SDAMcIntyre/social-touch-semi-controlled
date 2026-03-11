# Plan: Commit Multi-Snapshot Forearm Registration Branch

**Date:** 2026-03-06
**Author:** Basil Duvernoy
**Status:** Draft
**Branch:** `feature/multi-snapshot-forearm-registration`

---

## Overview

Organize and commit all pending changes on the `feature/multi-snapshot-forearm-registration`
branch into atomic, dependency-ordered commits. The branch contains work from six plans
(three active, two completed, one draft) spanning registration, curation, DAG integration,
rotatable ROI, and launcher UI improvements. A structured commit sequence ensures each
commit is independently buildable and traceable to its originating plan.

## Problem Statement

The branch has accumulated 20+ modified/new files across six different feature plans. All
changes are currently uncommitted (mix of staged and unstaged). Committing everything as a
single commit would obscure the distinct concerns, make code review difficult, and
complicate future bisection or revert operations.

## Goals

### In Scope
1. Define a sequence of atomic commits, each covering a single logical concern
2. Order commits so each is independently buildable (respects inter-file dependencies)
3. Map every file to exactly one commit (no file appears in two commits)
4. Follow Conventional Commits format per project commit procedure
5. Ensure no commit introduces import errors or broken references

### Out of Scope
- Running tests or manual verification of the features themselves
- Modifying any source code (commits capture the current state as-is)
- Pushing to remote or creating a pull request
- Moving plan documents between `active/`/`completed/` directories

## Success Criteria

- [x] Every modified/new file is assigned to exactly one commit
- [x] Commit order respects import and dependency chains
- [x] Each commit message follows `<type>(<scope>): <subject>` format
- [x] No commit introduces a broken import (files that export symbols are committed
      before or alongside files that import them)

---

## Technical Design

### Approach

Group files by their originating plan, then order commits so that foundational modules
(data model changes, new subpackages) are committed before the pipeline scripts that
import them. Where two plans touch the same file, assign the file to the commit where
its primary change belongs and note the overlap.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| One commit per plan (6 commits) | Clean 1:1 plan mapping | Some plans share files (`__init__.py` exports); hard to split cleanly | Rejected |
| One commit per file | Maximum granularity | Too many commits (20+); no logical grouping | Rejected |
| Group by dependency layer (9 commits) | Each commit is buildable; logical grouping; traceable to plans | Some commits span two plans | **Chosen** |
| Single monolithic commit | Simple | Obscures six distinct concerns; hard to review/revert | Rejected |

### Architecture Changes

No architecture changes. This plan organizes existing changes into commits.

---

## Implementation Plan

### Phase 1: Rotatable ROI foundation
**Goal:** Commit the new ROI widget and its integration into the forearm data model.

**Commit:**
```
feat(forearm-extraction): add rotatable ROI selection with angle persistence
```

Cross-cutting enhancement with no dedicated plan. Introduces `FrameROIRotatable`,
adds `angle_deg` to the data model and serialization layer, switches
`define_extraction_parameters` from `FrameROISquare` to `FrameROIRotatable` with
centre-based coordinates and AABB computation from rotated corners. Also adds
skip-if-saved logic to avoid re-opening the ROI GUI when parameters already exist.

- [x] Identify all files

**Files:**
- `code/src/preprocessing/common/gui/frame_roi_rotatable.py` -- **new** FrameROIRotatable widget
- `code/src/preprocessing/common/__init__.py` -- add FrameROIRotatable export
- `code/src/preprocessing/forearm_extraction/models/forearm_parameters.py` -- add `angle_deg` to RegionOfInterest
- `code/src/preprocessing/forearm_extraction/data_access/forearm_frame_parameters_filehandler.py` -- deserialize `angle_deg` (default 0.0)
- `code/scripts/_3_preprocessing/_3_forearm_extraction/define_extraction_parameters.py` -- switch to FrameROIRotatable, AABB from rotated corners, skip-if-saved

**Dependencies:** None

---

### Phase 2: Per-step skip guards in sub-functions
**Goal:** Commit the `force_processing` + `should_process_task` additions to each forearm sub-function.

**Commit:**
```
feat(forearm-extraction): add force-processing skip guards to sub-functions
```

Phase 1 of **forearm-per-step-force-processing** plan. Each sub-function gains a
`force_processing` kwarg and a `should_process_task()` guard that skips execution
when output files already exist. No caller changes yet -- callers are updated in
Phase 5.

- [x] Identify all files

**Files:**
- `code/scripts/_3_preprocessing/_3_forearm_extraction/extract_participant_forearm.py` -- add force_processing + should_process_task
- `code/scripts/_3_preprocessing/_3_forearm_extraction/clean_forearm_pointcloud.py` -- add force_processing + should_process_task, remove manual FileNotFoundError
- `code/scripts/_3_preprocessing/_3_forearm_extraction/define_normals.py` -- add force_processing + should_process_task
- `code/scripts/_3_preprocessing/_3_forearm_extraction/define_forearm_mesh.py` -- add force_processing + should_process_task

**Dependencies:** None (parallel with Phase 1)

---

### Phase 3: Curation subpackage and script
**Goal:** Commit the new curation module (data layer + GUI + orchestration script).

**Commit:**
```
feat(forearm-extraction): add manual point cloud curation GUI and data layer
```

Phases 1-3 of **forearm-manual-curation-gui** plan. Self-contained curation
subpackage: metadata filehandler for JSON persistence, PyQt5+PyVista interactive
GUI for point removal, and script-level orchestration function with skip-if-exists.
Not yet wired into the pipeline (that happens in Phase 5).

- [x] Identify all files

**Files:**
- `code/src/preprocessing/forearm_extraction/curation/__init__.py` -- **new** package init with exports
- `code/src/preprocessing/forearm_extraction/curation/curation_metadata_filehandler.py` -- **new** JSON load/save for curation metadata
- `code/src/preprocessing/forearm_extraction/curation/forearm_curation_gui.py` -- **new** ForearmCurationGUI (QMainWindow + QtInteractor)
- `code/scripts/_3_preprocessing/_3_forearm_extraction/curate_forearm_pointcloud.py` -- **new** orchestration function

**Dependencies:** None (parallel with Phases 1-2)

---

### Phase 4: Registration subpackage, workbench, and data model
**Goal:** Commit the complete registration module including ICP logic, CSV transformer,
orchestration, viewer, workbench GUI, and catalog extensions.

**Commit:**
```
feat(forearm-extraction): add ICP registration module with workbench GUI
```

Covers Phases 1-3 of **multi-snapshot-forearm-registration**, all phases of
**registration-workbench-gui**, and all phases of **workbench-driven-registration**.
Includes `ForearmRegistrator` (point-to-plane ICP), `csv_spatial_transformer`
(transform contact columns), `register_session_forearms` (orchestration with
interactive/headless paths), `registration_viewer` (read-only inspection),
`RegistrationWorkbench` (dual-viewport parameter tuning with Accept/Result flow).
Extends `ForearmCatalog` with `get_unified_pointcloud()` and
`load_registration_transforms()`. Updates `forearm_extraction/__init__.py` with
exports for both registration and curation subpackages.

- [x] Identify all files

**Files:**
- `code/src/preprocessing/forearm_extraction/registration/__init__.py` -- **new** package init with exports
- `code/src/preprocessing/forearm_extraction/registration/forearm_registrator.py` -- **new** ForearmRegistrator class
- `code/src/preprocessing/forearm_extraction/registration/csv_spatial_transformer.py` -- **new** transform_unified_csv()
- `code/src/preprocessing/forearm_extraction/registration/register_session_forearms.py` -- **new** orchestration (workbench + headless paths)
- `code/src/preprocessing/forearm_extraction/registration/registration_viewer.py` -- **new** read-only Open3D viewer
- `code/src/preprocessing/forearm_extraction/registration/registration_workbench.py` -- **new** dual-viewport workbench with RegistrationResult
- `code/src/preprocessing/forearm_extraction/models/forearm_catalog.py` -- add get_unified_pointcloud(), load_registration_transforms()
- `code/src/preprocessing/forearm_extraction/__init__.py` -- add exports for registration + curation subpackages

**Dependencies:** Phase 3 (curation exports are added to `__init__.py` here)

---

### Phase 5: Manual pipeline integration with DAG config
**Goal:** Commit the pipeline rewrite that wires curation, registration, and DAG config
into the manual forearm extraction pipeline.

**Commit:**
```
feat(forearm-extraction): integrate curation, registration, and DAG config into manual pipeline
```

Covers Phase 4 of **forearm-manual-curation-gui**, Phase 2 of
**forearm-per-step-force-processing**, and Phase 3 (manual integration) of
**multi-snapshot-forearm-registration**. Major rewrite of
`preprocess_pipeline_extract_forearm_manual.py`: removes `FORCE_PROCESSING` constant,
adds `setup_environment()`, integrates `DagConfigHandler`/`TaskExecutor`, expands
`execute_frame_batch` to 5 steps (adds curate between extract and clean), adds
registration as session-level step 7. Creates `apply_registration_transform` script
function and the YAML DAG config.

- [x] Identify all files

**Files:**
- `code/scripts/_3_preprocessing/_3_forearm_extraction/__init__.py` -- add exports for curate_forearm_pointcloud, apply_registration_transform
- `code/scripts/_3_preprocessing/_3_forearm_extraction/apply_registration_transform.py` -- **new** block-order-aware transform application
- `code/scripts/preprocess_pipeline_extract_forearm_manual.py` -- major rewrite (DAG, curation step, registration step)
- `configs/preprocess_forearm_manual_dag.yaml` -- **new** YAML DAG config for manual pipeline

**Dependencies:** Phase 2 (sub-functions accept force_processing), Phase 3 (curation
module exists), Phase 4 (registration module exists)

---

### Phase 6: Automated pipeline integration
**Goal:** Commit the new registration transform flow in the automated Kinect pipeline.

**Commit:**
```
feat(kinect-auto): add registration transform step to automated pipeline
```

Phase 4 of **multi-snapshot-forearm-registration**. Adds
`transform_to_registered_frame` `@flow` function (step 8b), inserts it as a pipeline
stage after `compute_somatosensory_characteristics`, updates `unify_processed_data`
to consume `registered_somatosensory_path`. DAG config gains the new task entry and
updated dependency for `unify_processed_data`.

- [x] Identify all files

**Files:**
- `code/scripts/preprocess_workflow_kinect_auto.py` -- add @flow + pipeline stage entry
- `configs/preprocess_workflow_kinect_auto_dag.yaml` -- add task, update unify_processed_data depends_on

**Dependencies:** Phase 5 (apply_registration_transform script function exists)

---

### Phase 7: Downstream merging integration
**Goal:** Commit the preference for registered CSV in the merging pipeline.

**Commit:**
```
feat(merging): prefer registered CSV in neuron-to-kinect merge pipeline
```

Phase 5 of **multi-snapshot-forearm-registration**. `resolve_filenames()` checks for
`_unified_registered.csv` before falling back to `_unified.csv`. Backward-compatible:
single-forearm sessions produce no registered CSV, so the original path is used.

- [x] Identify all files

**Files:**
- `code/scripts/merging_pipeline_neuron_to_kinect_auto.py` -- prefer registered CSV in resolve_filenames()
- `configs/merging_pipeline_neuron_to_kinect_auto_dag.yaml` -- minor config change (if any diff)

**Dependencies:** Phase 6

---

### Phase 8: DAG launcher UI improvements
**Goal:** Commit the column swap and abort button in the launcher window.

**Commit:**
```
feat(dag-launcher): swap task/kinect columns and add abort button
```

Covers **dag-launcher-swap-columns** (tasks center, kinect dirs right) and
**dag-launcher-abort-button** (red Abort button to terminate subprocess, cleanup
in closeEvent). Both changes are in the same file and are small UI improvements.

- [x] Identify all files

**Files:**
- `code/src/utils/gui/dag_launcher/launcher_window.py` -- swap addWidget order + stretch factors, add Abort button

**Dependencies:** None (parallel with all other phases)

---

### Phase 9: Knowledge base documentation
**Goal:** Commit the ICP registration knowledge base note.

**Commit:**
```
docs(knowledge-base): add ICP registration constraints note for forearm clouds
```

Phase 6 of **multi-snapshot-forearm-registration**. Documents ICP convergence
constraints, default parameters (`max_correspondence_distance=0.10`,
`icp_max_iteration=200`), fitness thresholds, and the point-to-plane method choice.

- [x] Identify all files

**Files:**
- `docs/development/knowledge-base/note-forearm-icp-registration.md` -- **new** knowledge base note
- `docs/development/knowledge-base/README.md` -- add table entry

**Dependencies:** None (parallel with all other phases)

---

## Testing Plan

### Pre-Commit Verification
- [ ] `git diff --cached` reviewed before each commit to confirm only intended files are staged
- [ ] `git status` after each commit confirms remaining files match expectation
- [ ] No `__pycache__`, `.env`, or credential files are staged

### Post-Commit Verification
- [ ] `git log --oneline` shows 9 commits in correct order with correct messages
- [ ] No commit message contains `Co-Authored-By` (per CLAUDE.md authorship rule)
- [ ] Import chain is valid: running `python -c "from preprocessing.forearm_extraction import ForearmRegistrator"` after Phase 4 commit does not error

---

## Documentation Plan

- [ ] No additional documentation changes needed (docs are part of the commits themselves)

---

## Rollback Plan

1. Each commit is independent and revertable via `git revert <sha>`
2. If a commit is found to have a broken import, `git reset HEAD~1` to unstage and
   reassign the missing file to the correct commit
3. No data migrations or external state changes

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Shared `__init__.py` exports committed before the module they export | Medium | High | Phase 4 commits `forearm_extraction/__init__.py` (which exports both curation and registration) after Phase 3 (curation) creates the module |
| `configs/preprocess_workflow_kinect_auto_dag.yaml` contains unrelated config tweaks mixed with registration task | Low | Low | Accept: the config tweaks are minor (directory path, force flag) and logically belong with the DAG update |
| `configs/merging_pipeline_neuron_to_kinect_auto_dag.yaml` has no visible diff | Low | Low | Verify with `git diff` before staging; skip from Phase 7 if truly empty |

---

## References

- Originating plans:
  - `docs/development/plans/active/multi-snapshot-forearm-registration.md`
  - `docs/development/plans/active/forearm-manual-curation-gui.md`
  - `docs/development/plans/active/forearm-per-step-force-processing.md`
  - `docs/development/plans/active/dag-launcher-swap-columns.md`
  - `docs/development/plans/completed/registration-workbench-gui.md`
  - `docs/development/plans/completed/workbench-driven-registration.md`
  - `docs/development/plans/completed/dag-launcher-abort-button.md`
- Commit procedure: `docs/development/git/commit-procedure.md`

---

## Commit Dependency Graph

```
Phase 1 (ROI) ────┐
Phase 2 (skip) ───┤
Phase 3 (CUR) ────┼──> Phase 4 (REG) ──> Phase 5 (manual pipeline) ──> Phase 6 (auto) ──> Phase 7 (merging)
                   │
Phase 8 (launcher) ┘   (independent)
Phase 9 (docs)         (independent)
```
