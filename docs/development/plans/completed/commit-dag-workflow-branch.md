# Plan: Commit DAG Workflow Category Grouping Branch

**Date:** 2026-03-06
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-03-06
**Branch:** `feature/dag-workflow-category-grouping`

---

## Overview

**What:** Organize all uncommitted changes on the `feature/dag-workflow-category-grouping` branch into atomic commits grouped by logical concern.
**Why:** The branch has accumulated changes from multiple completed plans (fix-dag-launcher-config-script-mismatch, dag-workflow-category-grouping addendum, forearm GUI fixes, DAG launcher forearm support, forearm pipeline include-list, view_merged rename) plus plan doc housekeeping. A single bulk commit would be unreadable.
**How:** Stage and commit each logical group separately, following the project's conventional-commit format.

## Problem Statement

The branch has 11 modified/deleted tracked files and 10 untracked files spanning 7 distinct concerns. Committing everything at once would make git history unusable for bisect or review. The changes need to be decomposed into atomic commits that each address one plan or concern.

## Goals

### In Scope
1. Create one atomic commit per logical change group (7 commits total)
2. Follow conventional-commit format with appropriate scopes
3. Move/add plan docs to their correct lifecycle directories

### Out of Scope
- Any code changes beyond what is already in the working tree
- Pushing to remote or creating a PR
- Merging into dev or main

## Success Criteria

- [x] Each commit touches only files related to one concern
- [x] All uncommitted changes are committed (clean working tree)
- [x] Commit messages follow `type(scope): subject` format
- [x] Plan docs are in their correct directories (completed/ or pending/)

---

## Technical Design

### Approach

Analyze `git status` and `git diff` to map each changed file to its logical concern, then stage and commit in dependency order (foundational changes first, dependent changes after).

### Architecture Changes

None. This is a commit-organization plan, not a code-change plan.

---

## Implementation Plan

### Commit 1: Rename forearm DAG config to match script (fix-dag-launcher-config-script-mismatch)

**Message:** `fix(dag-launcher): rename forearm config to match script naming convention`

**Files:**
- `configs/preprocess_forearm_manual_dag.yaml` -> `configs/preprocess_pipeline_extract_forearm_manual_dag.yaml` (already staged rename)
- `code/scripts/preprocess_pipeline_extract_forearm_manual.py` — update config path string (line 67)

**Rationale:** The rename is the foundation for subsequent commits that reference the new filename. Must come first.

---

### Commit 2: Rename view_merged_neural_kinect to merging_view_neural_kinect

**Message:** `refactor(neural-kinect): rename view_merged to merging_view for category grouping`

**Files:**
- `code/scripts/view_merged_neural_kinect.py` (delete)
- `code/scripts/merging_view_neural_kinect.py` (add — identical except config path on line 198)
- `configs/view_merged_neural_kinect_dag.yaml` (delete)
- `configs/merging_view_neural_kinect_dag.yaml` (add — identical content)

**Rationale:** Renames the script+config so the `merging_` prefix groups it with other merging workflows in the category-grouped GUI.

---

### Commit 3: Add new workflow stems to ordered list (dag-workflow-category-grouping addendum)

**Message:** `feat(dag-launcher): add forearm and merging_view stems to ordered workflow list`

**Files:**
- `code/src/utils/gui/dag_launcher/workflow_selector.py` — add 2 entries to `_ORDERED_STEMS`

**Rationale:** Depends on commits 1-2 (the stems must exist). Completes the category grouping feature.

---

### Commit 4: Relax representative frame validation in forearm GUI

**Message:** `fix(forearm-gui): allow representative frame from any valid index, not just group members`

**Files:**
- `code/src/preprocessing/forearm_extraction/gui/group_edit_dialog.py` — change validation from `in unique_frames` to `0 <= rep < total_frames`
- `code/src/preprocessing/forearm_extraction/gui/multivideo_frames_selector.py` — same validation relaxation + remove mark-rep restriction

**Rationale:** Independent bugfix for the forearm curation GUI. No dependency on other commits.

---

### Commit 5: Add forearm config support to DAG launcher GUI

**Message:** `feat(dag-launcher): support forearm_configs directory and include-list model in GUI`

**Files:**
- `code/src/utils/gui/dag_launcher/kinect_directory_selector.py` — detect forearm_configs root, show root-level YAMLs, add `get_checked_forearm_files()`
- `code/src/utils/gui/dag_launcher/launcher_window.py` — route forearm workflows to include-list instead of exclude-list
- `code/src/utils/pipeline/dag_config_model.py` — add `get_config_dir_root_name()`, `get/set_forearm_config_files()`, extend `get_kinect_directories()` for forearm

**Rationale:** Extends the DAG launcher to handle forearm workflows natively. Depends on commit 1 (renamed config).

---

### Commit 6: Add forearm_config_files include-list to pipeline script and DAG YAML

**Message:** `feat(forearm-pipeline): filter session configs via forearm_config_files include-list`

**Files:**
- `code/scripts/preprocess_pipeline_extract_forearm_manual.py` — add `forearm_config_files` parameter to `_discover_session_configs()`
- `configs/preprocess_pipeline_extract_forearm_manual_dag.yaml` — add `forearm_config_files` key with example entry

**Rationale:** The pipeline-side counterpart to commit 5. Reads the include-list that the GUI writes.

---

### Commit 7: Plan docs housekeeping

**Message:** `docs(plans): move completed plans and add pending plan docs`

**Files (add):**
- `docs/development/plans/completed/dag-gui-forearm-pipeline-mapping.md`
- `docs/development/plans/completed/exclude-files-missing-scripts.md`
- `docs/development/plans/completed/fix-dag-launcher-config-script-mismatch.md`
- `docs/development/plans/completed/forearm-per-step-force-processing.md`
- `docs/development/plans/completed/forearm-remove-backward-compat.md`
- `docs/development/plans/completed/pipeline-dag-validation-fixes.md`
- `docs/development/plans/pending/dag-workflow-category-grouping.md`
- `docs/development/plans/pending/exclude-files-missing-scripts.md`

**Files (delete):**
- `docs/development/plans/active/forearm-per-step-force-processing.md` (moved to completed)

**Rationale:** Housekeeping commit. Captures plan lifecycle transitions. Independent of code changes but placed last for cleanliness.

---

## Testing Plan

### Manual Verification
- [ ] After each commit, run `git status` to confirm only intended files were staged
- [ ] After all commits, `git status` shows a clean working tree
- [ ] `git log --oneline -8` shows 7 new commits with correct messages
- [ ] Run `python code/scripts/launch_dag_config_gui.py` to verify the GUI still works

---

## Rollback Plan

`git reset --soft HEAD~7` to undo all commits while preserving changes in the working tree.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Staging the wrong file in a commit | Low | Low | Review `git diff --cached` before each commit |
| Dependency order wrong (commit references file not yet renamed) | Low | Medium | Commits ordered so renames come first |

---

## References

- Completed plans: `docs/development/plans/completed/fix-dag-launcher-config-script-mismatch.md`, `dag-gui-forearm-pipeline-mapping.md`, etc.
- Commit procedure: `.claude/skills/commit-procedure/`
