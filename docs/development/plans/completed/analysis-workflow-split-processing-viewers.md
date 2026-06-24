# Plan: Split Analysis Workflow into Processing and Viewers Scripts

**Date:** 2026-05-04
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-05-05 07:06
**Base Branch:** `feature/contact-projection-first-architecture`
**Branch:** `feature/analysis-workflow-split-processing-viewers`

---

## Overview

The analysis workflow currently lives in a single script (`analysis_workflow.py`) with a `--mode processing|viewers|all` flag introduced in `f822e39`. This plan replaces that flag with two physical scripts and two distinct GUI entries, mirroring the established merging pipeline pattern (`merging_pipeline_neuron_to_kinect_auto.py` + `merging_pipeline_neuron_to_kinect_visualisation.py`).

## Problem Statement

The current single-script approach with `--mode` is a CLI workaround, not a structural separation. The GUI launcher still shows one "Analysis Workflow" button, so users cannot select processing vs. viewers as independent workflows. The merging pipeline demonstrates the correct pattern: two scripts, two DAG configs, two GUI entries.

## Goals

### In Scope
1. Create `analysis_workflow_processing.py` — entry point for processing tasks only
2. Create `analysis_workflow_viewers.py` — entry point for viewer tasks only
3. Split `analyse_workflow_dag.yaml` into two separate DAG configs
4. Register both scripts in `launcher.yaml` as two distinct GUI buttons
5. Remove the `--mode` flag from the user-facing interface

### Out of Scope
- Changing any flow function logic inside `analysis_workflow.py`
- Changing task categories (`processing`, `viewer`, `viewer_support`)
- Removing `analysis_workflow.py` (kept as shared implementation module)
- Removing `analyse_workflow_dag.yaml` (kept for backward-compatible manual CLI use)

## Success Criteria

- [ ] GUI shows two buttons under "Analysis": "Analysis [Processing]" and "Analysis [Viewers]"
- [ ] `analysis_workflow_processing.py --help` shows no `--mode` flag
- [ ] `analysis_workflow_viewers.py --help` shows no `--mode` flag
- [ ] Running the processing script executes only `category: processing` tasks
- [ ] Running the viewers script executes only `category: viewer` and `category: viewer_support` tasks
- [ ] Each new script accepts `--dag-config` pointing to its dedicated DAG YAML

---

## Technical Design

### Approach

Keep all flow functions and `run_batch_analysis()` in `analysis_workflow.py` (unchanged). Create two thin entry-point scripts that import `run_batch_analysis` and call it with a hardcoded mode. This avoids code duplication while producing the two physical files the user expects. `analysis_workflow.py` becomes an internal module (not registered in the GUI).

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Thin wrappers importing `analysis_workflow` | No code duplication; flow logic in one place | `analysis_workflow.py` persists as a non-entry-point | **Chosen** |
| Full self-contained split | Each script independently readable | ~400 lines duplicated across both scripts | Rejected |
| Keep single script, just add two launcher entries with `--mode` arg | Trivial | GUI passes `--dag-config` only; no way to pass `--mode` via launcher | Rejected |

### Architecture Changes

**New files:**
- `code/scripts/analysis_workflow_processing.py` — thin launcher, imports core, mode="processing"
- `code/scripts/analysis_workflow_viewers.py` — thin launcher, imports core, mode="viewers"
- `configs/analyse_workflow_processing_dag.yaml` — processing tasks only
- `configs/analyse_workflow_viewers_dag.yaml` — viewer + viewer_support tasks only

**Modified files:**
- `configs/launcher.yaml` — replace one Analysis entry with two

**Unchanged (demoted to internal module):**
- `code/scripts/analysis_workflow.py` — all flow logic stays here; removed from GUI
- `configs/analyse_workflow_dag.yaml` — kept for backward-compatible manual use; removed from GUI

### Script structure

`analysis_workflow_processing.py` skeleton:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from analysis_workflow import run_batch_analysis   # import shared core
# ... same DagConfigHandler + forearm_configs setup as analysis_workflow.main()
# argparse: --dag-config only
run_batch_analysis(dag_handler, forearm_configs, mode="processing")
```

`analysis_workflow_viewers.py` skeleton:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from analysis_workflow import run_batch_analysis
# argparse: --dag-config only
run_batch_analysis(dag_handler, forearm_configs, mode="viewers")
```

### DAG config split

**`analyse_workflow_processing_dag.yaml`** — retains:
- All global config keys (`forearm_configs`, etc.)
- Tasks with `category: processing`: `summarize_session_blocks`, `map_receptive_fields_simple`, `touch_preparation`, `touch_series_transforms`, `touch_feature_extraction`, `touch_clustering`, `touch_comparing`, `analyse_ap_efficacy`, `extract_receptive_fields_clustered`, `visualize_receptive_fields_clustered`
- Deprecated task `map_receptive_fields_clustered`
- All `depends_on` entries between processing tasks are preserved

**`analyse_workflow_viewers_dag.yaml`** — retains:
- All global config keys
- Tasks with `category: viewer_support` and `category: viewer`: `precompute_explorer_caches`, `explore_preparation`, `explore_rf_feature_space`, `explore_touch_playback`, `explore_rf_gallery`
- `depends_on` entries referencing processing tasks are **removed** (processing outputs assumed to exist on disk)

### Launcher update

```yaml
- name: Analysis
  workflows:
    - name: Analysis [Processing]
      script: code/scripts/analysis_workflow_processing.py
      dag_config: configs/analyse_workflow_processing_dag.yaml
    - name: Analysis [Viewers]
      script: code/scripts/analysis_workflow_viewers.py
      dag_config: configs/analyse_workflow_viewers_dag.yaml
```

---

## Implementation Plan

### Phase 1: DAG Config Split
**Goal:** Produce the two DAG YAML files from the existing combined one.
**Started:** 2026-05-05
**Completed:** 2026-05-05

- [x] Read `configs/analyse_workflow_dag.yaml` in full
- [x] Create `configs/analyse_workflow_processing_dag.yaml` — processing tasks only
- [x] Create `configs/analyse_workflow_viewers_dag.yaml` — viewer + viewer_support tasks, cross-boundary `depends_on` removed

**Files Modified:**
- `configs/analyse_workflow_processing_dag.yaml` — created
- `configs/analyse_workflow_viewers_dag.yaml` — created

**Dependencies:** None

### Phase 2: Entry-Point Scripts
**Goal:** Create the two thin launcher scripts.
**Started:** 2026-05-05
**Completed:** 2026-05-05

- [x] Read `code/scripts/analysis_workflow.py` `main()` to extract exact init pattern (DagConfigHandler construction, forearm_configs resolution, Prefect server setup)
- [x] Create `code/scripts/analysis_workflow_processing.py`
- [x] Create `code/scripts/analysis_workflow_viewers.py`

**Files Modified:**
- `code/scripts/analysis_workflow_processing.py` — created
- `code/scripts/analysis_workflow_viewers.py` — created

**Dependencies:** Phase 1

### Phase 3: Launcher Registration
**Goal:** Wire both scripts into the GUI.
**Started:** 2026-05-05
**Completed:** 2026-05-05

- [x] Edit `configs/launcher.yaml`: replace the single Analysis entry with the two new entries
- [x] Verify the old `analysis_workflow.py` entry is removed

**Files Modified:**
- `configs/launcher.yaml` — two Analysis entries replacing one

**Dependencies:** Phase 2

---

## Testing Plan

### Manual Verification
- [ ] Launch GUI (`python code/scripts/launch_pipeline_gui.py`) → two buttons "Analysis [Processing]" and "Analysis [Viewers]" appear under Analysis
- [ ] `python code/scripts/analysis_workflow_processing.py --help` → `--mode` absent from output
- [ ] `python code/scripts/analysis_workflow_viewers.py --help` → `--mode` absent from output
- [ ] Enable `summarize_session_blocks` in processing DAG, run processing script → task executes without opening any GUI
- [ ] Enable `explore_touch_playback` in viewers DAG, run viewers script → TouchPlaybackExplorer launches

### Edge Cases
- [ ] Running processing script with viewers DAG → only processing tasks exist in that DAG, nothing runs (graceful no-op)
- [ ] Running viewers script with processing DAG → only processing tasks exist, nothing runs (graceful no-op)

---

## Documentation Plan

- [ ] Update `docs/development/plans/active/analysis-workflow-processing-viewer-separation.md` to note that `--mode` flag was superseded by the script split (append a note at the bottom of that plan)

---

## Rollback Plan

All changes are additive (new files) except the `launcher.yaml` edit:
1. Revert `configs/launcher.yaml` to restore the single "Analysis Workflow" entry
2. Delete `analysis_workflow_processing.py`, `analysis_workflow_viewers.py`
3. Delete `analyse_workflow_processing_dag.yaml`, `analyse_workflow_viewers_dag.yaml`

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `sys.path` insert in new scripts causes import conflicts | Low | Medium | Test imports immediately after creating scripts; use `Path(__file__).parent` which is deterministic |
| Viewer DAG tasks with removed `depends_on` silently run out of order | Low | Low | Viewer tasks are independent of each other; order in YAML is sufficient |

---

## References

- Related Plan: `docs/development/plans/active/analysis-workflow-processing-viewer-separation.md`
- Pattern reference: `code/scripts/merging_pipeline_neuron_to_kinect_auto.py` and `merging_pipeline_neuron_to_kinect_visualisation.py`
