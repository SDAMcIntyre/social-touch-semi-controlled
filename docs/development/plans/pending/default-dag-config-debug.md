# Plan: Default DAG Config for Standalone Script Debugging

**Date:** 2026-03-17
**Author:** Basil Duvernoy
**Status:** Draft
**Branch:** `feature/default-dag-config-debug`

---

## Overview

Make `--dag-config` optional in all 10 main workflow scripts so they can be launched directly via VS Code's debugger (F5) without needing to configure launch arguments. When `--dag-config` is omitted, each script falls back to its corresponding default DAG YAML — the same one declared in `configs/launcher.yaml`.

## Problem Statement

All workflow scripts require `--dag-config <path>` as a mandatory argparse argument. This was fine when scripts were always launched through the DAG Launcher GUI (which passes `--dag-config` automatically), but it blocks standalone debugging in VS Code: hitting F5 on any workflow script immediately fails with a missing required argument error.

Developers must either manually configure `launch.json` entries with `args` for each script or run scripts exclusively through the GUI, losing the ability to set breakpoints and step through code interactively.

## Goals

### In Scope
1. Make `--dag-config` optional with a sensible default in all 10 workflow scripts
2. Preserve full backward compatibility — passing `--dag-config` explicitly still works identically
3. GUI launcher behavior is unchanged (it always passes `--dag-config` explicitly)

### Out of Scope
- Adding VS Code `launch.json` entries for each script (the user can configure these if desired)
- Adding `__main__` blocks to individual processing functions (e.g., `track_handstickers_roi.py`)
- Changes to the DAG Launcher GUI or `launcher.yaml`

## Success Criteria

- [ ] All 10 workflow scripts accept `--dag-config` as optional with a default
- [ ] Running `python code/scripts/<script>.py` from the project root (no args) starts processing using the default DAG config
- [ ] Running `python code/scripts/<script>.py --dag-config <path>` works identically to before
- [ ] GUI launcher behavior is unchanged

---

## Technical Design

### Approach

In each script's argparse setup, change:

```python
parser.add_argument("--dag-config", type=Path, required=True)
```

to:

```python
parser.add_argument("--dag-config", type=Path,
                    default=Path("configs/<corresponding_dag_config>.yaml"))
```

This is a one-line change per script. The default path is a relative path that works when the script is run from the project root (which is both the VS Code workspace root and the `cwd` set by the GUI launcher).

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Default argparse value | Minimal change, no new code, backward compatible | Requires running from project root | **Chosen** |
| Auto-discover config from script name | More flexible, works from any directory | Over-engineered for the need, adds complexity | Rejected |
| VS Code `launch.json` entries only | No code changes | Must maintain launch configs in sync, doesn't help CLI use | Rejected |

### Architecture Changes

None — this is a one-line behavioral change in each script's argument parser. No new modules, classes, or interfaces.

---

## Implementation Plan

### Phase 1: Update All 10 Scripts
**Goal:** Make `--dag-config` optional with correct defaults

**Tasks:**
- [ ] Update `preprocess_workflow_kinect_auto.py` — default: `configs/preprocess_workflow_kinect_auto_dag.yaml`
- [ ] Update `preprocess_workflow_kinect_manual.py` — default: `configs/preprocess_workflow_kinect_manual_dag.yaml`
- [ ] Update `preprocess_workflow_kinect_visualisation.py` — default: `configs/preprocess_workflow_kinect_visualisation_dag.yaml`
- [ ] Update `preprocess_pipeline_extract_forearm_manual.py` — default: `configs/preprocess_pipeline_extract_forearm_manual_dag.yaml`
- [ ] Update `primary_workflow_kinect_auto.py` — default: `configs/primary_workflow_kinect_auto_dag.yaml`
- [ ] Update `postprocess_workflow_kinect_auto.py` — default: `configs/postprocess_workflow_kinect_auto_dag.yaml`
- [ ] Update `postprocess_visualization.py` — default: `configs/postprocess_visualization_dag.yaml`
- [ ] Update `analysis_workflow.py` — default: `configs/analyse_workflow_dag.yaml`
- [ ] Update `merging_pipeline_neuron_to_kinect_auto.py` — default: `configs/merging_pipeline_neuron_to_kinect_auto_dag.yaml`
- [ ] Update `merging_pipeline_neuron_to_kinect_visualisation.py` — default: `configs/merging_pipeline_neuron_to_kinect_visualisation_dag.yaml`

**Files Modified:**
- `code/scripts/preprocess_workflow_kinect_auto.py` — argparse default
- `code/scripts/preprocess_workflow_kinect_manual.py` — argparse default
- `code/scripts/preprocess_workflow_kinect_visualisation.py` — argparse default
- `code/scripts/preprocess_pipeline_extract_forearm_manual.py` — argparse default
- `code/scripts/primary_workflow_kinect_auto.py` — argparse default
- `code/scripts/postprocess_workflow_kinect_auto.py` — argparse default
- `code/scripts/postprocess_visualization.py` — argparse default
- `code/scripts/analysis_workflow.py` — argparse default
- `code/scripts/merging_pipeline_neuron_to_kinect_auto.py` — argparse default
- `code/scripts/merging_pipeline_neuron_to_kinect_visualisation.py` — argparse default

**Dependencies:** None

---

## Testing Plan

### Manual Verification
- [ ] Pick one script (e.g., `preprocess_workflow_kinect_auto.py`), run it from project root with no arguments — verify it picks up the default config and starts
- [ ] Run the same script with `--dag-config` explicitly — verify identical behavior to before
- [ ] Launch the same workflow through the DAG Launcher GUI — verify no change in behavior
- [ ] Open a workflow script in VS Code, hit F5 (with no launch.json args) — verify it starts debugging with the default config

### Edge Cases
- [ ] Running from a different working directory (should fail gracefully with "config not found" rather than a cryptic argparse error)

---

## Documentation Plan

- [ ] No documentation changes needed — this is a developer ergonomics improvement with no user-facing API change

---

## Rollback Plan

1. Revert the single-line change in each script (restore `required=True`, remove `default=...`)
2. No data or config changes to reverse

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Script run from wrong directory, default config not found | Low | Low | Existing `FileNotFoundError` handling in each script already prints a clear error message |
| Default path drifts out of sync with `launcher.yaml` | Low | Low | Mapping is obvious from naming convention; could add a comment referencing `launcher.yaml` |

---

## References

- `configs/launcher.yaml` — single source of truth for script↔config mapping
- `.vscode/launch.json` — existing debug configurations (currently only 2 entries)
