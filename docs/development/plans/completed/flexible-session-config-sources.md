# Plan: Flexible Session Config Sources in DAG YAML Files

**Date:** 2026-03-09
**Author:** Basil Duvernoy
**Status:** Draft
**Branch:** `feature/flexible-session-config-sources`

---

## Overview

**What:** Replace the rigid `kinect_configs_directory` / `forearm_configs_directory` DAG
parameters with flexible `kinect_configs` / `forearm_configs` keys that accept a folder,
a specific YAML file, or a flow-style list `[...]` mixing both.

**Why:** Current parameters only accept a single directory. Selecting specific files
requires auxiliary keys (`exclude_files`, `forearm_config_files`) that add complexity.
The "directory" naming is inaccurate once the value can be a file.

**How:** Introduce a single resolver utility, update the model/GUI/scripts to use the new
key format, and migrate all 9 DAG YAML configs.

## Problem Statement

- `kinect_configs_directory` accepts only a directory path; selecting individual files
  requires a separate `exclude_files` list (subtract model).
- `forearm_configs_directory` accepts only a directory; selecting files requires a
  separate `forearm_config_files` list (include model).
- Two different selection models (include vs exclude) for the same conceptual operation.
- The analysis workflow uses a third variant (`kinect_configs_directories`, plural).
- Three key names for one concept makes the codebase harder to reason about.

## Goals

### In Scope
1. Rename parameters to `kinect_configs` / `forearm_configs`
2. Accept string or flow-style list; each entry can be a directory or a `.yaml` file
3. Paths are relative to the implied root (`configs/kinect_configs/` or `configs/forearm_configs/`)
4. Eliminate `exclude_files` and `forearm_config_files` auxiliary keys
5. Update DagConfigModel, GUI selector, all 9 pipeline scripts, and all 9 DAG YAML files
6. Preserve ruamel.yaml round-trip fidelity (comments, ordering)

### Out of Scope
- Adding new GUI features beyond adapting the existing selector to the new format
- Changing how individual session configs are loaded (`KinectConfigFileHandler`, `ForearmConfigFileHandler`)
- Backward-compatibility shim for old key names (clean migration, all configs are versioned)

## Success Criteria

- [ ] All 9 DAG YAML files use `kinect_configs` or `forearm_configs` with the new format
- [ ] No references to `kinect_configs_directory`, `kinect_configs_directories`, `forearm_configs_directory`, `exclude_files`, or `forearm_config_files` remain in code or configs
- [ ] GUI populates tree correctly and saves flow-style `[item1, item2]` lists
- [ ] Each pipeline script resolves the same session files as before migration
- [ ] `merging_view_neural_kinect` (previously using `exclude_files`) produces identical file list

---

## Technical Design

### New YAML Format

```yaml
# Single directory — glob all *.yaml inside
kinect_configs: "valid_configs_ST13-01"

# Single file
kinect_configs: "valid_configs_ST13-01/kinect_config_block-order02.yaml"

# Mixed list (flow style)
kinect_configs: [valid_configs_ST13-01, valid_configs_ST13-02/specific_block.yaml]

# Forearm — specific files
forearm_configs: [session_2022-06-22_ST18-01.yaml, session_2022-06-22_ST18-02.yaml]

# Forearm — entire root directory
forearm_configs: "."
```

Flow-style lists via `ruamel.yaml.CommentedSeq` + `.fa.set_flow_style()`.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| `kinect_configs` / `forearm_configs` (chosen) | Shortest names, self-documenting | Slight ambiguity with `configs/kinect_configs/` dir name | **Chosen** |
| `kinect_session_configs` / `forearm_session_configs` | More explicit | Verbose, redundant with "configs" in path | Rejected |
| Single `session_configs` + `config_type` field | Fully generic | Adds required field, breaks mental model | Rejected |
| Keep auxiliary keys alongside new format | Backward compat | Two selection models persist, added complexity | Rejected |

### Architecture Changes

**New module:**
- `code/src/utils/pipeline/session_config_resolver.py` — single `resolve_session_configs()` function

**Modified modules:**
- `code/src/utils/pipeline/dag_config_model.py` — replace 6 methods with 3 unified ones
- `code/src/utils/gui/dag_launcher/kinect_directory_selector.py` — adapt to unified format, rename class
- `code/src/utils/gui/dag_launcher/launcher_window.py` — simplify selection handler

**Replaced functions:**
- `get_block_files()` in `kinect_config_filehandler.py:105-111`
- `_discover_session_configs()` in `preprocess_pipeline_extract_forearm_manual.py:425-448`

### Resolver Function Signature

```python
def resolve_session_configs(
    entries: str | list[str],
    config_root: Path,        # e.g. configs_dir / "kinect_configs"
) -> list[Path]:
    """Resolve a mix of directories and files into a flat sorted list of YAML paths."""
```

Logic: normalize to list → for each entry: directory → `sorted(glob("*.yaml"))`,
file → append directly, else → raise `FileNotFoundError` → deduplicate preserving order.

---

## Implementation Plan

### Phase 1: Resolver utility + DagConfigModel
**Goal:** Core logic that everything else depends on.

- [ ] Create `code/src/utils/pipeline/session_config_resolver.py` with `resolve_session_configs()`
- [ ] Update `dag_config_model.py`: replace lines 45–138 with `get_config_type()`, `get_config_entries()`, `set_config_entries()`
- [ ] `set_config_entries()` writes flow-style lists via `CommentedSeq.fa.set_flow_style()`

**Files created:**
- `code/src/utils/pipeline/session_config_resolver.py`

**Files modified:**
- `code/src/utils/pipeline/dag_config_model.py` — replace 6 directory/exclude/include methods with 3 unified methods

**Dependencies:** None

---

### Phase 2: Pipeline scripts (parallelizable — 3 subagent batches)
**Goal:** Update all 9 scripts to use `resolve_session_configs()`.

Each script follows the same mechanical pattern:
```python
# Before
dir_name = dag_handler.get_parameter('kinect_configs_directory')
files = get_block_files(configs_dir / dir_name)
excluded = set(dag_handler.get_parameter('exclude_files', []) or [])
files = [f for f in files if f.name not in excluded]

# After
from utils.pipeline.session_config_resolver import resolve_session_configs
entries = dag_handler.get_parameter('kinect_configs')  # or 'forearm_configs'
files = resolve_session_configs(entries, configs_dir / "kinect_configs")
```

**Subagent batch A** (kinect preprocess + primary):
- [ ] `code/scripts/preprocess_workflow_kinect_auto.py`
- [ ] `code/scripts/preprocess_workflow_kinect_manual.py`
- [ ] `code/scripts/preprocess_workflow_kinect_visualisation.py`
- [ ] `code/scripts/primary_workflow_kinect_auto.py`

**Subagent batch B** (merging + postprocess + analysis):
- [ ] `code/scripts/merging_pipeline_neuron_to_kinect_auto.py`
- [ ] `code/scripts/merging_view_neural_kinect.py` — also remove `exclude_files` consumption
- [ ] `code/scripts/postprocess_workflow_kinect_auto.py`
- [ ] `code/scripts/analysis_workflow.py` — change from `kinect_configs_directories` (plural) to `kinect_configs`

**Subagent batch C** (forearm + cleanup):
- [ ] `code/scripts/preprocess_pipeline_extract_forearm_manual.py` — uses `forearm_configs`, remove `_discover_session_configs()` and `forearm_config_files` consumption
- [ ] `code/src/primary_processing/data_access/kinect_config_filehandler.py` — remove `get_block_files()` (now unused)

**Dependencies:** Phase 1

---

### Phase 3: GUI adaptation
**Goal:** Update selector widget and launcher window.

- [ ] Rename `KinectDirectorySelector` → `SessionConfigSelector` in `kinect_directory_selector.py`
- [ ] `populate()`: derive check state from `model.get_config_entries()` — dir entry → check entire subtree, file entry → check specific file
- [ ] `get_selection() -> list[str]`: fully-checked dir → emit dir name, partially-checked dir → emit individual `subdir/file.yaml` paths
- [ ] Remove `get_checked_forearm_files()` (merged into `get_selection()`)
- [ ] Update `launcher_window.py`: simplify `_on_kinect_changed()` to single `model.set_config_entries(selector.get_selection())` call
- [ ] Update imports and attribute names for renamed class

**Files modified:**
- `code/src/utils/gui/dag_launcher/kinect_directory_selector.py` — rename class, rewrite populate/selection
- `code/src/utils/gui/dag_launcher/launcher_window.py` — simplify handler, update imports

**Dependencies:** Phase 1

---

### Phase 4: YAML config migration (parallelizable with Phase 3)
**Goal:** Migrate all DAG YAML files to new key names and format.

For each file: rename key, strip root prefix from paths, use flow-style list.

- [ ] `configs/preprocess_workflow_kinect_auto_dag.yaml` — `kinect_configs_directory: ""` → `kinect_configs: ""`
- [ ] `configs/primary_workflow_kinect_auto_dag.yaml` — strip `kinect_configs/` prefix
- [ ] `configs/preprocess_workflow_kinect_manual_dag.yaml` — strip prefix
- [ ] `configs/preprocess_workflow_kinect_visualisation_dag.yaml` — strip prefix
- [ ] `configs/merging_pipeline_neuron_to_kinect_auto_dag.yaml` — strip prefix
- [ ] `configs/merging_view_neural_kinect_dag.yaml` — strip prefix, compute effective file list from directory minus `exclude_files`, remove `exclude_files` key
- [ ] `configs/postprocess_workflow_kinect_auto_dag.yaml` — strip prefix
- [ ] `configs/analyse_workflow_dag.yaml` — rename `kinect_configs_directories` → `kinect_configs`, strip prefixes, flow-style list
- [ ] `configs/preprocess_pipeline_extract_forearm_manual_dag.yaml` — rename `forearm_configs_directory` → `forearm_configs`, fold `forearm_config_files` into main key, remove auxiliary key

**Dependencies:** None (can run in parallel with Phases 2–3)

---

## Parallelization Guide for Subagents

Phases 2, 3, and 4 are independent of each other (all depend only on Phase 1).
During implementation, launch up to 4 background agents simultaneously:

| Agent | Phase | Scope | Context needed |
|-------|-------|-------|----------------|
| A | 2 | Scripts batch A (4 kinect preprocess/primary scripts) | Resolver function signature, old→new pattern |
| B | 2 | Scripts batch B (4 merging/postprocess/analysis scripts) | Same as A, plus `exclude_files` removal |
| C | 2+cleanup | Forearm script + `get_block_files` removal | Resolver signature, forearm key name |
| D | 4 | All 9 YAML config files | New key names, prefix stripping rules |

Phase 3 (GUI) should be done in the main context as it requires careful coordination
between two tightly coupled files.

---

## Testing Plan

### Manual Verification
- [ ] Launch GUI → select a kinect workflow → verify tree populates with correct check states
- [ ] Launch GUI → select forearm workflow → verify tree populates correctly
- [ ] Check/uncheck files in GUI → save → verify YAML output uses `kinect_configs: [...]` flow-style
- [ ] Partially check a directory → save → verify individual file paths are written (not dir name)
- [ ] Fully check a directory → save → verify dir name is written (not individual files)

### Script Verification
- [ ] Run `preprocess_workflow_kinect_auto.py` with migrated config → same session files discovered
- [ ] Run `preprocess_pipeline_extract_forearm_manual.py` with migrated config → same session files
- [ ] Run `merging_view_neural_kinect.py` → verify previously-excluded files are still absent

### Edge Cases
- [ ] Empty string value (`kinect_configs: ""`) → raises or returns empty list gracefully
- [ ] Single file entry (string, not list) → resolves to one file
- [ ] Single directory entry (string) → globs all `*.yaml` inside
- [ ] Mixed list with nonexistent entry → raises `FileNotFoundError` with clear message
- [ ] `forearm_configs: "."` → globs root forearm_configs directory

---

## Documentation Plan

- [ ] Update `CLAUDE.md` if any architectural conventions change
- [ ] Inline comments on `resolve_session_configs()` explaining the resolution logic

---

## Rollback Plan

1. All changes are in a single feature branch — revert by not merging
2. No data migrations, no database changes — purely code + config YAML
3. If a single script breaks: revert that script's changes and its DAG YAML to old key names

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| GUI tree doesn't map cleanly to mixed dir/file entries | Medium | Medium | Tree already shows dirs with children; fully-checked dir → dir name, partial → file paths maps naturally |
| `merging_view_neural_kinect` exclude→include migration misses a file | Low | Medium | Manually verify the effective file list before and after migration |
| ruamel.yaml flow-style list breaks comment preservation | Low | Low | Test round-trip on all 9 YAML files; flow style only affects the value, not surrounding comments |
| `analysis_workflow.py` plural key has different semantics | Low | Low | New format is already a list — same resolution logic applies |

---

## References

- Internal draft: `.claude/plans/swift-sleeping-flamingo.md`
- Knowledge base: `docs/development/knowledge-base/note-cupy-import-order.md` — entry-point guard (already handled in `launch_dag_config_gui.py`)
