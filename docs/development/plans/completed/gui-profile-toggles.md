# Plan: GUI Profile Toggles for Extraction & Clustering

**Date:** 2026-03-17
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-03-18 11:29
**Branch:** `feature/unified-touch-analysis-pipeline`

---

## Overview

Add per-profile enable/disable checkboxes to the DAG launcher TaskPanel so users can toggle individual extraction profiles (max, statistical, temporal, mos) and clustering profiles (kmeans, dbscan) directly from the GUI. Currently these nested dict options render as uneditable grey text.

## Problem Statement

The `unified_touch_analysis` task exposes `extraction_profiles` and `clustering_profiles` as nested dict options in the DAG YAML. The TaskPanel only supports boolean checkboxes and read-only text for non-boolean values. Users must manually edit the YAML file to add or remove profiles, which is error-prone and defeats the purpose of the GUI launcher.

## Goals

### In Scope
1. Per-profile checkboxes in the TaskPanel for extraction and clustering profiles
2. Round-trip persistence of enabled/disabled state via the YAML config
3. Pipeline-side filtering to skip disabled profiles

### Out of Scope
- Editing profile parameters (e.g., changing `youngs_modulus_kpa`) from the GUI
- Adding new profile types from the GUI
- Reordering profiles

## Success Criteria

- [ ] TaskPanel displays one checkbox column per profile (e.g., "Extract: Max", "Cluster: Kmeans")
- [ ] Toggling a checkbox persists `enabled: false` in the profile's YAML dict
- [ ] Re-enabling removes the `enabled` key (absence = enabled, backward compatible)
- [ ] Pipeline skips profiles with `enabled: false`
- [ ] Existing YAML files without `enabled` keys work unchanged (all profiles default to enabled)

---

## Technical Design

### Approach

**YAML convention:** Each profile dict gains an optional `enabled` key. When `false`, the profile is skipped. When absent or `true`, the profile runs. Re-enabling deletes the key to keep YAML clean.

```yaml
extraction_profiles:
  max:
    method: max
  temporal:
    enabled: false          # toggled off in GUI
    method: temporal
```

**GUI:** The TaskPanel's column discovery logic is refactored to detect "profile container" options (dicts of dicts with `method` keys) and expand them into individual checkbox columns instead of a single grey-text column.

**Pipeline:** A two-line filter removes disabled profiles before the extraction/clustering loops.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| `enabled: false` in profile dict | Minimal YAML change, preserves config for re-enabling, backward compatible | `enabled` key passed to extractors (harmless — they ignore unknown keys) | **Chosen** |
| Move disabled profiles to `_disabled_extraction_profiles` sibling key | Pipeline needs no changes | Complex model logic, ugly YAML, two keys to maintain | Rejected |
| Remove disabled profiles from YAML entirely | Clean YAML, no pipeline changes | Custom config lost when toggled off then on; no round-trip fidelity | Rejected |

### Architecture Changes

No new modules. Changes are localized to three existing files plus a helper function and two model methods.

**Profile detection heuristic:** A value is a "profile container" if it is a non-empty dict where every value is itself a dict containing a `"method"` key. This correctly excludes `grouping_columns` (a list), `force_processing` (a bool), and `monitor` (a bool).

---

## Implementation Plan

### Phase 1: Model Layer
**Goal:** Add profile-level read/write methods to `DagConfigModel`

- [ ] Add `get_profile_names(task_name, option_key) -> list[str]`
- [ ] Add `get_profile_enabled(task_name, option_key, profile_name) -> bool` — returns `profile.get("enabled", True)`
- [ ] Add `set_profile_enabled(task_name, option_key, profile_name, enabled)` — sets `enabled: false` when disabling; deletes the key when enabling

**Files Modified:**
- `code/src/utils/pipeline/dag_config_model.py` — add three methods

**Dependencies:** None

### Phase 2: GUI Layer
**Goal:** Render per-profile checkbox columns in TaskPanel

- [ ] Add `_is_profile_dict(val)` module-level helper
- [ ] Add `_profile_header(option_key, profile_name)` helper for column headers (e.g., "Extract: Max", "Cluster: Kmeans")
- [ ] Replace `_option_keys: list[str]` with `_column_spec: list[tuple]` where each entry is `("simple", key)` or `("profile", key, profile_name)`
- [ ] Refactor column discovery in `populate()` to expand profile containers into individual entries
- [ ] Refactor cell rendering: profile columns get checkboxes via `model.get_profile_enabled()`; simple columns unchanged
- [ ] Add `_make_profile_handler()` factory calling `model.set_profile_enabled()`
- [ ] Update `_on_cell_clicked()` depends_col calculation to use `len(self._column_spec)`

**Files Modified:**
- `code/src/utils/gui/dag_launcher/task_panel.py` — refactor column discovery, rendering, and handler logic

**Dependencies:** Phase 1

### Phase 3: Pipeline Layer
**Goal:** Skip disabled profiles during execution

- [ ] In `run_unified_touch_analysis()`, filter `extraction_profiles` and `clustering_profiles` dicts to exclude entries where `cfg.get("enabled", True)` is `False`

**Files Modified:**
- `code/src/analysis/touch_analytics/unified_pipeline.py` — add two-line filter after reading profiles from options

**Dependencies:** None (can be done in parallel with Phase 1-2)

---

## Testing Plan

### Manual Verification
- [ ] Launch GUI (`python code/scripts/launch_pipeline_gui.py`), select "Analysis Workflow"
- [ ] Verify profile columns appear with checkboxes (4 extraction + 2 clustering)
- [ ] Toggle off `temporal` and `dbscan`, click save — verify YAML has `enabled: false` on those profiles
- [ ] Reload the workflow — verify toggle states persist
- [ ] Run the pipeline — verify only enabled profiles produce output directories
- [ ] Verify other tasks (summarize_session_blocks, analyse_ap_efficacy, map_receptive_fields) still render correctly with grey N/A cells in profile columns

### Edge Cases
- [ ] All profiles disabled — pipeline should produce no extraction/clustering output without crashing
- [ ] YAML without any `enabled` keys (pre-existing config) — all profiles should default to enabled
- [ ] Task with no `extraction_profiles` option — profile columns show grey N/A cells

---

## Documentation Plan

- [ ] No external documentation needed — GUI is self-explanatory
- [ ] Inline comments on `_is_profile_dict` heuristic

---

## Rollback Plan

1. Revert commits on the feature branch
2. No YAML migration needed — `enabled` keys are simply ignored if the old code is restored
3. No data impact — this only affects which profiles run, not the output format

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `_is_profile_dict` heuristic matches unintended options | Low | Med | Heuristic requires all values to be dicts with `method` key; no current options match this pattern besides profiles |
| `enabled` key leaks into extractor/clusterer config | Low | Low | Extractors/clusterers use `.get()` for known keys only; `enabled` is ignored |
| Column explosion if many profiles are added in future | Low | Low | Columns auto-resize via `ResizeToContents`; manageable up to ~10 profiles |

---

## References

- Related Plan: `docs/development/plans/pending/unified-pipeline-progress-reporting.md`
- Unified pipeline commit: `1903036`
