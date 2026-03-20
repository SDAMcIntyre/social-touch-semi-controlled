# Plan: Composable Feature GUI & Dynamic Heatmaps

**Created:** 2026-03-20
**Completed:** 2026-03-20 19:51
**Author:** Basil Duvernoy
**Status:** Completed
**Branch:** `feature/composable-feature-extraction` (amends existing plan)

---

## Overview

**What:** Amend the active `composable-feature-extraction` plan to (A) add full CRUD GUI support for feature combinations in the TaskPanel, and (B) make heatmap/matrix generation dynamic based on available features instead of hardcoded `*_max` columns.

**Why:** The existing plan's Phase 7 (GUI) only covers enable/disable of pre-defined combinations — there is no way to create a new combination (e.g., "cluster by mean only") from the GUI. Phase 3 only renames column references but doesn't make heatmaps work with non-max aggregations, causing heatmaps to be silently skipped when other feature sets are used.

**How:** Extend the TaskPanel with new column types for feature dicts and feature combinations (including create/delete), add a `FeatureCombinationDialog`, and replace hardcoded heatmap column names with dynamic resolution from DataFrame columns.

## Problem Statement

1. **GUI gap:** A user wanting to cluster by mean only must manually edit YAML to add `mean_only: {features: [mean]}` to the clustering task's `feature_combinations`. The GUI has no affordance for creating, editing, or deleting combinations.

2. **Heatmap gap:** `matrix_generation.py` hardcodes `velocity_max`, `depth_max`, `area_max` as heatmap axes. `touch_config.py` hardcodes `depth_max`, `area_max`, `velocity_max`, `acceleration_max` for discretization. When only `mean` features are present, all column lookups fail silently — no heatmaps are generated, and the matrix degrades to a trivial 1D breakdown.

## Goals

### In Scope
1. GUI to create, edit, and delete feature combinations in downstream tasks
2. GUI to show per-feature checkboxes (with param editing) for the extraction task
3. Dynamic heatmap/matrix column resolution based on available kinematic aggregations
4. One heatmap set generated per kinematic aggregation in multi-aggregation combinations
5. DagConfigModel CRUD methods for dict-of-dicts option entries

### Out of Scope
- Adding new feature types or aggregation functions
- Changing clustering/comparing algorithm logic
- GUI for configuring heatmap axis signals (velocity, depth, area are still fixed — only the aggregation suffix is dynamic)
- Drag-and-drop reordering of combinations

## Success Criteria

- [ ] From the GUI, user can create a new feature combination (e.g., `mean_only` with `features: [mean]`), and it appears as a new column in the TaskPanel
- [ ] From the GUI, user can delete an existing feature combination with confirmation
- [ ] From the GUI, user can edit the features list of an existing combination via a checklist dialog
- [ ] Extraction task shows per-feature checkboxes; MoS feature shows "..." button for params
- [ ] Heatmaps are generated when only `mean` features are present, using `velocity_mean`, `depth_mean`, `area_mean` axes
- [ ] Combination `[max, mean]` produces two sets of heatmaps — one per aggregation
- [ ] Combination `[temporal]` only → heatmaps gracefully skipped (no kinematic columns)
- [ ] Combination `[max]` → identical output to current code (backward compat)
- [ ] All GUI changes persist correctly through save/reload cycle

---

## Technical Design

### Approach

**Part A (GUI):** Extend the TaskPanel's column detection and rendering system with two new dict-shape recognizers (`_is_feature_dict`, `_is_feature_combinations_dict`) alongside the existing `_is_profile_dict`. Add a `FeatureCombinationDialog` for create/edit operations. Add CRUD methods to `DagConfigModel`.

**Part B (Heatmaps):** Replace hardcoded column references with dynamic detection of kinematic aggregations present in the DataFrame. Convert `DISCRETIZATION_CONFIG` into a factory function. Loop over detected aggregations to produce one heatmap set per aggregation.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Raw YAML editor for combinations (existing `YamlEditDialog`) | No new widgets needed | Not user-friendly, error-prone, no feature validation | Rejected |
| Dedicated `FeatureCombinationDialog` with checklist | Guided UX, validates feature names, prevents typos | New widget to build | **Chosen** |
| Configurable heatmap axes in DAG YAML | Full user control | Over-engineered for current needs, adds config complexity | Rejected |
| Auto-detect aggregations from DataFrame columns | Dynamic, no config needed, backward compatible | Slightly more logic in matrix_generation | **Chosen** |

### Architecture Changes

#### New detection functions in `task_panel.py`

```python
def _is_feature_dict(val: Any) -> bool:
    """Dict of dicts with 'enabled' but NOT 'method' or 'features' keys."""

def _is_feature_combinations_dict(val: Any) -> bool:
    """Dict of dicts each containing a 'features' key."""
```

Detection priority in `populate()`:
1. `_is_profile_dict()` → `("profile", key, name)` columns (existing)
2. `_is_feature_dict()` → `("feature", key, name)` columns
3. `_is_feature_combinations_dict()` → `("combination", key, name)` + `("combination_add", key)` columns
4. Fallback → `("simple", key)` (existing)

#### New dialog: `FeatureCombinationDialog`

```
+------------------------------------------+
|  Feature Combination - [task_name]       |
+------------------------------------------+
|  Name: [_______________]  (or label)     |
|                                          |
|  Select features:                        |
|  [x] max                                 |
|  [x] mean                                |
|  [ ] min                                 |
|  [ ] median                              |
|  [ ] std                                 |
|  [ ] range                               |
|  [ ] skewness                            |
|  [ ] temporal                            |
|  [ ] mechanics_of_solids                 |
|                                          |
|  [OK]  [Cancel]                          |
+------------------------------------------+
```

Two modes: **Create** (editable name input) and **Edit** (read-only name label).

Feature list sourced from `AGGREGATION_NAMES` and `EXTRACTOR_REGISTRY` in `feature_extraction/__init__.py`.

Validation: name must be non-empty `[a-z0-9_]`, no conflicts; at least one feature selected.

#### DagConfigModel new methods

- `add_combination(task_name, option_key, combo_name, combo_config)` — insert into dict-of-dicts
- `remove_combination(task_name, option_key, combo_name)` — delete entry
- `get_combination_features(task_name, option_key, combo_name) -> list[str]`
- `set_combination_features(task_name, option_key, combo_name, features)` — write `CommentedSeq` with flow style

#### Dynamic heatmap resolution

Replace static `DISCRETIZATION_CONFIG` with factory:
```python
KINEMATIC_SIGNALS = ('depth', 'area', 'velocity', 'acceleration')

def get_discretization_config(aggregation: str) -> dict:
    return {
        'continuous_vars': {
            f'{sig}_{aggregation}': {'method': 'qcut', 'q': 3}
            for sig in KINEMATIC_SIGNALS
        },
        'categorical_vars': ['type_metadata', 'direction'],
    }

# Backward compat
DISCRETIZATION_CONFIG = get_discretization_config('max')
```

In `matrix_generation.py`, detect aggregations from column names:
```python
def _detect_kinematic_aggregations(columns: list[str]) -> list[str]:
    """Return aggregation suffixes found in column names matching {signal}_{agg}."""
```

Loop over detected aggregations, generating one full set of outputs per aggregation.

### Knowledge Base

- `note-qt-itemchanged-signal-recursion.md` — When modifying TaskPanel items programmatically inside event handlers, always guard with `blockSignals(True/False)` to prevent infinite re-entry. Apply this pattern to all new handlers.
- No other notes applicable.

---

## Implementation Plan

### Phase 1: Dynamic Heatmap Columns (amends existing Phase 3)
**Goal:** Make heatmap/matrix generation work with any kinematic aggregation, not just `max`.
**Started:** 2026-03-20 **Completed:** 2026-03-20

- [x] Add `KINEMATIC_SIGNALS` constant and `get_discretization_config(aggregation)` factory to `touch_config.py`
- [x] Keep `DISCRETIZATION_CONFIG` as backward-compat alias
- [x] Add `_detect_kinematic_aggregations(columns)` to `matrix_generation.py`
- [x] Refactor `_generate_matrix_internal()` to loop over detected aggregations
- [x] Parameterize `hierarchy_order` and heatmap axes with aggregation suffix
- [x] Generate outputs in aggregation-specific subdirectories or with filename suffixes for multi-aggregation cases

**Files Modified:**
- `code/src/analysis/touch_analytics/touch_config.py` — factory function, `KINEMATIC_SIGNALS`
- `code/src/analysis/touch_analytics/matrix_generation.py` — dynamic column resolution, aggregation loop

**Dependencies:** None (can proceed independently)

### Phase 2: DagConfigModel CRUD Methods
**Goal:** Add model-layer support for creating, reading, updating, and deleting combination entries.
**Started:** 2026-03-20 **Completed:** 2026-03-20

- [x] Add `add_combination()` method
- [x] Add `remove_combination()` method
- [x] Add `get_combination_features()` method
- [x] Add `set_combination_features()` method with `CommentedSeq` flow-style formatting

**Files Modified:**
- `code/src/utils/pipeline/dag_config_model.py` — four new methods

**Dependencies:** None

### Phase 3: FeatureCombinationDialog
**Goal:** Create the dialog widget for creating and editing feature combinations.
**Started:** 2026-03-20 **Completed:** 2026-03-20

- [x] Create `feature_combination_dialog.py` with create/edit modes
- [x] Implement feature checklist from `AGGREGATION_NAMES` + `EXTRACTOR_REGISTRY`
- [x] Add name validation (format, uniqueness) and feature validation (at least one selected)
- [x] Error label for invalid input (same pattern as `YamlEditDialog`)

**Files Modified:**
- `code/src/utils/gui/dag_launcher/feature_combination_dialog.py` — **NEW**

**Dependencies:** None

### Phase 4: TaskPanel Extension (amends existing Phase 7)
**Goal:** Extend TaskPanel to render feature dicts, combination dicts, and management controls.
**Started:** 2026-03-20 **Completed:** 2026-03-20

- [x] Add `_is_feature_dict()` and `_is_feature_combinations_dict()` detection functions
- [x] Extend column spec with `("feature", ...)`, `("combination", ...)`, `("combination_add", ...)` types
- [x] Implement feature column rendering: checkbox + optional "..." params button
- [x] Implement combination column rendering: checkbox + clickable features-preview label
- [x] Implement "+" column: button that opens `FeatureCombinationDialog` in create mode
- [x] Implement right-click context menu on combination cells for deletion (with `QMessageBox` confirmation)
- [x] Add handler factories: `_make_feature_params_handler`, `_make_combination_edit_handler`, `_make_combination_add_handler`, `_make_combination_delete_handler`
- [x] Re-populate table after structural changes (add/delete)
- [x] Apply `blockSignals` guard pattern per knowledge base note

**Files Modified:**
- `code/src/utils/gui/dag_launcher/task_panel.py` — detection, rendering, handlers, context menu

**Dependencies:** Phase 2, Phase 3

---

## Testing Plan

### Manual Verification

#### Heatmaps
- [ ] Run matrix generation with `[mean]` combination → heatmaps use `velocity_mean` x `depth_mean`, `velocity_mean` x `area_mean`
- [ ] Run with `[max, mean]` → two sets of heatmaps generated (one per aggregation)
- [ ] Run with `[temporal]` only → heatmaps gracefully skipped (no kinematic columns present)
- [ ] Run with `[max]` → identical behavior to current code (backward compat)

#### GUI
- [ ] Open DAG launcher → extraction task shows per-feature checkboxes (max, mean, ..., temporal, mos)
- [ ] MoS checkbox has "..." button → click opens YamlEditDialog with `youngs_modulus_kpa` etc.
- [ ] Toggle feature checkbox → `enabled` state persists to YAML on save
- [ ] Clustering task shows existing combination columns with checkboxes
- [ ] Click features preview label → `FeatureCombinationDialog` opens with correct checkboxes
- [ ] Click "+" button → create `mean_only` with `[mean]` → new column appears
- [ ] Right-click combination cell → Delete → confirmation → column removed
- [ ] Save config → reload → all changes preserved
- [ ] Create combination with invalid name (spaces, uppercase) → error shown
- [ ] Create combination with no features selected → error shown
- [ ] Create combination with duplicate name → error shown

### Edge Cases
- [ ] Task has no `feature_combinations` key → "+" column is N/A (greyed out)
- [ ] All combinations deleted → only "+" column remains for that option key
- [ ] Feature combination references feature not in extraction → works (validation is at runtime, not GUI)

---

## Documentation Plan

- [ ] Update `composable-feature-extraction.md` Phase 3 and Phase 7 to reference this amendment
- [ ] Add inline comments in `FeatureCombinationDialog` documenting create vs edit mode
- [ ] Update DAG config YAML comments to document `feature_combinations` format

---

## Rollback Plan

1. Revert `touch_config.py` to static `DISCRETIZATION_CONFIG`
2. Revert `matrix_generation.py` to hardcoded column names
3. Delete `feature_combination_dialog.py`
4. Revert `task_panel.py` detection and rendering changes
5. Revert `dag_config_model.py` CRUD methods

All changes are in code files with no data migration — rollback is a clean `git revert`.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Detection heuristic false positive on generic dicts | Low | Medium | Three distinct functions with specific key requirements (`method`, `enabled`+no-method, `features`) |
| `itemChanged` signal recursion in new handlers | Medium | High | Apply `blockSignals` guard pattern per knowledge base note |
| Column count explosion with many combinations | Low | Medium | Combinations are typically 2-5 per task; table scrolls horizontally |
| Feature list constant drifts from registry | Low | Low | Import directly from `feature_extraction.__init__` |
| Table rebuild flicker on add/delete | Medium | Low | `blockSignals(True)` during rebuild |

---

## References

- Parent plan: `docs/development/plans/active/composable-feature-extraction.md` (Phases 3 and 7)
- Knowledge base: `docs/development/knowledge-base/note-qt-itemchanged-signal-recursion.md`
- `code/src/utils/gui/dag_launcher/task_panel.py` — TaskPanel widget
- `code/src/utils/gui/dag_launcher/yaml_edit_dialog.py` — pattern for new dialog
- `code/src/utils/pipeline/dag_config_model.py` — round-trip YAML model
- `code/src/analysis/touch_analytics/matrix_generation.py` — heatmap generation
- `code/src/analysis/touch_analytics/touch_config.py` — discretization config
- `code/src/analysis/touch_analytics/feature_extraction/__init__.py` — `AGGREGATION_NAMES`, `EXTRACTOR_REGISTRY`
