# Plan: Remove `_transformed` Column Dead Code

**Date:** 2026-03-16
**Author:** Basil Duvernoy
**Status:** Draft
**Branch:** `chore/clean-touch-analytics-dead-code` (extends existing work)

---

## Overview

Remove all dead code related to `_transformed` column names: the `resolve_column()` helper,
the `use_transformed` parameters throughout the analysis layer, the legacy
`transform_unified_csv()` function, and their exports. This completes the migration to
in-place spatial transforms that was started in the `fix-analysis-pipeline-use-transformed-and-forearm-rf` plan.

## Problem Statement

The codebase had a legacy pattern where `transform_unified_csv()` created
`*_transformed` columns alongside originals (e.g. `contact_location_x_transformed`).
The newer postprocessing pipeline uses `transform_spatial_columns_in_place()` which
overwrites originals — no `_transformed` columns exist in modern output.

A prior plan (`fix-analysis-pipeline-use-transformed-and-forearm-rf`) hardcoded
`use_transformed=False` at call sites but explicitly deferred cleaning the downstream
function signatures. This leaves behind:

1. `resolve_column()` — called with `use_transformed=False` everywhere, making it a
   no-op wrapper that just returns `base`
2. `use_transformed` parameters in 3 function signatures — always `False`
3. `transform_unified_csv()` — creates `*_transformed` columns that no downstream code reads
4. Dead option forwarding in `postprocess_workflow_kinect_auto.py` for a removed DAG key

## Goals

### In Scope
1. Remove `resolve_column()` usage and function definition
2. Remove `use_transformed` parameter from all function signatures and call sites
3. Replace `transform_unified_csv()` call with `transform_spatial_columns_in_place()` in preprocessing step 3
4. Delete `transform_unified_csv()` and `resolve_column()` from source and all export chains
5. Remove dead `use_transformed` option forwarding in postprocess workflow

### Out of Scope
- Changing the transform math or spatial column names
- Removing `transform_spatial_columns_in_place()` or `transform_spatial_columns_scheduled()`
- Adding unit tests (no test infrastructure exists for this package)

## Success Criteria

- [ ] No Python file references `resolve_column` or `transform_unified_csv`
- [ ] No Python file in `code/src/` or `code/scripts/` references `use_transformed`
- [ ] `apply_registration_transform.py` uses `transform_spatial_columns_in_place()` and produces CSVs with in-place transformed columns (no `*_transformed` suffix columns)
- [ ] All modified files pass `python -m py_compile`

---

## Technical Design

### Approach

Direct deletion and replacement — no new modules, no architectural changes. The in-place
transform function (`transform_spatial_columns_in_place`) is a drop-in replacement since
it uses the same `_XYZ_GROUPS`, `apply_rigid_transform()`, and `contact_points` parsing
as the legacy `transform_unified_csv`.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Delete `transform_unified_csv`, replace with in-place | Clean, completes migration | Preprocessing step 3 output loses `_transformed` columns | **Chosen** — those columns are never consumed |
| Keep `transform_unified_csv` but stop adding `_transformed` suffix | Less invasive | Maintains a near-duplicate of `transform_spatial_columns_in_place` | Rejected |

### Architecture Changes

No new modules or classes. Only deletions and simplifications of existing code.

### Knowledge Base Constraints

No applicable knowledge base notes.

---

## Implementation Plan

### Phase 1: Remove `resolve_column()` usage and `use_transformed` parameter
**Goal:** Eliminate all calls to `resolve_column()`, drop the dead `use_transformed` parameter from all signatures and call sites.

**Tasks:**
- [ ] Task 1.1 — **`touch_analysis.py`**: Delete `resolve_column` import (line 10). Remove `_LOCATION_BASES`/`loc_cols` dict (lines 48-50). Use string literals `'contact_location_x'`/`y`/`z` directly at lines 95-97. Remove `use_transformed` from `generate_unified_summary()` signature (line 17), forwarding call (line 35), and `_process_touch_analysis()` signature (line 37)
- [ ] Task 1.2 — **`rf_data_loader.py`**: Delete `resolve_column` import (lines 18-20). Replace `resolve_column(header_df, col.points, use_transformed)` (line 231) with `col.points`. Remove `use_transformed` from `load_grouped_spatial_data()` signature (line 153) and its docstring (lines 173-175)
- [ ] Task 1.3 — **`analysis_workflow.py`**: Remove `use_transformed=False` from call sites (lines 113, 271)
- [ ] Task 1.4 — **`postprocess_workflow_kinect_auto.py`**: Delete dead `if 'use_transformed' in options:` block (lines 243-244)

**Files Modified:**
- `code/src/analysis/touch_analytics/touch_analysis.py` — Remove import, resolve_column usage, use_transformed parameter
- `code/src/analysis/receptive_field_mapping/rf_data_loader.py` — Remove import, resolve_column usage, use_transformed parameter
- `code/scripts/analysis_workflow.py` — Remove use_transformed=False from call sites
- `code/scripts/postprocess_workflow_kinect_auto.py` — Remove dead option forwarding

**Dependencies:** None

### Phase 2: Replace `transform_unified_csv()` and delete dead functions
**Goal:** Switch preprocessing step 3 to the in-place function, then delete the legacy function and `resolve_column()` from source and all export chains.

**Tasks:**
- [ ] Task 2.1 — **`apply_registration_transform.py`**: Change import from `transform_unified_csv` to `transform_spatial_columns_in_place`. Add `import numpy as np` and `import pandas as pd`. Replace line 83 with: read CSV, `np.asarray(transform_4x4)`, call `transform_spatial_columns_in_place(df, T)`, write CSV
- [ ] Task 2.2 — **`csv_spatial_transformer.py`**: Delete `resolve_column()` (lines 95-114) and `transform_unified_csv()` (lines 398-462). Update module docstring (lines 1-7) to remove mentions of legacy function and resolve_column
- [ ] Task 2.3 — **`registration/__init__.py`**: Remove `resolve_column` and `transform_unified_csv` from imports and `__all__`. Update module docstring
- [ ] Task 2.4 — **`forearm_extraction/__init__.py`**: Remove `resolve_column` and `transform_unified_csv` from imports and `__all__`

**Files Modified:**
- `code/scripts/_3_preprocessing/_3_forearm_extraction/apply_registration_transform.py` — Replace legacy call with in-place transform
- `code/src/preprocessing/forearm_extraction/registration/csv_spatial_transformer.py` — Delete two functions, update docstring
- `code/src/preprocessing/forearm_extraction/registration/__init__.py` — Clean exports
- `code/src/preprocessing/forearm_extraction/__init__.py` — Clean exports

**Dependencies:** Phase 1 (all consumers must be gone before deleting the functions)

---

## Commit Strategy

**Commit A (Phase 1):** `chore(analysis): remove resolve_column and use_transformed dead code`

**Commit B (Phase 2):** `chore(registration): replace legacy transform_unified_csv with in-place transform`

---

## Testing Plan

### Unit Tests
- [ ] No unit test infrastructure exists for these packages — compilation checks only

### Integration Tests
- [ ] `python -m py_compile` on every modified file (8 files total)

### Manual Verification
- [ ] `grep -r "resolve_column" code/` returns zero Python hits
- [ ] `grep -r "transform_unified_csv" code/` returns zero Python hits
- [ ] `grep -r "use_transformed" code/src/ code/scripts/` returns zero hits
- [ ] Run preprocessing step 3 on one session — output CSV has base spatial columns with transformed values, no `*_transformed` suffix columns
- [ ] Run analysis workflow on one session — identical output (was already using `use_transformed=False`)

### Edge Cases
- [ ] Single-forearm session (no transforms file) — `apply_registration_transform.py` returns early before reaching the modified code, so no change in behavior

---

## Documentation Plan

- [ ] Update the existing active plan at `docs/development/plans/active/clean-touch-analytics-dead-code.md` to reference this expanded scope
- [ ] No external documentation changes needed (internal cleanup)

---

## Rollback Plan

1. `git revert <commit>` — changes span only analysis and registration files
2. No data migrations, no schema changes, no external state affected

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Old notebooks read `*_transformed` columns from step-3 CSVs | Low | Low | Those columns were already absent from postprocessing output |
| `transform_spatial_columns_in_place` differs from legacy function | Very Low | Medium | Same `_XYZ_GROUPS` and `apply_rigid_transform`; in-place version also handles `sticker_*` columns |
| External import of `resolve_column` | Very Low | Low | Grep confirms only 2 consumers, both addressed in Phase 1 |

---

## References

- Prior plan: `docs/development/plans/completed/fix-analysis-pipeline-use-transformed-and-forearm-rf.md`
- Active branch plan: `docs/development/plans/active/clean-touch-analytics-dead-code.md`
- Legacy function: `code/src/preprocessing/forearm_extraction/registration/csv_spatial_transformer.py` lines 398-462
- In-place replacement: `code/src/preprocessing/forearm_extraction/registration/csv_spatial_transformer.py` lines 265-312
- Preprocessing caller: `code/scripts/_3_preprocessing/_3_forearm_extraction/apply_registration_transform.py`
