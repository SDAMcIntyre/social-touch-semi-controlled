# Plan: Fix Postprocess Visualization CSV Filename

**Created:** 2026-03-13
**Approved:** —
**Completed:** —
**Author:** Basil Duvernoy
**Status:** Draft
**Branch:** `feature/postprocessing-visualization-pipeline`

---

## Overview

**What:** Fix the `contact_projected_csv` path in the postprocessed viewer workflow script so it resolves to the actual file produced by the pipeline.

**Why:** The viewer aborts with `Missing required file 'contact_projected_csv'` because the filename was constructed without the `_pca-xyz` suffix that the `set_xyz_reference_from_gestures` stage appends.

**How:** Change one string in `resolve_postprocessed_paths()` — replace `_merged_data.csv` with `_merged_data_pca-xyz.csv`.

## Problem Statement

`postprocess_visualization.py::resolve_postprocessed_paths()` builds the contact-projected CSV path as:

```
blocks_contact_projected/{session_id}_semicontrolled_{block_id}_merged_data.csv
```

But the actual pipeline filename chain is:

| Stage | Output filename |
|-------|----------------|
| 1 apply_icp_registration | `…_merged_data.csv` |
| 2 set_xyz_reference_from_gestures | `…_merged_data_pca-xyz.csv` ← suffix added here |
| 4 project_contacts_onto_forearm | `…_merged_data_pca-xyz.csv` ← preserved (output name = input name) |

The file the viewer needs is the stage 4 output, which retains the `_pca-xyz` suffix.

## Goals

### In Scope
1. Fix the `merged_name` variable in `resolve_postprocessed_paths()` to use the correct `_pca-xyz.csv` suffix.

### Out of Scope
- Changes to the postprocessing pipeline itself
- Changes to any other path in `resolve_postprocessed_paths()` (`forearm_pca_ply`, `pca_calib_json` are already correct)

## Success Criteria

- [ ] Running `view_postprocessed_simple` no longer logs `Missing required file 'contact_projected_csv'`
- [ ] The viewer launches and renders the forearm PLY with animated contact points

---

## Technical Design

### Approach

Single-string fix in `resolve_postprocessed_paths()`. No architectural changes.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Fix the string in `resolve_postprocessed_paths` | Minimal, targeted | None | **Chosen** |
| Scan `blocks_contact_projected/` for any CSV matching a partial name | Robust to future suffix changes | Fragile if multiple files match | Rejected |

### Architecture Changes

None — only one line in the workflow script changes.

---

## Implementation Plan

### Phase 1: Fix filename
**Goal:** Correct the `merged_name` suffix

- [ ] In `code/scripts/postprocess_visualization.py`, change `resolve_postprocessed_paths()`:

```python
# Before
merged_name = (
    f"{config.session_id}_semicontrolled_{config.block_id}_merged_data.csv"
)

# After
merged_name = (
    f"{config.session_id}_semicontrolled_{config.block_id}_merged_data_pca-xyz.csv"
)
```

**Files Modified:**
- `code/scripts/postprocess_visualization.py` — fix `merged_name` suffix

**Dependencies:** None

---

## Testing Plan

### Manual Verification
- [ ] Run `python code/scripts/postprocess_visualization.py --dag-config configs/postprocess_visualization_dag.yaml` with `view_postprocessed_simple: enabled: true`
- [ ] Viewer launches without "Missing required file" error
- [ ] Forearm PLY renders as static surface; contact points animate per frame

### Edge Cases
- [ ] Session where `blocks_contact_projected/` does not yet exist → should still print a clear "Missing required file" message (existing path-existence check handles this)

---

## Documentation Plan

- [ ] No documentation changes needed (internal pipeline fix, no public API change)

---

## Rollback Plan

1. Revert the one-line change in `postprocess_visualization.py`.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| A session was processed with a different suffix convention | Low | Med | Check actual files on disk for one session before running batch |

---

## References

- Active feature: `docs/development/plans/active/postprocessing-visualization-pipeline.md`
- Pipeline stage: `code/scripts/_5_postprocessing/set_xyz_reference_from_gestures.py` (suffix defined at line 41: `output_csv_suffix = "_pca-xyz.csv"`)
- Pipeline stage: `code/scripts/_5_postprocessing/project_contacts_onto_forearm.py` (output name = input name, line 159: `output_csv = output_dir / input_csv.name`)
