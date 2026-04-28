# Plan: GMM Cluster Gallery — Show Per-Cluster Feature Ranges

**Date:** 2026-04-28
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `feature/gmm-gallery-feature-ranges`

---

## Overview

The RF cluster gallery viewer renders a text overlay and thumbnail tooltip for each cluster via `description_summary_line()`. For binning/kmeans/dbscan/hierarchical this shows meaningful parameters (bin feature + range, k, eps). For GMM clusters the function currently outputs only the bare word `"gmm"` — no component count, no covariance type, no feature names, no per-cluster value ranges.

## Problem Statement

When a GMM run completes (e.g. on `pressure_velocity_mean`), the gallery cannot answer two basic questions from the overlay alone: *which features were clustered?* and *what value ranges does this cluster span?* The data is already computed and stored in `cluster_description.json` (`display_ranges`, reduction `retained_columns`) but is never rendered, because `_format_generation_params()` has no GMM branch and short-circuits before reaching the legacy range formatter.

## Goals

### In Scope
1. Add a GMM branch to `_build_cluster_description()` that writes `k`, `covariance_type`, and `features` (retained column names from reduction metadata) into `generation_params`.
2. Add a GMM branch to `_format_generation_params()` that renders those fields **plus** the already-computed `display_ranges` (per-cluster min/max per feature type).
3. Keep display behaviour for all other algorithms exactly unchanged.

### Out of Scope
- Displaying GMM component means/covariances or BIC scores.
- Changes to how `display_ranges` is computed.
- Adding feature-range display to non-GMM algorithms (kmeans, dbscan, hierarchical).
- New tests for `description_summary_line()` (coverage gap, tracked separately).

## Success Criteria

- [ ] Gallery text overlay for a GMM cluster shows: `gmm — k: N — cov: <type> — features: <col1>, <col2> — <feature_type1>: [min, max] — <feature_type2>: [min, max]`
- [ ] Thumbnail tooltip reflects the same summary line.
- [ ] Overlay for all non-GMM algorithms is visually unchanged.
- [ ] `pytest code/tests/` passes with no regressions.

---

## Technical Design

### Approach

Two small, self-contained changes to `rf_cluster_pipeline.py`:

1. **`_build_cluster_description()`** — inside the `if metadata_json_path` block, add an `elif algo == 'gmm':` branch parallel to the existing binning/kmeans/dbscan/hierarchical branches. Populate `generation['k']`, `generation['covariance_type']`, and `generation['features']` from the already-loaded `cluster_metadata.json`.

2. **`_format_generation_params()`** — add an `elif algo == 'gmm':` branch. Emit `k`, `cov`, `features`, then iterate `desc.get('display_ranges')` to append per-cluster range lines. `display_ranges` is already populated by the existing `_build_cluster_description()` logic (lines 304–327) — no new computation needed.

The `display_ranges` block is scoped to the GMM branch only; all other algorithms are untouched.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Show `display_ranges` for ALL algorithms | Consistent information density | Changes visible output for users of kmeans/dbscan/hierarchical, potentially noisy | Rejected |
| Show raw `feature_ranges` (per-column) instead of `display_ranges` | More granular | Very verbose for multi-column feature types (e.g. hand_velocity x/y/z) | Rejected |
| Show reduction `retained_columns` only (no ranges) | Simple | Doesn't answer "what values does this cluster span?" | Rejected |

### Architecture Changes

No new modules. Single file modified:

```
rf_cluster_pipeline.py
  _build_cluster_description()  ← add elif algo == 'gmm' branch
  _format_generation_params()   ← add elif algo == 'gmm' branch
```

---

## Implementation Plan

### Phase 1: Add GMM generation_params capture

**Goal:** Populate `generation_params` with GMM-specific fields during cluster description build.

- [x] Task 1.1 — In `_build_cluster_description()`, after the `elif algo == 'type_stratified':` block (≈line 396) and before `desc['generation_params'] = generation` (≈line 400), add:

  ```python
  elif algo == 'gmm':
      generation['k'] = meta.get('k')
      generation['covariance_type'] = meta.get('covariance_type')
      retained = meta.get('reduction', {}).get('retained_columns', [])
      if retained:
          generation['features'] = retained
  ```

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` — add `elif algo == 'gmm'` block

**Dependencies:** None

### Phase 2: Add GMM display branch

**Goal:** Render GMM generation_params and display_ranges in the gallery overlay.

- [x] Task 2.1 — In `_format_generation_params()`, after the `elif algo == 'hierarchical':` block (≈line 452) and before the final `else:`, add:

  ```python
  elif algo == 'gmm':
      header = 'gmm'
      if gen.get('k') is not None:
          parts.append(f"k: {gen['k']}")
      if gen.get('covariance_type'):
          parts.append(f"cov: {gen['covariance_type']}")
      if gen.get('features'):
          parts.append(f"features: {', '.join(gen['features'])}")
      dr = desc.get('display_ranges') or {}
      for label, r in dr.items():
          parts.append(f"{label}: [{r['min']}, {r['max']}]")
  ```

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` — add `elif algo == 'gmm'` display block

**Dependencies:** Phase 1

---

## Testing Plan

### Manual Verification
- [ ] Re-run `visualize_receptive_fields_clustered` (or force re-extraction) to regenerate `cluster_description.json` with the new `generation_params` fields.
- [ ] Open the gallery viewer — each GMM cluster overlay must show k, covariance type, feature names, and per-cluster feature ranges.
- [ ] Hover over thumbnails — tooltip summary line must reflect the new info.
- [ ] Open a non-GMM cluster (e.g. binning or kmeans) — confirm overlay is unchanged.

### Regression
- [ ] `pytest code/tests/` — full suite must pass.

### Edge Cases
- [ ] GMM cluster where `display_ranges` is empty (no numeric feature columns) — overlay shows only `gmm — k: N — cov: full — features: ...` without crashing.
- [ ] Old `cluster_description.json` on disk (written before this change, no `features` key in `generation_params`) — `gen.get('features')` returns `None`, branch skips gracefully.

---

## Documentation Plan

- [ ] No public API changes — no docs updates required.

---

## Rollback Plan

Single-file change. To revert: remove the two `elif algo == 'gmm':` blocks added in Phase 1 and Phase 2. No data migration needed — the `cluster_description.json` files on disk will simply have the extra fields ignored by the old code.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `retained_columns` absent from old `cluster_metadata.json` (run before reduction metadata was serialised) | Low | Low | `meta.get('reduction', {}).get('retained_columns', [])` returns `[]`; branch skips `features` key silently |
| `display_ranges` absent from old `cluster_description.json` | Low | Low | `desc.get('display_ranges') or {}` returns `{}`; loop is a no-op |

---

## References

- Discovered during gallery inspection: `feature/gmm-clusterer` branch
- Related file: `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py`
