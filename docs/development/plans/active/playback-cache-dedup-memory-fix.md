# Plan: Playback Cache Dedup Memory Fix

**Date:** 2026-05-06
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/touch-population-explorer-improvements`
**Branch:** `feature/touch-population-explorer-improvements`

---

## Overview

The `_save_playback_cache()` function in `touch_playback_data.py` crashes with
a `MemoryError` when concatenating per-frame contact-point arrays into flat
numpy arrays. For large sessions (358M expanded contact points), peak memory
exceeds 16 GiB. The fix replaces the flat expansion with a deduplicated cache
format that stores each unique contact configuration once and uses per-frame
indices, reducing memory from ~17 GiB to ~10 MB.

## Problem Statement

- `_save_playback_cache` flattens every frame's contact points and vertex
  indices into four parallel flat arrays (`cp_pts_data`, `cp_vertex_data`,
  `cp_pts_frame_touch`, `cp_pts_frame_idx`), each with 358M entries at int64
  or float64 precision.
- The `np.concatenate(vtx_parts)` call at line 143 fails with:
  `Unable to allocate 2.67 GiB for an array with shape (358195153,)`.
- This kills the `map_single_touch_rf` pipeline even though the `PlaybackData`
  result has already been computed — the cache is an optimization, not a
  requirement.
- **Root cause:** At 1 kHz sampling, consecutive frames within a touch event
  share identical `contact_points` strings. The CSV parser (lines 504-507)
  correctly reuses the same ndarray reference for consecutive identical
  strings. But the cache saver iterates every frame and expands all references
  into flat lists, destroying the memory sharing.

## Goals

### In Scope
1. Replace the flat-expansion cache format with a deduplicated format that
   stores unique contact-point groups once
2. Reduce array dtypes (int64 -> int32, float64 -> float32) for cache storage
3. Update the cache loader to reconstruct `TouchEvent` objects from the new
   format
4. Bump `_CACHE_SCHEMA_VERSION` so old caches are automatically recomputed

### Out of Scope
- Changing the in-memory `TouchEvent` / `PlaybackData` data model (the
  deduplication is cache-internal)
- Content-based deduplication across touch events (object-identity dedup via
  `id()` is sufficient since the parser already shares references for
  consecutive identical strings)
- Optimizing the CSV parsing itself (already addressed in prior work, see
  `bug-rf-explorer-contact-parsing-scale.md`)

## Success Criteria

- [ ] `map_single_touch_rf` completes on the session that triggered the crash
      (no `MemoryError`)
- [ ] Cache round-trip produces identical `PlaybackData` (same RF map results)
- [ ] Old schema-v2 caches are detected and recomputed as v3

---

## Technical Design

### Approach

Use object-identity deduplication (`id()`) during cache save. The CSV parser
already reuses the same ndarray object for consecutive identical
`contact_points` strings within a touch event, so `id(vtx_array)` identifies
frames that share identical data. The cache stores:

- **Unique arrays:** Each distinct `(pts, vtx)` pair stored once, indexed by
  group ID.
- **Per-frame mapping:** Each frame with contact points records which group it
  belongs to, plus its touch index and frame-within-touch index.

On load, the mapping is inverted: for each frame, look up its group and slice
the corresponding pts/vtx from the unique arrays.

Dtypes are reduced for storage only (`int32` for indices, `float32` for
coordinates). The loader casts back to the types expected by `TouchEvent`
(`int64`, `float64`).

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Object-identity dedup (`id()`) | Simple, catches dominant pattern (consecutive identical frames within a touch), no content comparison overhead | Misses cross-touch duplicates with identical content but different objects | **Chosen** — sufficient for the problem |
| Content-based dedup (`vtx.tobytes()`) | Catches all duplicates including cross-touch | Extra hashing cost per frame; more complex | Rejected — marginal gain for this data pattern |
| Dtype reduction only (no dedup) | Trivial 1-line changes | Only halves memory (8.5 GiB -> 4.3 GiB); still too large for 358M points | Rejected — insufficient alone |
| Skip cache for large sessions | Zero implementation effort | Loses caching benefit entirely for large sessions | Rejected — defeats the purpose |

### Architecture Changes

No new modules. Single file modified: `touch_playback_data.py`.

**New cache arrays (v3 format):**
```
cp_unique_pts        (total_unique_pts, 3) float32  — unique contact coords
cp_unique_vtx        (total_unique_pts,)   int32    — unique vertex indices
cp_unique_offsets    (n_groups + 1,)       int32    — cumulative counts for slicing
cp_frame_group       (n_contact_frames,)   int32    — group index per frame
cp_frame_touch       (n_contact_frames,)   int32    — touch index per frame
cp_frame_fi          (n_contact_frames,)   int32    — frame-within-touch index
```

**Removed arrays (v2 format):**
```
cp_pts_data          (total_pts, 3) float64
cp_vertex_data       (total_pts,)   int64
cp_pts_frame_touch   (total_pts,)   int64
cp_pts_frame_idx     (total_pts,)   int64
```

---

## Implementation Plan

### Phase 1: Deduplicated cache format
**Goal:** Replace the flat-expansion save/load with a deduplicated format.
**Started:** 2026-05-06
**Completed:** 2026-05-06

- [x] 1.1 Bump `_CACHE_SCHEMA_VERSION` from 2 to 3 (line 19)
- [x] 1.2 Rewrite `_save_playback_cache` contact-point section (lines 119-150):
  - Build `seen: dict[int, int]` mapping `id(vtx_array)` to group index
  - For each frame with `k > 0` contact points: look up or assign group index,
    append `(group_idx, ti, fi)` to per-frame metadata
  - After loop: concatenate unique pts/vtx (small), build offset array,
    build frame mapping arrays
  - Use `float32` for pts, `int32` for all index arrays
- [x] 1.3 Update `np.savez_compressed` call (lines 156-172) to store new array
      names and remove old ones
- [x] 1.4 Rewrite `_load_playback_cache` (lines 179-377):
  - Update `required_keys` to new names
  - Load unique arrays + frame mapping
  - Reconstruct per-frame `frame_contact_pts` (float64) and
    `frame_vertex_indices` (int64) lists by slicing unique arrays via offsets
  - Frames not in the mapping get empty-array defaults
  - Shape validation for new arrays

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/touch_playback_data.py` — rewrite
  `_save_playback_cache` and `_load_playback_cache`, bump schema version

**Dependencies:** None

---

## Testing Plan

### Manual Verification
- [ ] Run `map_single_touch_rf` on the session that triggered the crash —
      confirm no `MemoryError`
- [ ] Delete the `.npz` cache, re-run — confirm cache is created and
      subsequent loads use it (check "cache hit" log line)
- [ ] Verify the single-touch RF pipeline produces identical RF map results
      after round-trip through the new cache

### Edge Cases
- [ ] Session with zero contact points — empty cache arrays, no crash
- [ ] Session with all-identical contact strings — single unique group, all
      frames reference index 0
- [ ] Old schema-v2 cache file exists — loader detects version mismatch,
      recomputes and saves as v3

---

## Documentation Plan

- [ ] Add knowledge-base note if the dedup pattern is reusable elsewhere
- [ ] No CLAUDE.md changes needed (no architectural changes)

---

## Rollback Plan

1. Revert the single commit on `touch_playback_data.py`
2. Delete any v3 `.npz` cache files (they will be regenerated as v2 on next
   run with the old code)
3. No data migration needed — caches are derived artifacts

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `id()` reuse after garbage collection gives false dedup | Low | High (corrupt data) | Arrays are alive in `TouchEvent` objects throughout `_save_playback_cache` — no GC possible during the loop |
| int32 overflow for vertex indices | Very Low | High | Forearm meshes have ~10k-100k vertices; int32 max is 2.1B |
| float32 precision loss in contact coordinates | Low | Low | Coordinates are used for KDTree nearest-vertex lookup, which was already done at parse time; stored coords are for display only |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 | ~1 hour | None |

---

## References

- Related knowledge-base note: `docs/development/knowledge-base/bug-rf-explorer-contact-parsing-scale.md`
- Error traceback: `rf_single_touch_pipeline.py:186 -> touch_playback_data.py:143`
