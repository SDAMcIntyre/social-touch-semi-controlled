# Plan: RF Explorer — Multi-Contact-Point Vertex Mapping

**Created:** 2026-05-03 00:00
**Approved:** —
**Completed:** —
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `feature/rf-explorer-multi-contact-vertex-mapping`

---

## Overview

**What:** Replace the single-centroid vertex mapping in the RF Feature-Space
Explorer with a per-contact-point mapping that uses every raw contact point
stored in the `contact_points` CSV column.

**Why:** The current loader reads `contact_location_x/y/z` (one aggregated
centroid per frame) and maps each frame to a single forearm vertex. The raw
`contact_points` column holds all individual contact points detected per
Kinect frame (format: `[[x1 y1 z1] [x2 y2 z2] …]`), which gives a more
accurate spatial representation of where contact occurred — especially for
larger-area strokes where multiple points are detected simultaneously.

**How:** Extend `ExplorerData` with two new parallel arrays (`cp_frame_idx`,
`cp_vertex_idx`) that expand the per-frame contact data to per-contact-point
granularity. The 3D heatmap accumulation in `_render_3d` and
`_apply_filter_update` indexes these arrays instead of `frame_vertex_idx`.
The scatter plot and frame-count label remain source-frame granularity.

---

## Problem Statement

`load_explorer_data` currently does:

```python
contact_pts = df[["contact_location_x", "contact_location_y",
                   "contact_location_z"]].to_numpy()
…
distances, frame_vertex_idx = tree.query(rotated_contacts)
```

One KDTree query per frame yields one vertex per frame.  The spike for that
frame is then credited to that single vertex.  When a stroke covers several
square centimetres of skin, only the centroid vertex receives the spike credit
— the surrounding vertices on the actual contact patch receive nothing.
`contact_points` already holds the full set of contact points per frame; it
is simply not being read.

---

## Goals

### In Scope

1. Parse `contact_points` column via the existing `parse_contact_points`
   utility and map every individual contact point to its nearest forearm
   vertex.
2. Replace `frame_vertex_idx` in `ExplorerData` with `cp_frame_idx` /
   `cp_vertex_idx` so that the heatmap accumulation reflects all contact
   points, not just the centroid.
3. Update the sidecar `.npz` cache to store and load the new arrays.
4. Keep source-frame arrays (`pressure`, `velocity_signed`, `gesture_types`,
   `spikes`) at frame granularity — the scatter and frame-count label are
   unchanged.

### Out of Scope

- Changes to the scatter plot axes or visual appearance.
- Per-contact-point weighting (all contact points for a frame receive the
  same spike weight as today).
- Changes to any other consumer of `ExplorerData` (`rf_cluster_pipeline.py`
  passes the object through to the GUI without inspecting its fields).
- Backfill / migration of existing `.npz` caches (stale caches are handled by
  cache-key detection; old caches recompute automatically on next load).

---

## Success Criteria

- [ ] Explorer launches and renders the 3D heatmap correctly for all sessions.
- [ ] For a frame with N contact points, exactly N vertex contributions are
  accumulated (verifiable via `diagnose_contact_points.py` point counts vs.
  heatmap density).
- [ ] Scatter plot and frame-count label are unchanged in behaviour.
- [ ] Checkbox toggle and filter-rectangle drag update the heatmap correctly.
- [ ] Session switching resets all state correctly (no stale arrays).
- [ ] Old `.npz` caches (lacking `cp_frame_idx`) are detected and silently
  recomputed on first load; a fresh cache is written in the new format.
- [ ] Frames whose contact points all exceed the 15 mm distance threshold are
  excluded from the scatter (consistent with current behaviour); individual
  bad contact points within a frame are dropped with a warning.

---

## Technical Design

### Approach

**Data model change — `ExplorerData`:**

Replace the single `frame_vertex_idx: np.ndarray` (shape `(n_frames,)`) with
two parallel contact-point–level arrays:

| Field | Shape | Meaning |
|-------|-------|---------|
| `cp_frame_idx` | `(n_contact_pts,)` int64 | Index into source-frame arrays for each contact point |
| `cp_vertex_idx` | `(n_contact_pts,)` int64 | Nearest forearm vertex index for each contact point |

`n_contact_pts ≥ n_frames` (each frame contributes ≥ 1 contact point).

**Loading — `load_explorer_data`:**

1. Parse `contact_points` column row by row using `parse_contact_points`.
2. Frames with zero parsed points are excluded entirely (equivalent to the
   current `contact_location_x.notna()` filter).
3. Flatten all contact points into `(N_total, 3)` array; build a parallel
   `row_idx` array mapping each point back to its source row index.
4. Rotate all points via `compute_tangent_plane_rotation`.
5. Single batch KDTree query → `distances`, `vertex_idx`.
6. Drop individual contact points exceeding 15 mm (warn); after dropping, any
   source frame that has zero surviving contact points is also removed from
   the source-frame arrays and from `cp_frame_idx`.
7. Re-index `cp_frame_idx` to be contiguous (0-based) after dropping frames.

**Heatmap accumulation — `_render_3d` and `_apply_filter_update`:**

```python
# _render_3d (full session baseline)
cp_spikes = self._data.spikes[self._data.cp_frame_idx]
spike_density = np.bincount(
    self._data.cp_vertex_idx,
    weights=cp_spikes.astype(float),
    minlength=n_verts,
)

# _apply_filter_update (filter-rectangle + checkbox)
frame_mask = rect_mask & type_mask          # (n_frames,)
cp_mask = frame_mask[self._data.cp_frame_idx]  # (n_contact_pts,)
cp_spikes = self._data.spikes[self._data.cp_frame_idx]
vertex_spikes = np.bincount(
    self._data.cp_vertex_idx[cp_mask],
    weights=cp_spikes[cp_mask].astype(float),
    minlength=n_verts,
)
```

**Cache versioning:**

The sidecar filename stays `<stem>_explorer_cache.npz`.  The load function
checks whether `"cp_frame_idx"` is a key in the loaded archive; if not, it
returns `None` (cache miss → recompute + write new cache).  No separate
version field is needed.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Separate cp_frame_idx / cp_vertex_idx arrays** | Clean split between frame-level scatter and point-level heatmap; bincount still 1D; n_frames stays meaningful | Slightly more loading complexity | **Chosen** |
| **Ragged list-of-arrays for frame_vertex_idx** | Preserves frame-indexed structure | Cannot use np.bincount directly; requires explicit loop or object array | Rejected — performance regression |
| **Expand all source-frame arrays to contact-point granularity** | Single uniform array length | `n_frames` becomes meaningless; scatter would plot duplicate points; frame-count label wrong | Rejected — breaks scatter semantics |
| **Keep using contact_location_x/y/z centroid** | No change needed | Loses spatial accuracy of multi-point contact | Rejected — motivation of this plan |

### Architecture Changes

**Modified files:**

- `code/src/analysis/receptive_field_mapping/rf_explorer_data.py`
  - `ExplorerData`: remove `frame_vertex_idx`, add `cp_frame_idx` and
    `cp_vertex_idx`
  - `load_explorer_data`: parse `contact_points` column; batch KDTree; build
    two arrays; frame-level bad-distance exclusion
  - `_save_explorer_cache`: save `cp_frame_idx`, `cp_vertex_idx` instead of
    `frame_vertex_idx`
  - `_load_explorer_cache`: validate new keys; return `None` on cache miss
    (absent `cp_frame_idx`)

- `code/src/analysis/receptive_field_mapping/gui/rf_feature_space_explorer.py`
  - `_render_3d`: expand spikes via `cp_frame_idx` before `bincount`
  - `_apply_filter_update`: expand `frame_mask` via `cp_frame_idx` before
    `bincount`

**No changes needed:**

- `rf_cluster_pipeline.py` — passes `ExplorerData` to GUI without field access
- `rf_data_loader.py` — `parse_contact_points` already exists; used as-is
- All other pipeline modules — do not consume `ExplorerData`

---

## Implementation Plan

### Phase 1: Data model and loader
**Goal:** Update `ExplorerData` and `load_explorer_data` to produce
`cp_frame_idx` / `cp_vertex_idx` from the `contact_points` column.
**Started:** 2026-05-03
**Completed:** 2026-05-03

- [x] Task 1.1 — `ExplorerData`: replace `frame_vertex_idx` with
  `cp_frame_idx: np.ndarray` and `cp_vertex_idx: np.ndarray`; update
  docstring
- [x] Task 1.2 — `load_explorer_data`: replace `contact_location_x/y/z` read
  with `parse_contact_points` loop; build flat `all_pts` and `cp_src_row`
  arrays
- [x] Task 1.3 — `load_explorer_data`: single batch KDTree query on
  `rotated_contacts`; apply 15 mm distance threshold per contact point
  (warn); drop source frames with zero surviving points; re-index
  `cp_frame_idx` to be contiguous
- [x] Task 1.4 — `_save_explorer_cache`: write `cp_frame_idx`, `cp_vertex_idx`
  instead of `frame_vertex_idx`
- [x] Task 1.5 — `_load_explorer_cache`: add `"cp_frame_idx" not in npz →
  return None` guard; update shape validation for new arrays

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_explorer_data.py`

**Dependencies:** None

### Phase 2: GUI heatmap accumulation
**Goal:** Update `_render_3d` and `_apply_filter_update` to use the new
contact-point arrays.
**Started:** 2026-05-03
**Completed:** 2026-05-03

- [x] Task 2.1 — `_render_3d`: replace `self._data.frame_vertex_idx` usage
  with `cp_frame_idx` + `cp_vertex_idx` expansion pattern
- [x] Task 2.2 — `_apply_filter_update`: build `frame_mask` as before, then
  expand via `cp_frame_idx` to `cp_mask` before `bincount`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/rf_feature_space_explorer.py`

**Dependencies:** Phase 1

---

## Testing Plan

### Manual Verification

- [ ] Launch explorer, confirm 3D heatmap renders without crash for at least
  two sessions
- [ ] Run `diagnose_contact_points.py` on one series-augmented CSV; verify
  total point count matches the expanded `cp_vertex_idx` length (logged via
  `len(data.cp_vertex_idx)` or a debug print)
- [ ] Drag filter rectangle — heatmap updates correctly
- [ ] Toggle gesture-type checkboxes including uncheck-all / re-check-all
- [ ] Switch sessions — no crash, heatmap resets correctly
- [ ] Delete an existing `_explorer_cache.npz`, reopen explorer — cache is
  rebuilt in new format without error
- [ ] Open explorer with an existing old-format `_explorer_cache.npz` — it is
  detected as stale and silently recomputed

### Edge Cases

- [ ] Session where some frames have `contact_points = "[]"` or NaN —
  those frames are excluded, others load normally
- [ ] Session where every contact point exceeds 15 mm — `ValueError` raised
  (fail-fast; no empty heatmap silently rendered)
- [ ] Session with only single-point frames (`contact_points` count = 1
  everywhere) — behaviour identical to current centroid approach
- [ ] Session with a frame having 10+ contact points — all points mapped,
  no crash

---

## Documentation Plan

- [ ] Update `note-rf-feature-space-explorer-gui-components.md` — reflect
  `cp_frame_idx` / `cp_vertex_idx` replacing `frame_vertex_idx`

---

## Rollback Plan

All changes are confined to two files:

1. Revert `rf_explorer_data.py` — restores centroid-based loading and
   `frame_vertex_idx`
2. Revert `rf_feature_space_explorer.py` — restores original bincount calls

Old-format `.npz` caches (if any were overwritten) will recompute
automatically on first load of the reverted code, because the reverted loader
will not find `frame_vertex_idx` in a new-format cache and will fall back to
recomputing (the stale-cache guard works symmetrically in both directions).

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `contact_points` column absent from older prepared CSVs | Low | High | `load_explorer_data` raises `ValueError` immediately (fail-fast); do not fall back to `contact_location_x/y/z` silently |
| High contact-point count per frame degrades load time | Low | Low | Single batch KDTree query — cost scales with total points, not per-frame; expect <1 s for a full session |
| Re-indexing `cp_frame_idx` after dropping bad frames introduces off-by-one | Medium | High | Unit-test the remapping: verify `data.spikes[data.cp_frame_idx]` aligns with manually constructed expected array |
| Old-format cache not detected → stale data served | Low | High | Cache-miss guard checks for `"cp_frame_idx"` key; any old cache lacking it is recomputed |

---

## References

- Active parent plan: `docs/development/plans/active/rf-explorer-event-and-scalar-fix.md`
- `parse_contact_points`: `code/src/analysis/receptive_field_mapping/rf_data_loader.py:118`
- `diagnose_contact_points.py`: `code/scripts/diagnose_contact_points.py`
- `ExplorerData`: `code/src/analysis/receptive_field_mapping/rf_explorer_data.py:25`
