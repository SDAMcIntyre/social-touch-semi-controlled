# Plan: Fix Contact Detection False Positives (Winding + Reference Forearm)

**Date:** 2026-05-11
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/hand-model-overlay-exporter`
**Branch:** `fix/contact-detection-false-positives`

---

## Overview

At frame 502 of `2022-06-16_ST15-01_semicontrolled_block-order04_kinect`, the
Neural-Kinect Viewer shows two contact-point clusters (one per finger) even
though the MANO handmesh is visually NOT touching the reference forearm PLY.
A thorough investigation identified two independent root causes: (H1) negative
scale values produced by the `scaled_procrustes` alignment flip the MANO mesh
winding order, inverting the Open3D signed-distance sign and creating false
positive contacts; (H2) the reference forearm fed to the contact detector may
be a stale PLY from a preceding block when the current block has no own
snapshot.

---

## Problem Statement

### H1 — Negative scale inverts MANO mesh winding

`HandMotionManager._calculate_alignment_procrustes` computes:

```
scale = Σ(P_rotated · Q) / Σ(P · P)
```

The denominator (sum of squares) is always positive. The numerator can be
negative for degenerate sticker frames (collinear stickers, tracking noise,
re-labelling flip), producing `scale < 0`.

In `__getitem__` the rotation block is then multiplied by this scalar:

```python
matrix[:3, :3] *= scale   # scale < 0 → det(matrix) < 0 → winding FLIPPED
mesh.transform(matrix)
```

Open3D's `RaycastingScene.compute_signed_distance` determines interior/exterior
via winding-number ray casting. With inverted winding the "inside" and "outside"
swap:

- Forearm vertices **outside** the hand → `signed_dist < 0` (falsely "inside")
- `inside_mask = all(dist < 1e-5)` triggers on triangles nowhere near the hand
- **False positive contact_points generated at every affected frame**

### H2 — Stale reference forearm from preceding block

`get_forearms_with_fallback` walks backwards through earlier blocks when
block-order04 has no own PLY snapshot. If the participant repositioned their arm
between blocks, the stale PLY no longer represents the actual arm position, and
the contact detector silently operates on the wrong surface. The viewer displays
the same stale PLY as "reference", so the hand may appear far from the PLY at a
frame where contact was—or was not—genuinely detected.

### Impact

Both issues corrupt `contact_points`, `contact_location_x/y/z`,
`contact_detected`, `contact_area`, and `contact_depth` in the somatosensory
CSV. Downstream RF-mapping and neural-touch analyses ingest these columns
directly; false positives inflate contact area and distort receptive-field maps.

---

## Goals

### In Scope

1. Diagnose and guard against negative scale in `_calculate_alignment_procrustes`
   so MANO mesh winding is always preserved.
2. Detect and surface (via loud warning / exception) when a stale cross-block
   fallback forearm is used, so the operator is aware before running contact
   detection.
3. Add a lightweight diagnostic utility to inspect the scales array of any NPZ
   file and flag suspicious frames.

### Out of Scope

- Reprocessing historical session data (operator decision after fix lands).
- Fixing the cubic-interpolation overshoot in RF Explorer (tracked separately in
  `bug-rf-explorer-nearest-vertex-distance.md`).
- Changing the contact detection algorithm (EPSILON, penetration model).
- GUI changes to the Neural-Kinect Viewer beyond what is needed to verify the fix.

---

## Success Criteria

- [x] `_calculate_alignment_procrustes` never produces `scale ≤ 0`; any such
      frame raises an explicit warning and falls back to the previous frame's
      scale (fail-fast: raises if no previous frame exists either).
- [x] `get_forearms_with_fallback` logs a visible WARNING (via `logging.warning`)
      when a cross-block fallback forearm is used, printing the source block and
      frame ID so the operator can decide whether to capture a new PLY.
- [x] A diagnostic script can be invoked to inspect any NPZ and report
      `(frame_id, scale)` for all frames where `scale < 0` or `abs(scale) > 3.0`
      (suspiciously large/inverted).
- [ ] At frame 502 of ST15-01 block-order04, the Neural-Kinect Viewer no longer
      shows spurious contact_points when the handmesh is visually detached from
      the reference PLY — **after the somatosensory CSV is reprocessed with the
      fix**.
- [ ] `pytest code/tests/` passes without new failures.

---

## Technical Design

### Approach

**H1 fix — clamp scale to positive in `_calculate_alignment_procrustes`**

After computing `scale = numerator / (denominator + 1e-8)`:
1. If `scale ≤ 0`: log a warning with the frame timestamp, then use the previous
   frame's scale (held via `HandMotionManager.process_frame` state). If no
   previous scale exists, raise `ValueError` (first frame cannot be degenerate).
2. Additionally clamp to a plausible range, e.g. `[0.3, 5.0]`, with a warning
   for out-of-range values, to catch erroneous magnitudes.

This is a pure arithmetic guard with zero change to the alignment strategy or
data format. The NPZ serialisation is unchanged.

**H2 fix — warn on cross-block fallback in `get_forearms_with_fallback`**

After step 2a (walk backwards to insert a key-0 entry from a preceding block),
emit `logging.warning(...)` with the source video filename, source block, and
the current block so the operator can react.

No logic change — fallback behaviour is preserved; only observability improves.

**Diagnostic script**

`code/scripts/__misc/inspect_handmodel_scales.py` — standalone, reads any NPZ
and prints a table of suspicious frames.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Clamp scale to `[0.3, 5.0]` with warning | Simple; preserves physical meaning | Silent clamp may mask upstream tracking bug | Chosen — with explicit warning |
| `abs(scale)` unconditionally | Simplest code | Silently accepts mirror-reflected frames, wrong semantics | Rejected |
| Raise hard error on negative scale | Fail-fast | Stops entire session processing for noisy frames | Rejected — too disruptive for a single bad frame |
| Flip column of R when `scale < 0` | Geometrically correct mirror fix | Complex; mirrors are not physically meaningful for hand tracking | Rejected — hand-tracking geometry should not produce mirrors |

### Architecture Changes

Only `hand_motion_manager.py` and `forearm_catalog.py` are modified. Both are
internal to the preprocessing package. No public API change.

```
code/src/preprocessing/motion_analysis/hand_tracking/models/hand_motion_manager.py
  _calculate_alignment_procrustes()  ← add scale clamp + warning

code/src/preprocessing/forearm_extraction/models/forearm_catalog.py
  get_forearms_with_fallback()       ← add cross-block warning

code/scripts/__misc/inspect_handmodel_scales.py  ← new diagnostic
```

---

## Implementation Plan

### Phase 1: Diagnose the specific session

**Goal:** Confirm which hypotheses are active for ST15-01 block-order04 before
touching production code.

- [ ] 1.1 — Load `*_handmodel_motion.npz` for ST15-01 block-order04, inspect
      `data["scales"]`. Record min, max, negative-frame indices.
- [ ] 1.2 — Inspect `*_arm_roi_metadata.json` for ST15-01. List all forearm
      entries; confirm whether any belong to block-order04.
- [ ] 1.3 — Compare row count of somatosensory CSV (`*_contact_and_kinematic_data.csv`)
      against `len(kinect_rows)` in merged CSV. Record any mismatch.

**Files Modified:** None (read-only diagnostic)

**Dependencies:** None

---

### Phase 2: Fix negative-scale winding bug (H1)

**Goal:** Prevent MANO mesh winding inversion for any future processing.

**Started:** 2026-05-11
**Completed:** 2026-05-11

- [x] 2.1 — In `_calculate_alignment_procrustes` (line ~315 of
      `hand_motion_manager.py`), after computing `scale`:
      ```python
      _SCALE_MIN, _SCALE_MAX = 0.3, 5.0
      if scale <= 0:
          logging.warning(
              "Procrustes produced non-positive scale=%.4f at frame %d; "
              "using previous frame scale.", scale, len(self.scales)
          )
          scale = self.scales[-1] if self.scales else None
          if scale is None:
              raise ValueError(
                  "First frame has non-positive scale from procrustes alignment. "
                  "Check sticker coordinates."
              )
      elif not (_SCALE_MIN <= scale <= _SCALE_MAX):
          logging.warning(
              "Procrustes scale=%.4f out of plausible range [%.1f, %.1f] "
              "at frame %d.", scale, _SCALE_MIN, _SCALE_MAX, len(self.scales)
          )
      ```
- [x] 2.2 — Verify that `rigid_basis` mode is unaffected (it always returns
      `scale=1.0`; no change needed).
- [x] 2.3 — Write unit test in `code/tests/` confirming that anti-aligned
      sticker input produces a positive (clamped) scale and emits a warning,
      not a negative scale.

**Files Modified:**
- `code/src/preprocessing/motion_analysis/hand_tracking/models/hand_motion_manager.py`
  — scale guard in `_calculate_alignment_procrustes`
- `code/tests/test_hand_motion_manager.py` (create or extend)

**Dependencies:** Phase 1 (confirming H1 is active)

---

### Phase 3: Warn on cross-block forearm fallback (H2)

**Goal:** Surface stale-reference-forearm situations loudly.

**Started:** 2026-05-11
**Completed:** 2026-05-11

- [x] 3.1 — In `get_forearms_with_fallback` (line ~296 of `forearm_catalog.py`),
      after inserting `forearms[0] = geometry` via the 2a walk-back:
      ```python
      logging.warning(
          "No forearm snapshot found for block %d of '%s'. "
          "Using forearm from block %d (frame %d) as fallback. "
          "If the arm repositioned between blocks, capture a new PLY.",
          identifier.block_number,
          current_video_filename,
          prev_block,
          best.representative_frame_id,
      )
      ```
- [ ] 3.2 — Optionally upgrade to `logging.error` if a distance check between
      the fallback PLY centroid and the current Kinect cloud exceeds a threshold
      (deferred — not in this plan).

**Files Modified:**
- `code/src/preprocessing/forearm_extraction/models/forearm_catalog.py`
  — warning in `get_forearms_with_fallback` step 2a

**Dependencies:** None (independent of Phase 2)

---

### Phase 4: Diagnostic script + knowledge-base note

**Goal:** Provide a reproducible diagnostic tool and document the root cause.

**Started:** 2026-05-11
**Completed:** 2026-05-11

- [x] 4.1 — Create `code/scripts/__misc/inspect_handmodel_scales.py`:
      CLI that loads any NPZ, prints per-frame scale statistics, and highlights
      frames where `scale < 0` or `abs(scale) > 3.0`.
- [x] 4.2 — Create `docs/development/knowledge-base/bug-contact-detection-winding-inversion.md`
      documenting the negative-scale mechanism, the session where it was
      discovered, and the fix applied.

**Files Modified:**
- `code/scripts/__misc/inspect_handmodel_scales.py` (new)
- `docs/development/knowledge-base/bug-contact-detection-winding-inversion.md` (new)

**Dependencies:** Phase 2

---

## Testing Plan

### Unit Tests

- [x] `test_procrustes_negative_scale_clamped` — supply degenerate (all-same)
      sticker coordinates to `_calculate_alignment_procrustes`; assert `scale > 0`
      and that a warning was emitted.
- [x] `test_procrustes_normal_scale_unchanged` — normal sticker data; assert
      scale is within `[0.3, 5.0]` and no warning is emitted.
- [x] `test_get_forearms_with_fallback_cross_block_warning` — mock catalog with
      entries only in block-order03; call for block-order04; assert warning
      message is logged.

### Integration Tests

- [ ] Run `compute_somatosensory_characteristics` end-to-end on a short clip of
      ST15-01 block-order04 (frames 490–510). Confirm no false positive contacts
      at frame 502 after fix.

### Manual Verification

- [ ] Open Neural-Kinect Viewer (transformed mode) for ST15-01 block-order04.
      Navigate to frame 502. Confirm contact_points actors are empty or spatially
      co-located with a finger visually touching the forearm PLY.
- [ ] Run `inspect_handmodel_scales.py` on the NPZ; confirm no negative-scale
      frames remain after reprocessing.

### Edge Cases

- [ ] First-frame procrustes produces negative scale → `ValueError` raised
      (no previous scale to fall back to).
- [ ] All stickers collinear → degenerate basis; `rigid_basis` fallback handles
      gracefully (cross-product protection already exists in `_create_basis`).
- [ ] Single-forearm session → `get_forearms_with_fallback` has no cross-block
      walk; no spurious warning emitted.

---

## Documentation Plan

- [x] Create knowledge-base note:
      `docs/development/knowledge-base/bug-contact-detection-winding-inversion.md`
- [ ] No CLAUDE.md changes needed — the fix is an internal arithmetic guard.
- [ ] No user-guide changes needed — diagnostic script is internal tooling.

---

## Rollback Plan

1. Both changes are confined to two source files and one new script.
2. Reverting: `git revert` the fix commits or `git checkout HEAD~N -- <file>`.
3. No data migrations, no schema changes, no CSV format changes.
4. Somatosensory CSVs generated with the broken code should be regenerated
   after the fix; reprocessing is idempotent (`force_processing=False` skips
   frames already processed, so only affected sessions need to be re-run).

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Negative scale is legitimate for some sessions (genuine mirror reflection) | Low | High | Inspect all ST15/ST16 NPZ files before releasing; promote to error only if no session legitimately needs scale < 0 |
| Scale clamp `[0.3, 5.0]` too tight for extreme hand sizes | Low | Med | Choose bounds from empirical range across all sessions; adjust as needed |
| H1 is not the root cause for this session (rigid_basis used, scale always 1.0) | Med | Med | Phase 1 diagnosis confirms before code change; H2 warning is still valuable independently |
| Reprocessing somatosensory CSV takes hours per session | Med | Low | Idempotent pipeline; only re-run affected sessions (confirmed by Phase 1 diagnosis) |

---

## References

- Investigation plan: `.claude/plans/we-must-investigate-from-declarative-meerkat.md`
- Knowledge base: `docs/development/knowledge-base/note-somatosensory-units-and-calculations.md`
- Knowledge base: `docs/development/knowledge-base/note-forearm-icp-registration.md`
- Related bug: `docs/development/knowledge-base/bug-rf-explorer-nearest-vertex-distance.md`
- Key source files:
  - `code/src/preprocessing/motion_analysis/hand_tracking/models/hand_motion_manager.py:283–329`
  - `code/src/preprocessing/motion_analysis/tactile_quantification/model/objects_interaction_processor.py:96–147`
  - `code/src/preprocessing/forearm_extraction/models/forearm_catalog.py:244–325`
