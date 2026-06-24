# Plan: RF Gallery Persistent Camera Settings

**Date:** 2026-04-30
**Author:** Basil Duvernoy
**Status:** Draft
**Base Branch:** `feature/per-type-clustering-outputs`
**Branch:** `feature/rf-gallery-persistent-camera-settings`

---

## Overview

The RF cluster gallery viewer stores per-session camera state in memory only (`self._session_cameras`), discarding it on close. This means every gallery session resets to a default orientation regardless of prior manual adjustments. This plan adds load-on-init and save-on-close persistence for `_session_cameras` using a JSON sidecar file, following the identical pattern already used by `_session_thresholds` (Delaunay thresholds).

## Problem Statement

- When the gallery viewer closes, all manually adjusted camera angles, zoom levels, and focal points are lost.
- On next launch, every session resets to either `view_xy()` (if tangent rotation exists) or `view_isometric()`.
- There is no mechanism to retain per-session camera orientations across pipeline re-runs.
- Users must manually re-orient each session's view every time they open the gallery.

## Goals

### In Scope
1. Persist `_session_cameras` (position, focal point, up vector, view angle) to `{output_base_dir}/session_cameras.json` on gallery close.
2. Load persisted camera settings at gallery init so prior manual orientations are restored.
3. Capture the currently-displayed session's camera in `closeEvent` before saving — currently only captured on session switch, not on close.

### Out of Scope
- Merging with the auto-computed `camera_params.json` from `rf_camera_angle_task.py` (separate concern).
- Per-cluster-label camera settings (sessions only, matching the existing `_session_cameras` key structure).
- A UI button to reset / clear saved cameras.

## Success Criteria

- [ ] Opening the gallery after manually adjusting a session's camera restores the same orientation.
- [ ] Sessions never previously viewed still fall back to the existing defaults (`view_xy` / `view_isometric`).
- [ ] `session_cameras.json` is written to `output_base_dir` on gallery close.
- [ ] A clean first-launch (no file present) produces no error.
- [ ] The currently-displayed session's camera is correctly captured on close (not just on session switch).

---

## Technical Design

### Approach

Mirror the existing Delaunay threshold persistence pattern exactly — it already solves the same problem for a different per-session value:

| Component | Thresholds (existing) | Cameras (new) |
|-----------|----------------------|---------------|
| I/O helpers | `load_delaunay_thresholds()` / `save_delaunay_thresholds()` in `rf_extraction_io.py` | `load_session_cameras()` / `save_session_cameras()` (same file) |
| Viewer init | `self._session_thresholds = self._load_thresholds()` | `self._session_cameras = self._load_cameras()` |
| Viewer methods | `_load_thresholds()` / `_save_thresholds()` | `_load_cameras()` / `_save_cameras()` |
| `closeEvent` | `self._save_thresholds()` | `self._save_cameras()` (+ capture current session first) |
| Sidecar file | `session_thresholds.json` | `session_cameras.json` |

The only deviation from the pattern: in `closeEvent`, capture the currently-displayed session's camera before saving, since `_load_cell()` only captures camera state on session *switch*.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| JSON sidecar in `output_base_dir` | Matches existing threshold pattern; human-readable; no new dependencies | None | **Chosen** |
| Reuse `camera_params.json` | Reuses existing file | Auto-computed by pipeline task, not for user adjustments | Rejected |
| Qt `QSettings` | OS-native persistence | Ties camera to machine/user; breaks if output dir moves | Rejected |

### Architecture Changes

No new modules. Two existing files modified:

```
rf_extraction_io.py              — add load_session_cameras() / save_session_cameras()
rf_cluster_gallery_viewer.py     — modify __init__, add _load_cameras()/_save_cameras(), update closeEvent
```

**Sidecar location:** `{self._gallery_data.output_base_dir}/session_cameras.json`

**Camera dict format** (unchanged from existing `_capture_camera()` return value):
```json
{
  "session_id_1": {
    "camera_position": [x, y, z],
    "focal_point":     [x, y, z],
    "up_vector":       [x, y, z],
    "view_angle":      30.0
  }
}
```

**Output dir access:** `self._gallery_data.output_base_dir` (`Path`) — same attribute used by `_load_thresholds()` / `_save_thresholds()`.

### Knowledge Base

No applicable notes — no existing entries cover PyVista camera persistence or Qt settings serialization patterns.

---

## Implementation Plan

### Phase 1: I/O helpers
**Goal:** Add `load_session_cameras` and `save_session_cameras` to `rf_extraction_io.py`.

- [ ] Add `load_session_cameras(output_dir: Path) -> Dict[str, dict]` — returns `{}` if file absent; raises `ValueError` with path on malformed JSON (fail-fast).
- [ ] Add `save_session_cameras(output_dir: Path, cameras: Dict[str, dict]) -> None` — writes with `indent=2`, creates dir if needed.

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_extraction_io.py` — two new functions placed after `save_delaunay_thresholds`

**Dependencies:** None

### Phase 2: Gallery viewer wiring
**Goal:** Connect the new I/O helpers to the viewer lifecycle.

- [ ] In `__init__` (line ~240): change `self._session_cameras: Dict[str, dict] = {}` → `self._session_cameras: Dict[str, dict] = self._load_cameras()`
- [ ] Add `_load_cameras(self) -> Dict[str, dict]` — delegates to `load_session_cameras(self._gallery_data.output_base_dir)`
- [ ] Add `_save_cameras(self) -> None` — delegates to `save_session_cameras(self._gallery_data.output_base_dir, self._session_cameras)`
- [ ] Update `closeEvent` (line ~1163) to capture current session then save:

```python
def closeEvent(self, event) -> None:  # noqa: N802
    if self._current_cell is not None:
        self._session_cameras[self._current_cell.session_id] = self._capture_camera()
    self._save_cameras()
    self._save_thresholds()
    self.plotter.close()
    super().closeEvent(event)
```

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/rf_cluster_gallery_viewer.py` — `__init__`, `closeEvent`, two new methods

**Dependencies:** Phase 1

---

## Testing Plan

### Unit Tests
- [ ] `load_session_cameras` with missing file returns `{}`
- [ ] `load_session_cameras` with valid JSON returns the correct dict
- [ ] `load_session_cameras` with malformed JSON raises `ValueError` containing the path
- [ ] `save_session_cameras` writes correct JSON to the expected path
- [ ] Round-trip: save then load returns the same dict

### Manual Verification
- [ ] Open gallery, navigate to a session, adjust camera orientation manually.
- [ ] Close the gallery; verify `session_cameras.json` appears in `output_base_dir`.
- [ ] Reopen the gallery; confirm the adjusted session opens with the saved orientation.
- [ ] Confirm sessions not yet viewed fall back to defaults (`view_xy` or `view_isometric`).
- [ ] Confirm first launch (no `session_cameras.json`) produces no error.

### Edge Cases
- [ ] `session_cameras.json` exists but contains `{}` — all sessions use defaults.
- [ ] Closing the gallery immediately on first open (no session switch ever triggered) — currently-displayed session's camera is captured correctly.

---

## Documentation Plan

- [ ] No user-facing docs needed — internal persistence behaviour.
- [ ] No CLAUDE.md update needed — no architectural pattern change.

---

## Rollback Plan

1. Delete `session_cameras.json` from `output_base_dir` — resets all sessions to defaults on next open.
2. Revert `rf_extraction_io.py` and `rf_cluster_gallery_viewer.py` to pre-change state.
3. No data migration required — the file is purely additive.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Stale cameras if mesh geometry changes between runs | Low | Low | Camera is per-session-id; user re-orients manually (no automatic invalidation needed) |
| `output_base_dir` not writable | Very low | Medium | Let `open()` raise `OSError` — consistent with fail-fast convention |
| JSON parse error on corrupted file | Very low | Low | Raise `ValueError` with path in message rather than silently returning `{}` |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: I/O helpers | ~30 min | None |
| Phase 2: Viewer wiring | ~30 min | Phase 1 |
