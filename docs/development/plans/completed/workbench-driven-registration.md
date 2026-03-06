# Plan: Workbench-Driven Registration with Full Parameter Persistence

**Date:** 2026-03-05
**Author:** Basil Duvernoy
**Status:** Completed
**Branch:** `feature/multi-snapshot-forearm-registration`

---

## Overview

`register_session_forearms` currently runs ICP registration with hardcoded function-argument parameters, then optionally opens a read-only viewer for post-hoc inspection. The `RegistrationWorkbench` GUI already exists for interactive parameter exploration but is standalone -- its results cannot be consumed by the orchestrator. This plan integrates the workbench into the orchestration flow so the user tunes parameters interactively before committing, and ensures all process parameters (`registration_method`, `max_correspondence_distance`, `icp_max_iteration`) are persisted in the transforms JSON.

## Problem Statement

1. **No interactive tuning in the pipeline** -- `register_session_forearms` accepts parameters as function args and runs a single ICP pass. Users must guess good parameters or re-run the whole pipeline to try different settings.
2. **Incomplete metadata** -- the transforms JSON only saves `mode` and `canonical_key`, omitting the 3 ICP process parameters. This makes it impossible to reproduce or audit a registration result from the saved artifacts alone.
3. **Redundant viewer** -- the read-only `show_registration_viewer` is called after registration, but the workbench already provides superior visualization with dual synced viewports.

## Goals

### In Scope
1. `register_session_forearms` launches `RegistrationWorkbench` when `visualize=True`, letting the user explore parameters and accept results interactively
2. All ICP process parameters (`registration_method`, `max_correspondence_distance`, `icp_max_iteration`) are persisted in the transforms JSON under a `"parameters"` key
3. Headless path (`visualize=False`) preserved for automated pipeline use
4. Cancellation (closing workbench without Accept) aborts cleanly, writing no artifacts

### Out of Scope
- Deleting `registration_viewer.py` (still useful for standalone inspection elsewhere)
- Adding `max_correspondence_distance` / `icp_max_iteration` as function-level args to `register_session_forearms` (use ForearmRegistrator defaults in headless mode; can be added later)

## Success Criteria

- [ ] Interactive mode: workbench opens, user tunes parameters, clicks Accept, unified PLY + transforms JSON are saved
- [ ] Transforms JSON includes `"parameters"` key with `registration_method`, `max_correspondence_distance`, `icp_max_iteration`
- [ ] Cancellation (X-button close without Accept): returns `None`, no artifacts written
- [ ] Headless mode (`visualize=False`): unchanged behavior, parameters still saved in JSON
- [ ] Existing `load_transforms()` consumers unaffected (additive schema change)

---

## Technical Design

### Approach

Add an "Accept" button to `RegistrationWorkbench` that closes the GUI and exposes results via a `RegistrationResult` dataclass. `register_session_forearms` instantiates the workbench with the loaded clouds, blocks on `.show()`, then reads results if accepted. Extend `ForearmRegistrator.save_transforms` with an optional `process_parameters` dict for the new JSON fields.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Dataclass result + Accept button | Type-safe, explicit accept/cancel semantics, widget values cached before close | Slightly more code in workbench | **Chosen** |
| `.show()` returns a dict | Simple | Stringly-typed, no accept/cancel signal, fragile | Rejected |
| Public properties only (no Accept) | Minimal change | No way to distinguish accept vs cancel; unclear if results are final | Rejected |

### Architecture Changes

**New type:** `RegistrationResult` frozen dataclass in `registration_workbench.py`

**Modified files:**
- `registration_workbench.py` -- Accept button, cached parameters, `RegistrationResult`, `accepted` property, `get_result()` method
- `forearm_registrator.py` -- `save_transforms` gains `process_parameters` kwarg
- `register_session_forearms.py` -- interactive path via workbench, headless path unchanged, `show_registration_viewer` import removed
- `registration/__init__.py` -- export `RegistrationResult`

**New JSON schema (additive):**

```json
{
    "mode": "reference",
    "canonical_key": "video_stem:42",
    "parameters": {
        "registration_method": "vanilla",
        "max_correspondence_distance": 0.10,
        "icp_max_iteration": 200
    },
    "transforms": {
        "video_stem:42": { "matrix_4x4": [[...]], "fitness": 1.0 }
    }
}
```

### Knowledge Base Constraints

- From `note-forearm-icp-registration.md`: default `max_correspondence_distance=0.10`, `icp_max_iteration=200`, fitness threshold 0.9. These defaults are preserved.
- From `note-open3d-scenewidget-layout.md`: SceneWidget must be direct Window child. Accept button goes in the existing `top_panel` row, not in a new nested container.

---

## Implementation Plan

### Phase 1: RegistrationResult dataclass + Accept button in workbench
**Goal:** Make the workbench capable of returning structured results to callers.

- [ ] Add `RegistrationResult` frozen dataclass at module level with fields: `mode`, `canonical_key`, `registration_method`, `max_correspondence_distance`, `icp_max_iteration`, `transforms`, `unified_cloud`
- [ ] Add `_accepted: bool = False` and `_btn_accept` to `__init__`
- [ ] Add cached parameter fields: `_last_mode`, `_last_method`, `_last_max_dist`, `_last_max_iter` (widgets may be invalid after `close()`)
- [ ] In `_build_gui`: add blue "Accept" button after Help button, disabled by default
- [ ] In `_on_process_clicked`: cache parameter values from widgets; enable Accept button
- [ ] Add `_on_accept_clicked`: set `_accepted = True`, call `self._win.close()`
- [ ] Add `accepted` property and `get_result() -> Optional[RegistrationResult]` method that reads cached values

**Files Modified:**
- `code/src/preprocessing/forearm_extraction/registration/registration_workbench.py` -- dataclass, button, properties (~40 lines added)

**Dependencies:** None

### Phase 2: Extend save_transforms with process parameters
**Goal:** Persist all ICP parameters in the transforms JSON.

- [ ] Add `process_parameters: Optional[Dict[str, object]] = None` kwarg to `save_transforms`
- [ ] Insert `"parameters"` key in JSON payload when provided
- [ ] Update docstring schema block to document the new key

**Files Modified:**
- `code/src/preprocessing/forearm_extraction/registration/forearm_registrator.py` -- `save_transforms` method (~10 lines changed)

**Dependencies:** None (can be done in parallel with Phase 1)

### Phase 3: Rewrite register_session_forearms orchestration
**Goal:** Route interactive mode through workbench, keep headless mode as-is.

- [ ] Replace `from .registration_viewer import show_registration_viewer` with `from .registration_workbench import RegistrationWorkbench`
- [ ] **Interactive path** (`visualize=True`): skip mode validation and canonical resolution (workbench handles these interactively); instantiate `RegistrationWorkbench(clouds)`, call `.show()`, check `.accepted`, read `.get_result()`; on cancel return `None`
- [ ] **Headless path** (`visualize=False`): keep existing registrator logic unchanged
- [ ] Both paths: build `process_parameters` dict and pass to `save_transforms`
- [ ] Remove `show_registration_viewer` call (line 183-184)
- [ ] Update docstring to document interactive behavior and cancellation

**Files Modified:**
- `code/src/preprocessing/forearm_extraction/registration/register_session_forearms.py` -- major rewrite of lines 125-186

**Dependencies:** Phase 1, Phase 2

### Phase 4: Update exports
**Goal:** Make `RegistrationResult` available from the package.

- [ ] Add `RegistrationResult` to `__init__.py` imports and `__all__`

**Files Modified:**
- `code/src/preprocessing/forearm_extraction/registration/__init__.py` -- 2 lines changed

**Dependencies:** Phase 1

---

## Testing Plan

### Manual Verification
- [ ] Run `preprocess_pipeline_extract_forearm_manual.py` for a multi-forearm session -- workbench opens, tune parameters, click Accept -- verify unified PLY and transforms JSON saved with `"parameters"` key
- [ ] Close workbench via X button (cancel) -- verify function returns `None`, no artifacts written
- [ ] Run with `visualize=False` (headless) -- verify registration completes and JSON includes `"parameters"`
- [ ] Load a saved transforms JSON via `ForearmRegistrator.load_transforms` and verify `"parameters"` dict is present
- [ ] Run standalone `registration_workbench.py __main__` test to verify Accept button renders and works

### Edge Cases
- [ ] Single-forearm session -- registration skipped before workbench opens (no change in behavior)
- [ ] Process clicked multiple times with different parameters -- Accept uses last run's results
- [ ] Accept button stays disabled until Process completes at least once
- [ ] `gui.Window.close()` cleanly exits `app.run()` -- if not, fall back to `gui.Application.instance.quit()`

---

## Documentation Plan

- [ ] Update help text in workbench (`_HELP_TEXT`) to document Accept button
- [ ] Update `note-forearm-icp-registration.md` to document `"parameters"` key in JSON schema

---

## Rollback Plan

1. Revert changes to `register_session_forearms.py` to restore direct `ForearmRegistrator` usage and `show_registration_viewer` call
2. Revert `save_transforms` signature to remove `process_parameters` kwarg
3. Remove `RegistrationResult` dataclass and Accept button from workbench
4. No data migration needed -- existing transforms JSON files without `"parameters"` key continue to work

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `gui.Window.close()` doesn't cleanly exit `app.run()` | Med | Med | Fall back to `gui.Application.instance.quit()` |
| Widget state invalid after `close()` | Med | High | Cache all parameter values in `_on_process_clicked`, never read widgets in `get_result()` |
| Cancellation leaves no artifacts, confusing downstream steps | Low | Low | `should_process_task` will re-trigger registration on next run |
| Additive JSON schema breaks existing consumers | Low | Low | `load_transforms` returns full dict; consumers accessing `["transforms"]` are unaffected |

---

## References

- Active plan: `docs/development/plans/active/multi-snapshot-forearm-registration.md`
- Completed plan: `docs/development/plans/completed/registration-workbench-gui.md`
- Knowledge base: `docs/development/knowledge-base/note-forearm-icp-registration.md`
