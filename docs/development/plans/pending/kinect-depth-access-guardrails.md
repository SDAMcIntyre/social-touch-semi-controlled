# Plan: Kinect depth access — close remaining bypasses and guard the invariant

**Created:** 2026-04-14 18:00
**Approved:** 2026-04-14 18:00
**Completed:** —
**Author:** Basil Duvernoy
**Status:** Approved
**Branch:** `feature/kinect-parallax-tolerance-band`

---

## Overview

**What:** Delete the one remaining dead-code path that reads raw Kinect depth
via `pyk4a`, flip the `KinectRgbDepthViewer.parallax_correction` default to
`True` so the widget is safe-by-default, and document the "single data-access
path" invariant so future contributors do not silently reintroduce bypasses.

**Why:** The sibling plan `kinect-frame-parallax-correction.md` pushes the
median RGB↔depth shift into `KinectFrame`. That correction only helps if
every depth read flows through `KinectFrame` / `KinectMKV` /
`KinectPointCloudView`. A two-agent audit confirmed production paths are
clean, but found one dead bypass, one inverted default, and no documented
rule — three small gaps that let the invariant quietly decay over time.

**How:** One deletion in `extract_participant_forearm.py`, one default flip
in `kinect_rgb_depth_viewer.py`, one new KB note, one paragraph added to
`CLAUDE.md`, and a refresh of the existing parallax KB note's §6 audit
table.

## Problem Statement

- **Dead-code leak.** `show_video()` at
  `code/scripts/_3_preprocessing/_3_forearm_extraction/extract_participant_forearm.py:170-197`
  imports `pyk4a.PyK4APlayback` locally and reads `capture.transformed_depth`.
  Grep confirms it is never called, but the code still typechecks and will
  drift out of sync with the blessed layer.
- **Inverted default.** `KinectRgbDepthViewer(parallax_correction=False)` is
  the default (`code/src/preprocessing/common/gui/kinect_rgb_depth_viewer.py:84`).
  Its only current caller (the audit script) passes `False` explicitly, so
  the default is load-bearing only for future / naïve callers — which will
  silently receive raw depth.
- **No guardrail for new bypasses.** Nothing in `CLAUDE.md` or the knowledge
  base documents the "all depth through `KinectFrame`/`KinectMKV`"
  invariant, so a future commit that adds a new `pyk4a` import outside the
  data-access layer has no reason to be flagged in review.

## Goals

### In Scope
1. Delete dead `show_video()` in `extract_participant_forearm.py` (and the
   three helpers it was the sole caller of: `_create_monitoring_frame`,
   `_display_frame`, `_resize_with_padding`).
2. Flip `KinectRgbDepthViewer.__init__` default to `parallax_correction=True`;
   update its docstring and the accompanying inline comment.
3. Confirm `audit_rgb_depth_registration.py` still passes
   `parallax_correction=False` explicitly; add a one-line comment referring
   to the new KB note.
4. Create `docs/development/knowledge-base/note-kinect-depth-access-single-path.md`
   stating the invariant, the `pyk4a` allowlist, and the grep-based audit
   command.
5. Add a short rule block to `CLAUDE.md` pointing at the new KB note.
6. Refresh §6 of `note-azure-kinect-rgb-depth-parallax.md` to remove the
   deleted `show_video` row.

### Out of Scope
- The `KinectFrame` correction itself — handled by
  `kinect-frame-parallax-correction.md`.
- Refactoring any of the legitimate `pyk4a` callers in the allowlist
  (`mkv_stream_analyze.py`, `extract_color_to_mp4.py`,
  `xyz_metadata_model.py`, `debug_opengl.py`). These are non-depth or
  structural-validation uses and are intentional.
- Automated lint/CI enforcement of the rule. The guardrail is
  documentation-only; a test can be added later if reviews miss violations.

## Success Criteria

- [ ] `show_video` is gone; file still parses; `grep show_video code/` returns
      no references.
- [ ] `KinectRgbDepthViewer()` (no kwargs) returns corrected frames;
      `audit_rgb_depth_registration.py` still recovers Δu ≈ +10 px median
      because it opts out explicitly.
- [ ] `grep -rnw "pyk4a" code/` output matches the allowlist in the new KB
      note exactly. No new rows, no missing rows.
- [ ] `CLAUDE.md` contains a rule block linking to
      `note-kinect-depth-access-single-path.md`.
- [ ] §6 of `note-azure-kinect-rgb-depth-parallax.md` no longer references
      `extract_participant_forearm.show_video`.

---

## Technical Design

### Approach

Treat the data-access layer as a one-way gate: after this plan, the only way
to read Kinect depth is through `KinectFrame` / `KinectMKV` /
`KinectPointCloudView`. Any `pyk4a` import outside
`code/src/preprocessing/common/data_access/` must appear in an explicit
allowlist in the KB note with a justification.

The three remedies are mechanical and each has a single point of failure:

1. **Deletion** closes the only known leak. Because the function is dead, no
   downstream caller breaks.
2. **Default flip** aligns the viewer with the "safe-by-default" principle
   already adopted for `KinectFrame` / `KinectMKV`. The one caller that
   actually needs raw data already passes `False` explicitly, so the flip is
   no-op for the current pipeline but closes the naïve-misuse hole.
3. **Documentation** makes the invariant discoverable. Reviewers and future
   contributors have a single grep command and a KB allowlist to compare
   against.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|---|---|---|---|
| Delete `show_video` + flip default + KB note (chosen) | Closes the known leak, aligns defaults, documents the rule | Documentation drift risk over time | **Chosen** |
| Only delete `show_video`, leave viewer default | Minimal surface | Inverted default still ticks for any future naïve reuse | Rejected |
| Add a pytest that scans `code/` for disallowed `pyk4a` imports | Machine-enforced | Maintenance cost; false positives in test files; user explicitly chose doc-only guardrail | Rejected (may revisit if reviews miss violations) |
| Move viewer out of the GUI layer into diagnostics-only | Physically prevents naïve reuse | High churn for a one-line change | Rejected |

### Architecture Changes

```
code/src/preprocessing/common/gui/
└── kinect_rgb_depth_viewer.py        — default flip + docstring/comment rewrite

code/scripts/_3_preprocessing/_3_forearm_extraction/
└── extract_participant_forearm.py    — delete show_video + helpers + dead pyk4a import

code/scripts/_3_preprocessing/_0_kinect_diagnostics/
└── audit_rgb_depth_registration.py   — one-line KB reference comment

docs/development/knowledge-base/
├── note-kinect-depth-access-single-path.md   — NEW: invariant + allowlist
└── note-azure-kinect-rgb-depth-parallax.md   — §6 row removal

CLAUDE.md                               — new rule block linking to KB note
```

---

## Implementation Plan

### Phase 1: Remove the dead bypass
**Goal:** Close the one real leak.
**Started:** —
**Completed:** —

- [ ] Delete `show_video` (lines 170–197) in
      `code/scripts/_3_preprocessing/_3_forearm_extraction/extract_participant_forearm.py`.
- [ ] Delete `_create_monitoring_frame`, `_display_frame`, `_resize_with_padding`
      in the same file (grep confirms they are only called by the removed
      `show_video`).
- [ ] Remove the local `from pyk4a import PyK4APlayback` (it lived inside
      `show_video`; it disappears with the function).
- [ ] Parse check:
      `python -c "import ast, pathlib; ast.parse(pathlib.Path('code/scripts/_3_preprocessing/_3_forearm_extraction/extract_participant_forearm.py').read_text())"`.

**Files Modified:**
- `code/scripts/_3_preprocessing/_3_forearm_extraction/extract_participant_forearm.py` — delete dead helpers and the raw-pyk4a path.

**Dependencies:** None.

### Phase 2: Flip the viewer default
**Goal:** Safe-by-default for any future caller.
**Started:** —
**Completed:** —

- [ ] In `code/src/preprocessing/common/gui/kinect_rgb_depth_viewer.py`:
  - Change the `__init__` signature to `parallax_correction: bool = True`.
  - Rewrite the docstring block for `parallax_correction` (lines 71–76):
    *"Defaults to `True` so the widget displays corrected depth. Pass
    `False` only for RGB↔depth registration audits that must measure the
    raw pyk4a offset."*
  - Rewrite the inline comment (lines 91–93) to match the new default.
- [ ] In `code/scripts/_3_preprocessing/_0_kinect_diagnostics/audit_rgb_depth_registration.py`:
  - Confirm `parallax_correction=False` is still passed explicitly (it is).
  - Add a comment on that line:
    `# explicit raw read — see note-kinect-depth-access-single-path.md §"Audit-mode opt-out"`.

**Files Modified:**
- `code/src/preprocessing/common/gui/kinect_rgb_depth_viewer.py` — default + docs.
- `code/scripts/_3_preprocessing/_0_kinect_diagnostics/audit_rgb_depth_registration.py` — one-line KB reference.

**Dependencies:** Phase 1 (no functional dependency, just ordering for review).

### Phase 3: Document the invariant
**Goal:** Make the rule discoverable; inventory existing exemptions.
**Started:** —
**Completed:** —

- [ ] Create `docs/development/knowledge-base/note-kinect-depth-access-single-path.md`
      with these sections:
  - **Invariant.** All code reading Kinect depth pixels MUST go through
    `KinectFrame` / `KinectMKV` / `KinectPointCloudView`. Direct `pyk4a`
    imports outside `code/src/preprocessing/common/data_access/` require a
    justification in the allowlist below.
  - **Why it matters.** Links to `note-azure-kinect-rgb-depth-parallax.md`
    and `plans/active/kinect-frame-parallax-correction.md`. A single
    unblessed reader silently reintroduces ~10 px median offset → metres of
    XYZ error at depth edges.
  - **Allowlist** (table of files importing `pyk4a` outside
    `data_access/`, what they do, and why it's OK):
    - `code/src/primary_processing/mkv_video_management/mkv_stream_analyze.py`
      — `PyK4APlayback` for MKV stream-structure validation only.
    - `code/scripts/_2_primary_processing/_2_generate_rgb_depth_video/extract_color_to_mp4.py`
      — RGB-only extraction; reads `capture.color`, never touches depth.
    - `code/src/preprocessing/stickers_analysis/xyz/models/xyz_metadata_model.py`
      — reads `playback.configuration` (metadata), no frame pixel reads.
    - `code/scripts/_3_preprocessing/_3_forearm_extraction/debug_opengl.py`
      — `__import__("pyk4a")` availability probe only.
  - **Audit-mode opt-out.** The single explicit pathway for raw reads:
    `KinectMKV(..., parallax_correction=False)` or
    `KinectRgbDepthViewer(..., parallax_correction=False)`. Used only by
    `audit_rgb_depth_registration.py`. New callers must not default to
    `False`.
  - **How to detect drift.** Run
    `grep -rn "from pyk4a\|import pyk4a" code/` and diff against the
    allowlist; any new row is a regression unless accompanied by a plan.
- [ ] Append a rule block to `CLAUDE.md`:

  ```markdown
  ## Kinect depth access — single path

  All code reading Kinect depth must go through `KinectFrame` /
  `KinectMKV` / `KinectPointCloudView`. Raw `pyk4a` reads are allowed
  only in the files enumerated in
  [`note-kinect-depth-access-single-path.md`](docs/development/knowledge-base/note-kinect-depth-access-single-path.md).
  ```

- [ ] In `docs/development/knowledge-base/note-azure-kinect-rgb-depth-parallax.md`
      §6: remove the `extract_participant_forearm.show_video` row (if
      present); confirm `KinectFrame` / `KinectPointCloudView` rows read
      `mitigated (median shift)`.
- [ ] Update `docs/development/knowledge-base/README.md` index to list the
      new KB note.

**Files Modified:**
- `docs/development/knowledge-base/note-kinect-depth-access-single-path.md` — new.
- `docs/development/knowledge-base/README.md` — add index entry.
- `docs/development/knowledge-base/note-azure-kinect-rgb-depth-parallax.md` — §6 refresh.
- `CLAUDE.md` — new rule block.

**Dependencies:** Phases 1 and 2 (so the allowlist reflects the post-plan state).

### Phase 4: Verification
**Goal:** Confirm invariant holds in tree.
**Started:** —
**Completed:** —

- [ ] `grep -rn "from pyk4a\|import pyk4a" code/` — output matches the KB
      allowlist (4 rows) plus the blessed `data_access/kinect_mkv_manager.py`.
- [ ] Re-run `audit_rgb_depth_registration.py` on one previously-measured
      ST13 session; confirm Δu median ≈ +10 px (audit still sees raw data
      despite the default flip).
- [ ] Smoke-run `extract_participant_forearm.py` on one ST13 session
      (monitor + non-monitor paths) — no broken imports.
- [ ] Existing unit tests still pass:
      `pytest code/tests/test_parallax_correction.py code/tests/test_ellipse_depth_extractor.py`.

**Dependencies:** Phases 1–3.

---

## Testing Plan

### Unit Tests
- None added. Existing tests cover the correction primitive and consumers.

### Integration Tests
- [ ] Audit re-run on a previously-measured session recovers +10 px median Δu.
- [ ] Forearm extraction smoke test on one ST13 session produces the same
      `.ply` size/shape as pre-change.

### Manual Verification
- [ ] `KinectRgbDepthViewer(mkv_path)` (no kwargs) opens and shows depth
      shifted relative to a run with `parallax_correction=False`.

---

## Documentation Plan

- [ ] Create `note-kinect-depth-access-single-path.md`.
- [ ] Update `CLAUDE.md` with rule block.
- [ ] Refresh §6 of `note-azure-kinect-rgb-depth-parallax.md`.
- [ ] Update `docs/development/knowledge-base/README.md` index.

---

## Rollback Plan

1. `git revert` the implementation commits — each phase is an atomic change
   set (deletion, flip, docs).
2. No data migrations, no on-disk format change.
3. If the viewer default flip surfaces a caller that relied on the old
   `False` default, either (a) have that caller pass `False` explicitly, or
   (b) revert just Phase 2.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Viewer default flip breaks a hidden caller | Very Low | Low | Grep-audited; the single caller passes `False` explicitly. |
| Helper deletion (`_display_frame`, `_resize_with_padding`, `_create_monitoring_frame`) removes still-used code | Low | Low | Grep confirms usage is confined to `show_video`; parse check after delete. |
| KB allowlist drifts as new files import `pyk4a` | Medium | Low–Med | Documented grep command in the KB note; CLAUDE.md rule prompts reviewers. |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|---|---|---|
| Phase 1 | ~15 min | None |
| Phase 2 | ~15 min | Phase 1 |
| Phase 3 | ~45 min | Phases 1–2 |
| Phase 4 | ~30 min (+ smoke-test runtime) | Phases 1–3 |

---

## References

- Parent plan: [`plans/active/kinect-frame-parallax-correction.md`](../active/kinect-frame-parallax-correction.md)
- KB note (to refresh): [`knowledge-base/note-azure-kinect-rgb-depth-parallax.md`](../../knowledge-base/note-azure-kinect-rgb-depth-parallax.md)
- Completed parent: [`plans/completed/kinect-rgb-depth-parallax-tolerance-band.md`](../completed/kinect-rgb-depth-parallax-tolerance-band.md)
