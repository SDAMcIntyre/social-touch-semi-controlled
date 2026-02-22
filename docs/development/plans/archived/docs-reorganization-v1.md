# Plan: Docs Folder Reorganization

**Date:** 2026-02-22
**Author:** —
**Status:** Draft
**Branch:** `docs/reorganize-structure`

---

## Overview

Reorganize the `docs/` folder to separate developer-facing documentation from
end-user documentation. All implementation and coding artefacts (bug reports,
engineering notes, feature plans, workflow guides) are consolidated under a
single `development/` subtree. Bug incident reports and resolved-pattern notes
are merged into a unified `knowledge-base/`.

## Problem Statement

The current layout scatters developer artefacts across three top-level siblings
(`bugs/`, `dev-notes/`, `plans/`) alongside the `development/` folder that
already exists for workflow guides. This creates three problems:

1. **Cognitive overhead** — a new contributor must scan four separate top-level
   directories to get a full picture of the project's engineering knowledge.
2. **Redundant cross-linking** — `bugs/` and `dev-notes/` cover two halves of
   the same incident (what went wrong / how to fix it), forcing every entry to
   maintain manual cross-references that can drift.
3. **No clear user/developer boundary** — there is no structural signal for
   what belongs to developers versus what is relevant to software users; adding
   user-facing docs in the future would be ambiguous.

## Goals

### In Scope

1. Merge `docs/bugs/` and `docs/dev-notes/` into `docs/development/knowledge-base/`.
2. Move `docs/plans/` to `docs/development/plans/`.
3. Update all internal cross-references broken by the moves (CLAUDE.md,
   planning-procedure.md, dev-notes README, plan files that reference sibling paths).
4. Define and document the `completed/` vs `archived/` distinction in
   `planning-procedure.md`.

### Out of Scope

- Converting bug-report files to the 7-section dev-notes format (content
  migration; can follow as a separate task).
- Adding a `docs/README.md` navigation index (can follow once the target
  structure is stable).
- Resolving the orphaned `kinect_frame_averaging_insight.docx` binary.
- Any changes to code, tests, or non-docs files.

## Success Criteria

- [ ] `docs/bugs/` and `docs/dev-notes/` no longer exist at the top level.
- [ ] `docs/plans/` no longer exists at the top level.
- [ ] All moved files are reachable under `docs/development/`.
- [ ] `CLAUDE.md` references updated to new paths.
- [ ] `planning-procedure.md` references updated to new paths.
- [ ] `dev-notes` README index links updated.
- [ ] No broken relative links in any moved markdown file (spot-checked).
- [ ] `planning-procedure.md` includes a written distinction between
  `completed/` and `archived/`.

---

## Technical Design

### Approach

Expand the existing `development/` folder to become the single root for all
developer-facing documentation. This avoids introducing a new top-level
directory and preserves the existing sub-structure (`git/`,
`planning-procedure.md`) without renaming anything.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Expand existing `development/` | No new top-level dir; naming already accurate | Slightly deeper paths for plans | **Chosen** |
| New `dev/` folder | Shorter name | Renames something that already exists; two very similar names (`dev/`, `development/`) | Rejected |
| Keep flat top-level structure with a `docs/README.md` index | Minimal moves | Does not create the user/developer boundary the project wants | Rejected |

### Architecture Changes

**Before:**

```
docs/
├── bugs/
│   ├── cupy-bool8-import-order.md
│   ├── hue-circle-drag-not-working.md
│   └── neural-kinect-viewer-initial-render.md
├── dev-notes/
│   ├── README.md
│   ├── cupy-import-order.md
│   └── open3d-scenewidget-layout.md
├── development/
│   ├── git/
│   │   ├── commit-procedure.md
│   │   └── git-workflow.md
│   └── planning-procedure.md
├── plans/
│   ├── active/
│   │   ├── hue-circle-integration-sandbox.md
│   │   ├── neural-kinect-viewer/
│   │   └── neural-kinect-viewer-fix-initial-render/
│   ├── archived/
│   │   └── hue-circle-pyqt-sandbox.md
│   └── completed/
│       └── [7 completed plan files]
└── kinect_frame_averaging_insight.docx
```

**After:**

```
docs/
├── development/
│   ├── knowledge-base/
│   │   ├── README.md                              ← from dev-notes/README.md
│   │   ├── cupy-import-order.md                   ← from dev-notes/
│   │   ├── open3d-scenewidget-layout.md            ← from dev-notes/
│   │   ├── cupy-bool8-import-order.md              ← from bugs/
│   │   ├── hue-circle-drag-not-working.md          ← from bugs/
│   │   └── neural-kinect-viewer-initial-render.md  ← from bugs/
│   ├── plans/
│   │   ├── active/
│   │   │   ├── hue-circle-integration-sandbox.md
│   │   │   ├── neural-kinect-viewer/
│   │   │   └── neural-kinect-viewer-fix-initial-render/
│   │   ├── archived/
│   │   │   └── hue-circle-pyqt-sandbox.md
│   │   └── completed/
│   │       └── [7 completed plan files]
│   ├── git/
│   │   ├── commit-procedure.md
│   │   └── git-workflow.md
│   └── planning-procedure.md
└── kinect_frame_averaging_insight.docx             ← unchanged (out of scope)
```

**Files with internal references that must be updated:**

| File | References to update |
|------|---------------------|
| `CLAUDE.md` | `docs/dev-notes/cupy-import-order.md` → `docs/development/knowledge-base/cupy-import-order.md` |
| `CLAUDE.md` | `docs/development/planning-procedure.md` — path unchanged; no edit needed |
| `development/planning-procedure.md` | `docs/plans/active/`, `docs/plans/completed/` → `docs/development/plans/…` |
| `development/planning-procedure.md` | `docs/dev-notes/README.md` dev-notes check prompt → `docs/development/knowledge-base/README.md` |
| `development/knowledge-base/README.md` | All `../bugs/` and `../dev-notes/` cross-links → new relative paths |
| `bugs/cupy-bool8-import-order.md` | `docs/dev-notes/cupy-import-order.md` → `docs/development/knowledge-base/cupy-import-order.md` |
| Individual plan files | Any sibling-relative paths pointing to `docs/plans/` or `docs/dev-notes/` |

---

## Implementation Plan

### Phase 1: Move files

**Goal:** Perform all filesystem moves in one pass with no content edits.

- [ ] Create `docs/development/knowledge-base/`
- [ ] Move `docs/dev-notes/*` → `docs/development/knowledge-base/`
- [ ] Move `docs/bugs/*` → `docs/development/knowledge-base/`
- [ ] Move `docs/plans/` → `docs/development/plans/`
- [ ] Delete now-empty `docs/dev-notes/` and `docs/bugs/` directories

**Files Modified:**
All files listed in the Architecture Changes table above (moved, not edited).

**Dependencies:** None

### Phase 2: Update cross-references

**Goal:** Restore all broken internal links introduced by Phase 1.

- [ ] Update `CLAUDE.md` — dev-notes path reference
- [ ] Update `development/planning-procedure.md` — plans paths + dev-notes check prompt
- [ ] Update `development/knowledge-base/README.md` — all internal links
- [ ] Update `development/knowledge-base/cupy-bool8-import-order.md` — dev-note cross-link
- [ ] Spot-check cross-links in `development/plans/active/` and `development/plans/completed/` files
- [ ] Add `completed/` vs `archived/` definition to `planning-procedure.md`

**Files Modified:**
`CLAUDE.md`, `docs/development/planning-procedure.md`,
`docs/development/knowledge-base/README.md`,
`docs/development/knowledge-base/cupy-bool8-import-order.md`,
any plan files with broken paths.

**Dependencies:** Phase 1

---

## Testing Plan

### Manual Verification

- [ ] From repo root, confirm `docs/bugs/` and `docs/dev-notes/` are gone.
- [ ] From repo root, confirm `docs/plans/` is gone.
- [ ] Open `CLAUDE.md` and verify both doc links resolve to existing files.
- [ ] Open `planning-procedure.md`, follow every internal markdown link.
- [ ] Open `knowledge-base/README.md`, verify all index links resolve.
- [ ] Open one active plan and one completed plan; verify any cross-references resolve.

### Edge Cases

- [ ] Phased plan directories (`neural-kinect-viewer/`,
  `neural-kinect-viewer-fix-initial-render/`) retain their internal relative
  links after the parent directory move.

---

## Documentation Plan

- [ ] Update `CLAUDE.md` with new paths (done in Phase 2).
- [ ] Update `planning-procedure.md` Quick Start section directory listing
  to reflect new `docs/development/plans/active/` path.

---

## Rollback Plan

All changes are filesystem moves and text edits tracked by git.

1. `git diff --stat` before committing to verify only expected files changed.
2. If anything is wrong: `git restore --staged .` and `git checkout .` to
   discard all unstaged edits; `git clean -fd docs/` to remove newly created
   directories.
3. No code, database, or binary state is affected — full rollback is a single
   `git reset --hard`.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Broken link in a plan file missed during spot-check | Medium | Low — docs only | Check every file that has a `docs/` absolute or `../` relative path |
| `planning-procedure.md` Quick Start still shows old `docs/plans/active/` path | Medium | Medium — misleads future contributors | Update explicitly in Phase 2 checklist |
| Phased plan sub-files have relative links that break after directory reparenting | Low | Low | Test one phased plan manually before closing Phase 2 |

---

## References

- Analysis that motivated this plan: conversation with Claude Code, 2026-02-22
- `docs/development/planning-procedure.md` — planning template used here
- `docs/dev-notes/README.md` — source for knowledge-base README content
