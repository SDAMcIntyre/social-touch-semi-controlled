# Plan: Docs Folder Reorganization

**Date:** 2026-02-22
**Author:** —
**Status:** Completed
**Branch:** `docs/reorganize-structure`

---

## Overview

Reorganize the `docs/` folder so that all developer-facing artefacts live under
a single `development/` subtree. Bug reports and engineering notes are merged
into `knowledge-base/` with a naming convention that preserves their categorical
distinction. Feature plans move into `development/plans/`.

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
3. Update **every** internal cross-reference broken by the moves — in CLAUDE.md,
   planning-procedure.md, the knowledge-base README, bug/dev-note files, and
   any plan files with sibling-relative paths.
4. Add a filename prefix convention (`bug-` / `note-`) to knowledge-base files
   so the original category remains discoverable without reading each file.
5. Define and document the `completed/` vs `archived/` distinction in
   `planning-procedure.md`.
6. Rewrite the knowledge-base README workflow instructions ("Adding a new note",
   bug cross-referencing) to match the merged structure.

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
- [ ] **All four** CLAUDE.md path references updated (`dev-notes/` link,
  `dev-notes/` prose mention, `plans/active/` path, `plans/completed/` path).
- [ ] `planning-procedure.md` references updated — Quick Start, Directory
  Structure, template block, Best Practices subagent prompt, and
  `dev-notes/README.md` mentions (at least 6 occurrences).
- [ ] Knowledge-base README index links, workflow instructions, and subagent
  prompt template all updated.
- [ ] No broken relative links in any moved markdown file (verified by
  automated grep — see Testing Plan).
- [ ] `planning-procedure.md` includes a written `completed/` vs `archived/`
  definition.
- [ ] This plan document itself is filed under
  `docs/development/plans/completed/docs-reorganization.md` after shipping.

---

## Technical Design

### Approach

Expand the existing `development/` folder to become the single root for all
developer-facing documentation. This avoids introducing a new top-level
directory and preserves the existing sub-structure (`git/`,
`planning-procedure.md`) without renaming anything.

Knowledge-base files from `bugs/` are renamed with a `bug-` prefix and files
from `dev-notes/` with a `note-` prefix so that `ls` or a README index
immediately communicates the document type without requiring the reader to
open each file.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Expand existing `development/` | No new top-level dir; naming already accurate | Slightly deeper paths for plans | **Chosen** |
| New `dev/` folder | Shorter name | Renames something that already exists; two very similar names (`dev/`, `development/`) | Rejected |
| Keep flat top-level structure with a `docs/README.md` index | Minimal moves | Does not create the user/developer boundary the project wants | Rejected |
| Merge into `knowledge-base/` without prefix convention | Fewer renames | Loses bug-vs-pattern distinction; flat folder becomes opaque | Rejected |

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
│   │   ├── neural-kinect-viewer/           (4 phase files)
│   │   └── neural-kinect-viewer-fix-initial-render/  (3 phase files)
│   ├── archived/
│   │   └── hue-circle-pyqt-sandbox.md
│   └── completed/
│       ├── fix-initial-render.md
│       ├── forearm-frame-averaging.md
│       ├── gpu_cupy_setup_and_viewer_improvements.md
│       ├── hsv-range-slider-ui.md
│       ├── neural-kinect-viewer-files.md
│       ├── neural-kinect-viewer.md
│       └── viewer-frame-rate-optimization.md
├── docs-reorganization.md                  ← this plan (misplaced)
└── kinect_frame_averaging_insight.docx
```

**After:**

```
docs/
├── development/
│   ├── knowledge-base/
│   │   ├── README.md                              ← rewritten from dev-notes/README.md
│   │   ├── note-cupy-import-order.md              ← from dev-notes/cupy-import-order.md
│   │   ├── note-open3d-scenewidget-layout.md      ← from dev-notes/open3d-scenewidget-layout.md
│   │   ├── bug-cupy-bool8-import-order.md         ← from bugs/cupy-bool8-import-order.md
│   │   ├── bug-hue-circle-drag-not-working.md     ← from bugs/hue-circle-drag-not-working.md
│   │   └── bug-neural-kinect-viewer-initial-render.md  ← from bugs/
│   ├── plans/
│   │   ├── active/
│   │   │   ├── hue-circle-integration-sandbox.md
│   │   │   ├── neural-kinect-viewer/
│   │   │   └── neural-kinect-viewer-fix-initial-render/
│   │   ├── archived/
│   │   │   └── hue-circle-pyqt-sandbox.md
│   │   └── completed/
│   │       ├── docs-reorganization.md             ← this plan, moved here on completion
│   │       ├── fix-initial-render.md
│   │       └── ... (6 more completed plans)
│   ├── git/
│   │   ├── commit-procedure.md
│   │   └── git-workflow.md
│   └── planning-procedure.md
└── kinect_frame_averaging_insight.docx             ← unchanged (out of scope)
```

**Exhaustive list of files with internal references that must be updated:**

All paths below are repo-root-relative.

| File | Reference | Old path | New path |
|------|-----------|----------|----------|
| `CLAUDE.md` | CuPy link | `docs/dev-notes/cupy-import-order.md` | `docs/development/knowledge-base/note-cupy-import-order.md` |
| `CLAUDE.md` | Dev Notes prose | `docs/dev-notes/` | `docs/development/knowledge-base/` |
| `CLAUDE.md` | Plan active path | `docs/plans/active/[feature-name].md` | `docs/development/plans/active/[feature-name].md` |
| `CLAUDE.md` | Plan completed path | `docs/plans/completed/` | `docs/development/plans/completed/` |
| `docs/development/planning-procedure.md` | Quick Start step 1 | `docs/plans/active/` | `docs/development/plans/active/` |
| `docs/development/planning-procedure.md` | Quick Start step 4 | `docs/plans/README.md` | (remove — file doesn't exist) |
| `docs/development/planning-procedure.md` | Directory Structure block | `docs/plans/` tree | `docs/development/plans/` tree |
| `docs/development/planning-procedure.md` | Template block | `docs/plans/` paths | `docs/development/plans/` paths |
| `docs/development/planning-procedure.md` | Best Practices subagent prompt | `docs/dev-notes/README.md` | `docs/development/knowledge-base/README.md` |
| `docs/development/planning-procedure.md` | Post-Implementation step 2 | `docs/plans/completed/` | `docs/development/plans/completed/` |
| `docs/development/planning-procedure.md` | Naming and Organization | `active/` directory | `docs/development/plans/active/` |
| `docs/development/knowledge-base/README.md` | Index table links | `cupy-import-order.md`, etc. | `note-cupy-import-order.md`, etc. |
| `docs/development/knowledge-base/README.md` | "Adding a new note" instructions | `docs/dev-notes/<slug>.md` | `docs/development/knowledge-base/note-<slug>.md` or `bug-<slug>.md` |
| `docs/development/knowledge-base/README.md` | Bug cross-reference workflow | `../bugs/` references | Rewrite for merged folder |
| `docs/development/knowledge-base/bug-cupy-bool8-import-order.md` | Dev-note cross-link | `docs/dev-notes/cupy-import-order.md` | `note-cupy-import-order.md` (sibling) |
| Plan files in `active/` and `completed/` | Any `docs/plans/` or `docs/dev-notes/` references | Various | Audit each file |

### Completed vs Archived — proposed definitions

To be added to `planning-procedure.md`:

- **`completed/`** — Plans for features that were **shipped as designed**. The
  plan accurately reflects what was built. These are the canonical design
  records and should be kept indefinitely.
- **`archived/`** — Plans that were **superseded, abandoned, or substantially
  rewritten** before completion. They preserve decision history (why an
  approach was rejected) but do not describe the current system.

---

## Implementation Plan

### Phase 1: Move and rename files

**Goal:** Perform all filesystem moves in one pass with no content edits.

- [ ] Create `docs/development/knowledge-base/`
- [ ] Move and rename `docs/dev-notes/README.md` → `docs/development/knowledge-base/README.md`
- [ ] Move and rename `docs/dev-notes/cupy-import-order.md` → `docs/development/knowledge-base/note-cupy-import-order.md`
- [ ] Move and rename `docs/dev-notes/open3d-scenewidget-layout.md` → `docs/development/knowledge-base/note-open3d-scenewidget-layout.md`
- [ ] Move and rename `docs/bugs/cupy-bool8-import-order.md` → `docs/development/knowledge-base/bug-cupy-bool8-import-order.md`
- [ ] Move and rename `docs/bugs/hue-circle-drag-not-working.md` → `docs/development/knowledge-base/bug-hue-circle-drag-not-working.md`
- [ ] Move and rename `docs/bugs/neural-kinect-viewer-initial-render.md` → `docs/development/knowledge-base/bug-neural-kinect-viewer-initial-render.md`
- [ ] Move `docs/plans/` → `docs/development/plans/` (entire subtree)
- [ ] Move `docs/docs-reorganization.md` → `docs/development/plans/active/docs-reorganization.md`
- [ ] Delete now-empty `docs/dev-notes/` and `docs/bugs/` directories

**Files Modified:** All files listed above (moved/renamed, not edited).

**Dependencies:** None

### Phase 2: Update cross-references

**Goal:** Restore all broken internal links introduced by Phase 1.

- [ ] Update `CLAUDE.md` — all four references (see reference table above)
- [ ] Update `docs/development/planning-procedure.md` — all ~8 occurrences
  (Quick Start, Directory Structure, template block, Best Practices prompt,
  Post-Implementation, Naming section)
- [ ] Remove the nonexistent `docs/plans/README.md` reference from
  `planning-procedure.md` Quick Start step 4
- [ ] Rewrite `docs/development/knowledge-base/README.md`:
  - Update index table links to use `note-` / `bug-` prefixed filenames
  - Rewrite "Adding a new note" section for `knowledge-base/` path and
    `note-` / `bug-` naming convention
  - Remove or rewrite bug cross-referencing instructions (no longer separate folders)
  - Update subagent prompt template path
- [ ] Update `docs/development/knowledge-base/bug-cupy-bool8-import-order.md` — sibling link
- [ ] Audit every file in `docs/development/plans/active/` and
  `docs/development/plans/completed/` for stale `docs/plans/` or
  `docs/dev-notes/` references — fix any found
- [ ] Add `completed/` vs `archived/` definition to `planning-procedure.md`
  (use text from Technical Design section above)

**Files Modified:**
`CLAUDE.md`, `docs/development/planning-procedure.md`,
`docs/development/knowledge-base/README.md`,
`docs/development/knowledge-base/bug-cupy-bool8-import-order.md`,
any plan files with broken paths.

**Dependencies:** Phase 1

### Phase 3: Self-filing

**Goal:** Move this plan to its final location.

- [ ] Move `docs/development/plans/active/docs-reorganization.md` →
  `docs/development/plans/completed/docs-reorganization.md`
- [ ] Update status to `Completed` in the plan frontmatter

**Dependencies:** Phase 2

---

## Testing Plan

### Automated Verification

- [ ] Run `grep -r 'docs/bugs/' .` from repo root — expect zero matches.
- [ ] Run `grep -r 'docs/dev-notes/' .` from repo root — expect zero matches.
- [ ] Run `grep -r 'docs/plans/' .` from repo root — expect zero matches
  (all should now say `docs/development/plans/`).

### Manual Verification

- [ ] From repo root, confirm `docs/bugs/` and `docs/dev-notes/` are gone.
- [ ] From repo root, confirm `docs/plans/` is gone.
- [ ] Open `CLAUDE.md` and verify all four doc links resolve to existing files.
- [ ] Open `planning-procedure.md`, follow every internal markdown link.
- [ ] Open `knowledge-base/README.md`, verify all index links resolve.
- [ ] Open one active plan and one completed plan; verify any cross-references
  resolve.

### Edge Cases

- [ ] Phased plan directories (`neural-kinect-viewer/`,
  `neural-kinect-viewer-fix-initial-render/`) retain their internal relative
  links after the parent directory move.

---

## Documentation Plan

- [ ] Update `CLAUDE.md` with new paths (done in Phase 2).
- [ ] Update `planning-procedure.md` Quick Start, Directory Structure, template,
  and Best Practices sections to reflect new paths (done in Phase 2).
- [ ] Rewrite knowledge-base README workflow sections (done in Phase 2).

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
| Missed CLAUDE.md reference breaks AI-assisted sessions project-wide | Medium | **High** — every Claude Code session starts with wrong paths | Exhaustive reference table above; automated grep verification in testing plan |
| Broken link in a plan file missed during audit | Medium | Low — docs only, no runtime effect | Automated `grep -r` for old path prefixes catches all occurrences |
| `planning-procedure.md` template block still shows old paths | Medium | Medium — new plans get created in wrong location | Enumerate all occurrences explicitly in Phase 2 checklist |
| Phased plan sub-files have relative links that break after directory reparenting | Low | Low | Test one phased plan manually before closing Phase 2 |
| `bug-` / `note-` prefix rename breaks external bookmarks or browser history | Low | Low — internal docs only | Acceptable trade-off for long-term discoverability |

---

## References

- Analysis that motivated this plan: conversation with Claude Code, 2026-02-22
- `docs/development/planning-procedure.md` — planning template used here
- `docs/dev-notes/README.md` — source for knowledge-base README content
