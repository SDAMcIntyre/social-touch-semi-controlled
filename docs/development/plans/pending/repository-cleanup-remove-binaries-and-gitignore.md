# Plan: Repository Cleanup — Remove Large Binaries & Update .gitignore

**Date:** 2026-03-13
**Author:** Basil Duvernoy
**Status:** Draft
**Branch:** `chore/repository-cleanup-remove-binaries`

---

## Overview

Remove large binary files (PPTX, DOCX, redundant ZIP) from git tracking, update `.gitignore` to prevent re-addition, and clean up stale worktrees. This reduces repo bloat by ~54 MB and establishes guardrails against future binary accumulation.

## Problem Statement

The repository tracks several large binary files that don't belong in version control:

- **Two PowerPoint presentations** (`docs/design/MNG_tracking_analysis-Shan-*.pptx`) totalling **51.2 MB** — research presentations that should live on a shared drive or cloud storage
- **Two Word documents** (`docs/design/*.docx`) — research notes that are not code artifacts
- **A redundant ZIP archive** (`media/cues_social_touch_semi_controlled.zip`, **2.7 MB**) — contains the same audio cues already tracked individually in `media/cues/`

Additionally, `.gitignore` lacks patterns for `.pptx` and `.docx`, so these file types could be accidentally re-added. Three stale git worktrees from previous Claude sessions also need cleanup.

These files inflate clone size and are not useful for development workflows.

## Goals

### In Scope
1. Remove the 5 identified binary files from git tracking
2. Add `.pptx` and `.docx` patterns to `.gitignore`
3. Verify `k4a.log` is properly gitignored
4. Clean up 3 stale git worktrees
5. Document the dev → main merge gap (282 commits) for future action

### Out of Scope
- Merging `dev` into `main` (to be done separately when ready)
- Moving `.pkl` model files to Git LFS (needed by imported hand tracking code; risky to change)
- Migrating `media/cues/` to LFS (staying in git per user decision)
- Plan document lifecycle moves (user will handle manually)
- Rewriting git history with BFG or `git filter-repo` (files remain in history; this only removes from HEAD)

## Success Criteria

- [ ] `git ls-files -- "*.pptx" "*.docx"` returns empty
- [ ] `media/cues_social_touch_semi_controlled.zip` is no longer tracked
- [ ] `.gitignore` contains `*.pptx` and `*.docx` patterns
- [ ] `k4a.log` confirmed ignored (not showing in `git status`)
- [ ] `git worktree list` shows only the main worktree
- [ ] All 5 removed files still exist locally on disk (just untracked)

---

## Technical Design

### Approach

Use `git rm --cached` to remove files from the index without deleting them from disk. Update `.gitignore` to prevent re-addition. Commit as a single cleanup commit on `dev`.

This is the simplest and safest approach — it does not rewrite history, so existing clones are unaffected. The files will remain in git history (they'll still be downloaded on a full clone) but won't be present in the working tree of future checkouts.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| `git rm --cached` + `.gitignore` | Simple, safe, no history rewrite | Files still in history (~54 MB on full clone) | **Chosen** |
| BFG Repo Cleaner / `git filter-repo` | Removes from history too, truly reduces clone size | Requires force-push, breaks all existing clones, risky for shared repo | Rejected — disproportionate risk for the benefit |
| Git LFS migration | Keeps files tracked but stored efficiently | Requires LFS setup on remote, adds complexity | Rejected — files don't need to be tracked at all |

### Architecture Changes

No code changes. Only affects:
- `.gitignore` — add 2 patterns
- Git index — remove 5 files from tracking

---

## Implementation Plan

### Phase 1: Worktree Cleanup
**Goal:** Remove stale worktrees from previous sessions

- [ ] Remove worktree `.claude/worktrees/agent-a3dff8eb`
- [ ] Remove worktree `.claude/worktrees/agent-a835f01d`
- [ ] Remove worktree `.claude/worktrees/agent-ac638211`
- [ ] Prune any remaining worktree references

**Dependencies:** None

### Phase 2: Remove Binary Files from Tracking
**Goal:** Untrack the 5 identified binary files without deleting them locally

- [ ] `git rm --cached docs/design/MNG_tracking_analysis-Shan-0403.pptx`
- [ ] `git rm --cached docs/design/MNG_tracking_analysis-Shan-0506.pptx`
- [ ] `git rm --cached docs/design/kinect_frame_averaging_insight.docx`
- [ ] `git rm --cached docs/design/social_touch_research_notes.docx`
- [ ] `git rm --cached media/cues_social_touch_semi_controlled.zip`

**Files Modified:**
- Git index only — files removed from tracking

**Dependencies:** None (can run in parallel with Phase 1)

### Phase 3: Update .gitignore
**Goal:** Prevent binary office files from being re-added

- [ ] Add `*.pptx` pattern to `.gitignore`
- [ ] Add `*.docx` pattern to `.gitignore`
- [ ] Verify `k4a.log` is caught by existing `*.log` pattern

**Files Modified:**
- `.gitignore` — add 2 lines in a new "Office documents" section

**Dependencies:** Phase 2 (gitignore should be updated after files are untracked)

### Phase 4: Commit
**Goal:** Create a single cleanup commit

- [ ] Stage `.gitignore` changes and the 5 file removals
- [ ] Commit with message: `chore: remove large binary files from tracking and update .gitignore`

**Dependencies:** Phase 2 + Phase 3

---

## Testing Plan

### Manual Verification
- [ ] `git ls-files -- "*.pptx" "*.docx"` returns no results
- [ ] `git ls-files -- "media/cues_social_touch_semi_controlled.zip"` returns no results
- [ ] `git status` shows the 5 files are NOT listed as untracked (gitignored)
- [ ] `ls docs/design/MNG_tracking_analysis-Shan-0403.pptx` confirms file still exists locally
- [ ] `git worktree list` shows only the main working tree
- [ ] `k4a.log` does not appear in `git status`

### Edge Cases
- [ ] Verify that `.gitignore` patterns don't accidentally exclude any tracked `.yaml` or other wanted files
- [ ] Confirm no other `.pptx` or `.docx` files exist elsewhere in the tree that should stay tracked

---

## Documentation Plan

- [ ] No README/CLAUDE.md updates needed — this is a housekeeping change
- [ ] The commit message documents the rationale

---

## Rollback Plan

1. **Before push:** `git reset HEAD~1` to undo the commit, then `git checkout HEAD -- .gitignore` to restore the old gitignore
2. **After push:** `git revert <commit-hash>` — this will re-add the files to tracking
3. **Data safety:** Files are never deleted from disk (`--cached` flag), so no data loss is possible

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Accidentally delete local files | Low | Medium | Using `git rm --cached` (not `git rm`) to preserve local copies |
| Gitignore pattern too broad | Low | Low | Patterns are specific (`*.pptx`, `*.docx`) and don't overlap with any tracked code files |
| Someone re-adds binary files later | Medium | Low | `.gitignore` prevents this; also documented in commit message |

---

## Appendix: Repository State Context

### dev → main gap (for future reference, not addressed in this plan)

`dev` is **282 commits ahead** of `main`. All dev work is pushed to `origin/dev`. This merge will also clean up:
- `code/src/preprocessing/stickers_analysis/__init__ copy.py` (accidental file only on main)
- `packages.txt` (2-line note only on main)
- `code/.../colorspace_metadata_template.json` (434 KB, deleted on dev)
- `pyproject.toml` location (moved from `code/` to root on dev)

### Plan lifecycle mismatches (not addressed in this plan)

These tracked plans are in the wrong lifecycle directory:
- `docs/development/plans/active/postprocessing-column-consolidation.md` → should be `completed/`
- `docs/development/plans/active/receptive-field-mapping.md` → should be `completed/`
- `docs/development/plans/pending/contact-point-forearm-projection.md` → should be `completed/`

### Lower-priority binary files (not addressed in this plan)

| File | Size | Notes |
|------|------|-------|
| `docs/design/semicontrolled_postprocessing_diagram.drawio` + `.drawio.png` | 380 KB | Binary diagram |
| `docs/design/specifications/ui.svg` | 103 KB | Binary SVG |
| `.pkl` model files (MANO_LEFT, MANO_RIGHT, hand_mesh_model) | 8 MB total | Needed by imported hand tracking code |

---

## References

- Related analysis: `.claude/plans/wondrous-plotting-seahorse.md` (scratch audit notes)
