# Plan: Branch Cleanup and Archival

**Date:** 2026-03-04
**Author:** Basil Duvernoy
**Status:** Completed
**Branch:** `chore/branch-cleanup-archival`

---

## Overview

Consolidate 20+ stale remote branches (from Aug 2025–Jan 2026) into a clean, maintainable remote state. This cleanup reduces cognitive load when reviewing branch status, clarifies which work is active vs. archived, and aligns branch management with the project's planning-driven workflow. The archival will be documented in `docs/development/plans/` to preserve decision history.

## Problem Statement

The repository has accumulated stale remote branches representing completed work, abandoned experiments, and cross-developer collaboration. These branches:
- Clutter `git branch -r` output (23 branches when only ~8 are actively maintained)
- Create confusion about what work is current vs. historical
- Represent different development phases and collaboration contexts (work from Sarah, SDAMcIntyre, internal infrastructure experiments)
- Are not integrated with the project's planning-driven workflow for feature tracking

Stale branches should be archived with full documentation so future developers can understand historical decisions.

## Goals

### In Scope
1. Audit all remote branches and categorize them (keep vs. archive)
2. Create an archive document in `docs/development/plans/archived/` with branch manifest
3. Delete archived remote branches using `git push origin --delete`
4. Delete corresponding local tracking branches (after remote cleanup confirmed)
5. Create an audit trail showing what was archived and why

### Out of Scope
- Rewriting or recovering content from archived branches (they remain in git history)
- Automated branch cleanup scripts (one-time manual operation)
- Changes to branch protection policies or naming conventions
- Migrating archived branch work to new branches

## Success Criteria

- [ ] Archive documentation created at `docs/development/plans/archived/2026-03-branch-cleanup-record.md`
- [ ] Archive document includes all branch names, commit hashes, dates, and category annotations
- [ ] All targeted remote branches deleted successfully (13 branches removed)
- [ ] Local tracking branches cleaned up (4 local branches deleted)
- [ ] Remote branch count reduced from 23 to ~10
- [ ] `git log --all --graph --oneline` confirms main dev line is clean
- [ ] Plan moved to `completed/` after verification

---

## Technical Design

### Approach

**Branching Strategy Reference:** The project follows a planning-driven workflow where:
- Feature branches are created from `dev` per the planning procedure
- Completed features are tracked via planning documents (`docs/development/plans/completed/`)
- Stale branches represent work that predates the current planning system or was abandoned

**Cleanup Strategy:**
1. Categorize branches by recency and status (last commit date, merge status)
2. Branches not touched since 2026-02-20 (2+ weeks) or earlier are candidates for archival
3. Keep recent branches (Feb 2026+) unless explicitly unmerged for extended periods
4. Document each archived branch with commit hash, date, and category for future reference
5. Delete remotes, then prune local tracking refs

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Delete without docs** | Fast, simple | Lose context, hard to recover if needed | Rejected |
| **Tag branches before delete** | Preserves commit objects, easy recovery | Clutters git tag space, not a standard practice | Rejected |
| **Archive doc only (keep branches)** | No deletion risk | Defeats purpose of cleanup; branches still visible | Rejected |
| **Delete + document (chosen)** | Cleans up remote, preserves decision history, aligns with planning workflow | Requires careful categorization upfront | **Chosen** |

### Architecture Changes

No code or structural changes. This is a repository maintenance operation.

**New artifact:**
- `docs/development/plans/archived/2026-03-branch-cleanup-record.md` — Manifest of archived branches with commit hashes, dates, categories, and context notes

---

## Implementation Plan

### Phase 1: Branch Audit and Classification
**Goal:** Categorize all branches and create archive documentation

- [ ] List all remote branches with dates and commit messages
- [ ] Classify branches as KEEP or ARCHIVE based on:
  - Recency (recent = last 2 weeks, archival candidates = 2+ weeks old or older)
  - Merge status (already merged into `dev` = safe to archive)
  - Purpose (feature, fix, refactor, infrastructure, collaboration)
- [ ] Create draft archive documentation with all branch metadata
- [ ] User review: confirm deletion list (especially `origin/chore/project-housekeeping`)

**Files Modified:**
- `docs/development/plans/archived/2026-03-branch-cleanup-record.md` — New archive manifest

**Dependencies:** None

### Phase 2: Remote Branch Deletion
**Goal:** Delete archived branches from origin

- [ ] Verify no active work depends on branches to be deleted
- [ ] Delete each archived remote branch: `git push origin --delete [branch-name]` (13 branches)
- [ ] Verify deletion succeeded: `git branch -r | wc -l` (should decrease to ~10)
- [ ] Confirm `origin/main` and `origin/dev` still exist and are correct

**Files Modified:** None (git operations only)

**Dependencies:** Phase 1

### Phase 3: Local Branch Cleanup
**Goal:** Remove local tracking branches for deleted remotes

- [ ] Delete local branches that tracked deleted remotes:
  - `git branch -d docs/reorganize-structure`
  - `git branch -d docs/planning-workflow-update`
  - `git branch -d chore/repo-readability-reorganization`
  - `git branch -d feature/sticker-depth-edge-gradient-bias`
- [ ] Prune stale tracking refs: `git remote prune origin`
- [ ] Verify clean state: `git branch -v` (should list only current dev)

**Files Modified:** None (git operations only)

**Dependencies:** Phase 2

### Phase 4: Documentation and Verification
**Goal:** Finalize archive documentation and move plan to completed

- [ ] Update archive document with final deletion summary
- [ ] Run final verification: `git log --all --graph --oneline -20`
- [ ] Commit archive document with message: `docs(git): archive stale branches from 2025–early 2026`
- [ ] Move this plan file to `docs/development/plans/completed/branch-cleanup-and-archival.md`
- [ ] Update plan status to `Completed`

**Files Modified:**
- `docs/development/plans/archived/2026-03-branch-cleanup-record.md` — Final version

**Dependencies:** Phase 3

---

## Testing Plan

### Verification Steps

- [ ] **Before cleanup:** `git branch -r | wc -l` shows ~23 branches
- [ ] **After remote delete:** `git branch -r | wc -l` shows ~10 branches
- [ ] **After local cleanup:** `git branch -v` shows only `dev` (and possibly `main`)
- [ ] **Commit integrity:** `git log --all --graph --oneline -30` shows clean history without deleted branches
- [ ] **Critical branches present:**
  - `origin/main` exists and points to correct commit
  - `origin/dev` exists and points to correct commit (HEAD)
  - All recent feature branches (Feb 2026+) still exist and correct

### Edge Cases

- [ ] No commits to deleted branches after archival
- [ ] Archive document is readable and complete
- [ ] User can reference archive record to understand historical branch purposes

---

## Documentation Plan

- [ ] Create archive manifest: `docs/development/plans/archived/2026-03-branch-cleanup-record.md`
- [ ] Archive manifest includes branch categories with brief explanatory notes
- [ ] Move completed plan to `docs/development/plans/completed/branch-cleanup-and-archival.md`
- [ ] No README changes needed (cleanup is internal housekeeping)

---

## Rollback Plan

If something goes wrong during cleanup:

1. **Before Phase 2:** Archive document exists but no branches deleted — safe to abort
2. **After Phase 2:** Deleted branches are still in git reflog for recovery:
   ```bash
   git reflog
   git branch [branch-name] [reflog-hash]
   git push origin [branch-name]
   ```
3. **After Phase 3:** Local branches deleted but can be recovered from `git reflog`

**No data loss risk** — deleted branches remain accessible in git history; deletion is non-destructive at the object level.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| User depends on a branch being kept | Medium | High (breaks workflow) | Phase 1 includes user review; ask about `origin/chore/project-housekeeping` specifically |
| Incorrect categorization of branch status | Low | Medium (might delete active work) | Phase 1 lists all branches with dates/messages; user approves before deletion |
| Local cleanup leaves tracking refs | Low | Low (cosmetic; fixed by `git remote prune`) | Phase 3 includes prune command; verification step confirms clean state |
| Deletion fails due to protected branches | Very Low | High (stops cleanup) | GitHub protection rules only apply to `main` and `dev`; deleting feature branches should succeed |

---

## User Decisions (Captured)

✅ **Unmerged `origin/chore/project-housekeeping` (Feb 2026):** Include in archive docs for user review; user decides whether to delete after seeing context

✅ **Local branch cleanup:** Delete all local tracking branches after remote deletion:
- `docs/reorganize-structure`
- `docs/planning-workflow-update`
- `chore/repo-readability-reorganization`
- `feature/sticker-depth-edge-gradient-bias`

✅ **Archive detail level:** Medium — table with branch names, commit hashes, dates, messages, plus category notes (feature, fix, experiment, infrastructure, collaboration)

---

## Branches to Archive

### Delete (13 remote branches)

| Branch | Last Commit | Date | Category | Reason |
|--------|-------------|------|----------|--------|
| `origin/dev-improve-sticker-tracking` | d40e518 | 2025-12-12 | Feature | Incomplete; superseded by newer sticker-tracking work |
| `origin/dev-fix-gui-normal-pointcloud` | a09606e | 2025-11-27 | Fix | Merged (PR #45), no longer needed |
| `origin/docker-implementation` | 14b5bde | 2025-10-04 | Infrastructure | Old infrastructure experiment; not actively used |
| `origin/dev_background-removal_basil` | f50bbfc | 2025-09-03 | Feature | Merged (PR #7); archived work from Sept 2025 |
| `origin/sarah_separate_background` | e6a50dc | 2025-08-29 | Collaboration | Cross-developer collaboration; work merged or completed |
| `origin/sarah_sticker_depth_position` | 639bb0d | 2025-08-28 | Collaboration | Cross-developer collaboration; work merged or completed |
| `origin/tiff-export-fix` | 7e844fe | 2025-08-28 | Fix | Stale fix; no recent activity; likely merged |
| `origin/install_and_project_folder_improvements` | 90642ba | 2025-08-28 | Chore | Infrastructure/setup work; archived |
| `origin/performance-testing` | 082b304 | 2023-01-16 | Experiment | Extremely old performance testing branch (Jan 2023) |
| `origin/chore/project-housekeeping` | 6dd44f2 | 2026-02-20 | Chore | Unmerged for 2+ weeks; review and decide |
| *(optional on user input)* | | | | |

### Keep (10 remote branches)

| Branch | Last Commit | Date | Status |
|--------|-------------|------|--------|
| `origin/main` | 48d497f | 2025-12-03 | Production baseline |
| `origin/dev` | 4094fbe | 2026-02-24 | Active integration |
| `origin/docs/planning-workflow-update` | 90eb902 | 2026-02-23 | Recent docs |
| `origin/docs/reorganize-structure` | 14dd335 | 2026-02-22 | Recent docs |
| `origin/chore/repo-readability-reorganization` | e4bafce | 2026-02-22 | Recent chore |
| `origin/feature/sticker-depth-edge-gradient-bias` | c99088c | 2026-02-23 | Recent feature |
| `origin/feature/neural-kinect-viewer` | abf97c4 | 2026-02-18 | Recent feature |
| `origin/fix/initial-render-clipping` | 3734dd2 | 2026-02-19 | Recent fix |
| `origin/fix/somatosensory-custom-colors-import` | fe3228b | 2026-02-18 | Recent fix |
| `origin/HEAD` | (points to main) | — | Metadata |

---

## References

- **Git Workflow:** `docs/development/git/git-workflow.md`
- **Planning Procedure:** `docs/development/planning-procedure.md`
- **Project Conventions:** `CLAUDE.md`
