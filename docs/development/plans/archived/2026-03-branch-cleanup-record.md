# Branch Cleanup Record — March 2026

**Date:** 2026-03-04
**Author:** Basil Duvernoy
**Related Plan:** `docs/development/plans/completed/branch-cleanup-and-archival.md`

---

## Summary

Archived 10 stale remote branches accumulated from Jan 2023 through Feb 2026. These branches represented completed features, merged PRs, cross-developer collaboration, and abandoned experiments. The cleanup reduced the remote branch count from 20 to 10.

---

## Archived Branches

| Branch | Tip Commit | Date | Category | Notes |
|--------|-----------|------|----------|-------|
| `dev-improve-sticker-tracking` | d40e518 | 2025-12-12 | Feature | Incomplete; superseded by `feature/sticker-depth-edge-gradient-bias` which was merged into `dev` |
| `dev-fix-gui-normal-pointcloud` | a09606e | 2025-11-27 | Fix | Merged via PR #45 (forearm pointcloud cleaning); no longer needed |
| `docker-implementation` | 14b5bde | 2025-10-04 | Infrastructure | Old Docker infrastructure experiment; tip commit is a merge from `SDAMcIntyre/hand_selection_hotfix`, not actively maintained |
| `dev_background-removal_basil` | f50bbfc | 2025-09-03 | Feature | Merged via PR #7 (forearm extraction ROI); work fully integrated |
| `sarah_separate_background` | e6a50dc | 2025-08-29 | Collaboration | Cross-developer (Sarah) background separation work; completed collaboration |
| `sarah_sticker_depth_position` | 639bb0d | 2025-08-28 | Collaboration | Cross-developer (Sarah) sticker depth position work; completed collaboration |
| `tiff-export-fix` | 7e844fe | 2025-08-28 | Fix | Stale fix with no recent activity; work superseded or abandoned |
| `install_and_project_folder_improvements` | 90642ba | 2025-08-27 | Chore | Setup/install improvements; work integrated into main development line |
| `performance-testing` | 082b304 | 2023-01-16 | Experiment | Extremely old (Jan 2023) performance testing branch; no ongoing relevance |
| `chore/project-housekeeping` | 6dd44f2 | 2026-02-20 | Chore | All commits fully merged into `dev`; branch had no unmerged history relative to `dev` |

---

## Branches Retained

| Branch | Tip Commit | Date | Status |
|--------|-----------|------|--------|
| `main` | 48d497f | 2025-12-03 | Production baseline |
| `dev` | 4094fbe | 2026-02-24 | Active integration branch |
| `docs/planning-workflow-update` | 90eb902 | 2026-02-23 | Recent docs work |
| `docs/reorganize-structure` | 14dd335 | 2026-02-22 | Recent docs work |
| `chore/repo-readability-reorganization` | e4bafce | 2026-02-22 | Recent chore work |
| `feature/sticker-depth-edge-gradient-bias` | c99088c | 2026-02-23 | Recent feature work |
| `feature/neural-kinect-viewer` | abf97c4 | 2026-02-18 | Recent feature work |
| `fix/initial-render-clipping` | 3734dd2 | 2026-02-19 | Recent fix |
| `fix/somatosensory-custom-colors-import` | fe3228b | 2026-02-18 | Recent fix |
| `HEAD` | → main | — | Metadata ref |

---

## Recovery

All archived branches remain accessible in git history. To recover a branch:

```bash
git branch <branch-name> <tip-commit-hash>
git push origin <branch-name>
```

Example: `git branch dev-improve-sticker-tracking d40e518`
