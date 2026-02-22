# Plan: Repository Readability Reorganization

**Date:** 2026-02-22
**Author:** Claude Opus 4.6
**Status:** Completed 2026-02-22
**Branch:** `chore/repo-readability-reorganization`

---

## Overview

A housekeeping initiative to bring the repository into a clean, conventional
state: remove stale artifacts, consolidate three scattered documentation
directories into one, apply PEP 8 naming to the two remaining PascalCase
Python files, normalise the misspelled `analyse/` package, add missing
`__init__.py` sentinels, fix docstring errors, and add missing READMEs for
key directories. Five independently committable phases, ordered from zero-risk
to progressively more impactful changes.

## Problem Statement

Over time the repository has accumulated:

| Issue | Impact |
|-------|--------|
| Copy-files (`* copy*.py`), build artifacts (`code/build/`), personal workspace files | Visual noise; confuses new contributors |
| `.gitignore` gaps that let some of these be tracked | Recurrence risk |
| Three top-level documentation roots (`docs/`, `documentation/`, `skeleton/`) | Unclear canonical location; discoverability |
| Two PascalCase Python source files (`ROIManager.py`, `KinectLEDValidation.py`) | PEP 8 violation; inconsistency with 174 other files |
| `analyse/` package (British) while rest of codebase uses American English | Confusing imports |
| 12 directories missing `__init__.py` | Implicit namespace packages where explicit packages are intended |
| Wrong module path and typo in `stickers_analysis/__init__.py` | Misleading developer documentation |
| No README for `configs/` (40+ YAML files) or `code/scripts/` (10 entry points) | Onboarding friction |
| Orphaned root-level files (`packages.txt`, `k4a.log`, `*.code-workspace`) | Clutter |

None affect runtime correctness today, but they degrade maintainability
and first-impression readability.

## Goals

### In Scope

1. Delete all stale/copy files and build artifacts
2. Update `.gitignore` to prevent recurrence
3. Move `documentation/` into `docs/design/` and `skeleton/` into `docs/skeleton/` (keep ignored)
4. Rename two PascalCase files to snake_case, update imports
5. Rename `analyse/` to `analysis/`, update the one external import
6. Add missing `__init__.py` files (or remove empty directories)
7. Fix `stickers_analysis/__init__.py` comments and docstring
8. Add README files for `configs/` and `code/scripts/`
9. Remove or absorb orphaned root-level files

### Out of Scope

- Refactoring module internals or APIs
- Changing any runtime behaviour
- Reorganising `code/src/` package hierarchy beyond the `analyse` rename
- Adding a formal `tests/` directory (separate initiative)
- Standardising `LED` vs `Led` class-name capitalisation (cosmetic; low priority)
- Upgrading dependencies or adding CI/CD

## Success Criteria

- [ ] No files matching `* copy*` exist in the tracked tree
- [ ] `code/build/` is not tracked (may exist locally as build artifact)
- [ ] Only one top-level documentation directory (`docs/`) exists; `documentation/` and `skeleton/` are gone from root
- [ ] `docs/skeleton/` is gitignored
- [ ] No PascalCase `.py` filenames remain under `code/src/`
- [ ] `python -c "from preprocessing.led_analysis import ROIManager, LedSignalValidator"` succeeds
- [ ] `python -c "from analysis.touch_analytics import generate_unified_summary"` succeeds
- [ ] Every directory under `code/src/` that contains `.py` files has an `__init__.py`
- [ ] `grep -r 'from analyse' code/ --include='*.py'` returns zero results
- [ ] `packages.txt`, `secondary-window-theme.code-workspace` no longer tracked
- [ ] `configs/README.md` and `code/scripts/README.md` exist

---

## Technical Design

### Approach

Straight-line file operations (delete, move, rename) plus minimal edits to
import statements and docstrings. No new abstractions, no dependency changes.

Each phase is a single atomic commit. Phases are ordered so earlier ones have
strictly lower risk.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Single commit for everything | Simpler history | Hard to review; hard to bisect | Rejected |
| Skip `analyse/` rename | Less churn | Leaves permanent inconsistency | Rejected |
| Convert to top-level `src/` layout | Cleaner | Massive blast radius; separate initiative | Rejected |
| Add linter pre-commit hook in same PR | Prevents future drift | Orthogonal scope | Rejected (future work) |

### Knowledge Base Relevance Check

Reviewed all entries in `docs/development/knowledge-base/README.md`:

- **note-cupy-import-order** — Not affected. No new import-order changes; renamed files preserve existing import structure.
- **note-open3d-scenewidget-layout** — Not relevant.
- **bug-cupy-bool8-import-order** — Not affected.
- **bug-hue-circle-drag-not-working** — Not relevant.
- **bug-neural-kinect-viewer-initial-render** — Not relevant.

No knowledge-base constraints apply to this plan.

---

## Implementation Plan

### Phase 1: Delete stale files and fix `.gitignore`

**Goal:** Remove all dead files and tighten ignore rules. Zero functional impact.

**Commit:** `chore: remove stale copy-files, build artifacts, and tighten .gitignore`

**Tasks:**

- [ ] Delete untracked copy file:
  - `code/src/preprocessing/forearm_extraction/arm_segmentation copy.py`
- [ ] `git rm` tracked copy files:
  - `code/scripts/__misc/test_interactive_xyz_stickers_position_short copy.py`
  - `code/scripts/__misc/test_interactive_xyz_stickers_position_short copy 2.py`
  - `code/build/lib/preprocessing/stickers_analysis/__init__ copy.py`
- [ ] Remove build artifacts from index: `git rm -r --cached code/build/`
  then `rm -rf code/build/`
- [ ] `git rm packages.txt` (content absorbed into `code/scripts/README.md` in Phase 5)
- [ ] `git rm secondary-window-theme.code-workspace`
- [ ] `git rm --cached k4a.log` (already matched by `*.log` rule; remove from index only)
- [ ] Add `*.code-workspace` to `.gitignore`

**Files modified:** `.gitignore` (1 line added)
**Files deleted:** 4 copy-files, `code/build/` tree, `packages.txt`, `*.code-workspace`, `k4a.log` (from index)

**Dependencies:** None

---

### Phase 2: Consolidate documentation directories

**Goal:** Single `docs/` root for all documentation.

**Commit:** `docs: consolidate documentation/ and skeleton/ under docs/`

**Tasks:**

- [ ] Create `docs/design/` and `docs/design/specifications/`
- [ ] `git mv` all 5 files from `documentation/` into `docs/design/`:
  - `documentation/MNG_tracking_analysis-Shan-0403.pptx` → `docs/design/`
  - `documentation/MNG_tracking_analysis-Shan-0506.pptx` → `docs/design/`
  - `documentation/semicontrolled_postprocessing_diagram.drawio` → `docs/design/`
  - `documentation/semicontrolled_postprocessing_diagram.drawio.png` → `docs/design/`
  - `documentation/specifications/ui.svg` → `docs/design/specifications/`
- [ ] Move orphaned `docs/kinect_frame_averaging_insight.docx` → `docs/design/`
- [ ] Move `skeleton/` → `docs/skeleton/` (filesystem move; contents are gitignored)
- [ ] Update `.gitignore`: `/skeleton/` → `/docs/skeleton/`
- [ ] Update `docs/development/planning-procedure.md`: fix the relative link to
  `skeleton/topics/04-planning-process.md` (line 492) — adjust path or remove
  if inaccessible since it's gitignored
- [ ] Create `docs/design/README.md` — short index of the legacy design files

**Files moved:** 6 files from `documentation/`, 1 within `docs/`, `skeleton/` directory
**Files created:** `docs/design/README.md`
**Files modified:** `.gitignore` (1 line changed), `docs/development/planning-procedure.md` (1 link)

**Dependencies:** None

---

### Phase 3: Python source renames

**Goal:** PEP 8 file naming; consistent American English package name.

**Commit:** `refactor: rename PascalCase files to snake_case and analyse/ to analysis/`

**Tasks:**

- [ ] Rename PascalCase files:
  - `git mv code/src/preprocessing/led_analysis/core/ROIManager.py code/src/preprocessing/led_analysis/core/roi_manager.py`
  - `git mv code/src/preprocessing/led_analysis/gui/KinectLEDValidation.py code/src/preprocessing/led_analysis/gui/kinect_led_validation.py`
- [ ] Update `code/src/preprocessing/led_analysis/__init__.py`:
  - Line 1: `from .core.ROIManager import ROIManager` → `from .core.roi_manager import ROIManager`
  - Line 7: `from .gui.KinectLEDValidation import LedSignalValidator` → `from .gui.kinect_led_validation import LedSignalValidator`
- [ ] Rename package: `git mv code/src/analyse code/src/analysis`
- [ ] Rename script: `git mv code/scripts/analyse_workflow.py code/scripts/analysis_workflow.py`
- [ ] Update `code/scripts/analysis_workflow.py` line 29:
  `from analyse.touch_analytics import (` → `from analysis.touch_analytics import (`

**Blast radius verification (pre-commit):**
- `grep -r 'from analyse\b' code/ --include='*.py'` — must return nothing
- `grep -r 'ROIManager\.py\|KinectLEDValidation\.py' code/ --include='*.py'` — must return nothing

**Files renamed:** 4 files
**Files modified:** 2 files (`led_analysis/__init__.py`, `analysis_workflow.py`)

**Dependencies:** None (independent of Phases 1-2)

---

### Phase 4: Package structure fixes

**Goal:** Every Python directory that contains `.py` files has an `__init__.py`;
fix docstring errors.

**Commit:** `fix: add missing __init__.py files and fix stickers_analysis docstring`

**Tasks:**

- [ ] Create empty `__init__.py` files (single line: blank or comment) in:
  1. `code/src/analysis/__init__.py`
  2. `code/src/preprocessing/common/data_access/__init__.py`
  3. `code/src/preprocessing/common/gui/__init__.py`
  4. `code/src/preprocessing/forearm_extraction/data_access/__init__.py`
  5. `code/src/preprocessing/forearm_extraction/models/__init__.py`
  6. `code/src/preprocessing/motion_analysis/hand_tracking/hamer_liu_client/__init__.py`
  7. `code/src/primary_processing/data_access/__init__.py`
  8. `code/src/primary_processing/models/__init__.py`
  9. `code/src/utils/gui/__init__.py`
  10. `code/src/utils/pipeline/__init__.py`
  11. `code/src/utils/pipeline/monitoring/__init__.py`
- [ ] Delete empty directory:
  `code/src/preprocessing/motion_analysis/tactile_quantification/data_access/`
  (confirmed empty and unreferenced)
- [ ] Fix `code/src/preprocessing/stickers_analysis/__init__.py`:
  - Line 1: `# preprocessing/stickers_analysis/roi/__init__.py` →
    `# preprocessing/stickers_analysis/__init__.py`
  - Docstring: `from he 'roi' namespace, decoupling their code from .roiour internal file structure.` →
    `from the 'roi' namespace, decoupling their code from our internal file structure.`
  - `__all__` comment: `# Define what gets imported with 'from .roi. import *'` →
    `# Define what gets imported with 'from stickers_analysis import *'`

**Files created:** 11 `__init__.py` files
**Files modified:** 1 file (`stickers_analysis/__init__.py`)
**Directories deleted:** 1 empty directory

**Dependencies:** Phase 3 (needs `analysis/` to exist before creating its `__init__.py`)

---

### Phase 5: Documentation additions

**Goal:** Add missing README files for `configs/` and `code/scripts/`.

**Commit:** `docs: add README files for configs/ and code/scripts/`

**Tasks:**

- [ ] Create `configs/README.md`:
  - Describe purpose (YAML config files for the processing pipeline)
  - Explain directory structure (`forearm_configs/`, `kinect_configs/`, `_dag_templates/`)
  - Note the `.gitignore` rule (per-session configs are ignored; only root-level configs tracked)
  - Reference config-loading code (`primary_processing.KinectConfigFileHandler`, etc.)
- [ ] Create `code/scripts/README.md`:
  - Describe the numbered subdirectory convention (`_1_acquisition/` through `_5_postprocessing/`)
  - List the top-level workflow scripts and their purpose
  - Note the `__misc/` directory (ad-hoc test/development scripts, gitignored)
  - Include the ffprobe dependency note (absorbed from deleted `packages.txt`):
    "LED ROI analysis requires `ffprobe` (FFmpeg) on PATH."

**Files created:** 2 README files

**Dependencies:** Phase 1 (absorbs `packages.txt` content)

---

## Testing Plan

### After Phase 1

- [ ] `git status` shows clean working tree
- [ ] `test ! -f packages.txt && test ! -d code/build/` — confirmed gone

### After Phase 3

- [ ] `python -c "from preprocessing.led_analysis import ROIManager, LedSignalValidator; print('OK')"` succeeds
- [ ] `python -c "from analysis.touch_analytics import generate_unified_summary; print('OK')"` succeeds
- [ ] `grep -r 'from analyse\b' code/ --include='*.py'` — zero results
- [ ] `grep -r 'ROIManager\.py\|KinectLEDValidation\.py' code/ --include='*.py'` — zero results

### After Phase 4

- [ ] `python -c "from preprocessing.stickers_analysis import ROIAnnotationFileHandler; print('OK')"` succeeds
- [ ] Spot-check that new `__init__.py` files exist in all 11 directories

### After all phases

- [ ] Only one top-level doc directory: `ls -d docs/ documentation/ skeleton/` → only `docs/` exists
- [ ] Full import smoke test:
  ```bash
  python -c "from preprocessing.led_analysis import *"
  python -c "from preprocessing.stickers_analysis import *"
  python -c "from analysis.touch_analytics import *"
  python -c "from primary_processing import *"
  ```

---

## Documentation Plan

- [ ] Phase 2 creates `docs/design/README.md`
- [ ] Phase 5 creates `configs/README.md` and `code/scripts/README.md`
- [ ] This plan document moves to `docs/development/plans/completed/` when shipped

---

## Rollback Plan

Each phase is a single commit. Rollback for any phase:

```bash
git revert <commit-hash>
```

Phases are ordered by increasing risk and independently revertible — a failure
in Phase 3 does not require reverting Phases 1-2.

For Phase 3 (Python renames), if downstream breakage is discovered after merge:
1. `git revert` the Phase 3 commit
2. Investigate the missed import reference
3. Re-apply with the fix included

No database, binary state, or external service is affected by any phase.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Missed import reference to `analyse` | Low | Medium — runtime ImportError | Pre-commit `grep -r 'from analyse\b'` across entire repo |
| Missed import of PascalCase filename | Low | Medium — runtime ImportError | Pre-commit `grep -r 'ROIManager\.py\|KinectLEDValidation\.py'` |
| `skeleton/` move breaks someone's local setup | Very low | Low — already gitignored, local-only | Document in commit message |
| `code/build/` removal from index confuses developer | Very low | None — regenerated by `pip install -e .` | Note in commit message |
| Adding `__init__.py` changes namespace package semantics | Very low | Low — none of these are namespace packages | Verify no `pkgutil`-style namespace usage |
| `analyse_workflow.py` rename breaks external references | Low | Low — internal tooling only | Document in commit message |

---

## References

- Previous reorganization: `docs/development/plans/completed/docs-reorganization.md`
- Planning procedure: `docs/development/planning-procedure.md`
- Knowledge base: `docs/development/knowledge-base/README.md`
