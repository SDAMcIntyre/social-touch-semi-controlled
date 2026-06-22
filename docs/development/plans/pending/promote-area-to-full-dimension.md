# Plan: Promote Contact Area to Full L3 Analysis Dimension

**Date:** 2026-06-21
**Author:** Basil Duvernoy
**Status:** Draft
**Base Branch:** `feature/family2-response-summary`
**Branch:** `docs/promote-area-dimension`

---

## Overview

Contact area is currently framed as a "dismissal variable" across the article-scoping docs and development plans — its sole purpose is to show a null result and justify dropping it from further analysis. This plan promotes area to a full L3 dimension, equal to velocity and depth, across all documentation. The pipeline already supports area tuning end-to-end; only documentation framing needs to change.

## Problem Statement

The figure specification (`07-figure-specification.md`) treats area differently from velocity and depth at L3:
- Velocity/depth: "1D tuning curve + 2D grid"
- Area: "1D tuning curve only — to dismiss it"

A dedicated "Area Dismissal Figure" section instructs placing the null result early to justify dropping area. Family 1's success criteria expect "no significant correlation." Family 2 doesn't mention area tuning at all. This framing contradicts the decision to treat area as a full analysis dimension.

## Goals

### In Scope
1. Remove all "dismissal" framing for contact area across article-scoping docs
2. Make area equal to velocity and depth at L3 in the figure specification
3. Add area tuning curves to Family 2 plan (currently missing)
4. Update Family 1 plan success criteria to treat area as exploratory, not null-hypothesis

### Out of Scope
- Code changes (pipeline already supports `contact_area_mean` in both spatial and response tuning)
- 2D grids involving area (same sparsity constraint as velocity x depth — ~295 touches/session makes 2D infeasible)
- Pressure calibration issues (separate concern)
- Temporal Approach B (deferred, separate plan)

## Success Criteria

- [ ] `07-figure-specification.md` L3 lists area with same notation as velocity and depth
- [ ] "Area Dismissal Figure" section removed or reframed as regular tuning analysis
- [ ] `family1-rf-spatial-figures.md` success criteria treat area as exploratory (not null-hypothesis)
- [ ] `family2-approach-a-response-summary.md` includes area in L3 tuning curves
- [ ] `05-article-structure.md` integrates area into main results (not as a standalone dismissal)
- [ ] No "dismiss" or "dropping area" language remains in any updated file

---

## Technical Design

### Approach

Documentation-only edits across 4 files. Each edit replaces dismissal framing with equal-status framing matching velocity and depth. No new sections needed — area slots into existing L3 structures.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Docs-only (remove dismissal framing) | No code changes, pipeline ready, minimal risk | None identified | Chosen |
| Docs + add 2D area grids | More thorough coverage | 2D grids infeasible due to data sparsity | Rejected |
| Keep dismissal but add temporal check | Partial promotion | Inconsistent — either area matters or it doesn't | Rejected |

### Architecture Changes

None. All changes are in markdown documentation files.

---

## Implementation Plan

### Phase 1: Figure Specification
**Goal:** Make area equal to velocity and depth in the L0-L3 hierarchy

- [ ] Update L3 entry from "Impact of area (1D tuning curve only — to dismiss it)" to "Impact of area (1D tuning curve)"
- [ ] Remove or reframe the "Area Dismissal Figure" section (lines 92-97) — convert to a regular L3 tuning note if any content is worth keeping
- [ ] Verify velocity and depth entries match the new area entry in structure

**Files Modified:**
- `docs/article-scoping/07-figure-specification.md` — L3 hierarchy line + Area Dismissal section

**Dependencies:** None

### Phase 2: Development Plans
**Goal:** Update both active family plans to treat area equally

- [ ] `family1-rf-spatial-figures.md`: Update success criteria — remove "shows flat/non-significant relationship" for area; replace with exploratory framing matching velocity/depth criteria
- [ ] `family1-rf-spatial-figures.md`: Update manual verification — remove expectation of "no significant correlation (p > 0.05)"
- [ ] `family2-approach-a-response-summary.md`: Add area to L3 tuning curves (currently velocity and depth only); mirror the existing velocity/depth entries

**Files Modified:**
- `docs/development/plans/active/family1-rf-spatial-figures.md` — success criteria + verification section
- `docs/development/plans/active/family2-approach-a-response-summary.md` — add area to L3

**Dependencies:** Phase 1 (figure spec establishes the canonical framing)

### Phase 3: Article Structure
**Goal:** Integrate area into the main results narrative

- [ ] Update the results section to list area alongside velocity/depth as a regular tuning parameter
- [ ] Remove any "novel — underexplored" hedging that implies area will be dismissed

**Files Modified:**
- `docs/article-scoping/05-article-structure.md` — results section bullet points

**Dependencies:** Phase 1

---

## Testing Plan

### Manual Verification
- [ ] Search all updated files for "dismiss" — zero hits expected
- [ ] Confirm L3 in `07-figure-specification.md` shows area, velocity, depth with identical structure
- [ ] Confirm `family2-approach-a-response-summary.md` now references area tuning curves
- [ ] Verify DAG config (`configs/analyse_workflow_processing_dag.yaml`) already has `contact_area_mean` in both `stimulus_response_tuning.tuning_features` and `spatial_tuning_rf_metrics.tuning_features` (read-only check, no changes)

### Edge Cases
- [ ] Ensure no dismissal framing was introduced elsewhere in docs (grep across `docs/article-scoping/` and `docs/development/plans/`)

---

## Documentation Plan

- [ ] This IS the documentation change — no additional docs needed
- [ ] No CLAUDE.md updates required (no architectural changes)

---

## Rollback Plan

1. `git revert` the single docs commit
2. No data, code, or config dependencies — pure markdown changes

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Area genuinely has no signal (flat relationship confirmed in data) | Medium | Low | Framing change is about analytical stance, not results; if area shows no signal, that's a finding worth reporting rather than a pre-assumed dismissal |
| Inconsistency with other docs not caught | Low | Low | Grep for "dismiss" across all docs after edits |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Figure spec | ~15 min | None |
| Phase 2: Dev plans | ~20 min | Phase 1 |
| Phase 3: Article structure | ~10 min | Phase 1 |

---

## References

- Analysis discussion plan: `.claude/plans/analyse-docs-article-scoping-i-want-fancy-liskov.md`
- Pipeline capabilities doc: `docs/article-scoping/02-pipeline-capabilities.md`
- DAG config (area already enabled): `configs/analyse_workflow_processing_dag.yaml`
- Knowledge base: `docs/development/knowledge-base/note-somatosensory-units-and-calculations.md` (area units = mm², 30 Hz sampling)
