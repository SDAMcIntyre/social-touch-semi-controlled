# Plan: Colormapped UV Diagnostic Panels (Steps 4-5)

**Date:** 2026-05-21
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-05-24 18:03
**Base Branch:** `feature/slim-uv-diagnostic-figures`
**Branch:** `feature/slim-uv-colormapped-diagnostic-panels`

---

## Overview

Replace the flat blue scatter dots in SLIM UV diagnostic steps 4-5 with
colormapped triangulated meshes (PolyCollection + viridis) so the researcher
can track how mesh regions move between UV initialisation and final SLIM
optimisation. Steps 1-3 already render readable 3D meshes; step 6 already
uses colormapped PolyCollection for distortion. Steps 4-5 are the gap.

## Problem Statement

Steps 4 and 5 of the SLIM UV diagnostic figures render UV coordinates as
`ax.scatter(..., c="steelblue", s=1)` — a cloud of uniform blue dots with no
mesh connectivity and no value-based coloring. This makes it impossible to see:

- Which regions compress or stretch during SLIM optimisation
- How the mesh topology is preserved between init and final UV
- Whether specific areas flip or fold

The 3D panel in step 4 (left) already colors the mesh by init-u with viridis
and is highly readable. The 2D panels should match.

## Goals

### In Scope
1. Replace scatter with PolyCollection + triplot in step 4 right panel
2. Replace scatter with PolyCollection + triplot in step 5 both panels
3. Color all UV panels by init-u (viridis) for spatial correspondence

### Out of Scope
- Changing steps 1-3 (3D mesh views — already good)
- Changing step 6 (distortion panels — already colormapped)
- Changing the existing QC figures (`save_slim_qc_figures`)
- Adding colorbars (the coloring is for spatial correspondence, not quantitative reading)

## Success Criteria

- [x] Step 4 right panel shows triangulated mesh colored by init-u (not blue dots)
- [x] Step 5 both panels show triangulated meshes colored by init-u
- [x] Same vertex retains the same color across steps 4-5 panels
- [x] Thin white mesh edges visible for connectivity
- [ ] All 12 sessions generate diagnostic figures without error

---

## Technical Design

### Approach

Use the `PolyCollection(polys, array=face_u, cmap="viridis")` pattern already
established in `plot_slim_distortion_panel()` (lines 109, 125). Add thin white
`triplot` edges for mesh connectivity, matching the QC panel pattern (line 74).
Color by init-u (the u-coordinate from UV initialisation) so vertex identity
is preserved across all panels.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| PolyCollection + viridis init-u | Consistent with step 6 pattern; shows mesh + spatial correspondence | Slightly more computation | **Chosen** |
| Scatter colored by init-u | Minimal code change | No mesh edges, dots can overlap/gap | Rejected |
| tripcolor instead of PolyCollection | Single-call convenience | Not the established pattern in this file | Rejected |

### Architecture Changes

None. Single file modified, no new functions, no new dependencies.

### Knowledge Base Constraints

From `note-3d-to-2d-surface-projection-algorithms.md`:
- Use per-face mean of vertex values for face coloring (`face_u = u_norm[F].mean(axis=1)`)
- Viridis is appropriate for perceptual uniformity

---

## Implementation Plan

### Phase 1: Replace scatter with colormapped mesh
**Goal:** All UV panels in steps 4-5 render as colormapped triangulated meshes.

- [x] Step 4 (`_plot_step4_uv_init`): move `face_u` computation before both panels; replace right-panel scatter with PolyCollection + triplot
- [x] Step 5 (`_plot_step5_slim_final`): add `F` parameter; compute per-face init-u; replace both scatter calls with PolyCollection + triplot
- [x] Call site (`save_slim_diagnostic_figures`): pass `F_trim` to step 5

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/surface/slim_qc_figures.py`
  - `_plot_step4_uv_init` (lines 333-379): replace scatter at line 370 with PolyCollection + triplot + autoscale_view
  - `_plot_step5_slim_final` (lines 382-408): add `F` param, replace scatter at lines 393 and 399 with PolyCollection + triplot + autoscale_view
  - `save_slim_diagnostic_figures` (line 503): pass `F_trim` to step 5 call

**Dependencies:** None

---

## Testing Plan

### Unit Tests
- [ ] Existing `test_forearm_slim_uv.py` passes (no regressions)

### Manual Verification
- [ ] Run `precompute_forearm_slim_uv` with `save_diagnostics: true` on one session
- [ ] Open `step4_uv_initialization.png` — right panel shows colormapped triangulated mesh
- [ ] Open `step5_slim_uv_final.png` — both panels show colormapped triangulated meshes
- [ ] Colors match between step 4 right, step 5 left, and step 5 right (same init-u mapping)
- [ ] White mesh edges are visible but not distracting

---

## Documentation Plan

- [ ] No documentation changes needed (this is a visual quality improvement to existing figures)

---

## Rollback Plan

Revert the single file `slim_qc_figures.py`. No data migrations, no API changes.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| PolyCollection rendering slower than scatter for large meshes | Low | Low | Already used in step 6 without issue; meshes are <15k faces |
| `autoscale_view()` changes axis limits vs previous scatter | Low | Low | Same method used in distortion panel; produces correct bounds |
