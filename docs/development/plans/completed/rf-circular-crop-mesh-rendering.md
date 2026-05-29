# Plan: RF Circular Crop Mesh Rendering

**Date:** 2026-05-27
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-05-29 08:51
**Base Branch:** `feature/stimulus-iff-tuning-curves`
**Branch:** `feature/rf-circular-crop-mesh-rendering`

---

## Overview

Replace the scatter pointcloud forearm background with a solid triangulated mesh surface in the circular RF population crop renderer, and make the heatmap overlay fully opaque. Both fixes target the same function in a single file.

## Problem Statement

The `_rf_population_<type>_circular_centroid.png` (and `_peak`) images have two visual defects:

1. **Forearm rendered as pointcloud** — `ax.scatter()` draws individual UV vertices with visible gaps between them, instead of a continuous surface. The SLIM mesh faces are already passed to the function but unused for rendering.
2. **Heatmap is semi-transparent** — `alpha=0.8` on the `pcolormesh` lets the scatter dots bleed through beneath the heatmap overlay. Only the heatmap should be visible in the contacted region.

## Goals

### In Scope
1. Render the forearm background as a solid triangulated mesh surface using `PolyCollection`
2. Make the heatmap overlay fully opaque (`alpha=1.0`) so it fully occludes the forearm beneath it
3. Update the function docstring to reflect the new rendering approach

### Out of Scope
- Changing the heatmap colormap, normalization, or interpolation method
- Modifying the circular clip path logic
- Changes to the pipeline call site or function signature
- Changes to other renderers (2D scatter+interpolated panels, composite, etc.)

## Success Criteria

- [ ] Forearm background is a continuous surface with skin-coloured triangles (no gaps between points)
- [ ] Heatmap fully occludes the forearm surface beneath it (no forearm features visible through the heatmap)
- [ ] Circular boundary is clean — no triangle edges protrude beyond the circle
- [ ] Grey fallback (`vertex_colors=None`) still produces a solid grey surface

---

## Technical Design

### Approach

Use `matplotlib.collections.PolyCollection` to render filled triangles from the SLIM UV mesh, with per-face colours averaged from vertex colours. This is the established pattern in the codebase — used in `slim_qc_figures.py`, `compare_flattening_methods.py`, and `flatten_forearm_sandbox.py`.

The existing circular clip path (`matplotlib.patches.Circle` applied via `set_clip_path()`) handles both `PolyCollection` and `QuadMesh` (from `pcolormesh`) since both appear in `ax.collections`.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| `PolyCollection` with per-face RGBA | Established codebase pattern; direct RGBA control; clean edges | Per-face colour is flat (no Gouraud interpolation) | **Chosen** |
| `ax.tripcolor` with scalar mapping | Single-call convenience; supports Gouraud shading | Not the established pattern (explicitly rejected in `slim-uv-colormapped-diagnostic-panels.md`); requires scalar-to-colour indirection for RGBA vertex colours | Rejected |
| Increase scatter point size | Minimal code change | Still produces gaps at varying densities; no true surface | Rejected |

### Architecture Changes

None. The function signature, call site, and pipeline orchestration remain unchanged. This is a rendering-only fix within the existing function body.

---

## Implementation Plan

### Phase 1: Mesh rendering and opacity fix
**Goal:** Replace scatter with mesh surface and make heatmap opaque
**Started:** 2026-05-28
**Completed:** 2026-05-28

**Tasks:**
- [x] Task 1.1 — Add `from matplotlib.collections import PolyCollection` import at module top
- [x] Task 1.2 — Replace per-vertex distance mask with per-face mask: include any face where at least 1 vertex is within `radius_uv`
- [x] Task 1.3 — Replace `ax.scatter()` block (lines 627-638) with `PolyCollection` mesh rendering: build `(M, 3, 2)` triangle vertices array, compute per-face RGBA via `vertex_colors[selected_faces].mean(axis=1)`, handle grey fallback with `np.tile`
- [x] Task 1.4 — Change `alpha=0.8` to `alpha=1.0` on the `pcolormesh` call (line 645)
- [x] Task 1.5 — Update function docstring to reflect mesh surface (not scatter) and opaque heatmap (not alpha=0.8)

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rendering/rf_population_map_renderer.py` — replace scatter with PolyCollection in `render_population_rf_circular_crop()`, fix heatmap alpha

**Dependencies:** None

**Key patterns to reuse:**
- `compare_flattening_methods.py:186-193` — `colors[F].mean(axis=1)` + `PolyCollection(uv[F], facecolors=..., edgecolors="none")`
- `slim_qc_figures.py:31` — `PolyCollection` import

---

## Testing Plan

### Manual Verification
- [ ] Run the `spatial_extract_boundaries` pipeline task for a session with existing SLIM UV cache and inflection boundaries
- [ ] Open `_rf_population_all_circular_centroid.png` — confirm solid forearm surface with skin texture, no gaps
- [ ] Open `_rf_population_all_circular_peak.png` — same verification
- [ ] Confirm heatmap region fully occludes forearm (no dots/texture visible through heatmap)
- [ ] Confirm circular boundary is clean (no triangle edges beyond circle)
- [ ] Test with a session where `vertex_colors=None` — confirm solid grey surface

### Edge Cases
- [ ] Empty face mask (tiny radius or off-centre) — `PolyCollection([])` should render nothing without error
- [ ] Session with very sparse mesh near circle boundary — clip path should handle partial triangles

---

## Documentation Plan

- [ ] Docstring update included in implementation (Task 1.5)
- [ ] No other documentation changes needed — this is a rendering bugfix

---

## Rollback Plan

Single file, single function. Revert the commit to restore the scatter-based rendering.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Flat per-face colour produces visible faceting | Low | Low | SLIM mesh has high triangle density (~10K+ faces); faceting is imperceptible at 300 DPI |
| PolyCollection doesn't auto-update axis limits | Low | Low | Axis limits are already set explicitly (lines 652-654), not auto-scaled |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 | ~30 min | None |

---

## References

- Established pattern: `code/scripts/compare_flattening_methods.py:186-193`
- Established pattern: `code/src/analysis/receptive_field_mapping/surface/slim_qc_figures.py:31`
- Target function: `code/src/analysis/receptive_field_mapping/rendering/rf_population_map_renderer.py:555-659`
- Call site: `code/src/analysis/receptive_field_mapping/pipelines/rf_population_response_field_pipeline.py:582-596`

---
