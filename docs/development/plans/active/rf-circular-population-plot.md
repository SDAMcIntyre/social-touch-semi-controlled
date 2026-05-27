# Plan: Circular RF Population Plot with Skin Colors

**Date:** 2026-05-27
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/stimulus-compare-sessions`
**Branch:** `feature/rf-circular-population-plot`

---

## Overview

Add a publication-quality circular RF heatmap output to the `spatial_extract_boundaries` pipeline. Each session produces two transparent-background PNGs — one centered on the boundary centroid, one on the hotspot peak — showing a 10 cm diameter circular crop of the interpolated population heatmap with forearm skin texture as background.

## Problem Statement

The current `_rf_population_{gtype}_interpolated.png` outputs are rectangular, black-background diagnostic plots with axes, titles, and grey vertex clouds. They are not suitable for publication figures or presentations. A clean, circular, transparent-background rendering centered on the response field would allow direct compositing into figures and posters.

Additionally, the grey vertex background (#404040) hides the forearm's natural skin texture. PLY vertex colors are already loaded elsewhere in the codebase (SLIM precompute, GUI viewers) but are not used by the pipeline renderer.

## Goals

### In Scope
1. New renderer function producing a circular, transparent, decoration-free heatmap PNG
2. UV-to-mm scale conversion utility (SLIM UV → mm) for the 10 cm diameter crop
3. PLY vertex skin colors as background for forearm regions without heatmap data
4. Two output files per session: centroid-centered and peak-centered

### Out of Scope
- Changing the existing rectangular standalone or composite PNGs
- Circular crops for gesture types other than `'all'` (can be added later)
- Storing vertex colors in the SLIM UV cache NPZ
- Interactive GUI circular crop

## Success Criteria

- [ ] Pipeline produces `{session_id}_rf_population_all_circular_centroid.png` and `{session_id}_rf_population_all_circular_peak.png` per session
- [ ] PNGs have transparent background (RGBA with alpha = 0 outside circle)
- [ ] No axes, titles, spines, or text in the output
- [ ] Circular crop is 10 cm diameter in 3D world-space units
- [ ] Forearm skin colors visible for background vertices within the circle
- [ ] Graceful grey fallback when PLY has no vertex colors
- [ ] Existing pipeline outputs unchanged

---

## Technical Design

### Approach

Render in SLIM UV space (already flattened) with a matplotlib `Circle` clip path for pixel-perfect circular edges. Convert 10 cm diameter (50 mm radius) to UV units via the median mesh edge-length ratio between 3D and UV. Load PLY vertex colors at pipeline time using the existing `nearest_orig_for_slim` KDTree mapping (already computed).

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| UV edge-length ratio scale (A) | Single scalar, stable via median, computed once | Assumes quasi-isometric SLIM (true by design) | Chosen |
| 3D distance masking (B) | Exact in 3D | Requires back-projecting every grid cell from UV to 3D; expensive and fragile at mesh boundaries | Rejected |
| PIL post-processing crop | Avoids matplotlib clip-path quirks with blitting | Unnecessary complexity for a static render; blitting concern is for animation only | Rejected |
| Store vertex colors in SLIM UV cache NPZ | Avoids reloading PLY at pipeline time | Breaks cache schema, forces re-precompute for all sessions | Rejected — load from PLY instead |

### Architecture Changes

No new modules or classes. Two new public functions in the existing renderer module; two new fields on the existing pipeline dataclass.

```
rf_population_map_renderer.py
  + compute_uv_to_mm_scale()       — edge-length-ratio UV↔mm conversion
  + render_population_rf_circular_crop()  — transparent circular heatmap PNG

rf_population_response_field_pipeline.py
  ~ _SessionCompositeData           — add forearm_ply_path, slim_vertex_colors
  ~ Pass 1 body                     — load PLY colors via existing KDTree
  ~ Pass 2 body                     — call circular renderer for 'all' gesture
```

---

## Implementation Plan

### Phase 1: Renderer functions
**Goal:** Add UV-to-mm scale utility and circular crop renderer
**Started:** 2026-05-27
**Completed:** 2026-05-27

- [x] Add `compute_uv_to_mm_scale(forearm_uv, forearm_V, forearm_faces) → float` — extract unique edges from faces, compute median(3D length / UV length), filter degenerate UV edges (< 1e-12)
- [x] Add `render_population_rf_circular_crop(u_grid, v_grid, interp_grid, forearm_uv, forearm_V, forearm_faces, center_uv, radius_mm, vmax, vmin, output_path, vertex_colors=None, heatmap_space="linear", dpi=300)` — transparent figure, circular clip, skin-color background scatter, heatmap pcolormesh

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rendering/rf_population_map_renderer.py` — two new public functions

**Dependencies:** None

### Phase 2: Pipeline integration
**Goal:** Load PLY vertex colors in Pass 1, call circular renderer in Pass 2
**Started:** 2026-05-27
**Completed:** 2026-05-27

- [x] Extend `_SessionCompositeData` with `forearm_ply_path: Path | None` and `slim_vertex_colors: np.ndarray | None`
- [x] In Pass 1: load PLY colors via `load_forearm_vertex_colors(forearm_ply_path)`, map to SLIM vertices using `nearest_orig_for_slim` (already computed), convert to (N,4) RGBA float64, store in dataclass
- [x] In Pass 2: after standalone interpolated PNGs, call `render_population_rf_circular_crop` twice for `'all'` gesture — once with `boundary.centroid_uv`, once with `boundary.peak_uv`
- [x] Raise `ValueError` if `'all'` inflection boundary is `None` (requires `inflection_sigma`)
- [x] Append both new PNGs to `sd.produced` for sentinel tracking
- [x] Update imports: add `render_population_rf_circular_crop` and `load_forearm_vertex_colors`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_population_response_field_pipeline.py` — dataclass extension, PLY color loading, circular renderer calls, imports

**Dependencies:** Phase 1

---

## Testing Plan

### Manual Verification
- [ ] Run `spatial_extract_boundaries` on at least one session with `inflection_sigma` set
- [ ] Confirm two new PNGs appear in `4_analysed/spatial_extract_boundaries/{session_id}/`
- [ ] Open PNGs in an image viewer that displays alpha — verify transparent background outside circle
- [ ] Verify no axes, titles, spines, or text
- [ ] Verify skin-colored forearm texture visible within circle (not grey)
- [ ] Verify heatmap (jet colormap) overlays correctly within circle
- [ ] Confirm existing rectangular PNGs, composites, and colorbar are unchanged
- [ ] Test with a PLY that has no vertex colors — verify grey fallback works

### Edge Cases
- [ ] Session where `inflection_sigma` is `None` — verify `ValueError` is raised with clear message
- [ ] Session with very small RF (heatmap mostly within circle) — verify rendering is reasonable
- [ ] Session with RF centroid near mesh edge — verify clip doesn't produce artifacts

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` — mention circular crop outputs in Population response fields section

---

## Rollback Plan

Feature adds new outputs without modifying existing ones. Rollback is simply reverting the feature branch commits — no data migration, no breaking changes. Existing sentinel JSONs will not include the circular PNGs, so re-running with the reverted code produces identical output.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| SLIM UV distortion makes 10 cm circle non-circular on forearm | Low | Low | SLIM is quasi-isometric by design; median edge ratio handles minor distortion. Visual check during verification. |
| 150x150 grid too coarse for 10 cm crop (low resolution within circle) | Medium | Low | Evaluate empirically; if needed, increase grid resolution for the circular render only (future enhancement). |
| PLY vertex colors missing for some sessions | Low | Low | Grey fallback is already handled; `load_forearm_vertex_colors` returns `None` gracefully. |
| `tight_layout` interference with clip path | Low | Medium | Per knowledge base note, use explicit figure sizing with `bbox_inches='tight'` on save, not `tight_layout=True`. |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Renderer functions | ~60 lines | None |
| Phase 2: Pipeline integration | ~30 lines | Phase 1 |

---

## References

- Knowledge base: `note-neural-kinect-viewer-blitting.md` — avoid `tight_layout=True` in matplotlib
- Existing renderer: `rf_population_map_renderer.py::render_population_rf_standalone_interpolated` — pattern reference
- PLY color loading: `rf_data_loader.py::load_forearm_vertex_colors` — reused utility
