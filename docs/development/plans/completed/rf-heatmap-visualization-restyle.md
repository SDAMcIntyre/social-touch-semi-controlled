# Plan: RF Heatmap Visualization Restyle

**Date:** 2026-04-13
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `feature/rf-heatmap-visualization-restyle`

---

## Overview

The `render_forearm_heatmap()` function renders the forearm pointcloud as a
near-invisible background and overlays spike counts on a linear scale with a
warm-only colormap. This plan restyles the visualization so the forearm is
clearly visible with its real skin-tone colors, the spike heatmap uses a
log-scale blue-to-red colormap, and the 3D scene has a black background.

## Problem Statement

The current rendering parameters (`s=0.5`, `alpha=0.3` for the forearm) make
the forearm background nearly invisible on the white matplotlib figure. It is
impossible to tell whether the PLY carries real RGB colors. The linear-scale
`YlOrRd` colormap also compresses low-count regions, making spatial patterns
hard to read when spike counts span several orders of magnitude.

## Goals

### In Scope
1. Black background for the 3D scene (figure, axes panes, grid)
2. Forearm pointcloud rendered at the same point size as the spike overlay,
   fully opaque, showing real RGB colors from the PLY
3. Spike heatmap using `RdYlBu_r` colormap with log-scale normalization
4. Colorbar and all text (labels, ticks, title) styled for readability on
   black background

### Out of Scope
- Changing the forearm PLY data pipeline (colors are preserved end-to-end)
- Adding new rendering backends (stays matplotlib Axes3D)
- Modifying camera angle logic or surface-normal computation

## Success Criteria

- [ ] Forearm skin-tone colors visible in generated PNG (not grey/invisible)
- [ ] Forearm and spike overlay points are the same size
- [ ] Spike counts rendered on a log scale from blue (low) through yellow to
      red (high) using `RdYlBu_r`
- [ ] 3D scene background is black; all text is white and readable
- [ ] Interactive mode (`show_interactive: true`) also renders correctly

---

## Technical Design

### Approach

Modify the single `render_forearm_heatmap()` function in
`rf_cluster_visualizer.py`. All changes are to matplotlib scatter and axes
styling parameters — no architectural changes, no new modules.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Restyle matplotlib scatter params | Minimal change, no new deps | Limited by matplotlib 3D quality | **Chosen** |
| Switch to PyVista rendering | Better 3D quality, lighting | New dependency in analysis path, heavier | Rejected |
| Switch to Open3D visualizer | Native pointcloud support | Non-interactive PNG export is harder | Rejected |

### Architecture Changes

None. Single function modification in one file.

---

## Implementation Plan

### Phase 1: Restyle render_forearm_heatmap()

**Goal:** Update all rendering parameters in the existing function.

**Tasks:**
- [x] Task 1.1 — Set figure and axes facecolor to black; set 3D pane wall
      colors to black
- [x] Task 1.2 — Set all axis labels, tick labels, and title color to white
- [x] Task 1.3 — Change forearm scatter: `s=20`, `alpha=1.0`, keep
      `c=point_colors` with `'lightgrey'` fallback
- [x] Task 1.4 — Change spike scatter: `cmap='RdYlBu_r'`, replace
      `vmin`/`vmax` with `norm=LogNorm(vmin=max(1, counts.min()), vmax=counts.max())`
- [x] Task 1.5 — Style colorbar: white label text, white tick labels
- [x] Task 1.6 — Update `fig.savefig()` to pass
      `facecolor=fig.get_facecolor()` so the black background persists in the
      PNG output

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_cluster_visualizer.py` —
  modify `render_forearm_heatmap()` (lines ~125-191)

**Dependencies:** None

---

## Testing Plan

### Manual Verification
- [ ] Run analysis workflow with `map_receptive_fields_simple` enabled and
      `force_processing: true`
- [ ] Open a generated `*_rf_simple.png` and visually confirm:
    - Black background
    - Forearm visible with skin-tone RGB colors, same point size as spike dots
    - Spike overlay colored blue-to-red on log scale
    - Colorbar text readable on black background
- [ ] Run with `show_interactive: true` and confirm the interactive window
      also displays correctly

### Edge Cases
- [ ] Session where forearm PLY has no colors — should fall back to
      `'lightgrey'` (existing behaviour, unchanged)
- [ ] Session with all spike counts equal (LogNorm with vmin == vmax) — must
      not crash; clamp or degrade gracefully

---

## Documentation Plan

- [ ] No README/CLAUDE.md changes needed (internal visualization tweak)

---

## Rollback Plan

Single-file change. Revert the commit on `rf_cluster_visualizer.py`.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| LogNorm crashes when all counts are identical | Low | Med | Clamp `vmin` < `vmax`; fall back to linear if needed |
| Forearm PLY on disk lacks colors (stale data) | Med | Low | Existing `'lightgrey'` fallback handles this; reprocessing fixes it |
| `s=20` too large for dense pointclouds | Low | Low | Stride subsampling already limits to ~10k points |

---

## References

- Visualizer source: `code/src/analysis/receptive_field_mapping/rf_cluster_visualizer.py`
- Consumer: `code/src/analysis/receptive_field_mapping/rf_simple_pipeline.py`
- Consumer: `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py`
