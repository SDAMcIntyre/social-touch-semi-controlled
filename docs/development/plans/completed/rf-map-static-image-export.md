# Plan: RF Map Static Image Export

**Date:** 2026-03-13
**Author:** Basil Duvernoy
**Status:** Completed
**Branch:** `dev` (commit `afcd3d5`)

---

## Overview

Add static PNG image generation to the `map_receptive_fields` task so that every
run produces a visual output alongside the existing CSV and JSON data files.
Currently, visualization is only available via an interactive Open3D viewer gated
behind the `monitor` flag; static images will be generated unconditionally.

## Problem Statement

When the RF mapping pipeline runs, it produces CSVs and a summary JSON but no
visual artefact. To inspect results, a user must either re-run with `monitor=True`
(which launches an interactive Open3D GUI) or manually load the CSV into an
external tool. A static image per group would make results immediately reviewable
without additional tooling.

## Goals

### In Scope
1. Generate a static PNG per group showing clustered RF points on the forearm
2. Reuse the existing HSV colour scheme from the interactive viewer
3. Include the forearm reference point cloud as a background when available
4. Save images to the same output directory as the CSVs

### Out of Scope
- Replacing or modifying the existing interactive Open3D viewer
- Adding new configuration options or DAG keys
- Generating multi-session aggregated RF images

## Success Criteria

- [ ] Each `map_receptive_fields` run produces one PNG per group in `4_analysed/receptive_field_maps/`
- [ ] Images show two orthogonal 2D projections (XY top-down and XZ side view)
- [ ] Cluster colours match the interactive viewer's HSV scheme
- [ ] Forearm reference appears as a subtle grey background when PLY is available
- [ ] Images render headlessly (no GUI window, `Agg` backend)
- [ ] Existing interactive viewer is unaffected

---

## Technical Design

### Approach

Use matplotlib 2D scatter plots with two orthogonal projections (XY and XZ) in a
1x2 subplot layout. This matches the project's established static-image pattern
(`reporting.py`) and avoids Open3D offscreen rendering, which is fragile on
headless/Windows systems.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Matplotlib 2D projections | Project standard, headless-safe, lightweight | Not true 3D | **Chosen** |
| Matplotlib Axes3D | True 3D rotation | Rendering artefacts, z-ordering issues | Rejected |
| Open3D OffscreenRenderer | Matches interactive view exactly | Fragile on Windows/headless, new dependency pattern | Rejected |

### Architecture Changes

No new modules or files. Two existing files are modified:

- **`rf_visualizer.py`** — new `save_rf_map_image` static method added to `RFVisualizer`
- **`analysis_workflow.py`** — forearm loading hoisted before group loop; unconditional call to `save_rf_map_image` added after `_save_group_csv`

---

## Implementation Plan

### Phase 1: Add static image method to RFVisualizer

**Goal:** Implement `save_rf_map_image` that renders and saves a PNG.

**Tasks:**
- [ ] Add `save_rf_map_image(rf_result, output_dir, forearm_pcd=None) -> Optional[Path]` static method
- [ ] Use lazy `matplotlib.use('Agg')` + `import matplotlib.pyplot as plt` inside the method to avoid polluting the module-level namespace
- [ ] Reuse `_BASE_HUES` and the same HSV brightness-from-selectivity formula as `_build_rf_point_cloud`
- [ ] Render forearm as `lightgray, s=0.5, alpha=0.3, rasterized=True` background
- [ ] Render cluster points as `c=colors, s=8, zorder=2` foreground
- [ ] Two panels: XY (top-down) and XZ (side), both with `aspect='equal'` and light grid
- [ ] Title: `"RF Map — {group_label} | N clusters, M touches"`
- [ ] Save with `bbox_inches='tight', dpi=150` then `plt.close(fig)`
- [ ] Return output path or `None` if no clusters

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_visualizer.py` — add method (~50 lines)

**Dependencies:** None

### Phase 2: Integrate into analysis workflow

**Goal:** Wire the new method into `map_receptive_fields_flow`.

**Tasks:**
- [ ] Hoist `session_id` extraction and `_load_forearm_pcd()` call from inside the `if monitor` block to **before** the group loop (load once per session)
- [ ] Add `RFVisualizer.save_rf_map_image(rf_result, rf_output_dir, forearm_pcd)` after `_save_group_csv(rf_result, rf_output_dir)` (unconditional, not gated by `monitor`)
- [ ] Remove the now-redundant forearm loading from the `if monitor` block
- [ ] Keep `if monitor` block unchanged otherwise (it still gates the interactive viewer)

**Files Modified:**
- `code/scripts/analysis_workflow.py` — ~10 lines changed around lines 280-300

**Dependencies:** Phase 1

---

## Testing Plan

### Manual Verification
- [ ] Run the analysis workflow on a session with known RF clusters; confirm PNGs appear in `4_analysed/receptive_field_maps/`
- [ ] Verify image shows both XY and XZ panels with correctly coloured clusters
- [ ] Verify forearm background renders when PLY exists
- [ ] Run on a session with no clusters; confirm no PNG is created and no error is raised
- [ ] Run with `monitor=True`; confirm both static image and interactive viewer still work
- [ ] Run with `monitor=False`; confirm static image is generated without any GUI window

### Edge Cases
- [ ] Session with no forearm PLY — image should render clusters only, no crash
- [ ] Group with a single cluster — one colour, no issues
- [ ] Group with >10 clusters — colours wrap around via `idx % len(_BASE_HUES)`

---

## Documentation Plan

- [ ] No external documentation changes needed (internal pipeline output)

---

## Rollback Plan

Revert the two modified files. No migrations, no config changes, no breaking changes.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Large forearm clouds slow down matplotlib | Low | Low | `rasterized=True` + small point size keeps render time reasonable |
| `matplotlib.use('Agg')` conflicts with later interactive plots in same process | Low | Med | Lazy import inside method body; Open3D viewer uses its own rendering, not matplotlib |

---

## References

- Existing pattern: `code/src/analysis/touch_analytics/reporting.py` (`VisualReportingStrategy`)
- Colour scheme: `code/src/analysis/receptive_field_mapping/rf_visualizer.py` (`_BASE_HUES`, `_build_rf_point_cloud`)
- Integration target: `code/scripts/analysis_workflow.py` (`map_receptive_fields_flow`, lines 204-338)
