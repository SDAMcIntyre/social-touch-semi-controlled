# Plan: Population RF Standalone Interpolated Heatmaps

**Date:** 2026-05-22
**Author:** Basil Duvernoy
**Status:** In Progress
**Started:** 2026-05-22
**Base Branch:** `feature/slim-uv-per-session-config-gui`
**Branch:** `feature/population-rf-standalone-heatmaps`

---

## Overview

Add standalone interpolated heatmap PNGs (one per gesture type) as the primary
output of the population response field pipeline. Enforce equal aspect ratio on
all UV-space axes so the forearm shape is geometrically correct. Reorganize
the output directory so only the primary heatmaps sit at the session root, with
aggregated multi-panel views and inflection diagnostics in dedicated subfolders.

## Problem Statement

The current pipeline produces **2-panel PNGs** (scatter + interpolated) per
gesture type and multi-panel composite PNGs — all dumped into the session root
folder alongside inflection diagnostic snapshots. This creates three issues:

1. **No standalone interpolated heatmap image** — the interpolated heatmap
   (the primary deliverable for boundary analysis) is always paired with a
   scatter panel, making it harder to compare across sessions or embed in
   reports.
2. **Distorted aspect ratio** — UV-space axes have no `set_aspect('equal')`
   call, so the forearm shape is stretched by the figure's aspect ratio rather
   than reflecting true spatial proportions (UV coordinates are in mm).
3. **Cluttered output folder** — diagnostic inflection snapshots (5 per gesture
   type) and aggregated multi-panel views are mixed with the primary results,
   making it hard to find the key outputs at a glance.

## Goals

### In Scope
1. Render one standalone interpolated heatmap PNG per gesture type (all, tap,
   stroke_proximal, stroke_distal) with optional inflection boundary overlay
2. Set equal aspect ratio on all UV-space matplotlib axes across the pipeline
3. Move 2-panel and composite PNGs to an `aggregated/` subfolder
4. Move inflection diagnostic snapshots to an `inspection/` subfolder

### Out of Scope
- Changing the interpolation algorithm or grid resolution (stays at 150x150)
- Modifying the inflection boundary detection logic
- Adding new metrics or data outputs to the NPZ
- Changing the composite or 2-panel renderers beyond aspect ratio

## Success Criteria

- [ ] Each session output contains 4 `*_interpolated.png` files in the root
- [ ] Each standalone PNG is a single-panel interpolated heatmap with equal aspect
- [ ] `aggregated/` contains the 2-panel and composite PNGs (also with equal aspect)
- [ ] `inspection/` contains all inflection diagnostic PNGs
- [ ] No PNGs other than `*_interpolated.png` remain in the session root
- [ ] Existing tests pass unchanged

---

## Technical Design

### Approach

Add a lightweight single-panel renderer
(`render_population_rf_standalone_interpolated`) that takes a precomputed
interpolated grid and renders it as a standalone figure with equal aspect ratio
and optional boundary overlay. In the pipeline, adjust output paths so existing
multi-panel PNGs land in `aggregated/` and inflection snapshots land in
`inspection/`. Apply `set_aspect('equal')` to all existing UV-space renderers.

The knowledge base confirms UV coordinates are in mm
(`note-3d-to-2d-surface-projection-algorithms.md`), so `set_aspect('equal')`
correctly represents the physical geometry.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| New standalone renderer | Clean single-panel output, no disruption to existing renderers | One more function | **Chosen** |
| Modify 2-panel renderer to optionally hide scatter | Reuses existing code | Adds conditional complexity, harder to maintain | Rejected |
| Extract interpolated panel from composite | No new renderer | Composite uses global UV limits, not per-gesture; different visual result | Rejected |

### Architecture Changes

**New function** in `rf_population_map_renderer.py`:
- `render_population_rf_standalone_interpolated()` — single-panel interpolated
  heatmap with boundary overlay, black background, jet colormap, equal aspect

**Modified files:**
- `rf_population_map_renderer.py` — new function + `set_aspect('equal')` on
  existing renderers
- `rendering/__init__.py` — export new function
- `rf_population_response_field_pipeline.py` — call new renderer, route
  existing outputs to subfolders
- `rf_inflection_boundary.py` — `set_aspect('equal')` on step 5 UV snapshot

**Output directory structure change:**
```
{session_id}/
├── *_interpolated.png          # NEW primary output (4 files)
├── *.npz, *_done.json          # data + sentinel (unchanged)
├── aggregated/                 # MOVED from root
│   ├── *_rf_population_{gtype}.png         # 2-panel
│   ├── *_scatter_composite.png             # composite
│   └── *_interpolated_composite.png        # composite
└── inspection/                 # MOVED from root
    └── inflection_*_step*.png              # diagnostics
```

---

## Implementation Plan

### Phase 1: Renderer changes
**Goal:** Add standalone renderer and equal aspect ratio to all UV-space axes

**Tasks:**
- [x] Task 1.1 — Add `render_population_rf_standalone_interpolated()` after
  `render_population_rf_map()` in `rf_population_map_renderer.py`
- [x] Task 1.2 — Add `ax.set_aspect('equal')` in `render_population_rf_map()`
  inside the `for ax in axes:` loop (after spine styling, line ~251)
- [x] Task 1.3 — Add `ax.set_aspect('equal')` in
  `render_population_rf_composite()` inside the per-panel loop (after spine
  styling, line ~415)
- [x] Task 1.4 — Add `ax.set_aspect("equal")` in `_render_step_uv()` in
  `rf_inflection_boundary.py` (after `ax.set_ylabel("V")`, line ~461)
- [x] Task 1.5 — Export `render_population_rf_standalone_interpolated` from
  `rendering/__init__.py`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rendering/rf_population_map_renderer.py` —
  new function (~50 lines), two `set_aspect` insertions
- `code/src/analysis/receptive_field_mapping/metrics/rf_inflection_boundary.py` —
  one `set_aspect` insertion
- `code/src/analysis/receptive_field_mapping/rendering/__init__.py` —
  add to import + `__all__`

**Dependencies:** None

### Phase 2: Pipeline output reorganization
**Goal:** Wire up the new renderer and route outputs to subfolders

**Tasks:**
- [x] Task 2.1 — Update imports in pipeline to include new renderer
- [x] Task 2.2 — After `output_dir.mkdir()` (Pass 1, line ~251), create
  `aggregated/` and `inspection/` subdirectories
- [x] Task 2.3 — Change 2-panel PNG path from `output_dir / ...` to
  `aggregated_dir / ...` (line ~261)
- [x] Task 2.4 — Change `snapshot_dir=output_dir` to
  `snapshot_dir=inspection_dir` in Pass 1 (line ~272)
- [x] Task 2.5 — After `render_population_rf_map()`, call
  `render_population_rf_standalone_interpolated()` with output path in session
  root (`output_dir / f'{session_id}_rf_population_{gtype}_interpolated.png'`)
- [x] Task 2.6 — In Pass 2, create `aggregated/` and `inspection/`
  subdirectories for each session
- [x] Task 2.7 — Change composite PNG path to `aggregated_dir / ...`
  (line ~354-356)
- [x] Task 2.8 — Change `snapshot_dir=sd.output_dir` to
  `snapshot_dir=inspection_dir` in Pass 2 (line ~374)

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_population_response_field_pipeline.py` —
  import update, subdirectory creation, path changes, new renderer call

**Dependencies:** Phase 1

---

## Testing Plan

### Unit Tests
- [ ] Existing `pytest code/tests/test_rf_inflection_boundary.py` passes
  unchanged (snapshot tests use `tmp_path` directly, not affected by pipeline
  path changes)

### Manual Verification
- [ ] Run pipeline on one session with `inflection_sigma` set
- [ ] Confirm 4 `*_interpolated.png` files in session root, each single-panel
  with equal aspect ratio
- [ ] Confirm `aggregated/` contains 2-panel + composite PNGs with equal aspect
- [ ] Confirm `inspection/` contains all `inflection_*` diagnostic PNGs
- [ ] Confirm no stray PNGs in session root besides `*_interpolated.png`
- [ ] Open a standalone PNG and visually verify the forearm is not
  stretched — UV axes should be geometrically proportional

### Edge Cases
- [ ] Session with no touches for a gesture type — that gesture is skipped,
  no standalone PNG produced (existing behavior, unchanged)
- [ ] `inflection_sigma=None` — no inflection boundary overlay, no
  `inspection/` snapshots needed (but directory is still created harmlessly)

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` — note new output layout in the
  population response fields section

---

## Rollback Plan

1. Revert the single commit on the feature branch
2. No data migrations — output files are regenerated by re-running the pipeline
   with `force_processing=True`

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `set_aspect('equal')` causes excess whitespace in figures with extreme UV ranges | Low | Low | `bbox_inches='tight'` is already used in all `savefig` calls, so saved PNGs are cropped to content |
| `_save_inflection_snapshots` fails because `inspection/` dir not created | Low | Med | `inspection_dir.mkdir(exist_ok=True)` called before any `compute_inflection_boundary()` |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 — Renderer changes | ~30 min | None |
| Phase 2 — Pipeline reorganization | ~20 min | Phase 1 |

---

## References

- Approved design: `.claude/plans/analyse-the-esxtraction-opulation-lucky-newt.md`
- Knowledge base: `docs/development/knowledge-base/note-3d-to-2d-surface-projection-algorithms.md`
- Knowledge base: `docs/development/knowledge-base/note-analysis-pipeline-coordinate-spaces.md`

---
