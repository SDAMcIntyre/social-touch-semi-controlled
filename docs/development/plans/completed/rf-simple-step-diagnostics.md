# Plan: RF Simple Pipeline Step-by-Step Diagnostic Figures

**Date:** 2026-05-11
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-05-12 10:42
**Base Branch:** `refactor/rf-simple-use-population-data`
**Branch:** `feature/rf-simple-step-diagnostics`

---

## Overview

**What:** Add inline diagnostic figure generation at each step of the RF simple pipeline, producing multi-panel PNGs that visualize intermediate data.
**Why:** After the PopulationData refactoring (commit e2d2ae7), the scatter plot output shows wrong spatial distribution / distorted projection. Without visibility into intermediate data, it is impossible to pinpoint where the issue originates.
**How:** Create a new `rf_simple_diagnostics.py` module with pure functions that take intermediate pipeline data and produce matplotlib figures, wired into `run_simple_rf_mapping()` via a `save_diagnostics` option.

## Problem Statement

- The RF simple pipeline was refactored to use `PopulationData` instead of inline CSV parsing. The resulting 2D scatter plot is spatially wrong.
- The pipeline has no intermediate visibility — data flows through 4 stages (load, extract, aggregate, project) with only the final PNG as output.
- Without step-by-step figures, debugging requires manually inserting print statements and re-running, which is slow and error-prone.
- Suspected issues include: centroid mismatch (projection centroid from ALL contacts vs spike-only), KDTree threshold change (15mm vs old 2mm), and contact point parsing differences.

## Goals

### In Scope
1. Diagnostic figure generation at each of the 4 pipeline steps
2. PNG output to `diagnostics/` subfolder alongside existing session output
3. Optional interactive display via `show_interactive` flag
4. DAG config integration via `save_diagnostics` option

### Out of Scope
- Interactive PyQt5 viewer (may be added later if static figures are insufficient)
- Fixing the actual scatter plot issue (diagnostics first, then fix)
- Modifying `PopulationData` or `load_population_data()` internals
- Diagnostics for the cluster RF pipeline (separate scope)

## Success Criteria

- [ ] Running `map_receptive_fields_simple` with `save_diagnostics: true` produces 4 PNG files in `<session_out>/diagnostics/`
- [ ] Step 1 figure shows all contact positions (blue) and spike positions (red) in XY and XZ views with summary counts
- [ ] Step 3 figure shows the projection centroid vs spike-only centroid, making centroid mismatch immediately visible
- [ ] Step 4 figure shows projected UV coordinates with forearm boundary, enabling direct comparison with the final scatter panel
- [ ] `show_interactive: true` displays figures as matplotlib windows
- [ ] Diagnostic failures never abort the pipeline (wrapped in try/except)
- [ ] Pipeline runs identically when `save_diagnostics: false` (default)

---

## Technical Design

### Approach

Pure-function diagnostic module: each function takes intermediate numpy/pandas data and returns a `matplotlib.figure.Figure`. An orchestrator function calls all 4 and handles save/show. The pipeline calls the orchestrator at the appropriate point, passing accumulated intermediate variables.

This approach was chosen because:
- Pure functions are testable without running the pipeline
- Returning Figure objects lets the caller decide save vs show vs both
- No new classes, no new viewer infrastructure — minimal footprint
- Captures exact intermediate state at the point of computation

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Pure diagnostic functions (inline PNGs) | Simple, captures exact state, no new infrastructure | Not interactive, static views only | **Chosen** |
| Interactive PyQt5 viewer | Full 3D rotation, dynamic exploration | Heavy implementation, requires persisting all intermediate state, delays debugging | Rejected (can add later) |
| Logging + CSV dumps at each step | Lightweight, no matplotlib dependency | Requires separate analysis tooling to interpret | Rejected |

### Architecture Changes

New module only — no architectural changes to existing code.

```
code/src/analysis/receptive_field_mapping/
├── rf_simple_diagnostics.py    # NEW — 4 diagnostic functions + orchestrator
├── rf_simple_pipeline.py       # MODIFIED — add save_diagnostics param + calls
├── rf_projection.py            # READ ONLY — reuse project_to_2d() in Step 4
└── rf_cluster_visualizer.py    # READ ONLY — reuse camera_settings_to_rotation()
```

Output structure per session:
```
4_analysed/receptive_field_maps_simple/<session_id>/
├── spike_positions.csv                          # existing
├── <session_id>_rf_simple_cylindrical_unwrap.png # existing
├── rf_simple_summary.json                        # existing
└── diagnostics/                                  # NEW
    ├── step1_population_data.png
    ├── step2_spike_extraction.png
    ├── step3_aggregation.png
    └── step4_projection.png
```

### Diagnostic Figure Specifications

**Step 1 — Population Data Loading** (`step1_population_data.png`, 2x2 panels):
- Top-left (XY view): All contact vertex positions (blue) + spike positions (red) on forearm silhouette (gray)
- Top-right (XZ view): Same, side view — together with XY shows 3D spatial coverage
- Bottom-left (histogram): Contact points per touch, spike-touches in red overlay
- Bottom-right (text): Total touches (T), contact points (C), spike contacts, unique spike vertices, mesh vertices (V), gesture type breakdown

**Step 2 — Spike Extraction** (`step2_spike_extraction.png`, 1x3 panels):
- Left (XY view): Forearm vertices (gray), spike positions (red, sized by occurrence count)
- Center (XZ view): Same, side view
- Right (histogram): Spike vertex reuse — how many times each vertex appears in `spike_vertex_indices`

**Step 3 — Aggregation** (`step3_aggregation.png`, 2x2 panels):
- Top-left (XY scatter): `spike_counts_df` positions colored by `spike_count` (RdYlBu_r, LogNorm)
- Top-right (XZ scatter): Same, side view
- Bottom-left (centroid comparison): XY view with 3 marked points — red: projection centroid (`neuron_contacts_xyz.mean`), green: spike-only centroid (`spike_xyz.mean`), blue: mesh centroid (`forearm_vertices.mean`). Annotated with pairwise distances in mm.
- Bottom-right (histogram): Distribution of `spike_count` values (log-scale x-axis)

**Step 4 — 2D Projection** (`step4_projection.png`, 1x3 panels):
- Left (UV scatter): All forearm vertices projected to UV (gray, subsampled 1:5) + spike vertices (colored by spike_count). Shows whether spikes fall within forearm boundary.
- Center (UV spike-only): Only spike UV points, colored by spike_count. Direct comparison target for the final scatter panel.
- Right (text): Projection method, centroid coords, camera rotation matrix, cylinder axis direction, mean radius.

**Visual conventions** (matching existing RF output): black background (`#1a1a1a`), white labels/ticks, `RdYlBu_r` colormap, `LogNorm` for spike counts.

### Reused Code

- `rf_projection.project_to_2d()` — Step 4 projection (same function the renderer uses)
- `rf_cluster_visualizer.camera_settings_to_rotation()` — Step 4 rotation matrix
- `rf_projection.fit_cylinder_axis()` — Step 4 metadata (axis direction, mean radius)
- `rf_projection._compute_local_radius()` — Step 4 metadata (local radius)

---

## Implementation Plan

### Phase 1: Diagnostic Module
**Goal:** Create `rf_simple_diagnostics.py` with all figure-generation functions.
**Started:** 2026-05-11
**Completed:** 2026-05-11

**Tasks:**
- [x] 1.1 — Create `diagnose_population_data()`: 2x2 figure (XY, XZ, histogram, text)
- [x] 1.2 — Create `diagnose_spike_extraction()`: 1x3 figure (XY, XZ, reuse histogram)
- [x] 1.3 — Create `diagnose_aggregation()`: 2x2 figure (XY colored, XZ colored, centroid comparison, count distribution)
- [x] 1.4 — Create `diagnose_projection()`: 1x3 figure (full UV, spike UV, metadata text)
- [x] 1.5 — Create `run_diagnostics()` orchestrator: calls all 4 functions, saves PNGs, optionally shows

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_simple_diagnostics.py` — new file

**Dependencies:** None

### Phase 2: Pipeline Integration
**Goal:** Wire diagnostics into `run_simple_rf_mapping()`.
**Started:** 2026-05-11
**Completed:** 2026-05-11

**Tasks:**
- [x] 2.1 — Add `save_diagnostics: bool = False` parameter to `run_simple_rf_mapping()`
- [x] 2.2 — Import diagnostic module and call `run_diagnostics()` after all intermediate data is computed (after line ~189, before sentinel write)
- [x] 2.3 — Pass `show_interactive` to the `show` parameter of `run_diagnostics()`
- [x] 2.4 — Wrap diagnostic call in try/except with `logger.warning` so failures never abort the pipeline

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_simple_pipeline.py` — add parameter, import, inline call

**Dependencies:** Phase 1

### Phase 3: DAG + Workflow Wiring
**Goal:** Make `save_diagnostics` configurable from the DAG YAML and threaded through Prefect.
**Started:** 2026-05-11
**Completed:** 2026-05-11

**Tasks:**
- [x] 3.1 — Add `save_diagnostics: true` under `map_receptive_fields_simple.options` in `configs/analyse_workflow_processing_dag.yaml`
- [x] 3.2 — Add `save_diagnostics: bool = False` parameter to `map_receptive_fields_simple_flow()` in `analysis_workflow.py`
- [x] 3.3 — Pass `save_diagnostics` through to `run_simple_rf_mapping()`

**Files Modified:**
- `configs/analyse_workflow_processing_dag.yaml` — add option
- `code/scripts/analysis_workflow.py` — add parameter to flow, pass to pipeline

**Dependencies:** Phase 2

---

## Testing Plan

### Manual Verification
- [ ] Run `map_receptive_fields_simple` with `save_diagnostics: true`, `show_interactive: true` — verify 4 matplotlib windows appear
- [ ] Verify 4 PNG files are saved to `<session_out>/diagnostics/`
- [ ] Verify Step 1 text panel shows correct counts (cross-check with `rf_simple_summary.json`)
- [ ] Verify Step 3 centroid panel shows 3 distinct marked points with distance annotations
- [ ] Verify Step 4 UV scatter matches the spatial pattern in the final `_rf_simple_cylindrical_unwrap.png`
- [ ] Run with `save_diagnostics: false` — verify no `diagnostics/` folder is created and pipeline output is identical
- [ ] Intentionally break a diagnostic function (e.g., wrong array shape) — verify pipeline completes and logs a warning

### Edge Cases
- [ ] Session with 0 spikes — Steps 2-4 should produce empty/placeholder figures or be skipped gracefully
- [ ] Session with 1 spike vertex — Step 4 projection should handle degenerate case (single point, no hull)

---

## Documentation Plan

- [ ] No CLAUDE.md changes needed (no architectural change)
- [ ] No README changes needed (internal diagnostic tool)

---

## Rollback Plan

1. Delete `rf_simple_diagnostics.py`
2. Revert changes to `rf_simple_pipeline.py` (remove `save_diagnostics` param and diagnostic calls)
3. Revert changes to `analysis_workflow.py` and DAG YAML
4. All changes are additive — no risk of breaking existing functionality

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Diagnostic figures slow down pipeline | Low | Low | Diagnostic calls add ~1-2s per session; opt-in via `save_diagnostics` flag |
| matplotlib backend conflict (Agg vs interactive) | Med | Low | Use `plt.show()` only when `show=True`; figure creation works on any backend |
| Diagnostic failure aborts pipeline | Med | High | Wrap all diagnostic calls in try/except with logger.warning |
| Step 4 projection inconsistency with renderer | Low | Med | Reuse exact same `project_to_2d()` function and `camera_settings_to_rotation()` |

---

## References

- Related commit: e2d2ae7 (refactor: use PopulationData instead of inline parsing)
- Knowledge base: `note-3d-to-2d-surface-projection-algorithms.md`, `note-rf-cluster-visualization-overview.md`
- Pipeline source: `code/src/analysis/receptive_field_mapping/rf_simple_pipeline.py`
