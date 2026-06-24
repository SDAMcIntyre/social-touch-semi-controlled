# Plan: RF Cluster Visualization & Projection Improvements

**Created:** 2026-04-24
**Approved:** —
**Completed:** 2026-04-30
**Author:** Basil Duvernoy
**Status:** Completed
**Branch:** `feature/rf-cluster-visualization-improvements`

---

## Overview

Extend the cluster-based RF mapping renderer with (1) a spike-ratio colour-map alternative, (2) contextual metadata overlays (touch counts, % share of neuron's touches, cluster feature ranges), (3) two convex-hull perimeters of the neuron's contact distribution, and (4) a fix for spurious fill-in between disjoint hotspots in the 2D projection heatmap. Implemented in the existing `receptive_field_mapping/` modules — no new packages.

## Problem Statement

Current RF cluster heatmaps (`rf_cluster_visualizer.py` + `rf_2d_renderer.py`) expose only a single metric (raw spike count) with no spatial context about the neuron's overall contact distribution, no cluster-scoped touch counts on the figure, and a cubic `griddata` 2D interpolation that fills in empty space between disjoint RF regions — an artefact absent from the 3D render. Reviewers currently cannot judge whether a "hot" region represents high responsiveness (ratio close to 1) or merely many touches, nor whether a 2D heatmap's coloured region corresponds to real contacts or interpolation artefact.

## Goals

### In Scope
1. New selectable colour-map driver **spike ratio** = (unique touches that elicited ≥1 spike at this position) / (total touches in cluster for this neuron). Range [0, 1]. Rendered alongside existing spike-count output (both PNGs per cluster).
2. Metadata overlay on each figure: cluster-touch count for current neuron, total neuron touches across all clusters, % share, and the cluster's feature names with their min/max ranges.
3. Two convex-hull perimeters: (a) all contacts for current neuron across all clusters, (b) intersection with current cluster. Drawn in both the 2D panels and the 3D render.
4. 2D projection heatmap must render NaN (transparent) in regions far from any sample point, so disjoint RF hotspots read as disjoint — matching the 3D render.

### Out of Scope
- Alpha shapes / concave hulls (convex hulls only).
- Changing the clustering or feature-extraction pipeline.
- A GUI-selectable runtime toggle — `display_metric` decisions are per-config, both PNGs always emitted.
- Drawing additional RF metrics (ellipse fit, hotspot, Gaussian contour) on the figure — those remain in `rf_metrics.json` only.

## Success Criteria

- [ ] `spike_counts.csv` gains a `unique_touch_spike_count` column (backward-compatible addition; existing consumers ignore it).
- [ ] Each cluster folder contains both `<session>_rf_heatmap_count<suffix>.png` and `<session>_rf_heatmap_ratio<suffix>.png`.
- [ ] `cluster_description.json` gains `neuron_touches` (total for neuron across all clusters) and `neuron_cluster_touches` (for current neuron ∩ cluster) alongside existing `n_touches` (cluster total across all neurons).
- [ ] Each rendered figure shows a metadata text box containing: cluster touches / neuron touches (%), plus cluster feature names with `[min, max]`.
- [ ] Each rendered figure shows two convex-hull outlines (neuron-all-clusters in one colour, neuron∩cluster in another) in both 2D panels and the 3D plot.
- [ ] The 2D heatmap panel renders NaN (transparent) anywhere > `disjoint_mask_distance_mm` from the nearest sample point; disjoint hotspots in the data produce a visibly disjoint heatmap.
- [ ] A manual before/after comparison on a known multi-region session confirms the 2D artefact is gone.
- [ ] Fail-fast convention respected — no silent fallbacks added; missing data raises loudly per `CLAUDE.md`.

---

## Technical Design

### Approach

**Per-point spike-ratio tracking.** Inside `_extract_spike_contact_points` (`rf_cluster_pipeline.py:114`), when iterating the `Nerve_spike == 1` rows, also maintain a second counter keyed on `(x, y, z)` that records the set of unique `(block_order_id, trial_id, single_touch_id)` touches seen at each position. Its size per point gives the ratio numerator. The denominator (cluster touches *for this neuron*) is `len(cluster_df[cluster_df.session_id == current_session])`.

**Per-session renderer scoping.** The renderer already runs per-session inside a cluster loop (`rf_cluster_pipeline.py:546`). Session_id is effectively the neuron identifier in this codebase (one session = one recording = one neuron). All three counts (`neuron_touches`, `neuron_cluster_touches`, `cluster_touches`) are computed once inside the cluster loop from the clustered DataFrame before rendering.

**Metadata & perimeters passed into renderer.** A new lightweight dataclass `RFRenderContext` is added to `rf_cluster_visualizer.py` carrying: `neuron_touches`, `neuron_cluster_touches`, `neuron_contacts_xyz` (N×3 ndarray of `mean_contact_x/y/z` for neuron across clusters), `neuron_cluster_contacts_xyz` (subset for current cluster), and `feature_ranges` (dict from `cluster_description`). Both `render_forearm_heatmap()` and `render_2d_heatmap()` accept an optional `render_context: RFRenderContext`. This avoids a long parameter explosion while keeping the renderer signature honest.

**Convex hulls.** Computed with `scipy.spatial.ConvexHull` on the projected 2D points (for 2D panels) and on raw XYZ points (for 3D — drawn as a closed 3D polyline on the vertices of the 3D hull via `ax.plot3D`). Fewer than 3 points → skip that perimeter with a `logger.info`, not an exception (not a pipeline assumption violation — legitimate case for neurons with sparse contacts).

**2D projection fix.** In `rf_2d_renderer.py:132`, after the existing cubic `griddata` call, build a `scipy.spatial.cKDTree` on `uv_points` and query the distance of every grid cell to its nearest sample. Cells with `dist > disjoint_mask_distance_mm` (default 8 mm, matching the `map_scalars_to_mesh` radius of 5 mm with a small margin) are set to `np.nan`. `pcolormesh` already renders NaN as transparent, so no further changes needed.

**DAG config.** `configs/analyse_workflow_dag.yaml` under `map_receptive_fields_clustered` adds one optional parameter:
- `disjoint_mask_distance_mm: 8.0`
- (No toggle for spike-ratio output — both PNGs always emitted.)

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Both PNGs per cluster (count + ratio) | No re-runs needed for research; both views side-by-side | Doubles render time per cluster | **Chosen** |
| Config toggle `display_metric` | Slightly faster per run | Requires re-run to compare the two | Rejected |
| Convex hull perimeter | Simple, scipy-native, robust | Misses concave structure in bimodal RFs | **Chosen** |
| Alpha shape perimeter | Tighter fit for non-convex distributions | New dependency (`alphashape`), tuning the α parameter | Rejected |
| Distance-to-nearest-sample mask (2D fix) | 3 lines, mirrors 3D mesh radius logic, respects disjoint topology | Threshold needs tuning; a per-cluster hull mask is more principled | **Chosen** — simplest honest fix |
| Per-cluster convex-hull mask on grid | Topology-aware, zero in gaps by construction | Harder for multi-modal single-cluster data; more code | Rejected for now |
| Switch `griddata` to `method='nearest'` | No bridging at all | Blocky/ugly; loses the smooth-heatmap value | Rejected |
| Compute ratio at renderer from existing `spike_count` | No schema change | `spike_count` counts spike *samples*, not unique touches — wrong numerator | Rejected (schema change unavoidable) |

### Architecture Changes

No new modules. Changes are local to the existing `receptive_field_mapping/` package:

```
code/src/analysis/receptive_field_mapping/
├── rf_cluster_pipeline.py        — extract unique-touch spike counts;
│                                   compute neuron/cluster touch counts;
│                                   collect neuron & cluster contact XYZ;
│                                   render count AND ratio PNGs per cluster
├── rf_cluster_visualizer.py      — new RFRenderContext dataclass;
│                                   draw 3D convex-hull outlines;
│                                   draw metadata text block;
│                                   accept display_metric parameter
├── rf_2d_renderer.py             — NaN-mask 2D heatmap by distance;
│                                   draw 2D convex-hull outlines;
│                                   draw metadata text block;
│                                   accept display_metric parameter
└── (no new files)
```

**`spike_counts.csv` schema change** — adds one column `unique_touch_spike_count`. Existing `rf_metrics.py`, `compute_rf_metrics()`, and any GUI consumers that read only `spike_count` continue to work unchanged.

**`cluster_description.json` schema change** — adds `neuron_cluster_touches: {session_id: int}` and `neuron_touches: {session_id: int}` maps (one entry per session in cluster). Existing `_description_summary_line()` need not consume these; the renderer pulls from `RFRenderContext` directly.

**Existing utilities reused (no duplication):**
- `scipy.spatial.ConvexHull` (already indirect via scipy used elsewhere)
- `scipy.spatial.cKDTree` (pattern already used in `rf_surface_utils.map_scalars_to_mesh`)
- `project_to_2d()` in `rf_projection.py` — for projecting perimeter vertices
- `cluster_description['feature_ranges']` — already computed, just rendered now

---

## Implementation Plan

### Phase 1: Pipeline — extend data flow
**Goal:** Capture unique-touch-spike counts and neuron/cluster contact sets before rendering.
**Started:** 2026-04-24
**Completed:** 2026-04-24

- [x] 1.1 — In `rf_cluster_pipeline.py:114` `_extract_spike_contact_points`, keep the existing `spike_counter: Counter[(x,y,z)]` and add `unique_touch_counter: DefaultDict[(x,y,z), set[tuple]]`. After the `Nerve_spike == 1` filter, iterate rows (not just `contact_points`) so `(block_order_id, trial_id, single_touch_id)` is available; for each parsed `pt` add the touch key to that position's set. Return both counters.
- [x] 1.2 — Update `_aggregate_spike_counts` (line 185) to accept a list of `(spike_counter, unique_touch_counter)` pairs and emit a DataFrame with columns `x, y, z, spike_count, unique_touch_spike_count`. Preserve descending-by-`spike_count` sort.
- [x] 1.3 — In the per-cluster loop (around `rf_cluster_pipeline.py:450`), compute a dict `neuron_cluster_touches: dict[session_id, int]` from `cluster_df.groupby('session_id').size()` **before** the session loop, and `neuron_touches: dict[session_id, int]` from the full `clustered_df.groupby('session_id').size()`.
- [x] 1.4 — Collect two per-session contact arrays from `clustered_df`: `neuron_contacts_xyz[session_id]` (all rows for that session across all clusters) and `neuron_cluster_contacts_xyz[session_id]` (rows matching current `cluster_label`). Source columns: `mean_contact_x`, `mean_contact_y`, `mean_contact_z`.
- [x] 1.5 — Extend `_build_cluster_description` to include `neuron_touches` and `neuron_cluster_touches` dicts in the saved JSON. Do not change `n_touches` (still cluster total).
- [x] 1.6 — Verify the aggregation CSV read (`_NEEDED_COLS`) already includes the touch-key columns; it does (`block_order_id`, `trial_id`, `single_touch_id`).

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` — main changes
- `code/src/analysis/receptive_field_mapping/rf_data_loader.py` — no change expected (reuse `parse_contact_points`)

**Dependencies:** None.

### Phase 2: Renderer — context object, metadata, perimeters, display_metric
**Goal:** Both renderers accept a context object, draw metadata and perimeters, and support spike_count vs spike_ratio.
**Started:** 2026-04-24
**Completed:** 2026-04-24

- [x] 2.1 — In `rf_cluster_visualizer.py`, add `@dataclass RFRenderContext` with fields:
  - `neuron_touches: int`
  - `neuron_cluster_touches: int`
  - `neuron_contacts_xyz: np.ndarray`  # (N, 3)
  - `neuron_cluster_contacts_xyz: np.ndarray`  # (M, 3)
  - `feature_ranges: dict[str, dict]`  # from cluster_description
- [x] 2.2 — Add `display_metric: Literal["spike_count", "spike_ratio"] = "spike_count"` parameter to both `render_forearm_heatmap()` and `render_2d_heatmap()`. When `spike_ratio`, use `unique_touch_spike_count / neuron_cluster_touches` as the colour driver with `norm = None` and `vmin=0, vmax=1`. Keep `RdYlBu_r`. Colorbar label switches between "Spike count" and "Spike ratio".
- [x] 2.3 — Build a helper `_format_metadata_overlay(context) -> str` that produces multi-line text:
  ```
  Touches: 42 / 318 (13.2%)
  contact_area_mean: [0.0, 45.0]
  pressure_mean: [0.1, 5.0]
  ```
  Added as a left-aligned text box (figure-level in 2D, axes-level in 3D) in `#aaaaaa`, monospace, small font.
- [x] 2.4 — Draw convex hulls. Helper `_draw_hull_2d(ax, points_2d, color, label)` uses `scipy.spatial.ConvexHull` and closes the polygon. `_draw_hull_3d(ax, points_3d, color, label)` plots the 3D hull's edge list. Both gracefully skip if fewer than 3 points (log info, no raise — this is a legitimate data shape, not a pipeline contract violation).
- [x] 2.5 — In `render_2d_heatmap()`, after computing `uv_points` for spike contacts, project `context.neuron_contacts_xyz` and `context.neuron_cluster_contacts_xyz` via `project_to_2d()` (passed through from caller) and draw hulls on both panels.
- [x] 2.6 — In `render_forearm_heatmap()` (3D branch), draw both hulls as 3D polylines on the main axes.
- [x] 2.7 — Fix 2D projection artefact: after `griddata(...)` at `rf_2d_renderer.py:132`, build `cKDTree(uv_points)`, query `grid_points`, and set cells with `dist > disjoint_mask_distance_mm` to NaN. Default 8.0 mm, parameter threaded through from the DAG config (see Phase 3). Keep the existing `np.clip(..., vmin, None)` for lower-bound safety.

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_cluster_visualizer.py`
- `code/src/analysis/receptive_field_mapping/rf_2d_renderer.py`

**Dependencies:** Phase 1.

### Phase 3: Wiring — always emit both PNGs; DAG config plumbing
**Goal:** Cluster pipeline emits count + ratio PNGs per session; `disjoint_mask_distance_mm` configurable.
**Started:** 2026-04-24
**Completed:** 2026-04-24

- [x] 3.1 — In `rf_cluster_pipeline.py` (render loop around line 546), for each `session_id` build an `RFRenderContext` and call `render_forearm_heatmap()` twice: once with `display_metric="spike_count"` → `<session>_rf_heatmap_count<suffix>.png`, once with `display_metric="spike_ratio"` → `<session>_rf_heatmap_ratio<suffix>.png`. Rename the existing file pattern accordingly (see backward-compatibility note in Rollback Plan).
- [x] 3.2 — Thread `disjoint_mask_distance_mm` from DAG config (`map_receptive_fields_clustered` params) → `map_receptive_fields_clustered_task` → `run_rf_cluster_mapping()` → `render_forearm_heatmap()` → `render_2d_heatmap()`.
- [x] 3.3 — Update `configs/analyse_workflow_dag.yaml` under `map_receptive_fields_clustered`:
  ```yaml
  disjoint_mask_distance_mm: 8.0
  ```
- [x] 3.4 — Update `rf_cluster_summary.json` per-cluster `summary_data` dict to include the new fields so downstream tooling sees them: `n_unique_touch_points: int` (rows in `spike_counts.csv` with `unique_touch_spike_count > 0`).

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py`
- `configs/analyse_workflow_dag.yaml`
- `code/src/analysis/receptive_field_mapping/map_receptive_fields_clustered_task.py` (or equivalent Prefect task wrapper — verify filename during implementation)

**Dependencies:** Phase 2.

### Phase 4: Manual verification
**Goal:** Confirm outputs match expectations on a known multi-region session.
**Started:** —
**Completed:** 2026-04-30

- [ ] 4.1 — Pick a session from `kinematics_simple_mean` + `type_stratified` that has multiple clusters (e.g., ST13-01 or ST14-01).
- [ ] 4.2 — Run the `map_receptive_fields_clustered` stage end-to-end via the existing DAG launcher (GUI) on that session.
- [ ] 4.3 — Visually verify per cluster folder:
  - `spike_counts.csv` has the new `unique_touch_spike_count` column, with values ≤ `spike_count`.
  - Both `_count.png` and `_ratio.png` exist and show consistent hotspot locations.
  - Metadata overlay shows touch counts and feature ranges.
  - Two convex-hull outlines visible in both 2D panels and 3D.
  - For a known multi-region RF: the 2D heatmap shows transparent gaps between hotspots, matching the 3D view.
- [ ] 4.4 — Check `cluster_description.json` includes `neuron_touches` and `neuron_cluster_touches` dicts.
- [ ] 4.5 — `rf_cluster_summary.json` remains valid JSON and includes the new summary field.

**Dependencies:** Phase 3.

---

## Testing Plan

### Unit Tests

Place new tests next to existing tests (check `tests/` layout during implementation — likely `tests/analysis/receptive_field_mapping/`).

- [ ] `test_extract_spike_contact_points_unique_touches` — synthetic aggregated DataFrame with 3 touches, one touch fires 3 spikes at the same position → `spike_count=3`, `unique_touch_spike_count=1`.
- [ ] `test_extract_spike_contact_points_multi_touch_same_position` — 2 different touches each fire 1 spike at same position → both counters = 2.
- [ ] `test_aggregate_spike_counts_preserves_unique_col` — pooling across sessions sums both counters correctly.
- [ ] `test_apply_distance_mask_preserves_hotspots` — synthetic grid + two disjoint sample clusters; cells inside threshold stay finite, between-cluster cells become NaN.
- [ ] `test_convex_hull_skip_when_too_few_points` — 0, 1, 2 input points → log info, no raise.
- [ ] `test_rfrendercontext_metadata_overlay_formatting` — snapshot test of the text string produced from a known context.

### Integration Tests

- [ ] `test_rf_cluster_pipeline_end_to_end_writes_both_pngs` — on a tiny fixture session, confirm both `_count.png` and `_ratio.png` land in the cluster folder with non-zero size.
- [ ] `test_cluster_description_contains_neuron_counts` — the emitted `cluster_description.json` contains the two new keys with integer values.

### Manual Verification
(covered in Phase 4 above)

### Edge Cases

- [ ] Cluster with zero spikes — `spike_counts.csv` empty; renderer should skip as it does today (covered by existing empty-check at `rf_cluster_visualizer.py:86`).
- [ ] Cluster with 1–2 contacts for a neuron — convex hulls skipped with log, metadata still rendered.
- [ ] Single-region RF — distance mask leaves the entire interpolated area visible (since all grid cells near samples).
- [ ] `unique_touch_spike_count` equals `neuron_cluster_touches` (every touch fired at this position) → ratio = 1.0, correctly rendered at the top of the colorbar.
- [ ] Multiple sessions per cluster — each session renders its own pair of PNGs, with per-session counts and perimeters.

---

## Documentation Plan

- [ ] Update `code/src/analysis/receptive_field_mapping/README.md` (if present; otherwise a docstring in `rf_cluster_pipeline.py`) describing the new `spike_counts.csv` schema and the two PNG outputs.
- [ ] No CLAUDE.md update needed — this is a local feature, not an architectural convention.
- [ ] Memory update after completion: amend `project_postprocessing_pipeline_state.md` to note the count/ratio dual output and the distance-masked 2D heatmap.
- [ ] No changelog file required (project convention doesn't use per-feature changelogs — verify during implementation and skip if so).

---

## Rollback Plan

1. **Before deployment:** Work on `feature/rf-cluster-visualization-improvements`. Merge is `--no-ff` per project convention; a single revert commit restores previous behaviour.
2. **Data considerations:**
   - `spike_counts.csv` gains a column — additive only, existing readers unaffected.
   - `cluster_description.json` gains two keys — additive only.
   - PNG filename scheme changes (`_rf_heatmap{suffix}.png` → `_rf_heatmap_count{suffix}.png` + `_rf_heatmap_ratio{suffix}.png`). On rollback, any downstream referencing the old path breaks — mitigated by grepping for the pattern before merging; verify no internal tooling hard-codes the old name during implementation.
   - No regenerated data is *wrong* under the old schema — re-running under old code simply overwrites with the old single PNG.
3. **Rollback procedure:** `git revert -m 1 <merge-sha>` on dev; old `spike_counts.csv` files keep the extra column (harmless) until their clusters are re-processed.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Downstream consumer hard-codes old `_rf_heatmap<suffix>.png` filename | Low | Med | Grep for the pattern before merging; update any references in the same PR. |
| `disjoint_mask_distance_mm=8.0` is wrong for some session geometries | Med | Med | Expose as DAG param (already planned); document the empirical basis (matches `map_scalars_to_mesh` radius + margin) in code comment. |
| Scipy `ConvexHull` raises on collinear points | Low | Low | Guard with `QhullError` catch → log info + skip that perimeter. |
| Memory blow-up from `unique_touch_counter` sets on very large sessions | Low | Low | Sets of `(int, int, int)` tuples — cheap. No expected issue at realistic scale (50k spike rows). |
| `mean_contact_x/y/z` missing from some clustered CSVs (older format) | Low | High (crash) | Per `CLAUDE.md` fail-fast convention: raise `ValueError` loudly naming the missing columns. Do not silently fall back. |
| Per-session contact sets are in raw XYZ; 3D hull must be drawn in the same frame as the forearm mesh | Med | Med | Apply the same tangent-plane rotation that the 3D renderer applies to spike contacts before drawing the 3D hull. Covered by task 2.6. |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 (pipeline data flow) | 0.5 day | None |
| Phase 2 (renderer + overlays + perimeters + 2D mask) | 1 day | Phase 1 |
| Phase 3 (wiring + config) | 0.25 day | Phase 2 |
| Phase 4 (manual verification) | 0.25 day | Phase 3 |

---

## References

- Planning procedure: `docs/development/planning-procedure.md`
- Fail-fast convention: `CLAUDE.md` (pipeline fail-fast section)
- 3D→2D projection catalogue: `docs/development/knowledge-base/note-3d-to-2d-surface-projection-algorithms.md`
- Related recent work: tangent-plane RF alignment, zero-padded cluster folder names, RF metrics module (see `memory/project_postprocessing_pipeline_state.md`).

---

## Critical Files (for implementation)

| File | Role |
|------|------|
| `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` | Pipeline orchestration; `_extract_spike_contact_points`, `_aggregate_spike_counts`, `_build_cluster_description`, render loop. |
| `code/src/analysis/receptive_field_mapping/rf_cluster_visualizer.py` | 3D renderer entry + projection dispatch. `render_forearm_heatmap()` at L45. |
| `code/src/analysis/receptive_field_mapping/rf_2d_renderer.py` | 2D panel renderer, interpolated heatmap, `griddata(method='cubic')` at L132. |
| `code/src/analysis/receptive_field_mapping/rf_projection.py` | `project_to_2d()` for projecting hull vertices in 2D panels. |
| `code/src/analysis/receptive_field_mapping/rf_surface_utils.py` | Reference pattern: KDTree-based radius queries already in use (`map_scalars_to_mesh`). |
| `code/src/analysis/receptive_field_mapping/rf_data_loader.py` | `parse_contact_points` reused unchanged. |
| `configs/analyse_workflow_dag.yaml` | `map_receptive_fields_clustered` params add `disjoint_mask_distance_mm`. |
