# Plan: RF cluster hulls from aggregated CSV `contact_points`

**Date:** 2026-04-25
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `feature/rf-cluster-visualization-improvements` (extends the active plan `rf-cluster-per-neuron-projection-frame.md`)

---

## Context

The fix in `rf-cluster-per-neuron-projection-frame.md` made the projection frame consistent across hull, spike heatmap, and forearm background. After that fix, the convex-hull perimeters STILL fail to enclose the heatmap. Manual code-trace identifies a structural set mismatch:

| Layer | Today's source | What it is |
|---|---|---|
| **Heatmap points** (`spike_counts_df`) | `_extract_spike_contact_points` parses `contact_points` from the aggregated CSV at every spike row (`Nerve_spike == 1`) | Many individual `(x, y, z)` per touch (the per-frame contact cloud) |
| **Hull points** (`neuron_contacts_xyz` / `neuron_cluster_contacts_xyz`) | `mean_contact_x/y/z` columns of `pooled_touch_summary_clustered.csv` | ONE centroid per touch — mean over frames of `contact_location_*`, which is itself the per-frame mean of the `contact_points` cloud |

The hull is mean-of-means across touches; the heatmap is the raw per-frame cloud. The convex hull of touch grand-centroids cannot enclose the cloud it summarises — this is structural, not a coordinate-frame issue.

**Fix direction:** keep reading the existing **aggregated CSV** (`*_semicontrolled_aggregated_session.csv`), but build hulls from its `contact_points` column instead of from `clustered_df.mean_contact_*`. Same source file as today, same source as the heatmap, just a second pass over the parsed cells (no Nerve_spike filter) to populate the hull arrays.

This guarantees by construction: `heatmap_pts ⊆ neuron_cluster_contacts ⊆ neuron_contacts`.

---

## Goals

### In Scope

1. Build `neuron_contacts_xyz[sid]` and `neuron_cluster_contacts_xyz[sid]` from **parsed `contact_points`** of the aggregated CSV (not `mean_contact_*` of `clustered_df`).
2. Read each session's aggregated CSV exactly **once per `(combo, clusterer)` pass**, with the union of all touch_keys for that session across all clusters; reuse the parsed dataframe across the per-cluster loop for both spike counting and hull construction.
3. Drop the now-unused `mean_contact_x/y/z` validation block in `rf_cluster_pipeline.py:491-498` (after confirming no other consumer in this stage needs it).
4. mm-level dedup of hull point clouds (round to ~1 mm, `np.unique`) to keep memory bounded.
5. Fail-fast (per CLAUDE.md): raise `ValueError` if a session's aggregated CSV produces zero parseable `contact_points` across all its touches.

### Out of Scope

- Switching to `_series_augmented.csv` (rejected — keep the existing source).
- `compute_rf_metrics` — already consumes `pooled_df` (the heatmap source), no change needed.
- The 3D `R = compute_tangent_plane_rotation(...)` rotation logic in `rf_cluster_visualizer.py` — unaffected.
- Cross-session comparability or persisting the projection frame to disk.

---

## Success Criteria

- [ ] Every rendered 2D heatmap: orange perimeter encloses every finite-colour cell; blue perimeter encloses orange.
- [ ] Set-containment assertion (verifiable in code): after mm-rounding, `set(spike_xyz) ⊆ set(neuron_cluster_contacts_xyz) ⊆ set(neuron_contacts_xyz)` for every `(session, cluster)` rendered.
- [ ] Each session's aggregated CSV is read exactly once per `(combo, clusterer)` pass, regardless of cluster count.
- [ ] `rf_cluster_summary.json` and `cluster_description.json` are byte-for-byte unchanged (pipeline only changes the renderer's hull source).
- [ ] Pipeline raises a clear `ValueError` with the session_id when the aggregated CSV yields zero parseable contact points for the requested touches.

---

## Technical Design

### Data flow change

**Today:**

```
aggregated CSV ── _extract_spike_contact_points (per cluster, ffill, Nerve_spike==1)
        ▼
spike Counter ─── heatmap

clustered_df.mean_contact_*  (separate path, mean-of-means)
        ▼
neuron_contacts_xyz / neuron_cluster_contacts_xyz ─── hulls
```

**After:**

```
aggregated CSV ── load once per session (with usecols), inline ffill of contact_points
        │
        ├── filter to all touches of the session, parse contact_points → neuron_contacts_xyz ─── blue hull
        │
        └── per cluster: filter to that cluster's touch_keys
              ├── parse all contact_points (not Nerve_spike-filtered) → neuron_cluster_contacts_xyz ─── orange hull
              └── filter Nerve_spike==1, parse contact_points → spike Counter ─── heatmap
```

Single source. Containment by construction.

### Refactor sketch (`rf_cluster_pipeline.py`)

1. **Replace `_extract_spike_contact_points`** with a session-level reader + per-touch-key parser pair, both operating on an in-memory dataframe:

   ```python
   def _load_session_aggregated(agg_csv: Path) -> pd.DataFrame:
       """Load the aggregated CSV with required columns, ffill contact_points per
       (block, trial, single_touch). Raises ValueError on missing columns / empty df."""
       # usecols=_NEEDED_COLS, validate columns, group-ffill contact_points
       ...

   def _parse_contacts_for_keys(
       df_session: pd.DataFrame,
       touch_keys: list[tuple] | None,  # None => all touches
       dedup_mm: float = 1.0,
   ) -> tuple[Counter, defaultdict, np.ndarray]:
       """Filter df_session to touch_keys (or all if None), then return:
         - spike_counter: from rows where Nerve_spike == 1
         - unique_touch_counter: ditto
         - all_contacts_xyz: mm-rounded unique cloud across ALL filtered rows
                             (NOT spike-filtered)."""
       ...
   ```

2. **Hoist session-level loading above the cluster loop.** For each session, read its aggregated CSV once with the union of all touch_keys for that session across all clusters (i.e., `clustered_df[clustered_df.session_id == sid][_KEY_COLS]`). Build `neuron_contacts_xyz[sid]` immediately.

3. **Inside the cluster loop**, replace the existing `_extract_spike_contact_points(...)` call (line 581) with `_parse_contacts_for_keys(session_dfs[sid], touch_keys=cluster_touch_keys)`. Harvest both the spike counter (heatmap) and the cluster contact cloud (orange hull).

4. **Drop dead code:**
   - `mean_contact_x/y/z` validation (lines 491-498).
   - `clustered_df`-based `neuron_contacts_xyz` (lines 504-511).
   - `clustered_df`-based `neuron_cluster_contacts_xyz` (lines 534-544).
   - `clustered_df` is still needed for `clustered_by_label`, `neuron_touches`, `neuron_cluster_touches`, and `_build_cluster_description`.

### `rf_cluster_visualizer.py`

- `RFRenderContext` field types unchanged. Update docstring on `neuron_contacts_xyz` / `neuron_cluster_contacts_xyz` to: "Unique mm-rounded individual contact points parsed from the session's aggregated CSV `contact_points` column. Invariant: heatmap spike contacts ⊆ neuron_cluster_contacts_xyz ⊆ neuron_contacts_xyz."
- No code change to the renderer itself — `projection_centroid = neuron_contacts_xyz.mean(axis=0)` still works (mean shifts numerically but the projection frame remains stable per session).

### mm-level dedup

```python
def _unique_mm(points_xyz: np.ndarray, mm: float = 1.0) -> np.ndarray:
    if len(points_xyz) == 0:
        return points_xyz
    rounded = np.round(points_xyz / mm) * mm
    return np.unique(rounded, axis=0)
```

Applied at the end of `_parse_contacts_for_keys`. A typical session yields O(10⁵) raw points → O(10³) unique mm cells; comfortably fits in memory.

---

## Files Modified

- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` — refactor `_extract_spike_contact_points` into `_load_session_aggregated` + `_parse_contacts_for_keys`; hoist session-level loading; drop the `mean_contact_*` path; harvest hull arrays from parsed `contact_points`.
- `code/src/analysis/receptive_field_mapping/rf_cluster_visualizer.py` — docstring update on `RFRenderContext` only.

No DAG / config / workflow-script changes (we keep the existing aggregated-CSV input).

---

## Verification

### End-to-end (manual)

1. Run `map_receptive_fields_clustered` via the GUI launcher on the multi-cluster session that exposed the bug.
2. Open both `_count.png` and `_ratio.png` for two clusters of one session: orange perimeter encloses every coloured cell; blue encloses orange.
3. Cross-cluster axis bounds remain identical (shared projection frame from prior fix).
4. Re-run with `projection_method="cylindrical_unwrap"`; enclosure must hold.

### Programmatic assertion

After parsing, before rendering, assert in code (debug log only — not a hard exception, since dedup rounding could place a heatmap point on a cluster cell boundary):

```python
spike_set = set(map(tuple, _round_mm(pooled_df[['x','y','z']].to_numpy())))
cluster_set = set(map(tuple, neuron_cluster_contacts_xyz[sid]))
neuron_set = set(map(tuple, neuron_contacts_xyz[sid]))
if not (spike_set <= cluster_set <= neuron_set):
    logger.warning("RF hull invariant violated for session=%s, cluster=%s", sid, cluster_label)
```

### Fail-fast cases

- Aggregated CSV missing for a session → existing skip-with-warning behaviour preserved.
- Aggregated CSV present but `contact_points` column absent → existing skip-with-warning preserved.
- Aggregated CSV present, column present, but **zero rows match the requested touch_keys** → preserve today's silent-empty (returns empty Counter / empty xyz).
- Aggregated CSV present, column present, but **all parsed cells are empty `[]`** for a session whose touch_keys do match → `ValueError` per CLAUDE.md (data is malformed, not a sparse-cluster case).

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Reading the entire aggregated CSV for all touches per session is slower than today's per-cluster filtered read | Med | Low | One read per session per `(combo, clusterer)` instead of N reads (one per cluster). Net improvement for ≥2 clusters. Use `usecols=_NEEDED_COLS`. |
| Memory: parsing all `contact_points` for a session before mm-dedup may spike | Low | Low | Stream-parse: iterate rows, parse each cell, accumulate into a `set[tuple[int, int, int]]` of mm-rounded ints. Single pass; no giant intermediate array. |
| `clustered_df.session_id` set ≠ aggregated-CSV set (orphan sessions) | Low | Med | Existing `session_dir_map` gating preserved; missing aggregated CSV still warn-and-skip. |
| Subtle change in spike-counter values vs. today's `_extract_spike_contact_points` | Low | High | Spike-counter logic preserved verbatim inside `_parse_contacts_for_keys` (same Nerve_spike==1 filter, same parse, same Counter accumulation). Only the surrounding scaffold changes. |
| Loss of `clustered_df.mean_contact_*` triggers downstream failure | Low | Low | Audit confirmed: only the rf_cluster_pipeline stage references `mean_contact_*` in this pipeline; `_build_cluster_description` does not need it. |

---

## Plan Lifecycle

This is the **third** plan in the visualization improvement sequence:

1. `rf-cluster-visualization-improvements.md` (active) — original visualization work.
2. `rf-cluster-per-neuron-projection-frame.md` (active) — projection-frame fix; Phase 3 manual verification surfaced the structural mismatch this plan addresses.
3. **This plan** — finishes the enclosure guarantee by switching the hull data source from `mean_contact_*` to parsed `contact_points`, both pulled from the same aggregated CSV that already feeds the heatmap.

When all three are complete, all three move to `completed/` together.

---

## References

- Predecessor plan: `docs/development/plans/active/rf-cluster-per-neuron-projection-frame.md`
- Aggregated CSV reader (today): `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py:114` (`_extract_spike_contact_points`)
- `parse_contact_points`: `code/src/analysis/receptive_field_mapping/rf_data_loader.py:47`
- `mean_contact_*` derivation chain (for context on why centroid-of-centroids fails to enclose): `code/src/analysis/touch_analytics/extraction_pipeline.py:498` ← `code/src/preprocessing/motion_analysis/tactile_quantification/model/objects_interaction_processor.py:182`
- Fail-fast convention: `CLAUDE.md` — "Fail-fast pipeline — no silent fallbacks"
