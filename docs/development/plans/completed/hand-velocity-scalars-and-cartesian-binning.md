# Plan: Hand-velocity scalar timeseries features + Cartesian-product binning clusterer

**Created:** 2026-04-29 11:08
**Approved:** —
**Completed:** 2026-04-30
**Author:** Basil Duvernoy
**Status:** Completed
**Branch:** `feature/hand-velocity-scalars-and-cartesian-binning`

---

## Overview

Add two new per-frame hand-velocity scalar timeseries (signed projection onto the per-touch principal motion axis, and unsigned amplitude) at the series-transform stage, and a new clustering strategy `cartesian_binning` that bins every input feature and emits one cluster per occupied combination of bin indices. These two changes together unlock richer scalar feature extraction and cross-feature joint binning analyses on touch events.

## Problem Statement

**Current limitations:**

1. **No first-class signed/unsigned 1-D velocity series.** The pipeline emits 3-D component velocities (`hand_velocity_{x,y,z}`) and uses the magnitude `‖v‖` only as a *temporary intermediate* inside `compute_mos_series` (`series_pipeline.py:225`). Downstream stages cannot consume "speed" or "signed velocity along the motion axis" without re-deriving them ad hoc. This makes natural features (mean speed, peak speed, signed mean motion through stroke reversals) unreachable from `StatisticalExtractor`, and forces every consumer to re-implement the projection.

2. **`BinningClusterer` discards cross-feature structure.** The existing strategy bins each feature independently but only retains the *highest-variance* feature's bin indices as the cluster label (`binning_clusterer.py:84`). Bin arrays for the other features are kept in metadata but never form clusters. When the experimenter wants to ask "what does a touch with high pressure AND high speed AND distal direction look like?", there is currently no way to express this as a clustering. We need a strategy that creates one cluster per occupied combination of feature bins.

**Why it matters:** Touch-feature analyses depend on the speed and the direction of motion as primary kinematic descriptors. Without persisted scalar series, the analysis pipeline cannot answer simple questions about speed without re-running ad-hoc code. And without joint binning, the experimenter cannot stratify clusters by combinations of features, which is the natural starting point for cross-tabulation analyses on the touch dataset.

## Goals

### In Scope

1. New per-frame column `hand_velocity_amplitude` in `<session>_series_augmented.csv` — `‖(vx, vy, vz)‖₂`, mm/s, always ≥ 0.
2. New per-frame column `hand_velocity_signed` in `<session>_series_augmented.csv` — signed projection of velocity onto a per-touch principal axis (PCA on velocity samples, sign convention aligned with mean velocity).
3. New `CartesianBinningClusterer` clustering strategy that:
   - bins every numeric feature (reusing the existing `pd.cut`/`pd.qcut` patterns),
   - assigns dense int cluster ids `0..K-1` over occupied bin combinations only,
   - supports a global `n_bins` plus optional `n_bins_per_feature` overrides.
4. YAML config wiring: enable both new transforms in the analysis DAG; make `cartesian_binning` available as a clustering method in `analyse_workflow_dag.yaml` (off by default for now).

### Out of Scope

- New feature extractors. The `StatisticalExtractor` already aggregates any numeric column (mean, max, min, std, median, range, skewness), so the two new series automatically gain `*_mean`, `*_max`, … per-touch features once they appear in the augmented CSV.
- Alternate sign conventions for the signed velocity (e.g. forearm long axis, stroke direction inferred elsewhere). The user chose per-touch PCA; other conventions can be added later as separate transforms.
- Refactoring `BinningClusterer` to share code with `CartesianBinningClusterer`. We intentionally keep the diff tight; minor duplication is acceptable.
- Heatmap / RF-mapping-side adaptations to the new cluster ids (the existing pipeline already accepts integer labels so no adaptation should be required).

## Success Criteria

- [ ] `<session>_series_augmented.csv` contains `hand_velocity_amplitude` ≥ 0 everywhere, equal (within 1e-6) to `sqrt(vx² + vy² + vz²)`.
- [ ] `<session>_series_augmented.csv` contains `hand_velocity_signed`, with `|hand_velocity_signed| ≤ hand_velocity_amplitude + 1e-6` per row, and per-touch `mean(hand_velocity_signed) ≥ 0` (sign convention).
- [ ] `StatisticalExtractor` produces `hand_velocity_amplitude_{mean,max,...}` and `hand_velocity_signed_{mean,max,...}` per-touch features without code changes.
- [ ] `CLUSTERER_REGISTRY` exposes `'cartesian_binning'`; `get_clusterer('cartesian_binning')` returns a fresh instance.
- [ ] Running `touch_clustering` with `cartesian_binning` enabled on a cluster group produces a clustering CSV with dense integer `cluster_label` values `0..K-1` and one `bin_<feature>` column per binned feature.
- [ ] `n_bins_per_feature` overrides propagate into the per-feature bin counts; an unknown override key raises `ValueError` listing the offending names.
- [ ] All existing tests under `code/tests/` continue to pass (`pytest code/tests/`).

---

## Technical Design

### Approach

**Velocity scalars (Part 1)** — add a new module `representation/series_level/velocity_scalar.py` with two pure functions (`compute_velocity_amplitude`, `compute_velocity_signed`) and wire them into `_transform_session` in `series_pipeline.py` exactly mirroring the existing `hand_velocity` / `hand_acceleration` toggle pattern.

For the signed variant, the per-touch principal motion axis `u` is the first right-singular vector of the centred velocity matrix `V_c = V − mean(V)`. We then **flip `u` if `mean(V) · u < 0`** so that `+` consistently means "in the dominant direction of motion". Projection uses **raw (uncentred) velocity** so that the value is comparable to the components and to the amplitude:

```python
_, s, Vt = np.linalg.svd(V - V.mean(axis=0), full_matrices=False)
u = Vt[0]
if np.dot(V.mean(axis=0), u) < 0:
    u = -u
hand_velocity_signed = V @ u
```

Edge case: when `s[0] < 1e-12` (effectively no motion), emit zeros for the touch (mathematically correct; not a fail-fast scenario).

**Cartesian binning (Part 2)** — add `clustering/cartesian_binning_clusterer.py` containing `CartesianBinningClusterer(TouchClusterer)`. The `fit_predict` method:
1. Bins each column independently via `pd.cut`/`pd.qcut`, reading per-feature `n_bins` from `n_bins_per_feature.get(col, n_bins_default)`.
2. Stacks per-column bin arrays into a `(n_touches, n_features)` matrix `B`.
3. Calls `np.unique(B, axis=0, return_inverse=True)` — the `inverse` array is the dense cluster id, naturally `0..K-1` over occupied combinations only.
4. Stores per-cluster bin-tuple mapping in metadata (`cluster_combinations`) for later inspection and emits the same `extra_columns = {'bin_<col>': bin_array}` shape that the pipeline already plumbs into the output CSV (`clustering_pipeline.py:619-625`).

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Velocity-1a sign reference: per-touch PCA** | Robust to forearm orientation; works without external direction inputs; aligns with "dominant motion axis" wording | PCA sign ambiguity needs convention; degenerate axis when motion is ~zero (handled with eps fallback) | **Chosen** (user choice) |
| Velocity-1a sign reference: forearm long axis | Stable across all touches in a session; direction is interpretable | Requires the forearm long-axis vector to be available at series-transform time; not currently exposed as a per-session input | Rejected |
| Velocity-1a sign reference: existing `direction.py` stroke-direction inference | Reuses existing module; stroke direction is already used | Module currently emits a coarse {proximal, distal} *label*, not a numeric axis; requires extending it | Rejected (out of scope) |
| **Cartesian-2: dense ids over occupied combos only** | Compact ID space; no empty clusters; matches pipeline label conventions | Cluster IDs are not stable across runs unless the dataset is identical (occupied-combos depend on data) | **Chosen** (user choice) |
| Cartesian-2: full Cartesian space (incl. empty combos) | IDs are reproducible given (n_bins, feature_order) regardless of data | Tens of thousands of unused IDs for >2 features × ~5 bins; hostile to downstream tooling that iterates clusters | Rejected |
| Cartesian-2: tuple-string labels | Self-describing; mirrors `TypeStratifiedClusterer` pattern | Object-dtype labels add overhead; many downstream tools assume int IDs; harder to diff across runs | Rejected |
| **Cartesian-2: per-feature `n_bins` override + global default** | Lets sparse/coarse features use fewer bins, fine-grained ones use more | Adds a config knob | **Chosen** (user choice) |
| Cartesian-2: single global `n_bins` (matches existing `BinningClusterer`) | Simplest config | Forces all features to share a coarseness; defeats much of the point of joint binning | Rejected |
| **Velocity insertion level: series-level columns** | Persisted per-frame; auto-aggregated by `StatisticalExtractor`; one place to compute | Adds two columns to every session CSV | **Chosen** (user choice) |
| Velocity insertion level: feature-extraction only | No CSV bloat; per-touch shape | Loses per-frame timeseries; cannot be visualized over time; can't be re-aggregated with new statistics later | Rejected |

### Knowledge-Base Relevance Check

Reviewed `docs/development/knowledge-base/` headings:
- `note-cupy-import-order.md` — N/A (no CuPy imports).
- `note-kinect-depth-access-single-path.md` — N/A (analysis stage; no Kinect depth reads).
- `note-git-merge-autonomous-fast-forward.md` — applies at merge time, not implementation.

The fail-fast convention from root `CLAUDE.md` is the primary constraint: **no silent fallbacks**. Applied to this plan:
- Unknown `n_bins_per_feature` keys → raise `ValueError` listing them.
- `n_bins < 2` → raise `ValueError`.
- Velocity SVD failure on NaN inputs → let `np.linalg.svd`'s exception propagate.
- Zero-motion touch (`s[0] < 1e-12`) → emit zeros (mathematically defined, not an error).

### Architecture Changes

**New files:**

```
code/src/analysis/touch_analytics/
├── representation/series_level/
│   └── velocity_scalar.py            — NEW: amplitude + signed projection helpers
└── clustering/
    ├── cartesian_binning_clusterer.py — NEW: CartesianBinningClusterer
    └── test_cartesian_binning_clusterer.py — NEW (optional but recommended)
```

**Modified files:**

- `code/src/analysis/touch_analytics/series_pipeline.py` — wire two new transform toggles, compute scalars per touch, concat columns into the output DataFrame, extend log lines and drop-input logic.
- `code/src/analysis/touch_analytics/clustering/__init__.py` — register `CartesianBinningClusterer` in `CLUSTERER_REGISTRY` and `__all__`.
- `configs/analyse_workflow_dag.yaml` — enable `hand_velocity_amplitude` and `hand_velocity_signed` transforms; add a `cartesian_binning` clustering method entry under one cluster group (e.g. `kinematics_simple_mean`) with `enabled: false`.

**Reused (unchanged):**

- `representation/series_level/kinematics.py` — `HAND_VELOCITY_COLUMNS`, `compute_velocity`, `get_kinematics`.
- `preparation/grouping.py::group_touches` — already used by `_transform_session` for per-touch iteration.
- `clustering/base.py` — `TouchClusterer`, `ClusteringContext`.
- `clustering_pipeline.py` `extra_columns` plumbing (lines 619–625) — accepts the new clusterer's metadata shape unchanged.

---

## Implementation Plan

### Phase 1: Hand-velocity scalar series transforms
**Goal:** Persist `hand_velocity_amplitude` and `hand_velocity_signed` per frame in the augmented CSV.
**Started:** 2026-04-29
**Completed:** 2026-04-29

- [x] 1.1 — Create `representation/series_level/velocity_scalar.py` with `compute_velocity_amplitude(vel_df)`, `compute_velocity_signed(vel_df)`, `HAND_VELOCITY_AMPLITUDE_COLUMN`, `HAND_VELOCITY_SIGNED_COLUMN`.
- [x] 1.2 — In `series_pipeline.py`, parse two new transform configs (`hand_velocity_amplitude`, `hand_velocity_signed`) with `enabled` and `drop_used_inputs` keys, mirroring existing toggles (lines 77–97).
- [x] 1.3 — In `_transform_session`, gate computation on the new flags inside the existing per-touch loop (lines 207–230) — when enabled, append the scalar series to two new dicts keyed by `id(group)`.
- [x] 1.4 — After the loop, concat each scalar dict into a column on `df` (mirroring the `vel_series` block at line 237).
- [x] 1.5 — Update the start-of-pipeline log line (lines 101–109) to include the two new toggles.
- [x] 1.6 — Update `need_velocity` (line 199) to also include the two new flags.

**Files Modified:**
- `code/src/analysis/touch_analytics/representation/series_level/velocity_scalar.py` — new.
- `code/src/analysis/touch_analytics/series_pipeline.py` — toggle handling, per-touch loop, concat, log line.

**Dependencies:** None.

### Phase 2: Cartesian-product binning clusterer
**Goal:** Provide a clustering strategy that emits one cluster per occupied combination of feature bins.
**Started:** 2026-04-29
**Completed:** 2026-04-29

- [x] 2.1 — Create `clustering/cartesian_binning_clusterer.py` with `CartesianBinningClusterer(TouchClusterer)` implementing `fit_predict`. Reuse the per-column binning logic pattern from `binning_clusterer.py` (do not refactor — duplicate inline).
- [x] 2.2 — Resolve per-feature `n_bins`: `n_bins_per_feature.get(col, n_bins_default)`. Validate up-front: any key in `n_bins_per_feature` that is not in `feature_df.columns` → `raise ValueError`.
- [x] 2.3 — Skip constant columns with a logged warning (matches `BinningClusterer`).
- [x] 2.4 — Stack per-column bin arrays into `(n_touches, n_features)`; call `np.unique(B, axis=0, return_inverse=True)` to get dense ids and the unique combination matrix.
- [x] 2.5 — Populate metadata: `algorithm`, `params`, `n_bins_default`, `n_bins_per_feature` (resolved), `bin_method`, `binned_features`, `bin_edges`, `n_clusters`, `cluster_combinations`, `extra_columns` (`bin_<col>` per column).
- [x] 2.6 — Register in `clustering/__init__.py`: import, add to `CLUSTERER_REGISTRY['cartesian_binning']`, add to `__all__`.
- [x] 2.7 — Add a unit test `code/src/analysis/touch_analytics/clustering/test_cartesian_binning_clusterer.py` covering: dense ids 0..K-1, occupied-only behaviour, per-feature override, unknown override key raises, constant column skipped, `n_bins < 2` raises.

**Files Modified:**
- `code/src/analysis/touch_analytics/clustering/cartesian_binning_clusterer.py` — new.
- `code/src/analysis/touch_analytics/clustering/test_cartesian_binning_clusterer.py` — new.
- `code/src/analysis/touch_analytics/clustering/__init__.py` — registry + `__all__` update.

**Dependencies:** None (independent of Phase 1).

### Phase 3: Configuration & integration
**Goal:** Wire the new components into the production DAG config and verify end-to-end.
**Started:** 2026-04-29
**Completed:** 2026-04-29

- [x] 3.1 — In `configs/analyse_workflow_dag.yaml`, under `touch_series_transforms.options.transforms`, add `hand_velocity_amplitude: {enabled: true, drop_used_inputs: false}` and `hand_velocity_signed: {enabled: true, drop_used_inputs: false}`.
- [x] 3.2 — In `configs/analyse_workflow_dag.yaml`, under `cluster_groups.kinematics_simple_mean.clustering_methods`, add a `cartesian_binning` entry with `method: cartesian_binning`, `n_bins: 5`, an example `n_bins_per_feature` mapping, `bin_method: equal_width`, `enabled: false`.
- [x] 3.3 — Run `pytest code/tests/` — confirm no regressions and any new tests pass.
- [ ] 3.4 — Launch GUI and run `touch_series_transforms` for one session; inspect the augmented CSV.
- [ ] 3.5 — Run `touch_feature_extraction` and confirm the two new columns are picked up by `StatisticalExtractor` automatically.
- [ ] 3.6 — Enable `cartesian_binning` for one cluster group, run `touch_clustering`, and inspect the clustering CSV.

**Files Modified:**
- `configs/analyse_workflow_dag.yaml` — transform toggles + cartesian_binning clustering method entry.

**Dependencies:** Phases 1 and 2.

---

## Testing Plan

### Unit Tests

- [ ] `test_cartesian_binning_clusterer.py::test_dense_ids_over_occupied_combos` — 2 features × 3 bins, only 4 actually-occupied combos in synthetic data → `K == 4`, labels are `{0,1,2,3}`.
- [ ] `test_cartesian_binning_clusterer.py::test_n_bins_per_feature_override` — set per-feature override to 2 vs 5; confirm `bin_<col>` arrays respect the override.
- [ ] `test_cartesian_binning_clusterer.py::test_unknown_override_raises` — `n_bins_per_feature: {nonexistent_col: 3}` → `ValueError`.
- [ ] `test_cartesian_binning_clusterer.py::test_constant_column_skipped` — a constant feature is dropped from binning with a warning; remaining columns drive the combos.
- [ ] `test_cartesian_binning_clusterer.py::test_n_bins_too_small_raises` — `n_bins = 1` → `ValueError`.
- [ ] *(optional)* `test_velocity_scalar.py::test_amplitude_matches_norm` — amplitude equals `sqrt(vx² + vy² + vz²)`.
- [ ] *(optional)* `test_velocity_scalar.py::test_signed_aligned_with_mean` — for a one-way motion sample, signed projections all positive.
- [ ] *(optional)* `test_velocity_scalar.py::test_zero_motion_returns_zeros` — all-zero velocity → all-zero signed output.

### Integration Tests

- [ ] After running `touch_series_transforms` on one real session, the augmented CSV contains both new columns and `hand_velocity_amplitude == sqrt(vx²+vy²+vz²)` within float tolerance.
- [ ] After running `touch_feature_extraction`, the per-feature CSV contains `hand_velocity_amplitude_{mean,max,...}` and `hand_velocity_signed_{mean,max,...}` columns automatically (no code change to extractors).
- [ ] After running `touch_clustering` with `cartesian_binning` enabled, the clustering CSV's `cluster_label` column has dense integer values whose maximum is `K-1` where `K` matches `metadata.n_clusters`.

### Manual Verification

- [ ] Inspect a sample touch group in the augmented CSV: confirm `mean(hand_velocity_signed)` is positive over the touch (sign convention).
- [ ] For a stroke session, plot `hand_velocity_signed` over time within one touch — expect it to dip near zero or go negative briefly during reversals.
- [ ] Open the cartesian-binning clustering output, group by `cluster_label`, and check that within a cluster the `bin_<feature>` columns are constant (one value per feature per cluster) — proves the Cartesian-product invariant.

### Edge Cases

- [ ] Zero-motion touch → `hand_velocity_signed` all zeros; no exception.
- [ ] Single-frame touch (rare; should already be filtered) — confirm no division-by-zero.
- [ ] All feature columns constant → `CartesianBinningClusterer` returns a single cluster (id 0) with a logged warning, matching `BinningClusterer`.
- [ ] `n_bins_per_feature` referencing a feature filtered out by the reduction stage → `ValueError` (because the feature is not present in `feature_df` at clustering time).

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` to mention `hand_velocity_amplitude` and `hand_velocity_signed` in the Stage 2 description, and `cartesian_binning` in the Stage 4 list.
- [ ] Update memory note `project_analysis_pipeline_state.md` once shipped to reflect the new transforms and clusterer.
- [ ] No user-facing guide change needed (these are research-pipeline internals).
- [ ] No changelog file convention is currently in use in this repo.

---

## Rollback Plan

These changes are additive and gated by config toggles, so rollback is low-risk.

1. **Before deployment:**
   - Set `hand_velocity_amplitude.enabled: false` and `hand_velocity_signed.enabled: false` in `configs/analyse_workflow_dag.yaml` to disable the new transforms.
   - Remove `cartesian_binning` entries (or set `enabled: false`) from `cluster_groups`.

2. **Data considerations:**
   - No schema migrations. Augmented CSVs without the new columns remain valid; consumers (`StatisticalExtractor`, clustering) skip absent columns.
   - Re-running with the toggles off regenerates CSVs without the new columns.

3. **Rollback procedure:**
   - `git revert <feature-commit-range>` on the feature branch.
   - Delete `<output>/series_transforms/*_series_augmented.csv` and `<output>/touch_features/`, `<output>/touch_clustering/` to force clean regeneration on next pipeline run.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| PCA sign ambiguity flips sign across similar touches in unexpected ways | Med | Med | Sign-flip rule (`mean(V)·u < 0 → flip`) gives a deterministic and intuitive convention; document the convention in the function docstring; add a unit test that covers a one-way motion sample. |
| Per-touch SVD over many touches × ~hundreds of frames could be slow | Low | Low | SVD on a 3-column matrix is `O(n)` per touch; should be negligible relative to existing per-touch work. Will benchmark on a real session and only optimize if it dominates the stage. |
| Cartesian-product clustering produces too many clusters when features are uncorrelated and `n_bins` is high | Med | Low | Document recommended `n_bins ≤ 5` and per-feature overrides for noisy features; emit a warning when `K > 1000`. |
| Adding two new columns to augmented CSVs invalidates existing caches; users could be surprised by re-runs | Med | Low | The `should_process_task` check already keys on inputs/outputs; users will see a one-time regeneration. Mention in the changelog/PR description. |
| Constant column detection edge case: `pd.qcut` with `duplicates='drop'` may produce 0 bins for ties-heavy features and break downstream `np.unique` axis=0 | Low | Med | After per-column binning, validate that every retained column has ≥ 2 distinct bin indices; otherwise drop with a warning matching `BinningClusterer`'s constant-column path. |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|------------------|--------------|
| Phase 1 (velocity scalars) | ~3 hours | None |
| Phase 2 (cartesian binning clusterer + tests) | ~4 hours | None |
| Phase 3 (config + integration verification) | ~1 hour | Phases 1 and 2 |

Total: ~1 day of focused work.

---

## References

- Existing strategy reference: `code/src/analysis/touch_analytics/clustering/binning_clusterer.py`
- Base interface: `code/src/analysis/touch_analytics/clustering/base.py`
- Pipeline integration: `code/src/analysis/touch_analytics/clustering_pipeline.py:573-625`
- Series-level pattern reference: `code/src/analysis/touch_analytics/series_pipeline.py:144-273`, `representation/series_level/kinematics.py`
- Related completed plans: `docs/development/plans/completed/binning-clustering-strategy.md`, `docs/development/plans/completed/split-analysis-pipeline.md`
- Analysis-stage CLAUDE notes: `code/src/analysis/CLAUDE.md`
