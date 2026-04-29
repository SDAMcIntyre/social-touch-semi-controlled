# Plan: GMM 2-D Feature-Space View (pressure × signed velocity)

**Date:** 2026-04-29
**Author:** Basil Duvernoy
**Status:** Completed
**Branch:** `feature/gmm-feature-space-view`

---

## Overview

Add a single PNG per `<cluster_group>/gmm/` output that shows touches as a
scatter colored by `cluster_label`, with each fitted GMM component's mean
and 1σ/2σ covariance ellipses overlaid on a 2-D plane defined by
`pressure_mean` (x) and a chosen *signed* velocity component (y, default
`hand_velocity_y_mean`). Ellipses are produced by analytically marginalising
the K-component GMM from the full scaled feature space onto the chosen
2-D subspace, then inverse-transforming to physical units.

## Problem Statement

The recently-added GMM clusterer (`feature/gmm-clusterer`) operates on
`pressure_velocity_mean` — a group that resolves to 4 numeric columns
(`pressure_mean`, `hand_velocity_x_mean`, `hand_velocity_y_mean`,
`hand_velocity_z_mean`). It emits hard labels and per-component soft
posteriors but **no visualisation in the feature space the algorithm
operated on**. The existing `_generate_session_heatmaps()` in
`clustering_pipeline.py:706` produces 2-D *density* maps but is not
GMM-aware (no cluster colors, no component ellipses). The RF gallery
viewer shows clusters on the 3-D forearm mesh but never on the touch
features themselves. As a result there is no way to inspect, sanity-check,
or report the GMM partition without writing ad-hoc notebook code.

## Goals

### In Scope
1. Persist enough information in `cluster_metadata.json` to re-construct the fitted GMM (means, covariances, scaler params) without re-fitting
2. Render `feature_space.png` next to `pooled_touch_summary_clustered.csv` for every GMM clustering result
3. Configurable axes via the `cluster_group.visualization` block (default: `pressure_mean` × `hand_velocity_y_mean`)
4. Correct marginalisation of the K-component GMM onto the chosen 2-D subspace, in physical units

### Out of Scope
- Per-session feature-space PNGs (clustering is pooled — pooled view is the natural primary artifact)
- Posterior-probability heatmap shading or contour plots (deferred)
- Pairwise scatter matrix or PCA(2) views
- Embedding the view in the RF gallery viewer or making it interactive
- Equivalent views for non-GMM clusterers (scatter without ellipses could be added later)
- Exposing the `visualization` block in the `cluster_group_dialog.py` GUI

## Success Criteria

- [ ] `cluster_metadata.json` for any GMM run contains JSON-serializable `means`, `covariances`, `scaler_mean`, `scaler_scale`, `retained_columns`
- [ ] `feature_space.png` is written alongside `pooled_touch_summary_clustered.csv` whenever `metadata['algorithm'] == 'gmm'`
- [ ] Plot axes are in physical units (not standardized space)
- [ ] 1σ ellipses enclose ≈ 39 % of each cluster's points; 2σ ≈ 86 % (within ~5 pp of theoretical for healthy fits)
- [ ] `force_processing: false` re-uses the existing PNG; `force_processing: true` regenerates it
- [ ] Pipeline raises `ValueError`/`KeyError` (per fail-fast convention) on missing metadata or unknown feature names — no silent fallback

---

## Technical Design

### Approach

The GMM is fit in scaled D-dimensional space (D = 4 for `pressure_velocity_mean`).
The marginal of a multivariate Gaussian onto any subset of axes is exact:
sub-vector of the mean and the corresponding sub-matrix of the covariance.
For component k:

```
indices = [retained_columns.index(x_feature), retained_columns.index(y_feature)]
μ_2d_scaled = means[k][indices]
Σ_2d_scaled = covariances[k][np.ix_(indices, indices)]

# Inverse StandardScaler (per-axis affine)
s = scaler_scale[indices]
m = scaler_mean[indices]
μ_2d_orig = μ_2d_scaled * s + m
Σ_2d_orig = diag(s) @ Σ_2d_scaled @ diag(s)
```

Eigendecomposition of `Σ_2d_orig` gives ellipse semi-axes (`√eigenvalue`)
and rotation (angle of dominant eigenvector). Plot at 1σ and 2σ.

This works correctly for `covariance_type ∈ {full, tied, diag, spherical}`
because all four shapes are returned by sklearn as full DxD matrices when
re-expanded — except `tied` which returns a single shared covariance: the
GMM clusterer must broadcast it to per-component matrices before persisting.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|---|---|---|---|
| **Marginalise + inverse-transform from persisted means/covariances** | Exact, no re-fit, deterministic, runs offline from the CSV+JSON | Requires extending GMM metadata; small JSON growth | **Chosen** |
| Re-fit a fresh GMM on the 2 chosen columns at render time | No metadata change | Different fit (different K, different ellipses) — confuses users; loses the BIC-selected model | Rejected |
| Density estimate via `predict_proba` evaluated on a grid (no ellipses) | Avoids the marginalisation math | Soft-probability shading is a separate, future feature; doesn't show *what GMM fit*; user explicitly chose ellipses | Rejected |
| Velocity magnitude on Y (unsigned) | True 2-D, easy axes | Loses stroke direction — user explicitly asked to keep the sign | Rejected |
| PCA(2) of the scaled feature matrix | Captures most variance | PC axes not interpretable in physical units | Rejected |

### Architecture Changes

No new modules required, but a new file is the cleanest fit:

```
code/src/analysis/touch_analytics/clustering/
├── feature_space_renderer.py   ← NEW (matplotlib renderer)
├── gmm_clusterer.py            ← extended metadata
└── ...
code/src/analysis/touch_analytics/
└── clustering_pipeline.py      ← scaler param injection + render hook
```

Integration touchpoints:

- `gmm_clusterer.py::fit_predict` returns `means`, `covariances`,
  `feature_columns` in metadata (alongside existing keys).
- `clustering_pipeline.py` enriches metadata with scaler params from the
  `ReductionPipeline` it already holds, then dispatches on
  `metadata['algorithm']` to call the new renderer.
- The renderer is self-contained (no pipeline imports) so it can also be
  driven from a notebook.

---

## Implementation Plan

### Phase 1: GMM metadata enrichment
**Goal:** GMM's `fit_predict` returns enough info to draw ellipses

**Tasks:**
- [x] Task 1.1 — In `gmm_clusterer.py::fit_predict`, extract `best_model.means_` (shape K×D), `best_model.covariances_` (shape K×D×D, broadcasting the single matrix from `covariance_type='tied'` to per-component matrices), and `feature_df.columns.tolist()`. Add to metadata as `means`, `covariances`, `feature_columns`. All values must be plain Python lists for JSON serializability.
- [x] Task 1.2 — Update the early-exit branch (`n < 2 * min_touches`) to return the same keys with single-component placeholders consistent with K=1.
- [x] Task 1.3 — Update docstring to document the new metadata keys.

**Files Modified:**
- `code/src/analysis/touch_analytics/clustering/gmm_clusterer.py` — extend metadata return value

**Dependencies:** None

### Phase 2: Pipeline metadata injection + render dispatch
**Goal:** Pipeline writes scaler params and triggers the renderer for GMM runs

**Tasks:**
- [x] Task 2.1 — In `clustering_pipeline.py`, after `clusterer.fit_predict(...)` and before writing `cluster_metadata.json`, read `mean_` and `scale_` from the fitted `ReductionPipeline.scaler_` (or the equivalent attribute) and `retained_columns` from the reduction step. Add to metadata as `scaler_mean`, `scaler_scale`, `retained_columns`. Raise `ValueError` if the scaler is not a `StandardScaler`-compatible affine (the visualisation math assumes per-axis affine — fail-fast).
- [x] Task 2.2 — After `pooled_touch_summary_clustered.csv` and `cluster_metadata.json` are written, if `metadata['algorithm'] == 'gmm'`, look up `cluster_groups[<group>].visualization.x_feature` / `.y_feature`. Default to `pressure_mean` and the first `hand_velocity_*_mean` column found in `retained_columns`. Pass these plus the result_df, metadata, and output dir to `render_gmm_feature_space(...)`.
- [x] Task 2.3 — Validate that both axis names are in `retained_columns` — raise `KeyError` listing available columns if not.

**Files Modified:**
- `code/src/analysis/touch_analytics/clustering_pipeline.py` — metadata enrichment block, render dispatch
- `configs/analyse_workflow_dag.yaml` — add `visualization: { x_feature: pressure_mean, y_feature: hand_velocity_y_mean }` under `pressure_velocity_mean`

**Dependencies:** Phase 1

### Phase 3: 2-D feature-space renderer
**Goal:** Standalone matplotlib renderer that produces `feature_space.png`

**Tasks:**
- [x] Task 3.1 — Create `feature_space_renderer.py` with `render_gmm_feature_space(result_df, metadata, x_feature, y_feature, output_path)`. Use `matplotlib.use('Agg')` at module top (matches `rf_2d_renderer.py:5-9`).
- [x] Task 3.2 — Implement `_marginalise_2d(means, covariances, scaler_mean, scaler_scale, retained_columns, x_feature, y_feature)` returning per-component `(mu_2d_orig, sigma_2d_orig)` in physical units.
- [x] Task 3.3 — Implement `_ellipse_from_cov(mu, cov, n_sigma)` returning a `matplotlib.patches.Ellipse` via eigendecomposition (semi-axes = `n_sigma * sqrt(eigenvalues)`, angle from dominant eigenvector). Raise `ValueError` if `cov` is not positive-definite (fail-fast for degenerate `tied`/`spherical` edges).
- [x] Task 3.4 — Render: scatter colored by `cluster_label` (alpha 0.5), per-component mean marker (×, larger), 1σ ellipse (solid), 2σ ellipse (dashed). Title: `gmm — k=K — cov=<type> — BIC=<best_bic>`. Legend: cluster id and size. Black background to match existing presentation style (`rf_2d_renderer.py`).

**Files Modified:**
- `code/src/analysis/touch_analytics/clustering/feature_space_renderer.py` — NEW (~150 lines)

**Dependencies:** Phase 2

### Phase 4: Tests
**Goal:** Verify metadata schema, marginalisation math, and rendering

**Tasks:**
- [x] Task 4.1 — Extend `test_gmm_clusterer.py::test_gmm_basic_multimodal` to assert `means`, `covariances`, `feature_columns` present with correct shapes.
- [x] Task 4.2 — Extend `test_gmm_metadata_json_serializable` to assert the new keys round-trip through `json.dumps`/`json.loads`.
- [x] Task 4.3 — Create `test_feature_space_renderer.py`: synthetic 4-D data with known means/covariances; assert `_marginalise_2d` recovers the exact 2×2 sub-matrix after scaling; assert `_ellipse_from_cov` returns the correct angle for an axis-aligned diagonal covariance; assert renderer writes a non-empty PNG.
- [x] Task 4.4 — Add a `KeyError` test: requested `y_feature` not in `retained_columns` raises with helpful message.

**Files Modified:**
- `code/tests/test_gmm_clusterer.py` — extend existing tests
- `code/tests/test_feature_space_renderer.py` — NEW (~120 lines)

**Dependencies:** Phase 3

---

## Testing Plan

### Unit Tests
- [ ] GMM metadata schema: `means` (K×D), `covariances` (K×D×D), `feature_columns` (D), all JSON-serializable
- [ ] `_marginalise_2d` round-trip: scaled→original→scaled returns input within `1e-9`
- [ ] `_ellipse_from_cov` axis-aligned diagonal: rotation is 0°, semi-axes match `n_sigma * sqrt(diag(cov))`
- [ ] `_ellipse_from_cov` raises `ValueError` for a non-PSD matrix
- [ ] Renderer produces a non-empty PNG with the expected DPI/size for synthetic 3-cluster data
- [ ] Pipeline raises `KeyError` if `visualization.y_feature` is not in `retained_columns`
- [ ] Pipeline raises `ValueError` if the reduction scaler is not a per-axis affine

### Integration Tests
- [ ] Run `touch_clustering` on a session with `pressure_velocity_mean.gmm` enabled. Verify `feature_space.png` exists, `cluster_metadata.json` has all new keys, BIC and K match between metadata and PNG title.
- [ ] Re-run with `force_processing: false` — PNG mtime unchanged.
- [ ] Re-run with `force_processing: true` — PNG mtime updated.

### Manual Verification
- [ ] Visually inspect `feature_space.png`: clusters look plausible vs. point density, ellipses encompass the right fraction of points (≈ 39 % at 1σ).
- [ ] Switch `covariance_type` between `full`, `tied`, `diag`, `spherical` and re-run — for `spherical`, ellipses become circles in scaled space (may stretch in physical units when `scale_` differs across axes — this is correct behavior, not a bug).
- [ ] Override `visualization.y_feature` to `hand_velocity_x_mean` and confirm the plot updates accordingly.

### Edge Cases
- [ ] Dataset with `n < 2 * min_touches` (early-exit branch): metadata still has `means`/`covariances` (single component placeholders), and the renderer either skips with a logged warning or draws a single-cluster plot — pick one, document it.
- [ ] Dataset where one component collapses to ~0 variance: 2σ ellipse may still draw correctly; if eigenvalues are < 1e-12, the renderer raises rather than producing a malformed plot.

---

## Documentation Plan

- [ ] Add a one-line entry under "Stage 4 — Clustering" in `code/src/analysis/CLAUDE.md` mentioning the GMM-only `feature_space.png` artifact
- [ ] Docstring on `render_gmm_feature_space` documenting the marginalisation + inverse-transform math
- [ ] Brief note in the GMM section of `cluster_metadata.json` schema (in-code docstring on `gmm_clusterer.py`) listing the new keys

---

## Rollback Plan

1. Revert the feature branch — no schema migrations: `cluster_metadata.json` and `feature_space.png` are regenerated on each run.
2. The new metadata keys are additive; consumers that don't know about them ignore them.
3. If only the renderer is buggy: gate it behind a `visualization.enabled: true|false` flag in the cluster_group spec (default true) so it can be disabled without removing the metadata changes.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Bootstrap stability re-runs GMM 20× per group; persisting K means/covariances per call inflates memory transiently | Low | Low | Discard non-best models inside `fit_predict`; only the winner is stored. Already done today — no change. |
| `cluster_metadata.json` size growth (D×D matrices × K components) | Low | Low | Worst case K=15, D=6 → 540 floats per group — negligible. |
| User changes `visualization.y_feature` to something not in `retained_columns` (e.g. variance filter dropped it) | Medium | Low | Fail-fast `KeyError` listing actual `retained_columns`. |
| `tied`/`spherical` covariance produces near-degenerate 2-D submatrix | Low | Medium | `_ellipse_from_cov` raises on non-PSD; user sees a clear error and can switch to `full`/`diag`. |
| Velocity-axis sign convention differs across studies (forearm orientation) | Medium | Low | `visualization.y_feature` is configurable per cluster group. |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|---|---|---|
| Phase 1: GMM metadata enrichment | ~30 lines | None |
| Phase 2: Pipeline injection + dispatch | ~40 lines | Phase 1 |
| Phase 3: Feature-space renderer | ~150 lines | Phase 2 |
| Phase 4: Tests | ~150 lines | Phase 3 |

---

## References

- `code/src/analysis/touch_analytics/clustering/gmm_clusterer.py` — current GMM implementation
- `code/src/analysis/touch_analytics/clustering_pipeline.py` — call site (line 596) and existing `_generate_session_heatmaps` (line 706)
- `code/src/analysis/touch_analytics/reduction/pipeline.py` — `ReductionPipeline` exposing the fitted scaler
- `code/src/analysis/receptive_field_mapping/rf_2d_renderer.py` — reference matplotlib + Agg backend pattern
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py::_format_generation_params` — already pulls `k`, `covariance_type`, `features` from GMM metadata (commit `08d837a`)
- `docs/development/plans/active/gmm-clusterer.md` — original GMM clusterer plan (this is the visualisation follow-up)
- `configs/analyse_workflow_dag.yaml` — `pressure_velocity_mean` cluster_group definition
