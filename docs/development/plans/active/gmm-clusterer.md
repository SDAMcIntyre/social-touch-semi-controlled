# Plan: GMM Clusterer Plugin

**Created:** 2026-04-28 18:00
**Approved:** —
**Completed:** —
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `feature/gmm-clusterer`

---

## Overview

**What:** Add a Gaussian Mixture Model (GMM) clusterer as a 6th plugin to the touch_analytics clustering architecture, using BIC-based model selection to automatically determine the optimal number of components.

**Why:** The current clustering toolbox lacks a method that directly models the Gaussian/multi-Gaussian distributions observed in touch features. After investigating 10+ scientific clustering methods, GMM + BIC emerged as the theoretically optimal fit for the project's data profile (500-2000 touches, 2-4 features, Gaussian-like distributions, meaningful partitioning for downstream RF mapping).

**How:** Implement `GMMClusterer(TouchClusterer)` following the existing plugin pattern — BIC sweep across K=1..max_components, adaptive minimum-component enforcement, soft membership probabilities stored via the `extra_columns` mechanism.

## Problem Statement

- The five existing clusterers (binning, K-Means, DBSCAN, hierarchical, type-stratified) do not exploit the known Gaussian structure of touch features. K-Means assumes spherical clusters; DBSCAN uses a fragile global epsilon; binning has no distance metric; hierarchical uses coverage constraints but not distributional modeling.
- Touch features (pressure, velocity, contact depth/area) follow Gaussian or multi-Gaussian distributions as observed in histogram inspection. A clustering method that directly models this structure will produce more accurate, interpretable, and stable partitions.
- The downstream RF mapping pipeline depends on cluster quality — better clusters yield more meaningful per-cluster receptive fields.

## Goals

### In Scope
1. Implement `GMMClusterer` conforming to the `TouchClusterer` interface with BIC-based model selection
2. Support configurable covariance type, max components, and minimum component size via DAG YAML
3. Store soft membership probabilities as `extra_columns` in the output CSV for downstream use
4. Report full BIC curve, component weights, and convergence info in cluster metadata
5. Register in `CLUSTERER_REGISTRY` and add DAG config entries
6. Comprehensive test suite covering normal operation and edge cases

### Out of Scope
- Bayesian GMM (Dirichlet Process) — future plan for cross-checking K selection
- HDBSCAN as DBSCAN replacement — separate enhancement
- Nadaraya-Watson kernel regression slider for continuous RF visualization — separate feature
- Soft-assignment-weighted RF mapping — future enhancement using the probability columns
- Anderson-Darling post-hoc Gaussianity validation — future diagnostic tool

## Success Criteria

- [ ] `GMMClusterer` passes all unit tests (10 test cases)
- [ ] GMM registered as `"gmm"` in `CLUSTERER_REGISTRY` and instantiable via `get_clusterer("gmm")`
- [ ] DAG config includes `gmm` entry under `pressure_velocity_mean` cluster group
- [ ] Pipeline produces output CSV with `cluster_label` and `gmm_prob_*` columns under `4_analysed/touch_clusters/<group>/gmm/`
- [ ] `cluster_metadata.json` contains `bic_scores`, `component_weights`, `convergence`, `internal_metrics`, and `stability` sections
- [ ] Bootstrap stability (ARI) completes without error

---

## Technical Design

### Approach

Implement GMM using `sklearn.mixture.GaussianMixture`. For each clustering run:
1. Sweep `n_components` from 1 to `max_components` (clamped by data size and `min_touches_per_component`)
2. Fit each model with `n_init` random initializations and `random_state=42`
3. Select K that minimizes BIC
4. Enforce minimum component size by removing violated K values from candidates and selecting next-best BIC
5. Store hard labels (for pipeline compatibility) and soft probabilities (as `extra_columns`)

Cache all fitted models during the BIC sweep to avoid re-fitting the winner.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **GMM + BIC** | Directly models Gaussian structure; principled K selection; soft assignments; 2-4D sweet spot | BIC is asymptotic (less reliable at N~500); bootstrap re-runs are slower | **Chosen** |
| **Bayesian GMM (DPGMM)** | Auto K selection without sweep; regularized by priors | Slower (variational inference); weight_concentration_prior needs tuning | Deferred — future cross-check |
| **HDBSCAN** | Auto K; noise detection; varying density | Does not exploit Gaussian structure; transductive | Deferred — complementary method |
| **K-Means (existing)** | Fast; simple | Spherical assumption; no soft assignments; strictly dominated by GMM for Gaussian data | Already exists; GMM supersedes for this data type |
| **Mean Shift** | Non-parametric mode-seeking; auto K | No soft assignments; bandwidth heuristic fragile; no distributional modeling | Rejected — GMM superior for Gaussian data |

### Architecture Changes

No architectural changes required. GMM plugs into the existing architecture:

```
code/src/analysis/touch_analytics/clustering/
├── __init__.py              ← add import + registry entry
├── base.py                  (unchanged — GMM conforms to TouchClusterer)
├── binning_clusterer.py     (unchanged)
├── dbscan_clusterer.py      (unchanged)
├── hierarchical_clusterer.py (unchanged)
├── kmeans_clusterer.py      (unchanged — used as code style reference)
├── type_stratified_clusterer.py (unchanged — can wrap GMM as base_method)
└── gmm_clusterer.py         ← NEW
```

**Integration points (all existing, no changes needed):**
- `clustering_pipeline.py` line 571: `get_clusterer(method)` — instantiates `GMMClusterer` when `method: gmm`
- `clustering_pipeline.py` line 596: `clusterer.fit_predict(scaled_df, config, context)` — GMM conforms to this signature
- `clustering_pipeline.py` line 608: `metadata.pop('extra_columns', None)` — writes `gmm_prob_*` columns to CSV
- `evaluation/internal_metrics.py`: silhouette, DB, CH work on hard labels (filters noise=-1)
- `evaluation/stability.py`: bootstrap ARI calls `fit_predict` on subsamples — GMM re-runs BIC per subsample (correct but slower)

---

## Implementation Plan

### Phase 1: Core Implementation
**Goal:** Create the `GMMClusterer` class with BIC model selection and soft assignments
**Started:** 2026-04-28
**Completed:** 2026-04-28

- [x] Task 1.1 — Create `gmm_clusterer.py` with module constants (`_DEFAULT_MAX_COMPONENTS=15`, `_DEFAULT_COVARIANCE_TYPE="full"`, `_DEFAULT_N_INIT=10`, `_DEFAULT_MIN_TOUCHES_PER_COMPONENT=30`)
- [x] Task 1.2 — Implement `GMMClusterer.fit_predict()` with: early guard for too-few samples, BIC sweep loop caching fitted models, best-K selection, adaptive min-component enforcement, soft assignment extraction
- [x] Task 1.3 — Implement `_build_metadata()` helper returning algorithm, params, k, cluster_sizes, bic_scores, covariance_type, component_weights, convergence, extra_columns
- [x] Task 1.4 — Handle edge cases: single-component result (K=1), convergence failure (raise `RuntimeError` per fail-fast convention)

**Files Modified:**
- `code/src/analysis/touch_analytics/clustering/gmm_clusterer.py` — NEW (~100 lines)

**Dependencies:** None

### Phase 2: Registration and Configuration
**Goal:** Make GMM available to the pipeline via registry and DAG config
**Started:** 2026-04-28
**Completed:** 2026-04-28

- [x] Task 2.1 — Add `from .gmm_clusterer import GMMClusterer` import to `clustering/__init__.py`
- [x] Task 2.2 — Add `'gmm': GMMClusterer` to `CLUSTERER_REGISTRY` and `'GMMClusterer'` to `__all__`
- [x] Task 2.3 — Add `gmm` clustering method entry under `pressure_velocity_mean` in `analyse_workflow_dag.yaml`

**Files Modified:**
- `code/src/analysis/touch_analytics/clustering/__init__.py` — add import, registry entry, `__all__` entry
- `configs/analyse_workflow_dag.yaml` — add `gmm` method config block

**Dependencies:** Phase 1

### Phase 3: Test Suite
**Goal:** Validate GMM clusterer with comprehensive tests covering normal and edge cases
**Started:** 2026-04-28
**Completed:** 2026-04-28

- [x] Task 3.1 — Create `test_gmm_clusterer.py` with synthetic data fixtures
- [x] Task 3.2 — Implement 10 test cases (see Testing Plan below)
- [x] Task 3.3 — Run tests and verify all pass

**Files Modified:**
- `code/tests/test_gmm_clusterer.py` — NEW (~200 lines)

**Dependencies:** Phase 2

---

## Testing Plan

### Unit Tests
- [ ] `test_gmm_basic_multimodal` — 3-Gaussian synthetic data (300 pts, 2 features); K found between 2-5, metadata keys present, convergence True
- [ ] `test_gmm_single_component` — unimodal blob (100 pts); BIC selects K=1, labels all-zero
- [ ] `test_gmm_min_touches_enforcement` — imbalanced 2-component data (200+10 pts); adaptive K reduction respects `min_touches_per_component=30`
- [ ] `test_gmm_too_few_samples` — 20 samples with `min_touches_per_component=30`; warning logged, all assigned to cluster 0
- [ ] `test_gmm_covariance_types` — parametrize over `['full', 'tied', 'diag', 'spherical']`; valid labels and convergence for each
- [ ] `test_gmm_convergence_failure` — mock `converged_=False`; `RuntimeError` raised with descriptive message
- [ ] `test_gmm_registry` — `'gmm' in CLUSTERER_REGISTRY` and `isinstance(get_clusterer('gmm'), GMMClusterer)`
- [ ] `test_gmm_soft_assignments_shape` — `extra_columns` has exactly K probability columns, each row sums to ~1.0
- [ ] `test_gmm_metadata_json_serializable` — `json.dumps(metadata)` succeeds without error
- [ ] `test_gmm_deterministic` — two runs with same data and config produce identical labels

### Integration Tests
- [ ] Enable `gmm` in DAG config, run `analysis_workflow.py`, verify output CSV and metadata JSON are written correctly
- [ ] Verify RF visualization pipeline picks up GMM clusters when `extract_receptive_fields_clustered` is enabled

### Manual Verification
- [ ] Run pipeline on `pressure_velocity_mean` with `gmm` enabled, inspect output CSV for `cluster_label` and `gmm_prob_*` columns
- [ ] Inspect `cluster_metadata.json` for BIC curve, component weights, internal metrics, and stability scores
- [ ] Compare GMM cluster assignments visually against binning results for reasonableness

### Edge Cases
- [ ] Dataset with all identical feature values (constant columns) — should warn and assign single cluster
- [ ] Dataset with exactly `min_touches_per_component * 2` samples — boundary for BIC sweep range
- [ ] `type_stratified` wrapper with `base_method: gmm` — verify GMM works as delegated clusterer

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` — add `gmm` to the clustering methods list in the Stage map
- [ ] Docstring in `gmm_clusterer.py` — document config keys, scaling assumption, BIC selection, edge cases

---

## Rollback Plan

1. Delete `gmm_clusterer.py` and `test_gmm_clusterer.py`
2. Remove import, registry entry, and `__all__` entry from `clustering/__init__.py`
3. Remove `gmm` config block from `analyse_workflow_dag.yaml`
4. No data migration needed — GMM produces new output directories, no existing outputs are modified

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Bootstrap stability slow (BIC sweep x 20 rounds x n_init=10) | Medium | Low | Document in docstring. User can reduce `n_rounds`, `max_components`, or `n_init`. Future: `reuse_k_in_bootstrap` config option. |
| ConvergenceWarning on small/degenerate subsamples | Low | Low | Fail-fast `RuntimeError` on `converged_=False`. Pipeline try/except catches per-combination. User can switch to `covariance_type: "diag"`. |
| BIC selects wrong K for small N (~500) | Low | Medium | Cross-check with Bayesian GMM (future plan). Compare against other clusterer results. |
| Probability columns bloat output CSV | Low | Low | With K=5 and 2000 rows: 10,000 extra cells — negligible. K capped at `max_components`. |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Core implementation | ~100 lines | None |
| Phase 2: Registration + config | ~10 lines across 2 files | Phase 1 |
| Phase 3: Test suite | ~200 lines | Phase 2 |

---

## References

- Related Plans: `docs/development/plans/pending/rf-cluster-gallery-enhancements.md` (downstream RF visualization)
- scikit-learn docs: [GaussianMixture](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html), [BIC model selection](https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html)
- Code style reference: `code/src/analysis/touch_analytics/clustering/kmeans_clusterer.py`
- Extra columns pattern: `code/src/analysis/touch_analytics/clustering/binning_clusterer.py`
