# Plan: Align touch_analytics with five-stage clustering guidelines

**Created:** 2026-04-23 14:00
**Approved:** —
**Completed:** 2026-04-24 17:07
**Author:** Basil Duvernoy
**Status:** Completed
**Branch:** `feature/align-clustering-pipeline-with-guidelines`

---

## Overview

**What:** Refactor `code/src/analysis/touch_analytics/` so that its module layout and data flow mirror the five-stage pipeline defined in `docs/design/timeseries_clustering_pipeline_guidelines.md` (preparation → representation 2a/2b → reduction → clustering → evaluation).

**Why:** The pipeline is currently a two-bucket `feature_extraction/` + `clustering/` shape. Series-level work (2a) and scalar characterisation (2b) are fused inside the extractors; feature scaling (stage 3) is inlined three times across clusterers; there is no internal evaluation (stage 5). This matches several anti-patterns the guidelines call out and makes it hard to add Path-A (sequence-aware) clusterers later.

**How:** Introduce dedicated `preparation/`, `representation/` (with `series_level/` and `feature_characterization/` submodules), `reduction/`, and `evaluation/` packages. Keep `clustering/` but mark each clusterer with its Path (A or B) and remove the inlined `StandardScaler` calls. Expose new `reduction` and `evaluation` blocks in `configs/analyse_workflow_dag.yaml`. Preserve the public surface that `rf_cluster_pipeline.py` and `comparing_pipeline.py` depend on (CSV path layout, `cluster_metadata.json` keys, `pipeline_shared` symbols).

## Problem Statement

The five-stage guidelines were adopted project-wide but the existing analysis module pre-dates them. Concretely:

- **Stage 1 is diffuse.** Block-id synthesis, direction inference, and touch grouping sit inside `extraction_pipeline.py` (`_extract_session` L183–190 for block-id synthesis, `_extract_all_touches` L283 for grouping, L295–299 for direction). There is no dedicated data-preparation module, so any future resampling / imputation / denoising would have nowhere clean to go.
- **Stage 2a and 2b are fused.** `feature_extraction/kinematics.py` returns `pd.Series` (velocity, acceleration) — genuine 2a outputs — but those series are *immediately* consumed by `statistical_extractor.py` / `mos_extractor.py` / `pressure_extractor.py`, which collapse them to scalars in the same call. Nothing persists a series-level artefact, and no provenance metadata records the 2a→2b chain. This is the "one amorphous feature-extraction step" anti-pattern from the guidelines.
- **Stage 3 is absent.** `StandardScaler` is instantiated inline in `clustering/kmeans_clusterer.py:38–39`, `clustering/hierarchical_clusterer.py:74–75`, and `clustering/dbscan_clusterer.py:33–34`. Variance/correlation filtering and PCA/UMAP have no landing spot.
- **Stage 4 commits implicitly to Path B.** `clustering/base.py:8–33` defines `TouchClusterer.fit_predict(feature_df, config)` — a scalar feature matrix — without declaring that this is Path B. A future DTW k-means or k-Shape implementation has no documented interface.
- **Stage 5 is missing.** No silhouette, Davies–Bouldin, Calinski–Harabasz, or stability check anywhere. Clustering outputs only a label column and algorithm-specific metadata. The guidelines explicitly name "reporting a clustering result without at least one internal metric and a stability check" as an anti-pattern.
- **Runtime config injection is implicit.** `clustering_pipeline.py:345–359` mutates the profile dict with `_sensor_labels`, `_type_labels`, `_direction_labels`. Consumers (`HierarchicalClusterer`, `TypeStratifiedClusterer`) pick them up by string key. This is brittle and fails on static typing.

## Goals

### In Scope

1. Introduce an explicit five-stage module layout visible at import time under `code/src/analysis/touch_analytics/`.
2. Move `StandardScaler` out of individual clusterers into a shared Stage-3 step used by the clustering orchestrator.
3. Add a Stage-5 evaluation module that produces at least silhouette score plus one stability measure per clustering run and writes the result alongside `cluster_metadata.json`.
4. Make `TouchClusterer` subclasses declare their Path (A or B) so future sequence-aware clusterers have a defined interface.
5. Replace the runtime string-key injection of `_type_labels` / `_direction_labels` / `_sensor_labels` with an explicit context object passed into `fit_predict`.
6. Extend `configs/analyse_workflow_dag.yaml` with new `reduction` and `evaluation` sub-sections under `touch_clustering`, keeping existing keys backward-compatible where reasonable.
7. Preserve the on-disk output contract (`pooled_touch_summary_clustered.csv` + `cluster_metadata.json` at `<output_dir>/<combination>/<clusterer>/`) that `rf_cluster_pipeline.py` and `comparing_pipeline.py` depend on.
8. Preserve the `pipeline_shared.py` symbols (`SHARED_COLUMNS`, `filter_enabled_profiles`, `session_id_from_path`) imported by `rf_cluster_pipeline.py:28–32`.

### Out of Scope

- Implementing an actual Path-A clusterer (DTW k-means, k-Shape). The refactor only reserves the interface.
- Modifying any code outside `code/src/analysis/touch_analytics/` except the two lines in `configs/analyse_workflow_dag.yaml` (`touch_feature_extraction`, `touch_clustering`). `receptive_field_mapping/` is explicitly not modified.
- Upstream preprocessing, Kinect data access, or session-aggregated-CSV format changes.
- Adding new features, extractors, or feature combinations beyond what already exists.
- Changing the `feature_combinations` cross-product semantics.
- GUI updates (`feature_combination_dialog.py:19` imports `AGGREGATION_NAMES` and `EXTRACTOR_REGISTRY` — those names must survive unchanged).

## Success Criteria

- [ ] Every stage in `docs/design/timeseries_clustering_pipeline_guidelines.md` is implemented by a named submodule under `touch_analytics/` (one module per stage, except Stage 2 which has two).
- [ ] `StandardScaler` no longer appears inside any `clustering/*.py` file.
- [ ] Each `TouchClusterer` subclass has a class-level attribute (e.g. `PATH: Literal["A","B"]`) declaring its path.
- [ ] `configs/analyse_workflow_dag.yaml`'s `touch_clustering` block contains `reduction` and `evaluation` sub-sections, with sensible defaults that reproduce current behaviour.
- [ ] Running the smoke test (see Verification) produces `cluster_metadata.json` containing at least `silhouette_score`, `davies_bouldin_score`, and a `stability` sub-dict with a bootstrap/k-fold agreement score.
- [ ] `rf_cluster_pipeline.py` and `comparing_pipeline.py` run unchanged on the smoke-test session and read the new outputs without modification.
- [ ] Any intentional change in cluster labels vs. the pre-refactor baseline is documented in the PR description with a named reason (e.g. "scaler moved outside the stratified split → stratum statistics now computed on shared scaling").
- [ ] Relevance check against `docs/development/knowledge-base/` is performed and any applicable notes (confirmed: `note-cupy-import-order.md`) are respected in new entry-point modules.

---

## Technical Design

### Approach

Adopt the five-stage layout verbatim. Each stage owns one concern, one module, one well-typed input/output:

```
code/src/analysis/touch_analytics/
├── preparation/                   # Stage 1
│   ├── __init__.py
│   ├── block_id.py                # Moved from extraction_pipeline.py:183–190
│   ├── direction.py               # Moved from extraction_pipeline.py:295–299 + touch_category_extractor.py:46–52 (de-duplicated)
│   ├── grouping.py                # Moved from extraction_pipeline.py:283
│   └── loader.py                  # CSV read, dtype checks, NaN policy
├── representation/                # Stage 2
│   ├── __init__.py
│   ├── series_level/              # 2a: series → series
│   │   ├── __init__.py
│   │   └── kinematics.py          # MOVED from feature_extraction/kinematics.py
│   └── feature_characterization/  # 2b: series → scalar vector
│       ├── __init__.py
│       ├── base.py                # MOVED from feature_extraction/base.py
│       ├── statistical.py         # MOVED from feature_extraction/statistical_extractor.py
│       ├── mos.py                 # MOVED from feature_extraction/mos_extractor.py
│       ├── pressure.py            # MOVED from feature_extraction/pressure_extractor.py
│       ├── temporal.py            # MOVED from feature_extraction/temporal_extractor.py
│       └── touch_category.py      # MOVED from feature_extraction/touch_category_extractor.py
├── reduction/                     # Stage 3 (NEW)
│   ├── __init__.py
│   ├── scaling.py                 # StandardScaler, RobustScaler behind a factory
│   └── pipeline.py                # Orchestrates (optional) variance-filter → scale → (optional) PCA
├── clustering/                    # Stage 4 (kept, cleaned up)
│   ├── __init__.py                # Registry unchanged
│   ├── base.py                    # TouchClusterer + PATH attribute + ClusteringContext
│   ├── kmeans_clusterer.py        # Scaling removed
│   ├── dbscan_clusterer.py        # Scaling removed
│   ├── hierarchical_clusterer.py  # Scaling removed; sensor labels via context
│   ├── binning_clusterer.py       # No change (no scaling used)
│   └── type_stratified_clusterer.py  # Uses ClusteringContext instead of _type_labels keys
├── evaluation/                    # Stage 5 (NEW)
│   ├── __init__.py
│   ├── internal_metrics.py        # silhouette, Davies–Bouldin, Calinski–Harabasz
│   └── stability.py               # Bootstrap resampling + Adjusted Rand Index agreement
├── pipeline_shared.py             # UNCHANGED — imported by rf_cluster_pipeline.py
├── extraction_pipeline.py         # Slimmed: orchestrates preparation + representation; same run_feature_extraction signature
├── clustering_pipeline.py         # Orchestrates reduction → clustering → evaluation; same run_clustering signature
└── comparing_pipeline.py          # UNCHANGED (out of scope)
```

Key design decisions:

- **`feature_extraction/` is retired as a directory**, but its public names (`AGGREGATION_NAMES`, `EXTRACTOR_REGISTRY`, `FeatureExtractor`, every `*Extractor` class, `get_feature_extractor`, `get_extractor`) are re-exported from `representation/feature_characterization/__init__.py` *and* from a thin `feature_extraction/__init__.py` shim module so `feature_combination_dialog.py:19` continues to work untouched. The shim module emits no deprecation warning (it is considered equivalent, not a transitional stub).
- **`ClusteringContext`** is a dataclass passed as a third positional argument to `fit_predict(feature_df, config, context)`. It carries `sensor_labels`, `type_labels`, `direction_labels`. The `_*_labels` string keys are removed from `config`. `TouchClusterer.fit_predict` signature change is a public-API change — call-sites are enumerated under Migration Strategy.
- **`PATH` attribute.** `TouchClusterer` declares `PATH: ClassVar[Literal["A", "B"]]`. All current subclasses set `PATH = "B"`. A future DTW k-means sets `PATH = "A"` and receives a sequence iterator rather than a feature matrix (out of scope for this plan, but the interface is reserved).
- **Stage-3 scaling runs in the orchestrator**, before the per-clusterer `fit_predict`. `TypeStratifiedClusterer` inherits the shared scaling for the whole matrix; this is a deliberate change from the current behaviour where each stratum would have been scaled independently if scalers were per-stratum (they are not today — `TypeStratifiedClusterer` currently delegates to the base clusterer's scaler, which runs once per stratum). The new behaviour (single scaling before split) is called out as an intentional cluster-label change.
- **Evaluation runs in the orchestrator**, after `fit_predict`. It writes `silhouette_score`, `davies_bouldin_score`, `calinski_harabasz_score`, and `stability.bootstrap_ari` into `cluster_metadata.json` alongside existing keys. Downstream consumers (`rf_cluster_pipeline.py`, `comparing_pipeline.py`) read specific keys by name (`algorithm`, `session_coverage`, `dispersion_per_stratum`, `per_type`, `primary_feature`, `bin_edges`) — adding new keys is backward-compatible.
- **History preservation.** All file moves use `git mv` so blame survives. Package-level `__init__.py` files are re-written rather than moved.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Five-stage module layout (chosen) | One-to-one with guidelines; future Path A has a home; scaling de-duplicated | Breaking change to `TouchClusterer.fit_predict` signature (third arg) | **Chosen** |
| Keep `feature_extraction/` and `clustering/`, add only `reduction/` + `evaluation/` | Minimal movement; no breaking imports | Stage 1 stays diffuse; 2a/2b stay fused; doesn't address the core anti-pattern | Rejected |
| Full rewrite with new public API (`TouchPipeline` class) | Clean slate | Requires migrating `rf_cluster_pipeline.py` and `comparing_pipeline.py` (out of scope) | Rejected |
| Dissolve `clustering/` into `evaluation/` (one "stage 4+5" module) | Fewer modules | Conflates grouping with scoring, exactly the anti-pattern we're fixing | Rejected |

### Architecture Changes

**New public types:**

- `touch_analytics.clustering.base.ClusteringContext` — dataclass with optional `sensor_labels: np.ndarray | None`, `type_labels: np.ndarray | None`, `direction_labels: np.ndarray | None`.
- `touch_analytics.clustering.base.TouchClusterer.PATH: ClassVar[Literal["A","B"]]`.
- `touch_analytics.clustering.base.TouchClusterer.fit_predict(feature_df, config, context)` — third arg is new.
- `touch_analytics.reduction.pipeline.ReductionPipeline` — `fit_transform(feature_df, config) -> (np.ndarray, ReductionMetadata)`.
- `touch_analytics.evaluation.internal_metrics.compute_internal_metrics(X, labels) -> dict`.
- `touch_analytics.evaluation.stability.bootstrap_stability(clusterer, X, config, context, n_rounds=20) -> dict`.

**Preserved public surface** (must not change):

- `touch_analytics.run_feature_extraction`, `run_clustering`, `run_comparing` — signatures and return types unchanged.
- `touch_analytics.pipeline_shared.{SHARED_COLUMNS, filter_enabled_profiles, session_id_from_path}`.
- `touch_analytics.feature_extraction.{AGGREGATION_NAMES, EXTRACTOR_REGISTRY}` — re-exported from the new location via a shim.
- Clustering output path layout: `<output_dir>/<combination>/<clusterer>/{pooled_touch_summary_clustered.csv,cluster_metadata.json,heatmaps/*.png}`.
- `cluster_metadata.json` keys currently read by `rf_cluster_pipeline.py` (L257–298) and `comparing_pipeline.py` (L154–163): `algorithm`, `session_coverage`, `dispersion_per_stratum`, `per_type` (when type-stratified), `primary_feature`, `bin_edges`. New keys may be added.

### Config migration (`configs/analyse_workflow_dag.yaml`)

**Unchanged keys:** all of `touch_feature_extraction.options.features` (feature definitions survive verbatim), `touch_clustering.options.feature_combinations` (cross-product semantics unchanged), and every existing `clustering_profiles.<name>.method` / parameter.

**New keys under `touch_clustering.options`:**

```yaml
touch_clustering:
  options:
    feature_combinations: {...}       # unchanged
    clustering_profiles: {...}        # unchanged
    reduction:                        # NEW — applied before every clusterer
      scaler: standard                # standard | robust | none
      variance_filter: null           # null | {threshold: float}
      decomposition: null             # null | {method: pca, n_components: int}
    evaluation:                       # NEW — computed after every clusterer
      internal_metrics: [silhouette, davies_bouldin, calinski_harabasz]
      stability:
        method: bootstrap             # bootstrap | null
        n_rounds: 20
        subsample_fraction: 0.8
        score: adjusted_rand_index
```

Defaults are chosen so behaviour matches the current pipeline: `scaler: standard` reproduces today's inlined `StandardScaler`; `variance_filter` and `decomposition` default to null; evaluation runs by default.

**Session-YAML migration:** no migration needed. Session YAMLs under `configs/kinect_configs/` and `configs/forearm_configs/` do not reference clustering internals — they are session descriptors resolved by `session_config_resolver.py`. A one-shot migration script is *not* warranted.

**Backwards compatibility for the DAG YAML:** if `reduction` or `evaluation` blocks are absent, the orchestrator fills in the defaults above so pre-refactor DAG configs keep working.

---

## Implementation Plan

### Phase 1: Stage-3 reduction + Stage-5 evaluation (self-contained, no breaking changes)
**Goal:** Introduce `reduction/` and `evaluation/` modules and route the orchestrator through them, without yet moving existing files. At the end of this phase the behaviour is equivalent (scaling still happens, just once, in the orchestrator) and new metrics appear in `cluster_metadata.json`.
**Started:** 2026-04-23
**Completed:** 2026-04-23

- [x] Create `reduction/scaling.py` with `get_scaler(name: str)` factory (standard, robust, none).
- [x] Create `reduction/pipeline.py` with `ReductionPipeline.fit_transform`.
- [x] Create `evaluation/internal_metrics.py` with silhouette + Davies–Bouldin + Calinski–Harabasz.
- [x] Create `evaluation/stability.py` with bootstrap-resampling Adjusted Rand Index.
- [x] Route `clustering_pipeline._cluster_combination` through the new reduction pipeline (L342–343: where `feature_df = pooled[feature_cols].dropna()` happens).
- [x] Remove the inlined `StandardScaler` from `clustering/kmeans_clusterer.py:38–39`, `clustering/hierarchical_clusterer.py:74–75`, `clustering/dbscan_clusterer.py:33–34`. The clusterers now receive a scaled array from the orchestrator.
- [x] After `fit_predict`, call evaluation module and merge its dict into `metadata` before writing `cluster_metadata.json` (at `clustering_pipeline.py:386`).
- [x] Add the `reduction:` and `evaluation:` blocks to `configs/analyse_workflow_dag.yaml` with defaults that match current behaviour.

**Files Modified:**
- `code/src/analysis/touch_analytics/reduction/__init__.py` — new
- `code/src/analysis/touch_analytics/reduction/scaling.py` — new
- `code/src/analysis/touch_analytics/reduction/pipeline.py` — new
- `code/src/analysis/touch_analytics/evaluation/__init__.py` — new
- `code/src/analysis/touch_analytics/evaluation/internal_metrics.py` — new
- `code/src/analysis/touch_analytics/evaluation/stability.py` — new
- `code/src/analysis/touch_analytics/clustering_pipeline.py` — route through reduction + evaluation
- `code/src/analysis/touch_analytics/clustering/kmeans_clusterer.py` — drop StandardScaler
- `code/src/analysis/touch_analytics/clustering/hierarchical_clusterer.py` — drop StandardScaler
- `code/src/analysis/touch_analytics/clustering/dbscan_clusterer.py` — drop StandardScaler
- `configs/analyse_workflow_dag.yaml` — add `reduction`, `evaluation` blocks

**Dependencies:** None

### Phase 2: Stage-1 preparation + Stage-2 representation split
**Goal:** Create `preparation/` and `representation/` module layout; `git mv` existing files into their new homes; de-duplicate direction inference between `extraction_pipeline.py` and `touch_category_extractor.py`. `feature_extraction/` remains as a shim that re-exports from the new locations so outside callers (notably `feature_combination_dialog.py:19`) keep working unchanged.
**Started:** 2026-04-23
**Completed:** 2026-04-23

- [x] `git mv code/src/analysis/touch_analytics/feature_extraction/kinematics.py → representation/series_level/kinematics.py`.
- [x] `git mv` the remaining `feature_extraction/*.py` files under `representation/feature_characterization/`.
- [x] Create `preparation/block_id.py`, `preparation/direction.py`, `preparation/grouping.py`, `preparation/loader.py` — move the relevant blocks out of `extraction_pipeline.py` (`_extract_session` L183–190, `_extract_all_touches` L283 / L295–299, `pd.read_csv` at L175).
- [x] De-duplicate direction inference. `touch_category_extractor.py:46–52` and `extraction_pipeline.py:295–299` both compute the same thing — both call into `preparation/direction.py::infer_direction(group)`.
- [x] Add a thin `feature_extraction/__init__.py` that re-exports `AGGREGATION_NAMES`, `EXTRACTOR_REGISTRY`, `FeatureExtractor`, all `*Extractor` classes, `get_feature_extractor`, `get_extractor` from `representation.feature_characterization`. The shim keeps `feature_combination_dialog.py:19` working verbatim.
- [x] Update `touch_analytics/__init__.py` to re-export from the new modules; `run_feature_extraction` / `run_clustering` / `run_comparing` exports unchanged.

**Files Modified:**
- `code/src/analysis/touch_analytics/preparation/*.py` — new (from `extraction_pipeline.py` extractions)
- `code/src/analysis/touch_analytics/representation/series_level/kinematics.py` — moved
- `code/src/analysis/touch_analytics/representation/feature_characterization/*.py` — moved
- `code/src/analysis/touch_analytics/representation/feature_characterization/touch_category.py` — also updated to call `preparation.direction.infer_direction`
- `code/src/analysis/touch_analytics/feature_extraction/__init__.py` — rewritten as a shim
- `code/src/analysis/touch_analytics/extraction_pipeline.py` — slimmed; calls `preparation.*`
- `code/src/analysis/touch_analytics/__init__.py` — updated re-exports

**Dependencies:** Phase 1

### Phase 3: Path declaration + ClusteringContext
**Goal:** Remove the `_type_labels` / `_direction_labels` / `_sensor_labels` runtime string-key convention. Add `PATH` attribute to every clusterer.
**Started:** 2026-04-23
**Completed:** 2026-04-23

- [x] Add `ClusteringContext` dataclass to `clustering/base.py`.
- [x] Change `TouchClusterer.fit_predict(feature_df, config)` signature to `fit_predict(feature_df, config, context)`. Add `PATH: ClassVar[Literal["A","B"]] = "B"` to the ABC.
- [x] Update all five clusterers to accept the context argument. `HierarchicalClusterer._satisfies_coverage` now reads `context.sensor_labels` instead of `config['_sensor_labels']`. `TypeStratifiedClusterer._build_group_keys` now reads `context.type_labels` / `context.direction_labels`.
- [x] Update `clustering_pipeline._cluster_combination` (L345–359 in current file) to construct a `ClusteringContext` from the pooled DataFrame and pass it into `fit_predict`. Remove the `_*_labels` keys from the dict that is spread into `config`.
- [x] Update `clustering/test_type_stratified_clusterer.py` to use the new signature.

**Files Modified:**
- `code/src/analysis/touch_analytics/clustering/base.py` — add `ClusteringContext` + `PATH` + 3-arg signature
- `code/src/analysis/touch_analytics/clustering/kmeans_clusterer.py` — accept context, `PATH = "B"`
- `code/src/analysis/touch_analytics/clustering/dbscan_clusterer.py` — same
- `code/src/analysis/touch_analytics/clustering/hierarchical_clusterer.py` — read sensor labels from context
- `code/src/analysis/touch_analytics/clustering/binning_clusterer.py` — accept context (ignored), `PATH = "B"`
- `code/src/analysis/touch_analytics/clustering/type_stratified_clusterer.py` — read type/direction from context
- `code/src/analysis/touch_analytics/clustering_pipeline.py` — construct + pass `ClusteringContext`
- `code/src/analysis/touch_analytics/clustering/test_type_stratified_clusterer.py` — update tests

**Dependencies:** Phase 1, Phase 2

---

## Migration Strategy — Public API Surface

**Internal call-sites to update** (exhaustive list, all inside the refactor scope):

| File | Symbol / Line | Change |
|------|---------------|--------|
| `touch_analytics/__init__.py:3–19` | Re-exports from moved modules | Update import paths |
| `touch_analytics/extraction_pipeline.py:183–190, 283, 295–299` | Block-id synthesis, grouping, direction | Moved into `preparation/` |
| `touch_analytics/clustering_pipeline.py:342–359` | Feature-df construction + `_*_labels` injection | Route through `ReductionPipeline`; construct `ClusteringContext` |
| `touch_analytics/clustering_pipeline.py:362, 386` | `fit_predict` call + metadata write | 3-arg call; merge evaluation results into metadata |
| `touch_analytics/clustering/kmeans_clusterer.py:38–39` | Inlined StandardScaler | Removed |
| `touch_analytics/clustering/hierarchical_clusterer.py:74–75, 58, 99–104` | Inlined StandardScaler + `_sensor_labels` | Removed; sensor labels from context |
| `touch_analytics/clustering/dbscan_clusterer.py:33–34` | Inlined StandardScaler | Removed |
| `touch_analytics/clustering/type_stratified_clusterer.py:51–60, 62, 78–79` | `_type_labels` / `_direction_labels` | Read from context; signature change |
| `touch_analytics/clustering/test_type_stratified_clusterer.py` (all tests) | Old `fit_predict(df, config)` calls | Update to 3-arg form |
| `touch_analytics/representation/feature_characterization/touch_category.py:46–52` (after move) | Direction inference duplicate | Delegate to `preparation.direction.infer_direction` |

**External consumers** (out of scope to modify — the plan preserves the API they depend on):

| File | Dependency | Preservation strategy |
|------|-----------|----------------------|
| `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py:28–32` | `from analysis.touch_analytics.pipeline_shared import SHARED_COLUMNS, filter_enabled_profiles, session_id_from_path` | `pipeline_shared.py` is explicitly not moved and not modified |
| `rf_cluster_pipeline.py:375–390` | Reads `pooled_touch_summary_clustered.csv` and `cluster_metadata.json` from hardcoded path layout | Output path layout preserved verbatim |
| `rf_cluster_pipeline.py:257–298` | Parses `algorithm`, `per_type`, `primary_feature`, `bin_edges` from metadata | All keys preserved; new keys (silhouette, etc.) are additive |
| `code/src/utils/gui/dag_launcher/feature_combination_dialog.py:19` | `from analysis.touch_analytics.feature_extraction import AGGREGATION_NAMES, EXTRACTOR_REGISTRY` | `feature_extraction/__init__.py` becomes a shim that re-exports from the new location |
| `touch_analytics/comparing_pipeline.py:122–163` | Reads `pooled_touch_summary_clustered.csv` + parses `dispersion_per_stratum` | Output contract preserved; no changes to `comparing_pipeline.py` |

**Intentional cluster-label changes called out in PR:**

- Moving `StandardScaler` out of individual clusterers means `TypeStratifiedClusterer` now scales the whole pooled matrix *before* stratification (current code re-scales inside each per-type delegate). Within-stratum numeric behaviour differs: feature means/stds are now computed across all strata. Cluster labels *may* shift for borderline touches. This is the documented trade-off — scaling once on the full matrix is the Stage-3 contract, and per-stratum scaling was an incidental consequence of inlined scalers. The PR description names this explicitly.

---

## Knowledge-Base Relevance Check

Per the planning procedure, ran a relevance scan of `docs/development/knowledge-base/`:

- **`note-cupy-import-order.md` — applies.** New entry-point modules (`reduction/pipeline.py`, `evaluation/internal_metrics.py`) import sklearn utilities. If any of them ever grow to import preprocessing utilities that transitively touch CuPy, the guarded `import cupy` must come first. Entry-point scripts that drive the pipeline already honour this; the new modules are leaves and do not need to guard themselves unless they gain preprocessing imports.
- **`note-kinect-depth-access-single-path.md` — does not apply.** The refactor is downstream of data access; no raw Kinect depth is read.
- **`note-somatosensory-units-and-calculations.md` — does not apply directly.** No new unit math is introduced; moved feature extractors preserve current unit conventions (mm, mm/s, mm/s², mJ, mN·s).
- **`note-git-merge-autonomous-fast-forward.md` — applies operationally.** The refactor branch must be merged with `--no-ff`.
- All other notes in the knowledge base reviewed and found irrelevant (GUI, 3D rendering, preprocessing-specific bugs).

---

## Testing Plan

### Unit Tests

- [ ] `reduction/test_scaling.py` — standard / robust / none factories return correct sklearn-compatible objects; `ReductionPipeline.fit_transform` is idempotent under identical inputs.
- [ ] `reduction/test_pipeline.py` — variance-filter threshold drops low-variance columns; PCA `n_components` honoured.
- [ ] `evaluation/test_internal_metrics.py` — silhouette / Davies–Bouldin / Calinski–Harabasz on a synthetic 3-blob dataset produce expected ranges.
- [ ] `evaluation/test_stability.py` — bootstrap ARI on a synthetic well-separated dataset is near 1.0; on random data is near 0.
- [ ] `clustering/test_type_stratified_clusterer.py` — updated for the 3-arg `fit_predict`; all existing 11 cases still pass.
- [ ] `preparation/test_direction.py` — `infer_direction` matches current behaviour for proximal / distal / static cases (migrated from `extraction_pipeline.py:295–299`).
- [ ] `preparation/test_block_id.py` — block-id regex extraction matches current behaviour (migrated from `extraction_pipeline.py:183–190`).

### Integration Tests

- [ ] Running `run_feature_extraction` on a pre-split session produces byte-identical per-feature CSVs to pre-refactor (numerical feature extraction is unchanged).
- [ ] Running `run_clustering` on the smoke-test session produces `cluster_metadata.json` with all pre-refactor keys *plus* the new evaluation keys.
- [ ] `rf_cluster_pipeline.py` runs end-to-end on the smoke-test session without modification.

### Manual Verification — Smoke Test

**Session:** `valid_configs_ST13-01/kinect_config_2022-06-14_ST13-01_semicontrolled_block-order01.yaml` paired with forearm config `session_2022-06-14_ST13-01.yaml` (smallest available session; already identified during exploration).

**Steps:**

1. Run the DAG against the ST13-01 session with `touch_feature_extraction` + `touch_clustering` enabled and the `only_mean` + `binning` profile.
2. Inspect outputs at `<output_dir>/only_mean/binning/`:
   - `pooled_touch_summary_clustered.csv` — must contain `cluster_label` column, same schema as before plus any `bin_*` extras.
   - `cluster_metadata.json` — must contain `algorithm`, `session_coverage`, `dispersion_per_stratum`, `primary_feature`, `bin_edges` (pre-existing) AND `silhouette_score`, `davies_bouldin_score`, `calinski_harabasz_score`, `stability.bootstrap_ari` (new).
   - `heatmaps/*.png` — unchanged count and shape.
3. Re-enable `type_stratified` profile with `base_method: binning` and repeat. `cluster_metadata.json` must still contain `algorithm: type_stratified` and a `per_type` dict.
4. Re-enable `kmeans` profile. Confirm metadata has `k`, and internal-metrics keys are present.
5. Run `rf_cluster_pipeline.py` on the same session. It must produce receptive-field outputs without error — this confirms the output contract survived.
6. Compare cluster labels against a saved pre-refactor baseline. Any differences must be accounted for by the documented scaling change; if differences appear for profiles that did not use scaling (e.g. `binning`), that is a regression.

**Pass criteria:**

- Pipeline runs green end-to-end on ST13-01.
- All four clustering profiles (`binning`, `kmeans`, `hierarchical`, `type_stratified`) emit `silhouette_score` in `cluster_metadata.json`.
- `rf_cluster_pipeline.py` produces the same RF-mapping outputs it produces today.
- Any cluster-label differences are documented in the PR with a named reason.

---

## Documentation Plan

- [ ] Update `docs/design/timeseries_clustering_pipeline_guidelines.md` cross-reference: add an appendix pointing to the `touch_analytics/` module layout that implements the guidelines.
- [ ] Update `CLAUDE.md` if the public API surface of `touch_analytics` changes in a way GUI / downstream code would need to know about (likely not, given the shim strategy).
- [ ] Docstring in `clustering/base.py` — explain `ClusteringContext` and the `PATH` attribute so future Path-A clusterer authors find the extension point.
- [ ] Memory: update `project_analysis_pipeline_state.md` after completion to reflect the new layout.

No new user-guide needed: internal refactor only.

---

## Rollback Plan

Plan phases are additive in Phase 1 and structural in Phases 2–3. Rollback options:

1. **During Phase 1** — revert the phase's commits. The new `reduction/` and `evaluation/` directories can be deleted; the inlined `StandardScaler` restored. Safe and isolated.
2. **During Phase 2** — `git mv` operations are trivially reversed by `git revert` on the phase commits (history preserved). The shim module in `feature_extraction/__init__.py` can be deleted, and the original files restored.
3. **During Phase 3** — revert the phase's commits to restore the 2-arg `fit_predict` signature and the `_*_labels` string-key convention. The tests that were updated will revert alongside.

No data migrations involved; outputs on disk use the same path layout throughout, so rolling back does not invalidate any previously-generated clustering results.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Moving `StandardScaler` outside the clusterer changes `TypeStratifiedClusterer` labels for borderline touches | High | Med | Call it out explicitly in PR; verify the pattern of label changes is consistent with scaling-before-stratification |
| Shim module in `feature_extraction/__init__.py` missed a symbol imported elsewhere | Med | Med | `grep -r "from analysis.touch_analytics.feature_extraction"` before declaring Phase 2 complete; exhaustive list is: `AGGREGATION_NAMES`, `EXTRACTOR_REGISTRY`, `FeatureExtractor`, `StatisticalExtractor`, `TemporalExtractor`, `MechanicsOfSolidsExtractor`, `PressureExtractor`, `TouchCategoryExtractor`, `get_feature_extractor`, `get_extractor` |
| `cluster_metadata.json` adds a key that breaks `json.load` in a pinned-schema consumer | Low | High | Survey consumers before Phase 1 (already done: `rf_cluster_pipeline.py` and `comparing_pipeline.py` both use `.get(...)` patterns, not strict schemas); add new keys only, do not rename |
| Stability bootstrap is slow enough to dominate pipeline runtime on large sessions | Med | Low | Default `n_rounds: 20` + `subsample_fraction: 0.8` (configurable); allow `evaluation.stability.method: null` to disable |
| ClusteringContext signature change missed at a third-party call site | Low | Med | Only two call paths construct a clusterer: `clustering_pipeline._cluster_combination` and `TypeStratifiedClusterer.fit_predict` (delegates). Grep for `.fit_predict(` under `analysis/` confirms this. |
| Tests in `test_type_stratified_clusterer.py` lock in the old signature | High | Low | Update tests as part of Phase 3; this is expected, not surprise breakage |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 — Reduction + Evaluation | ~1 day | None |
| Phase 2 — Preparation + Representation split | ~1 day | Phase 1 |
| Phase 3 — Path declaration + ClusteringContext | ~0.5 day | Phase 2 |

---

## References

- Design document: `docs/design/timeseries_clustering_pipeline_guidelines.md`
- Planning procedure: `docs/development/planning-procedure.md`
- Related plan (recently completed): `docs/development/plans/completed/` — RF cluster quantitative metrics
- Related active plan: `docs/development/plans/active/type-stratified-clustering.md`
- Knowledge base: `docs/development/knowledge-base/note-cupy-import-order.md`
