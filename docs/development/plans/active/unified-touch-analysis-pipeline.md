# Plan: Unified Touch Analysis Pipeline

**Created:** 2026-03-16 18:00
**Approved:** —
**Completed:** —
**Author:** Basil Duvernoy
**Status:** Draft
**Branch:** `feature/unified-touch-analysis-pipeline`

---

## Overview

**What:** Merge `summarize_touches_per_session` and `analyse_number_single_touches` into a single unified pipeline that supports pluggable feature extraction methods and pluggable clustering methods.

**Why:** The current pipeline is rigid — it only extracts kinematics via `max()` and groups touches via fixed binning (`qcut(q=3)` / `pd.cut(bins=20)`). Investigating how different computational methods characterize and cluster single touches requires a framework that can run multiple approaches and persist their results independently.

**How:** Introduce a `FeatureExtractor` ABC with a registry of concrete extractors (max, statistical, temporal, mechanics-of-solids), a `TouchClusterer` ABC with adaptive clustering algorithms (K-means, DBSCAN), and a unified Prefect flow that drives both from YAML-configured profiles. Each extraction+clustering combination writes to its own subdirectory under `unified_touches/`.

## Problem Statement

- **Single extraction method:** `_process_touch_analysis()` reduces each touch to 4 kinematic features using only `max()` — discarding signal shape, duration, physical interpretation, and statistical spread.
- **Fixed binning instead of clustering:** The downstream matrix step uses `qcut(q=3)` (3 bins per variable) and `pd.cut(bins=20)` (heatmap), producing systematic grids rather than data-driven clusters. These bins don't adapt to the actual distribution of touches.
- **No cluster persistence:** The binning is applied at visualization time and not stored back on individual touches, making it impossible to use grouping information in downstream analyses.
- **Two separate workflow steps** for what is conceptually one operation (characterize touches, then group them), making configuration and data flow unnecessarily complex.

## Goals

### In Scope

1. Pluggable feature extraction framework with 4 concrete extractors: max (current behavior), statistical summaries, temporal features, mechanics of solids
2. Pluggable clustering framework with 2 concrete clusterers: adaptive K-means, DBSCAN — both respecting a configurable `min_touches_per_cluster` constraint
3. Global clustering (all sessions pooled) with session identity preserved
4. YAML-driven profile selection — no code changes needed to enable/disable extraction or clustering methods
5. Per-method subdirectory output under `unified_touches/` so methods don't overwrite each other
6. Backward compatibility: `max` profile writes to existing root `unified_touches/` path so downstream tasks (`analyse_ap_efficacy`, `map_receptive_fields`) work unchanged
7. Single merged Prefect flow replacing both `summarize_touches_per_session` and `analyse_number_single_touches`

### Out of Scope

- Timeseries extractor and DTW clustering (deferred to optional follow-up phase)
- Hertzian or viscoelastic contact models (start with simple uniaxial elastic)
- Per-session clustering (global only in this plan)
- Automatic method comparison / evaluation framework
- Changes to `analyse_ap_efficacy` or `map_receptive_fields` internals
- GPU acceleration for clustering

## Success Criteria

- [ ] `max` extraction profile produces output identical to current `generate_unified_summary()` (byte-level CSV diff)
- [ ] `analyse_ap_efficacy` and `map_receptive_fields` run without modification on new pipeline output
- [ ] At least 4 extraction profiles (max, statistical, temporal, mos) produce valid per-session CSVs
- [ ] At least 2 clustering methods (K-means, DBSCAN) produce valid `pooled_touch_summary_clustered.csv` with `cluster_label` column
- [ ] All cluster sizes >= `min_touches_per_cluster` (or documented exception for DBSCAN noise)
- [ ] Each extraction+clustering combination writes to its own subdirectory without overwriting others
- [ ] Pipeline is idempotent: second run with unchanged inputs skips processing
- [ ] All profiles configurable via `analyse_workflow_dag.yaml` without code changes

---

## Technical Design

### Approach

A registry-based plugin architecture where extractors and clusterers are discovered by method name from YAML config. This keeps the orchestration code generic while allowing new methods to be added by creating a single new file and registering it.

The pipeline runs in two phases:
1. **Extraction** (per-session, parallelizable): for each session and each enabled extraction profile, run the extractor on every touch group and save a per-session CSV
2. **Clustering** (global, per-profile): for each extraction profile, pool all session CSVs, run each enabled clustering method, save pooled results + metadata

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Plugin registry (ABC + dict) | Extensible, no orchestrator changes for new methods, clear interface | More files/boilerplate upfront | **Chosen** |
| Config-driven if/elif chain | Simpler initial implementation | Grows unwieldy, violates OCP, hard to test in isolation | Rejected |
| Separate pipeline per method | Maximum isolation | Massive code duplication, config explosion | Rejected |
| Column-stamped single CSV | One file, all methods together | Wide CSV becomes unmanageable, coupling between methods | Rejected — user chose subdirectory approach |

### Architecture Changes

```
code/src/analysis/touch_analytics/
├── feature_extraction/             # NEW — pluggable extractors
│   ├── __init__.py                 # EXTRACTOR_REGISTRY + get_extractor()
│   ├── base.py                     # ABC: FeatureExtractor.extract(group, config) -> dict
│   ├── kinematics.py               # Shared: compute_velocity(), compute_acceleration()
│   ├── max_extractor.py            # Current behavior (max of depth/area/vel/accel)
│   ├── statistical_extractor.py    # mean, median, std, min, max, range, skewness
│   ├── temporal_extractor.py       # duration, time_to_peak, AUC, onset/offset slope
│   └── mos_extractor.py            # Mechanics of solids: strain, stress, energy, impulse
├── clustering/                     # NEW — pluggable clusterers
│   ├── __init__.py                 # CLUSTERER_REGISTRY + get_clusterer()
│   ├── base.py                     # ABC: TouchClusterer.fit_predict(df, config) -> (labels, metadata)
│   ├── kmeans_clusterer.py         # Adaptive K-means with min_touches_per_cluster
│   └── dbscan_clusterer.py         # DBSCAN with auto-eps option
├── unified_pipeline.py             # NEW — orchestrates extraction + clustering
├── touch_analysis.py               # EXISTING — refactored to delegate to extractors
├── matrix_generation.py            # EXISTING — optionally called as post-step
├── reporting.py                    # EXISTING — unchanged
├── touch_config.py                 # EXISTING — kept for backward compat
├── session_summary.py              # EXISTING — unchanged
└── __init__.py                     # MODIFIED — export new public API
```

#### Feature Extraction Interface

```python
class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, group: pd.DataFrame, config: dict) -> dict:
        """Given frames for one (trial_id, single_touch_id), return feature dict."""
```

Common columns (trial_id, single_touch_id, block_order_id, type_metadata, direction, spike_elicited, mean_contact_x/y/z) are computed once in shared orchestration code, not per-extractor. Extractors only produce method-specific kinematic features.

#### Shared Kinematics Utility

Extracted from `touch_analysis.py:76-83`:
- `compute_velocity_magnitudes(group) -> pd.Series` — 3D sticker position diff magnitudes
- `compute_acceleration_magnitudes(velocity_magnitudes) -> pd.Series` — velocity diff magnitudes

#### Mechanics of Solids Extractor

Derives physically meaningful quantities from kinematics + assumed tissue properties (configurable in YAML):

| Metric | Formula | Units | Aggregation |
|--------|---------|-------|-------------|
| strain | `depth / skin_thickness` | dimensionless | max, mean |
| stress | `E * strain` | kPa | max, mean |
| strain_rate | `velocity / (skin_thickness * fps)` | 1/s | max |
| elastic_energy | `0.5 * E * strain^2 * area * skin_thickness` | mJ | max, total |
| impulse | `sum(stress * area * dt)` over frames | mN·s | total |

Default tissue parameters: `youngs_modulus_kpa=100.0`, `poissons_ratio=0.45`, `skin_thickness_mm=1.5`.

#### Clustering Interface

```python
class TouchClusterer(ABC):
    @abstractmethod
    def fit_predict(self, feature_df: pd.DataFrame, config: dict) -> Tuple[np.ndarray, dict]:
        """Return (cluster_labels, metadata_dict)."""
```

**Adaptive K-means:** starts at `k_max = total_touches // min_touches_per_cluster`, decrements k until all clusters meet minimum size (floor k=2). Features standardized with `StandardScaler`.

**DBSCAN:** `min_samples = min_touches_per_cluster`, `eps = auto` via k-distance knee detection. Noise points (label=-1) are valid.

#### Output Structure

```
4_analysed/unified_touches/
  # Backward-compat root (max profile copies here)
  ST13-01_semicontrolled_touch_summary.csv
  ...
  # Per-method subdirectories
  <profile_name>/
    ST13-01_semicontrolled_touch_summary.csv
    ...
    clustering/
      <clusterer_name>/
        pooled_touch_summary_clustered.csv   # all sessions, with cluster_label
        cluster_metadata.json                # k, sizes, centroids, session coverage
```

#### DAG Configuration

```yaml
unified_touch_analysis:
  enabled: true
  options:
    force_processing: false
    extraction_profiles:
      max:
        method: max
      statistical:
        method: statistical
        aggregations: [mean, median, std, min, max, range, skewness]
      temporal:
        method: temporal
      mos:
        method: mechanics_of_solids
        youngs_modulus_kpa: 100.0
        poissons_ratio: 0.45
        skin_thickness_mm: 1.5
    clustering_profiles:
      kmeans:
        method: kmeans
        min_touches_per_cluster: 30
      dbscan:
        method: dbscan
        eps: auto
        min_touches_per_cluster: 30
  depends_on: []
```

---

## Implementation Plan

### Phase 1: Extractor Framework + Max Extractor
**Goal:** Create the pluggable feature extraction interface and prove the `max` extractor reproduces current behavior identically.
**Started:** —
**Completed:** —

- [ ] Task 1.1 — Create `FeatureExtractor` ABC in `feature_extraction/base.py`
- [ ] Task 1.2 — Extract shared kinematics (velocity, acceleration) from `touch_analysis.py:76-83` into `feature_extraction/kinematics.py`
- [ ] Task 1.3 — Implement `MaxExtractor` in `feature_extraction/max_extractor.py` (replicates `touch_analysis.py:69-119`)
- [ ] Task 1.4 — Create `EXTRACTOR_REGISTRY` and `get_extractor()` factory in `feature_extraction/__init__.py`
- [ ] Task 1.5 — Validate: run `MaxExtractor` on a session and diff against current `generate_unified_summary()` output

**Files Created:**
- `code/src/analysis/touch_analytics/feature_extraction/__init__.py` — Registry and factory
- `code/src/analysis/touch_analytics/feature_extraction/base.py` — ABC definition
- `code/src/analysis/touch_analytics/feature_extraction/kinematics.py` — Shared velocity/acceleration computation
- `code/src/analysis/touch_analytics/feature_extraction/max_extractor.py` — Max-based extraction (current behavior)

**Dependencies:** None

### Phase 2: Additional Scalar Extractors
**Goal:** Implement statistical, temporal, and mechanics-of-solids extractors.
**Started:** —
**Completed:** —

- [ ] Task 2.1 — Implement `StatisticalExtractor`: mean, median, std, min, max, range, skewness per kinematic variable (depth, area, velocity, acceleration). Uses `scipy.stats.skew`.
- [ ] Task 2.2 — Implement `TemporalExtractor`: touch duration (frame count), time_to_peak_depth, time_to_peak_area, AUC via `np.trapz` for depth and area curves, onset/offset slope via linear fit on first/last N frames.
- [ ] Task 2.3 — Implement `MechanicsOfSolidsExtractor`: per-frame strain, stress, strain_rate, elastic_energy, impulse using configurable tissue properties. Aggregates per touch (max, mean, total as appropriate).
- [ ] Task 2.4 — Register all 3 extractors in `EXTRACTOR_REGISTRY`

**Files Created:**
- `code/src/analysis/touch_analytics/feature_extraction/statistical_extractor.py`
- `code/src/analysis/touch_analytics/feature_extraction/temporal_extractor.py`
- `code/src/analysis/touch_analytics/feature_extraction/mos_extractor.py`

**Dependencies:** Phase 1

### Phase 3: Clustering Framework
**Goal:** Create pluggable clustering with adaptive cluster count respecting `min_touches_per_cluster`.
**Started:** —
**Completed:** —

- [ ] Task 3.1 — Create `TouchClusterer` ABC in `clustering/base.py`
- [ ] Task 3.2 — Implement `KMeansClusterer` with adaptive k reduction loop and `StandardScaler` normalization
- [ ] Task 3.3 — Implement `DBSCANClusterer` with auto-eps (k-distance knee detection via `NearestNeighbors`)
- [ ] Task 3.4 — Create `CLUSTERER_REGISTRY` and `get_clusterer()` factory in `clustering/__init__.py`
- [ ] Task 3.5 — Implement `cluster_metadata.json` serialization (algorithm, params, k, per-cluster sizes, session coverage)

**Files Created:**
- `code/src/analysis/touch_analytics/clustering/__init__.py` — Registry and factory
- `code/src/analysis/touch_analytics/clustering/base.py` — ABC definition
- `code/src/analysis/touch_analytics/clustering/kmeans_clusterer.py` — Adaptive K-means
- `code/src/analysis/touch_analytics/clustering/dbscan_clusterer.py` — DBSCAN with auto-eps

**Dependencies:** Phase 1

### Phase 4: Merged Pipeline Orchestration
**Goal:** Wire extraction + clustering into a single Prefect flow driven by YAML config, replacing both old tasks.
**Started:** —
**Completed:** —

- [ ] Task 4.1 — Create `unified_pipeline.py` with `run_unified_touch_analysis(input_items, options, force)`:
  - Step A: Per session × per extraction profile → run extractor → save per-session CSV to `unified_touches/<profile>/`
  - Step B: If `max` profile active → also write to root `unified_touches/` (backward compat)
  - Step C: Per profile → pool all session CSVs → run each clustering profile → save pooled CSV + metadata to `unified_touches/<profile>/clustering/<clusterer>/`
  - Step D (optional): generate count matrix via existing `generate_touch_summary_matrix()`
- [ ] Task 4.2 — Add `unified_touch_analysis_flow` Prefect flow in `analysis_workflow.py`
- [ ] Task 4.3 — Register `"unified_touch_analysis"` in `available_tasks` in `run_batch_analysis()`
- [ ] Task 4.4 — Add deprecation warnings to `summarize_touches_per_session_flow` and `analyse_number_single_touches_flow`
- [ ] Task 4.5 — Update `configs/analyse_workflow_dag.yaml`: add `unified_touch_analysis` task, update downstream `depends_on`
- [ ] Task 4.6 — Update `touch_analytics/__init__.py` exports

**Files Created:**
- `code/src/analysis/touch_analytics/unified_pipeline.py` — Orchestration logic

**Files Modified:**
- `code/scripts/analysis_workflow.py` — New flow, updated available_tasks, deprecation warnings
- `code/src/analysis/touch_analytics/__init__.py` — Export new public API
- `configs/analyse_workflow_dag.yaml` — New task config, updated dependencies

**Dependencies:** Phases 1, 2, 3

### Phase 5: Integration Testing
**Goal:** Verify backward compatibility and end-to-end correctness.
**Started:** —
**Completed:** —

- [ ] Task 5.1 — Run `max` profile only, diff output against current `summarize_touches_per_session` output — must be identical
- [ ] Task 5.2 — Run `analyse_ap_efficacy` and `map_receptive_fields` on new output — must complete without error
- [ ] Task 5.3 — Run all 4 extraction profiles + 2 clustering methods — verify directory structure matches spec
- [ ] Task 5.4 — Test idempotency: run twice with unchanged inputs, verify second run skips
- [ ] Task 5.5 — Test `force_processing: true` forces re-run
- [ ] Task 5.6 — Test edge case: session with zero valid touches (all touch_id == 0)
- [ ] Task 5.7 — Test edge case: too few touches for `min_touches_per_cluster` — verify graceful degradation (k=2 or single cluster)

**Dependencies:** Phase 4

### Phase 6 (Optional): Timeseries Extractor + DTW Clustering
**Goal:** Add raw time-series preservation and DTW-based clustering. Ship phases 1-5 first.
**Started:** —
**Completed:** —

- [ ] Task 6.1 — Implement `TimeseriesExtractor`: saves raw per-touch kinematic signals to individual CSVs in `unified_touches/timeseries/signals/`, returns `signal_path`
- [ ] Task 6.2 — Implement `DTWClusterer`: computes pairwise DTW distance matrix (cached as `.npy`), applies agglomerative clustering on precomputed distances
- [ ] Task 6.3 — Register both in their respective registries

**Files Created:**
- `code/src/analysis/touch_analytics/feature_extraction/timeseries_extractor.py`
- `code/src/analysis/touch_analytics/clustering/dtw_clusterer.py`

**Notes:** DTW is O(n^2) — cache distance matrix for idempotency. May require `dtaidistance` or `tslearn`.

**Dependencies:** Phases 1, 3, 4

---

## Testing Plan

### Unit Tests

- [ ] `MaxExtractor.extract()` produces identical values to current `_process_touch_analysis()` for a known touch group
- [ ] `StatisticalExtractor` produces correct mean/median/std/skewness for synthetic data
- [ ] `MechanicsOfSolidsExtractor` produces correct strain/stress/energy for a known depth+area+velocity profile
- [ ] `KMeansClusterer` respects `min_touches_per_cluster` — no cluster smaller than threshold
- [ ] `KMeansClusterer` adaptive loop terminates at k=2 floor
- [ ] `DBSCANClusterer` auto-eps produces valid clusters (non-empty, noise labeled -1)
- [ ] `get_extractor("max")` returns `MaxExtractor` instance
- [ ] `get_clusterer("kmeans")` returns `KMeansClusterer` instance
- [ ] Unknown method name raises `KeyError`

### Integration Tests

- [ ] Full pipeline with `max` profile + `kmeans` clustering produces expected directory structure
- [ ] Downstream `map_receptive_fields` consumes root `unified_touches/` CSVs without error
- [ ] Multiple extraction profiles run in sequence without interference

### Manual Verification

- [ ] Run pipeline on ST13-01 session with all profiles enabled — inspect output directories and CSV contents
- [ ] Open `cluster_metadata.json` — verify cluster sizes, centroids, session coverage make sense
- [ ] Diff `max` profile output against legacy `generate_unified_summary()` output — must be identical

### Edge Cases

- [ ] Session with 0 valid touches (all `single_touch_id == 0`) — produces empty CSV, no crash
- [ ] Session with 1 touch — extraction succeeds, clustering skips gracefully (too few for any cluster)
- [ ] `min_touches_per_cluster` larger than total touches — pipeline reports warning, produces k=1 or skips clustering
- [ ] Missing `Nerve_spike` column — `spike_elicited` defaults to 0 (existing behavior preserved)

---

## Documentation Plan

- [ ] Update `CLAUDE.md` memory section with new `unified_pipeline.py` location and extraction/clustering architecture
- [ ] Add inline docstrings to all new ABC classes and concrete implementations
- [ ] Document YAML profile schema in comments within `analyse_workflow_dag.yaml`
- [ ] Add knowledge base note if MoS tissue property assumptions need documenting: `docs/development/knowledge-base/note-mos-tissue-assumptions.md`

---

## Rollback Plan

1. **Before deployment:**
   - Old flow functions (`summarize_touches_per_session_flow`, `analyse_number_single_touches_flow`) are preserved with deprecation warnings — revert by re-enabling them in DAG YAML
   - No existing files are deleted, only new files added + minor modifications to `analysis_workflow.py` and `__init__.py`

2. **Data considerations:**
   - No database migrations
   - New output subdirectories (`unified_touches/max/`, etc.) can be deleted without affecting legacy root-level CSVs
   - Root `unified_touches/` CSVs are written identically to before (backward-compat dual-write)

3. **Rollback procedure:**
   - Revert the merge commit on `dev`
   - Re-enable `summarize_touches_per_session` and `analyse_number_single_touches` in `analyse_workflow_dag.yaml`
   - Delete any new subdirectories under `unified_touches/` if desired

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| DTW distance matrix O(n^2) slow for large datasets | Medium | High | Deferred to optional Phase 6; cache as `.npy`; progress logging |
| New dependencies (`scipy.stats`) missing in environment | Low | Medium | Graceful import with warning; `scipy` likely already available |
| MoS assumed tissue properties are inaccurate | Medium | Low | All values configurable in YAML; document assumptions; start with literature defaults |
| Backward-compat dual-write creates maintenance burden | Low | Low | Document clearly; plan to deprecate root-level writes in future |
| Adaptive K-means loop slow for very large k_max | Low | Low | k_max bounded by total_touches / min_touches; typically < 100 iterations |
| DBSCAN auto-eps selects poor eps value | Medium | Medium | Store eps in metadata for inspection; allow manual override in YAML |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Extractor framework + max | ~4 files, ~200 LOC | None |
| Phase 2: Additional extractors | ~3 files, ~300 LOC | Phase 1 |
| Phase 3: Clustering framework | ~4 files, ~250 LOC | Phase 1 |
| Phase 4: Pipeline orchestration | ~1 new + 3 modified files, ~200 LOC | Phases 1-3 |
| Phase 5: Integration testing | Manual verification | Phase 4 |
| Phase 6 (Optional): Timeseries + DTW | ~2 files, ~200 LOC | Phases 1, 3, 4 |

---

## References

- Current extraction logic: `code/src/analysis/touch_analytics/touch_analysis.py:37-141`
- Current binning config: `code/src/analysis/touch_analytics/touch_config.py`
- Current matrix generation: `code/src/analysis/touch_analytics/matrix_generation.py`
- Current visualization: `code/src/analysis/touch_analytics/reporting.py` (num_bins=20)
- Flow orchestration: `code/scripts/analysis_workflow.py`
- DAG config: `configs/analyse_workflow_dag.yaml`
- Idempotency utility: `code/src/utils/should_process_task.py`
- Units reference: `docs/development/knowledge-base/note-somatosensory-units-and-calculations.md`
