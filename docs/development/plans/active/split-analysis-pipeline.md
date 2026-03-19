# Plan: Split Analysis Pipeline into Independent Extraction, Clustering, and Comparing Tasks

**Created:** 2026-03-18
**Approved:** —
**Completed:** —
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `feature/split-analysis-pipeline`

---

## Overview

**What:** Split `unified_touch_analysis` into 3 independent DAG tasks (`touch_feature_extraction`, `touch_clustering`, `touch_comparing`), add a hierarchical clusterer with adaptive coverage-constrained dendrogram cut, and implement a comparing stage with bias/precision/distribution tests on `spike_elicited` across sessions (neurons).

**Why:** The monolithic pipeline cannot run stages independently, lacks the comparing stage entirely, and uses only flat-partition clusterers (KMeans, DBSCAN) rather than the hierarchical approach with coverage-aware stratum selection described in `docs/design/inter_neuron_agreement_methodology.md`.

**How:** Decompose `unified_pipeline.py` into three standalone pipeline modules that communicate via filesystem artifacts (CSVs, JSONs), each discoverable by the next stage via directory scanning. Follow the existing registry pattern (ABC + factory) for the new comparing stage and hierarchical clusterer.

## Problem Statement

1. **Monolithic orchestration:** `run_unified_touch_analysis()` couples extraction and clustering — you cannot re-cluster without re-extracting, nor compare without re-running everything.
2. **Missing Stage 3:** No statistical comparison exists. After clustering, the pipeline stops. There is no mechanism to ask "do neurons agree on response within similar-stimulus strata?"
3. **No hierarchical clustering:** The design doc calls for Ward's linkage + adaptive non-uniform dendrogram cut with coverage constraint. Current clusterers (KMeans, DBSCAN, Binning) produce flat partitions without coverage guarantees.
4. **DAG rigidity:** `unified_touch_analysis` is one task in the DAG config. Downstream tasks (`analyse_ap_efficacy`, `map_receptive_fields`) depend on it wholesale when they only need extraction output.

## Goals

### In Scope

1. Split `unified_pipeline.py` into `extraction_pipeline.py`, `clustering_pipeline.py`, `comparing_pipeline.py` + `pipeline_shared.py`
2. Three new Prefect flows in `analysis_workflow.py` with independent DAG task entries
3. Remove `unified_touch_analysis` as a DAG task (replace entirely with the 3 split tasks)
4. New `HierarchicalClusterer` — Ward's linkage, adaptive non-uniform dendrogram cut, coverage constraint (min instances per sensor x min sensor types)
5. New `comparing/` package with `ComparisonStrategy` ABC + 3 concrete strategies (bias, precision, distribution)
6. Comparing operates on `spike_elicited` per cluster, with `session_id` as the neuron/sensor column
7. Dispersion-weighted cross-stratum synthesis
8. Update `configs/analyse_workflow_dag.yaml` with split task structure

### Out of Scope

- Changes to existing extractors (MaxExtractor, StatisticalExtractor, TemporalExtractor, MechanicsOfSolidsExtractor)
- Changes to existing clusterers (KMeansClusterer, DBSCANClusterer, BinningClusterer)
- Changes to `analyse_ap_efficacy` or `map_receptive_fields` internals
- Visualization of comparison results (follow-up)
- DTW/timeseries extractor (already out of scope from previous plan)
- GPU acceleration for clustering or comparison

## Success Criteria

- [ ] Each of the 3 stages runs independently via its own DAG task entry
- [ ] `touch_clustering` discovers extraction output on disk (no explicit file list required from extraction)
- [ ] `touch_comparing` discovers clustered output on disk
- [ ] `HierarchicalClusterer` produces strata where each valid stratum has >= N instances per sensor for >= M sensor types
- [ ] Non-exploitable strata (failing coverage) are tagged and excluded from comparison
- [ ] `BiasComparator` runs ANOVA + Tukey HSD on `spike_elicited` across `session_id` within each stratum
- [ ] `PrecisionComparator` runs Levene's test on `spike_elicited` variance across sessions
- [ ] `DistributionComparator` runs pairwise KS tests on `spike_elicited` distributions
- [ ] Synthesis report weights results by inverse stratum dispersion
- [ ] Existing `analyse_ap_efficacy` and `map_receptive_fields` work with updated `depends_on: [touch_feature_extraction]`
- [ ] Pipeline is idempotent at each stage (second run skips if outputs are up-to-date)

---

## Technical Design

### Approach

Decompose the monolithic `unified_pipeline.py` into three standalone pipeline modules that communicate via filesystem artifacts (CSVs, JSONs). Each module discovers its inputs by scanning known directory patterns. Follow the existing registry pattern (ABC + factory) for the new comparing stage and new hierarchical clusterer.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Keep `unified_pipeline` as backward-compat wrapper | No DAG config changes needed | Confusing two ways to do same thing | **Rejected** — replace entirely |
| Comparing only works with hierarchical clusterer | Purest methodology alignment | Limits flexibility; other clusterers still produce valid strata | **Rejected** — build comparing for all clusterers, recommend hierarchical |
| Pass sensor labels via extended `TouchClusterer` ABC | Cleaner interface | Breaks existing clusterer implementations | **Rejected** — use `config['_sensor_labels']` injection |
| Keep stages in one file with public entry points | Fewer files | Defeats the goal of independent modules | **Rejected** |

### Architecture Changes

#### New Module Structure

```
code/src/analysis/touch_analytics/
├── pipeline_shared.py              # NEW — shared constants, utilities
├── extraction_pipeline.py          # NEW — standalone extraction orchestrator
├── clustering_pipeline.py          # NEW — standalone clustering orchestrator
├── comparing_pipeline.py           # NEW — standalone comparing orchestrator
├── unified_pipeline.py             # DELETED (replaced by the 3 above)
├── comparing/                      # NEW — Stage 3 registry
│   ├── __init__.py
│   ├── base.py                     # ComparisonStrategy ABC + ComparisonResult
│   ├── bias_comparator.py          # ANOVA + Tukey HSD
│   ├── precision_comparator.py     # Levene's test
│   ├── distribution_comparator.py  # Pairwise KS tests
│   └── synthesis.py                # Dispersion-weighted aggregation
├── clustering/
│   ├── ...                         # Existing files untouched
│   └── hierarchical_clusterer.py   # NEW — Ward + adaptive coverage cut
├── feature_extraction/             # UNCHANGED
├── reporting.py                    # UNCHANGED
└── __init__.py                     # UPDATED exports
```

#### Data Flow Between Stages

```
Stage 1 (Extraction)
  Input:  aggregated session CSVs (from 3_merged/)
  Output: 4_analysed/touch_features/<profile>/<session>_touch_summary.csv
     |
     v
Stage 2 (Clustering)
  Input:  discovers touch_features/<profile>/*_touch_summary.csv via glob
  Output: 4_analysed/touch_clusters/<profile>/<clusterer>/
            pooled_touch_summary_clustered.csv
            cluster_metadata.json   (includes dispersion per stratum)
     |
     v
Stage 3 (Comparing)
  Input:  discovers pooled_touch_summary_clustered.csv + cluster_metadata.json from touch_clusters/
  Output: 4_analysed/touch_comparisons/<profile>/<clusterer>/
            <strategy>_results.json     (per-stratum test results)
            synthesis_report.json       (dispersion-weighted global summary)
```

#### pipeline_shared.py

Extracted from `unified_pipeline.py`:
- `SHARED_COLUMNS` (line 67-73)
- `_TqdmLineWrapper` (lines 33-58)
- `session_id_from_path()` (lines 518-523, made public)
- Per-stage default option dicts

#### extraction_pipeline.py

```python
def run_feature_extraction(
    input_items: list[tuple[Path, Path]],
    extraction_profiles: dict,
    output_dir: Path,
    force: bool = False,
) -> dict[str, list[Path]]:
    """Returns {profile_name: [session_csv_paths]}."""
```

Logic moved from `_extract_session()` (lines 180-291) and `_extract_all_touches()` (lines 294-358). Uses existing `get_extractor()` factory. Idempotency via `should_process_task()`.

#### clustering_pipeline.py

```python
def run_clustering(
    output_dir: Path,
    extraction_profiles: dict,   # only names needed, for directory scanning
    clustering_profiles: dict,
    force: bool = False,
) -> dict[str, list[Path]]:
    """Discovers extraction CSVs on disk. Returns {profile/clusterer: [csv_paths]}."""
```

Logic moved from `_cluster_profile()` (lines 365-511) and heatmap helpers (lines 526-607). Uses existing `get_clusterer()` factory. Discovery: `glob(output_dir / profile_name / '*_touch_summary.csv')`.

#### HierarchicalClusterer

```python
# clustering/hierarchical_clusterer.py
class HierarchicalClusterer(TouchClusterer):
    def fit_predict(self, feature_df, config) -> tuple[np.ndarray, dict]:
        """
        Ward's linkage -> full dendrogram -> adaptive non-uniform cut.

        Config keys:
            min_instances_per_sensor (int): N in coverage constraint
            min_sensor_types (int): M in coverage constraint
            _sensor_labels (np.ndarray): injected by caller at runtime
        """
```

**Algorithm (from design doc Stage 2):**
1. Compute linkage matrix with Ward's method on z-scored features
2. Build dendrogram tree as a binary tree structure
3. Recursive top-down cut:
   - At each node, check if both children independently satisfy coverage constraint
   - If yes: descend into both (finer strata)
   - If no: current node becomes a stratum (if it itself satisfies constraint)
   - If current node fails constraint: mark as non-exploitable
4. Tag each stratum with dispersion (within-cluster variance)
5. Return labels + metadata including `dispersion_per_stratum`, `non_exploitable_count`

**Sensor label injection:** The `TouchClusterer.fit_predict()` interface receives only numeric `feature_df`. To preserve the ABC, the caller (`clustering_pipeline.py`) injects sensor labels via `config['_sensor_labels']`. The `_` prefix convention signals a runtime-injected key, not a YAML-configured one. Existing clusterers ignore unknown config keys.

#### Comparing Stage

**`comparing/base.py`:**

```python
@dataclass
class ComparisonResult:
    strategy_name: str
    stratum_label: int | str
    stratum_dispersion: float
    n_sensors: int
    n_observations: int
    test_statistic: float
    p_value: float
    effect_size: float | None
    detail: dict              # strategy-specific (e.g., pairwise post-hoc table)

class ComparisonStrategy(ABC):
    @abstractmethod
    def compare(
        self,
        stratum_df: pd.DataFrame,
        sensor_col: str,
        measurement_col: str,
        config: dict,
    ) -> ComparisonResult:
        """Compare measurement across sensors within one stratum."""
```

**Concrete strategies:**

| Strategy | Registry key | Test | Detail output |
|----------|-------------|------|---------------|
| `BiasComparator` | `bias` | One-way ANOVA (`scipy.stats.f_oneway`); Tukey HSD post-hoc (`scipy.stats.tukey_hsd`) | Per-pair mean difference, CI, p-value |
| `PrecisionComparator` | `precision` | Levene's test (`scipy.stats.levene`) | Per-sensor variance, F-stat |
| `DistributionComparator` | `distribution` | Pairwise 2-sample KS (`scipy.stats.ks_2samp`) | Per-pair KS statistic, p-value |

**`synthesis.py`:**

```python
def synthesize_across_strata(
    results: list[ComparisonResult],
    weighting: str = 'inverse_dispersion',
) -> pd.DataFrame:
    """
    Aggregate per-stratum results into a global summary.
    Weight by 1/dispersion: tight strata carry more evidential weight.
    Returns DataFrame with columns: strategy, stratum, p_value, weight, weighted_p, ...
    """
```

#### comparing_pipeline.py

```python
def run_comparing(
    output_dir: Path,
    extraction_profiles: dict,
    clustering_profiles: dict,
    comparing_profiles: dict,
    min_instances_per_sensor: int = 5,
    min_sensor_types: int = 2,
    force: bool = False,
) -> list[Path]:
    """Discovers clustered CSVs, runs comparisons, writes results."""
```

**Orchestration:**
1. For each `(extraction_profile, clustering_profile)`:
   - Load `pooled_touch_summary_clustered.csv` and `cluster_metadata.json`
   - Group by `cluster_label`
   - For each stratum, check coverage: >= `min_instances_per_sensor` rows per `session_id`, for >= `min_sensor_types` distinct `session_id` values
   - Skip non-exploitable strata (log warning)
   - For each enabled comparing_profile, run `comparator.compare()`
2. Run `synthesize_across_strata()` on collected results
3. Write outputs to `comparisons/` subdirectory

#### Prefect Flows (analysis_workflow.py)

```python
@flow(name="touch_feature_extraction")
def touch_feature_extraction_flow(
    input_items, force_processing=False, extraction_profiles=None,
) -> list[Path]: ...

@flow(name="touch_clustering")
def touch_clustering_flow(
    input_items, force_processing=False,
    extraction_profiles=None, clustering_profiles=None,
) -> list[Path]: ...

@flow(name="touch_comparing")
def touch_comparing_flow(
    input_items, force_processing=False,
    extraction_profiles=None, clustering_profiles=None,
    comparing_profiles=None, min_instances_per_sensor=5, min_sensor_types=2,
) -> list[Path]: ...
```

**Updated `available_tasks`:**
```python
available_tasks = [
    ("summarize_session_blocks", summarize_session_blocks_flow),
    ("touch_feature_extraction", touch_feature_extraction_flow),
    ("touch_clustering", touch_clustering_flow),
    ("touch_comparing", touch_comparing_flow),
    ("analyse_ap_efficacy", analyse_ap_efficacy_flow),
    ("map_receptive_fields", map_receptive_fields_flow),
]
```

Remove `unified_touch_analysis_flow` entirely.

#### DAG YAML Config (analyse_workflow_dag.yaml)

```yaml
tasks:
  summarize_session_blocks:
    enabled: false
    options:
      force_processing: false
    depends_on: []

  touch_feature_extraction:
    enabled: true
    options:
      force_processing: true
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
    depends_on: []

  touch_clustering:
    enabled: true
    options:
      force_processing: true
      extraction_profiles:
        max: { method: max }
        statistical: { method: statistical }
        temporal: { method: temporal }
        mos: { method: mechanics_of_solids }
      clustering_profiles:
        kmeans:
          method: kmeans
          min_touches_per_cluster: 30
        dbscan:
          method: dbscan
          eps: auto
          min_touches_per_cluster: 30
        hierarchical:
          method: hierarchical
          min_instances_per_sensor: 5
          min_sensor_types: 2
          sensor_col: session_id
    depends_on: ["touch_feature_extraction"]

  touch_comparing:
    enabled: false
    options:
      force_processing: false
      extraction_profiles:
        max: { method: max }
      clustering_profiles:
        hierarchical: { method: hierarchical }
      comparing_profiles:
        bias:
          method: bias
          measurement_col: spike_elicited
          sensor_col: session_id
          alpha: 0.05
        precision:
          method: precision
          measurement_col: spike_elicited
          sensor_col: session_id
        distribution:
          method: distribution
          measurement_col: spike_elicited
          sensor_col: session_id
      min_instances_per_sensor: 5
      min_sensor_types: 2
    depends_on: ["touch_clustering"]

  analyse_ap_efficacy:
    enabled: false
    options:
      force_processing: false
    depends_on: ["touch_feature_extraction"]

  map_receptive_fields:
    enabled: false
    options:
      grouping_columns: ['type_metadata', 'direction']
      monitor: false
      force_processing: false
    depends_on: ["touch_feature_extraction"]
```

### Architecture Constraints

- **Unit consistency:** All somatosensory metrics are in mm, mm/frame, and mm^2 (see `docs/development/knowledge-base/note-somatosensory-units-and-calculations.md`). Standardization (z-score) in the hierarchical clusterer must account for these heterogeneous units.
- **CuPy import order:** Not applicable — no GPU acceleration in this plan. If added later, CuPy must be imported before preprocessing packages (see CLAUDE.md).

### GUI Integration Notes

The DAG Launcher GUI (`launcher_window.py` → `TaskPanel`) auto-renders task options from the YAML config. The split pipeline requires **no GUI code changes**, but pipeline modules must honor the GUI's profile enable/disable contract.

1. **No GUI code changes needed** — `TaskPanel._is_profile_dict()` auto-detects profile dicts (dict-of-dicts with `method` key). `comparing_profiles` follows this convention and will render as checkbox columns automatically.

2. **Profile enable/disable contract** — When a user unchecks a profile checkbox in the TaskPanel, `DagConfigModel.set_profile_enabled()` writes `enabled: false` into that profile dict in the YAML. All pipeline modules **must** filter these out using `filter_enabled_profiles()` from `pipeline_shared.py` before iterating over profiles. Failure to filter means disabled profiles are silently executed.

3. **Simple scalars** — `min_instances_per_sensor` and `min_sensor_types` render as editable cells automatically (they are plain int values under `options:`).

4. **launcher.yaml** — No changes needed. The launcher already points to `analysis_workflow.py` + the DAG config; new tasks appear automatically.

5. **DagConfigHandler vs DagConfigModel** — The GUI uses `DagConfigModel` (ruamel round-trip, preserves comments); the script uses `DagConfigHandler` (`safe_load`). Both see `enabled: false` as a plain dict key. The filtering responsibility lies in the pipeline modules, not the config layer.

---

## Implementation Plan

### Phase 1: Extract Shared Utilities + Split Extraction
**Goal:** Create `pipeline_shared.py` and `extraction_pipeline.py`, verify extraction runs standalone.
**Started:** 2026-03-18
**Completed:** 2026-03-18

- [x] Task 1.1 — Create `pipeline_shared.py`: move `SHARED_COLUMNS`, `_TqdmLineWrapper`, `session_id_from_path()`, default option dicts from `unified_pipeline.py`. Add `filter_enabled_profiles()` utility for the GUI enable/disable contract.
- [x] Task 1.2 — Create `extraction_pipeline.py`: move `_extract_session()`, `_extract_all_touches()`, wrap in public `run_feature_extraction()`. Use `filter_enabled_profiles()` from `pipeline_shared` instead of inline dict comprehension.
- [x] Task 1.3 — Add `touch_feature_extraction_flow` to `analysis_workflow.py`
- [ ] Task 1.4 — Verify: extraction runs standalone and produces identical output

**Files created:**
- `code/src/analysis/touch_analytics/pipeline_shared.py`
- `code/src/analysis/touch_analytics/extraction_pipeline.py`

**Files modified:**
- `code/scripts/analysis_workflow.py` — add `touch_feature_extraction_flow`
- `code/src/analysis/touch_analytics/__init__.py` — add new export

**Dependencies:** None

### Phase 2: Split Clustering
**Goal:** Create `clustering_pipeline.py` with directory-scanning discovery, verify clustering runs standalone on previously-extracted data.
**Started:** 2026-03-18
**Completed:** 2026-03-18

- [x] Task 2.1 — Create `clustering_pipeline.py`: move `_cluster_profile()`, heatmap helpers, wrap in public `run_clustering()` with glob-based input discovery. Call `filter_enabled_profiles()` on both `extraction_profiles` and `clustering_profiles` at the top of `run_clustering()`.
- [x] Task 2.2 — Add `touch_clustering_flow` to `analysis_workflow.py`
- [ ] Task 2.3 — Verify: clustering discovers extraction CSVs and produces identical output

**Files created:**
- `code/src/analysis/touch_analytics/clustering_pipeline.py`

**Files modified:**
- `code/scripts/analysis_workflow.py` — add `touch_clustering_flow`
- `code/src/analysis/touch_analytics/__init__.py` — add new export

**Dependencies:** Phase 1

### Phase 3: Hierarchical Clusterer
**Goal:** Implement Ward's linkage + adaptive non-uniform dendrogram cut with coverage constraint.
**Started:** 2026-03-18
**Completed:** 2026-03-18

- [x] Task 3.1 — Create `clustering/hierarchical_clusterer.py` implementing `TouchClusterer` ABC
- [x] Task 3.2 — Implement recursive coverage-constrained cut algorithm (design doc Stage 2)
- [x] Task 3.3 — Add dispersion tagging (within-cluster variance) to metadata
- [x] Task 3.4 — Register `'hierarchical'` in `CLUSTERER_REGISTRY`
- [x] Task 3.5 — Inject `_sensor_labels` from `clustering_pipeline.py` when `sensor_col` is configured
- [ ] Task 3.6 — Verify: hierarchical clusterer produces valid strata with coverage guarantees

**Files created:**
- `code/src/analysis/touch_analytics/clustering/hierarchical_clusterer.py`

**Files modified:**
- `code/src/analysis/touch_analytics/clustering/__init__.py` — register new clusterer
- `code/src/analysis/touch_analytics/clustering_pipeline.py` — sensor label injection logic

**Dependencies:** Phase 2; `scipy.cluster.hierarchy` (linkage, dendrogram)

### Phase 4: Comparing Stage
**Goal:** Implement the comparing framework and 3 concrete strategies.
**Started:** 2026-03-18
**Completed:** 2026-03-18

- [x] Task 4.1 — Create `comparing/base.py`: `ComparisonStrategy` ABC + `ComparisonResult` dataclass
- [x] Task 4.2 — Create `comparing/__init__.py`: `COMPARATOR_REGISTRY` + `get_comparator()`
- [x] Task 4.3 — Implement `BiasComparator`: `scipy.stats.f_oneway` + `scipy.stats.tukey_hsd`
- [x] Task 4.4 — Implement `PrecisionComparator`: `scipy.stats.levene`
- [x] Task 4.5 — Implement `DistributionComparator`: pairwise `scipy.stats.ks_2samp`
- [x] Task 4.6 — Implement `synthesis.py`: `synthesize_across_strata()` with inverse-dispersion weighting
- [x] Task 4.7 — Create `comparing_pipeline.py`: orchestrates discovery -> coverage check -> comparison -> synthesis -> output. Call `filter_enabled_profiles()` on all three profile dicts (`extraction_profiles`, `clustering_profiles`, `comparing_profiles`) at the top of `run_comparing()`.
- [x] Task 4.8 — Add `touch_comparing_flow` to `analysis_workflow.py`

**Files created:**
- `code/src/analysis/touch_analytics/comparing/__init__.py`
- `code/src/analysis/touch_analytics/comparing/base.py`
- `code/src/analysis/touch_analytics/comparing/bias_comparator.py`
- `code/src/analysis/touch_analytics/comparing/precision_comparator.py`
- `code/src/analysis/touch_analytics/comparing/distribution_comparator.py`
- `code/src/analysis/touch_analytics/comparing/synthesis.py`
- `code/src/analysis/touch_analytics/comparing_pipeline.py`

**Files modified:**
- `code/scripts/analysis_workflow.py` — add `touch_comparing_flow`
- `code/src/analysis/touch_analytics/__init__.py` — add new exports

**Dependencies:** Phase 2; `scipy.stats` (f_oneway, tukey_hsd, levene, ks_2samp)

### Phase 5: DAG Config + Cleanup
**Goal:** Update DAG YAML, remove old unified_pipeline.py, update task dependencies.
**Started:** 2026-03-18
**Completed:** 2026-03-18

- [x] Task 5.1 — Rewrite `configs/analyse_workflow_dag.yaml` with 3 split tasks
- [x] Task 5.2 — Update `available_tasks` in `analysis_workflow.py`: remove `unified_touch_analysis`, add 3 new tasks
- [x] Task 5.3 — Update kwargs forwarding in `run_batch_analysis()` for new option keys: add `if "comparing_profiles" in options`, `if "min_instances_per_sensor" in options`, `if "min_sensor_types" in options` forwarding blocks alongside the existing `extraction_profiles` / `clustering_profiles` blocks
- [x] Task 5.4 — Update `analyse_ap_efficacy` and `map_receptive_fields` `depends_on` to `["touch_feature_extraction"]`
- [x] Task 5.5 — Delete `unified_pipeline.py` (all logic now in the 3 split modules)
- [x] Task 5.6 — Update `touch_analytics/__init__.py` exports: remove `run_unified_touch_analysis`, add `run_feature_extraction`, `run_clustering`, `run_comparing`
- [x] Task 5.7 — Fix `_collect_unified_files()` in `analysis_workflow.py` to scan profile subdirectories (or accept profile parameter)

**Files modified:**
- `configs/analyse_workflow_dag.yaml` — restructured task entries
- `code/scripts/analysis_workflow.py` — updated available_tasks, kwargs forwarding
- `code/src/analysis/touch_analytics/__init__.py` — updated exports

**Files deleted:**
- `code/src/analysis/touch_analytics/unified_pipeline.py`

**Dependencies:** Phases 1-4

### Phase 6: Integration Testing
**Goal:** Verify end-to-end correctness of the split pipeline.

- [ ] Task 6.1 — Run extraction alone -> verify per-session CSVs match previous output
- [ ] Task 6.2 — Run clustering alone (on pre-existing extraction output) -> verify clustered CSVs match
- [ ] Task 6.3 — Run hierarchical clustering -> verify coverage constraint holds in all valid strata
- [ ] Task 6.4 — Run comparing -> verify JSON outputs with ANOVA/Levene/KS results
- [ ] Task 6.5 — Run full chain via `depends_on` -> verify all 3 stages execute in order
- [ ] Task 6.6 — Verify `analyse_ap_efficacy` and `map_receptive_fields` still find their inputs
- [ ] Task 6.7 — Test idempotency: second run skips at each stage

**Dependencies:** Phase 5

---

## Testing Plan

### Unit Tests
- [ ] `HierarchicalClusterer` with synthetic data: verify coverage constraint enforced
- [ ] `HierarchicalClusterer` with insufficient data: verify non-exploitable strata tagged
- [ ] `BiasComparator.compare()` with known means: verify ANOVA detects difference
- [ ] `PrecisionComparator.compare()` with known variances: verify Levene's detects difference
- [ ] `DistributionComparator.compare()` with known distributions: verify KS detects difference
- [ ] `synthesize_across_strata()`: verify dispersion weighting math
- [ ] Coverage check logic: edge cases (0 sensors, 1 sensor, exactly at threshold)

### Integration Tests
- [ ] Extraction -> Clustering -> Comparing full chain on real data
- [ ] Each stage discovers inputs from previous stage's filesystem output
- [ ] Downstream tasks (`analyse_ap_efficacy`, `map_receptive_fields`) consume extraction output correctly

### Manual Verification
- [ ] Inspect `cluster_metadata.json` for hierarchical clusterer: dispersion values, non-exploitable count
- [ ] Inspect `comparisons/bias_results.json`: ANOVA p-values make domain sense
- [ ] Inspect `synthesis_report.json`: dispersion weighting produces sensible global summary

### Edge Cases
- [ ] Session with 0 valid touches (all `single_touch_id == 0`) — extraction produces empty CSV, no crash
- [ ] Single session in dataset — hierarchical clusterer tags all strata as non-exploitable (only 1 sensor)
- [ ] All strata fail coverage — comparing outputs empty results with warning, no crash
- [ ] `spike_elicited` is constant (all 0 or all 1) within a stratum — ANOVA returns NaN, handled gracefully

---

## Documentation Plan

- [ ] Update `CLAUDE.md` memory section with new pipeline module locations
- [ ] Add inline docstrings to all new ABC classes and concrete implementations
- [ ] Document YAML profile schema in comments within `analyse_workflow_dag.yaml`
- [ ] Add knowledge base note if hierarchical clusterer coverage constraint assumptions need documenting

---

## Rollback Plan

1. **Before deployment:**
   - All new files can be deleted without affecting existing code
   - `unified_pipeline.py` can be restored from git (only deleted in Phase 5)

2. **Data considerations:**
   - No database migrations
   - New output subdirectories (`comparisons/`) can be deleted without affecting existing outputs
   - Extraction and clustering output paths are unchanged

3. **Rollback procedure:**
   - Revert the merge commit on `dev`
   - Restore original `analyse_workflow_dag.yaml` from git
   - Existing extractors and clusterers are completely untouched throughout

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `spike_elicited` is binary — ANOVA on binary data is debatable | Medium | Medium | ANOVA is robust to non-normality with large N; KS test is distribution-free; document limitation |
| Too few sessions per stratum for meaningful comparison | Medium | High | Coverage constraint filters out strata; report `non_exploitable_count` prominently |
| `scipy.stats.tukey_hsd` requires scipy >= 1.8 | Low | Low | Check version in environment.yml; fallback to `statsmodels` if needed |
| Hierarchical clustering O(n^2) memory for large datasets | Low | Medium | Touch counts per session are moderate (~hundreds); log warning if N > threshold |
| Profile config duplication across task YAML entries | Low | Low | Clustering/comparing only need profile *names* for directory discovery; can simplify to list |
| TaskPanel column explosion (~10 profile checkboxes across 3 tasks) | Low | Low | Existing behavior (greyed-out columns for non-applicable profiles); follow-up: collapsible profile groups |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Shared utilities + extraction split | ~2 new files, ~250 LOC | None |
| Phase 2: Clustering split | ~1 new file, ~200 LOC | Phase 1 |
| Phase 3: Hierarchical clusterer | ~1 new file, ~200 LOC | Phase 2 |
| Phase 4: Comparing stage | ~7 new files, ~400 LOC | Phase 2 |
| Phase 5: DAG config + cleanup | ~3 modified files, ~100 LOC | Phases 1-4 |
| Phase 6: Integration testing | Manual verification | Phase 5 |

---

## References

- Design methodology: `docs/design/inter_neuron_agreement_methodology.md`
- Previous unified pipeline plan: `docs/development/plans/completed/unified-touch-analysis-pipeline.md`
- Current unified pipeline: `code/src/analysis/touch_analytics/unified_pipeline.py`
- Prefect flows: `code/scripts/analysis_workflow.py`
- DAG config: `configs/analyse_workflow_dag.yaml`
- Extractor registry: `code/src/analysis/touch_analytics/feature_extraction/__init__.py`
- Clusterer registry: `code/src/analysis/touch_analytics/clustering/__init__.py`
- Idempotency utility: `code/src/utils/should_process_task.py`
- Unit conventions: `docs/development/knowledge-base/note-somatosensory-units-and-calculations.md`
- GUI task panel: `code/src/utils/gui/dag_launcher/task_panel.py`
- GUI launcher window: `code/src/utils/gui/dag_launcher/launcher_window.py`
- GUI config model: `code/src/utils/pipeline/dag_config_model.py`
