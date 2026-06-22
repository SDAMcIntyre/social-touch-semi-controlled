# Implementation Report: 2026-05-28 to 2026-06-09

**Generated:** 2026-06-09
**Period:** 2026-05-28 to 2026-06-09 (12 days)
**Total commits:** 36 (across all branches)

---

## Summary

Over the past 12 days, the social-touch-semi-controlled project underwent a major expansion of the **stimulus-response analysis pipeline** and significant **visual/scientific quality improvements**. The work spans 36 commits across 15+ feature branches, with 13 plans completed and 1 actively in progress. The primary themes were:

1. **Stimulus tuning analysis** -- A full suite of IFF tuning curve tasks was built from scratch, progressively enriched with error bars, designed-level composition bars, spike-count metrics, instruction-level categorical analysis, neuron-type coloring, and raw-dots scatter plots.
2. **Spatial analysis improvements** -- Circular RF population crop rendering, IFF metric selection (mean vs max), and cross-session comparison tasks were added.
3. **Publication readiness** -- The jet colormap was replaced with perceptually uniform alternatives (inferno), UV-space distances were converted to physical millimetres, and comprehensive article-scoping documentation was created.
4. **Nomenclature and organisation** -- All ~24 analysis tasks were renamed with a systematic prefix scheme (spatial_, stimulus_, cross_, touch_), output directories were realigned, and IFF-specific naming is being generalized to "response" terminology.
5. **Documentation** -- 5 new knowledge-base investigation notes, 6 article-scoping chapters, 3 architecture reference documents, and a literature reference library with 15+ PDFs and analysis summaries were added.

---

## Branches and Merges

### Merged into dev (chronological order)

| Date | Branch | Merge Commit | Description |
|------|--------|-------------|-------------|
| 2026-05-29 | `refactor/analysis-task-nomenclature` | `70287c4` | Renamed all ~24 analysis tasks with systematic prefixes |
| 2026-05-29 | `refactor/analysis-output-dirs` | `2d28245` | Aligned output directories with new task nomenclature (cascade-merged subordinate branches) |
| 2026-06-01 | `feature/stimulus-iff-tuning-enrichment` | `955d4ff` | Full stimulus tuning enrichment chain (cascade of 5 sub-branches) |
| 2026-06-09 | `feature/uv-to-mm-conversion` | `d632790` | UV-to-mm conversion + jet colormap replacement (cascade of 2 sub-branches) |

### Cascade merge chains

**Chain 1 (merged 2026-05-29):** Output directory alignment
```
feature/rf-circular-crop-mesh-rendering
  -> feature/stimulus-iff-tuning-curves
    -> feature/iff-metric-mean-max-selection
      -> feature/rf-circular-population-plot
        -> feature/stimulus-compare-sessions
          -> refactor/analysis-output-dirs -> dev
```

**Chain 2 (merged 2026-06-01):** Stimulus tuning enrichment
```
feature/stimulus-iff-tuning-enrichment (base)
  -> feature/stimulus-tuning-spike-count-metric
    -> feature/stimulus-iff-instruction-tuning
      -> feature/iff-tuning-neuron-type-coloring
        -> feature/iff-tuning-raw-dots -> dev
```

**Chain 3 (merged 2026-06-09):** Visual/scientific quality
```
feature/uv-to-mm-conversion
  -> feature/replace-jet-colormap -> dev
```

### Active branches

| Branch | Status | Description |
|--------|--------|-------------|
| `feature/rename-iff-to-response-tuning` | **Active, in progress** | Renaming IFF tuning tasks to "response tuning"; 1 commit + uncommitted changes |

### Other local branches (not active in this period)

| Branch | Description |
|--------|-------------|
| `feature/rf-stroke-virtual-gesture-subset` | Pre-existing, no activity since May 28 |
| `main` | Production branch, not updated in this period |

---

## Completed Plans (13)

All plans follow the lifecycle: pending -> active -> completed. Listed in merge order.

### 2026-05-29 -- Nomenclature and Organisation Batch

| Plan | Branch | Summary |
|------|--------|---------|
| **RF Circular Crop Mesh Rendering** | `feature/rf-circular-crop-mesh-rendering` | Replaced scatter pointcloud forearm background with solid triangulated mesh surface in circular RF population crop renderer; made heatmap overlay fully opaque |
| **IFF Metric Selection (Mean vs Max)** | `feature/iff-metric-mean-max-selection` | Added `iff_metric` option ("mean" / "max" / "both") so both mean and max IFF are produced as separate outputs across spatial and stimulus paths |
| **IFF Tuning Curves Pipeline** | `feature/stimulus-iff-tuning-curves` | New analysis task generating binned tuning curves showing touch features vs mean neural firing rate, per-session and cross-session overlays, faceted by gesture subset |
| **Circular RF Population Plot** | `feature/rf-circular-population-plot` | Publication-quality circular transparent-background RF heatmap PNGs centered on boundary centroid and hotspot peak, with forearm skin texture background |
| **Cross-Session Touch Feature Comparison** | `feature/stimulus-compare-sessions` | New task pooling per-session touch feature CSVs for cross-session box+strip comparison plots and summary grids |
| **Analysis Task Nomenclature Redesign** | `refactor/analysis-task-nomenclature` | Renamed all ~24 analysis tasks with spatial_, stimulus_, cross_, touch_ prefixes; added DAG graph category colors |

### 2026-06-01 -- Stimulus Tuning Enrichment Batch

| Plan | Branch | Summary |
|------|--------|---------|
| **IFF Tuning Enrichment** (STD, level bars, overlapping bins, CSV) | `feature/stimulus-iff-tuning-enrichment` | Added per-bin STD error bars, stacked designed-level composition bars, overlapping bin windows, and CSV export to IFF tuning curves |
| **Spike-Count Metric** | `feature/stimulus-tuning-spike-count-metric` | Generalized Y-axis from IFF-only to response_metric selector supporting iff_mean, iff_max, spike_count_mean, spike_count_total; new SpikeCountExtractor |
| **IFF Instruction Tuning** | `feature/stimulus-iff-instruction-tuning` | New task rendering mean IFF bar charts grouped by designed metadata instruction levels (contact area, speed, force) with cross-session overlays |
| **Neuron Type Coloring** | `feature/iff-tuning-neuron-type-coloring` | Colored tuning curves by neuron type (read from MNG-DataSummary.xlsx) with a shared hardcoded neuron-type color map across both tuning tasks |
| **Raw Dots Binning Strategy** | `feature/iff-tuning-raw-dots` | Added `raw_dots` alternative to sliding_window binning -- individual touches as scatter dots with polynomial regression fit lines and R-squared annotations |

### 2026-06-09 -- Visual and Scientific Quality

| Plan | Branch | Summary |
|------|--------|---------|
| **Replace Jet Colormap** | `feature/replace-jet-colormap` | Replaced all hardcoded `jet` colormaps with `inferno` (perceptually uniform); made colormap user-selectable from DAG launcher dropdown |
| **UV-to-mm Conversion** | `feature/uv-to-mm-conversion` | Converted all SLIM UV-parameter-space distances, offsets, PCA axes, area, and perimeter to physical millimetres in comparison pipelines |

---

## Active Plans (3)

| Plan | Branch | Status | Summary |
|------|--------|--------|---------|
| **Rename IFF to Response Tuning** | `feature/rename-iff-to-response-tuning` | In progress | Renaming task keys, output dirs, Python modules from "iff" to "response"; enriching output filenames with feature/gesture/metric; connecting same-session dots in overlay plots |
| **RF Explorer Handoff** | `feature/rf-explorer-event-and-scalar-fix` | Handoff doc (pre-existing) | Documents 6 mouse event and scalar bar bugs that were fixed in the RF Feature Space Explorer GUI |
| **XY-Deduplication Forearm and Contacts** | `feature/contact-projection-first-architecture` | In progress (pre-existing, started 2026-05-05) | Deduplicating forearm pointcloud vertices and per-frame contact points by (x,y) position, keeping the lowest z |

---

## Pending Plans (3 new in this period)

| Plan | Date Created | Summary |
|------|-------------|---------|
| **Feature Ranges GUI Button** | 2026-05-13 | Move standalone feature range inspection script into the pipeline as a GUI button within the population RF grid task panel |
| **RF Inflection Basin Opening** | 2026-05-20 | Add morphological opening step to negative-Laplacian flood-fill basin to remove thin protrusions before contour extraction |
| **RF Output Directory Alignment** | 2026-05-27 | Rename 20 output subdirectories under `4_analysed/` to mirror new task naming; extract hardcoded paths into constants module |

---

## Detailed Changelog

### 2026-05-29 -- Nomenclature, Spatial Rendering, and New Analysis Tasks

**`74f9d3d`** (2026-05-29) `feat(rf-circular-crop-mesh-rendering): extend iff-both mode and analysis renderers`
- Modified 11 files: `analysis_workflow_processing.py`, 5 pipeline files, 2 renderers, `clustering_pipeline.py`, `task_detail_panel.py`, DAG config
- Extended IFF-both mode across all analysis renderers; added RF circular crop mesh rendering

**`49e70f4`** (2026-05-29) `docs(rf-circular-crop-mesh-rendering): mark plan as completed`

**`452d9a0`** (2026-05-29) Merge `feature/rf-circular-crop-mesh-rendering` into `feature/stimulus-iff-tuning-curves`

**`84c01c6`** (2026-05-29) `docs(iff-metric-mean-max-selection): mark plan as completed`

**`03284c0`** (2026-05-29) Merge `feature/iff-metric-mean-max-selection` into `feature/stimulus-iff-tuning-curves`

**`1bfb88d`** (2026-05-29) `docs(stimulus-iff-tuning-curves): mark plan as completed`

**`fa70d61`** (2026-05-29) Merge `feature/stimulus-iff-tuning-curves` into `feature/rf-circular-population-plot`

**`d97446f`** (2026-05-29) `docs(rf-circular-population-plot): mark plan as completed`

**`6d91307`** (2026-05-29) Merge `feature/rf-circular-population-plot` into `feature/stimulus-compare-sessions`

**`bdcfd66`** (2026-05-29) `docs(stimulus-compare-sessions): mark plan as completed`

**`4f19c87`** (2026-05-29) Merge `feature/stimulus-compare-sessions` into `refactor/analysis-output-dirs`

**`ad11568`** (2026-05-29) `docs(analysis-task-nomenclature): mark plan as completed`

**`70287c4`** (2026-05-29) Merge `refactor/analysis-task-nomenclature` into `dev`

**`2d28245`** (2026-05-29) Merge `refactor/analysis-output-dirs` into `dev`

### 2026-05-30 to 2026-06-01 -- Stimulus Tuning Enrichment Chain

**`b5dd0ea`** (2026-05-30) `feat(stimulus-iff-tuning): add STD error bars, stacked level bars, overlapping bins, CSV export to IFF tuning curves`
- 5 files, +738/-44 lines
- Added per-bin IFF standard deviation with error bars, stacked designed-level composition bars, overlapping bin windows, CSV export of binned data

**`006b8ff`** (2026-05-31) `feat(stimulus-tuning): add spike-count metric to IFF tuning curves`
- 8 files, +819/-328 lines
- Generalized Y-axis to response_metric selector; new `SpikeCountExtractor` producing per-touch `Nerve_spike_count` column; 4 response metric modes

**`ac62852`** (2026-05-31) `feat(stimulus-iff-instruction-tuning): add IFF instruction tuning bar chart task`
- 7 files, +1176 lines (all new)
- New pipeline (`rf_iff_instruction_tuning_pipeline.py`, 500 lines) and renderer (295 lines); per-session bar charts and cross-session overlays by instruction level

**`efb068c`** (2026-06-01) `feat(iff-tuning): color IFF tuning curves by neuron type`
- 8 files, +607/-41 lines
- New `neuron_type_colors.py` module (150 lines) with hardcoded neuron-type color map; integrated into both tuning and instruction tuning tasks

**`ce1c1ae`** (2026-06-01) `feat(iff-tuning): add raw_dots binning strategy to iff tuning`
- 4 files, +727/-111 lines
- Major refactor of `rf_iff_tuning_pipeline.py` (422 lines changed); new renderer methods for scatter + polynomial fit + R-squared

**`7f10dba`** (2026-06-01) `feat(iff-tuning-raw-dots): wire flow layer options, update neuron colors, plan doc`
- 6 files, +63/-15 lines
- Wired binning_strategy through flow layer options; updated neuron color assignments

**`98461e4` to `955d4ff`** (2026-06-01) Plan completion markers and cascade merges (6 plans marked completed, 5 branch merges cascading into dev)

### 2026-06-08 to 2026-06-09 -- Visual Quality and Unit Conversion

**`c62491d`** (2026-06-08) `feat(uv-to-mm-conversion): convert UV-space distances and PCA axes to millimetres`
- 4 files, +328/-76 lines
- Applied `compute_uv_to_mm_scale()` to proximal-distal center and session boundary comparison pipelines; renamed CSV columns from `_uv` to `_mm`

**`d031b8b`** (2026-06-08) `feat(replace-jet-colormap): replace jet with perceptually uniform colormap`
- 14 files, +338/-34 lines
- Replaced `jet` with `inferno` in all renderers, pipelines, GUI viewers, and DAG config; added `cmap` parameter to 4 renderer functions

**`127a092`** (2026-06-09) `chore(replace-jet-colormap): commit outstanding working-tree changes`
- 65 files, +6832/-34 lines
- Large commit including: `.gitignore` updates, `migrate_output_dirs.py` migration script, full article-scoping documentation suite (6 chapters + README), literature reference library (15+ PDFs + analysis summaries), 3 architecture reference documents, 5 knowledge-base investigation notes, 3 new pending plans

**`39c2ee8`** (2026-06-09) `docs(replace-jet-colormap): mark plan as completed`

**`67140f5`** (2026-06-09) Merge `feature/replace-jet-colormap` into `feature/uv-to-mm-conversion`

**`bb437dd`** (2026-06-09) `docs(uv-to-mm-conversion): mark plan as completed`

**`d632790`** (2026-06-09) Merge `feature/uv-to-mm-conversion` into `dev`

### 2026-06-09 -- Active Work (current branch)

**`de49441`** (2026-06-09) `refactor(rename-iff-to-response-tuning): rename IFF tuning tasks to response tuning and enrich output filenames`
- 9 files, +445/-128 lines
- Renamed `rf_iff_tuning_pipeline.py` -> `rf_response_tuning_pipeline.py`
- Renamed `rf_iff_instruction_tuning_pipeline.py` -> `rf_response_instruction_tuning_pipeline.py`
- Renamed corresponding renderer files
- Updated task keys, output dirs, constants, flows, and DAG config

---

## Knowledge Base Updates (5 new notes)

All added in the `127a092` commit (2026-06-09), though investigation dates vary:

| Note | Date | Status | Summary |
|------|------|--------|---------|
| `investigation-jet-colormap-perceptual-problems.md` | 2026-06-08 | Research complete | Literature review of 15 sources and 67 claims on perceptual problems with the jet (rainbow) colormap; documented current codebase usage in 5 renderer functions |
| `investigation-caret-pals-for-forearm-flattening.md` | 2026-05-15 | Exploratory | Investigated whether Caret/PALS cortical flattening approach could replace the cylindrical unwrap used in RF mapping; compared with current SLIM UV parameterization |
| `investigation-handmesh-over-smoothing.md` | 2026-05-12 | Resolved | Root cause: YAML parameter changes were never written to the config file; One-Euro filter at beta=0.007 was effectively a fixed ~1.3 Hz low-pass |
| `investigation-hotspot-images-not-produced.md` | 2026-05-26 | Fixed | Hotspot PNGs not produced because stale NPZ files lacked peak_uv keys and upstream task had failed with TypeError |
| `investigation-slim-uv-delaunay-quality.md` | 2026-05-22 | Open | Investigating why SLIM UV pipeline with mesh_method delaunay produces poor-quality triangulation compared to preprocessing pipeline using the same algorithm |

---

## Documentation Additions

### Article Scoping (new section, `docs/article-scoping/`)

A complete article-scoping framework was added with 6 chapters and supporting materials:

| File | Content |
|------|---------|
| `README.md` | Overview and navigation for the article-scoping section |
| `01-dataset-overview.md` | Dataset description and experimental design |
| `02-pipeline-capabilities.md` | Pipeline capabilities summary |
| `03-literature-comparison.md` | Literature comparison and positioning |
| `04-novelty-assessment.md` | Novelty assessment against existing work |
| `05-article-structure.md` | Proposed article structure |
| `06-open-issues.md` | Open issues and gaps to address |

### Literature References (`docs/article-scoping/references/`)

- 15+ PDFs of key publications (Johansson and Vallbo 1979/1980/1983, Vallbo et al 1995/1999, Ackerley et al 2014, Loken et al 2009, Bai et al 2015, Kuehn et al 2019, Xu et al 2025, and others)
- `SUMMARY.md` -- comprehensive literature reference summary
- `classification-receptive-field.md` and `classification-receptive-field-extended.md` -- RF classification frameworks
- `classification-stimulus-tuning.md` -- stimulus tuning classification framework
- `johansson-vallbo-1983-reference-abstracts.md` -- detailed reference abstracts
- `vallbo-et-al-1995-analysis.md` -- analysis of Vallbo et al 1995

### Architecture Documentation (`docs/architecture/`)

| File | Lines | Content |
|------|-------|---------|
| `analysis-pipeline-outcomes.md` | 688 | Catalogue of analysis pipeline outputs |
| `dag-graph-view-reference.md` | 217 | DAG graph visualization reference |
| `gui-dag-launcher-reference.md` | 1448 | Comprehensive GUI DAG launcher reference |

---

## Current State

### Branch: `feature/rename-iff-to-response-tuning` (checked out)

**Last commit:** `de49441` (2026-06-09 12:07) -- rename IFF tuning tasks to response tuning

**Uncommitted staged changes:** None

**Uncommitted unstaged changes (6 modified files):**

| File | Nature of Changes |
|------|-------------------|
| `code/scripts/launch_pipeline_gui.py` | Added logging setup, exception hook for unhandled Qt exceptions (+23 lines) |
| `code/src/analysis/receptive_field_mapping/gui/touch_playback_explorer.py` | Minor additions (+8 lines) |
| `code/src/analysis/receptive_field_mapping/pipelines/rf_population_response_field_pipeline.py` | Additions (+15 lines) |
| `code/src/utils/gui/dag_launcher/dag_graph_items.py` | Minor addition (+1 line) |
| `configs/analyse_workflow_processing_dag.yaml` | Toggled `force_processing` flags (false for spatial tasks, true for response tuning) |
| `docs/article-scoping/references/classification-receptive-field-extended.md` | Expanded RF classification content (+29 lines) |

**Untracked files (3):**

| File | Content |
|------|---------|
| `docs/article-scoping/references/analysis-hairy-skin-literature.md` | Hairy skin literature analysis |
| `docs/article-scoping/references/analysis-hairy-skin-myelinated-literature.md` | Hairy skin myelinated afferent literature analysis |
| `docs/article-scoping/references/plan-investigation-hairy-skin-myelinated.md` | Investigation plan for hairy skin myelinated afferents |

### dev vs main

The `dev` branch is significantly ahead of `main` (hundreds of commits). No merges to `main` occurred in this period. The `main` branch has not been updated.

### Summary statistics for this period

- **36 commits** (including merges)
- **13 plans completed**, 1 in progress, 3 new pending
- **15+ feature/refactor branches** created and merged
- **5 knowledge-base notes** added
- **~30 new documentation files** (article scoping, architecture, literature)
- Key file additions: `neuron_type_colors.py`, `rf_iff_instruction_tuning_pipeline.py` (now `rf_response_instruction_tuning_pipeline.py`), `spike_count.py`, `migrate_output_dirs.py`
- Primary areas modified: `code/src/analysis/receptive_field_mapping/`, `code/scripts/analysis_workflow_processing.py`, `configs/analyse_workflow_processing_dag.yaml`
