# Article Figure Specification — Hierarchical RF & Stimulus Sensitivity Analysis

## Context

The project needs a complete set of publication-quality figures for a scientific article characterizing spatial response fields and stimulus sensitivity across 5 afferent subtypes (SA-I, SA-II, HFA, Field-LTMR, CT; 11 neurons total). The dataset is organized hierarchically by gesture type (tap vs stroke) and stroke direction (distal vs proximal), and the figures must reflect this tree structure at every analysis level. Two parallel analysis families — **RF spatial features** and **stimulus sensitivity** — follow the same hierarchy, with a cross-neuron comparison layer on top.

This plan is a **specification document** — it defines what figures to generate, their content, and the infrastructure changes needed. It does NOT define implementation phases yet; those will come after the spec is validated.

---

## Figure Hierarchy — Overview

Both families follow the same data-splitting tree:

```
Per neuron (11 neurons, grouped by 5 types)
├── L0: All stimuli (baseline RF / response)
├── L1: Tap vs Stroke → comparison
├── L2: Stroke Distal vs Stroke Proximal → comparison
└── L3: Within each {tap, stroke_distal, stroke_proximal}:
    ├── Impact of velocity (1D tuning curve + 2D grid)
    ├── Impact of depth (1D tuning curve + 2D grid)
    └── Impact of area (1D tuning curve only — to dismiss it)
```

**Figure organization:** Per-level figures with all neurons on the same figure, grouped/colored by neuron type (Okabe-Ito scheme). Per-neuron detailed breakdowns go to supplementary.

**Cross-neuron comparison:** At every level, neurons are grouped by type (SA-I, SA-II, CT, Field, HFA) with individual data points visible (strip/swarm style). Non-parametric stats (Kruskal-Wallis, Mann-Whitney) reported despite small n.

---

## Family 1 — RF Spatial Features

**Core metrics (bar charts, mandatory at every level):**
- Inflection boundary area (mm²)
- Circularity (4π·area/perimeter²)
- PCA major/minor axis ratio (elongation)
- PCA orientation angle (degrees)
- Centroid shift between compared conditions (UV units → mm when available)

**Visual panels (contour overlays, to be explored):**
- Option A: One example neuron's heatmap + contour per condition
- Option B: All neurons' contours overlaid per condition
- Both options should be implemented; final choice based on rendering quality

### L0 — RF Overview (all stimuli)

| Panel | Content |
|-------|---------|
| Heatmap | SLIM UV interpolated heatmap per neuron (IFF mode) |
| Contour | Inflection boundary overlay |
| Metrics table | Area, circularity, PCA ratio, PCA angle per neuron |
| Cross-neuron | Grouped bar chart of all 5 metrics, neurons by type |

### L1 — Tap vs Stroke Comparison

| Panel | Content |
|-------|---------|
| Contour overlay | Tap contour vs Stroke contour per neuron |
| Metric bars | Paired bars (tap/stroke) for area, circularity, PCA ratio, PCA angle |
| Centroid shift | Distance between tap centroid and stroke centroid |
| Cross-neuron | Same metrics grouped by type, condition as hue |

### L2 — Distal Stroke vs Proximal Stroke Comparison

| Panel | Content |
|-------|---------|
| Contour overlay | Distal vs Proximal contour per neuron |
| Metric bars | Paired bars (distal/proximal) for area, circularity, PCA ratio, PCA angle |
| Centroid shift | Proximal-distal centroid distance (already computed in pipeline) |
| Cross-neuron | Same metrics grouped by type |

### L3 — Impact of Stimulus Parameters on RF

For each gesture subset {tap, stroke_distal, stroke_proximal}:

**2D Grid Heatmaps (velocity × depth):**
- Existing `cross_render_grid_metrics` pipeline with `velocity_depth_2d` group
- Metrics per cell: mean IFF, area, circularity, centroid shift from baseline
- One heatmap grid per neuron, per gesture subset
- Cross-neuron: overlay or side-by-side by type

**1D Tuning Curves (scatter + fit):**
- X = stimulus parameter (velocity, depth, or area), Y = RF metric
- One dot per touch, polynomial fit line
- Fit degree: **to be investigated** — generate linear (1), quadratic (2), cubic (3), degree-4 fits simultaneously; compare R² and biological interpretability to select best
- Three stimulus parameters:
  - **Velocity**: full analysis
  - **Depth**: full analysis
  - **Area**: single summary figure to demonstrate no significant RF impact → then area is dropped from subsequent analysis

**Area Dismissal Figure:**
- Single summary figure across all neurons
- 1D tuning curve: contact_area vs RF area (or circularity)
- Show flat/non-significant relationship
- Brief statistical test (Spearman correlation, p-value)
- Placed early in results to justify dropping area dimension

---

## Family 2 — Stimulus Sensitivity (Temporal Features)

Same L0→L3 hierarchy as Family 1, but the Y-axis is neural responsiveness.

**Response metrics (three, shown as separate panels/rows):**
- Mean IFF (Hz) — `Nerve_freq_mean`
- Max IFF (Hz) — `Nerve_freq_max`
- Spike count — `Nerve_spike_count`

### Approach A — Summary Metrics (existing infrastructure)

At each hierarchy level, bar charts of the three response metrics:

| Level | Comparison |
|-------|-----------|
| L0 | Baseline response per neuron (all stimuli) |
| L1 | Tap vs Stroke response |
| L2 | Distal vs Proximal stroke response |
| L3 | 1D tuning curves: velocity/depth/area → mean IFF / max IFF / spike count |

L3 uses the `stimulus_response_tuning` pipeline (raw_dots mode).

### Approach B — Response Dynamics (NEW — needs to be built)

**Concept:** Align IFF time series to contact onset, average within conditions, overlay to compare temporal response profiles.

**What exists:**
- Per-frame IFF data in `TouchEvent.frame_iff` (ndarray, one value per frame)
- Per-frame spike data in `TouchEvent.frame_spikes` (bool array)
- Per-frame contact points with vertex indices
- Frame-level gesture type and contact detection flags

**What needs to be built:**
1. **Temporal alignment**: Align each touch's IFF trace to contact onset (frame 0 = first contact frame). Normalize trace length or truncate to a common window.
2. **Condition-wise averaging**: Group touches by condition (gesture type, velocity bin, depth bin), compute mean ± SEM IFF trace per condition.
3. **Overlay rendering**: Matplotlib figure with multiple mean IFF traces overlaid, color-coded by condition, with shaded SEM bands.
4. **Similarity metric**: Quantify how similar/different two temporal profiles are — options include Pearson correlation, DTW distance, or RMSE between mean traces.

**Hierarchy application:**
| Level | Overlay comparison |
|-------|-------------------|
| L1 | Mean IFF trace for tap vs stroke |
| L2 | Mean IFF trace for distal vs proximal |
| L3 | Mean IFF traces per velocity bin (e.g., slow/medium/fast) |
| L3 | Mean IFF traces per depth bin (e.g., shallow/medium/deep) |

---

## Cross-Neuron Comparison Layer

At **every** hierarchy level (L0–L3), a cross-neuron figure:

- **Visual**: Grouped bar/strip chart, X = neuron type (5 groups), individual neuron dots visible
- **Color**: Okabe-Ito neuron type palette (SAI=orange, SAII=sky blue, CT=bluish green, Field=blue, HFA=vermillion)
- **Statistics**: Kruskal-Wallis across types + Mann-Whitney pairwise (report p-values; acknowledge small n in text)
- **Metrics**: whichever metrics are relevant at that level (RF metrics for Family 1, response metrics for Family 2)

---

## Fit Function Investigation (L3 Tuning Curves)

Current config: degree-4 polynomial. The plan is to **expand fitting options** and compare:

| Fit type | Degree | Use case |
|----------|--------|----------|
| Linear | 1 | Monotonic relationships (e.g., Aβ velocity) |
| Quadratic | 2 | Inverted-U shapes (e.g., CT velocity tuning) |
| Cubic | 3 | Asymmetric peaks |
| Quartic | 4 | Current default — flexible |
| (Future) Exponential/sigmoid | — | If polynomial fits poorly |

**Implementation:** Generate all fits in parallel, report R² and AIC/BIC for each, render all on the same scatter plot with a legend. The scientist selects the best fit per neuron × feature combination based on interpretability and goodness-of-fit.

**Infrastructure change:** The `fit_degree` option should accept a list (e.g., `[1, 2, 3, 4]`) instead of a single int, and the renderer should overlay all fits.

---

## DAG Configuration Changes

The following task options need modification or addition:

1. **`stimulus_response_tuning`**: accept `fit_degree` as a list; add RF metrics (area, circularity) as Y-axis options alongside IFF metrics
2. **`cross_map_feature_grid`**: ensure `velocity_depth_2d` is enabled; add per-gesture-type output
3. **`spatial_extract_boundaries`**: ensure all 5 gesture subsets are produced (all, tap, stroke, stroke_proximal, stroke_distal)
4. **`contact_area` dismissal**: add a dedicated task or sub-option in `stimulus_response_tuning` for area vs RF metric scatter

---

## New Infrastructure Required

### 1. Temporal Alignment & Response Dynamics Module
- **Location**: `code/src/analysis/receptive_field_mapping/pipelines/rf_response_dynamics_pipeline.py`
- **Renderer**: `code/src/analysis/receptive_field_mapping/rendering/rf_response_dynamics_renderer.py`
- Reads `TouchEvent.frame_iff` from playback data, aligns to contact onset, groups by condition, renders overlays

### 2. Multi-Fit Tuning Curve Extension
- **Modify**: `rf_response_tuning_pipeline.py` and `rf_response_tuning_renderer.py`
- Accept list of fit degrees, render all on same axes, report comparison table

### 3. Cross-Neuron Comparison Renderer
- **Location**: `code/src/analysis/receptive_field_mapping/rendering/rf_cross_neuron_renderer.py`
- Grouped bar/strip charts with Okabe-Ito colors, statistical annotations
- Reusable across both families at every hierarchy level

---

## Estimated Figure Count (generate all, curate later)

| Category | Figures |
|----------|---------|
| Family 1 RF: L0 overview | 1 |
| Family 1 RF: L1 tap vs stroke | 1–2 |
| Family 1 RF: L2 distal vs proximal | 1–2 |
| Family 1 RF: L3 velocity/depth grid | 2–4 |
| Family 1 RF: L3 tuning curves | 2–3 |
| Family 1 RF: Area dismissal | 1 |
| Family 2 Sensitivity: L0–L2 summary metrics | 3 |
| Family 2 Sensitivity: L3 tuning curves | 2–3 |
| Family 2 Dynamics: L1–L3 IFF traces | 3–4 |
| Cross-neuron comparison panels | embedded in above |
| **Supplementary: per-neuron detailed** | 11+ |
| **Total estimate** | ~16–22 main + supplementary |

---

## Verification

- Run `stimulus_response_tuning` with multi-fit degrees on one neuron; confirm all fits render on same axes
- Run `cross_render_grid_metrics` with `velocity_depth_2d` for all gesture subsets; confirm per-gesture heatmaps
- Build temporal alignment prototype on one neuron's `TouchEvent.frame_iff` data; confirm traces align sensibly
- Render one cross-neuron grouped bar chart with Okabe-Ito colors; confirm type grouping and stats annotations
- Compare area vs RF metric scatter for all neurons; confirm no significant correlation

---

## Open Questions

1. **Centroid shift units**: Currently in UV units — do we need mm conversion before publication?
2. **Temporal window for dynamics**: Fixed window (e.g., 0–2s from contact onset) or variable per touch duration?
3. **Fit selection policy**: Per-neuron × per-feature (most flexible) or one fit type per feature across all neurons (simpler to interpret)?
4. **Depth as deactivatable dimension**: Should the DAG option allow toggling depth on/off in L3 analysis, similar to the area dismissal pattern?
