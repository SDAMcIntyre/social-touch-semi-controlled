# Time-Series Clustering Pipeline — Guidelines

## Purpose

This document defines a clear, segmented pipeline for multi-timeseries clustering. The main design goal is to separate what is currently an overloaded "feature extraction" step into well-defined stages, so that each stage has a single responsibility and a well-defined output type that constrains the downstream choices.

## Reference framing

The structure below follows the consensus framing from the time-series clustering literature:

- Räsänen & Kolehmainen (2009), Wang, Smith & Hyndman (2006) — four-step feature-based clustering framework (input preparation → feature extraction → feature reduction → clustering/evaluation).
- Decade review of time-series clustering (arXiv 2412.20582, 2024) — the three pillars: *representation, dissimilarity measure, clustering procedure*, treated as coupled choices rather than independent stages.
- Kreienkamp et al. (2024, *Multivariate Behavioral Research*) — applied restatement of the same four-step structure.

## Pipeline overview

The pipeline is split into five stages. Stages 2 and 4 are internally coupled: the type of object produced by Stage 2 constrains which algorithms are valid in Stage 4.

```
Stage 1: Data loading & preparation
Stage 2: Representation
  ├── 2a. Series-level transformations   (series → series)
  └── 2b. Feature characterization        (series → scalar vector)
Stage 3: Feature reduction / selection   (optional)
Stage 4: Clustering                       (algorithm choice constrained by 2a vs 2b)
Stage 5: Cluster analysis & evaluation
```

---

## Stage 1 — Data loading & preparation

**Responsibility:** ingest the raw multi-timeseries data and produce clean, aligned, comparable series. No features are produced at this stage.

**Typical operations:**

- Loading and parsing the raw multi-timeseries source.
- Temporal alignment across series and across channels.
- Resampling (up- or down-sampling) to a common rate.
- Missing-value imputation (linear, spline, time-based interpolation, or model-based).
- Outlier handling.
- Denoising / smoothing (moving average, low-pass filter, wavelet denoising).
- Normalization or scaling when globally appropriate (note: z-normalization per series is usually deferred to Stage 2a, because it is a representation choice, not a data-quality choice).

**Output:** a cleaned set of multi-timeseries, same type as the input but reliable.

**What does *not* belong here:** anything that changes the semantic content of a series (combining channels, Fourier transforms, deriving indicators). Those are Stage 2a operations.

---

## Stage 2 — Representation

This is the stage most commonly conflated. It decides *what kind of object represents each entity* going into clustering. Split it explicitly into two parallel sub-stages. Either sub-stage may be used alone, both may be used, and 2b is commonly applied *on top of* 2a outputs.

### 2a. Series-level transformations (series → series)

**Responsibility:** transform sequences into other sequences that preserve temporal structure but are more informative, more compact, or more cluster-friendly.

**Output type:** one or more sequences per entity (still indexed by time or by frequency bin).

**Examples:**

- Keeping the raw series as-is (valid representation choice).
- Combining input channels into a derived series: ratio of two channels, linear projection (first PCA component over channels at each timestep), sum/difference, etc.
- Z-normalization per series (recommended before DTW or k-Shape).
- Windowing / subsequence extraction (sliding window, shapelets).
- Frequency-domain transforms: FFT, wavelet decomposition.
- Symbolic representations: SAX, PAA.
- Detrending, differencing.

### 2b. Feature characterization (series → scalar vector)

**Responsibility:** collapse each entity into a fixed-length vector of scalar descriptors. The time dimension is intentionally discarded.

**Output type:** one row of scalar features per entity (tabular data).

**Examples:**

- Basic statistical aggregates: mean, median, min, max, variance, standard deviation.
- Distributional descriptors: skewness, kurtosis, quantiles.
- Temporal descriptors: trend slope, seasonality strength, autocorrelation at lag k, partial autocorrelation.
- Spectral descriptors: dominant frequency, spectral entropy, band energy ratios.
- Complexity / nonlinearity measures: sample entropy, approximate entropy, Hurst exponent.
- Library-provided feature sets: **tsfresh**, **catch22**, **tsfeatures**, **hctsa**.

**Note on chaining 2a → 2b:** it is perfectly valid (and common) to apply 2b on a 2a output — e.g. the spectral entropy of a wavelet coefficient series, or the mean of a derived ratio channel. When this happens, be explicit about it in code and documentation: the final object is still a scalar vector (a 2b output), even though its inputs went through 2a.

---

## Stage 3 — Feature reduction / selection (optional)

**Responsibility:** reduce dimensionality or remove uninformative/redundant features before clustering. Applies mostly to 2b outputs, occasionally to 2a (e.g. keeping top-k Fourier coefficients).

**Typical operations:**

- Variance filtering, correlation-based pruning.
- Univariate feature scoring (mutual information, ANOVA F-score if any weak supervision is available).
- PCA, ICA, or UMAP on the feature matrix.
- Scaling (StandardScaler, RobustScaler, QuantileTransformer) before distance-based clustering.

**Output:** a lower-dimensional or better-conditioned version of the Stage 2 output, same type.

---

## Stage 4 — Clustering

**Responsibility:** group entities based on similarity. The algorithm choice is **constrained by** the Stage 2 output type. Picking an algorithm implies a representation, and vice versa — treat this coupling as explicit.

### Path A — Stage 2a output (sequences) → sequence-aware clusterers

Requires a distance/similarity function that operates on sequences.

**Examples:**

- **DTW k-means** — k-means variant using Dynamic Time Warping as the distance and DBA (DTW Barycenter Averaging) for centroid computation.
- **k-Shape** — uses a normalized cross-correlation–based shape distance; assumes z-normalized series.
- **KernelKMeans with GAK kernel** — Global Alignment Kernel, a differentiable DTW-like kernel.
- **k-medoids (PAM) with DTW** — avoids averaging, so it works with any pairwise distance.
- **Hierarchical clustering with DTW / LCSS / MSM / cross-correlation** as the linkage distance.
- **Density-based clustering (DBSCAN, OPTICS) with a sequence distance** — requires a precomputed distance matrix.

### Path B — Stage 2b output (scalar feature vectors) → vector-space clusterers

Standard tabular clustering on the feature matrix. The time dimension is gone; Euclidean-family distances are appropriate after scaling.

**Examples:**

- **k-means** on scaled features.
- **Gaussian Mixture Models (GMM)**.
- **Hierarchical / agglomerative clustering** with Euclidean, Manhattan, cosine, or Mahalanobis distance.
- **DBSCAN / HDBSCAN** on scaled features.
- **Spectral clustering** on a similarity graph built from feature-vector distances.
- **Self-Organizing Maps (SOM)** — used in the original Characteristic-Based Clustering work by Wang, Smith & Hyndman.

### Rule at the Stage 4 interface

A clustering routine should consume **either** a set of sequences with a sequence-appropriate metric, **or** a feature matrix with a vector metric — not an undocumented mix. If both sources of information are needed, use one of the two supported patterns:

1. **Ensemble / consensus clustering** — run Path A and Path B independently and combine the partitions (co-occurrence matrix, consensus clustering).
2. **Composite distance** — define an explicit weighted distance `d = α·d_shape + β·d_features` and document the weights. This is only honest if the weights are motivated.

---

## Stage 5 — Cluster analysis & evaluation

**Responsibility:** judge cluster quality and produce interpretable summaries.

**Internal metrics (no ground truth):**

- Silhouette score.
- Davies–Bouldin index.
- Calinski–Harabasz index.
- Within-cluster sum of squares / inertia (for k-means-like methods).
- Stability under resampling / bootstrap.

**Interpretation outputs:**

- Centroid or medoid of each cluster (a representative series for Path A, a representative feature vector for Path B).
- Per-cluster feature-distribution summaries.
- Visualizations: overlay plots of series per cluster, 2D projections (PCA/UMAP/t-SNE) of feature vectors coloured by cluster.

---

## Decision guide

Use this to pick the path before writing code:

- **Do you care about *trajectories looking alike* (rise/fall pattern, timing, shape)?** → Path A. Stage 2a transformations + sequence-aware clusterer. Skip 2b unless you are building a composite distance.
- **Do you care about *summary behaviour* (averages, volatility, trend direction, seasonality strength)?** → Path B. Stage 2b feature characterization + vector-space clusterer.
- **Do you care about both?** → Decide explicitly: ensemble two clusterings, or build a composite distance with documented weights.
- **Series are of different lengths or unaligned?** → Either Path B (length drops out when you aggregate), or Path A with DTW / LCSS (which tolerate length differences). Avoid Euclidean directly on raw sequences.
- **Very long series, performance-constrained?** → Path B is almost always faster; Path A with DTW is quadratic in series length unless bounded (Sakoe–Chiba, Itakura).

---

## Anti-patterns to avoid

- **One amorphous "feature extraction" step** that mixes series outputs and scalar outputs, then hands an ambiguous object to the clusterer. This is the original problem that motivated these guidelines.
- **Using Euclidean distance directly on raw unaligned sequences.** It is shift-sensitive and length-sensitive; use DTW, k-Shape, or go to Path B.
- **Skipping z-normalization before DTW / k-Shape.** Amplitude differences will dominate and shape similarity will be lost.
- **Computing normalization statistics on the full dataset before a train/test split** (when there is any downstream supervised evaluation). Fit scalers on training data only.
- **Reporting a clustering result without at least one internal metric and a stability check.**
- **Silently chaining 2a into 2b without documenting it.** Reviewers and future-you need to know the provenance of every feature.
