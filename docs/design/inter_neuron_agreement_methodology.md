# Inter-Sensor Agreement and Bias Detection Under Process Variability

## Problem Statement

You have an observational dataset where a single source produces **processes** (observable phenomena evolving over time). Each process instance has its outcome recorded by one of several **sensor types**, with one sensor per process instance. You want to know: **"Given similar processes, do different sensors report different measurements?"**

Since the process varies from instance to instance, any naive comparison of sensor readings across the full dataset would confound genuine sensor differences with differences in the underlying process. The process must therefore be controlled for. This is fundamentally an **inter-sensor agreement and bias detection** problem, where the process acts as a confounding variable.

---

## Pipeline

### Stage 1 — Feature Extraction

Each process instance must be encoded as a numerical representation suitable for distance computation. Critically, this representation must use **only process characteristics**, strictly excluding any measurement or outcome data. Including sensor readings at this stage would contaminate the stratification and invalidate all downstream comparisons.

**Scalar features** — Summary statistics extracted from each process instance: duration, peak amplitude, rate of change, energy, spatial extent, or any domain-relevant parameter. These form a fixed-length vector per instance.

**Temporal features** — When the shape or dynamics of the process matter (two instances may share the same summary statistics but follow very different paths), the raw time series must be preserved or compressed. Options include Fourier coefficients, wavelet decompositions, or simply retaining the resampled series for direct comparison using time-series distance metrics.

**Standardization** — All features must be brought to a common scale before distance computation. Center-reduce (z-score) is the minimum. If features have different units or dynamic ranges, failing to standardize will cause high-magnitude features to dominate the distance metric and effectively silence the others.

**Composite distance** — When mixing scalar and temporal features, a weighted combination of distances is needed. For instance, Euclidean distance on the scalar feature vector combined with Dynamic Time Warping on the temporal profile, with an explicit weighting parameter controlling the relative importance. This weight is a design decision that should be justified by domain knowledge or sensitivity analysis.

The output of this stage is either a feature matrix (N instances × D features) or a precomputed pairwise distance matrix (N × N).

---

### Stage 2 — Clustering

The goal is to partition all process instances into **strata** of similar processes, subject to a **coverage constraint**: each stratum must contain enough representatives from multiple sensor types to support statistical comparison.

**Hierarchical agglomerative clustering** is the chosen method because it produces a full dendrogram — a nested tree of merges from individual instances up to a single root cluster — which can be cut at any level without re-running the algorithm. This flexibility is essential for the adaptive stratification that follows.

**Linkage method** — Ward's method is recommended as the default. It minimizes within-cluster variance at each merge step, producing compact, roughly spherical clusters that are well-suited to stratification. Single linkage tends to produce elongated chains where instances at opposite ends of the same cluster may be very dissimilar, which undermines the "similar process" guarantee. Complete linkage produces compact clusters but is sensitive to outliers.

**Adaptive non-uniform cut** — Rather than cutting the dendrogram at a single height (which forces all strata to the same granularity), the tree is cut at variable depths using a recursive algorithm driven by the coverage constraint.

The algorithm proceeds as follows. Starting from the root, at each node, evaluate whether the subtree satisfies the coverage constraint: at least *N* instances per sensor type, for at least *M* sensor types present. If both child branches independently satisfy the constraint, descend into both — splitting improves homogeneity without sacrificing coverage. If splitting would break the constraint in either branch, stop descending and declare the current node as a stratum. If the current node itself does not satisfy the constraint, it is marked as non-exploitable and its instances are set aside.

This produces strata of **variable granularity**: fine-grained in regions of the feature space where data is dense and multiple sensors are well-represented, coarser in sparser regions. The partition emerges from the data and the constraint, with no arbitrary number of clusters to select.

**Dispersion tagging** — Each stratum is tagged with an internal dispersion measure (within-cluster variance, maximum pairwise distance, or diameter). This quantifies how loosely "similar" is defined within that stratum and will be used downstream to weight the confidence of comparisons.

The output of this stage is a set of strata, each containing a list of process instances, their sensor labels, and a dispersion score.

---

### Stage 3 — Comparing

Within each valid stratum, sensor measurements are compared. Since the processes within a stratum are approximately equivalent, any systematic difference in measurements is attributable to the sensor itself.

**Bias detection** — Compare the central tendency of measurements across sensor types. If sensor A consistently reports higher values than sensor B within the same stratum, this indicates a calibration offset. ANOVA tests whether the means differ across sensors; pairwise post-hoc tests (Tukey HSD, Dunn's test) identify which specific sensor pairs diverge.

**Precision comparison** — Even when sensors agree on average, their variability may differ. Levene's test or Bartlett's test assesses whether measurement variance is homogeneous across sensors. A sensor with higher variance is less precise, even if unbiased.

**Distribution shape** — Beyond mean and variance, the full distribution of measurements may differ. Kolmogorov-Smirnov or Anderson-Darling tests detect distributional differences that ANOVA would miss, such as skewness, heavy tails, or multimodality in one sensor's readings.

**Regime-dependent behavior** — The most valuable output of this architecture is that comparisons are stratified. Sensors may agree perfectly in one process regime and diverge in another. Reporting results per stratum — with strata ordered or grouped by their position in the process feature space — reveals where sensors are interchangeable and where they are not.

**Dispersion-weighted synthesis** — When aggregating findings across strata, the dispersion score from Stage 2 serves as a confidence weight. A significant sensor difference found in a tight, homogeneous stratum (low dispersion) carries more evidential weight than one found in a broad stratum where "similar process" is loosely defined. This can be formalized as a weighted meta-analysis across strata, or simply reported descriptively with dispersion as context.

**Global summary** — The final output is a matrix of sensor-pair comparisons across process regimes, answering questions such as: "Sensors A and B agree everywhere. Sensor C is biased high in regime X but not in regime Y. Sensor D is noisier than all others regardless of regime."

---

## Design Parameters

The pipeline has a small number of user-defined parameters that control its behavior:

- **Feature set and distance metric** — what defines "similar process"
- **Linkage method** — how clusters are formed (Ward's recommended)
- **N** — minimum instances per sensor per stratum (controls statistical power)
- **M** — minimum number of sensor types per stratum (controls comparison breadth)
- **Dispersion weighting scheme** — how to aggregate findings across strata of varying quality

All of these should be documented and, ideally, subjected to sensitivity analysis to verify that conclusions are robust to reasonable variations.
