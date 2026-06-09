# Recommended Article Structure

**Option C: Combined spatial + stimulus paper** (preferred).
Fallback: Option D (two papers) if reviewers request splitting.

---

## Title Direction

"Response field mapping of human low-threshold mechanoreceptors during
naturalistic social touch using depth-camera videography and SLIM UV
surface parameterization"

---

## Proposed Structure

### 1. Introduction

- Classical RF mapping (probe) vs. response field (naturalistic touch)
- Formally define "response field" and justify the distinction
- Gap: spatial RF properties under naturalistic touch are unexplored
- Gap: Aβ hairy-skin stimulus tuning with naturalistic stimuli is sparse
- Aim: characterize spatial response fields and stimulus tuning across 5
  afferent subtypes using depth-camera videography + SLIM UV parameterization

### 2. Methods

- Experimental setup: microneurography + Azure Kinect depth on dorsal forearm
- SLIM (Symmetric-Dirichlet) UV parameterization: quasi-isometric
  (angle-and-area preserving) 2D flattening of 3D forearm mesh
- Inflection-contour boundary extraction (novel contribution)
- Per-touch feature extraction: velocity, contact area, pressure, depth
- Tuning curve computation + instruction-level analysis
- Unit classification: conduction velocity + response properties → subtype

This section can stand alone as a methodological contribution.

### 3. Results — Spatial Response Fields

- Per-subtype response field areas, shapes, centroids (Table + Figure)
- Response field areas ~17–210× larger than classical probe RFs — explain why
- Circularity and PCA ellipse per subtype and gesture type
- Gesture-dependent centroid shifts (proximal vs. distal, mean 8.6 ± 7.0 UV units)
- Cross-session boundary overlays per subtype
- Comparison table: this study vs. Vallbo 1995 / Bai et al. 2015 (mouse)

### 4. Results — Stimulus Tuning

- Velocity tuning curves per subtype (CT inverted-U vs. Aβ monotonic)
- Contact area tuning per subtype (novel — underexplored parameter)
- Instruction-level effects (speed/force/area metadata → IFF)
- 1D velocity-parameterized response fields (cross_render_sessions)
- Spike count tuning (complementary to IFF)

### 5. Discussion

- vs. Gerling 2025: complementary (discrimination vs. spatial organization)
- vs. Vallbo 1995 / Bai et al. 2015 (mouse): response field vs. classical RF
- Response field as ecologically valid measure of skin sensitivity
- Subtype differences in spatial organization and tuning
- Methodological contribution: SLIM UV + inflection contour
- Limitations: n=2 per subtype, pressure calibration, ST15-01 outlier

---

## Suggested Figures

| Figure | Content | Data source |
|--------|---------|-------------|
| Fig 1 | Methods overview: Kinect → mesh → SLIM UV → inflection contour | Pipeline diagrams |
| Fig 2 | Per-subtype response field overlays + boundary metrics | `spatial_extract_boundaries` |
| Fig 3 | Gesture-dependent centroid shifts (scatter plot) | `spatial_compare_rf_centers` |
| Fig 4 | Tuning curves per subtype: velocity, contact area | `stimulus_iff_tuning_curves` |
| Fig 5 | Instruction-level bar charts | `stimulus_iff_instruction_tuning` |
| Fig 6 | 1D velocity-parameterized RF heatmap across sessions | `cross_render_sessions` |
| Fig S1 | All individual session heatmaps per gesture (supplementary) | `spatial_extract_boundaries` |
| Fig S2 | Radar plots per subtype (supplementary) | `stimulus_render_radar` |

---

## Suggested Tables

| Table | Content |
|-------|---------|
| Table 1 | Session inventory: subtype, conduction velocity, n touches |
| Table 2 | Response field metrics per subtype: area, perimeter, circularity, PCA |
| Table 3 | This study vs. literature: RF sizes by subtype |
| Table 4 | Centroid shift per subtype: proximal-distal distance |
| Table 5 | Tuning curve parameters: R², slope, optimal feature value per subtype |

---

## Core References

| Reference | Year | Journal | Comparison point |
|-----------|------|---------|-----------------|
| Vallbo, Olausson, Wessberg, Kakuda | 1995 | J Physiol | Hairy Aβ RF sizes |
| Wessberg, Olausson, Fernström, Vallbo | 2003 | J Neurophysiol 89:1567–1575 | CT RF sizes |
| Johansson & Vallbo | 1979 | J Physiol 286:283–300 | Glabrous Aβ RF sizes (mm²) |
| Johansson & Vallbo | 1983 | Trends Neurosci | Glabrous tactile sensibility review |
| Löken et al. | 2009 | Nat Neurosci | CT velocity inverted-U |
| Ackerley et al. | 2014 | J Neurosci | CT velocity × temperature |
| Bai et al. (Ginty lab) | 2015 | Cell 163(7):1783–1795 | Aβ field-LTMR discovery (mouse) |
| Kuehn et al. (Ginty lab) | 2019 | PNAS 116(19):9168–9177 | LTMR tiling/somatotopy (mouse) |
| Hauser & Gerling | 2019 | IEEE WHC | Naturalistic touch tracking |
| Ebbesen & Froemke | 2022 | Nat Commun | 3D videography RF mapping (mice) |
| Xu, Hauser, Nagi et al. | 2025 | bioRxiv | Aβ social touch discrimination |
| Marshall & McGlone | 2020 | Neuroscience Insights (vol 15) | CT review |
| Abraira & Ginty | 2013 | Neuron | Sensory neurons of touch review |

---

## Target Journals (by scope)

| Journal | Fit | Notes |
|---------|-----|-------|
| Journal of Neurophysiology | Strong | Methods + physiology; Vallbo/Wessberg published here |
| eLife | Strong | Open access; novel methods + biological insight |
| Journal of Neuroscience | Moderate | Higher bar; needs strong mechanistic insight |
| PNAS | Moderate | Kuehn et al. 2019 LTMR-tiling paper published here; needs broad significance |
| J Neurosci Methods | Fallback | If framed as methods paper only |
