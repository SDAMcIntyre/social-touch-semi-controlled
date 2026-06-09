# Novelty Assessment

What is novel, what extends existing work, what replicates known findings.

---

## High Novelty (no precedent)

### 1. Inflection-contour response field boundaries

Laplacian zero-crossing on Gaussian-smoothed population heatmap, computed on
a UV-parameterized skin surface. Produces quantitative boundary polygons with
area, perimeter, circularity, and PCA ellipse metrics. No published method
defines tactile afferent RF boundaries this way.

### 2. SLIM (Symmetric-Dirichlet) UV mapping for skin RF analysis

SLIM (Symmetric-Dirichlet) UV parameterization applied to the forearm mesh
for 2D RF rendering and boundary extraction. This is a quasi-isometric
(angle-and-area preserving) flattening, not a strictly conformal map.
Literature search found no published applications of this surface
parameterization to somatosensory receptor mapping on body skin surfaces.

### 3. Gesture-dependent response field shift

Mean 8.6 ± 7.0 UV-unit centroid migration between proximal and distal stroke
directions (range 0.6–23.5 UV units, excluding outlier; SLIM UV parameter
space, not physical mm). Published data compare neural firing
rates across gestures (Gerling 2025), but not how the spatial response field
itself changes shape and location with stroke direction.

### 4. Per-subtype response field shape metrics

Circularity, PCA aspect ratio, PCA orientation, and area reported per
afferent subtype (SA-I, SA-II, HFA, Field-LTMR, CT) on hairy forearm skin.
These quantitative shape descriptors have not been reported for any afferent
subtype in published literature.

### 5. "Response field" definition for peripheral afferents

No established formal distinction between "response field" and "receptive
field" exists in somatosensory peripheral literature. This paper can formally
define "response field" as the skin area where naturalistic touch elicits a
neural response — distinct from the classical "receptive field" mapped with
punctate monofilament probes.

---

## Moderate Novelty (extends sparse literature)

### 6. Aβ stimulus tuning on hairy forearm with naturalistic touch

Most published Aβ velocity/force tuning data is from glabrous skin or uses
controlled mechanical stimulators. Naturalistic-touch tuning curves on hairy
forearm are sparse — only the Gerling lab has published in this space, and
their focus is temporal discrimination, not tuning curves per se.

### 7. Contact area as a tuning parameter

Contact area effects on tactile afferent firing are acknowledged as
underexplored in the literature for both CT and Aβ. This study has per-touch
contact area measurements (3–100 mm²) and relates them to IFF via tuning
curves — data that does not exist for specific Aβ subtypes on hairy skin.

### 8. 1D velocity-parameterized response fields

Via `cross_render_sessions`: partitioning touches by velocity bin and
computing per-bin population response fields. Shows how the spatial response
field changes as a function of stimulus velocity. No published equivalent
for peripheral afferents.

---

## Replication with New Method (valuable but not primary novelty)

### 9. CT velocity tuning (inverted-U)

Well established by Löken 2009, Ackerley 2014. This study has only 2 CT
units. Confirming the inverted-U with naturalistic delivery is valuable but
cannot be the main finding.

### 10. Aβ monotonic velocity response

Expected pattern (Vallbo 1995). Confirmed with naturalistic touch stimulus
across multiple Aβ subtypes. Adds ecological validity to the established
finding.

---

## Positioning vs. Closest Competitor

**Gerling lab (Virginia)** — Xu, Hauser, Nagi et al. 2025:

| Dimension | Gerling 2025 | This study |
|-----------|-------------|------------|
| Question | Can Aβ afferents discriminate social gestures? | Where on the skin does a unit respond, and how does this change with gesture? |
| Output | Classification accuracy, temporal discrimination | Spatial response field boundaries, shape metrics, centroid shifts |
| Method | Leap Motion joint tracking | Kinect depth → 3D mesh → SLIM UV |
| Analysis | Firing rate patterns over time | Spatial heatmaps + inflection contours |
| Subtypes | SA-II + HFA best discriminators | Per-subtype spatial characterization |

**Verdict: complementary, not competing.** They answer "can it discriminate?"
We answer "where on the skin, and what shape?"
