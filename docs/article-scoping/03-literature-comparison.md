# Literature Comparison

Comparison data from the existing literature on tactile afferent RF mapping
and stimulus-response tuning, organized for direct comparison with the
pipeline output data.

**See also:**
- [RF Classification](references/classification-receptive-field.md) —
  species-tagged classification of all references by whether they report
  receptive field data
- [Stimulus-Tuning Classification](references/classification-stimulus-tuning.md) —
  classification by stimulus domain (velocity, temperature, contact area,
  gesture, force, pleasantness) with a cross-reference summary matrix

---

## Response Field Sizes — Classical vs. This Study

| Afferent type | Site | Classical RF (probe) | This study (naturalistic) | Factor |
|---------------|------|---------------------|--------------------|--------|
| SA-I | Forearm | ~4 mm² (2–4 spots) [verify in full text] | 288–838 mm² | ~70–210× |
| SA-II | Forearm | ~36 mm² (diffuse) [verify in full text] | 609–614 mm² | ~17× |
| HFA | Forearm | multiple hair follicles | 510–734 mm² | — |
| Field-LTMR | Forearm | 3–4 mm² (spotty, 20–180 endings) | 534–599 mm² | ~134–200× |
| CT (C-LTMR) | Forearm | 1–35 mm² (mean 14.7 ± 10.5 [verify in full text]) | 632–662 mm² | ~19–45× |
| SA-I | Hand (glabrous) | 11.0 mm² (Johansson & Vallbo 1980) | — | — |
| SA-II | Hand (glabrous) | 58.9 mm² (Johansson & Vallbo 1980) | — | — |
| FA-I/RA | Hand (glabrous) | 12.6 mm² (Johansson & Vallbo 1980) | — | — |
| FA-II/PC | Hand (glabrous) | 101.3 mm² (Johansson & Vallbo 1980) | — | — |

**Why the ~17–210× difference is expected:**
- Classical RFs: punctate 5 mN monofilament, single static contact point
- This study: full hand contact (3–100 mm² area), dynamic stroking/tapping,
  aggregated across many touches
- The response field captures "where on the forearm does this unit respond
  to naturalistic touch" — a fundamentally different and more ecologically
  valid question

**Population overlap context (Johansson & Vallbo 1980):** Even with
classical punctate probes on glabrous skin, a single point stimulus on the
fingertip co-activates ~34 RA + SA-I fields spread over ~8 mm. Mean
overlap: ~20 fields at fingertip, ~9 at finger, ~6 at palm (all from their
population model of glabrous hand, not hairy skin). *Our interpretation:*
if punctate probes already recruit dozens of units, large-area naturalistic
contact on hairy skin should recruit far more, and the resulting response
fields likely reflect population-level recruitment rather than single-unit
RF expansion.

---

## Stimulus-Response Tuning — Literature Baseline

| Parameter | CT (literature) | Aβ (literature) | This study |
|-----------|----------------|-----------------|------------|
| Velocity tuning | Inverted-U, peak 1–10 cm/s | Monotonic increase | 2.7–10+ cm/s |
| Optimal velocity | ~3 cm/s (Löken 2009) | No peak (linear ↑) | TBD from curves |
| Force threshold | 0.3–2.5 mN (Wessberg 2003) | 0.3–40 mN [source to confirm] | Pressure uncalibrated |
| Contact area | Underexplored | Underexplored | 3–100 mm² |
| Temperature | Optimal 32°C (Ackerley 2014) | Insensitive | Not measured |
| Gesture discrim. | Not studied at RF level | SA-II + HFA best (Gerling 2025) | Tap vs. stroke |

> Note: the Aβ "0.3–40 mN" force threshold is unsourced/unverified and is
> flagged [source to confirm]. The Gerling/Xu preprint reports Aβ von Frey
> thresholds ≤ ~1.6 mN, which does not match the "0.3–40 mN" range.

### Per-subtype velocity tuning (from literature)

- **CT:** Inverted-U curve peaking at ~3 cm/s; decreases <0.3 and >30 cm/s.
  Correlates with pleasantness ratings (r ≈ 0.7–0.8 [verify in full text]).
  Löken et al. 2009.
- **Aβ (all subtypes):** Monotonic linear increase from 0.3–30 cm/s.
  Power-function relationship between velocity and firing rate. Vallbo 1995
  [verify in full text — velocity tuning may belong to a different
  Vallbo/Edin velocity paper].
- **SA-II:** Responds to indentation AND skin stretch/shear. Discriminates
  tap vs. stroke at perceptual accuracy. Gerling 2025.
- **HFA (hair follicle):** Rapidly adapting; fires during dynamic phase of
  hair deflection. Discriminates social gestures. Gerling 2025.
- **Field-LTMR:** Highly sensitive to gentle stroking; unresponsive to hair
  deflection. Hotspot–insensitive-patch organization. Bai et al. 2015 (mouse).

### Contact area effects (from literature)

- 13 dB threshold reduction when contact area increases from 0.05 to 1.5 cm²
  (probability summation + neural integration). Gescheider et al. 2005.
- Contact force encoded in firing rates; texture in spatial distribution and
  spike timing. Saal et al. 2017.
- Population overlap model: a single point stimulus on the fingertip
  co-activates ~34 RA + SA-I fields; disc stimuli <2 mm activate similar
  numbers. Johansson & Vallbo 1980.
- Contact area effects underexplored for specific Aβ subtypes on hairy skin.

---

## Methodological Landscape

| Approach | Who | Sensors | RF mapping? | Boundary extraction? |
|----------|-----|---------|-------------|---------------------|
| 5 mN scanning probe | Vallbo et al. 1995 | Monofilament | Yes (punctate) | Manual |
| Lightweight probe | Wessberg et al. 2003 | 0.23mm tracks | Yes (CT) | Manual |
| Leap Motion + microneurography | Gerling lab 2019 | Stereo IR | No (firing rates) | No |
| 3D videography + deep learning | Ebbesen & Froemke 2022 | Depth cameras | Yes (mice, social) | No |
| **Kinect + SLIM UV + inflection contour** | **This study** | **Azure Kinect depth** | **Yes (naturalistic)** | **Automated, quantitative** |

### Key differentiators of this study's approach

1. **Depth-camera-based 3D mesh** — not joint tracking (Leap Motion) or
   manual probe positioning
2. **SLIM (Symmetric-Dirichlet) UV parameterization** — quasi-isometric
   (angle-and-area preserving) surface flattening; no published application to
   somatosensory RF mapping
3. **Automated boundary extraction** — Laplacian-based inflection contour
   with quantitative shape metrics (area, perimeter, circularity, PCA)
4. **Per-gesture response fields** — computed separately for tap, stroke
   proximal, stroke distal; enables within-unit comparison

---

## Terminology: "Response Field" vs. "Receptive Field"

**No established formal distinction** exists in somatosensory peripheral
afferent literature. The term is used in:

- **Motor neuroscience:** response field = movement space (not sensory space)
- **Audition:** spectro-temporal receptive field (STRF)
- **Central somatosensory:** "population receptive field" (pRF) used for
  cortical mapping (Bayesian pRF models)

**Opportunity:** This paper could formally define "response field" for
peripheral afferents as the skin area where naturalistic touch elicits a
response, distinct from the classical "receptive field" mapped with punctate
monofilament probes.

---

## Reference Table

### Spatial RF data

| Reference | Year | Journal | Key finding | Download |
|-----------|------|---------|-------------|----------|
| Vallbo, Olausson, Wessberg, Kakuda | 1995 | J Physiol 483:783–95 | Hairy skin Aβ RF sizes: SA-I ~4 mm², SA-II ~36 mm² [verify in full text], 5 unit types | [PMC (free)](https://pmc.ncbi.nlm.nih.gov/articles/PMC1157818/) |
| Wessberg, Olausson, Fernström, Vallbo | 2003 | J Neurophysiol 89:1567–1575 | CT RF sizes: 1–35 mm² (mean 14.7 [verify in full text]), 1–9 responsive hotspots | [J Neurophysiol (free)](https://journals.physiology.org/doi/full/10.1152/jn.00256.2002) |
| Johansson & Vallbo | 1979 | J Physiol 286:283–300 | Glabrous Aβ RF sizes: SA-I ~11, SA-II ~59, FA-I ~12.6, FA-II ~101 mm² | [Wiley](https://physoc.onlinelibrary.wiley.com/doi/abs/10.1113/jphysiol.1979.sp012619) |
| Johansson & Vallbo | 1980 | Brain Res 184:353–366 | Glabrous Aβ RF sizes (255 units): RA 12.6, SA-I 11.0, SA-II 58.9, PC 101.3 mm²; population overlap model | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/0006899380908045) · [PubMed](https://pubmed.ncbi.nlm.nih.gov/7353161/) |
| Johansson & Vallbo | 1983 | Trends Neurosci 6:27–31 | Glabrous tactile sensibility review (general framing) | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/0166223683900115) · [Free PDF (NYU)](https://www.cns.nyu.edu/~david/courses/sm12/Readings/Johansson1983.pdf) |
| Bai et al. (Ginty lab) | 2015 | Cell 163(7):1783–1795 | Aβ field-LTMR (mouse): 20–180 circumferential endings, spotty hotspot organization | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0092867415016220) · [PubMed](https://pubmed.ncbi.nlm.nih.gov/26687362/) |
| Kuehn et al. (Ginty lab) | 2019 | PNAS 116(19):9168–9177 | LTMR tiling and somatotopic alignment; self-avoidance (mouse) | [PNAS (free)](https://www.pnas.org/content/116/19/9168) · [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6511030/) |
| Ebbesen & Froemke | 2022 | Nat Commun 13:593 | 3D videography social RF mapping in mice | [Nature (open access)](https://www.nature.com/articles/s41467-022-28153-7) · [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8807631/) |

### Stimulus-response tuning

| Reference | Year | Journal | Key finding | Download |
|-----------|------|---------|-------------|----------|
| Löken, Wessberg, Morrison, McGlone, Olausson | 2009 | Nat Neurosci 12:547–8 | CT velocity inverted-U (peak ~3 cm/s) + pleasantness correlation | [Nature](https://www.nature.com/articles/nn.2312) · [PubMed](https://pubmed.ncbi.nlm.nih.gov/19363489/) |
| Ackerley et al. | 2014 | J Neurosci 34:2879–83 | CT velocity × temperature: optimal at 1–10 cm/s, 32°C | [J Neurosci (free)](https://www.jneurosci.org/content/34/8/2879) · [PubMed](https://pubmed.ncbi.nlm.nih.gov/24553929/) |
| Vallbo, Olausson, Wessberg | 1999 | J Neurophysiol 81:2753–63 | CT = second tactile system in hairy skin | [J Neurophysiol (free)](https://journals.physiology.org/doi/full/10.1152/jn.1999.81.6.2753) · [PubMed](https://pubmed.ncbi.nlm.nih.gov/10368395/) |
| Xu, Hauser, Nagi et al. | 2025 | bioRxiv | SA-II + HFA discriminate social gestures at 3–4 s timescale | [bioRxiv (free)](https://www.biorxiv.org/content/10.1101/2023.07.22.549516v1) · [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11952579/) |
| Hauser & Gerling | 2019 | IEEE WHC 2019:592–7 | Human-to-human touch tracking + afferent responses | [ResearchGate](https://www.researchgate.net/publication/334042081_From_Human-to-Human_Touch_to_Peripheral_Nerve_Responses) |

### Contact area and texture (cited in-text)

| Reference | Year | Journal | Key finding | Download |
|-----------|------|---------|-------------|----------|
| Gescheider, Bolanowski, Pope, Verrillo | 2005 | Somatosens Mot Res 22(4):255–68 | 13 dB threshold reduction with contact area increase; probability summation + neural integration | [Taylor & Francis](https://www.tandfonline.com/doi/abs/10.1080/08990220500420236) · [PubMed](https://pubmed.ncbi.nlm.nih.gov/16503579/) |
| Saal, Delhaye, Rayhaun, Bensmaia | 2017 | PNAS 114(28):E5693–E5702 | TouchSim: contact force in firing rates, texture in spike timing | [PNAS (free)](https://www.pnas.org/content/early/2017/06/20/1704856114) · [PubMed](https://pubmed.ncbi.nlm.nih.gov/28652360/) |

### Reviews and context

| Reference | Year | Journal | Scope | Download |
|-----------|------|---------|-------|----------|
| Marshall & McGlone | 2020 | Neuroscience Insights (vol 15) | CT: Enigmatic Spinal Pathway review | [SAGE (open access)](https://journals.sagepub.com/doi/10.1177/2633105520925072) · [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7265072/) |
| Morrison | 2016 | Springer (book chapter) | CT Afferent-Mediated Affective Touch: Brain Networks and Functional Hypotheses | [Springer](https://link.springer.com/chapter/10.1007/978-1-4939-6418-5_12) |
| Abraira & Ginty | 2013 | Neuron 79(4):618–39 | Sensory neurons of touch (comprehensive review) | [Cell Press (free)](https://www.cell.com/neuron/fulltext/S0896-6273(13)00710-1) · [PubMed](https://pubmed.ncbi.nlm.nih.gov/23972592/) |
