# Classification — Stimulus-Response Tuning

Classification of all reference articles that report stimulus-response tuning
data — velocity curves, force thresholds, temperature modulation, contact
area effects, gesture discrimination — grouped by stimulus domain and species.

---

## Velocity Tuning

### Human — microneurography

| Reference | Year | Afferents | Velocities tested | Delivery | Key finding |
|-----------|------|-----------|-------------------|----------|-------------|
| Löken et al. | 2009 | CT (n=20), SAI (15), SAII (8), Hair (7) | 0.2, 3, 10, 30 cm/s | Robot brush, 0.2–0.4 N | CT: inverted-U peak ~1–3 cm/s (R²=0.71); Aβ: monotonic increase |
| Ackerley et al. | 2014 | CT (n=20), myelinated (n=8) | 0.3, 1, 3, 10, 30 cm/s | Mechano-thermal probe (Dancer Design), 0.4 N | CT: inverted-U peak 3 cm/s at all temperatures; hair units: monotonic, no temperature modulation |
| Vallbo et al. | 1999 | Low-threshold C (n=27), C nociceptors (n=11) | Moving stimuli (various) | Manual brush, plotter-controlled probe | CT vigorous to slow stroking (50–100 imp/s), poor to fast stimuli — opposite of Aβ |

### Human — psychophysics only

| Reference | Year | Participants | Velocities tested | Key finding |
|-----------|------|-------------|-------------------|-------------|
| Croy et al. | 2020 | 127 (reanalysis of 5 studies) | 0.3, 1, 3, 10, 30 cm/s | Group-level inverted-U pleasantness peak at 3 cm/s; only 42% of individuals show significant quadratic fit |

---

## Temperature Tuning

| Reference | Year | Afferents | Temperatures tested | Key finding |
|-----------|------|-----------|---------------------|-------------|
| Ackerley et al. | 2014 | CT (n=20), myelinated (n=8) | 18°C, 32°C, 42°C | Significant velocity × temperature interaction; CT firing strongest at warm (42°C) but pleasantness–firing correlation only at neutral (32°C); myelinated afferents insensitive to temperature |

---

## Contact Area

### Human — psychophysics

| Reference | Year | Site | Areas tested | Key finding |
|-----------|------|------|-------------|-------------|
| Gescheider et al. | 2005 | Thenar eminence (glabrous) | 0.05 vs 1.5 cm² | ~13 dB threshold reduction: ~8 dB neural integration + ~5 dB probability summation; ~3 dB per doubling of contactor size |

### Human — microneurography (naturalistic)

| Reference | Year | Afferents | Contact area range | Key finding |
|-----------|------|-----------|-------------------|-------------|
| Hauser & Gerling | 2019 | CT, PC (case study) | Varied by gesture (1-finger to whole-hand) | CT fired primarily during high shear velocity + large contact area (stroking); PC fired at each tap and at RF crossing |

---

## Gesture Discrimination

| Reference | Year | Afferents | Gestures tested | Delivery | Key finding |
|-----------|------|-----------|----------------|----------|-------------|
| Xu et al. | 2025 | SA-II (4), HFA (7), Field (5), CT (6), FA-II (5), MS (12) | 6 social expressions: attention, happiness, calming, love, sadness, gratitude | Human-to-human | SA-II + HFA: 70–80% accuracy (1D-CNN, SVM), matching human perception; CT, FA-II near chance; 3–4 s of firing sufficient |
| Hauser & Gerling | 2019 | CT, PC | 4 gestures: 1-finger tap, multi-finger tap, multi-finger stroke, whole-hand hold | Human-to-human | CT: selective for stroking at high shear velocity; PC: synchronous with each tap |

---

## Force / Threshold

| Reference | Year | Afferents | Force range | Key finding |
|-----------|------|-----------|------------|-------------|
| Vallbo et al. | 1999 | Low-threshold C (n=27), C nociceptors (n=11) | 0.1–80 mN (von Frey) | CT thresholds 0.3–2.5 mN (median 1.3 mN), bimodally separable from nociceptors (~40 mN); CT does not discriminate blunt vs. pin-prick |
| Löken et al. | 2009 | CT, SAI, SAII, Hair | 0.2–0.4 N (brush) | No significant force effect on firing rate at these levels |

---

## Adaptation / Fatigue

| Reference | Year | Afferents | Key finding |
|-----------|------|-----------|-------------|
| Vallbo et al. | 1999 | CT | Delayed acceleration response: after initial adaptation, CT resumes irregular firing peaking ~40 imp/s over ~40 s; prominent fatigue with repeated stimulation |
| Xu et al. | 2025 | SA-II, HFA | SA-II: sustained, slowly decaying firing during stroking/holding; HFA: onset/offset transients only |

---

## Pleasantness (Perception ↔ Neural Correlation)

| Reference | Year | Correlation | Key finding |
|-----------|------|-------------|-------------|
| Löken et al. | 2009 | CT firing ↔ pleasantness | R² = 0.70 (p = 0.0063); no correlation for Aβ; pleasantness lower on palm vs. forearm |
| Ackerley et al. | 2014 | CT firing ↔ pleasantness (temperature-dependent) | Significant correlation only at neutral 32°C (p = 0.003, R² = 0.96); not at cool or warm |
| Croy et al. | 2020 | Group-level only | Inverted-U robust at group level; high inter-individual variability (43% show no significant quadratic fit individually) |

---

## Computational Model

| Reference | Year | Species | Afferents modelled | Key finding |
|-----------|------|---------|-------------------|-------------|
| Saal et al. | 2017 | Macaque (fit) → Human (applied) | SA1, RA, PC (~12,500 tiled on hand) | Firing rates R² ~0.91; temporal precision 2–8 ms; force encoded in rates, texture in spike timing; SA2 excluded; glabrous only |

---

## Summary Matrix

Which stimulus parameters each paper covers:

| Reference | Velocity | Force | Temperature | Contact area | Gesture | Frequency | Pleasantness | Adaptation |
|-----------|:--------:|:-----:|:-----------:|:------------:|:-------:|:---------:|:------------:|:----------:|
| Löken 2009 | ✓ | · | · | · | · | · | ✓ | · |
| Ackerley 2014 | ✓ | · | ✓ | · | · | · | ✓ | · |
| Vallbo 1999 | ✓ | ✓ | · | ✓ | · | · | · | ✓ |
| Xu 2025 | · | · | · | · | ✓ | · | · | ✓ |
| Hauser & Gerling 2019 | ✓ | · | · | ✓ | ✓ | · | · | · |
| Croy 2020 | ✓ | · | · | · | · | · | ✓ | · |
| Gescheider 2005 | · | · | · | ✓ | · | ✓ | · | · |
| Saal 2017 | ✓ | ✓ | · | · | · | ✓ | · | ✓ |

All articles in this classification are **human** (or human-targeted model),
except Saal et al. 2017 which fits to **macaque** data and applies to human hand.
