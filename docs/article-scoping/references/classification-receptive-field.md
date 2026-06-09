# Classification — Receptive Field

Classification of all reference articles by whether they report receptive
field (RF) data — size, shape, density, tiling, or boundary mapping — and
by species.

---

## Articles WITH Receptive Field Data

### Human

| Reference | Year | Journal | Site | Afferents | RF content |
|-----------|------|---------|------|-----------|------------|
| Vallbo, Olausson, Wessberg, Kakuda | 1995 | J Physiol 483:783–95 | Hairy forearm | SA-I, SA-II, HFA, Field, Pacinian | Scanning probe RF sizes: SA-I 11 mm², SA-II 1.4 mm², HFA 113 mm², Field 78 mm² |
| Wessberg, Olausson, Fernström, Vallbo | 2003 | J Neurophysiol 89:1567–75 | Hairy forearm | CT | 1–9 hotspots per unit, 1–35 mm² (mean 14.7 ± 10.5), force-dependent |
| Johansson & Vallbo | 1979 | J Physiol 286:283–300 | Glabrous hand | RA, SA-I, SA-II, PC | Density mapping: 334 units, proximo-distal gradient, ~17,000 units/hand |
| Johansson & Vallbo | 1980 | Brain Res 184:353–366 | Glabrous hand | RA, SA-I, SA-II, PC | RF sizes: RA 12.6 mm², SA-I 11.0 mm², SA-II 58.9 mm², PC 101.3 mm²; population tiling model |
| Johansson & Vallbo | 1983 | Trends Neurosci 6:27–32 | Glabrous hand | FA-I, FA-II, SA-I, SA-II | Review — 4-type classification with RF size/density framework |

### Mouse

| Reference | Year | Journal | Site | Afferents | RF content |
|-----------|------|---------|------|-----------|------------|
| Bai et al. (Ginty lab) | 2015 | Cell 163(7):1783–95 | Dorsal thigh hairy skin | Aβ Field-LTMR | 20–180 circumferential endings per unit, ~3.1 mm² spotty hotspot arrays (also confirmed in cat, dog, macaque) |
| Kuehn et al. (Ginty lab) | 2019 | PNAS 116(19):9168–77 | Trunk hairy skin | C-LTMR, Aδ-LTMR, Aβ Field/SA1/RA-LTMR | Tiling and self-avoidance across 5 subtypes; central projection somatotopy |
| Ebbesen & Froemke | 2022 | Nat Commun 13:593 | Full body (tracked) | Cortical (barrel cortex), not peripheral | 3D videography social RF mapping — cortical level, not peripheral afferents |

### Multi-species (review)

| Reference | Year | Journal | Species | RF content |
|-----------|------|---------|---------|------------|
| Abraira & Ginty | 2013 | Neuron 79(4):618–39 | Mouse (primary), human comparisons | LTMR taxonomy with RF properties; hair follicle innervation patterns; dorsal horn columnar processing |

---

## Articles WITHOUT Receptive Field Data

### Human — stimulus-response tuning

| Reference | Year | Journal | Topic |
|-----------|------|---------|-------|
| Löken et al. | 2009 | Nat Neurosci 12:547–8 | CT velocity inverted-U (peak ~3 cm/s) + pleasantness correlation |
| Ackerley et al. | 2014 | J Neurosci 34:2879–83 | CT velocity × temperature: optimal at 1–10 cm/s, 32°C |
| Vallbo, Olausson, Wessberg | 1999 | J Neurophysiol 81:2753–63 | CT characterisation: thresholds, velocity preference, fatigue (quotes 1–35 mm² RF range but does not map RFs) |
| Xu et al. | 2025 | bioRxiv | SA-II + HFA gesture discrimination via temporal coding (1D-CNN, SVM) |
| Hauser & Gerling | 2019 | IEEE WHC 2019:592–7 | Touch tracking + firing rates (CT, PC) — no RF mapping |
| Croy et al. | 2020 | Neuroscience 464:33–43 | Individual variability of pleasantness ratings across velocities — psychophysics only, no neural recording |
| Gescheider et al. | 2005 | Somatosens Mot Res 22(4):255–68 | Contact area vs. detection threshold (Pacinian channel) — psychophysics |

### Computational (monkey → human model)

| Reference | Year | Journal | Topic |
|-----------|------|---------|-------|
| Saal et al. | 2017 | PNAS 114(28):E5693–E5702 | TouchSim: skin mechanics + spiking model fit to macaque data, applied to human hand; uses RF density as input, does not report new RF measurements |

### Reviews (no original RF data)

| Reference | Year | Journal | Topic |
|-----------|------|---------|-------|
| Marshall & McGlone | 2020 | Neurosci Insights vol 15 | CT spinal pathway — anterolateral cordotomy preserves pleasantness curve |
| Morrison | 2016 | Springer (book chapter) | CT affective touch: posterior insula, social bonding, brain networks |

---

## Coverage check

21 PDFs in `references/` → 20 articles + 1 ZIP (Cell Press bundle, not yet
extracted). All 20 individual articles are classified above. The
`johansson-vallbo-1983-reference-abstracts.md` file contains abstracts of
papers cited *within* Johansson & Vallbo 1983 and is not a standalone article.
