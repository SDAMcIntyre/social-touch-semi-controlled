# Investigation: Perceptual Problems with the `jet` (Rainbow) Colormap

**Date:** 2026-06-08
**Status:** Research complete — replacement not yet implemented

---

## Context

RF population heatmaps (circular centroid, composite, standalone interpolated,
etc.) all use the `jet` colormap. A colleague flagged that this colormap type
can cause visual illusions and convey incorrect information. A literature review
was conducted across peer-reviewed articles, empirical user studies, journal
editorial policies, and authoritative software documentation to assess the
validity of this concern.

The review covered 15 sources spanning 5 research angles (perceptual science
foundations, landmark papers, journal editorial policies, colourblind
accessibility and empirical user studies, perceptually uniform alternatives).
67 distinct claims were extracted; the top 25 are presented below, organised by
topic with full attribution.

### Current usage in codebase

**File:** `code/src/analysis/receptive_field_mapping/rendering/rf_population_map_renderer.py`

| Function | Approx. line | Configurable? |
|----------|--------------|---------------|
| `render_population_rf_map()` | ~211 | Yes (`cmap` parameter, default `"jet"`) |
| `render_population_rf_standalone_interpolated()` | ~449 | No (hardcoded `'jet'`) |
| `render_population_rf_colorbar()` | ~491 | No (hardcoded `'jet'`) |
| `render_population_rf_circular_crop()` | ~645 | No (hardcoded `'jet'`) |
| `render_population_rf_composite()` | ~774, ~789, ~814 | No (hardcoded `'jet'`) |

**DAG config** (`configs/analyse_workflow_processing_dag.yaml`):
- `spatial_extract_boundaries` task (line ~195): `cmap: jet`
- `spatial_compare_rf_centers` task (line ~221): `cmap: jet`

---

## 1. Perceptual Distortion: False Boundaries, Hidden Features, and Attention Bias

### 1.1 The fundamental mechanism: non-monotonic lightness (L\*)

The human visual system is far more sensitive to changes in lightness than to
changes in hue. When a colormap is used to encode a continuous scalar field
(such as an RF population density), viewers primarily perceive the lightness
channel to read data values and detect gradients. The jet colormap has **widely
varying, non-monotonic lightness (L\* in CIELAB colour space)** throughout its
range. This makes it a poor choice for perceptual data representation because
equal steps in data do not map to equal steps in perceived colour difference.

> "The jet colormap has widely varying lightness (L\*) values throughout its
> range, making it a poor choice for perceptual data representation because the
> human brain perceives changes in lightness as changes in data far more readily
> than changes in hue."
> — Matplotlib documentation [10]

### 1.2 Three simultaneous artefacts

Crameri, Shephard & Heron (2020) identify three distinct perceptual artefacts
that rainbow colormaps create simultaneously:

1. **False boundaries (banding).** Colormaps with non-monotonic L\* values
   create perceptual banding artefacts — regions where data appears to have
   boundaries or bands that do not exist in the underlying data. This occurs
   because plateaus or kinks in the L\* function cause the viewer to perceive
   false structure. In jet specifically, the cyan-to-green and yellow-to-red
   transitions contain abrupt L\* shifts that produce the appearance of sharp
   edges in smooth, continuous data. Viewers perceive contour lines that do not
   exist.

2. **Hidden features (masking).** Conversely, L\* plateaus — regions where
   lightness barely changes despite data value changes — make real gradients
   invisible. Small-scale variations in the data are suppressed. Real structures
   that become clearly visible with a perceptually uniform colormap are hidden
   by jet.

3. **No intuitive ordering.** Rainbow colormaps prevent any visual intuitive
   ordering of the data set because the human eye cannot naturally sort the
   spectral hues into a magnitude sequence. Unlike a light-to-dark ramp, there
   is no innate "green is more than blue" ordering.

> "Rainbow colormaps create three distinct perceptual artifacts simultaneously:
> they add artificial boundaries to some parts of the data range, hide
> small-scale variations elsewhere, and prevent any visual intuitive ordering in
> the data set."
> — Crameri, Shephard & Heron (2020), Nature Communications [2]

> "Colormaps with non-monotonic lightness (L\*) values, such as jet, create
> perceptual banding artifacts — regions where data appears to have boundaries
> or bands that do not exist in the underlying data — because plateaus or kinks
> in the L\* function cause the viewer to perceive false structure."
> — Matplotlib documentation [10]

### 1.3 Disproportionate visual attention to yellow and cyan

Rainbow colormaps have non-uniform luminance that disproportionately and
misleadingly draws visual attention to the yellow and cyan elements. These hue
bands are perceptually brightest due to L\* peaks, causing the viewer's eye to
be drawn to those data ranges regardless of their actual importance in the
underlying data.

> "Rainbow colormaps have non-uniform luminance that disproportionately and
> misleadingly draws visual attention to the yellow and cyan elements, creating
> perceptual distortion in the data representation."
> — Stauffer et al. (2022), Geoscience Communication [3]

### 1.4 Quantified distortion: >7% of displayed data variation

The distortion is not merely qualitative. Crameri et al. (2020) measured the
perceptual non-uniformity of rainbow colormaps using the CIEDE2000 colour
difference metric. They found that colour-introduced blind interpretation can
diverge from an objective representation by more than seven percent of the
total displayed data variation. This means that a viewer reading a jet-coloured
heatmap can misread data values by up to 7.5% of the full data range purely
due to the colour encoding — before any other source of measurement error.

> "Rainbow colormaps introduce quantifiable data distortion: colour-introduced
> blind interpretation can diverge from an objective representation by more than
> seven percent of the displayed data variation, as measured by the CIEDE2000
> perceptual uniformity metric."
> — Crameri, Shephard & Heron (2020), Nature Communications [2]

> "Rainbow colormaps introduce quantifiable visual distortion of up to 7.5% of
> total displayed data variation due to non-uniform perceptual encoding, where
> abrupt changes in lightness and saturation create unintended emphasis on
> certain data ranges."
> — Nuez et al. (2021), HESS [1]

> "Rainbow colormaps cause discordant false coloring that can produce visual
> errors of up to 7.5% of the total displayed data variation, meaning viewers
> misread data values by that margin due to perceptual non-uniformity."
> — Nuez et al. (2021), HESS [1]

### 1.5 Implications for RF population heatmaps

In the context of this project's RF population heatmaps, these artefacts mean:

- Apparent sharp contours in the centroid density maps may be artefacts of jet's
  L\* kinks rather than genuine receptive field boundaries.
- Subtle gradients in firing rate density — potentially the most scientifically
  interesting features — may be invisible in the cyan and yellow bands.
- Viewers will unconsciously over-weight mid-range data values (mapped to yellow
  and cyan) relative to the extremes.

---

## 2. Empirical Evidence: Jet Performs Worst in Controlled Studies

### 2.1 Triplet comparison study (Smart & Szafir, UW Interactive Data Lab)

Smart & Szafir conducted a controlled empirical study testing 9 different
colormaps using triplet comparison tasks — participants were shown a reference
colour and asked to judge which of two other colours was closer in data value.

Key findings:

- **Jet performed worst overall** among all nine colormaps tested, in both
  response time and error rate. The authors concluded it "should be jettisoned."

- **Viridis** exhibited consistently low error across all conditions and was
  significantly less error-prone than both a single-hue (blues) and jet
  (p < 0.001). Viridis was designed in CAM02-UCS colour space to ensure
  perceptual uniformity.

- **Jet's occasional good performance is an artefact of categorical colour
  naming, not perceptual uniformity.** Jet's poor overall performance is
  partially rescued in a narrow isoluminant region around reference value 50,
  where colour name boundaries happen to align with data value differences,
  yielding error rates as low as 3.5%. This demonstrates that when jet appears
  to "work," it is because viewers use categorical landmarks ("that's yellow,"
  "that's green") rather than actually perceiving the continuous data mapping.
  This advantage does not generalise to the rest of the data range.

> "In empirical triplet comparison tasks, the rainbow colormap (jet) performed
> worst overall among all nine colormaps tested in both response time and error
> rate, leading the authors to conclude it 'should be jettisoned'."
> — Smart & Szafir, UW IDL [8]

> "Viridis, a perceptually uniform multi-hue colormap designed in CAM02-UCS
> color space, exhibited consistently low error across all conditions and was
> significantly less error-prone than both blues and jet (p < 0.001)."
> — Smart & Szafir, UW IDL [8]

> "Jet's poor overall performance is partially rescued in a narrow isoluminant
> region around reference value 50, where color name boundaries happen to align
> with data value differences, yielding error rates as low as 3.5% —
> demonstrating that jet's occasional good performance is an artifact of
> categorical color naming rather than perceptual uniformity."
> — Smart & Szafir, UW IDL [8]

### 2.2 Broader empirical consensus

Multiple independent studies converge on the same result: rainbow colormaps are
empirically slower and more error-prone for quantitative judgment tasks compared
to perceptually uniform alternatives. They can both emphasise artificial
gradients in smooth data and hide real structures that become visible only with
perceptually uniform colormaps.

> "Empirical judgment studies have found the rainbow colormap to be perceptually
> much slower and more error-prone for quantitative reading tasks compared to
> single-hue colormaps."
> — Nuez et al. (2021), HESS [1]

> "Rainbow colormaps are empirically slower and more error-prone for
> quantitative judgments than perceptually uniform alternatives, and can both
> emphasize artificial gradients in smooth data and hide real structures that
> become visible only with perceptually uniform colormaps."
> — Nuez et al. (2021), HESS [4]

---

## 3. Colour Vision Deficiency (CVD) Accessibility

### 3.1 Prevalence of CVD

Colour vision deficiency is not rare:

- Worldwide, **0.5% of women and 8% of men** have some form of colour vision
  deficiency.
- The most common forms (deuteranopia and protanopia) specifically impair
  **red-green discrimination** — the exact axis that the jet/rainbow colormap
  relies on most heavily.
- When considering the global population, up to **4%** of all people have CVD.

> "Rainbow colormaps are inaccessible to a large fraction of readers because
> worldwide 0.5% of women and 8% of men have colour-vision deficiency, and
> colormaps including both red and green at similar lightness cannot be read by
> these individuals."
> — Crameri, Shephard & Heron (2020), Nature Communications [2]

> "The most common form of color vision deficiency involves differentiating red
> and green, so colormaps containing both red and green (such as rainbow/jet)
> should be avoided to ensure accessibility."
> — Matplotlib documentation [10]

### 3.2 Impact on peer review

The simultaneous use of red and green in the rainbow colormap obstructs
unbiased data access for colour-vision-deficient readers. In a typical
peer-review scenario — a male editor and two male reviewers — there is up to a
**22.1% probability** that at least one person cannot correctly interpret
rainbow-coloured figures. This means that roughly one in five all-male review
panels will include someone who cannot fully read jet-coloured visualisations.

> "The simultaneous use of red and green in the rainbow colormap obstructs
> unbiased data access for the 8-10% of males and 0.4-0.5% of females with
> color vision deficiency, meaning a typical all-male editorial team of three
> has up to a 22.1% chance that at least one member cannot correctly interpret
> rainbow-colored figures."
> — Nuez et al. (2021), HESS [1]

> "Color vision deficiency affects up to 4% of the world's population, and in a
> typical editorial review scenario with a male editor and two male reviewers,
> there is up to a 22.1% chance that at least one person has a color vision
> deficiency, making rainbow colormaps' red-green reliance a significant
> accessibility barrier in peer review."
> — Nuez et al. (2021), HESS [4]

### 3.3 Greyscale printing

Jet is also unreadable when printed in greyscale. Because its lightness profile
is non-monotonic, converting jet to greyscale produces inversions — regions of
high data value can appear darker than regions of lower data value, and vice
versa. A perceptually uniform colormap with monotonically increasing lightness
remains interpretable in greyscale.

---

## 4. Prevalence of Misuse and Resistance to Change

### 4.1 Survey evidence from geosciences

Two large systematic surveys quantify how deeply entrenched the rainbow
colormap remains in scientific practice:

**Stauffer et al. (2022)** surveyed **2,638 geoscience papers** and found:
- **34%** used rainbow colour maps specifically.
- **55%** contained at least one problematic visualisation (rainbow or
  red-green palette without alternatives).
- Rainbow colormap usage has remained remarkably stable over 15 years, dropping
  only from **31% in 2005 to 29% in 2020**. This indicates that awareness
  campaigns and published criticisms have had minimal practical impact on
  researcher behaviour.

> "A survey of 2,638 geoscience papers found that 34% used rainbow colour maps
> specifically, and 55% contained at least one problematic visualization
> (rainbow or red-green without alternatives), showing widespread continued
> misuse despite decades of criticism."
> — Stauffer et al. (2022), Geoscience Communication [3]

> "Rainbow colormap usage in geosciences has remained remarkably stable over 15
> years, dropping only from 31% in 2005 to 29% in 2020, indicating that
> awareness campaigns and published criticisms have had minimal practical impact
> on researcher behavior."
> — Stauffer et al. (2022), Geoscience Communication [3]

**Nuez et al. (2021)** surveyed **797 HESS papers** (2005-2020) and found:
- **24%** contained rainbow colormaps.
- **23%** had red-green elements without adequate distinction.
- In total, **47%** had some form of visualisation issue.
- Only **3.4%** of reviewer comments addressed colour problems — reviewers
  almost never flag problematic colormaps.

> "A systematic survey of 797 HESS papers (2005-2020) found that 24% contained
> rainbow colormaps and 23% had red-green elements without adequate distinction,
> totaling 47% with some form of visualization issue, while only 3.4% of
> reviewer comments addressed color problems."
> — Nuez et al. (2021), HESS [4]

### 4.2 Implication

The fact that jet remains widespread does not validate its use — it reflects
inertia and the absence of reviewer gatekeeping, not scientific defensibility.
Switching away from jet is both scientifically correct and increasingly expected
by journals.

---

## 5. Journal and Community Policies

### 5.1 Nature

Nature's figure preparation guidelines now recommend perceptually uniform
colormaps and advise against rainbow/jet colormaps.

> Source: Nature Research Figure Guide [5]

### 5.2 AGU (American Geophysical Union)

AGU has published recommendations against rainbow colormaps in their author
guidelines for journal submissions.

> Source: AGU text and graphics requirements [6]

### 5.3 EGU (European Geosciences Union)

EGU Copernicus journals have published editorials and multiple survey papers
documenting the problem with rainbow colormaps, and their journal editors have
advocated for perceptually uniform alternatives.

> Source: EGU Geodynamics Division blog [7]

### 5.4 Matplotlib

Matplotlib changed its default colormap from `jet` to `viridis` in version 2.0
(released 2017), explicitly citing perceptual uniformity and CVD accessibility
as the reasons. The matplotlib documentation now contains an extended discussion
of why jet is problematic and recommends perceptually uniform alternatives.

> Source: Matplotlib colormap documentation [10]

---

## 6. Recommended Alternatives

### 6.1 Overview of perceptually uniform colormaps

Perceptually uniform colormaps are recommended as the best choice for
scientific data because **equal steps in data correspond to equal perceived
steps in colour**, achieved through monotonically increasing lightness. They
report true data variations, reduce complexity, and are accessible for people
with colour-vision deficiencies.

> "Perceptually uniform colormaps (viridis, plasma, inferno, magma, cividis)
> are recommended as the best choice for scientific data because equal steps in
> data correspond to equal perceived steps in color, achieved through
> monotonically increasing lightness."
> — Matplotlib documentation [10]

> "Perceptually uniform colormaps such as viridis, magma, plasma, inferno,
> cividis, and batlow report true data variations, reduce complexity, and are
> accessible for people with colour-vision deficiencies; cividis in particular
> is mathematically optimized for readers with different colour-vision
> deficiencies."
> — Crameri, Shephard & Heron (2020), Nature Communications [2]

### 6.2 Comparison table

| Colormap | Type | Perceptually uniform | CVD-safe | B&W printable | Colour range | Availability |
|----------|------|---------------------|----------|---------------|--------------|--------------|
| **`batlow`** | Sequential, multi-hue | Yes | Yes | Yes | Blue-green-yellow-red (wide gamut) | `cmcrameri` package or Crameri's website |
| `viridis` | Sequential, multi-hue | Yes | Yes | Yes | Purple-green-yellow | Built into matplotlib (default since 2.0) |
| `inferno` | Sequential, multi-hue | Yes | Yes | Yes | Black-purple-red-yellow-white | Built into matplotlib |
| `plasma` | Sequential, multi-hue | Yes | Yes | Yes | Purple-red-yellow | Built into matplotlib |
| `magma` | Sequential, multi-hue | Yes | Yes | Yes | Black-purple-pink-yellow-white | Built into matplotlib |
| `cividis` | Sequential, two-hue | Yes | Optimised | Yes | Blue-yellow (muted) | Built into matplotlib |

### 6.3 Detailed notes on each alternative

**`batlow`** (Crameri) — Crameri's scientific colour maps are perceptually
uniform and ordered, designed to represent data without visual distortion. They
are readable by colour-vision-deficient and colour-blind people, and remain
interpretable when printed in black and white. `batlow` is the closest to jet's
"colourful" aesthetic while being scientifically correct — it spans a wide hue
range (blue through green, yellow, to red) with monotonically increasing
lightness. It was designed specifically to be a scientifically defensible
replacement for rainbow colormaps.

> "Crameri's scientific colour maps are perceptually uniform and ordered,
> designed to represent data without visual distortion, contrasting with
> colormaps like jet that introduce such distortion."
> — Crameri scientific colour maps website [11]

> "Scientific colour maps are designed to be readable by colour-vision deficient
> and colour-blind people, and remain interpretable when printed in black and
> white — addressing a key failure mode of rainbow/jet colormaps."
> — Crameri scientific colour maps website [11]

Available via: `pip install cmcrameri`, then `from cmcrameri import cm` (use as
`cm.batlow`). Alternatively, Crameri provides standalone colour map data files
at https://www.fabiocrameri.ch/colourmaps/ that can be loaded into matplotlib
directly. The `batlow` page with full details is at
https://www.fabiocrameri.ch/batlow/.

**`viridis`** (matplotlib default) — Designed in CAM02-UCS colour space to
ensure perceptual uniformity. The most extensively tested alternative; has been
matplotlib's default since version 2.0. Exhibits consistently low error across
all conditions in empirical studies. Purple-to-green-to-yellow hue range. The
design rationale is documented at https://bids.github.io/colormap/.

**`inferno`** — Part of the same matplotlib perceptual colormap family as
viridis. Higher contrast than viridis (black-to-yellow range), making it
particularly suitable for heatmaps where strong visual impact is needed.

**`cividis`** — Developed by optimising the viridis colormap specifically for
colour vision deficiency. It enables nearly identical visual-data
interpretation for both CVD and normal vision viewers while being perceptually
uniform in hue and brightness and increasing in brightness linearly. If maximum
CVD accessibility is the primary concern, cividis is the strongest choice.

> "The cividis colormap was developed by optimizing the viridis colormap
> specifically for color vision deficiency, and it enables nearly identical
> visual-data interpretation for both CVD and normal vision viewers while being
> perceptually uniform in hue and brightness and increasing in brightness
> linearly."
> — Nuez & Wickham (2018), PLOS ONE [12]

### 6.4 Recommendation for this project

For RF population heatmaps, **`batlow`** or **`inferno`** are the strongest
candidates:

- Both preserve the intuitive cold-to-hot mapping that researchers expect from
  a density heatmap.
- Both are perceptually uniform — equal data steps produce equal visual steps.
- Both are CVD-safe and print well in greyscale.
- `batlow` provides the widest hue range (most "colourful" appearance), closest
  to the aesthetic familiarity of jet.
- `inferno` provides the highest contrast (black-to-white lightness range),
  best for picking out fine detail.

---

## 7. Key References (Full List)

### Peer-reviewed articles

| # | Authors | Title | Year | Venue | DOI / URL |
|---|---------|-------|------|-------|-----------|
| [A] | Rogowitz, B.E. & Treinish, L.A. | "Data Visualization: The End of the Rainbow" | 1998 | IEEE Spectrum | — |
| [B] | Borland, D. & Taylor, R.M. | "Rainbow Color Map (Still) Considered Harmful" | 2007 | IEEE Computer Graphics & Applications | — |
| [C] | Thyng, K.M. et al. | "True colors of oceanography: Guidelines for effective and accurate colormap selection" | 2016 | Oceanography, 29(3) | — |
| [D] | Nuez, J.R. & Wickham, C.R. | "Optimizing colormaps with consideration for color vision deficiency to enable accurate interpretation of scientific data" | 2018 | PLOS ONE | https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0199239 |
| [E] | **Crameri, F., Shephard, G.E. & Heron, P.J.** | **"The misuse of colour in science communication"** | **2020** | **Nature Communications 11, 5444** | https://www.nature.com/articles/s41467-020-19160-7 |
| [F] | Smart, S. & Szafir, D.A. | "Measuring the Effectiveness of Quantitative Color Encodings" | 2020 | UW Interactive Data Lab | https://idl.uw.edu/papers/quantitative-color |
| [G] | Nuez, J.R. et al. | "Colour issues in geoscience publications" (survey of HESS papers) | 2021 | Hydrology and Earth System Sciences 25, 4549 | https://hess.copernicus.org/articles/25/4549/2021/ |
| [H] | Stauffer, R. et al. | "Somewhere over the rainbow: How to make effective use of colors in meteorological visualizations" (survey of geoscience papers) | 2022 | Geoscience Communication 5, 83 | https://gc.copernicus.org/articles/5/83/2022/ |

### Journal policies and guidelines

| # | Organisation | Resource | URL |
|---|-------------|----------|-----|
| [I] | Nature | Figure Preparation Guide — specifications | https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/ |
| [J] | AGU | Text and Graphics Requirements for Authors | https://www.agu.org/publications/authors/journals/text-graphics-requirements |
| [K] | EGU | "The Rainbow Colour Map" (Geodynamics blog) | https://blogs.egu.eu/divisions/gd/2017/08/23/the-rainbow-colour-map/ |

### Software documentation and tools

| # | Resource | URL |
|---|----------|-----|
| [L] | Matplotlib colormap documentation | https://matplotlib.org/stable/users/explain/colors/colormaps.html |
| [M] | Crameri Scientific Colour Maps | https://www.fabiocrameri.ch/colourmaps/ |
| [N] | Crameri `batlow` colormap | https://www.fabiocrameri.ch/batlow/ |
| [O] | BIDS viridis design rationale | https://bids.github.io/colormap/ |
| [P] | Petroff, M. "Discernibility of rainbow colormaps" | https://mpetroff.net/2019/08/discernibility-of-rainbow-colormaps/ |

### In-text citation key

The bracketed numbers [1]-[15] used in quoted claims above map to the source
list in the order they were retrieved during the literature review:

1. https://hess.copernicus.org/articles/25/4549/2021/hess-25-4549-2021.html — Nuez et al. (2021), HESS (primary)
2. https://www.nature.com/articles/s41467-020-19160-7 — Crameri et al. (2020), Nature Communications (primary)
3. https://gc.copernicus.org/articles/5/83/2022/ — Stauffer et al. (2022), Geoscience Communication (primary)
4. https://hess.copernicus.org/articles/25/4549/2021/ — Nuez et al. (2021), HESS (primary, alternate URL)
5. https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/ — Nature figure guide (primary)
6. https://www.agu.org/publications/authors/journals/text-graphics-requirements — AGU guidelines
7. https://blogs.egu.eu/divisions/gd/2017/08/23/the-rainbow-colour-map/ — EGU blog
8. https://idl.uw.edu/papers/quantitative-color — Smart & Szafir (2020), UW IDL (primary)
9. https://mpetroff.net/2019/08/discernibility-of-rainbow-colormaps/ — Petroff blog
10. https://matplotlib.org/stable/users/explain/colors/colormaps.html — Matplotlib docs (primary)
11. https://www.fabiocrameri.ch/colourmaps/ — Crameri scientific colour maps (primary)
12. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0199239 — Nuez & Wickham (2018), PLOS ONE (primary)
13. https://www.researchgate.net/publication/345156662 — Crameri et al. (2020) on ResearchGate (primary)
14. https://bids.github.io/colormap/ — BIDS viridis (primary)
15. https://www.fabiocrameri.ch/batlow/ — Crameri batlow (primary)

---

## 8. Summary

The scientific consensus against the rainbow/jet colormap is unambiguous and
supported by multiple independent lines of evidence:

1. **Creates false boundaries** in smooth data — RF heatmaps likely show
   apparent contours at jet's L\* kinks (cyan-green, yellow-red transitions)
   that are not present in the underlying data.
2. **Hides real gradients** in the cyan and yellow L\* plateau regions — subtle
   RF density variations in these value ranges are invisible.
3. **Distorts perceived values by >7%** of the total data range (Crameri et al.
   2020, Nature Communications) as measured by CIEDE2000.
4. **Performs worst** among 9 colormaps in controlled perceptual studies with
   statistically significant differences (Smart & Szafir, p < 0.001 vs
   viridis).
5. **Inaccessible to ~8% of male viewers** due to red-green CVD; ~22%
   probability that a male review panel includes someone affected.
6. **Unreadable in greyscale** due to non-monotonic lightness inversions.
7. **Abandoned as default** by matplotlib (2017) and explicitly discouraged by
   Nature, AGU, and EGU journal guidelines.
8. **Prevalence reflects inertia, not validity** — usage barely declined from
   31% to 29% over 15 years despite sustained criticism, and only 3.4% of
   reviewers flag the issue.

Replacement with a perceptually uniform colormap (`batlow`, `inferno`, or
`viridis`) is recommended.
