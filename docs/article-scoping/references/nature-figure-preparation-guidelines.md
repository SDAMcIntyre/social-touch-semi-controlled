# Nature Journal Figure Preparation Guidelines

Comprehensive reference for figure design standards across Nature Portfolio
journals (Nature, Nature Methods, Nature Communications, Scientific Reports,
etc.).  Compiled from official Springer Nature sources and the *Points of
Significance* / *Points of View* columns.

**Primary sources**

- [Nature Research Figure Guide -- Specifications](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/)
- [Nature Research Figure Guide -- Building & Exporting Panels](https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/)
- [Nature Research Figure Guide -- Extended Data](https://research-figure-guide.nature.com/figures/extended-data-formatting-guidelines/)
- [Nature Research Figure Guide -- Image Integrity](https://research-figure-guide.nature.com/figures/image-integrity/)
- [Nature Research Figure Guide -- Top 10 Ways to Delay Your Paper](https://research-figure-guide.nature.com/figures/top-10-ways-to-delay-your-paper/)
- [Nature Formatting Guide](https://www.nature.com/nature/for-authors/formatting-guide)
- [Nature Final Submission](https://www.nature.com/nature/for-authors/final-submission)
- Wong, B. "Points of view: Colour blindness." *Nature Methods* **8**, 441 (2011). [doi:10.1038/nmeth.1618](https://doi.org/10.1038/nmeth.1618)
- [Springer Nature Artwork Submission Instructions](https://support.springernature.com/en/support/solutions/articles/6000083109-artwork-submission-instructions)

---

## 1. Figure Dimensions

### 1.1 Nature Portfolio journals

| Width category        | mm        | inches    |
|-----------------------|-----------|-----------|
| Single column         | 89        | 3.50      |
| 1.5 column            | 120--136  | 4.72--5.35|
| Double column (full)  | 183       | 7.20      |
| **Maximum height**    | **247**   | **9.72**  |
| Recommended max height| 170       | 6.69      |

The 170 mm recommended maximum leaves space for the figure legend below; the
247 mm value is the absolute physical page limit.

**Extended Data figures:** single column 89 mm, two column 180 mm; maximum
page area 180 x 170 mm; file size limit 10 MB.

**Main figure file size limit:** 50 MB.

### 1.2 Broader Springer journals (for comparison)

| Width category   | Large-format journals | Small-format journals |
|------------------|-----------------------|-----------------------|
| Single column    | 84 mm                 | 119 mm                |
| Double column    | 174 mm                | --                    |
| Max height       | 234 mm                | 195 mm                |

---

## 2. Typography

### 2.1 Font families

| Context                    | Font                                   |
|----------------------------|----------------------------------------|
| All figure text (default)  | **Helvetica** or **Arial** (sans-serif)|
| Greek letters / glyphs     | **Symbol** font                        |
| Amino-acid sequences       | **Courier** (monospaced), lines of 50 or 100 characters |

The same font family must be used consistently across **all** figures in the
manuscript.

### 2.2 Font sizes

| Element                              | Size      | Style                             |
|--------------------------------------|-----------|-----------------------------------|
| Panel labels (a, b, c ...)           | **8 pt**  | Bold, upright (not italic), lowercase |
| All other text (axis labels, ticks, legends, annotations) | **5--7 pt** | Regular weight |
| Absolute minimum readable text       | **5 pt**  | At final printed size             |

### 2.3 Font embedding and editability

- All text **must remain editable** -- never outline or convert to shapes.
- Embed fonts as **TrueType 2 or Type 42** (not TrueType 3).
- For matplotlib users: `matplotlib.rcParams['pdf.fonttype'] = 42`.
- **No colored text.** Use colored symbols / boxes with black text labels
  instead.
- Text contrast ratio must exceed **4.5:1** (WCAG 2.1 Level AA).

> **Manuscript body text** (not figures): 12 pt Times New Roman, per the
> formatting guide.

---

## 3. Line Weights

| Parameter            | Value       |
|----------------------|-------------|
| Minimum line weight  | **0.25 pt** |
| Maximum line weight  | **1.0 pt**  |

Lines thinner than 0.25 pt may vanish during print production.  This applies
to axes, tick marks, data lines, arrows, and scale bars.  All lines must
remain as **editable vector artwork** (do not rasterise).

For broader Springer journals the minimum is 0.1 mm (0.3 pt).

---

## 4. Resolution (DPI)

### 4.1 Nature Portfolio

| Content type                      | DPI                                |
|-----------------------------------|------------------------------------|
| Photographic / halftone images    | 300 minimum; 450+ recommended      |
| Combination (photo + line art)    | 600                                |
| Line art / graphs (rasterised)    | 600+ (vector PDF/EPS preferred)    |
| Online proofs maximum             | 450                                |
| Extended Data maximum             | 300 (capped, not a minimum)        |

### 4.2 Broader Springer journals

| Content type            | DPI   |
|-------------------------|-------|
| Line art (bitmap)       | 1200  |
| Halftone (photographs)  | 300   |
| Combination             | 600   |

**Best practice:** Submit line art and graphs as **vector** (PDF or EPS),
making DPI irrelevant.  Reserve raster for photographic content only.

---

## 5. File Formats

### 5.1 Main figures

**Preferred (vector, editable):**  `.ai`, `.eps`, `.pdf`

**Acceptable:** layered `.psd`, `.svg`, `.ps`, PowerPoint (convert to PDF),
Excel

**NOT accepted:** JPEG, TIFF, PNG (bitmap-only formats), Canvas, DeltaGraph,
TeX, ChemDraw, SigmaPlot, CorelDraw native formats

All components must be **embedded** (not linked).

### 5.2 Extended Data figures

**Preferred:** JPEG (highest quality setting)

**Acceptable:** TIFF, EPS

### 5.3 Color mode

Submit in **RGB** (not CMYK).  Nature handles the CMYK conversion for print
automatically.  RGB preserves a wider gamut, especially for fluorescence
images viewed digitally.

---

## 6. Color and Accessibility

### 6.1 Mandatory rules

Nature requires figures to meet **WCAG 2.1 Level AA** accessibility
standards.

- **No red-green combinations** (affects ~8 % of male readers).
- **No rainbow / jet colormaps** (misleading in greyscale, unreadable for
  CVD).
- **Never encode meaning with color alone** -- always add redundant encoding
  via shape, line style, pattern, or text label.
- Text contrast ratio > **4.5:1**.
- **No colored text** -- use keys / keylines with high-contrast black or
  white labels.

### 6.2 Recommended categorical palette -- Okabe-Ito

From Wong, B. *Nature Methods* **8**, 441 (2011):

| Color          | Hex       | RGB             |
|----------------|-----------|-----------------|
| Black          | `#000000` | (0, 0, 0)       |
| Orange         | `#E69F00` | (230, 159, 0)   |
| Sky Blue       | `#56B4E9` | (86, 180, 233)  |
| Bluish Green   | `#009E73` | (0, 158, 115)   |
| Yellow         | `#F0E442` | (240, 228, 66)  |
| Blue           | `#0072B2` | (0, 114, 178)   |
| Vermillion     | `#D55E00` | (213, 94, 0)    |
| Reddish Purple | `#CC79A7` | (204, 121, 167) |

This palette remains distinguishable under protanopia, deuteranopia, and
tritanopia, and has distinct luminance values suitable for greyscale printing.

### 6.3 Additional categorical palettes -- Paul Tol (SRON)

| Palette        | Max categories | Hex codes |
|----------------|----------------|-----------|
| Bright         | 7  | `#4477AA, #EE6677, #228833, #CCBB44, #66CCEE, #AA3377, #BBBBBB` |
| High-contrast  | 3  | `#004488, #DDAA33, #BB5566` |
| Muted          | 9  | `#CC6677, #332288, #DDCC77, #117733, #88CCEE, #882255, #44AA99, #999933, #AA4499` |

Source: <https://personal.sron.nl/~pault/>

### 6.4 Continuous / sequential data

Use **perceptually uniform** colormaps:

- **viridis** -- general-purpose, CVD-safe
- **cividis** -- optimised for deuteranopia
- **inferno**, **magma**, **plasma** -- CVD-safe alternatives

### 6.5 Testing tools

- **Color Oracle** (desktop, free) -- real-time CVD simulation
- **Coblis** (web) -- upload image, view CVD simulations
- **Viz Palette** (web) -- validates palette accessibility
- **Greyscale conversion** -- if still distinguishable in B&W, it passes the
  strictest test

### 6.6 Fluorescence microscopy

Recolor from red/green to **green/magenta** for accessibility.

---

## 7. Panel Labeling and Layout

### 7.1 Panel labels

- **Lowercase** letters: a, b, c, d ...
- **8 pt bold**, upright (not italic)
- Placed consistently (typically upper-left of each panel)
- Nature uses lowercase; contrast with Science and Cell which use uppercase
  (A, B, C)

### 7.2 Multi-panel layout

- Arrange panels alphabetically in **reading order** (left-to-right, top-to-bottom).
- Group related data together.
- Maintain **consistent spacing** between panels.
- **Align axes** across comparative panels.
- Size individual panels based on content -- do not make panels
  disproportionately large for minor information.
- **Minimise white space** without overcrowding.
- Design compact figures; tall, narrow figures waste page space.

### 7.3 Elements to avoid

- Background gridlines
- Drop shadows
- Decorative icons (use text labels instead)
- Pattern fills (use solid colours)
- Overlapping text
- Text on busy / low-contrast backgrounds

---

## 8. Scale Bars

- Use **scale bars** rather than magnification factors on all microscopy /
  micrograph panels.
- Scale bars and their text must be on a **separate, editable layer** -- do
  not flatten into the image.
- State scale bar length in the figure legend (Nature Communications,
  Scientific Data) or on the bar itself (Scientific Reports).
- Place consistently (commonly bottom-right).
- Magnification must also be stated in the Methods section.

---

## 9. Axis and Graph Formatting

- Include **axis lines and tick marks** on all graphs.
- Label every axis with **units in parentheses**: `Data (unit)`.
- Use **lowercase type, first letter capitalised, no full stop** at the end
  of labels.
- Keep all text within the 5--7 pt range.
- Line weights 0.25--1.0 pt.
- **No** background gridlines, drop shadows, patterns, or decorative
  elements.

---

## 10. Statistical Figure Best Practices

From *Nature Methods* "Points of Significance" and *Nature Biomedical
Engineering* editorials.

### 10.1 Bar charts

- Bar charts are appropriate for **counts only**.
- For means / medians: use **scatter + error** plots or **box plots**.
- Bar charts misleadingly emphasise the distance from zero rather than
  between-group differences.

### 10.2 Individual data points

- **Always show individual data points** overlaid on summary statistics,
  especially for small samples (n < ~100).
- For large samples (n > 100), box-and-whisker plots are preferable.

### 10.3 Box plots and error bars

- Box plots require **at least 5 data points**; for 3--5, use
  mean-and-error plots.
- Always **define error bars** in the legend: SD, SEM, or CI -- and how they
  were calculated.
- State whether centre values are **mean or median**.

### 10.4 Key references

- "Kick the bar chart habit." *Nature Methods* **11**, 113 (2014).
  [doi:10.1038/nmeth.2837](https://doi.org/10.1038/nmeth.2837)
- "Show the dots in plots." *Nature Biomedical Engineering* **1**, 0079
  (2017). [doi:10.1038/s41551-017-0079](https://doi.org/10.1038/s41551-017-0079)
- [Points of Significance collection](https://www.nature.com/collections/qghhqm/pointsofsignificance)

---

## 11. Figure Legend Requirements

Each legend must contain:

1. **Brief title** for the whole figure.
2. **Short description of each panel** and the symbols used.
3. **Sample size:** exact *n* for each group/condition, with a definition of
   how *n* is defined (e.g. "x cells from x slices from x animals").
4. **Error bar definition:** what the error bars represent (SD, SEM, CI) and
   how they were calculated; description of centre values (median or mean).
5. **Statistical test** used and *P* values.
6. **Number of independent replicates.**
7. **Defined abbreviations.**

**Maximum length:** fewer than **300 words** per legend.

Example structure:
> **Figure 1. [Title].** (**a**) [Panel description]. (**b**) [Methods/data].
> (**c**) [n = 3, mean +/- s.d., \*\*\**P* < 0.001, two-tailed Student's
> *t*-test].

---

## 12. Image Integrity Rules

- **Generative AI content in figures is NOT permitted.**
- Prohibited Photoshop tools: Spot Healing Brush, Remove Tool, Healing Brush,
  Patch Tool, Content-Aware Move, Eraser, Clone Stamp, Generative Fill.
- Brightness / contrast adjustments only if applied **equally to the entire
  image** and equally to controls.
- Gel / Western blot: spliced or non-adjacent lanes must be **clearly
  marked**; submit unprocessed originals with accepted version.

---

## 13. Common Rejection and Delay Reasons

From Nature's "Top 10 Ways to Delay Your Paper":

| # | Issue | Detail |
|---|-------|--------|
| 1 | Wrong figure dimensions | Even a 2 mm mismatch triggers rejection |
| 2 | Merged / flattened layers | Rasterising editable elements |
| 3 | Text too small | Below 5 pt at final print size |
| 4 | Text outlined / corrupted | Converted to shapes, uneditable |
| 5 | Accessibility failures | Red-green schemes, coloured text |
| 6 | Low-resolution images | Blurry raster where vector was needed |
| 7 | Missing / unlinked panels | Improperly embedded graphics |
| 8 | Image integrity problems | Undisclosed manipulation, splicing |
| 9 | Chemical structure errors | Wrong size or uneditable |
| 10| Wrong file format | JPEG for line art, unsupported formats |

Additional common issues:
- Non-standard fonts (Calibri, Open Sans, Times instead of Arial/Helvetica)
- Inconsistent styling across figures in the same paper
- Missing panel labels or scale bars
- JPEG compression artifacts

---

## 14. Pre-Submission Checklist

**Dimensions and format**
- [ ] Widths match Nature column spec (89 / 120--136 / 183 mm)
- [ ] Height within 247 mm (ideally under 170 mm)
- [ ] File format is .ai, .eps, or .pdf (not JPEG/TIFF/PNG for main figures)
- [ ] File size under 50 MB per figure
- [ ] RGB colour space (not CMYK)

**Typography**
- [ ] Font is Arial or Helvetica throughout all figures
- [ ] All text 5--7 pt; panel labels 8 pt bold lowercase
- [ ] All text editable (not outlined / converted to shapes)
- [ ] Font embedding: TrueType 2 or 42

**Lines and resolution**
- [ ] Line weights within 0.25--1.0 pt
- [ ] Photos at 300+ DPI at final print size
- [ ] Line art at 600+ DPI or vector format

**Panels and layout**
- [ ] Panel labels present, bold, lowercase, consistent placement
- [ ] Panels arranged alphabetically in reading order
- [ ] Scale bars on all microscopy panels (separate editable layer)
- [ ] No gridlines, drop shadows, or decorative elements

**Colour and accessibility**
- [ ] No red-green colour combinations
- [ ] Colourblind-safe palette used (Okabe-Ito or viridis/cividis)
- [ ] Colour not the sole encoding -- redundant shapes/patterns/labels
- [ ] Text contrast ratio > 4.5:1
- [ ] Tested with CVD simulator (Color Oracle, Coblis)

**Statistical content**
- [ ] Individual data points shown alongside summary statistics
- [ ] Error bars defined in legend (SD / SEM / CI)
- [ ] Sample sizes (exact *n*) stated in legend
- [ ] Statistical test and *P* values in legend
- [ ] Number of independent replicates stated

**Legend**
- [ ] Under 300 words
- [ ] Title, panel descriptions, abbreviations defined

**Integrity**
- [ ] No generative AI used in figure creation
- [ ] All adjustments applied uniformly and disclosed
- [ ] Spliced lanes clearly marked

---

## 15. Matplotlib Configuration for Nature Compliance

```python
import matplotlib as mpl
import matplotlib.pyplot as plt

# --- Font embedding (critical) ---
mpl.rcParams['pdf.fonttype'] = 42      # TrueType, not Type 3
mpl.rcParams['ps.fonttype'] = 42

# --- Fonts ---
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Arial']
mpl.rcParams['font.size'] = 7          # default body text
mpl.rcParams['axes.labelsize'] = 7     # axis labels
mpl.rcParams['xtick.labelsize'] = 6    # tick labels
mpl.rcParams['ytick.labelsize'] = 6
mpl.rcParams['legend.fontsize'] = 6

# --- Line widths (0.25--1.0 pt range) ---
mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['lines.linewidth'] = 0.75
mpl.rcParams['xtick.major.width'] = 0.5
mpl.rcParams['ytick.major.width'] = 0.5

# --- Okabe-Ito colour cycle ---
OKABE_ITO = [
    '#E69F00',  # orange
    '#56B4E9',  # sky blue
    '#009E73',  # bluish green
    '#F0E442',  # yellow
    '#0072B2',  # blue
    '#D55E00',  # vermillion
    '#CC79A7',  # reddish purple
    '#000000',  # black
]
mpl.rcParams['axes.prop_cycle'] = plt.cycler(color=OKABE_ITO)

# --- Figure size constants (mm to inches) ---
SINGLE_COL = 89 / 25.4    # 3.503 in
ONEHALF_COL_MIN = 120 / 25.4  # 4.724 in
ONEHALF_COL_MAX = 136 / 25.4  # 5.354 in
DOUBLE_COL = 183 / 25.4   # 7.205 in
MAX_HEIGHT = 170 / 25.4   # 6.693 in
```

---

## 16. Points of View Column -- Key Articles for Figure Design

From Bang Wong's *Nature Methods* column:

**Colour**
- "Color coding" (Aug 2010)
- "Color blindness" (Jun 2011) -- *the* reference for the Okabe-Ito palette
- "Avoiding color" (Jul 2011)
- "Mapping quantitative data to color" (Aug 2012)
- "Heat maps" (Mar 2012)

**Layout and composition**
- "Layout" (Oct 2011)
- "Gestalt principles Part 1 & 2" (Nov--Dec 2010)
- "Negative space" (Jan 2011)
- "The design process" (Dec 2011)
- "Elements of visual style" (May 2013)

**Figure elements**
- "Typography" (Apr 2011)
- "Axes, ticks and grids" (Mar 2013)
- "Labels and callouts" (Apr 2013)
- "Plotting symbols" (Jun 2013)

**Plot types**
- "Bar charts and box plots" (Feb 2014)
- "Temporal data" (Feb 2015)
- "Unentangling complex plots" (Jul 2015)

**Clarity**
- "Simplify to clarify" (Aug 2011)
- "Design of data figures" (Sep 2010)
- "Points of review Part 1 & 2" (Feb--Mar 2011)

Full listing: [Methagora -- Data Visualization Points of View](http://blogs.nature.com/methagora/2013/07/data-visualization-points-of-view.html)

---

## 17. Nature vs. Science vs. Cell -- Quick Comparison

| Specification       | Nature               | Science              | Cell                 |
|---------------------|----------------------|----------------------|----------------------|
| Panel labels        | **lowercase** bold   | UPPERCASE            | UPPERCASE            |
| Panel label font    | 8 pt Helvetica/Arial | 6--8 pt Times        | 6--8 pt Avenir/Arial |
| Figure text         | 5--7 pt sans-serif   | 6--8 pt              | 5--8 pt              |
| Single column width | 89 mm                | 85 mm                | 85 mm                |
| Double column width | 183 mm               | 178 mm               | 178 mm               |
| Preferred format    | PDF / EPS / AI       | EPS / PDF            | PDF / EPS / AI       |
