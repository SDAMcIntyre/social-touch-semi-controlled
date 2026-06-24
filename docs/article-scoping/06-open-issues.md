# Open Issues to Resolve Before Paper

---

## 1. Pressure Calibration

**Status:** Known unit issue.
**Problem:** Pipeline reports pressure as 0.03–0.53 Pa, which is unrealistically
low (atmospheric pressure is ~101,325 Pa; finger contact typically produces
kPa-range pressures).
**Impact:** Pressure tuning curves and pressure-related comparisons with
literature are unreliable until fixed.
**Action:** Investigate the pressure computation pipeline; determine correct
unit conversion.

---

## 2. ST15-01 Outlier

**Status:** Known persistent outlier across all analyses.
**Session:** ST15-01 (Field-LTMR, conduction velocity 47 m/s).
**Extreme values:** Stroke RF area = 4,587 mm² (~5–11× other stroke RFs),
centroid shift = 37.3 UV units (~1.6–62× others), hotspot shift = 70.4 UV units.
(Centroid- and hotspot-shift distances are in SLIM UV parameter space, not
physical mm — physical-mm conversion pending. RF area remains in mm².)
**Possible explanation:** Field-LTMRs have the most expansive innervation
territory (up to 180 circumferential endings per Bai et al. 2015 — note this
is mouse data, so the comparison is a cross-species extrapolation). However,
the magnitude of the outlier suggests data quality issues rather than
biological variation.
**Action:** Exclude from group statistics. Optionally include as a separate
case in supplementary material with explicit discussion.

---

## 3. Per-Subtype Sample Size

**Status:** Inherent limitation.
**Problem:** n=2 per subtype (SA-I, HFA, Field-LTMR, CT) and n=3 for SA-II.
Too thin for inferential statistics (t-tests, ANOVA).
**Options:**
- Frame as descriptive characterization, not inferential statistics
- Pool subtypes: "slowly adapting" (SA-I + SA-II, n=5) vs. "rapidly adapting"
  (HFA + Field-LTMR, n=4) for more statistical power
- Include CT (n=2) as a qualitative comparison only
**Action:** Decide on pooling strategy before drafting results.

---

## 4. Feature Grid Feasibility

**Status:** 2D grids not feasible; 1D velocity grid works.
**Problem:** Total 7,223 touches, median 295/session. Configured 2D grids
have 1,100–10,000 cells → ~73–97% empty (per session, ≤1 touch/cell:
velocity×depth 10,000 cells ≈ 295/10,000 filled → ~97% empty;
velocity×pressure 1,100 cells ≈ 295/1,100 filled → ~73% empty).
**Working alternative:** 1D velocity grid via `cross_render_sessions`
(20 bins, step=25 mm/s) already enabled and functional.
**Optional:** Coarse 2D grid (5×5 = 25 cells) for the 2 largest sessions
(ST16-05: 1,983 touches, ST18-01: 1,836 touches) as supplementary analysis.
**Action:** Use 1D velocity grid for main paper. Optionally enable coarse 2D
for supplementary material.

---

## 5. Conduction Velocity Validation

**Status:** Data available, not yet used in paper.
**Available data:** Per-subtype conduction velocities from
`semicontrol_unit-name_to_unit-type.csv`:
- SA-I: 57 m/s, SA-II: 53 m/s, HFA: 35 m/s, Field-LTMR: 47 m/s, CT: 0.9 m/s
- Electrode-endorgan distance: 40 cm for all units
**Action:** Include in methods table for unit classification validation. The
clear separation between Aβ (35–57 m/s) and CT (0.9 m/s) confirms subtyping.

---

## 6. Response Field vs. Receptive Field Framing

**Status:** Terminological decision needed.
**Finding:** No established formal distinction exists in somatosensory
peripheral literature. The term needs explicit definition in the paper.
**Proposed definition:** "Response field" = the skin area where naturalistic
touch (hand stroking or tapping) elicits a response from a single afferent,
as opposed to the classical "receptive field" delineated by punctate
monofilament probes.
**Action:** Decide on terminology and write a clear definition paragraph in
the Introduction or Methods.
