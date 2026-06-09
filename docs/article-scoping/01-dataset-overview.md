# Dataset Overview

## Session Inventory

12 sessions across 5 subjects, recorded 2022-06-14 to 2022-06-22.
Database root: `F:\liu-onedrive-nospecial-carac\_Teams\Social touch Kinect MNG\02_data\semi-controlled\`
Subtype source: `1_primary\nerve\semicontrol_unit-name_to_unit-type.csv`

### Afferent Subtype Mapping

| Session | Subtype | Conduction vel. | Category |
|---------|---------|-----------------|----------|
| ST14-01 | SA-I | 57 m/s | Aβ |
| ST16-05 | SA-I | 57 m/s | Aβ |
| ST13-01 | SA-II | 53 m/s | Aβ |
| ST16-02 | SA-II | 53 m/s | Aβ |
| ST18-04 | SA-II | 53 m/s | Aβ |
| ST13-03 | HFA | 35 m/s | Aβ |
| ST14-02 | HFA | 35 m/s | Aβ |
| ST14-04 | Field-LTMR | 47 m/s | Aβ |
| ~~ST15-01~~ | ~~Field-LTMR~~ | ~~47 m/s~~ | ~~Aβ (outlier)~~ |
| ST18-01 | Field-LTMR | 47 m/s | Aβ |
| ST13-02 | CT | 0.9 m/s | C-fiber |
| ST16-03 | CT | 0.9 m/s | C-fiber |

**Breakdown:** SA-I (2), SA-II (3), HFA (2), Field-LTMR (2 + 1 outlier), CT (2).

ST15-01 is a **persistent outlier** across all analyses — excluded from group
statistics. It is a Field-LTMR, which have the most expansive innervation
territory (up to 180 circumferential endings per Bai et al. 2015 — note this
is mouse data, so the comparison is a cross-species extrapolation), possibly
explaining its extreme values.

---

## Response Field Areas (mm²) — `iff_mean`

| Session | Subtype | All | Stroke | Tap | Prox. | Dist. |
|---------|---------|-----|--------|-----|-------|-------|
| ST14-01 | SA-I | 838 | 835 | 957 | 713 | 1008 |
| ST16-05 | SA-I | 288 | 667 | 733 | 708 | 321 |
| ST13-01 | SA-II | 609 | 487 | 1059 | 424 | 476 |
| ST16-02 | SA-II | 614 | 568 | 1840 | 521 | 544 |
| ST18-04 | SA-II | 613 | 506 | 868 | 596 | 298 |
| ST13-03 | HFA | 734 | 646 | 577 | 587 | 735 |
| ST14-02 | HFA | 510 | 524 | 329 | 482 | 1364 |
| ST14-04 | Field | 599 | 409 | 842 | 423 | 534 |
| ~~ST15-01~~ | ~~Field~~ | ~~400~~ | ~~**4587**~~ | ~~1819~~ | ~~132~~ | ~~1158~~ |
| ST18-01 | Field | 534 | 447 | 354 | 366 | 461 |
| ST13-02 | CT | 662 | 868 | 261 | 568 | 371 |
| ST16-03 | CT | 632 | 632 | 286 | 603 | 569 |

Typical range: **288–1364 mm²** (excluding ST15-01).
Circularity: 0.15–0.82 (strokes more circular; taps more dispersed).
PCA aspect ratio: 1.03–2.03.

---

## Proximal-Distal Centroid Shifts

| Session | Subtype | Centroid dist. (UV units) | Hotspot dist. (UV units) |
|---------|---------|--------------------|--------------------|
| ST14-01 | SA-I | 2.0 | — |
| ST16-05 | SA-I | 8.7 | 7.1 |
| ST13-01 | SA-II | 8.0 | — |
| ST16-02 | SA-II | 6.3 | — |
| ST18-04 | SA-II | 8.0 | — |
| ST13-03 | HFA | 0.6 | — |
| ST14-02 | HFA | 23.5 | — |
| ST14-04 | Field | 5.6 | — |
| ~~ST15-01~~ | ~~Field~~ | ~~37.3~~ | ~~70.4~~ |
| ST18-01 | Field | 4.6 | — |
| ST13-02 | CT | 20.6 | — |
| ST16-03 | CT | 6.9 | — |

**Mean centroid shift (excl. ST15-01): 8.6 ± 7.0 UV units** (range 0.6–23.5 UV units).

> Note: centroid-shift and hotspot-shift distances are measured in the SLIM UV
> parameter space (columns `proximal_distal_distance_uv` and
> `centroid_hotspot_distance_uv_stroke`), not physical millimetres
> (physical-mm conversion pending).

---

## Per-Subtype Summary (excluding ST15-01)

| Subtype | n | Mean RF area (mm²) | Centroid shift (UV units) | Literature RF (probe) |
|---------|---|--------------------|--------------------|----------------------|
| SA-I | 2 | 563 | 5.4 | ~4 mm² (Vallbo 1995) [verify in full text] |
| SA-II | 3 | 612 | 7.4 | ~36 mm² (Vallbo 1995) [verify in full text] |
| HFA | 2 | 622 | 12.1 | hair follicle territory |
| Field-LTMR | 2 | 567 | 5.1 | 3–4 mm² (Bai et al. 2015, mouse) |
| CT | 2 | 647 | 13.8 | 1–35 mm² (Wessberg 2003) |

---

## Stimulus Parameters During IFF Activity

- Contact area: 3–100 mm²
- Contact depth: 0.5–3.2 mm
- Hand velocity: 27–100+ mm/s (2.7–10+ cm/s)
- IFF during firing: 8.5–23.2 Hz
- Pressure: 0.03–0.53 Pa (**known calibration issue**)

> Note: these are central-tendency ranges computed over `mean_during_iff`-filtered
> touch aggregations (per-touch means during IFF activity), not raw per-touch
> min/max extremes.

---

## Touch Counts

Total touches across all sessions: **7,223**.

Largest sessions: ST16-05 (SA-I, 1,983 touches), ST18-01 (Field, 1,836 touches).
Median per session: 295.
