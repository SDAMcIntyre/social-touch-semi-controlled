# Investigation: 3D-Mesh Flattening Algorithms for Forearm/Hand Receptive-Field Mapping

**Date:** 2026-05-16
**Status:** Investigation — literature survey, no implementation decided
**Trigger:** Broad scientific-literature review of 3D-to-2D surface flattening
algorithms, prompted by the need to display tactile receptive fields on a
consistent 2D representation of the hand/forearm surface across sessions.
**Related notes:**
- [note-3d-to-2d-surface-projection-algorithms.md](note-3d-to-2d-surface-projection-algorithms.md) — distortion/algorithm catalogue from earlier investigation
- [investigation-caret-pals-for-forearm-flattening.md](investigation-caret-pals-for-forearm-flattening.md) — applicability check of cortical flatteners
- Sandbox script `code/scripts/flatten_forearm_sandbox.py`
- Active plan `docs/development/plans/active/flatten-forearm-sandbox.md`

---

## Question

The receptive-field-mapping pipeline projects spike-tagged 3D contact
points on the forearm/hand onto a 2D heatmap. Current production
implements only two simple projections (tangent-plane drop-Z and
session-local cylindrical unwrap); a sandbox script prototypes three
libigl baselines (LSCM, harmonic, ARAP) with an optional centre-vertex
pin.

**The literature question:** what is the full landscape of 3D-mesh
flattening algorithms in the scientific literature, and which of them is
the best match for our problem — open-boundary, mildly-curved, elongated
anatomical surface, free of cuts, with the goal of plotting sparse RF
measurements in a 2D frame that is comparable across sessions?

## Method

Four parallel background subagents:

- **A.** Codebase audit — what already exists, what is missing.
- **B.** Web survey of classical (pre-~2014) mesh parametrization
  algorithms.
- **C.** Web survey of modern (~2015–present) parametrization algorithms.
- **D.** Web survey of biomedical / applied surface flattening (cortical
  neuroscience, dermatology, garments, RF mapping, hand templates).

Agents B–C–D used live WebSearch + WebFetch where available. Agent B was
inadvertently sandboxed before web tools loaded and produced its
bibliography from established knowledge; URLs in the Classical table
below have NOT been live-verified and must be re-checked before
publication citation.

## Findings A — Codebase state

### What already exists
- **Algorithm catalogue note:** [`note-3d-to-2d-surface-projection-algorithms.md`](note-3d-to-2d-surface-projection-algorithms.md) — 5 options (tangent-plane, exponential map, geodesic MDS, LSCM/ARAP, cylindrical unwrap) with measured tradeoffs.
- **Production projection registry:** `code/src/analysis/receptive_field_mapping/rf_projection.py`
  - `project_tangent_plane()` lines 21–34
  - `project_cylindrical_unwrap()` lines 93–177 — local K=500 PCA cylinder fit, `u = r·θ`, `v = h`
  - `project_to_2d()` dispatcher lines 190–223 — only `{"tangent_plane", "cylindrical_unwrap"}`
- **Production consumers:** `rf_simple_pipeline.py:251–254`, `rf_cluster_visualizer.py:189/194/211/224`
- **Mesh source:** `load_or_build_forearm_mesh()` at `rf_surface_utils.py:23–128` — Ball-Pivoting Algorithm via Open3D, ~10–50k vertices, OBJ-cached
- **Sandbox:** `code/scripts/flatten_forearm_sandbox.py` — LSCM/Harmonic/ARAP via libigl, with optional `center_vid` pinned to UV origin
- **Plans:** `active/flatten-forearm-sandbox.md` (Phases 1–3 done, integration deferred); pending `investigate-rf-projection.md` (tangent-plane diagnostic + centroid-bug fix)

### What is missing
- No exponential map, no geodesic MDS, no modern distortion-bounded method
  (SLIM, BFF, Progressive) is implemented anywhere
- No template-registration path (MANO, SMPL-H, statistical shape model)
- No cross-session RF comparison — every session re-projects with its own
  cylinder fit or rotation matrix, so 2D coordinates are not commensurable
  across sessions

## Findings B — Classical mesh parametrization (foundational, pre-~2014)

URLs reconstructed from memory by Agent B; verify before citation. Author /
year / venue triples are reliable.

| Algorithm | Year | Distortion | Boundary | Implementation |
|---|---|---|---|---|
| **Tutte barycentric embedding** | 1963 | Fixed (no minimisation) | Convex polygon | libigl `igl::map_vertices_to_circle` + `igl::harmonic` |
| **Floater shape-preserving** | 1997 | Fixed boundary, generalised barycentric | Convex polygon | Floater group; libigl |
| **Mean-value coordinates** | Floater 2003 | Fixed boundary | Convex polygon | libigl, CGAL |
| **Harmonic maps** | Eck, DeRose, Duchamp et al. 1995 | Conformal-flavoured fixed-boundary | Unit disk | `igl::harmonic(k=1)` — used in sandbox |
| **DCP — Discrete Conformal Parameterization** | Desbrun, Meyer, Alliez 2002 | Conformal | Free / mixed | OpenMesh, MATLAB |
| **LSCM — Least-Squares Conformal Maps** | Lévy 2002 | Conformal | Free, 2 pinned | `igl::lscm` — used in sandbox |
| **ABF / ABF++** | Sheffer et al. 2001 / 2005 | Angle-based conformal | Free | OpenMesh, custom |
| **MIPS** | Hormann / Greiner 2000 | Most Isometric | Free | Few public refs |
| **ARAP parametrization** | Liu, Zhang, Wang 2008 | As-rigid-as-possible isometric | Free | `igl::arap_*` — used in sandbox |
| **SCP — Spectral Conformal Parameterization** | Mullen et al. 2008 | Conformal (single eigenproblem) | Free | libigl |
| **CETM — Conformal Equivalence of Triangle Meshes** | Springborn, Schröder, Pinkall 2008 | Exact discrete conformal | Cones | MATLAB / C++ |
| **Discrete Ricci flow** | Jin, Kim, Luo, Gu 2008 | Conformal via curvature flow | Any | Research code |
| **Disk conformal map** | Choi & Lui 2014 / 2018 | Conformal | Free, 2-point gauge | MATLAB toolbox |
| **Stretch-minimizing param.** | Sander, Snyder, Gortler, Hoppe 2001 | Isometric (L² / L∞ stretch) | Free | Foundational signal-encoding paper |
| **Iso-charts** | Zhou, Snyder, Guo, Shum 2004 | Isometric multi-chart | Multi-chart | Microsoft Research |
| **Linear ABF** | Zayer et al. 2007 | Angle-based, linear | Free | Custom |
| **MAPS multiresolution** | Lee, Sweldens, Schröder, Cowsar, Dobkin 1998 | Hierarchical mesh-to-disk | Disk | Caltech ref impl |
| **Circle patterns** | Kharevych, Springborn, Schröder 2006 | Conformal | Cones | C++ |
| **Seamless / MIQ** | Bommes, Zimmer, Kobbelt 2009 | Seamless integer-grid | Auto cuts | libigl |

Worth reading at the survey level:
- **Floater & Hormann 2005** — "Surface parameterization: a tutorial and survey"
- **Sheffer, Praun, Rose 2006** — "Mesh parameterization methods and their applications"
- **Hormann, Lévy, Sheffer SIGGRAPH 2007 course** — comprehensive tutorial

**Classical-family recommendation for the hand/forearm problem:** ARAP as
a quick isometric baseline (already in sandbox); conformal options (LSCM,
SCP) only if angle preservation matters more than isometry. None of these
is the **modern** state-of-the-art.

## Findings C — Modern parametrization (~2015–present)

URLs live-verified by Agent C.

| Algorithm | Year | Distortion / objective | Boundary | Implementation |
|---|---|---|---|---|
| **BFF — Boundary First Flattening** | Sawhney & Crane, SIGGRAPH 2017 [arXiv 1704.06873](https://arxiv.org/abs/1704.06873) | Conformal; min-area mode + disk-uniformization | Free; user-prescribed length/angles; cones | C++ ref + **geometry-central** `parameterizeBFF` |
| **OptCuts** | Li, Kaufman et al., SIGGRAPH Asia 2018 [project](https://www.cs.ubc.ca/labs/imager/tr/2018/OptCuts/) | Symmetric Dirichlet with hard distortion bound | Auto cut placement; bijective | `liminchen/OptCuts` (libigl) |
| **SLIM — Scalable Locally Injective Mappings** | Rabinovich, Poranne, Panozzo, Sorkine-Hornung, ACM TOG 2017 [PDF](https://cims.nyu.edu/gcl/papers/SLIM2017.pdf) | Symmetric Dirichlet / isometric / conformal; flip-free | Free + arbitrary positional pins | **libigl** `igl::slim_*`; Python bindings; tutorial #709 |
| **Progressive Parameterizations** | Liu, Ye, Ni, Fu, SIGGRAPH 2018 [project](http://staff.ustc.edu.cn/~fuxm/projects/ProgressivePara/) | Isometric, foldover-free | Disk, free boundary | C++ on project page; benchmarked on 20 712 meshes |
| **AQP — Accelerated Quadratic Proxy** | Kovalsky, Galun, Lipman, SIGGRAPH 2016 | ARAP / isometric / conformal | Free + positional | MATLAB ref impl |
| **Bounded Distortion Maps** | Lipman, SIGGRAPH 2012 [PDF](https://www.wisdom.weizmann.ac.il/~ylipman/BoundedDistortion/bounded_distortion_may_1.pdf) | Provable conformal-distortion bound + injectivity | Free + positional | MATLAB; foundation of later proxies |
| **Foldover-free Maps in 50 Lines** | Garanzha et al., SIGGRAPH 2021 [arXiv 2102.03069](https://arxiv.org/abs/2102.03069) | Local injectivity + low isometric distortion | Boundary fixed | 50-line C++; `ultimaille` |
| **Efficient Bijective Parameterizations** | Su, Ye, Liu, Fu, SIGGRAPH 2020 | Symmetric Dirichlet, globally bijective | Free | C++ at USTC GCL |
| **Scaffold** | Jiang, Schaefer, Panozzo, ACM TOG 2017 | Symmetric Dirichlet / ARAP, globally bijective | Free + chart-packing | libigl-adjacent |
| **CEPS — Discrete Conformal Equivalence of Polyhedral Surfaces** | Gillespie, Springborn, Crane, ACM TOG 2021 [repo](https://github.com/MarkGillespie/CEPS) | Exact discrete conformal, locally injective | Any topology; user cones | Open-source C++ |
| **Efficient & Robust Discrete Conformal Equivalence with Boundary** | Campen, Capouellez et al., ACM TOG 2021 [PDF](https://cims.nyu.edu/gcl/papers/2021-Conformal.pdf) | Exact discrete conformal with boundary | Boundary curvature + cones | C++ |
| **Seamless w/ Arbitrary Cones** | Campen et al., ACM TOG 2019 | Seamless / integer-grid | Auto cuts | NYU/GCL C++ |
| **Penner-coordinate Seamless** | Capouellez, Zorin et al., ACM TOG 2024 | Seamless, locally injective, convex feasible set | Boundary, cones | Research release |
| **Hyperbolic Orbifold Tutte** | Aigerman & Lipman, SIGGRAPH Asia 2016 [repo](https://github.com/noamaig/hyperbolic_orbifolds) | Conformal-flavoured, hyperbolic | Cones replace free boundary | MATLAB |
| **Choi free-boundary disk conformal** | Choi & Lui, JSC 2022 | Conformal | Free, 2-pt gauge | MATLAB by G. Choi |
| **Quasi-conformal / Beltrami-coefficient methods** | Lui, Choi et al. (ongoing) | Quasi-conformal with bounded Beltrami norm | Boundary + landmarks | Lui group MATLAB |
| **Neural Surface Maps** | Morreale, Aigerman, Kim, Mitra, CVPR 2021 [arXiv 2103.16942](https://arxiv.org/abs/2103.16942) | Differentiable conformal / isometric | Boundary via pretrained chart | PyTorch |
| **Neural Jacobian Fields** | Aigerman et al., SIGGRAPH 2022 [arXiv 2205.02904](https://arxiv.org/abs/2205.02904) | Symmetric Dirichlet / ARAP via differentiable Poisson | Boundary via chart; pins via loss | PyTorch |
| **Flatten Anything Model / ParaPoint** | Zhang, Hou, Tan et al., NeurIPS 2024 [arXiv 2405.14633](https://arxiv.org/abs/2405.14633) | Free-boundary, distortion-loss driven | Auto cuts | PyTorch |
| **Nuvo** | Srinivasan et al. 2024 [project](https://pratulsrinivasan.github.io/nuvo/) | Distortion-balanced multi-chart | Auto segmentation | Reference impl |
| **Advancing Front Mapping** | Livesu et al. 2023 [arXiv 2305.11552](https://arxiv.org/html/2305.11552) | Guaranteed local injectivity | Boundary = target polygon | `cinolib` C++ |

**Modern-family recommendation for the hand/forearm problem:**

1. **SLIM** (symmetric-Dirichlet) — primary. Native support for arbitrary
   positional pins (matches the centre-vertex constraint already in the
   sandbox); libigl Python bindings; flip-free guaranteed. Initialise from
   a Tutte/harmonic embedding.
2. **BFF (min-area mode)** — strong conformal alternative. Apply a 2D
   similarity afterwards to land the chosen interior vertex at origin and
   a reference direction along +x.
3. **Progressive Parameterizations** — fallback when SLIM stalls on bad
   triangulations.

**Deliberately not recommended** for this problem:
- OptCuts (introduces cuts on an already-open patch)
- Neural methods (training overhead, no provable injectivity)
- CETM / CEPS / seamless families (oriented toward quad meshing)

## Findings D — Biomedical / applied surface flattening

The mature applied domain is **cortical surface flattening** for
neuroimaging — it has thirty years of head start on us.

| Method | Domain | Reference | Implementation |
|---|---|---|---|
| **FreeSurfer `mris_flatten`** | Cortical surface | Fischl, Sereno, Dale 1999 [paper](https://www.sciencedirect.com/science/article/abs/pii/S1053811998903962) | Open source [wiki](https://surfer.nmr.mgh.harvard.edu/fswiki/mris_flatten) |
| **CARET / Connectome Workbench** | Cortical surface | Van Essen lab, 2012 review [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3288593/) | Open source |
| **PALS-B12 atlas** | Cortical surface | Van Essen 2005 [PubMed](https://pubmed.ncbi.nlm.nih.gov/16172003/) | Canonical "register-everyone-to-one-template" precedent |
| **BrainSuite SVReg** | Cortical surface | [Project](https://brainsuite.org/processing/svreg/details/) | Open source |
| **SUMA / AFNI** | Cortical surface | [Project](https://afni.nimh.nih.gov/Suma) | Open source |
| **Pycortex** | Cortical surface | Gao et al. 2015 [Frontiers](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2015.00023/full) | Sparse-matrix pixel-wise flat-map renderer |
| **Circle packing flattening** | Cerebellum | Hurdal & Stephenson 2009 [PubMed](https://pubmed.ncbi.nlm.nih.gov/19049882/) | Closest published non-cortex precedent |
| **Garment / developable approximation** | Apparel | Wang 2002, Decaudin 2006 | Wrong philosophy — assumes developable surface |
| **CHOIR / Mancini 2014 body-maps** | Pain mapping | Mancini 2014 [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4143958/) | Hand-drawn 2D; no flattening involved |
| **TouchSim** | Tactile RF simulation | Saal/Bensmaia 2017 [PNAS](https://www.pnas.org/doi/abs/10.1073/pnas.1704856114) | Open source [BensmaiaLab/touchSim](https://github.com/BensmaiaLab/touchSim) — **directly relevant precedent**, canonical 2D glabrous-hand template |
| **Vallbo / Johansson hand drawings** | Tactile microneurography | Johansson 1983 | Hand-drawn 2D outline |
| **Wessberg / Olausson CT afferents** | Hairy-skin microneurography | Wessberg 2003 [PubMed](https://pubmed.ncbi.nlm.nih.gov/12626628/) | Local 2D probe coordinates only |
| **MANO** | Hand shape model | Romero 2017 [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de/) | Parametric mesh with manually-authored UV |
| **SMPL-X / SMPL+H** | Full-body + hand | Pavlakos 2019 [vchoutas/smplx](https://github.com/vchoutas/smplx) | Embeds MANO |
| **Handy** | Hand shape model | Potamias 2023 [Project](https://rolpotamias.github.io/Handy/) | Scales MANO to 1200 subjects |
| **van de Giessen articulating hand SSM** | Hand SSM | [Springer chapter](https://link.springer.com/chapter/10.1007/978-3-319-94223-0_41) | Skeleton + canonical-pose alignment |

**Biomedical-family recommendation:**
- **MANO** (or **Handy**) + per-vertex UV from the template — the "register,
  don't re-flatten" philosophy that PALS taught cortical neuroscience.
  Inter-session comparison becomes automatic.
- **TouchSim** is the closest published precedent: it already plots
  simulated tactile afferents on a canonical 2D hand template, exactly the
  workflow the project would inherit.

## Synthesis — three philosophical tracks

The agents converge on three distinct strategies. They are not mutually
exclusive; they answer different questions.

### Track A — "Register, don't re-flatten" (template-based)

**Strategy.** Fit MANO (or Handy) to each session's hand mesh once.
Receptive-field locations are sampled at the nearest MANO vertex and
inherit the template's authored (u, v). Plotting is just look-up.

**Pros.**
- Cross-session RF comparison is automatic (everyone shares one UV).
- The template's UV is hand-authored — the 2D image actually looks like a
  hand outline, publishable directly.
- No per-session flattening to debug.
- Matches both PALS (cortical) and TouchSim (tactile) precedents.

**Cons.**
- Requires a robust MANO fit per session (the existing pipeline already
  does MANO hand tracking — leverage that).
- Loses session-specific anatomy: if subject's hand differs from MANO
  mean, RFs are projected onto a "best fit" that may smear true location.
- Forearm is not in MANO — needs extension or a separate forearm SSM
  (van de Giessen-style).

### Track B — Per-session flattening (modern parametriser)

**Strategy.** Replace the LSCM/Harmonic/ARAP baseline in
`flatten_forearm_sandbox.py` with **SLIM (symmetric-Dirichlet)** plus the
existing centre-vertex pin mechanism. Initialise from harmonic; pin
centre vertex to `(0,0)` and one boundary vertex to `(1,0)` for
orientation; iterate until convergence.

**Pros.**
- Native interior-vertex pin via SLIM's positional constraints API.
- Provably flip-free; lower isometric distortion than LSCM/ARAP.
- Already in libigl — minimal new dependency surface.
- Per-session anatomy preserved exactly.

**Cons.**
- 2D coordinates remain per-session (no cross-session comparison).
- Need a principled choice of centre vertex (anatomical landmark? centroid?
  user-picked?).
- Doesn't address "what does the publication figure look like?" — output
  is a blob shaped like that session's forearm.

### Track C — Keep simple baselines + add diagnostics

**Strategy.** The existing knowledge-base note already documents that for
20–60 mm patches the tangent-plane projection is sufficient (<1 ms,
negligible distortion). Add the **exponential map** as a second option
(already documented, not implemented). Keep `cylindrical_unwrap` for the
whole-forearm view. Ship diagnostic plots that quantify per-session
distortion.

**Pros.** Cheapest path. No new dependencies, no new algorithms. Closes
the pending `investigate-rf-projection.md`.

**Cons.** Doesn't unlock cross-session comparison or
publication-quality figures. Doesn't justify the work already done in the
sandbox.

## Recommendation

**Two-stage path, ordered by reversibility:**

1. **First, verify the sandbox's premise** with a small experimental
   study on 1–2 sessions: render LSCM, ARAP, harmonic, and (newly added)
   SLIM with the centre-pin, alongside the existing tangent-plane /
   cylindrical-unwrap baselines. Use the sandbox's per-face distortion
   plots (`compute_face_distortion`, lines 633–687 of
   `flatten_forearm_sandbox.py`) to quantify each. This is a small,
   reversible experiment with no new install requirements — SLIM is
   already part of libigl's Python bindings.

2. **Then choose between Track A and Track B for production** based on
   what that study shows. The literature strongly suggests **Track A
   ("register to MANO") is the right long-term answer** — it matches the
   PALS precedent in cortical neuroscience and the TouchSim precedent in
   tactile RF mapping — but **Track B is the smaller engineering step** and
   keeps per-session anatomy intact. Re-evaluate after stage 1.

The "do nothing new" Track C is a fine fallback if stage 1 results are
ambiguous or downstream stakeholders do not demand more.

## Open questions / verification needed

- **Classical-literature URLs:** Agent B's bibliography is reconstructed
  from established knowledge — re-verify every URL with WebSearch before
  citing in a paper.
- **MANO availability of forearm:** confirm whether the pipeline's existing
  MANO fits cover the forearm or only the wrist-and-hand. If they don't,
  Track A needs a forearm SSM addition.
- **Centre-vertex semantics:** Track B leaves open the question of what
  the user-picked centre vertex should anatomically represent. Defining it
  (e.g. dorsal midpoint between MCP joints) would make per-session SLIM
  outputs roughly comparable.
- **Geodesic-vs-2D distance check:** before any choice is committed,
  measure geodesic distance on the 3D mesh vs Euclidean distance in the 2D
  flattened image for ~50 random pairs per session. Target relative error
  <5 % for short geodesics, <15 % for long ones.

## Stage-1 results (2026-05-16)

Script: `code/scripts/compare_flattening_methods.py`

### What was run
Six methods compared on session `2022-06-17_ST16-05` forearm mesh:
LSCM, ARAP, Harmonic, SLIM (symmetric-Dirichlet, 40 iterations), TangentPlane (PCA rotation), Cylindrical (PCA rotation).

### Artefacts
Generated next to the input PLY (path: see `PLY_PATH` constant in the script):
- `*_compare_{timestamp}.png` — 7-panel comparison figure (3D + one UV per method)
- `*_compare_{timestamp}_distortion.png` — 6-column × 2-row distortion maps
- `*_compare_{timestamp}_summary.csv` — per-method conformal/area distortion statistics

### Results summary
Results are recorded in the CSV artefacts above. To be updated after running on session `2022-06-16_ST15-01` for cross-session ranking consistency check.

### Track recommendation (pending cross-session run)
Stage-1 comparison to be reviewed before committing to Track A or Track B. If SLIM's `conformal_p95` is ≤ the other three mesh parametrisers (as the literature predicts), proceed with Track B ("per-session SLIM") as the production pivot. Track A ("register to MANO") remains the longer-term goal.

## Cross-references

- Algorithm-only catalogue: [note-3d-to-2d-surface-projection-algorithms.md](note-3d-to-2d-surface-projection-algorithms.md)
- Why CARET isn't the right import: [investigation-caret-pals-for-forearm-flattening.md](investigation-caret-pals-for-forearm-flattening.md)
- Sandbox implementation: `code/scripts/flatten_forearm_sandbox.py`
- Compare script: `code/scripts/compare_flattening_methods.py`
- Active plan: `docs/development/plans/active/flatten-forearm-sandbox.md`
- Pending diagnostic plan: `docs/development/plans/pending/investigate-rf-projection.md`
