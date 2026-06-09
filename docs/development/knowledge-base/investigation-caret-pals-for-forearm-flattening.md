# Investigation: Caret/PALS Cortical Flattening — Applicability to Forearm Surface Mapping

**Date:** 2026-05-15
**Status:** Investigation — exploratory, no implementation decided
**Trigger:** Question raised during planning: could the Caret/PALS
surface-based cortical flattening approach be adapted to replace the current
cylindrical unwrap used in receptive-field mapping?
**Related note:** [note-3d-to-2d-surface-projection-algorithms.md](note-3d-to-2d-surface-projection-algorithms.md) — algorithm catalogue covering the same projection problem from a distortion/algorithm angle.

---

## Question

The receptive-field-mapping pipeline projects spike-tagged 3D contact points
on the forearm onto a 2D heatmap. The current method (`cylindrical_unwrap`)
is a session-local PCA-axis cylinder unwrap. The question:

1. What does Caret/PALS-style surface-based cortical flattening provide that
   our current cylindrical unwrap does not?
2. Could it (or its modern algorithmic descendants) be applied to forearm
   surface mapping?

## Method

Two parallel investigations:
- **A.** Web research on Caret/PALS, its algorithmic core, preconditions,
  and modern successors.
- **B.** Code audit of the current 3D→2D projection in
  `code/src/analysis/receptive_field_mapping/`.

## Findings A — Caret/PALS (web)

**What it is.** Caret (Van Essen Lab) is brain-mapping software; PALS
(Population-Average, Landmark- and Surface-based) is the associated
population atlas. Pipeline: **inflation → cutting (medial-wall seam) →
flattening → landmark-based registration to atlas**.

**Status.** Caret is no longer actively developed. Successor is **Connectome
Workbench** (HCP); modern atlas is **Conte69**. The same flattening
algorithms live in **FreeSurfer `mris_flatten`** and general-purpose mesh
libs (libigl LSCM/ARAP, geometry-central).

**Algorithmic core.** Conformal mapping (angle-preserving, e.g. LSCM),
metric/spring-mass relaxation (FreeSurfer style), multi-resolution
landmark-constrained smoothing. Topological precondition: input must be a
manifold genus-0 surface, with an explicit seam cut so the closed surface
becomes a simply-connected disk.

**Inputs / preconditions.** Watertight manifold cortical mesh (~150k
vertices), anatomical landmarks (sulci/gyri) for atlas registration, the
PALS/Conte69 atlas as registration target.

**Outputs.** Per-vertex 2D coordinates, registration map to the population
atlas, per-vertex distortion maps (angular and metric).

**Applicability outside cortex.** No mainstream precedent for Caret on
limbs/skin, but the underlying algorithms (LSCM, ARAP, conformal, Ricci
flow, harmonic) are topology-agnostic. Medical-imaging precedent in colon
flattening and surgical-simulation skin meshes. For a forearm:
- It is roughly cylindrical, not genus-0 as captured — needs an explicit
  topological cut (long axis, or wrist/elbow) to become flattenable.
- No sulci/gyri → automatic landmark detection fails. Manual fiducials
  (wrist crease, elbow, ulnar seam) would be required for cross-subject
  registration.

**Sources.**
- Caret overview — https://pmc.ncbi.nlm.nih.gov/articles/PMC3288593/
- PALS atlas — https://www.sciencedirect.com/science/article/abs/pii/S1053811905004945
- Flattening method comparison — https://surfer.nmr.mgh.harvard.edu/ftp/articles/ju_l_28p869_2005.pdf
- LSCM for cortex — https://www.math.fsu.edu/~mhurdal/papers/isbi2004/isbi2004_prepreint.pdf
- Connectome Workbench — https://www.humanconnectome.org/software/connectome-workbench
- libigl parameterization — https://libigl.github.io/tutorial/

## Findings B — current projection in this codebase

**Where it lives.**
- Primary: `code/src/analysis/receptive_field_mapping/.../rf_projection.py`
  - `project_cylindrical_unwrap()` at rf_projection.py:90–177
  - `project_to_2d()` dispatcher at rf_projection.py:190–223
  - Cylinder-axis fit `fit_cylinder_axis()` at rf_projection.py:37–66
- Callers: `rf_cluster_visualizer.py:189`, `rf_simple_pipeline.py:238`
- Render: `rf_2d_renderer.py:80–316`
- DAG: `configs/analyse_workflow_processing_dag.yaml` (default
  `cylindrical_unwrap` at line 33; legacy `map_receptive_fields_clustered`
  at line 363)

**Input.** Per-contact `PopulationData`
(`touch_population_data.py:48–85`):
- `cp_vertex_idx` (N,) — forearm-mesh vertex indices, snapped via KDTree
  with a 15 mm radius
- `cp_spike` (N,) — boolean spike mask
- 3D coords: `forearm_vertices[cp_vertex_idx]` (mm, RF-centred)
- Forearm mesh: PLY point cloud, ~2–10k vertices

**Projection method.** Cylindrical unwrap. Fit a cylinder to the local
neighbourhood around the contact centroid: SVD on K=500 nearest neighbours
→ principal axis; mean radius from radial distances. If camera settings
exist, axis/up/seam come from the camera rotation matrix; else legacy PCA
reference frame. Per-point transform: height `v = (p − c)·axis`,
`θ = atan2(r_y, r_x)`, arc length `u = r·θ`. Output `(u, v)` in mm.

**Output.** (N, 2) scatter, interpolated onto a fixed 100×100 cubic grid
via `scipy.interpolate.griddata` (`rf_2d_renderer.py:227`). u ≈ 50–80 mm,
v ≈ 100–150 mm. One heatmap per (session, cluster).

**Population mapping.** None. Each session is independent. Cylinder axis
and radius are refit locally per neuron. No atlas, no template, no
cross-subject alignment.

**Visualisation.** matplotlib via `rf_2d_renderer.render_2d_heatmap()`
(scatter + interpolated heatmap, convex-hull overlay). 3D fallback via
PyVista offscreen. Interactive GUIs:
`gui/rf_cluster_gallery_viewer.py`, `gui/touch_playback_explorer.py`.

**Mesh availability (verified).** A forearm mesh **is** produced upstream
by `code/scripts/_3_preprocessing/_3_forearm_extraction/define_forearm_mesh.py:36–179`
(2.5D Delaunay, OBJ in `{session_processed_path}/forearm_pointclouds/`),
but the surface is **not watertight** — it is a topographic patch with
open boundary. The RF pipeline currently **ignores** this file and
rebuilds via Ball Pivoting on-the-fly
(`rf_surface_utils.py:23–128`, cached as `{stem}_mesh_bpa.obj`).

## Side-by-side comparison

| Dimension | Current (cylindrical unwrap) | Caret/PALS family |
|---|---|---|
| **Input geometry** | Sparse PLY point cloud, ~2–10k pts | Dense manifold mesh, ~150k verts |
| **Topology assumed** | Local cylinder (open) | Genus-0 closed surface + explicit cut |
| **Math** | PCA axis + `atan2`, arc length | LSCM / ARAP / metric relaxation |
| **Distortion handling** | None — grows away from seam | Explicitly minimised (angle or area) |
| **Anatomical landmarks** | Camera rotation, contact centroid | Sulci/gyri (cortex) or manual fiducials |
| **Cross-subject registration** | None — per-session map | Native — registered to atlas |
| **Mesh requirements** | Tolerates point clouds | Needs watertight manifold mesh |
| **Compute cost** | Cheap, closed-form | Iterative, minutes per surface |
| **Distortion metrics** | Not exposed | Per-vertex angular & metric maps |
| **Maintained tooling** | In-house, simple | libigl / FreeSurfer / Workbench |
| **Output 2D coords** | Arc-length × axial mm | Atlas-registered 2D mm |
| **Per-trial vs population** | Per-(session, cluster) only | Population atlas built in |

### Where they diverge most
1. **Topology & data.** Cylindrical unwrap accepts a point cloud and a
   locally cylindrical assumption; Caret needs a proper closed manifold
   mesh and an explicit seam cut.
2. **Distortion.** Cylindrical unwrap accepts whatever distortion arises
   from non-constant radius; Caret methods minimise it with quantified
   per-vertex error maps.
3. **Population alignment.** Biggest qualitative gap. Caret/PALS exists
   precisely to make subjects comparable in a common 2D frame; the
   current code has no analogue.
4. **Anatomical anchoring.** Cortex has sulci as natural landmarks; the
   forearm has none, so any Caret-style port needs manual fiducials.

## Are anatomical landmarks required to unwrap a watertight mesh?

**No, not for single-subject unwrap.** Landmarks are only required for
cross-subject correspondence (atlas-style).

| Goal | Landmarks needed? | What is actually needed |
|---|---|---|
| Flatten single closed (genus-0) mesh | No anatomical landmarks | A seam cut (geometric) + 2 fixed vertices for LSCM gauge |
| Flatten disk-topology patch | No anatomical landmarks | 2 fixed boundary vertices for LSCM |
| Compare maps across sessions (PALS-style) | **Yes** | Anatomical correspondences (wrist/elbow/ulnar seam) |

The "fixed vertices" LSCM needs are a **mathematical** gauge to remove
translation/rotation/scale freedom; they have no anatomical meaning.

For the forearm specifically:
- The existing Delaunay patch is already disk-topology → LSCM works
  directly with no landmarks.
- A closed cylinder (wrist→elbow) needs a seam, but the seam can be
  derived geometrically (e.g. geodesic from wrist to elbow along the
  ulnar side).
- Only atlas-style cross-subject pooling forces manual landmark
  annotation.

## Where this overlaps with existing notes

- [note-3d-to-2d-surface-projection-algorithms.md](note-3d-to-2d-surface-projection-algorithms.md)
  covers the algorithm catalogue (tangent-plane, exponential map,
  geodesic MDS, LSCM, ARAP, cylindrical unwrap) with distortion analysis
  and Python library recommendations. **This investigation complements
  that note** by framing the question in terms of the Caret/PALS
  philosophy (population atlas, anatomical landmarks, cortical lineage)
  rather than algorithm-by-algorithm distortion trade-offs.

## Status / next concrete step

This is investigation-only — no implementation decision has been made.
When/if action is taken, the most natural first step would be a
proof-of-concept LSCM flatten of the existing Delaunay patch using
libigl (`igl.lscm(vertices, faces, b, bc)` — see the existing
algorithm-catalogue note for the call signature), overlay the same RF
heatmap, and compare distortion against the current cylindrical unwrap.
No atlas work in scope at that stage.
