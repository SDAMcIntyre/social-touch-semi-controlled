# Report: SLIM UV Mesh Cleaning Steps and Their Session Origins

**Date:** 2026-05-21
**Author:** Basil Duvernoy

---

## Purpose

This report documents the eight sequential mesh cleaning steps in `clean_mesh()`
(`surface/slim_helpers.py:579`), explains why each exists, traces each back to the
recording session(s) that exposed the need for it, and explains why no step can be
removed or merged with another.

---

## Background

The SLIM (Scalable Locally Injective Mappings) parameterization unfolds a 3D forearm
mesh into a 2D UV map for receptive-field analysis. SLIM requires its input mesh to be:

1. **Manifold** — every edge shared by exactly 2 faces
2. **Single-boundary** — exactly one boundary loop (disk topology, Euler chi = 1)
3. **Well-conditioned** — no degenerate (zero-area or extreme aspect-ratio) faces

The input meshes come from Ball Pivoting Algorithm (BPA) reconstruction on Kinect
point clouds, which routinely violates all three properties. The eight cleaning steps
systematically remove each class of violation.

---

## The Eight Steps

### Step 1 — Keep largest connected component

```
slim_helpers.py:610–617
Origin: feature/slim-forearm-projection (2026-05-18)
Sessions: all 12
```

BPA reconstruction on partial point clouds produces satellite mesh fragments
disconnected from the main forearm surface. `trimesh.split(only_watertight=False)`
partitions the mesh into connected components; only the largest (by vertex count) is
retained.

**Affected sessions:** Universal — all 12 sessions have at least minor disconnected
fragments from point cloud noise at scan boundaries.

---

### Step 2 — Fix face winding

```
slim_helpers.py:620
Origin: feature/slim-forearm-projection (2026-05-18)
Sessions: all 12
```

`trimesh.repair.fix_winding()` ensures consistent outward normal orientation across
all faces. BPA produces mixed clockwise/counter-clockwise face winding at seam edges
where the ball pivoting algorithm reverses direction.

**Why it matters:** SLIM and harmonic-map initialization compute signed triangle areas.
Inconsistent winding causes sign flips that break the bijective parameterization
guarantee (Rado-Kneser-Choquet theorem).

**Affected sessions:** Universal — winding inconsistencies are inherent to BPA on
partial scans.

---

### Step 3 — Remove orphan vertices

```
slim_helpers.py:631
Origin: feature/slim-forearm-projection (2026-05-18)
Sessions: all 12
```

`trimesh.remove_unreferenced_vertices()` discards vertices not referenced by any face.
Component selection (step 1) and winding repair (step 2) can leave vertices without
associated faces.

**Why it matters:** Orphan vertices waste memory and can produce indexing errors in
libigl's cotangent-weight Laplacian solver (harmonic initialization).

**Affected sessions:** Universal.

---

### Step 4 — Remove non-manifold edges (Open3D)

```
slim_helpers.py:641–651
Origin: feature/slim-forearm-projection (2026-05-18)
Sessions: all 12 (critical for ST13-02, ST14-02, ST14-04, ST16-03)
```

Four Open3D operations in sequence:
1. `remove_duplicated_vertices()` — merge identical vertex positions
2. `remove_duplicated_triangles()` — remove duplicate faces
3. `remove_degenerate_triangles()` — drop zero-area triangles
4. `remove_non_manifold_edges()` — remove faces where an edge is shared by 3+ faces

BPA frequently produces non-manifold edges at seam overlap regions. libigl's LSCM,
harmonic, and ARAP solvers require strictly manifold edge topology.

**Affected sessions:** All 12 have some non-manifold edges. The effect is most
pronounced in sessions where BPA seams overlap heavily:

| Session | Non-manifold faces removed | Interior holes created |
|---------|---------------------------|----------------------|
| ST13-02 | significant | 4 holes (sizes 5, 8, 9, 13) |
| ST14-02 | significant | 10 holes (thumb-forearm junction) |
| ST14-04 | significant | 13 holes |
| ST16-03 | minor | 1 hole |
| Others | minor | 0 holes |

**Side effect:** Removing non-manifold faces can punch interior holes in the mesh,
breaking disk topology. Steps 7 and 8 address this.

---

### Step 5 — Pinch-vertex repair

```
slim_helpers.py:670–704
Origin: fix(slim-forearm-projection) (2026-05-18, commit e64e6b9)
Sessions: ST13-02, ST14-01, ST14-04 (primarily)
```

A "pinch" vertex appears as the source of 2+ boundary edges — it is a vertex where the
boundary self-touches, making the mesh locally non-manifold at the boundary. These arise
from inconsistent face winding near BPA seam edges that survive step 4.

The repair iterates up to 10 passes:
1. Find pinch vertices via `igl.boundary_facets` source-count analysis
2. Remove all faces adjacent to pinch vertices
3. Re-take the largest connected component
4. Repeat until no pinch vertices remain

**Why iterative:** Removing faces around one pinch vertex can expose a previously hidden
pinch vertex underneath.

**Affected sessions:**

| Session | Pinch passes | Faces removed |
|---------|-------------|---------------|
| ST13-02 | 1 | ~8 |
| ST14-01 | 1 | ~4 |
| ST14-04 | 2 | ~12 |
| Others | 0 | 0 (no pinch vertices) |

**Why it can't be merged with step 4:** Pinch vertices are *created* by step 4's
non-manifold edge removal. They cannot be detected before step 4 runs.

---

### Step 6 — Sliver face removal (aspect ratio > 10)

```
slim_helpers.py:706–715, function at 499–549
Origin: feature/slim-uv-delaunay-hole-filling (2026-05-21, commit d1940bc)
Sessions: ST14-02 (primarily), benefits all sessions
```

BPA produces elongated "sliver" triangles at mesh boundaries where point-cloud density
drops off. These are faces with `longest_edge / shortest_edge > 10.0`.

**Effect on SLIM:** Sliver triangles cause:
- Ill-conditioned Hessian matrices during SLIM iteration (slow/no convergence)
- Conformal distortion hotspots in the UV map
- Tutte flip failures that trigger expensive trim-retry loops

The function removes all faces exceeding the aspect-ratio threshold, then takes the
largest connected component and removes orphan vertices.

**Discovery context:** ST14-02 had 10 interior holes with 178 centroid-fan triangles
from hole filling (old approach). Many of these fan triangles were extreme slivers.
Investigation traced the problem upstream — BPA itself was producing slivers even
before hole filling. Removing BPA slivers *before* hole filling allows the Delaunay
filler (step 8) to patch gaps with well-conditioned replacement triangles.

**Affected sessions:**

| Session | Sliver faces removed | Notes |
|---------|---------------------|-------|
| ST14-02 | ~30 | Thumb-forearm boundary region |
| ST18-04 | ~15 | Boundary region near wrist |
| Others | 0–5 | Minor edge cleanup |

**Why it must come before steps 7-8:** Sliver removal can create new boundary gaps and
holes. Steps 7 (stitching) and 8 (hole filling) handle these downstream effects.

---

### Step 7 — Boundary-gap stitching

```
slim_helpers.py:722–733, function at 351–496
Origin: feature/slim-uv-boundary-gap-stitching (2026-05-21, commit 65ae562)
Sessions: ST14-02 (critical fix), ST13-01 (regression guard)
```

This step distinguishes between two types of secondary boundary loops:

- **Junction gaps:** Narrow strips of missing faces where an appendage (thumb, finger)
  connects to the forearm body. Their boundary vertices are geometrically close to the
  main boundary.
- **True interior holes:** Missing faces deep inside the mesh (from non-manifold
  removal). Their boundary vertices are far from the main boundary.

Junction gaps must be *stitched* (welded back to the main boundary). True interior
holes must be *filled* (step 8). If junction gaps are filled instead of stitched, the
appendage perimeter is converted from boundary to interior, and the main boundary loop
bypasses the appendage entirely.

**Algorithm:**
1. Build KDTree of main-boundary vertex positions
2. For each secondary boundary loop, query nearest main-boundary vertex for each loop vertex
3. If >= 2 vertices are within 5.0 mm (`_BOUNDARY_GAP_PROXIMITY_MM`) AND the ratio of
   close vertices to total loop vertices < 0.25 (`_BOUNDARY_GAP_MAX_CLOSE_RATIO`):
   classify as junction gap and weld
4. Replace close secondary vertex IDs with their nearest main-boundary counterpart in
   the face array
5. Remove degenerate faces (where 2+ vertices collapsed to the same ID)
6. Fix winding, take largest component
7. Repeat up to 3 passes (welding can reveal new boundary configurations)

**Discovery context — ST14-02:** The thumb-forearm junction had 10 secondary boundary
loops created by non-manifold/pinch/sliver removal. Without stitching, hole filling
enclosed the thumb, producing:
- Boundary loop that skipped the thumb entirely (visible in step 3 diagnostic)
- Conformal distortion sigma1/sigma2 >> 3.0 (vs ~1.175 in correct sessions like ST14-01)
- Flipped triangles in harmonic init, triggering Tutte fallback + mesh trimming

**Regression — ST13-01:** The first implementation had no ratio guard. ST13-01 had two
tiny secondary loops where *most* vertices were close to the main boundary (high close
ratio). These were incorrectly classified as junction gaps and welded, fragmenting the
mesh and producing 293 flipped triangles. The `close_ratio < 0.25` guard was added to
prevent false positives on small boundary artifacts.

**Affected sessions:**

| Session | Secondary loops | Stitched | Notes |
|---------|----------------|----------|-------|
| ST14-02 | 10 | yes (all junction gaps) | Thumb-forearm junction |
| ST13-01 | 2 | no (ratio guard blocks) | Small artifacts, not junction gaps |
| Others | 0 | no-op | No secondary loops to process |

**Why it must come before step 8:** Once a junction gap is filled by step 8, the
distinction between gap and hole is lost. The appendage perimeter becomes interior, and
no post-hoc fix can recover it.

---

### Step 8 — Interior hole filling

```
slim_helpers.py:738–748, function at 268–336
Origin: fix(slim-forearm-projection) (2026-05-18, commit e64e6b9)
Upgraded: feature/slim-uv-delaunay-hole-filling (2026-05-21, commit d1940bc)
Sessions: ST13-02, ST14-02, ST14-04, ST16-03 (required), ST18-04 (benefits)
```

After steps 4-7, the mesh may still have interior holes (non-largest boundary loops)
that break the single-boundary disk topology required for harmonic/Tutte UV
initialization. This step fills all remaining non-largest boundary loops.

**Algorithm (iterative, up to 5 passes):**
1. Find all boundary loops via `_find_boundary_loops()`
2. Identify the largest loop (real outer boundary)
3. For each smaller loop:
   - If > 4 vertices: use Delaunay hole filling (`_fill_hole_delaunay`)
   - If <= 4 vertices: use centroid-fan triangulation
4. Fix winding
5. Repeat until only one boundary loop remains

**Delaunay hole filling** (`_fill_hole_delaunay`, line 105):
- Projects hole-boundary vertices onto best-fit plane (SVD)
- Runs `scipy.spatial.Delaunay` in 2D on projected points
- Filters simplices whose centroid falls outside the boundary polygon
- For large holes (diameter > 3x median boundary edge length): inserts a regular grid
  of Steiner points at median spacing to maintain triangle size uniformity
- Falls back to centroid-fan if plane is degenerate (nearly collinear boundary)

**Why Delaunay replaced centroid-fan:** The original centroid-fan approach (2026-05-18)
inserted a single centroid vertex and fanned triangles around the loop. For large holes,
this created extreme aspect-ratio "starburst" triangles. ST14-02 with 10 holes generated
178 fan triangles, many with aspect ratios >> 10, degrading SLIM convergence and creating
distortion hotspots.

**Affected sessions:**

| Session | Holes | Euler chi before | Euler chi after | Method |
|---------|-------|------------------|-----------------|--------|
| ST13-02 | 4 (sizes 5, 8, 9, 13 verts) | -4 | 1 | Delaunay (3) + fan (1) |
| ST14-02 | 10 (after stitching: remaining true holes) | -5 | 1 | Delaunay |
| ST14-04 | 13 | -16 | -3 (genus) | Delaunay (10) + fan (3) |
| ST16-03 | 1 | 0 | 1 | Delaunay |
| Others | 0 | 1 | 1 | No-op |

---

## Step Dependencies and Ordering Constraints

The eight steps form a strict causal chain. Each step can create defects that subsequent
steps handle, and no step can be reordered or merged without breaking the pipeline.

```
Step 1 (largest component)
  │ creates: orphan vertices from dropped fragments
  ▼
Step 2 (fix winding)
  │ creates: (none — pure repair)
  ▼
Step 3 (remove orphans)
  │ creates: (none — pure cleanup)
  ▼
Step 4 (non-manifold removal)
  │ creates: pinch vertices, interior holes, exposed slivers
  ▼
Step 5 (pinch repair)
  │ creates: additional small holes (faces removed around pinch vertices)
  ▼
Step 6 (sliver removal)
  │ creates: new boundary gaps and holes (faces removed)
  ▼
Step 7 (boundary-gap stitching)
  │ creates: degenerate faces removed, loops merged
  │ passes through: true interior holes (not stitched)
  ▼
Step 8 (interior hole filling)
  │ creates: disk topology (single boundary loop, chi = 1)
  ▼
  Output → harmonic/Tutte init → SLIM
```

**Why no step can be removed:**

| Removal candidate | Failure mode |
|-------------------|-------------|
| Skip step 1 | SLIM receives multiple disconnected components; undefined behaviour |
| Skip step 2 | Mixed face winding; signed triangle areas inconsistent; SLIM bijectivity violated |
| Skip step 3 | Orphan vertices cause indexing errors in libigl Laplacian solver |
| Skip step 4 | Non-manifold edges; libigl harmonic/ARAP/LSCM crash or produce garbage |
| Skip step 5 | Pinch vertices at boundary; LSCM fails with "singular matrix" error |
| Skip step 6 | Slivers degrade SLIM convergence; Tutte flips trigger trim-retry cascade |
| Skip step 7 | Junction gaps filled as holes; appendages enclosed; extreme distortion |
| Skip step 8 | Multiple boundary loops; harmonic/Tutte init violates disk topology assumption |

**Why no steps can be merged:**

| Merge candidate | Reason it fails |
|-----------------|----------------|
| Steps 1–4 into one Open3D call | Open3D `remove_non_manifold_edges` requires single-component input; step 1 must run first |
| Steps 4+5 | Pinch vertices are created by step 4; can't detect them before step 4 runs |
| Steps 6+7 | Sliver removal creates new gaps; stitching needs to see the post-sliver boundary |
| Steps 7+8 | Stitching must classify loops *before* filling; once filled, gap vs hole distinction is lost |

---

## Session Summary Matrix

This matrix shows which cleaning steps perform non-trivial work for each session.
A dash (—) means the step is a no-op for that session.

| Session | Step 4: holes | Step 5: pinch | Step 6: slivers | Step 7: stitch | Step 8: fill | SLIM path |
|---------|:------------:|:------------:|:--------------:|:-------------:|:-----------:|-----------|
| ST13-01 | — | — | minor | blocked (ratio) | — | harmonic → SLIM |
| ST13-02 | 4 holes | 1 pass | minor | — | 4 holes | harmonic → Tutte → trim(1) → SLIM |
| ST13-03 | — | — | — | — | — | harmonic → SLIM |
| ST14-01 | — | 1 pass | minor | — | — | harmonic → Tutte → trim(1) → SLIM |
| **ST14-02** | **10 holes** | minor | **~30 faces** | **10 loops welded** | **remaining holes** | harmonic → Tutte → SLIM |
| ST14-04 | 13 holes | 2 passes | minor | — | 13 holes | harmonic → Tutte → trim(1) → SLIM |
| ST15-01 | — | — | minor | — | — | harmonic → Tutte → trim(1) → SLIM |
| ST16-02 | — | — | — | — | — | harmonic → SLIM |
| ST16-03 | 1 hole | — | — | — | 1 hole | harmonic → SLIM |
| ST16-05 | — | — | — | — | — | harmonic → SLIM |
| ST18-01 | — | — | minor | — | — | harmonic → Tutte → trim(1) → SLIM |
| ST18-04 | — | — | ~15 faces | — | — | harmonic → Tutte → trim(2) → SLIM |

**Key observation:** ST14-02 is the only session that exercises all 8 steps
non-trivially. It is the most topologically complex mesh (thumb-forearm junction with
extensive non-manifold regions) and has been the primary driver for steps 6, 7, and
the Delaunay upgrade to step 8.

---

## Development Chronology

| Date | Commit | Steps added/modified | Trigger |
|------|--------|---------------------|---------|
| 2026-05-18 | `d63c0a9` | Steps 1–4 (basic cleanup) | Initial SLIM production integration; all 12 sessions |
| 2026-05-18 | `e64e6b9` | Step 5 (pinch repair), step 8 v1 (centroid-fan hole fill) | ST13-02, ST14-02, ST14-04, ST16-03 failing with non-manifold flips |
| 2026-05-21 | `d1940bc` | Step 6 (sliver removal), step 8 v2 (Delaunay upgrade) | ST14-02 starburst artifacts from centroid-fan in large holes |
| 2026-05-21 | `65ae562` | Step 7 (boundary-gap stitching) | ST14-02 thumb enclosed by hole filling; ST13-01 regression guard |

---

## Related Documents

- Knowledge base: [`bug-slim-uv-non-manifold-flip.md`](bug-slim-uv-non-manifold-flip.md) — three-layer failure analysis and fix for 4 failing sessions
- Knowledge base: [`note-mesh-parameterization-interior-pin-foldovers.md`](note-mesh-parameterization-interior-pin-foldovers.md) — why interior pins cause foldovers; boundary-only constraint design
- Knowledge base: [`note-igl-slim-api-version-mismatch.md`](note-igl-slim-api-version-mismatch.md) — nanobind API for `slim_precompute` / `slim_solve`
- Knowledge base: [`investigation-mesh-flattening-algorithms-survey.md`](investigation-mesh-flattening-algorithms-survey.md) — literature survey motivating SLIM as production method
- Active plan: [`slim-uv-boundary-gap-stitching.md`](../plans/active/slim-uv-boundary-gap-stitching.md) — current branch plan for step 7
- Active plan: [`slim-uv-delaunay-hole-filling.md`](../plans/active/slim-uv-delaunay-hole-filling.md) — plan for step 6 + Delaunay upgrade to step 8
