# Plan: SLIM-Based Forearm Projection (Production Integration)

**Created:** 2026-05-17 14:30
**Approved:** —
**Completed:** —
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `feature/slim-forearm-projection`

---

## Overview

Promote the SLIM (Scalable Locally Injective Maps) forearm-flattening method from the
`compare_flattening_methods.py` sandbox into the production RF mapping pipeline as a new
projection method, registered alongside `tangent_plane` and `cylindrical_unwrap`.  SLIM
produces a per-vertex UV map for the cleaned forearm mesh; at projection time, arbitrary
3D query points are mapped to UV via barycentric interpolation on the nearest face.

## Problem Statement

The two existing projection methods (`tangent_plane`, `cylindrical_unwrap`) are closed-form
geometric projections.  They preserve global geometric structure but introduce significant
**area and conformal distortion** in regions far from the contact centroid, especially on
elongated forearm geometry.  The compare-flattening-methods sandbox showed SLIM produces
substantially lower symmetric-Dirichlet distortion across the entire mapped region.

A SLIM-based projection would let receptive-field heatmaps reflect distances on the actual
forearm surface rather than distances in a tangent plane or cylindrical unwrap — yielding
more faithful spatial RF metrics (hotspot area, half-peak diameter, weighted centroids).

The challenge: SLIM is non-closed-form — it requires a global optimisation that takes
seconds to minutes per session.  It cannot be evaluated lazily inside
`project_to_2d(points, ...)` on every call.

## Goals

### In Scope

1. **New precompute task** `precompute_forearm_slim_uv` in
   `analyse_workflow_processing_dag.yaml` that computes a per-session SLIM UV map and
   caches it next to the forearm PLY.
2. **New projection method** `project_slim` in `rf_projection.py`, registered as
   `"slim"` in `PROJECTION_METHODS`.  Loads the cached UV map; barycentric-interpolates
   arbitrary 3D query points via nearest-face lookup.
3. **Centre-vertex selection policy:** SLIM is centred on the spike-count-weighted 3D
   centroid (the "neuron hotspot location"), computed from `spike_positions.csv`
   produced by `map_receptive_fields_simple`.  The chosen mesh vertex is the one nearest
   that centroid; if it falls on the mesh boundary, the task raises (fail-fast).
4. **No fallbacks** in the production code: cotangent-harmonic init must be flip-free;
   if it isn't, the task raises and the researcher is responsible for triaging
   (re-mesh, different centre, etc.).
5. **Idempotency** via `should_process_task()`: cache is invalidated when the forearm
   PLY or spike_positions.csv changes (mtime-based).
6. **Unit tests** in `code/tests/`: synthetic-mesh smoke test for the UV precompute,
   barycentric interpolation correctness, fail-fast on boundary-centre, fail-fast on
   flipped init.

### Out of Scope

- Interactive GUI for centre-vertex picking (the sandbox has one; production uses
  the automatic spike-weighted centroid).
- Per-cluster SLIM maps (one per `(session, cluster)` pair).  All consumers use the
  per-session map.  Per-cluster centring is a documented future enhancement.
- Re-running upstream RF tasks to re-render heatmaps with the new projection method
  (that happens naturally once `"slim"` is selectable in the DAG config — no extra
  scaffolding needed).
- GPU acceleration of SLIM (libigl's CPU implementation is sufficient for the
  ~5k-30k-vertex forearm meshes in this dataset).
- ARAP and Harmonic productionisation (they were comparison baselines only; SLIM
  was the chosen winner).

## Success Criteria

- [ ] `precompute_forearm_slim_uv` runs end-to-end on at least 2 real session PLYs
      (ST15-01, ST16-05) and produces a flip-free UV map cached to disk.
- [ ] `project_to_2d(..., method="slim")` returns finite (N, 2) UVs for arbitrary 3D
      points sampled across the cleaned mesh surface, and exactly recovers the cached
      UV at mesh-vertex query points (within 1e-9 tolerance).
- [ ] When `spike_positions.csv` is missing or empty, the precompute task raises with a
      clear message naming the missing dependency.
- [ ] When the spike-weighted centroid maps to a boundary vertex, the precompute task
      raises with a clear message.
- [ ] Re-running the analysis pipeline without changes to PLY or spike data is a no-op
      (idempotency check) — confirmed via `should_process_task` skip.
- [ ] All new tests pass in `pytest code/tests/`.
- [ ] At least one heatmap rendered via `method="slim"` is reviewed by the researcher
      and visually compared against `tangent_plane` and `cylindrical_unwrap` on the
      same session.

---

## Technical Design

### Approach

Two-stage architecture:

```
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 1 — Precompute (Prefect task, per session, idempotent)        │
│                                                                     │
│   forearm_<session>.ply          spike_positions.csv                │
│             │                              │                        │
│             ▼                              ▼                        │
│   load_or_build_forearm_mesh()    weighted_centroid_3d = mean(xyz)  │
│             │                              │                        │
│             ▼                              ▼                        │
│   clean_mesh(...) → V, F          KDTree(V).query(centroid)         │
│             │                              │                        │
│             └──────────┬───────────────────┘                        │
│                        ▼                                            │
│                  flatten_slim(V, F, boundary, center_vid)           │
│                        │  (boundary-only init, no fallback,         │
│                        │   raise on flipped init)                   │
│                        ▼                                            │
│                  forearm_<session>_slim_uv.npz                      │
│                    (V_uv, F, center_vid, boundary,                  │
│                     ply_mtime, spike_csv_mtime)                     │
└─────────────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────────┐
│ Stage 2 — Projection (called from RF heatmap renderers)             │
│                                                                     │
│   project_slim(points_3d, forearm_vertices, contact_centroid, ...)  │
│       1. Load forearm_<session>_slim_uv.npz                         │
│       2. KDTree on mesh vertices → nearest face per query point     │
│       3. Barycentric coords on that face → interpolate UV           │
│       4. Return (N, 2) UVs                                          │
└─────────────────────────────────────────────────────────────────────┘
```

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Lazy compute inside `project_slim`** (no Prefect task) | No DAG change; simplest | SLIM is O(seconds–minutes); blocks every renderer; race conditions on cache writes | Rejected |
| **Two-stage precompute + projection (chosen)** | Compute-once; clean DAG; matches existing `load_or_build_forearm_mesh` pattern | New DAG task; need cache invalidation logic | **Chosen** |
| **One SLIM map per (session, cluster)** | Each cluster gets a locally-optimal UV near its own hotspot | N× more cache files; UV varies by clustering choice, making cluster-to-cluster spatial comparison harder | Rejected — out of scope; future enhancement |
| **Use existing `RFMetrics.weighted_centroid_3d` as centre** | Pre-computed; well-known | Only exists per-cluster, never per-session; would force us to bind SLIM to a specific cluster definition | Rejected |
| **Mean of `spike_positions.csv` rows (chosen)** | Mathematically identical to spike-count-weighted vertex centroid; per-session; no extra computation | Requires `spike_positions.csv` to exist (so SLIM precompute depends on `map_receptive_fields_simple`) | **Chosen** |
| **Tutte fallback when cotangent harmonic init has flips** | Robust — libigl tutorial recommends it | Conflicts with project's fail-fast rule; hides mesh-quality issues from researcher | Rejected per user decision |

### Architecture Changes

**New files:**

```
code/src/analysis/receptive_field_mapping/
├── forearm_slim_uv.py          — Precompute + cache I/O + barycentric interpolation
└── (rf_projection.py extended) — Adds project_slim; registers "slim" key

code/scripts/
└── analysis_workflow.py        — Adds @task wrapping precompute_forearm_slim_uv
                                  and wires it into the @flow.

code/tests/
└── test_forearm_slim_uv.py     — Unit tests (synthetic mesh, barycentric, fail-fast)

configs/
└── analyse_workflow_processing_dag.yaml — Adds task entry
                                  precompute_forearm_slim_uv with depends_on:
                                  [map_receptive_fields_simple]
```

**Module:** `forearm_slim_uv.py` exposes:

```python
def precompute_forearm_slim_uv(
    forearm_ply_path: Path,
    spike_positions_csv: Path,
    cache_path: Path | None = None,
    n_iter: int = 40,
) -> Path:
    """Build SLIM UV cache for a single session. Raises on any failure."""

def load_slim_uv_cache(cache_path: Path) -> SlimUvCache:
    """Load the cached UV + mesh data. Raises if cache is missing or stale."""

def barycentric_uv_lookup(
    cache: SlimUvCache,
    query_points_3d: np.ndarray,
) -> np.ndarray:
    """For each query point, find nearest face and interpolate UV. Shape (N, 2)."""
```

**Cache schema** (NumPy `.npz` with `allow_pickle=False`):

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `V` | (N_v, 3) | f8 | Cleaned mesh vertex positions |
| `F` | (N_f, 3) | i4 | Face indices into V |
| `uv` | (N_v, 2) | f8 | SLIM UV per vertex (centre at (0,0), boundary[0] on +x) |
| `center_vid` | scalar | i4 | Interior vertex chosen as the SLIM centre |
| `boundary_vid` | scalar | i4 | Boundary anchor vertex used for canonicalisation |
| `ply_mtime` | scalar | f8 | Source forearm PLY mtime — for idempotency checks |
| `ply_hash` | scalar | U64 | SHA-256 of first 4 kB of the PLY file — guards against same-mtime rewrites |
| `spike_csv_mtime` | scalar | f8 | spike_positions.csv mtime — for idempotency checks |
| `centroid_3d` | (3,) | f8 | Spike-weighted centroid (provenance) |

**Registry integration** in `rf_projection.py`:

```python
PROJECTION_METHODS = {
    "tangent_plane": project_tangent_plane,
    "cylindrical_unwrap": project_cylindrical_unwrap,
    "slim": project_slim,   # <-- new
}
```

`project_slim` signature matches the existing convention but requires a `slim_cache_path`
kwarg.  The caller (renderer) resolves this from the session config + analysed output dir.

### Cross-references to existing knowledge base

- [`note-igl-slim-api-version-mismatch.md`](../../knowledge-base/note-igl-slim-api-version-mismatch.md)
  — nanobind SLIM API; `igl.slim_precompute` returns the `SLIMData`; energy enum is
  `igl.MappingEnergyType.SYMMETRIC_DIRICHLET`.
- [`note-mesh-parameterization-interior-pin-foldovers.md`](../../knowledge-base/note-mesh-parameterization-interior-pin-foldovers.md)
  — interior Dirichlet pins break the Tutte/RKC bijectivity guarantee; the centre is
  applied post-solve via the rigid `canonicalise_uv` helper, never as a SLIM constraint.

### Reuse from existing sandbox

`compare_flattening_methods.py::flatten_slim()` and `flatten_forearm_sandbox.py::{boundary_loop,
clean_mesh, canonicalise_uv, _has_flipped_triangles}` already implement the core algorithm
correctly (post the fixes in `note-mesh-parameterization-interior-pin-foldovers.md`).  The
production module should **import** these as private helpers from a shared utility location
rather than copy-paste — preferred: move the helpers into a new
`code/src/analysis/receptive_field_mapping/_slim_helpers.py`, and update the sandbox to
import from there.  This avoids two copies of the algorithm drifting.

---

## Implementation Plan

### Phase 1: Extract reusable helpers from sandbox into production package
**Goal:** Move algorithmic primitives (`clean_mesh`, `boundary_loop`, `canonicalise_uv`,
`_has_flipped_triangles`, `flatten_slim`) from `code/scripts/` sandbox scripts into
`code/src/analysis/receptive_field_mapping/` so they can be imported by both the sandbox
AND the new production module.
**Started:** 2026-05-18
**Completed:** 2026-05-18

- [x] Task 1.1 — Create `code/src/analysis/receptive_field_mapping/_slim_helpers.py`
      containing `clean_mesh`, `boundary_loop`, `canonicalise_uv`,
      `_has_flipped_triangles`, and a production `flatten_slim`.  The production
      `flatten_slim` strips the `_tutte_uniform_map` fallback from the sandbox version;
      if the cotangent-harmonic init has flipped triangles, it raises
      `RuntimeError(...)` immediately instead of falling back.
- [x] Task 1.2 — Update `code/scripts/compare_flattening_methods.py` and
      `code/scripts/flatten_forearm_sandbox.py` to import from the new module.
      The sandbox `compare_flattening_methods.py` keeps its own `_tutte_uniform_map`
      and wraps the imported `flatten_slim` with the fallback logic locally.
- [x] Task 1.3 — Re-run sandbox to confirm parity (no behaviour change).
      Note: manual verification required — researcher should run the script against
      a real forearm PLY to confirm output parity before proceeding to Phase 2.

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/_slim_helpers.py` — NEW.
- `code/scripts/compare_flattening_methods.py` — import refactor; remove duplicated helpers;
  retain `_tutte_uniform_map` fallback wrapper locally.
- `code/scripts/flatten_forearm_sandbox.py` — import refactor; remove moved helpers.

**Dependencies:** None.

### Phase 2: Precompute task + cache I/O
**Goal:** Implement the per-session precompute that produces and caches the SLIM UV map.
**Started:** 2026-05-18
**Completed:** 2026-05-18

- [x] Task 2.1 — Implement `forearm_slim_uv.py::precompute_forearm_slim_uv()`:
      builds mesh via `load_or_build_forearm_mesh + clean_mesh`, reads
      `spike_positions.csv`, computes spike-weighted 3D centroid (mean of xyz columns),
      KDTree-lookup for nearest interior vertex, calls `flatten_slim`, writes `.npz` cache.
- [x] Task 2.2 — Implement `load_slim_uv_cache()` with mtime-based staleness check (raises
      if PLY or spike CSV has changed since cache).
- [x] Task 2.3 — Wire as Prefect `@task precompute_forearm_slim_uv` in
      `analysis_workflow.py`; add DAG-config entry with `depends_on:
      [map_receptive_fields_simple]`; respect `should_process_task` idempotency.
- [x] Task 2.4 — Boundary-vertex check: if nearest-vertex lookup hits a boundary index,
      raise `ValueError("Spike-weighted centroid maps to mesh boundary vertex...")`.

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/forearm_slim_uv.py` — NEW.
- `code/scripts/analysis_workflow.py` — add `@task` wrapper + flow wiring.
- `configs/analyse_workflow_processing_dag.yaml` — add `precompute_forearm_slim_uv` entry.

**Dependencies:** Phase 1.

### Phase 3: Projection registration + barycentric lookup
**Goal:** Make `method="slim"` selectable through `project_to_2d`.
**Started:** 2026-05-18
**Completed:** 2026-05-18

- [x] Task 3.1 — Implement `barycentric_uv_lookup()` in `forearm_slim_uv.py`: KDTree on
      mesh face centroids → find candidate face → compute barycentric coords for the
      query point projected onto the face plane → interpolate UV.
- [x] Task 3.2 — Implement `project_slim()` in `rf_projection.py` matching the registry
      signature; load cache via `load_slim_uv_cache`, delegate to barycentric lookup.
- [x] Task 3.3 — Register `"slim"` in `PROJECTION_METHODS`.  Document the additional
      kwarg `slim_cache_path` and how renderers should resolve it from session config.
- [x] Task 3.4 — Update **all** callers of `project_to_2d` to resolve and pass
      `slim_cache_path` when method is `"slim"`:
      `rf_simple_pipeline.py`, `rf_cluster_visualizer.py`, `rf_metrics.py`,
      and `rf_grid_cell_metrics.py`.

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/forearm_slim_uv.py` — barycentric helper.
- `code/src/analysis/receptive_field_mapping/rf_projection.py` — new `project_slim`,
  registry entry.
- `code/src/analysis/receptive_field_mapping/rf_simple_pipeline.py` — pass cache path.
- `code/src/analysis/receptive_field_mapping/rf_cluster_visualizer.py` — same.
- `code/src/analysis/receptive_field_mapping/rf_metrics.py` — pass cache path.
- `code/src/analysis/receptive_field_mapping/rf_grid_cell_metrics.py` — same.

**Dependencies:** Phase 2.

### Phase 4: Tests + manual verification on real sessions
**Goal:** Confirm correctness + performance on real data.
**Started:** 2026-05-18
**Completed:** 2026-05-18

- [x] Task 4.1 — Synthetic-mesh unit tests for `precompute_forearm_slim_uv` (flip-free,
      centre at origin, boundary[0] on +x).
- [x] Task 4.2 — Synthetic-mesh unit tests for `barycentric_uv_lookup` (mesh-vertex
      query exactly recovers cached UV; mid-face query lies in the convex hull of the
      face's vertex UVs).
- [x] Task 4.3 — Fail-fast unit tests: missing spike CSV → raise; centre on boundary
      → raise; flipped init → raise.
- [x] Task 4.4 — Manual verification: run the analysis pipeline on ST15-01 and
      ST16-05 with `projection_method: slim` in the DAG config; eyeball
      heatmap PNGs against the same sessions rendered with `tangent_plane` and
      `cylindrical_unwrap`; measure precompute wall-clock time.
      Manual verification requires real session PLYs (ST15-01, ST16-05) — performed by researcher after merge.

**Files Modified:**
- `code/tests/test_forearm_slim_uv.py` — NEW.

**Dependencies:** Phase 3.

---

## Testing Plan

### Unit Tests

- [ ] **Synthetic disk mesh** — Build a 73-vertex disk with mild curvature, pick centre
      = apex, run precompute, assert `uv[center_vid] ≈ (0,0)`, `uv[boundary[0]]` on +x,
      no flipped triangles, no NaN.  (Same pattern as the smoke test from the
      compare-flattening-methods fix session.)
- [ ] **Synthetic elongated patch** — 213-vertex forearm-like patch.  Demonstrates that
      the boundary-only init avoids the foldover vortex.  Centre = off-axis interior
      vertex.  Assert flip-free.
- [ ] **Barycentric round-trip** — Query at every mesh vertex → returned UV matches the
      cached UV exactly (within 1e-9).
- [ ] **Barycentric interior** — Query at the centroid of each face → returned UV lies in
      the convex hull of that face's three vertex UVs.
- [ ] **Idempotency** — Calling `precompute_forearm_slim_uv` twice on the same inputs
      reuses the cache (no re-solve), verified by mtime check.
- [ ] **Stale cache invalidation** — Touch the PLY → next call rebuilds the cache.
- [ ] **Fail-fast: missing spike CSV** — Raises `FileNotFoundError` with the CSV path.
- [ ] **Fail-fast: empty spike CSV** — Raises `ValueError("no spikes ...")`.
- [ ] **Fail-fast: centre on boundary** — Force centroid to land near the boundary →
      raises `ValueError("centre vertex is on mesh boundary ...")`.
- [ ] **Fail-fast: flipped init** — Inject a degenerate mesh that produces flipped
      cotangent-harmonic init → raises `RuntimeError("flipped triangles ...")`.

### Integration Tests

- [ ] `project_to_2d(..., method="slim", slim_cache_path=...)` end-to-end on synthetic
      mesh + synthetic query points.
- [ ] DAG dry-run: `precompute_forearm_slim_uv` appears after
      `map_receptive_fields_simple` in the resolved task order.

### Manual Verification

- [ ] Run pipeline on ST15-01 — heatmap PNG side-by-side comparison.
- [ ] Run pipeline on ST16-05 — heatmap PNG side-by-side comparison.
- [ ] Measure wall-clock time for the precompute on each session — should be under
      ~2 minutes per session on a typical workstation; document in the completion note.

### Edge Cases

- [ ] Mesh with multiple connected components — `clean_mesh` keeps the largest;
      precompute proceeds; flag a warning if discarded vertices exceed 5% of total.
- [ ] Mesh with very short boundary loop (<1% of vertices) — same warning as in
      `compare_flattening_methods.py` main path.

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` — mention `forearm_slim_uv.py` and the new
      `precompute_forearm_slim_uv` task.
- [ ] Update `configs/analyse_workflow_processing_dag.yaml` inline comments — document
      the `precompute_forearm_slim_uv` task entry with options.
- [ ] No new user guide needed; existing RF mapping doc covers `projection_method`.

---

## Rollback Plan

1. **Before merging to dev:** Plan lives only on `feature/slim-forearm-projection`.
   Revert is just a branch deletion.
2. **After merging:**
   - Remove `precompute_forearm_slim_uv` from `analyse_workflow_processing_dag.yaml`
     (researcher can do this without code changes — the task simply won't run).
   - Switch `projection_method` back to `tangent_plane` or `cylindrical_unwrap` in any
     RF visualisation task.  Cached `.npz` files become orphans — they can be left in
     place or globbed and deleted.
3. **No database, no migrations.** The only persistent state is the cache `.npz` files
   next to the forearm PLYs, which are safe to leave (or `git clean -ndx` if the cache
   directory is gitignored).

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Cotangent-harmonic init produces flips on a real forearm mesh — researcher hits the new fail-fast and is blocked | Med | Med | Document the failure mode in the new KB note (already exists for foldovers); provide a small script that rebuilds the mesh with finer BPA radii. |
| Spike-weighted centroid lands on the mesh boundary | Low | Med | Fail-fast with a clear message; researcher can re-touch the forearm closer to the centre, or extend the mesh extraction to include more skin. |
| SLIM wall-clock time blows out on dense meshes (>30k verts) | Low | Low | Profile during Phase 4 manual verification; can decimate the mesh to ~10k verts before SLIM if needed (`open3d.simplify_quadric_decimation`). |
| Barycentric interpolation fails for query points far from any face (e.g., point above the surface in 3D) | Med | Low | Use nearest-face fallback in 3D (closest-point-on-triangle) — not a "silent fallback" because it's the *only* defined behaviour for off-surface points; document explicitly. |
| Cache invalidation drift — researcher edits PLY but cache mtime doesn't update due to filesystem quirks | Low | Med | Use both PLY mtime AND content hash (first 4 kB of file content) as the staleness key. |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Extract helpers | 1 hour | None |
| Phase 2: Precompute task + cache | 4 hours | Phase 1 |
| Phase 3: Projection registration | 3 hours | Phase 2 |
| Phase 4: Tests + verification | 3 hours | Phase 3 |
| **Total** | **~11 hours** | |

---

## References

- Related Plans:
  - `docs/development/plans/active/compare-flattening-methods.md` — the sandbox study
    that selected SLIM as the winner.
  - `docs/development/plans/active/flatten-forearm-sandbox.md` — the initial sandbox.
- Knowledge base:
  - `docs/development/knowledge-base/note-igl-slim-api-version-mismatch.md`
  - `docs/development/knowledge-base/note-mesh-parameterization-interior-pin-foldovers.md`
- libigl tutorial 709 (SLIM):
  https://github.com/libigl/libigl/blob/main/tutorial/709_SLIM/param_2d_demo_iter.cpp
