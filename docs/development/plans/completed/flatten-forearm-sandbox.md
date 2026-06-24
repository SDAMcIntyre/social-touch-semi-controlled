# Plan: Flatten-Forearm Sandbox Script

**Date:** 2026-05-15
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-05-18 20:33
**Base Branch:** `feature/nerve-data-extraction-pipeline`
**Branch:** `feature/flatten-forearm-sandbox`

---

## Overview

Add a standalone development sandbox script
`code/scripts/flatten_forearm_sandbox.py` that loads a single forearm
PLY (path hard-coded at the top, user-edited), builds a triangle mesh,
and runs three working flattening baselines side-by-side (LSCM, ARAP,
harmonic). The script provides a working iteration environment for
developing the new flattening technique — outside the Prefect DAGs and
outside the receptive-field-mapping package.

## Problem Statement

The receptive-field-mapping pipeline currently projects 3D contact
points onto a 2D map via a session-local cylindrical unwrap
(`code/src/analysis/receptive_field_mapping/.../rf_projection.py:90–177`).
The investigation in
`docs/development/knowledge-base/investigation-caret-pals-for-forearm-flattening.md`
identified LSCM/ARAP-style conformal flattening as the natural
algorithmic successor. Before any pipeline change, we need a low-cost
environment for iterating on flattening code: load one PLY, see the
flattened result, tweak, repeat. The full DAG / config / Prefect
stack is too heavy for that iteration loop, and the current pipeline
has no flattening implementations beyond `cylindrical_unwrap`.

## Goals

### In Scope
1. New standalone script `code/scripts/flatten_forearm_sandbox.py`
   with a single hard-coded `PLY_PATH` string at the top of the file.
2. Mesh construction reusing existing helpers
   (`load_or_build_forearm_mesh()` for BPA; `define_forearm_mesh()`
   for 2.5D Delaunay as alternative).
3. Three working flattening baselines via `libigl`: LSCM, ARAP,
   harmonic.
4. A single matplotlib figure that shows the 3D mesh and the three 2D
   flattenings side-by-side, runnable as a one-shot:
   `python code/scripts/flatten_forearm_sandbox.py`.
5. Add `libigl` to `requirements.txt`.

### Out of Scope
- Spike / contact overlay on the 2D map (deferred — "pure geometry
  first" per user).
- Integration into the receptive-field-mapping DAG or any Prefect
  flow.
- Distortion-metric computation (per-vertex angular / area error
  maps).
- Cross-subject atlas / population alignment (PALS-style).
- Cylindrical-unwrap baseline (the production algorithm being
  compared against — left as a follow-up).
- Caching libigl outputs to disk.

## Success Criteria

- [ ] `pip install -r requirements.txt` in a fresh env installs
      `libigl` successfully.
- [ ] After editing `PLY_PATH` to a valid forearm PLY, running
      `python code/scripts/flatten_forearm_sandbox.py` opens a single
      matplotlib window with four panels (1× 3D + 3× 2D).
- [ ] All three 2D flattenings (LSCM, ARAP, harmonic) produce
      non-degenerate output (no NaNs, no collapsed-to-a-line).
- [ ] The script raises a clear, fail-fast error if `PLY_PATH` does
      not exist or the mesh is empty after cleanup (no silent
      fallbacks).
- [ ] Switching `MESH_METHOD` between `"bpa"` and `"delaunay"` works
      without other edits.

---

## Technical Design

### Approach

Single-file Python script in `code/scripts/`, matching the existing
sandbox-script convention (`code/scripts/sandbox_ply_viewer.py`,
`code/scripts/__misc/hue_circle_v9_integration_sandbox.py`). The
script imports two existing in-repo helpers for mesh building, then
calls `libigl` for flattening. All three flattenings share the same
boundary-loop computation; visualisation uses `matplotlib` with
`mpl_toolkits.mplot3d` for the 3D panel and
`matplotlib.tri.Triangulation` for the 2D panels.

The forearm mesh produced upstream by `define_forearm_mesh.py` is a
disk-topology open patch — which is the canonical LSCM input — so
no extra topological cut is required.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|---|---|---|---|
| Standalone script in `code/scripts/` | Matches sandbox convention; zero pipeline coupling; quickest to iterate on | Lives outside the production codepath; some duplication if/when promoted | **Chosen** |
| Add a new entry to `rf_projection.PROJECTION_METHODS` | Goes through the production dispatcher | Pipeline-coupled (config, DAG, idempotence wrappers) — kills the iteration loop the user wants | Rejected |
| Jupyter notebook | Inline plots, REPL-friendly | No `notebooks/` directory exists in repo; would establish a new convention | Rejected |
| Hand-rolled LSCM in numpy/scipy | Zero new dependency | ~100 lines of finicky sparse-system code, no ARAP/harmonic for free | Rejected |
| `potpourri3d` instead of `libigl` | Lighter install | LSCM only, no ARAP; less established | Rejected |

### Architecture Changes

**New file:**
- `code/scripts/flatten_forearm_sandbox.py` — single-file script,
  ~150–200 lines, structured as: header constants → `load_pcd()` →
  `build_mesh()` → `clean_mesh()` → `flatten_lscm()` →
  `flatten_arap()` → `flatten_harmonic()` → `plot_panels()` →
  `if __name__ == "__main__":` block.

**Modified file:**
- `requirements.txt` — add `libigl` line. No version pin on first
  pass; pin later if breakage occurs.

**Integration points (read-only):**
- `code/src/analysis/receptive_field_mapping/rf_surface_utils.py:23` —
  `load_or_build_forearm_mesh(forearm_ply_path: Path,
  mesh_cache_path: Path = None) -> Optional[trimesh.Trimesh]` (BPA).
- `code/scripts/_3_preprocessing/_3_forearm_extraction/define_forearm_mesh.py:36` —
  `define_forearm_mesh(source, output_path=None, show=False,
  force_processing=True) -> trimesh.Trimesh` (2.5D Delaunay).
- `o3d.io.read_point_cloud(str(path))` — canonical PLY loader,
  pattern used in 18 files across the repo.

**Knowledge-base references** (no change, but consulted):
- `docs/development/knowledge-base/note-3d-to-2d-surface-projection-algorithms.md`
  — algorithm catalogue (LSCM/ARAP/harmonic signatures already
  documented).
- `docs/development/knowledge-base/investigation-caret-pals-for-forearm-flattening.md`
  — motivation / context (this work is the "next concrete step"
  proposed at the bottom of that note).

**CuPy import-order constraint:** Not applicable — the script does
not import from `preprocessing.*`. No `import cupy` line needed.

---

## Implementation Plan

### Phase 1: Bootstrap — dependency, skeleton, PLY load, mesh build
**Goal:** Script can be run end-to-end to load a PLY and produce a
clean trimesh, with fail-fast errors on bad input.
**Started:** 2026-05-15
**Completed:** 2026-05-15

- [x] Task 1.1 — Add `libigl` to `requirements.txt`.
- [x] Task 1.2 — Create `code/scripts/flatten_forearm_sandbox.py`
      with module docstring, `PLY_PATH` and `MESH_METHOD` constants
      at the top, and the standard
      `if __name__ == "__main__":` entry point.
- [x] Task 1.3 — Implement `load_pcd(path: str) -> np.ndarray`
      wrapping `o3d.io.read_point_cloud()`. Raise
      `FileNotFoundError` if path missing.
- [x] Task 1.4 — Implement `build_mesh(ply_path: Path, method: str)
      -> trimesh.Trimesh` dispatching to either
      `load_or_build_forearm_mesh()` or `define_forearm_mesh()`.
- [x] Task 1.5 — Implement `clean_mesh(mesh) -> (V, F)`: keep
      largest connected component (`mesh.split(only_watertight=False)`,
      pick max-vertex submesh), `trimesh.repair.fix_winding`, drop
      unreferenced vertices, return numpy arrays. Raise
      `ValueError` if zero faces remain.

**Files Modified:**
- `requirements.txt` — add `libigl`.
- `code/scripts/flatten_forearm_sandbox.py` — **new** file, skeleton
  + Phase 1 functions.

**Dependencies:** None.

### Phase 2: Flattening baselines
**Goal:** Three callable flattening functions, each returning `(N,
2)` UV coordinates with no NaNs on a valid mesh.
**Started:** 2026-05-15
**Completed:** 2026-05-15

- [x] Task 2.1 — `boundary_loop(F)` helper wrapping
      `igl.boundary_loop`. Raise `RuntimeError` if no boundary
      (closed surface — unexpected here).
- [x] Task 2.2 — `flatten_lscm(V, F, boundary) -> np.ndarray`:
      pin two boundary vertices (`boundary[0]` → `(0, 0)`,
      `boundary[len//2]` → `(1, 0)`), call `igl.lscm`. Assert no
      NaNs in output.
- [x] Task 2.3 — `flatten_harmonic(V, F, boundary) -> np.ndarray`:
      map boundary uniformly to unit circle, call `igl.harmonic`
      with `k=1`. Assert no NaNs.
- [x] Task 2.4 — `flatten_arap(V, F, boundary) -> np.ndarray`:
      initialise with harmonic output, call `igl.ARAP(V, F, dim=2,
      b=...)`, iterate `arap.solve(...)` ~20 times or until
      convergence. Assert no NaNs.

**Files Modified:**
- `code/scripts/flatten_forearm_sandbox.py` — add Phase 2 functions.

**Dependencies:** Phase 1.

### Phase 3: Visualisation + entry point
**Goal:** Single matplotlib window with 4 panels appears on run.
**Started:** 2026-05-15
**Completed:** 2026-05-15

- [x] Task 3.1 — `plot_panels(V, F, uv_lscm, uv_arap, uv_harmonic)`:
      `plt.figure(figsize=(16, 4))`, four subplots:
      `(141, projection='3d')` for the input mesh,
      `(142)`/`(143)`/`(144)` for LSCM/ARAP/harmonic.
- [x] Task 3.2 — Each 2D panel: render triangles with
      `matplotlib.tri.Triangulation(uv[:, 0], uv[:, 1], F)` and
      `plt.triplot()` (wireframe). Title each panel with the method
      name. `set_aspect('equal')`.
- [x] Task 3.3 — Wire `if __name__ == "__main__":` block:
      load → mesh → clean → boundary → 3× flatten → plot →
      `plt.show()`.

**Files Modified:**
- `code/scripts/flatten_forearm_sandbox.py` — add Phase 3.

**Dependencies:** Phase 2.

---

## Testing Plan

### Manual Verification
- [ ] `pip install -r requirements.txt` succeeds in a fresh conda env
      on Windows (target platform per `CLAUDE.md`).
- [ ] Edit `PLY_PATH` to a real session's forearm PLY (e.g. under
      `{session_processed_path}/forearm_pointclouds/*.ply`); run
      `python code/scripts/flatten_forearm_sandbox.py`; confirm
      matplotlib window opens with 4 populated panels.
- [ ] Set `MESH_METHOD = "delaunay"`, re-run, confirm the panels
      regenerate without other code edits.
- [ ] Set `PLY_PATH` to a non-existent file → confirm
      `FileNotFoundError` is raised with a clear message and the
      script exits non-zero (no silent fallback per project
      convention).

### Edge Cases
- [ ] PLY with disconnected components — `clean_mesh()` keeps the
      largest; confirm the panels reflect that component.
- [ ] Very small mesh (< 50 vertices) — confirm libigl flattenings
      still complete or fail-fast with a clear error.
- [ ] PLY with no normals — `o3d.io.read_point_cloud` tolerates
      this; mesh helpers re-estimate. Confirm end-to-end run
      succeeds.

### Out of scope for testing
- Unit tests under `code/tests/` — this is a developer-only
  sandbox, not a production module. No automated test fixture.

---

## Documentation Plan

- [ ] Top-of-file docstring in `flatten_forearm_sandbox.py`
      explaining: purpose, how to use (edit `PLY_PATH`, run), what
      each flattening method does in one line, and a pointer to
      `docs/development/knowledge-base/note-3d-to-2d-surface-projection-algorithms.md`.
- [ ] No `README.md` / `CLAUDE.md` updates (sandbox is dev-only, not
      a user-facing feature).
- [ ] No new knowledge-base note (the two existing notes already
      cover the algorithm catalogue and the Caret/PALS
      investigation).

---

## Rollback Plan

All changes are additive and trivially reversible:
1. Delete `code/scripts/flatten_forearm_sandbox.py`.
2. Revert the `libigl` line in `requirements.txt`.
3. (Optional) `pip uninstall libigl` in the conda env.

No data migrations, no config changes, no public API changes.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `libigl` install fails on Windows / conda env | Med | Med | Document the install path in the script header. If breakage, pin a known-working version. Fallback: drop libigl and implement LSCM in numpy (rejected for now, but viable). |
| BPA / Delaunay mesh has non-manifold edges → libigl crashes | Med | Med | `clean_mesh()` keeps largest component + fixes winding. If libigl still complains, fail-fast with a clear error message naming the offending mesh issue. |
| Multiple disjoint boundary loops (degenerate mesh) | Low | Low | `igl.boundary_loop` returns the longest; document this and fail-fast if length is 0. |
| Forearm PLY too large (50k+ vertices) → libigl slow | Low | Low | Sandbox is single-shot; minutes are acceptable. Optional `downsample` in clean step if it becomes a problem. |
| User picks the wrong PLY (e.g. raw scan, not segmented forearm) | Med | Low | Fail-fast message in `clean_mesh()` if mesh is implausibly large; user can re-edit `PLY_PATH`. |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|---|---|---|
| Phase 1: Bootstrap + load + mesh | Small (~60 lines + 1-line req) | None |
| Phase 2: Three flattening baselines | Small (~50 lines, libigl-driven) | Phase 1 |
| Phase 3: Visualisation | Small (~40 lines, matplotlib) | Phase 2 |

Total: roughly 150–200 lines, single file. Implementable in one
sitting once approved.

---

## References

- Investigation note:
  `docs/development/knowledge-base/investigation-caret-pals-for-forearm-flattening.md`
- Algorithm catalogue:
  `docs/development/knowledge-base/note-3d-to-2d-surface-projection-algorithms.md`
- Existing mesh builders:
  `code/src/analysis/receptive_field_mapping/rf_surface_utils.py:23`,
  `code/scripts/_3_preprocessing/_3_forearm_extraction/define_forearm_mesh.py:36`
- Existing sandbox-script convention:
  `code/scripts/sandbox_ply_viewer.py`
- Production projection (for context, not modified):
  `code/src/analysis/receptive_field_mapping/.../rf_projection.py:90–177`
