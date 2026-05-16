# Plan: Compare Six Forearm Flattening Methods (SLIM Stage-1 Study)

**Date:** 2026-05-16
**Author:** Basil Duvernoy
**Status:** In Progress
**Started:** 2026-05-16
**Base Branch:** `feature/center-point-picker-gui`
**Branch:** `feature/compare-flattening-methods`

---

## Overview

Build a single sandbox script that flattens the same forearm mesh with six methods — LSCM, ARAP, Harmonic, SLIM (new), tangent-plane, and cylindrical-unwrap — and emits side-by-side figures plus a per-method distortion summary. Runs on two confirmed session PLYs. Verifies the survey's claim that SLIM has the lowest isometric distortion of the parametric methods, before any production wire-in is committed.

## Problem Statement

The production RF pipeline uses two simple 3D→2D projections (tangent-plane and cylindrical-unwrap). A sandbox prototypes three libigl baselines (LSCM, harmonic, ARAP) with an interactive centre-vertex pin but is not integrated. The recent literature survey [`investigation-mesh-flattening-algorithms-survey.md`](../../knowledge-base/investigation-mesh-flattening-algorithms-survey.md) identifies SLIM (Rabinovich et al., ACM TOG 2017) as the modern best-fit for our use case — symmetric-Dirichlet energy, flip-free, natively accepts arbitrary interior-vertex pins — and recommends a head-to-head comparison before any production change.

Without this comparison, we cannot:
1. Justify replacing the LSCM/ARAP baseline in the sandbox with SLIM.
2. Quantify how much distortion the production tangent-plane and cylindrical-unwrap projections incur relative to mesh parametrisers.
3. Decide between Track A ("register to MANO") and Track B ("per-session SLIM") for production.

## Goals

### In Scope
1. Add SLIM as a fourth flattening method, with the existing centre-vertex pin mechanism.
2. Run all six methods on the **same cleaned mesh `(V, F)`** for at least two sessions.
3. Produce comparable per-face distortion metrics (conformal σ₁/σ₂, log₂ area) across all six methods.
4. Persist artefacts (PNGs + summary CSV) so the survey note can be updated with a "stage-1 result" section.

### Out of Scope
- Wiring any method into the production pipeline (`rf_projection.py` registry stays unchanged).
- MANO / template-registration work (Track A — deferred).
- Geodesic-vs-Euclidean distance scatter plots (added complexity for marginal gain at stage 1; can be a follow-up).
- New unit tests in `code/tests/` — the deliverable is exploratory.
- Changing the existing sandbox script `flatten_forearm_sandbox.py` (it stays as-is; the new script imports from it).

## Success Criteria

- [ ] New script `code/scripts/compare_flattening_methods.py` runs end-to-end on at least two forearm PLYs and produces three artefacts per session (comparison PNG, distortion PNG, summary CSV).
- [ ] SLIM is implemented as `flatten_slim(V, F, boundary, center_vid)` using `igl.SLIMData` / `igl.slim_precompute` / `igl.slim_solve`; raises on NaN; no fallback (fail-fast convention).
- [ ] All six methods produce a UV-per-vertex on the same cleaned `(V, F)` so `compute_face_distortion` is reusable for every method.
- [ ] Summary CSV reports `conformal_median`, `conformal_p95`, `area_median`, `area_p95` per method.
- [ ] The pinned centre vertex (red dot) lands at the UV origin in the four mesh-parametric panels (LSCM, ARAP, Harmonic, SLIM).
- [ ] Cross-session sanity: relative method ranking is consistent across the two sessions (or the inconsistency is documented and a third session is added).
- [ ] The survey note is updated with a stage-1 results section linking to the generated artefacts.

---

## Technical Design

### Approach

A single standalone sandbox script that:
1. Reuses `flatten_forearm_sandbox.py`'s mesh-loading and cleaning pipeline by importing its functions directly.
2. Adds one new function `flatten_slim()` (analogous to the existing `flatten_arap()` at lines 587–630 of the sandbox).
3. Applies the production point-set projections (`project_tangent_plane`, `project_cylindrical_unwrap`) to `V` as the point set, with a PCA-derived rotation matrix as a stand-in for the saved RF-camera rotation.
4. Routes every method through `compute_face_distortion(V, F, uv)` and tallies summary statistics.
5. Saves PNGs + CSV next to the input PLY (same convention as `flatten_forearm_sandbox.py`).

Same script structure, same UX (interactive picker, hard-coded `PLY_PATH` constant) — so a Basil who has used the existing sandbox can use this one without learning anything new.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Extend `flatten_forearm_sandbox.py` in place (add SLIM + production projections inside the existing file) | One file, no duplication of imports/CLI | File already ~890 lines; mixes "sandbox for new algorithms" with "comparison harness for production methods" | Rejected |
| **New sandbox `compare_flattening_methods.py` that imports from the existing sandbox** | Clear separation of concerns; existing sandbox unchanged; matches project's "multiple sandbox scripts" pattern | One more file to maintain | **Chosen** |
| Jupyter notebook for the comparison | Inline display, easy iteration | Notebooks are not in this repo's pattern; harder to re-run reproducibly | Rejected |
| Add `slim` to `rf_projection.py` dispatcher immediately | Closest to production | Premature — this plan is explicitly verification, not integration | Rejected |

### Architecture Changes

No architectural changes. Only one new file. No edits to:
- `code/src/analysis/receptive_field_mapping/rf_projection.py`
- `code/scripts/flatten_forearm_sandbox.py`
- Any DAG config or `configs/*.yaml`.

```
code/scripts/
├── flatten_forearm_sandbox.py            ← unchanged; imported by the new script
└── compare_flattening_methods.py         ← NEW
```

### Reused functions (no re-implementation)

From `code/scripts/flatten_forearm_sandbox.py`:
- `load_pcd` (lines 102–141)
- `build_mesh` (lines 352–412, `MESH_METHOD = "bpa"`)
- `clean_mesh` (lines 415–505)
- `pick_center_point` (lines 271–312)
- `boundary_loop` (lines 515–523)
- `flatten_lscm`, `flatten_harmonic`, `flatten_arap` (lines 526–630)
- `compute_face_distortion` (lines 633–687)
- `plot_panels`, `plot_distortion_panels` patterns (lines 699–817)
- `show_mesh_inspector` (lines 315–349) — optional final view

From `code/src/analysis/receptive_field_mapping/rf_projection.py`:
- `project_tangent_plane(points_3d, forearm_vertices, contact_centroid, rotation_matrix, **kwargs)` — lines 21–34. Requires a non-None `rotation_matrix` (raises `ValueError` otherwise).
- `project_cylindrical_unwrap(points_3d, forearm_vertices, contact_centroid, per_point_radius=False, rotation_matrix=None, **kwargs)` — lines 93–177. Falls back to PCA if `rotation_matrix` is None; passes a synthesised one for parity.

### SLIM integration sketch

Installed `libigl` exposes `igl.SLIMData`, `igl.slim_precompute`, `igl.slim_solve` (verified via `python -c "import igl; print([s for s in dir(igl) if 'slim' in s.lower()])"`).

```python
def flatten_slim(V, F, boundary, center_vid=None, n_iter=40):
    uv_init = flatten_harmonic(V, F, boundary, center_vid=center_vid)
    if center_vid is not None:
        b = np.array([int(boundary[0]), int(center_vid)], dtype=np.int32)
        bc = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    else:
        b = np.array([int(boundary[0]), int(boundary[len(boundary)//2])], dtype=np.int32)
        bc = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float64)
    data = igl.SLIMData()
    igl.slim_precompute(V, F, uv_init, data,
                        igl.SLIM_ENERGY_TYPE_SYMMETRIC_DIRICHLET,
                        b, bc, soft_p=1e5)
    uv = uv_init
    for _ in range(n_iter):
        uv = igl.slim_solve(data, 1)
    if np.any(np.isnan(uv)):
        raise RuntimeError("SLIM produced NaN values — mesh may be degenerate.")
    return uv.astype(np.float64)
```

Exact constant name `igl.SLIM_ENERGY_TYPE_SYMMETRIC_DIRICHLET` should be verified at first run; the Python binding may use a numeric enum. Fail-fast on first error — no fallback (CLAUDE.md "fail-fast pipeline" convention).

### Applying point-set projections to the whole mesh

```python
centered = V - V.mean(axis=0)
_, _, Vt = np.linalg.svd(centered, full_matrices=False)
rotation_matrix = Vt  # PCA principal axes; stand-in for saved RF-camera rotation

uv_tangent = project_tangent_plane(
    points_3d=V, forearm_vertices=V,
    contact_centroid=V.mean(axis=0),
    rotation_matrix=rotation_matrix,
)
uv_cyl = project_cylindrical_unwrap(
    points_3d=V, forearm_vertices=V,
    contact_centroid=V.mean(axis=0),
    per_point_radius=False,
    rotation_matrix=rotation_matrix,
)
```

The synthesised rotation matrix is documented in the script with a comment explaining why we are not using the saved RF-camera matrix for this geometry-only comparison.

---

## Implementation Plan

### Phase 1: SLIM addition
**Goal:** Add a fourth mesh-parametric method to match the existing three.

- [x] Create `code/scripts/compare_flattening_methods.py` with the standard sandbox header (PLY_PATH constant, CENTER_POINT constant).
- [x] Import the reused functions from `flatten_forearm_sandbox`.
- [x] Implement `flatten_slim(V, F, boundary, center_vid)` per the sketch above.
- [x] Smoke test: run only LSCM + SLIM on session `2022-06-17_ST16-05`, confirm SLIM converges, confirm pinned centre vertex is at `(0, 0)` in the SLIM panel.

**Files Modified:**
- `code/scripts/compare_flattening_methods.py` — new file, ~120 lines at this phase

**Dependencies:** None

### Phase 2: Six-method orchestration
**Goal:** All six methods produce a UV-per-vertex on the same cleaned `(V, F)`.

- [x] Add the PCA-rotation-matrix synthesis helper.
- [x] Wire `project_tangent_plane(V, V, V.mean(axis=0), rotation_matrix=...)` and `project_cylindrical_unwrap(...)` calls.
- [x] Build the `uvs: dict[str, np.ndarray]` mapping `{"LSCM", "ARAP", "Harmonic", "SLIM", "TangentPlane", "Cylindrical"} -> uv_array`.
- [x] Compute `compute_face_distortion(V, F, uv)` for every entry; collect `(conformal, area)` arrays.
- [x] Persist `{stem}_compare_{timestamp}_summary.csv` with `method, conformal_median, conformal_p95, area_median, area_p95`.

**Files Modified:**
- `code/scripts/compare_flattening_methods.py` — extended to ~200 lines

**Dependencies:** Phase 1

### Phase 3: Figures and second-session verification
**Goal:** Produce side-by-side figures and confirm cross-session stability.

- [x] Extend `plot_panels` to a 7-panel layout (3D input + 6 UVs); reuse the existing `plot_panels` pattern or write a small wrapper.
- [x] Extend `plot_distortion_panels` to a 6-column × 2-row layout (one column per method).
- [x] Save `{stem}_compare_{timestamp}.png` and `{stem}_compare_{timestamp}_distortion.png` next to the input PLY; open them via `os.startfile` per the project's debug convention (`feedback-debugpy-matplotlib-backend` memory).
- [x] Run on the second session `2022-06-16_ST15-01`; compare summary CSVs across the two sessions; check that the relative method ranking is consistent.
- [x] Update [`investigation-mesh-flattening-algorithms-survey.md`](../../knowledge-base/investigation-mesh-flattening-algorithms-survey.md) with a "Stage-1 results" section that links to the saved artefacts and notes the chosen track for stage 2.

**Files Modified:**
- `code/scripts/compare_flattening_methods.py` — final size ~250–300 lines
- `docs/development/knowledge-base/investigation-mesh-flattening-algorithms-survey.md` — append a "Stage-1 results" section

**Dependencies:** Phase 2

---

## Testing Plan

### Unit Tests
None. This is an exploratory comparison script; no production code is touched.

### Integration Tests
None. The plan explicitly excludes pipeline wiring.

### Manual Verification
- [ ] Run `python code/scripts/compare_flattening_methods.py` on session `2022-06-17_ST16-05`.
  - Confirms the script loads, picker opens, six methods run, three artefacts appear.
- [ ] Run on the second session `2022-06-16_ST15-01`.
  - Same checks; confirms portability.
- [ ] Visually inspect the 7-panel comparison PNG: every panel shows a forearm-shaped blob; centre vertex marker lands at UV origin in LSCM/ARAP/Harmonic/SLIM.
- [ ] Open the summary CSV and confirm SLIM's `conformal_p95` is `<=` the other three mesh parametrisers (literature claim from the survey).
- [ ] Diff the two sessions' summary CSVs; confirm method ranking is consistent.

### Edge Cases
- [ ] **SLIM constant name wrong**: if `igl.SLIM_ENERGY_TYPE_SYMMETRIC_DIRICHLET` does not exist in the installed binding, the script raises `AttributeError` immediately — expected; do not add a fallback. Fix by inspecting `dir(igl)` and substituting the correct name (likely a numeric `0`/`1`/`2`).
- [ ] **Mesh has multiple boundary loops after cleaning**: `boundary_loop` returns only the longest; documented and acceptable (LSCM/SLIM tolerate small holes). Surface as a warning if `len(boundary) < V.shape[0] * 0.01`.
- [ ] **Centre vertex on boundary**: `flatten_harmonic` / SLIM raise — already handled in the sandbox.
- [ ] **Production projection emits NaN/inf for vertices behind the camera**: `compute_face_distortion` would surface this via `degen` masking. If too many faces are masked, log a warning.

---

## Documentation Plan

- [x] Update `docs/development/knowledge-base/investigation-mesh-flattening-algorithms-survey.md` with a "Stage-1 results" section linking to the saved PNGs and CSV.
- [ ] No README.md or CLAUDE.md change — this is an internal sandbox.
- [ ] No `docs/guides/` entry — script is exploratory, not user-facing.
- [ ] No changelog entry — no shipped feature.
- [ ] Inline docstrings on `flatten_slim` and the orchestration `main` to describe inputs, outputs, and the SLIM API caveat.

---

## Rollback Plan

Trivial: this plan adds **exactly one file** (`code/scripts/compare_flattening_methods.py`) plus an append-only edit to one knowledge-base note.

1. **Before merge:** delete `code/scripts/compare_flattening_methods.py`. No callers exist; pipeline unaffected.
2. **Data considerations:** none — script only writes PNGs and CSVs to the OneDrive data directory next to the input PLYs. Deleting them has no pipeline side-effect.
3. **Rollback procedure:** `git revert <feature-commit>` on the merge commit.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| SLIM Python binding API differs from the libigl tutorial #709 pattern | Med | Low | Fail-fast at first call; verify constant names with `dir(igl)` before fixing. Documented as Edge Case above. |
| PCA-derived rotation matrix biases the production-projection panels unfairly | Low | Med | Comment explaining the stand-in; if it skews results, swap in the saved RF-camera rotation matrix from the corresponding session config in a follow-up. |
| SLIM stalls on degenerate triangulations from BPA | Low | Med | Same `clean_mesh` pipeline as LSCM/ARAP already handles pinch vertices; if it stalls, fall back to the literature's recommended secondary (Progressive Parameterizations) — but as a separate plan, not a fallback in this script. |
| Two-session comparison is inconclusive (different method ranking per session) | Med | Med | Add a third session before drawing conclusions; documented in Success Criteria. |
| Picked centre vertex is not anatomically meaningful → SLIM UVs are not comparable across sessions | High (for this plan) | Low (deferred concern) | Stage 1 is geometry-only; cross-session SLIM comparison is explicitly a stage-2 concern (Track A vs B decision). |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 — SLIM addition | 1 hour | None |
| Phase 2 — Six-method orchestration | 1–2 hours | Phase 1 |
| Phase 3 — Figures + second session + survey-note update | 1 hour | Phase 2 |
| **Total** | **3–4 hours** | |

---

## References

- Survey: [`docs/development/knowledge-base/investigation-mesh-flattening-algorithms-survey.md`](../../knowledge-base/investigation-mesh-flattening-algorithms-survey.md) — origin of the stage-1 recommendation.
- Algorithm catalogue: [`docs/development/knowledge-base/note-3d-to-2d-surface-projection-algorithms.md`](../../knowledge-base/note-3d-to-2d-surface-projection-algorithms.md).
- Related but distinct: [`docs/development/plans/pending/investigate-rf-projection.md`](./investigate-rf-projection.md) — diagnostic + bug fix for the existing tangent-plane projection, complementary to this comparison.
- Existing sandbox: `code/scripts/flatten_forearm_sandbox.py`.
- Existing sandbox plan: `docs/development/plans/active/flatten-forearm-sandbox.md` (Phases 1–3 done; integration deferred — this plan is part of that deferred work).
- SLIM paper: Rabinovich, Poranne, Panozzo, Sorkine-Hornung, "Scalable Locally Injective Mappings", ACM TOG 2017. [PDF](https://cims.nyu.edu/gcl/papers/SLIM2017.pdf).
- libigl SLIM tutorial: tutorial #709, [https://libigl.github.io/tutorial/#scalable-locally-injective-maps](https://libigl.github.io/tutorial/#scalable-locally-injective-maps).
