# Plan: Per-Vertex Contact Depth — Proof of Concept

**Date:** 2026-08-12
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `dev`
**Branch:** `feature/per-vertex-contact-depth-poc`

---

## Overview

Today the contact between the hand mesh and the forearm mesh is summarised per frame by a
**single scalar** `contact_depth = max(|signed distance|)`. This PoC elevates that to a
**per-vertex signed depth field** over the contact patch, and renders it as an interactive 3D view
for **one recording**: contact vertices coloured by depth, a colourbar legend in millimetres, and a
frame time slider.

The per-vertex values already exist inside the current computation and are destroyed by a single
`np.max`. The scientific work is therefore not the geometry — it is proving the field is
trustworthy and making it visible.

## Problem Statement

Every frame with contact carries one IFF (instantaneous firing frequency) value from the recorded
afferent. Today **all vertices in a contact patch are treated as equally responsible** for that
IFF. This is physically wrong: for a spherical fingertip the stress is concentrated at the patch
centre and falls to zero at the rim. Without a per-vertex quantity there is no basis on which to
weight each vertex's contribution.

The scalar has a second, sharper pathology. `max(|d|)` reports the deepest interpenetration
**anywhere** in the patch. If a knuckle digs in 4 mm at one end of the patch while the recorded
afferent's receptive field sits under 0.5 mm at the other end, the frame is labelled "4 mm" and the
neuron's actual mechanical drive is misrepresented. The pipeline already computes an RF centre
(`center_on_receptive_field`), yet cannot currently answer *"what was the indentation at the RF?"*

Full background and the decision history: [`docs/development/brainstorms/per-vertex-contact-depth.md`](../../brainstorms/per-vertex-contact-depth.md).

## Goals

### In Scope

1. Extract the per-vertex signed depth field as a **pure, dependency-light function** that takes
   geometry and returns arrays — no config objects, no file paths, no rendering imports.
2. Make the existing scalar `contact_depth` **derived from** that field, so the definition of
   interpenetration exists in exactly one place.
3. Prove correctness with an **exact regression invariant** (`max(|field|) == legacy scalar`,
   bit-identical) plus analytic known-answer tests.
4. A **standalone PoC driver** that runs one recording end-to-end and holds the per-frame field in
   memory.
5. An **interactive PyQt5 + PyVista viewer**: contact vertices coloured by depth, perceptually
   uniform colourmap, colourbar labelled in mm, frame slider, play/pause, forearm and hand shown as
   context geometry.
6. Fix the **long-form data contract** (column names, dtypes, units, sign convention) even though
   nothing is written to disk in this PoC.

### Out of Scope

- **The Parquet sidecar writer** — the contract is fixed here; the writer is not built.
- **Plumbing the field through the postprocessing chain** (ICP, dedup, projection onto
  forearm-of-reference, PCA calibration, RF-centring) and the parallel spatial transformer that
  would require.
- **DAG task integration** — no new task in any `configs/*_dag.yaml`, no Prefect flow.
- **The IFF weighting kernel itself** — this PoC produces the raw field the kernel will later
  consume; it does not decide the depth→weight transform.
- **Contact-mechanics smoothing** (Hertzian correction) — raw geometric field only.
- **Multi-recording / whole-session batch** — one recording at a time.
- **Retiring the `contact_points` CSV column** — untouched.
- **Fixing the non-watertight hand mesh** — see Risks; the PoC exposes it, it does not repair it.

## Success Criteria

- [x] `signed_contact_depth_mm()` exists as a pure function importing neither PyVista, nor Qt, nor
      any repo loader/config class.
- [x] `ObjectsInteractionProcessor._calculate_intersection_volume` derives `contact_depth` from the
      field rather than computing it independently — no second `np.max` over raw distances.
- [ ] Regression test: over a full recording, the recomputed scalar equals the value in the
      committed reference CSV for **every** frame, with `np.array_equal` on the float64 values
      (bit-identical, not `allclose`). *(Test written; skips until a recording bundle is supplied
      via `SOCIAL_TOUCH_CONTACT_REFERENCE_DIR`. The equivalent bit-identity claim over a synthetic
      40-frame sweep is covered and passing.)*
- [ ] Analytic test: signed distance for a point-vs-plane and a sphere-vs-sphere configuration
      matches the closed-form value to `rtol=1e-6`. *(Point-vs-plane meets it. Sphere-vs-sphere
      cannot: the inscribed-mesh sag dominates at any tractable resolution, so it is asserted
      against that derived bound plus second-order convergence — see Phase 1 deviation 8.)*
- [x] Determinism test: the same frame computed twice returns bit-identical arrays.
- [ ] The field is index-aligned with the existing `contact_points` array: `len(depths) ==
      len(contact_points)` asserted for every frame of a full recording.
- [ ] Viewer launches on one recording, shows contact vertices coloured by depth with a colourbar
      reading **"Penetration depth (mm)"**, and the slider scrubs all frames without the colour
      scale changing.
- [ ] Colour limits are computed **once** over the whole recording and are constant while scrubbing
      (verified by reading the same vertex's colour at two different frames with equal depth).
- [ ] Frames with zero contact render without a blank/collapsed viewport.
- [ ] Scrubbing the slider across a 3000-frame recording stays responsive (no per-frame SDF
      recomputation — the field is precomputed before the window opens).

## Definitions

- **Contact patch**: the set of forearm-terrain vertices belonging to at least one triangle whose
  **all three** vertices satisfy `signed_distance < EPSILON` (`EPSILON = 1e-5`). This is the
  *existing* definition and is deliberately unchanged, so `contact_area` stays consistent. Formally
  `np.unique(active_tris)` — which is exactly the array that already populates `contact_points`.
- **Signed depth**: the value returned by `RaycastingScene.compute_signed_distance()` for a forearm
  vertex queried against a scene built from the **hand** mesh. **Negative means penetrating.**
  Units: **millimetres** (Kinect-native; no conversion anywhere in this pipeline).
- **Penetration depth**: `-signed_depth`, i.e. a positive magnitude in mm. This is the quantity
  *displayed* and the one the colourbar is labelled with. Storage remains signed.
- **Depth field**: the 1-D array of signed depths, one per contact-patch vertex, index-aligned with
  that frame's `contact_points` coordinate array.
- **Space 1 / Kinect native**: the raw Kinect coordinate frame, in mm, before ICP registration, PCA
  calibration and RF-centring. All PoC geometry lives here. Pairing a Space-1 hand mesh with a
  downstream forearm PLY (`forearm_rf_centered/`, `forearm_pca_calibrated/`) silently produces
  garbage depths.
- **Trustworthy** (as used in Success Criteria): satisfies the exact regression invariant, the
  analytic known-answer tests, and the determinism check — not "looks plausible in the viewer".

---

## Technical Design

### Approach

Three strictly separated filters, in the pipe-and-filter shape guide 01 recommends for work that
will later become an orchestrated task *without rewrite*:

```
[1] load_recording_geometry()   →  hand meshes, forearm meshes, timestamps
[2] signed_contact_depth_mm()   →  (points (N,3) float64, depths (N,) float64)   ← PURE
[3] ContactDepthFieldViewer     →  interactive 3D render                          ← PURE SINK
```

Filter [2] is the future postprocessing task and must be callable with nothing but geometry.
Filter [3] must not compute anything — it receives a finished field and renders it. Imports flow
one way: [3] → contract ← [2]; neither imports the other.

**The computation is extracted, not duplicated.** `_calculate_intersection_volume` is refactored to
call the new pure function and derive its scalar as `max(|field|)`. This is the DRY-critical
choice: duplicating the SDF logic into a standalone PoC module would create two definitions of
interpenetration that silently drift. The exact regression invariant is what makes this refactor
safe to perform on production code.

**Rendering uses PyVista exclusively.** This is not a preference — Open3D's Filament GUI crashes on
this machine's driver (`wglCreateContextAttribs() failed, error 87`) for
`gui.Application.create_window`, `gui.SceneWidget`, `rendering.Open3DScene` *and*
`rendering.OffscreenRenderer`. Open3D geometry/compute (`RaycastingScene`,
`compute_signed_distance`, PLY/OBJ I/O) is unaffected. See
`docs/development/knowledge-base/issue-open3d-filament-opengl-crash.md`.

**Colour limits are global, not per-frame.** `clim` is computed once over the whole recording
before the window opens. Per-frame autoscaling would make the animation lie about relative depth
(guide 02 §9, "Visualization as a pure sink"), and it independently dodges the PyVista scalar-bar
defect where `clim` only ever expands.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Extract field into pure fn; scalar derived from it** | One definition of interpenetration; exact invariant provable; the pure fn is the future postprocessing task, promotable unchanged | Touches production code | **Chosen** |
| Duplicate SDF logic in a standalone PoC module | Zero risk to the production path | Two definitions that will drift; change amplification on sign/units; unpromotable | Rejected |
| Change patch definition to per-vertex `d < 0` | More natural for a field; softer, more realistic rim | Diverges from `contact_area`; breaks the exact regression invariant that makes the PoC trustworthy | Rejected (revisit post-PoC) |
| Open3D GUI (`SceneWidget`) for the viewer | Same library as the compute; no mesh conversion | **Crashes on this driver**, all variants, all documented workarounds failed | Rejected — infeasible |
| Subclass existing `SceneViewer` + new `ScalarColoredPointCloudSequence` | Far less code; slider/checkbox scaffolding free | `SceneViewer._update_plot()` calls `plotter.clear()` every frame — the exact anti-pattern that corrupts the scalar bar; no camera persistence | Rejected |
| Copy `PostprocessedSceneViewer` skeleton | Smallest complete `QtInteractor`+`QSlider`+play+drag-throttle+deferred-render viewer; already parses contact points per frame | ~700 lines copied; adds a 6th `_parse_contact_points_cell` clone if done carelessly | **Chosen** (do not clone the parser — PoC reads geometry, not CSV) |
| VTK-native `plotter.add_slider_widget` | No Qt wiring | Wrong for a Qt window; repo convention is `QSlider`; overlays inside the interactor | Rejected |
| Compute SDF lazily in the slider callback | No upfront cost; instant open | Interactivity leaks into the compute stage; scrubbing becomes unusable | Rejected — precompute all frames |
| `jet` colourmap | Familiar | Non-monotonic lightness, false boundaries, CVD-hostile, discouraged by Nature/AGU/EGU | Rejected — use `inferno` |

### Architecture & Module Contracts

| Module / layer | Responsibility | Inputs → Outputs | Must NOT know about |
|----------------|----------------|------------------|---------------------|
| `contact_depth_field.py` (new, in `tactile_quantification/model/`) | Compute the per-vertex signed depth field for one frame | `(hand_mesh: o3d TriangleMesh, forearm_mesh: o3d TriangleMesh)` → `ContactDepthFrame` | File paths, config objects, DataFrames, PyVista, Qt, session/block/recording identity |
| `ContactDepthFrame` (frozen dataclass, same module) | The inter-stage contract | fields: `points (N,3) float64`, `signed_depth_mm (N,) float64`, `triangle_areas_mm2`, `total_area_mm2`, `mean_location` | Anything — it is data only |
| `ObjectsInteractionProcessor` (modified) | Per-frame orchestration + the existing CSV row schema | unchanged public signature | How the field is computed (delegates) |
| `poc_contact_depth_field.py` (new, in `code/scripts/_3_preprocessing/_4_somatosensory_quantification/`) | Load one recording, run the field over all frames, launch the viewer | 6 paths → in-memory `List[ContactDepthFrame]` → viewer | Rendering internals; the SDF math |
| `ContactDepthFieldViewer` (new, in `preprocessing/motion_analysis/tactile_quantification/gui/`) | Render a precomputed field series interactively | `List[ContactDepthFrame]`, context meshes, `clim` → a window | Open3D, file paths, how depth was computed, **any statistic it must derive itself** |

```
code/src/preprocessing/motion_analysis/tactile_quantification/
├── model/
│   ├── contact_depth_field.py          # NEW — pure SDF field + ContactDepthFrame
│   └── objects_interaction_processor.py # MODIFIED — delegates, derives scalar
└── gui/
    └── contact_depth_field_viewer.py    # NEW — PyQt5 + pyvistaqt.QtInteractor

code/scripts/_3_preprocessing/_4_somatosensory_quantification/
└── poc_contact_depth_field.py           # NEW — standalone driver, hardcoded-path __main__

code/tests/
└── test_contact_depth_field.py          # NEW — analytic + regression + determinism
```

Contract signature:

```python
@dataclass(frozen=True)
class ContactDepthFrame:
    frame_index: int
    time_s: float
    points: np.ndarray            # (N, 3) float64, mm, Kinect Space 1
    signed_depth_mm: np.ndarray   # (N,)   float64, negative = penetrating
    total_area_mm2: float
    mean_location: np.ndarray     # (3,)   float64, mm

def signed_contact_depth_mm(
    hand_mesh: o3d.geometry.TriangleMesh,
    forearm_mesh: o3d.geometry.TriangleMesh,
    *,
    epsilon: float = 1e-5,
) -> Optional[ContactDepthFrame]:
    """Return the per-vertex contact depth field, or None if no contact.

    None means 'no contact this frame' — a legitimate, documented empty result.
    It never means 'computation failed'; failures raise.
    """
```

The long-form record fixed here (the future Parquet schema, not written in this PoC):

| column | dtype | meaning |
|--------|-------|---------|
| `frame_index` | `int32` | Kinect frame index |
| `time_s` | `float64` | seconds |
| `x`, `y`, `z` | `float32` | contact vertex position, mm, Space 1 |
| `signed_depth_mm` | `float32` | negative = penetrating |

Note `float32` at rest (0.1 µm precision at mm scale) but `float64` in memory, so the regression
invariant stays bit-exact against the existing `float64` scalar.

### Constraints inherited from the knowledge base

Each of these is a documented, already-paid-for lesson; violating one reintroduces a solved bug.

- **CuPy first.** The driver script must `import cupy` (guarded) *before* any `preprocessing.*`
  import, at module top. Lazy import inside a function does not work — Open3D/pyk4a mutate NumPy's
  dtype registry at DLL-load time. (`note-cupy-import-order.md`)
- **Load hand poses via `HandMotionManager`, never `np.load` on the NPZ directly.** The manager
  carries the guard against negative Procrustes scale; a negative-determinant transform inverts the
  winding number and makes *every* vertex report as penetrating.
  (`bug-contact-detection-winding-inversion.md`)
- **Stay in Space 1.** Pair with the per-block `forearm_pointclouds/*_mesh.obj`, never
  `forearm_rf_centered/` or `forearm_pca_calibrated/`. (`note-spatial-alignment-pipeline.md`)
- **Surface, don't swallow, the forearm fallback warning.** `get_forearms_with_fallback` may hand
  back a *preceding block's* forearm mesh. The PoC must print which mesh it actually used.
- **No `rf_camera_settings.json` dependency.** Existing 3D viewers fail fast without it; a Space-1
  PoC has no business inheriting that coupling. Derive the camera from the contact bbox or use
  `view_isometric()`. (`note-rf-camera-settings-connections.md`)
- **`QtInteractor` is the sole central widget.** The slider goes in a bottom bar / `QDockWidget` /
  `QSplitter` — never overlaid in the same grid cell. VTK's OpenGL window captures mouse at OS
  level and no `raise_()`/translucency/focus trick routes events past it.
  (`note-rf-explorer-post-layout-bugfix-status.md`)
- **Never `remove_actor` + `add_mesh` per frame.** `add_mesh(..., copy_mesh=False)` once, then
  update `mesh["depth"]` in place and set `actor.mapper.scalar_range` directly. PyVista's
  keep-extremum scalar-bar logic only ever expands `clim`; one all-zero frame permanently corrupts
  the LUT. (Verified on PyVista 0.46.1 — the pinned version.)
- **Call `ResetCameraClippingRange()` on every empty→non-empty bounds transition**, and seed an
  invisible `pv.Box` bounds proxy. Frame 0 commonly has no contact; without this the viewport is
  blank until the user rotates. (`bug-neural-kinect-viewer-initial-render.md`)
- **Deferred first render.** `showEvent` → `QTimer.singleShot(0, ...)` → `interactor.Initialize()`,
  `render_window.SetSize(...)`, then draw frame 0. Without it the first VTK render has zero pixels.
- **Debounce the slider** with a single-shot `QTimer` (30–80 ms) between the control and the
  scene update, following `postprocessed_scene_viewer.py:791-813`.
- **Do not import batch-rendering modules** that call `matplotlib.use('Agg')` at module top — it
  preempts `Qt5Agg` and breaks the interactive window.
- **Colourmap `inferno`**, not `jet`. (`investigation-jet-colormap-perceptual-problems.md`)

### Fail-fast raise sites

Per the project mandate, and guide 02 §8. Each of these currently has no check:

- Hand mesh transform has `det(M) <= 0` → `ValueError` (winding inversion; would invert every sign).
- **All** queried forearm vertices report `d < 0` → `ValueError` (near-certain winding inversion,
  not a real whole-arm contact).
- `len(points) != len(signed_depth_mm)` → `AssertionError` (contract violation).
- `NaN`/`inf` anywhere in the field → `ValueError` (failed MANO fit propagating).
- `max(|depth|)` outside a plausible mm range (sanity band, e.g. `> 200 mm`) → `ValueError` with
  the value in the message. A max of `0.03` vs `30` is the m-vs-mm tell.
- Forearm mesh dict empty for the block → existing `ValueError`, kept.
- A frame whose hand pose is absent → explicit absent marker, **never** a zero-depth frame.
  "Zero" and "absent" must not collapse: once this field weights neural firing rate, an
  absent-read-as-zero silently down-weights real spikes.

---

## Implementation Plan

### Phase 1: Extract the field (pure function + invariant)
**Goal:** The per-vertex field exists, is proven bit-identical to the legacy scalar, and production
behaviour is unchanged.

**Started:** 2026-08-12
**Completed:** 2026-08-12

- [x] 1.1 — Create `contact_depth_field.py` with `ContactDepthFrame` and `signed_contact_depth_mm()`,
      lifting the AABB broad phase + `RaycastingScene` narrow phase + triangle mask verbatim from
      `objects_interaction_processor.py:88-190`.
- [x] 1.2 — Add the fail-fast raise sites listed above.
- [x] 1.3 — Refactor `_calculate_intersection_volume` to call it and derive
      `contact_depth = np.max(np.abs(frame.signed_depth_mm))`. Public signature unchanged.
- [x] 1.4 — Write analytic known-answer tests (point-vs-plane, sphere-vs-sphere).
- [x] 1.5 — Write the determinism test (same frame twice → `np.array_equal`).
- [x] 1.6 — Write the regression test against a committed reference CSV for one recording.

**Files Modified:**
- `code/src/preprocessing/motion_analysis/tactile_quantification/model/contact_depth_field.py` — new
- `code/src/preprocessing/motion_analysis/tactile_quantification/model/objects_interaction_processor.py` — delegate + derive
- `code/tests/test_contact_depth_field.py` — new
- `code/tests/conftest.py` — stub the `preprocessing.motion_analysis` facade root (unplanned, see below)

**Dependencies:** None

**Phase 1 deviations from the plan as written** (each deliberate; flagged for review):

1. **`ContactDepthFrame.frame_index` / `time_s` are `Optional`.** The plan's contract block gives
   them as `int` / `float`, but the plan's own signature for `signed_contact_depth_mm()` passes no
   frame identity, and `ObjectsInteractionProcessor` does not know the frame index either. They are
   keyword-only parameters defaulting to `None` — an explicit *unlabelled* marker, never a
   fabricated `0`, per the plan's own "zero and absent must not collapse" rule.
2. **`ContactDepthFrame` carries a seventh field, `normals`.** The existing `contact_info`
   visualization dict returns the forearm vertex normals sliced at the contact indices. Those
   indices are local to the AABB crop, which now lives inside the field function, so without this
   field the processor would have to redo the crop — duplicating the broad phase and creating a
   second place where the patch is defined. (The plan's Architecture table and its contract block
   already disagree on the exact field list.)
3. **`det(M) <= 0` is checked by `validate_pose_transform()`, not inside the field function.**
   `signed_contact_depth_mm()` receives an already-transformed mesh and cannot see the transform.
   The validator is exported for the Phase 2 driver to call at the point where the pose matrix is
   known. It computes the 3×3 determinant by scalar triple product rather than `np.linalg.det` —
   the LAPACK path hard-crashes in this environment (see below).
4. **`process_single_frame(current_mesh=None)` now raises instead of returning `empty_structure()`.**
   This is the plan's "a frame whose hand pose is absent → never a zero-depth frame" raise site. It
   changes no production output: `ObjectsInteractionController.run()` dereferences
   `current_mesh.vertices` before calling, so the `None` branch is unreachable from the pipeline.
5. **The `try/except Exception` around `TriangleMesh.from_legacy` was removed**, replaced by explicit
   empty-mesh preconditions. A bare `except` returning an empty result is exactly the silent
   fallback the project mandate forbids.
6. **`code/tests/conftest.py` gained a stub for `preprocessing.motion_analysis`.** That package's
   `__init__.py` is a facade that eagerly imports the GUI layer and the HaMeR client, so the new
   leaf module could not be imported on its own. This follows the file's existing pattern for
   heavyweight package roots.
7. **Task 1.6 has two tests, not one.** No reference CSV is committable (recording size, participant
   data), so the real-recording regression skips with an explicit reason unless
   `SOCIAL_TOUCH_CONTACT_REFERENCE_DIR` points at a bundle. Standing in for it — and running
   always — is a characterisation test that replays the pre-refactor algorithm, transcribed verbatim
   as an oracle, over a seeded 40-frame pose sweep and asserts every CSV field matches
   bit-identically.
8. **The sphere-vs-sphere known-answer test asserts convergence, not `rtol=1e-6`.** An inscribed
   sphere mesh under-reports penetration by `depth · (π/resolution)² / 8` — a property of the
   fixture, not of the code. The test asserts agreement within that derived bound at two
   resolutions *and* that the error quarters when the resolution doubles, which is a stronger claim
   than a single loosened tolerance would be.

**Environment note (pre-existing, not introduced here):** in the `social-touch` conda env, MKL fails
to load (`OSError 0xc06d007f`), so `np.linalg`/scipy-backed paths fail or hard-crash the interpreter.
On `dev` at `8c9e572` the suite already reports 28 failures in `test_deduplicate_xy_points.py` and a
fatal crash in `test_hand_motion_manager.py`. Phase 1 leaves that set exactly unchanged.

### Phase 2: Standalone driver
**Goal:** One recording runs end-to-end and produces an in-memory field series.

**Started:** 2026-08-12
**Completed:** 2026-08-12

- [x] 2.1 — `poc_contact_depth_field.py` with the guarded CuPy import at module top.
- [x] 2.2 — Load geometry via `HandMotionManager` (**not** raw `np.load`),
      `HandMetadataFileHandler`, `ForearmFrameParametersFileHandler`, `ForearmCatalog`,
      `get_forearms_with_fallback(use_mesh=True)`.
- [x] 2.3 — Iterate frames replicating the controller's semantics: `update_reference()` when
      `frame_id` is a key in the forearm dict; `remove_vertices_by_index(excluded_vertex_ids)`
      before contact computation.
- [x] 2.4 — Print which forearm mesh was used per block (surface the fallback).
- [x] 2.5 — Compute global `clim` over the whole recording; report frame count, contact-frame
      count, and the depth range.
- [x] 2.6 — `__main__` block with hardcoded paths following `stabilise_hand_motion.py:128-141`.

**Files Modified:**
- `code/scripts/_3_preprocessing/_4_somatosensory_quantification/poc_contact_depth_field.py` — new
- `code/tests/test_contact_depth_field.py` — one-line fix to the Phase 1 regression test
  (unplanned, see deviation 5)

**Dependencies:** Phase 1

**Phase 2 execution results** (real data, `social-touch` conda env, Open3D 0.19.0):

| Recording | Frames | Contact | No contact | Absent pose | Signed depth range (mm) | Field RAM |
|-----------|--------|---------|------------|-------------|--------------------------|-----------|
| `ST14-01 / block-order-01` | 3143 | 1208 | 1935 | 0 | `[-12.908786, +0.000004]` | **7.67 MB** |
| `ST13-03 / block-order-02` (fallback case) | 2434 | 560 | 1874 | 0 | `[-6.185465, -0.000015]` | 4.20 MB |

The 7.67 MB figure answers the plan's open memory question: a whole recording of fields is
**~8 MB of array payload**, roughly 2.5 kB/frame, so the future Parquet sidecar is small and
whole-session batching is not memory-constrained. (Points + depths + normals only; Python object
overhead excluded, ~200 extra bytes per frame.)

The `ST13-03` run exercises the forearm fallback and a mid-recording reference switch: key `0` is
reported as `FALLBACK — not a snapshot of this block` alongside the loader's own two log records
naming block 1, and key `74` names the block's own `*_frames_0074-0108_avg_N35_mesh.obj`.

**Regression invariant, measured (not merely written):** driven over `ST14-01 / block-order-01`
against the committed `*_contact_and_kinematic_data.csv`, the driver reproduces
`contact_detected` on **3143/3143** frames and `contact_depth` **bit-identically**
(`np.array_equal`, max abs diff `0.0`) on all **1208** contact frames, and
`len(signed_depth_mm) == len(contact_points)` holds on every one of them. The Success Criteria
checkboxes stay unticked because the *committed test* still skips without a bundle — this was a
driver-level measurement, not the test running.

**Open finding for Phase 3 (manual verification):** no absent-pose frames and no carried-forward
poses occurred in either recording, so the "zero vs absent" hazard is not exercised by this data.
The sign-speckle question is untouched — `max(|d|)` is bit-identical but says nothing about the
per-vertex sign distribution, which only the viewer will reveal.

**Phase 2 deviations from the plan as written** (each deliberate; flagged for review):

1. **Hand meshes are not materialised as a list.** The plan's filter `[1]` says "hand meshes";
   `RecordingGeometry` instead exposes `hand_mesh(i)`, which builds one on demand. A 3143-frame
   recording is ~110 MB of Open3D meshes that are each consumed exactly once, and the viewer needs
   arbitrary single frames, not all of them at once. One accessor also means the compute stage and
   the viewer cannot disagree about what "the hand at frame *i*" is. It returns a fresh object
   every call, which is what makes the in-place `remove_vertices_by_index` safe.
2. **The 4x4 pose matrix is rebuilt in the driver.** `validate_pose_transform()` must see the
   transform, but `HandMotionManager` applies `T·R·S` inside `__getitem__` and exposes no accessor
   for it. `RecordingGeometry.pose_matrix()` mirrors those four lines. This is genuine, knowing
   duplication; the clean fix is a `pose_matrix(index)` accessor on `HandMotionManager`, which is
   out of scope here.
3. **Three-state `FrameStatus`, and a fourth diagnostic that is deliberately *not* a state.**
   `CONTACT` / `NO_CONTACT` / `POSE_ABSENT` keep "zero" and "absent" apart as the plan requires.
   But `HandMotionManager` has no representation for an absent pose: on sticker dropout it copies
   the previous frame's transform, so a dropped frame is indistinguishable from a tracked one after
   `load()`. `POSE_ABSENT` therefore fires only on genuinely detectable absence (non-finite pose or
   vertices), and the bit-identical-pose-repeat count is reported separately as a *diagnostic*,
   explicitly labelled as the NPZ's dropout signature and explicitly not used to classify frames.
   Inventing a classification from a heuristic would have been worse than reporting the gap.
4. **Frame 0 must carry a forearm reference or the driver raises.** The controller seeds its
   processor from `next(iter(references_mesh))` — insertion order, not key 0 — which is a silent
   fallback. `get_forearms_with_fallback` documents key `0` as guaranteed, so the driver asserts it
   instead. Behaviourally identical on all real data; louder if the catalog ever changes shape.
5. **One-line fix to `code/tests/test_contact_depth_field.py` (a Phase 1 file).** Its regression
   test read the reference CSV with a bare `pd.read_csv`. pandas' default float parser perturbs
   ~9% of the values (105/1208 frames here) by exactly one ULP, so the test's `np.array_equal`
   bit-identity assertion would have failed spuriously the moment a real bundle was supplied —
   against values the pipeline had written correctly. Adding `float_precision="round_trip"` is what
   turned the measurement above from "max abs diff 1.8e-15" into "bit-identical".
6. **`get_forearms_with_fallback` provenance is captured from the logging stream.** It returns
   geometry only and emits the fallback's origin as `logging.info`/`logging.warning` on the root
   logger. Task 2.4 is satisfied by attaching a temporary handler around the call and re-emitting
   the records in the run report, rather than re-deriving the answer in parallel — a parallel
   derivation would be a second definition of "which mesh was used" and could drift.
7. **Environment: the conda env is `social-touch`, not `social-touch-env`** (CLAUDE.md names the
   latter; only the former exists and carries Open3D 0.19.0). CuPy is **not installed** in it, so
   the guarded import took its `except` branch on every run — the import-order constraint is still
   honoured for machines that do have it. Console output needs `PYTHONIOENCODING=utf-8`: an
   unrelated module on the import path prints an emoji that the cp1252 default codec cannot encode.
8. **Importing the driver transitively imports PyVista (0.47.1).** The driver itself imports
   neither PyVista nor Qt, but `preprocessing.forearm_extraction.__init__` is a facade that pulls
   `point_cloud_visualizer`. Two consequences for Phase 3: the boundary is enforced by this file
   only, not by the package; and the installed PyVista is **0.47.1**, not the 0.46.1 the plan's
   scalar-bar findings were version-verified against — re-verify the `clim` behaviour there.

### Phase 3: Viewer
**Goal:** Interactive 3D view with colourbar and time slider.

- [ ] 3.1 — `ContactDepthFieldViewer(QMainWindow)`, skeleton adapted from
      `postprocessed_scene_viewer.py`: `QtInteractor` central, bottom bar with `QSlider` +
      frame label + play/pause.
- [ ] 3.2 — Context actors: forearm mesh (grey, opaque) and hand mesh (translucent), so the
      coloured patch is anatomically interpretable rather than dots in space.
- [ ] 3.3 — Contact actor via `add_mesh(..., scalars='penetration_depth_mm', cmap='inferno',
      clim=<global>, show_scalar_bar=True, scalar_bar_args={'title': 'Penetration depth (mm)'},
      copy_mesh=False)`. **This is new ground — no existing viewer in the repo uses a scalar bar.**
- [ ] 3.4 — `_update_frame()`: update points and the scalar array **in place**; never
      `clear()`/`remove_actor()`; `ResetCameraClippingRange()` then `render()`.
- [ ] 3.5 — Invisible `pv.Box` bounds proxy + empty→non-empty clipping-range handling for
      zero-contact frames.
- [ ] 3.6 — Deferred first render via `showEvent` + `QTimer.singleShot(0, ...)`.
- [ ] 3.7 — 30–80 ms debounce `QTimer` on slider drag; play timer at 33 ms.
- [ ] 3.8 — On-screen readout: frame index, time, contact vertex count (guide 03 §7 — show the
      denominator; a 3-vertex patch and a 300-vertex patch must not look equally authoritative).
- [ ] 3.9 — `closeEvent` → `plotter.close()` to release VTK resources.

**Files Modified:**
- `code/src/preprocessing/motion_analysis/tactile_quantification/gui/contact_depth_field_viewer.py` — new
- `code/src/preprocessing/motion_analysis/tactile_quantification/gui/__init__.py` — export

**Dependencies:** Phase 2

---

## Testing Plan

### Unit Tests
- [x] Point-vs-plane: signed distance equals the closed-form value, `rtol=1e-6`.
- [x] Sphere-vs-sphere: penetration depth equals `r1 + r2 - |c1 - c2|` within the inscribed-mesh sag
      bound, with second-order convergence asserted (see Phase 1 deviation 8).
- [x] Sign convention: a vertex known to be inside the hand mesh returns a **negative** value.
- [x] Contract: `len(points) == len(signed_depth_mm)` for a synthetic contact.
- [x] No contact → returns `None`, not an empty-array frame and not a zero-depth frame.
- [x] Determinism: same inputs twice → `np.array_equal` on the depth array.
- [x] Raise on `NaN` in hand vertices.
- [x] Raise when all queried vertices are negative (winding-inversion sentinel).
- [x] Raise when `max|depth|` exceeds the sanity band.
- [x] Raise on an absent hand pose rather than reporting zero depth.
- [x] `validate_pose_transform()` rejects `det <= 0`, non-finite and non-4×4 transforms.
- [x] Contact patch of exactly one triangle → exactly three contact vertices.

### Integration Tests
- [ ] **The invariant:** for every frame of one full recording, `max(|field|)` equals the
      `contact_depth` in the committed reference CSV, bit-identical (`np.array_equal`, not
      `allclose`). This is the single test that makes the field trustworthy. *(Written; skips
      pending a recording bundle.)*
- [ ] `len(field)` equals the parsed `contact_points` length for every frame of that recording.
      *(Written; skips pending a recording bundle. Asserted and passing over the synthetic sweep.)*
- [x] Refactored `ObjectsInteractionProcessor` reproduces the legacy algorithm **in full** — every
      column, not just depth — over a seeded 40-frame pose sweep, proving the extraction changed
      nothing. The oracle is the pre-refactor code transcribed verbatim.

### Manual Verification
- [ ] Launch on one recording; confirm the colourbar reads "Penetration depth (mm)" with a sensible
      numeric range.
- [ ] Scrub the slider start→end: colour scale does not change; no flicker; no blank viewport.
- [ ] Land on a zero-contact frame: window stays rendered, context geometry visible.
- [ ] Visually confirm the deepest colour sits near the **centre** of a fingertip contact patch,
      not at its rim. *If it does not, that is a finding, not a bug to hide* — it would mean the
      geometric field disagrees with the Hertzian expectation and the weighting premise needs
      revisiting.
- [ ] Inspect for sign **speckle** (isolated vertices with inverted sign) — the expected signature
      of the non-watertight hand mesh. Record whether it occurs and how often.
- [ ] Run from a plain terminal, not only under the VS Code debugger (interactive plot windows and
      GPU/CuPy availability both behave differently under debugpy).

### Edge Cases
- [ ] Frame 0 with no contact (the documented blank-viewport trigger).
- [ ] Recording where `get_forearms_with_fallback` falls back to a previous block's mesh.
- [ ] Contact patch of exactly one triangle (3 vertices).
- [ ] Hand mesh entirely inside the forearm mesh.
- [ ] Recording with zero contact frames throughout — viewer must open and say so, not crash.

---

## Documentation Plan

- [x] Module docstring in `poc_contact_depth_field.py` listing the deliberate PoC debt explicitly:
      no persistence, whole recording in memory, hardcoded paths, no DAG integration.
- [x] Short decision record for **sign convention, units, and query direction** — the three things
      a future maintainer will otherwise re-derive incorrectly. *(Full record in
      `contact_depth_field.py`; one-line restatement plus the Space-1 pairing rule in the driver.)*
- [ ] Update `docs/development/brainstorms/per-vertex-contact-depth.md` → Status: Handed off.
- [ ] Add changelog entry `docs/changelogs/per-vertex-contact-depth-poc.md`.
- [x] Record which NPZ variant was consumed (raw vs `_stabilised`) — depth magnitudes are sensitive
      to the pose-smoothing configuration. *(Both Phase 2 runs used the **raw**
      `_handmodel_motion.npz`; the variant is a named constant in the `__main__` config block and
      the file name is echoed in every run report.)*
- [ ] **Not** updating CLAUDE.md — no architectural change lands in this PoC.

---

## Rollback Plan

1. **Phase 3 / Phase 2 only:** purely additive files. Delete
   `contact_depth_field_viewer.py`, `poc_contact_depth_field.py` and the `gui/__init__.py` export.
   Nothing else references them.
2. **Phase 1** is the only phase touching production code. `_calculate_intersection_volume` is a
   pure refactor guarded by the full-CSV reproduction test; revert that single file to restore the
   previous behaviour exactly.
3. **Data considerations:** none. No migration, no schema change, no file written by this PoC.
   Existing CSVs are neither read-modified nor invalidated — output is byte-identical by
   construction, which the integration test enforces.
4. **Rollback procedure:** `git revert` the Phase 1 commit; Phases 2–3 can be left in place
   harmlessly or deleted wholesale.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Hand mesh is not watertight** after `remove_vertices_by_index(excluded_vertex_ids)` — signed distance is formally undefined on an open mesh, and returns a *plausible number* rather than erroring | High | High | Pre-existing, inherited, **not fixed here**. The per-vertex field makes it visible for the first time (sign speckle) where `max()` hid it. Manual verification explicitly looks for it; findings feed the follow-up plan. Do not silently repair. |
| Winding inversion from negative Procrustes scale flips every sign | Med | High | Load via `HandMotionManager` (carries the guard); add the all-negative sentinel raise; `inspect_handmodel_scales.py` exists as a diagnostic |
| Refactor changes production output | Low | High | Full-CSV reproduction test, bit-identical, over a whole recording — not a spot check |
| PyVista scalar-bar `clim` corruption on an empty frame | Med | Med | Global fixed `clim`; in-place scalar update; never remove/add actor. Documented and version-verified on the pinned 0.46.1 |
| Blank viewport on zero-contact frame 0 | High | Low | Bounds proxy + `ResetCameraClippingRange()` on empty→non-empty |
| Slider unresponsive on long recordings | Med | Med | Precompute all frames before opening; debounce timer; no SDF in the callback |
| Memory: whole recording of fields held in RAM | Low | Med | One recording ≈ 3k frames × ~300 pts × 4 float64 ≈ tens of MB. Report the figure in Phase 2.5; if it surprises, that number sizes the future Parquet sidecar |
| Depth field contradicts the Hertzian premise (peak not central) | Med | High | This is the PoC's actual scientific question. Surfaced by manual verification. A negative result is a valid, valuable outcome — the geometric overlap profile is parabolic and its support is ~√2× the true Hertz contact radius, so disagreement is *expected* in degree if not in kind |
| Colourmap misleads (per-frame autoscale) | Low | High | Global `clim`, asserted in Success Criteria |
| PoC becomes permanent unreviewed infrastructure | Med | Med | Debt listed explicitly in the module docstring; Phase 1 is written to production standard precisely so promotion is a move, not a rewrite |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 — extract field + invariant | ~250 LOC + ~200 LOC tests | None |
| Phase 2 — standalone driver | ~150 LOC | Phase 1 |
| Phase 3 — viewer | ~450 LOC | Phase 2 |

---

## References

- Brainstorm: [`docs/development/brainstorms/per-vertex-contact-depth.md`](../../brainstorms/per-vertex-contact-depth.md)
- `docs/development/knowledge-base/note-somatosensory-units-and-calculations.md` — mm everywhere; existing formula
- `docs/development/knowledge-base/bug-contact-detection-winding-inversion.md` — sign inversion
- `docs/development/knowledge-base/issue-open3d-filament-opengl-crash.md` — why PyVista renders
- `docs/development/knowledge-base/note-rf-explorer-post-layout-bugfix-status.md` — Qt/VTK layout, scalar update
- `docs/development/knowledge-base/bug-neural-kinect-viewer-initial-render.md` — blank viewport
- `docs/development/knowledge-base/note-spatial-alignment-pipeline.md` — coordinate Space 1
- `docs/development/knowledge-base/investigation-jet-colormap-perceptual-problems.md` — colourmap
- `docs/development/knowledge-base/note-cupy-import-order.md` — import order
- Guides: `~/.claude/knowledge/data-pipeline-engineering/` 01 (classification), 02 (§1–§11 checklist), 03 (§4–§7 scientific correctness), 05 (§3, §6, §7 craftsmanship)
