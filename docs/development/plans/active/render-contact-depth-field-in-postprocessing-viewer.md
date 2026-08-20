# Plan: Render the Contact Depth Field in the Postprocessing Stage Viewer

**Date:** 2026-08-19
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/depth-field-postprocessing`
**Branch:** `feature/depth-field-stage-viewer`

> **Base-branch note.** This feature consumes the per-stage sidecars produced by
> `feature/depth-field-postprocessing`. If that branch is merged into `dev` before this one opens,
> rebase the base to `dev` and update this field — the dependency is on the *artifacts*, not on the
> branch itself.

---

## Overview

`PostprocessingStageViewer` shows six postprocessing stages but draws contact points in flat red,
reading only the CSV's `%.1f`-quantised `contact_points` blob. The depth-field branch now writes a
per-vertex `signed_depth_mm` sidecar next to **five** of those six stage CSVs, and nothing displays
it. This plan colours the contact points by penetration depth in the stage viewer, reusing the
adapter, colormap and clim discipline already built and tested for the Neural+Kinect viewer, and adds
an optional forearm-PLY colouring mode driven by `vertex_id`.

## Problem Statement

The depth field is propagated through all five postprocessing spatial stages, schema-v2 stamped with
`vertex_id` and reference-PLY provenance, and verified numerically — 36/36 row-count checks, bitwise
depth preservation, a 0.09999771 mm `vertex_id`↔PLY residual. But it has **never been looked at** in
any space after Kinect Space 1.

Phase 9 of `propagate-contact-depth-field-through-postprocessing.md` records this as the one class of
evidence it could not produce:

> The **render check** — no visual confirmation that the RF-centred field sits on the RF-centred
> forearm surface. The 0.09999771 mm `vertex_id` residual is the numerical stand-in for it.

The corresponding Testing Plan checkbox in that plan is still unticked. A numerical residual proves
the index and the coordinates agree with each other; it does not prove either agrees with the
anatomy. A depth patch that is internally consistent but sitting on the wrong side of the forearm, or
mirrored, or translated by a stage's origin, would pass every check that branch ran.

Concretely, today:

- `postprocessing_stage_viewer.py:397-404` adds the contact actor with `color="red"` — no `scalars=`,
  no `cmap=`, no `clim=`, and `plotter.scalar_bars` is never touched anywhere in the module.
- Four postprocessing viewers each carry a private copy of `_parse_contact_points_cell`
  (`postprocessing_stage_viewer.py:76`, `postprocessed_scene_viewer.py:106`,
  `before_after_step_viewer.py:64`) and none of them opens a parquet.
- The visualisation DAG (`configs/postprocess_visualization_dag.yaml`) has no depth task and no
  depth option.

The cost is not cosmetic. The depth field exists to serve the IFF weighting
`w(depth_i) x RF_sensitivity(position_i)` in the RF-centred frame. Shipping that weight off a field
no one has ever seen in that frame means a spatial defect would first surface as a wrong scientific
result, downstream, in the other repository.

## Goals

### In Scope

1. Colour the stage viewer's contact points by `penetration_depth_mm` for every stage whose sidecar
   exists, with a **global** (whole-recording, per-stage) colour range and a scalar bar.
2. Resolve each stage's sidecar through the existing `depth_field_path_for_csv()` and hand the viewer
   a **lazy loader**, never a path and never a loaded table.
3. Join depth rows to the displayed frame by Kinect `frame_index`, never by row position.
4. Validate each stage's declared `coordinate_space` against the stage the user selected, and raise
   on a mismatch.
5. Surface absence explicitly — a disabled control with a message naming the task that produces the
   artifact — and distinguish it from a corrupt sidecar, which must raise.
6. A "Colour by depth" toggle per the `neural_kinect_scene_viewer` precedent, with the flat-red
   rendering retained as the off state.
7. *(Separable, Phase 5)* Colour the forearm PLY itself by joining `vertex_id`, for the three stages
   that carry one.

### Out of Scope

- **Any recomputation in the GUI.** No re-deriving depth, colour range, nearest vertex, or per-frame
  maxima. If a number is needed that the sidecar does not carry, the *producer's* contract is
  extended — see `depth_field_stage_io.py:27-34` on why re-derivation against the parquet picks
  different vertices than the CSV did.
- **A nearest-vertex fallback for stages without `vertex_id`.** Stages 0-2 have no `vertex_id`; that
  is a fact about the pipeline, not a gap to paper over.
- **Recording-wide aggregation onto the forearm** ("which vertices were ever touched, how deep"). A
  genuinely useful view, but it is an aggregation — it belongs in an adapter with its own plan, not
  bolted onto a per-frame renderer.
- **The other three postprocessing viewers** (`postprocessed_scene_viewer`, `before_after_step_viewer`,
  `forearm_stage_inspector`). One viewer, done properly, first.
- **Adding a `blocks_filtered/` stage to the dropdown.** See Definitions — stage 0 legitimately has no
  sidecar. Adding the filtered stage is a separate, defensible change; it is not this one.
- **Streaming / row-group parquet reads.** No streaming reader exists in the repo; the mitigation on
  offer is deferred whole-file reads plus a bounded cache. Building one is not justified at this size.
- **Deduplicating `_parse_contact_points_cell` across the four viewers.** Noted as debt; not this plan.
- **Any change to the depth-field producers or the postprocessing DAG.** This is a read-only consumer.

## Success Criteria

- [ ] Opening any of the five sidecar-bearing stages colours the contact points by penetration depth,
      with a scalar bar titled "Penetration depth (mm)". *(Phase 6: the **state** is measured on real
      ST14-02 artifacts for all five stages — mapper bound to `penetration_depth_mm`, `scalar_range`
      = `series.clim_penetration_mm`, exactly one bar titled "Penetration depth (mm)". Left unticked
      because "colours" is a visual claim and nothing was seen; that is Phase 6.3.)*
- [x] The colour range for a given stage is identical on every frame of that stage — asserted by
      reading `actor.mapper.scalar_range` on at least three frames including one with no contact and
      one immediately after a stage switch. *(Phase 3: six frames, three of them empty, plus one
      post-switch; all `(0.25, 9.5)`.)*
- [x] Switching stages replaces the scalar bar and its range; no bar from a previous stage survives,
      and `plotter.scalar_bars` contains exactly one depth bar at any time. *(Phase 3: 1 → 1 → 0 → 1
      across stages 5 → 1 → 0 → 4.)*
- [x] Stage 0 (`blocks_merged/`) shows a disabled "Colour by depth" checkbox whose tooltip names the
      producing task, and renders flat-red contact points without error. *(Phase 4: box disabled and
      unchecked; tooltip byte-identical to `StageDepthField.message`, which names
      `filter_contact_depth_field_by_neural_quality`'s output directory and points at stage 1; the
      contact actor is in flat-red mode with no scalar bar; no exception. The word "flat-red" here is
      `scalar_visibility == False` with `GetProperty()` colour `(1, 0, 0)` — seeing it is Phase 6.)*
- [x] A frame's drawn depth values equal `series.frame(kinect_df["frame_index"].iloc[pos])[1]`
      exactly — verified by a headless test on a synthetic sidecar whose `frame_index` values are
      deliberately non-contiguous and not equal to row position.
- [x] A sidecar whose `coordinate_space` disagrees with the selected stage raises `ValueError` naming
      both spaces and the file.
- [x] A present-but-undecodable sidecar raises; it does not fall back to flat colour.
- [x] Constructing the six stage loaders reads zero bytes — asserted with a counting reporter, as
      `test_neural_kinect_depth_field_view.py:441` does for blocks.
- [x] *(Phase 5)* On stages 3-5, the forearm PLY colours by the current frame's `vertex_id` join, and
      untouched vertices are visually distinct from zero-depth vertices. *(Phase 5: the join is
      measured — vertex 4 lit at 0.50 mm on frame 7, back to NaN on frame 19; a genuine 0.00 mm
      contact survives as 0.0 while every untouched vertex is NaN, and NaN is bound to an explicit
      `nan_color` of #a0a0a0 rather than clamped to the bottom of the ramp. \Visually\ — seeing the
      grey next to the shallowest inferno — is Phase 6.)*
- [ ] **The render check that Phase 9 could not perform:** on `2022-06-15_ST14-02` the RF-centred
      depth patch is visually seated on the RF-centred forearm surface, on the correct side, moving
      coherently with the hand across the block. Recorded with screenshots in this plan.
- [x] Full suite green; the new adapter tests import no Qt, VTK or Open3D. *(Phase 6.7: 674 passed,
      7 skipped, `MKL_THREADING_LAYER=TBB`; the Phase 2 static AST test asserts the leaf and its
      adapter import no GUI toolkit.)*

## Definitions

- **Stage** — one entry of `STAGE_LABELS` (`postprocessing_stage_viewer.py:43-50`); six of them. Not
  the same as a *coordinate space* (four) nor a *pipeline output directory* (seven, because
  `blocks_filtered/` exists and is not in the dropdown).
- **Sidecar-bearing stage** — a stage whose CSV has a sibling parquet under
  `depth_field_path_for_csv()`. Concretely stages 1-5 (`blocks_registered`, `blocks_deduped`,
  `blocks_projected`, `blocks_pca_calibrated`, `blocks_rf_centered`). **Stage 0 is not one**: it reads
  `blocks_merged/`, which is written *before* `filter_contact_depth_field_by_neural_quality`, whose
  output sidecar lands in `blocks_filtered/`. This is expected, not a bug, and must resolve to the
  ordinary absent state.
- **Global clim** — `ContactDepthFieldSeries.clim_penetration_mm`, computed once over the entire
  recording for that stage (`contact_depth_field_series.py:340-341`). Testably: the value passed to
  `add_mesh(clim=...)` and re-asserted after every `DeepCopy` is `series.clim_penetration_mm`, and is
  never recomputed from a frame's own data.
- **Absent** — the sidecar file does not exist. Testably: `ContactDepthFieldResolution.series is None`
  with a non-empty `message`. Distinct from **corrupt** (file exists, cannot be decoded → raise) and
  from **no contact this frame** (`series.frame(i) is None` → draw nothing).
- **Depth colouring active** — `self._depth_series is not None and self._colour_contact_by_depth`; a
  data fact conjoined with a view preference, kept as separate attributes per
  `neural_kinect_scene_viewer.py:1484-1491`.
- **Wrong-space** — `series.coordinate_space != EXPECTED_SPACE_BY_STAGE[idx]`. The expected map is
  stages 1, 2, 3 → `icp_registered`; stage 4 → `pca_calibrated`; stage 5 → `rf_centered`.

---

## Technical Design

### Approach

Mirror the Neural+Kinect viewer's depth integration, which is already the in-tree answer to this exact
problem in a different viewer. Three properties make it the right thing to copy rather than adapt:

1. **The adapter is a leaf.** `merging/contact_depth_field_series.py` imports pandas, numpy and the io
   module — no Qt, no VTK. Its ~40 tests run headless. Anything the stage viewer needs that the
   adapter can provide should be provided *there*, keeping the widget's addition to render code only.
2. **The injection contract is a zero-arg loader, not a path.** `make_contact_depth_field_loader`
   builds a closure that reads nothing; `BoundedContactDepthFieldCache` bounds residency. The stage
   viewer has the same shape of problem as the batch viewer (six stages instead of a hundred blocks),
   so the same contract applies without modification.
3. **The layering is already established.** `postprocessing_stage_viewer.py:36` imports
   `NeuralDataPanel` from `merging.gui.neural_kinect_scene_viewer`, and
   `postprocessed_scene_viewer.py:79-80` and `before_after_step_viewer.py:56` do the same. Importing
   `merging.contact_depth_field_series` introduces no new dependency direction. The stage-boundary
   rule in CLAUDE.md forbids importing from *analysis*, which this does not.

The path-resolution hook is `resolve_stage_paths()` in `postprocess_visualization.py:252-316`, which
already builds all six `StagePaths`. It gains one field per stage, built from the CSV path it has
already computed. The viewer never learns that parquet files exist.

Two deliberate departures from the merging precedent:

- **Per-stage rather than per-block bar lifecycle.** `_on_stage_changed` calls `plotter.clear()`
  (`:516`), which destroys the scalar bar; and each stage carries its own global clim. So the bar is
  registered inside `_init_actors()` — which is already re-run per stage switch — and the "remove the
  previous bar first" step of `neural_kinect_scene_viewer.py:1769-1770` should be unnecessary here
  because `clear()` has already done it. This must be *verified*, not assumed, hence a success
  criterion and a fallback task.
- **Space validation.** The merging viewer sees only `kinect_space_1` and carries `coordinate_space`
  "so a caller can refuse to draw a wrong-space field" without any caller doing so. This viewer sees
  four spaces across six stages, so it becomes the first caller that actually refuses. Fail-fast per
  CLAUDE.md.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Reuse `merging.contact_depth_field_series` via a loader on `StagePaths` | Zero new format knowledge in the GUI; ~40 existing tests cover the read path; layering precedent already in the target file | `ContactDepthFieldSeries` drops `vertex_id`, so Phase 5 needs the DTO widened | **Chosen** |
| Promote the series module to a shared package first | "Cleaner" layering on paper | Pure churn — `postprocessing → merging` imports already exist in three files; renames a well-tested module for no behavioural gain; YAGNI (`05` §6) | Rejected |
| Read the parquet directly in `_load_stage_data` | Fewest files touched | Puts schema knowledge in a 685-line widget that already does loading, UI, actors, camera and playback — the change-amplification smell (`05` §4); violates "visualization as a pure sink" (`02` §9); untestable without Qt | Rejected |
| Colour from the CSV's existing `contact_depth` scalar | No new I/O at all | `contact_depth` is one scalar **per frame** (the max), not per vertex — it cannot colour a patch. It is also `%.1f` text | Rejected — answers a different question |
| Per-frame autoscaled colour range | Every frame uses the full colormap; looks vivid | Makes the animation lie about relative depth; rejected twice already with measurements (`contact_depth_field_viewer.py:46-62`) | Rejected |
| Nearest-vertex snapping to give stages 0-2 a PLY colouring | Uniform feature across all stages | Exactly the re-derivation `depth_field_stage_io.py:27-34` forbids; `bug-rf-explorer-nearest-vertex-distance.md` measured 29 mm off-surface overshoot from the analogous shortcut | Rejected |
| A separate new depth-only viewer | No risk to the stage viewer | A seventh viewer to maintain; loses the whole point, which is seeing depth *at each stage* next to the geometry that stage produced | Rejected |

### Architecture Constraints (from the knowledge base)

Binding, with sources:

1. **`inferno`, never `jet`** — `investigation-jet-colormap-perceptual-problems.md`. Import
   `COLORMAP` rather than restating it.
2. **Explicit `clim` on every `add_mesh`, re-asserted after every `DeepCopy`** — measured on the
   installed PyVista 0.47.1 (`contact_depth_field_viewer.py:46-62`): without it the mapper tracks each
   frame's own data range.
3. **`allow_empty_mesh` for zero-point frames**; the empty PolyData must still carry the scalar array
   or the mapper loses its binding (`contact_depth_field_viewer.py:797-805`).
4. **Do not replace the `pv.Box` bounds proxy with `pv.Sphere()`** — `pv.Sphere()` hard-crashes this
   interpreter (`0xC06D007F`). The existing proxy at `postprocessing_stage_viewer.py:432-441` is
   deliberate.
5. **Negate exactly once, in the adapter.** `penetration_depth_mm = -signed_depth_mm`
   (`contact_depth_field_series.py:329`); the display clim is `(-signed_max, -signed_min)`, since
   negation reverses order. The viewer performs no arithmetic on depth.
6. **Never ffill or interpolate the field** — row *i* of frame *n* is not the same vertex as row *i*
   of frame *n+1* (`contact_depth_field_io.py:92-99`).
7. **`validate_vertex_ids_against_reference` before any join**, and it takes a vertex *count*, never a
   mesh (`contact_depth_field_io.py:772-881`). A re-dedup at a different epsilon renumbers every
   vertex silently.
8. **Camera comes from scene bounds** — do not introduce `rf_camera_settings.json`, an analysis-stage
   artifact (`note-rf-camera-settings-connections.md`). Extend the existing `_compute_camera_params`
   (`:563`) if anything.
9. **No new blanket `try/except`** around the parquet read; the module already has three
   (`:462-464`, `:593-594`, `:609-610`) and `02` §8 forbids adding a fourth.
10. **`blockSignals` guard when seeding the checkbox**, so a fieldless stage does not overwrite the
    persisted preference (`neural_kinect_scene_viewer.py:1583-1586`; recursion hazard documented in
    `note-qt-itemchanged-signal-recursion.md`).

### Architecture & Module Contracts

| Module / layer | Responsibility | Inputs → Outputs | Must NOT know about |
|----------------|----------------|------------------|---------------------|
| `merging/contact_depth_field_series.py` *(existing; widened in Ph.5)* | Parquet → validated in-memory series with a global clim | `Path` → `ContactDepthFieldSeries` \| absent-report | Qt, VTK, stages, sessions, file layout |
| `postprocessing/gui/stage_depth_field.py` *(new leaf)* | Stage-aware policy: expected space per stage, space validation, absent/disabled messaging, PLY scalar assembly | `(stage_idx, loader)` → `StageDepthField` | Qt, VTK, plotters, actors, widgets |
| `postprocess_visualization.py::resolve_stage_paths` *(existing)* | Turn a `KinectConfig` into six `StagePaths`, now each with a loader | `KinectConfig` → `List[StagePaths]` | Parquet schema, colormaps, rendering |
| `postprocessing/gui/postprocessing_stage_viewer.py` *(existing)* | Render; own actors, widgets, camera, playback | `StagePaths[]` → a window | Parquet, schema versions, sign convention, clim arithmetic, path derivation |

```
StagePaths (postprocessing_stage_viewer.py:61-68) gains one field:

    depth_field_loader: Optional[ContactDepthFieldLoader] = None
        # zero-arg callable; returns ContactDepthFieldSeries or None.
        # Default None keeps every existing construction site valid.

New leaf — code/src/postprocessing/gui/stage_depth_field.py:

    EXPECTED_SPACE_BY_STAGE: Mapping[int, str]
        # {1,2,3: icp_registered, 4: pca_calibrated, 5: rf_centered}
        # stage 0 is absent from the map by construction

    @dataclass(frozen=True)
    class StageDepthField:
        series: Optional[ContactDepthFieldSeries]
        message: str                                 # non-empty; shown as the checkbox tooltip
        @property is_present -> bool

    def resolve_stage_depth_field(stage_idx: int,
                                  loader: Optional[ContactDepthFieldLoader]) -> StageDepthField
        # Calls the loader, validates coordinate_space against EXPECTED_SPACE_BY_STAGE,
        # raises ValueError on mismatch, TypeError on a wrong return type.

    def forearm_depth_scalars(series, frame_index, vertex_count) -> Optional[np.ndarray]   # Phase 5
        # (V,) float array, NaN where untouched, for PLY colouring. Pure numpy.
```

The new leaf lives under `gui/` for locality but imports nothing from Qt or VTK — the same arrangement
that makes `contact_depth_field_series.py` testable while sitting next to viewers.

---

## Implementation Plan

### Phase 1: Wire the loader through the stage contract
**Goal:** Every stage carries a lazy depth-field loader. Nothing renders yet, nothing reads yet.

**Started:** 2026-08-20T06:05Z  **Completed:** 2026-08-20T06:40Z

- [x] 1.1 — Add `depth_field_loader: Optional[ContactDepthFieldLoader] = None` to `StagePaths`.
      Defaulted, so no existing construction site breaks.
- [x] 1.2 — In `resolve_stage_paths`, build one loader per stage from the CSV path it already
      computes, via `depth_field_path_for_csv(csv)`. Guard `csv_path is None` → loader `None`.
      Wrap the `ValueError` from a non-conforming CSV name with the stage label; do not swallow it.
- [x] 1.3 — Construct a single `BoundedContactDepthFieldCache` shared by the six loaders; name the
      `maxsize` constant with its rationale in a comment (six stages of one block, ~12 MB per stage
      on disk per the Phase 9 sizing table — larger once parsed into per-frame arrays, so residency
      is bounded rather than unlimited).
- [x] 1.4 — Route the loaders' `report` callback to the same `print` channel the script already uses,
      so an absent sidecar announces itself once in the console.
- [x] 1.5 — Assert by test that building all six loaders performs zero reads.

**Files Modified:**
- `code/src/postprocessing/gui/postprocessing_stage_viewer.py` — `StagePaths` gains one field
- `code/scripts/postprocess_visualization.py` — `resolve_stage_paths` builds loaders; new imports
- `code/tests/test_stage_depth_field.py` *(new)* — laziness and path-pairing tests

**Dependencies:** None

### Phase 2: The stage-aware leaf
**Goal:** All policy — expected space, validation, messaging — in one Qt-free module with tests.

**Started:** 2026-08-20T06:41Z  **Completed:** 2026-08-20T06:51Z

- [x] 2.1 — Create `code/src/postprocessing/gui/stage_depth_field.py` with
      `EXPECTED_SPACE_BY_STAGE`, `StageDepthField`, `resolve_stage_depth_field`.
- [x] 2.2 — Space validation: raise `ValueError` naming the stage label, the expected space, the
      declared space and the file path. Stage 0 is absent from the map — resolving it returns the
      absent state without consulting a loader.
- [x] 2.3 — Type guard on the loader's return, mirroring `neural_kinect_scene_viewer.py:1178-1188`
      (`TypeError`, with the "drawing whatever this is" rationale).
- [x] 2.4 — Absent messaging: for stages 1-5 name the postprocessing DAG task that produces the
      sidecar; for stage 0 state plainly that `blocks_merged/` precedes the depth field and point at
      the ICP-registered stage. Every message non-empty.
- [x] 2.5 — Let a corrupt-but-present sidecar propagate as `ValueError`. Explicitly no `except`.
- [x] 2.6 — Unit tests on synthetic sidecars written with the production writer: correct space per
      stage, each mismatch pair, absent, corrupt, wrong return type, stage 0.

**Contract correction — the signature is `(stage_idx, loader, sidecar_path)`.**
Task 2.2 requires the mismatch error to name the *file*, but a
`ContactDepthFieldLoader` is a zero-argument callable that deliberately hides its path, and
`ContactDepthFieldSeries` carries none either — so the path is unreachable through the two-argument
signature this plan's contract table stated. The signature is widened by one **messaging-only**
parameter: the leaf never opens it, and passing a loader without it raises rather than degrading the
message. Phase 3 must therefore have `StagePaths` carry the sidecar path beside the loader, derived
where the loader already is (`resolve_stage_paths`); the widget forwards two opaque fields and still
derives no parquet path of its own.

**Two departures from the stated Files Modified, both to keep the leaf headless:**
- `STAGE_LABELS` now *lives* in `stage_depth_field.py` and is re-exported from
  `postprocessing_stage_viewer.py`, because the validation messages name stages and a second copy of
  the six strings would drift. Every existing importer is unaffected.
- `postprocessing/gui/__init__.py` eagerly imports the four Qt/VTK/Open3D viewers, so the export
  added there cannot be the test import path. `conftest.py` stubs the `postprocessing.gui` package
  root — the mechanism it already uses for five other heavyweight roots — so the leaf imports on its
  own. A static AST test asserts the leaf and its adapter import no GUI toolkit; `sys.modules` would
  be useless for this in a full-suite run.

**Files Modified:**
- `code/src/postprocessing/gui/stage_depth_field.py` *(new, ~120 LOC)*
- `code/src/postprocessing/gui/__init__.py` — export
- `code/src/postprocessing/gui/postprocessing_stage_viewer.py` — `STAGE_LABELS` moved to the leaf and
  re-exported
- `code/tests/conftest.py` — stub the `postprocessing.gui` root
- `code/tests/test_stage_depth_field.py` — extended

**Dependencies:** Phase 1 (for the loader type; developable in parallel)

### Phase 3: Depth-coloured contact points
**Goal:** The feature, visible.

**Started:** 2026-08-20T07:05Z  **Completed:** 2026-08-20T07:55Z

- [x] 3.1 — In `_load_stage_data`, call
      `resolve_stage_depth_field(stage_idx, sp.depth_field_loader, sp.depth_field_path)`
      and store `self._depth_field`. No parquet knowledge enters the widget.
- [x] 3.2 — Build the **frame_index map**: `self._frame_indices = self._kinect_df["frame_index"]`,
      and look up `series.frame(int(self._frame_indices.iloc[pos]))`. Raise if the column is missing
      rather than falling back to positional indexing — this is the single most dangerous shortcut
      available here.
- [x] 3.3 — In `_init_actors`, when depth colouring is active, add the contact actor with
      `scalars=CONTACT_SCALAR_NAME`, `cmap=COLORMAP`, `clim=series.clim_penetration_mm`,
      `show_scalar_bar=True` and the `scalar_bar_args` shape from
      `contact_depth_field_viewer.py:506-522` (white text on the black background). Keep
      `GetProperty().SetColor(1,0,0)` so the flat-red off state is intact.
- [x] 3.4 — In `_update_frame`, set the scalar array on the replacement PolyData before `DeepCopy`,
      `set_active_scalars`, then re-assert `actor.mapper.scalar_range = clim`.
- [x] 3.5 — Empty-frame handling: an `_empty_contact_polydata()` equivalent that still carries the
      scalar array; keep `allow_empty_mesh`.
- [x] 3.6 — Verify the scalar bar does not survive a stage switch, and that exactly one exists after
      each switch. If `plotter.clear()` turns out not to remove it, add the explicit
      `remove_scalar_bar` guard from `neural_kinect_scene_viewer.py:1769-1770`.
- [x] 3.7 — Confirm the depth series replaces the CSV blob as the geometry source when active (as
      `neural_kinect_scene_viewer.py:2075-2086` does), so the drawn points are the parquet's float32
      coordinates, not the `%.1f` CSV ones — and note in a comment that the two differ.

**The join was extracted, not written inline.** Task 3.2's expression lives in the Phase 2 leaf as
two pure functions — `kinect_frame_indices(kinect_df, source)` and
`depth_frame_at_position(series, frame_indices, position)` — because the viewer cannot be
constructed headlessly in this environment (see below) and the positional-join hazard is precisely
the thing that must be tested rather than argued. The widget calls both and holds no join logic of
its own. `kinect_frame_indices` raises on a missing column, on a missing/non-finite value and on a
non-integral one; `depth_frame_at_position` raises `IndexError` rather than clamping a position and
`TypeError` when called without a series. There is no positional path anywhere.

**The sidecar path is now a `StagePaths` field**, per the Phase 2 contract correction:
`depth_field_path` is derived in `resolve_stage_paths` beside the loader (new
`_resolve_stage_sidecar_path`, split out of `_build_stage_depth_field_loader`) and forwarded
verbatim. The widget derives no parquet path.

**3.6 measured, not assumed:** `plotter.clear()` **does** destroy the scalar bar on the installed
PyVista 0.47.1 — after a switch to stage 0 `plotter.scalar_bars` is empty, and after a switch to any
sidecar-bearing stage it holds exactly one `"Penetration depth (mm)"` bar. The
`remove_scalar_bar` guard from `neural_kinect_scene_viewer.py:1769-1770` is therefore **not**
added; adding it would be dead code. A comment in `_init_actors` records why.

**Headless verification of the render path.** `QtInteractor` cannot initialise under
`QT_QPA_PLATFORM=offscreen` in this environment — VTK's `RenderWindowInteractor.initialize` blows
the stack (`0xC00000FD`). Substituting an off-screen `pv.Plotter` for the `QtInteractor` class only
(every other line executed is the viewer's own) drove the real
`_load_stage_data` → `_init_actors` → `_update_frame` → `_on_stage_changed` path on a synthetic
six-stage session and measured:

| Property | Measured |
|----------|----------|
| Global clim, stage 5 | `(0.25, 9.5)` = `series.clim_penetration_mm`, never recomputed |
| `mapper.scalar_range` over 6 frames (3 of them empty) | `(0.25, 9.5)` on every one |
| `mapper.scalar_range` immediately after a stage switch | `(0.25, 9.5)` |
| `plotter.scalar_bars` on stages 5 → 1 → 0 → 4 | 1 → 1 → 0 → 1 bar |
| Geometry source, stage 5 frame 7 | `(1.53, 2.57, 3.51)` — the parquet float32, not the CSV's `%.1f` `(1.5, 2.6, 3.5)` |
| Stage 0 | flat red, CSV blob geometry, no bar, absent-message present |
| Reads performed across the four stage visits | 3 — one per sidecar-bearing stage opened |

Not verified here, and left to Phase 6: anything requiring a real window — visible colour, bar
legibility, playback frame rate, and the render check itself.

**Files Modified:**
- `code/src/postprocessing/gui/postprocessing_stage_viewer.py` — `StagePaths.depth_field_path`,
  `_load_stage_data`, `_init_actors`, `_update_frame`, `_empty_contact_polydata`, new imports of
  `CONTACT_SCALAR_NAME` / `SCALAR_BAR_TITLE` / `COLORMAP` and of `contact_polydata`
- `code/src/postprocessing/gui/stage_depth_field.py` — `FRAME_INDEX_COLUMN`,
  `kinect_frame_indices`, `depth_frame_at_position`
- `code/scripts/postprocess_visualization.py` — `_resolve_stage_sidecar_path`; the six
  `StagePaths` carry their sidecar path
- `code/tests/test_stage_depth_field.py` — the frame-join section

**Dependencies:** Phases 1, 2

### Phase 4: The "Colour by depth" control
**Goal:** User control, and honest absence.

**Started:** 2026-08-20T07:56Z  **Completed:** 2026-08-20T08:20Z

- [x] 4.1 — Extend `_add_group` with an `extra_widgets` parameter, matching the merging viewer's
      `_add_object_group` signature so the two panels stay recognisably the same.
- [x] 4.2 — Add the `QCheckBox("Colour by depth")` to the Contact Points group;
      `setEnabled(self._depth_field.is_present)`; tooltip = the stage's actual `low..high` mm when
      present, else `StageDepthField.message`.
- [x] 4.3 — Seed `setChecked(...)` inside `blockSignals(True)` so a stage without a field does not
      clear the preference on switch.
- [x] 4.4 — Handler toggles `mapper.scalar_visibility` and the bar's visibility without re-adding
      actors (`_apply_contact_scalar_mode`, `neural_kinect_scene_viewer.py:2324-2339`); re-assert
      `scalar_range` on the way through.
- [x] 4.5 — Register the layer in the `actor_map` at `:652` if a new actor is introduced (it should
      not be — the same contact actor is recoloured).
- [x] 4.6 — Persist the preference across stage switches as a plain attribute.

**4.5 required no change, and that was measured rather than assumed.** Both modes drive the *same*
contact actor, so `_on_point_size_changed`'s actor map needs no new entry; the smoke drive asserts
`len(plotter.renderer.actors)` is unchanged across a full off → on round trip.

**The two facts are two attributes.** `self._colour_contact_by_depth` (the view preference, set in
`__init__` and never touched by `_load_stage_data`) is conjoined with `self._depth_series is not None`
(the data fact) in a new `_depth_colouring_active` property, mirroring
`neural_kinect_scene_viewer.py:1484-1491`. A stage with no sidecar and a stage the user unchecked are
therefore distinguishable, and neither can overwrite the other.

**`_init_actors` applies the preference after adding the actor.** The actor is still built in depth
mode whenever a field exists — the `clim`, the cmap and the bar are registered exactly as Phase 3 left
them — and `_apply_contact_scalar_mode()` then switches scalar visibility off if the user's preference
says so. Entering a stage with the box unchecked therefore opens flat-red with the bar hidden, rather
than flashing coloured for a frame. `_update_frame` continues to re-assert only `scalar_range`, as the
merging viewer does: `DeepCopy` replaces the *dataset*, and `scalar_visibility` lives on the mapper,
so it survives untouched — asserted after a frame change in the off state.

**Headless verification.** The same substitution Phase 3 used (an off-screen `pv.Plotter` for the
`QtInteractor` class only; `QtInteractor` still cannot initialise offscreen here) drove the real
`__init__` → `_build_right_panel_controls` → `_init_actors` → checkbox → `_on_stage_changed` path on a
synthetic six-stage session. The checkbox was located by walking the panel *layout* rather than
`findChildren`, because a stage switch `deleteLater()`s the previous boxes and they stay children
until the DeferredDelete is delivered — `findChildren` hands back the previous stage's widget and
quietly tests nothing.

| Property | Measured |
|----------|----------|
| Stage 5 on open | box enabled, checked; `scalar_visibility` on; bar visible; range `(0.25, 9.5)` |
| Tooltip, field present | names the stage's own `0.25` to `9.50` mm range |
| Tooltip, field absent (stage 0) | byte-identical to `StageDepthField.message`; box disabled, unchecked |
| Unchecked | `scalar_visibility` off, bar `SetVisibility(0)`, range still `(0.25, 9.5)` |
| Re-checked | `scalar_visibility` on, bar visible, range still `(0.25, 9.5)` |
| Actor count across the off → on round trip | unchanged — no `remove_actor` / `add_mesh` cycle |
| Frame change while unchecked | stays off; range still `(0.25, 9.5)` |
| Preference OFF, walked 5 → 0 → 4 | still off on stage 4; bar hidden — stage 0 did not clobber it |
| Preference ON, walked 4 → 0 → 1 | still on on stage 1; exactly one bar; range `(0.25, 9.5)` |

Not verified here, and left to Phase 6: anything requiring a real window — that the points are
*visibly* flat red when off and *visibly* inferno-ramped when on, and that the bar is legible.

**Files Modified:**
- `code/src/postprocessing/gui/postprocessing_stage_viewer.py` — `__init__`
  (`_colour_contact_by_depth`), `_depth_colouring_active`, `_add_group`,
  `_build_right_panel_controls`, `_init_actors`, `_apply_contact_scalar_mode`,
  `_on_contact_depth_colour_changed`

**Dependencies:** Phase 3

### Phase 5: Forearm PLY colouring by `vertex_id` *(separable)*
**Goal:** Depth on the surface, not just on the patch. **Revertible on its own** — Phases 1-4 deliver
the core value without it.

**Started:** 2026-08-20T08:35Z  **Completed:** 2026-08-20T09:30Z

- [x] 5.1 — Widen `ContactDepthFieldSeries` with
      `vertex_id_by_frame: Optional[Dict[int, np.ndarray]] = None`, populated only when the table
      carries the column. A pure addition; v1 and `vertex_id`-less v2 sidecars keep working, and every
      existing merging test must stay green untouched.
- [x] 5.2 — Carry the reference-PLY provenance triple onto the series so the join can be validated.
- [x] 5.3 — `forearm_depth_scalars(series, frame_index, vertex_count) -> Optional[np.ndarray]` in the
      leaf: call `validate_vertex_ids_against_reference` with `len(ply.points)` **first**, then
      scatter. NaN for untouched vertices, so they are visually distinct from a genuine 0 mm.
- [x] 5.4 — Switch the forearm actor from `scalars="colors", rgb=True` (`:377-395`) to scalar mode
      when the layer is on, and back when off. Only for stages 3-5; the control is absent, not merely
      disabled, on stages 0-2 — there is no `vertex_id` and no honest thing to show.
- [x] 5.5 — Handle `nan_color` explicitly so untouched vertices read as the plain forearm.
- [x] 5.6 — Tests: correct scatter, NaN placement, provenance mismatch raises, absent `vertex_id`
      returns `None` rather than an empty array.

**5.1 was a pure addition, and that is measured, not asserted.** `git diff` on
`test_neural_kinect_depth_field_view.py` is `41 insertions(+), 0 deletions(-)` — every existing test in
that file passes **unmodified**; the only change is two appended tests stating the property the others
imply (a merging sidecar carries no vertex identity; the widened fields are optional at construction).
The two new fields are appended and defaulted, and every construction site in the tree already used
keyword arguments, so positional construction could not break either.

**The two halves travel together, and the DTO enforces it.** `vertex_id_by_frame` without
`reference_ply_provenance` raises: an index whose mesh identity is unknown cannot be validated against
anything, and provenance for an absent index describes nothing. The loader populates both or neither,
keyed on `VERTEX_ID_COLUMN in table.columns`. `int32` is **asserted, not requested** — passing
`dtype=np.int32` to `to_numpy` would silently narrow a column that had no business being int64, and the
whole value of the index is that it is the one the projection stage wrote.

**Stages 3-5 is a consequence, not a constant.** The control is gated on
`series.has_vertex_ids` — a fact read off the artifact — rather than on a hardcoded `{3, 4, 5}`. The two
coincide today; where they would not is a v1 sidecar sitting at stage 4, and there the data-driven test
is the correct one. `_forearm_depth_available` conjoins three independent facts: a field, a `vertex_id`
in it, and a forearm with vertices for the index to address.

**Duplicate `vertex_id` within one frame resolves to the deeper, not to the last row.** This is not a
degenerate case: `depth_field_stage_io.py:731-734` states that projection "is a per-point
nearest-neighbour lookup with no uniqueness constraint, so two rows may legitimately address one
vertex". Plain fancy-index assignment would resolve that by row order — an arbitrary choice that
changes under the loader's own frame sort. `np.maximum.at` over a `-inf`-seeded set of touched entries
is order-independent, and "how hard was this piece of skin pressed" is the conservative reading. NaN
still means untouched: seeding with `-inf` rather than leaving NaN is what stops `maximum` propagating
NaN over the touched vertices.

**One expression turns a slider position into a frame key, and now two joins share it.**
`kinect_frame_at_position(frame_indices, position)` was extracted out of `depth_frame_at_position`,
which now delegates to it. `forearm_depth_scalars` takes a **Kinect frame index**, not a position, so
the widget converts through that one function; had it passed a position, the surface would have been
painted from a different frame than the patch drawn on top of it, with nothing on screen to say so.

**One scalar bar serves both layers.** They paint the same field on the same fixed scale, and PyVista
keys bars by title, so a second `"Penetration depth (mm)"` bar is not available anyway. The forearm is
added with `show_scalar_bar=False` and the bar's visibility became
`_depth_colouring_active or _forearm_depth_colouring_active` (`_apply_depth_scalar_bar_visibility`) —
otherwise unchecking the contact layer would hide the legend for a forearm still mapped to it.

**Mode changes re-add the actor; frame changes do not.** Direct-RGB and mapped-scalar colouring are
different mapper configurations, not different arrays, so the on/off switch cannot be a
`scalar_visibility` toggle the way the contact layer's is — `_add_forearm_actor` re-enters `add_mesh`
under the same `name="forearm"`. Per-*frame* updates overwrite the array in place under the name the
mapper is bound to, so playback never cycles `remove_actor`/`add_mesh`; the actor count is measured
unchanged across a full off → on round trip.

**The layer defaults OFF**, unlike the contact layer. The PLY's own vertex colours are the anatomical
context the contact patch has to be judged against — the render check Phase 6.3 owes is precisely
"is the patch seated on the surface" — and a surface repainted every frame would replace that context
by default rather than on request. The preference is a plain attribute
(`_colour_forearm_by_depth`), separate from the data fact, and survives a walk through stages that
cannot offer the control.

**`nan_color` is `#a0a0a0`** — the same neutral grey this module already paints a colourless PLY with.
Without it the LUT clamps NaN to the bottom of the ramp, which would render "nobody touched this"
identically to "touched at the shallowest depth in the recording".

**Headless verification.** The Phase 3/4 substitution again (an off-screen `pv.Plotter` for the
`QtInteractor` class only; `QtInteractor` still cannot initialise offscreen here, `0xC00000FD`) drove
the real `_load_stage_data` → `_build_right_panel_controls` → `_init_actors` → checkbox →
`_update_frame` → `_on_stage_changed` path on a synthetic six-stage session whose stages 3-5 carry
`vertex_id` and whose stages 1-2 do not.

| Property | Measured |
|----------|----------|
| Stage 5 on open | Forearm group offers the control, unchecked; mapper array `colors` |
| Layer ON, frame 7 | mapper + active scalars `forearm_penetration_depth_mm`; range `(0.25, 9.5)` = the contact clim |
| Frame 7, vertex 4 (addressed twice, 0.25 and 0.50 mm) | `0.50` — the deeper |
| Frame 7, all other vertices | `NaN`, and vertex 9's genuine `0.00` mm stays `0.0` (unit test) |
| Frame 19 after frame 7 | vertex 0 = `4.0`, vertex 4 back to `NaN` — no paint persists |
| Frame 3 (no contact) | every vertex `NaN` |
| Actor count across the off → on round trip | unchanged |
| Contact OFF / forearm ON | bar still visible; contact OFF / forearm OFF | bar hidden |
| Stages 0, 1, 2 | the control is **absent** from the Forearm group, not disabled |
| Preference ON, walked 5 → 1 → 0 → 3 | still on at stage 3; exactly one bar; vertex 4 = `0.50` |
| Toggled off | active scalars back to `colors` |
| Forearm of 15 vertices vs a sidecar declaring 12 (every id still **in range**) | `ValueError: vertex_id provenance mismatch` — only the count check can catch this |
| Suite | 674 passed, 7 skipped (Phase 4 baseline 656/7; +18 here) |

Not verified here, and left to Phase 6: anything requiring a real window — that untouched skin is
*visibly* the plain grey, that the patch and the surface agree in colour where they overlap, and the
render check itself.

**Files Modified:**
- `code/src/merging/contact_depth_field_series.py` — `vertex_id_by_frame`,
  `reference_ply_provenance`, `has_vertex_ids`, `_validate_vertex_identity`; the loader populates both
- `code/src/postprocessing/gui/stage_depth_field.py` — `FOREARM_DEPTH_SCALAR_NAME`,
  `forearm_depth_scalars`, `kinect_frame_at_position`
- `code/src/postprocessing/gui/postprocessing_stage_viewer.py` — `_colour_forearm_by_depth`,
  `_forearm_depth_available`, `_forearm_depth_colouring_active`, `_add_forearm_actor`,
  `_update_forearm_depth_scalars`, `_apply_depth_scalar_bar_visibility`,
  `_on_forearm_depth_colour_changed`, the Forearm group's conditional control
- `code/tests/test_stage_depth_field.py` — section 6, the forearm join
- `code/tests/test_neural_kinect_depth_field_view.py` — two appended tests; **no existing test touched**

**Dependencies:** Phase 4

### Phase 6: Verification — including the render check Phase 9 owed
**Goal:** Evidence, not argument.

- [ ] 6.1 — Drive `2022-06-15_ST14-02` (the translating RF branch, `status: "ok"`) through
      `view_postprocessing_stages`; walk all six stages.
- [ ] 6.2 — Screenshot each sidecar-bearing stage; record the clim per stage. Because depth is
      bitwise preserved through every transform, the clim should be **identical** across stages 1-5 —
      a difference is a defect and must be investigated, not accepted.
- [ ] 6.3 — **The render check.** Confirm the RF-centred depth patch sits on the RF-centred forearm
      surface, on the anatomically correct side, and tracks the hand across the block. Tick the
      corresponding box in `propagate-contact-depth-field-through-postprocessing.md`.
- [ ] 6.4 — Drive `2022-06-14_ST13-01` (the no-cluster passthrough) and confirm stage 5 behaves
      correctly when RF-centring is a passthrough copy.
- [ ] 6.5 — Confirm stage 0's disabled control and its message on both sessions.
- [x] 6.6 — Record peak memory across a full six-stage walk, to confirm the cache bound holds.
- [x] 6.7 — Full suite; record pass/skip counts against the current 585 passed / 7 skipped baseline.

**Started:** 2026-08-20T10:05Z  **Completed (headless half):** 2026-08-20T11:20Z

#### Verification Results — headless, against REAL artifacts

Everything below was measured on the production artifacts under
`.../02_data/semi-controlled/3_merged/`, not on synthetic fixtures. Phases 1-5 were verified against
synthetic sidecars; this is the first time the feature has met the real pipeline's output.

**Environment — a real condition of reproducing these numbers.**

| Condition | Value |
|-----------|-------|
| Interpreter | `D:/Programming/anaconda3/envs/social-touch/python.exe` |
| `MKL_THREADING_LAYER` | **must** be `TBB`. Without it a broken Intel OpenMP runtime kills the interpreter with `0xc06d007f` as soon as anything imports `sklearn`/`threadpoolctl` |
| `PYTHONUTF8` | `1` — unrelated to this feature, but `forearm_extraction/.../point_cloud_visualizer.py:10` prints an emoji at import and a `cp1252` console raises `UnicodeEncodeError` before any of this code runs |
| `QtInteractor` | **still cannot initialise here** — VTK's `RenderWindowInteractor.initialize` blows the stack (`0xC00000FD`), offscreen included. Every render-path measurement below substitutes an off-screen `pv.Plotter` **for the `QtInteractor` class only**; every other line executed is the viewer's own |
| PyVista / pandas | 0.47.1 / 2.3.3 |

##### 6.1 (headless half) — the data path over real ST14-02, all six stages, two blocks

`resolve_stage_paths` → `resolve_stage_depth_field` → `kinect_frame_indices` →
`depth_frame_at_position` → `forearm_depth_scalars`, driven for every stage of
`block-order-01` and `block-order-05`.

`2022-06-15_ST14-02`, **block-order-01** — CSV 73 100 rows / 2 193 Kinect frames (stage 0: 3 149):

| Stage | Sidecar dir | MB | Rows | Contact frames | `coordinate_space` | `vertex_id` | `clim_penetration_mm` |
|-------|-------------|----|------|----------------|--------------------|-------------|------------------------|
| 0 Merged (Raw) | `blocks_merged/` | — | — | — | *(absent)* | — | — |
| 1 ICP Registered | `blocks_registered/` | 7.07 | 647 169 | 983 | `icp_registered` | no | `(-0.0, 18.62079620361328)` |
| 2 Deduplicated | `blocks_deduped/` | 4.64 | 423 111 | 983 | `icp_registered` | no | `(-0.0, 18.62079620361328)` |
| 3 Contact Projected | `blocks_projected/` | 5.23 | 423 111 | 983 | `icp_registered` | **yes** | `(-0.0, 18.62079620361328)` |
| 4 PCA Calibrated | `blocks_pca_calibrated/` | 5.23 | 423 111 | 983 | `pca_calibrated` | **yes** | `(-0.0, 18.62079620361328)` |
| 5 RF Centered | `blocks_rf_centered/` | 5.23 | 423 111 | 983 | `rf_centered` | **yes** | `(-0.0, 18.62079620361328)` |

`2022-06-15_ST14-02`, **block-order-05** (the largest block) — CSV 98 766 rows / 2 963 Kinect frames:

| Stage | MB | Rows | Contact frames | Space | `vertex_id` | `clim_penetration_mm` |
|-------|----|------|----------------|-------|-------------|------------------------|
| 1 | 41.94 | 3 632 693 | 2 678 | `icp_registered` | no | `(-5.7220458984375e-06, 22.561479568481445)` |
| 2 | 25.96 | 2 273 340 | 2 678 | `icp_registered` | no | `(-5.394796517066425e-06, 22.561479568481445)` |
| 3 | 29.54 | 2 273 340 | 2 678 | `icp_registered` | **yes** | `(-5.394796517066425e-06, 22.561479568481445)` |
| 4 | 29.56 | 2 273 340 | 2 678 | `pca_calibrated` | **yes** | `(-5.394796517066425e-06, 22.561479568481445)` |
| 5 | 29.56 | 2 273 340 | 2 678 | `rf_centered` | **yes** | `(-5.394796517066425e-06, 22.561479568481445)` |

Join properties, every stage of both blocks:

| Property | Measured |
|----------|----------|
| `kinect_frame_indices` on the real CSV | succeeds; no missing, non-finite or non-integral `frame_index` in any of the four blocks driven |
| `frame_index` range | `[0, n-1]`, strictly increasing, all unique — i.e. on **these** artifacts it happens to coincide with row position, so real data cannot by itself distinguish the correct join from the positional shortcut. The synthetic non-contiguous fixture of Phase 3 remains the test that can |
| `depth_frame_at_position(series, fi, p)` vs `series.frame(fi[p])` | identical on the first 50 probe positions of every stage of every block (bitwise, points and depths) |
| 400 probe positions, ST14-02 b01 | 179 with contact, 221 empty; every drawn depth inside the stage's own clim |
| 400 probe positions, ST14-02 b05 | 362 with contact, 38 empty; same |
| Geometry source when depth colouring is on | the parquet's float32 coordinates, not the CSV `%.1f` blob (Phase 3 measurement, unchanged) |

##### 6.2 — the clim-identity check. **The clims are NOT identical, and it is not a defect in the field.**

Measured `clim_penetration_mm` across the sidecar-bearing stages of four real blocks, compared by
IEEE-754 hex so no formatting hides a difference:

| Block | Stages 1-5 clim identical? | Where it differs |
|-------|---------------------------|-------------------|
| ST14-02 `block-order-01` | **yes, bitwise** (`-0x0.0p+0` … `0x1.29eec80000000p+4` on all five) | — |
| ST14-02 `block-order-05` | **no** | stage 1 low `-5.7220458984375e-06`; stages 2-5 low `-5.394796517066425e-06`. High bitwise identical (`0x1.68fbd20000000p+4`) on all five |
| ST13-01 `block-order-01` | **yes, bitwise** across stages 1-4 (stage 5 raises — see 6.4) | — |
| ST13-01 `block-order-03` | **no** | stage 1 low `0.0`; stages 2-4 low `6.748888699803501e-05`. High bitwise identical (`0x1.df431c0000000p+2`) |

The plan said a difference "is a defect and must be investigated, not accepted". It was
investigated, and the premise it rests on is the thing that is wrong.

**The measurement.** Comparing the `(frame_index, signed_depth_mm)` multiset of `blocks_registered/`
against `blocks_deduped/`, with the depth compared by its 64-bit pattern rather than by value:

| Block | Registered rows | Deduped rows | Dropped | Deduped pairs **not** accounted for in registered |
|-------|-----------------|--------------|---------|---------------------------------------------------|
| ST14-02 b01 | 647 169 | 423 111 | 224 058 | **0** |
| ST14-02 b05 | 3 632 693 | 2 273 340 | 1 359 353 | **0** |
| ST13-01 b01 | 187 056 | 83 492 | 103 564 | **0** |
| ST13-01 b03 | 804 323 | 335 528 | 468 795 | **0** |

So every depth value that survives deduplication survives it **bitwise**; the deduped field is a
strict sub-multiset of the registered one. And the *deep* end is preserved exactly in all four
blocks — `min(signed_depth_mm)`, which is `clim_penetration_mm[1]`, is bitwise identical
registered↔deduped everywhere.

What moves is the *shallow* end — `max(signed_depth_mm)`, i.e. `-clim_penetration_mm[0]`, the vertex
that was barely touching or a fraction of a micron outside the surface — and it moves precisely when
deduplication drops the single row that carried it:

| Block | Registered `signed max` | On frame | That frame: reg rows → ded rows | Value still present after dedup? | Deduped `signed max` |
|-------|--------------------------|----------|----------------------------------|----------------------------------|-----------------------|
| ST14-02 b01 | `0.0` | 470 | 286 → 173 | **yes** | `0.0` |
| ST13-01 b01 | `-0.0` | 354 | 259 → 110 | no, but another row carries `-0.0` | `-0.0` |
| ST14-02 b05 | `5.7220458984375e-06` | 1260 | 4097 → 2389 | **no** | `5.394796517066425e-06` |
| ST13-01 b03 | `-0.0` | 878 | 311 → 117 | **no** | `-6.748888699803501e-05` |

**Conclusion: the plan's expectation was too strong, not the artifact wrong.** "Depth is bitwise
preserved through every transform" is a statement about *values*; `clim` is the min/max over a *row
set*, and deduplication is a set-shrinking transform. Preserving every surviving value does not
preserve the extremes of the set. Stages 2→3→4→5 change no rows at all and their clims are
consequently bitwise identical in all four blocks — the only boundary that can move the clim is
1→2, and only at its shallow end.

Magnitude, for the record:

| Block | Δ clim low | Δ as a fraction of the block's full clim span |
|-------|-----------|-----------------------------------------------|
| ST14-02 b05 | 3.272493e-07 mm | 1.45e-08 |
| ST13-01 b03 | 6.748889e-05 mm | 9.01e-06 |

Both are far below any visible difference in an inferno ramp and far below the depth field's own
resolution. **No action is warranted**; what should change is this plan's stated expectation, and
this section is that correction.

##### 6.4 (headless half) — **DEFECT: stage 5 of `2022-06-14_ST13-01` cannot be opened at all**

Driving the no-cluster passthrough session, stage 5 raises before anything is drawn:

```
ValueError: Stage 5 ('RF Centered') expects a contact depth field in coordinate space
'rf_centered', but '...\3_merged\2022-06-14_ST13-01\blocks_rf_centered\
2022-06-14_ST13-01_semicontrolled_block-order-01_contact_depth_field_pca-xyz.parquet'
declares 'pca_calibrated'. Refusing to draw it: ...
```

Measured facts:

| Fact | Measurement |
|------|-------------|
| Blocks affected | **all four** of ST13-01 (`block-order-01/02/03/04`) |
| `rf_center_origin.json` | `{"status": "no_cluster_found", "rf_center": null, "total_points_evaluated": 1506, "points_above_threshold": 2}` |
| `blocks_rf_centered/` sidecar vs `blocks_pca_calibrated/` sidecar | **md5-identical** on all four blocks (`ee0c5f4496e3`, `5cfa10964cb9`, `30df842335e9`, `f1ec65ad8b35`) — a byte copy |
| Declared `coordinate_space` in the copied file | `pca_calibrated`, schema v2 |
| Same check on ST14-02 (`status: "ok"`) | the two files differ (`36ffcc6d471c` vs `be2c4dde1efe`) and stage 5 declares `rf_centered` — correct |
| Where the copy is made | `code/scripts/_5_postprocessing/center_on_receptive_field.py:370` `_copy_field_unchanged` |
| Where the expectation is stated | `code/src/postprocessing/gui/stage_depth_field.py:149` `EXPECTED_SPACE_BY_STAGE[5] = COORDINATE_SPACE_RF_CENTERED` |

**This is a genuine contract collision between two branches, not a coding slip on either side.**
`_copy_field_unchanged`'s docstring argues the copy deliberately: *"the points did not move, so the
file's declared `coordinate_space` — `pca_calibrated` — is still the truth, and copying the bytes is
the only way to guarantee nothing was restamped on the way past."* The viewer's map states the
opposite invariant — that whatever sits in `blocks_rf_centered/` declares `rf_centered` — and both
positions are defensible in isolation. They cannot both hold.

Blast radius in the viewer: `_load_stage_data` is called unguarded at
`postprocessing_stage_viewer.py:864`, inside the `currentIndexChanged` slot, so the `ValueError`
propagates out of a Qt slot. `self._current_stage_idx` has already been set to `5` on line 863, so
after the failure the viewer's idea of the current stage disagrees with the data it is displaying
(measured: the subsequent switch to stage 3 still worked, and stages 0-4 remain fully functional).

**Not fixed here — Phase 6 is verification only.** Recording the options rather than choosing one:

1. Restamp `coordinate_space` to `rf_centered` in the passthrough (a read-modify-write, contradicting
   `_copy_field_unchanged`'s stated reason for being a byte copy), or
2. make `EXPECTED_SPACE_BY_STAGE[5]` accept `pca_calibrated` **when** `rf_center_origin.json` says
   `no_cluster_found` — which requires the viewer to read that file, i.e. new policy in the leaf, or
3. introduce a fourth space name meaning "RF-centring was a no-op".

Option 2 is the only one that keeps both branches' stated invariants; it is also the only one that
adds a new input to the leaf. This belongs in its own change with its own plan.

##### 6.5 (headless half) — stage 0 resolves to the ordinary absent state on both real sessions

| Property | ST14-02 | ST13-01 |
|----------|---------|---------|
| `blocks_merged/` sidecar path derived | yes (`..._merged_data` → `..._contact_depth_field.parquet`) | yes |
| That file exists | **no** | **no** |
| `StageDepthField.is_present` | `False` | `False` |
| Checkbox enabled / checked | `False` / `False` | `False` / `False` |
| Tooltip byte-identical to `StageDepthField.message` | **yes** | **yes** |
| Message names the producing task | yes — names `blocks_filtered/`, the directory `filter_contact_depth_field_by_neural_quality` writes, and points the user at stage 1 | yes |
| `plotter.scalar_bars` | `[]` | `[]` |
| Contact actor | `scalar_visibility == False`, `GetProperty()` colour `(1.0, 0.0, 0.0)` | same |
| Exception | none | none |

##### Widget render path, driven over real artifacts

Substituting an off-screen `pv.Plotter` for the `QtInteractor` class only, the real
`__init__` → `_load_stage_data` → `_build_ui` → `_init_actors` → `_update_frame` →
`_on_stage_changed` path was driven on `ST14-02 block-order-01` (opened at stage 5) and
`ST13-01 block-order-01` (opened at stage 4, since stage 5 raises).

| Property | ST14-02 b01 |
|----------|-------------|
| `mapper.scalar_range` on open @ stage 5 | `(-0.0, 18.62079620361328)` = `series.clim_penetration_mm` |
| `mapper.scalar_range` over 8 frames (4 empty incl. first and last, 3 with contact, plus a re-visit of frame 0) | `(-0.0, 18.62079620361328)` on every one |
| Points drawn, positions 77 / 78 / 79 | 53 / 95 / 144 |
| `plotter.scalar_bars` on stages 5 → 4 → 3 → 2 → 1 → 0 | `1 → 1 → 1 → 1 → 1 → 0`, always exactly `"Penetration depth (mm)"` |
| `mapper.scalar_range` after each of those five switches | unchanged; stage 0 falls back to VTK's default `(0.0, 1.0)` with `scalar_visibility == False` |
| Contact "Colour by depth" enabled on stages 1-5 / stage 0 | `True` / `False` |
| Forearm "Colour by depth" **present** on stages 3, 4, 5; **absent** on 0, 1, 2 | confirmed (absent, not disabled) |
| Forearm PLY loaded at stage 5 | `forearm_rf_centered/2022-06-15_ST14-02_forearm.ply`, **10 712 points** |
| Contact toggle OFF → ON round trip | `scalar_visibility` `False`→`True`; `scalar_range` unchanged; **actor count unchanged (5 → 5)** |
| Forearm layer ON, position 77 (Kinect frame 77) | mapper array and active scalars both `forearm_penetration_depth_mm`; `mapper.scalar_range` = the contact clim; **53 lit / 10 659 NaN**; depths 0.00702 … 4.26 mm |
| Forearm layer ON, a no-contact frame | array all-NaN — no paint persists from the previous frame |
| Forearm layer OFF | active scalars back to `colors`; actor count unchanged |
| Scalar bar with contact OFF but forearm ON | still visible (one bar serves both layers) |

`ST13-01 block-order-01`, same drive from stage 4: clim `(0.0, 6.510202407836914)` on open and on
every frame; forearm PLY `forearm_pca_calibrated/2022-06-14_ST13-01_forearm.ply`, **1 807 points**;
forearm layer ON at position 168 → 3 lit / 1 804 NaN, 0.329 … 0.912 mm; stage 5 raises as above;
stages 3, 2, 1, 0 then all switch cleanly with exactly one bar (zero on stage 0).

##### 6.3's numeric half — `vertex_id` against the REAL forearm PLYs

`forearm_depth_scalars` was called with `len(pv.read(ply).points)` of the **actual** stage PLYs, which
is the check that would catch a provenance mismatch on real data:

| Session | Stage | Forearm PLY | `len(ply.points)` | `reference_ply_vertex_count` in the sidecar | Match |
|---------|-------|-------------|-------------------|---------------------------------------------|-------|
| ST14-02 | 3 | `forearm_deduped/2022-06-15_ST14-02_forearm.ply` | 10 712 | `10712` | ✔ |
| ST14-02 | 4 | `forearm_pca_calibrated/2022-06-15_ST14-02_forearm.ply` | 10 712 | `10712` | ✔ |
| ST14-02 | 5 | `forearm_rf_centered/2022-06-15_ST14-02_forearm.ply` | 10 712 | `10712` | ✔ |
| ST13-01 | 3 | `forearm_deduped/2022-06-14_ST13-01_forearm.ply` | 1 807 | `1807` | ✔ |
| ST13-01 | 4 | `forearm_pca_calibrated/2022-06-14_ST13-01_forearm.ply` | 1 807 | `1807` | ✔ |

Provenance recorded in every stage-3/4/5 sidecar of both sessions:
`{reference_ply: <session>_forearm.ply, reference_ply_vertex_count: 10712|1807, dedup_epsilon: 0.5}`.
Verified on both blocks of ST14-02 and both blocks of ST13-01:

- the scatter lights only vertices the frame touched — ST14-02 b01 frame 82: **223 lit / 10 489 NaN**
  from 225 contact rows (two rows address one vertex and resolve to the deeper); ST14-02 b05 frame
  126: **152 lit / 10 560 NaN** from 160 rows; ST13-01 b03 frame 134: **19 lit / 1 788 NaN** from 20 rows;
- a frame with no contact still validates and returns an **all-NaN** array (so a re-deduplicated
  forearm is refused on the first frame drawn, not the first frame that happens to touch);
- calling with `vertex_count + 1` raises `ValueError` on **every** stage of **every** block driven —
  the count check is live against real provenance, not only against the synthetic fixture.

##### 6.6 — memory across a full six-stage walk, and the cache bound

`STAGE_DEPTH_FIELD_CACHE_SIZE == len(STAGE_LABELS) == 6`, so a walk of the six stages **never
evicts**: the bound is "one full sweep resident", and that is what these numbers price.

| Drive | Peak working set | RSS at exit |
|-------|------------------|-------------|
| Data path only, ST14-02 b01, six stages | **587.3 MB** | 422.0 MB |
| Data path only, ST14-02 b05 (largest block), six stages | **1 118.6 MB** | 585.0 MB |
| Data path only, ST13-01 b01 + b03, twelve stage resolutions | 627.9 MB | 443.5 MB |
| Full widget drive (VTK + real PLYs), ST14-02 b01, six stages + toggles | **741.2 MB** | 643.9 MB |
| Full widget drive, ST14-02 b05, six stages + toggles | **1 378.5 MB** | 1 169.2 MB |
| 14-visit walk `5,4,3,2,1,0,5,4,3,2,1,0,3,5` on b05 | 1 201.4 MB | 933.5 MB |

Read accounting on that 14-visit walk, counted by wrapping `read_contact_depth_field`:

| Point | Reads |
|-------|-------|
| After constructing all six loaders | **0** |
| After the first full sweep (stages 5→0) | **5** — one per sidecar-bearing stage, none for stage 0 |
| After 14 visits total | **5**, 5 distinct files — **every revisit was served from the cache** |

The bound therefore holds in the sense it was written for (residency is a property of the constant,
not of how long the window stays open), and a full sweep of the largest block in the dataset costs
~1.4 GB peak with VTK in the process. **Eviction itself is still untested** — with `maxsize` equal to
the stage count it cannot occur on any real walk; the corresponding Testing-Plan box stays unticked.

##### 6.7 — full suite

`MKL_THREADING_LAYER=TBB pytest -q` → **674 passed, 7 skipped in 25.50s**, identical to the Phase 5
baseline. No test was added, removed or modified in this phase (Phase 6 is verification only; the
working tree carries no source, test or config change from it).

#### What was NOT verified — every one of these needs a real window

Stated plainly so nothing here is mistaken for the render check:

- **Nothing was seen.** No colour, no ramp, no scalar bar, no grey untouched skin, no geometry was
  put in front of a human eye. `QtInteractor` cannot initialise in this environment (`0xC00000FD`),
  so "the mapper is bound to `penetration_depth_mm` with clim X and a bar titled Y" is the whole of
  what was established. Whether that renders as an inferno-ramped patch seated on the forearm is
  exactly the open question, and it is exactly what Phase 9 owed.
- **6.1's GUI walk, 6.2's screenshots, 6.3 the render check, and 6.5's visual confirmation remain
  undone**, and their boxes are deliberately left unticked.
- **Playback frame rate** on the largest block is unmeasured — the timer never ran.
- **The `wglMakeCurrent` hazard** at `postprocessing_stage_viewer.py:523-527` is untested: the
  deferred `QTimer.singleShot(0, ...)` tick was invoked synchronously by the harness, so the generation
  guard and the real OpenGL context release were never exercised. Rapid stage switching during
  playback likewise.
- **Cache eviction** — see 6.6.
- **Real data cannot distinguish the frame-index join from a positional one.** On all four blocks
  driven, `frame_index` equals row position exactly. The join is correct — it was measured against
  `series.frame(fi[p])` — but the artifact that would *catch* a regression here is still only the
  synthetic non-contiguous fixture from Phase 3.

#### Remaining for a real window (human)

Everything below needs a machine where `QtInteractor` initialises. Do them in this order.

**Launch.**

```bash
conda activate social-touch-env
cd F:/GitHub/touch_projects/social-touch-semi-controlled
python code/scripts/launch_pipeline_gui.py     # Postprocess category
```

or drive the DAG directly:

- config: `configs/postprocess_visualization_dag.yaml`
- task: `view_postprocessing_stages`
- set `kinect_configs` to `kinect_configs/valid_configs_ST14-02` (start with
  `kinect_config_2022-06-15_ST14-02_semicontrolled_block-order01.yaml`; `block-order05` is the
  3.6 M-row stress case)

**Step 1 — 6.1, the six-stage walk.** Open at stage 0 and step 0 → 1 → 2 → 3 → 4 → 5, then back
5 → 0. Look for: no VTK context error on any switch (the `wglMakeCurrent` hazard); exactly one scalar
bar titled "Penetration depth (mm)" on stages 1-5 and none on stage 0; the bar's numeric endpoints
reading `-0.00` and `18.62` on block-order-01 at *every* stage 1-5.
**FAILURE** = a crash or a black frame on switch, two bars at once, a bar surviving onto stage 0, or
a bar whose numbers change between stages 2, 3, 4 and 5. (Between stage 1 and stage 2 a change in the
*low* endpoint below display precision is expected — see 6.2 above; on block-order-01 there is none
at all, on block-order-05 it is 3e-07 mm and cannot be visible.)

**Step 2 — 6.2, screenshots.** One screenshot per stage 1-5 of `block-order-01`, on a frame with
substantial contact (positions 77-79 have 53, 95 and 144 contact points; positions 82, 104, 109 have
225-240 in the deduped stages). Paste them into this section. Include the scalar bar in the frame.

**Step 3 — 6.3, THE RENDER CHECK — the one thing this whole plan exists for.** On stage 5
(`RF Centered`) of `2022-06-15_ST14-02`, with "Colour by depth" ON for the contact points:

1. Scrub to a frame with a large patch and confirm the coloured patch is **on the forearm surface** —
   not floating beside it, not behind it, not inside it, not mirrored across the arm's long axis.
2. Rotate the camera 180° and confirm the patch is on the side of the arm the hand is on.
3. Turn the **Forearm** group's "Colour by depth" ON. Where the patch overlaps the surface, the two
   must agree in colour; untouched skin must be the plain neutral grey (`#a0a0a0`), visibly distinct
   from the darkest end of the inferno ramp — that distinction is "nobody touched this" vs "touched
   at the shallowest depth in the recording", and it is the one thing `nan_color` exists for.
4. Play the block through and confirm the patch **tracks the hand coherently** — it moves with the
   stroke rather than jumping, wrapping around the mesh, or staying put.
5. Compare against stage 4 (`PCA Calibrated`): the patch must sit in the same anatomical place on the
   arm, only the origin of the axes having moved.

**FAILURE of the render check** = any of: the patch is off the surface by more than the mesh's own
thickness; the patch is on the wrong side of the arm; the patch is mirrored or translated relative to
where the hand visibly is; the patch does not move with the hand; the patch sits somewhere different
at stage 5 than at stage 4 relative to the anatomy. **If it fires, this plan stops.** The finding goes
back to `propagate-contact-depth-field-through-postprocessing.md` as a defect in the propagation, and
the viewer must *not* be adjusted to make the picture look right.

If it passes, tick 6.3 here **and** the render-check box in Phase 9 of
`propagate-contact-depth-field-through-postprocessing.md`, and attach the screenshots to both.

**Step 4 — 6.5 visual, stage 0.** On stage 0 confirm the contact points render **visibly flat red**
with no bar, and that the "Colour by depth" box is greyed out and its tooltip (hover) names
`blocks_filtered/` and points at stage 1. The state was measured headlessly; only "visibly" is left.

**Step 5 — toggles.** Uncheck "Colour by depth" on stage 5: the points must turn flat red and the bar
must disappear. Re-check: inferno and bar return, with the *same* endpoints as before.
**FAILURE** = the colours change scale on the way back, or the bar comes back with different numbers.

**Step 6 — playback rate.** Play `block-order-05` (2 963 frames, 2.27 M sidecar rows) end to end at
stage 5 with both depth layers on. Record the observed frame rate here.
**FAILURE** = a rate low enough that scrubbing is unusable, which would put the per-frame scalar
update back on the table.

**Step 7 — 6.4 visual, ST13-01.** Point `kinect_configs` at
`kinect_configs/valid_configs_ST13-01` and walk stages 0-4. **Stage 5 will raise** — that is the
defect recorded in 6.4 above, it is expected until that contract collision is resolved, and reaching
it will take the window down. Do not select stage 5 unless you want to see the failure mode.

**Files Modified:** none (verification only); results recorded in this plan

**Dependencies:** Phase 5

---

## Testing Plan

### Unit Tests (headless — no Qt, no VTK, no Open3D)
- [x] `depth_field_path_for_csv` pairs each of the six stage CSV names correctly, including the
      `_pca-xyz` fork on stages 4-5.
- [x] Building six loaders reads zero bytes (counting reporter, per
      `test_neural_kinect_depth_field_view.py:441`).
- [x] `resolve_stage_depth_field` returns the series for each correct (stage, space) pair.
- [x] Every wrong (stage, space) pair raises `ValueError` naming both spaces.
- [x] Stage 0 returns absent without calling a loader.
- [x] Missing sidecar → absent with a non-empty message; corrupt sidecar → raises.
- [x] A loader returning a non-series raises `TypeError`.
- [ ] The cache evicts at its bound and re-reads an evicted stage. *(Phase 6.6: **cannot occur on a
      real walk** — `STAGE_DEPTH_FIELD_CACHE_SIZE == len(STAGE_LABELS) == 6`, and a 14-visit walk over
      one block performed 5 reads and zero re-reads. Testing eviction needs a deliberately undersized
      cache; left unticked rather than quietly reinterpreted.)*
- [x] *(Ph.5)* `forearm_depth_scalars` scatters to the right vertices; NaN elsewhere; provenance
      mismatch raises; no `vertex_id` → `None`. *(Plus: the mismatch raises on a no-contact frame too,
      so a re-deduplicated forearm is refused on the first frame drawn rather than the first frame
      that happens to touch; and a vertex addressed twice in one frame resolves to the deeper, not to
      whichever row the sort left last.)*
- [x] *(Ph.5)* Every existing test in `test_neural_kinect_depth_field_view.py` passes **unmodified**
      after the DTO widening. *(`git diff`: 41 insertions, 0 deletions — two appended tests, no
      existing line touched.)*

### Integration Tests
- [x] Frame join: a synthetic sidecar with non-contiguous `frame_index` values that differ from row
      position; assert the depths drawn at slider position *p* are those of `frame_index.iloc[p]`,
      not of frame *p*. **This test is the whole point of Phase 3.2** — it is the one that fails if
      the positional shortcut is taken. *(Plus a guard-the-guard test asserting the fixture can still
      tell the two joins apart.)*
- [x] A `_kinect_df` without a `frame_index` column raises rather than falling back.
- [x] Six `StagePaths` built by `resolve_stage_paths` against a synthetic session tree carry the
      expected loaders, with stage 0's resolving to a non-existent path. *(Phase 6 did this against
      **real** session trees instead, which is strictly stronger: four blocks across ST14-02 and
      ST13-01, six `StagePaths` each, stage 0's sidecar path derived and non-existent every time,
      stages 1-5 pairing to the files that are actually on disk.)*

### Manual Verification
- [ ] All six stages open without a VTK context error on stage switch (the `wglMakeCurrent` hazard
      the generation guard at `:523-527` exists for).
- [ ] Colour range visibly constant across frames within a stage; a deep frame and a shallow frame
      differ in colour, not in scale.
- [ ] Toggling "Colour by depth" off returns flat red and hides the bar; on restores both.
- [ ] Playback runs at an acceptable frame rate on the largest block (ST14-02 block 05, 3.6 M rows
      pre-dedup).
- [ ] Screenshots recorded for stages 1-5 on ST14-02.

### Edge Cases
- [x] A frame with no contact — draws nothing, does not corrupt the LUT, does not raise.
- [x] The first frame of a stage having no contact (mapper binding on an empty PolyData).
- [ ] A stage whose sidecar exists but whose CSV does not, and the reverse.
- [ ] A single-depth-value recording → degenerate but valid clim (covered upstream at
      `test_neural_kinect_depth_field_view.py:263`).
- [ ] Rapid stage switching during playback (generation guard).
- [ ] A session where `blocks_rf_centered/` is a passthrough copy (ST13-01). **Tested and it
      FAILS** — stage 5 raises `ValueError` on all four ST13-01 blocks because the passthrough copies
      the sidecar byte-for-byte with `coordinate_space: pca_calibrated`. See Phase 6.4 for the
      measurements and the three options; not fixed here.
- [x] *(Ph.5)* A forearm PLY whose vertex count disagrees with `reference_ply_vertex_count` — must
      raise, since this is the silent-renumbering hazard. *(Driven through the widget with a 15-vertex
      forearm against a sidecar declaring 12: every id is still in range, so only the count check
      catches it, and it does.)*

---

## Documentation Plan

- [ ] Add a knowledge-base note on the stage ↔ space ↔ sidecar mapping, since the six / four / seven
      mismatch (stages / spaces / directories) is exactly the kind of thing that will be
      misremembered.
- [ ] Update `note-spatial-alignment-pipeline.md` — already flagged stale by the depth-field plan's
      Documentation Plan; add the sidecar column to its space table.
- [ ] **Update `docs/development/knowledge-base/README.md`** — the index is stale, and four unindexed
      notes (including `investigation-jet-colormap-perceptual-problems.md`) are more relevant to GUI
      work than half the indexed ones. Small fix, real recurring cost.
- [ ] Document the new option in `configs/postprocess_visualization_dag.yaml`'s header comment.
- [ ] Tick the render-check box in
      `docs/development/plans/active/propagate-contact-depth-field-through-postprocessing.md` and
      record the screenshots there — that plan's Phase 9 is where the debt is recorded.
- [ ] Changelog entry `docs/changelogs/depth-field-stage-viewer.md`.
- [ ] No CLAUDE.md change — no new architectural rule is introduced.

---

## Rollback Plan

1. **Before deployment:** the feature is read-only — it opens artifacts and draws them. No pipeline
   stage, no producer, and no on-disk artifact is written or modified by any phase.
2. **Data considerations:** none. No migration, no schema change, no re-run required. Reverting leaves
   every sidecar and CSV byte-identical.
3. **Rollback procedure:** phases are independently revertible in reverse order. Phase 5 is the only
   one that touches a module outside postprocessing (`contact_depth_field_series.py`); its widening is
   a defaulted optional field, so reverting it cannot break the merging viewer. Reverting Phases 1-4
   returns the viewer to flat-red contact points; `StagePaths.depth_field_loader` is defaulted, so
   even a partial revert leaves the dataclass constructible.
4. **Kill switch:** setting `depth_field_loader=None` for every stage in `resolve_stage_paths`
   disables the whole feature at one line, without touching the viewer.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Positional frame join** — slider position used instead of `frame_index`; the CSV is upsampled ~33x to the nerve rate, so this silently draws another frame's depths | **High** | **High** | Phase 3.2 is written against it; a dedicated integration test with non-contiguous, non-positional `frame_index`; raise rather than fall back when the column is absent |
| Per-frame autoscale creeps back via a forgotten `clim` after `DeepCopy` | Med | **High** | Explicit `clim` at `add_mesh`, re-assert `scalar_range` after every swap, plus a success criterion measured on three frames including one post-switch |
| Scalar bar leaks or duplicates across stage switches | Med | Med | Success criterion asserts exactly one bar; explicit `remove_scalar_bar` guard as the fallback if `clear()` proves insufficient |
| Widening `ContactDepthFieldSeries` breaks the merging viewer | Low | **High** | Optional field, defaulted `None`; the existing ~40 tests must pass **unmodified** — that is the acceptance test, and Phase 5 is separately revertible |
| `vertex_id` join against a mismatched PLY silently mis-colours | Low | **High** | `validate_vertex_ids_against_reference` called with `len(ply.points)` before any scatter; epsilon/count provenance is exactly what it checks |
| Memory — six stages of 10^5-10^6 vertices | Med | Med | `BoundedContactDepthFieldCache`; deferred loaders; peak measured in Phase 6.6. No streaming reader exists and building one is out of scope |
| Stage 0's absent sidecar read as a bug and "fixed" with a `blocks_filtered/` hack | Med | Med | Defined in Definitions, asserted by test, stated in Out of Scope |
| VTK context error on stage switch (`wglMakeCurrent`) aggravated by new actors | Med | Med | Keep the existing `QTimer.singleShot(0, ...)` + generation guard; recolour the existing contact actor rather than adding a new one |
| The 685-line viewer grows further | **High** | Med | All policy in the Qt-free leaf; the widget gains render code only (`05` §1/§4) |
| Playback frame rate degrades with per-frame scalar arrays | Med | Low | In-place `DeepCopy` update, no `remove_actor`/`add_mesh` cycle; measured in Manual Verification |
| **The render check reveals a genuine spatial defect in the depth-field branch** | Low | **High** | That is the *purpose*. If it fires, this plan stops and the finding goes back to the depth-field plan as a defect — do not adjust the viewer to make the picture look right |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 — wire the loader | ~50 LOC + ~60 test | None |
| Phase 2 — stage-aware leaf | ~120 LOC + ~150 test | Phase 1 (parallelisable) |
| Phase 3 — depth-coloured points | ~110 LOC + ~80 test | Phases 1, 2 |
| Phase 4 — the control | ~70 LOC | Phase 3 |
| Phase 5 — PLY colouring *(separable)* | ~130 LOC + ~110 test | Phase 4 |
| Phase 6 — verification | measurement only | Phase 5 |

---

## References

- **The debt this pays:**
  `docs/development/plans/active/propagate-contact-depth-field-through-postprocessing.md`
  Phase 9, "Not verified" — the render check
- **Motivating requirement:** `docs/development/brainstorms/per-vertex-contact-depth.md` — IFF
  weighting `w(depth_i) x RF_sensitivity(position_i)` in the RF-centred frame
- **The precedent to mirror:** `code/src/merging/gui/neural_kinect_scene_viewer.py` (injection
  `:245`, guard `:1178-1188`, actor `:1763-1814`, mode switch `:2324-2339`, checkbox `:1570-1601`)
  and its loader factory `code/scripts/merging_pipeline_neuron_to_kinect_visualisation.py:233-285`
- **The adapter to reuse:** `code/src/merging/contact_depth_field_series.py`
- **The conventions to import:**
  `code/src/preprocessing/motion_analysis/tactile_quantification/gui/contact_depth_field_viewer.py`
  (`COLORMAP` `:111`, `CONTACT_SCALAR_NAME` `:104`, `SCALAR_BAR_TITLE` `:108`, PyVista 0.47.1
  measurements `:46-62`, scalar-bar args `:506-522`, empty PolyData `:797-805`)
- **The resolver:** `code/src/postprocessing/depth_field_stage_io.py:1141-1186`
- **The format:**
  `code/src/preprocessing/motion_analysis/tactile_quantification/io/contact_depth_field_io.py`
  (schema `:254-268`, spaces `:201-223`, `validate_vertex_ids_against_reference` `:772-881`)
- **The test shape to copy:** `code/tests/test_neural_kinect_depth_field_view.py`
- Knowledge base: `investigation-jet-colormap-perceptual-problems.md`,
  `bug-rf-explorer-nearest-vertex-distance.md`, `note-artifact-serialization-formats.md`,
  `note-qt-itemchanged-signal-recursion.md`, `note-rf-camera-settings-connections.md`,
  `note-spatial-alignment-pipeline.md`
- Guides: `~/.claude/knowledge/data-pipeline-engineering/` 02 (§2 layering, §3 DTOs, §6 normalization,
  §8 error boundaries, §9 visualization as a pure sink), 05 (§1 SRP, §3 DRY, §4 change amplification,
  §6 YAGNI, §8 tests)

---
