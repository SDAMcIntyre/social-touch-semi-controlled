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
      with a scalar bar titled "Penetration depth (mm)".
- [ ] The colour range for a given stage is identical on every frame of that stage — asserted by
      reading `actor.mapper.scalar_range` on at least three frames including one with no contact and
      one immediately after a stage switch.
- [ ] Switching stages replaces the scalar bar and its range; no bar from a previous stage survives,
      and `plotter.scalar_bars` contains exactly one depth bar at any time.
- [ ] Stage 0 (`blocks_merged/`) shows a disabled "Colour by depth" checkbox whose tooltip names the
      producing task, and renders flat-red contact points without error.
- [ ] A frame's drawn depth values equal `series.frame(kinect_df["frame_index"].iloc[pos])[1]`
      exactly — verified by a headless test on a synthetic sidecar whose `frame_index` values are
      deliberately non-contiguous and not equal to row position.
- [ ] A sidecar whose `coordinate_space` disagrees with the selected stage raises `ValueError` naming
      both spaces and the file.
- [ ] A present-but-undecodable sidecar raises; it does not fall back to flat colour.
- [x] Constructing the six stage loaders reads zero bytes — asserted with a counting reporter, as
      `test_neural_kinect_depth_field_view.py:441` does for blocks.
- [ ] *(Phase 5)* On stages 3-5, the forearm PLY colours by the current frame's `vertex_id` join, and
      untouched vertices are visually distinct from zero-depth vertices.
- [ ] **The render check that Phase 9 could not perform:** on `2022-06-15_ST14-02` the RF-centred
      depth patch is visually seated on the RF-centred forearm surface, on the correct side, moving
      coherently with the hand across the block. Recorded with screenshots in this plan.
- [ ] Full suite green; the new adapter tests import no Qt, VTK or Open3D.

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

- [ ] 3.1 — In `_load_stage_data`, call `resolve_stage_depth_field(stage_idx, sp.depth_field_loader)`
      and store `self._depth_field`. No parquet knowledge enters the widget.
- [ ] 3.2 — Build the **frame_index map**: `self._frame_indices = self._kinect_df["frame_index"]`,
      and look up `series.frame(int(self._frame_indices.iloc[pos]))`. Raise if the column is missing
      rather than falling back to positional indexing — this is the single most dangerous shortcut
      available here.
- [ ] 3.3 — In `_init_actors`, when depth colouring is active, add the contact actor with
      `scalars=CONTACT_SCALAR_NAME`, `cmap=COLORMAP`, `clim=series.clim_penetration_mm`,
      `show_scalar_bar=True` and the `scalar_bar_args` shape from
      `contact_depth_field_viewer.py:506-522` (white text on the black background). Keep
      `GetProperty().SetColor(1,0,0)` so the flat-red off state is intact.
- [ ] 3.4 — In `_update_frame`, set the scalar array on the replacement PolyData before `DeepCopy`,
      `set_active_scalars`, then re-assert `actor.mapper.scalar_range = clim`.
- [ ] 3.5 — Empty-frame handling: an `_empty_contact_polydata()` equivalent that still carries the
      scalar array; keep `allow_empty_mesh`.
- [ ] 3.6 — Verify the scalar bar does not survive a stage switch, and that exactly one exists after
      each switch. If `plotter.clear()` turns out not to remove it, add the explicit
      `remove_scalar_bar` guard from `neural_kinect_scene_viewer.py:1769-1770`.
- [ ] 3.7 — Confirm the depth series replaces the CSV blob as the geometry source when active (as
      `neural_kinect_scene_viewer.py:2075-2086` does), so the drawn points are the parquet's float32
      coordinates, not the `%.1f` CSV ones — and note in a comment that the two differ.

**Files Modified:**
- `code/src/postprocessing/gui/postprocessing_stage_viewer.py` — `_load_stage_data`, `_init_actors`,
  `_update_frame`, new imports of `CONTACT_SCALAR_NAME` / `SCALAR_BAR_TITLE` / `COLORMAP`

**Dependencies:** Phases 1, 2

### Phase 4: The "Colour by depth" control
**Goal:** User control, and honest absence.

- [ ] 4.1 — Extend `_add_group` with an `extra_widgets` parameter, matching the merging viewer's
      `_add_object_group` signature so the two panels stay recognisably the same.
- [ ] 4.2 — Add the `QCheckBox("Colour by depth")` to the Contact Points group;
      `setEnabled(self._depth_field.is_present)`; tooltip = the stage's actual `low..high` mm when
      present, else `StageDepthField.message`.
- [ ] 4.3 — Seed `setChecked(...)` inside `blockSignals(True)` so a stage without a field does not
      clear the preference on switch.
- [ ] 4.4 — Handler toggles `mapper.scalar_visibility` and the bar's visibility without re-adding
      actors (`_apply_contact_scalar_mode`, `neural_kinect_scene_viewer.py:2324-2339`); re-assert
      `scalar_range` on the way through.
- [ ] 4.5 — Register the layer in the `actor_map` at `:652` if a new actor is introduced (it should
      not be — the same contact actor is recoloured).
- [ ] 4.6 — Persist the preference across stage switches as a plain attribute.

**Files Modified:**
- `code/src/postprocessing/gui/postprocessing_stage_viewer.py` — `_build_right_panel_controls`,
  `_add_group`, new handler

**Dependencies:** Phase 3

### Phase 5: Forearm PLY colouring by `vertex_id` *(separable)*
**Goal:** Depth on the surface, not just on the patch. **Revertible on its own** — Phases 1-4 deliver
the core value without it.

- [ ] 5.1 — Widen `ContactDepthFieldSeries` with
      `vertex_id_by_frame: Optional[Dict[int, np.ndarray]] = None`, populated only when the table
      carries the column. A pure addition; v1 and `vertex_id`-less v2 sidecars keep working, and every
      existing merging test must stay green untouched.
- [ ] 5.2 — Carry the reference-PLY provenance triple onto the series so the join can be validated.
- [ ] 5.3 — `forearm_depth_scalars(series, frame_index, vertex_count) -> Optional[np.ndarray]` in the
      leaf: call `validate_vertex_ids_against_reference` with `len(ply.points)` **first**, then
      scatter. NaN for untouched vertices, so they are visually distinct from a genuine 0 mm.
- [ ] 5.4 — Switch the forearm actor from `scalars="colors", rgb=True` (`:377-395`) to scalar mode
      when the layer is on, and back when off. Only for stages 3-5; the control is absent, not merely
      disabled, on stages 0-2 — there is no `vertex_id` and no honest thing to show.
- [ ] 5.5 — Handle `nan_color` explicitly so untouched vertices read as the plain forearm.
- [ ] 5.6 — Tests: correct scatter, NaN placement, provenance mismatch raises, absent `vertex_id`
      returns `None` rather than an empty array.

**Files Modified:**
- `code/src/merging/contact_depth_field_series.py` — optional field + provenance
- `code/src/postprocessing/gui/stage_depth_field.py` — `forearm_depth_scalars`
- `code/src/postprocessing/gui/postprocessing_stage_viewer.py` — forearm actor scalar mode
- `code/tests/test_stage_depth_field.py`, `code/tests/test_neural_kinect_depth_field_view.py`

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
- [ ] 6.6 — Record peak memory across a full six-stage walk, to confirm the cache bound holds.
- [ ] 6.7 — Full suite; record pass/skip counts against the current 585 passed / 7 skipped baseline.

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
- [ ] The cache evicts at its bound and re-reads an evicted stage.
- [ ] *(Ph.5)* `forearm_depth_scalars` scatters to the right vertices; NaN elsewhere; provenance
      mismatch raises; no `vertex_id` → `None`.
- [ ] *(Ph.5)* Every existing test in `test_neural_kinect_depth_field_view.py` passes **unmodified**
      after the DTO widening.

### Integration Tests
- [ ] Frame join: a synthetic sidecar with non-contiguous `frame_index` values that differ from row
      position; assert the depths drawn at slider position *p* are those of `frame_index.iloc[p]`,
      not of frame *p*. **This test is the whole point of Phase 3.2** — it is the one that fails if
      the positional shortcut is taken.
- [ ] A `_kinect_df` without a `frame_index` column raises rather than falling back.
- [ ] Six `StagePaths` built by `resolve_stage_paths` against a synthetic session tree carry the
      expected loaders, with stage 0's resolving to a non-existent path.

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
- [ ] A frame with no contact — draws nothing, does not corrupt the LUT, does not raise.
- [ ] The first frame of a stage having no contact (mapper binding on an empty PolyData).
- [ ] A stage whose sidecar exists but whose CSV does not, and the reverse.
- [ ] A single-depth-value recording → degenerate but valid clim (covered upstream at
      `test_neural_kinect_depth_field_view.py:263`).
- [ ] Rapid stage switching during playback (generation guard).
- [ ] A session where `blocks_rf_centered/` is a passthrough copy (ST13-01).
- [ ] *(Ph.5)* A forearm PLY whose vertex count disagrees with `reference_ply_vertex_count` — must
      raise, since this is the silent-renumbering hazard.

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
