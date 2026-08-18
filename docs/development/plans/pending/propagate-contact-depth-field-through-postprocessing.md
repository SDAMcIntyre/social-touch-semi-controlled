# Plan: Propagate Contact Depth Field Through Postprocessing

**Date:** 2026-08-18
**Created:** 2026-08-18 19:00
**Approved:** —
**Completed:** —
**Author:** Basil Duvernoy
**Status:** Draft
**Base Branch:** `feature/remove-prefect-orchestration`
**Branch:** `feature/depth-field-postprocessing`

---

## Overview

Carry the per-vertex contact depth field through the five postprocessing spatial stages, so it arrives
in the RF-centred frame alongside the CSV it belongs to. The field gains a `vertex_id` column at the
projection stage — an integer index into the session's reference forearm — which is **invariant across
Spaces 2, 3 and 4** and therefore does more for downstream consumers than the coordinates themselves.

## Problem Statement

The depth field currently stops at `blocks_filtered/` in **Kinect Space 1**. Postprocessing then moves
every contact coordinate through four more frames, and the field's `x/y/z` become wrong-space the
moment `apply_icp_registration` runs. The file says so in its own metadata.

This blocks the requirement the whole line of work exists to serve. From
`docs/development/brainstorms/per-vertex-contact-depth.md`: the IFF weight on vertex *i* is
`w(depth_i) x RF_sensitivity(position_i)` — two fields multiplying — so the depth field must be
expressible in the **RF-centred frame**. It "cannot be a preprocessing-only debug artefact".

There is a second, larger opportunity that the investigation surfaced. Three of the five stages
(`forearm_deduped/` → `forearm_pca_calibrated/` → `forearm_rf_centered/`) transform the reference
forearm PLY **in place, in file order, with no reordering, filtering or count change**
(`set_xyz_reference_from_gestures.py:335-349`, `center_on_receptive_field.py:192-196`). Vertex *i* is
therefore the same physical vertex in all three spaces. `project_contacts_onto_forearm.py:93` already
computes exactly that index for every contact point — and discards it.

Persisting it makes the depth field **space-independent**: a consumer joins `vertex_id` against
whichever forearm PLY it wants and gets correct coordinates in that space, instead of trusting a
coordinate triple that four transforms have rounded to 0.1 mm at three separate points.

It also fixes a live defect. `center_on_receptive_field.py:85-88` aggregates spike counts keyed on
**float coordinate tuples** parsed from the `%.1f` `contact_points` blob — precisely the pattern
`completed/rf-heatmap-vertex-index-aggregation.md` diagnosed as splitting one physical vertex across
several keys — and its output defines the origin of Space 4 for the entire session.

## Goals

### In Scope

1. Propagate the depth field through all five spatial stages and into `blocks_rf_centered/`.
2. Add `vertex_id` to the schema at the projection stage, with the reference-PLY provenance needed to
   make it verifiable (schema version bump).
3. Reuse the CSV's own index mappings at the two non-replayable stages — dedup and projection —
   rather than recomputing them against the parquet.
4. Implement the agreed dedup reduction rule: the dropped vertex's row is removed and the survivor
   inherits the collapsed group's **maximum magnitude**.
5. Restamp `coordinate_space` at every stage that moves points, so the declared space is always the
   actual one.
6. Idempotency and `clean_task_outputs` for the second artifact at every stage.
7. **Prerequisite hardening** of the dedup epsilon hazards that would silently repoint every
   persisted `vertex_id` (see Phase 1).

### Out of Scope

- **A session-level aggregated parquet.** `frame_index` is per-block, so a session file needs a
  `block_order_id` column, which the exact-column validator makes a second schema decision. The
  terminal artifact is per-block `blocks_rf_centered/*.parquet`; the aggregated CSV already carries
  `block_order_id` for joining. Revisit if the analysis repo asks for it.
- **Recomputing `signed_depth_mm` anywhere.** See Definitions — depth is preserved, never re-measured.
- **Changing what the PCA fit or the RF estimate consume.** Feeding contact vertices into either
  would change the coordinate system itself and invalidate every existing `rf_center_origin.json`.
- **Depth-weighting the RF centre.** Scientifically tempting, out of scope, and a behaviour change to
  the origin of Space 4.
- **The IFF weighting kernel** — lives in the analysis repo.
- **Reinstating unique-vertex projection assignment.** `completed/unique-vertex-projection-assignment.md`
  is a superseded plan (its Hungarian assignment was removed by `ef7cf75` as a silent point-loss bug);
  its `In Progress` header is stale metadata. Not revived here.
- **Back-filling.** Users re-run the pipeline, per `fix-contact-points-csv-corruption.md`.
- **Fixing `projection_stats.csv`'s partial-overwrite defect** and its mislabelled `session` column.
  Documented in Risks, repaired separately.

## Success Criteria

- [ ] `blocks_rf_centered/<session>_semicontrolled_<block>_..._contact_depth_field.parquet` exists for
      every block the postprocess DAG processes.
- [ ] Its metadata declares `coordinate_space = "rf_centered"`, and every intermediate declares the
      space it is actually in — never the space it came from.
- [ ] **Row-count agreement at every stage**: for each frame, the parquet's row count equals the
      parsed length of that frame's `contact_points` cell in the CSV written by the same stage. This
      is the single check that catches desynchronisation, and it must hold at all five stages.
- [ ] After dedup, `max(|signed_depth_mm|)` per frame still equals the CSV's `contact_depth` for that
      frame. The max-magnitude reduction rule is what preserves this; assert it.
- [ ] `vertex_id` is present from the projection stage onward, is `int32`, and every value is a valid
      index into the reference PLY (`0 <= vertex_id < n_vertices`).
- [ ] Metadata records `reference_ply`, `reference_ply_vertex_count` and `dedup_epsilon`, and the
      reader **raises** on a vertex-count mismatch against the PLY it is joined to.
- [ ] Resolving `vertex_id` against `forearm_rf_centered/*.ply` reproduces the parquet's own `x/y/z`
      to within the PLY's 0.1 mm rounding — the cross-check that the index and the coordinates agree.
- [ ] The three replayable stages (ICP, PCA, RF-centring) leave `signed_depth_mm` **bitwise
      unchanged**; only coordinates move.
- [ ] Deleting only a stage's parquet re-runs that stage; `clean_task_outputs` removes both artifacts.
- [ ] A passthrough branch (ICP with no transforms, RF-centring with no cluster) copies the parquet as
      well as the CSV, so the two never land in different spaces.
- [ ] Full test suite does not regress from its count at branch point (**352 passed, 7 skipped**).

## Definitions

- **Replayable stage**: one whose transform is a persisted matrix or offset applicable to an `(N,3)`
  array — ICP, PCA calibration, RF-centring. The parquet is transformed by the same object the CSV
  uses, never a re-derived one.
- **Index-consuming stage**: dedup and projection. Neither is a coordinate transform; each computes a
  per-row index mapping. The parquet **must** be filtered/re-addressed by the CSV's own mapping.
- **Preserved depth**: `signed_depth_mm` is a measurement of how far the hand penetrated at a vertex.
  Projection re-addresses that measurement to a canonical vertex; it does not re-measure. Testable:
  the value carried into `blocks_rf_centered/` is bit-identical to the value in `blocks_filtered/`
  for the same (frame, vertex), except where the dedup rule deliberately replaced it with a group
  maximum.
- **`vertex_id`**: `int32` row index into the reference forearm PLY **in file order**, assigned at
  projection. Meaningless without `reference_ply` + `reference_ply_vertex_count` + `dedup_epsilon`.
- **Reference PLY identity**: the triple above. A `vertex_id` from a PLY deduped at a different
  epsilon is silently wrong; recording the identity is what turns that into a loud failure.
- **Desynchronisation**: the parquet and the CSV disagreeing about how many contact vertices a frame
  has. Always an error, never tolerated, checked at every stage.
- **Space names in metadata**: `kinect_space_1` → `icp_registered` → (dedup, projection: unchanged
  space) → `pca_calibrated` → `rf_centered`.

---

## Technical Design

### Approach

Two mechanisms, applied per stage according to which kind of stage it is.

```
blocks_filtered/      Space 1   depth field as produced
   |  [1] apply_icp_registration      REPLAYABLE  — same per-frame-segment schedule as the CSV
blocks_registered/    Space 2
   |  [2] deduplicate_xy              INDEX       — CSV's kept_indices + DBSCAN labels; max-magnitude rule
blocks_deduped/       Space 2
   |  [3] project_contacts_onto_forearm  INDEX    — CSV's KD-tree indices; ASSIGNS vertex_id
blocks_projected/     Space 2   (+ vertex_id from here on)
   |  [4] calibrate_pca_xyz           REPLAYABLE  — same CalibrationResult as the CSV
blocks_pca_calibrated/ Space 3
   |  [5] center_on_receptive_field   REPLAYABLE  — same translation as the CSV
blocks_rf_centered/   Space 4   terminal artifact
```

**The load-bearing rule: never recompute an index against the parquet.** CSV contact points are
float64 parsed from a `%.1f`-quantised string; the parquet holds full float32. Re-running DBSCAN or
the KD-tree on the parquet's own coordinates would pick different clusters and different nearest
vertices wherever two candidates are near-equidistant — and would do so **silently**, producing two
artifacts that describe different geometry with no error. Both index-consuming stages therefore
export their mapping and the parquet consumes it.

Once `vertex_id` exists, stages 4 and 5 are strictly optional for correctness — a consumer could
resolve coordinates from the PLY instead. They are still done, because the parquet should be usable
without a PLY join and because the CSV and parquet must remain checkable against each other at every
stage.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Propagate through all five stages, reusing the CSV's index mappings, and assign `vertex_id` at projection** | Field arrives in Space 4; `vertex_id` is space-invariant and fixes the RF float-key defect; every stage stays checkable against the CSV | Touches five stages; needs a schema bump | **Chosen** |
| Keep the field in Space 1; store only the transform chain and let consumers replay it | No stage changes | Dedup and projection are not replayable from stored state — that is the whole difficulty. Also pushes the burden onto the separate analytics repo, which the artifact exists to serve | Rejected |
| Assign `vertex_id` and **stop** — do not transform coordinates after projection | Half the work; `vertex_id` carries the spatial meaning anyway | The parquet would declare Space 2 coordinates while sitting in a `blocks_rf_centered/` directory. Every consumer would need the PLY join. Row-count checks against the later CSVs would still be wanted, so the stages get touched regardless | Rejected — but this is the fallback if stages 4-5 prove costly |
| Re-run dedup / KD-tree independently on the parquet | Simple, no cross-stage plumbing | Silent desynchronisation from float32-vs-quantised divergence. Exactly the failure class the fail-fast rule exists to prevent | Rejected |
| Aggregate to one session-level parquet | Mirrors the aggregated CSV | `frame_index` is per-block, so it needs `block_order_id`; the exact-column validator makes that a second schema decision. The aggregated CSV already carries the block key | Rejected — out of scope |
| Recompute `signed_depth_mm` after projection | Coordinates and depth stay mutually consistent | Would measure penetration against a forearm the hand never touched. Depth is a measurement, not a derived coordinate | Rejected |

### Architecture & Module Contracts

| Module / layer | Responsibility | Inputs → Outputs | Must NOT know about |
|----------------|----------------|------------------|---------------------|
| `contact_depth_field_io.py` (modified) | Schema v2: optional `vertex_id`, reference-PLY metadata, vertex-count validation | table + metadata → parquet | Stages, sessions, PLYs, DAGs |
| `depth_field_stage_io.py` (new, `code/src/postprocessing/`) | Per-stage read → transform → write of the sidecar, and the row-count agreement check against a CSV | `(in_parquet, out_parquet, transform_or_mapping, csv_for_check)` → parquet | Which stage called it, Prefect, `KinectConfig` |
| `deduplicate_xy_points.py` (modified) | Additionally return `kept_indices` **and** DBSCAN `labels` per row | unchanged CSV behaviour + a mapping | The depth field, parquet, metadata |
| `project_contacts_onto_forearm.py` (modified) | Additionally return the per-row KD-tree `indices` | unchanged CSV behaviour + a mapping | The depth field's schema |
| the five stage scripts (modified) | Drive both artifacts from one mapping/transform | CSV + parquet in → CSV + parquet out | Each other |
| `postprocess_workflow_kinect_auto.py` (modified) | Path arithmetic, context binding, idempotency | `KinectConfig` → paths | Serialisation, transform mathematics |

The `depth_field_stage_io` leaf exists for the same reason `contact_depth_field_series.py` and
`frame_navigation.py` do: three of the five stage scripts import `open3d` or `matplotlib` at module
scope and are **not importable in the unit-test environment**. Logic placed in them cannot be tested.

### Constraints from the investigation

- **Empty-list trap.** `postprocess_workflow_kinect_auto.py:388-391` skips a stage entirely — CSVs
  included — if any `list` parameter is empty. Never pass an empty list of parquet paths; pass
  `None` or filter before binding.
- **Passthrough branches must copy the parquet.** `apply_icp_registration.py:77` (no transforms file)
  and `center_on_receptive_field.py:267-280` (no RF cluster) copy CSVs through untransformed. If the
  parquet is not copied there, the two artifacts end up in different spaces with no error.
- **`parse_contact_points` drops malformed points silently** (`csv_spatial_transformer.py:47-55`:
  `if len(parts) == 3` with no `else`). This is why the row-count agreement check is a success
  criterion at every stage and not just at the end.
- **float32 dtype.** `x/y/z` are float32; transforms must run in float64 and cast back, or the
  writer's dtype validation rejects the table.
- **`np.round(..., 1)` on the PLYs** at stages 4 and 5 can make distinct vertices coincident *by
  value* while remaining distinct *by index* — the reason index-keying is mandatory.
- **PCA drops blocks that fail gesture segmentation** (`set_xyz_reference_from_gestures.py:200-203`)
  from `loaded_data`. A parquet loop keyed off `loaded_data` inherits that silent drop; key off
  `input_files`.
- **`apply_full_transform` mutates in place** — pass a copy, as the CSV path does at `:317`.
- **Idempotency boundaries differ per stage**: ICP, dedup, PCA and RF check per *session*; projection
  checks per *block*. Match each stage's existing boundary rather than imposing one.

---

## Implementation Plan

### Phase 1: Harden the dedup epsilon hazards
**Goal:** A persisted `vertex_id` cannot be silently repointed by an unrecorded or drifted epsilon.
**Started:** —  **Completed:** —

These are pre-existing defects, but `vertex_id` makes them dangerous: an epsilon change re-deduplicates
the forearm and renumbers every vertex, with nothing on disk to detect it.

- [ ] 1.1 — Reconcile the epsilon defaults. `deduplicate_xy_flow` defaults `epsilon=5.0`
      (`postprocess_workflow_kinect_auto.py:101`) against the YAML's `0.5` — dropping the key is a
      silent 10x change. Make the flow require it, or make the defaults agree.
- [ ] 1.2 — Record the **effective** epsilon on disk. Under `monitor: true` it is chosen interactively
      (`:126`) and survives only in a log line, making the run unreproducible.
- [ ] 1.3 — Return `kept_indices` from `deduplicate_forearm_ply` (already in scope at
      `deduplicate_xy_points.py:279`, currently dropped at `:295-299`) so the source→deduped vertex
      mapping is recoverable.
- [ ] 1.4 — Record `n_vertices` and the epsilon alongside the deduped PLY, so a `vertex_id` can be
      validated against the PLY it claims to index.

**Files Modified:** `code/scripts/_5_postprocessing/deduplicate_xy_points.py`,
`code/scripts/postprocess_workflow_kinect_auto.py`

**Dependencies:** None

### Phase 2: Schema v2 — `vertex_id` and reference-PLY provenance
**Goal:** The format can carry a vertex id and prove which PLY it indexes.
**Started:** —  **Completed:** —

- [ ] 2.1 — `SCHEMA_VERSION = "2"`; `SUPPORTED_SCHEMA_VERSIONS = {"1", "2"}` so existing Space-1
      artifacts still read.
- [ ] 2.2 — Optional `vertex_id` (`int32`) column, absent before projection and present after.
      The current validator demands an exact column list; widen it to accept both shapes and reject
      anything else.
- [ ] 2.3 — Metadata keys `reference_ply`, `reference_ply_vertex_count`, `dedup_epsilon`, and the
      space names listed in Definitions.
- [ ] 2.4 — A reader helper that raises on a vertex-count mismatch when joining to a PLY.
- [ ] 2.5 — Tests: v1 files still read; v2 round-trips with and without `vertex_id`; out-of-range
      `vertex_id` raises; vertex-count mismatch raises; unknown version still raises.

**Files Modified:** `.../tactile_quantification/io/contact_depth_field_io.py`,
`code/tests/test_contact_depth_field_io.py`

**Dependencies:** None (parallel with Phase 1)

### Phase 3: Surface the index mappings
**Goal:** Both index-consuming stages expose their mapping, with the CSV path byte-identical.
**Started:** —  **Completed:** —

- [ ] 3.1 — `deduplicate_contact_points_csv`: switch `:345` to `return_indices=True` and accumulate
      per-frame `kept_indices`. `deduped_array` is the same array either way (`:57`), so the CSV is
      unchanged.
- [ ] 3.2 — Additionally surface the DBSCAN `labels` (local at `:42`). `kept_indices` alone supports
      dropping but **not** the max-magnitude inheritance rule — that needs to know which rows
      collapsed into which survivor.
- [ ] 3.3 — `_project_single_csv`: accumulate the per-row KD-tree `indices` (already bound at `:93`)
      alongside the existing `all_distances` accumulator and widen the return.
- [ ] 3.4 — Assert the CSV outputs of both stages are byte-identical to a pre-change run.
- [ ] 3.5 — Tests for both mappings against the existing algorithm tests
      (`test_deduplicate_xy_points.py:447-504` already covers the `return_indices` contract).

**Files Modified:** `code/scripts/_5_postprocessing/deduplicate_xy_points.py`,
`code/scripts/_5_postprocessing/project_contacts_onto_forearm.py`, their tests

**Dependencies:** None

### Phase 4: The stage io leaf
**Goal:** A testable module that reads, transforms and writes the sidecar per stage.
**Started:** —  **Completed:** —

- [ ] 4.1 — New `code/src/postprocessing/depth_field_stage_io.py`: apply a 4x4, apply a
      `CalibrationResult`, apply a per-frame index filter with the max-magnitude rule, and apply a
      per-frame vertex re-addressing.
- [ ] 4.2 — The row-count agreement check: given a parquet and the CSV written by the same stage,
      assert per-frame parity and raise naming the first offending frame.
- [ ] 4.3 — float64 compute, float32 cast-back, dtype preserved.
- [ ] 4.4 — Tests on synthetic tables — no Open3D, no PyQt5, no recording.

**Files Modified:** `code/src/postprocessing/depth_field_stage_io.py` (new),
`code/tests/test_depth_field_stage_io.py` (new)

**Dependencies:** Phase 2

### Phase 5: ICP propagation (Space 1 → 2)
**Goal:** The field arrives in `blocks_registered/`.
**Started:** —  **Completed:** —

- [ ] 5.1 — Reuse the **same** `schedule` object the CSV uses (`apply_icp_registration.py:89`), and
      the CSV-derived `max_frame` from `:87` — the parquet holds only contacting frames and would
      otherwise build a different schedule.
- [ ] 5.2 — Vectorised per-segment mask; `signed_depth_mm` untouched.
- [ ] 5.3 — Handle the no-transforms passthrough at `:77`.
- [ ] 5.4 — Restamp `coordinate_space = "icp_registered"`.
- [ ] 5.5 — Idempotency at the session boundary, matching `:53-60`.

**Files Modified:** `code/scripts/_5_postprocessing/apply_icp_registration.py`

**Dependencies:** Phases 3, 4

### Phase 6: Dedup and projection (the index-consuming stages)
**Goal:** The field survives row removal and vertex re-addressing, and gains `vertex_id`.
**Started:** —  **Completed:** —

- [ ] 6.1 — Dedup: filter parquet rows by the CSV's per-frame `kept_indices`; the survivor's
      `signed_depth_mm` becomes the maximum magnitude of its DBSCAN group. Never re-run DBSCAN.
- [ ] 6.2 — Assert after dedup that per-frame `max(|signed_depth_mm|)` still equals the CSV's
      `contact_depth` — this is what the reduction rule was chosen to preserve.
- [ ] 6.3 — The epsilon used must be the one actually applied, including an interactive override
      (`postprocess_workflow_kinect_auto.py:126`), so the hook lives inside that wrapper.
- [ ] 6.4 — Projection: re-address each parquet row's `x/y/z` using the CSV's KD-tree `indices`, and
      write `vertex_id`. Never re-query the tree.
- [ ] 6.5 — Stamp `reference_ply`, `reference_ply_vertex_count`, `dedup_epsilon`.
- [ ] 6.6 — `signed_depth_mm` preserved through both stages except where 6.1 deliberately replaces it.
- [ ] 6.7 — Idempotency: session boundary for dedup, **block** boundary for projection
      (`project_contacts_onto_forearm.py:173`).

**Files Modified:** `code/scripts/_5_postprocessing/deduplicate_xy_points.py`,
`code/scripts/_5_postprocessing/project_contacts_onto_forearm.py`,
`code/scripts/postprocess_workflow_kinect_auto.py`

**Dependencies:** Phase 5

### Phase 7: PCA and RF-centring (Space 2 → 3 → 4)
**Goal:** The field reaches `blocks_rf_centered/`.
**Started:** —  **Completed:** —

- [ ] 7.1 — PCA: one vectorised `apply_full_transform` over the whole `x/y/z` block, using the same
      `CalibrationResult` the CSV uses. Pass a copy — it mutates in place.
- [ ] 7.2 — Key the parquet loop off `input_files`, **not** `loaded_data`, which silently omits
      blocks that failed gesture segmentation.
- [ ] 7.3 — Handle the `_pca-xyz` filename fork with an explicit parquet suffix constant.
- [ ] 7.4 — RF-centring: same translation matrix; handle the no-cluster passthrough at `:267-280`.
- [ ] 7.5 — Restamp `pca_calibrated` then `rf_centered`.
- [ ] 7.6 — Confirm `vertex_id` is carried through both stages untouched.
- [ ] 7.7 — Do **not** feed the field into the PCA fit or the RF estimate.

**Files Modified:** `code/scripts/_5_postprocessing/set_xyz_reference_from_gestures.py`,
`code/scripts/_5_postprocessing/center_on_receptive_field.py`

**Dependencies:** Phase 6

### Phase 8: Wiring and idempotency
**Goal:** The DAG produces the artifact at every stage.
**Started:** —  **Completed:** —

- [ ] 8.1 — Resolve `source_parquets` alongside `session_input_files`
      (`postprocess_workflow_kinect_auto.py:243-254`) and seed it into `context` at `:265-267`, index-
      aligned with `session_configs`.
- [ ] 8.2 — Extend each stage's `outputs` list and returned tuple; binding is positional, so existing
      keys must keep their order.
- [ ] 8.3 — Add the parquets to `input_paths` / `output_paths` / `clean_task_outputs` per stage, at
      that stage's own boundary.
- [ ] 8.4 — **Never pass an empty list** — the guard at `:388-391` would silently skip the whole
      stage, CSVs included.
- [ ] 8.5 — Decide and document the DAG config's session list: the depth field is a hard input, so a
      session lacking one (`ST15-01` has merged CSVs but no field) must be excluded explicitly rather
      than failing mid-run.

**Files Modified:** `code/scripts/postprocess_workflow_kinect_auto.py`,
`configs/postprocess_workflow_kinect_auto_dag.yaml`

**Dependencies:** Phase 7

### Phase 9: Verification
**Goal:** Measured evidence at every stage, not argument.
**Started:** —  **Completed:** —

- [ ] 9.1 — Run one session end to end.
- [ ] 9.2 — Row-count agreement between parquet and CSV at **all five** stages.
- [ ] 9.3 — `signed_depth_mm` bitwise unchanged through ICP, PCA and RF-centring.
- [ ] 9.4 — After dedup, per-frame `max(|signed_depth_mm|)` equals the CSV's `contact_depth`.
- [ ] 9.5 — Every `vertex_id` in range; resolving it against `forearm_rf_centered/*.ply` reproduces
      the parquet's `x/y/z` within the PLY's 0.1 mm rounding.
- [ ] 9.6 — The CSVs at every stage are byte-identical to a pre-change run of the same inputs.
- [ ] 9.7 — Delete one stage's parquet, re-run, confirm regeneration; re-run clean, confirm skip.
- [ ] 9.8 — Record artifact sizes per stage and the dataset-wide total.

**Files Modified:** none (verification only)

**Dependencies:** Phase 8

---

## Testing Plan

### Unit Tests
- [ ] Schema v1 files still read; v2 round-trips with and without `vertex_id`.
- [ ] Out-of-range `vertex_id` raises; PLY vertex-count mismatch raises.
- [ ] Rigid transform on a synthetic table leaves `signed_depth_mm` bitwise unchanged.
- [ ] Index filter drops exactly the expected rows and applies the max-magnitude rule.
- [ ] Row-count agreement check raises, naming the first offending frame.
- [ ] Vertex re-addressing preserves row order.
- [ ] float32 dtypes survive a float64 transform round-trip.

### Integration Tests
- [ ] Per-stage row-count agreement on a real block.
- [ ] Depth preservation through the three replayable stages on real data.
- [ ] `vertex_id` resolves against each of the three forearm PLYs to the same physical vertex.
- [ ] `should_process_task` behaviour asserted directly for each stage's new output.

### Manual Verification
- [ ] Open a `blocks_rf_centered/` parquet with no repo import; confirm metadata alone states space,
      units, sign convention, reference PLY and epsilon.
- [ ] Render the RF-centred field against the RF-centred forearm and confirm the contact patch sits
      on the surface.
- [ ] Re-run the DAG unchanged; confirm every stage skips.

### Edge Cases
- [ ] Block whose ICP has no transforms (passthrough) — parquet copied, space unchanged.
- [ ] Session whose RF estimation fails (`no_cluster_found`) — parquet copied, space stays Space 3,
      and the metadata says so.
- [ ] Block dropped by PCA gesture segmentation.
- [ ] Frame whose every contact vertex is removed by dedup.
- [ ] A malformed `contact_points` cell — must fail the row-count check loudly, not silently misalign.

---

## Documentation Plan

- [ ] Module docstrings for the new leaf and the schema v2 changes.
- [ ] Changelog `docs/changelogs/depth-field-postprocessing.md`.
- [ ] **Rewrite `docs/development/knowledge-base/note-spatial-alignment-pipeline.md`** — it is stale:
      wrong stage order, and directory names that no longer exist. Already flagged twice; this plan
      depends on the corrected version.
- [ ] Knowledge-base note on the `vertex_id` scheme: why the index is invariant across Spaces 2-4,
      why coordinates are not, and what the reference-PLY identity is for.
- [ ] Update `contact-depth-field-sidecar.md` and `filter-contact-depth-field-by-neural-quality.md`
      Out-of-Scope sections to point here — this is the successor both of them named.
- [ ] Mark `completed/unique-vertex-projection-assignment.md` as superseded, correcting its stale
      `In Progress` header.

---

## Rollback Plan

1. **Purely additive artifacts.** Nothing consumes the postprocessed parquets yet; deleting them has
   no downstream effect.
2. Phase-per-commit: revert from the last stage backwards. Each stage is independent in the revert
   direction because each reads the previous stage's output from disk.
3. Phase 1 should be **kept regardless** — the epsilon hazards are real without this work.
4. Phase 2 is backward compatible: schema v1 files still read, so reverting later phases leaves the
   Space-1 artifacts valid.
5. **Data considerations:** no migration. Existing CSVs are neither read-modified nor invalidated.
   Existing Space-1 parquets remain valid v1 files.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Silent desynchronisation** — parquet and CSV disagree on a frame's vertex count | Med | **High** | Row-count agreement asserted at every stage, a success criterion, and the reason `parse_contact_points`'s silent point-dropping is called out. Never re-run an index computation on the parquet |
| **Epsilon change silently repoints every `vertex_id`** — `should_process_task` never sees the YAML | Med | High | Phase 1 records the effective epsilon and the PLY vertex count; the reader raises on mismatch |
| Empty-list guard silently skips a whole stage including CSVs | Med | High | Never pass an empty list (8.4); pass `None` |
| Passthrough branch moves the CSV but not the parquet | Med | High | Explicit tasks at 5.3 and 7.4, and two edge cases |
| Depth-weighting creeps into the RF estimate or the PCA fit | Low | **High** | Out of Scope and an explicit task (7.7). Either would move the origin of Space 4 and invalidate every existing `rf_center_origin.json` |
| `float32` round-trip degrades coordinates | Low | Med | float64 compute, cast back; dtype assertions |
| Zero existing test coverage of these five stages | **Certain** | Med | Logic lives in the testable leaf (Phase 4); three stage scripts are not importable in the unit-test environment, so integration coverage is via real-data runs |
| Scope creep into fixing `projection_stats.csv` or the RF float-key defect | Med | Med | Both documented, both explicitly out of scope. The RF defect becomes fixable *after* this lands, as its own plan |
| Five stages x two artifacts is a lot of surface | High | Med | Two mechanisms only (replayable vs index-consuming), both in one leaf module; per-stage phases with independent revert |
| Schema v2 breaks an existing reader | Low | Med | v1 stays in `SUPPORTED_SCHEMA_VERSIONS`; `vertex_id` optional; explicit round-trip tests both ways |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 — epsilon hardening | ~40 LOC | None |
| Phase 2 — schema v2 | ~90 LOC + ~120 test | None |
| Phase 3 — surface the mappings | ~60 LOC + ~80 test | None |
| Phase 4 — stage io leaf | ~200 LOC + ~220 test | Phase 2 |
| Phase 5 — ICP | ~70 LOC | Phases 3, 4 |
| Phase 6 — dedup + projection | ~160 LOC | Phase 5 |
| Phase 7 — PCA + RF | ~110 LOC | Phase 6 |
| Phase 8 — wiring | ~90 LOC | Phase 7 |
| Phase 9 — verification | measurement only | Phase 8 |

---

## References

- Predecessors: `docs/development/plans/active/contact-depth-field-sidecar.md`,
  `docs/development/plans/active/filter-contact-depth-field-by-neural-quality.md` — both name this as
  their successor
- Motivating requirement: `docs/development/brainstorms/per-vertex-contact-depth.md` (IFF weighting
  requires the field in the RF-centred frame; Fork B — depth travels with its coordinates)
- Vertex-index rationale: `docs/development/plans/completed/rf-heatmap-vertex-index-aggregation.md`
- Superseded, do not revive: `docs/development/plans/completed/unique-vertex-projection-assignment.md`
- Stage architecture: `docs/development/plans/completed/postprocessing-column-consolidation.md`
  (no shadow columns; subfolder per step)
- Serialization decisions: `docs/development/knowledge-base/note-artifact-serialization-formats.md`
- Coordinate spaces: `docs/development/knowledge-base/note-spatial-alignment-pipeline.md` (stale — see
  Documentation Plan)
- The in-tree precedent for carrying this artifact through a stage:
  `code/scripts/_4_merging/filter_contact_depth_field_by_neural_quality.py`
- Guides: `~/.claude/knowledge/data-pipeline-engineering/` 02 (§1 pipe-and-filter, §3 DTOs,
  §5 idempotency, §8 error boundaries, §11 provenance), 03 (§6 missing vs zero), 05 (§YAGNI)
