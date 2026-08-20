# Plan: Propagate Contact Depth Field Through Postprocessing

**Date:** 2026-08-18
**Created:** 2026-08-18 19:00
**Approved:** —
**Completed:** 2026-08-20 14:39
**Author:** Basil Duvernoy
**Status:** Completed
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
      *Holds for all 10 blocks of the 2 sessions driven (ST13-01, ST14-02); the other 9 of the DAG's
      11 sessions were not driven, so this is unproven at DAG scope.*
- [x] Its metadata declares `coordinate_space = "rf_centered"`, and every intermediate declares the
      space it is actually in — never the space it came from. *ST14-02's terminal artifact declares
      `rf_centered`; ST13-01 takes the no-cluster passthrough and correctly declares `pca_calibrated`.*
- [x] **Row-count agreement at every stage**: for each frame, the parquet's row count equals the
      parsed length of that frame's `contact_points` cell in the CSV written by the same stage. This
      is the single check that catches desynchronisation, and it must hold at all five stages.
      *36/36 checks on ST14-02, 24/24 on ST13-01.*
- [x] After dedup, `max(|signed_depth_mm|)` per frame still equals the CSV's `contact_depth` for that
      frame. The max-magnitude reduction rule is what preserves this; assert it. *Max disagreement
      7.105e-15 mm over 10 199 frames; the field's own per-frame maximum is unchanged to 0.000e+00.*
- [x] `vertex_id` is present from the projection stage onward, is `int32`, and every value is a valid
      index into the reference PLY (`0 <= vertex_id < n_vertices`). *`[0, 10711]` of 10 712 across
      5.68 M rows.*
- [x] Metadata records `reference_ply`, `reference_ply_vertex_count` and `dedup_epsilon`, and the
      reader **raises** on a vertex-count mismatch against the PLY it is joined to. *Stamped values
      measured on real output; the raising behaviour is covered by the Phase 2 unit tests.*
- [x] Resolving `vertex_id` against `forearm_rf_centered/*.ply` reproduces the parquet's own `x/y/z`
      to within the PLY's 0.1 mm rounding — the cross-check that the index and the coordinates agree.
      *0.09999771 mm (two roundings) and 0.04999390 mm (one).*
- [x] The three replayable stages (ICP, PCA, RF-centring) leave `signed_depth_mm` **bitwise
      unchanged**; only coordinates move. *Values and dtype, on every block of both sessions —
      projection too.*
- [x] Deleting only a stage's parquet re-runs that stage; `clean_task_outputs` removes both artifacts.
      *Measured per stage on ST13-01 in Phase 8; a clean whole-session re-run of ST14-02 skips all
      seven tasks.*
- [x] A passthrough branch (ICP with no transforms, RF-centring with no cluster) copies the parquet as
      well as the CSV, so the two never land in different spaces. *Both branches measured in Phases 5
      and 7; ST13-01's end-to-end drive exercises the RF one.*
- [x] Full test suite does not regress from its count at branch point (**352 passed, 7 skipped**).
      *585 passed, 7 skipped.*

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
**Started:** 2026-08-18 13:20  **Completed:** 2026-08-18 13:50

These are pre-existing defects, but `vertex_id` makes them dangerous: an epsilon change re-deduplicates
the forearm and renumbers every vertex, with nothing on disk to detect it.

- [x] 1.1 — Reconcile the epsilon defaults. `deduplicate_xy_flow` defaults `epsilon=5.0`
      (`postprocess_workflow_kinect_auto.py:101`) against the YAML's `0.5` — dropping the key is a
      silent 10x change. Make the flow require it, or make the defaults agree.
      *Done: `monitor` and `epsilon` are now required keyword-only arguments of `deduplicate_xy_flow`
      (the `monitor=True` default diverged from the YAML's `false` the same way), validated on entry;
      `deduplicate_forearm_ply`'s own `epsilon=0.35` default was removed for the same reason. The YAML
      values were not changed.*
- [x] 1.2 — Record the **effective** epsilon on disk. Under `monitor: true` it is chosen interactively
      (`:126`) and survives only in a log line, making the run unreproducible.
      *Done: `forearm_deduped/<stem>_dedup_metadata.json`, written beside the PLY the epsilon
      produced, also recording `epsilon_source` (`dag_config` vs `interactive_monitor`).*
- [x] 1.3 — Return `kept_indices` from `deduplicate_forearm_ply` (already in scope at
      `deduplicate_xy_points.py:279`, currently dropped at `:295-299`) so the source→deduped vertex
      mapping is recoverable. *Done: added as a fourth dict key; existing by-key readers unaffected.*
- [x] 1.4 — Record `n_vertices` and the epsilon alongside the deduped PLY, so a `vertex_id` can be
      validated against the PLY it claims to index. *Done: `n_vertices_original` /
      `n_vertices_deduped` / `n_vertices_removed` in the same sidecar, with the
      `deduped + removed == original` invariant asserted at write time.*

**Not done in this phase (deliberate):** `should_process_task` still does not see the DAG YAML, so
changing `epsilon` and re-running does not re-trigger dedup. Per the Risks table, the mitigation is
the Phase 2 reader raising on a recorded-vs-actual mismatch, not a new re-run trigger.

**Files Modified:** `code/scripts/_5_postprocessing/deduplicate_xy_points.py`,
`code/scripts/_5_postprocessing/__init__.py`,
`code/scripts/postprocess_workflow_kinect_auto.py`, `code/tests/test_deduplicate_xy_points.py`

**Dependencies:** None

### Phase 2: Schema v2 — `vertex_id` and reference-PLY provenance
**Goal:** The format can carry a vertex id and prove which PLY it indexes.
**Started:** 2026-08-18 13:52  **Completed:** 2026-08-18 14:05

- [x] 2.1 — `SCHEMA_VERSION = "2"`; `SUPPORTED_SCHEMA_VERSIONS = {"1", "2"}` so existing Space-1
      artifacts still read. *Done. v2 is a pure widening — a v1-shaped table is a valid v2 payload —
      so neither writer needed a change to start stamping it. Verified against the real v1 files under
      `3_merged/2022-06-14_ST13-01/blocks_filtered/`: all four read back with the v1 column list and
      `schema_version = "1"`.*
- [x] 2.2 — Optional `vertex_id` (`int32`) column, absent before projection and present after.
      The current validator demands an exact column list; widen it to accept both shapes and reject
      anything else. *Done: `_VALID_COLUMN_LAYOUTS` holds exactly two tuples — the required six, and
      the required six followed by `vertex_id`. `COLUMN_DTYPES` deliberately still means "the columns
      that are always there" (the merging filter iterates it), with `OPTIONAL_COLUMN_DTYPES` and
      `ALL_COLUMN_DTYPES` alongside. A misplaced `vertex_id` raises "wrong order"; any other column
      raises `unexpected=[...]`.*
- [x] 2.3 — Metadata keys `reference_ply`, `reference_ply_vertex_count`, `dedup_epsilon`, and the
      space names listed in Definitions. *Done, with three enforcement decisions recorded below.*
- [x] 2.4 — A reader helper that raises on a vertex-count mismatch when joining to a PLY.
      *Done: `validate_vertex_ids_against_reference(table, metadata, *, reference_vertex_count,
      reference_description)`. It takes a **count**, never a mesh, so the module stays Open3D-free;
      the count check runs before the range check because in-range ids against the wrong mesh are the
      dangerous case, and its message quotes the recorded `reference_ply` and `dedup_epsilon` so the
      failure points at the epsilon change rather than at the symptom.*
- [x] 2.5 — Tests: v1 files still read; v2 round-trips with and without `vertex_id`; out-of-range
      `vertex_id` raises; vertex-count mismatch raises; unknown version still raises. *Done — 34 new
      tests. Also: `vertex_id` in the wrong position, an arbitrary extra column, `int64` instead of
      `int32`, a misspelled coordinate space, a malformed count/epsilon, and a purity assertion that
      the module imports no geometry engine. The pre-existing "rejects an extra column" test used
      `vertex_id` as its example and was retargeted to an arbitrary column, with a second case
      proving the widening did not open the schema to anything else.*

**Schema decisions taken in this phase:**

1. **Space names are validated against a closed set**, at write time only. `COORDINATE_SPACES` =
   {`kinect_space_1`, `icp_registered`, `pca_calibrated`, `rf_centered`}. A free-text space would let
   `"rf_centred"` reach disk and read back as an unrecognised-but-accepted string, at which point a
   consumer guesses the frame or skips the file; a rejected write is the cheaper failure. Validation
   is *not* applied on read: the name does not affect decoding, and refusing an old name would break
   backward compatibility over a string the reader never uses.
2. **`vertex_id` requires its full provenance triple and a version that knows the column.** Writing
   the column with any of `reference_ply` / `reference_ply_vertex_count` / `dedup_epsilon` missing
   raises, and so does writing it under `schema_version = "1"`. The second rule is what catches the
   projection stage carrying its input's metadata through verbatim instead of restamping — the exact
   mistake the merging task's `CARRIED_METADATA_KEYS` pattern invites. The reader mirrors it: a file
   whose columns contradict its declared version is refused rather than half-believed.
3. **Numeric metadata is parsed, not just present.** `reference_ply_vertex_count` must round-trip to
   a positive int and `dedup_epsilon` to a finite positive float, both still stored as `str` per the
   existing writer contract.

**Compatibility confirmed:** `code/scripts/_4_merging/filter_contact_depth_field_by_neural_quality.py`
is unaffected. It iterates `COLUMN_DTYPES` (still the six required columns) for its bitwise
preservation assertion, and carries `schema_version` through verbatim, so a v1 input still yields a
v1 output. Its test file passes unchanged.

**Files Modified:** `.../tactile_quantification/io/contact_depth_field_io.py`,
`code/tests/test_contact_depth_field_io.py`

**Dependencies:** None (parallel with Phase 1)

### Phase 3: Surface the index mappings
**Goal:** Both index-consuming stages expose their mapping, with the CSV path byte-identical.
**Started:** 2026-08-18 13:55  **Completed:** 2026-08-18 14:10

- [x] 3.1 — `deduplicate_contact_points_csv`: accumulate the per-frame `kept_indices`.
      *Done, but by a cleaner route than a `return_indices=True` call: the clustering moved into a new
      `deduplicate_xy_mapping(points, epsilon) -> DedupMapping`, and `deduplicate_xy` became a thin
      application of what it returns (`points[kept_indices]`). One implementation of the clustering,
      so the CSV is unchanged by construction rather than by argument.*
- [x] 3.2 — Additionally surface the DBSCAN `labels`. *Done as the second field of the frozen
      `DedupMapping` — one entry per **input** point, so the collapsed group of each survivor is
      recoverable, which is what the max-magnitude inheritance rule needs. A second parallel flag was
      rejected: `return_indices=True` keeps its exact 3-tuple contract (its tests are untouched) and
      the mapping is reached through the new function instead.*
- [x] 3.3 — `_project_single_csv`: accumulate the per-row KD-tree `indices` and widen the return.
      *Done: it now returns a frozen `ProjectionResult(distances, vertex_indices)`. `distances` is the
      array it used to return, unchanged; `vertex_indices` is `frame_index → (M,) intp`.*
- [x] 3.4 — Assert the CSV outputs of both stages are byte-identical to a pre-change run. *Done —
      evidence below.*
- [x] 3.5 — Tests for both mappings. *Done — 35 new tests: 8 for `deduplicate_xy_mapping` (including
      that it agrees with `deduplicate_xy` on a 200-point random cloud), 12 for the dedup CSV mapping,
      and a new `test_project_contacts_onto_forearm.py` (15) that builds a real 12-vertex lattice and
      checks the indices against hand-computed nearest vertices and against the coordinates the CSV
      itself was given.*

**Both mappings are keyed by `frame_index`, never by row position.** The merged CSV is upsampled to
the nerve rate, so a row index is meaningless outside one particular file, and every other part of the
pipeline aligns on `frame_index`. The column arrives as float64 (non-Kinect rows hold NaN), so a
contact-bearing row with a NaN or fractional `frame_index` raises rather than becoming a wrong key,
and two contact-bearing rows sharing a `frame_index` raise as ambiguous. Verified on real data
(`ST13-01`, blocks 01-03): `frame_index` is unique across every contact-bearing row.

**Byte-identity evidence (3.4).** `git show HEAD:<path>` extracted both pre-change stage modules to
the scratchpad; a harness imported the old and new copies by file path in separate processes and ran
both over the same inputs — two real truncated blocks from
`3_merged/2022-06-14_ST13-01/` (block-order-01, 30 000 rows, 258 contact frames, 49 577 points;
block-order-02, 40 000 rows, 568 contact frames, 277 175 points), projecting onto the session's real
`forearm_deduped/*.ply`. All four output CSVs compare equal under `cmp` and sha256:

| stage | block | sha256 (identical before and after) |
|-------|-------|--------------------------------------|
| dedup | 01 | `717b1332b537f1f130bd91daa7a220cb0f55a6a791968f38d882bacb11235775` |
| projection | 01 | `a4454ef8fad1d1fd8119c70e3408eec32cb5c4064a85d4eb035c61090a027122` |
| dedup | 02 | `f6f9aa87c30d875acd70b8d7200a5da59acc1e1b88613b94a6f8abc64bd1a7e8` |
| projection | 02 | `931c0db7f5d5a9e8c81140f6421016328322b3312f0c51f6443efdd5112ace7f` |

**Deliberate duplication:** the `frame_index` coercion helper exists in both stage scripts. Sharing it
would mean one stage importing from the other, which the module contracts forbid; the shared version
belongs in the Phase 4 leaf if a third stage ever needs it.

**Files Modified:** `code/scripts/_5_postprocessing/deduplicate_xy_points.py`,
`code/scripts/_5_postprocessing/project_contacts_onto_forearm.py`,
`code/scripts/_5_postprocessing/__init__.py`, `code/tests/test_deduplicate_xy_points.py`,
`code/tests/test_project_contacts_onto_forearm.py` (new)

**Dependencies:** None

### Phase 4: The stage io leaf
**Goal:** A testable module that reads, transforms and writes the sidecar per stage.
**Started:** 2026-08-18 13:35  **Completed:** 2026-08-18 14:23

- [x] 4.1 — New `code/src/postprocessing/depth_field_stage_io.py`: apply a 4x4, apply a
      `CalibrationResult`, apply a per-frame index filter with the max-magnitude rule, and apply a
      per-frame vertex re-addressing. *Done as five pure functions —
      `apply_rigid_transform_to_field`, `apply_transform_schedule_to_field`,
      `apply_pca_calibration_to_field`, `apply_dedup_mapping_to_field`,
      `apply_vertex_addressing_to_field` — each taking and returning a schema-conforming table and
      never mutating its arguments. The 4x4 comes in both flavours because the ICP stage applies a
      **schedule**, not a matrix; the schedule path reproduces
      `transform_spatial_columns_scheduled`'s half-open segmentation exactly, transforms each row
      from its original coordinates (segments are disjoint), and leaves frames below the first
      `start_frame` alone. Both delegate to `csv_spatial_transformer.apply_rigid_transform`, so the
      two artifacts are moved by the same arithmetic and not merely the same matrix.*
- [x] 4.2 — The row-count agreement check: given a parquet and the CSV written by the same stage,
      assert per-frame parity and raise naming the first offending frame. *Done:
      `assert_row_counts_agree_with_csv(table, csv_path)`, over the two public counters
      `field_row_counts_by_frame` and `csv_contact_point_counts_by_frame`. The CSV side reads only
      `frame_index` and `contact_points` and parses with `parse_contact_points` itself, so it counts
      what the stage actually saw — a stricter parser would report a number no stage ever worked
      with. The message names the lowest offending frame, both counts, and how many frames disagree
      in total.*
- [x] 4.3 — float64 compute, float32 cast-back, dtype preserved. *Done in one place,
      `_with_coordinates`, which is the only writer of `x/y/z`: it rejects a wrong-shaped, non-finite
      or float32-overflowing result and casts exactly once, at the end. `signed_depth_mm` is
      untouched by every path except the dedup rule, asserted bitwise.*
- [x] 4.4 — Tests on synthetic tables — no Open3D, no PyQt5, no recording. *Done — 76 tests.*

**The dedup reduction rule, stated unambiguously.** A survivor inherits the **minimum** signed depth
of its DBSCAN group — the most negative value, which under the `negative_is_penetrating` convention
is the deepest penetration. The value is taken verbatim from the group; nothing is averaged or
recomputed. On real data this is the same row "largest absolute value" would pick: the field's
positive values, where they exist at all, are bounded by the contact-detection epsilon
(`inside_mask = np.all(tri_distances < epsilon, axis=1)`), so a positive can never be the largest
magnitude in a group that contains a penetrating vertex. Where a synthetic group mixes signs the two
readings diverge, and the module keeps the penetrating row; a test asserts that, and a 20-case
randomised test asserts the invariant the rule exists for — per-frame `max(|signed_depth_mm|)` is
unchanged by deduplication, which is what keeps 6.2 / 9.4 true against the CSV's `contact_depth`.

**Purity, and the conftest change it required.** The leaf imports no geometry engine and no GUI
toolkit: vertex re-addressing takes a vertex *array*, the count check takes a *CSV path*. But
`csv_spatial_transformer` — numpy, pandas and `re` only — sits under
`preprocessing/forearm_extraction/registration/__init__.py`, which imports open3d and
`open3d.visualization.gui`, so importing the leaf dragged both in. `conftest.py` now stubs
`preprocessing.forearm_extraction.registration` the same way it already stubs its parent, and a
subprocess test proves the leaf imports with neither `open3d` nor `PyQt5` in `sys.modules`. The
stub-lifting dance in `test_contact_depth_field.py::_load_processor` had to lift the subpackage stub
as well — the parent's own `__init__` re-exports from `.registration` — or its six processor tests
would have skipped instead of running.

**Public surface:** `apply_rigid_transform_to_field`, `apply_transform_schedule_to_field`,
`apply_pca_calibration_to_field`, `apply_dedup_mapping_to_field`, `apply_vertex_addressing_to_field`,
`assert_row_counts_agree_with_csv`, `field_row_counts_by_frame`,
`csv_contact_point_counts_by_frame`; the constants `FRAME_INDEX_COLUMN`, `CONTACT_POINTS_COLUMN`,
`COORDINATE_COLUMNS`, `DEPTH_COLUMN`; and the types `TransformSchedule` and `DedupMappingLike`.
`DedupMappingLike` is a `Protocol`, not an import: the stage scripts depend on this module, so the
module cannot depend on them, and `deduplicate_xy_points.DedupMapping` satisfies it as written —
Phase 6 passes `frame_mappings` straight through with no conversion.

**Files Modified:** `code/src/postprocessing/depth_field_stage_io.py` (new),
`code/tests/test_depth_field_stage_io.py` (new), `code/tests/conftest.py`,
`code/tests/test_contact_depth_field.py`

**Dependencies:** Phase 2

**Verification:** full suite **517 passed, 7 skipped** (441 + 76 new; skip count unchanged from the
Phase 3 baseline).

### Phase 5: ICP propagation (Space 1 → 2)
**Goal:** The field arrives in `blocks_registered/`.
**Started:** 2026-08-18 14:15  **Completed:** 2026-08-18 14:40

- [x] 5.1 — Reuse the **same** `schedule` object the CSV uses (`apply_icp_registration.py:89`), and
      the CSV-derived `max_frame` from `:87` — the parquet holds only contacting frames and would
      otherwise build a different schedule. *Done: the loop body is unchanged up to and including the
      `get_transform_schedule` call; the sidecar is handed that same `schedule` object.*
- [x] 5.2 — Vectorised per-segment mask; `signed_depth_mm` untouched. *Done via
      `apply_transform_schedule_to_field`, plus an explicit stage-level assertion
      (`_assert_depth_preserved`) that the column survived values and dtype alike — a silently
      re-derived depth would be indistinguishable from a measured one in the output file.*
- [x] 5.3 — Handle the no-transforms passthrough at `:77`. *Done, and the second one at `:98` (a
      block with no applicable transform) as well. Both use a **byte copy**, not a read-modify-write:
      the points did not move, so `coordinate_space` stays `kinect_space_1`, and copying the bytes is
      the only way to guarantee no key was restamped in passing.*
- [x] 5.4 — Restamp `coordinate_space = "icp_registered"`. *Done for the transformed branch only.
      Every other metadata key is carried through verbatim — including `pipeline_stage = "merging"`,
      deliberately left alone: only the coordinates changed, and re-deriving provenance a stage did
      not produce is what lets the two artifacts drift apart.*
- [x] 5.5 — Idempotency at the session boundary, matching `:53-60`. *Done: both artifacts in
      `input_paths`, `output_paths` and `clean_task_outputs`, so deleting only the parquet re-runs the
      whole session and the two can never be regenerated out of step.*
- [x] 5.6 — `assert_row_counts_agree_with_csv` after every block, against the CSV this stage just
      wrote. *Done on both branches. It passed on all four real ST13-01 blocks (187k / 923k / 804k /
      403k rows) — the first real-data confirmation that the sidecar and the merged CSV agree
      frame-by-frame.*

**Signature change.** `apply_icp_registration` now returns
`Tuple[List[Path], List[Path]]` — `(csv_paths, parquet_paths)` — and takes a new keyword-only
`input_parquets: Optional[Sequence[Path]] = None`. Phase 8 binds the second return slot and passes
`source_parquets`; until then the executor's positional `outputs` binding keeps `registered_files`
pointing at the CSVs, so the workflow is correct unchanged apart from the flow wrapper's annotation.
When `input_parquets` is omitted the paths are derived by the new module-level
`depth_field_path_for_csv`, which is the naming rule the merging pipeline used to write them
(`..._merged_data.csv` → `..._contact_depth_field.parquet`, same directory) and which holds at every
later stage too, since each stage writes its CSV under the input's own name. This is a derivation,
not a fallback: the derived path is then required to exist.

**Fail-fast:** a block whose sidecar is absent raises `FileNotFoundError` naming the file, before
`should_process_task` is consulted. The depth field is a hard input of postprocessing; skipping the
block would leave a gap nothing downstream would notice.

**Byte-identity evidence.** `git show HEAD:code/scripts/_5_postprocessing/apply_icp_registration.py`
extracted the pre-change module to the scratchpad; a harness ran old and new over the same real
inputs — the four `blocks_filtered/` CSVs of `2022-06-14_ST13-01`, a genuinely multi-forearm session
whose schedules have 3, 1, 2 and 1 segments — into separate output directories. All four CSVs are
sha256-identical:

| block | sha256 (identical before and after) |
|-------|--------------------------------------|
| 01 | `a7f6a4d327373bfd1d4e9273d792c8000e4508c5f6cd6ecdf54dad18ed117774` |
| 02 | `a41500da5a7661cabb9446e1a431575782a9f165dfcdde4ad69639398321feb2` |
| 03 | `af597c647267912681a63a9d38125ee8ad9a61bfbc09c4bec30de6b63c94dba0` |
| 04 | `1b17f8528e52e59c312a976459e866809859ef7a4aaab4142aa53642859c7700` |

Both passthrough branches were exercised separately (a session directory with no transforms file;
a block whose number precedes every snapshot, giving an empty schedule) and their CSVs are
sha256-identical to the pre-change module's on the same inputs, with the sidecar byte-identical to
its input and still declaring `kinect_space_1`.

**Cross-check that the two artifacts landed in the same space.** For each block, the registered
parquet's coordinates were compared against the registered CSV's parsed `contact_points` over 200
contact frames. Worst per-axis disagreement: 0.050 / 0.086 / 0.086 / 0.093 mm — inside the 0.137 mm
bound implied by the CSV's double quantisation (its Space-1 `contact_points` were already `%.1f`,
the rotation spreads that ≤ 0.05·√3 across axes, and the stage re-rounds on write) against the
parquet's unquantised float32. `signed_depth_mm` is bitwise unchanged on all four blocks; `x/y/z`
moved by up to 5.5 mm and are still float32.

**Files Modified:** `code/scripts/_5_postprocessing/apply_icp_registration.py`,
`code/scripts/_5_postprocessing/__init__.py`,
`code/scripts/postprocess_workflow_kinect_auto.py` (flow wrapper return annotation only)

**Dependencies:** Phases 3, 4

**Verification:** full suite **517 passed, 7 skipped** — unchanged from the Phase 4 baseline, same
seven skip reasons.

### Phase 6: Dedup and projection (the index-consuming stages)
**Goal:** The field survives row removal and vertex re-addressing, and gains `vertex_id`.
**Started:** 2026-08-18 14:30  **Completed:** 2026-08-18 15:05

- [x] 6.1 — Dedup: filter parquet rows by the CSV's per-frame `kept_indices`; the survivor's
      `signed_depth_mm` becomes the maximum magnitude of its DBSCAN group. Never re-run DBSCAN.
      *Done: `deduplicate_contact_depth_field(input_parquet, output_parquet, output_csv,
      frame_mappings)` in the dedup stage script, consuming `frame_mappings` from the same
      `deduplicate_contact_points_csv` call. On real ST13-01 blocks it drops 56.5% / 61.9% of the
      sidecar's rows — exactly the fraction the CSV lost.*
- [x] 6.2 — Assert after dedup that per-frame `max(|signed_depth_mm|)` still equals the CSV's
      `contact_depth` — this is what the reduction rule was chosen to preserve. *Done:
      `assert_max_depth_agrees_with_csv` in the Phase 4 leaf, over the new public counters
      `csv_contact_depth_by_frame` and `field_max_depth_magnitude_by_frame`. Measured on real data:
      max absolute disagreement **8.88e-16 mm** over 258 + 568 frames (max relative 4.5e-16), i.e.
      one ULP of the CSV's decimal round trip.*
- [x] 6.3 — The epsilon used must be the one actually applied, including an interactive override
      (`postprocess_workflow_kinect_auto.py:126`), so the hook lives inside that wrapper. *Done, and
      by construction rather than by threading a number: the parquet is reduced by the mapping the
      CSV run returned, so it cannot have been clustered at a different epsilon. The `dedup_epsilon`
      that reaches metadata at 6.5 is read from the Phase 1 JSON sidecar, never re-derived.*
- [x] 6.4 — Projection: re-address each parquet row's `x/y/z` using the CSV's KD-tree `indices`, and
      write `vertex_id`. Never re-query the tree. *Done: `_write_projected_field` consumes
      `ProjectionResult.vertex_indices`. The tree is queried exactly once per block, by the CSV path.*
- [x] 6.5 — Stamp `reference_ply`, `reference_ply_vertex_count`, `dedup_epsilon`. *Done, plus
      `schema_version` bumped to `"2"` — Phase 2 refuses the column under version 1. The count is
      also checked against the PLY actually loaded before anything is written.*
- [x] 6.6 — `signed_depth_mm` preserved through both stages except where 6.1 deliberately replaces it.
      *Done, asserted at both stage boundaries: `_assert_depth_only_reduced` (dedup — every surviving
      value must be one the frame already held, so nothing is averaged or invented) and
      `_assert_depth_preserved` (projection — bitwise, dtype included). Bitwise preservation through
      projection confirmed on both real blocks.*
- [x] 6.7 — Idempotency: session boundary for dedup, **block** boundary for projection
      (`project_contacts_onto_forearm.py:173`). *Done and exercised on real data: an unchanged
      re-run of `deduplicate_xy_flow` rewrites nothing; deleting one parquet regenerates the whole
      session; the projection stage skips or re-runs per block, and `clean_task_outputs` removes both
      artifacts at each stage.*

**Signature changes.**

- `project_contacts_onto_forearm` now returns `Tuple[List[Path], List[Path]]` — `(csv_paths,
  parquet_paths)` — and takes a keyword-only `input_parquets: Optional[Sequence[Path]] = None`,
  derived from the CSV paths when omitted. The missing-PLY and empty-PLY early returns became
  `([], [])`. Phase 8 binds the second slot; until then the positional `outputs` binding keeps
  `projected_files` on the CSVs.
- `deduplicate_xy_flow` now returns `(deduped_csvs, deduped_forearm_ply, deduped_depth_fields)` —
  a third slot appended, so the first two keep their meaning.
- `project_contacts_onto_registered_forearm_flow` mirrors the stage and returns a 2-tuple.
- New: `deduplicate_contact_depth_field` (dedup stage), exported from `_5_postprocessing`.

**Two module moves, forced by the "stage scripts must not know about each other" contract.**

1. **`postprocessing/forearm_dedup_metadata.py` (new).** The Phase 1 provenance sidecar is *written*
   by the dedup stage and now *read* by the projection stage, which needs the epsilon and the vertex
   count to stamp at 6.5. The constants, the path rule, the writer and a new
   `read_forearm_dedup_metadata` moved into a leaf; `deduplicate_xy_points` re-exports all of them,
   so `_5_postprocessing/__init__`, the workflow and the existing tests are unaffected. The reader
   parses every field rather than fetching it: an unverified epsilon stamped onto a `vertex_id`
   artifact is worse than no artifact.
2. **`depth_field_path_for_csv` moved from `apply_icp_registration` into the Phase 4 leaf**, for the
   same reason — projection needs the CSV/sidecar pairing rule too. `apply_icp_registration` imports
   and re-exports it, so the Phase 5 public surface is unchanged.

**A defect the 6.2 check found immediately.** `contact_depth` is `0.0`, not blank, on every row where
nothing touched the arm, so a first cut keyed on "finite `contact_depth`" reported 642 phantom frames.
Membership is now decided by the same `contact_points` cell the row-count check parses, so the two
checks agree on which frames exist rather than each having its own idea.

**Byte-identity evidence.** `git show HEAD:<path>` extracted both pre-change stage modules to the
scratchpad; a harness imported old and new by file path and ran both over the same real inputs — two
truncated `2022-06-14_ST13-01` blocks (block-order-01: 30 000 rows, 258 contact frames, 49 577 field
rows; block-order-02: 40 000 rows, 568 contact frames, 277 175 field rows), projected onto a forearm
PLY deduplicated at the DAG's epsilon of 0.5 (3 718 → 1 807 vertices). All four CSVs are
sha256-identical:

| stage | block | sha256 (identical before and after) |
|-------|-------|--------------------------------------|
| dedup | 01 | `c1f6d9b0384c65a1f7bae381653bcaf2fe6b4acb86044b8f0e6490279d112226` |
| dedup | 02 | `1b212eb51f565bc1338709aad3940b964d674cb841f00f9e3d488511bee95b99` |
| projection | 01 | `b2a2a891cfb04a8f5e7e1bbc86b91b083c3750c869fc15e3cae75d5154a7a85e` |
| projection | 02 | `28c10b07d29149740b4ca8846844ee2e05da3c8ba220165209a03e43e7853c6e` |

**Cross-checks on the projected sidecar** (real data, both blocks): `vertex_id` is `int32`, in range
(`[556, 1223]` and `[7, 1546]` against 1 807 vertices), and resolving it against the reference PLY
reproduces the parquet's own `x/y/z` to within **3.05e-05 mm** — the float32 storage rounding, and
the cross-check that the index and the coordinates agree. `signed_depth_mm` is bitwise unchanged by
projection. `coordinate_space` is carried through untouched by both stages, as specified.

**Fail-fast:** a block whose sidecar is absent raises `FileNotFoundError` naming the file — in the
dedup flow wrapper *before* the idempotency check, and in the projection stage's
`_resolve_input_parquets`. A missing or count-mismatched forearm provenance sidecar raises before any
`vertex_id` is written. A frame in the field with no mapping entry, or a mapping entry for an absent
frame, raises from the leaf.

**Files Modified:** `code/scripts/_5_postprocessing/deduplicate_xy_points.py`,
`code/scripts/_5_postprocessing/project_contacts_onto_forearm.py`,
`code/scripts/_5_postprocessing/apply_icp_registration.py`,
`code/scripts/_5_postprocessing/__init__.py`,
`code/scripts/postprocess_workflow_kinect_auto.py`,
`code/src/postprocessing/depth_field_stage_io.py`,
`code/src/postprocessing/forearm_dedup_metadata.py` (new),
`code/tests/test_depth_field_stage_io.py`, `code/tests/test_deduplicate_xy_points.py`,
`code/tests/test_project_contacts_onto_forearm.py`,
`code/tests/test_forearm_dedup_metadata.py` (new)

**Dependencies:** Phase 5

**Verification:** full suite **581 passed, 7 skipped** (517 + 64 new; skip count and reasons unchanged
from the Phase 5 baseline).

### Phase 7: PCA and RF-centring (Space 2 → 3 → 4)
**Goal:** The field reaches `blocks_rf_centered/`.
**Started:** 2026-08-18 14:45  **Completed:** 2026-08-18 15:19

- [x] 7.1 — PCA: one vectorised `apply_full_transform` over the whole `x/y/z` block, using the same
      `CalibrationResult` the CSV uses. Pass a copy — it mutates in place.
      *Done: `_write_calibrated_field` is handed the very `calib_result` object the CSV loop used —
      no re-fit, no reload of the JSON just written. The private copy is made inside the Phase 4 leaf's
      `apply_pca_calibration_to_field`, which is the single place that knows the mutation.*
- [x] 7.2 — Key the parquet loop off `input_files`, **not** `loaded_data`, which silently omits
      blocks that failed gesture segmentation. *Done, and the silent drop is converted into a loud
      one: `_write_calibrated_field` raises `FileNotFoundError` naming the CSV that was never written
      and the `loaded_data` mechanism that dropped it. Exercised on real data — a block whose
      `single_touch_id` is 0 throughout raises instead of yielding a session with three CSVs and four
      sidecars.*
- [x] 7.3 — Handle the `_pca-xyz` filename fork with an explicit parquet suffix constant.
      *Done: `CalibrationConfig.output_parquet_suffix = "_pca-xyz.parquet"`, beside
      `output_csv_suffix`, both applied with the same `f"{p.stem}{suffix}"` shape. The two are then
      **checked against each other**: `_assert_output_names_pair` requires the sidecar name this stage
      produces to equal what `depth_field_path_for_csv` resolves the CSV to, because that function is
      how the next stage finds its inputs. The leaf's pairing rule was generalised from a suffix swap
      to a marker substitution (`_merged_data` → `_contact_depth_field`) so it survives the fork;
      three new tests cover the forked name, a non-CSV name and an ambiguous double-marker name.*
- [x] 7.4 — RF-centring: same translation matrix; handle the no-cluster passthrough at `:267-280`.
      *Done: `_translate_single_field` applies the same `T` from `_build_translation_matrix`, and the
      passthrough branch copies the sidecar beside the CSVs it copies. `shutil.copy2`, not
      `copyfile` — matching the CSV copy two lines above it, so an unchanged re-run reproduces the
      previous outputs exactly, mtimes included, instead of making the sidecar look newer than the CSV
      it belongs to. Verified: the passthrough sidecars are sha256-identical to their inputs and still
      declare `pca_calibrated`.*
- [x] 7.5 — Restamp `pca_calibrated` then `rf_centered`. *Done, and `pipeline_stage` with them — see
      below.*
- [x] 7.6 — Confirm `vertex_id` is carried through both stages untouched. *Done, asserted at both
      stage boundaries by `_assert_depth_and_vertex_preserved` (values and `int32` dtype), which also
      raises if the column is **absent** — a field reaching these stages unprojected is a pipeline
      error, not something to work around. Confirmed on real data, and cross-checked the other way:
      resolving the terminal `vertex_id` against `forearm_rf_centered/*.ply` reproduces the parquet's
      own `x/y/z` to 0.0975 mm, inside the two 0.1 mm PLY roundings that separate them.*
- [x] 7.7 — Do **not** feed the field into the PCA fit or the RF estimate. *Done — neither
      `compute_calibration`'s inputs (`tapping_segments` / `stroking_segments`, built only from
      sticker columns) nor `_compute_rf_center` was touched. Proved by the byte-identity evidence
      below: `pca-xyz_transformation-matrices.json` and `rf_center_origin.json` are sha256-identical to
      the pre-change run, so neither coordinate system moved.*

**`pipeline_stage` restamped to `"postprocessing"`.** It was carried through as `"merging"` from the
merging-stage filter, which by `blocks_rf_centered/` is simply false. Both Phase 7 stages now set it to
`"postprocessing"` — the coarse stage name, not the per-task one, because the artifact's task-level
provenance is already carried by the keys that mean something (`reference_ply`, `dedup_epsilon`,
`coordinate_space`), and a per-task value would need restamping in five places to stay honest. The
writer validates `coordinate_space` against a closed set but treats `pipeline_stage` as free text, so
no schema change was needed. The constant lives in the Phase 4 leaf
(`PIPELINE_STAGE_POSTPROCESSING`) so the two stages spell it identically without importing each other.
**Known gap:** `blocks_registered/`, `blocks_deduped/` and `blocks_projected/` still read `"merging"`
— Phase 5 deliberately left it alone and that decision was not reversed here. The terminal artifact and
`blocks_pca_calibrated/` are correct.

**Signature changes.**

- `calibrate_pca_xyz` now returns `Tuple[List[Path], Path, Path, List[Path]]` — a fourth slot,
  `(csvs, output_dir, forearm_ply, parquets)` — and takes a keyword-only
  `input_parquets: Optional[Sequence[Path]] = None`, derived from the CSV paths when omitted. The
  early returns (`should_process_task` skip, insufficient gesture data) grew the fourth slot too.
- `center_on_receptive_field` now returns `Tuple[List[Path], List[Path]]` — `(csv_paths,
  parquet_paths)` — and takes the same keyword-only `input_parquets`. The no-input early return
  became `([], [])`.
- Output binding is positional and ignores extra slots, so `pca_files` / `pca_output_dir` /
  `pca_forearm` / `rf_files` keep their meaning until Phase 8 binds the new ones; only the two flow
  wrappers' return annotations changed.
- `depth_field_path_for_csv` (Phase 4 leaf) now pairs by marker substitution rather than suffix
  swap, so it resolves `*_merged_data_pca-xyz.csv` as well as `*_merged_data.csv`. New:
  `MERGED_CSV_MARKER`, `DEPTH_FIELD_MARKER`, `PIPELINE_STAGE_POSTPROCESSING`.
- `_5_postprocessing` additionally exports `CalibrationConfig`, `COORDINATE_SPACE_AFTER_PCA` and
  `COORDINATE_SPACE_AFTER_RF_CENTERING`.

**Byte-identity evidence.** `git show HEAD:<path>` extracted both pre-change stage modules to the
scratchpad; a harness imported old and new by file path and ran both over the same real inputs — all
four `blocks_projected/` CSVs of `2022-06-14_ST13-01` (106k / 105k / 100k / 44k rows; 1 030 / 1 900 /
2 790 / 1 014 contact frames), with the schema-v2 sidecars reconstructed from each CSV's own
`contact_points` against the session's real `forearm_deduped/*.ply` (1 807 vertices). Every output of
both stages is sha256-identical, including the two files that define the coordinate systems:

| stage | artifact | sha256 (identical before and after) |
|-------|----------|--------------------------------------|
| PCA | block-order-01 csv | `20fd272ba5b2d23d425076ec5d3e3ce471fa6394ee5a1223dccf0791b0e16628` |
| PCA | block-order-02 csv | `d7ccb73c837775160dc72b8c8a608d067a0cdfe6fef4be312d0023681669765f` |
| PCA | block-order-03 csv | `5cbf09670f23a992eed00fe22f17652ee5d73a4a3976ad69ef8ea5fd356bb46b` |
| PCA | block-order-04 csv | `7e510e09d34e11feed898de7043eb7ab5602d066a07a56445f9c65f672265bde` |
| PCA | `pca-xyz_transformation-matrices.json` | `48e600d402069abfcc9bde0ab100be87f1a8b7f241636194c5fa2255bdaa0418` |
| PCA | `forearm_pca_calibrated/*.ply` | `a76a951a49c6a3f96e8779f1f555e3484de3e3d2c312ccf9869881a58dcfc5a1` |
| RF (translating) | block-order-01 csv | `debd34e2a7eb6d06adc4b59a26caa02879ecc7d75decc27335792c7c20daf5a5` |
| RF (translating) | block-order-02 csv | `dc198a7c3e0b8d2552a5f31678aaea33b642c9685814920c235a441c3f24f9ed` |
| RF (translating) | block-order-03 csv | `9dc5a1526507827d2ba8da2f09e3c07aa919cb8f23b2369936e0a41c9b5bb4fc` |
| RF (translating) | block-order-04 csv | `75aac7ad8a63365059d3b055194c08fd2a469bd339fe6b34f5566b6a077039cd` |
| RF (translating) | `rf_center_origin.json` | `ad61830073b837fa0664fff3cf17709308c7bae975045643c031149218b9c50e` |
| RF (translating) | `forearm_rf_centered/*.ply` | `4b819dccd45d809901f9147bc718d4de8d425e0d8439c185eb48a43cf9f39946` |

ST13-01's selectivity never clears the default DBSCAN threshold, so a plain run takes the
**passthrough** branch. Both branches were therefore exercised: the passthrough one on the default
config (all four CSVs, `rf_center_origin.json` and the PLY sha256-identical old vs new; the sidecars
byte-identical to their inputs and still declaring `pca_calibrated`), and the translating one with a
permissive `SelectivityDBSCANConfig` injected identically into both modules — the table above.

**Cross-checks on the terminal sidecar** (real data, all four blocks). `signed_depth_mm` is **bitwise**
unchanged through both stages, dtype included. `vertex_id` is unchanged and still `int32`, in range
(`[13, 1713]` against 1 807 vertices). `x/y/z` are still float32 and moved up to 890 mm. The RF
translation is exactly `-rf_center` on every row of every block, identical to 4 decimal places across
all axes and all 983 k rows. `coordinate_space` reads `icp_registered → pca_calibrated → rf_centered`;
`pipeline_stage` reads `merging → postprocessing → postprocessing`; every other metadata key is
unchanged from the input. Row-count agreement against the stage's own CSV is asserted inside both
stages and passed on every block.

**Idempotency**, measured on the real fixture: an unchanged re-run of either stage reproduces its
outputs exactly (byte- and mtime-identical); deleting only a parquet regenerates the whole session's
pair at both stages; both artifacts are in `input_paths`, `output_paths` and `clean_task_outputs`.

**Files Modified:** `code/scripts/_5_postprocessing/set_xyz_reference_from_gestures.py`,
`code/scripts/_5_postprocessing/center_on_receptive_field.py`,
`code/scripts/_5_postprocessing/__init__.py`,
`code/scripts/postprocess_workflow_kinect_auto.py` (flow wrapper return annotations only),
`code/src/postprocessing/depth_field_stage_io.py`, `code/tests/test_depth_field_stage_io.py`

**Dependencies:** Phase 6

**Verification:** full suite **584 passed, 7 skipped** (581 + 3 new; skip count and reasons unchanged
from the Phase 6 baseline).

### Phase 8: Wiring and idempotency
**Goal:** The DAG produces the artifact at every stage.
**Started:** 2026-08-18 15:20  **Completed:** 2026-08-18 15:40

- [x] 8.1 — Resolve `source_parquets` alongside `session_input_files` and seed it into `context`,
      index-aligned with `session_configs`. *Done in the same loop over `session_configs` that builds
      the CSV list, so the two can only ever be built in step, using `depth_field_path_for_csv`.
      Existence is required there, with a `FileNotFoundError` naming the file and naming the merging
      task that produces it — the same treatment the CSV already got. The field is a hard input, and a
      session that lacks one belongs out of the DAG's `kinect_configs` list, not discovered mid-run.*
- [x] 8.2 — Extend each stage's `outputs` list and pass the parquet inputs through `params`; binding
      is positional, so existing keys keep their order. *Done — appended `registered_depth_fields`,
      `deduped_depth_fields`, `projected_depth_fields`, `pca_depth_fields`, `rf_depth_fields`. Every
      existing key kept its index, so `registered_files` / `deduped_files` / `deduped_forearm` /
      `projected_files` / `pca_files` / `pca_output_dir` / `pca_forearm` / `rf_files` still mean what
      they meant. The five flow wrappers gained a keyword-only `input_parquets`, forwarded to the
      stage function they wrap; `deduplicate_xy_flow` — the one stage whose sidecar handling lives in
      the workflow file — takes it too and length-checks it against `input_files` before use.*
- [x] 8.3 — Both artifacts in `should_process_task` / `clean_task_outputs`, at each stage's own
      boundary. *Verified end to end rather than re-implemented: Phases 5-7 put the parquets on both
      sides inside each stage function, so this phase measured the result instead of duplicating it.
      On the real ST13-01 session, deleting one sidecar at each of the five stages
      (`blocks_registered/` 01, `blocks_deduped/` 02, `blocks_projected/` 03,
      `blocks_pca_calibrated/` 04, `blocks_rf_centered/` 01) and re-driving regenerated all five, each
      byte-for-byte the same size as the file removed. An unchanged re-run skips ICP, dedup,
      projection (per block) and PCA.*
- [x] 8.4 — Never pass an empty list. *Done through one named helper, `_nonempty_list`, applied to
      every parquet list bound into `params`: `[]` and `None` both become `None`. The guard's own code
      is quoted in that helper's docstring, with the failure it would cause — the stage skipped whole,
      the previous stage's outputs left bound to the next stage's inputs — spelled out, and a pointer
      to it added beside the guard itself. `None` is not a fallback: the stage then derives the
      sidecar paths from its CSV paths and **requires** them to exist, so an absent field is a named
      `FileNotFoundError` rather than a silent skip. Confirmed on the real drive: the guard printed
      nothing and all seven tasks ran.*
- [x] 8.5 — The DAG config's session list. *`valid_configs_ST15-01` removed from `kinect_configs`,
      with the reason recorded in the file. Two independent reasons, both fatal:
      `3_merged/2022-06-16_ST15-01/blocks_filtered/` holds 4 merged CSVs and **0**
      `*_contact_depth_field.parquet`, and the config group had already been renamed
      `_NOT_ENOUGH_DATA_valid_configs_ST15-01`, so `resolve_session_configs()` raised
      `FileNotFoundError` on the stale entry before any session ran at all — the DAG as listed was
      unrunnable. A survey of the other eleven listed sessions found a 1:1 CSV/parquet pairing in
      every one (4/4, 8/8, 10/10, 9/9, 6/6, 3/3, 15/15, 5/5, 16/16, 16/16, 7/7); the list now resolves
      to 99 block configs.*
- [x] `pipeline_stage` restamp completed. *Phase 7 left `blocks_registered/`, `blocks_deduped/` and
      `blocks_projected/` reporting `"merging"`. All three now stamp `PIPELINE_STAGE_POSTPROCESSING`,
      so no intermediate misreports where the artifact has been. This **reverses one Phase 5
      decision**: `_copy_field_unchanged` was a byte copy, and is now a read-modify-write that
      restamps exactly one key. `coordinate_space` is still carried through untouched on that branch —
      the points genuinely did not move — but a passthrough block that kept `"merging"` while its
      transformed siblings said `"postprocessing"` would make the stamp mean nothing. The CSV on that
      same branch is already round-tripped through pandas rather than byte-copied, so the sidecar now
      matches it.*

**Idempotency finding, pre-existing and out of scope.** `center_on_receptive_field` never skips on its
**no-cluster passthrough** branch. `shutil.copy2` preserves each input's mtime, so the outputs inherit
the *spread* of input mtimes: on ST13-01 the oldest output is 15:30:14 (block-01 CSV) while the latest
input is 15:30:25 (block-04 parquet), and `should_process_task`'s `latest_input > oldest_output` test
is therefore true forever. This predates Phase 8 — the CSV copies alone produce it — and the re-run is
idempotent in content (byte- and mtime-identical outputs), merely not skipped. Not repaired here; this
phase is wiring only.

**Files Modified:** `code/scripts/postprocess_workflow_kinect_auto.py`,
`configs/postprocess_workflow_kinect_auto_dag.yaml`,
`code/scripts/_5_postprocessing/apply_icp_registration.py`,
`code/scripts/_5_postprocessing/deduplicate_xy_points.py`,
`code/scripts/_5_postprocessing/project_contacts_onto_forearm.py`,
`code/tests/test_deduplicate_xy_points.py`, `code/tests/test_project_contacts_onto_forearm.py`

**Dependencies:** Phase 7

**Verification:** `run_single_session_postprocessing` driven over the **whole real session**
`2022-06-14_ST13-01` (4 blocks), through the actual DAG config, with no GUI monitor — every one of the
seven tasks completed and all five spatial stages wrote 4 sidecars each:

| stage | rows/block | schema | `coordinate_space` | `pipeline_stage` | `vertex_id` |
|-------|-----------|--------|--------------------|------------------|-------------|
| `blocks_filtered/` (input) | 187 056 / 923 205 / 804 323 / 402 753 | 1 | `kinect_space_1` | `merging` | — |
| `blocks_registered/` | unchanged | 1 | `icp_registered` | `postprocessing` | — |
| `blocks_deduped/` | 83 492 / 389 724 / 335 528 / 174 170 | 1 | `icp_registered` | `postprocessing` | — |
| `blocks_projected/` | unchanged | 2 | `icp_registered` | `postprocessing` | `int32`, in `[13, 1713]` of 1 807 |
| `blocks_pca_calibrated/` | unchanged | 2 | `pca_calibrated` | `postprocessing` | unchanged |
| `blocks_rf_centered/` | unchanged | 2 | `pca_calibrated`¹ | `postprocessing` | unchanged |

¹ ST13-01's selectivity never clears the DBSCAN threshold, so RF-centring took its documented
**passthrough** branch: the sidecar is copied and correctly still declares `pca_calibrated`. The
translating branch was exercised in Phase 7 and is not re-run here.

Measured on that output: row-count agreement between each parquet and the CSV of the same stage holds
at **all five stages, all four blocks** (24 checks). `signed_depth_mm` is **bitwise** unchanged (dtype
included) through ICP, projection, PCA and RF-centring. Per-frame `max(|signed_depth_mm|)` is
identical before and after dedup — max difference **0.000e+00** over 1 030 + 1 900 + 2 790 + 1 014
frames. `vertex_id` is unchanged through PCA and RF-centring. `x/y/z` are still `float32` and moved up
to 886.8 mm at the PCA stage. `reference_ply`, `reference_ply_vertex_count = 1807` and
`dedup_epsilon = 0.5` are stamped from projection onward. The empty-list guard printed nothing on any
run. Full suite **585 passed, 7 skipped** (584 + 1 new test; skip count unchanged and the same seven
reasons — five reference-recording bundles, two `analyse_workflow_*` configs).

### Phase 9: Verification
**Goal:** Measured evidence at every stage, not argument.
**Started:** 2026-08-18 15:40  **Completed:** 2026-08-18 16:10

- [x] 9.1 — Run one session end to end. *Two sessions, so the evidence is not single-session:
      `2022-06-14_ST13-01` (4 blocks) in Phase 8, and `2022-06-15_ST14-02` (6 blocks) here — chosen
      because its `rf_center_origin.json` reads `status: "ok"`, which is what puts the **RF
      translating branch** on the end-to-end path for the first time. 7/7 tasks, 6 sidecars at every
      one of the five stages.*
- [x] 9.2 — Row-count agreement between parquet and CSV at **all five** stages. *36/36 on ST14-02
      (six blocks x the five stages plus the Space-1 input), on top of Phase 8's 24/24 on ST13-01.*
- [x] 9.3 — `signed_depth_mm` bitwise unchanged through ICP, PCA and RF-centring. *All six ST14-02
      blocks, all four transitions including projection, values and dtype alike.*
- [x] 9.4 — After dedup, per-frame `max(|signed_depth_mm|)` equals the CSV's `contact_depth`.
      *Max absolute disagreement 7.105e-15 mm over 10 199 frames; the field's own per-frame maximum
      is unchanged by dedup to **0.000e+00** despite 34.6-45.4% of rows being dropped.*
- [x] 9.5 — Every `vertex_id` in range; resolving it against `forearm_rf_centered/*.ply` reproduces
      the parquet's `x/y/z` within the PLY's 0.1 mm rounding. *Measured on genuine end-to-end output
      rather than Phase 7's reconstruction: **0.09999771 mm** on ST14-02 (5.68 M rows, 10 712
      vertices) and **0.04999390 mm** on ST13-01 (983 k rows, 1 807 vertices).*
- [x] 9.6 — The CSVs at every stage are byte-identical to a pre-change run of the same inputs.
      *Proven end-to-end through the whole chain — see the controlled experiment below. 36/36
      artifacts identical.*
- [x] 9.7 — Delete one stage's parquet, re-run, confirm regeneration; re-run clean, confirm skip.
      *Regeneration measured per stage on ST13-01 in Phase 8. The clean-re-run half is measured here
      on ST14-02: **all seven tasks skipped**, in 0.04 s.*
- [x] 9.8 — Record artifact sizes per stage. *Per stage, for both driven sessions, below. The
      **dataset-wide** total is **not** measured — only 2 of the DAG's 11 sessions were driven.*

**Files Modified:** none (verification only)

**Dependencies:** Phase 8

---

#### Verification Results (2026-08-18)

Session `2022-06-15_ST14-02`, driven through `run_single_session_postprocessing` on the real DAG
config with no GUI monitor. All 7 tasks completed; 6 sidecars landed at each of the five stages.

**Per-stage shape, metadata and row-count agreement (9.1, 9.2).** Rows are summed across the six
blocks; the per-block breakdown is in `verify_2022-06-15_ST14-02.json`.

| stage | rows (6 blocks) | schema | `coordinate_space` | `pipeline_stage` | `vertex_id` | row-count checks |
|-------|----------------|--------|--------------------|------------------|-------------|------------------|
| `blocks_filtered/` (input) | 9 284 464 | 1 | `kinect_space_1` | `merging` | — | 6/6 |
| `blocks_registered/` | 9 284 464 | 1 | `icp_registered` | `postprocessing` | — | 6/6 |
| `blocks_deduped/` | 5 683 526 | 1 | `icp_registered` | `postprocessing` | — | 6/6 |
| `blocks_projected/` | 5 683 526 | 2 | `icp_registered` | `postprocessing` | `int32`, `[0,10711]` of 10 712 | 6/6 |
| `blocks_pca_calibrated/` | 5 683 526 | 2 | `pca_calibrated` | `postprocessing` | unchanged | 6/6 |
| `blocks_rf_centered/` | 5 683 526 | 2 | **`rf_centered`** | `postprocessing` | unchanged | 6/6 |

`reference_ply_vertex_count = 10712` and `dedup_epsilon = 0.5` are stamped from projection onward and
match the PLY actually on disk. This is the first end-to-end run whose terminal artifact declares
`rf_centered` — ST13-01 takes the documented no-cluster passthrough and correctly stops at
`pca_calibrated`.

**Depth preservation (9.3, 9.4).** Per block, `signed_depth_mm` bitwise unchanged (dtype included)
across all four transitions:

| block | rows in → out of dedup | dropped | ICP | projection | PCA | RF | `max\|depth\|` reg vs ded | ded vs CSV `contact_depth` |
|-------|----------------------|---------|-----|-----------|-----|----|----------------------|---------------------------|
| 01 | 647 169 → 423 111 | 34.6% | OK | OK | OK | OK | 0.000e+00 over 983 frames | 7.105e-15 |
| 02 | 2 175 484 → 1 413 058 | 35.0% | OK | OK | OK | OK | 0.000e+00 over 1 534 frames | 7.105e-15 |
| 04 | 1 987 640 → 1 086 204 | 45.4% | OK | OK | OK | OK | 0.000e+00 over 2 750 frames | 7.105e-15 |
| 05 | 3 632 693 → 2 273 340 | 37.4% | OK | OK | OK | OK | 0.000e+00 over 2 678 frames | 7.105e-15 |
| 06 | 338 901 → 202 167 | 40.3% | OK | OK | OK | OK | 0.000e+00 over 1 103 frames | 1.776e-15 |
| 08 | 502 577 → 285 646 | 43.2% | OK | OK | OK | OK | 0.000e+00 over 1 151 frames | 1.776e-15 |

The `contact_depth` residual is one ULP of the CSV's decimal round trip, matching Phase 6's
measurement. No frame was lost at any stage — the frame count is identical on both sides of dedup.

**The RF translating branch, end to end.** ST14-02's `rf_center_origin.json` reads
`status: "ok"`, `rf_center = [19.950000000000003, 355.0375, 400.3875]` (5 clusters found, dominant
cluster 59 points, mean selectivity 0.712). Every row of every block is translated by exactly
`-rf_center`: the per-axis shift is constant at `-19.9500 / -355.0375 / -400.3875` and
`max|shift − (−rf_center)|` is **3.05e-06 mm** across 5.68 M rows (1.72e-06 mm on the two smaller
blocks) — the float32 storage rounding of coordinates of that magnitude, nothing more. This is the
branch that Phase 7 could only reach with an injected permissive `SelectivityDBSCANConfig`.

**`vertex_id` resolution (9.5).**

| session | vertices | rows | `max\|PLY[vertex_id] − parquet x/y/z\|` | roundings between them |
|---------|---------|------|--------------------------------------|------------------------|
| ST14-02 (translating) | 10 712 | 5 683 526 | 0.09999771 mm | two (`np.round(...,1)` at PCA **and** at RF) |
| ST13-01 (passthrough) | 1 807 | 982 914 | 0.04999390 mm | one (PCA only; RF copies the PLY) |

Both sit just inside their bound and the two differ by exactly the factor the branch predicts, which
is the cross-check that the index and the coordinates agree rather than merely both existing.

**9.6 — byte-identity, the controlled experiment.** The pre-Phase-1 versions of all five stage
modules (`git show f6bba02:...`, the parent of `a5f64de`) were loaded by file path and driven over
ST14-02's real `blocks_filtered/` inputs into a separate output tree, in the **same interpreter and
the same library versions** as the new chain, and every artifact sha256-compared. The old
`deduplicate_xy_flow` lived in the workflow file and had different required kwargs, so its body was
replicated from `workflow_old.py` verbatim, minus the monitor branch the DAG disables and an
idempotency check `force_processing=True` bypasses anyway. **36/36 artifacts identical** — 31 CSVs
across the five stages, plus the three forearm PLYs, `pca-xyz_transformation-matrices.json` and
`rf_center_origin.json`. The last two matter most: the files that *define* Spaces 3 and 4 did not
move, so 7.7 holds on a second session.

**A confound found and attributed, not waved away.** The session's on-disk CSVs from the 2026-05-13
production run were hashed *before* this drive overwrote them. All 32 differ from the new output,
while the three PLYs and `rf_center_origin.json` are identical to May. Rather than assume, the
pre-change ICP module was re-run today over an **untouched** session (`2022-06-15_ST14-04`, whose
outputs are still the 2026-05-13 ones) and compared against its own May output: **0/3 reproduce it**,
with sizes differing by 18-51 bytes on files of 6-15 MB. The pre-change code therefore no longer
reproduces its own May bytes, so the drift predates this branch and cannot be caused by it —
consistent with a float-formatting change in a dependency between May and now (current env: pandas
2.3.3, numpy 2.2.6, scipy 1.15.2, scikit-learn 1.7.2, open3d 0.19.0). The controlled same-environment
old-vs-new comparison above is the evidence for 9.6; the May hashes are recorded only to explain why
they are not.

Independently, `git diff` over the five stage modules between the code state at the May run
(`f3de8b6`) and the pre-Phase-1 ref is **one import-path line** in `center_on_receptive_field.py`
(`analysis.receptive_field_mapping` → `postprocessing.receptive_field.rf_clustering`, moved verbatim
by `82e5083`), so the May outputs were produced by numerically the same stage code.

**9.7 — idempotency.** An unchanged re-run of the whole session skipped **every** task:
`fetch_forearm_of_reference`, `apply_icp_registration`, `deduplicate_xy`,
`project_contacts_onto_forearm` (all six blocks), `calibrate_pca_xyz`, `center_on_receptive_field`
and `aggregate_session`. Note this includes RF-centring: the Phase 8 finding that
`center_on_receptive_field` never skips is specific to its **no-cluster passthrough** branch, where
`shutil.copy2` propagates the spread of input mtimes. The translating branch writes its outputs
normally and skips correctly.

**9.8 — sidecar sizes per stage (MB).**

| stage | ST13-01 (4 blocks) | ST14-02 (6 blocks) | total |
|-------|-------------------|--------------------|-------|
| `blocks_filtered/` | 20.6 | 91.6 | 112.3 |
| `blocks_registered/` | 22.1 | 105.4 | 127.4 |
| `blocks_deduped/` | 10.2 | 63.6 | 73.8 |
| `blocks_projected/` | 11.3 | 71.3 | 82.7 |
| `blocks_pca_calibrated/` | 11.3 | 71.4 | 82.7 |
| `blocks_rf_centered/` | 11.3 | 71.4 | 82.7 |
| **total, 2 sessions** | 86.8 | 474.7 | **561.6** |

Registration *grows* the file (22.1 vs 20.6 MB) because the transform destroys the compressibility of
the quantised Space-1 coordinates; dedup then more than halves it; projection grows it again by
adding `vertex_id` and re-addressing coordinates to a small vertex set. The dataset-wide total is not
measured — 9 of the DAG's 11 sessions were not driven.

**Test suite:** 585 passed, 7 skipped — unchanged from the Phase 8 baseline, same seven skip reasons.

**Not verified in this phase**, and left unticked above where it applies:

- The other **9 of 11** sessions in the DAG's `kinect_configs`. Only ST13-01 and ST14-02 were driven,
  so the dataset-wide artifact total and "every block the DAG processes" are unproven at DAG scope.
- The **Testing Plan's Manual Verification and Edge Cases** sections. Three edge cases were exercised
  incidentally across Phases 5-8 (ICP passthrough, RF `no_cluster_found`, a block dropped by PCA
  gesture segmentation); "a frame whose every contact vertex is removed by dedup" and "a malformed
  `contact_points` cell" were not constructed on real data.
- The **render check** — no visual confirmation that the RF-centred field sits on the RF-centred
  forearm surface. The 0.09999771 mm `vertex_id` residual is the numerical stand-in for it.

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
