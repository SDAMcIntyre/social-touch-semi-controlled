# Plan: Filter Contact Depth Field by Neural Quality

**Date:** 2026-08-18
**Created:** 2026-08-18 10:00
**Approved:** —
**Completed:** —
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/per-vertex-contact-depth-poc`
**Branch:** `feature/filter-contact-depth-field-by-neural-quality`

---

## Overview

The per-vertex contact depth field sidecar is produced in preprocessing and stops there. This plan
adds one merging-stage task that applies the **same neural-quality trial exclusion** the merged CSV
already receives to the depth field, and writes the result into
`3_merged/<session>/blocks_filtered/` — the directory postprocessing reads from. No coordinate
transformation is involved; merging performs none.

## Problem Statement

`<video_stem>_contact_depth_field.parquet` currently exists only in
`2_processed/kinect/<session>/<block>/kinematics_analysis/`. Postprocessing resolves its inputs
**exclusively** from `3_merged/<session>/blocks_filtered/`
(`postprocess_workflow_kinect_auto.py:255-262`), so the sidecar is invisible to every stage
downstream of preprocessing. It cannot be transformed, aligned, or consumed.

This matters beyond convenience. The motivating requirement recorded in
`docs/development/brainstorms/per-vertex-contact-depth.md` is that the IFF weight on vertex *i* is
`w(depth_i) x RF_sensitivity(position_i)` — two fields multiplying — so the depth field must
ultimately be expressible in the RF-centred frame. It "cannot be a preprocessing-only debug
artefact". Reaching the merged tree is the first link in that chain, and the only one requiring no
scientific decisions.

**The task is a filter, not a file move.** `filter_by_neural_quality` discards trials the
experimenter marked unusable in `semicontrolled_data-collection_quality-check.xlsx`. Measured across
the 103 blocks that currently have both `blocks_merged/` and `blocks_filtered/` artifacts on disk:
**69 blocks are unchanged, 34 lose rows**, the worst (`ST13-03/block-order-08`) retaining only
**27.5%** of its original size, with twelve blocks below 42%. A depth field copied without that
exclusion applied would carry per-vertex data for neurally unusable trials into postprocessing, and
would break the invariant "every sidecar frame appears in the CSV" on a third of the dataset. The
exclusion is the substance of this task; relocating the file is the incidental part.

## Goals

### In Scope

1. A new merging DAG task `filter_contact_depth_field_by_neural_quality`, depending on
   `filter_by_neural_quality`, that applies that task's trial exclusion to the Space-1 depth field.
2. Row selection driven by the set of `frame_index` values surviving into the filtered merged CSV,
   so the two artifacts are excluded by exactly the same criterion with no reimplementation of the
   xlsx-parsing logic.
3. A DataFrame-level writer in the existing `contact_depth_field_io` module, so a table read from
   disk can be written back without reconstructing `ContactDepthFrame` objects.
4. Provenance metadata recording that the neural-quality filter was applied and how many frames it
   removed.
5. Idempotency consistent with the rest of the repo (`should_process_task` / `clean_task_outputs`).
6. **Prerequisite fix:** the merging DAG names a config group that does not exist, which aborts the
   pipeline at startup. See Phase 1.

### Out of Scope

- **All coordinate transformation.** ICP, dedup, projection, PCA calibration and RF-centring are the
  successor plan. The artifact this plan produces remains in **Kinect Space 1** and says so in its
  metadata.
- **Persisting `vertex_id`** (the parked "stable forearm-of-reference vertex ids" work). It belongs
  with projection, which is postprocessing.
- **The dedup reduction rule for depth.** Decided in principle — drop the row, survivor inherits the
  collapsed group's maximum magnitude — but implemented in the successor plan, not here.
- **Re-deriving the exclusion from the quality xlsx.** This task reads the surviving frames from the
  already-filtered CSV. Parsing the xlsx a second time would be a second source of truth.
- **Upsampling the depth field to the nerve sampling rate.** Merging scatters Kinect rows ~33x into a
  nerve-rate frame (`merge_neural_and_kinect_data.py:142-148`). Replicating that in a per-vertex
  table would multiply ~1 GB by ~33 for zero information gain. The field stays at Kinect frame rate
  and joins to anchor rows on `frame_index`.
- **A session-level aggregated parquet.** Whether the analysis repo wants one alongside
  `<session>_semicontrolled_aggregated_session.csv` is undecided; aggregation is a postprocessing
  task (`aggregate_session`) in any case.
- **Back-filling existing merged artifacts.** Per `completed/fix-contact-points-csv-corruption.md`,
  users re-run the pipeline.
- **Adding the depth field to the `input_paths` of `unify_dataset` or `filter_by_neural_quality`.**
  Regenerating a depth field must not invalidate merged CSVs that do not depend on it.
- **Fixing the `filter_by_neural_quality` staleness hole** (see Risks). Documented, not repaired here.
- **Reconciling the merging DAG's session list with the analysis repo's.** Merging covers 9 groups
  and omits `ST13-01/02/03`; that is a separate curation decision.

## Success Criteria

- [ ] `<session_id>_semicontrolled_<block_id>_contact_depth_field.parquet` is produced in
      `3_merged/<session_id>/blocks_filtered/` for every block the merging DAG processes.
- [ ] For every retained row, `x`, `y`, `z` and `signed_depth_mm` are **bitwise identical** to the
      Space-1 sidecar (`np.array_equal`), and dtypes are unchanged. This task removes rows; it must
      not alter a single surviving value.
- [ ] The `frame_index` set of the output equals the Space-1 frame set intersected with the filtered
      CSV's non-NaN `frame_index` set. No frame in one and not the other.
- [ ] On a block with no Not2Use trials, output row count equals input row count exactly.
- [ ] On a block with Not2Use trials, output row count < input row count, and every removed
      `frame_index` is absent from the filtered CSV.
- [ ] File metadata declares `coordinate_space = "kinect_space_1"` (unchanged — no transform is
      applied), retains the original `source_recording` and `produced_by`, and adds
      `pipeline_stage = "merging"`, `neural_quality_filtered = "true"` and `frames_dropped`.
- [ ] `read_contact_depth_field()` reads the filtered file unmodified, and rejects it if
      `schema_version` is unknown.
- [ ] Deleting only the filtered parquet and re-running regenerates it; re-running unchanged skips.
- [ ] `clean_task_outputs` removes it, so a failed run leaves no partial artifact.
- [ ] The merging pipeline starts and completes on at least one real session (currently impossible —
      see Phase 1).

## Definitions

- **Neural-quality filtering**: the trial exclusion performed by `filter_by_neural_quality` —
  truncation from the first Not2Use trial onward when `discard_from_first_not2use: true`
  (`filter_merged_by_neural_quality.py:149-155`), otherwise boolean removal of individual Not2Use
  trial rows (`:169-171`). This task applies **the outcome** of that exclusion, never its logic.
- **Surviving frame**: a `frame_index` value present and non-NaN in that block's
  `blocks_filtered/*_merged_data.csv`. NaN `frame_index` rows are the ~32 of every 33 interpolated
  nerve-rate rows and carry no Kinect data; they are not frames.
- **Retained row**: a Space-1 depth-field row whose `frame_index` is a surviving frame.
- **Value-preserving**: no column of a retained row is recomputed, rescaled, rounded, or reordered.
  Testable as `np.array_equal` on all four numeric columns.
- **Space 1 / `kinect_space_1`**: the raw Kinect frame in mm, before ICP registration, PCA
  calibration and RF-centring. Unchanged by this plan.
- **Empty result**: zero retained rows. An error, not a valid artifact — see Error Boundaries.

---

## Technical Design

### Approach

One new filter in the pipe-and-filter chain, after `filter_by_neural_quality` and before
postprocessing, with an explicit input/output contract and no hidden state.

```
[preprocess]  <video_stem>_contact_depth_field.parquet          (Space 1, all frames)
                              |
[merging]     filter_contact_depth_field_by_neural_quality
                  reads blocks_filtered/*.csv for the surviving frame set
                              |
              blocks_filtered/<session>_semicontrolled_<block>_contact_depth_field.parquet
                              |
[postprocess] (successor plan: ICP -> dedup -> projection -> PCA -> RF-centring)
```

The exclusion criterion is taken from the already-filtered CSV rather than re-derived from the
quality xlsx. That keeps one source of truth: if `filter_by_neural_quality` changes its rule, or its
`discard_from_first_not2use` option is flipped, the depth field follows automatically with no
matching change here.

Both input paths and the output path resolve directly from `KinectConfig`
(`video_processed_output_dir`, `session_merged_output_dir`, `session_id`, `block_id`), so the task
needs no filename regex and no `block-order-01` / `block-order01` reconciliation — the two spellings
are each available as a config attribute. This satisfies the rule that a skip decision be resolvable
from path arithmetic and `stat` alone (`completed/fast-skip-check-extraction-pipeline.md`).

The surviving frame set is read with `usecols=["frame_index"]`; the filtered CSV reaches 112 MB on
the largest block and must not be loaded whole.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **New merging task applying the CSV's surviving frame set to the depth field** | One source of truth for the exclusion; everything postprocessing needs in one directory; participates in DAG, idempotency and cleanup like every other task | One more task to maintain; duplicates ~1 GB across the dataset | **Chosen** |
| Straight `shutil.copy2`, no filtering | Trivial; no CSV read | Measured: 34/103 blocks would carry depth data for neurally unusable trials, worst case ~72% surplus. Breaks a success criterion of the producing plan and hands postprocessing rows it must silently ignore | Rejected |
| Re-read the quality xlsx and re-derive the exclusion | Independent of CSV state | Two implementations of one rule, guaranteed to diverge; would have to duplicate the `discard_from_first_not2use` branch and the unit/block regex parsing | Rejected |
| No merging task; postprocessing reaches back into `2_processed/` | No duplication; no new artifact | Postprocessing would gain knowledge of the preprocessing tree layout, crossing a stage boundary the repo keeps clean, and would still have to apply the neural-quality exclusion itself — in the wrong stage | Rejected |
| Merge the depth field into the merged CSV as columns | One artifact | Ragged per-vertex data cannot be one row per frame. Exactly the pathology that produced the `contact_points` string blob, explicitly prohibited by `note-artifact-serialization-formats.md:51-55` | Rejected |
| Upsample the depth field to nerve rate alongside the CSV | Row-aligned with the merged CSV | ~33x on ~1 GB for no information; the field is not interpolable (`contact_depth_field_io.py:43-50`), so added rows could only be NaN | Rejected |
| Filter against `blocks_merged/` instead of `blocks_filtered/` | Slightly simpler dependency | Would reintroduce exactly the trials the previous task removed | Rejected |

### Architecture & Module Contracts

| Module / layer | Responsibility | Inputs → Outputs | Must NOT know about |
|----------------|----------------|------------------|---------------------|
| `contact_depth_field_io.write_contact_depth_field_table` (new) | Serialise a schema-conforming DataFrame with caller-supplied metadata | `(DataFrame, Path, metadata: dict[str,str])` → parquet file | Sessions, configs, DAGs, Prefect, the CSV, neural quality, why rows were selected |
| `filter_contact_depth_field_by_neural_quality.py` (new) | Determine surviving frames from the filtered CSV and write the reduced depth field | `(depth_field_path, filtered_csv_path, output_path)` → parquet file | Prefect, `KinectConfig`, parquet internals, the quality xlsx, how the field was computed |
| `filter_contact_depth_field_by_neural_quality_flow` (new, in `merging_pipeline_neuron_to_kinect_auto.py`) | Path arithmetic from config | `KinectConfig` → three paths | Serialisation, filtering semantics |
| `filter_merged_by_neural_quality.py` | unchanged | — | — |

```
code/src/preprocessing/motion_analysis/tactile_quantification/io/
└── contact_depth_field_io.py                          # MODIFIED — add write_contact_depth_field_table()

code/scripts/_4_merging/
└── filter_contact_depth_field_by_neural_quality.py    # NEW — the filter

code/scripts/
└── merging_pipeline_neuron_to_kinect_auto.py          # MODIFIED — flow + stage registration

configs/
└── merging_pipeline_neuron_to_kinect_auto_dag.yaml    # MODIFIED — new task + ST15-01 fix
```

Target signatures:

```python
def write_contact_depth_field_table(
    table: pd.DataFrame,
    output_path: Path,
    *,
    metadata: Dict[str, str],
) -> None:
    """Write a schema-conforming long-form table with explicit file metadata.

    Validates columns, order and dtypes against COLUMN_DTYPES before writing;
    raises on mismatch rather than coercing. Overwrites wholesale; never appends.
    """

def filter_contact_depth_field_by_neural_quality(
    depth_field_path: Path,
    filtered_csv_path: Path,
    output_path: Path,
    *,
    force_processing: bool = False,
) -> Optional[Path]:
    """Reduce the Space-1 depth field to the frames that survived neural-quality filtering."""
```

### Constraints from the knowledge base

- **Precision is a schema decision** (`note-artifact-serialization-formats.md`). This task must not
  round, downcast or re-quantise. `signed_depth_mm` stays float64 so the frame-level invariant
  `max(|signed_depth_mm|) == contact_depth` stays bit-exact. Any test comparing against a CSV must
  use `float_precision="round_trip"`.
- **Never align by coordinate value** (`contact-depth-field-sidecar.md:274-276`). Row selection is by
  `frame_index` membership only. `contact_points` is quantised to `%.1f` and must never be used to
  match rows.
- **The field is not forward-fillable or interpolable** (`bug-rf-explorer-nearest-vertex-distance.md`;
  cubic interpolation measured overshooting 29 mm off-surface). This is the substantive reason the
  field is not upsampled with the CSV, and it belongs in the new module docstring.
- **Zero vs absent must not collapse** (`contact-depth-field-sidecar.md`). A block whose every frame
  is excluded is *absent*, and must raise rather than write a zero-row file that a later run would
  treat as valid and complete.
- **Subfolder-per-step, no shadow columns** (`completed/postprocessing-column-consolidation.md`). The
  filtered file lives beside the CSV it belongs to and adds no suffixed column family.
- **New artifact, new task's `output_paths`** — listed there and paired with `clean_task_outputs`.
  The first-run trap does not apply here (the task itself is new), but the pairing still does.
- **CuPy import order** — the new script imports no geometry engine and is not a workflow entry
  point, so the constraint should not bite. Confirm during implementation; do not assume.

### Error boundaries (fail-fast, per CLAUDE.md)

| Condition | Behaviour |
|-----------|-----------|
| Space-1 depth field missing | Raise via `should_process_task`'s missing-input path. The block was never processed by `compute_somatosensory_characteristics`; a broken dependency graph, not a skip. |
| Filtered CSV missing | Same. |
| Filtered CSV has no `frame_index` column | Raise, naming the file and listing the columns found. |
| Zero retained rows (every trial Not2Use) | Raise. An empty artifact is indistinguishable from a complete one on the next run. |
| Depth field contains a frame absent from the CSV | Expected and normal — that is what the filter removes. Counted, reported, not raised. |
| CSV has a `contact_detected` frame absent from the depth field | Raise. The two artifacts disagree about the same recording; one is stale. |

---

## Implementation Plan

### Phase 1: Unblock the merging pipeline
**Goal:** The merging DAG can start at all.
**Started:** 2026-08-18 00:00  **Completed:** 2026-08-18 00:30

The DAG names `valid_configs_ST15-01`, but that directory was renamed to
`_NOT_ENOUGH_DATA_valid_configs_ST15-01`. `resolve_session_configs` raises `FileNotFoundError` on any
entry that does not resolve (`session_config_resolver.py:53-56`), so the pipeline aborts before task
one. Nothing in this plan can be tested until this is fixed.

- [x] 1.1 — Remove `valid_configs_ST15-01` from `parameters.kinect_configs`.
- [x] 1.2 — Confirm the remaining 8 group names all resolve; record the resulting block count.
      Confirmed: all 8 remaining groups (`valid_configs_ST14-01`, `-02`, `-04`, `valid_configs_ST16-02`,
      `-03`, `-05`, `valid_configs_ST18-01`, `-04`) resolve under `configs/kinect_configs/`. Total
      block YAML files across the 8 groups: **77** (ST14-01: 9, ST14-02: 6, ST14-04: 3, ST16-02: 15,
      ST16-03: 5, ST16-05: 16, ST18-01: 16, ST18-04: 7).
- [x] 1.3 — Run the merging pipeline unchanged on one session to establish a working baseline
      *before* adding anything. Per the task's stated acceptance bar, verified directly:
      `resolve_session_configs()` called with the DAG's (corrected) `kinect_configs` list and
      `configs/kinect_configs/` as root returns 77 resolved paths with no exception. A fuller
      Prefect-flow run was not attempted — the environment's shared Prefect database
      (`~/.prefect/prefect.db`) is incompatible with this env's prefect version and is documented as
      out of scope to fix; the resolver-level check is the acceptance bar for this phase and is
      sufficient to prove the DAG can now start.

**Files Modified:**
- `configs/merging_pipeline_neuron_to_kinect_auto_dag.yaml` — drop the non-existent group

**Dependencies:** None

### Phase 2: The DataFrame writer
**Goal:** A table read from disk can be written back with explicit metadata, unit-tested in isolation.
**Started:** 2026-08-18 11:00  **Completed:** 2026-08-18 11:40

- [x] 2.1 — Add `write_contact_depth_field_table(table, output_path, *, metadata)` to
      `contact_depth_field_io.py`, reusing `_SCHEMA_FIELDS` and the temp-file-plus-`os.replace`
      write already used by `write_contact_depth_field`. The atomic write was factored out into
      `_write_arrow_table_atomically()`, now shared by both writers; `write_contact_depth_field`'s
      signature and behaviour are unchanged. The arrow table is built column-by-column against
      `_SCHEMA_FIELDS` rather than via `Table.from_pandas`, so the DataFrame index and pandas'
      own schema metadata cannot leak in alongside the caller's metadata.
- [x] 2.2 — Validate the incoming DataFrame against `COLUMN_DTYPES` — names, order and dtypes — and
      raise on mismatch rather than coercing silently. `_validate_table_against_schema()`
      distinguishes a missing/unexpected column (names it) from a mis-ordered one (says so), and
      names the offending column and both dtypes on a dtype mismatch. Empty and `None` tables are
      refused here too.
- [x] 2.3 — Require `schema_version` in the supplied metadata and reject an unknown value at write
      time, mirroring the reader. `_validate_supplied_metadata()` also refuses non-`str` keys or
      values, so `frames_dropped=17` must be stringified by the caller deliberately rather than by
      this writer guessing a format.
- [x] 2.4 — Extend the module docstring: two writers now exist; neither knows why rows were selected.
      New "Two writers, one schema" section; the purity contract is intact and now also names
      schema-conforming DataFrames.
- [x] 2.5 — Tests: round-trip a reduced table; dtype preservation; metadata round-trip including the
      new keys; wrong column order raises; wrong dtype raises; empty table raises. 19 tests added in
      a new section 6 of `test_contact_depth_field_io.py`; existing tests untouched apart from the
      import list and the renumbering of the integration section to 7. Bit-identity is asserted with
      `np.array_equal` on all six columns, and the float32-depth guard proves the assertion is not
      vacuous.

`write_contact_depth_field_table` is added to the module's `__all__`. It is deliberately **not**
re-exported from `io/__init__.py`: that package's docstring states it is "deliberately empty of
re-exports" so no caller drags in a serialisation dependency it does not need, and
`write_contact_depth_field` is not exported there either.

**Files Modified:**
- `code/src/preprocessing/motion_analysis/tactile_quantification/io/contact_depth_field_io.py`
- `code/tests/test_contact_depth_field_io.py` — extend

**Dependencies:** None (independent of Phase 1)

### Phase 3: The neural-quality filter
**Goal:** A pure, testable function that produces the reduced depth field.
**Started:** 2026-08-18 12:00  **Completed:** 2026-08-18 12:50

- [x] 3.1 — New `code/scripts/_4_merging/filter_contact_depth_field_by_neural_quality.py` with the
      signature above. Returns `output_path` when it wrote, `None` when the output was already
      up to date — the `Optional` in the signature is the skip case, matching
      `align_and_merge_neural_and_kinect`. It imports neither Prefect nor `KinectConfig`, and the
      only preprocessing import is the `contact_depth_field_io` leaf (numpy/pandas/pyarrow only),
      so the CuPy import-order constraint does not bite: verified that
      `merging_pipeline_neuron_to_kinect_auto.py` imports no CuPy at all.
- [x] 3.2 — `should_process_task(input_paths=[depth_field_path, filtered_csv_path], output_paths=[output_path], force=force_processing)`
      then `clean_task_outputs([output_path])`, in that order, as the sibling merging tasks do.
- [x] 3.3 — The CSV header is read first (`nrows=0`) so a missing column can be reported by name
      against the columns actually found; the body is then read with
      `usecols=["frame_index", "contact_detected"]`. `contact_detected` is needed for the
      stale-artifact check of 3.5 and costs nothing extra in the same pass. NaN `frame_index` rows
      are dropped before anything else, so the NaN `contact_detected` they also carry is never
      examined.
- [x] 3.4 — Selection is `np.isin` on `frame_index` alone — never a coordinate value. The input
      columns are copied out of the DataFrame immediately after the read, so the bit-identity
      assertion compares the written rows against what came off disk rather than against a view of
      themselves; dtype and `np.array_equal` are both checked, for all six columns.
- [x] 3.5 — Every row implemented: missing input raises through `should_process_task`; a missing
      `frame_index` or `contact_detected` column raises naming the file and listing the columns
      found; zero retained rows raises; a CSV contact frame absent from the sidecar raises as a
      staleness disagreement; a sidecar frame absent from the CSV is the filter working and is
      counted into `frames_dropped` and logged. Two further violations raise rather than being
      papered over: a non-integral `frame_index`, and a `contact_detected` that is missing or
      non-numeric on a frame row (treating it as "no contact" would silently weaken the
      staleness check).
- [x] 3.6 — `CARRIED_METADATA_KEYS` carries the six through verbatim — `coordinate_space` included
      rather than restated, since re-deriving it here is how the declared space would drift from the
      actual one — and raises if the input omits any of them. `frames_dropped` counts frames, not
      rows, and is stringified at the call site as Phase 2's writer requires.
- [x] 3.7 — Exported from `code/scripts/_4_merging/__init__.py`.
- [x] 3.8 — 17 tests in `code/tests/test_filter_contact_depth_field_by_neural_quality.py`, all on
      synthetic `tmp_path` fixtures. The CSV fixture reproduces the real shape — one anchor row per
      frame followed by interpolated rows that are NaN in *every* Kinect column, `contact_detected`
      included. Bit-identity is asserted on raw bytes (`.tobytes()`), not `np.allclose`.

**Files Modified:**
- `code/scripts/_4_merging/filter_contact_depth_field_by_neural_quality.py` — new
- `code/scripts/_4_merging/__init__.py` — export
- `code/tests/test_filter_contact_depth_field_by_neural_quality.py` — new

**Dependencies:** Phase 2

### Phase 4: Pipeline wiring
**Goal:** The task runs as part of the merging DAG.
**Started:** 2026-08-18 13:10  **Completed:** 2026-08-18 13:45

- [x] 4.1 — `filter_contact_depth_field_by_neural_quality_flow(...)` added to
      `merging_pipeline_neuron_to_kinect_auto.py`, decorated `@task(name="12. Filter Contact Depth
      Field by Neural Quality")` — the file's two existing "flow" wrappers (`unify_dataset`,
      `filter_by_neural_quality_flow`) are `@task`-decorated with numbered names, so the new one
      follows that, not the `@flow` used for the two orchestrating entry points. The wrapper takes
      the three resolved paths, not the config: **all** path arithmetic went into the existing
      `resolve_filenames` helper, which the file already uses to centralise naming conventions.
      Three keys added there: `depth_field_path`, `filtered_csv_path`, `depth_field_output_path`.
      The two block-id spellings are each taken straight from the config attribute that carries them
      (`source_video.stem` → `block-order02`, `block_id` → `block-order-02`); nothing converts
      between them. `filtered_csv_path` also replaces the inline derivation that
      `filter_by_neural_quality` was doing in `run_single_session_pipeline`, so the filtered CSV path
      is now computed in exactly one place and the two tasks cannot drift apart.
- [x] 4.2 — Registered in `run_single_session_pipeline` immediately after `filter_by_neural_quality`,
      inside the same `try` block, using the identical `dag_handler.can_run(...)` →
      `get_task_options(...)` → `options.get('force_processing', False)` → call →
      `dag_handler.mark_completed(...)` sequence as its two siblings. No extra existence checks are
      added: the task's own `should_process_task` raises on a missing input, which is the documented
      error boundary.
- [x] 4.3 — DAG entry added after `filter_by_neural_quality`, with a four-line comment in the style of
      the neighbouring entries. The file was edited as text, so comments, key order and the flow-style
      `kinect_configs` sequence are untouched. The comment is deliberately ASCII-only:
      `DagConfigHandler` opens the YAML with `open(path, 'r')` and no explicit encoding, so a UTF-8
      em-dash would be silently mojibaked under the Windows default code page.
- [x] 4.4 — Verified directly against `DagConfigHandler` (no Prefect run needed). With
      `enabled: false` the task's `can_run` returns `False` and nothing raises, so the stage is
      skipped and the flow continues to its success return; with the entry deleted entirely it also
      returns `False`, printing the handler's "not found in DAG configuration" warning. Also
      asserted: `can_run` is `False` while `filter_by_neural_quality` is uncompleted and `True` once
      it is, and the three resolved paths match the spec exactly for a synthetic config.

**Files Modified:**
- `code/scripts/merging_pipeline_neuron_to_kinect_auto.py` — flow + stage registration
- `configs/merging_pipeline_neuron_to_kinect_auto_dag.yaml` — new task entry

**Dependencies:** Phase 3

### Phase 5: Verification on real data
**Goal:** Measured evidence, not argument.
**Started:** 2026-08-18 14:00  **Completed:** 2026-08-18 14:35

- [x] 5.1 — Run on `ST14-01`. All 9 of its blocks hold both a Space-1 sidecar and a
      `blocks_filtered/` CSV, and all 9 were processed: 4 lose no frame at all, 5 lose frames
      (`block-order-03` retains 30.0% of its rows, `block-order-10` 33.3%, `block-order-02` 35.4%).
      The plan's earlier percentages were *CSV byte* ratios; the numbers here are depth-field **row**
      ratios, which is why they differ by a few points. The run was then extended to all 77 blocks of
      the eight merging-DAG config groups — every one succeeded.
- [x] 5.2 — Bitwise equality asserted **independently of the task's own internal check**: the output
      parquet was re-read and compared against the Space-1 parquet reduced by the CSV frame set, using
      `np.array_equal` *and* a raw `.tobytes()` comparison, on all six columns (`frame_index`,
      `time_s`, `x`, `y`, `z`, `signed_depth_mm`), with dtypes compared too. 77/77 blocks pass on
      every column; no approximate comparison was used anywhere.
- [x] 5.3 — Frame-set agreement confirmed in both directions on all 77 blocks: the output frame set
      equals `Space-1 frames ∩ CSV non-NaN frames` exactly (zero extra, zero missing), no output frame
      is absent from the CSV, and **zero** CSV frames marked `contact_detected` are absent from the
      output. The 66,753 CSV frames dataset-wide that are absent from the output are the non-contact
      frames the sidecar never contained.
- [x] 5.4 — Sizes recorded per block and per session (tables below). Dataset-wide over the 77 DAG
      blocks: **971.5 MB Space-1 → 809.5 MB filtered (−16.7%)**. The plan's "1,030 MB baseline"
      re-measured as 1,082.7 MB / 1,032.5 MiB across all **99** blocks on disk — the 22-block gap is
      the sessions the merging DAG does not cover (`ST13-*` etc.), which therefore produce no filtered
      copy at all.
- [x] 5.5 — `ST14-01/block-order-05`'s filtered parquet deleted, re-run regenerated it
      **byte-identical** (1,008,080 bytes, md5 `e3cfe159f1ba0ce1b7520d54fb40c0c8` before and after);
      an immediate second call returned `None` and left mtime and md5 untouched. A whole-set re-run of
      all 77 blocks returned `None` 77/77 and changed 0 mtimes.

**Files Modified:** none (verification only)

**Dependencies:** Phase 4, Phase 1

#### Verification Results (2026-08-18)

**How it was run.** Option A: a scratchpad driver (not committed) that imports `resolve_filenames`
from `merging_pipeline_neuron_to_kinect_auto.py` and feeds it a `KinectConfig` built exactly as
`run_batch_processing` builds one (`KinectConfigFileHandler.load_and_resolve_config` →
`KinectConfig(config_data=..., database_path=<project data root>)`), then calls the Phase-3 function
per block. Path arithmetic is therefore the *real* Phase-4 arithmetic, not a re-implementation.
Prefect was deliberately not driven (the shared `~/.prefect/prefect.db` is incompatible with this
environment's Prefect, as recorded in Phase 1.3), so **the DAG task wrapper and `dag_handler` gating
were not exercised on real data here** — they were verified directly against `DagConfigHandler` in
Phase 4.4. Every CSV used for comparison was read with `float_precision="round_trip"`.

**ST14-01 — per block** (rows/frames from the parquets; `keep%` is retained rows / Space-1 rows):

| Block | S1 rows | Retained rows | S1 frames | Retained frames | Frames dropped | Out MB | S1 MB | keep% | Bitwise | Frame-set |
|-------|--------:|--------------:|----------:|----------------:|---------------:|-------:|------:|------:|:-------:|:---------:|
| block-order-01 |   251,434 |   125,497 | 1,208 |   753 |   455 |  1.32 |  2.46 |  49.9% | PASS | PASS |
| block-order-02 | 1,502,134 |   531,627 | 1,425 |   602 |   823 |  5.40 | 15.27 |  35.4% | PASS | PASS |
| block-order-03 |   858,856 |   257,248 | 2,668 |   934 | 1,734 |  2.48 |  8.12 |  30.0% | PASS | PASS |
| block-order-05 |    99,658 |    99,658 |   784 |   784 |     0 |  1.01 |  1.01 | 100.0% | PASS | PASS |
| block-order-06 |   544,290 |   544,290 | 1,052 | 1,052 |     0 |  4.78 |  4.78 | 100.0% | PASS | PASS |
| block-order-07 |   178,967 |   178,967 | 1,108 | 1,108 |     0 |  1.66 |  1.66 | 100.0% | PASS | PASS |
| block-order-08 | 1,021,821 | 1,021,821 | 1,011 | 1,011 |     0 |  9.61 |  9.61 | 100.0% | PASS | PASS |
| block-order-09 |   963,164 |   382,400 | 1,454 |   654 |   800 |  3.79 |  9.27 |  39.7% | PASS | PASS |
| block-order-10 | 2,018,873 |   672,789 | 2,761 |   950 | 1,811 |  6.62 | 20.01 |  33.3% | PASS | PASS |
| **ST14-01 total** | **7,439,197** | **3,814,297** | | | | **36.7** | **72.2** | **51.3%** | 9/9 | 9/9 |

Metadata checked on every block: `coordinate_space` still `kinect_space_1`, `units`,
`sign_convention`, `source_recording`, `produced_by` and `schema_version` carried through verbatim,
plus `pipeline_stage="merging"`, `neural_quality_filtered="true"` and a `frames_dropped` that matches
the independently recomputed count. 77/77 pass.

**All eight merging-DAG config groups — per session:**

| Session | Blocks | Blocks losing frames | S1 MB | Filtered MB | S1 rows | Retained rows |
|---------|-------:|---------------------:|------:|------------:|--------:|--------------:|
| 2022-06-15_ST14-01 |  9 | 5 |  72.2 |  36.7 |  7,439,197 |  3,814,297 |
| 2022-06-15_ST14-02 |  6 | 2 |  98.5 |  91.6 |  9,960,012 |  9,284,464 |
| 2022-06-15_ST14-04 |  3 | 3 |  23.3 |  11.6 |  2,380,133 |  1,183,848 |
| 2022-06-17_ST16-02 | 15 | 7 | 228.3 | 158.8 | 22,680,274 | 15,853,953 |
| 2022-06-17_ST16-03 |  5 | 2 |  24.0 |  18.7 |  2,598,770 |  2,022,298 |
| 2022-06-17_ST16-05 | 16 | 1 | 287.5 | 276.1 | 28,671,871 | 27,575,817 |
| 2022-06-22_ST18-01 | 16 | 2 | 170.4 | 161.8 | 19,770,270 | 18,773,020 |
| 2022-06-22_ST18-04 |  7 | 3 |  67.2 |  54.1 |  6,686,427 |  5,366,840 |
| **Total** | **77** | **25** | **971.5** | **809.5** | **100,186,954** | **83,874,537** |

**Dataset-level totals (5.4).**

- 77/77 blocks processed; **0 failures, 0 skipped for missing input** — every block of the eight
  groups had both a Space-1 sidecar and a `blocks_filtered/` CSV.
- Space-1 across the 77 DAG blocks: **971.5 MB**. Filtered output: **809.5 MB** — a **16.7%**
  reduction on disk.
- Rows: 100,186,954 → 83,874,537 (**83.7% retained**). Frames: 113,685 → 93,600 (**82.3% retained**).
- 52 blocks lose no frame (output row-count-identical); 25 lose frames; worst retention
  `ST14-01/block-order-03` at **30.0%** of rows.
- Whole-dataset baseline re-measured: 99 Space-1 sidecars totalling **1,082.7 MB (1,032.5 MiB)** —
  the plan's "1,030 MB" figure was MiB, and is confirmed. Only 77 of those 99 blocks are covered by
  the merging DAG, so 22 blocks (~111 MB of Space-1 data) currently gain no filtered copy; that is
  the already-documented session-list divergence, not a fault of this task.

**Idempotency (5.5).** Delete → re-run → byte-identical regeneration (same size, same md5);
re-run again → `None` returned, nothing written, mtime and md5 unchanged. Extended to the whole set:
77/77 returned `None` on an unchanged re-run and 0 files were rewritten.

**Manual read with no repo import.** `pyarrow.parquet.read_metadata` alone on a filtered file returns
`schema_version=1`, `coordinate_space=kinect_space_1`, `units=mm`,
`sign_convention=negative_is_penetrating`, `source_recording`, `produced_by`,
`pipeline_stage=merging`, `neural_quality_filtered=true`, `frames_dropped`, `source_artifact`; the
schema reads back as `frame_index:int32, time_s:double, x:float, y:float, z:float,
signed_depth_mm:double`. All 77 `blocks_filtered/` directories now hold the CSV and the parquet side
by side.

**Not verified in this phase (stated plainly):** the Prefect flow itself was not executed, so the
`@task` wrapper, the DAG `can_run`/`mark_completed` gating and `force_processing` plumbing remain
verified only by the direct `DagConfigHandler` checks of Phase 4.4; and no block in the eight groups
exercised the zero-retained-rows error path on real data (it is covered by unit tests only).

---

## Testing Plan

### Unit Tests
- [x] `write_contact_depth_field_table` round-trips a table with dtypes preserved.
- [x] Wrong column order raises; wrong dtype raises; empty table raises.
- [x] Metadata round-trips, including `pipeline_stage`, `neural_quality_filtered`, `frames_dropped`.
- [x] Writing with an unknown `schema_version` raises.
- [x] Surviving set equal to all frames returns a row-identical table.
- [x] Partial surviving set removes exactly the complement.
- [x] NaN `frame_index` values in the CSV are excluded from the surviving set.
- [x] Zero retained rows raises.
- [x] A `contact_detected` frame in the CSV but absent from the depth field raises.

### Integration Tests
- [x] Against a real block: retained `x/y/z/signed_depth_mm` are `np.array_equal` to Space 1 —
      verified on 77 real blocks, plus `frame_index`/`time_s`, plus a raw-bytes comparison (Phase 5.2).
- [x] Against a real block: output frame set equals CSV non-NaN frame set intersected with Space 1 —
      77/77, both directions, with zero CSV contact frames missing from the output (Phase 5.3).
- [x] A no-Not2Use block produces a row-count-identical output — 52 of the 77 blocks dropped zero
      frames and are row-identical (e.g. `ST14-01/block-order-05`, 99,658 rows in and out).
- [x] `should_process_task` returns True when only the filtered parquet is missing, False on a clean
      re-run — asserted directly, so it runs without a recording.
- [x] `clean_task_outputs` removes the filtered parquet.

### Manual Verification
- [x] Open a filtered parquet in pandas with no repo import; confirm metadata alone states units,
      space, sign convention, and that the neural-quality filter was applied — done with
      `pyarrow.parquet` alone (pandas' own parquet backend); all keys present (Phase 5).
- [x] Confirm `blocks_filtered/` contains the CSV and the parquet side by side for every block —
      77/77 of the merging DAG's blocks. (The 22 blocks outside the DAG's config groups have no
      filtered parquet, by design of the session list.)
- [ ] Re-run the merging pipeline unchanged; confirm the task skips and writes nothing. **Partially
      done:** the task *function* skipped on all 77 blocks (returned `None`, 0 mtimes changed), but
      the Prefect flow itself was not run — the shared Prefect database is incompatible with this
      environment, as recorded in Phase 1.3. Left unticked deliberately.

### Edge Cases
- [ ] Block with zero contact frames throughout — the Space-1 file does not exist (the producer
      raises rather than writing an empty file), so this surfaces as a missing input.
- [x] Block where every trial is Not2Use — zero retained rows, must raise.
- [ ] Block where `filter_by_neural_quality` took the no-Not2Use `shutil.copy2` branch — output must
      still be produced, row-identical.
- [ ] Windows-locked output file during `clean_task_outputs` (warns and continues by design).
- [ ] `PermissionError` mid-write leaves no half-file a later run would accept (temp-file-plus-
      `os.replace` already guarantees this).

---

## Documentation Plan

- [x] Module docstring in `filter_contact_depth_field_by_neural_quality.py`: what the filter removes
      and why, why the exclusion is read from the CSV rather than the xlsx, why the field is not
      upsampled, why selection is by `frame_index` only.
- [ ] Changelog entry `docs/changelogs/filter-contact-depth-field-by-neural-quality.md`.
- [ ] Update `docs/development/plans/active/contact-depth-field-sidecar.md` Out-of-Scope to point here.
- [ ] **Correct `note-spatial-alignment-pipeline.md`** — it is stale: it lists
      `set_xyz_reference_from_gestures` before projection, names directories that no longer exist
      (`blocks_registered_deduped/`, `blocks_registered_projected/`) and a task
      (`export_forearm_pca_calibrated`) that no longer exists. The live order is
      dedup -> project -> PCA. Cheap to fix, and the successor plan depends on the corrected version.
- [ ] Note in the knowledge base that `blocks_filtered/` is now a two-artifact directory.
- [ ] **Not** updating CLAUDE.md — no architectural boundary moves.

---

## Rollback Plan

1. **Purely additive artifact.** Nothing consumes the filtered parquet yet; deleting it has no
   downstream effect.
2. Phase 4 alone: revert the flow + DAG commit. The task stops running; merged CSVs are untouched by
   construction — this task never writes a CSV.
3. Phase 3 alone: delete the script and its tests.
4. Phase 2 alone: revert the io module addition. `write_contact_depth_field` is unchanged, so the
   producing task is unaffected.
5. Phase 1 is an independent bug fix and should be kept regardless.
6. **Data considerations:** no migration. No existing artifact is read-modified or invalidated. The
   filtered parquets can be deleted wholesale with a glob.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Merging pipeline cannot run at all (`ST15-01` group missing) | **Certain** — verified | High | Phase 1 fixes it first; Phase 1.3 establishes a working baseline before anything is added |
| `filter_by_neural_quality` staleness hole: its no-Not2Use branch copies gated only on existence (`filter_merged_by_neural_quality.py:119-125`), so a regenerated input does not refresh it | Med | Med | Out of scope to fix, but this task takes the *filtered* CSV as an `input_paths` entry, so a stale CSV cannot silently yield a fresh-looking depth field. Document it |
| Depth field and CSV disagree because one is stale | Med | High | CSV-contact-frame-not-in-depth-field raises rather than producing a quietly incomplete artifact |
| Duplicating ~1 GB across the dataset | High | Low | Measured 1,030 MB in Space 1 across 99 blocks; the filtered copy is strictly smaller. Disk is not the constraint |
| Silent dtype coercion by pandas/pyarrow on round-trip | Low | High | Writer validates against `COLUMN_DTYPES` and raises; a round-trip test asserts bit-identity |
| Reading the ~112 MB filtered CSV per block is slow | Med | Low | `usecols=["frame_index"]` |
| Depth field regenerated later, filtered copy goes stale | Med | Med | It is in the task's `input_paths`, so `should_process_task` catches it by mtime |
| Merging DAG's session list diverges from the analysis repo's (9 vs 11 groups; `ST13-*` absent) | High | Low | Explicitly out of scope; flagged so it is not mistaken for an oversight |
| `block-order-01` vs `block-order01` spelling mismatch | Low | Med | Both spellings come from `KinectConfig` attributes; no string surgery in this task |
| Task name implies it only relocates a file | — | Med | Named `filter_contact_depth_field_by_neural_quality` precisely so the exclusion, not the copy, is what the name states |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 — unblock merging | ~1 LOC + one baseline run | None |
| Phase 2 — DataFrame writer | ~70 LOC + ~90 test | None |
| Phase 3 — neural-quality filter | ~130 LOC + ~150 test | Phase 2 |
| Phase 4 — pipeline wiring | ~45 LOC | Phase 3 |
| Phase 5 — verification | measurement only | Phases 1, 4 |

---

## References

- Producer: `docs/development/plans/active/contact-depth-field-sidecar.md`
- Predecessor: `docs/development/plans/active/per-vertex-contact-depth-poc.md`
- Motivating requirement: `docs/development/brainstorms/per-vertex-contact-depth.md` (IFF weighting
  requires the field in the RF-centred frame)
- Serialization decision record: `docs/development/knowledge-base/note-artifact-serialization-formats.md`
- Coordinate spaces: `docs/development/knowledge-base/note-spatial-alignment-pipeline.md` (stale — see
  Documentation Plan)
- Why columnar: `docs/development/knowledge-base/bug-rf-explorer-contact-parsing-scale.md`
- Not interpolable: `docs/development/knowledge-base/bug-rf-explorer-nearest-vertex-distance.md`
- Idempotency mechanics: `docs/development/plans/completed/stale-output-cleanup.md`,
  `keep-stale-outputs.md`, `fast-skip-check-extraction-pipeline.md`
- Migration precedent: `docs/development/plans/completed/fix-contact-points-csv-corruption.md`
- Known drift in the postprocessing workflow file: `docs/development/plans/pending/postprocess-workflow-update.md`
- Guides: `~/.claude/knowledge/data-pipeline-engineering/` 02 (§1 pipe-and-filter, §3 DTOs,
  §5 idempotency, §8 error boundaries, §11 provenance), 05 (§YAGNI)

---

## Modified Files

<!-- auto-generated by /plan-implement — do not edit manually -->
- code/scripts/_4_merging/__init__.py
- code/scripts/_4_merging/filter_contact_depth_field_by_neural_quality.py
- code/scripts/merging_pipeline_neuron_to_kinect_auto.py
- code/src/preprocessing/motion_analysis/tactile_quantification/io/contact_depth_field_io.py
- code/tests/test_contact_depth_field_io.py
- code/tests/test_filter_contact_depth_field_by_neural_quality.py
- configs/merging_pipeline_neuron_to_kinect_auto_dag.yaml
- docs/development/plans/active/filter-contact-depth-field-by-neural-quality.md
