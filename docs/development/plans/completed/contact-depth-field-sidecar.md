# Plan: Contact Depth Field Sidecar

**Date:** 2026-08-12
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-08-20 14:51
**Base Branch:** `dev`
**Branch:** `feature/per-vertex-contact-depth-poc` *(continue on the existing branch — do NOT create a new one; invoke with `implement here`)*

---

## Overview

The per-vertex contact depth field is **already computed in production** — `ObjectsInteractionProcessor`
calls `signed_contact_depth_mm()` on every frame — and is then discarded after `max()`. This plan
persists it: a long-form columnar sidecar written alongside the existing
`*_contact_and_kinematic_data.csv` by the `compute_somatosensory_characteristics` task.

Nothing else changes. The CSV stays byte-identical, no postprocessing step is touched, and no DAG
toggle is added.

## Problem Statement

[`per-vertex-contact-depth-poc`](../active/per-vertex-contact-depth-poc.md) proved the field is
trustworthy (bit-identical to the legacy scalar on 3143/3143 real frames) and that it is
centre-peaked, which is what makes it usable as a weight. But it exists only inside a single
function call and dies there. Every recording processed since that PoC landed has computed the
field and thrown it away.

Until it is on disk, nothing downstream can use it: not depth-at-RF, not per-vertex IFF weighting,
not cross-trial spatial maps. Persistence is the smallest change that unblocks all of them.

There is also a second, quieter cost. The only per-vertex payload that survives today is
`contact_points`, an `[[x y z] ...]` blob in one CSV cell quantised to `%.1f`. Parsing it at scale
took 15+ minutes on 619k rows and needed two rounds of optimisation
(`bug-rf-explorer-contact-parsing-scale.md`). A columnar long-form table removes that entire
problem class for the new data.

## Goals

### In Scope

1. Surface the `ContactDepthFrame` from `ObjectsInteractionProcessor` up through
   `ObjectsInteractionController.run()`, populating `frame_index` and `time_s` (already accepted as
   kwargs by `signed_contact_depth_mm()`, currently populated by no caller).
2. A self-describing long-form sidecar written next to the CSV, one row per contact vertex per
   frame, with provenance metadata.
3. A reader for it, so tests and future consumers do not each reinvent one.
4. Correct idempotency: the new path participates in `should_process_task` and
   `clean_task_outputs`, so existing sessions reprocess rather than silently skipping.
5. Registry/flow wiring so the path lands in the pipeline context under
   `contact_depth_field_path`.

### Out of Scope

- **Transporting the field through the postprocessing chain** (ICP, dedup, projection onto
  forearm-of-reference, PCA calibration, RF-centring). The sidecar is written in **Kinect Space 1**
  and will become stale-space the moment postprocessing runs. This is why the file declares its
  coordinate space explicitly.
- **Persisting stable forearm-of-reference vertex ids** — the successor plan.
- **The dedup reduction rule** — not needed to ship this; deliberately unanswered.
- **The IFF weighting kernel** — lives in the analysis repo.
- **Back-filling existing artifacts.** Per `fix-contact-points-csv-corruption.md`'s precedent,
  users re-run the pipeline.
- **Renaming or altering the CSV.** `apply_registration_transform.py:74-77` string-matches on
  `_contact_and_kinematic_data.csv`.
- **A new DAG toggle.** Per `population-rf-vertex-data-export.md`, the artifact is unconditional
  when the task runs.
- **Fixing the `monitor=True` write-barrier defect** (see Risks) — documented, not repaired here.

## Success Criteria

- [ ] `<video_stem>_contact_depth_field.parquet` is produced by
      `compute_somatosensory_characteristics` for a real recording.
- [ ] Row count equals the total contact-vertex count over the recording (251 434 for
      `ST14-01/block-order-01`, from the PoC measurement).
- [ ] **The CSV is byte-identical before and after this change** for the same input, verified by
      hashing the output of a full run on both sides.
- [ ] Every `frame_index` present in the sidecar has `contact_detected == True` in the CSV, and
      every `contact_detected == True` frame has at least one sidecar row. No frame appears in one
      and not the other.
- [ ] For every frame, `max(|signed_depth_mm|)` from the sidecar equals the CSV `contact_depth`
      **bit-identically** (`np.array_equal`), reading the CSV with
      `float_precision="round_trip"`. This is why depth is stored at float64 — see Definitions.
- [ ] Per frame, sidecar row count equals the parsed length of that frame's `contact_points` cell.
- [ ] Deleting only the sidecar and re-running the task regenerates it (i.e. a missing sidecar
      forces reprocessing rather than skipping).
- [ ] `clean_task_outputs` removes both artifacts, so a failed run leaves neither.
- [ ] File-level metadata records schema version, coordinate space, units, sign convention and
      source recording; a reader can determine all five without consulting this document.
- [ ] `contact_depth_field_path` is present in the pipeline context after the task runs, and
      `somatosensory_chars_path` still binds to the CSV (positional binding — order matters).
- [ ] Round-trip: writing then reading returns arrays equal to the input, with dtypes preserved.

## Definitions

- **Sidecar**: `<video_stem>_contact_depth_field.parquet`, in the same
  `.../block-order-NN/kinematics_analysis/` directory as the CSV, sharing its `<video_stem>` stem.
- **Long form**: exactly one row per (frame, contact vertex). A frame with no contact contributes
  **zero rows**. Frame-level facts are *not* stored here — they live in the CSV, which has one row
  per frame. The two join on `frame_index`.
- **Contact vertex**: unchanged from the PoC — a forearm vertex belonging to at least one triangle
  whose all three vertices satisfy `signed_distance < EPSILON` (`1e-5`).
- **Signed depth**: negative = penetrating. Millimetres, Kinect-native, no conversion anywhere.
  Stored signed; `penetration_depth_mm = -signed_depth_mm` is a display-only derivation.
- **Kinect Space 1**: the raw Kinect frame in mm, before ICP registration, PCA calibration and
  RF-centring. The sidecar's `x/y/z` are in this space and **become wrong-space if interpreted
  after postprocessing has run**. The file says so in its metadata.
- **Byte-identical CSV**: `hashlib.sha256` of the written file matches pre-change output for the
  same inputs. Not "looks the same", not "same columns".
- **Zero vs absent**: a frame with no contact has zero rows and `contact_detected == False` in the
  CSV — that is *zero*. A frame with an unusable hand pose is *absent* and must **raise**, never be
  written as zero. See Technical Design.

---

## Technical Design

### Approach

Three seams, in dependency order:

```
[1] processor/controller widen   →  ContactDepthFrame escapes run()
[2] io module (write + read)     →  pure, no pipeline knowledge          ← testable in isolation
[3] task + flow + registry       →  path arithmetic and idempotency
```

Seam [2] is a **pure I/O module** that takes a list of `ContactDepthFrame` and a path. It knows
nothing about sessions, configs, DAGs or the CSV. That is what makes it unit-testable without a
recording and promotable if the artifact ever moves.

**Provenance lives in file metadata, not in columns.** Parquet supports schema-level key-value
metadata; writing `schema_version`, `coordinate_space`, `units`, `sign_convention` and
`source_recording` there keeps the row payload minimal while making the file self-describing
(guide 02 §11). Constant-valued columns would also work and dictionary-encode to near-zero, but
metadata is the semantically correct home and avoids implying the value could vary per row.

**Depth is stored at float64, not float32.** This is a deliberate departure from the schema the PoC
plan fixed. float32 would silently break the plan's headline invariant: `max(|signed_depth_mm|)`
recovered from a float32 column cannot equal the float64 `contact_depth` in the CSV bit-identically,
only approximately. Preserving an exact, checkable invariant end-to-end is worth the bytes — the
measured payload is 7.67 MB per recording, so this is roughly 8→12 MB. `x/y/z` stay float32
(0.1 µm at mm scale, against a CSV counterpart already quantised to 0.1 mm).

**An unusable hand pose raises rather than being recorded.** The PoC driver modelled a three-state
`FrameStatus` including `POSE_ABSENT`, but that enum lives in a `code/scripts/` module a `src`
package must not import — and more importantly, `POSE_ABSENT` fires only on genuinely non-finite
pose data, which under the fail-fast mandate is an error, not a state to serialise. Raising means
"no rows for frame N" has exactly one meaning: no contact. That is what keeps zero and absent from
collapsing, which matters because an absent-read-as-zero would silently down-weight real spikes
once this field weights IFF.

### Schema

| column | dtype | meaning |
|--------|-------|---------|
| `frame_index` | `int32` | Kinect frame index; joins to the CSV's `frame_index` |
| `time_s` | `float64` | seconds, from the controller's timestamps |
| `x`, `y`, `z` | `float32` | contact vertex position, mm, Kinect Space 1 |
| `signed_depth_mm` | `float64` | negative = penetrating; float64 preserves the exact invariant |

File-level metadata (`schema.with_metadata`):

| key | value |
|-----|-------|
| `schema_version` | `"1"` |
| `coordinate_space` | `"kinect_space_1"` |
| `units` | `"mm"` |
| `sign_convention` | `"negative_is_penetrating"` |
| `source_recording` | `<video_stem>` |
| `produced_by` | `"compute_somatosensory_characteristics"` |

`schema_version` is present from day one on the direct evidence of prior art:
`population-rf-vertex-data-export.md` shipped without one and had its file renamed and schema
expanded by a follow-on plan five days later.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Parquet (pyarrow)** long form | Self-describing schema + metadata; dtypes guaranteed on round-trip; columnar, so the downstream analytics repo can filter/join without loading everything; ragged frames need no padding or offsets; kills the string-blob parsing problem class | **New hard dependency** — zero parquet/pyarrow usage anywhere in the repo today | **Chosen** |
| `savez_compressed` `.npz` (three flat arrays) | **Zero new dependency**; the repo's incumbent sidecar pattern (`bug-rf-explorer-contact-parsing-scale.md`, `note-rf-camera-settings-connections.md`); `population-rf-vertex-data-export.md` chose exactly this and rejected HDF5 as "new dependency, overkill" | numpy-only container — the eventual consumer is a *separate analytics repo*; no schema, no metadata, no column types, no partial reads; every consumer re-derives the layout | **Rejected**, but it is a genuine contender and the decision is reversible — seam [2] isolates the format behind a writer/reader pair |
| HDF5 | Hierarchical, partial reads | Already rejected in this repo for this exact reason (new dependency, overkill) | Rejected |
| Extra columns on the existing CSV | No new artifact | Ragged data cannot be one row per frame; would reintroduce the string-blob pathology at larger scale | Rejected |
| Constant-valued provenance columns instead of file metadata | Readable by any parquet reader without metadata support | Implies the value could vary per row; inflates the schema | Rejected |
| Store `signed_depth_mm` as float32 | ~4 MB smaller per recording | Breaks the bit-exact invariant against the CSV — the single strongest correctness check this feature has | Rejected |

**On the dependency.** `pyarrow` must be added to `requirements.txt` and the `pip:` block of
`environment.yml`. `pyproject.toml`'s dependency list is a minimal subset that does not even name
`open3d`, so it is left alone. If the dependency is judged unacceptable at review, seam [2] is the
only place that changes — swap the writer/reader bodies for `savez_compressed` and every other file
in this plan stands.

### Architecture & Module Contracts

| Module / layer | Responsibility | Inputs → Outputs | Must NOT know about |
|----------------|----------------|------------------|---------------------|
| `contact_depth_field_io.py` (new) | Serialise/deserialise a field series | `(List[ContactDepthFrame], Path, metadata dict)` → file; `Path` → `pd.DataFrame` | Sessions, configs, DAGs, the CSV, Prefect, the pipeline at all |
| `ObjectsInteractionProcessor` (modified) | Per-frame orchestration | gains `frame_index`/`time_s` params; returns the `ContactDepthFrame` alongside existing dicts | The sidecar, its format, its path |
| `ObjectsInteractionController` (modified) | Frame loop | `run()` additionally returns the field series | The sidecar format and path |
| `compute_somatosensory_characteristics` (modified) | Task entry point | gains `output_parquet_path` param; writes both artifacts | How the field was computed |
| `compute_somatosensory_characteristics_flow` (modified) | Path arithmetic | builds both paths; returns `(csv, parquet)` | Serialisation details |

```
code/src/preprocessing/motion_analysis/tactile_quantification/
├── io/
│   ├── __init__.py                       # NEW
│   └── contact_depth_field_io.py         # NEW — write_contact_depth_field / read_contact_depth_field
├── model/
│   └── objects_interaction_processor.py  # MODIFIED — surface the frame, accept frame identity
└── core/
    └── objects_interaction_controller.py # MODIFIED — collect and return the series
```

Target signatures:

```python
def write_contact_depth_field(
    frames: Sequence[ContactDepthFrame],
    output_path: Path,
    *,
    source_recording: str,
) -> None:
    """Write the long-form sidecar. Overwrites wholesale; never appends."""

def read_contact_depth_field(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return (long-form rows, file metadata). Raises if schema_version is unknown."""
```

### Idempotency

`should_process_task` (`code/src/utils/should_process_task.py:55`) already accepts
`PathInput = Union[Path, List[Path]]`, so the change is `output_paths=[csv, parquet]` — no utility
change needed. Its rules make this correct automatically:

- **any** output missing → reprocess. This is what solves the first-run trap: after this ships,
  every existing session has a current CSV but **no sidecar**, so the task would otherwise skip
  forever and never produce one. Listing the sidecar makes a missing sidecar force the rerun. No
  `force_processing` sweep needed.
- staleness compares newest input against **oldest** output, so a sidecar written after the CSV
  cannot mask a stale CSV.

`clean_task_outputs([csv, parquet])` then removes both, keeping the existing "a failed run leaves
nothing on disk" behaviour intact.

**Deliberately not changed:** `input_paths` currently omits `hand_metadata_path` and
`forearm_pointcloud_dir`, so edits to those do not trigger reprocessing. That is a pre-existing
staleness gap, out of scope here, and widening it would force a large unplanned reprocessing sweep.
Noted so it is not mistaken for an oversight.

### Registry wiring

Binding is **positional** (`preprocess_workflow_kinect_auto.py:628-638`), and
`unify_processed_data` consumes `somatosensory_chars_path` (`:602`). Therefore the CSV must stay
**first** in both the returned tuple and the `outputs` list:

```python
"outputs": ["somatosensory_chars_path", "contact_depth_field_path"],
```

The two-output tuple return is an established in-repo pattern —
`generate_3d_hand_in_motion` (`:553-559`) and `generate_stimuli_metadata` (`:540`) both do it.
**No YAML change**: `configs/preprocess_workflow_kinect_auto_dag.yaml:127-133` declares only
`enabled`, `options`, `depends_on` — output paths are built in Python.

### Constraints from the knowledge base

- **Winding-inversion sentinel must fire before the writer.** A `det(M) < 0` pose inverts the
  winding number and makes *every* forearm vertex report as penetrating
  (`bug-contact-detection-winding-inversion.md`). At scalar level that corrupted one number per
  frame; at per-vertex level it emits tens of thousands of rows. The existing sentinel in
  `signed_contact_depth_mm()` already raises upstream of any write — verify that ordering holds and
  do not add a `try/except` that would let a corrupt frame through.
- **Never align sidecar rows to `contact_points` by coordinate value.** `serialize_contact_points`
  writes `%.1f`, so the CSV blob is quantised to 0.1 mm while the sidecar carries full precision
  (`note-spatial-alignment-pipeline.md`, Invariant 3). Align by index/order only.
- **`float_precision="round_trip"`** on any `pd.read_csv` used in a test asserting exact equality —
  the default parser perturbs ~9% of values by one ULP, measured on this very data during the PoC.
- **Units are mm throughout**, unconverted from the Kinect SDK
  (`note-somatosensory-units-and-calculations.md`). Do not propagate the known `'Area (cm^2)'`
  mislabel at `compute_somatosensory_characteristics.py:150` into the new artifact's metadata.
- **The field is not forward-fillable.** `bug-rf-explorer-nearest-vertex-distance.md` records cubic
  interpolation overshooting 29 mm off-surface at touch boundaries. Per-vertex rows must never be
  interpolated or ffilled the way a scalar can be — state it in the module docstring.
- **CuPy import order** — only bites if the parquet engine gets imported at an entry point; the
  writer lives in `src`, so guarded `import cupy` ordering in scripts is unaffected. Verify.
- **Skip check must remain resolvable from path arithmetic alone**
  (`fast-skip-check-extraction-pipeline.md`). The sidecar path derives from `<video_stem>` exactly
  like the CSV — keep it that way; never gate it on anything read from the CSV.

---

## Implementation Plan

### Phase 1: Widen the seam
**Goal:** The `ContactDepthFrame` escapes `run()` with frame identity populated.

- [x] 1.1 — `signed_contact_depth_mm()` already accepts `frame_index`/`time_s`; thread real values
      from `process_single_frame` into it (no caller populates them today).
- [x] 1.2 — `process_single_frame(current_mesh, frame_index, time_s, _debug=False)` returns the
      `ContactDepthFrame` (or `None`) alongside the existing `(contact_quantities, contact_info)`.
- [x] 1.3 — `ObjectsInteractionController.run()` collects the frames and returns
      `(df, field_series, vis_artifacts)`.
- [x] 1.4 — Update the PoC driver's call site so it keeps working. **No-op**: the driver
      reimplements the frame loop against `signed_contact_depth_mm()` directly and never calls
      `process_single_frame`; that function's signature is unchanged. Verified by grep.
- [x] 1.5 — Tests: frame identity is populated and matches the DataFrame's `frame_index`/`time`;
      the series contains exactly the contact frames.

**Files Modified:**
- `.../model/objects_interaction_processor.py` — thread identity, surface the frame
- `.../core/objects_interaction_controller.py` — collect and return the series
- `code/scripts/_3_preprocessing/_4_somatosensory_quantification/poc_contact_depth_field.py` — call site
- `code/tests/test_contact_depth_field.py` — extend

**Dependencies:** None

### Phase 2: The I/O module
**Goal:** A pure, unit-tested writer and reader with no pipeline knowledge.

- [x] 2.1 — Add `pyarrow` to `requirements.txt` and the `pip:` block of `environment.yml`.
- [x] 2.2 — `contact_depth_field_io.py` with `write_contact_depth_field` and
      `read_contact_depth_field`, the schema above, and file-level metadata.
- [x] 2.3 — Raise on unknown `schema_version` at read time; raise on empty/`None` frames rather
      than writing a zero-row file silently.
- [x] 2.4 — Module docstring recording sign convention, units, coordinate space, and the
      not-forward-fillable warning.
- [x] 2.5 — Tests: round-trip equality and dtype preservation; metadata round-trip; row count;
      unknown-schema-version raise; empty-input raise; a frame with a single contact vertex.

**Files Modified:**
- `.../tactile_quantification/io/__init__.py` — new
- `.../tactile_quantification/io/contact_depth_field_io.py` — new
- `requirements.txt`, `environment.yml` — add pyarrow
- `code/tests/test_contact_depth_field_io.py` — new

**Dependencies:** Phase 1

### Phase 3: Pipeline wiring
**Goal:** The sidecar is produced by the real task, with correct idempotency.

- [x] 3.1 — `compute_somatosensory_characteristics` gains `output_parquet_path: Path`.
- [x] 3.2 — `should_process_task(output_paths=[csv, parquet], ...)` and
      `clean_task_outputs([csv, parquet])`.
- [x] 3.3 — Write the sidecar immediately adjacent to `results_df.to_csv(...)` (`:173`) so both
      artifacts succeed or fail together.
- [x] 3.4 — Flow builds both paths inline (matching the repo's `<stem> + "_<suffix>"` idiom) and
      returns `(csv_path, parquet_path)`; return type becomes `tuple[Path, Path]`.
- [x] 3.5 — Registry: `"outputs": ["somatosensory_chars_path", "contact_depth_field_path"]`, CSV
      first.
- [x] 3.6 — Integration tests written in `code/tests/test_contact_depth_field_io.py`, gated on
      `SOCIAL_TOUCH_CONTACT_REFERENCE_DIR`: frame-set agreement both directions, per-frame
      `max(|depth|)` bit-identical to `contact_depth`, per-frame row count equals parsed
      `contact_points` length, metadata declares the coordinate space. The
      delete-sidecar-and-rerun and stale-CSV behaviours are asserted directly against
      `should_process_task`, so they run without a recording.
      **Not automated:** the CSV sha256 byte-identity check and the context-binding check —
      both need a full Prefect run against a real recording. See Verification Status below.

**Files Modified:**
- `code/scripts/_3_preprocessing/_4_somatosensory_quantification/compute_somatosensory_characteristics.py`
- `code/scripts/preprocess_workflow_kinect_auto.py` — flow + registry
- `code/tests/test_contact_depth_field_io.py` — integration cases

**Dependencies:** Phase 2

---

## Verification Status (2026-08-12)

Implemented on `feature/per-vertex-contact-depth-poc`. Two things about the environment bound what
could be executed, and both are stated here rather than left implicit.

**The conda env `social-touch-env` was not present on this machine.** The available interpreter has
`pandas`/`numpy`/`pytest` but no `open3d`, `PyQt5`, `prefect` or `cv2`.

| Check | Status |
|-------|--------|
| Writer/reader unit tests (24 tests) | **Executed, passing** — they need only numpy/pandas/pyarrow |
| Idempotency: missing sidecar forces reprocess; newer sidecar cannot mask a stale CSV; `clean_task_outputs` removes both | **Executed, passing** — asserted directly against `should_process_task` |
| Controller identity threading + series collection | **Verified out-of-band** with a stubbed-geometry harness (fake `open3d`, monkeypatched `signed_contact_depth_mm`): identity threaded, series holds exactly the contacting frames, per-frame `max(|depth|)` matches the DataFrame's `contact_depth`, sidecar round-trips. The committed tests for this (`test_contact_depth_field.py`) require real Open3D and **skip** in this environment |
| Every file compiles | **Executed** (`py_compile`) |
| CSV byte-identity (sha256) before/after | **Not run.** Needs a full pipeline run on a real recording. The change is structurally byte-safe — `results_df.to_csv(...)` is untouched and the sidecar write is appended after it — but that is an argument, not a measurement |
| Row count ≈ 251 434 and 10–15 MB on `ST14-01/block-order-01` | **Not run.** Same reason |
| Context binding (`somatosensory_chars_path` = CSV, `contact_depth_field_path` = sidecar) | **Not run.** Needs Prefect. Guarded by construction: CSV first in both the returned tuple and the `outputs` list |

**Before merging, run the task once on `ST14-01/block-order-01`** and confirm the three unrun rows.
Note that merging triggers a reprocessing sweep of every existing session — by design, since that is
how already-processed recordings gain a sidecar.

### Deviations from the plan as written

- **`ContactDepthFrame` is imported under `TYPE_CHECKING`** in the io module rather than at runtime.
  The writer duck-types four plain attributes and needs no geometry engine; this keeps a pure
  serialisation module importable and testable without Open3D, which is what let the unit tests run
  at all here. It strengthens the purity contract the plan asked for.
- **`produced_by` is a module constant**, not a parameter. The plan's target signature takes only
  `source_recording`, and a provenance string every caller must supply correctly is one that will
  eventually be supplied incorrectly.
- **The knowledge-base note is `note-artifact-serialization-formats.md`**, written as general
  guidance for the next artifact plan rather than as a note about this one specifically.

---

## Testing Plan

### Unit Tests
- [ ] Round-trip: write → read returns equal arrays with dtypes preserved.
- [ ] File metadata round-trips all six keys.
- [ ] Row count equals the sum of per-frame contact-vertex counts.
- [ ] Reading a file with an unknown `schema_version` raises.
- [ ] Writing an empty series raises rather than producing a zero-row file.
- [ ] A frame with exactly one contact vertex writes exactly one row.
- [ ] `frame_index`/`time_s` populated by the controller match the DataFrame's own columns.
- [ ] `signed_depth_mm` survives round-trip bit-identically at float64.

### Integration Tests
- [ ] **CSV byte-identical** (sha256) before and after the change on the same inputs.
- [ ] Frame-set agreement both directions against the CSV: no frame in one and not the other.
- [ ] Per frame, `max(|signed_depth_mm|)` equals CSV `contact_depth` bit-identically
      (`np.array_equal`, CSV read with `float_precision="round_trip"`).
- [ ] Per frame, sidecar row count equals the parsed `contact_points` length.
- [ ] Delete only the sidecar, re-run → it is regenerated (the first-run trap).
- [ ] Delete only the CSV, re-run → both are regenerated.
- [ ] Context binding: `somatosensory_chars_path` is the CSV, `contact_depth_field_path` is the
      sidecar.

### Manual Verification
- [ ] Run the task on `ST14-01/block-order-01`; confirm ~251 434 rows and a file size around
      10–15 MB.
- [ ] Open the sidecar in pandas without any repo import and confirm it is self-describing —
      units, space and sign convention readable from metadata alone.
- [ ] Re-run the task unchanged; confirm it skips and writes nothing.

### Edge Cases
- [ ] Recording with zero contact frames throughout — writer raises rather than emitting an empty
      file; task surfaces it clearly.
- [ ] `clean_task_outputs` on a Windows-locked file (it warns and continues by design).
- [ ] A `PermissionError` mid-write leaves no half-file that a later run would treat as valid.

---

## Documentation Plan

- [ ] Module docstring in `contact_depth_field_io.py`: schema, sign convention, units, coordinate
      space, not-forward-fillable warning.
- [ ] Changelog entry `docs/changelogs/contact-depth-field-sidecar.md`.
- [ ] Update the PoC plan's Out-of-Scope note to point here.
- [ ] Note in `docs/development/knowledge-base/` that the repo now has a parquet dependency and
      what the incumbent `.npz` alternative was — future artifact plans will face the same choice.
- [ ] **Not** updating CLAUDE.md — no architectural boundary moves.

---

## Rollback Plan

1. **Purely additive artifact.** Deleting the sidecar has no downstream impact — nothing consumes
   `contact_depth_field_path` yet.
2. Phase 3 alone: revert the task/flow/registry commit. The CSV path is untouched by construction
   (the byte-identity test is what guarantees this).
3. Phase 2 alone: delete the io module and drop `pyarrow` from the env files.
4. Phase 1 is the only phase touching shared call signatures; revert it last, and the PoC driver
   with it.
5. **Data considerations:** no migration. Existing CSVs are neither read-modified nor invalidated.
   Existing recordings simply have no sidecar until reprocessed — and after this ships they will
   reprocess automatically, since a missing output forces the rerun.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **New `pyarrow` dependency rejected at review** | Med | Med | Format is isolated behind seam [2]; swapping to `savez_compressed` changes two function bodies and nothing else. Alternatives table states the case both ways |
| **First-run skip** — existing sessions have current CSVs, so the task skips and no sidecar is ever produced | High if missed | High | Sidecar listed in `output_paths`; missing output forces reprocess. Explicit integration test |
| Adding the sidecar to `output_paths` triggers a **full reprocessing sweep** of every session | High | Med | This is intended, not accidental — it is the only way existing recordings gain a sidecar. Flag the compute cost before merging; it is the same cost as the original run |
| Winding inversion emits tens of thousands of corrupt rows instead of one corrupt number | Low | High | Existing sentinel raises upstream of the write; verify ordering, add no `try/except` around it |
| Sidecar interpreted in the wrong coordinate space after postprocessing runs | Med | High | `coordinate_space` in file metadata; reader surfaces it; stated in the module docstring and Out of Scope |
| Positional registry binding silently swaps the two paths | Low | High | CSV first in both tuple and `outputs` list; explicit context-binding test |
| `monitor=True` exits before the write | Low | Med | **Pre-existing defect**: `sys.exit(app.exec_())` is raised inside an `except AttributeError:` clause, so the sibling `except SystemExit: pass` does not catch it, and the CSV is already never written in that path. The DAG sets `monitor: false`. Placing the sidecar write beside `to_csv` means it inherits exactly the existing behaviour rather than a new one. Documented, not fixed here |
| Schema needs to change later | Med | Low | `schema_version` from day one, reader raises on unknown — directly on the evidence of `population-rf-vertex-data-export.md`'s rename five days after shipping |
| float32/float64 confusion breaks the exact invariant | Low | High | Depth is float64 at rest by explicit decision; a round-trip test asserts bit-identity |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 — widen the seam | ~120 LOC + ~80 test | None |
| Phase 2 — io module | ~180 LOC + ~200 test | Phase 1 |
| Phase 3 — pipeline wiring | ~80 LOC + ~150 test | Phase 2 |

---

## References

- Predecessor: [`docs/development/plans/active/per-vertex-contact-depth-poc.md`](../active/per-vertex-contact-depth-poc.md) — PR #72
- Brainstorm: [`docs/development/brainstorms/per-vertex-contact-depth.md`](../../brainstorms/per-vertex-contact-depth.md)
- Prior art (artifact added to an existing task): `docs/development/plans/completed/population-rf-vertex-data-export.md`
- Idempotency mechanics: `docs/development/plans/completed/stale-output-cleanup.md`,
  `keep-stale-outputs.md`, `fast-skip-check-extraction-pipeline.md`
- Serialization decision history: `docs/development/plans/completed/fix-contact-points-csv-corruption.md`
- `docs/development/knowledge-base/note-spatial-alignment-pipeline.md` — Space 1, `%.1f` quantisation
- `docs/development/knowledge-base/bug-contact-detection-winding-inversion.md` — the volume risk
- `docs/development/knowledge-base/bug-rf-explorer-contact-parsing-scale.md` — why columnar
- `docs/development/knowledge-base/note-somatosensory-units-and-calculations.md` — mm throughout
- Guides: `~/.claude/knowledge/data-pipeline-engineering/` 02 (§3 DTOs, §5 idempotency, §8 error
  boundaries, §11 provenance), 03 (§1 reproducibility, §6 missing vs zero), 05 (§6 YAGNI)
