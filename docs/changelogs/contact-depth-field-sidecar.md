# The per-vertex contact depth field is now persisted

**Date:** 2026-08-12
**Branch:** `feature/per-vertex-contact-depth-poc`
**Plan:** [`docs/development/plans/active/contact-depth-field-sidecar.md`](../development/plans/active/contact-depth-field-sidecar.md)

`compute_somatosensory_characteristics` now writes a second artifact alongside
`<video_stem>_contact_and_kinematic_data.csv`:

```
<block>/kinematics_analysis/<video_stem>_contact_depth_field.parquet
```

The per-vertex signed depth field was already computed on every frame — the
predecessor PoC made `signed_contact_depth_mm()` the single definition of
interpenetration, and every scalar the CSV reports is derived from it. It was
then discarded after `max()`. It is now kept.

**The CSV is unchanged.** No column added, no column renamed, no value moved.
`apply_registration_transform.py` string-matches on
`_contact_and_kinematic_data.csv`, and nothing about that path was touched.

## The artifact

**Long form:** exactly one row per (frame, contact vertex). A frame with no
contact contributes **zero rows** — frame-level facts stay in the CSV, which has
one row per frame. The two join on `frame_index`.

| column | dtype | meaning |
|--------|-------|---------|
| `frame_index` | `int32` | Kinect frame index; joins to the CSV |
| `time_s` | `float64` | seconds |
| `x`, `y`, `z` | `float32` | contact vertex position, mm, Kinect Space 1 |
| `signed_depth_mm` | `float64` | negative = penetrating |

File-level metadata (parquet schema key–value pairs) carries `schema_version`,
`coordinate_space`, `units`, `sign_convention`, `source_recording` and
`produced_by`, so a reader can determine all of them without consulting any
document in this repo.

**Depth is float64 on purpose.** The invariant that makes this artifact
checkable is that per frame, `max(|signed_depth_mm|)` equals the CSV's
`contact_depth` *bit-identically*. float32 would degrade that to "approximately".
Cost is roughly 8 MB → 12 MB per recording.

**`x/y/z` are in Kinect Space 1** — before ICP registration, PCA calibration and
RF-centring. They become wrong-space the moment postprocessing runs. That is why
the file declares `coordinate_space` rather than leaving it to convention.
Transporting the field through the postprocessing chain is explicitly *not* part
of this change.

**The field is not forward-fillable and not interpolable.** The contact patch
changes membership frame to frame, so row *i* of frame *n* is not the same vertex
as row *i* of frame *n+1*. Join on `frame_index`; never ffill.

## Two behaviour changes to expect

**Every existing session will reprocess this task once.** The sidecar is listed
in `should_process_task(output_paths=[csv, parquet])`, so a session with a
current CSV and no sidecar no longer skips. This is the only way already-processed
recordings gain the artifact — it is intended, and it costs the same as the
original run. `clean_task_outputs` removes both artifacts, so a failed run still
leaves nothing on disk.

**`pyarrow` is a new hard dependency**, added to `requirements.txt` and the
`pip:` block of `environment.yml`. `pyproject.toml` is a deliberately minimal
subset and was left alone. This is the repo's first parquet file; the incumbent
pattern was `savez_compressed`. See
[`note-artifact-serialization-formats.md`](../development/knowledge-base/note-artifact-serialization-formats.md)
for when to pick which — the format is isolated behind one writer/reader pair,
so reversing the choice changes two function bodies.

## What changed in the code

| File | Change |
|------|--------|
| `.../tactile_quantification/io/contact_depth_field_io.py` | **New.** `write_contact_depth_field` / `read_contact_depth_field`. Pure: knows a sequence of frames and a path, nothing about sessions, configs, DAGs, Prefect or the CSV. Imports its DTO under `TYPE_CHECKING` only, so serialisation never drags in Open3D |
| `.../model/objects_interaction_processor.py` | `process_single_frame(current_mesh, frame_index, time_s, _debug=False)` now returns `(contact_data, viz, ContactDepthFrame \| None)`. Frame identity is threaded into `signed_contact_depth_mm()`, which has accepted it since the PoC but had no caller populating it |
| `.../core/objects_interaction_controller.py` | `run()` returns `(df, field_series, vis_artifacts)`. The series is sparse — one entry per *contacting* frame |
| `compute_somatosensory_characteristics.py` | Gains `output_parquet_path`; both artifacts in `should_process_task` / `clean_task_outputs`; the sidecar is written immediately beside `results_df.to_csv(...)` so the two succeed or fail together |
| `preprocess_workflow_kinect_auto.py` | Flow builds both paths and returns `(csv, parquet)`; registry `"outputs": ["somatosensory_chars_path", "contact_depth_field_path"]` |

No DAG YAML change: `configs/preprocess_workflow_kinect_auto_dag.yaml` declares
only `enabled`, `options` and `depends_on`; output paths are built in Python. The
artifact is unconditional when the task runs — there is no toggle for it.

**Registry binding is positional.** The CSV must stay first in both the returned
tuple and the `outputs` list, because `unify_processed_data` consumes
`somatosensory_chars_path`.

## Fail-fast behaviour

- An **unusable hand pose raises**; it is never written as a zero-depth frame.
  "No rows for frame N" therefore has exactly one meaning: no contact. That
  matters because an absent-read-as-zero would silently down-weight real spikes
  once this field weights instantaneous firing rate.
- A **recording with no contact at all** raises rather than producing a zero-row
  file that a later run would accept as complete.
- An **unlabelled frame** (no `frame_index` / `time_s`) is refused: it cannot be
  joined to the CSV.
- The **winding-inversion sentinel** in `signed_contact_depth_mm()` still fires
  upstream of any write, and no `try/except` was added around it. At scalar
  level an inverted pose corrupted one number per frame; at per-vertex level it
  would emit tens of thousands of rows.
- The writer stages to a `.partial` sibling and `os.replace`s into position, so a
  failure mid-write leaves either the previous file or nothing.

## Tests

`code/tests/test_contact_depth_field_io.py` (new) covers the writer/reader in
isolation — round-trip values and dtypes, all six metadata keys, row accounting,
bit-identical float64 depth, every raise site, no-half-file on failure — plus the
idempotency behaviours the two-output list buys (a missing sidecar forces
reprocessing; a newer sidecar cannot mask a stale CSV).

`code/tests/test_contact_depth_field.py` gains controller-level tests asserting
the field series holds exactly the contacting frames with identity matching the
DataFrame's own `frame_index` / `time`.

Tests that need a real recording (frame-set agreement with the CSV in both
directions, per-frame bit-identity against `contact_depth`, row count versus the
parsed `contact_points` cell) are gated on
`SOCIAL_TOUCH_CONTACT_REFERENCE_DIR` and skip loudly when the bundle is absent —
it is not committable (size, participant data).

Any `pd.read_csv` in those assertions uses `float_precision="round_trip"`: the
default parser perturbs ~9% of a real recording's values by one ULP.

## Rollback

Purely additive. Nothing consumes `contact_depth_field_path` yet, so deleting
the sidecar has no downstream effect. Revert the pipeline-wiring commit alone
and the CSV path is untouched by construction. No migration: existing CSVs are
neither read-modified nor invalidated; existing recordings simply have no
sidecar until they reprocess, which they now do automatically.
