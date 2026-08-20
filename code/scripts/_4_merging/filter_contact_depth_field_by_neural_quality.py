"""Apply the neural-quality trial exclusion to the per-vertex contact depth field.

What this filter removes
------------------------
``filter_by_neural_quality`` discards the trials the experimenter marked
``Not2Use`` in ``semicontrolled_data-collection_quality-check.xlsx``, dropping
rows from the merged CSV.  The Space-1 contact depth field sidecar produced in
preprocessing knows nothing about that exclusion: it still carries per-vertex
rows for trials the nerve recording cannot support.  This task applies the
*outcome* of that exclusion to the sidecar so the two artifacts in
``blocks_filtered/`` describe the same set of frames.

Measured on the 103 blocks holding both a merged and a filtered CSV, 34 lose
rows, the worst retaining 27.5% of its original size.  Copying the sidecar
without the exclusion would hand postprocessing per-vertex data for neurally
unusable trials and break the invariant "every sidecar frame appears in the
CSV" on a third of the dataset.  The exclusion is the substance of this task;
relocating the file is the incidental part.

Why the exclusion is read from the CSV, not the xlsx
-----------------------------------------------------
The surviving frames are read from the already-filtered CSV.  Re-parsing the
quality xlsx would be a second implementation of one rule — it would have to
duplicate the ``discard_from_first_not2use`` branch and the unit/block filename
parsing, and the two copies would diverge the first time either changed.  Read
from the CSV, this task follows automatically whenever
``filter_by_neural_quality`` changes its rule or its options.

Why the field is not upsampled
-------------------------------
Merging scatters Kinect rows ~33x into the nerve-rate frame, which is why
``frame_index`` is NaN on roughly 32 of every 33 rows of the merged CSV.  Those
NaN rows are interpolated nerve-rate samples carrying no Kinect data; they are
not frames.  Replicating that upsampling in a per-vertex table would multiply
~1 GB by ~33 for zero information gain, and the field is not interpolable in
any case (the contact patch changes membership frame to frame, so row *i* of
frame *n* and row *i* of frame *n+1* are not the same vertex; cubic
interpolation across a touch boundary has been measured overshooting 29 mm
off-surface).  The field stays at Kinect frame rate and joins to the CSV's
anchor rows on ``frame_index``.

Why selection is by ``frame_index`` only
-----------------------------------------
Never by coordinate value.  The CSV's ``contact_points`` column is quantised to
``%.1f`` by ``serialize_contact_points``, so matching a sidecar row to a CSV row
by position would compare a float32 millimetre coordinate against a rounded
decimal string.  ``frame_index`` is the only exact join key the two artifacts
share.

No coordinate transformation
-----------------------------
Merging performs none.  The output stays in **Kinect Space 1** and says so in
its metadata, which is carried through from the input file unchanged.  Rows are
removed; not one surviving value is recomputed, rescaled, rounded or reordered.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from preprocessing.motion_analysis.tactile_quantification.io.contact_depth_field_io import (
    COLUMN_DTYPES,
    read_contact_depth_field,
    write_contact_depth_field_table,
)
from utils.should_process_task import should_process_task, clean_task_outputs

logger = logging.getLogger(__name__)


#: Metadata keys carried from the Space-1 sidecar to the filtered one, verbatim.
#: ``coordinate_space`` is in this list rather than being restated: this task
#: applies no spatial transform, so the space it declares must be the space the
#: input declared, and re-deriving it here would let the two drift apart.
CARRIED_METADATA_KEYS: Tuple[str, ...] = (
    "schema_version",
    "coordinate_space",
    "units",
    "sign_convention",
    "source_recording",
    "produced_by",
)

#: Columns read from the merged CSV.  It reaches 112 MB on the largest block,
#: so it is never read whole: ``frame_index`` gives the surviving frame set and
#: ``contact_detected`` lets a stale-artifact disagreement be detected.
_REQUIRED_CSV_COLUMNS: Tuple[str, ...] = ("frame_index", "contact_detected")


def _read_frame_rows(filtered_csv_path: Path) -> pd.DataFrame:
    """Return the CSV's Kinect-frame rows: ``frame_index`` and ``contact_detected``.

    Only the two needed columns are parsed, and only the rows carrying a real
    ``frame_index`` are returned — the NaN ones are the interpolated nerve-rate
    samples and are not frames.

    Raises:
        ValueError: If either required column is absent, if a ``frame_index``
            is not integral, or if a frame row carries a non-boolean or missing
            ``contact_detected``.
    """
    header = pd.read_csv(filtered_csv_path, nrows=0)
    missing = [c for c in _REQUIRED_CSV_COLUMNS if c not in header.columns]
    if missing:
        raise ValueError(
            f"{filtered_csv_path} has no {missing} column(s). The filtered merged "
            f"CSV is the sole source of the surviving frame set, and "
            f"'contact_detected' is what makes a stale sidecar detectable. "
            f"Columns found: {list(header.columns)}."
        )

    csv = pd.read_csv(filtered_csv_path, usecols=list(_REQUIRED_CSV_COLUMNS))
    frame_rows = csv[csv["frame_index"].notna()]

    if len(frame_rows) == 0:
        raise ValueError(
            f"{filtered_csv_path} carries no non-NaN 'frame_index' value: it "
            "describes no Kinect frame at all. Refusing to derive a surviving "
            "frame set from it."
        )

    raw_frames = frame_rows["frame_index"].to_numpy()
    if not np.array_equal(raw_frames, np.rint(raw_frames)):
        non_integral = raw_frames[raw_frames != np.rint(raw_frames)]
        raise ValueError(
            f"{filtered_csv_path} carries non-integral 'frame_index' values "
            f"(e.g. {non_integral[:5].tolist()}). A frame index identifies a "
            "Kinect frame; a fractional one means the column was interpolated, "
            "and interpolated frame indices cannot be joined to the sidecar."
        )

    contact = frame_rows["contact_detected"]
    if contact.dtype.kind not in "biuf":
        raise ValueError(
            f"{filtered_csv_path} column 'contact_detected' has dtype "
            f"{contact.dtype}, which is neither boolean nor numeric. It is a "
            "binary state flag; refusing to guess how to interpret it."
        )
    if contact.isna().any():
        n_missing = int(contact.isna().sum())
        raise ValueError(
            f"{filtered_csv_path}: {n_missing} row(s) carry a 'frame_index' but "
            "no 'contact_detected'. Every Kinect frame row states whether "
            "contact was detected; a missing flag means the CSV is malformed, "
            "and treating it as 'no contact' would silently weaken the "
            "sidecar-agreement check below."
        )

    return frame_rows


def _surviving_frames(frame_rows: pd.DataFrame) -> np.ndarray:
    """Return the sorted unique frame indices that survived neural filtering."""
    return np.unique(frame_rows["frame_index"].to_numpy().astype(np.int64))


def _contact_frames(frame_rows: pd.DataFrame) -> np.ndarray:
    """Return the sorted unique frame indices the CSV reports as contacting."""
    contacting = frame_rows[frame_rows["contact_detected"].astype(bool)]
    return np.unique(contacting["frame_index"].to_numpy().astype(np.int64))


def filter_contact_depth_field_by_neural_quality(
    depth_field_path: Path,
    filtered_csv_path: Path,
    output_path: Path,
    *,
    force_processing: bool = False,
) -> Optional[Path]:
    """Reduce the Space-1 depth field to the frames that survived neural-quality filtering.

    Args:
        depth_field_path: The Space-1 ``*_contact_depth_field.parquet`` sidecar
            written by ``compute_somatosensory_characteristics``.
        filtered_csv_path: The block's ``blocks_filtered/*_merged_data.csv``,
            already stripped of Not2Use trials by ``filter_by_neural_quality``.
            Its non-NaN ``frame_index`` values *are* the exclusion criterion.
        output_path: Destination ``.parquet``. Its parent is created if absent.
        force_processing: Rewrite even when the output is present and current.

    Returns:
        ``output_path`` when the filtered sidecar was written, or ``None`` when
        the existing output was already up to date and nothing was written.

    Raises:
        FileNotFoundError: If either input is missing — a broken dependency
            graph, not a skip.
        ValueError: If the CSV lacks a required column; if the CSV reports a
            contacting frame the sidecar does not contain (the two artifacts
            disagree about the same recording, so one is stale); if no row is
            retained (an empty artifact is indistinguishable from a complete one
            on the next run); or if the input sidecar's metadata omits a key
            this task must carry through.
    """
    depth_field_path = Path(depth_field_path)
    filtered_csv_path = Path(filtered_csv_path)
    output_path = Path(output_path)

    if not should_process_task(
        input_paths=[depth_field_path, filtered_csv_path],
        output_paths=[output_path],
        force=force_processing,
    ):
        logger.info(f"Already up-to-date: {output_path.name}")
        return None
    clean_task_outputs([output_path])

    table, source_metadata = read_contact_depth_field(depth_field_path)

    # Captured before any selection so the bit-identity check below compares the
    # written rows against what was actually read off disk, not against a view
    # of themselves.
    original_columns: Dict[str, np.ndarray] = {
        column: table[column].to_numpy(copy=True) for column in COLUMN_DTYPES
    }

    frame_rows = _read_frame_rows(filtered_csv_path)
    surviving = _surviving_frames(frame_rows)
    csv_contact_frames = _contact_frames(frame_rows)

    depth_frames = np.unique(original_columns["frame_index"].astype(np.int64))

    # The two artifacts must agree in this direction: a frame the CSV reports as
    # contacting owes at least one vertex row to the sidecar.  The other
    # direction is the filter doing its job and is merely counted.
    absent_from_field = np.setdiff1d(csv_contact_frames, depth_frames)
    if absent_from_field.size:
        raise ValueError(
            f"{filtered_csv_path.name} reports contact on "
            f"{absent_from_field.size} frame(s) absent from "
            f"{depth_field_path.name} (e.g. {absent_from_field[:10].tolist()}). "
            "The two artifacts describe the same recording, so one of them is "
            "stale. Refusing to write a quietly incomplete depth field; "
            "regenerate the sidecar or the merged CSV."
        )

    retained_mask = np.isin(original_columns["frame_index"].astype(np.int64), surviving)
    retained = table[retained_mask]

    if len(retained) == 0:
        raise ValueError(
            f"No row of {depth_field_path.name} survives the neural-quality "
            f"filter: none of its {depth_frames.size} frame(s) appears in "
            f"{filtered_csv_path.name}. Refusing to write a zero-row sidecar, "
            "which a later run would treat as a valid, complete artifact — "
            "zero rows and an absent artifact must not collapse into the same "
            "thing on disk."
        )

    # This task removes rows. It must never alter a surviving value, so assert
    # that rather than trusting it: every column, values and dtype alike.
    for column in COLUMN_DTYPES:
        expected = original_columns[column][retained_mask]
        actual = retained[column].to_numpy()
        if actual.dtype != expected.dtype:
            raise ValueError(
                f"Row selection changed the dtype of {column!r} from "
                f"{expected.dtype} to {actual.dtype}. Precision is a schema "
                "decision here; a filter must not touch it."
            )
        if not np.array_equal(actual, expected):
            raise ValueError(
                f"Row selection altered the values of {column!r}. This task "
                "removes rows and must leave every surviving value bitwise "
                "unchanged."
            )

    retained_frames = np.unique(original_columns["frame_index"][retained_mask].astype(np.int64))
    frames_dropped = int(depth_frames.size - retained_frames.size)
    rows_dropped = int(len(table) - len(retained))

    missing_metadata = [k for k in CARRIED_METADATA_KEYS if k not in source_metadata]
    if missing_metadata:
        raise ValueError(
            f"{depth_field_path} carries no {missing_metadata} in its file "
            f"metadata. Provenance is carried through this task verbatim, not "
            f"reconstructed from convention. Metadata found: "
            f"{sorted(source_metadata)}."
        )

    metadata: Dict[str, str] = {key: source_metadata[key] for key in CARRIED_METADATA_KEYS}
    metadata.update(
        {
            "pipeline_stage": "merging",
            "neural_quality_filtered": "true",
            "frames_dropped": str(frames_dropped),
            "source_artifact": str(depth_field_path),
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_contact_depth_field_table(retained, output_path, metadata=metadata)

    logger.info(
        f"{output_path.name}: dropped {frames_dropped} of {depth_frames.size} frames "
        f"({rows_dropped} of {len(table)} rows) excluded by neural-quality filtering, "
        f"{retained_frames.size} frames / {len(retained)} rows retained"
    )
    return output_path
