"""Postprocessing step 0: Apply ICP registration transforms to merged CSVs.

Reads per-block CSVs from ``blocks_merged/``, applies the pre-computed
4x4 ICP registration transform to the spatial columns (overwriting originals),
and writes the results to ``blocks_registered/``.

Single-forearm sessions (no ``registration_transforms.json``) pass through
unchanged with files copied as-is.

The per-vertex contact depth field
----------------------------------
Each block carries a second artifact beside its CSV — the Space-1
``*_contact_depth_field.parquet`` sidecar produced in preprocessing and reduced
to the neurally usable frames during merging.  This stage is the first of the
five postprocessing stages that move contact coordinates, so it is the first
that must move the sidecar too.  If it did not, the two artifacts would sit in
the same directory describing the same contact vertices in **different
coordinate spaces**, with nothing on disk to say so.

Three rules govern how it is moved here:

1. **The same schedule object the CSV uses.**  The schedule is built from the
   CSV's ``max_frame``.  The parquet holds rows only for *contacting* frames,
   so a schedule rebuilt from it would end at a different frame and could
   segment the block differently — the two artifacts would then diverge
   silently.  The schedule is therefore built once, from the CSV, and applied
   to both.
2. **Passthrough is all-or-nothing.**  There are two branches that write the
   CSV through untransformed: a session with no transforms file at all, and a
   block with no applicable transform.  In both, the sidecar is copied byte for
   byte, which keeps ``coordinate_space = "kinect_space_1"`` — the correct
   declaration, because those points genuinely did not move.  Restamping it
   would be a lie; transforming it would put it in a space its CSV is not in.
3. **Depth is not a coordinate.**  ``signed_depth_mm`` measures how far the
   hand penetrated the forearm.  A rigid transform moves both bodies together
   and cannot change it, so it is carried through bitwise and asserted to be.
"""
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.should_process_task import should_process_task, clean_task_outputs
from preprocessing.forearm_extraction import (
    ForearmRegistrator,
    get_transform_schedule,
    transform_spatial_columns_scheduled,
)
from preprocessing.motion_analysis.tactile_quantification.io.contact_depth_field_io import (
    read_contact_depth_field,
    write_contact_depth_field_table,
)
from postprocessing.depth_field_stage_io import (
    DEPTH_COLUMN,
    apply_transform_schedule_to_field,
    assert_row_counts_agree_with_csv,
)
from primary_processing import KinectConfig

logger = logging.getLogger(__name__)


#: The space the depth field declares once this stage has transformed it.  It is
#: a member of ``contact_depth_field_io.COORDINATE_SPACES``, which the writer
#: validates, so a typo here is a rejected write rather than a file whose
#: declared frame nothing recognises.
COORDINATE_SPACE_AFTER_ICP: str = "icp_registered"

#: The two halves of a block's filename.  The sidecar sits beside its CSV and
#: shares its stem up to the suffix: ``<block>_merged_data.csv`` alongside
#: ``<block>_contact_depth_field.parquet``.  Both names are produced by
#: ``merging_pipeline_neuron_to_kinect_auto._resolve_paths``.
_MERGED_CSV_SUFFIX: str = "_merged_data.csv"
_DEPTH_FIELD_SUFFIX: str = "_contact_depth_field.parquet"


def depth_field_path_for_csv(csv_path: Path) -> Path:
    """Return the contact-depth-field sidecar that belongs to *csv_path*.

    The sidecar always sits in the same directory as the CSV it describes and
    differs only in suffix.  Every postprocessing stage writes its CSV under the
    input's own name, so this derivation holds at every stage, not just this one.

    Args:
        csv_path: A block's ``*_merged_data.csv``.

    Returns:
        The sibling ``*_contact_depth_field.parquet`` path.  Existence is not
        checked here.

    Raises:
        ValueError: If *csv_path* does not end in ``_merged_data.csv``.  The
            name is the join between the two artifacts; guessing at an
            unrecognised one would pair a CSV with the wrong sidecar.
    """
    csv_path = Path(csv_path)
    if not csv_path.name.endswith(_MERGED_CSV_SUFFIX):
        raise ValueError(
            f"{csv_path.name!r} does not end in {_MERGED_CSV_SUFFIX!r}, so the "
            "contact depth field sidecar that belongs to it cannot be named. "
            "The two artifacts are paired by filename stem; refusing to guess."
        )
    stem = csv_path.name[: -len(_MERGED_CSV_SUFFIX)]
    return csv_path.with_name(f"{stem}{_DEPTH_FIELD_SUFFIX}")


def _resolve_input_parquets(
    input_files: Sequence[Path], input_parquets: Optional[Sequence[Path]]
) -> List[Path]:
    """Return the sidecar path for every input CSV, proven to exist.

    Args:
        input_files: The block CSVs this stage will process.
        input_parquets: Explicit sidecar paths, index-aligned with
            *input_files*.  When ``None`` they are derived with
            :func:`depth_field_path_for_csv` — the same rule the caller would
            apply, kept in one place so the stage and its workflow cannot
            disagree about which file belongs to which block.

    Returns:
        One existing sidecar path per input CSV, in the same order.

    Raises:
        ValueError: If *input_parquets* is supplied with a different length.
        FileNotFoundError: If any sidecar is absent.  The depth field is a hard
            input of postprocessing, not an optional extra: a block without one
            cannot produce the terminal artifact, and skipping it would leave a
            gap nothing downstream would notice.
    """
    if input_parquets is None:
        resolved = [depth_field_path_for_csv(path) for path in input_files]
    else:
        resolved = [Path(path) for path in input_parquets]
        if len(resolved) != len(input_files):
            raise ValueError(
                f"input_parquets has {len(resolved)} entr(ies) but input_files "
                f"has {len(input_files)}. The two lists are matched by position, "
                "one sidecar per block CSV, so a length mismatch would pair a "
                "block with another block's depth field."
            )

    missing = [path for path in resolved if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Contact depth field sidecar(s) missing: "
            f"{[str(path) for path in missing]}. The per-vertex depth field is a "
            "required input of postprocessing — it is produced by the merging "
            "pipeline's filter_contact_depth_field_by_neural_quality task and "
            "must sit beside the block CSV it describes. Re-run merging for the "
            "affected block(s) rather than processing without it."
        )
    return resolved


def _assert_depth_preserved(
    original: pd.DataFrame, moved: pd.DataFrame, *, parquet_name: str
) -> None:
    """Raise unless ``signed_depth_mm`` survived the transform bit for bit.

    A rigid transform moves the hand and the forearm together, so the distance
    the hand penetrated the surface is invariant under it.  The leaf module
    preserves the column by construction; this asserts it at the stage boundary
    anyway, because a silently re-derived depth would be indistinguishable from
    a measured one in the output file.
    """
    before = original[DEPTH_COLUMN].to_numpy()
    after = moved[DEPTH_COLUMN].to_numpy()
    if after.dtype != before.dtype or not np.array_equal(after, before):
        raise ValueError(
            f"{parquet_name}: the ICP transform altered {DEPTH_COLUMN!r} "
            f"(dtype {before.dtype} -> {after.dtype}). Penetration depth is a "
            "measurement and is invariant under a rigid transform; only "
            "coordinates may move at this stage."
        )


def _write_transformed_field(
    input_parquet: Path,
    output_parquet: Path,
    output_csv: Path,
    schedule,
) -> None:
    """Move a sidecar by *schedule*, restamp its space, and check it against the CSV.

    Args:
        input_parquet: The Space-1 sidecar for this block.
        output_parquet: Destination in ``blocks_registered/``.
        output_csv: The registered CSV **this stage just wrote** for the same
            block — the only CSV the row-count check is meaningful against.
        schedule: The very schedule object applied to the CSV, built from the
            CSV's frame range.

    Raises:
        ValueError: If the transform disturbed ``signed_depth_mm``, or if the
            written field and the written CSV disagree about any frame's
            contact-point count.
    """
    table, source_metadata = read_contact_depth_field(input_parquet)
    moved = apply_transform_schedule_to_field(table, schedule)
    _assert_depth_preserved(table, moved, parquet_name=input_parquet.name)

    # Provenance is carried through verbatim; only the declared space changes,
    # because only the coordinates changed.
    metadata: Dict[str, str] = dict(source_metadata)
    metadata["coordinate_space"] = COORDINATE_SPACE_AFTER_ICP

    write_contact_depth_field_table(moved, output_parquet, metadata=metadata)
    assert_row_counts_agree_with_csv(moved, output_csv)


def _copy_field_unchanged(
    input_parquet: Path, output_parquet: Path, output_csv: Path
) -> None:
    """Copy a sidecar through a passthrough branch and check it against the CSV.

    A byte copy, not a read-modify-write: the points did not move, so the file's
    declared ``coordinate_space`` — ``kinect_space_1`` — is still the truth, and
    copying the bytes is the only way to guarantee no key was restamped on the
    way past.

    Raises:
        ValueError: If the copied field and the CSV this stage wrote disagree
            about any frame's contact-point count.
    """
    shutil.copyfile(input_parquet, output_parquet)
    table, _ = read_contact_depth_field(output_parquet)
    assert_row_counts_agree_with_csv(table, output_csv)


def apply_icp_registration(
    input_files: List[Path],
    session_configs: List["KinectConfig"],
    output_dir: Path,
    *,
    input_parquets: Optional[Sequence[Path]] = None,
    force_processing: bool = False,
) -> Tuple[List[Path], List[Path]]:
    """Apply ICP registration transforms to merged CSVs and their depth fields.

    For multi-forearm sessions, loads ``registration_transforms.json`` from
    ``forearm_pointclouds/`` and applies the appropriate per-block 4x4 rigid
    transform to spatial columns in place.  Single-forearm sessions (no
    transforms file) have their files copied to *output_dir* unchanged.

    Each block's per-vertex contact depth field is moved by the same schedule as
    its CSV, or copied through in exactly the cases the CSV is copied through.

    Args:
        input_files: Per-block merged CSVs (from ``blocks_filtered/``).
        session_configs: KinectConfig objects (one per block, same session).
        output_dir: Destination directory (``blocks_registered/``).
        input_parquets: Per-block contact depth field sidecars, index-aligned
            with *input_files*.  When omitted they are derived from the CSV
            paths — the sidecar sits beside its CSV — which is the same set of
            paths a caller would resolve.
        force_processing: Re-run even if outputs are up-to-date.

    Returns:
        ``(csv_paths, parquet_paths)``: the output CSVs in *output_dir*, and
        their depth field sidecars, index-aligned with each other.

    Raises:
        FileNotFoundError: If a block's depth field sidecar is missing.
        ValueError: If *input_parquets* is misaligned with *input_files*, if the
            transform disturbs ``signed_depth_mm``, or if any written pair of
            artifacts disagrees about a frame's contact-point count.
    """
    resolved_parquets = _resolve_input_parquets(input_files, input_parquets)

    # Idempotency check: output filenames mirror input filenames, for both
    # artifacts.  The boundary is the session, matching this stage's existing
    # behaviour — a stale or absent sidecar re-runs the whole session, so the
    # two artifacts can never be regenerated out of step with each other.
    expected_outputs = [output_dir / f.name for f in input_files]
    expected_parquet_outputs = [output_dir / p.name for p in resolved_parquets]

    if not should_process_task(
        input_paths=[*input_files, *resolved_parquets],
        output_paths=[*expected_outputs, *expected_parquet_outputs],
        force=force_processing,
    ):
        logger.info("[%s] ICP registration up-to-date. Skipping.", output_dir.name)
        return expected_outputs, expected_parquet_outputs
    clean_task_outputs([*expected_outputs, *expected_parquet_outputs])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load registration transforms (keyed by session from first config)
    first_config = session_configs[0]
    forearm_pc_dir = first_config.session_processed_output_dir / "forearm_pointclouds"
    transforms_path = forearm_pc_dir / f"{first_config.session_id}_registration_transforms.json"
    transforms_data = ForearmRegistrator.load_transforms(transforms_path)

    if transforms_data is None:
        logger.info(
            "[%s] No registration transforms found (%s). "
            "Single-forearm session — copying files unchanged.",
            output_dir.name,
            transforms_path.name,
        )
        for input_path, input_parquet in zip(input_files, resolved_parquets):
            out_csv = output_dir / input_path.name
            pd.read_csv(input_path).to_csv(out_csv, index=False)
            _copy_field_unchanged(
                input_parquet, output_dir / input_parquet.name, out_csv
            )
        return expected_outputs, expected_parquet_outputs

    # Multi-forearm: apply scheduled transforms per block
    output_paths = []
    output_parquet_paths = []
    for input_path, input_parquet, config in zip(
        input_files, resolved_parquets, session_configs
    ):
        out_path   = output_dir / input_path.name
        out_parquet = output_dir / input_parquet.name
        video_stem = f"{config.session_id}_semicontrolled_{config.block_id}".replace("block-order-", "block-order")
        df         = pd.read_csv(input_path)

        max_frame = int(df["frame_index"].dropna().max()) if "frame_index" in df.columns and df["frame_index"].notna().any() else 0

        schedule = get_transform_schedule(transforms_data["transforms"], video_stem, max_frame)

        if not schedule:
            logger.warning(
                "[%s] No applicable transform for '%s' "
                "(no preceding snapshot). Passing through unchanged.",
                output_dir.name,
                video_stem,
            )
            df.to_csv(out_path, index=False)
            output_paths.append(out_path)
            # The CSV was not moved, so the sidecar must not be either: it keeps
            # kinect_space_1, which is what it is still in.
            _copy_field_unchanged(input_parquet, out_parquet, out_path)
            output_parquet_paths.append(out_parquet)
            continue

        logger.info(
            "[%s] Applying %d transform segment(s) for '%s'.",
            output_dir.name,
            len(schedule),
            video_stem,
        )
        transform_spatial_columns_scheduled(df, schedule).to_csv(out_path, index=False)
        output_paths.append(out_path)
        # The same schedule object, and therefore the same segment boundaries
        # the CSV's max_frame produced.
        _write_transformed_field(input_parquet, out_parquet, out_path, schedule)
        output_parquet_paths.append(out_parquet)
        logger.info("[%s] Wrote registered CSV: %s", output_dir.name, out_path.name)
        logger.info(
            "[%s] Wrote registered depth field: %s", output_dir.name, out_parquet.name
        )

    return output_paths, output_parquet_paths
