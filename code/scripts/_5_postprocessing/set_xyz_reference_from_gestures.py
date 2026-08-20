"""Postprocessing step 4: PCA-calibrate the XYZ reference from gesture data.

Fits a two-step PCA on the session's tapping and stroking sticker trajectories,
then applies the resulting transform to every block CSV and to the forearm PLY,
so the session's coordinate axes are anatomically meaningful.

The per-vertex contact depth field
----------------------------------
This is a **replayable** stage: the transform is a
:class:`CalibrationResult` applicable to any ``(N, 3)`` array, so each block's
depth field sidecar is moved by the *same* result object the CSV was moved by,
in one vectorised call.  Four rules govern how:

1. **The fit never sees the field.**  Feeding contact vertices into
   :meth:`PCACalibrationEngine.compute_calibration` would change the coordinate
   system itself and invalidate every ``rf_center_origin.json`` already on disk.
   The field is a passenger here, not an input.
2. **The parquet loop is keyed off ``input_files``, never ``loaded_data``.**
   ``GestureDataLoader.load_and_segment`` returns ``None`` for a block that
   fails gesture segmentation, and that block is silently absent from
   ``loaded_data``.  A parquet loop keyed off that dict would inherit the silent
   drop and produce a session missing sidecars with nothing raised.
3. **The filename fork is explicit.**  This is the one stage that renames its
   outputs (``_pca-xyz``), so the sidecar suffix is a declared field of
   :class:`CalibrationConfig` beside ``output_csv_suffix``, and the name it
   produces is checked against the leaf's CSV/sidecar pairing rule — the next
   stage derives its inputs from that rule and would otherwise not find them.
4. **Depth is not a coordinate.**  ``signed_depth_mm`` measures penetration; a
   rotation does not re-measure it.  It is carried through bitwise and asserted
   to be, as is ``vertex_id`` — an index into a forearm this stage transforms in
   place, in file order, so the index stays valid and must not be renumbered.
"""

import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import open3d as o3d
from pathlib import Path
from typing import List, Sequence, Tuple, Dict, Optional
from dataclasses import dataclass, field
from sklearn.decomposition import PCA

# Import the idempotency check utility
from utils.should_process_task import should_process_task, clean_task_outputs

from postprocessing.xyz_reference_from_gestures import (
    PCACalibrationEngine,
    CalibrationResult,
    CalibrationVisualizer,
    Trajectory3DVisualizer
)
from postprocessing.depth_field_stage_io import (
    DEPTH_COLUMN,
    PIPELINE_STAGE_POSTPROCESSING,
    apply_pca_calibration_to_field,
    assert_row_counts_agree_with_csv,
    depth_field_path_for_csv,
)
from preprocessing.forearm_extraction.registration.csv_spatial_transformer import (
    parse_contact_points,
    serialize_contact_points,
)
from preprocessing.motion_analysis.tactile_quantification.io.contact_depth_field_io import (
    VERTEX_ID_COLUMN,
    read_contact_depth_field,
    write_contact_depth_field_table,
)

# Setup module-level logger
logger = logging.getLogger(__name__)

#: The space the depth field declares once this stage has calibrated it.  A
#: member of ``contact_depth_field_io.COORDINATE_SPACES``, which the writer
#: validates, so a typo here is a rejected write rather than a file whose
#: declared frame nothing recognises.
COORDINATE_SPACE_AFTER_PCA: str = "pca_calibrated"

# --- 1. Configuration Layer ---

@dataclass(frozen=True)
class CalibrationConfig:
    """Immutable configuration for gesture calibration."""
    col_type: str = 'type_metadata'
    col_touch_id: str = 'single_touch_id'
    col_contact_area: str = 'contact_area_metadata'

    val_tapping: str = 'tap'
    val_stroking: str = 'stroke'
    val_one_finger_tip: str = 'one finger tip'

    output_json_name: str = "pca-xyz_transformation-matrices.json"
    output_csv_suffix: str = "_pca-xyz.csv"

    #: The sidecar counterpart of ``output_csv_suffix``.  This stage is the only
    #: one that renames its outputs, and both artifacts must fork together or
    #: the next stage's CSV/sidecar pairing rule stops resolving.  Declared
    #: rather than derived inline: ``Path.stem`` on ``x.parquet`` is ``x``, so
    #: the same ``f"{p.stem}{suffix}"`` shape builds both names, and writing the
    #: suffix out once is what stops the two spellings drifting apart.
    output_parquet_suffix: str = "_pca-xyz.parquet"

    target_colors: Tuple[str, ...] = ('blue', 'yellow', 'green')

# --- 2. Data Ingestion Layer ---

class GestureDataLoader:
    """Handles file I/O and initial data segmentation."""

    def __init__(self, config: CalibrationConfig):
        self.config = config

    def load_and_segment(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Loads a CSV and performs initial cleaning.
        Returns None if required columns are missing or file is empty.
        """
        if not file_path.exists():
            logger.warning(f"Input file not found: {file_path}")
            return None

        try:
            # dynamically build columns for all target colors + metadata
            data_cols = []
            for color in self.config.target_colors:
                data_cols.append(f"sticker_{color}_position_x")
                data_cols.append(f"sticker_{color}_position_y")
                data_cols.append(f"sticker_{color}_position_z")

            meta_cols = [
                self.config.col_type,
                self.config.col_touch_id,
                self.config.col_contact_area,
            ]

            df = pd.read_csv(file_path, usecols=data_cols + meta_cols)

            # Drop rows missing the columns required for every subsequent operation.
            df_clean = df.dropna(subset=[self.config.col_type, self.config.col_touch_id])

            # single_touch_id == 0 → no valid touch (codebase convention).
            df_clean = df_clean[df_clean[self.config.col_touch_id] != 0]

            if df_clean.empty:
                return None
            return df_clean

        except ValueError as ve:
            logger.debug(f"Skipping {file_path.name}: {ve}")
            return None
        except Exception as e:
            logger.error(f"Error reading {file_path.name}: {e}")
            return None



def _extract_touch_data(
    touch_group: pd.DataFrame,
    config: CalibrationConfig,
) -> Optional[np.ndarray]:
    """
    Extracts mean-centred 3-D coordinates from one touch event.

    Sticker selection is driven by contact_area_metadata:
      - 'one finger tip'  →  blue sticker only
      - anything else     →  green + yellow stickers (concatenated)

    Each sticker's contribution is independently mean-centred before
    concatenation so that absolute position differences between stickers
    do not bias the PCA.

    Returns an (N, 3) ndarray, or None if no valid rows exist.
    """
    contact_type_series = touch_group[config.col_contact_area].dropna()
    if contact_type_series.empty:
        return None
    contact_type = contact_type_series.iloc[0]

    colors_to_use = ['blue'] if contact_type == config.val_one_finger_tip else ['green', 'yellow']

    centered_arrays = []
    for color in colors_to_use:
        cx, cy, cz = (f"sticker_{color}_position_{ax}" for ax in 'xyz')
        if not all(c in touch_group.columns for c in [cx, cy, cz]):
            continue
        coords = touch_group[[cx, cy, cz]].dropna().values
        if len(coords) == 0:
            continue
        centered_arrays.append(coords - coords.mean(axis=0))

    return np.vstack(centered_arrays) if centered_arrays else None


# --- 4b. The per-vertex contact depth field ---

def _resolve_input_parquets(
    input_files: Sequence[Path], input_parquets: Optional[Sequence[Path]]
) -> List[Path]:
    """Return the sidecar path for every input CSV, proven to exist.

    Args:
        input_files: The block CSVs this stage will calibrate.
        input_parquets: Explicit sidecar paths, index-aligned with
            *input_files*.  When ``None`` they are derived with
            ``depth_field_path_for_csv`` — the sidecar sits beside its CSV, and
            keeping the rule in one place stops the stage and its caller
            disagreeing about which file belongs to which block.

    Returns:
        One existing sidecar path per input CSV, in the same order.

    Raises:
        ValueError: If *input_parquets* is supplied with a different length.
        FileNotFoundError: If any sidecar is absent.  The depth field is a hard
            input of postprocessing; calibrating without it would leave a block
            with no path to the terminal artifact and nothing would notice.
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
            "required input of this stage — it is written beside the block CSV "
            "by project_contacts_onto_forearm. Re-run that stage for the "
            "affected block(s) rather than calibrating without it."
        )
    return resolved


def _output_parquet_path(
    input_parquet: Path, output_dir: Path, config: CalibrationConfig
) -> Path:
    """Name this stage's sidecar output for one block.

    ``Path.stem`` on ``x.parquet`` is ``x``, so the sidecar name is built with
    the same ``f"{stem}{suffix}"`` shape as the CSV name — the ``_pca-xyz`` fork
    happens once, to both artifacts, from two declared constants.
    """
    return output_dir / f"{input_parquet.stem}{config.output_parquet_suffix}"


def _assert_output_names_pair(output_csv: Path, output_parquet: Path) -> None:
    """Raise unless *output_parquet* is the sidecar *output_csv* resolves to.

    ``depth_field_path_for_csv`` is how the next stage finds its sidecar inputs.
    If the two naming rules ever disagreed, ``center_on_receptive_field`` would
    look for a file this stage never wrote and the failure would surface a stage
    late, pointing at the wrong code.
    """
    expected = depth_field_path_for_csv(output_csv)
    if expected.name != output_parquet.name:
        raise ValueError(
            f"This stage writes {output_csv.name!r} and {output_parquet.name!r}, "
            f"but the CSV/sidecar pairing rule resolves that CSV to "
            f"{expected.name!r}. The next stage derives its sidecar inputs from "
            "that rule, so the two suffixes must fork together: check "
            "CalibrationConfig.output_csv_suffix against "
            "CalibrationConfig.output_parquet_suffix."
        )


def _assert_depth_and_vertex_preserved(
    original: pd.DataFrame, moved: pd.DataFrame, *, parquet_name: str
) -> None:
    """Raise unless the measurement and the vertex identity survived bit for bit.

    ``signed_depth_mm`` is how far the hand penetrated the forearm.  A PCA
    calibration rotates and translates both bodies together and cannot change
    it.  ``vertex_id`` is a row index into the session's reference forearm, which
    this stage transforms **in place, in file order, with no reordering or count
    change** — so index *i* still names the same physical vertex afterwards and
    must not be recomputed or renumbered.

    The leaf module preserves both columns by construction; asserting it at the
    stage boundary anyway is what makes a silently re-derived value distinguishable
    from a measured one in the output file.
    """
    before_depth = original[DEPTH_COLUMN].to_numpy()
    after_depth = moved[DEPTH_COLUMN].to_numpy()
    if after_depth.dtype != before_depth.dtype or not np.array_equal(
        after_depth, before_depth
    ):
        raise ValueError(
            f"{parquet_name}: PCA calibration altered {DEPTH_COLUMN!r} (dtype "
            f"{before_depth.dtype} -> {after_depth.dtype}). Penetration depth is "
            "a measurement and is invariant under a rigid transform; only "
            "coordinates may move at this stage."
        )

    if VERTEX_ID_COLUMN not in original.columns:
        raise ValueError(
            f"{parquet_name}: the depth field carries no {VERTEX_ID_COLUMN!r} "
            "column. It is assigned by project_contacts_onto_forearm, which runs "
            "before this stage, so a field without one has not been projected — "
            "re-run that stage rather than calibrating an unaddressed field."
        )
    before_id = original[VERTEX_ID_COLUMN].to_numpy()
    after_id = moved[VERTEX_ID_COLUMN].to_numpy()
    if after_id.dtype != before_id.dtype or not np.array_equal(after_id, before_id):
        raise ValueError(
            f"{parquet_name}: PCA calibration altered {VERTEX_ID_COLUMN!r} (dtype "
            f"{before_id.dtype} -> {after_id.dtype}). The forearm is transformed "
            "in place and in file order, so vertex i is the same physical vertex "
            "before and after; the index must be carried through untouched."
        )


def _write_calibrated_field(
    input_parquet: Path,
    output_parquet: Path,
    output_csv: Path,
    calibration: CalibrationResult,
) -> None:
    """Calibrate a sidecar, restamp its space, and check it against the CSV.

    Args:
        input_parquet: The block's depth field as the projection stage left it.
        output_parquet: Destination in ``blocks_pca_calibrated/``.
        output_csv: The calibrated CSV **this stage just wrote** for the same
            block — the only CSV the row-count check is meaningful against.
        calibration: The very ``CalibrationResult`` applied to the CSV.  Never a
            result re-fitted from the JSON or from the sidecar's own points: a
            second fit would define a different coordinate system from the same
            session's data, silently.

    Raises:
        FileNotFoundError: If *output_csv* was not written — which happens when
            gesture segmentation dropped the block from ``loaded_data``.
        ValueError: If the transform disturbed ``signed_depth_mm`` or
            ``vertex_id``, or if the written field and the written CSV disagree
            about any frame's contact-point count.
    """
    if not output_csv.exists():
        raise FileNotFoundError(
            f"{output_csv} was not written, but its depth field sidecar "
            f"({input_parquet.name}) is about to be. The block was dropped from "
            "the calibration's loaded_data — gesture segmentation found no "
            "usable rows, or the CSV write failed — and a session that is "
            "missing one artifact of a pair must say so rather than carry a "
            "half-processed block forward."
        )

    table, source_metadata = read_contact_depth_field(input_parquet)
    moved = apply_pca_calibration_to_field(table, calibration)
    _assert_depth_and_vertex_preserved(table, moved, parquet_name=input_parquet.name)

    # Provenance is carried through verbatim — schema version, reference PLY,
    # vertex count, epsilon — because none of it changed.  Two keys do: the
    # coordinates are in a new space, and the file has now demonstrably been
    # through postprocessing rather than stopping at merging.
    metadata: Dict[str, str] = dict(source_metadata)
    metadata["coordinate_space"] = COORDINATE_SPACE_AFTER_PCA
    metadata["pipeline_stage"] = PIPELINE_STAGE_POSTPROCESSING

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    write_contact_depth_field_table(moved, output_parquet, metadata=metadata)
    assert_row_counts_agree_with_csv(moved, output_csv)


# --- 5. Orchestrator ---

def calibrate_pca_xyz(
    input_files: List[Path],
    output_dir: Path,
    forearm_ply_path: Path,
    forearm_output_dir: Path,
    *,
    input_parquets: Optional[Sequence[Path]] = None,
    force_processing: bool = False,
    monitor: bool = False,
    monitor_segment: bool = False,
) -> Tuple[List[Path], Path, Path, List[Path]]:
    """
    Orchestrates the global PCA analysis pipeline and applies the resulting
    transform to the forearm PLY.

    Each block's per-vertex contact depth field is moved by the same
    ``CalibrationResult`` as its CSV, in one vectorised call, and lands beside it
    under the same ``_pca-xyz`` fork.

    Args:
        input_files: CSV block files to transform.
        output_dir: Destination for transformed CSVs and calibration JSON.
        forearm_ply_path: Source forearm PLY (from forearm_source/).
        forearm_output_dir: Destination for the PCA-calibrated forearm PLY.
        input_parquets: Per-block contact depth field sidecars, index-aligned
            with *input_files*.  When omitted they are derived from the CSV
            paths — the sidecar sits beside its CSV.
        force_processing: Re-run even if all outputs are up-to-date.
        monitor: Display global aggregate visualisation after calibration.
        monitor_segment: Display per-segment visualisation during calibration.

    Returns:
        A 4-tuple: ``(generated_csv_files, output_dir, forearm_output_ply_path,
        generated_parquet_files)``.  The stage's ``outputs`` list binds
        positionally, so the first three slots keep their existing meaning and
        the fourth is appended.

    Raises:
        FileNotFoundError: If a block's depth field sidecar is missing, or if a
            block was dropped from the calibration but its sidecar was not.
        ValueError: If *input_parquets* is misaligned with *input_files*, if the
            calibration disturbs ``signed_depth_mm`` or ``vertex_id``, or if any
            written pair of artifacts disagrees about a frame's contact-point
            count.
    """
    logger.info(f"[{output_dir.name}] Starting Global PCA pipeline on {len(input_files)} files.")

    config = CalibrationConfig()

    resolved_parquets = _resolve_input_parquets(input_files, input_parquets)

    # Prepare expected outputs for idempotency
    json_output_path = output_dir / config.output_json_name
    expected_output_files = [output_dir / f"{p.stem}{config.output_csv_suffix}" for p in input_files]
    expected_output_parquets = [
        _output_parquet_path(p, output_dir, config) for p in resolved_parquets
    ]
    for csv_out, parquet_out in zip(expected_output_files, expected_output_parquets):
        _assert_output_names_pair(csv_out, parquet_out)

    # session_id is encoded in the forearm PLY stem: "{session_id}_forearm"
    forearm_stem = forearm_ply_path.stem  # e.g. "ST13-03_forearm"
    forearm_output_path = forearm_output_dir / f"{forearm_stem}.ply"

    # Both artifacts sit on both sides of the idempotency check, at this stage's
    # own boundary — the session — so deleting either one regenerates the pair
    # and they can never be produced out of step with each other.
    all_check_outputs = (
        expected_output_files
        + expected_output_parquets
        + [json_output_path, forearm_output_path]
    )

    if not should_process_task(
        input_paths=input_files + resolved_parquets + [forearm_ply_path],
        output_paths=all_check_outputs,
        force=force_processing
    ):
        logger.info(f"[{output_dir.name}] Task up-to-date. Skipping.")
        return (
            expected_output_files,
            output_dir,
            forearm_output_path,
            expected_output_parquets,
        )
    clean_task_outputs(all_check_outputs)
    output_dir.mkdir(parents=True, exist_ok=True)
    forearm_output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Components
    loader = GestureDataLoader(config)

    # 1. Load Data (into memory map)
    loaded_data: Dict[Path, pd.DataFrame] = {}

    logger.info("Phase 1: Ingesting data...")

    tapping_segments = []
    stroking_segments = []

    for input_path in input_files:
        df = loader.load_and_segment(input_path)
        if df is not None:
            loaded_data[input_path] = df

            df_tap = df[df[config.col_type] == config.val_tapping]
            df_stroke = df[df[config.col_type] == config.val_stroking]

            for _, group in df_tap.groupby(config.col_touch_id):
                data = _extract_touch_data(group, config)
                if data is not None:
                    tapping_segments.append(data)

            for _, group in df_stroke.groupby(config.col_touch_id):
                data = _extract_touch_data(group, config)
                if data is not None:
                    stroking_segments.append(data)

    if not tapping_segments or not stroking_segments:
        logger.error("Insufficient data for calibration (missing tap or stroke segments).")
        return [], output_dir, forearm_output_path, []

    # 2. Compute Calibration
    logger.info("Computing PCA Matrices...")
    all_tapping = np.vstack(tapping_segments)
    all_stroking = np.vstack(stroking_segments)

    calib_result = PCACalibrationEngine.compute_calibration(all_tapping, all_stroking)

    # 3. Save Calibration
    with open(json_output_path, 'w') as f:
        json.dump(calib_result.to_dict(), f, indent=4)

    # 4. Monitor
    if monitor_segment:
        try:
            logger.info("Starting Interactive Segment Visualization.")
            logger.info("Close the visualization window to proceed to the next segment.")

            # --- Visualize Tapping Segments (Step 1 Aligned) ---
            logger.info(">>> Visualizing Tapping Segments (Step 1: Z-Aligned)")
            for i, seg in enumerate(tapping_segments):
                transformed_seg = PCACalibrationEngine.apply_step1_transform(seg, calib_result)
                logger.info(f"Displaying Tapping Segment {i+1}/{len(tapping_segments)}")
                viz = Trajectory3DVisualizer(transformed_seg)
                viz.show()

            # --- Visualize Stroking Segments (Full Aligned) ---
            logger.info(">>> Visualizing Stroking Segments (Final: XY-Aligned)")
            for i, seg in enumerate(stroking_segments):
                transformed_seg = PCACalibrationEngine.apply_full_transform(seg, calib_result)
                logger.info(f"Displaying Stroking Segment {i+1}/{len(stroking_segments)}")
                viz = Trajectory3DVisualizer(transformed_seg)
                viz.show()
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            pass

    if monitor:
        try:
            logger.info("Displaying Global Aggregate Summary...")
            CalibrationVisualizer.visualize(all_tapping, all_stroking, calib_result, output_dir.name)
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            pass

    # 5. Apply Transformation to loaded data and Save
    logger.info("Phase 2: Applying transformation to [Blue, Yellow, Green] and saving files...")
    generated_files = []

    for input_path, df in loaded_data.items():
        output_path = output_dir / f"{input_path.stem}{config.output_csv_suffix}"

        try:
            # 1. Read the full original file to preserve original structure (and NaNs)
            df_out = pd.read_csv(input_path)

            # 2. Apply transformation to each color defined in config
            for color in config.target_colors:
                c_x = f"sticker_{color}_position_x"
                c_y = f"sticker_{color}_position_y"
                c_z = f"sticker_{color}_position_z"

                if not all(c in df_out.columns for c in [c_x, c_y, c_z]):
                    logger.warning(f"Skipping color '{color}' for {input_path.name}: Columns missing.")
                    continue

                coords = df_out[[c_x, c_y, c_z]].to_numpy(dtype=np.float64)
                mask = ~np.isnan(coords).any(axis=1)
                if mask.any():
                    coords_transformed = coords.copy()
                    coords_transformed[mask] = PCACalibrationEngine.apply_full_transform(
                        coords[mask], calib_result
                    )
                    df_out[c_x] = coords_transformed[:, 0]
                    df_out[c_y] = coords_transformed[:, 1]
                    df_out[c_z] = coords_transformed[:, 2]

            # 3. Apply transformation to contact_location_x/y/z
            loc_cols = ["contact_location_x", "contact_location_y", "contact_location_z"]
            if all(c in df_out.columns for c in loc_cols):
                loc = df_out[loc_cols].to_numpy(dtype=np.float64)
                mask = ~np.isnan(loc).any(axis=1)
                if mask.any():
                    loc_transformed = loc.copy()
                    loc_transformed[mask] = PCACalibrationEngine.apply_full_transform(
                        loc[mask], calib_result
                    )
                    for i, col in enumerate(loc_cols):
                        df_out[col] = loc_transformed[:, i]

            # 4. Apply transformation to contact_points
            if "contact_points" in df_out.columns:
                new_contact_points = []
                for cell in df_out["contact_points"]:
                    pts = parse_contact_points(cell)
                    if pts:
                        arr = np.asarray(pts, dtype=np.float64).copy()
                        arr = PCACalibrationEngine.apply_full_transform(arr, calib_result)
                        new_contact_points.append(
                            serialize_contact_points([tuple(row) for row in arr.tolist()])
                        )
                    else:
                        new_contact_points.append(cell)
                df_out["contact_points"] = new_contact_points

            df_out.to_csv(output_path, index=False)
            generated_files.append(output_path)

        except Exception as e:
            logger.error(f"Failed to save processed file {input_path.name}: {e}")

    # 5b. Apply the same calibration to each block's contact depth field.
    #
    # Keyed off ``input_files``, deliberately, and never off ``loaded_data``:
    # a block whose gesture segmentation returned None is silently absent from
    # that dict, and a loop over it would produce a session missing sidecars
    # with nothing raised. Iterating the inputs instead turns the drop into the
    # explicit FileNotFoundError raised by _write_calibrated_field, which names
    # the CSV that was never written.
    logger.info("Phase 2b: Applying transformation to the contact depth fields...")
    generated_parquets = []

    for input_parquet, output_parquet, output_csv in zip(
        resolved_parquets, expected_output_parquets, expected_output_files
    ):
        # The same calib_result object the CSVs were transformed with — never a
        # re-fit, never a reload of the JSON just written.
        _write_calibrated_field(
            input_parquet, output_parquet, output_csv, calib_result
        )
        generated_parquets.append(output_parquet)
        logger.info(f"Wrote PCA-calibrated depth field: {output_parquet.name}")

    # 6. Apply PCA transform to forearm PLY
    logger.info(f"Phase 3: Applying PCA transform to forearm PLY: {forearm_ply_path.name}")

    pcd = o3d.io.read_point_cloud(str(forearm_ply_path))
    vertices = np.asarray(pcd.points)
    if len(vertices) == 0:
        raise ValueError(
            f"Forearm PLY has no points: {forearm_ply_path}. "
            "Cannot apply PCA transform to an empty point cloud."
        )

    transformed = PCACalibrationEngine.apply_full_transform(vertices.copy(), calib_result)

    out_pcd = o3d.geometry.PointCloud()
    out_pcd.points = o3d.utility.Vector3dVector(np.round(transformed, 1))
    if pcd.has_colors():
        out_pcd.colors = pcd.colors
    o3d.io.write_point_cloud(str(forearm_output_path), out_pcd)

    logger.info(f"[{output_dir.name}] Wrote PCA-calibrated forearm PLY: {forearm_output_path.name}")

    return generated_files, output_dir, forearm_output_path, generated_parquets


# Backward-compatibility alias (used during transition; callers should migrate to calibrate_pca_xyz)
set_xyz_reference_from_gestures = calibrate_pca_xyz
