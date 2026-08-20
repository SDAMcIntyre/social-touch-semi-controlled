"""Postprocessing step 5: Center coordinate origin on the receptive field.

For each session, estimates the receptive field (RF) center from the
projected block CSVs using selectivity-weighted DBSCAN clustering, then
translates all spatial columns and the forearm PLY by the RF center offset
so that the coordinate origin sits at the RF center.

If RF estimation fails (no valid DBSCAN cluster), a warning is logged and
input data is passed through unchanged.

The per-vertex contact depth field
----------------------------------
This is the last of the five spatial stages, so ``blocks_rf_centered/`` is where
the depth field sidecar comes to rest — the terminal artifact the whole
propagation exists to produce.  It is a **replayable** stage: the transform is
one 4x4 translation, so each block's sidecar is moved by the *same* matrix its
CSV was moved by.

Three rules govern how:

1. **The RF estimate never sees the field.**  ``_compute_rf_center`` reads the
   CSVs only.  Feeding contact vertices — let alone depth-weighting them — into
   the estimate would move the origin of this space and invalidate every
   ``rf_center_origin.json`` already on disk.  The field is a passenger.
2. **Passthrough copies the sidecar too.**  When RF estimation fails the CSVs
   are copied through untranslated; the sidecar is copied with them, byte for
   byte, which keeps ``coordinate_space = "pca_calibrated"`` — the correct
   declaration, because those points genuinely did not move.  A read-modify-write
   here could restamp a key in passing; a byte copy cannot.
3. **Depth and vertex identity are not coordinates.**  ``signed_depth_mm`` is a
   penetration measurement and a translation does not re-measure it;
   ``vertex_id`` indexes a forearm this stage translates in place, in file order,
   so index *i* still names the same physical vertex afterwards.  Both are
   carried through bitwise and asserted to be.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d
import pandas as pd

from postprocessing.depth_field_stage_io import (
    DEPTH_COLUMN,
    PIPELINE_STAGE_POSTPROCESSING,
    apply_rigid_transform_to_field,
    assert_row_counts_agree_with_csv,
    depth_field_path_for_csv,
)
from postprocessing.receptive_field.rf_clustering import (
    GroupedSpatialData,
    RFMappingEngine,
    SelectivityDBSCANConfig,
)
from preprocessing.forearm_extraction.registration.csv_spatial_transformer import (
    parse_contact_points,
    transform_spatial_columns_in_place,
)
from preprocessing.motion_analysis.tactile_quantification.io.contact_depth_field_io import (
    VERTEX_ID_COLUMN,
    read_contact_depth_field,
    write_contact_depth_field_table,
)
from utils.should_process_task import clean_task_outputs, should_process_task

logger = logging.getLogger(__name__)

#: The space the depth field declares once this stage has translated it — the
#: terminal space of the whole pipeline.  A member of
#: ``contact_depth_field_io.COORDINATE_SPACES``, which the writer validates, so
#: a typo here is a rejected write rather than a file whose declared frame
#: nothing recognises.
COORDINATE_SPACE_AFTER_RF_CENTERING: str = "rf_centered"


def _compute_rf_center(
    block_csvs: List[Path],
    config: SelectivityDBSCANConfig,
) -> Tuple[Optional[np.ndarray], dict]:
    """Estimate the RF center from block CSVs via DBSCAN + top spike-count filter.

    Reads all block CSVs, accumulates spike and total contact-point counts
    per unique 3D position, computes per-point selectivity (spike/total),
    and clusters the high-selectivity points with DBSCAN.  The RF center is
    the spike-count-weighted centroid of the top 10% highest spike-count
    points within the dominant (largest) cluster.

    Args:
        block_csvs: Block-level CSVs from ``blocks_contact_projected/``.
        config: Selectivity threshold and DBSCAN parameters.

    Returns:
        Tuple of:
        - RF center as a ``(3,)`` float64 array, or ``None`` on failure.
        - Metadata dict suitable for JSON serialisation.
    """
    gsd = GroupedSpatialData(group_label="all")
    required_cols = {"single_touch_id", "Nerve_spike", "contact_points"}

    for csv_path in block_csvs:
        if not csv_path.exists():
            logger.warning("Block CSV not found, skipping: %s", csv_path)
            continue

        try:
            available = set(pd.read_csv(csv_path, nrows=0).columns)
            missing = required_cols - available
            if missing:
                logger.warning(
                    "Skipping %s: missing columns %s", csv_path.name, missing
                )
                continue
            df = pd.read_csv(csv_path, usecols=list(required_cols))
        except Exception:
            logger.exception("Error reading %s", csv_path.name)
            continue

        touch_ids = df["single_touch_id"].to_numpy()
        spikes = df["Nerve_spike"].to_numpy()
        raw_points = df["contact_points"]

        for idx in range(len(df)):
            if touch_ids[idx] == 0:
                continue
            pts = parse_contact_points(raw_points.iat[idx])
            if not pts:
                continue
            gsd.total_counts.update(pts)
            if spikes[idx] == 1:
                gsd.spike_counts.update(pts)

    if not gsd.total_counts:
        logger.warning(
            "No contact points found in block CSVs — cannot estimate RF center."
        )
        return None, {"status": "no_contact_points", "rf_center": None}

    selectivity = RFMappingEngine.compute_selectivity(gsd.spike_counts, gsd.total_counts)
    rf_result = RFMappingEngine.cluster_receptive_field(
        selectivity_scores=selectivity,
        config=config,
        group_label="all",
    )

    if not rf_result.clusters:
        logger.warning(
            "DBSCAN found no valid clusters (evaluated %d points, %d above threshold).",
            rf_result.total_points_evaluated,
            rf_result.points_above_threshold,
        )
        return None, {
            "status": "no_cluster_found",
            "rf_center": None,
            "total_points_evaluated": rf_result.total_points_evaluated,
            "points_above_threshold": rf_result.points_above_threshold,
        }

    TOP_FRACTION = 0.1

    dominant = max(rf_result.clusters, key=lambda c: c.point_count)

    spike_counts_arr = np.array([
        gsd.spike_counts.get(tuple(pt), 0)
        for pt in dominant.points
    ])

    n_top = max(1, int(np.ceil(len(spike_counts_arr) * TOP_FRACTION)))
    top_indices = np.argsort(spike_counts_arr)[-n_top:]

    top_points = dominant.points[top_indices]
    top_weights = spike_counts_arr[top_indices]

    if top_weights.sum() > 0:
        rf_center = np.average(top_points, weights=top_weights, axis=0)
    else:
        rf_center = np.mean(top_points, axis=0)

    return rf_center.astype(np.float64), {
        "status": "ok",
        "rf_center": rf_center.tolist(),
        "dominant_cluster_id": int(dominant.cluster_id),
        "dominant_cluster_point_count": int(dominant.point_count),
        "dominant_cluster_mean_selectivity": float(dominant.mean_selectivity),
        "rf_center_top_fraction": TOP_FRACTION,
        "rf_center_top_n_points": int(n_top),
        "n_clusters_found": len(rf_result.clusters),
        "total_points_evaluated": rf_result.total_points_evaluated,
        "points_above_threshold": rf_result.points_above_threshold,
    }


def _build_translation_matrix(offset: np.ndarray) -> np.ndarray:
    """Build a 4x4 matrix that translates points by ``-offset``.

    Applying this matrix moves the RF center (at *offset*) to the origin.
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = -offset
    return T


def _translate_single_csv(
    input_csv: Path,
    output_csv: Path,
    translation_matrix: np.ndarray,
) -> None:
    """Apply a 4x4 translation to all spatial columns in one block CSV.

    Args:
        input_csv: Source block CSV.
        output_csv: Destination CSV (parent created if needed).
        translation_matrix: 4x4 translation; identity is a no-op.
    """
    df = pd.read_csv(input_csv)
    transform_spatial_columns_in_place(df, translation_matrix)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)


def _translate_forearm_ply(
    input_ply: Path,
    output_ply: Path,
    offset: np.ndarray,
) -> None:
    """Translate a forearm PLY by ``-offset`` (RF center → origin).

    Args:
        input_ply: Source PCA-calibrated forearm PLY.
        output_ply: Destination PLY (parent created if needed).
        offset: RF center in mm; vertices are shifted by ``-offset``.
    """
    pcd = o3d.io.read_point_cloud(str(input_ply))
    pcd.translate(-offset, relative=True)
    pcd.points = o3d.utility.Vector3dVector(np.round(np.asarray(pcd.points), 1))
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_ply), pcd)


def _resolve_input_parquets(
    input_files: Sequence[Path], input_parquets: Optional[Sequence[Path]]
) -> List[Path]:
    """Return the sidecar path for every input CSV, proven to exist.

    Args:
        input_files: The block CSVs this stage will translate.
        input_parquets: Explicit sidecar paths, index-aligned with
            *input_files*.  When ``None`` they are derived with
            ``depth_field_path_for_csv``, which resolves the ``_pca-xyz`` fork
            the previous stage applied to both artifacts alike.

    Returns:
        One existing sidecar path per input CSV, in the same order.

    Raises:
        ValueError: If *input_parquets* is supplied with a different length.
        FileNotFoundError: If any sidecar is absent.  The depth field is a hard
            input of postprocessing; a block that reaches this stage without one
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
            "required input of this stage — it is written beside the block CSV "
            "by calibrate_pca_xyz. Re-run that stage for the affected block(s) "
            "rather than centring without it."
        )
    return resolved


def _assert_depth_and_vertex_preserved(
    original: pd.DataFrame, moved: pd.DataFrame, *, parquet_name: str
) -> None:
    """Raise unless the measurement and the vertex identity survived bit for bit.

    ``signed_depth_mm`` is how far the hand penetrated the forearm; translating
    the whole scene does not change it.  ``vertex_id`` is a row index into the
    session's reference forearm, which this stage translates **in place, in file
    order, with no reordering or count change** — index *i* still names the same
    physical vertex afterwards and must not be recomputed or renumbered.

    The leaf module preserves both columns by construction; asserting it at the
    stage boundary anyway is what makes a silently re-derived value
    distinguishable from a measured one in the output file.
    """
    before_depth = original[DEPTH_COLUMN].to_numpy()
    after_depth = moved[DEPTH_COLUMN].to_numpy()
    if after_depth.dtype != before_depth.dtype or not np.array_equal(
        after_depth, before_depth
    ):
        raise ValueError(
            f"{parquet_name}: RF centring altered {DEPTH_COLUMN!r} (dtype "
            f"{before_depth.dtype} -> {after_depth.dtype}). Penetration depth is "
            "a measurement and is invariant under a translation; only "
            "coordinates may move at this stage."
        )

    if VERTEX_ID_COLUMN not in original.columns:
        raise ValueError(
            f"{parquet_name}: the depth field carries no {VERTEX_ID_COLUMN!r} "
            "column. It is assigned by project_contacts_onto_forearm, two stages "
            "earlier, so a field without one has not been projected — re-run "
            "that stage rather than centring an unaddressed field."
        )
    before_id = original[VERTEX_ID_COLUMN].to_numpy()
    after_id = moved[VERTEX_ID_COLUMN].to_numpy()
    if after_id.dtype != before_id.dtype or not np.array_equal(after_id, before_id):
        raise ValueError(
            f"{parquet_name}: RF centring altered {VERTEX_ID_COLUMN!r} (dtype "
            f"{before_id.dtype} -> {after_id.dtype}). The forearm is translated "
            "in place and in file order, so vertex i is the same physical vertex "
            "before and after; the index must be carried through untouched."
        )


def _translate_single_field(
    input_parquet: Path,
    output_parquet: Path,
    output_csv: Path,
    translation_matrix: np.ndarray,
) -> None:
    """Translate a sidecar, restamp its space, and check it against the CSV.

    Args:
        input_parquet: The block's depth field as the PCA stage left it.
        output_parquet: Destination in ``blocks_rf_centered/``.
        output_csv: The RF-centred CSV **this stage just wrote** for the same
            block — the only CSV the row-count check is meaningful against.
        translation_matrix: The very 4x4 matrix applied to the CSV.

    Raises:
        ValueError: If the translation disturbed ``signed_depth_mm`` or
            ``vertex_id``, or if the written field and the written CSV disagree
            about any frame's contact-point count.
    """
    table, source_metadata = read_contact_depth_field(input_parquet)
    moved = apply_rigid_transform_to_field(table, translation_matrix)
    _assert_depth_and_vertex_preserved(table, moved, parquet_name=input_parquet.name)

    # Provenance is carried through verbatim — schema version, reference PLY,
    # vertex count, epsilon — because none of it changed. Only the declared
    # space does, because only the coordinates did.
    metadata: Dict[str, str] = dict(source_metadata)
    metadata["coordinate_space"] = COORDINATE_SPACE_AFTER_RF_CENTERING
    metadata["pipeline_stage"] = PIPELINE_STAGE_POSTPROCESSING

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    write_contact_depth_field_table(moved, output_parquet, metadata=metadata)
    assert_row_counts_agree_with_csv(moved, output_csv)


def _copy_field_unchanged(
    input_parquet: Path, output_parquet: Path, output_csv: Path
) -> None:
    """Copy a sidecar through the no-cluster passthrough and check it.

    A byte copy, not a read-modify-write: the points did not move, so the file's
    declared ``coordinate_space`` — ``pca_calibrated`` — is still the truth, and
    copying the bytes is the only way to guarantee nothing was restamped on the
    way past.  The CSVs beside it are copied through for the same reason.

    ``copy2`` rather than ``copyfile``, matching the CSV copy two lines above it
    in the caller: the timestamp is copied as well as the bytes, so a re-run of
    an unchanged session reproduces the previous run's outputs exactly, mtimes
    included, instead of making the sidecar look newer than the CSV it belongs
    to.

    Raises:
        ValueError: If the copied field and the CSV this stage produced disagree
            about any frame's contact-point count.
    """
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_parquet, output_parquet)
    table, _ = read_contact_depth_field(output_parquet)
    assert_row_counts_agree_with_csv(table, output_csv)


def center_on_receptive_field(
    input_files: List[Path],
    forearm_ply_path: Path,
    output_dir: Path,
    forearm_output_dir: Path,
    rf_origin_path: Path,
    *,
    input_parquets: Optional[Sequence[Path]] = None,
    force_processing: bool = False,
) -> Tuple[List[Path], List[Path]]:
    """Center all spatial data on the receptive field coordinate origin.

    Estimates the RF center from *input_files* (block CSVs from
    ``blocks_contact_projected/``) using selectivity-weighted DBSCAN, then
    translates all spatial columns in each CSV and the forearm PLY so that
    the RF center becomes ``(0, 0, 0)``.

    When RF estimation fails (no cluster above threshold), a warning is logged,
    ``rf_origin_path`` is written with ``status: "no_cluster_found"``, and
    the input files are copied to *output_dir* unchanged so downstream tasks
    can still run.

    Args:
        input_files: Block CSVs from ``blocks_contact_projected/``.
        forearm_ply_path: PCA-calibrated forearm PLY (stage 3 output).
        output_dir: Output directory for RF-centered block CSVs
            (``blocks_rf_centered/``).
        forearm_output_dir: Output directory for the translated forearm PLY
            (``forearm_rf_centered/``).
        rf_origin_path: Destination for the per-session
            ``rf_center_origin.json`` metadata file.
        input_parquets: Per-block contact depth field sidecars, index-aligned
            with *input_files*.  When omitted they are derived from the CSV
            paths — the sidecar sits beside its CSV.
        force_processing: Re-run even if outputs are up-to-date.

    Returns:
        ``(csv_paths, parquet_paths)``: the output CSVs in *output_dir* and
        their depth field sidecars, index-aligned with each other.

    Raises:
        FileNotFoundError: If a block's depth field sidecar is missing.
        ValueError: If *input_parquets* is misaligned with *input_files*, if the
            translation disturbs ``signed_depth_mm`` or ``vertex_id``, or if any
            written pair of artifacts disagrees about a frame's contact-point
            count.
    """
    if not input_files:
        logger.warning("center_on_receptive_field: no input files provided.")
        return [], []

    resolved_parquets = _resolve_input_parquets(input_files, input_parquets)

    output_csvs = [output_dir / f.name for f in input_files]
    output_parquets = [output_dir / p.name for p in resolved_parquets]
    forearm_output_ply = (
        forearm_output_dir / forearm_ply_path.name
        if forearm_ply_path.exists()
        else None
    )

    # Both artifacts sit on both sides of the idempotency check, at this stage's
    # own boundary — the session — so deleting either one regenerates the pair
    # and they can never be produced out of step with each other.
    all_outputs: List[Path] = (
        list(output_csvs) + list(output_parquets) + [rf_origin_path]
    )
    if forearm_output_ply:
        all_outputs.append(forearm_output_ply)

    input_paths: List[Path] = list(input_files) + list(resolved_parquets)
    if forearm_ply_path.exists():
        input_paths.append(forearm_ply_path)

    if not should_process_task(
        input_paths=input_paths,
        output_paths=all_outputs,
        force=force_processing,
    ):
        logger.info("RF centering up-to-date. Skipping session.")
        return output_csvs, output_parquets

    clean_task_outputs(all_outputs)

    # --- Compute RF center ---
    dbscan_config = SelectivityDBSCANConfig()
    rf_center, metadata = _compute_rf_center(input_files, dbscan_config)

    if rf_center is None:
        logger.warning(
            "RF center estimation failed (status: %s). Passing data through unchanged.",
            metadata.get("status"),
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        for src, dst in zip(input_files, output_csvs):
            shutil.copy2(src, dst)
        # The CSVs were not translated, so the sidecars must not be either: they
        # keep pca_calibrated, which is the space they are still in.
        for src_parquet, dst_parquet, dst_csv in zip(
            resolved_parquets, output_parquets, output_csvs
        ):
            _copy_field_unchanged(src_parquet, dst_parquet, dst_csv)
        if forearm_output_ply and forearm_ply_path.exists():
            forearm_output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(forearm_ply_path, forearm_output_ply)
        rf_origin_path.parent.mkdir(parents=True, exist_ok=True)
        rf_origin_path.write_text(json.dumps(metadata, indent=2))
        return output_csvs, output_parquets

    logger.info(
        "RF center estimated at [%.2f, %.2f, %.2f] mm "
        "(cluster %d, %d points, mean selectivity %.3f).",
        rf_center[0], rf_center[1], rf_center[2],
        metadata["dominant_cluster_id"],
        metadata["dominant_cluster_point_count"],
        metadata["dominant_cluster_mean_selectivity"],
    )

    T = _build_translation_matrix(rf_center)

    # --- Translate block CSVs and their depth fields ---
    output_dir.mkdir(parents=True, exist_ok=True)
    for src, dst, src_parquet, dst_parquet in zip(
        input_files, output_csvs, resolved_parquets, output_parquets
    ):
        _translate_single_csv(src, dst, T)
        logger.info("RF-centered block CSV: %s", dst.name)
        # The same 4x4 matrix, applied to the same points, in the same order.
        _translate_single_field(src_parquet, dst_parquet, dst, T)
        logger.info("RF-centered depth field: %s", dst_parquet.name)

    # --- Translate forearm PLY ---
    if forearm_ply_path.exists() and forearm_output_ply:
        _translate_forearm_ply(forearm_ply_path, forearm_output_ply, rf_center)
        logger.info("RF-centered forearm PLY: %s", forearm_output_ply.name)
    else:
        logger.warning(
            "Forearm PLY not found (%s); skipping PLY translation.", forearm_ply_path
        )

    # --- Write RF origin metadata ---
    rf_origin_path.parent.mkdir(parents=True, exist_ok=True)
    rf_origin_path.write_text(json.dumps(metadata, indent=2))
    logger.info("Wrote RF origin metadata: %s", rf_origin_path.name)

    return output_csvs, output_parquets
