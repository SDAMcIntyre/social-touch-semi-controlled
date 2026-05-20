import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.pipeline.shared_constants import session_id_from_path
from analysis.receptive_field_mapping.rendering.rf_boundary_comparison_renderer import (
    render_boundary_contour_overlay,
    render_boundary_metric_panels,
    render_session_gesture_heatmap,
)

logger = logging.getLogger(__name__)

PANEL_METRICS = [
    'area_mm2',
    'perimeter_mm',
    'circularity',
    'pca_major_uv',
    'pca_minor_uv',
    'pca_aspect_ratio',
    'pca_orientation_deg',
    'mean_iff_on_contour',
]


def run_session_rf_boundary_comparison(
    session_configs: list[tuple[Path, Path]],
    force_processing: bool = False,
) -> None:
    if not session_configs:
        raise ValueError("[Session RF Boundary Comparison] session_configs is empty.")

    _, db_path = session_configs[0]
    db_path = Path(db_path)
    output_dir = db_path / '4_analysed' / 'session_rf_boundary_comparison'
    sentinel_path = output_dir / 'session_rf_boundary_comparison_done.json'

    if sentinel_path.exists() and not force_processing:
        logger.info("[Session RF Boundary Comparison] up-to-date, skipping.")
        return

    contour_overlays_dir = output_dir / 'contour_overlays'
    metric_panels_dir = output_dir / 'metric_panels'
    heatmap_dir = output_dir / 'session_gesture_heatmaps'

    for d in (output_dir, contour_overlays_dir, metric_panels_dir, heatmap_dir):
        d.mkdir(parents=True, exist_ok=True)

    logger.info("[Session RF Boundary Comparison] building summary DataFrame from %d sessions...", len(session_configs))
    df, contour_data, centroid_data = _build_summary_dataframe(session_configs)

    csv_path = output_dir / 'session_rf_boundary_summary.csv'
    df.to_csv(csv_path, index=False)
    logger.info("[Session RF Boundary Comparison] wrote %s", csv_path.name)

    for gtype, contours in contour_data.items():
        if not contours:
            continue
        centroids = centroid_data.get(gtype, {})
        render_boundary_contour_overlay(
            contours=contours,
            centroids=centroids,
            gesture_type=gtype,
            output_path=contour_overlays_dir / f'contour_overlay_{gtype}.png',
        )

    for gtype in df['gesture_type'].unique():
        gdf = df[df['gesture_type'] == gtype]
        render_boundary_metric_panels(
            df=gdf,
            gesture_type=gtype,
            metrics=PANEL_METRICS,
            output_path=metric_panels_dir / f'metric_panels_{gtype}.png',
        )

    for metric in PANEL_METRICS:
        render_session_gesture_heatmap(
            df=df,
            metric_name=metric,
            output_path=heatmap_dir / f'heatmap_{metric}.png',
            cluster_sessions=(df['session_id'].nunique() >= 3),
        )

    _write_sentinel(sentinel_path, n_sessions=len(session_configs))


def _load_boundary_metrics_from_npz(
    npz_path: Path,
    session_id: str,
) -> tuple[list[dict], dict[str, np.ndarray], dict[str, np.ndarray]]:
    npz = np.load(npz_path, allow_pickle=True)
    gesture_types = list(npz['gesture_types'])

    rows: list[dict] = []
    contours_by_gtype: dict[str, np.ndarray] = {}
    centroids_by_gtype: dict[str, np.ndarray] = {}

    all_nan = True

    for gtype in gesture_types:
        contour_key = f'boundary_contour_uv_{gtype}'
        has_boundary = contour_key in npz

        if has_boundary:
            all_nan = False

            contour = npz[contour_key].astype(np.float64)
            centroid = npz[f'boundary_centroid_uv_{gtype}'].astype(np.float64)
            contours_by_gtype[gtype] = contour
            centroids_by_gtype[gtype] = centroid

            area_mm2 = float(npz[f'boundary_area_xyz_mm2_{gtype}'])
            perimeter_mm = float(npz[f'boundary_perimeter_xyz_mm_{gtype}'])
            circularity = float(npz[f'boundary_circularity_{gtype}'])
            pca_major_uv = float(npz[f'boundary_pca_major_uv_{gtype}'])
            pca_minor_uv = float(npz[f'boundary_pca_minor_uv_{gtype}'])
            pca_orientation_deg = float(npz[f'boundary_pca_orientation_deg_{gtype}'])
            mean_iff_on_contour = float(npz[f'boundary_mean_iff_on_contour_{gtype}'])
            area_uv = float(npz[f'boundary_area_uv_{gtype}'])
            perimeter_uv = float(npz[f'boundary_perimeter_uv_{gtype}'])
            centroid_xyz = npz[f'boundary_centroid_xyz_{gtype}'].astype(np.float64)

            pca_aspect_ratio = pca_major_uv / pca_minor_uv if pca_minor_uv != 0.0 else float('nan')

            row = {
                'session_id': session_id,
                'gesture_type': gtype,
                'area_mm2': area_mm2,
                'perimeter_mm': perimeter_mm,
                'circularity': circularity,
                'pca_major_uv': pca_major_uv,
                'pca_minor_uv': pca_minor_uv,
                'pca_aspect_ratio': pca_aspect_ratio,
                'pca_orientation_deg': pca_orientation_deg,
                'mean_iff_on_contour': mean_iff_on_contour,
                'centroid_x_mm': float(centroid_xyz[0]),
                'centroid_y_mm': float(centroid_xyz[1]),
                'centroid_z_mm': float(centroid_xyz[2]),
                'area_uv': area_uv,
                'perimeter_uv': perimeter_uv,
            }
        else:
            row = {
                'session_id': session_id,
                'gesture_type': gtype,
                'area_mm2': float('nan'),
                'perimeter_mm': float('nan'),
                'circularity': float('nan'),
                'pca_major_uv': float('nan'),
                'pca_minor_uv': float('nan'),
                'pca_aspect_ratio': float('nan'),
                'pca_orientation_deg': float('nan'),
                'mean_iff_on_contour': float('nan'),
                'centroid_x_mm': float('nan'),
                'centroid_y_mm': float('nan'),
                'centroid_z_mm': float('nan'),
                'area_uv': float('nan'),
                'perimeter_uv': float('nan'),
            }

        rows.append(row)

    if all_nan:
        logger.warning(
            "[Session RF Boundary Comparison] %s: no boundaries detected — all rows will be NaN",
            session_id,
        )

    return rows, contours_by_gtype, centroids_by_gtype


def _build_summary_dataframe(
    session_configs: list[tuple[Path, Path]],
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    all_rows: list[dict] = []
    contour_data: dict[str, dict[str, np.ndarray]] = {}
    centroid_data: dict[str, dict[str, np.ndarray]] = {}

    for csv_path, db_path in session_configs:
        csv_path = Path(csv_path)
        db_path = Path(db_path)

        session_id = session_id_from_path(csv_path)
        npz_path = (
            db_path / '4_analysed' / 'population_response_fields'
            / session_id / f'{session_id}_population_response_fields.npz'
        )

        if not npz_path.exists():
            raise FileNotFoundError(
                f"[Session RF Boundary Comparison] {session_id}: NPZ not found at "
                f"{npz_path} — run extract_population_rf_response_field_boundaries first."
            )

        logger.info("[Session RF Boundary Comparison] loading %s", session_id)
        rows, contours_by_gtype, centroids_by_gtype = _load_boundary_metrics_from_npz(
            npz_path, session_id
        )
        all_rows.extend(rows)

        for gtype, contour in contours_by_gtype.items():
            contour_data.setdefault(gtype, {})[session_id] = contour
        for gtype, centroid in centroids_by_gtype.items():
            centroid_data.setdefault(gtype, {})[session_id] = centroid

    df = pd.DataFrame(all_rows)
    df = df.astype({
        'session_id': str,
        'gesture_type': str,
        'area_mm2': np.float64,
        'perimeter_mm': np.float64,
        'circularity': np.float64,
        'pca_major_uv': np.float64,
        'pca_minor_uv': np.float64,
        'pca_aspect_ratio': np.float64,
        'pca_orientation_deg': np.float64,
        'mean_iff_on_contour': np.float64,
        'centroid_x_mm': np.float64,
        'centroid_y_mm': np.float64,
        'centroid_z_mm': np.float64,
        'area_uv': np.float64,
        'perimeter_uv': np.float64,
    })

    return df, contour_data, centroid_data


def _write_sentinel(sentinel_path: Path, n_sessions: int) -> None:
    sentinel_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sentinel_path, 'w') as f:
        json.dump({'done': True, 'n_sessions': n_sessions}, f)
    logger.info("[Session RF Boundary Comparison] wrote sentinel → %s", sentinel_path.name)
