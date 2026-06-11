import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.pipeline.output_dirs import SPATIAL_EXTRACT_BOUNDARIES
from analysis.pipeline.shared_constants import IFF_METRICS, session_id_from_path
from analysis.receptive_field_mapping.rendering.neuron_type_colors import (
    SessionColorScheme,
    build_session_color_scheme,
)
from analysis.receptive_field_mapping.rendering.rf_boundary_comparison_renderer import (
    render_boundary_contour_overlay,
    render_boundary_metric_panels,
    render_session_gesture_heatmap,
)
from analysis.receptive_field_mapping.rendering.rf_population_map_renderer import (
    compute_uv_to_mm_scale,
    render_population_rf_circular_crop,
)

logger = logging.getLogger(__name__)

PANEL_METRICS = [
    'area_mm2',
    'perimeter_mm',
    'circularity',
    'pca_major_mm',
    'pca_minor_mm',
    'pca_aspect_ratio',
    'pca_orientation_deg',
    'mean_iff_on_contour',
]


def run_session_rf_boundary_comparison(
    session_configs: list[tuple[Path, Path]],
    output_dir: Path,
    force_processing: bool = False,
    iff_metric: str = "mean",
    neuron_summary_xlsx: Path | None = None,
    heatmap_space: str = "linear",
    cmap: str = "inferno",
) -> None:
    if iff_metric not in IFF_METRICS:
        raise ValueError(
            f"[Session RF Boundary Comparison] Invalid iff_metric {iff_metric!r}. "
            f"Expected one of {IFF_METRICS}."
        )
    if not session_configs:
        raise ValueError("[Session RF Boundary Comparison] session_configs is empty.")
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
    df, contour_data, centroid_data = _build_summary_dataframe(session_configs, iff_metric=iff_metric)

    csv_path = output_dir / 'session_rf_boundary_summary.csv'
    df.to_csv(csv_path, index=False)
    logger.info("[Session RF Boundary Comparison] wrote %s", csv_path.name)

    # Build per-session color scheme when neuron_summary_xlsx is provided.
    # When None, session_colors and neuron_type_legend remain None and the
    # renderer falls back to uniform steelblue (explicitly required by plan).
    session_colors: dict[str, str] | None = None
    neuron_type_legend: dict[str, str] | None = None
    if neuron_summary_xlsx is not None:
        session_ids = df['session_id'].unique().tolist()
        scheme: SessionColorScheme = build_session_color_scheme(session_ids, neuron_summary_xlsx)
        session_colors = scheme.session_color
        neuron_type_legend = scheme.type_color

    metric_limits = _global_metric_limits(df, PANEL_METRICS)
    uv_limits = _global_uv_limits(contour_data)

    for gtype, contours in contour_data.items():
        if not contours:
            continue
        centroids = centroid_data.get(gtype, {})
        render_boundary_contour_overlay(
            contours=contours,
            centroids=centroids,
            gesture_type=gtype,
            output_path=contour_overlays_dir / f'contour_overlay_{gtype}.png',
            uv_limits=uv_limits,
        )

    for gtype in df['gesture_type'].unique():
        gdf = df[df['gesture_type'] == gtype]
        render_boundary_metric_panels(
            df=gdf,
            gesture_type=gtype,
            metrics=PANEL_METRICS,
            output_path=metric_panels_dir / f'metric_panels_{gtype}.png',
            metric_limits=metric_limits,
            session_colors=session_colors,
            neuron_type_legend=neuron_type_legend,
        )

    for metric in PANEL_METRICS:
        render_session_gesture_heatmap(
            df=df,
            metric_name=metric,
            output_path=heatmap_dir / f'heatmap_{metric}.png',
            cluster_sessions=(df['session_id'].nunique() >= 3),
        )

    # --- Circular crops per session × gesture type ---
    circular_crops_root = output_dir / 'circular_crops'

    # Collect all grid_z values across sessions and gtypes to compute global vmin/vmax
    all_grid_z_values: list[np.ndarray] = []
    render_data_per_session: list[tuple[str, dict]] = []

    for sc_csv_path, sc_db_path in session_configs:
        sc_csv_path = Path(sc_csv_path)
        sc_db_path = Path(sc_db_path)
        session_id = session_id_from_path(sc_csv_path)
        npz_path = (
            sc_db_path / '4_analysed' / SPATIAL_EXTRACT_BOUNDARIES
            / f"iff_{iff_metric}" / session_id / f'{session_id}_population_response_fields.npz'
        )
        render_data = _load_heatmap_rendering_data_from_npz(npz_path)
        render_data_per_session.append((session_id, render_data))
        for gdata in render_data['per_gtype'].values():
            if gdata['centroid_uv'] is None:
                continue
            finite_z = gdata['grid_z'][np.isfinite(gdata['grid_z']) & (gdata['grid_z'] > 0)]
            if finite_z.size > 0:
                all_grid_z_values.append(finite_z)

    if all_grid_z_values:
        all_z_concat = np.concatenate(all_grid_z_values)
        global_vmax = float(all_z_concat.max())
        vmin_candidates = all_z_concat[all_z_concat > 0]
        global_vmin = float(vmin_candidates.min()) if vmin_candidates.size > 0 else global_vmax * 1e-3

        for session_id, render_data in render_data_per_session:
            session_crops_dir = circular_crops_root / session_id
            session_crops_dir.mkdir(parents=True, exist_ok=True)
            for gtype, gdata in render_data['per_gtype'].items():
                if gdata['centroid_uv'] is None:
                    continue
                crop_path = session_crops_dir / f'{session_id}_rf_circular_centroid_{gtype}.png'
                render_population_rf_circular_crop(
                    u_grid=gdata['grid_u'],
                    v_grid=gdata['grid_v'],
                    interp_grid=gdata['grid_z'],
                    forearm_uv=render_data['forearm_uv'],
                    forearm_V=render_data['forearm_V'],
                    forearm_faces=render_data['forearm_faces'],
                    center_uv=gdata['centroid_uv'],
                    radius_mm=50.0,
                    vmax=global_vmax,
                    vmin=global_vmin,
                    output_path=crop_path,
                    vertex_colors=render_data['slim_vertex_colors'],
                    heatmap_space=heatmap_space,
                    cmap=cmap,
                    dpi=300,
                )
                logger.info(
                    "[Session RF Boundary Comparison] %s: saved circular crop → %s",
                    session_id, crop_path.name,
                )
    else:
        logger.warning(
            "[Session RF Boundary Comparison] no valid heatmap data found across sessions "
            "— skipping circular crop rendering."
        )

    _write_sentinel(sentinel_path, n_sessions=len(session_configs))


def _global_metric_limits(
    df: pd.DataFrame,
    metrics: list[str],
) -> dict[str, tuple[float, float]]:
    """Per-metric (lo, hi) over every (session, gesture) row, with a 5% margin.

    Positive-only metrics are anchored at 0 so bar baselines stay visible.
    Metrics with no finite values raise — fail-fast per pipeline convention.
    """
    limits: dict[str, tuple[float, float]] = {}
    for m in metrics:
        col = df[m].to_numpy(dtype=float)
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            raise ValueError(
                f"_global_metric_limits: metric '{m}' has no finite values across "
                f"any session × gesture row — cannot determine shared limits."
            )
        lo = float(finite.min())
        hi = float(finite.max())
        if lo == hi:
            pad = abs(lo) * 0.05 if lo != 0.0 else 1.0
            limits[m] = (lo - pad, hi + pad)
            continue
        span = hi - lo
        margin = 0.05 * span
        lo_out = lo - margin if lo < 0.0 else 0.0
        hi_out = hi + margin if hi > 0.0 else 0.0
        limits[m] = (lo_out, hi_out)
    return limits


def _global_uv_limits(
    contour_data: dict[str, dict[str, np.ndarray]],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """(xlim, ylim) covering every contour across every session × gesture, with margin."""
    all_pts: list[np.ndarray] = []
    for gtype_contours in contour_data.values():
        for contour in gtype_contours.values():
            all_pts.append(contour)
    if not all_pts:
        return None
    pts = np.vstack(all_pts)
    u_min, u_max = float(pts[:, 0].min()), float(pts[:, 0].max())
    v_min, v_max = float(pts[:, 1].min()), float(pts[:, 1].max())
    u_margin = 0.05 * (u_max - u_min) if u_max > u_min else 1.0
    v_margin = 0.05 * (v_max - v_min) if v_max > v_min else 1.0
    return (u_min - u_margin, u_max + u_margin), (v_min - v_margin, v_max + v_margin)


def _load_boundary_metrics_from_npz(
    npz_path: Path,
    session_id: str,
) -> tuple[list[dict], dict[str, np.ndarray], dict[str, np.ndarray]]:
    npz = np.load(npz_path, allow_pickle=True)
    gesture_types = list(npz['gesture_types'])

    forearm_uv = npz['forearm_uv'].astype(np.float64)
    forearm_V = npz['forearm_V'].astype(np.float64)
    forearm_faces = npz['forearm_faces'].astype(np.int32)
    uv_to_mm = compute_uv_to_mm_scale(forearm_uv, forearm_V, forearm_faces)

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
            pca_major_mm = float(npz[f'boundary_pca_major_uv_{gtype}']) * uv_to_mm
            pca_minor_mm = float(npz[f'boundary_pca_minor_uv_{gtype}']) * uv_to_mm
            pca_orientation_deg = float(npz[f'boundary_pca_orientation_deg_{gtype}'])
            mean_iff_on_contour = float(npz[f'boundary_mean_iff_on_contour_{gtype}'])
            area_uv_mm2 = float(npz[f'boundary_area_uv_{gtype}']) * uv_to_mm ** 2
            perimeter_uv_mm = float(npz[f'boundary_perimeter_uv_{gtype}']) * uv_to_mm
            centroid_xyz = npz[f'boundary_centroid_xyz_{gtype}'].astype(np.float64)

            pca_aspect_ratio = pca_major_mm / pca_minor_mm if pca_minor_mm != 0.0 else float('nan')

            row = {
                'session_id': session_id,
                'gesture_type': gtype,
                'uv_to_mm_scale': uv_to_mm,
                'area_mm2': area_mm2,
                'perimeter_mm': perimeter_mm,
                'circularity': circularity,
                'pca_major_mm': pca_major_mm,
                'pca_minor_mm': pca_minor_mm,
                'pca_aspect_ratio': pca_aspect_ratio,
                'pca_orientation_deg': pca_orientation_deg,
                'mean_iff_on_contour': mean_iff_on_contour,
                'centroid_x_mm': float(centroid_xyz[0]),
                'centroid_y_mm': float(centroid_xyz[1]),
                'centroid_z_mm': float(centroid_xyz[2]),
                'area_uv_mm2': area_uv_mm2,
                'perimeter_uv_mm': perimeter_uv_mm,
            }
        else:
            row = {
                'session_id': session_id,
                'gesture_type': gtype,
                'uv_to_mm_scale': uv_to_mm,
                'area_mm2': float('nan'),
                'perimeter_mm': float('nan'),
                'circularity': float('nan'),
                'pca_major_mm': float('nan'),
                'pca_minor_mm': float('nan'),
                'pca_aspect_ratio': float('nan'),
                'pca_orientation_deg': float('nan'),
                'mean_iff_on_contour': float('nan'),
                'centroid_x_mm': float('nan'),
                'centroid_y_mm': float('nan'),
                'centroid_z_mm': float('nan'),
                'area_uv_mm2': float('nan'),
                'perimeter_uv_mm': float('nan'),
            }

        rows.append(row)

    if all_nan:
        logger.warning(
            "[Session RF Boundary Comparison] %s: no boundaries detected — all rows will be NaN",
            session_id,
        )

    return rows, contours_by_gtype, centroids_by_gtype


def _load_heatmap_rendering_data_from_npz(npz_path: Path) -> dict:
    """Load mesh and per-gesture heatmap arrays from a session NPZ for circular crop rendering.

    Returns a dict with:
    - ``forearm_uv`` (N, 2) float64
    - ``forearm_V``  (N, 3) float64
    - ``forearm_faces`` (F, 3) int32
    - ``slim_vertex_colors`` (N, 4) float64, or zeros array if key absent
    - ``per_gtype`` dict keyed by gesture type string, each value a dict with:
        - ``grid_u``, ``grid_v``, ``grid_z`` — interpolated heatmap meshgrids
        - ``centroid_uv`` — (2,) float64 centroid, or None if key absent for that gtype
    """
    npz = np.load(npz_path, allow_pickle=True)

    forearm_uv = npz['forearm_uv'].astype(np.float64)
    forearm_V = npz['forearm_V'].astype(np.float64)
    forearm_faces = npz['forearm_faces'].astype(np.int32)

    if 'slim_vertex_colors' in npz:
        slim_vertex_colors = npz['slim_vertex_colors'].astype(np.float64)
    else:
        slim_vertex_colors = np.zeros((forearm_V.shape[0], 4), dtype=np.float64)

    # Discover gesture types from grid_u_* keys
    npz_keys = set(npz.files)
    gtypes = [k[len('grid_u_'):] for k in npz_keys if k.startswith('grid_u_')]

    per_gtype: dict[str, dict] = {}
    for gtype in gtypes:
        centroid_key = f'boundary_centroid_uv_{gtype}'
        centroid_uv = (
            npz[centroid_key].astype(np.float64)
            if centroid_key in npz_keys
            else None
        )
        per_gtype[gtype] = {
            'grid_u': npz[f'grid_u_{gtype}'],
            'grid_v': npz[f'grid_v_{gtype}'],
            'grid_z': npz[f'grid_z_{gtype}'],
            'centroid_uv': centroid_uv,
        }

    return {
        'forearm_uv': forearm_uv,
        'forearm_V': forearm_V,
        'forearm_faces': forearm_faces,
        'slim_vertex_colors': slim_vertex_colors,
        'per_gtype': per_gtype,
    }


def _build_summary_dataframe(
    session_configs: list[tuple[Path, Path]],
    iff_metric: str = "mean",
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    all_rows: list[dict] = []
    contour_data: dict[str, dict[str, np.ndarray]] = {}
    centroid_data: dict[str, dict[str, np.ndarray]] = {}

    for csv_path, db_path in session_configs:
        csv_path = Path(csv_path)
        db_path = Path(db_path)

        session_id = session_id_from_path(csv_path)
        npz_path = (
            db_path / '4_analysed' / SPATIAL_EXTRACT_BOUNDARIES
            / f"iff_{iff_metric}" / session_id / f'{session_id}_population_response_fields.npz'
        )

        if not npz_path.exists():
            raise FileNotFoundError(
                f"[Session RF Boundary Comparison] {session_id}: NPZ not found at "
                f"{npz_path} — run spatial_extract_boundaries first."
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
        'uv_to_mm_scale': np.float64,
        'area_mm2': np.float64,
        'perimeter_mm': np.float64,
        'circularity': np.float64,
        'pca_major_mm': np.float64,
        'pca_minor_mm': np.float64,
        'pca_aspect_ratio': np.float64,
        'pca_orientation_deg': np.float64,
        'mean_iff_on_contour': np.float64,
        'centroid_x_mm': np.float64,
        'centroid_y_mm': np.float64,
        'centroid_z_mm': np.float64,
        'area_uv_mm2': np.float64,
        'perimeter_uv_mm': np.float64,
    })

    return df, contour_data, centroid_data


def _write_sentinel(sentinel_path: Path, n_sessions: int) -> None:
    sentinel_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sentinel_path, 'w') as f:
        json.dump({'done': True, 'n_sessions': n_sessions}, f)
    logger.info("[Session RF Boundary Comparison] wrote sentinel → %s", sentinel_path.name)
