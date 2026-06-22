"""Extract 1D RF cross-section profiles and boundary crossings from population heatmap grids."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.pipeline.output_dirs import SPATIAL_EXTRACT_BOUNDARIES
from analysis.pipeline.shared_constants import IFF_METRICS, session_id_from_path
from analysis.receptive_field_mapping.rendering.rf_profile_renderer import (
    render_profile_strip_with_boundary,
    render_representative_profiles,
)

logger = logging.getLogger(__name__)


def find_boundary_u_crossings(
    contour_uv: np.ndarray | None,
    v_target: float,
) -> np.ndarray:
    """Return sorted U-intercepts where a closed UV contour crosses a constant-V scanline."""
    if contour_uv is None or len(contour_uv) == 0:
        return np.empty(0, dtype=np.float64)

    closed = np.concatenate([contour_uv, contour_uv[:1]], axis=0)
    u1 = closed[:-1, 0]
    v1 = closed[:-1, 1]
    u2 = closed[1:, 0]
    v2 = closed[1:, 1]

    # One-sided strict inequality avoids double-counting at vertices
    straddles = (v1 <= v_target) & (v2 > v_target) | (v2 <= v_target) & (v1 > v_target)
    if not np.any(straddles):
        return np.empty(0, dtype=np.float64)

    u1_s = u1[straddles]
    v1_s = v1[straddles]
    u2_s = u2[straddles]
    v2_s = v2[straddles]

    u_cross = u1_s + (v_target - v1_s) * (u2_s - u1_s) / (v2_s - v1_s)
    return np.sort(u_cross)


def _extract_profiles_for_gesture(
    session_id: str,
    gtype: str,
    npz: np.lib.npyio.NpzFile,
) -> pd.DataFrame:
    """Extract 150 constant-V profiles and boundary crossings for one gesture type."""
    grid_u = npz[f'grid_u_{gtype}']
    grid_v = npz[f'grid_v_{gtype}']
    grid_z = npz[f'grid_z_{gtype}']

    contour_key = f'boundary_contour_uv_{gtype}'
    contour_uv = npz[contour_key] if contour_key in npz else None

    n_cols = grid_v.shape[1]
    rows: list[dict] = []

    for j in range(n_cols):
        iff_values = grid_z[:, j]
        v_value = float(grid_v[0, j])

        valid_mask = ~np.isnan(iff_values)
        n_valid = int(np.sum(valid_mask))

        if n_valid == 0:
            profile_max = float('nan')
            profile_mean = float('nan')
        else:
            profile_max = float(np.nanmax(iff_values))
            profile_mean = float(np.nanmean(iff_values))

        crossings = find_boundary_u_crossings(contour_uv, v_value)
        n_crossings = len(crossings)

        if n_crossings >= 1:
            boundary_u_left = float(crossings[0])
            boundary_u_right = float(crossings[-1])
        else:
            boundary_u_left = float('nan')
            boundary_u_right = float('nan')

        if n_crossings >= 2:
            boundary_width = boundary_u_right - boundary_u_left
        else:
            boundary_width = float('nan')

        boundary_u_all = ';'.join(f'{u:.6f}' for u in crossings) if n_crossings > 0 else ''

        rows.append({
            'session_id': session_id,
            'gesture_type': gtype,
            'v_index': j,
            'v_value_mm': v_value,
            'n_valid': n_valid,
            'profile_max': profile_max,
            'profile_mean': profile_mean,
            'n_boundary_crossings': n_crossings,
            'boundary_u_left_mm': boundary_u_left,
            'boundary_u_right_mm': boundary_u_right,
            'boundary_width_mm': boundary_width,
            'boundary_u_all': boundary_u_all,
        })

    return pd.DataFrame(rows)


def run_rf_profile_extraction(
    session_configs: list[tuple[Path, Path]],
    output_dir: Path,
    force_processing: bool = False,
    iff_metric: str = "mean",
) -> None:
    """Extract 1D RF profiles and boundary crossings from population heatmap grids."""
    if iff_metric not in IFF_METRICS:
        raise ValueError(
            f"[RF Profile Extraction] Invalid iff_metric {iff_metric!r}. "
            f"Expected one of {IFF_METRICS}."
        )
    if not session_configs:
        raise ValueError("[RF Profile Extraction] session_configs is empty.")

    for csv_path, database_path in session_configs:
        csv_path = Path(csv_path)
        database_path = Path(database_path)
        session_id = session_id_from_path(csv_path)

        session_output_dir = output_dir / f"iff_{iff_metric}" / session_id
        sentinel = session_output_dir / f'{session_id}_rf_profiles_done.json'

        if sentinel.exists() and not force_processing:
            print(f"[RF Profile Extraction] {session_id}: up-to-date, skipping.")
            continue

        npz_path = (
            database_path / '4_analysed' / SPATIAL_EXTRACT_BOUNDARIES
            / f"iff_{iff_metric}" / session_id
            / f'{session_id}_population_response_fields.npz'
        )
        if not npz_path.exists():
            raise FileNotFoundError(
                f"[RF Profile Extraction] {session_id}: NPZ not found at "
                f"{npz_path} — run spatial_extract_boundaries first."
            )

        print(f"[RF Profile Extraction] {session_id}: processing...")

        npz = np.load(npz_path, allow_pickle=True)
        gesture_types = list(npz['gesture_types'])

        session_output_dir.mkdir(parents=True, exist_ok=True)
        produced_csvs: list[str] = []
        produced_pngs: list[str] = []

        for gtype in gesture_types:
            grid_key = f'grid_u_{gtype}'
            if grid_key not in npz:
                logger.warning(
                    "[RF Profile Extraction] %s: grid key %r missing from NPZ, skipping gesture %r.",
                    session_id, grid_key, gtype,
                )
                continue

            df = _extract_profiles_for_gesture(session_id, gtype, npz)
            csv_out = session_output_dir / f'{session_id}_rf_profiles_{gtype}.csv'
            df.to_csv(csv_out, index=False)
            produced_csvs.append(str(csv_out))
            logger.info(
                "[RF Profile Extraction] %s/%s: wrote %d profiles to %s",
                session_id, gtype, len(df), csv_out.name,
            )

            contour_key = f'boundary_contour_uv_{gtype}'
            contour_uv = npz[contour_key] if contour_key in npz else None

            crossings_per_v: dict[int, np.ndarray] = {}
            for _, row in df.iterrows():
                if row['n_boundary_crossings'] > 0:
                    crossings_per_v[int(row['v_index'])] = np.array(
                        [float(x) for x in row['boundary_u_all'].split(';')]
                    )

            strip_path = session_output_dir / f'{session_id}_rf_profile_strip_{gtype}.png'
            render_profile_strip_with_boundary(
                grid_u=npz[f'grid_u_{gtype}'],
                grid_v=npz[f'grid_v_{gtype}'],
                grid_z=npz[f'grid_z_{gtype}'],
                contour_uv=contour_uv,
                crossings_per_v=crossings_per_v,
                output_path=strip_path,
                title=f"{session_id} | {gtype} | RF Profile Strip",
            )
            produced_pngs.append(str(strip_path))

            rep_path = session_output_dir / f'{session_id}_rf_profiles_representative_{gtype}.png'
            render_representative_profiles(
                grid_u=npz[f'grid_u_{gtype}'],
                grid_v=npz[f'grid_v_{gtype}'],
                grid_z=npz[f'grid_z_{gtype}'],
                contour_uv=contour_uv,
                output_path=rep_path,
                title=f"{session_id} | {gtype} | Representative 1D Profiles",
            )
            produced_pngs.append(str(rep_path))

        sentinel_data = {
            'session_id': session_id,
            'gesture_types': gesture_types,
            'n_csvs': len(produced_csvs),
            'csvs': produced_csvs,
            'n_pngs': len(produced_pngs),
            'pngs': produced_pngs,
        }
        with open(sentinel, 'w') as f:
            json.dump(sentinel_data, f, indent=2)

        print(
            f"[RF Profile Extraction] {session_id}: done — "
            f"{len(produced_csvs)} CSVs, {len(produced_pngs)} PNGs written."
        )
