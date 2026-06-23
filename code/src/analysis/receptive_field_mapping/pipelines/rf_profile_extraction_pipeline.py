"""Extract 1D RF cross-section profiles and boundary crossings from population heatmap grids."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.pipeline.output_dirs import SPATIAL_EXTRACT_BOUNDARIES
from analysis.pipeline.shared_constants import IFF_METRICS, session_id_from_path
from analysis.receptive_field_mapping.rendering.rf_profile_renderer import (
    render_center_axis_profile,
    render_gradient_profile,
    render_laplacian_context,
    render_laplacian_profile,
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


def find_boundary_v_crossings(
    contour_uv: np.ndarray | None,
    u_target: float,
) -> np.ndarray:
    """Return sorted V-intercepts where a closed UV contour crosses a constant-U scanline."""
    if contour_uv is None or len(contour_uv) == 0:
        return np.empty(0, dtype=np.float64)

    closed = np.concatenate([contour_uv, contour_uv[:1]], axis=0)
    u1 = closed[:-1, 0]
    v1 = closed[:-1, 1]
    u2 = closed[1:, 0]
    v2 = closed[1:, 1]

    straddles = (u1 <= u_target) & (u2 > u_target) | (u2 <= u_target) & (u1 > u_target)
    if not np.any(straddles):
        return np.empty(0, dtype=np.float64)

    u1_s, v1_s = u1[straddles], v1[straddles]
    u2_s, v2_s = u2[straddles], v2[straddles]

    v_cross = v1_s + (u_target - u1_s) * (v2_s - v1_s) / (u2_s - u1_s)
    return np.sort(v_cross)


def _extract_profiles_for_gesture(
    session_id: str,
    gtype: str,
    npz: np.lib.npyio.NpzFile,
) -> pd.DataFrame:
    """Extract 150 constant-V profiles and boundary crossings for one gesture type."""
    grid_u = npz[f'grid_u_{gtype}']
    grid_v = npz[f'grid_v_{gtype}']
    grid_z = npz[f'grid_z_{gtype}']

    inflection_contour_key = f'inflection_contour_uv_{gtype}'
    if inflection_contour_key in npz:
        contour_uv = npz[inflection_contour_key]
    elif f'boundary_contour_uv_{gtype}' in npz:
        contour_uv = npz[f'boundary_contour_uv_{gtype}']
    else:
        contour_uv = None

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


_METHODOLOGY_MD = """\
# RF Profile Extraction: Methodology

## What this pipeline produces

1D cross-section profiles extracted from 2D population RF heatmaps, with
boundary crossing annotations derived from the inflection contour.

## Distinction between profile data and boundary computation

The profiles and the boundary contour are computed from **different
representations** of the same underlying heatmap.

### Profile IFF values (the plotted curve)

The IFF values shown in each profile are sampled directly from `grid_z`,
the **raw interpolated** 150x150 heatmap produced by harmonic interpolation
of per-vertex IFF onto a regular UV grid. No additional smoothing is applied
to these values.

### Boundary contour (the red crossing markers)

The inflection boundary is computed from a **Gaussian-smoothed** version of
the same `grid_z` (default sigma = 4 grid cells). The algorithm:

1. NaN-aware Gaussian smoothing normalised by a weight mask.
2. Nearest-neighbour extrapolation of remaining NaN cells.
3. Laplacian computation (second spatial derivative in both U and V).
4. Flood-fill of the connected negative-Laplacian region around the peak.
5. Marching-squares contour extraction at the basin boundary.

The contour therefore represents the **Laplacian zero-crossing** of the
smoothed field: the ring where the surface transitions from concave (near
the peak) to convex (on the flanks).

## Consequence for profile interpretation

Because the boundary was derived from the smoothed field while the profile
plots the raw field, the crossing markers may not fall exactly at the
inflection points of the displayed 1D curve. Two factors contribute:

- **Smoothing mismatch**: the Gaussian pre-smoothing shifts the effective
  inflection location relative to the raw data.
- **2D vs 1D Laplacian**: the 2D Laplacian
  (d^2z/du^2 + d^2z/dv^2) includes the off-axis curvature term, so
  the zero-crossing of the 2D Laplacian along a 1D slice does not generally
  coincide with the zero of the 1D second derivative (d^2z/du^2 alone).

## Current design decision

The profiles intentionally display the **raw interpolated** IFF values
rather than the smoothed ones. This preserves the original signal detail for
visual inspection. The boundary crossings are overlaid as-is from the
smoothed-field contour, providing a spatial reference for the RF extent
without implying that the crossing falls at a specific feature of the raw
curve.

A future refinement could overlay both representations (raw as a faint
background, smoothed as the primary line) so that the crossings visually
align with the displayed curve.

## Output formats

Each profile figure is saved in two formats:
- **PNG** (150 dpi) for quick inspection and screen display.
- **SVG** (vector) for publication-quality figure preparation.
"""


def _write_methodology_md(output_dir: Path) -> None:
    """Write the methodology markdown file at the output directory root."""
    md_path = output_dir / 'rf_profile_extraction_methodology.md'
    md_path.write_text(_METHODOLOGY_MD, encoding='utf-8')


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

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_methodology_md(output_dir)

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
        produced_svgs: list[str] = []

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

            gradient_contour_key = f'gradient_contour_uv_{gtype}'
            gradient_contour_uv = npz[gradient_contour_key] if gradient_contour_key in npz else None

            grid_u = npz[f'grid_u_{gtype}']
            grid_v = npz[f'grid_v_{gtype}']
            grid_z = npz[f'grid_z_{gtype}']
            u_coords = grid_u[:, 0]
            v_coords = grid_v[0, :]

            centroid_key = f'boundary_centroid_uv_{gtype}'
            peak_key = f'boundary_peak_uv_{gtype}'
            centroid_uv = npz[centroid_key] if centroid_key in npz else None
            peak_uv = npz[peak_key] if peak_key in npz else None

            centers = [('centroid', centroid_uv), ('peak', peak_uv)]
            has_any_center = any(c is not None for _, c in centers)

            if not has_any_center:
                logger.warning(
                    "[RF Profile Extraction] %s/%s: no centroid or peak found, skipping profile rendering.",
                    session_id, gtype,
                )
                continue

            smoothed_key = f'smoothed_{gtype}'
            laplacian_key = f'laplacian_{gtype}'
            has_laplacian = smoothed_key in npz and laplacian_key in npz
            if has_laplacian:
                smoothed = npz[smoothed_key]
                laplacian = npz[laplacian_key]

            gradient_mag_key = f'gradient_mag_{gtype}'
            has_gradient = gradient_mag_key in npz
            if has_gradient:
                gradient_mag = npz[gradient_mag_key]

            if has_laplacian:
                laplacian_2d_dir = session_output_dir / 'laplacian_2d'
                laplacian_2d_dir.mkdir(parents=True, exist_ok=True)

            for center_type, center in centers:
                if center is None:
                    continue

                # U-profile: IFF vs U at constant V = center_v
                j = int(np.argmin(np.abs(v_coords - center[1])))
                iff_u = grid_z[:, j]
                crossings_u = find_boundary_u_crossings(contour_uv, v_coords[j])
                grad_crossings_u = find_boundary_u_crossings(gradient_contour_uv, v_coords[j])
                u_png = session_output_dir / f'{session_id}_rf_profile_u_{center_type}_{gtype}.png'
                render_center_axis_profile(
                    coords=u_coords,
                    iff_values=iff_u,
                    boundary_crossings=crossings_u,
                    output_path=u_png,
                    title=f"{session_id} | {gtype} | IFF vs U at {center_type} (V={v_coords[j]:.1f} mm)",
                    xlabel="U (mm)",
                    gradient_crossings=grad_crossings_u if len(grad_crossings_u) > 0 else None,
                )
                produced_pngs.append(str(u_png))
                produced_svgs.append(str(u_png.with_suffix('.svg')))

                if has_laplacian:
                    smoothed_u = smoothed[:, j]
                    u_smooth_png = session_output_dir / f'{session_id}_rf_profile_smoothed_u_{center_type}_{gtype}.png'
                    render_center_axis_profile(
                        coords=u_coords,
                        iff_values=smoothed_u,
                        boundary_crossings=crossings_u,
                        output_path=u_smooth_png,
                        title=f"{session_id} | {gtype} | Smoothed IFF vs U at {center_type} (V={v_coords[j]:.1f} mm)",
                        xlabel="U (mm)",
                        ylabel="Smoothed IFF (Hz)",
                        gradient_crossings=grad_crossings_u if len(grad_crossings_u) > 0 else None,
                    )
                    produced_pngs.append(str(u_smooth_png))
                    produced_svgs.append(str(u_smooth_png.with_suffix('.svg')))

                    lap_u = laplacian[:, j]
                    u_lap1d_png = session_output_dir / f'{session_id}_rf_laplacian_1d_u_{center_type}_{gtype}.png'
                    render_laplacian_profile(
                        coords=u_coords,
                        lap_values=lap_u,
                        boundary_crossings=crossings_u,
                        output_path=u_lap1d_png,
                        title=f"{session_id} | {gtype} | Laplacian vs U at {center_type} (V={v_coords[j]:.1f} mm)",
                        xlabel="U (mm)",
                    )
                    produced_pngs.append(str(u_lap1d_png))
                    produced_svgs.append(str(u_lap1d_png.with_suffix('.svg')))

                    u_lap_png = laplacian_2d_dir / f'{session_id}_rf_laplacian_2d_u_{center_type}_{gtype}.png'
                    render_laplacian_context(
                        grid_u=grid_u, grid_v=grid_v,
                        laplacian=laplacian, smoothed=smoothed,
                        contour_uv=contour_uv,
                        scanline_coord=float(v_coords[j]),
                        scanline_axis="U",
                        boundary_crossings=crossings_u,
                        output_path=u_lap_png,
                        title=f"{session_id} | {gtype} | Laplacian context — U-profile at {center_type} (V={v_coords[j]:.1f} mm)",
                    )
                    produced_pngs.append(str(u_lap_png))
                    produced_svgs.append(str(u_lap_png.with_suffix('.svg')))

                if has_gradient:
                    grad_u = gradient_mag[:, j]
                    u_grad_png = session_output_dir / f'{session_id}_rf_gradient_1d_u_{center_type}_{gtype}.png'
                    render_gradient_profile(
                        coords=u_coords,
                        grad_values=grad_u,
                        gradient_crossings=grad_crossings_u,
                        output_path=u_grad_png,
                        title=f"{session_id} | {gtype} | |∇IFF| vs U at {center_type} (V={v_coords[j]:.1f} mm)",
                        xlabel="U (mm)",
                    )
                    produced_pngs.append(str(u_grad_png))
                    produced_svgs.append(str(u_grad_png.with_suffix('.svg')))

                # V-profile: IFF vs V at constant U = center_u
                i = int(np.argmin(np.abs(u_coords - center[0])))
                iff_v = grid_z[i, :]
                crossings_v = find_boundary_v_crossings(contour_uv, u_coords[i])
                grad_crossings_v = find_boundary_v_crossings(gradient_contour_uv, u_coords[i])
                v_png = session_output_dir / f'{session_id}_rf_profile_v_{center_type}_{gtype}.png'
                render_center_axis_profile(
                    coords=v_coords,
                    iff_values=iff_v,
                    boundary_crossings=crossings_v,
                    output_path=v_png,
                    title=f"{session_id} | {gtype} | IFF vs V at {center_type} (U={u_coords[i]:.1f} mm)",
                    xlabel="V (mm)",
                    gradient_crossings=grad_crossings_v if len(grad_crossings_v) > 0 else None,
                )
                produced_pngs.append(str(v_png))
                produced_svgs.append(str(v_png.with_suffix('.svg')))

                if has_laplacian:
                    smoothed_v = smoothed[i, :]
                    v_smooth_png = session_output_dir / f'{session_id}_rf_profile_smoothed_v_{center_type}_{gtype}.png'
                    render_center_axis_profile(
                        coords=v_coords,
                        iff_values=smoothed_v,
                        boundary_crossings=crossings_v,
                        output_path=v_smooth_png,
                        title=f"{session_id} | {gtype} | Smoothed IFF vs V at {center_type} (U={u_coords[i]:.1f} mm)",
                        xlabel="V (mm)",
                        ylabel="Smoothed IFF (Hz)",
                        gradient_crossings=grad_crossings_v if len(grad_crossings_v) > 0 else None,
                    )
                    produced_pngs.append(str(v_smooth_png))
                    produced_svgs.append(str(v_smooth_png.with_suffix('.svg')))

                    lap_v = laplacian[i, :]
                    v_lap1d_png = session_output_dir / f'{session_id}_rf_laplacian_1d_v_{center_type}_{gtype}.png'
                    render_laplacian_profile(
                        coords=v_coords,
                        lap_values=lap_v,
                        boundary_crossings=crossings_v,
                        output_path=v_lap1d_png,
                        title=f"{session_id} | {gtype} | Laplacian vs V at {center_type} (U={u_coords[i]:.1f} mm)",
                        xlabel="V (mm)",
                    )
                    produced_pngs.append(str(v_lap1d_png))
                    produced_svgs.append(str(v_lap1d_png.with_suffix('.svg')))

                    v_lap_png = laplacian_2d_dir / f'{session_id}_rf_laplacian_2d_v_{center_type}_{gtype}.png'
                    render_laplacian_context(
                        grid_u=grid_u, grid_v=grid_v,
                        laplacian=laplacian, smoothed=smoothed,
                        contour_uv=contour_uv,
                        scanline_coord=float(u_coords[i]),
                        scanline_axis="V",
                        boundary_crossings=crossings_v,
                        output_path=v_lap_png,
                        title=f"{session_id} | {gtype} | Laplacian context — V-profile at {center_type} (U={u_coords[i]:.1f} mm)",
                    )
                    produced_pngs.append(str(v_lap_png))
                    produced_svgs.append(str(v_lap_png.with_suffix('.svg')))

                if has_gradient:
                    grad_v = gradient_mag[i, :]
                    v_grad_png = session_output_dir / f'{session_id}_rf_gradient_1d_v_{center_type}_{gtype}.png'
                    render_gradient_profile(
                        coords=v_coords,
                        grad_values=grad_v,
                        gradient_crossings=grad_crossings_v,
                        output_path=v_grad_png,
                        title=f"{session_id} | {gtype} | |∇IFF| vs V at {center_type} (U={u_coords[i]:.1f} mm)",
                        xlabel="V (mm)",
                    )
                    produced_pngs.append(str(v_grad_png))
                    produced_svgs.append(str(v_grad_png.with_suffix('.svg')))

        sentinel_data = {
            'session_id': session_id,
            'gesture_types': gesture_types,
            'n_csvs': len(produced_csvs),
            'csvs': produced_csvs,
            'n_pngs': len(produced_pngs),
            'pngs': produced_pngs,
            'n_svgs': len(produced_svgs),
            'svgs': produced_svgs,
        }
        with open(sentinel, 'w') as f:
            json.dump(sentinel_data, f, indent=2)

        print(
            f"[RF Profile Extraction] {session_id}: done — "
            f"{len(produced_csvs)} CSVs, {len(produced_pngs)} PNGs, "
            f"{len(produced_svgs)} SVGs written."
        )
