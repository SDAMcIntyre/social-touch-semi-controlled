"""
Pure, UI-free helper that rebuilds every intermediate variable for one frame
of the depth-weighted XYZ aggregation pipeline.

Reuses ``EllipseDepthExtractor``'s static helpers verbatim, so the values
returned here are guaranteed to match the extractor's output for the same
inputs.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..core.xyz_extractor_ellipse_depth import EllipseDepthExtractor

_MIN_VALID_PIXELS = 3


def recompute_frame_diagnostics(
    point_cloud: Optional[np.ndarray],
    tracked_obj_row: pd.Series,
    sticker_diameter_mm: float = 10.0,
    depth_weight_sigma: float = 0.3,
) -> Dict[str, Any]:
    """Rebuild every intermediate variable for one frame.

    Reuses ``EllipseDepthExtractor._build_ellipse_mask``,
    ``_sample_depth_within_mask``, and ``_weighted_median`` verbatim,
    so the returned ``weighted_z`` is guaranteed to equal what the extractor
    would produce for the same inputs.

    Args:
        point_cloud: (H, W, 3) XYZ array in mm, or ``None`` when the frame
            is unavailable.
        tracked_obj_row: A single row of a ConsolidatedTracksManager
            DataFrame for one sticker.
        sticker_diameter_mm: Sticker-size guard threshold (mm).
        depth_weight_sigma: Exponential decay rate for depth weighting.

    Returns:
        Dictionary with the following keys:

        ``mask`` — boolean (H, W) ellipse mask, or ``None`` on fallback.
        ``xs``, ``ys``, ``zs`` — 1-D arrays of valid sampled pixels in mm.
        ``fallback_triggered`` — True when fewer than 3 valid depth pixels
            were found (matches extractor fallback condition).
        ``z_std``, ``z_min``, ``z_max``, ``z_range`` — depth statistics over
            the full sample (NaN on fallback).
        ``z_range_clipped`` — True when the sticker-size guard fired.
        ``candidate_xs``, ``candidate_ys``, ``candidate_zs`` — pixels after
            applying the guard (same as ``xs/ys/zs`` when guard did not fire).
        ``clipped_xs``, ``clipped_ys``, ``clipped_zs`` — pixels removed by
            the guard (empty when guard did not fire).
        ``cz_min``, ``cz_max`` — depth range over candidates.
        ``z_norm`` — per-candidate normalised depth (0 = nearest camera).
        ``weights`` — per-candidate exponential weights.
        ``uniform_weights`` — True when all candidates have identical z.
        ``weighted_x``, ``weighted_y``, ``weighted_z`` — weighted-median
            result in mm.
        ``plain_x``, ``plain_y``, ``plain_z`` — plain median result in mm
            (uniform weights over candidates).
    """
    px = tracked_obj_row.get("center_x", np.nan)
    py = tracked_obj_row.get("center_y", np.nan)
    axes_major = tracked_obj_row.get("axes_major", np.nan)
    axes_minor = tracked_obj_row.get("axes_minor", np.nan)
    angle = tracked_obj_row.get("angle", 0.0)

    mask: Optional[np.ndarray] = None
    xs = np.array([], dtype=float)
    ys = np.array([], dtype=float)
    zs = np.array([], dtype=float)

    can_use_ellipse = (
        point_cloud is not None
        and pd.notna(axes_major)
        and pd.notna(axes_minor)
        and pd.notna(px)
        and pd.notna(py)
    )

    if can_use_ellipse:
        mask = EllipseDepthExtractor._build_ellipse_mask(
            shape=point_cloud.shape[:2],
            center_x=float(px),
            center_y=float(py),
            axes_major=float(axes_major),
            axes_minor=float(axes_minor),
            angle=float(angle) if pd.notna(angle) else 0.0,
        )
        xs, ys, zs = EllipseDepthExtractor._sample_depth_within_mask(point_cloud, mask)

    if len(zs) < _MIN_VALID_PIXELS:
        _nan = float("nan")
        _empty = np.array([], dtype=float)
        return {
            "mask": mask,
            "xs": xs,
            "ys": ys,
            "zs": zs,
            "fallback_triggered": True,
            "z_std": _nan,
            "z_min": _nan,
            "z_max": _nan,
            "z_range": _nan,
            "z_range_clipped": False,
            "candidate_xs": _empty,
            "candidate_ys": _empty,
            "candidate_zs": _empty,
            "clipped_xs": _empty,
            "clipped_ys": _empty,
            "clipped_zs": _empty,
            "cz_min": _nan,
            "cz_max": _nan,
            "z_norm": _empty,
            "weights": _empty,
            "uniform_weights": False,
            "weighted_x": _nan,
            "weighted_y": _nan,
            "weighted_z": _nan,
            "plain_x": _nan,
            "plain_y": _nan,
            "plain_z": _nan,
        }

    z_std = float(np.std(zs))
    z_min = float(zs.min())
    z_max = float(zs.max())
    z_range = z_max - z_min

    if z_range > sticker_diameter_mm:
        size_mask = zs <= z_min + sticker_diameter_mm
        candidate_xs = xs[size_mask]
        candidate_ys = ys[size_mask]
        candidate_zs = zs[size_mask]
        clipped_xs = xs[~size_mask]
        clipped_ys = ys[~size_mask]
        clipped_zs = zs[~size_mask]
        z_range_clipped = True
    else:
        candidate_xs = xs
        candidate_ys = ys
        candidate_zs = zs
        clipped_xs = np.array([], dtype=float)
        clipped_ys = np.array([], dtype=float)
        clipped_zs = np.array([], dtype=float)
        z_range_clipped = False

    cz_min = float(candidate_zs.min())
    cz_max = float(candidate_zs.max())

    if cz_max == cz_min:
        z_norm = np.zeros(len(candidate_zs), dtype=float)
        weights = np.ones(len(candidate_zs), dtype=float)
        uniform_weights = True
    else:
        z_norm = (candidate_zs - cz_min) / (cz_max - cz_min)
        weights = np.exp(-z_norm / depth_weight_sigma)
        uniform_weights = False

    weighted_x = EllipseDepthExtractor._weighted_median(candidate_xs, weights)
    weighted_y = EllipseDepthExtractor._weighted_median(candidate_ys, weights)
    weighted_z = EllipseDepthExtractor._weighted_median(candidate_zs, weights)

    uniform_w = np.ones(len(candidate_zs), dtype=float)
    plain_x = EllipseDepthExtractor._weighted_median(candidate_xs, uniform_w)
    plain_y = EllipseDepthExtractor._weighted_median(candidate_ys, uniform_w)
    plain_z = EllipseDepthExtractor._weighted_median(candidate_zs, uniform_w)

    return {
        "mask": mask,
        "xs": xs,
        "ys": ys,
        "zs": zs,
        "fallback_triggered": False,
        "z_std": z_std,
        "z_min": z_min,
        "z_max": z_max,
        "z_range": z_range,
        "z_range_clipped": z_range_clipped,
        "candidate_xs": candidate_xs,
        "candidate_ys": candidate_ys,
        "candidate_zs": candidate_zs,
        "clipped_xs": clipped_xs,
        "clipped_ys": clipped_ys,
        "clipped_zs": clipped_zs,
        "cz_min": cz_min,
        "cz_max": cz_max,
        "z_norm": z_norm,
        "weights": weights,
        "uniform_weights": uniform_weights,
        "weighted_x": weighted_x,
        "weighted_y": weighted_y,
        "weighted_z": weighted_z,
        "plain_x": plain_x,
        "plain_y": plain_y,
        "plain_z": plain_z,
    }
