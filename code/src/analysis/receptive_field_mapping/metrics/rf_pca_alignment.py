"""PCA-based rigid UV alignment for population response fields.

Computes and applies an unweighted 2D PCA alignment: shifts the centroid of
above-threshold vertices to the origin and rotates so the major axis aligns
with U.
"""

from __future__ import annotations

import numpy as np


def compute_rf_pca_alignment(
    forearm_uv: np.ndarray,
    heatmap: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute PCA alignment from above-threshold heatmap vertices.

    Returns (center, rotation_matrix, angle_deg).
    """
    mask = heatmap > 0
    n_valid = int(mask.sum())
    if n_valid < 3:
        raise ValueError(
            f"fewer than 3 above-threshold vertices: cannot compute PCA alignment"
        )

    pts = forearm_uv[mask]
    center = pts.mean(axis=0)
    diff = pts - center
    cov = (diff.T @ diff) / n_valid

    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    major_axis = eigenvectors[:, -1].copy()
    if major_axis[0] < 0:
        major_axis = -major_axis

    minor_axis = eigenvectors[:, -2].copy()
    if minor_axis[1] < 0:
        minor_axis = -minor_axis

    rotation_matrix = np.stack([major_axis, minor_axis], axis=0)
    angle_deg = float(np.degrees(np.arctan2(major_axis[1], major_axis[0])))

    return center, rotation_matrix, angle_deg


def apply_uv_alignment(
    uv: np.ndarray,
    center: np.ndarray,
    rotation_matrix: np.ndarray,
) -> np.ndarray:
    """Apply rigid shift+rotation: (uv - center) @ rotation_matrix.T."""
    return (uv - center) @ rotation_matrix.T
