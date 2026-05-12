"""2D surface projection infrastructure for RF heatmaps.

Registry pattern: PROJECTION_METHODS maps string keys to pure functions.
Each function has signature:
    (points_3d, forearm_vertices, contact_centroid, **kwargs) -> np.ndarray (N, 2)

To add a new projection method, implement the function and register it at the
bottom of this module via PROJECTION_METHODS[key] = function.
"""

import logging

import numpy as np
from scipy.spatial import KDTree

from .tangent_plane_alignment import align_points

logger = logging.getLogger(__name__)


def project_tangent_plane(
    points_3d: np.ndarray,
    forearm_vertices: np.ndarray,
    contact_centroid: np.ndarray,
    rotation_matrix: np.ndarray = None,
    **kwargs,
) -> np.ndarray:
    if rotation_matrix is None:
        raise ValueError(
            "project_tangent_plane: rotation_matrix is required. "
            "Run 'set_rf_camera_settings' before any downstream RF task."
        )
    rotated = align_points(points_3d, rotation_matrix)
    return rotated[:, :2]


def fit_cylinder_axis(
    forearm_vertices: np.ndarray,
    contact_centroid: np.ndarray,
) -> tuple:
    """Fit a cylinder axis to the forearm point cloud via PCA.

    Returns
    -------
    (axis, axis_point, mean_radius)
        axis: unit 3-vector along the cylinder long axis
        axis_point: (3,) point on the cylinder axis (mean of the local PCA neighborhood)
        mean_radius: mean radial distance from the axis (in mm)
    """
    # Use local neighborhood around contact centroid for more robust axis fit
    k = min(500, len(forearm_vertices))
    tree = KDTree(forearm_vertices)
    _, idx = tree.query(contact_centroid, k=k)
    local_pts = forearm_vertices[idx]

    axis_point = local_pts.mean(axis=0)
    centered = local_pts - axis_point
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    axis = Vt[0]
    axis = axis / np.linalg.norm(axis)

    proj_len = centered @ axis
    radial = centered - np.outer(proj_len, axis)
    mean_radius = float(np.mean(np.linalg.norm(radial, axis=1)))

    return axis, axis_point, mean_radius


def _compute_local_radius(
    forearm_vertices: np.ndarray,
    contact_centroid: np.ndarray,
    axis: np.ndarray,
) -> tuple:
    """Compute mean cylinder radius from local forearm vertices.

    Returns (axis_point, mean_radius) where axis_point is the centroid of the
    local neighborhood (a point on/near the cylinder axis).
    """
    k = min(500, len(forearm_vertices))
    tree = KDTree(forearm_vertices)
    _, idx = tree.query(contact_centroid, k=k)
    local_pts = forearm_vertices[idx]

    axis_point = local_pts.mean(axis=0)
    centered = local_pts - axis_point
    proj_len = centered @ axis
    radial = centered - np.outer(proj_len, axis)
    mean_radius = float(np.mean(np.linalg.norm(radial, axis=1)))

    return axis_point, mean_radius


def project_cylindrical_unwrap(
    points_3d: np.ndarray,
    forearm_vertices: np.ndarray,
    contact_centroid: np.ndarray,
    per_point_radius: bool = False,
    rotation_matrix: np.ndarray = None,
    **kwargs,
) -> np.ndarray:
    """Unwrap 3D points from a cylinder surface to 2D (u = r*theta, v = h) coords.

    When ``rotation_matrix`` is provided (from saved RF camera settings), the
    camera frame defines the projection geometry entirely:

    - **Cylinder axis** = R[0] (camera right) — the forearm longitudinal direction.
    - **theta=0**      = −R[2] (camera-facing) — seam falls on the far side.
    - **Angular up**   = R[1] (camera up).

    When ``rotation_matrix`` is *None*, falls back to PCA axis fitting.

    Parameters
    ----------
    per_point_radius:
        If True, use the per-point radial distance from the axis instead of the
        global mean radius. Useful for tapered geometry.
    rotation_matrix:
        (3, 3) camera rotation from ``camera_settings_to_rotation()``.
        When provided, the camera frame defines axis and angular reference.
        When *None*, PCA axis fitting is used (legacy behaviour).

    Returns
    -------
    (N, 2) array with columns [u (arc_length_mm), v (height_mm)].
    """
    if forearm_vertices is None or contact_centroid is None:
        logger.warning("project_cylindrical_unwrap: no forearm vertices — cannot fit cylinder.")
        return points_3d[:, :2]

    if rotation_matrix is not None:
        # Camera frame defines the projection: R[0]=axis, -R[2]=theta=0, R[1]=up
        axis = rotation_matrix[0]
        x_rad = -rotation_matrix[2]
        y_rad = rotation_matrix[1]
        axis_point, mean_radius = _compute_local_radius(
            forearm_vertices, contact_centroid, axis,
        )
    else:
        axis, axis_point, mean_radius = fit_cylinder_axis(
            forearm_vertices, contact_centroid,
        )
        centroid_delta = contact_centroid - axis_point
        centroid_radial = centroid_delta - (centroid_delta @ axis) * axis
        centroid_radial_norm = np.linalg.norm(centroid_radial)
        if centroid_radial_norm < 1e-8:
            x_rad = np.array([1.0, 0.0, 0.0])
            x_rad -= np.dot(x_rad, axis) * axis
            x_rad /= np.linalg.norm(x_rad)
        else:
            x_rad = centroid_radial / centroid_radial_norm
        y_rad = np.cross(axis, x_rad)
        y_rad /= np.linalg.norm(y_rad)

    # --- Center on axis point ---
    delta = points_3d - axis_point

    # --- Height (v) = projection along cylinder axis ---
    v = delta @ axis

    # --- Radial component ---
    radial = delta - np.outer(v, axis)

    # --- theta = angle relative to x_rad, wrapped to (-pi, pi] ---
    rx = radial @ x_rad
    ry = radial @ y_rad
    theta = np.arctan2(ry, rx)

    # --- Arc length u = r * theta ---
    if per_point_radius:
        r = np.linalg.norm(radial, axis=1)
        r = np.where(r < 1e-8, mean_radius, r)
    else:
        r = mean_radius

    u = r * theta

    return np.column_stack([u, v])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PROJECTION_METHODS: dict = {
    "tangent_plane": project_tangent_plane,
    "cylindrical_unwrap": project_cylindrical_unwrap,
}


def project_to_2d(
    points_3d: np.ndarray,
    forearm_vertices: np.ndarray,
    contact_centroid: np.ndarray,
    method: str = "tangent_plane",
    rotation_matrix: np.ndarray = None,
    **kwargs,
) -> np.ndarray:
    """Dispatch to the requested 2D projection method.

    Parameters
    ----------
    points_3d:
        (N, 3) array of 3D contact points in mm.
    forearm_vertices:
        (M, 3) array of forearm PLY vertices in mm.
    contact_centroid:
        (3,) centroid of contact points — used for axis/plane fitting.
    method:
        Key into PROJECTION_METHODS. Raises KeyError for unknown methods.

    Returns
    -------
    (N, 2) array of projected (u, v) coordinates in mm.
    """
    if method not in PROJECTION_METHODS:
        raise KeyError(
            f"Unknown projection method '{method}'. "
            f"Available: {list(PROJECTION_METHODS.keys())}"
        )
    return PROJECTION_METHODS[method](
        points_3d, forearm_vertices, contact_centroid,
        rotation_matrix=rotation_matrix, **kwargs
    )
