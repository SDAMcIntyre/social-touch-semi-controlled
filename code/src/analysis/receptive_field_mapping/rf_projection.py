"""2D surface projection infrastructure for RF heatmaps.

Registry pattern: PROJECTION_METHODS maps string keys to pure functions.
Each function has signature:
    (points_3d, forearm_vertices, contact_centroid, **kwargs) -> np.ndarray (N, 2)

To add a new projection method, implement the function and register it at the
bottom of this module via PROJECTION_METHODS[key] = function.
"""

import logging

import numpy as np

from .tangent_plane_alignment import align_points, compute_tangent_plane_rotation

logger = logging.getLogger(__name__)


def project_tangent_plane(
    points_3d: np.ndarray,
    forearm_vertices: np.ndarray,
    contact_centroid: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """Project 3D points onto the local tangent plane and return 2D (u, v) coords.

    Wraps compute_tangent_plane_rotation() + align_points() + drop-Z slice.
    Falls back to raw XY if rotation cannot be computed.
    """
    if forearm_vertices is not None and contact_centroid is not None:
        R = compute_tangent_plane_rotation(forearm_vertices, contact_centroid)
        if R is not None:
            rotated = align_points(points_3d, R)
            return rotated[:, :2]

    logger.warning("project_tangent_plane: falling back to raw XY (no forearm vertices or centroid).")
    return points_3d[:, :2]


def fit_cylinder_axis(
    forearm_vertices: np.ndarray,
    contact_centroid: np.ndarray,
) -> tuple:
    """Fit a cylinder axis to the forearm point cloud via PCA.

    Returns
    -------
    (axis, center, mean_radius)
        axis: unit 3-vector along the cylinder long axis
        center: centroid projected onto the axis (scalar position)
        mean_radius: mean radial distance from the axis (in mm)
    """
    from scipy.spatial import KDTree

    # Use local neighborhood around contact centroid for more robust axis fit
    k = min(500, len(forearm_vertices))
    tree = KDTree(forearm_vertices)
    _, idx = tree.query(contact_centroid, k=k)
    local_pts = forearm_vertices[idx]

    centered = local_pts - local_pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    # First right singular vector = direction of maximum variance = cylinder axis
    axis = Vt[0]
    axis = axis / np.linalg.norm(axis)

    # Project vertices onto the axis to compute radial residuals
    proj_len = local_pts @ axis
    proj_pts = np.outer(proj_len, axis)
    radial = local_pts - proj_pts
    mean_radius = float(np.mean(np.linalg.norm(radial, axis=1)))

    # Center: project contact centroid onto the axis
    center = contact_centroid @ axis  # scalar height of centroid along axis

    return axis, center, mean_radius


def project_cylindrical_unwrap(
    points_3d: np.ndarray,
    forearm_vertices: np.ndarray,
    contact_centroid: np.ndarray,
    per_point_radius: bool = False,
    **kwargs,
) -> np.ndarray:
    """Unwrap 3D points from a cylinder surface to 2D (u = r*theta, v = h) coords.

    The cylinder axis is fitted from the forearm vertices via PCA. The seam
    (theta = ±pi) is placed opposite the RF contact centroid (at theta = 0),
    so it falls in the unobserved region for finger data where the Kinect only
    captures ~180 degrees.

    Parameters
    ----------
    per_point_radius:
        If True, use the per-point radial distance from the axis instead of the
        global mean radius. Useful for tapered geometry.

    Returns
    -------
    (N, 2) array with columns [u (arc_length_mm), v (height_mm)].
    """
    if forearm_vertices is None or contact_centroid is None:
        logger.warning("project_cylindrical_unwrap: no forearm vertices — cannot fit cylinder.")
        return points_3d[:, :2]

    axis, _center, mean_radius = fit_cylinder_axis(forearm_vertices, contact_centroid)

    # --- Height (v) = projection along cylinder axis ---
    v = points_3d @ axis

    # --- Radial component = residual after subtracting axis projection ---
    proj_pts = np.outer(points_3d @ axis, axis)
    radial = points_3d - proj_pts

    # --- Build a consistent radial frame: x_rad points from axis toward centroid ---
    centroid_proj = np.outer(np.array([contact_centroid @ axis]), axis)
    centroid_radial = contact_centroid - centroid_proj.squeeze()
    centroid_radial_norm = np.linalg.norm(centroid_radial)
    if centroid_radial_norm < 1e-8:
        x_rad = np.array([1.0, 0.0, 0.0])
        x_rad -= np.dot(x_rad, axis) * axis
        x_rad /= np.linalg.norm(x_rad)
    else:
        x_rad = centroid_radial / centroid_radial_norm

    y_rad = np.cross(axis, x_rad)
    y_rad /= np.linalg.norm(y_rad)

    # --- theta = angle relative to x_rad, wrapped to (-pi, pi] ---
    rx = radial @ x_rad
    ry = radial @ y_rad
    theta = np.arctan2(ry, rx)

    # Seam at pi, centroid at theta=0 (already satisfied by construction)

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
    return PROJECTION_METHODS[method](points_3d, forearm_vertices, contact_centroid, **kwargs)
