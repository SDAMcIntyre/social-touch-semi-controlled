import logging

import numpy as np

logger = logging.getLogger(__name__)


def _compute_surface_normal(
    forearm_vertices: np.ndarray,
    contact_centroid: np.ndarray,
    k: int = 50,
) -> np.ndarray:
    try:
        from scipy.spatial import KDTree

        tree = KDTree(forearm_vertices)
        _, idx = tree.query(contact_centroid, k=min(k, len(forearm_vertices)))
        neighbors = forearm_vertices[idx]
        centered = neighbors - neighbors.mean(axis=0)
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        normal = Vt[-1]
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            return None
        return normal / norm
    except Exception:
        logger.debug("Surface normal computation failed", exc_info=True)
        return None


def compute_tangent_plane_rotation(
    forearm_vertices: np.ndarray,
    contact_centroid: np.ndarray,
    k: int = 50,
) -> np.ndarray:
    z_new = _compute_surface_normal(forearm_vertices, contact_centroid, k=k)
    if z_new is None:
        return None

    if np.dot(z_new, np.array([0.0, 0.0, 1.0])) < 0:
        z_new = -z_new

    x_candidate = np.array([1.0, 0.0, 0.0])
    x_proj = x_candidate - np.dot(x_candidate, z_new) * z_new
    x_norm = np.linalg.norm(x_proj)
    if x_norm < 1e-6:
        x_candidate = np.array([0.0, 1.0, 0.0])
        x_proj = x_candidate - np.dot(x_candidate, z_new) * z_new
        x_norm = np.linalg.norm(x_proj)
    x_new = x_proj / x_norm

    y_new = np.cross(z_new, x_new)

    R = np.stack([x_new, y_new, z_new], axis=0)

    if np.linalg.det(R) < 0:
        z_new = -z_new
        y_new = np.cross(z_new, x_new)
        R = np.stack([x_new, y_new, z_new], axis=0)

    return R


def align_points(points: np.ndarray, R: np.ndarray) -> np.ndarray:
    return points @ R.T
