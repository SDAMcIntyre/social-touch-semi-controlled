import numpy as np


def align_points(points: np.ndarray, R: np.ndarray) -> np.ndarray:
    return points @ R.T


def camera_settings_to_rotation(camera_settings: dict) -> np.ndarray:
    """Derive a 3×3 rotation matrix from saved camera parameters.

    The returned R satisfies: align_points(pts, R) rotates into a frame where
    Z is the view direction and XY is the projection plane.

    Raises ValueError if the up vector is parallel to the view direction
    (degenerate camera configuration).
    """
    camera_position = np.array(camera_settings["camera_position"], dtype=np.float64)
    focal_point = np.array(camera_settings["focal_point"], dtype=np.float64)
    up_vector = np.array(camera_settings["up_vector"], dtype=np.float64)

    view_dir = focal_point - camera_position
    view_norm = np.linalg.norm(view_dir)
    if view_norm < 1e-8:
        raise ValueError(
            "camera_settings_to_rotation: camera_position and focal_point are identical"
        )
    z_new = view_dir / view_norm

    right = np.cross(up_vector, z_new)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        raise ValueError(
            "camera_settings_to_rotation: up_vector is parallel to view direction "
            "(degenerate camera configuration)"
        )
    x_new = right / right_norm
    y_new = np.cross(z_new, x_new)

    return np.stack([x_new, y_new, z_new], axis=0)
