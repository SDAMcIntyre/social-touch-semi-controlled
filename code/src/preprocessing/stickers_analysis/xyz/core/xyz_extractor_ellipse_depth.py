import cv2
import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple

from ...roi.models.roi_tracked_data import ROITrackedObjects
from .xyz_extractor_centroid import CentroidPointCloudExtractor
from .xyz_extractor_interface import XYZExtractorInterface

_MIN_VALID_PIXELS = 3
_DEFAULT_SPREAD_THRESHOLD_MM = 10.0
_SHALLOW_CLUSTER_PERCENTILE = 25.0
STICKER_DIAMETER_MM: float = 10.0


class EllipseDepthExtractor(XYZExtractorInterface):
    """
    Extracts 3D coordinates by sampling all depth points within the tracked
    ellipse and applying homogeneity-adaptive aggregation.

    For frames where the ellipse straddles a depth edge (fingertip, knuckle),
    the shallow-cluster path selects the lower percentile of z values to
    avoid the depth-gradient bias produced by the Kinect sensor at object
    boundaries.

    Falls back to single-pixel centroid extraction when ellipse data is
    unavailable or when fewer than three valid depth pixels are found within
    the mask.

    Output includes monitor columns ``z_std``, ``n_depth_pixels``, and
    ``z_range_clipped`` (True when the sticker-size guard fires).
    """

    def __init__(
        self,
        debug: bool = False,
        spread_threshold_mm: float = _DEFAULT_SPREAD_THRESHOLD_MM,
        sticker_diameter_mm: float = STICKER_DIAMETER_MM,
    ):
        self.debug = debug
        self.spread_threshold_mm = spread_threshold_mm
        self._sticker_diameter_mm = sticker_diameter_mm

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def extract(
        self,
        tracked_obj_row: pd.Series,
        point_cloud: np.ndarray,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Orchestrate mask → sample → aggregate pipeline with centroid fallback."""
        px = tracked_obj_row["center_x"]
        py = tracked_obj_row["center_y"]

        axes_major = tracked_obj_row.get("axes_major", np.nan)
        if pd.notna(axes_major) and point_cloud is not None:
            axes_minor = tracked_obj_row.get("axes_minor", np.nan)
            angle = tracked_obj_row.get("angle", 0.0)

            if pd.notna(axes_minor):
                mask = self._build_ellipse_mask(
                    shape=point_cloud.shape[:2],
                    center_x=float(px),
                    center_y=float(py),
                    axes_major=float(axes_major),
                    axes_minor=float(axes_minor),
                    angle=float(angle) if pd.notna(angle) else 0.0,
                )
                xs, ys, zs = self._sample_depth_within_mask(point_cloud, mask)

                if len(zs) >= _MIN_VALID_PIXELS:
                    x_mm, y_mm, z_mm, z_std, n_pixels, z_range_clipped = self._aggregate_depth(
                        xs, ys, zs, self.spread_threshold_mm, self._sticker_diameter_mm
                    )
                    coords_3d = {"x_mm": x_mm, "y_mm": y_mm, "z_mm": z_mm}
                    monitor_data = {
                        "px": px,
                        "py": py,
                        "z_std": z_std,
                        "n_depth_pixels": n_pixels,
                        "z_range_clipped": z_range_clipped,
                    }
                    return coords_3d, monitor_data

        # Fallback: single-pixel centroid extraction (ellipse unavailable or
        # mask yielded fewer than _MIN_VALID_PIXELS valid depth samples).
        x_mm, y_mm, z_mm = CentroidPointCloudExtractor.get_xyz_from_point_cloud(point_cloud, px, py)
        coords_3d = {"x_mm": x_mm, "y_mm": y_mm, "z_mm": z_mm}
        monitor_data = {"px": px, "py": py, "z_std": np.nan, "n_depth_pixels": 1, "z_range_clipped": False}
        return coords_3d, monitor_data

    def get_empty_result(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        coords_3d = {"x_mm": np.nan, "y_mm": np.nan, "z_mm": np.nan}
        monitor_data = {
            "px": np.nan,
            "py": np.nan,
            "z_std": np.nan,
            "n_depth_pixels": np.nan,
            "z_range_clipped": np.nan,
        }
        return coords_3d, monitor_data

    @classmethod
    def can_process(cls, tracked_data: Any) -> bool:
        if not isinstance(tracked_data, ROITrackedObjects):
            return False
        first_object_name = next(iter(tracked_data), None)
        if first_object_name:
            required_cols = {"roi_x", "roi_y", "roi_width", "roi_height", "status"}
            df_cols = set(tracked_data[first_object_name].columns)
            if not required_cols.issubset(df_cols):
                return False
        return True

    @classmethod
    def should_process_row(cls, tracked_obj_row: pd.Series) -> bool:
        invalid_statuses = {"Failed", "Black Frame", "Ignored"}
        return tracked_obj_row["status"] not in invalid_statuses

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_ellipse_mask(
        shape: Tuple[int, int],
        center_x: float,
        center_y: float,
        axes_major: float,
        axes_minor: float,
        angle: float,
    ) -> np.ndarray:
        """Return a boolean 2D mask with the filled ellipse set to True.

        Args:
            shape: (height, width) of the point cloud frame.
            center_x: Ellipse centre column in pixels.
            center_y: Ellipse centre row in pixels.
            axes_major: Semi-major axis length in pixels.
            axes_minor: Semi-minor axis length in pixels.
            angle: Rotation angle in degrees (OpenCV convention).

        Returns:
            Boolean array of shape ``shape``.
        """
        mask = np.zeros(shape, dtype=np.uint8)
        center = (int(round(center_x)), int(round(center_y)))
        # Clamp axes to at least 1 px so cv2.ellipse never receives (0, 0).
        axes = (max(1, int(round(axes_major))), max(1, int(round(axes_minor))))
        cv2.ellipse(mask, center, axes, angle, 0, 360, color=1, thickness=-1)
        return mask.astype(bool)

    @staticmethod
    def _sample_depth_within_mask(
        point_cloud: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract valid (non-zero, non-NaN) point cloud values within the mask.

        The mask is clipped to the point-cloud dimensions before sampling so
        that ellipses near the image border do not cause an index error.

        Args:
            point_cloud: (H, W, 3) array of XYZ coordinates in mm.
            mask: Boolean (H, W) mask; True selects pixels to sample.

        Returns:
            Three 1-D arrays ``(xs, ys, zs)`` of valid sample values in mm.
        """
        h, w, _ = point_cloud.shape
        clipped_mask = mask[:h, :w]

        sampled = point_cloud[clipped_mask]  # shape (N, 3)

        # Keep only pixels where the sensor returned a real reading.
        nonzero = np.any(sampled != 0, axis=1)
        nonan = ~np.any(np.isnan(sampled), axis=1)
        valid = nonzero & nonan

        sampled_valid = sampled[valid]
        return sampled_valid[:, 0], sampled_valid[:, 1], sampled_valid[:, 2]

    @staticmethod
    def _aggregate_depth(
        xs: np.ndarray,
        ys: np.ndarray,
        zs: np.ndarray,
        spread_threshold_mm: float,
        sticker_diameter_mm: float,
    ) -> Tuple[float, float, float, float, int, bool]:
        """Homogeneity-adaptive aggregation with sticker-size guard.

        **Homogeneous case** (z_std < threshold):
            The ellipse lies on a single surface plane. Select all valid
            samples as candidates.

        **Spread-out case** (z_std >= threshold):
            The ellipse straddles a depth edge (e.g. fingertip against
            background). Select the *shallow cluster* — the lower
            ``_SHALLOW_CLUSTER_PERCENTILE`` percent of z values — as
            candidates.

        **Sticker-size guard** (applied after either path above):
            The physical sticker diameter (~10 mm) is a hard upper bound on
            legitimate depth variation within the sticker area. If the z-range
            of the candidate cluster still exceeds ``sticker_diameter_mm``,
            some pixels are capturing the background depth ramp. Override the
            cluster by keeping only values ≤ z_min + sticker_diameter_mm
            (upper-Z, nearest-to-camera subset). ``z_range_clipped`` is set to
            True when the guard fires.

        Args:
            xs: 1-D array of x coordinates in mm.
            ys: 1-D array of y coordinates in mm.
            zs: 1-D array of z (depth) coordinates in mm.
            spread_threshold_mm: z std value above which the spread-out path
                is taken (default 10 mm ≈ 2× Kinect v2 noise floor).
            sticker_diameter_mm: Physical sticker diameter used as the z-range
                plausibility guard (default 10 mm).

        Returns:
            Tuple ``(x_mm, y_mm, z_mm, z_std, n_pixels, z_range_clipped)``.
        """
        z_std = float(np.std(zs))
        n_pixels = int(len(zs))

        if z_std < spread_threshold_mm:
            candidate_xs, candidate_ys, candidate_zs = xs, ys, zs
        else:
            # Shallow cluster: points nearest to the camera (smallest z).
            z_cutoff = float(np.percentile(zs, _SHALLOW_CLUSTER_PERCENTILE))
            shallow = zs <= z_cutoff
            candidate_xs = xs[shallow]
            candidate_ys = ys[shallow]
            candidate_zs = zs[shallow]

        # Sticker-size guard: z-range within the candidate cluster must not
        # exceed the physical sticker diameter.  Any excess means the cluster
        # still contains background-gradient pixels; keep only the
        # upper-Z (nearest-to-camera, minimum-Z) subset.
        z_range = float(candidate_zs.max() - candidate_zs.min())
        if z_range > sticker_diameter_mm:
            z_min = float(candidate_zs.min())
            size_mask = candidate_zs <= z_min + sticker_diameter_mm
            candidate_xs = candidate_xs[size_mask]
            candidate_ys = candidate_ys[size_mask]
            candidate_zs = candidate_zs[size_mask]
            z_range_clipped = True
        else:
            z_range_clipped = False

        x_mm = float(np.median(candidate_xs))
        y_mm = float(np.median(candidate_ys))
        z_mm = float(np.median(candidate_zs))

        return x_mm, y_mm, z_mm, z_std, n_pixels, z_range_clipped
