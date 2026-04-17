import cv2
import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple

from ...roi.models.roi_tracked_data import ROITrackedObjects
from .xyz_extractor_centroid import CentroidPointCloudExtractor
from .xyz_extractor_interface import XYZExtractorInterface

_MIN_VALID_PIXELS = 3
_DEPTH_WEIGHT_SIGMA = 0.3
STICKER_DIAMETER_MM: float = 10.0


class EllipseDepthExtractor(XYZExtractorInterface):
    """
    Extracts 3D coordinates by sampling all depth points within the tracked
    ellipse and applying depth-accuracy-weighted aggregation.

    Pixels nearer to the camera (lower z) receive higher weight via an
    exponential decay from z_min, correcting the systematic bias on tilted or
    curved surfaces where far-side pixels are pulled toward the background.

    Falls back to single-pixel centroid extraction when ellipse data is
    unavailable or when fewer than three valid depth pixels are found within
    the mask.

    Output includes monitor columns ``z_std``, ``n_depth_pixels``, and
    ``z_range_clipped`` (True when the sticker-size guard fires).
    """

    def __init__(
        self,
        debug: bool = False,
        sticker_diameter_mm: float = STICKER_DIAMETER_MM,
        depth_weight_sigma: float = _DEPTH_WEIGHT_SIGMA,
    ):
        self.debug = debug
        self._sticker_diameter_mm = sticker_diameter_mm
        self._depth_weight_sigma = depth_weight_sigma

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
                        xs, ys, zs, self._sticker_diameter_mm, self._depth_weight_sigma
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
            axes_major: Full major-axis length (diameter) in pixels, as stored
                in the consolidated tracks CSV by ``cv2.fitEllipse``.
            axes_minor: Full minor-axis length (diameter) in pixels.
            angle: Rotation angle in degrees (OpenCV convention).

        Returns:
            Boolean array of shape ``shape``.
        """
        mask = np.zeros(shape, dtype=np.uint8)
        center = (int(round(center_x)), int(round(center_y)))
        # cv2.ellipse expects semi-axes; the CSV stores full diameters.
        # The first axis is the one rotated by `angle`, which is `axes_minor`
        # per the cv2.fitEllipse convention used by fit_ellipses_on_correlation_videos.py.
        # Clamp to at least 1 px so cv2.ellipse never receives (0, 0).
        axes = (
            max(1, int(round(axes_minor / 2.0))),
            max(1, int(round(axes_major / 2.0))),
        )
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
        sticker_diameter_mm: float,
        depth_weight_sigma: float,
    ) -> Tuple[float, float, float, float, int, bool]:
        """Depth-accuracy-weighted aggregation with sticker-size guard.

        Assigns each sample a weight based on its relative z-position within
        the candidate set.  Pixels nearer to the camera (smaller z) receive
        higher weight via an exponential decay from z_min:

        .. code-block:: python

            z_norm = (z_i - z_min) / (z_max - z_min)   # 0 = nearest
            w_i    = exp(-z_norm / sigma)

        When all z values are identical (``z_max == z_min``), weights are set
        to 1.0 (uniform), reproducing plain-median behaviour.

        **Sticker-size guard** (applied before weighting):
            The physical sticker diameter is a hard upper bound on legitimate
            depth variation.  If ``z_max - z_min > sticker_diameter_mm``, keep
            only the subset with ``z <= z_min + sticker_diameter_mm``.
            ``z_range_clipped`` is True when the guard fires.

        Args:
            xs: 1-D array of x coordinates in mm.
            ys: 1-D array of y coordinates in mm.
            zs: 1-D array of z (depth) coordinates in mm.
            sticker_diameter_mm: Physical sticker diameter used as the z-range
                plausibility guard (default 10 mm).
            depth_weight_sigma: Exponential decay rate for depth weighting.
                Smaller values give a steeper decay (default 0.3).

        Returns:
            Tuple ``(x_mm, y_mm, z_mm, z_std, n_pixels, z_range_clipped)``.
        """
        z_std = float(np.std(zs))
        n_pixels = int(len(zs))

        # Sticker-size guard: keep only pixels within one sticker-diameter of
        # the nearest-camera (minimum-z) value.
        z_min = float(zs.min())
        z_max = float(zs.max())
        z_range = z_max - z_min
        if z_range > sticker_diameter_mm:
            size_mask = zs <= z_min + sticker_diameter_mm
            candidate_xs = xs[size_mask]
            candidate_ys = ys[size_mask]
            candidate_zs = zs[size_mask]
            z_range_clipped = True
        else:
            candidate_xs, candidate_ys, candidate_zs = xs, ys, zs
            z_range_clipped = False

        # Depth-accuracy weights: exponential decay from z_min.
        cz_min = float(candidate_zs.min())
        cz_max = float(candidate_zs.max())
        if cz_max == cz_min:
            weights = np.ones(len(candidate_zs))
        else:
            z_norm = (candidate_zs - cz_min) / (cz_max - cz_min)
            weights = np.exp(-z_norm / depth_weight_sigma)

        x_mm = EllipseDepthExtractor._weighted_median(candidate_xs, weights)
        y_mm = EllipseDepthExtractor._weighted_median(candidate_ys, weights)
        z_mm = EllipseDepthExtractor._weighted_median(candidate_zs, weights)

        return x_mm, y_mm, z_mm, z_std, n_pixels, z_range_clipped

    @staticmethod
    def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
        """Return the weighted median of ``values`` using ``weights``.

        The weighted median is the value minimising ``sum(w_i * |x_i - v|)``,
        found by sorting values, computing cumulative weights, and locating
        the point where cumulative weight first reaches 50 % of total.

        Args:
            values: 1-D array of values.
            weights: 1-D array of non-negative weights (same length as values).

        Returns:
            Weighted median as a float.
        """
        order = np.argsort(values)
        sorted_vals = values[order]
        sorted_weights = weights[order]
        cumulative = np.cumsum(sorted_weights)
        midpoint = cumulative[-1] / 2.0
        return float(sorted_vals[cumulative >= midpoint][0])
