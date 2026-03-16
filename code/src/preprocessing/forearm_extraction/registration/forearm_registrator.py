"""ICP-based forearm point cloud registration.

Implements :class:`ForearmRegistrator`, which aligns multiple forearm
snapshots captured during a session to a single canonical reference frame
using Open3D point-to-plane ICP, and provides helpers for persisting the
resulting unified point cloud and per-snapshot transform matrices.

Two unification modes are supported:

* ``"reference"`` — the canonical cloud is held fixed (identity transform);
  all other snapshots are rigidly aligned to it.
* ``"average"`` — all snapshots are first aligned to a temporary internal
  reference, then every transform is re-centred around the SE(3) mean of
  the group so that no single snapshot is privileged.

Six registration methods are available (selected via the ``method`` parameter
of :meth:`register_all` and :meth:`register_all_to_average`):

* ``"vanilla"`` *(default)* — point-to-plane ICP with equal weights.
* ``"robust"`` — point-to-plane ICP with a Tukey robust kernel to downweight
  fringe-point residuals.
* ``"multiscale"`` — coarse-to-fine ICP across three resolution levels.
* ``"generalized"`` — Generalized ICP (GICP) modelling local surface
  covariance.
* ``"trimmed"`` — removes fringe source points by nearest-neighbour distance
  before running ICP.
* ``"global"`` — FPFH + RANSAC global alignment followed by ICP refinement;
  handles larger displacements where identity initialisation fails.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d

from preprocessing.common import PointCloudDataHandler

logger = logging.getLogger(__name__)


class ForearmRegistrator:
    """Registers multiple forearm point cloud snapshots to a common reference frame.

    Uses Open3D point-to-plane ICP to align forearm snapshots captured at
    different times during a session.  The first snapshot (canonical) is held
    fixed; every other snapshot is aligned to it.

    Args:
        canonical_cloud: The reference point cloud (held fixed).
        max_correspondence_distance: ICP distance threshold (metres).
        icp_max_iteration: Maximum ICP iterations per registration.
    """

    _METHOD_MAP: Dict[str, str] = {
        "vanilla": "register",
        "robust": "register_robust",
        "multiscale": "register_multiscale",
        "generalized": "register_generalized",
        "trimmed": "register_trimmed",
        "global": "register_global_then_local",
    }

    def __init__(
        self,
        canonical_cloud: o3d.geometry.PointCloud,
        *,
        max_correspondence_distance: float = 0.10,
        icp_max_iteration: int = 200,
    ):
        self._canonical = canonical_cloud
        self._max_corr_dist = max_correspondence_distance
        self._max_iter = icp_max_iteration

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_target_normals(self) -> None:
        """Estimate normals on the canonical cloud if not already present."""
        if not self._canonical.has_normals():
            self._canonical.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
            )

    def _make_point_to_plane(
        self, use_robust_kernel: bool = False, kernel_k: float = 0.01
    ) -> o3d.pipelines.registration.TransformationEstimationPointToPlane:
        """Return a point-to-plane estimation, optionally with a Tukey kernel."""
        if use_robust_kernel:
            return o3d.pipelines.registration.TransformationEstimationPointToPlane(
                o3d.pipelines.registration.TukeyLoss(k=kernel_k)
            )
        return o3d.pipelines.registration.TransformationEstimationPointToPlane()

    def _compute_fpfh(
        self, cloud: o3d.geometry.PointCloud, voxel_size: float
    ) -> Tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
        """Downsample *cloud*, estimate normals, and compute FPFH features.

        Args:
            cloud: Input point cloud.
            voxel_size: Voxel size for downsampling (metres).

        Returns:
            ``(downsampled_cloud, fpfh_features)``
        """
        cloud_down = cloud.voxel_down_sample(voxel_size)
        cloud_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
        )
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            cloud_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100),
        )
        return cloud_down, fpfh

    # ------------------------------------------------------------------
    # Public API — registration methods
    # ------------------------------------------------------------------

    def register(
        self, source: o3d.geometry.PointCloud
    ) -> Tuple[np.ndarray, float]:
        """Register *source* to the canonical cloud via point-to-plane ICP.

        Returns:
            (transformation, fitness) where *transformation* is a 4x4 rigid
            matrix and *fitness* is the fraction of source points that found
            a correspondence within *max_correspondence_distance*.
        """
        self._ensure_target_normals()

        result = o3d.pipelines.registration.registration_icp(
            source,
            self._canonical,
            self._max_corr_dist,
            np.eye(4),
            self._make_point_to_plane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self._max_iter
            ),
        )
        logger.info(
            "ICP fitness=%.4f  RMSE=%.6f", result.fitness, result.inlier_rmse
        )
        return np.asarray(result.transformation), result.fitness

    def register_robust(
        self,
        source: o3d.geometry.PointCloud,
        *,
        kernel_k: float = 0.01,
    ) -> Tuple[np.ndarray, float]:
        """Register *source* using point-to-plane ICP with a Tukey robust kernel.

        The robust kernel downweights residuals larger than *kernel_k*, so
        fringe points whose true correspondences are absent in the target cloud
        contribute less to the estimated transform.

        Args:
            source: Source point cloud to align.
            kernel_k: Tukey loss scale parameter (metres).  Residuals beyond
                ~3k are effectively zeroed out.  Default ``0.01`` (1 cm) suits
                typical forearm-surface residuals.

        Returns:
            (transformation, fitness)
        """
        self._ensure_target_normals()

        result = o3d.pipelines.registration.registration_icp(
            source,
            self._canonical,
            self._max_corr_dist,
            np.eye(4),
            self._make_point_to_plane(use_robust_kernel=True, kernel_k=kernel_k),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self._max_iter
            ),
        )
        logger.info(
            "Robust ICP (k=%.4f) fitness=%.4f  RMSE=%.6f",
            kernel_k,
            result.fitness,
            result.inlier_rmse,
        )
        return np.asarray(result.transformation), result.fitness

    def register_multiscale(
        self,
        source: o3d.geometry.PointCloud,
        *,
        use_robust_kernel: bool = False,
        kernel_k: float = 0.01,
    ) -> Tuple[np.ndarray, float]:
        """Register *source* using coarse-to-fine multiscale ICP.

        Runs ICP at three progressively finer resolution levels
        ``[(voxel_size, max_correspondence_distance), ...]``, carrying the
        transform from each level as the initialisation for the next.

        Args:
            source: Source point cloud to align.
            use_robust_kernel: Apply a Tukey kernel at every scale.
            kernel_k: Tukey loss scale parameter (metres).

        Returns:
            (transformation, fitness) from the finest scale.
        """
        self._ensure_target_normals()

        scales = [(0.01, 0.05), (0.005, 0.02), (0.002, 0.01)]
        T = np.eye(4)
        fitness = 0.0

        for voxel_size, max_dist in scales:
            src_down = source.voxel_down_sample(voxel_size)
            tgt_down = self._canonical.voxel_down_sample(voxel_size)
            tgt_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=voxel_size * 2, max_nn=30
                )
            )

            result = o3d.pipelines.registration.registration_icp(
                src_down,
                tgt_down,
                max_dist,
                T,
                self._make_point_to_plane(
                    use_robust_kernel=use_robust_kernel, kernel_k=kernel_k
                ),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=self._max_iter
                ),
            )
            T = np.asarray(result.transformation)
            fitness = result.fitness
            logger.debug(
                "Multiscale ICP voxel=%.4f dist=%.4f fitness=%.4f RMSE=%.6f",
                voxel_size,
                max_dist,
                result.fitness,
                result.inlier_rmse,
            )

        logger.info("Multiscale ICP fitness=%.4f", fitness)
        return T, fitness

    def register_generalized(
        self, source: o3d.geometry.PointCloud
    ) -> Tuple[np.ndarray, float]:
        """Register *source* using Generalized ICP (GICP).

        GICP models the local surface covariance at each point, which
        implicitly downweights correspondences in uncertain or flat
        neighbourhoods.  Normals on the target are still required.

        Returns:
            (transformation, fitness)
        """
        self._ensure_target_normals()

        result = o3d.pipelines.registration.registration_generalized_icp(
            source,
            self._canonical,
            self._max_corr_dist,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self._max_iter
            ),
        )
        logger.info(
            "GICP fitness=%.4f  RMSE=%.6f", result.fitness, result.inlier_rmse
        )
        return np.asarray(result.transformation), result.fitness

    def register_trimmed(
        self,
        source: o3d.geometry.PointCloud,
        *,
        overlap_ratio: float = 0.85,
    ) -> Tuple[np.ndarray, float]:
        """Register *source* after removing fringe points by nearest-neighbour distance.

        Source points whose distance to the nearest target point ranks in the
        top ``1 - overlap_ratio`` fraction are assumed to be fringe points with
        no true correspondences and are excluded before ICP.  Falls back to the
        full cloud if fewer than 100 points survive trimming.

        Args:
            source: Source point cloud to align.
            overlap_ratio: Expected fractional overlap between source and
                target (0–1).  Default ``0.85`` retains the 85 % closest
                source points.

        Returns:
            (transformation, fitness) from ICP on the trimmed cloud.
        """
        self._ensure_target_normals()

        dists = np.asarray(source.compute_point_cloud_distance(self._canonical))
        threshold = np.percentile(dists, overlap_ratio * 100.0)
        keep_indices = np.where(dists <= threshold)[0]

        if len(keep_indices) < 100:
            logger.warning(
                "Trimmed cloud has only %d points (< 100); falling back to full cloud.",
                len(keep_indices),
            )
            trimmed = source
        else:
            trimmed = source.select_by_index(keep_indices.tolist())
            logger.debug(
                "Trimmed source: %d → %d points (overlap_ratio=%.2f).",
                len(source.points),
                len(trimmed.points),
                overlap_ratio,
            )

        result = o3d.pipelines.registration.registration_icp(
            trimmed,
            self._canonical,
            self._max_corr_dist,
            np.eye(4),
            self._make_point_to_plane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self._max_iter
            ),
        )
        logger.info(
            "Trimmed ICP fitness=%.4f  RMSE=%.6f", result.fitness, result.inlier_rmse
        )
        return np.asarray(result.transformation), result.fitness

    def register_global_then_local(
        self,
        source: o3d.geometry.PointCloud,
        *,
        voxel_size: float = 0.005,
    ) -> Tuple[np.ndarray, float]:
        """Register *source* via FPFH+RANSAC global alignment then ICP refinement.

        Suitable for larger displacements where identity initialisation fails.
        The global RANSAC step finds a coarse alignment from FPFH feature
        correspondences; ICP then refines it at full resolution.

        Args:
            source: Source point cloud to align.
            voxel_size: Voxel size for downsampling during feature extraction
                (metres).  Default ``0.005`` (5 mm).

        Returns:
            (transformation, fitness) from the ICP refinement step.
        """
        self._ensure_target_normals()

        src_down, src_fpfh = self._compute_fpfh(source, voxel_size)
        tgt_down, tgt_fpfh = self._compute_fpfh(self._canonical, voxel_size)

        distance_threshold = voxel_size * 1.5
        global_result = (
            o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                src_down,
                tgt_down,
                src_fpfh,
                tgt_fpfh,
                True,
                distance_threshold,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                3,
                [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                        0.9
                    ),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                        distance_threshold
                    ),
                ],
                o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
            )
        )
        T_global = np.asarray(global_result.transformation)
        logger.debug("RANSAC global fitness=%.4f", global_result.fitness)

        result = o3d.pipelines.registration.registration_icp(
            source,
            self._canonical,
            self._max_corr_dist,
            T_global,
            self._make_point_to_plane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self._max_iter
            ),
        )
        logger.info(
            "Global+Local ICP fitness=%.4f  RMSE=%.6f",
            result.fitness,
            result.inlier_rmse,
        )
        return np.asarray(result.transformation), result.fitness

    # ------------------------------------------------------------------
    # Public API — orchestration
    # ------------------------------------------------------------------

    def register_all(
        self,
        clouds: Dict[str, o3d.geometry.PointCloud],
        canonical_key: str,
        *,
        method: str = "vanilla",
    ) -> Dict[str, Tuple[np.ndarray, float]]:
        """Register every cloud in *clouds* to the canonical reference.

        The canonical cloud's own entry receives an identity transform with
        fitness 1.0.

        Args:
            clouds: Mapping ``{snapshot_key: point_cloud}`` where each key
                uniquely identifies a forearm snapshot (e.g.
                ``"video_stem:frame_id"``).
            canonical_key: The key in *clouds* that corresponds to the
                canonical cloud passed to the constructor.
            method: Registration method to use.  One of ``"vanilla"``,
                ``"robust"``, ``"multiscale"``, ``"generalized"``,
                ``"trimmed"``, ``"global"``.  Default ``"vanilla"``.

        Returns:
            ``{snapshot_key: (4x4_transform, fitness)}``.

        Raises:
            ValueError: If *method* is not a recognised registration method.
        """
        if method not in self._METHOD_MAP:
            raise ValueError(
                f"Unknown registration method {method!r}. "
                f"Expected one of: {sorted(self._METHOD_MAP)}"
            )
        register_fn = getattr(self, self._METHOD_MAP[method])

        transforms: Dict[str, Tuple[np.ndarray, float]] = {}

        for key, cloud in clouds.items():
            if key == canonical_key:
                transforms[key] = (np.eye(4), 1.0)
                logger.info(
                    "Snapshot '%s' is the canonical reference (identity).", key
                )
            else:
                T, fitness = register_fn(cloud)
                transforms[key] = (T, fitness)
                if fitness < 0.9:
                    logger.warning(
                        "Snapshot '%s' ICP fitness %.4f is below 0.9 threshold.",
                        key,
                        fitness,
                    )
        return transforms

    def register_all_to_average(
        self,
        clouds: Dict[str, o3d.geometry.PointCloud],
        canonical_key: str,
        *,
        method: str = "vanilla",
    ) -> Dict[str, Tuple[np.ndarray, float]]:
        """Register all clouds then re-centre transforms around their mean pose.

        Unlike :meth:`register_all`, which fixes the canonical cloud at the
        origin, this method computes the SE(3) mean of all aligned transforms
        and shifts the common frame to that centroid pose so that no single
        snapshot is privileged.

        Procedure:

        1. Align every non-canonical cloud to ``self._canonical`` via ICP
           (same as :meth:`register_all`).
        2. Compute the mean transform: average rotation via
           ``scipy.spatial.transform.Rotation.mean`` and average translation.
        3. Apply ``T_mean⁻¹`` to every transform so the output frame sits at
           the group's centroid pose.

        Args:
            clouds: Mapping ``{snapshot_key: point_cloud}``.
            canonical_key: Key of the cloud corresponding to
                ``self._canonical`` (receives identity before centring).
            method: Registration method to use (same options as
                :meth:`register_all`).  Default ``"vanilla"``.

        Returns:
            ``{snapshot_key: (4x4_transform, fitness)}`` centred on the mean.
            Fitness values are inherited from ICP (1.0 for the canonical entry).
        """
        from scipy.spatial.transform import Rotation  # optional dep, lazy import

        # Step 1 — register all to canonical (reuse existing logic)
        raw = self.register_all(clouds, canonical_key, method=method)

        # Step 2 — compute mean transform in SE(3) from non-canonical clouds only.
        # The canonical's transform is always I (an implementation artifact, not
        # a real observation), so including it would bias the mean toward itself.
        non_canonical_matrices = [
            T for k, (T, _) in raw.items() if k != canonical_key
        ]
        matrices_for_mean = non_canonical_matrices or [T for T, _ in raw.values()]
        mean_R = Rotation.from_matrix([T[:3, :3] for T in matrices_for_mean]).mean()
        mean_t = np.mean([T[:3, 3] for T in matrices_for_mean], axis=0)

        T_mean = np.eye(4)
        T_mean[:3, :3] = mean_R.as_matrix()
        T_mean[:3, 3] = mean_t
        T_mean_inv = np.linalg.inv(T_mean)

        # Step 3 — re-centre every transform around the mean
        centred: Dict[str, Tuple[np.ndarray, float]] = {}
        for key, (T, fitness) in raw.items():
            centred[key] = (T_mean_inv @ T, fitness)
            logger.info(
                "Snapshot '%s' centred transform (fitness=%.4f).", key, fitness
            )
        return centred

    def build_unified_cloud(
        self,
        clouds: Dict[str, o3d.geometry.PointCloud],
        transforms: Dict[str, Tuple[np.ndarray, float]],
        representative_key: str,
    ) -> o3d.geometry.PointCloud:
        """Return the representative cloud realigned to the common reference frame.

        All input clouds represent the same physical object captured at different
        positions/orientations.  Rather than concatenating them, this method
        applies the computed transform to one representative snapshot so the
        output is a single, correctly-oriented cloud of the object.

        Args:
            clouds: Mapping ``{snapshot_key: point_cloud}``.
            transforms: Per-snapshot ``{snapshot_key: (4x4_transform, fitness)}``
                as returned by :meth:`register_all` or
                :meth:`register_all_to_average`.
            representative_key: The snapshot to use as the output cloud.
                Typically the canonical key: in ``"reference"`` mode its
                transform is ``I`` (no movement); in ``"average"`` mode it
                carries ``T_mean⁻¹`` (shift to the mean frame).
        """
        cloud = clouds[representative_key]
        T, _ = transforms[representative_key]
        result = o3d.geometry.PointCloud(cloud)
        result.transform(T)
        logger.info(
            "Built unified cloud from '%s' in common frame (%d points).",
            representative_key,
            len(result.points),
        )
        return result

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    @staticmethod
    def save_unified_cloud(
        cloud: o3d.geometry.PointCloud, path: Path
    ) -> None:
        """Save the unified registered point cloud to *path* (PLY)."""
        PointCloudDataHandler.save(cloud, str(path))

    @staticmethod
    def save_transforms(
        transforms: Dict[str, Tuple[np.ndarray, float]],
        canonical_key: Optional[str],
        path: Path,
        *,
        mode: str = "reference",
        process_parameters: Optional[Dict[str, object]] = None,
    ) -> None:
        """Persist per-snapshot transforms as JSON.

        Schema::

            {
                "mode": "reference" | "average",
                "canonical_key": "<video_stem>:<frame_id>" | null,
                "parameters": {
                    "registration_method": "vanilla",
                    "max_correspondence_distance": 0.10,
                    "icp_max_iteration": 200
                },
                "transforms": {
                    "<video_stem>:<frame_id>": {
                        "matrix_4x4": [[...], ...],
                        "fitness": <float>
                    },
                    ...
                }
            }

        The ``"parameters"`` key is only present when *process_parameters* is
        provided.  Existing consumers that read only ``"transforms"`` are
        unaffected (additive schema change).

        Args:
            transforms: Per-snapshot ``{key: (4x4_matrix, fitness)}`` mapping.
            canonical_key: The ICP reference key.  ``None`` is written as
                ``null`` (used when no single cloud is privileged, e.g.
                ``mode="average"``).
            path: Destination JSON path.
            mode: Unification mode — ``"reference"`` or ``"average"``.
            process_parameters: Optional dict of ICP process parameters to
                persist (e.g. ``registration_method``,
                ``max_correspondence_distance``, ``icp_max_iteration``).
        """
        payload: Dict[str, object] = {
            "mode": mode,
            "canonical_key": canonical_key,
        }
        if process_parameters is not None:
            payload["parameters"] = process_parameters
        payload["transforms"] = {
            key: {
                "matrix_4x4": T.tolist(),
                "fitness": fitness,
            }
            for key, (T, fitness) in transforms.items()
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=4)
        logger.info("Saved registration transforms to %s", path)

    @staticmethod
    def load_transforms(
        path: Path,
    ) -> Optional[Dict[str, object]]:
        """Load a previously saved transforms JSON.

        Returns the full dict (with ``canonical_key`` and ``transforms``
        mapping) or ``None`` if the file does not exist.  Transform keys
        are composite strings ``"video_stem:frame_id"``.
        """
        if not path.exists():
            return None
        with open(path) as fh:
            raw = json.load(fh)
        # Convert matrix lists back to numpy arrays
        for key, entry in raw["transforms"].items():
            entry["matrix_4x4"] = np.asarray(entry["matrix_4x4"])
        return raw
