from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from scipy.spatial.transform import Rotation

from preprocessing.stickers_analysis.xyz.motion_correction.motion_filter_factory import (
    MotionFilterFactory,
)

_SCALE_MIN = 0.1
_SCALE_MAX = 10.0


class PoseStabilisation:
    """Applies a 4-step post-generation correction to a MANO hand-mesh pose stream.

    Corrects two artefacts caused by per-frame Procrustes sensitivity to depth noise:
    1. Size breathing — per-frame scale variation despite fixed physical hand size.
    2. Rotational jitter — high-frequency noise in the rotation stream.

    Translation is never filtered directly; it is re-derived from the anchor constraint
    after scale and rotation are finalised, preserving the exact world position of the
    blue sticker vertex.
    """

    @staticmethod
    def stabilise(
        vertices: np.ndarray,
        translations: np.ndarray,
        rotations_xyzw: np.ndarray,
        scales: np.ndarray,
        anchor_idx: int,
        fps: float,
        filter_method: Union[str, Any] = "butterworth",
        filter_params: Optional[Dict[str, Any]] = None,
        smooth_anchor: bool = True,
        anchor_filter_method: Union[str, Any] = "one_euro",
        anchor_filter_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply the 4-step pose correction to a full session.

        Args:
            vertices: (N, V, 3) array of untransformed mesh vertices per frame.
            translations: (N, 3) array of per-frame translations.
            rotations_xyzw: (N, 4) array of per-frame quaternions in XYZW order.
            scales: (N,) array of per-frame uniform scale factors.
            anchor_idx: Index into the vertex array for the blue sticker (anchor) vertex.
                        Typically ``sticker_vertex_indices[0]``.
            fps: Recording frame rate in Hz.
            filter_method: Filter name accepted by :class:`MotionFilterFactory`
                           (``"butterworth"`` or ``"savgol"``).
            filter_params: Forwarded to :meth:`MotionFilterFactory.get_filter` for the
                           rotation (and scale) filter.
            smooth_anchor: When ``True`` (default), the reconstructed anchor positions
                           ``t0`` are filtered before translation is re-derived. The
                           anchor invariant then becomes ``anchor_world == t0_smooth``
                           rather than ``anchor_world == t0_raw``.
            anchor_filter_method: Filter name for the anchor smoothing step. Defaults to
                                  ``"one_euro"``. Any filter registered in
                                  :class:`MotionFilterFactory` is accepted.
            anchor_filter_params: Filter params for the anchor smoothing step, in the
                                  same format as ``filter_params``. Defaults to
                                  ``{"one_euro": {"min_cutoff": 1.0, "beta": 0.007, "d_cutoff": 1.0}}``
                                  when ``None`` and ``smooth_anchor`` is ``True``.

        Returns:
            Tuple ``(translations_new, rotations_new, scales_new)`` where:
            - ``translations_new`` is (N, 3) float32
            - ``rotations_new`` is (N, 4) float32 XYZW
            - ``scales_new`` is (N,) float32 (all entries equal to the session median)

        Raises:
            ValueError: If ``N < min_frames_required`` for the chosen filter,
                        if no valid scales remain after range filtering, or if
                        NaN quaternions are detected in the input.
        """
        translations = np.asarray(translations, dtype=np.float64)
        rotations_xyzw = np.asarray(rotations_xyzw, dtype=np.float64)
        scales = np.asarray(scales, dtype=np.float64)
        vertices = np.asarray(vertices, dtype=np.float64)

        n_frames = len(translations)

        # Validate minimum frame count for the chosen filter before doing any work.
        filt = MotionFilterFactory.get_filter(filter_method, filter_params)
        min_frames = PoseStabilisation._min_frames_required(filter_method, filter_params)
        if n_frames < min_frames:
            raise ValueError(
                f"Session has {n_frames} frame(s) but the chosen filter "
                f"('{filter_method}') requires at least {min_frames} frames."
            )

        # --- Step 1: Reconstruct measured sticker positions (t0) ---
        # Inverse of the Procrustes translation formula: t = t0 - s * R @ s0
        # => t0 = t + s * R @ s0
        # Q: why not store t0 in the NPZ?  The raw NPZ does not have t0; re-deriving
        # it here avoids a schema change and keeps the raw NPZ minimal.
        R_orig = Rotation.from_quat(rotations_xyzw)
        s0 = vertices[:, anchor_idx, :]  # (N, 3) — anchor vertex per frame
        t0 = translations + scales[:, np.newaxis] * R_orig.apply(s0)  # (N, 3)

        # --- Step 1.5: Smooth anchor positions (t0) ---
        # Filtering t0 before translation re-derivation removes the dominant
        # noise source: raw sticker measurements feed directly into t_new via
        # t_new = t0 - s_stable * R_smooth @ s0. With smooth R and s, any t0
        # noise propagates unchanged into the final translation.
        if smooth_anchor:
            if anchor_filter_params is None:
                anchor_filter_params = {"one_euro": {"min_cutoff": 1.0, "beta": 0.007, "d_cutoff": 1.0}}
            anchor_filt = MotionFilterFactory.get_filter(anchor_filter_method, anchor_filter_params)
            for axis in range(3):
                t0[:, axis] = anchor_filt.filter(t0[:, axis], fps)

        # --- Step 2: Smooth rotations ---
        # Quaternions live on a double-cover of SO(3): q and -q represent the same
        # rotation. Naive interpolation or filtering can jump between the two covers,
        # creating a 2x-amplitude artefact. The sign-fixup loop below ensures all
        # quaternions point into the same hemisphere before converting to rotvec.
        rotations_fixed = rotations_xyzw.copy()
        for i in range(1, n_frames):
            if np.any(np.isnan(rotations_fixed[i])):
                raise ValueError(
                    f"NaN quaternion detected at frame {i}. "
                    "Repair or interpolate NaN frames before running stabilisation."
                )
            if np.dot(rotations_fixed[i], rotations_fixed[i - 1]) < 0.0:
                rotations_fixed[i] = -rotations_fixed[i]

        # Validate frame 0 for NaN as well (loop above only checks i >= 1).
        if np.any(np.isnan(rotations_fixed[0])):
            raise ValueError(
                "NaN quaternion detected at frame 0. "
                "Repair or interpolate NaN frames before running stabilisation."
            )

        rotvecs = Rotation.from_quat(rotations_fixed).as_rotvec()  # (N, 3)

        # Filter each of the 3 rotvec components independently.
        rotvecs_smooth = np.empty_like(rotvecs)
        for component in range(3):
            rotvecs_smooth[:, component] = filt.filter(rotvecs[:, component], fps)

        rotations_smooth = Rotation.from_rotvec(rotvecs_smooth)

        # --- Step 3: Lock scale to session median ---
        valid_mask = (scales > _SCALE_MIN) & (scales < _SCALE_MAX)
        if not np.any(valid_mask):
            raise ValueError(
                f"No valid scales remain after filtering for range "
                f"({_SCALE_MIN}, {_SCALE_MAX}). All {n_frames} frame scales are "
                "outside the expected range. Check the Procrustes alignment output."
            )
        s_stable = float(np.median(scales[valid_mask]))

        # --- Step 4: Re-derive translation from anchor constraint ---
        # t_new = t0 - s_stable * R_smooth @ s0
        # Substituting back confirms the anchor is preserved:
        # s_stable * R_smooth @ s0 + t_new = t0
        t_new = t0 - s_stable * rotations_smooth.apply(s0)  # (N, 3)

        translations_new = t_new.astype(np.float32)
        rotations_new = rotations_smooth.as_quat().astype(np.float32)
        scales_new = np.full(n_frames, s_stable, dtype=np.float32)

        return translations_new, rotations_new, scales_new

    @staticmethod
    def _min_frames_required(
        filter_method: Union[str, Any],
        filter_params: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Return the minimum number of frames needed by the given filter."""
        if filter_params is None:
            filter_params = {}

        method = filter_method if isinstance(filter_method, str) else str(filter_method)

        if method == "butterworth":
            bw_params = filter_params.get("butterworth", {})
            order = bw_params.get("order", 2)
            return 2 * order + 1

        # Savgol and any future filters: default to window_length if available,
        # otherwise 1 (no minimum beyond having at least one frame).
        if method == "savgol":
            sg_params = filter_params.get("savgol", {})
            return sg_params.get("window_length", 11)

        if method == "one_euro":
            return 2

        return 1
