"""Unit tests for HandMotionManager._calculate_alignment_procrustes scale guard
and PoseStabilisation.

Covers tasks 2.1 and 2.3 of:
  docs/development/plans/active/fix-contact-detection-false-positives.md
Covers Phase 6 of:
  docs/development/plans/active/stabilise-handmesh-pose.md
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_MODULE = "preprocessing.motion_analysis.hand_tracking.models.hand_motion_manager"

# Load the module directly to bypass the heavy motion_analysis/__init__.py which
# pulls in tactile_quantification → forearm_extraction SDK dependencies.
if _MODULE not in sys.modules:
    _mod_path = (
        _SRC
        / "preprocessing"
        / "motion_analysis"
        / "hand_tracking"
        / "models"
        / "hand_motion_manager.py"
    )
    # Ensure intermediate package stubs exist so relative imports in the module
    # resolve without executing their __init__.py files.
    for _pkg in (
        "preprocessing.motion_analysis",
        "preprocessing.motion_analysis.hand_tracking",
        "preprocessing.motion_analysis.hand_tracking.models",
    ):
        if _pkg not in sys.modules:
            _parts = _pkg.split(".")
            _pkg_dir = _SRC / Path(*_parts)
            _stub = types.ModuleType(_pkg)
            _stub.__path__ = [str(_pkg_dir)]
            _stub.__package__ = _pkg
            sys.modules[_pkg] = _stub

    _spec = importlib.util.spec_from_file_location(_MODULE, _mod_path)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_MODULE] = _mod
    _spec.loader.exec_module(_mod)

HandMotionManager = sys.modules[_MODULE].HandMotionManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager() -> HandMotionManager:
    return HandMotionManager(fps=30.0)


def _normal_stickers() -> tuple[np.ndarray, np.ndarray]:
    """Source and target sticker coordinates that produce a reasonable scale (~1.0)."""
    source = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.5],
    ], dtype=np.float32)
    target = np.array([
        [5.0, 5.0, 5.0],
        [6.0, 5.0, 5.0],
        [5.0, 6.0, 5.0],
        [5.5, 5.5, 5.5],
    ], dtype=np.float32)
    return source, target


def _degenerate_stickers() -> tuple[np.ndarray, np.ndarray]:
    """Sticker coordinates where all source points are identical (collapsed tracking).

    In the shifted frame P = source - source[0] = 0.  The Procrustes covariance
    matrix H = P.T @ Q = 0, so the SVD-derived R is the identity and P_rotated = 0.
    The numerator sum(P_rotated * Q) = 0 and the denominator sum(P*P) = 0, giving
    scale = 0 / (0 + 1e-8) ≈ 0.0, which satisfies `scale <= 0` and triggers the
    guard unconditionally.
    """
    source = np.array([
        [5.0, 5.0, 5.0],
        [5.0, 5.0, 5.0],
        [5.0, 5.0, 5.0],
        [5.0, 5.0, 5.0],
    ], dtype=np.float32)
    target = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return source, target


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestProcrustesScaleGuard:

    def test_procrustes_normal_scale_unchanged(self, caplog: pytest.LogCaptureFixture) -> None:
        """Normal sticker data produces a scale in [0.3, 5.0] and no warning."""
        manager = _make_manager()
        source, target = _normal_stickers()

        with caplog.at_level(logging.WARNING, logger=_MODULE):
            _, _, scale = manager._calculate_alignment_procrustes(source, target)

        assert 0.3 <= scale <= 5.0, f"Expected scale in [0.3, 5.0], got {scale}"
        assert caplog.records == [], (
            f"Expected no warnings for normal stickers, got: {[r.message for r in caplog.records]}"
        )

    def test_procrustes_negative_scale_clamped_to_previous(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Degenerate stickers produce scale=0; guard replaces with previous frame's scale."""
        manager = _make_manager()
        good_source, good_target = _normal_stickers()

        with caplog.at_level(logging.WARNING, logger=_MODULE):
            _, _, first_scale = manager._calculate_alignment_procrustes(good_source, good_target)

        assert first_scale > 0, "Setup: first frame must produce a positive scale"
        manager.scales.append(first_scale)

        bad_source, bad_target = _degenerate_stickers()
        with caplog.at_level(logging.WARNING, logger=_MODULE):
            _, _, recovered_scale = manager._calculate_alignment_procrustes(bad_source, bad_target)

        assert recovered_scale > 0, (
            f"Guard must produce a positive scale; got {recovered_scale}"
        )
        assert recovered_scale == pytest.approx(first_scale), (
            "Guard must return the previous frame's scale exactly"
        )
        warning_messages = [r.message for r in caplog.records]
        assert any("non-positive" in m for m in warning_messages), (
            f"Expected a 'non-positive' warning; got: {warning_messages}"
        )

    def test_procrustes_negative_scale_first_frame_raises(self) -> None:
        """If the very first frame produces scale <= 0, a ValueError must be raised
        because there is no previous scale to fall back to."""
        manager = _make_manager()
        assert manager.scales == [], "Manager must have no prior scales"

        bad_source, bad_target = _degenerate_stickers()

        with pytest.raises(ValueError, match="First frame has non-positive scale"):
            manager._calculate_alignment_procrustes(bad_source, bad_target)

    def test_rigid_basis_always_returns_unit_scale(self) -> None:
        """_calculate_alignment_basis always returns scale=1.0 regardless of input."""
        manager = _make_manager()
        source, target = _normal_stickers()
        _, _, scale = manager._calculate_alignment_basis(source, target)
        assert scale == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# PoseStabilisation loading
# ---------------------------------------------------------------------------
#
# Load PoseStabilisation and its dependency (MotionFilterFactory) directly from
# source files so no heavy __init__.py runs (same pattern as HandMotionManager
# above).  The filter stack only needs numpy + scipy.signal.

_MOTION_CORRECTION_PKG = (
    "preprocessing.stickers_analysis.xyz.motion_correction"
)
_FILTER_FACTORY_MODULE = f"{_MOTION_CORRECTION_PKG}.motion_filter_factory"
_FILTER_INTERFACE_MODULE = f"{_MOTION_CORRECTION_PKG}.motion_filter_interface"
_BUTTERWORTH_MODULE = f"{_MOTION_CORRECTION_PKG}.butterworth_filter"
_SAVGOL_MODULE = f"{_MOTION_CORRECTION_PKG}.savgol_filter"

_MOTION_CORRECTION_DIR = (
    _SRC
    / "preprocessing"
    / "stickers_analysis"
    / "xyz"
    / "motion_correction"
)

_STABILISATION_MODULE = (
    "preprocessing.motion_analysis.hand_tracking.pose_stabilisation"
)


def _load_module_from_file(dotted_name: str, file_path: Path) -> types.ModuleType:
    if dotted_name in sys.modules:
        return sys.modules[dotted_name]
    spec = importlib.util.spec_from_file_location(dotted_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ensure_motion_correction_stubs() -> None:
    """Ensure all intermediate stub packages exist for the motion_correction path."""
    for pkg in (
        "preprocessing.stickers_analysis.xyz",
        _MOTION_CORRECTION_PKG,
    ):
        if pkg not in sys.modules:
            parts = pkg.split(".")
            pkg_dir = _SRC / Path(*parts)
            stub = types.ModuleType(pkg)
            stub.__path__ = [str(pkg_dir)]
            stub.__package__ = pkg
            sys.modules[pkg] = stub


_ensure_motion_correction_stubs()

# Load the filter stack in dependency order (each subsequent module can resolve
# its relative imports because the parent package stub is already in sys.modules).
_load_module_from_file(
    _FILTER_INTERFACE_MODULE,
    _MOTION_CORRECTION_DIR / "motion_filter_interface.py",
)
_load_module_from_file(
    _BUTTERWORTH_MODULE,
    _MOTION_CORRECTION_DIR / "butterworth_filter.py",
)
_load_module_from_file(
    _SAVGOL_MODULE,
    _MOTION_CORRECTION_DIR / "savgol_filter.py",
)
_load_module_from_file(
    _FILTER_FACTORY_MODULE,
    _MOTION_CORRECTION_DIR / "motion_filter_factory.py",
)

# Now load PoseStabilisation itself.
_load_module_from_file(
    _STABILISATION_MODULE,
    _SRC
    / "preprocessing"
    / "motion_analysis"
    / "hand_tracking"
    / "pose_stabilisation.py",
)

PoseStabilisation = sys.modules[_STABILISATION_MODULE].PoseStabilisation
MotionFilterFactory = sys.modules[_FILTER_FACTORY_MODULE].MotionFilterFactory
_SCALE_MIN = sys.modules[_STABILISATION_MODULE]._SCALE_MIN
_SCALE_MAX = sys.modules[_STABILISATION_MODULE]._SCALE_MAX


# ---------------------------------------------------------------------------
# Synthetic data helpers for PoseStabilisation tests
# ---------------------------------------------------------------------------

_N_FRAMES = 30
_N_VERTICES = 10
_ANCHOR_IDX = 0
_FPS = 30.0
_RNG = np.random.default_rng(42)


def _identity_session(
    n_frames: int = _N_FRAMES,
    n_vertices: int = _N_VERTICES,
    anchor_idx: int = _ANCHOR_IDX,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (vertices, translations, rotations_xyzw, scales) for a static session.

    The rotation is the identity quaternion throughout.  The anchor vertex is
    placed at a fixed world position so the anchor invariant is trivially satisfied
    before and after stabilisation.
    """
    vertices = _RNG.uniform(-0.1, 0.1, (n_frames, n_vertices, 3))
    vertices[:, anchor_idx, :] = np.array([0.05, 0.02, 0.03])

    R = Rotation.identity()
    q = R.as_quat()  # XYZW
    rotations_xyzw = np.tile(q, (n_frames, 1))

    s0 = vertices[:, anchor_idx, :]
    t0 = np.tile(np.array([1.0, 2.0, 3.0]), (n_frames, 1))
    translations = t0 - scale * R.apply(s0)

    scales = np.full(n_frames, scale)
    return vertices, translations, rotations_xyzw, scales


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPoseStabilisation:

    def test_anchor_preserved(self) -> None:
        """After stabilisation with smooth_anchor=True (default), the anchor world
        position matches the smoothed t0 to < 1e-5 per frame."""
        vertices, translations, rotations_xyzw, scales = _identity_session(
            scale=1.0
        )
        t0_raw = translations + scales[:, np.newaxis] * Rotation.from_quat(
            rotations_xyzw
        ).apply(vertices[:, _ANCHOR_IDX, :])

        anchor_filter_params = {"butterworth": {"order": 2, "cutoff_hz": 5.0}}
        anchor_filt = MotionFilterFactory.get_filter("butterworth", anchor_filter_params)
        t0_smooth = t0_raw.copy()
        for axis in range(3):
            t0_smooth[:, axis] = anchor_filt.filter(t0_raw[:, axis], _FPS)

        translations_new, rotations_new, scales_new = PoseStabilisation.stabilise(
            vertices=vertices,
            translations=translations,
            rotations_xyzw=rotations_xyzw,
            scales=scales,
            anchor_idx=_ANCHOR_IDX,
            fps=_FPS,
        )

        R_smooth = Rotation.from_quat(rotations_new.astype(np.float64))
        s0 = vertices[:, _ANCHOR_IDX, :]
        anchor_reconstructed = (
            scales_new[:, np.newaxis] * R_smooth.apply(s0) + translations_new
        )
        err = np.linalg.norm(anchor_reconstructed - t0_smooth, axis=1)
        assert np.all(err < 1e-5), (
            f"Anchor invariant violated; max error = {err.max():.2e}"
        )

    def test_anchor_smoothing_disabled(self) -> None:
        """With smooth_anchor=False, the anchor world position matches the raw t0
        (original invariant) to < 1e-5 per frame."""
        vertices, translations, rotations_xyzw, scales = _identity_session(
            scale=1.0
        )
        t0_raw = translations + scales[:, np.newaxis] * Rotation.from_quat(
            rotations_xyzw
        ).apply(vertices[:, _ANCHOR_IDX, :])

        translations_new, rotations_new, scales_new = PoseStabilisation.stabilise(
            vertices=vertices,
            translations=translations,
            rotations_xyzw=rotations_xyzw,
            scales=scales,
            anchor_idx=_ANCHOR_IDX,
            fps=_FPS,
            smooth_anchor=False,
        )

        R_smooth = Rotation.from_quat(rotations_new.astype(np.float64))
        s0 = vertices[:, _ANCHOR_IDX, :]
        anchor_reconstructed = (
            scales_new[:, np.newaxis] * R_smooth.apply(s0) + translations_new
        )
        err = np.linalg.norm(anchor_reconstructed - t0_raw, axis=1)
        assert np.all(err < 1e-5), (
            f"Anchor invariant violated with smooth_anchor=False; max error = {err.max():.2e}"
        )

    def test_scale_locked_to_median(self) -> None:
        """All output scales equal the median of the input valid scales."""
        known_scale_pattern = np.array([1.0, 1.05, 0.95, 1.02, 0.98])
        n = _N_FRAMES
        known_scales = np.resize(known_scale_pattern, n)
        vertices = _RNG.uniform(-0.1, 0.1, (n, _N_VERTICES, 3))
        R = Rotation.identity()
        q = R.as_quat()
        rotations_xyzw = np.tile(q, (n, 1))
        s0 = vertices[:, _ANCHOR_IDX, :]
        t0 = np.ones((n, 3))
        translations = t0 - known_scales[:, np.newaxis] * R.apply(s0)

        _, _, scales_new = PoseStabilisation.stabilise(
            vertices=vertices,
            translations=translations,
            rotations_xyzw=rotations_xyzw,
            scales=known_scales,
            anchor_idx=_ANCHOR_IDX,
            fps=_FPS,
        )

        expected_median = float(np.median(known_scale_pattern))
        assert np.all(scales_new == pytest.approx(expected_median)), (
            f"Expected all scales == {expected_median}; got {scales_new}"
        )

    def test_scale_filters_out_of_range(self) -> None:
        """Out-of-range scales are excluded before computing the median.

        Builds _N_FRAMES scales by repeating a valid pattern and inserting two
        out-of-range values.  The median must equal the median of the valid
        values only.
        """
        n = _N_FRAMES
        valid_pattern = np.array([1.0, 1.05, 0.98])
        raw_scales = np.resize(valid_pattern, n).copy()
        raw_scales[0] = _SCALE_MIN * 0.5
        raw_scales[n // 2] = _SCALE_MAX * 2.0
        valid_mask = (raw_scales > _SCALE_MIN) & (raw_scales < _SCALE_MAX)
        valid_values = raw_scales[valid_mask]

        vertices = _RNG.uniform(-0.1, 0.1, (n, _N_VERTICES, 3))
        R = Rotation.identity()
        q = R.as_quat()
        rotations_xyzw = np.tile(q, (n, 1))
        s0 = vertices[:, _ANCHOR_IDX, :]
        t0 = np.ones((n, 3))
        translations = t0 - raw_scales[:, np.newaxis] * R.apply(s0)

        _, _, scales_new = PoseStabilisation.stabilise(
            vertices=vertices,
            translations=translations,
            rotations_xyzw=rotations_xyzw,
            scales=raw_scales,
            anchor_idx=_ANCHOR_IDX,
            fps=_FPS,
        )

        expected_median = float(np.median(valid_values))
        assert np.all(scales_new == pytest.approx(expected_median)), (
            f"Median should be computed from valid scales only; "
            f"expected {expected_median}, got {scales_new[0]}"
        )

    def test_raises_on_no_valid_scales(self) -> None:
        """ValueError is raised when all scales fall outside the valid range."""
        n = _N_FRAMES
        bad_scales = np.full(n, _SCALE_MAX * 5.0)
        vertices = _RNG.uniform(-0.1, 0.1, (n, _N_VERTICES, 3))
        R = Rotation.identity()
        q = R.as_quat()
        rotations_xyzw = np.tile(q, (n, 1))
        translations = np.zeros((n, 3))

        with pytest.raises(ValueError, match="No valid scales"):
            PoseStabilisation.stabilise(
                vertices=vertices,
                translations=translations,
                rotations_xyzw=rotations_xyzw,
                scales=bad_scales,
                anchor_idx=_ANCHOR_IDX,
                fps=_FPS,
            )

    def test_raises_on_short_session(self) -> None:
        """ValueError is raised when the session is shorter than the filter minimum."""
        filter_order = 2
        min_frames = 2 * filter_order + 1  # = 5 for order=2
        n = min_frames - 2  # 3 frames — below the threshold
        vertices = _RNG.uniform(-0.1, 0.1, (n, _N_VERTICES, 3))
        R = Rotation.identity()
        q = R.as_quat()
        rotations_xyzw = np.tile(q, (n, 1))
        translations = np.zeros((n, 3))
        scales = np.ones(n)

        with pytest.raises(ValueError, match="requires at least"):
            PoseStabilisation.stabilise(
                vertices=vertices,
                translations=translations,
                rotations_xyzw=rotations_xyzw,
                scales=scales,
                anchor_idx=_ANCHOR_IDX,
                fps=_FPS,
                filter_method="butterworth",
                filter_params={"butterworth": {"order": filter_order}},
            )

    def test_rotation_smoothed(self) -> None:
        """Output rotvecs have lower total variation than input after high-freq jitter."""
        t = np.linspace(0, 1, _N_FRAMES)
        slow_trend = 0.3 * t
        jitter = 0.15 * np.sin(2 * np.pi * 12 * t)
        rotvec_sequence = np.column_stack([
            slow_trend + jitter,
            np.zeros(_N_FRAMES),
            np.zeros(_N_FRAMES),
        ])
        rotations_in = Rotation.from_rotvec(rotvec_sequence)
        rotations_xyzw = rotations_in.as_quat()

        scale = 1.0
        vertices = _RNG.uniform(-0.1, 0.1, (_N_FRAMES, _N_VERTICES, 3))
        s0 = vertices[:, _ANCHOR_IDX, :]
        t0 = np.tile(np.array([1.0, 2.0, 3.0]), (_N_FRAMES, 1))
        translations = t0 - scale * rotations_in.apply(s0)
        scales = np.full(_N_FRAMES, scale)

        _, rotations_new, _ = PoseStabilisation.stabilise(
            vertices=vertices,
            translations=translations,
            rotations_xyzw=rotations_xyzw,
            scales=scales,
            anchor_idx=_ANCHOR_IDX,
            fps=_FPS,
        )

        rotvecs_out = Rotation.from_quat(rotations_new.astype(np.float64)).as_rotvec()
        tv_in = float(np.sum(np.abs(np.diff(rotvec_sequence[:, 0]))))
        tv_out = float(np.sum(np.abs(np.diff(rotvecs_out[:, 0]))))
        assert tv_out < tv_in, (
            f"Output should be smoother than input; TV in={tv_in:.4f}, out={tv_out:.4f}"
        )

    def test_raw_npz_unchanged(self) -> None:
        """Running stabilisation does not alter the raw NPZ on disk."""
        vertices, translations, rotations_xyzw, scales = _identity_session()

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            raw_path = Path(f.name)

        try:
            np.savez(
                raw_path,
                vertices=vertices,
                translations=translations,
                rotations=rotations_xyzw,
                scales=scales,
            )
            raw_bytes_before = raw_path.read_bytes()

            PoseStabilisation.stabilise(
                vertices=vertices,
                translations=translations,
                rotations_xyzw=rotations_xyzw,
                scales=scales,
                anchor_idx=_ANCHOR_IDX,
                fps=_FPS,
            )

            raw_bytes_after = raw_path.read_bytes()
            assert raw_bytes_before == raw_bytes_after, (
                "stabilise() must not modify the raw NPZ file on disk"
            )
        finally:
            raw_path.unlink(missing_ok=True)

    def test_sign_fixup(self) -> None:
        """A trajectory with a quaternion sign flip produces consistent output rotations."""
        angle_sequence = np.linspace(0.1, 0.6, _N_FRAMES)
        rotvecs = np.column_stack([
            angle_sequence,
            np.zeros(_N_FRAMES),
            np.zeros(_N_FRAMES),
        ])
        rotations_in = Rotation.from_rotvec(rotvecs)
        quats = rotations_in.as_quat().copy()

        flip_frame = _N_FRAMES // 2
        quats[flip_frame:] = -quats[flip_frame:]

        scale = 1.0
        vertices = _RNG.uniform(-0.1, 0.1, (_N_FRAMES, _N_VERTICES, 3))
        s0 = vertices[:, _ANCHOR_IDX, :]
        t0 = np.tile(np.array([1.0, 2.0, 3.0]), (_N_FRAMES, 1))
        translations = t0 - scale * rotations_in.apply(s0)
        scales = np.full(_N_FRAMES, scale)

        _, rotations_new, _ = PoseStabilisation.stabilise(
            vertices=vertices,
            translations=translations,
            rotations_xyzw=quats,
            scales=scales,
            anchor_idx=_ANCHOR_IDX,
            fps=_FPS,
        )

        R_out = Rotation.from_quat(rotations_new.astype(np.float64))
        R_expected = Rotation.from_rotvec(rotvecs)
        for i in range(_N_FRAMES):
            # The relative rotation between expected and output should be near identity.
            diff = (R_expected[i].inv() * R_out[i]).magnitude()
            assert diff < 0.05, (
                f"Frame {i}: output rotation deviates from expected by {diff:.4f} rad "
                "(sign-fixup artefact suspected)"
            )
