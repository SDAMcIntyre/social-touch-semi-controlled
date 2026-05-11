"""Unit tests for HandMotionManager._calculate_alignment_procrustes scale guard.

Covers tasks 2.1 and 2.3 of:
  docs/development/plans/active/fix-contact-detection-false-positives.md
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path

import numpy as np
import pytest

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
