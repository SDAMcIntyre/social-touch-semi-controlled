"""Unit tests for MeanDuringIffExtractor and MeanBeforeIffExtractor."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Stub analysis.touch_analytics to avoid running its heavy __init__.py.
for _pkg in ('analysis', 'analysis.touch_analytics', 'analysis.touch_analytics.representation'):
    if _pkg not in sys.modules:
        _mod = types.ModuleType(_pkg)
        _parts = _pkg.split('.')
        _pkg_dir = _SRC / Path(*_parts)
        if _pkg_dir.exists():
            _mod.__path__ = [str(_pkg_dir)]
        _mod.__package__ = _pkg
        sys.modules[_pkg] = _mod
    if '.' in _pkg:
        _parent_name, _child_name = _pkg.rsplit('.', 1)
        setattr(sys.modules[_parent_name], _child_name, sys.modules[_pkg])

from analysis.touch_analytics.representation.feature_characterization import (  # noqa: E402
    MeanDuringIffExtractor,
    MeanBeforeIffExtractor,
    get_feature_extractor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_group(n_frames: int = 400, **extra_cols) -> pd.DataFrame:
    """Return a minimal group DataFrame with common numeric columns."""
    rng = np.random.default_rng(0)
    data = {
        'Nerve_freq': np.zeros(n_frames),
        'contact_detected': np.zeros(n_frames, dtype=int),
        'pressure': rng.uniform(0.0, 1.0, size=n_frames),
        'velocity_x': rng.uniform(-1.0, 1.0, size=n_frames),
        # Orchestrator columns — must be excluded from features
        'block_order_id': np.ones(n_frames, dtype=int),
        'trial_id': np.ones(n_frames, dtype=int),
        'Nerve_spike': np.zeros(n_frames, dtype=int),
    }
    data.update(extra_cols)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# MeanDuringIffExtractor tests
# ---------------------------------------------------------------------------

class TestMeanDuringIffExtractor:

    def test_during_basic(self):
        """Mixed zero/non-zero Nerve_freq: means must use only non-zero frames."""
        group = _make_group(n_frames=10)
        # Frames 3, 4, 5 are active
        group.loc[3:5, 'Nerve_freq'] = 50.0
        group['pressure'] = np.arange(10, dtype=float)  # 0..9

        result = MeanDuringIffExtractor().extract(group, config={})

        active_pressures = group.loc[group['Nerve_freq'] > 0, 'pressure']
        expected_pressure_mean = active_pressures.mean()
        assert result['pressure_mean_during_iff'] == pytest.approx(expected_pressure_mean)

        # Nerve_freq itself should also be in output (it's numeric and not excluded)
        active_nerve = group.loc[group['Nerve_freq'] > 0, 'Nerve_freq']
        assert result['Nerve_freq_mean_during_iff'] == pytest.approx(active_nerve.mean())

    def test_during_fallback_contact(self):
        """Nerve_freq all zero; contact_detected has non-zero frames; use contact frames."""
        group = _make_group(n_frames=10)
        # contact active at frames 6, 7
        group.loc[6:7, 'contact_detected'] = 1
        group['pressure'] = np.arange(10, dtype=float)

        result = MeanDuringIffExtractor().extract(group, config={})

        expected = group.loc[group['contact_detected'] == 1, 'pressure'].mean()
        assert result['pressure_mean_during_iff'] == pytest.approx(expected)

    def test_during_both_zero(self):
        """Both Nerve_freq and contact_detected are all zero; output must be all NaN."""
        group = _make_group(n_frames=10)
        # contact_detected stays 0 (already default); Nerve_freq stays 0

        result = MeanDuringIffExtractor().extract(group, config={})

        for key, value in result.items():
            assert np.isnan(value), f"Expected NaN for {key}, got {value}"

    def test_during_all_active(self):
        """Nerve_freq > 0 for every frame; means equal overall column means."""
        group = _make_group(n_frames=20)
        group['Nerve_freq'] = 100.0  # all active

        result = MeanDuringIffExtractor().extract(group, config={})

        assert result['pressure_mean_during_iff'] == pytest.approx(group['pressure'].mean())
        assert result['velocity_x_mean_during_iff'] == pytest.approx(group['velocity_x'].mean())


# ---------------------------------------------------------------------------
# MeanBeforeIffExtractor tests
# ---------------------------------------------------------------------------

class TestMeanBeforeIffExtractor:

    def test_before_full_window(self):
        """First IFF frame at index 300; window is frames 50–299 (250-frame window)."""
        n = 400
        group = _make_group(n_frames=n)
        group.loc[300:, 'Nerve_freq'] = 80.0
        # Distinct pressure values so we can verify the exact window
        group['pressure'] = np.arange(n, dtype=float)

        result = MeanBeforeIffExtractor().extract(group, config={})

        expected = group['pressure'].iloc[50:300].mean()
        assert result['pressure_mean_before_iff'] == pytest.approx(expected)

    def test_before_clipped_window(self):
        """First IFF frame at index 100; window is clipped to frames 0–99."""
        n = 200
        group = _make_group(n_frames=n)
        group.loc[100:, 'Nerve_freq'] = 60.0
        group['pressure'] = np.arange(n, dtype=float)

        result = MeanBeforeIffExtractor().extract(group, config={})

        expected = group['pressure'].iloc[0:100].mean()
        assert result['pressure_mean_before_iff'] == pytest.approx(expected)

    def test_before_first_frame_active(self):
        """Activity starts at frame 0; all features must be NaN."""
        group = _make_group(n_frames=10)
        group['Nerve_freq'] = 50.0  # active from frame 0

        result = MeanBeforeIffExtractor().extract(group, config={})

        for key, value in result.items():
            assert np.isnan(value), f"Expected NaN for {key}, got {value}"

    def test_before_custom_window(self):
        """pre_iff_window_ms: 100 in config; verify 100-frame window is used."""
        n = 300
        group = _make_group(n_frames=n)
        group.loc[200:, 'Nerve_freq'] = 70.0
        group['pressure'] = np.arange(n, dtype=float)

        result = MeanBeforeIffExtractor().extract(group, config={'pre_iff_window_ms': 100})

        # Window: frames 100–199
        expected = group['pressure'].iloc[100:200].mean()
        assert result['pressure_mean_before_iff'] == pytest.approx(expected)

    def test_before_fallback_contact(self):
        """Nerve_freq all zero; fallback to contact_detected onset for before window."""
        n = 200
        group = _make_group(n_frames=n)
        # contact active from frame 100 onward
        group.loc[100:, 'contact_detected'] = 1
        group['pressure'] = np.arange(n, dtype=float)

        result = MeanBeforeIffExtractor().extract(group, config={})

        # Window: max(0, 100 - 250) = 0 through 99 → frames 0..99
        expected = group['pressure'].iloc[0:100].mean()
        assert result['pressure_mean_before_iff'] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Shared / error tests
# ---------------------------------------------------------------------------

class TestShared:

    def test_missing_nerve_freq_raises(self):
        """No Nerve_freq column must raise ValueError from either extractor."""
        group = pd.DataFrame({'pressure': [1.0, 2.0, 3.0]})

        with pytest.raises(ValueError, match="Nerve_freq"):
            MeanDuringIffExtractor().extract(group, config={})

        with pytest.raises(ValueError, match="Nerve_freq"):
            MeanBeforeIffExtractor().extract(group, config={})

    def test_excludes_orchestrator_columns(self):
        """block_order_id, trial_id, Nerve_spike must NOT appear in output keys."""
        group = _make_group(n_frames=10)
        group['Nerve_freq'] = 50.0  # all active

        during_result = MeanDuringIffExtractor().extract(group, config={})
        before_result = MeanBeforeIffExtractor().extract(group, config={})

        forbidden_prefixes = ('block_order_id', 'trial_id', 'Nerve_spike')
        for result, label in [(during_result, 'during'), (before_result, 'before')]:
            for key in result:
                for prefix in forbidden_prefixes:
                    assert not key.startswith(prefix), (
                        f"Orchestrator column '{prefix}' leaked into {label} output as '{key}'"
                    )

    def test_registry_lookup_during(self):
        """get_feature_extractor('mean_during_iff', {}) returns MeanDuringIffExtractor."""
        extractor = get_feature_extractor('mean_during_iff', {})
        assert isinstance(extractor, MeanDuringIffExtractor)

    def test_registry_lookup_before(self):
        """get_feature_extractor('mean_before_iff', {}) returns MeanBeforeIffExtractor."""
        extractor = get_feature_extractor('mean_before_iff', {})
        assert isinstance(extractor, MeanBeforeIffExtractor)
