"""Unit tests for adjust_nerve_conduction_velocity."""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
for _p in (_SRC, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Manually load utils.should_process_task before importing the module under test
# so conftest's bare-stub for 'utils' does not block it.
_mod_name = "utils.should_process_task"
if _mod_name not in sys.modules:
    _mod_path = _SRC / "utils" / "should_process_task.py"
    spec = importlib.util.spec_from_file_location(_mod_name, _mod_path)
    _mod = importlib.util.module_from_spec(spec)
    sys.modules[_mod_name] = _mod
    spec.loader.exec_module(_mod)

from _3_preprocessing._8_nerve_velocity_adjustment.adjust_nerve_conduction_velocity import (
    adjust_nerve_conduction_velocity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_N = 1000
_DT = 0.001  # 1 ms spacing → 1000 Hz

# ST14-01: velocity=50 m/s, distance=5cm → lag_sec=0.001 → lag_nsample=1
_UNIT = "ST14-01"
_VELOCITY = 50.0
_DISTANCE_CM = 5.0
_EXPECTED_LAG_SEC = (_DISTANCE_CM / 100) / _VELOCITY  # 0.001
_EXPECTED_LAG_NSAMPLE = 1


def _make_nerve_csv(tmp_path: Path, filename: str | None = None) -> Path:
    filename = filename or f"2022-06-15_{_UNIT}_semicontrolled_block-order01_nerve.csv"
    sec = np.arange(_N) * _DT
    df = pd.DataFrame({
        "Sec_FromStart": sec,
        "Nervespike1": np.ones(_N),
        "Freq": np.ones(_N) * 2.0,
    })
    p = tmp_path / filename
    df.to_csv(p, index=False)
    return p


def _make_metadata_csv(tmp_path: Path, velocity: float = _VELOCITY) -> Path:
    df = pd.DataFrame({
        "Unit_name": [_UNIT],
        "conduction_velocity (m/s)": [velocity],
        "electrode_endorgan_distance (cm)": [_DISTANCE_CM],
    })
    p = tmp_path / "metadata.csv"
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAdjustNerveConductionVelocity:

    def test_basic_shift(self, tmp_path: Path) -> None:
        nerve_csv = _make_nerve_csv(tmp_path)
        meta_csv = _make_metadata_csv(tmp_path)
        out_csv = tmp_path / "output" / "adjusted.csv"

        result = adjust_nerve_conduction_velocity(nerve_csv, meta_csv, out_csv)

        assert out_csv.exists()
        assert isinstance(result, dict)
        assert set(result.keys()) == {"filename", "lag_sec", "lag_nsample"}
        assert result["lag_nsample"] == _EXPECTED_LAG_NSAMPLE
        assert abs(result["lag_sec"] - _EXPECTED_LAG_SEC) < 1e-9

        output_df = pd.read_csv(out_csv)
        # After shifting by -1 and fill_value=0, the last row must be 0
        assert output_df["Nervespike1"].iloc[-1] == 0
        assert output_df["Freq"].iloc[-1] == 0
        # First N - lag_nsample rows should match original shifted values (all ones)
        assert output_df["Nervespike1"].iloc[0] == 1.0
        assert output_df["Freq"].iloc[0] == 2.0

    def test_idempotency_skip(self, tmp_path: Path) -> None:
        nerve_csv = _make_nerve_csv(tmp_path)
        meta_csv = _make_metadata_csv(tmp_path)
        out_csv = tmp_path / "adjusted.csv"

        # First run — produce output
        adjust_nerve_conduction_velocity(nerve_csv, meta_csv, out_csv)
        assert out_csv.exists()

        original_content = out_csv.read_text()

        # Wait to ensure any timestamp difference would be detectable
        time.sleep(0.01)

        # Second run with force=False — should skip
        result = adjust_nerve_conduction_velocity(nerve_csv, meta_csv, out_csv, force_processing=False)

        assert result is None
        assert out_csv.read_text() == original_content

    def test_force_reprocessing(self, tmp_path: Path) -> None:
        nerve_csv = _make_nerve_csv(tmp_path)
        meta_csv = _make_metadata_csv(tmp_path)
        out_csv = tmp_path / "adjusted.csv"

        # First run
        adjust_nerve_conduction_velocity(nerve_csv, meta_csv, out_csv)
        mtime_after_first = out_csv.stat().st_mtime

        time.sleep(0.05)

        # Second run with force=True — should regenerate
        result = adjust_nerve_conduction_velocity(nerve_csv, meta_csv, out_csv, force_processing=True)

        assert result is not None
        assert isinstance(result, dict)
        assert out_csv.stat().st_mtime >= mtime_after_first

    def test_missing_unit_raises(self, tmp_path: Path) -> None:
        nerve_csv = _make_nerve_csv(tmp_path)
        # Metadata has a different unit name
        df = pd.DataFrame({
            "Unit_name": ["XX00-99"],
            "conduction_velocity (m/s)": [50.0],
            "electrode_endorgan_distance (cm)": [5.0],
        })
        meta_csv = tmp_path / "metadata.csv"
        df.to_csv(meta_csv, index=False)
        out_csv = tmp_path / "adjusted.csv"

        with pytest.raises(ValueError, match="not found in metadata CSV"):
            adjust_nerve_conduction_velocity(nerve_csv, meta_csv, out_csv)

    def test_missing_column_nervespike1(self, tmp_path: Path) -> None:
        sec = np.arange(_N) * _DT
        df = pd.DataFrame({"Sec_FromStart": sec, "Freq": np.ones(_N)})
        nerve_csv = tmp_path / f"2022-06-15_{_UNIT}_nerve.csv"
        df.to_csv(nerve_csv, index=False)
        meta_csv = _make_metadata_csv(tmp_path)
        out_csv = tmp_path / "adjusted.csv"

        with pytest.raises(ValueError, match="Nervespike1"):
            adjust_nerve_conduction_velocity(nerve_csv, meta_csv, out_csv)

    def test_missing_column_freq(self, tmp_path: Path) -> None:
        sec = np.arange(_N) * _DT
        df = pd.DataFrame({"Sec_FromStart": sec, "Nervespike1": np.ones(_N)})
        nerve_csv = tmp_path / f"2022-06-15_{_UNIT}_nerve.csv"
        df.to_csv(nerve_csv, index=False)
        meta_csv = _make_metadata_csv(tmp_path)
        out_csv = tmp_path / "adjusted.csv"

        with pytest.raises(ValueError, match="Freq"):
            adjust_nerve_conduction_velocity(nerve_csv, meta_csv, out_csv)

    def test_fill_value_zero(self, tmp_path: Path) -> None:
        nerve_csv = _make_nerve_csv(tmp_path)
        meta_csv = _make_metadata_csv(tmp_path)
        out_csv = tmp_path / "adjusted.csv"

        result = adjust_nerve_conduction_velocity(nerve_csv, meta_csv, out_csv)
        lag = result["lag_nsample"]

        output_df = pd.read_csv(out_csv)
        # The last `lag` rows must be exactly 0, not NaN
        tail_nervespike = output_df["Nervespike1"].iloc[-lag:]
        tail_freq = output_df["Freq"].iloc[-lag:]

        assert not tail_nervespike.isna().any(), "Nervespike1 tail contains NaN instead of 0"
        assert not tail_freq.isna().any(), "Freq tail contains NaN instead of 0"
        assert (tail_nervespike == 0).all(), "Nervespike1 tail not filled with 0"
        assert (tail_freq == 0).all(), "Freq tail not filled with 0"

    def test_zero_velocity_raises(self, tmp_path: Path) -> None:
        nerve_csv = _make_nerve_csv(tmp_path)
        meta_csv = _make_metadata_csv(tmp_path, velocity=0.0)
        out_csv = tmp_path / "adjusted.csv"

        with pytest.raises(ValueError, match="zero"):
            adjust_nerve_conduction_velocity(nerve_csv, meta_csv, out_csv)
