"""Round-trip tests for RegionOfInterest centre-based fields.

Covers the unit-test items from the Testing Plan in:
  docs/development/plans/active/roi-gui-prefill-on-reprocess.md

Note: _find_existing_roi_for_group routing is verified by code review and
Phase 3 manual verification (the function's module-level imports require
hardware SDKs unavailable in this test environment).
"""

import json
import tempfile
from pathlib import Path

from preprocessing.forearm_extraction.models.forearm_parameters import (
    ForearmParameters,
    RegionOfInterest,
    Point,
)
from preprocessing.forearm_extraction.data_access.forearm_frame_parameters_filehandler import (
    ForearmFrameParametersFileHandler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_params(roi: RegionOfInterest) -> ForearmParameters:
    return ForearmParameters(
        video_filename="test_video.mp4",
        frame_ids=[10],
        representative_frame_id=10,
        region_of_interest=roi,
        frame_width=640,
        frame_height=480,
        fps=30.0,
        nframes=100,
        fourcc_str="mp4v",
    )


def _roundtrip(params: ForearmParameters) -> ForearmParameters:
    """Save a single ForearmParameters to a temp file and reload it."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        tmp_path = Path(f.name)
    try:
        ForearmFrameParametersFileHandler.save([params], tmp_path)
        loaded = ForearmFrameParametersFileHandler.load(tmp_path)
        assert loaded is not None and len(loaded) == 1
        return loaded[0]
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Tests: new-format round-trip (centre fields populated)
# ---------------------------------------------------------------------------

class TestNewFormatRoundtrip:
    def test_centre_fields_preserved_exactly(self):
        roi = RegionOfInterest(
            top_left_corner=Point(x=100, y=80),
            bottom_right_corner=Point(x=300, y=240),
            angle_deg=30.0,
            center_x=200.0,
            center_y=160.0,
            width=120.0,
            height=90.0,
        )
        loaded = _roundtrip(_make_params(roi)).region_of_interest
        assert abs(loaded.center_x - 200.0) < 1e-6
        assert abs(loaded.center_y - 160.0) < 1e-6
        assert abs(loaded.width  - 120.0) < 1e-6
        assert abs(loaded.height -  90.0) < 1e-6
        assert abs(loaded.angle_deg - 30.0) < 1e-6

    def test_aabb_still_preserved(self):
        roi = RegionOfInterest(
            top_left_corner=Point(x=50, y=30),
            bottom_right_corner=Point(x=250, y=200),
            angle_deg=45.0,
            center_x=150.0,
            center_y=115.0,
            width=100.0,
            height=80.0,
        )
        loaded = _roundtrip(_make_params(roi)).region_of_interest
        assert loaded.top_left_corner.x == 50
        assert loaded.top_left_corner.y == 30
        assert loaded.bottom_right_corner.x == 250
        assert loaded.bottom_right_corner.y == 200

    def test_zero_angle_roundtrip(self):
        roi = RegionOfInterest(
            top_left_corner=Point(x=0, y=0),
            bottom_right_corner=Point(x=100, y=60),
            angle_deg=0.0,
            center_x=50.0,
            center_y=30.0,
            width=100.0,
            height=60.0,
        )
        loaded = _roundtrip(_make_params(roi)).region_of_interest
        assert abs(loaded.center_x - 50.0) < 1e-6
        assert abs(loaded.angle_deg) < 1e-6

    def test_extreme_angle_roundtrip(self):
        roi = RegionOfInterest(
            top_left_corner=Point(x=10, y=10),
            bottom_right_corner=Point(x=110, y=110),
            angle_deg=89.0,
            center_x=60.0,
            center_y=60.0,
            width=80.0,
            height=80.0,
        )
        loaded = _roundtrip(_make_params(roi)).region_of_interest
        assert abs(loaded.angle_deg - 89.0) < 1e-6
        assert abs(loaded.center_x - 60.0) < 1e-6


# ---------------------------------------------------------------------------
# Tests: legacy-format round-trip (centre fields absent)
# ---------------------------------------------------------------------------

class TestLegacyFormatRoundtrip:
    def _write_legacy_json(self, path: Path) -> None:
        """Write a minimal JSON without the centre-based fields."""
        data = [
            {
                "video_filename": "legacy_video.mp4",
                "frame_ids": [5],
                "representative_frame_id": 5,
                "region_of_interest": {
                    "top_left_corner": {"x": 20, "y": 10},
                    "bottom_right_corner": {"x": 120, "y": 90},
                    "angle_deg": 15.0,
                },
                "frame_width": 320,
                "frame_height": 240,
                "fps": 25.0,
                "nframes": 50,
                "fourcc_str": "mp4v",
            }
        ]
        path.write_text(json.dumps(data))

    def test_legacy_loads_without_error(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            tmp_path = Path(f.name)
        try:
            self._write_legacy_json(tmp_path)
            loaded = ForearmFrameParametersFileHandler.load(tmp_path)
            assert loaded is not None and len(loaded) == 1
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_legacy_centre_fields_are_none(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            tmp_path = Path(f.name)
        try:
            self._write_legacy_json(tmp_path)
            roi = ForearmFrameParametersFileHandler.load(tmp_path)[0].region_of_interest
            assert roi.center_x is None
            assert roi.center_y is None
            assert roi.width is None
            assert roi.height is None
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_legacy_aabb_correct(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            tmp_path = Path(f.name)
        try:
            self._write_legacy_json(tmp_path)
            roi = ForearmFrameParametersFileHandler.load(tmp_path)[0].region_of_interest
            assert roi.top_left_corner.x == 20
            assert roi.top_left_corner.y == 10
            assert roi.angle_deg == 15.0
        finally:
            tmp_path.unlink(missing_ok=True)
