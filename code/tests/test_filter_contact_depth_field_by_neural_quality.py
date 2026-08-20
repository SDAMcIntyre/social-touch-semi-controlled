"""Tests for the merging-stage neural-quality filter on the contact depth field.

Everything here runs on synthetic fixtures in ``tmp_path``: a small Space-1
parquet sidecar and a merged CSV shaped like the real one (nerve-rate rows with
a NaN ``frame_index`` between the Kinect anchor rows).  No recording, no
Kinect SDK, no Open3D — the module under test knows only about two paths in and
one path out, which is exactly what makes it testable this way.

The assertions that matter are the ones about *not* changing things: a retained
row must reach the output bitwise unchanged, and provenance must be carried
through rather than reconstructed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow", reason="pyarrow is required for the parquet sidecar")

_SRC = Path(__file__).resolve().parent.parent / "src"
_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
for _p in (_SRC, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Load utils.should_process_task from its real file before importing the module
# under test, so conftest's bare stub for the 'utils' package root cannot block
# it.
_mod_name = "utils.should_process_task"
if _mod_name not in sys.modules:
    _mod_path = _SRC / "utils" / "should_process_task.py"
    _spec = importlib.util.spec_from_file_location(_mod_name, _mod_path)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_mod_name] = _mod
    _spec.loader.exec_module(_mod)

from preprocessing.motion_analysis.tactile_quantification.io.contact_depth_field_io import (  # noqa: E402
    COLUMN_DTYPES,
    COORDINATE_SPACE,
    PRODUCED_BY,
    SCHEMA_VERSION,
    SIGN_CONVENTION,
    UNITS,
    read_contact_depth_field,
    write_contact_depth_field_table,
)
from _4_merging.filter_contact_depth_field_by_neural_quality import (  # noqa: E402
    filter_contact_depth_field_by_neural_quality,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SOURCE_RECORDING = "2022-06-15_ST14-01_semicontrolled_block-order-02"

INPUT_METADATA: Dict[str, str] = {
    "schema_version": SCHEMA_VERSION,
    "coordinate_space": COORDINATE_SPACE,
    "units": UNITS,
    "sign_convention": SIGN_CONVENTION,
    "source_recording": SOURCE_RECORDING,
    "produced_by": PRODUCED_BY,
}


def _depth_table(frames: Sequence[int], vertices_per_frame: int = 3) -> pd.DataFrame:
    """Build a schema-conforming long-form table with deterministic values.

    Values are distinct per (frame, vertex) so a mis-selected row is visible,
    and irregular enough that a rounding or a downcast would show up in an
    ``np.array_equal`` comparison.
    """
    frame_index = []
    time_s = []
    x = []
    y = []
    z = []
    depth = []
    for frame in frames:
        for vertex in range(vertices_per_frame):
            frame_index.append(frame)
            time_s.append(frame / 30.0)
            x.append(100.0 + frame + vertex * 0.123456)
            y.append(-50.0 - frame - vertex * 0.654321)
            z.append(900.0 + frame * 0.5 + vertex)
            depth.append(-(frame * 0.0137 + vertex * 0.00931) - 1e-9)

    table = pd.DataFrame(
        {
            "frame_index": np.asarray(frame_index, dtype=np.int32),
            "time_s": np.asarray(time_s, dtype=np.float64),
            "x": np.asarray(x, dtype=np.float32),
            "y": np.asarray(y, dtype=np.float32),
            "z": np.asarray(z, dtype=np.float32),
            "signed_depth_mm": np.asarray(depth, dtype=np.float64),
        }
    )
    assert list(table.columns) == list(COLUMN_DTYPES)
    return table


def _write_depth_field(
    path: Path,
    frames: Sequence[int],
    *,
    vertices_per_frame: int = 3,
    metadata: Dict[str, str] = None,
) -> pd.DataFrame:
    table = _depth_table(frames, vertices_per_frame=vertices_per_frame)
    write_contact_depth_field_table(
        table, path, metadata=dict(INPUT_METADATA if metadata is None else metadata)
    )
    return table


def _write_merged_csv(
    path: Path,
    frames: Sequence[int],
    *,
    contact_frames: Sequence[int] = None,
    upsample: int = 3,
    frame_index_column: str = "frame_index",
) -> None:
    """Write a merged-CSV-shaped file: one anchor row per frame, NaN in between.

    ``upsample`` interpolated rows follow each anchor row, mirroring the ~33x
    scattering merging performs.  Those rows carry NaN in *every* Kinect column,
    ``contact_detected`` included — which is precisely the shape the filter must
    ignore rather than choke on.
    """
    if contact_frames is None:
        contact_frames = list(frames)
    contact_set = set(contact_frames)

    records = []
    for frame in frames:
        records.append(
            {
                frame_index_column: float(frame),
                "contact_detected": 1.0 if frame in contact_set else 0.0,
                "Nerve_freq": 12.5,
            }
        )
        for _ in range(upsample):
            records.append(
                {
                    frame_index_column: np.nan,
                    "contact_detected": np.nan,
                    "Nerve_freq": 12.5,
                }
            )
    pd.DataFrame.from_records(records).to_csv(path, index=False)


@pytest.fixture
def paths(tmp_path: Path) -> Dict[str, Path]:
    return {
        "depth_field": tmp_path / "space1_contact_depth_field.parquet",
        "csv": tmp_path / "ST14-01_semicontrolled_block-order-02_merged_data.csv",
        "output": tmp_path
        / "blocks_filtered"
        / "ST14-01_semicontrolled_block-order-02_contact_depth_field.parquet",
    }


# ---------------------------------------------------------------------------
# 1. Row selection
# ---------------------------------------------------------------------------


def test_all_frames_surviving_yields_a_row_identical_table(paths):
    frames = [10, 11, 12, 13]
    original = _write_depth_field(paths["depth_field"], frames)
    _write_merged_csv(paths["csv"], frames)

    returned = filter_contact_depth_field_by_neural_quality(
        paths["depth_field"], paths["csv"], paths["output"]
    )

    assert returned == paths["output"]
    result, _ = read_contact_depth_field(paths["output"])
    assert len(result) == len(original)
    for column, dtype in COLUMN_DTYPES.items():
        assert result[column].dtype == dtype
        assert np.array_equal(result[column].to_numpy(), original[column].to_numpy())


def test_partial_surviving_set_removes_exactly_the_complement(paths):
    frames = [10, 11, 12, 13, 14, 15]
    surviving = [10, 11, 12]
    original = _write_depth_field(paths["depth_field"], frames)
    _write_merged_csv(paths["csv"], surviving)

    filter_contact_depth_field_by_neural_quality(
        paths["depth_field"], paths["csv"], paths["output"]
    )

    result, _ = read_contact_depth_field(paths["output"])
    assert sorted(set(result["frame_index"].tolist())) == surviving
    assert len(result) == len(surviving) * 3

    expected = original[original["frame_index"].isin(surviving)]
    for column in COLUMN_DTYPES:
        assert np.array_equal(result[column].to_numpy(), expected[column].to_numpy())


def test_retained_values_are_bitwise_unchanged(paths):
    frames = [4, 5, 6, 7]
    original = _write_depth_field(paths["depth_field"], frames)
    _write_merged_csv(paths["csv"], [5, 7])

    filter_contact_depth_field_by_neural_quality(
        paths["depth_field"], paths["csv"], paths["output"]
    )

    result, _ = read_contact_depth_field(paths["output"])
    expected = original[original["frame_index"].isin([5, 7])]
    # Bit-identity, not approximate equality: this task removes rows and must
    # never touch a value, so a single rounded float is a failure.
    for column in ("x", "y", "z", "signed_depth_mm", "time_s"):
        assert result[column].to_numpy().tobytes() == expected[column].to_numpy().tobytes()


def test_nan_frame_index_rows_are_ignored_when_building_the_surviving_set(paths):
    frames = [20, 21, 22]
    _write_depth_field(paths["depth_field"], frames)
    # 30 interpolated NaN rows per anchor row, all with a NaN contact flag.
    _write_merged_csv(paths["csv"], [21], upsample=30)

    filter_contact_depth_field_by_neural_quality(
        paths["depth_field"], paths["csv"], paths["output"]
    )

    result, metadata = read_contact_depth_field(paths["output"])
    assert sorted(set(result["frame_index"].tolist())) == [21]
    assert metadata["frames_dropped"] == "2"


def test_depth_field_frame_absent_from_csv_is_counted_not_raised(paths):
    frames = [1, 2, 3, 4, 5]
    _write_depth_field(paths["depth_field"], frames)
    _write_merged_csv(paths["csv"], [1, 2])

    filter_contact_depth_field_by_neural_quality(
        paths["depth_field"], paths["csv"], paths["output"]
    )

    _, metadata = read_contact_depth_field(paths["output"])
    assert metadata["frames_dropped"] == "3"


# ---------------------------------------------------------------------------
# 2. Error boundaries
# ---------------------------------------------------------------------------


def test_missing_depth_field_raises(paths):
    _write_merged_csv(paths["csv"], [1, 2])
    with pytest.raises(FileNotFoundError):
        filter_contact_depth_field_by_neural_quality(
            paths["depth_field"], paths["csv"], paths["output"]
        )


def test_missing_filtered_csv_raises(paths):
    _write_depth_field(paths["depth_field"], [1, 2])
    with pytest.raises(FileNotFoundError):
        filter_contact_depth_field_by_neural_quality(
            paths["depth_field"], paths["csv"], paths["output"]
        )


def test_csv_without_frame_index_column_raises_naming_the_columns(paths):
    _write_depth_field(paths["depth_field"], [1, 2])
    _write_merged_csv(paths["csv"], [1, 2], frame_index_column="frame_id")

    with pytest.raises(ValueError, match="frame_index"):
        filter_contact_depth_field_by_neural_quality(
            paths["depth_field"], paths["csv"], paths["output"]
        )

    # The message must name the file and list what was actually found.
    with pytest.raises(ValueError, match="frame_id"):
        filter_contact_depth_field_by_neural_quality(
            paths["depth_field"], paths["csv"], paths["output"]
        )


def test_zero_retained_rows_raises(paths):
    _write_depth_field(paths["depth_field"], [10, 11, 12])
    # Frames the depth field never saw, and non-contacting so the
    # sidecar-disagreement check is not what fires.
    _write_merged_csv(paths["csv"], [90, 91], contact_frames=[])

    with pytest.raises(ValueError, match="survives the neural-quality filter"):
        filter_contact_depth_field_by_neural_quality(
            paths["depth_field"], paths["csv"], paths["output"]
        )
    assert not paths["output"].exists()


def test_csv_contact_frame_absent_from_depth_field_raises(paths):
    _write_depth_field(paths["depth_field"], [10, 11])
    _write_merged_csv(paths["csv"], [10, 11, 12], contact_frames=[10, 11, 12])

    with pytest.raises(ValueError, match="stale"):
        filter_contact_depth_field_by_neural_quality(
            paths["depth_field"], paths["csv"], paths["output"]
        )
    assert not paths["output"].exists()


def test_csv_non_contact_frame_absent_from_depth_field_is_fine(paths):
    """A surviving frame with no contact owes the sidecar no rows."""
    _write_depth_field(paths["depth_field"], [10, 11])
    _write_merged_csv(paths["csv"], [10, 11, 12], contact_frames=[10, 11])

    filter_contact_depth_field_by_neural_quality(
        paths["depth_field"], paths["csv"], paths["output"]
    )

    result, _ = read_contact_depth_field(paths["output"])
    assert sorted(set(result["frame_index"].tolist())) == [10, 11]


def test_input_metadata_missing_a_carried_key_raises(paths):
    incomplete = dict(INPUT_METADATA)
    del incomplete["source_recording"]
    _write_depth_field(paths["depth_field"], [1, 2], metadata=incomplete)
    _write_merged_csv(paths["csv"], [1, 2])

    with pytest.raises(ValueError, match="source_recording"):
        filter_contact_depth_field_by_neural_quality(
            paths["depth_field"], paths["csv"], paths["output"]
        )


# ---------------------------------------------------------------------------
# 3. Metadata / provenance
# ---------------------------------------------------------------------------


def test_metadata_carries_through_and_adds_the_merging_keys(paths):
    frames = [1, 2, 3, 4]
    _write_depth_field(paths["depth_field"], frames)
    _write_merged_csv(paths["csv"], [1, 3])

    filter_contact_depth_field_by_neural_quality(
        paths["depth_field"], paths["csv"], paths["output"]
    )

    _, metadata = read_contact_depth_field(paths["output"])
    for key, value in INPUT_METADATA.items():
        assert metadata[key] == value
    # No transform is applied here, so the space must not have moved.
    assert metadata["coordinate_space"] == "kinect_space_1"
    assert metadata["pipeline_stage"] == "merging"
    assert metadata["neural_quality_filtered"] == "true"
    assert metadata["frames_dropped"] == "2"
    assert metadata["source_artifact"] == str(paths["depth_field"])


# ---------------------------------------------------------------------------
# 4. Idempotency
# ---------------------------------------------------------------------------


def test_second_run_skips_and_writes_nothing(paths):
    _write_depth_field(paths["depth_field"], [1, 2, 3])
    _write_merged_csv(paths["csv"], [1, 2, 3])

    first = filter_contact_depth_field_by_neural_quality(
        paths["depth_field"], paths["csv"], paths["output"]
    )
    assert first == paths["output"]
    mtime = paths["output"].stat().st_mtime_ns

    second = filter_contact_depth_field_by_neural_quality(
        paths["depth_field"], paths["csv"], paths["output"]
    )
    assert second is None
    assert paths["output"].stat().st_mtime_ns == mtime


def test_force_processing_rewrites(paths):
    _write_depth_field(paths["depth_field"], [1, 2, 3])
    _write_merged_csv(paths["csv"], [1, 2, 3])

    filter_contact_depth_field_by_neural_quality(
        paths["depth_field"], paths["csv"], paths["output"]
    )
    returned = filter_contact_depth_field_by_neural_quality(
        paths["depth_field"], paths["csv"], paths["output"], force_processing=True
    )

    assert returned == paths["output"]
    result, _ = read_contact_depth_field(paths["output"])
    assert len(result) == 9


def test_deleting_only_the_output_regenerates_it(paths):
    _write_depth_field(paths["depth_field"], [1, 2, 3])
    _write_merged_csv(paths["csv"], [1, 2, 3])

    filter_contact_depth_field_by_neural_quality(
        paths["depth_field"], paths["csv"], paths["output"]
    )
    paths["output"].unlink()

    returned = filter_contact_depth_field_by_neural_quality(
        paths["depth_field"], paths["csv"], paths["output"]
    )
    assert returned == paths["output"]
    assert paths["output"].exists()


def test_a_stale_output_is_cleaned_before_being_rewritten(paths):
    _write_depth_field(paths["depth_field"], [1, 2, 3])
    _write_merged_csv(paths["csv"], [1, 2, 3])
    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    paths["output"].write_bytes(b"not a parquet file")

    filter_contact_depth_field_by_neural_quality(
        paths["depth_field"], paths["csv"], paths["output"], force_processing=True
    )

    result, metadata = read_contact_depth_field(paths["output"])
    assert len(result) == 9
    assert metadata["neural_quality_filtered"] == "true"
