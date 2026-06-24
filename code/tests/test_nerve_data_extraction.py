"""Unit tests for nerve data extraction pipeline (Phase 1 + Phase 2)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io

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

from _3_preprocessing._9_nerve_data_extraction.convert_nerve_mat_to_csv import (
    convert_nerve_mat_to_csv,
)
from _3_preprocessing._9_nerve_data_extraction.rename_nerve_to_block_order import (
    rename_nerve_to_block_order,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_N_ROWS = 4
_YYYYMMDD_ST14_01 = 20220616  # ST14-01 has -1 day correction → 20220615


def _make_mat_file(
    tmp_path: Path,
    session_key: str = "ST14-01",
    stimulus: str = "Semi_contr",
    n_blocks: int = 2,
    yyyymmdd: int = _YYYYMMDD_ST14_01,
) -> Path:
    """Create a synthetic .mat file using scipy.io.savemat and return its path.

    The structure mirrors the MATLAB nerve recording struct:
      S.Exp, S.UnitName, S.UnitNumber, S.IdxInDataInfo, S.UnitType,
      S.Stimulus, S.FullPeriod_D.ContD(b).D  (table with YYYYMMDD, Nervespike1 cols)
    """
    block_dtype = np.dtype([("YYYYMMDD", np.int32), ("Nervespike1", np.float64)])
    blocks = []
    for _ in range(n_blocks):
        D = np.zeros(_N_ROWS, dtype=block_dtype)
        D["YYYYMMDD"] = yyyymmdd
        D["Nervespike1"] = 1.0
        blocks.append({"D": D})

    ContD = np.empty((1, n_blocks), dtype=object)
    for b, blk in enumerate(blocks):
        ContD[0, b] = blk

    mat_data = {
        "S": {
            "Exp": np.array([[1]], dtype=np.int32),
            "UnitName": np.array([[session_key]], dtype=object),
            "UnitNumber": np.array([[1]], dtype=np.int32),
            "IdxInDataInfo": np.array([[1]], dtype=np.int32),
            "UnitType": np.array([["SA"]], dtype=object),
            "Stimulus": np.array([[stimulus]], dtype=object),
            "FullPeriod_D": {
                "ContD": ContD,
            },
        }
    }

    mat_path = tmp_path / f"{session_key}_nerve.mat"
    scipy.io.savemat(str(mat_path), mat_data)
    return mat_path


def _read_metadata(session_folder: Path, mat_stem: str) -> dict[str, str]:
    meta_file = session_folder / f"{mat_stem}_metadata.txt"
    assert meta_file.exists(), f"Metadata file not found: {meta_file}"
    result = {}
    for line in meta_file.read_text().splitlines():
        if ": " in line:
            k, v = line.split(": ", 1)
            result[k.strip()] = v.strip()
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNonSemiContrSkipped:

    def test_non_semicontr_returns_empty_list(self, tmp_path: Path) -> None:
        mat_path = _make_mat_file(tmp_path, stimulus="Passive")
        output_dir = tmp_path / "output"

        result = convert_nerve_mat_to_csv(mat_path, output_dir)

        assert result == [], f"Expected empty list for non-Semi_contr, got {result!r}"
        assert not any(output_dir.iterdir()), "No CSVs should be written for non-Semi_contr"

    def test_non_semicontr_does_not_write_csvs(self, tmp_path: Path) -> None:
        mat_path = _make_mat_file(tmp_path, stimulus="Active")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = convert_nerve_mat_to_csv(mat_path, output_dir)

        assert result == []
        csv_files = list(output_dir.rglob("*.csv"))
        assert len(csv_files) == 0


class TestDateCorrection:

    def test_date_correction_applied_for_st14_01(self, tmp_path: Path) -> None:
        mat_path = _make_mat_file(
            tmp_path, session_key="ST14-01", yyyymmdd=_YYYYMMDD_ST14_01
        )
        output_dir = tmp_path / "output"

        result = convert_nerve_mat_to_csv(mat_path, output_dir)

        assert result is not None
        assert len(result) == 1
        assert result[0]["date_corrected"] is True

        session_folder = Path(result[0]["session_folder"])
        # ST14-01 with -1 day: 20220616 → 20220615 → "2022-06-15"
        assert "2022-06-15" in session_folder.name, (
            f"Expected date-corrected folder name with '2022-06-15', got: {session_folder.name}"
        )

        # Verify YYYYMMDD in CSV reflects the correction
        csv_files = sorted(session_folder.glob("*_block*_table.csv"))
        assert len(csv_files) > 0
        import pandas as pd
        df = pd.read_csv(csv_files[0])
        assert (df["YYYYMMDD"] == 20220615).all(), (
            f"Expected YYYYMMDD=20220615 after correction, got: {df['YYYYMMDD'].unique()}"
        )

    def test_date_correction_applied_for_st16_02(self, tmp_path: Path) -> None:
        mat_path = _make_mat_file(
            tmp_path, session_key="ST16-02", yyyymmdd=20220620
        )
        output_dir = tmp_path / "output"

        result = convert_nerve_mat_to_csv(mat_path, output_dir)

        assert result is not None and len(result) == 1
        assert result[0]["date_corrected"] is True
        session_folder = Path(result[0]["session_folder"])
        # ST16-02 with +2 days: 20220620 → 20220622 → "2022-06-22"
        assert "2022-06-22" in session_folder.name

    def test_date_correction_skipped_for_unknown_session(self, tmp_path: Path) -> None:
        mat_path = _make_mat_file(
            tmp_path, session_key="ST15-01", yyyymmdd=20220616
        )
        output_dir = tmp_path / "output"

        result = convert_nerve_mat_to_csv(mat_path, output_dir)

        assert result is not None and len(result) == 1
        assert result[0]["date_corrected"] is False

        session_folder = Path(result[0]["session_folder"])
        assert "2022-06-16" in session_folder.name, (
            f"Expected unchanged date '2022-06-16', got: {session_folder.name}"
        )

        import pandas as pd
        csv_files = sorted(session_folder.glob("*_block*_table.csv"))
        df = pd.read_csv(csv_files[0])
        assert (df["YYYYMMDD"] == 20220616).all()


class TestCsvOutputWritten:

    def test_block_csvs_created(self, tmp_path: Path) -> None:
        n_blocks = 3
        mat_path = _make_mat_file(tmp_path, n_blocks=n_blocks)
        output_dir = tmp_path / "output"

        result = convert_nerve_mat_to_csv(mat_path, output_dir)

        assert result is not None and len(result) == 1
        session_folder = Path(result[0]["session_folder"])
        csv_files = sorted(session_folder.glob("*_block*_table.csv"))
        assert len(csv_files) == n_blocks, (
            f"Expected {n_blocks} block CSVs, found {len(csv_files)}"
        )

    def test_block_csv_has_expected_columns(self, tmp_path: Path) -> None:
        mat_path = _make_mat_file(tmp_path)
        output_dir = tmp_path / "output"

        result = convert_nerve_mat_to_csv(mat_path, output_dir)

        session_folder = Path(result[0]["session_folder"])
        csv_files = sorted(session_folder.glob("*_block*_table.csv"))
        import pandas as pd
        df = pd.read_csv(csv_files[0])
        assert "YYYYMMDD" in df.columns
        assert "Nervespike1" in df.columns
        assert len(df) == _N_ROWS

    def test_block_csv_naming_convention(self, tmp_path: Path) -> None:
        mat_path = _make_mat_file(tmp_path, n_blocks=2)
        output_dir = tmp_path / "output"
        mat_stem = mat_path.stem

        result = convert_nerve_mat_to_csv(mat_path, output_dir)

        session_folder = Path(result[0]["session_folder"])
        csv_files = sorted(session_folder.glob("*_block*_table.csv"))
        names = [f.name for f in csv_files]
        assert f"{mat_stem}_block1_table.csv" in names
        assert f"{mat_stem}_block2_table.csv" in names


class TestMetadataFile:

    def test_metadata_file_created(self, tmp_path: Path) -> None:
        mat_path = _make_mat_file(tmp_path)
        output_dir = tmp_path / "output"
        mat_stem = mat_path.stem

        result = convert_nerve_mat_to_csv(mat_path, output_dir)

        session_folder = Path(result[0]["session_folder"])
        meta_file = session_folder / f"{mat_stem}_metadata.txt"
        assert meta_file.exists(), f"Metadata file missing: {meta_file}"

    def test_metadata_file_contents(self, tmp_path: Path) -> None:
        mat_path = _make_mat_file(tmp_path, session_key="ST14-01", n_blocks=2)
        output_dir = tmp_path / "output"
        mat_stem = mat_path.stem

        result = convert_nerve_mat_to_csv(mat_path, output_dir)

        session_folder = Path(result[0]["session_folder"])
        meta = _read_metadata(session_folder, mat_stem)

        assert meta["Exp"] == "1"
        assert meta["UnitName"] == "ST14-01"
        assert meta["UnitNumber"] == "1"
        assert meta["IdxInDataInfo"] == "1"
        assert meta["UnitType"] == "SA"
        assert meta["Stimulus"] == "Semi_contr"
        assert meta["n_blocks"] == "2"
        assert meta["date_corrected"] == "True"

    def test_metadata_date_corrected_false_for_unknown_session(self, tmp_path: Path) -> None:
        mat_path = _make_mat_file(tmp_path, session_key="ST15-01")
        output_dir = tmp_path / "output"
        mat_stem = mat_path.stem

        result = convert_nerve_mat_to_csv(mat_path, output_dir)

        session_folder = Path(result[0]["session_folder"])
        meta = _read_metadata(session_folder, mat_stem)
        assert meta["date_corrected"] == "False"


class TestIdempotency:

    def test_idempotency_skip_returns_none(self, tmp_path: Path) -> None:
        mat_path = _make_mat_file(tmp_path)
        output_dir = tmp_path / "output"

        first = convert_nerve_mat_to_csv(mat_path, output_dir)
        assert first is not None

        second = convert_nerve_mat_to_csv(mat_path, output_dir, force_processing=False)
        assert second is None

    def test_idempotency_does_not_overwrite_outputs(self, tmp_path: Path) -> None:
        mat_path = _make_mat_file(tmp_path)
        output_dir = tmp_path / "output"

        result = convert_nerve_mat_to_csv(mat_path, output_dir)
        session_folder = Path(result[0]["session_folder"])
        csv_files_before = sorted(f.name for f in session_folder.glob("*.csv"))
        mtime_before = {f.name: f.stat().st_mtime for f in session_folder.glob("*.csv")}

        import time
        time.sleep(0.05)

        convert_nerve_mat_to_csv(mat_path, output_dir, force_processing=False)

        csv_files_after = sorted(f.name for f in session_folder.glob("*.csv"))
        assert csv_files_before == csv_files_after
        for fname, mtime in mtime_before.items():
            current_mtime = (session_folder / fname).stat().st_mtime
            assert current_mtime == mtime, (
                f"File {fname} was modified during idempotency skip"
            )

    def test_force_reprocessing_regenerates_outputs(self, tmp_path: Path) -> None:
        mat_path = _make_mat_file(tmp_path)
        output_dir = tmp_path / "output"

        first = convert_nerve_mat_to_csv(mat_path, output_dir)
        assert first is not None
        session_folder = Path(first[0]["session_folder"])
        csv_files_after_first = sorted(f.name for f in session_folder.glob("*.csv"))

        import time
        time.sleep(0.05)

        second = convert_nerve_mat_to_csv(mat_path, output_dir, force_processing=True)

        assert second is not None, "force=True should return results, not None"
        assert len(second) == 1
        new_session_folder = Path(second[0]["session_folder"])
        csv_files_after_force = sorted(f.name for f in new_session_folder.glob("*.csv"))
        assert csv_files_after_first == csv_files_after_force

    def test_force_reprocessing_clears_stale_outputs(self, tmp_path: Path) -> None:
        mat_path = _make_mat_file(tmp_path)
        output_dir = tmp_path / "output"

        first = convert_nerve_mat_to_csv(mat_path, output_dir)
        session_folder = Path(first[0]["session_folder"])

        # Inject a stale file that should be removed on force reprocess
        stale_file = session_folder / "stale_garbage.csv"
        stale_file.write_text("stale")
        assert stale_file.exists()

        import time
        time.sleep(0.05)

        convert_nerve_mat_to_csv(mat_path, output_dir, force_processing=True)

        # The whole output_dir was cleared and rebuilt — stale file must be gone
        assert not stale_file.exists(), (
            "Stale file should have been removed during force reprocessing"
        )


class TestReturnValue:

    def test_return_value_structure(self, tmp_path: Path) -> None:
        mat_path = _make_mat_file(tmp_path, n_blocks=2)
        output_dir = tmp_path / "output"

        result = convert_nerve_mat_to_csv(mat_path, output_dir)

        assert isinstance(result, list)
        assert len(result) == 1
        entry = result[0]
        assert set(entry.keys()) == {"mat_file", "session_folder", "n_blocks", "date_corrected"}
        assert entry["mat_file"] == mat_path.name
        assert entry["n_blocks"] == 2
        assert isinstance(entry["date_corrected"], bool)

    def test_n_blocks_matches_actual_output(self, tmp_path: Path) -> None:
        n_blocks = 4
        mat_path = _make_mat_file(tmp_path, n_blocks=n_blocks)
        output_dir = tmp_path / "output"

        result = convert_nerve_mat_to_csv(mat_path, output_dir)

        assert result[0]["n_blocks"] == n_blocks
        session_folder = Path(result[0]["session_folder"])
        csv_files = list(session_folder.glob("*_block*_table.csv"))
        assert len(csv_files) == n_blocks


# ---------------------------------------------------------------------------
# Helpers for Phase 2
# ---------------------------------------------------------------------------

_SESSION_ID_RENAME = "2022-06-16_ST15-01"
_MAT_STEM_RENAME = "ST15-01_zoom3_unit"


def _make_quality_check_xlsx(tmp_path: Path) -> Path:
    xlsx_path = tmp_path / "quality_check.xlsx"
    df_xlsx = pd.DataFrame(
        {
            "Zoom": [3, 3, 4],
            "Zoom Block ID": [1, 2, 1],
            "Block order": [3, 5, 7],
        }
    )
    df_xlsx.to_excel(xlsx_path, index=False)
    return xlsx_path


def _make_csv_session_dir(
    tmp_path: Path,
    session_id: str,
    mat_stem: str,
    zoom: int,
    block_ids: list[int],
) -> Path:
    session_dir = tmp_path / session_id
    session_dir.mkdir()
    for b in block_ids:
        csv = session_dir / f"{mat_stem}_block{b}_table.csv"
        csv.write_text("YYYYMMDD,Nervespike1\n20220616,1.0\n")
    meta = session_dir / f"{mat_stem}_metadata.txt"
    meta.write_text(f"Zoom: {zoom}\nUnitName: {session_id}\n")
    return session_dir


# ---------------------------------------------------------------------------
# Phase 2 Tests
# ---------------------------------------------------------------------------


class TestBlockOrderRename:

    def test_block_order_mapping(self, tmp_path: Path) -> None:
        csv_dir = tmp_path / "csv_files"
        csv_dir.mkdir()
        _make_csv_session_dir(
            csv_dir, _SESSION_ID_RENAME, _MAT_STEM_RENAME, zoom=3, block_ids=[2]
        )
        output_dir = tmp_path / "block_order"
        xlsx_path = _make_quality_check_xlsx(tmp_path)

        result = rename_nerve_to_block_order(
            csv_dir=csv_dir,
            output_dir=output_dir,
            quality_check_xlsx=xlsx_path,
            session_id=_SESSION_ID_RENAME,
        )

        assert result is not None
        assert len(result) == 1
        output_file = Path(result[0]["output_file"])
        assert output_file.exists()
        assert output_file.name == (
            f"{_SESSION_ID_RENAME}_semicontrolled_block-order05_nerve.csv"
        )

    def test_block_order_filename_format(self, tmp_path: Path) -> None:
        csv_dir = tmp_path / "csv_files"
        csv_dir.mkdir()
        _make_csv_session_dir(
            csv_dir, _SESSION_ID_RENAME, _MAT_STEM_RENAME, zoom=3, block_ids=[1]
        )
        output_dir = tmp_path / "block_order"
        xlsx_path = _make_quality_check_xlsx(tmp_path)

        result = rename_nerve_to_block_order(
            csv_dir=csv_dir,
            output_dir=output_dir,
            quality_check_xlsx=xlsx_path,
            session_id=_SESSION_ID_RENAME,
        )

        assert result is not None
        assert len(result) == 1
        output_name = Path(result[0]["output_file"]).name
        import re
        pattern = re.compile(
            rf"^{re.escape(_SESSION_ID_RENAME)}_semicontrolled_block-order(\d{{2}})_nerve\.csv$"
        )
        assert pattern.match(output_name), (
            f"Output filename '{output_name}' does not match expected pattern"
        )

    def test_missing_zoom_block_raises(self, tmp_path: Path) -> None:
        csv_dir = tmp_path / "csv_files"
        csv_dir.mkdir()
        _make_csv_session_dir(
            csv_dir, _SESSION_ID_RENAME, _MAT_STEM_RENAME, zoom=9, block_ids=[1]
        )
        output_dir = tmp_path / "block_order"
        xlsx_path = _make_quality_check_xlsx(tmp_path)

        with pytest.raises(ValueError, match="No block-order entry found"):
            rename_nerve_to_block_order(
                csv_dir=csv_dir,
                output_dir=output_dir,
                quality_check_xlsx=xlsx_path,
                session_id=_SESSION_ID_RENAME,
            )

    def test_idempotency_skip_rename(self, tmp_path: Path) -> None:
        csv_dir = tmp_path / "csv_files"
        csv_dir.mkdir()
        _make_csv_session_dir(
            csv_dir, _SESSION_ID_RENAME, _MAT_STEM_RENAME, zoom=3, block_ids=[2]
        )
        output_dir = tmp_path / "block_order"
        xlsx_path = _make_quality_check_xlsx(tmp_path)

        first = rename_nerve_to_block_order(
            csv_dir=csv_dir,
            output_dir=output_dir,
            quality_check_xlsx=xlsx_path,
            session_id=_SESSION_ID_RENAME,
        )
        assert first is not None

        second = rename_nerve_to_block_order(
            csv_dir=csv_dir,
            output_dir=output_dir,
            quality_check_xlsx=xlsx_path,
            session_id=_SESSION_ID_RENAME,
            force_processing=False,
        )
        assert second is None

    def test_force_reprocessing_rename(self, tmp_path: Path) -> None:
        csv_dir = tmp_path / "csv_files"
        csv_dir.mkdir()
        _make_csv_session_dir(
            csv_dir, _SESSION_ID_RENAME, _MAT_STEM_RENAME, zoom=3, block_ids=[2]
        )
        output_dir = tmp_path / "block_order"
        xlsx_path = _make_quality_check_xlsx(tmp_path)

        first = rename_nerve_to_block_order(
            csv_dir=csv_dir,
            output_dir=output_dir,
            quality_check_xlsx=xlsx_path,
            session_id=_SESSION_ID_RENAME,
        )
        assert first is not None

        output_session_dir = output_dir / _SESSION_ID_RENAME
        stale_file = output_session_dir / "stale_garbage.csv"
        stale_file.write_text("stale")

        second = rename_nerve_to_block_order(
            csv_dir=csv_dir,
            output_dir=output_dir,
            quality_check_xlsx=xlsx_path,
            session_id=_SESSION_ID_RENAME,
            force_processing=True,
        )

        assert second is not None, "force=True should return results, not None"
        assert len(second) == 1
        assert not stale_file.exists(), (
            "Stale file should have been removed during force reprocessing"
        )
