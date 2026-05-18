import shutil
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

_SRC = Path(__file__).resolve().parents[4] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.should_process_task import should_process_task

DATE_CORRECTIONS = {
    "ST14-01": timedelta(days=-1),
    "ST14-02": timedelta(days=-1),
    "ST16-01": timedelta(days=-1),
    "ST16-02": timedelta(days=+2),
}


def _apply_date_correction(yyyymmdd_int: int, correction: timedelta) -> int:
    d = date(yyyymmdd_int // 10000, (yyyymmdd_int // 100) % 100, yyyymmdd_int % 100)
    d_corrected = d + correction
    return d_corrected.year * 10000 + d_corrected.month * 100 + d_corrected.day


def _unwrap(arr):
    """Recursively index into (1,1) numpy arrays until reaching a non-(1,1) value."""
    while isinstance(arr, np.ndarray) and arr.ndim >= 2 and arr.shape[:2] == (1, 1):
        arr = arr[0, 0]
    return arr


def _extract_scalar(field_value):
    """Extract a Python scalar from scipy.io.loadmat nested array wrapping."""
    val = _unwrap(field_value)
    while isinstance(val, np.ndarray) and val.size == 1:
        val = val.flat[0]
    if isinstance(val, (np.bytes_, bytes)):
        return val.decode().strip()
    if isinstance(val, (np.str_, str)):
        return str(val).strip()
    if isinstance(val, np.generic):
        return val.item()
    return val


def _column_to_series(col_values) -> pd.Series:
    """Flatten a D-table column, which may be a (1, n_rows) array of (1,1) sub-arrays."""
    flat = col_values.flatten()
    scalars = []
    for v in flat:
        if isinstance(v, np.ndarray):
            scalars.append(v.flat[0])
        else:
            scalars.append(v)
    return pd.Series(scalars)


def _block_to_dataframe(D) -> pd.DataFrame:
    """Convert a scipy.io.loadmat structured array (MATLAB table) to a DataFrame.

    scipy may represent MATLAB tables as structured arrays where each field
    is itself a (1, n_rows) array of (1,1) element arrays, depending on the
    MATLAB file's internal representation.
    """
    if not hasattr(D, "dtype") or D.dtype.names is None:
        raise ValueError(
            f"Expected a numpy structured array for block table D, got {type(D)} "
            f"with dtype {getattr(D, 'dtype', 'unknown')}"
        )
    return pd.DataFrame(
        {name: _column_to_series(D[name]) for name in D.dtype.names}
    )


def convert_nerve_mat_to_csv(
    mat_path: Path,
    output_dir: Path,
    force_processing: bool = False,
) -> list[dict] | None:
    if not mat_path.exists():
        raise FileNotFoundError(f"Input .mat file does not exist: {mat_path}")

    if not force_processing and output_dir.exists() and any(output_dir.iterdir()):
        print("✅ Task outputs are up-to-date.")
        return None

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mat = scipy.io.loadmat(str(mat_path), squeeze_me=False)

    if "S" not in mat:
        raise ValueError(f"Expected top-level key 'S' in .mat file: {mat_path.name}")

    S = _unwrap(mat["S"])

    if not hasattr(S, "dtype") or S.dtype.names is None:
        raise ValueError(
            f"Expected a numpy structured array for S, got {type(S)}"
        )

    Exp = int(_extract_scalar(S["Exp"]))
    UnitName = str(_extract_scalar(S["UnitName"]))
    UnitNumber = int(_extract_scalar(S["UnitNumber"]))
    IdxInDataInfo = int(_extract_scalar(S["IdxInDataInfo"]))
    UnitType = str(_extract_scalar(S["UnitType"]))
    Stimulus = str(_extract_scalar(S["Stimulus"]))
    Zoom = _extract_scalar(S["Zoom"]) if "Zoom" in S.dtype.names else None

    if Stimulus != "Semi_contr":
        return []

    session_key = UnitName.split("_")[0]
    date_correction = DATE_CORRECTIONS.get(session_key)
    date_corrected = date_correction is not None

    FPD = _unwrap(S["FullPeriod_D"])

    if not hasattr(FPD, "dtype") or FPD.dtype.names is None:
        raise ValueError(
            f"Expected a numpy structured array for FullPeriod_D, got {type(FPD)}"
        )

    ContD_raw = FPD["ContD"]
    # ContD may be a (1,1) wrapper around the actual (1, n_blocks) object array
    ContD = _unwrap(ContD_raw)
    if isinstance(ContD, np.ndarray) and ContD.dtype == object and ContD.ndim == 2:
        pass  # Already a (1, n_blocks) object array — correct
    elif isinstance(ContD, np.ndarray) and ContD.dtype.names is not None:
        pass  # Already a structured array of blocks
    else:
        raise ValueError(
            f"Unexpected ContD type after unwrapping: {type(ContD)}, dtype={getattr(ContD, 'dtype', 'N/A')}"
        )

    if ContD.ndim != 2 or ContD.shape[0] != 1:
        raise ValueError(
            f"Expected ContD to be a 1×N array, got shape {ContD.shape}"
        )

    n_blocks = ContD.shape[1]

    def _extract_block_df(b: int) -> pd.DataFrame:
        block_entry = ContD[0, b]
        # block_entry may be a structured array with field 'D', or just the D table directly
        block = _unwrap(block_entry) if isinstance(block_entry, np.ndarray) else block_entry
        if hasattr(block, "dtype") and block.dtype.names is not None and "D" in block.dtype.names:
            D_wrapper = block["D"]
            D = _unwrap(D_wrapper)
        elif hasattr(block, "dtype") and block.dtype.names is not None:
            D = block
        else:
            raise ValueError(
                f"Cannot extract block table D from block {b}: unexpected structure {type(block)}"
            )
        return _block_to_dataframe(D)

    first_df = _extract_block_df(0)

    if "YYYYMMDD" not in first_df.columns:
        raise ValueError(
            f"Expected 'YYYYMMDD' column in block table D, got columns: {list(first_df.columns)}"
        )

    first_date_int = int(first_df["YYYYMMDD"].iloc[0])
    if date_corrected:
        first_date_int = _apply_date_correction(first_date_int, date_correction)

    d = date(first_date_int // 10000, (first_date_int // 100) % 100, first_date_int % 100)
    date_str = d.strftime("%Y-%m-%d")
    session_folder = output_dir / f"{date_str}_{UnitName}"
    session_folder.mkdir(parents=True, exist_ok=True)

    mat_stem = mat_path.stem

    for b in range(n_blocks):
        df = _extract_block_df(b)

        if date_corrected and "YYYYMMDD" in df.columns:
            df["YYYYMMDD"] = df["YYYYMMDD"].apply(
                lambda v: _apply_date_correction(int(v), date_correction)
            )

        csv_path = session_folder / f"{mat_stem}_block{b + 1}_table.csv"
        df.to_csv(csv_path, index=False)

    metadata_path = session_folder / f"{mat_stem}_metadata.txt"
    metadata_lines = [
        f"Exp: {Exp}",
        f"UnitName: {UnitName}",
        f"UnitNumber: {UnitNumber}",
        f"IdxInDataInfo: {IdxInDataInfo}",
        f"UnitType: {UnitType}",
        f"Stimulus: {Stimulus}",
        f"Zoom: {Zoom}",
        f"n_blocks: {n_blocks}",
        f"date_corrected: {date_corrected}",
    ]
    metadata_path.write_text("\n".join(metadata_lines) + "\n")

    return [
        {
            "mat_file": mat_path.name,
            "session_folder": str(session_folder),
            "n_blocks": n_blocks,
            "date_corrected": date_corrected,
        }
    ]
