import re
import shutil
import sys
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parents[4] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_EXPECTED_XLSX_COLUMNS = {"Zoom", "Zoom Block ID", "Block order"}

_BLOCK_RE = re.compile(r"_block(\d+)_table\.csv$")


def _parse_zoom_from_metadata(metadata_path: Path) -> int:
    for line in metadata_path.read_text().splitlines():
        if line.startswith("Zoom:"):
            value = line.split(":", 1)[1].strip()
            return int(value)
    raise ValueError(
        f"No 'Zoom:' line found in metadata file: {metadata_path}"
    )


def rename_nerve_to_block_order(
    csv_dir: Path,
    output_dir: Path,
    quality_check_xlsx: Path,
    session_id: str,
    force_processing: bool = False,
) -> list[dict] | None:
    input_session_dir = csv_dir / session_id
    output_session_dir = output_dir / session_id

    if not force_processing and output_session_dir.exists() and any(output_session_dir.iterdir()):
        print("✅ Task outputs are up-to-date.")
        return None

    if output_session_dir.exists():
        shutil.rmtree(output_session_dir)

    if not input_session_dir.exists():
        raise FileNotFoundError(
            f"Input session directory does not exist: {input_session_dir}"
        )

    output_session_dir.mkdir(parents=True, exist_ok=True)

    df_quality_control = pd.read_excel(quality_check_xlsx)

    missing_cols = _EXPECTED_XLSX_COLUMNS - set(df_quality_control.columns)
    if missing_cols:
        raise ValueError(
            f"Quality-check xlsx is missing expected columns: {sorted(missing_cols)}. "
            f"Found columns: {list(df_quality_control.columns)}"
        )

    lookup: dict[tuple[int, int], int] = {}
    for _, row in df_quality_control.iterrows():
        key = (int(row["Zoom"]), int(row["Zoom Block ID"]))
        lookup[key] = int(row["Block order"])

    results = []
    csv_files = sorted(input_session_dir.glob("*_table.csv"))

    for csv_path in csv_files:
        match = _BLOCK_RE.search(csv_path.name)
        if match is None:
            continue

        block_id = int(match.group(1))
        mat_stem = csv_path.name[: -len(f"_block{block_id}_table.csv")]

        metadata_path = input_session_dir / f"{mat_stem}_metadata.txt"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found for '{csv_path.name}': {metadata_path}"
            )

        zoom_id = _parse_zoom_from_metadata(metadata_path)

        key = (zoom_id, block_id)
        if key not in lookup:
            raise ValueError(
                f"No block-order entry found for (Zoom={zoom_id}, Zoom Block ID={block_id}) "
                f"in '{quality_check_xlsx.name}'. "
                f"Processing file: {csv_path.name}"
            )

        block_order = lookup[key]
        output_filename = (
            f"{session_id}_semicontrolled_block-order{block_order:02d}_nerve.csv"
        )
        output_path = output_session_dir / output_filename
        shutil.copy2(csv_path, output_path)

        results.append(
            {
                "input_file": str(csv_path),
                "output_file": str(output_path),
                "block_order": block_order,
            }
        )

    return results
