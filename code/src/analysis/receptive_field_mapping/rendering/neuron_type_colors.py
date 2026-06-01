# Edit NEURON_TYPE_COLORS / NEURON_TYPE_ORDER to change the color scheme for all IFF plots.

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# USER-EDITABLE BLOCK — change colors and order here; all IFF plots update.
# ---------------------------------------------------------------------------

NEURON_TYPE_COLORS: dict[str, str] = {
    "SAI":   "#e06c75",  # red
    "SAII":  "#61afef",  # blue
    "Field": "#98c379",  # green
    "HFA":   "#e5c07b",  # yellow/amber
    "CT":    "#c678dd",  # purple
}

NEURON_TYPE_ORDER: list[str] = ["SAI", "SAII", "Field", "HFA", "CT"]

# ---------------------------------------------------------------------------
# Private column-name targets for case-insensitive xlsx lookup
# ---------------------------------------------------------------------------

_NEURON_TYPE_COL = "neuron type"
_UNIT_NAME_COL = "unit name"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SessionColorScheme:
    session_color: dict[str, str]        # session_id -> hex color
    session_neuron_type: dict[str, str]  # session_id -> neuron type string
    type_color: dict[str, str]           # neuron_type -> hex color (only types present, in NEURON_TYPE_ORDER)


# ---------------------------------------------------------------------------
# Xlsx parser
# ---------------------------------------------------------------------------

def parse_neuron_summary_xlsx(xlsx_path: Path) -> dict[str, str]:
    """Parse MNG-DataSummary.xlsx and return {unit_name: neuron_type}.

    Column lookup is case-insensitive and strips surrounding whitespace.
    Raises ValueError if either required column is absent.
    """
    df = pd.read_excel(xlsx_path)

    # Case-insensitive, stripped column lookup
    col_map: dict[str, str] = {str(c).strip().lower(): str(c) for c in df.columns}

    if _UNIT_NAME_COL not in col_map:
        raise ValueError(
            f"Required column '{_UNIT_NAME_COL}' not found in {xlsx_path.name}. "
            f"Found columns: {list(df.columns)}"
        )
    if _NEURON_TYPE_COL not in col_map:
        raise ValueError(
            f"Required column '{_NEURON_TYPE_COL}' not found in {xlsx_path.name}. "
            f"Found columns: {list(df.columns)}"
        )

    unit_col = col_map[_UNIT_NAME_COL]
    type_col = col_map[_NEURON_TYPE_COL]

    df = df.dropna(subset=[unit_col, type_col])

    unit_to_type: dict[str, str] = {}
    for _, row in df.iterrows():
        unit_name = str(row[unit_col]).strip()
        neuron_type = str(row[type_col]).strip()
        unit_to_type[unit_name] = neuron_type

    return unit_to_type


# ---------------------------------------------------------------------------
# Session-id utilities
# ---------------------------------------------------------------------------

def unit_name_from_session_id(session_id: str) -> str:
    """Extract the unit name (e.g. 'ST13-01') from a session_id like '2022-06-14_ST13-01'.

    Raises ValueError if the expected pattern is not found.
    """
    match = re.search(r"(ST\d+-\d+)", session_id)
    if match is None:
        raise ValueError(
            f"Cannot extract unit name from session_id '{session_id}': "
            f"expected a token matching 'ST<digits>-<digits>'."
        )
    return match.group(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_session_color_scheme(
    session_ids: list[str],
    xlsx_path: Path,
) -> SessionColorScheme:
    """Map each session_id to its neuron-type color via MNG-DataSummary.xlsx.

    Fail-fast: raises ValueError for any unknown unit or unrecognised neuron type.
    """
    unit_to_type = parse_neuron_summary_xlsx(xlsx_path)

    session_color: dict[str, str] = {}
    session_neuron_type: dict[str, str] = {}

    for session_id in session_ids:
        unit_name = unit_name_from_session_id(session_id)

        if unit_name not in unit_to_type:
            raise ValueError(
                f"Unit '{unit_name}' (from session '{session_id}') "
                f"was not found in {xlsx_path.name}."
            )

        neuron_type = unit_to_type[unit_name]

        if neuron_type not in NEURON_TYPE_COLORS:
            raise ValueError(
                f"Neuron type '{neuron_type}' (session '{session_id}', unit '{unit_name}') "
                f"is not in NEURON_TYPE_COLORS. "
                f"Known types: {list(NEURON_TYPE_COLORS.keys())}."
            )

        session_color[session_id] = NEURON_TYPE_COLORS[neuron_type]
        session_neuron_type[session_id] = neuron_type

    # Build type_color: only types present in this session set, in NEURON_TYPE_ORDER
    present_types = {nt for nt in session_neuron_type.values()}
    type_color: dict[str, str] = {
        nt: NEURON_TYPE_COLORS[nt]
        for nt in NEURON_TYPE_ORDER
        if nt in present_types
    }

    return SessionColorScheme(
        session_color=session_color,
        session_neuron_type=session_neuron_type,
        type_color=type_color,
    )
