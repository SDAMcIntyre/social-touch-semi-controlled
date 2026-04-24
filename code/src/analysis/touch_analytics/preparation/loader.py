# preparation/loader.py
"""
Session CSV loader for semi-controlled touch data.

Extracted from extraction_pipeline._extract_session (line 175).

Centralises CSV loading so dtype checks and NaN policy are applied
consistently regardless of which pipeline stage reads a session file.
"""

import logging
from pathlib import Path

import pandas as pd


def load_session_csv(path: Path) -> pd.DataFrame:
    """
    Load a session touch CSV and return a DataFrame.

    The file is read with pandas defaults (dtype inference, keep NaN).
    Raises on file-not-found or parse errors rather than silently returning
    an empty DataFrame; callers that need to tolerate missing sessions should
    catch the exceptions explicitly.

    Parameters
    ----------
    path : Path
        Absolute path to the session CSV.

    Returns
    -------
    pd.DataFrame
        Session touch DataFrame as stored on disk; no column filtering or
        dtype coercion is applied here.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    Exception
        Any pandas CSV parse error is propagated unchanged.
    """
    logging.debug(f"load_session_csv: reading {path}")
    return pd.read_csv(path)
