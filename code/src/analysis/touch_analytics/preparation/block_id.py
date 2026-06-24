# preparation/block_id.py
"""
Block-ID synthesis for semi-controlled touch sessions.

Extracted from extraction_pipeline._extract_session (lines 183–190).
"""

from pathlib import Path
from typing import Union

import pandas as pd


def extract_block_id(df: pd.DataFrame) -> pd.Series:
    """
    Derive the ``block_order_id`` column when it is absent from *df*.

    If ``block_order_id`` is already present it is returned as-is.
    If ``source_block_file`` is present, extracts the integer from the
    ``_block-order-<N>_`` fragment of the filename.
    Otherwise returns a Series of ``None``.

    Parameters
    ----------
    df : pd.DataFrame
        Session touch DataFrame, as loaded from a session CSV.

    Returns
    -------
    pd.Series
        A Series aligned to *df* containing block-order IDs (strings or None).
    """
    if 'block_order_id' in df.columns:
        return df['block_order_id']
    if 'source_block_file' in df.columns:
        return (
            df['source_block_file'].astype(str)
            .str.extract(r'_block-order-(\d+)_', expand=False)
        )
    return pd.Series([None] * len(df), index=df.index)


def ensure_block_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of *df* with ``block_order_id`` guaranteed to exist.

    If the column is already present, the DataFrame is returned unchanged.
    Otherwise the column is synthesised via :func:`extract_block_id`.

    Parameters
    ----------
    df : pd.DataFrame
        Session touch DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``block_order_id`` column present.
    """
    if 'block_order_id' in df.columns:
        return df
    df = df.copy()
    df['block_order_id'] = extract_block_id(df)
    return df
