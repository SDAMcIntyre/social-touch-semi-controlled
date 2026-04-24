# preparation/grouping.py
"""
Touch grouping for semi-controlled session DataFrames.

Extracted from extraction_pipeline._extract_all_touches (line 283).
"""

from typing import List, Tuple

import pandas as pd


def group_touches(df: pd.DataFrame) -> List[Tuple[Tuple, pd.DataFrame]]:
    """
    Group a session DataFrame into per-touch sub-DataFrames.

    Groups by ``(block_order_id, trial_id, single_touch_id)`` — the canonical
    three-level key used throughout the extraction pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Session touch DataFrame with ``block_order_id``, ``trial_id``, and
        ``single_touch_id`` columns present.

    Returns
    -------
    list of ((block_order_id, trial_id, single_touch_id), group_df) tuples
        Materialised list (not a lazy iterator) so callers can index and
        report progress without consuming a generator.
    """
    return list(df.groupby(['block_order_id', 'trial_id', 'single_touch_id']))
