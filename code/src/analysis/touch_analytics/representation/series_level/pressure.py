# representation/series_level/pressure.py
from typing import List

import pandas as pd

PRESSURE_INPUT_COLUMNS: List[str] = ['contact_depth', 'contact_area']


def compute_geo_pressure(group: pd.DataFrame) -> pd.Series:
    """Geometric pressure per frame: contact_depth / contact_area (NaN where area <= 0)."""
    area = group['contact_area']
    depth = group['contact_depth']
    result = depth.where(area > 0, other=float('nan')) / area.where(area > 0, other=float('nan'))
    return result


def get_geo_pressure(group: pd.DataFrame) -> pd.Series:
    """Return pre-computed geo_pressure column if present, else compute from raw depth/area."""
    if 'geo_pressure' in group.columns:
        return group['geo_pressure']
    return compute_geo_pressure(group)
