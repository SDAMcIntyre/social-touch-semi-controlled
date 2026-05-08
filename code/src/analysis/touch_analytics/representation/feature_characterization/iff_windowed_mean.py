# representation/feature_characterization/iff_windowed_mean.py
import numpy as np
import pandas as pd
from .base import FeatureExtractor
from .statistical import _EXCLUDE_FROM_AGGREGATION


class _IffWindowedBase(FeatureExtractor):
    """
    Shared base for IFF-windowed extractors.

    Validates that 'Nerve_freq' is present, auto-discovers numeric columns
    (using the same exclusion set as StatisticalExtractor), and builds a
    two-tier activity mask:
      1. Primary:  Nerve_freq > 0
      2. Fallback: contact_detected == 1  (only when primary mask is all-False)
      3. Fail-fast: if fallback column also absent, raise ValueError
    """

    def _get_activity_mask_and_cols(
        self, group: pd.DataFrame
    ) -> tuple[pd.Series, list[str]]:
        if 'Nerve_freq' not in group.columns:
            raise ValueError(
                "_IffWindowedBase requires a 'Nerve_freq' column in the group DataFrame, "
                "but it was not found. Ensure the session CSV includes neural frequency data."
            )

        numeric_cols = [
            col for col in group.columns
            if col not in _EXCLUDE_FROM_AGGREGATION
            and pd.api.types.is_numeric_dtype(group[col])
        ]

        activity_mask = group['Nerve_freq'] > 0

        if not activity_mask.any():
            if 'contact_detected' not in group.columns:
                raise ValueError(
                    "IFF activity mask is all-False (Nerve_freq is zero for every frame) "
                    "and no 'contact_detected' fallback column is present. "
                    "Cannot determine activity window."
                )
            activity_mask = group['contact_detected'] == 1

        return activity_mask, numeric_cols

    def extract(self, group: pd.DataFrame, config: dict) -> dict:
        raise NotImplementedError  # pragma: no cover


class MeanDuringIffExtractor(_IffWindowedBase):
    """
    Computes the mean of each numeric column over frames where IFF activity is
    detected (Nerve_freq > 0, or contact_detected == 1 as fallback).

    Output columns: <col>_mean_during_iff
    """

    def extract(self, group: pd.DataFrame, config: dict) -> dict:
        activity_mask, numeric_cols = self._get_activity_mask_and_cols(group)

        if not activity_mask.any():
            return {f'{col}_mean_during_iff': np.nan for col in numeric_cols}

        active_frames = group.loc[activity_mask]
        return {
            f'{col}_mean_during_iff': active_frames[col].mean()
            for col in numeric_cols
        }


class MeanBeforeIffExtractor(_IffWindowedBase):
    """
    Computes the mean of each numeric column over the window of frames
    immediately before the first active frame (Nerve_freq > 0, or
    contact_detected == 1 as fallback).

    The window size is controlled by config['pre_iff_window_ms'] (default 250).
    If activity starts at the first frame (index 0), all features are NaN.

    Output columns: <col>_mean_before_iff
    """

    def extract(self, group: pd.DataFrame, config: dict) -> dict:
        pre_iff_window_ms = config.get('pre_iff_window_ms', 250)

        activity_mask, numeric_cols = self._get_activity_mask_and_cols(group)

        if not activity_mask.any():
            return {f'{col}_mean_before_iff': np.nan for col in numeric_cols}

        # iloc-based positional index of the first active frame
        first_active_iloc = activity_mask.values.argmax()

        if first_active_iloc == 0:
            return {f'{col}_mean_before_iff': np.nan for col in numeric_cols}

        window_start = max(0, first_active_iloc - pre_iff_window_ms)
        before_slice = group.iloc[window_start:first_active_iloc]

        return {
            f'{col}_mean_before_iff': before_slice[col].mean()
            for col in numeric_cols
        }
