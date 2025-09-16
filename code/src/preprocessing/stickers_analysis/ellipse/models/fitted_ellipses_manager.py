# Standard library imports
from __future__ import annotations
from functools import reduce
from typing import Dict

# Third-party imports
import pandas as pd

class FittedEllipsesManager:
    """
    Manages the collection and aggregation of ellipse fitting results in memory.
    This class is a pure data structure and has no knowledge of file I/O.
    """
    def __init__(self):
        """Initializes the manager with an empty dictionary to store results."""
        self._results: Dict[str, pd.DataFrame] = {}

    def add_result(self, object_name: str, result_df: pd.DataFrame) -> None:
        """
        Adds the ellipse fitting result for a specific object.

        Args:
            object_name (str): The name of the object (e.g., 'sticker_1').
            result_df (pd.DataFrame): The DataFrame containing results for this object.
        """
        if 'frame_number' not in result_df.columns:
            raise ValueError("Input DataFrame must contain a 'frame_number' column.")
        self._results[object_name] = result_df.copy()
    
    def get_all_results(self) -> Dict[str, pd.DataFrame]:
        """Returns the dictionary of all stored results."""
        return self._results
    
    def get_combined_dataframe(self) -> pd.DataFrame:
        """
        Combines all stored results into a single, wide-format DataFrame by
        **merging on the 'frame_number' column**. This guarantees a single,
        shared 'frame_number' column in the final output.

        Returns:
            pd.DataFrame: The aggregated DataFrame.
        """
        if not self._results:
            return pd.DataFrame()

        # 1. Prepare a list of DataFrames with data columns prefixed by object name.
        #    The 'frame_number' column is left untouched for merging.
        prefixed_dfs = []
        for name, df in self._results.items():
            cols_to_prefix = df.columns.drop('frame_number')
            rename_map = {col: f"{name}_{col}" for col in cols_to_prefix}
            prefixed_df = df.rename(columns=rename_map)
            prefixed_dfs.append(prefixed_df)
        
        if len(prefixed_dfs) == 1:
            return prefixed_dfs[0]

        # 2. Use functools.reduce to iteratively merge all DataFrames on the
        #    shared 'frame_number' column. An 'outer' join ensures no data is
        #    lost if frame counts differ between objects.
        merged_df = reduce(
            lambda left, right: pd.merge(left, right, on='frame_number', how='outer'),
            prefixed_dfs
        )

        return merged_df
