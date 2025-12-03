import logging
from pathlib import Path
from typing import Optional, Literal, Tuple, List, Set

import pandas as pd

# --- External Dependencies (Preserved) ---
# Assuming these exist in your project structure
try:
    from utils.should_process_task import should_process_task
    from preprocessing.led_analysis import LEDBlinkingFilesHandler
except ImportError:
    # Mocking for standalone analysis if files are missing in the generic context
    logging.warning("External modules not found. Mocking for architectural demonstration.")
    
    def should_process_task(input_paths, output_paths, force):
        return True

    class LEDBlinkingFilesHandler:
        def load_timeseries_from_csv(self, path, output_format):
            return pd.read_csv(path)

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


class TimeSeriesMerger:
    """
    Encapsulates logic for merging time-series DataFrames using various strategies
    (Positional, Time-Tolerant, Exact).
    """

    def __init__(
        self,
        key_column: str = 'time',
        decimal_precision: Optional[int] = 2,
        primary_time_source: Literal['df1', 'df2'] = 'df1'
    ):
        self.key_column = key_column
        self.decimal_precision = decimal_precision
        self.primary_time_source = primary_time_source

    def merge(
        self, 
        df1: pd.DataFrame, 
        df2: pd.DataFrame, 
        enforce_row_alignment: bool = True
    ) -> Tuple[pd.DataFrame, bool]:
        """
        Main entry point. Determines the optimal merge strategy based on input data state.
        """
        logger.info(f"Starting merge process on key: '{self.key_column}'")

        # 1. Validation
        if self.key_column not in df1.columns or self.key_column not in df2.columns:
            logger.error(f"Key column '{self.key_column}' not found in both DataFrames.")
            return pd.DataFrame(), False

        # 2. Strategy: Positional Alignment (Preferred)
        if enforce_row_alignment and len(df1) == len(df2):
            logger.info(f"Row counts match ({len(df1)}). Executing Positional Merge.")
            return self._merge_positional(df1, df2)

        # 3. Strategy: Fuzzy/Rounded Time Match (Fallback 1)
        if self.decimal_precision is not None:
            logger.info(f"Row mismatch. Executing Fuzzy Merge (Precision: {self.decimal_precision}).")
            return self._merge_time_tolerant(df1, df2)

        # 4. Strategy: Exact Match (Fallback 2)
        logger.info("Row mismatch and no precision set. Executing Exact Merge.")
        return self._merge_exact(df1, df2)

    def _merge_positional(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
        """Merges based on index/position, ignoring time mismatches."""
        # Reset index to ensure strict 0..N alignment without modifying originals
        df1_clean = df1.reset_index(drop=True)
        df2_clean = df2.reset_index(drop=True)

        merged_df = pd.merge(
            df1_clean, 
            df2_clean, 
            left_index=True, 
            right_index=True, 
            how='outer', 
            suffixes=('_df1', '_df2')
        )

        merged_df = self._consolidate_time_columns(merged_df)
        return merged_df, True

    def _merge_time_tolerant(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
        """Merges based on a rounded time key."""
        temp_key = f"_{self.key_column}_rounded"
        
        # Work on copies to avoid side effects
        df1_temp = df1.copy()
        df2_temp = df2.copy()

        try:
            df1_temp[temp_key] = df1_temp[self.key_column].round(self.decimal_precision)
            df2_temp[temp_key] = df2_temp[self.key_column].round(self.decimal_precision)
        except TypeError:
            logger.error(f"Column '{self.key_column}' is not numeric; cannot apply precision.")
            return pd.DataFrame(), False

        # Intersection check
        keys1 = set(df1_temp[temp_key].dropna())
        keys2 = set(df2_temp[temp_key].dropna())
        common = keys1.intersection(keys2)

        if not common:
            logger.error(f"No common time values found at precision {self.decimal_precision}.")
            return pd.DataFrame(), False

        logger.info(f"Found {len(common)} common rounded time values.")

        merged_df = pd.merge(
            df1_temp, 
            df2_temp, 
            on=temp_key, 
            how='outer', 
            suffixes=('_df1', '_df2')
        )
        
        merged_df.drop(columns=[temp_key], inplace=True)
        
        # Consolidate logic for fuzzy merge is slightly different (combine_first)
        time_col_df1 = f"{self.key_column}_df1"
        time_col_df2 = f"{self.key_column}_df2"
        
        if time_col_df1 in merged_df and time_col_df2 in merged_df:
            merged_df[self.key_column] = merged_df[time_col_df1].combine_first(merged_df[time_col_df2])
            merged_df.drop(columns=[time_col_df1, time_col_df2], inplace=True)
            
        return self._finalize_dataframe(merged_df), True

    def _merge_exact(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
        """Merges strictly on the key column."""
        keys1 = set(df1[self.key_column].dropna())
        keys2 = set(df2[self.key_column].dropna())
        
        if not keys1.intersection(keys2):
            logger.error("No common exact time values found.")
            return pd.DataFrame(), False

        merged_df = pd.merge(
            df1, 
            df2, 
            on=self.key_column, 
            how='outer', 
            suffixes=('_df1', '_df2')
        )
        return self._finalize_dataframe(merged_df), True

    def _consolidate_time_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper to clean up suffixed time columns after a positional merge."""
        time_col_df1 = f"{self.key_column}_df1"
        time_col_df2 = f"{self.key_column}_df2"

        source_col = time_col_df1 if self.primary_time_source == 'df1' else time_col_df2
        
        if source_col in df.columns:
            df[self.key_column] = df[source_col]

        cols_to_drop = [c for c in [time_col_df1, time_col_df2] if c in df.columns]
        df.drop(columns=cols_to_drop, inplace=True)
        
        return self._finalize_dataframe(df)

    def _finalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensures the key column is first."""
        if self.key_column in df.columns:
            cols = [self.key_column] + [c for c in df.columns if c != self.key_column]
            return df[cols]
        return df


class ContactTTLProcessor:
    """
    Service layer handling the specific business logic of joining 
    Contact Characteristics with TTL LED data.
    """
    
    def __init__(self):
        self.merger = TimeSeriesMerger(
            key_column='time',
            decimal_precision=2,
            primary_time_source='df1'
        )

    def run(self, contact_chars_path: Path, ttl_path: Path, output_path: Path, force: bool = False) -> bool:
        """
        Orchestrates the loading, cleaning, merging, and saving process.
        """
        if not should_process_task([contact_chars_path, ttl_path], [output_path], force=force):
            logger.info(f"Skipping task: Output '{output_path}' exists.")
            return True

        try:
            # 1. Load Data
            contact_df = pd.read_csv(contact_chars_path)
            led_df = self._load_led_data(ttl_path)

            # 2. Merge
            merged_df, success = self.merger.merge(
                contact_df, 
                led_df, 
                enforce_row_alignment=True
            )

            if not success:
                logger.error(f"Merge failed for inputs: '{contact_chars_path}', '{ttl_path}'")
                return False

            # 3. Post-Processing (Column Reordering)
            merged_df = self._reorder_columns(merged_df)

            # 4. Save
            self._save_output(merged_df, output_path)
            return True

        except FileNotFoundError as e:
            logger.error(f"File missing: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return False

    def _load_led_data(self, path: Path) -> pd.DataFrame:
        """Encapsulates LED specific loading and initial cleaning."""
        handler = LEDBlinkingFilesHandler()
        df = handler.load_timeseries_from_csv(path, output_format='dataframe')
        
        if 'green_levels' in df.columns:
            df.drop(columns='green_levels', inplace=True)
        return df

    def _reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handles specific column presentation logic."""
        cols = df.columns.tolist()
        if 'led_on' in cols and 'frame_index' in cols:
            cols.remove('led_on')
            target_idx = cols.index('frame_index') + 1
            cols.insert(target_idx, 'led_on')
            return df[cols]
        return df

    def _save_output(self, df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Successfully saved merged data to '{path}'.")


# --- Public API Wrapper (to maintain backward compatibility if needed) ---

def unify_contact_caracteristics_and_ttl(
    contact_chars_path: Path,
    ttl_path: Path,
    output_path: Path,
    *,
    force_processing: bool = False
) -> bool:
    processor = ContactTTLProcessor()
    success = processor.run(ttl_path, contact_chars_path, output_path, force=force_processing)
    return success