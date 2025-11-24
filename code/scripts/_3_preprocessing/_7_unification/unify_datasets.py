import logging
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd

from utils.should_process_task import should_process_task

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


class UnifiedDatasetProcessor:
    """
    Orchestrates the loading, validation, and horizontal merging of multiple
    pre-processed CSV datasets.
    
    Assumption: All input CSVs possess an identical number of rows and are 
    synchronously aligned (frame-by-frame).
    """

    def run(
        self, 
        inputs: Dict[str, Path], 
        output_path: Path, 
        force: bool = False
    ) -> bool:
        """
        Main execution flow.
        """
        input_paths_list = list(inputs.values())
        
        # Check cache/existence logic
        if not should_process_task(input_paths_list, [output_path], force=force):
            logger.info(f"Skipping task: Output '{output_path}' exists.")
            return True

        try:
            logger.info("Loading datasets for unification...")
            dataframes = self._load_dataframes(inputs)

            logger.info("Validating row alignment...")
            self._validate_row_counts(dataframes)

            logger.info("Merging datasets...")
            merged_df = self._merge_horizontal(dataframes)

            logger.info("Saving unified dataset...")
            self._save_output(merged_df, output_path)
            
            return True

        except FileNotFoundError as e:
            logger.error(f"File missing: {e}")
            return False
        except ValueError as e:
            logger.error(f"Data Validation Error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return False

    def _load_dataframes(self, inputs: Dict[str, Path]) -> Dict[str, pd.DataFrame]:
        """
        Loads all CSVs. 
        """
        dfs = {}
        for key, path in inputs.items():
            if not path.exists():
                raise FileNotFoundError(f"Input for '{key}' not found at: {path}")
            
            # Read CSV. Assuming standard format. 
            # We do not set an index_col to ensure we preserve all data row-by-row 
            # unless a specific column is guaranteed to be the index.
            df = pd.read_csv(path)
            dfs[key] = df
            logger.debug(f"Loaded '{key}': {len(df)} rows, {len(df.columns)} columns.")
        return dfs

    def _validate_row_counts(self, dataframes: Dict[str, pd.DataFrame]) -> None:
        """
        Ensures all dataframes have the exact same number of rows.
        Raises ValueError if mismatch found.
        """
        if not dataframes:
            raise ValueError("No dataframes provided to merge.")

        # Get length of the first dataframe
        first_key = next(iter(dataframes))
        target_len = len(dataframes[first_key])

        mismatches = []
        for key, df in dataframes.items():
            if len(df) != target_len:
                mismatches.append(f"{key}: {len(df)}")

        if mismatches:
            error_msg = (
                f"Row count mismatch detected. Expected {target_len} rows (based on {first_key}). "
                f"Found mismatches: {', '.join(mismatches)}."
            )
            raise ValueError(error_msg)

    def _merge_horizontal(self, dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Concatenates dataframes horizontally (axis=1).
        """
        # We convert the dict values to a list for concatenation
        # Note: We might want to handle duplicate column names here.
        # If 'time' or 'frame' exists in multiple, pandas will suffix them (time, time.1).
        # Given the requirement for simplicity, we accept suffixes, or we could drop duplicates.
        
        # Strategy: Concatenate all. 
        # Logic: To ensure determinism, we sort keys or rely on insertion order (Python 3.7+).
        # We prioritize 'contact_data' or 'led_data' as the "leftmost" if order matters, 
        # but pd.concat will just append.
        
        df_list = list(dataframes.values())
        
        # Reset indices to ensure strict row alignment (ignoring any existing index metadata)
        df_list_reset = [df.reset_index(drop=True) for df in df_list]
        
        merged_df = pd.concat(df_list_reset, axis=1)
        
        # Optional: Cleanup duplicate columns if they are identical.
        # For now, we leave them to ensure no data loss, as requested by "strict positional".
        
        return merged_df

    def _save_output(self, df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Successfully saved unified data to '{path}'. Dimensions: {df.shape}")


# --- Public API Wrapper ---

def unify_datasets(
    led_data: Path,
    contact_data: Path,
    trial_data: Path,
    touch_data: Path,
    stimuli_data: Path,
    output_path: Path,
    *,
    force_processing: bool = False
) -> bool:
    """
    Wrapper matching the signature required by the 'unify_processed_data' task 
    in the preprocessing workflow.
    """
    processor = UnifiedDatasetProcessor()
    
    # Map inputs to a dictionary for cleaner processing
    inputs = {
        'contact_data': contact_data, # Putting contact first as it usually contains primary kinematics
        'led_data': led_data,
        'trial_data': trial_data,
        'touch_data': touch_data,
        'stimuli_data': stimuli_data
    }
    
    success = processor.run(inputs, output_path, force=force_processing)
    return success