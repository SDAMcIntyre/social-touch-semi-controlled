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


def _load_datasets(inputs: Dict[str, Path]) -> Dict[str, pd.DataFrame]:
    """
    Loads all CSVs defined in the input dictionary.
    
    Args:
        inputs: Dictionary mapping dataset keys (names) to file paths.
        
    Returns:
        Dictionary mapping keys to loaded pandas DataFrames.
        
    Raises:
        FileNotFoundError: If a path does not exist.
    """
    dfs = {}
    for key, path in inputs.items():
        if not path.exists():
            raise FileNotFoundError(f"Input for '{key}' not found at: {path}")
        
        # Read CSV. We do not set an index_col to ensure we preserve all data 
        # row-by-row unless a specific column is guaranteed to be the index.
        df = pd.read_csv(path)
        dfs[key] = df
        logger.debug(f"Loaded '{key}': {len(df)} rows, {len(df.columns)} columns.")
    return dfs


def _validate_alignment(dataframes: Dict[str, pd.DataFrame]) -> None:
    """
    Ensures all dataframes have the exact same number of rows to maintain
    frame-by-frame synchronization.
    
    Args:
        dataframes: Dictionary of DataFrames to check.
        
    Raises:
        ValueError: If row counts differ between dataframes.
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


def _merge_dataframes(dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenates dataframes horizontally (axis=1) and removes duplicate columns.
    
    Args:
        dataframes: Dictionary of DataFrames to merge.
        
    Returns:
        A single DataFrame containing all columns from inputs, horizontally concatenated,
        with duplicate column names removed (keeping the first occurrence).
    """
    # Convert dict values to a list for concatenation
    df_list = list(dataframes.values())
    
    # Reset indices to ensure strict row alignment (ignoring existing index metadata)
    # and prevent pandas from aligning on potentially mismatched indices.
    df_list_reset = [df.reset_index(drop=True) for df in df_list]
    
    merged_df = pd.concat(df_list_reset, axis=1)
    
    # Identify duplicate column names
    # duplicated() returns True for duplicates (mark='first' is default, checking from left)
    is_duplicate = merged_df.columns.duplicated()
    
    if is_duplicate.any():
        logger.info(f"Removing {is_duplicate.sum()} duplicate columns found during merge.")
        # retain only columns that are NOT duplicates
        merged_df = merged_df.loc[:, ~is_duplicate]
    
    return merged_df


def _save_output(df: pd.DataFrame, path: Path) -> None:
    """
    Saves the dataframe to CSV, ensuring parent directories exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Successfully saved unified data to '{path}'. Dimensions: {df.shape}")


# --- Public API Wrapper ---

def unify_datasets(
    led_path: Path,
    contact_path: Path,
    trial_path: Path,
    single_touch_path: Path,
    stimuli_path: Path,
    output_path: Path,
    *,
    force_processing: bool = False
) -> bool:
    """
    Orchestrates the loading, validation, and horizontal merging of multiple
    pre-processed CSV datasets.
    
    Matches the signature required by the 'unify_processed_path' task 
    in the preprocessing workflow.
    """
    # Map inputs to a dictionary for cleaner processing
    inputs = {
        'led_path': led_path,
        'contact_path': contact_path, # Primary kinematics usually in contact path
        'trial_path': trial_path,
        'single_touch_path': single_touch_path,
        'stimuli_path': stimuli_path
    }
    
    # Check cache/existence logic using the utility function
    if not should_process_task(
        output_paths=[output_path], 
        input_paths=[contact_path, led_path, trial_path, single_touch_path, stimuli_path], 
        force=force_processing
    ):
        logger.info(f"Skipping task: Output '{output_path}' exists.")
        return True

    try:
        logger.info("Loading datasets for unification...")
        dataframes = _load_datasets(inputs)

        logger.info("Validating row alignment...")
        _validate_alignment(dataframes)

        logger.info("Merging datasets...")
        merged_df = _merge_dataframes(dataframes)

        logger.info("Saving unified dataset...")
        _save_output(merged_df, output_path)
        
        return True

    except FileNotFoundError as e:
        logger.error(f"File missing: {e}")
        raise
    except ValueError as e:
        logger.error(f"Data Validation Error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise