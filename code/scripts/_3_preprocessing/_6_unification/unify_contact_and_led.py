import logging
from pathlib import Path
import pandas as pd
from typing import Optional

# Configure basic logging to print to the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from utils.should_process_task import should_process_task
from preprocessing.led_analysis import (
    LEDBlinkingFilesHandler
)



def merge_dataframes_by_time_tolerant(
    df1: pd.DataFrame, 
    df2: pd.DataFrame, 
    key_column: str = 'time',
    decimal_precision: Optional[int] = None
) -> tuple[pd.DataFrame, bool]:
    """
    Merges two pandas DataFrames on a time column with optional tolerance for floats.

    If decimal_precision is provided, the key column is rounded to that number of
    decimal places before merging. This allows for merging on floating-point time
    values that are "close enough". An outer join is always used to ensure no 
    data is lost. After the merge, it ensures only a single time column remains.

    Args:
        df1 (pd.DataFrame): The first DataFrame. Its time values are prioritized.
        df2 (pd.DataFrame): The second DataFrame.
        key_column (str): The name of the common time column to merge on.
        decimal_precision (Optional[int]): The number of decimal places to round the
                                           key_column to for merging. If None, an
                                           exact match is performed. Defaults to None.

    Returns:
        tuple[pd.DataFrame, bool]: A tuple containing:
            - pd.DataFrame: A new DataFrame with the merged data. Empty on failure.
            - bool: True if the merge was successful, False otherwise.
    """
    print(f"🚀 Starting merge process on key: '{key_column}'")

    # 1. Existence Check: The key must be in both DataFrames.
    if key_column not in df1.columns or key_column not in df2.columns:
        print(f"❌ Error: Key column '{key_column}' not found in both DataFrames.")
        return pd.DataFrame(), False
    print("✅ Key column found in both DataFrames.")

    # --- Logic branch based on decimal_precision ---
    if decimal_precision is not None:
        print(f"🔬 Using float precision matching up to {decimal_precision} decimal places.")
        
        # Create temporary copies to avoid modifying the original DataFrames
        df1_temp = df1.copy()
        df2_temp = df2.copy()
        
        # Define a temporary key for the rounded values
        temp_merge_key = f"_{key_column}_rounded"
        
        # Create the new column with rounded time values in each temp DataFrame
        try:
            df1_temp[temp_merge_key] = df1_temp[key_column].round(decimal_precision)
            df2_temp[temp_merge_key] = df2_temp[key_column].round(decimal_precision)
        except TypeError:
            print(f"❌ Error: 'decimal_precision' is set, but the column '{key_column}' is not numeric.")
            return pd.DataFrame(), False

        # 2. Overlap Check (on the rounded key)
        keys1 = set(df1_temp[temp_merge_key].dropna())
        keys2 = set(df2_temp[temp_merge_key].dropna())

        if not keys1.intersection(keys2):
            print(f"❌ Error: No common time values found up to {decimal_precision} decimals. Merge aborted.")
            return pd.DataFrame(), False
        print(f"✅ Found {len(keys1.intersection(keys2))} common rounded time values. Proceeding with merge.")

        # 3. Perform the Merge using the temporary rounded key
        print("\nMerging DataFrames using an 'outer' join on the rounded key...")
        merged_df = pd.merge(df1_temp, df2_temp, on=temp_merge_key, how='outer', suffixes=('_df1', '_df2'))
        
        # Clean up by dropping the temporary merge key
        merged_df.drop(columns=[temp_merge_key], inplace=True)
        
        # --- Post-Merge Cleanup: Coalesce time columns ---
        # When merging on a temp key, the original 'key_column' from both
        # dataframes are preserved with suffixes (e.g., 'time_df1', 'time_df2').
        # We need to combine them into a single column.
        time_col_df1 = f"{key_column}_df1"
        time_col_df2 = f"{key_column}_df2"
        
        print(f"✨ Consolidating '{time_col_df1}' and '{time_col_df2}' into a single '{key_column}' column.")
        
        # Use the time from df1, but fill any missing values with the time from df2.
        # This handles rows that were unique to df2.
        merged_df[key_column] = merged_df[time_col_df1].combine_first(merged_df[time_col_df2])
        
        # Drop the now-redundant suffixed time columns
        merged_df.drop(columns=[time_col_df1, time_col_df2], inplace=True)

    else:
        # --- Original Exact Match Logic ---
        print("🔬 Using exact matching for the key.")
        
        # 2. Overlap Check
        keys1 = set(df1[key_column].dropna())
        keys2 = set(df2[key_column].dropna())
        
        if not keys1.intersection(keys2):
            print("❌ Error: No common time values found between the DataFrames. Merge aborted.")
            return pd.DataFrame(), False
        print(f"✅ Found {len(keys1.intersection(keys2))} common time values. Proceeding with merge.")

        # 3. Perform the Merge
        print("\nMerging DataFrames using an 'outer' join to preserve all data...")
        merged_df = pd.merge(df1, df2, on=key_column, how='outer', suffixes=('_df1', '_df2'))
    
    # Optional but good practice: move the key column to the front
    cols = [key_column] + [col for col in merged_df.columns if col != key_column]
    merged_df = merged_df[cols]
    
    print(f"\n🎉 Merge complete! New DataFrame has {merged_df.shape[0]} rows and {merged_df.shape[1]} columns.")
    return merged_df, True


def unify_contact_caracteristics_and_ttl(
        contact_chars_path: Path,
        ttl_path: Path,
        output_path: Path,
        *,
        force_processing: bool = False
) -> bool:
    """
    Merges contact characteristics and TTL LED data into a single file.

    Args:
        contact_chars_path: Path to the contact characteristics CSV.
        ttl_path: Path to the TTL data CSV.
        output_path: Path for the merged output CSV file.
        force_processing: If True, overwrites the output file even if it exists.

    Returns:
        True if the operation was successful (or skipped correctly), False otherwise.
    """
    if not should_process_task(
        input_paths=[contact_chars_path, ttl_path],
        output_paths=[output_path],
        force=force_processing
    ):
        logging.info(f"✅ Skipping task: Output file '{output_path}' already exists.")
        return True # Task is successfully skipped, so we return True.

    try:
        led_filehandler = LEDBlinkingFilesHandler()
        led_df = led_filehandler.load_timeseries_from_csv(ttl_path, output_format='dataframe')
        
        # Ensure 'green_levels' column exists before dropping
        if 'green_levels' in led_df.columns:
            led_df.drop(columns='green_levels', inplace=True)

        contact_chars_df = pd.read_csv(contact_chars_path)

        merged_df, success = merge_dataframes_by_time_tolerant(
            contact_chars_df, led_df, key_column='time', decimal_precision = 3
        )

        # Get the list of columns from the DataFrame
        cols = merged_df.columns.tolist()
        # Remove 'led_on' from its current position
        cols.remove('led_on')
        # Find the index of the 'frame_index' column and insert 'led_on' right after it
        cols.insert(cols.index('frame_index') + 1, 'led_on')
        # Apply the new column order to the DataFrame
        merged_df = merged_df[cols]

        # Verify the new order
        print(merged_df.columns)
        if success:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            merged_df.to_csv(output_path, index=False)
            logging.info(f"🎉 Successfully merged data to '{output_path}'.")
            return True
        else:
            # This is the main failure case you wanted to log.
            logging.error(
                f"❌ Failed to merge dataframes. "
                f"Input files: '{contact_chars_path}' and '{ttl_path}'."
            )
            return False

    except FileNotFoundError as e:
        logging.error(f"❌ File not found during processing: {e}")
        return False
    except Exception as e:
        logging.error(f"❌ An unexpected error occurred: {e}", exc_info=True)
        return False