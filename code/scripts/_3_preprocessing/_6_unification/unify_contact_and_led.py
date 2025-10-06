
from pathlib import Path
import pandas as pd

from utils.should_process_task import should_process_task
from preprocessing.led_analysis import (
    LEDBlinkingFilesHandler
)



def merge_dataframes_with_checks(df1: pd.DataFrame, df2: pd.DataFrame, key_column: str = 'time') -> pd.DataFrame:
    """
    Merges two pandas DataFrames on a specified key column after performing consistency checks.

    Args:
        df1 (pd.DataFrame): The first DataFrame (e.g., led_df).
        df2 (pd.DataFrame): The second DataFrame (e.g., contact_chars_df).
        key_column (str): The name of the common column to merge on. Defaults to 'time'.

    Returns:
        pd.DataFrame: A new DataFrame containing the merged data.
        
    Raises:
        ValueError: If the key column is not found in one or both DataFrames.
    """
    print(f"🚀 Starting merge process on key: '{key_column}'")

    # 1. Existence Check
    if key_column not in df1.columns or key_column not in df2.columns:
        raise ValueError(f"Error: Key column '{key_column}' not found in both DataFrames.")
    print("✅ Key column found in both DataFrames.")

    # 2. Data Type Check
    if df1[key_column].dtype != df2[key_column].dtype:
        print(f"⚠️ Warning: Key column '{key_column}' has different data types: "
              f"({df1.name}: {df1[key_column].dtype}, {df2.name}: {df2[key_column].dtype}). "
              f"Attempting to standardize to numeric.")
        # Attempt to convert to a common numeric type to prevent merge errors
        df1[key_column] = pd.to_numeric(df1[key_column], errors='coerce')
        df2[key_column] = pd.to_numeric(df2[key_column], errors='coerce')
        # Drop rows where conversion resulted in NaT/NaN if any
        df1.dropna(subset=[key_column], inplace=True)
        df2.dropna(subset=[key_column], inplace=True)
        print("   -> Data types standardized.")
    else:
        print(f"✅ Key columns have a consistent data type: {df1[key_column].dtype}.")

    # 3. Uniqueness Check
    if df1[key_column].duplicated().any():
        print(f"⚠️ Warning: Duplicate values found in the key column of the first DataFrame.")
    if df2[key_column].duplicated().any():
        print(f"⚠️ Warning: Duplicate values found in the key column of the second DataFrame.")
    else:
        print("✅ Key columns are unique within each DataFrame.")

    # 4. Overlap Analysis
    keys1 = set(df1[key_column])
    keys2 = set(df2[key_column])
    common_keys_count = len(keys1.intersection(keys2))
    unique_to_df1_count = len(keys1 - keys2)
    unique_to_df2_count = len(keys2 - keys1)

    print("\n📊 Data Overlap Report:")
    print(f"   - Common timestamps to be merged: {common_keys_count}")
    print(f"   - Timestamps unique to first DataFrame (will be dropped): {unique_to_df1_count}")
    print(f"   - Timestamps unique to second DataFrame (will be dropped): {unique_to_df2_count}")

    if common_keys_count == 0:
        print("\n❌ Error: No common timestamps found. Returning an empty DataFrame.")
        return pd.DataFrame()

    # Perform the merge
    print("\nMerging DataFrames using an 'inner' join...")
    merged_df = pd.merge(df1, df2, on=key_column, how='inner')
    print(f"✨ Merge complete! New DataFrame has {merged_df.shape[0]} rows and {merged_df.shape[1]} columns.")
    
    return merged_df


def unify_contact_caracteristics_and_ttl(
        contact_chars_path: Path,
        ttl_path: Path,
        output_path: Path,
        *,
        force_processing: bool = False
):
    if not should_process_task(
        input_paths=[contact_chars_path, ttl_path], 
        output_paths=[output_path], 
        force=force_processing):
        print(f"✅ Output file '{output_path}' already exists. Use force_processing to overwrite.")
        return

    led_filehandler = LEDBlinkingFilesHandler()
    led_df = led_filehandler.load_timeseries_from_csv(ttl_path, output_format='dataframe')
    led_df.drop(columns='green_levels', inplace=True)

    contact_chars_df = pd.read_csv(contact_chars_path)

    merged_df = merge_dataframes_with_checks(contact_chars_df, led_df, key_column='time')

    merged_df.to_csv(output_path, index=False)

    return output_path