from pathlib import Path
import pandas as pd
from utils.should_process_task import should_process_task

def add_stimuli_metadata_to_data(
    data_path: Path, 
    stimuli_path: Path, 
    output_path: Path, 
    force_processing: bool = False
):
    """
    Implements the logic to merge stimuli data into the main dataset.
    Utilizes timestamp-based verification to skip processing if outputs are up-to-date.
    Explicitly renames stimuli columns to ensure consistent naming conventions.
    """
    
    # Validation: Check if processing is required using should_process_task
    # Checks modification times of [data_path, stimuli_path] vs output_path
    if not should_process_task(
        input_paths=[data_path, stimuli_path], 
        output_paths=output_path, 
        force=force_processing
    ):
        print(f"Skipping stimuli metadata addition: Output file already exists and is up-to-date at {output_path}")
        return
        
    print(f"Reading data from {data_path} and stimuli from {stimuli_path}")

    # 1. Load Data and Stimuli
    data_df = pd.read_csv(data_path)
    stimuli_df = pd.read_csv(stimuli_path)

    # 2. Select required columns from stimuli
    stimuli_cols = ['trial_id', 'type', 'speed', 'contact_area', 'force']
    # Filter columns and ensure 'trial_id' is present for merging
    stimuli_metadata = stimuli_df[[col for col in stimuli_cols if col in stimuli_df.columns]].copy()
    
    if 'trial_id' not in stimuli_metadata.columns:
         raise KeyError(f"'trial_id' column not found in stimuli file: {stimuli_path.name}")

    # 3. Rename columns explicitly before merging
    # This ensures the suffix is applied to ALL stimuli columns, not just overlapping ones.
    suffix = '_metadata'
    rename_map = {
        col: f"{col}{suffix}" 
        for col in stimuli_metadata.columns 
        if col != 'trial_id'
    }
    
    stimuli_metadata_renamed = stimuli_metadata.rename(columns=rename_map)

    # 4. Merge dataframes on 'trial_id'
    # Use a left merge to keep all rows from the main data file.
    # Suffixes argument is removed as renaming is handled upstream.
    merged_df = data_df.merge(
        stimuli_metadata_renamed, 
        on='trial_id', 
        how='left'
    )
    
    # 5. Save the result
    # Ensure parent directory exists to avoid FileNotFoundError on write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    print(f"Merged data saved to {output_path}")

    return output_path