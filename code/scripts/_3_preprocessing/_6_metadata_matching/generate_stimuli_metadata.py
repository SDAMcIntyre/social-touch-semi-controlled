from pathlib import Path
import pandas as pd
from utils.should_process_task import should_process_task

def generate_stimuli_metadata_to_data(
    trial_id_path: Path, 
    stimuli_path: Path, 
    output_path: Path, 
    aligned_output_path: Path,
    force_processing: bool = False
):
    """
    Generates two files:
    1. A standalone metadata file containing unique stimuli parameters (output_path).
    2. An aligned stimuli data file matching the row count and order of the trial ID file (aligned_output_path).

    Args:
        trial_id_path (Path): Path to the file containing experimental trial IDs.
        stimuli_path (Path): Path to the source file containing stimuli parameters.
        output_path (Path): Destination path for the unique metadata file.
        aligned_output_path (Path): Destination path for the expanded, row-aligned stimuli data.
        force_processing (bool): If True, ignores timestamp checks and re-processes.
    """
    
    # Validation: Check if processing is required using should_process_task
    # We pass both output paths to ensure both are generated if either is missing/outdated
    if not should_process_task(
        input_paths=[trial_id_path, stimuli_path], 
        output_paths=[output_path, aligned_output_path], 
        force=force_processing
    ):
        print(f"Skipping stimuli generation: Output files exist and are up-to-date.")
        return output_path, aligned_output_path
        
    print(f"Reading trial IDs from {trial_id_path} and stimuli source from {stimuli_path}")

    # 1. Load Data
    trial_id_data = pd.read_csv(trial_id_path)
    raw_stimuli_source = pd.read_csv(stimuli_path)

    # 2. Identify Relevant Trials
    valid_trial_ids = trial_id_data['trial_id'].unique()

    # 3. Filter and Select Columns
    # target_columns defines the specific physical parameters we need to extract.
    target_columns = ['trial_id', 'type', 'speed', 'contact_area', 'force']
    
    # Verify availability of columns in the source file
    available_columns = [col for col in target_columns if col in raw_stimuli_source.columns]
    
    # Create a subset of the stimuli data containing only relevant columns
    extracted_stimuli_features = raw_stimuli_source[available_columns].copy()
    
    if 'trial_id' not in extracted_stimuli_features.columns:
          raise KeyError(f"'trial_id' column not found in stimuli file: {stimuli_path.name}")

    # ---------------------------------------------------------
    # Definition: Column Renaming Logic
    # ---------------------------------------------------------
    metadata_suffix = '_metadata'
    # Create a map to add suffixes to all columns except trial_id
    column_rename_map = {
        col: f"{col}{metadata_suffix}" 
        for col in extracted_stimuli_features.columns 
        if col != 'trial_id'
    }

    # ---------------------------------------------------------
    # Process 1: Unique Metadata (Original Logic)
    # ---------------------------------------------------------
    
    # Filter for Relevance: Keep only rows where 'trial_id' exists in experiment data
    relevant_stimuli_metadata = extracted_stimuli_features[
        extracted_stimuli_features['trial_id'].isin(valid_trial_ids)
    ].copy()

    # Apply renaming map
    formatted_metadata_output = relevant_stimuli_metadata.rename(columns=column_rename_map)

    # Save Unique Metadata
    output_path.parent.mkdir(parents=True, exist_ok=True)
    formatted_metadata_output.to_csv(output_path, index=False)
    print(f"Unique stimuli metadata saved to {output_path}")

    # ---------------------------------------------------------
    # Process 2: Aligned Data (New Logic for N Rows)
    # ---------------------------------------------------------

    # Merge strategy: Left join on trial_id_data to preserve N rows and order.
    # We use trial_id_data[['trial_id']] to act as the spine for the merge.
    
    aligned_stimuli_data = pd.merge(
        trial_id_data[['trial_id']],
        extracted_stimuli_features,
        on='trial_id',
        how='left'
    )
    
    # Validation: Ensure row count matches N
    original_row_count = len(trial_id_data)
    aligned_row_count = len(aligned_stimuli_data)
    
    if aligned_row_count != original_row_count:
        raise RuntimeError(
            f"Row mismatch in aligned output. Expected {original_row_count}, got {aligned_row_count}. "
            "Check for duplicate trial_ids in the source stimuli file."
        )

    # Modification: Apply suffix to aligned columns
    aligned_stimuli_data = aligned_stimuli_data.rename(columns=column_rename_map)

    # Modification: Remove trial_id from the aligned output
    if 'trial_id' in aligned_stimuli_data.columns:
        aligned_stimuli_data = aligned_stimuli_data.drop(columns=['trial_id'])

    # Save Aligned Data
    aligned_output_path.parent.mkdir(parents=True, exist_ok=True)
    aligned_stimuli_data.to_csv(aligned_output_path, index=False)
    print(f"Aligned stimuli data ({aligned_row_count} rows) saved to {aligned_output_path}")

    return output_path, aligned_output_path