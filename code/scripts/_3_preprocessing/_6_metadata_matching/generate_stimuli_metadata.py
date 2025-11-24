from pathlib import Path
import pandas as pd
from utils.should_process_task import should_process_task

def generate_stimuli_metadata_to_data(
    trial_id_path: Path, 
    stimuli_path: Path, 
    output_path: Path, 
    force_processing: bool = False
):
    """
    Generates a standalone metadata file containing stimuli parameters.
    
    This function reads the available stimuli configurations and filters them
    to match the trial IDs present in the primary experiment data. 
    It explicitly renames columns and saves only the metadata, without 
    merging it back into the primary raw data file.

    Args:
        trial_id_path (Path): Path to the file containing experimental trial IDs.
        stimuli_path (Path): Path to the source file containing stimuli parameters.
        output_path (Path): Destination path for the processed metadata file.
        force_processing (bool): If True, ignores timestamp checks and re-processes.
    """
    
    # Validation: Check if processing is required using should_process_task
    # Checks modification times of input files vs output_path
    if not should_process_task(
        input_paths=[trial_id_path, stimuli_path], 
        output_paths=output_path, 
        force=force_processing
    ):
        print(f"Skipping stimuli metadata generation: Output file already exists and is up-to-date at {output_path}")
        return output_path
        
    print(f"Reading trial IDs from {trial_id_path} and stimuli source from {stimuli_path}")

    # 1. Load Data
    # We load the experiment data only to identify which trial IDs are relevant.
    trial_id_data = pd.read_csv(trial_id_path)
    raw_stimuli_source = pd.read_csv(stimuli_path)

    # 2. Identify Relevant Trials
    # Extract unique trial IDs to ensure we only process metadata for existing trials.
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

    # 4. Filter for Relevance
    # Keep only the rows where 'trial_id' exists in the experiment data.
    relevant_stimuli_data = extracted_stimuli_features[
        extracted_stimuli_features['trial_id'].isin(valid_trial_ids)
    ].copy()

    # 5. Rename Columns
    # Apply '_metadata' suffix to all columns except the join key ('trial_id').
    metadata_suffix = '_metadata'
    column_rename_map = {
        col: f"{col}{metadata_suffix}" 
        for col in relevant_stimuli_data.columns 
        if col != 'trial_id'
    }
    
    formatted_metadata_output = relevant_stimuli_data.rename(columns=column_rename_map)

    # 6. Save the Result
    # This saves ONLY the metadata columns (and trial_id), detached from the original sensor data.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    formatted_metadata_output.to_csv(output_path, index=False)
    print(f"Stimuli metadata saved to {output_path}")

    return output_path