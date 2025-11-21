import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from sklearn.decomposition import PCA

# Import the idempotency check utility
from utils.should_process_task import should_process_task

from postprocessing.xyz_reference_from_gestures import (
    PCACalibrationEngine,
    CalibrationVisualizer,
    Trajectory3DVisualizer
)

# Setup module-level logger
logger = logging.getLogger(__name__)

# --- 1. Configuration Layer ---

@dataclass(frozen=True)
class CalibrationConfig:
    """Immutable configuration for gesture calibration."""
    col_x: str = 'sticker_blue_position_x'
    col_y: str = 'sticker_blue_position_y'
    col_z: str = 'sticker_blue_position_z'
    col_type: str = 'type_metadata'
    col_trial: str = 'trial_id'
    col_contact: str = 'contact_detected'
    
    val_tapping: str = 'tap'
    val_stroking: str = 'stroke'
    
    output_json_name: str = "pca-xyz_transformation-matrices.json"
    output_csv_suffix: str = "_pca-xyz.csv"

# --- 2. Data Ingestion Layer ---

class GestureDataLoader:
    """Handles file I/O and initial data segmentation."""
    
    def __init__(self, config: CalibrationConfig):
        self.config = config

    def load_and_segment(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Loads a CSV and performs initial cleaning.
        Returns None if required columns are missing or file is empty.
        """
        if not file_path.exists():
            logger.warning(f"Input file not found: {file_path}")
            return None
            
        try:
            usecols = [
                self.config.col_x, self.config.col_y, self.config.col_z,
                self.config.col_type, self.config.col_contact, self.config.col_trial
            ]
            df = pd.read_csv(file_path, usecols=usecols)
            
            # Drop NaNs only in critical columns
            # Note: This preserves the original index, which is crucial for merging back later.
            df_clean = df.dropna(subset=usecols)
            
            if df_clean.empty:
                return None
                
            return df_clean
            
        except ValueError as ve:
            logger.debug(f"Skipping {file_path.name}: {ve}")
            return None
        except Exception as e:
            logger.error(f"Error reading {file_path.name}: {e}")
            return None

    def extract_active_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Groups by trial_id and slices from first to last detected contact.
        """
        valid_segments = []
        
        for _, group in df.groupby(self.config.col_trial):
            contact_rows = group[group[self.config.col_contact] == 1]
            
            if contact_rows.empty:
                continue
                
            start_idx = contact_rows.index[0]
            end_idx = contact_rows.index[-1]
            
            # Slice includes the end index in pandas loc
            segment = group.loc[start_idx:end_idx]
            valid_segments.append(segment)
            
        if not valid_segments:
            return pd.DataFrame(columns=df.columns)
            
        return pd.concat(valid_segments)


# --- 5. Orchestrator ---

def set_xyz_reference_from_gestures(
    input_files: List[Path], 
    output_dir: Path,
    *,
    force_processing: bool = False,
    monitor: bool = False,
    monitor_segment: bool = False
) -> Tuple[List[Path], Path]:
    """
    Orchestrates the global PCA analysis pipeline.
    """
    logger.info(f"[{output_dir.name}] Starting Global PCA pipeline on {len(input_files)} files.")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    config = CalibrationConfig()
    
    # Prepare expected outputs for idempotency
    json_output_path = output_dir / config.output_json_name
    expected_output_files = [output_dir / f"{p.stem}{config.output_csv_suffix}" for p in input_files]
    
    all_check_outputs = expected_output_files + [json_output_path]
    
    if not should_process_task(
        input_paths=input_files,
        output_paths=all_check_outputs,
        force=force_processing
    ):
        logger.info(f"[{output_dir.name}] Task up-to-date. Skipping.")
        return expected_output_files, output_dir

    # Initialize Components
    loader = GestureDataLoader(config)
    
    # 1. Load Data (into memory map)
    loaded_data: Dict[Path, pd.DataFrame] = {}
    
    logger.info("Phase 1: Ingesting data...")
    
    tapping_segments = []
    stroking_segments = []

    for input_path in input_files:
        df = loader.load_and_segment(input_path)
        if df is not None:
            loaded_data[input_path] = df
            
            # Filter specific types for calibration calculation
            df_tap = df[df[config.col_type] == config.val_tapping]
            df_stroke = df[df[config.col_type] == config.val_stroking]
            
            # Apply segmentation logic
            tap_seg = loader.extract_active_segments(df_tap)
            stroke_seg = loader.extract_active_segments(df_stroke)
            
            # Store raw segments individually before grouping
            if not tap_seg.empty:
                # Group by trial to get individual arrays for sequential visualization
                for _, group in tap_seg.groupby(config.col_trial):
                    tapping_segments.append(group[[config.col_x, config.col_y, config.col_z]].values)
            
            if not stroke_seg.empty:
                for _, group in stroke_seg.groupby(config.col_trial):
                    stroking_segments.append(group[[config.col_x, config.col_y, config.col_z]].values)
            

    if not tapping_segments or not stroking_segments:
        logger.error("Insufficient data for calibration (missing tap or stroke segments).")
        return [], output_dir

    # 2. Compute Calibration
    logger.info("Computing PCA Matrices...")
    all_tapping = np.vstack(tapping_segments)
    all_stroking = np.vstack(stroking_segments)
    
    calib_result = PCACalibrationEngine.compute_calibration(all_tapping, all_stroking)
    
    # 3. Save Calibration
    with open(json_output_path, 'w') as f:
        json.dump(calib_result.to_dict(), f, indent=4)
    
    # 4. Monitor
    if monitor_segment:
        try:
            logger.info("Starting Interactive Segment Visualization.")
            logger.info("Close the visualization window to proceed to the next segment.")

            # --- Visualize Tapping Segments (Step 1 Aligned) ---
            # Tapping segments are used to flatten the Z-plane. We visualize them after Step 1 
            # transform to verify they are "flat".
            logger.info(">>> Visualizing Tapping Segments (Step 1: Z-Aligned)")
            for i, seg in enumerate(tapping_segments):
                # Transform raw segment using Step 1 (Z alignment)
                transformed_seg = PCACalibrationEngine.apply_step1_transform(seg, calib_result)
                
                logger.info(f"Displaying Tapping Segment {i+1}/{len(tapping_segments)}")
                viz = Trajectory3DVisualizer(transformed_seg)
                # Optionally, we could set a specific title if the Visualizer supported it publicly,
                # but the default window title is handled internally.
                viz.show()

            # --- Visualize Stroking Segments (Full Aligned) ---
            # Stroking segments are used to align X/Y. We visualize them after Full transform.
            logger.info(">>> Visualizing Stroking Segments (Final: XY-Aligned)")
            for i, seg in enumerate(stroking_segments):
                # Transform raw segment using Full Transform
                transformed_seg = PCACalibrationEngine.apply_full_transform(seg, calib_result)
                
                logger.info(f"Displaying Stroking Segment {i+1}/{len(stroking_segments)}")
                viz = Trajectory3DVisualizer(transformed_seg)
                viz.show()
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            # Ensure pipeline continues even if viz fails
            pass

    if monitor:
        try:
            # Visualize Aggregate Summary (Original implementation retained for global view)
            logger.info("Displaying Global Aggregate Summary...")
            CalibrationVisualizer.visualize(all_tapping, all_stroking, calib_result, output_dir.name)
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            # Ensure pipeline continues even if viz fails
            pass

    # 5. Apply Transformation to loaded data and Save
    logger.info("Phase 2: Applying transformation and saving files...")
    generated_files = []
    
    for input_path, df in loaded_data.items():
        output_path = output_dir / f"{input_path.stem}{config.output_csv_suffix}"
        
        try:
            # Extract coords from the cleaned DataFrame (df)
            # df matches the row count of transformed_coords
            coords = df[[config.col_x, config.col_y, config.col_z]].values
            
            # Apply Full Transform
            transformed_coords = PCACalibrationEngine.apply_full_transform(coords, calib_result)
            
            # Update DataFrame
            # 1. Read the full original file to preserve original structure (and NaNs)
            df_out = pd.read_csv(input_path) 
            
            # 2. Use index-based assignment (.loc).
            # 'df.index' contains the indices of rows that survived the dropna() in load_and_segment.
            # We map the transformed coordinates exactly to those rows in the output DataFrame.
            df_out.loc[df.index, config.col_x] = transformed_coords[:, 0]
            df_out.loc[df.index, config.col_y] = transformed_coords[:, 1]
            df_out.loc[df.index, config.col_z] = transformed_coords[:, 2]
            
            df_out.to_csv(output_path, index=False)
            generated_files.append(output_path)
            
        except Exception as e:
            logger.error(f"Failed to save processed file {input_path.name}: {e}")

    return generated_files, output_dir