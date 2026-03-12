import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from sklearn.decomposition import PCA

# Import the idempotency check utility
from utils.should_process_task import should_process_task

from postprocessing.xyz_reference_from_gestures import (
    PCACalibrationEngine,
    CalibrationVisualizer,
    Trajectory3DVisualizer
)
from preprocessing.forearm_extraction.registration.csv_spatial_transformer import (
    parse_contact_points,
    serialize_contact_points,
)

# Setup module-level logger
logger = logging.getLogger(__name__)

# --- 1. Configuration Layer ---

@dataclass(frozen=True)
class CalibrationConfig:
    """Immutable configuration for gesture calibration."""
    col_type: str = 'type_metadata'
    col_touch_id: str = 'single_touch_id'
    col_contact_area: str = 'contact_area_metadata'

    val_tapping: str = 'tap'
    val_stroking: str = 'stroke'
    val_one_finger_tip: str = 'one finger tip'

    output_json_name: str = "pca-xyz_transformation-matrices.json"
    output_csv_suffix: str = "_pca-xyz.csv"

    target_colors: Tuple[str, ...] = ('blue', 'yellow', 'green')

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
            # dynamically build columns for all target colors + metadata
            data_cols = []
            for color in self.config.target_colors:
                data_cols.append(f"sticker_{color}_position_x")
                data_cols.append(f"sticker_{color}_position_y")
                data_cols.append(f"sticker_{color}_position_z")

            meta_cols = [
                self.config.col_type,
                self.config.col_touch_id,
                self.config.col_contact_area,
            ]

            df = pd.read_csv(file_path, usecols=data_cols + meta_cols)

            # Drop rows missing the columns required for every subsequent operation.
            df_clean = df.dropna(subset=[self.config.col_type, self.config.col_touch_id])

            # single_touch_id == 0 → no valid touch (codebase convention).
            df_clean = df_clean[df_clean[self.config.col_touch_id] != 0]

            if df_clean.empty:
                return None
            return df_clean
            
        except ValueError as ve:
            logger.debug(f"Skipping {file_path.name}: {ve}")
            return None
        except Exception as e:
            logger.error(f"Error reading {file_path.name}: {e}")
            return None



def _extract_touch_data(
    touch_group: pd.DataFrame,
    config: CalibrationConfig,
) -> Optional[np.ndarray]:
    """
    Extracts mean-centred 3-D coordinates from one touch event.

    Sticker selection is driven by contact_area_metadata:
      - 'one finger tip'  →  blue sticker only
      - anything else     →  green + yellow stickers (concatenated)

    Each sticker's contribution is independently mean-centred before
    concatenation so that absolute position differences between stickers
    do not bias the PCA.

    Returns an (N, 3) ndarray, or None if no valid rows exist.
    """
    contact_type_series = touch_group[config.col_contact_area].dropna()
    if contact_type_series.empty:
        return None
    contact_type = contact_type_series.iloc[0]

    colors_to_use = ['blue'] if contact_type == config.val_one_finger_tip else ['green', 'yellow']

    centered_arrays = []
    for color in colors_to_use:
        cx, cy, cz = (f"sticker_{color}_position_{ax}" for ax in 'xyz')
        if not all(c in touch_group.columns for c in [cx, cy, cz]):
            continue
        coords = touch_group[[cx, cy, cz]].dropna().values
        if len(coords) == 0:
            continue
        centered_arrays.append(coords - coords.mean(axis=0))

    return np.vstack(centered_arrays) if centered_arrays else None


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

    output_dir.mkdir(parents=True, exist_ok=True)
    
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
            
            df_tap = df[df[config.col_type] == config.val_tapping]
            df_stroke = df[df[config.col_type] == config.val_stroking]

            for _, group in df_tap.groupby(config.col_touch_id):
                data = _extract_touch_data(group, config)
                if data is not None:
                    tapping_segments.append(data)

            for _, group in df_stroke.groupby(config.col_touch_id):
                data = _extract_touch_data(group, config)
                if data is not None:
                    stroking_segments.append(data)
            

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
            logger.info(">>> Visualizing Tapping Segments (Step 1: Z-Aligned)")
            for i, seg in enumerate(tapping_segments):
                transformed_seg = PCACalibrationEngine.apply_step1_transform(seg, calib_result)
                logger.info(f"Displaying Tapping Segment {i+1}/{len(tapping_segments)}")
                viz = Trajectory3DVisualizer(transformed_seg)
                viz.show()

            # --- Visualize Stroking Segments (Full Aligned) ---
            logger.info(">>> Visualizing Stroking Segments (Final: XY-Aligned)")
            for i, seg in enumerate(stroking_segments):
                transformed_seg = PCACalibrationEngine.apply_full_transform(seg, calib_result)
                logger.info(f"Displaying Stroking Segment {i+1}/{len(stroking_segments)}")
                viz = Trajectory3DVisualizer(transformed_seg)
                viz.show()
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            pass

    if monitor:
        try:
            logger.info("Displaying Global Aggregate Summary...")
            CalibrationVisualizer.visualize(all_tapping, all_stroking, calib_result, output_dir.name)
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            pass

    # 5. Apply Transformation to loaded data and Save
    logger.info("Phase 2: Applying transformation to [Blue, Yellow, Green] and saving files...")
    generated_files = []
    
    for input_path, df in loaded_data.items():
        output_path = output_dir / f"{input_path.stem}{config.output_csv_suffix}"
        
        try:
            # 1. Read the full original file to preserve original structure (and NaNs)
            df_out = pd.read_csv(input_path)

            # 2. Apply transformation to each color defined in config
            for color in config.target_colors:
                c_x = f"sticker_{color}_position_x"
                c_y = f"sticker_{color}_position_y"
                c_z = f"sticker_{color}_position_z"

                if not all(c in df_out.columns for c in [c_x, c_y, c_z]):
                    logger.warning(f"Skipping color '{color}' for {input_path.name}: Columns missing.")
                    continue

                coords = df_out[[c_x, c_y, c_z]].to_numpy(dtype=np.float64)
                mask = ~np.isnan(coords).any(axis=1)
                if mask.any():
                    coords_transformed = coords.copy()
                    coords_transformed[mask] = PCACalibrationEngine.apply_full_transform(
                        coords[mask], calib_result
                    )
                    df_out[c_x] = coords_transformed[:, 0]
                    df_out[c_y] = coords_transformed[:, 1]
                    df_out[c_z] = coords_transformed[:, 2]

            # 3. Apply transformation to contact_location_x/y/z
            loc_cols = ["contact_location_x", "contact_location_y", "contact_location_z"]
            if all(c in df_out.columns for c in loc_cols):
                loc = df_out[loc_cols].to_numpy(dtype=np.float64)
                mask = ~np.isnan(loc).any(axis=1)
                if mask.any():
                    loc_transformed = loc.copy()
                    loc_transformed[mask] = PCACalibrationEngine.apply_full_transform(
                        loc[mask], calib_result
                    )
                    for i, col in enumerate(loc_cols):
                        df_out[col] = loc_transformed[:, i]

            # 4. Apply transformation to contact_points
            if "contact_points" in df_out.columns:
                new_contact_points = []
                for cell in df_out["contact_points"]:
                    pts = parse_contact_points(cell)
                    if pts:
                        arr = np.asarray(pts, dtype=np.float64).copy()
                        arr = PCACalibrationEngine.apply_full_transform(arr, calib_result)
                        new_contact_points.append(
                            serialize_contact_points([tuple(row) for row in arr.tolist()])
                        )
                    else:
                        new_contact_points.append(cell)
                df_out["contact_points"] = new_contact_points

            df_out.to_csv(output_path, index=False)
            generated_files.append(output_path)
            
        except Exception as e:
            logger.error(f"Failed to save processed file {input_path.name}: {e}")

    return generated_files, output_dir