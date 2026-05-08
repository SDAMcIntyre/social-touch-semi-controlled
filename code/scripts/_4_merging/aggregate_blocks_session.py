import re
import shutil
import pandas as pd
from pathlib import Path
from typing import List, Optional
import logging

# Setup a basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from utils.should_process_task import should_process_task, clean_task_outputs

def aggregate_session_blocks(
    input_paths: List[Path],
    output_path: Path,
    *,
    forearm_ply_path: Optional[Path] = None,
    force_processing: bool = False
) -> Path:
    """
    Aggregates a specific list of block-level merged CSV files into one final CSV.
    
    Path resolution and file discovery are now decoupled from this function.
    It strictly handles validation, aggregation, and serialization.
    """
    
    # 1. Validation: Check if input files exist
    if not input_paths:
        logging.warning(f"⚠️ No input files provided for aggregation. Expected content for: {output_path.name}")
        # Return expected path, even if aggregation didn't happen
        return output_path

    # Derive session_id from output CSV name (pattern: {session_id}_semicontrolled_aggregated_session.csv)
    _csv_stem = output_path.stem  # e.g. "ST13-03_semicontrolled_aggregated_session"
    _session_id = _csv_stem.replace("_semicontrolled_aggregated_session", "")
    forearm_ply_dest = (
        output_path.parent / f"{_session_id}_forearm.ply"
        if forearm_ply_path is not None else None
    )

    # 2. Check if processing is required using should_process_task
    # Only the CSV inputs/output are checked — the forearm PLY is a side-effect copy
    # whose preserved mtime (shutil.copy2) would otherwise always trigger a false stale.
    if not should_process_task(
        input_paths=list(input_paths),
        output_paths=[output_path],
        force=force_processing
    ):
        logging.info(f"✅ Output file '{output_path.name}' already exists and is up-to-date. Use force_processing to overwrite.")
        return output_path
    clean_task_outputs(output_path)
    if forearm_ply_dest is not None:
        forearm_ply_dest.unlink(missing_ok=True)

    # Proceed with aggregation if check passed
    logging.info(f"Aggregating {len(input_paths)} blocks into {output_path.name}...")

    df_list = []
    for filename in input_paths:
        try:
            df = pd.read_csv(filename)
            # Add a column indicating the source file (block)
            df['source_block_file'] = filename.name
            match = re.search(r'block-order-(\d+)', filename.name)
            df['block_order_id'] = match.group(1) if match else None
            df_list.append(df)
        except Exception as e:
            logging.error(f"Failed to read {filename}: {e}")

    if not df_list:
        logging.warning("No valid dataframes could be loaded from the provided input paths.")
        return output_path

    merged_df = pd.concat(df_list, ignore_index=True)

    # Ensure directory exists before saving
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)

    logging.info(f"✅ Successfully aggregated {len(input_paths)} blocks into: {output_path.name}")

    if forearm_ply_path is not None and forearm_ply_path.exists() and forearm_ply_dest is not None:
        shutil.copy2(forearm_ply_path, forearm_ply_dest)
        logging.info(f"✅ Copied forearm PLY to session root: {forearm_ply_dest.name}")

    return output_path