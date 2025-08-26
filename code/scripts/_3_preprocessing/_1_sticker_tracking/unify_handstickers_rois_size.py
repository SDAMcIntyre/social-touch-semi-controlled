# Standard library imports
import json
import os
from typing import Any, Dict, Tuple

# Third-party imports
import pandas as pd

# Local application imports
from .utils.roi_to_position_funcs import parse_rois_from_dataframe


def get_max_roi_values(tracked_rois: pd.Series) -> Tuple[int, int, int, int]:
    """
    Calculates the maximum value for each column across all ROIs in a Series.

    This function is useful for finding the bounding box that encompasses all
    individual ROIs, for instance, to determine the required size of a canvas
    or a crop area.

    Args:
        tracked_rois: A pandas Series where each element is a list or tuple
                      representing an ROI, typically as (x, y, width, height).

    Returns:
        A tuple containing the maximum x, y, width, and height found across
        all ROIs. Returns (0, 0, 0, 0) if the input Series is empty.
    """
    valid_rois = tracked_rois.dropna()
    if valid_rois.empty:
        return (0, 0, 0, 0)
    roi_df = pd.DataFrame(
        valid_rois.tolist(),
        columns=['x', 'y', 'width', 'height']
    )
    max_values = roi_df.max()
    return (
        int(max_values['x']),
        int(max_values['y']),
        int(max_values['width']),
        int(max_values['height'])
    )


def unify_roi_size(tracked_rois: pd.Series) -> pd.Series:
    """
    Unifies the size of all ROIs in a Series to the maximum dimensions found.

    Each new ROI is centered on the same point as the original ROI. This is
    achieved by finding the maximum width and height across all valid ROIs
    and then resizing each ROI to these dimensions while adjusting its top-left
    (x, y) coordinate to maintain the original center.

    Args:
        tracked_rois: A pandas Series where each element is a list or tuple
                      representing an ROI, typically as (x, y, width, height).
                      The Series can contain non-ROI values like NaN, which
                      will be preserved.

    Returns:
        A new pandas Series with all valid ROIs resized to the maximum width and
        height, while maintaining their original centers. Non-ROI values are
        preserved in their original positions.
    """
    # 1. Isolate valid ROIs to perform calculations, ignoring NaNs.
    valid_rois = tracked_rois.dropna()
    if valid_rois.empty:
        return tracked_rois # Return original Series if no valid ROIs exist.

    # 2. Use the helper function to get the maximum width and height from all ROIs.
    # We only need the width and height for this operation.
    _, _, max_w, max_h = get_max_roi_values(valid_rois)

    def recenter_roi(roi: Any) -> Any:
        """Applies the resizing and recentering logic to a single ROI."""
        # Ensure the element is a valid ROI before processing.
        if isinstance(roi, (list, tuple)) and len(roi) == 4:
            x, y, w, h = roi
            
            # a. Calculate the original center point.
            center_x = x + w / 2
            center_y = y + h / 2
            
            # b. Calculate the new top-left corner using the max dimensions
            #    to keep the center point the same.
            new_x = center_x - max_w / 2
            new_y = center_y - max_h / 2
            
            # c. Return the new, unified ROI as a list of integers.
            return [int(new_x), int(new_y), int(max_w), int(max_h)]
        
        # If the element is not a valid ROI (e.g., NaN), return it unchanged.
        return roi

    # 3. Apply the recentering function to each element in the *original* Series.
    # This ensures that NaN values are preserved in their correct locations.
    return tracked_rois.apply(recenter_roi)


def get_obj_name(metadata: dict) -> list[str]:
    keys_list = list(metadata.keys())
    return keys_list


def _load_and_validate_inputs(md_path: str, tracking_path: str) -> Dict[str, Any]:
    """
    Responsibility: Loads and validates metadata and tracking data from files.
    """
    if not os.path.exists(md_path):
        raise FileNotFoundError(f"Metadata file not found at '{md_path}'.")
    try:
        with open(md_path, "r") as f:
            metadata = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        raise ValueError(f"Could not read or parse metadata file '{md_path}'.") from e
    tracking_df = pd.read_csv(tracking_path)
    print(f"✅ Successfully loaded metadata and tracking data.")
    return metadata, tracking_df


def unify_objects_rois_size(
    roi_path: str,
    md_path: str,
    output_roi_path: str
):
    """
    Main processing pipeline to unify ROI sizes for specified objects.
    
    It loads tracking data from a CSV, processes each object's ROIs to make 
    them uniform in size based on the largest dimensions found, and saves the 
    resulting DataFrame to a new CSV file.
    
    Args:
        roi_path (str): Path to the input CSV file with ROI tracking data.
        md_path (str): Path to the JSON metadata file.
        output_roi_path (str): Path where the output CSV file will be saved.
    """
    if os.path.exists(output_roi_path):
        print(f"Unified roi dataset found (file =  {output_roi_path}): skipping...")
        return

    print(f"--- Starting ROI Unification Process ---")
    # 1. Load all data from input files.
    # This helper function handles file existence and parsing errors.
    metadata, tracking_df = _load_and_validate_inputs(md_path, roi_path)

    # 2. Identify objects to process from metadata and initialize a results dictionary.
    object_names = get_obj_name(metadata['obj_to_track'])
    unified_roi_results = {}

    # 3. Process each object sequentially.
    for name in object_names:
        print(f"Processing '{name}'...")
        # Assumes parse_rois_from_dataframe extracts the relevant column as a Series.
        current_tracked_roi = parse_rois_from_dataframe(tracking_df, name)
        
        # Unify the ROI sizes for the current object.
        updated_tracked_roi = unify_roi_size(current_tracked_roi)
        print(f"✅ Finished processing for '{name}'.")

        # Store the resulting Series in the dictionary.
        unified_roi_results[name] = updated_tracked_roi

    # 4. Consolidate all processed Series into a single new DataFrame.
    # The keys of the dictionary become the column headers.
    print("\nConsolidating results into a final DataFrame...")
    result_df = pd.DataFrame(unified_roi_results)
    
    # 5. Save the resulting DataFrame to the specified output path.
    # The `index=False` argument prevents pandas from writing row indices to the CSV.
    try:
        result_df.to_csv(output_roi_path, index=False)
        print(f"✅ Successfully saved unified ROIs to '{output_roi_path}'")
    except IOError as e:
        print(f"❌ Error saving file: {e}")
        
    print(f"--- Processing Complete ---")