import json
import os
import logging
from dataclasses import asdict
from typing import Optional

# Import pandas
import pandas as pd

from ..models.roi_manual_annotation import (
    AnnotationData, 
    ROIProcessingStatus,
    TrackedObject
)

# Set up a basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ROIAnnotationFileHandler:
    """Handles loading from and saving AnnotationData to a JSON file."""

    @staticmethod
    def save(filepath: str, data: AnnotationData) -> None:
        """Saves the AnnotationData object to a JSON file."""
        try:
            # dataclasses.asdict converts the dataclass hierarchy to dicts, 
            # but leaves complex objects like pd.DataFrame instances intact.
            data_dict = asdict(data)
            if data_dict is None:
                raise ValueError("Failed to convert AnnotationData to dictionary.")
            
            # Correct iteration for modification
            objects_dict = data_dict.get("objects_to_track", {})
            for obj_name, obj_data in objects_dict.items():
                
                # 1. Handle ROIs (DataFrame -> Dict)
                if "rois" in obj_data and isinstance(obj_data["rois"], pd.DataFrame):
                    df = obj_data["rois"]
                    rois_dict = {}
                    for _, row in df.iterrows():
                        frame_id = str(int(row['frame_id']))
                        rois_dict[frame_id] = {
                            'x': int(row['roi_x']),
                            'y': int(row['roi_y']),
                            'w': int(row['roi_width']),
                            'h': int(row['roi_height'])
                        }
                    obj_data["rois"] = rois_dict

                # 2. Handle Ignore Starts (DataFrame -> List[int])
                if "ignore_starts" in obj_data and isinstance(obj_data["ignore_starts"], pd.DataFrame):
                    if not obj_data["ignore_starts"].empty:
                        obj_data["ignore_starts"] = obj_data["ignore_starts"]["frame_id"].astype(int).tolist()
                    else:
                        obj_data["ignore_starts"] = []

                # 3. Handle Ignore Stops (DataFrame -> List[int])
                if "ignore_stops" in obj_data and isinstance(obj_data["ignore_stops"], pd.DataFrame):
                    if not obj_data["ignore_stops"].empty:
                        obj_data["ignore_stops"] = obj_data["ignore_stops"]["frame_id"].astype(int).tolist()
                    else:
                        obj_data["ignore_stops"] = []

            with open(filepath, 'w') as f:
                json.dump(data_dict, f, indent=4, default=str)
            logger.info("Successfully saved annotations to '%s'.", filepath)
            
        except (IOError, ValueError, TypeError) as e:
            logger.error("Error saving to '%s': %s", filepath, e)
            raise

    @staticmethod
    def load(filepath: str) -> Optional[AnnotationData]:
        """
        Loads and constructs an AnnotationData object from a JSON file.
        Returns None if the file doesn't exist, is empty, or contains invalid JSON.
        """
        if not os.path.exists(filepath):
            logger.info("Annotation file not found: '%s'. Returning None.", filepath)
            return None

        try:
            with open(filepath, 'r') as f:
                content = f.read()
                if not content.strip():
                    logger.info("Annotation file is empty: '%s'. Returning None.", filepath)
                    return None
                raw_data = json.loads(content)

            if raw_data is None:
                logger.info("Annotation file contains null content: '%s'. Returning None.", filepath)
                return None
            
            objects = {}
            for name, obj_data in raw_data.get("objects_to_track", {}).items():
                # --- 1. Reconstruct ROIs DataFrame ---
                rois_list = []
                for frame_id_str, roi_data in obj_data.get("rois", {}).items():
                    try:
                        rois_list.append({
                            'frame_id': int(frame_id_str),
                            'roi_x': roi_data['x'],
                            'roi_y': roi_data['y'],
                            'roi_width': roi_data['w'],
                            'roi_height': roi_data['h']
                        })
                    except (ValueError, KeyError) as e:
                        logger.warning("Skipping malformed ROI entry for object '%s' frame '%s': %s", name, frame_id_str, e)

                if rois_list:
                    rois_df = pd.DataFrame(rois_list)
                else:
                    rois_df = pd.DataFrame(columns=['frame_id', 'roi_x', 'roi_y', 'roi_width', 'roi_height'])

                # --- 2. Reconstruct Ignore Starts DataFrame ---
                ignore_starts_list = obj_data.get("ignore_starts", [])
                if ignore_starts_list:
                    # Ensure all items are ints
                    ignore_starts_df = pd.DataFrame({'frame_id': [int(x) for x in ignore_starts_list]})
                else:
                    ignore_starts_df = pd.DataFrame(columns=['frame_id'])

                # --- 3. Reconstruct Ignore Stops DataFrame ---
                ignore_stops_list = obj_data.get("ignore_stops", [])
                if ignore_stops_list:
                    # Ensure all items are ints
                    ignore_stops_df = pd.DataFrame({'frame_id': [int(x) for x in ignore_stops_list]})
                else:
                    ignore_stops_df = pd.DataFrame(columns=['frame_id'])

                # --- 4. Construct TrackedObject ---
                status = ROIProcessingStatus(obj_data.get("status", "to be processed"))
                
                objects[name] = TrackedObject(
                    status=status.value, 
                    rois=rois_df,
                    ignore_starts=ignore_starts_df,
                    ignore_stops=ignore_stops_df
                )
            
            logger.info("Successfully loaded annotations from '%s'.", filepath)
            return AnnotationData(objects_to_track=objects)

        except (json.JSONDecodeError, IOError, ValueError) as e:
            logger.error("Critical error loading or parsing '%s': %s. Returning None.", filepath, e)
            return None


# ======================= Example Usage =======================

if __name__ == '__main__':
    from ..models.roi_manual_annotation import ROIAnnotationManager

    FILEPATH = "annotations_v2.json"
    annotation_data = ROIAnnotationFileHandler.load(FILEPATH)

    # If loading fails (returns None), start with a new, empty data object.
    if annotation_data is None:
        print(f"Could not load from '{FILEPATH}'. Starting with a new annotation session.")
        annotation_data = AnnotationData()
        
    manager = ROIAnnotationManager(annotation_data)

    try:
        OBJ_NAME = "car_01"
        if OBJ_NAME not in manager.get_object_names():
            manager.add_object(OBJ_NAME)
        
        # 1. Set ROIs
        manager.set_roi(OBJ_NAME, frame_id=100, x=10, y=20, width=30, height=40)
        manager.set_roi(OBJ_NAME, frame_id=101, x=12, y=22, width=30, height=40)
        
        # 2. Set Ignore Regions (New Functionality Test)
        manager.set_ignore_start(OBJ_NAME, frame_id=50)
        manager.set_ignore_stop(OBJ_NAME, frame_id=60)
        
        manager.update_status(OBJ_NAME, ROIProcessingStatus.COMPLETED)
        
        car_obj = manager.get_object(OBJ_NAME)
        if car_obj:
            print(f"\nDetails for {OBJ_NAME}:")
            print(f"  Status: {car_obj.status}")
            print(f"  ROI at frame 100: {manager.get_roi(OBJ_NAME, 100)}")
            print(f"  Ignore Starts: {car_obj.ignore_starts['frame_id'].tolist()}")
            print(f"  Ignore Stops: {car_obj.ignore_stops['frame_id'].tolist()}")

        # Save to disk
        ROIAnnotationFileHandler.save(FILEPATH, manager.data)
        print(f"\nOperations complete. Data saved to {FILEPATH}")
        
        # Reload to verify serialization
        reloaded_data = ROIAnnotationFileHandler.load(FILEPATH)
        if reloaded_data:
            reloaded_obj = reloaded_data.objects_to_track[OBJ_NAME]
            print("\n--- Verification after reload ---")
            print(f"  Ignore Starts: {reloaded_obj.ignore_starts['frame_id'].tolist()}")
            print(f"  Ignore Stops: {reloaded_obj.ignore_stops['frame_id'].tolist()}")

    except (ValueError, KeyError) as e:
        print(f"\nAn error occurred: {e}")