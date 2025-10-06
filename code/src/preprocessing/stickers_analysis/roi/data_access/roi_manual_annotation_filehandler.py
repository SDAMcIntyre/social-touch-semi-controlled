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
            data_dict = asdict(data)
            if data_dict is None:
                raise
            # Reconstruct the nested ROI dictionary format for JSON serialization.
            for obj_data in data_dict.get("objects_to_track", {}).values():
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
            
            with open(filepath, 'w') as f:
                json.dump(data_dict, f, indent=4, default=str)
            logger.info("Successfully saved annotations to '%s'.", filepath)
        except IOError as e:
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
                rois_list = []
                # In the JSON, `rois` is a dictionary where keys are frame_ids.
                # We iterate through its items.
                for frame_id_str, roi_data in obj_data.get("rois", {}).items():
                    try:
                        rois_list.append({
                            'frame_id': int(frame_id_str), # Key is the frame_id
                            'roi_x': roi_data['x'],
                            'roi_y': roi_data['y'],
                            'roi_width': roi_data['w'],
                            'roi_height': roi_data['h']
                        })
                    except (ValueError, KeyError) as e:
                        logger.warning("Skipping malformed ROI entry for object '%s' with frame_id '%s': %s", name, frame_id_str, e)

                # Create DataFrame from the list, ensuring columns exist even if empty.
                if rois_list:
                    rois_df = pd.DataFrame(rois_list)
                else:
                    rois_df = pd.DataFrame(columns=['frame_id', 'roi_x', 'roi_y', 'roi_width', 'roi_height'])

                status = ROIProcessingStatus(obj_data.get("status", "to be processed"))
                objects[name] = TrackedObject(status=status.value, rois=rois_df)
            
            logger.info("Successfully loaded annotations from '%s'.", filepath)
            return AnnotationData(objects_to_track=objects)

        except (json.JSONDecodeError, IOError) as e:
            logger.error("Critical error loading or parsing '%s': %s. Returning None.", filepath, e)
            return None


# ======================= Example Usage =======================

if __name__ == '__main__':
    from ..models.roi_manual_annotation import ROIAnnotationManager

    FILEPATH = "annotations_refactored.json"
    annotation_data = ROIAnnotationFileHandler.load(FILEPATH)

    # 💡 MODIFIED: If loading fails (returns None), start with a new, empty data object.
    if annotation_data is None:
        print(f"Could not load from '{FILEPATH}'. Starting with a new annotation session.")
        annotation_data = AnnotationData()
        
    manager = ROIAnnotationManager(annotation_data)

    try:
        if "car_01" not in manager.get_object_names():
            manager.add_object("car_01")
        
        # Call set_roi with individual integer arguments.
        manager.set_roi("car_01", frame_id=100, x=10, y=20, width=30, height=40)
        manager.set_roi("car_01", frame_id=101, x=12, y=22, width=30, height=40)
        
        manager.update_status("car_01", ROIProcessingStatus.COMPLETED)
        
        car_obj = manager.get_object("car_01")
        if car_obj:
            print(f"\nDetails for car_01:")
            print(f"  Status: {car_obj.status.name}")
            print(f"  ROI at frame 100: {manager.get_roi('car_01', 100)}")

        manager.remove_roi("car_01", 101)
        print(f"\ncar_01 ROIs after removing frame 101:")
        print(car_obj.rois)

    except (ValueError, KeyError) as e:
        print(f"\nAn error occurred: {e}")

    ROIAnnotationFileHandler.save(FILEPATH, manager.data)
    print(f"\nOperations complete. Data saved to {FILEPATH}")