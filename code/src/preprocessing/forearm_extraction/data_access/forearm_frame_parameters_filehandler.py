import json
from dataclasses import asdict
from pathlib import Path
from typing import Union, List

from ..models.forearm_parameters import (
    ForearmParameters,
    RegionOfInterest,
    Point
)

class ForearmFrameParametersFileHandler:
    """Handles the serialization and deserialization of a list of ForearmParameters."""

    @staticmethod
    def save(parameters: List[ForearmParameters], file_path: Union[str, Path]) -> None:
        """
        Saves a list of ForearmParameters objects to a single JSON file.

        Args:
            parameters (List[ForearmParameters]): The list of data objects to save.
            file_path (Union[str, Path]): The path to the output JSON file.
        """
        print(f"\n💾 Saving parameters for {len(parameters)} frames to '{file_path}'...")
        try:
            # Use a list comprehension to convert each dataclass instance to a dict
            data_to_save = [asdict(p) for p in parameters]
            with open(file_path, 'w') as f:
                json.dump(data_to_save, f, indent=4)
            print("🎉 Successfully saved parameters.")
        except IOError as e:
            print(f"❌ Error: Could not write to file '{file_path}'.\n{e}")

    @staticmethod
    def load(file_path: Union[str, Path]) -> List[ForearmParameters] | None:
        """
        Loads a list of forearm parameters from a JSON file.

        Args:
            file_path (Union[str, Path]): The path to the input JSON file.

        Returns:
            List[ForearmParameters] | None: A list of loaded data objects, or None if an error occurs.
        """
        print(f"\n📂 Loading parameters from '{file_path}'...")
        if not ForearmFrameParametersFileHandler.is_valid_structure(file_path):
            print(f"❌ Error: File '{file_path}' has an invalid or corrupted structure.")
            return None
        try:
            with open(file_path, 'r') as f:
                data_list = json.load(f)

            loaded_parameters = []
            for data in data_list:
                # Reconstruct the nested dataclasses from each dictionary in the list
                roi_data = data["region_of_interest"]
                roi = RegionOfInterest(
                    top_left_corner=Point(**roi_data["top_left_corner"]),
                    bottom_right_corner=Point(**roi_data["bottom_right_corner"])
                )
                
                # Remove the processed ROI dict to unpack the rest of the keys
                del data["region_of_interest"]
                
                parameter = ForearmParameters(region_of_interest=roi, **data)
                loaded_parameters.append(parameter)
            
            print(f"✅ Successfully loaded parameters for {len(loaded_parameters)} frames.")
            return loaded_parameters
        except (IOError, KeyError, TypeError, json.JSONDecodeError) as e:
            print(f"❌ Error: Could not read or process file '{file_path}'.\n{e}")
            return None

    @staticmethod
    def is_valid_structure(file_path: Union[str, Path]) -> bool:
        """
        Checks if a JSON file has the valid structure for a list of ForearmParameters.

        This method verifies that the file contains a JSON array, and that each object
        in the array contains the necessary keys and nested structures.

        Args:
            file_path (Union[str, Path]): The path to the JSON file to validate.

        Returns:
            bool: True if the file has a valid structure, False otherwise.
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # The root object must be a list
            if not isinstance(data, list):
                return False
            
            # An empty list is considered a valid file, no need to check elements
            if not data:
                return True

            # Iterate through and check the structure of EVERY element
            for item in data:
                # Check for all required keys at every level.
                # A KeyError or TypeError will be raised if any key is missing or is of the wrong type.
                _ = item["video_filename"]
                _ = item["frame_id"]
                _ = item["frame_width"]
                _ = item["frame_height"]
                _ = item["fps"]
                _ = item["nframes"]
                _ = item["fourcc_str"]
                
                roi_data = item["region_of_interest"]
                top_left = roi_data["top_left_corner"]
                bottom_right = roi_data["bottom_right_corner"]
                
                _ = top_left["x"]
                _ = top_left["y"]
                _ = bottom_right["x"]
                _ = bottom_right["y"]

            # If the loop completes without raising an exception, the structure is valid.
            return True
        except (FileNotFoundError, IOError, json.JSONDecodeError, KeyError, TypeError, IndexError):
            # Any of these exceptions indicate an invalid file or structure.
            return False