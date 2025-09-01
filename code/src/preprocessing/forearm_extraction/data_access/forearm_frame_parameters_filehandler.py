import json
from dataclasses import asdict
from pathlib import Path
from typing import Union

# Assuming these dataclasses are defined in ..models.forearm_parameters
# For demonstration, I'll define them here.
from dataclasses import dataclass

@dataclass
class Point:
    x: int
    y: int

@dataclass
class RegionOfInterest:
    top_left_corner: Point
    bottom_right_corner: Point

@dataclass
class VideoMetadata:
    reference_frame_idx: int
    region_of_interest: RegionOfInterest
    frame_width: int
    frame_height: int
    fps: float
    nframes: int
    fourcc_str: str


class ForearmFrameParametersFileHandler:
    """Handles the serialization, deserialization, and validation of VideoMetadata."""

    @staticmethod
    def save(metadata: VideoMetadata, file_path: Union[str, Path]) -> None:
        """
        Saves a VideoMetadata object to a JSON file.

        Args:
            metadata (VideoMetadata): The data object to save.
            file_path (Union[str, Path]): The path to the output JSON file.
        """
        print(f"\n💾 Saving metadata to '{file_path}'...")
        try:
            # dataclasses.asdict recursively converts the object to a dictionary
            with open(file_path, 'w') as f:
                json.dump(asdict(metadata), f, indent=4)
            print(f"🎉 Successfully saved metadata.")
        except IOError as e:
            print(f"❌ Error: Could not write to file '{file_path}'.\n{e}")

    @staticmethod
    def load(file_path: Union[str, Path]) -> VideoMetadata | None:
        """
        Loads video metadata from a JSON file into a VideoMetadata object.

        Args:
            file_path (Union[str, Path]): The path to the input JSON file.

        Returns:
            VideoMetadata | None: The loaded data object, or None if an error occurs.
        """
        print(f"\n📂 Loading metadata from '{file_path}'...")
        if not ForearmFrameParametersFileHandler.is_valid_structure(file_path):
             print(f"❌ Error: File '{file_path}' has an invalid or corrupted structure.")
             return None
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Reconstruct the nested dataclasses from the loaded dictionary
            roi_data = data["region_of_interest"]
            roi = RegionOfInterest(
                top_left_corner=Point(**roi_data["top_left_corner"]),
                bottom_right_corner=Point(**roi_data["bottom_right_corner"])
            )
            
            # Remove the processed ROI dict to unpack the rest
            del data["region_of_interest"]

            metadata = VideoMetadata(region_of_interest=roi, **data)
            
            print("✅ Successfully loaded metadata.")
            return metadata
        except (IOError, KeyError) as e:
            # Errors like IOError or KeyError should be caught by is_valid_structure,
            # but we keep this as a fallback.
            print(f"❌ Error: Could not read or process file '{file_path}'.\n{e}")
            return None

    @staticmethod
    def is_valid_structure(file_path: Union[str, Path]) -> bool:
        """
        Checks if a JSON file has the valid data structure for VideoMetadata.

        This method verifies that the file exists, is readable, is valid JSON,
        and contains all the necessary keys and nested structures to be
        successfully loaded as a VideoMetadata object.

        Args:
            file_path (Union[str, Path]): The path to the JSON file to validate.

        Returns:
            bool: True if the file has a valid structure, False otherwise.
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Check for all required keys at every level.
            # A KeyError will be raised if any key is missing.
            _ = data["reference_frame_idx"]
            _ = data["frame_width"]
            _ = data["frame_height"]
            _ = data["fps"]
            _ = data["nframes"]
            _ = data["fourcc_str"]
            
            roi_data = data["region_of_interest"]
            top_left = roi_data["top_left_corner"]
            bottom_right = roi_data["bottom_right_corner"]
            
            _ = top_left["x"]
            _ = top_left["y"]
            _ = bottom_right["x"]
            _ = bottom_right["y"]

            # An additional check could be for types, but for structure,
            # key checking is usually sufficient.
            return True
        except (FileNotFoundError, IOError, json.JSONDecodeError, KeyError, TypeError):
            # Any of these exceptions indicate an invalid file or structure.
            return False

# ----------------------------------------------------------------------------
# 3. Example Usage
# This demonstrates how to use the new validation method.
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Assume these values come from your application logic ---
    ref_frame_idx = 150
    tracking_data = {"x": 120, "y": 80, "width": 250, "height": 200}
    video_manager_mock = {
        "width": 1920,
        "height": 1080,
        "fps": 29.97,
        "nframes": 1800,
        "fourcc_str": "avc1"
    }
    metadata_path = Path("video_metadata.json")
    invalid_metadata_path = Path("invalid_metadata.json")
    non_existent_path = Path("non_existent_file.json")

    # --- Create and save a valid data object ---
    x, y, w, h = tracking_data["x"], tracking_data["y"], tracking_data["width"], tracking_data["height"]
    metadata_to_save = VideoMetadata(
        reference_frame_idx=ref_frame_idx,
        region_of_interest=RegionOfInterest(
            top_left_corner=Point(x=x, y=y),
            bottom_right_corner=Point(x=x + w, y=y + h)
        ),
        frame_width=video_manager_mock["width"],
        frame_height=video_manager_mock["height"],
        fps=video_manager_mock["fps"],
        nframes=video_manager_mock["nframes"],
        fourcc_str=video_manager_mock["fourcc_str"]
    )
    ForearmFrameParametersFileHandler.save(metadata_to_save, metadata_path)
    
    # --- Create and save an invalid data object (missing a key) ---
    with open(invalid_metadata_path, 'w') as f:
        json.dump({"reference_frame_idx": 100, "fps": 30.0}, f, indent=4)


    # --- Use the new validation method ---
    print("\n🧐 Performing structure validation checks...")
    print(f"'{metadata_path}' is valid: {ForearmFrameParametersFileHandler.is_valid_structure(metadata_path)}")
    print(f"'{invalid_metadata_path}' is valid: {ForearmFrameParametersFileHandler.is_valid_structure(invalid_metadata_path)}")
    print(f"'{non_existent_path}' is valid: {ForearmFrameParametersFileHandler.is_valid_structure(non_existent_path)}")

    # --- Attempt to load the valid data back ---
    loaded_metadata = ForearmFrameParametersFileHandler.load(metadata_path)
    if loaded_metadata:
        print("\n🔍 Verifying loaded data:")
        print(loaded_metadata)
        roi_width = loaded_metadata.region_of_interest.bottom_right_corner.x - \
                    loaded_metadata.region_of_interest.top_left_corner.x
        print(f"\nCalculated ROI width from loaded data: {roi_width}")

    # --- Attempt to load the invalid data ---
    ForearmFrameParametersFileHandler.load(invalid_metadata_path)