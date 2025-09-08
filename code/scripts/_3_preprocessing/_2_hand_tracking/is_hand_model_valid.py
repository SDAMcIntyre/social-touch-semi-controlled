import json
from pathlib import Path
from typing import List, Tuple



def is_hand_model_valid(
    metadata_path: Path,
    hand_models_dir: Path,
    expected_labels: List[str]
) -> Tuple[bool, List[str]]:
    """
    Validates a metadata JSON file against a set of rules.

    This function checks for the existence and correctness of the hand model file,
    the hand orientation, and the labels of the selected points.

    Args:
        metadata_path (Path):
            The path to the metadata JSON file to be validated.
        hand_models_dir (Path):
            The directory where the hand model .txt files are stored.
        expected_labels (List[str]):
            A list of strings representing the valid labels for selected points.

    Returns:
        Tuple[bool, List[str]]:
            A tuple containing:
            - A boolean indicating if the metadata is valid (True) or not (False).
            - A list of error messages. If the metadata is valid, the list is empty.
    """
    errors = []
    
    # 1. Check if metadata file exists and is valid JSON
    if not metadata_path.is_file():
        return False, [f"Metadata file not found at: {metadata_path}"]
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except json.JSONDecodeError:
        return False, [f"Failed to decode JSON from file: {metadata_path}"]
    except Exception as e:
        return False, [f"An unexpected error occurred while reading the file: {e}"]

    # 2. Validate the hand model file name
    model_name = metadata.get("selected_hand_model_name")
    if not model_name or not isinstance(model_name, str):
        errors.append("'selected_hand_model_name' key is missing or invalid.")
    else:
        expected_model_path = hand_models_dir / model_name
        if not expected_model_path.is_file():
            errors.append(
                f"Hand model file '{model_name}' does not exist in '{hand_models_dir}'."
            )

    # 3. Validate hand orientation
    orientation = metadata.get("hand_orientation")
    if orientation not in ["left", "right"]:
        errors.append(
            f"Invalid 'hand_orientation' value: '{orientation}'. Must be 'left' or 'right'."
        )

    # 4. Validate selected point labels
    selected_points = metadata.get("selected_points")
    if not isinstance(selected_points, list):
        errors.append("'selected_points' should be a list.")
    else:
        expected_labels_set = set(expected_labels)
        for i, point_data in enumerate(selected_points):
            if not isinstance(point_data, dict):
                errors.append(f"Item at index {i} in 'selected_points' is not a dictionary.")
                continue

            label = point_data.get("label")
            if not label:
                errors.append(f"Item at index {i} in 'selected_points' is missing a 'label'.")
            elif label not in expected_labels_set:
                errors.append(f"Invalid point label '{label}' found. It is not in the list of expected labels.")
    
    is_valid = not errors
    return is_valid, errors