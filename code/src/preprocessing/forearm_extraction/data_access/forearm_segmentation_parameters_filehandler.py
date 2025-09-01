import json
import math

# --- Helper logic for custom JSON serialization/deserialization ---

class _CustomJSONEncoder(json.JSONEncoder):
    """
    A private custom JSON encoder to handle non-standard values like infinity.
    This is called automatically by json.dump() when provided via the 'cls' argument.
    """
    def default(self, obj):
        if obj == math.inf:
            return "Infinity"
        if obj == -math.inf:
            return "-Infinity"
        # Let the base class default method handle other types or raise an error.
        return super().default(obj)

def _json_decoder_hook(dct: dict) -> dict:
    """
    A private decoder hook to be used with json.load()'s 'object_hook'.
    It scans dictionary values and converts string representations back to python objects.
    """
    for key, value in dct.items():
        if value == "Infinity":
            dct[key] = math.inf
        elif value == "-Infinity":
            dct[key] = -math.inf
    return dct


# --- Main Class ---

class ForearmSegmentationParamsFileHandler:
    """
    A class to manage loading and saving configuration data to and from JSON files,
    with special handling for non-standard JSON values like infinity.
    """
    @staticmethod
    def save(data: dict, file_path: str) -> bool:
        """
        Saves a dictionary to a JSON file using a custom encoder.

        Args:
            data (dict): The dictionary data to save.
            file_path (str): The path to the output JSON file.
        
        Returns:
            bool: True if saving was successful, False otherwise.
        """
        print(f"Attempting to save configuration to {file_path}...")
        try:
            with open(file_path, 'w') as f:
                # Use the custom encoder class via the 'cls' argument
                json.dump(data, f, indent=4, cls=_CustomJSONEncoder)
            print(f"✅ Configuration successfully saved.")
            return True
        except (IOError, TypeError) as e:
            print(f"❌ Error saving configuration: {e}")
            return False

    @staticmethod
    def load(file_path: str) -> dict | None:
        """
        Loads a dictionary from a JSON file using a custom decoder hook.

        Args:
            file_path (str): The path to the input JSON file.

        Returns:
            dict | None: The loaded dictionary, or None if an error occurred.
        """
        print(f"Attempting to load configuration from {file_path}...")
        try:
            with open(file_path, 'r') as f:
                # Use the custom decoder hook via the 'object_hook' argument
                data = json.load(f, object_hook=_json_decoder_hook)
            print(f"✅ Configuration successfully loaded.")
            return data
        except (IOError, json.JSONDecodeError) as e:
            print(f"❌ Error loading configuration: {e}")
            return None

# --- Example Usage ---
if __name__ == "__main__":
    # 1. Your original data
    original_params = {
        'down_sampling': {'enabled': False, 'leaf_size': 5.0},
        'box_filter': {'enabled': True, 'min_z': -math.inf, 'max_z': math.inf},
        'color_skin_filter': {'enabled': 1.0, 'hsv_lower_bound': [], 'hsv_upper_bound': []},
        'region_growing': {'dbscan_eps': 10.0, 'min_cluster_size': 500.0}
    }
    
    # 2. Define the file path for the config
    config_file = "segmenter_params.json"

    # 3. Instantiate the manager
    config_manager = ConfigManager()

    # 4. Save the data to the file
    config_manager.save(original_params, config_file)
    
    print("-" * 20)

    # 5. Load the data back from the file
    loaded_params = config_manager.load(config_file)

    # 6. Verify and print the result
    if loaded_params:
        print("\nOriginal parameters:")
        print(original_params)
        print("\nLoaded parameters:")
        print(loaded_params)
        
        # This assert confirms the round-trip was successful
        assert original_params == loaded_params
        print("\n✅ Verification successful: Loaded data is identical to the original.")