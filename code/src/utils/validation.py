import os

def create_success_flag_file(output_path: str) -> None:
    """Creates an empty .SUCCESS suffixed file to signal successful completion."""
    try:
        success_filepath = success_flag_file(output_path)
        with open(success_filepath, 'w'):
            pass # Create empty file
        print(f"✅ All objects tracked successfully. Created success flag at '{success_filepath}'")
    except IOError as e:
        print(f"⚠️ Could not create success flag file. Reason: {e}")

def success_flag_file(output_path: str):
    output_dir = os.path.dirname(output_path)
    output_basename = os.path.basename(output_path)
    success_filepath = os.path.join(output_dir, f"{output_basename}.SUCCESS")
    return success_filepath


def is_success(output_path: str):
    return os.path.exists(success_flag_file(output_path))
