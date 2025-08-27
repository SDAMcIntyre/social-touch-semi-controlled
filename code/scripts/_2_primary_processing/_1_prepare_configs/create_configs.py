# generate_configs_recursively.py
import yaml
from pathlib import Path
import re
import os
import tkinter as tk
from tkinter import filedialog

from config_manager import load_and_resolve_config
import utils.path_tools as path_tools

def parse_path_from_template(path_str: str, template: str) -> dict:
    """Parses a relative path string using a template to extract named components."""
    path_str_posix = path_str.replace(os.path.sep, '/')
    regex_pattern = re.sub(r'\{(\w+)\}', r'(?P<\1>[^/]+)', template)
    match = re.fullmatch(regex_pattern, path_str_posix)
    if not match:
        raise ValueError(f"Path '{path_str}' does not match template '{template}'")
    return match.groupdict()

def generate_yaml_config(source_video_abs_path: Path, project_config: dict, project_data_root: Path, project_code_root: Path):
    """
    Generates a single YAML configuration file for a given video.
    
    Args:
        source_video_abs_path (Path): The absolute path to the source video.
        project_config (dict): The loaded project configuration dictionary.
        project_data_root (Path): The absolute path to the data project's root directory.
        project_code_root (Path): The absolute path to the code project's root directory.
    """
    # Paths are made relative to the DATA root for template matching
    source_video_rel_path = source_video_abs_path.relative_to(project_data_root)
    source_template = project_config['block_path_templates']['source_video']
    extracted_parts = parse_path_from_template(str(source_video_rel_path), source_template)

    video_filename = extracted_parts['video_filename']
    stimulus_filename = video_filename.replace('_kinect.mkv', '_stimuli.csv')
    context = {
        **project_config['path_roots'],
        **extracted_parts,
        'stimulus_filename': stimulus_filename
    }
    templates = project_config['block_path_templates']
    stimulus_path_str = templates['stimulus_metadata'].format(**context)
    hand_models_dir_str = templates['hand_models_dir'].format(**context)
    video_primary_output_dir_str = templates['video_primary_output_dir'].format(**context)
    video_processed_output_dir_str = templates['video_processed_output_dir'].format(**context)
    config_data = {
        'source_video': source_video_rel_path.as_posix(),
        'stimulus_metadata': Path(stimulus_path_str).as_posix(),
        'hand_models_dir': Path(hand_models_dir_str).as_posix(),
        'video_primary_output_dir': Path(video_primary_output_dir_str).as_posix(),
        'video_processed_output_dir': Path(video_processed_output_dir_str).as_posix(),
    }

    output_yaml_filename = f"kinect_config_{Path(video_filename).stem.replace('_kinect', '')}.yaml"
    
    # 1. Define the target directory using the CODE root
    config_output_dir = project_code_root / 'configs' / 'kinect_configs'
    
    # 2. Ensure the directory exists
    config_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Construct the final path within the code project
    output_yaml_path = config_output_dir / output_yaml_filename
    # --- END MODIFIED SECTION ---

    with open(output_yaml_path, 'w') as f:
        yaml.dump(config_data, f, sort_keys=False, default_flow_style=False)

    print(f"✅ Generated config in code project's 'configs/': {output_yaml_filename}")


def get_project_data_root():
    """
    Determines the project data root directory.

    First, it tries to find the directory automatically. If that fails or the
    directory doesn't exist, it opens a GUI dialog for the user to select it.

    Returns:
        Path: The path to the project_data_root directory.
        None: If the user cancels the directory selection.
    """
    try:
        # Attempt to find the path automatically as before
        base_path = Path(path_tools.get_database_path())
        project_data_root = base_path / "semi-controlled"
        if project_data_root.is_dir():
            print(f"✅ Project DATA root automatically identified at: {project_data_root.resolve()}")
            return project_data_root
    except FileNotFoundError:
        # This case is hit if path_tools.get_database_path() fails. We'll pass
        # and let the GUI handler take over.
        pass

    # If automatic detection fails or the directory doesn't exist, prompt the user
    print("⚠️ Project DATA root not found automatically.")
    print("Please select your 'semi-controlled' data folder using the dialog window.")

    # Set up the Tkinter root window and hide it
    root = tk.Tk()
    root.withdraw()

    # Open the directory selection dialog
    selected_path = filedialog.askdirectory(
        title="Please Select the Project Data Folder"
    )

    if not selected_path:  # Handles the case where the user closes the dialog
        print("❌ No folder selected. Exiting program.")
        return None

    project_data_root = Path(selected_path)
    print(f"👍 Project DATA root set by user to: {project_data_root.resolve()}")
    return project_data_root


def main():
    """Main function to find all videos and generate configuration for each."""
    project_code_root = Path(__file__).resolve().parents[3]
    print(f"Project CODE root identified at: {project_code_root}")

    # 2. Determine Project DATA Root using the new helper function
    project_data_root = get_project_data_root()
    if not project_data_root:
        return  # Exit if no folder was selected

    # 3. Load Project Config from the determined data root
    try:
        project_config_path = project_data_root / "project_config.yaml"
        project_config = load_and_resolve_config(project_config_path)
    except FileNotFoundError:
        print(f"❌ Error: 'project_config.yaml' not found in the selected directory: {project_data_root.resolve()}")
        return

    # 4. Determine the Directory to Search (in the data project)
    search_dir = project_data_root / project_config['path_roots']['raw_root_kinect']
    if not search_dir.is_dir():
        print(f"❌ Error: Search directory not found at '{search_dir.resolve()}'")
        return

    # 5. Recursively Find and Process All Matching Videos
    print(f"\nScanning for videos in: {search_dir.resolve()}...")
    videos_found = list(search_dir.rglob('*_kinect.mkv'))

    if not videos_found:
        print("No videos found matching the pattern '*_kinect.mkv'.")
        return

    print(f"Found {len(videos_found)} video(s) to process.\n")

    success_count = 0
    fail_count = 0

    for video_path in videos_found:
        try:
            # Pass BOTH root paths to the generator function
            generate_yaml_config(video_path, project_config, project_data_root, project_code_root)
            success_count += 1
        except (ValueError, FileNotFoundError) as e:
            print(f"❗️ Failed to process {video_path.name}: {e}")
            fail_count += 1
    
    print(f"\n--- 🚀 Processing Complete ---")
    print(f"Successfully generated: {success_count} config(s)")
    print(f"Failed: {fail_count} config(s)")


if __name__ == '__main__':
    main()