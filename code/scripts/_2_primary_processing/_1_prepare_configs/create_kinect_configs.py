# generate_configs_recursively.py
import yaml
from pathlib import Path
import re
import os
from pydantic import ValidationError # Still useful if catching Pydantic errors explicitly

# Preserving original imports as requested
from primary_processing import KinectConfigFileHandler, KinectConfig
import utils.path_tools as path_tools

def parse_path_from_template(path_str: str, template: str) -> dict:
    """Parses a relative path string using a template to extract named components."""
    path_str_posix = path_str.replace(os.path.sep, '/')
    # Convert template placeholders {name} to regex groups (?P<name>[^/]+)
    regex_pattern = re.sub(r'\{(\w+)\}', r'(?P<\1>[^/]+)', template)
    match = re.fullmatch(regex_pattern, path_str_posix)
    if not match:
        raise ValueError(f"Path '{path_str}' does not match template '{template}'")
    return match.groupdict()

def generate_yaml_config(source_video_abs_path: Path, project_config: dict, project_data_root: Path, project_code_root: Path):
    """
    Generates a single YAML configuration file for a given video.
    Dynamically resolves all templates found in project_config['block_path_templates'].
    """
    # Paths are made relative to the DATA root for template matching
    source_video_rel_path = source_video_abs_path.relative_to(project_data_root)
    source_template = project_config['block_path_templates']['source_video']
    
    try:
        extracted_parts = parse_path_from_template(str(source_video_rel_path), source_template)
    except ValueError as e:
        print(f"Skipping {source_video_abs_path.name}: {e}")
        return

    video_filename = extracted_parts['video_filename']
    # Heuristic: assume stimulus filename matches video filename pattern
    stimulus_filename = video_filename.replace('_kinect.mkv', '_stimuli.csv')
    
    # Build context for string formatting.
    context = {
        **project_config['path_roots'],
        **extracted_parts,
        'stimulus_filename': stimulus_filename
    }
    
    # --- CONFIG DATA CONSTRUCTION ---
    # Initialize with base metadata that is not a path template
    # NOTE: extracted_parts must contain 'block_id' based on the project configuration template
    config_data = {
        'session_id': extracted_parts['session_id'],
        'block_id': extracted_parts['block_id'], # Added mapping
        'objects_to_track': project_config['parameters']['objects_to_track'],
        # We explicitly set source_video to the actual relative path found to ensure accuracy
        'source_video': source_video_rel_path.as_posix(),
    }

    # --- DYNAMIC PATH RESOLUTION ---
    templates = project_config['block_path_templates']
    
    for key, template_str in templates.items():
        # Skip source_video as it is already handled explicitly above
        if key == 'source_video':
            continue
            
        try:
            # Resolve the path template using the context dictionary
            resolved_path_str = template_str.format(**context)
            
            # Convert to Path and back to POSIX string to ensure cross-platform consistency
            config_data[key] = Path(resolved_path_str).as_posix()
            
        except KeyError as e:
            # If a template requires a key we couldn't extract or calculate, warn and skip
            print(f"⚠️ Warning: Could not resolve template '{key}'. Missing context variable: {e}")
            config_data[key] = None

    # --- VALIDATION (Updated to use KinectConfig) ---
    # Validate against KinectConfig, which checks the schema (SessionInputs) 
    # and ensures the paths are resolvable against the data root.
    try:
        # Attempt to create the final config object, using the project_data_root as the base.
        # If this succeeds, the config_data is valid.
        KinectConfig(config_data=config_data, database_path=project_data_root)
    except ValueError as e:
        # KinectConfig raises ValueError on Pydantic validation failure
        print(f"❌ Configuration Validation Failed for {video_filename} using KinectConfig:\n{e}")
        return

    # Construct output filename
    output_yaml_filename = f"kinect_config_{Path(video_filename).stem.replace('_kinect', '')}.yaml"
    
    # 1. Define the target directory using the CODE root
    config_output_dir = project_code_root / 'configs' / 'kinect_configs'
    
    # 2. Ensure the directory exists
    config_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Construct the final path within the code project
    output_yaml_path = config_output_dir / output_yaml_filename

    with open(output_yaml_path, 'w') as f:
        yaml.dump(config_data, f, sort_keys=False, default_flow_style=False)

    print(f"✅ Generated config in code project's 'configs/': {output_yaml_filename}")


def main():
    """Main function to find all videos and generate configuration for each."""
    # 1. Determine Project CODE Root
    # Assumes this script is located 4 levels deep relative to root in the code structure
    project_code_root = Path(__file__).resolve().parents[4]
    print(f"Project CODE root identified at: {project_code_root}")

    # 2. Determine Project DATA Root using the helper function
    project_data_root = path_tools.get_project_data_root()
    if not project_data_root:
        print("Operation cancelled by user.")
        return

    # 3. Load Project Config from the determined data root
    try:
        project_config_path = project_data_root / "project_config.yaml"
        project_config = KinectConfigFileHandler.load_and_resolve_config(project_config_path)
    except FileNotFoundError:
        print(f"❌ Error: 'project_config.yaml' not found in the selected directory: {project_data_root.resolve()}")
        return

    # 4. Determine the Directory to Search (in the data project)
    try:
        raw_root_name = project_config['path_roots']['raw_root']
        search_dir = project_data_root / raw_root_name / 'kinect'
    except KeyError as e:
        print(f"❌ Error: Missing key in project_config: {e}")
        return

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
            # generate_yaml_config now contains the KinectConfig validation step
            generate_yaml_config(video_path, project_config, project_data_root, project_code_root)
            success_count += 1
        except (ValueError, FileNotFoundError, KeyError) as e:
            if isinstance(e, KeyError) and 'session_id' in str(e):
                 print(f"❗️ Failed to process {video_path.name}: 'session_id' key not found in path template. Check your 'project_config.yaml'.")
            elif isinstance(e, KeyError) and 'block_id' in str(e):
                 print(f"❗️ Failed to process {video_path.name}: 'block_id' key not found in path template. Check your 'project_config.yaml'.")
            else:
                print(f"❗️ Failed to process {video_path.name}: {e}")
            fail_count += 1

    print(f"\n--- 🚀 Processing Complete ---")
    print(f"Successfully generated: {success_count} config(s)")
    print(f"Failed: {fail_count} config(s)")

if __name__ == '__main__':
    main()