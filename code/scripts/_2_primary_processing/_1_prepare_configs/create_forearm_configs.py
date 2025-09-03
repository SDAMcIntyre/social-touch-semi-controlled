import yaml
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

import utils.path_tools as path_tools
from primary_processing import (
    ForearmConfigFileHandler,
    ForearmConfig,
    KinectConfigFileHandler,
    KinectConfig
)

# --- 2. Core Logic to Group and Create Session Files ---

def create_session_configs(source_dir: Path, output_dir: Path, database_path: Path):
    """
    Scans a directory for YAML files, groups them by 'session_id',
    and creates a new summary config file for each session, including
    the session's primary and processed output paths.

    Args:
        source_dir (Path): The directory containing the original YAML config files.
        output_dir (Path): The directory where new session config files will be saved.
    """
    if not source_dir.is_dir():
        print(f"⚠️ Source directory not found: {source_dir}. Aborting.")
        return

    # Use defaultdict to group file paths and session metadata.
    # The structure will be: {session_id: {"file_paths": [...], "primary_path": Path, "processed_path": Path}}
    sessions: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"file_paths": [], "primary_path": None, "processed_path": None})
    
    print(f"\n🔍 Scanning for YAML files in '{source_dir}'...")
    
    source_files = list(source_dir.glob("*.yaml"))
    if not source_files:
        print("   -> No YAML files found.")
        return
        
    print(f"   -> Found {len(source_files)} files. Parsing for 'session_id'...")

    for config_file in source_files:
        try:
            data_config = KinectConfigFileHandler.load_and_resolve_config(config_file)
            kinect_config = KinectConfig(data_config, database_path)
            
            session_id = kinect_config.session_id
            current_primary_path = kinect_config.session_primary_output_dir
            current_processed_path = kinect_config.session_processed_output_dir
            
            if session_id:
                session_group = sessions[session_id]
                
                # Store the absolute path for an unambiguous link
                session_group["file_paths"].append(config_file.resolve())

                # --- MODIFICATION START ---
                # Check for path consistency and store the primary session path
                if session_group["primary_path"] is None:
                    session_group["primary_path"] = current_primary_path
                elif session_group["primary_path"] != current_primary_path:
                    print(f"   -> ⚠️ INCONSISTENCY ERROR for session '{session_id}' (primary path):")
                    print(f"        - Path from other files: {session_group['primary_path']}")
                    print(f"        - Path in '{config_file.name}': {current_primary_path}")
                    print("        -> Please resolve this conflict in your Kinect config files.")

                # Check for path consistency and store the processed session path
                if session_group["processed_path"] is None:
                    session_group["processed_path"] = current_processed_path
                elif session_group["processed_path"] != current_processed_path:
                    print(f"   -> ⚠️ INCONSISTENCY ERROR for session '{session_id}' (processed path):")
                    print(f"        - Path from other files: {session_group['processed_path']}")
                    print(f"        - Path in '{config_file.name}': {current_processed_path}")
                    print("        -> Please resolve this conflict in your Kinect config files.")
                # --- MODIFICATION END ---
                    
            else:
                print(f"   -> ⚠️ Skipping '{config_file.name}': 'session_id' key not found.")
        except yaml.YAMLError as e:
            print(f"   -> ❌ Error parsing '{config_file.name}': {e}. Skipping.")
        except Exception as e:
            print(f"   -> ❌ An unexpected error occurred with '{config_file.name}': {e}. Skipping.")

    if not sessions:
        print("\nNo sessions could be grouped. Exiting.")
        return
        
    print(f"\n✅ Found {len(sessions)} unique session(s). Generating new config files in '{output_dir}'...")

    for session_id, session_info in sessions.items():
        file_paths = session_info["file_paths"]
        primary_path = session_info["primary_path"]
        # --- MODIFICATION START ---
        processed_path = session_info["processed_path"]
        # --- MODIFICATION END ---
        
        # Sort paths for consistent output
        file_paths.sort()
        
        if primary_path is None or processed_path is None:
            print(f"   -> ⚠️ Skipping session '{session_id}' due to a missing primary or processed path.")
            continue

        # Create the session data object using our Pydantic model, now including both paths.
        # Note: Ensure the ForearmConfig Pydantic model itself has been updated to accept these fields.
        session_data = ForearmConfig(
            session_id=session_id,
            session_primary_path=primary_path,
            session_processed_path=processed_path,
            config_file_links=file_paths
        )
        
        # Define a clean output filename
        output_filename = f"session_{session_id}.yaml"
        output_path = output_dir / output_filename
        
        # Use our handler to save the file
        ForearmConfigFileHandler.save(session_data, output_path)

    print("\n🎉 All session configuration files have been created successfully.")


# --- 4. Main Execution Block ---

def main():
    """Main function to find all videos and generate configuration for each."""
    # Using a relative path from this script's location for robustness
    project_root = Path(__file__).resolve().parents[4]
    print(f"Project root identified at: {project_root}")

    configs_dir = project_root / "configs"
    SOURCE_CONFIGS_DIR = configs_dir / "kinect_configs"
    OUTPUT_SESSIONS_DIR = configs_dir / "forearm_configs"

    # Ensure the output directory exists
    OUTPUT_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    project_data_root = path_tools.get_project_data_root()

    create_session_configs(source_dir=SOURCE_CONFIGS_DIR, output_dir=OUTPUT_SESSIONS_DIR, database_path=project_data_root)

if __name__ == '__main__':
    main()