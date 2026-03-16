import logging
import pickle
import tkinter as tk
from pathlib import Path
from typing import Optional
import numpy as np  # Added for vector operations

# Import from local modules
from preprocessing.motion_analysis import (
    HamerCheckupSelector,
    MeshSequenceLoader
)
from utils.should_process_task import should_process_task

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def curate_hamer_hand_models(
    video_path: Path, 
    data_path: Path, 
    output_file_path: Path, 
    output_success_path: Path, 
    csv_path: Path,
    *,
    force_processing: bool = False
):
    """
    Launches the selector tool, allows user to select frames, 
    extracts the mesh data for those frames, and saves to output_file_path.
    """
    
    # --- Idempotency Check ---
    if not should_process_task(
        output_paths=[output_file_path, output_success_path],
        input_paths=[video_path, data_path, csv_path],
        force=force_processing
    ):
        logging.info(f"Skipping curation: Output '{output_file_path}' exists and is up to date.")
        return
    
    # --- State Sanitization ---
    try:
        if output_success_path.exists():
            output_success_path.unlink()
            logging.info(f"Invalidated previous success state: Removed '{output_success_path}'")
    except OSError as e:
        logging.error(f"Failed to remove existing success path: {e}")

    # Check input files before launching GUI
    if not Path(data_path).exists() or not Path(video_path).exists():
        logging.error("Error: Input files not found.")
        return

    root = tk.Tk()
    app: Optional[HamerCheckupSelector] = None
    
    # --- GUI Execution Context ---
    # We use try/finally to ensure the window is destroyed and resources are freed
    # even if an error occurs inside the app or the data extraction phase.
    try:
        app = HamerCheckupSelector(root, video_path, data_path, output_file_path, csv_path)
        
        # Start the blocking event loop
        root.mainloop()
        
        # NOTE: Code execution resumes here only after app.root.quit() is called.
        # At this point, the window still exists (it's just not looping).
        
    except Exception as e:
        logging.error(f"An unexpected error occurred during GUI execution: {e}")
        # Re-raise or handle, but ensure finally block runs
        return
    finally:
        # 1. Clean up application-specific resources (Video handles, Matplotlib figures)
        if app is not None:
            app.cleanup()
        
        # 2. Destroy the GUI window immediately to prevent "zombies"
        # This handles the root window and any embedded Toplevels.
        try:
            root.destroy()
        except tk.TclError:
            pass # Window might have been closed by user explicitly

    # --- Post-Processing: Extraction and Saving ---
    # We only proceed if the app was successfully initialized
    if app is None:
        return

    selected_indices = sorted(list(app.selected_frames))
    
    if not selected_indices:
        logging.warning("No frames were selected. No file generated.")
        return

    print(f"Extracting mesh data for {len(selected_indices)} frames...")
    
    # --- Transformation Constants ---
    scaling_factor = 0.930
    MESH_UNIT_CONVERSION_FACTOR = 1000.0
    
    mesh_library = {}
    
    # We iterate through selected frames and extract clean data
    for idx in selected_indices:
        mesh_data = app.data_manager.extract_clean_mesh_data(idx, use_default_vertices=True)
        
        if mesh_data:
            # Apply Coordinate System Transformation (Meters -> Millimeters) and Rescaling
            # We target the 'vertices' key specifically.
            if 'vertices' in mesh_data:
                # Ensure data is a numpy array for vectorized multiplication
                verts = mesh_data['vertices']
                if not isinstance(verts, np.ndarray):
                    verts = np.array(verts)
                
                # Apply: Meters -> mm (* 1000) -> Scale (* 0.930)
                mesh_data['vertices'] = verts * MESH_UNIT_CONVERSION_FACTOR * scaling_factor
            
            mesh_library[idx] = mesh_data
        else:
            logging.warning(f"Frame {idx} selected but has no valid mesh data.")

    output_data = {
        "max_frames": app.total_frames,
        "meshes": mesh_library
    }
    
    try:
        out_path = Path(output_file_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 1. Save the Mesh Data
        with open(out_path, 'wb') as f:
            pickle.dump(output_data, f)
        print(f"Successfully saved mesh data to: {out_path}")
        print(f"Dictionary keys: max_frames, meshes (count: {len(mesh_library)})")
        
        # 2. Check Exit Status from GUI
        if app.is_task_validated:
            logging.info("User validated selection. Generating success flag...")
            success_path = Path(output_success_path)
            success_path.touch()
            print(f"Success flag created at: {success_path}")
        else:
            logging.info("User saved progress (Validation not confirmed). No success flag generated.")
            
    except Exception as e:
        logging.error(f"Error saving output file: {e}")

if __name__ == "__main__":
    # === Configuration ===
    DATABASE_PATH = r'F:\OneDrive - Linköpings universitet\_Teams\Social touch Kinect MNG\02_data\semi-controlled'
    
    VIDEO_PATH = Path(DATABASE_PATH + r'\1_primary\kinect\2022-06-17_ST16-05\block-order-01\2022-06-17_ST16-05_semicontrolled_block-order01_kinect.mp4')
    DATA_PATH = Path(DATABASE_PATH + r'\2_processed\kinect\2022-06-17_ST16-05\block-order-01\kinematics_analysis\2022-06-17_ST16-05_semicontrolled_block-order01_kinect_handmodel_tracked_hands.pkl')
    TRIAL_PATH = Path(DATABASE_PATH + r'\2_processed\kinect\2022-06-17_ST16-05\block-order-01\temporal_segmentation\2022-06-17_ST16-05_semicontrolled_block-order01_kinect_trial-chunks.csv')
    
    OUTPUT_PATH = Path(DATABASE_PATH + r'\2_processed\kinect\2022-06-17_ST16-05\block-order-01\kinematics_analysis\2022-06-17_ST16-05_semicontrolled_block-order01_kinect_handmodel_tracked_hands_curated.pkl')
    OUTPUT_SUCCESS_PATH = Path(str(OUTPUT_PATH) + ".SUCCESS")

    FORCE_PROCESSING = True

    print("--- Launching Hamer Checkup Selector ---")
    
    curate_hamer_hand_models(
        video_path=VIDEO_PATH, 
        data_path=DATA_PATH, 
        output_file_path=OUTPUT_PATH, 
        output_success_path=OUTPUT_SUCCESS_PATH,
        csv_path=TRIAL_PATH,
        force_processing=FORCE_PROCESSING
    )

    if Path(OUTPUT_PATH).exists():
        print("\n" + "="*40)
        print("TESTING MeshSequenceLoader")
        print("="*40)
        
        loader = MeshSequenceLoader(OUTPUT_PATH)
        print(f"Total Video Frames: {len(loader)}")
        
        test_frame = 500
        mesh = loader[test_frame]
        
        if mesh:
            v_count = len(mesh['vertices'])
            print(f"Mesh for Frame {test_frame}: Vertices count = {v_count}")
        else:
            print(f"Mesh for Frame {test_frame}: None (No preceding selection found)")