# track_LED_state_changes.py

import matplotlib.pyplot as plt
import numpy as np
import traceback
from typing import Optional, Dict, Any
import logging
from pathlib import Path

# Setup a basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime-s - %(levelname)s - %(message)s')


from utils.should_process_task import should_process_task

from preprocessing.led_analysis import (
    LEDBlinkingAnalyzer,
    LEDBlinkingFilesHandler,
    WaitForButtonPressPopup
)

# ===================================================================
#  Helper Function (UI - Modified)
# ===================================================================
def plot_analysis_results(
    time_series_data: Dict[str, Any],
    video_name: str
) -> None:
    """
    Displays a plot of the analysis results from the time-series dictionary.
    
    Args:
        time_series_data (Dict[str, Any]): The dictionary containing time-series arrays.
        video_name (str): The name of the video file for the plot title.
    """
    time = time_series_data.get('time', np.array([]))
    green_levels = time_series_data.get('green_levels', np.array([]))
    led_on = time_series_data.get('led_on', np.array([]))

    # Guard against empty data
    if time.size == 0 or green_levels.size == 0:
        logging.warning("Cannot plot results; time-series data is empty.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 7))
    
    # Plot green channel intensity
    ax.plot(time, green_levels, label='Normalized Green Level', color='green', alpha=0.9)
    
    # Plot detected LED state, handling NaNs for occluded/corrupted frames
    # We create a copy to manipulate for plotting
    led_on_plot = led_on.copy()
    valid_led_on = led_on_plot[~np.isnan(led_on_plot)]
    if valid_led_on.size > 0:
        # Scale the 0-1 signal to the max of the green levels for visibility
        led_on_plot *= np.nanmax(green_levels)
        ax.plot(time, led_on_plot, label='Detected LED ON State', linestyle='--', color='red', alpha=0.8)

    ax.set_title(f"LED Blinking Analysis for {video_name}", fontsize=16)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Normalized Intensity / State", fontsize=12)
    ax.legend()
    ax.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    
    # Use ion() to allow script to continue after plot is shown
    plt.ion()
    plt.show()
    
    # Wait for user to close the plot
    WaitForButtonPressPopup()
    plt.close(fig)


# ===================================================================
#  Orchestrator (Ties everything together)
# ===================================================================
def track_led_states_changes(
    video_path: Path,
    csv_output_path: Path,
    metadata_output_path: Path,
    *,
    force_processing: bool = False,
    show_plot: bool = False
) -> None:
    """
    Orchestrates the full processing pipeline for a single video file.
    
    This function handles file checks, calls the analysis, saves the
    results using the dedicated handler, and optionally plots them.
    """
    # 1. Check if processing is needed    
    if not should_process_task(
         input_paths=[video_path], 
         output_paths=[csv_output_path, metadata_output_path], 
         force=force_processing):
        logging.info(f"Output files already exist. Skipping analysis for '{video_path.name}'.")
        return

    # 2. Run the core analysis (Computation)
    analyzer = None
    try:
        # Instantiate the analyzer with desired parameters
        analyzer = LEDBlinkingAnalyzer(
            video_path=video_path,
            occlusion_threshold=40.0,
            discard_black_frames=True,
            black_frame_threshold=1.0
        )
        # Run the analysis which populates the .results attribute
        analyzer.run_analysis(
            led_on_method="bimodal", 
            update_on_occlusion=False,
            show=True)
    except Exception as e:
        logging.error(f"Analysis failed for '{video_path.name}': {e}")
        traceback.print_exc()
        return None
    
    # 3. Handle results (I/O and UI logic)
    if analyzer.results is not None:
        # 3a. Save results using the dedicated FilesHandler
        logging.info(f"Saving analysis results for ...")
        saver = LEDBlinkingFilesHandler()
        saver.save_timeseries_to_csv(analyzer.time_series, csv_output_path)
        saver.save_metadata_to_json(analyzer.metadata, metadata_output_path)


        # 3b. Optionally plot the results
        if show_plot:
            logging.info("Generating plot...")
            # Pass the relevant data parts to the plotting function
            plot_analysis_results(analyzer.results['time_series_data'], video_path.name)
    else:
        logging.error(f"Failed to get analysis results for '{video_path.name}'. Nothing will be saved or plotted.")



# ===================================================================
#  STEP 4: Example Execution Block
# ===================================================================
if __name__ == '__main__':
    import cv2
    # --- Configuration ---
    # Create a dummy video file for demonstration purposes.
    # In a real scenario, this path would point to your actual video.
    dummy_video_path = Path("./dummy_led_video.mp4")
    if not dummy_video_path.exists():
        logging.info("Creating a dummy video file for demonstration...")
        # Create a 10-second, 320x240, 30 FPS video with a blinking green square
        width, height, fps, duration = 320, 240, 30, 10
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(dummy_video_path), fourcc, fps, (width, height))
        for i in range(fps * duration):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            # Blink the green square every 15 frames (0.5 seconds)
            if (i // 15) % 2 == 0:
                cv2.rectangle(frame, (140, 100), (180, 140), (0, 200, 0), -1) # Green square
            writer.write(frame)
        writer.release()
        logging.info(f"Dummy video created at '{dummy_video_path}'")

    # Define the output directory and a base name for the result files
    video_input_path = dummy_video_path
    results_directory = Path("./analysis_results")
    results_basename = video_input_path.stem + "_analysis"

    # --- Execution ---
    print("\n" + "="*50)
    print(f"Starting LED Blinking Analysis for: {video_input_path.name}")
    print(f"Results will be saved in: {results_directory}")
    print("="*50 + "\n")

    track_led_states_changes(
        video_path=video_input_path,
        output_dir=results_directory,
        file_basename=results_basename,
        force_processing=True,  # Set to True to always re-run analysis
        show_plot=True          # Set to True to see the visual results
    )

    print("\n" + "="*50)
    print("Processing finished.")
    print(f"Check for '{results_basename}.csv' and '{results_basename}_metadata.json' in the '{results_directory}' folder.")
    print("="*50 + "\n")