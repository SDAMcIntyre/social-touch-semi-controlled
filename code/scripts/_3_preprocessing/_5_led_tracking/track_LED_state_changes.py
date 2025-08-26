import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time
import traceback
from typing import Optional

from .utils.KinectLEDBlinkingMP4 import KinectLEDBlinkingMP4
from .utils.waitforbuttonpress_popup import WaitForButtonPressPopup

# ===================================================================
#  STEP 1: Core Logic (Analysis Only)
# ===================================================================
def analyze_led_blinking(
        video_path: Path,
        occlusion_threshold: int = 40
) -> Optional[KinectLEDBlinkingMP4]:
    """
    Performs the core analysis of LED blinking from a video file.
    This function is responsible ONLY for computation and returns the results.
    """
    try:
        processor = KinectLEDBlinkingMP4(video_path)
        processor.load_video()
        processor.find_bimodal_green_pixels()
        processor.monitor_bimodal_pixels_green()
        processor.process_led_on()
        # led_blink_processor.define_occlusion(threshold=occlusion_threshold)
        return processor
    except Exception as e:
        print(f"ERROR: Analysis failed for '{video_path.name}': {e}")
        traceback.print_exc()
        return None


# ===================================================================
#  STEP 2: Helper Functions (I/O and UI)
# ===================================================================
def save_analysis_results(
    processor: KinectLEDBlinkingMP4,
    csv_output_path: Path,
    metadata_output_path: Path
) -> None:
    """Saves the analysis results to CSV and metadata files."""
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    processor.save_result_csv(csv_output_path)
    
    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)
    processor.save_result_metadata(metadata_output_path)
    print(f"INFO: Successfully saved results for '{processor.video_path.name}'.")

def plot_analysis_results(processor: KinectLEDBlinkingMP4) -> None:
    """Displays a plot of the analysis results."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 6))
    plt.plot(processor.time, processor.green_levels, label='Green Channel Level')
    plt.plot(processor.time, processor.led_on * np.max(processor.green_levels),
             label='Detected LED State (ON=High)', linestyle='--', alpha=0.8)
    plt.title(f"LED Blinking Analysis for {processor.video_path.name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Pixel Intensity / State")
    plt.legend()
    plt.ion()
    plt.show()
    WaitForButtonPressPopup()
    plt.close()


# ===================================================================
#  STEP 3: Orchestrator (Ties everything together)
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
    results, and optionally plots them. It serves as the main entry point.
    """
    # 1. Handle file existence check (Application logic)
    if not force_processing and csv_output_path.exists():
        print(f"INFO: Output for '{video_path.name}' already exists. Skipping.")
        return

    # 2. Run the core analysis (Computation)
    analysis_results = analyze_led_blinking(video_path)

    # 3. Handle results (I/O and UI logic)
    if analysis_results:
        save_analysis_results(analysis_results, csv_output_path, metadata_output_path)
        if show_plot:
            plot_analysis_results(analysis_results)