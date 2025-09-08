import os
import pandas as pd
from typing import Dict, Any, Tuple

from utils.kinect_mkv_manager import KinectMKV

# Import the new processor and other required modules
from .xyz_extractor import Sticker3DPositionExtractor

# Type hints for injected dependencies
from ..gui.xyz_monitor_gui import XYZVisualizationHandler
from ...roi.models.roi_tracked_data import ROITrackedObjects


class XYZStickerOrchestrator:
    """
    Orchestrates the process of extracting 3D sticker positions by delegating
    responsibilities to specialized components.
    """

    def __init__(
        self,
        visualizer: XYZVisualizationHandler
    ):
        """
        Initializes the orchestrator with its dependencies.

        Args:
            visualizer: The handler for GUI visualization.
        """
        self.visualizer = visualizer

    def run(self, source_video_path: str, tracked_data: ROITrackedObjects) -> Tuple[pd.DataFrame, int]:
        """
        Executes the entire extraction and processing pipeline.

        Returns:
            A tuple containing:
            - A pandas DataFrame with a MultiIndex ('frame', 'sticker') containing
              the 3D coordinates and monitoring data for all objects.
            - The total number of frames processed.
        """
        print("Starting sticker 3D position extraction...")
        object_names = list(tracked_data.keys())

        # The results_data dictionary will be a temporary container
        results_data = {name: [] for name in object_names}
        
        try:
            with KinectMKV(source_video_path) as mkv:
                print(f"Successfully opened video with ~{len(mkv)} frames.")

                nframes_saved = 0
                # Iterate through each frame using a for loop
                for frame_index, frame in enumerate(mkv):
                    print(f"--- Processing Frame {frame_index} ---", end='\r')

                    for object_name in object_names:
                        tracked_obj_row = tracked_data[object_name].loc[frame_index]
                        if tracked_obj_row["status"] != "Failed":
                            (coords_3d, monitor_data) = Sticker3DPositionExtractor.get_xyz(
                                tracked_obj_row, 
                                frame.transformed_depth_point_cloud)
                        else:
                            (coords_3d, monitor_data) = Sticker3DPositionExtractor.get_empty_xyz()
                        results_data[object_name].append({**coords_3d, **monitor_data})

        except EOFError:
            # This is the expected way to end the loop
            print(f"\nEnd of file reached. A total of {nframes_saved} frames were saved.")
        except Exception as e:
            print(f"\nProcessing failed: {e}")
            raise
        finally:
            if self.visualizer:
                self.visualizer.release()
        
        final_df = self.structure_into_dataframe_dict(results_data)
        return final_df, frame_index

    def structure_into_dataframe_dict(self, results_data):
        """
        Converts collected data into a dictionary of DataFrames,
        one for each sticker.
        """
        print("\nStructuring results into a dictionary of DataFrames...")
        all_dfs = {}

        for sticker_name, frame_data_list in results_data.items():
            if not frame_data_list:
                continue
            
            # Create the DataFrame for this sticker
            df = pd.DataFrame(frame_data_list)
            
            # Rename the index to 'frame'
            df.index.name = 'frame'
            
            # Store it in the dictionary with the sticker name as the key
            all_dfs[sticker_name] = df

        if not all_dfs:
            print("Warning: No data was collected. Returning an empty dictionary.")

        print("DataFrame dictionary structuring complete.")
        return all_dfs