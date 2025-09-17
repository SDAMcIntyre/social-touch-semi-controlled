# modified file: xyz_extracting_orchestrator.py
from __future__ import annotations

import pandas as pd
from typing import Tuple, TYPE_CHECKING

from preprocessing.common import KinectMKV
if TYPE_CHECKING:
    from preprocessing.stickers_analysis import TrackedDataInterface
from .xyz_extractor_interface import XYZExtractorInterface

class XYZStickerOrchestrator:
    """
    Orchestrates the process of extracting 3D positions by delegating
    all data-specific logic to an injected XYZExtractorInterface instance.
    """
    def __init__(self):
        """
        Initializes the orchestrator with a specific extraction strategy.
        """
        pass
    
    @staticmethod
    def run(source_video_path: str, data_source: TrackedDataInterface, extractor: XYZExtractorInterface) -> Tuple[pd.DataFrame, int]:
        if not isinstance(extractor, XYZExtractorInterface):
            raise TypeError("The provided extractor does not implement the XYZExtractorInterface interface.")

        print(f"Starting 3D position extraction...")
        object_names = data_source.object_names
        results_data = {name: [] for name in object_names}
        frame_index = -1

        try:
            with KinectMKV(source_video_path) as mkv:
                for frame_index, frame in enumerate(mkv):
                    print(f"--- Processing Frame {frame_index} ---", end='\r')

                    # The loop is now generic 🚀
                    for object_name, tracked_obj_row in data_source.get_items_for_frame(frame_index):
                        if extractor.should_process_row(tracked_obj_row):
                            (coords_3d, monitor_data) = extractor.extract(
                                tracked_obj_row, 
                                frame.transformed_depth_point_cloud)
                        else:
                            (coords_3d, monitor_data) = extractor.get_empty_result()
                        
                        results_data[object_name].append({**coords_3d, **monitor_data})
        
        except EOFError:
            print(f"\nEnd of file reached. Processed {frame_index + 1} frames.")
        except Exception as e:
            print(f"\nProcessing failed at frame {frame_index}: {e}")
            raise
        
        final_dfs = XYZStickerOrchestrator.structure_into_dataframe_dict(results_data)
        return final_dfs, frame_index + 1
    
    @staticmethod
    def structure_into_dataframe_dict(results_data):
        """
        Converts collected data into a dictionary of DataFrames, one for each object.
        """
        print("\nStructuring results into a dictionary of DataFrames...")
        all_dfs = {}

        for obj_name, frame_data_list in results_data.items():
            if not frame_data_list:
                continue
            df = pd.DataFrame(frame_data_list)
            df.index.name = 'frame'
            all_dfs[obj_name] = df

        if not all_dfs:
            print("Warning: No data was collected. Returning an empty dictionary.")

        print("DataFrame dictionary structuring complete.")
        return all_dfs
