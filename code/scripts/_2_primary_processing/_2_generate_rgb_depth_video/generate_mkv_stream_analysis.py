import os
from pyk4a import K4AException

from primary_processing.mkv_video_management.mkv_stream_analyze import (
    MKVStreamAnalyzer, 
    save_report_to_csv
)
    
def generate_mkv_stream_analysis(
              input_video: str,
              output_csv_path:str
) -> str:
    
    if os.path.exists(output_csv_path):
        print(f"Analysis has already been done (result file: {output_csv_path}). Skipping")
        return output_csv_path
    
    try:
        # Use the class as a context manager
        with MKVStreamAnalyzer(input_video) as analyzer:
            # 1. The analyzer generates the results
            frame_report_generator = analyzer.analyze_frames()
            
            # 2. The separate writer function consumes the results and saves the file
            save_report_to_csv(frame_report_generator, output_csv_path)

    except (FileNotFoundError, K4AException, RuntimeError) as e:
        print(f"An error occurred: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

    return output_csv_path