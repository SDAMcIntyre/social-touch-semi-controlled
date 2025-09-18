import os
from pathlib import Path
from typing import Optional

from utils.should_process_task import should_process_task
from preprocessing.stickers_analysis import (
    ROIProcessingStatus,
    ROIAnnotationFileHandler,
    ROIAnnotationManager,
    TrackingOrchestrator,
    ROITrackedFileHandler,
)

def track_objects_in_video(
        video_path: str, 
        metadata_path: str, 
        output_path: str,
        *,
        force_processing: bool = False,
        gui_control: bool = False,
        show_gui: bool = False) -> Optional[str]:
    """
    Orchestrates the full video object tracking pipeline.

    This function loads object metadata, runs trackers on a video,
    and saves the resulting tracking data based on the provided settings.

    Args:
        video_path (str): Path to the input RGB video file.
        metadata_path (str): Path to the file containing object metadata.
        output_path (str): Path where the output tracking results will be saved.
        gui_control (bool): Enables GUI control features during the tracking process.
        show_gui (bool): Determines if the tracking GUI should be displayed.
        save_results (bool): If True, the final tracking results are saved to the output_path.

    Returns:
        Optional[str]: The output path if results were successfully saved, otherwise None.
    """
    # This block only runs if Step 1 determined that processing might be needed.
    print("🔎 Performing task-specific check on annotation status...")
    # we should exclude metadata_path as it is updated manually to assess the quality of the automatic process 
    need_to_process = should_process_task(output_paths=output_path, input_paths=video_path, force=force_processing)
    annotation_data_iohandler = ROIAnnotationFileHandler.load(metadata_path)
    annotation_manager = ROIAnnotationManager(annotation_data_iohandler)
    # Logic to check the JSON content.
    if not need_to_process and annotation_manager.are_no_objects_with_status(ROIProcessingStatus.TO_BE_PROCESSED):
        print("✅ Task-specific check passed: No objects require processing. Skipping.")
        return
    
    print(f"--- Starting processing for: {os.path.basename(video_path)} ---")

    orchestrator = TrackingOrchestrator(video_path=video_path, use_gui=show_gui)

    results = {}
    annotation_file_modified = False
    for object_name in annotation_manager.get_object_names():
        tracked_object = annotation_manager.get_object(object_name)

        if need_to_process or tracked_object.status == ROIProcessingStatus.TO_BE_PROCESSED.value:
            tracked_roi = orchestrator.run(labeled_rois=tracked_object.rois)

            # 1. Track the object
            results[object_name] = tracked_roi
            
            # 2. Filter the DataFrame for rows where status is 'manual'
            manual_rows = tracked_roi[tracked_roi['status'] == 'manual']
            
            # 3. Update the annotation metadata file content
            print("Iterating through rows where status is 'manual':")
            for index, row in manual_rows.iterrows():
                annotation_manager.set_roi(object_name, row['frame_id'], row['roi_x'], row['roi_y'], row['roi_width'], row['roi_height'])
            annotation_manager.update_status(object_name, ROIProcessingStatus.TO_BE_REVIEWED)
            annotation_file_modified = True
    
    if ROITrackedFileHandler.is_roi_tracked_objects(results): 
        tracked_data_iohandler = ROITrackedFileHandler(output_path)
        tracked_data_iohandler.save_all_data(results)

    if annotation_file_modified:
        ROIAnnotationFileHandler.save(metadata_path, annotation_manager.data)
        
    return output_path
