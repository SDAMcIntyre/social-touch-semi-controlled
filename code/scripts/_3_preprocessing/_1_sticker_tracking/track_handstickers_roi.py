import os
from pathlib import Path
from typing import Optional

from utils.should_process_task import should_process_task
from preprocessing.common.data_access.video_mp4_manager import VideoMP4Manager
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
    It attempts to load existing tracking data to allow for incremental updates.

    Args:
        video_path (str): Path to the input RGB video file.
        metadata_path (str): Path to the file containing object metadata.
        output_path (str): Path where the output tracking results will be saved.
        gui_control (bool): Enables GUI control features during the tracking process.
        show_gui (bool): Determines if the tracking GUI should be displayed.

    Returns:
        Optional[str]: The output path if results were successfully saved, otherwise None.
    """
    # This block only runs if Step 1 determined that processing might be needed.
    print("🔎 Performing task-specific check on annotation status...")
    
    need_to_process = should_process_task(output_paths=output_path, input_paths=video_path, force=force_processing)
    annotation_data_iohandler = ROIAnnotationFileHandler.load(metadata_path)
    annotation_manager = ROIAnnotationManager(annotation_data_iohandler)
    
    # Check if there are objects that explicitly require processing
    objects_to_process_exist = not annotation_manager.are_no_objects_with_status(ROIProcessingStatus.TO_BE_PROCESSED)

    if not need_to_process and not objects_to_process_exist:
        print("✅ Task-specific check passed: No objects require processing. Skipping.")
        return
    
    print(f"--- Starting processing for: {os.path.basename(video_path)} ---")

    # Initialize the FileHandler early to load existing data
    tracked_data_iohandler = ROITrackedFileHandler(output_path)
    
    # Load existing results if they exist; otherwise, start with an empty dict
    results = tracked_data_iohandler.load_all_data()
    if results is None:
        results = {}
        print("ℹ️  No existing tracking data found. Starting fresh.")
    else:
        print(f"ℹ️  Loaded existing tracking data for {len(results)} objects.")

    # Need to fetch total frames to calculate the processable mask
    try:
        # Open video temporarily to get total_frames
        temp_vm = VideoMP4Manager(video_path)
        total_frames = temp_vm.total_frames
        # VideoMP4Manager often relies on __del__ or context managers.
    except Exception as e:
        print(f"Error reading video info: {e}")
        return None

    orchestrator = TrackingOrchestrator(video_path=video_path, use_gui=show_gui)
    annotation_file_modified = False

    for object_name in annotation_manager.get_object_names():
        tracked_object = annotation_manager.get_object(object_name)

        # Process if the global 'need_to_process' is True OR if this specific object is flagged
        if need_to_process or tracked_object.status == ROIProcessingStatus.TO_BE_PROCESSED.value:
            
            print(f"Processing object: {object_name}")
            
            # Calculate which frames should be processed and which ignored
            frames_to_process = annotation_manager.get_frames_to_process(object_name, total_frames)
            
            tracked_roi = orchestrator.run(
                labeled_rois=tracked_object.rois,
                valid_frames=frames_to_process
            )

            # 1. Track the object - Overwrite existing data for this object
            results[object_name] = tracked_roi
            
            # 2. Filter the DataFrame for rows where status is 'manual'
            manual_rows = tracked_roi[tracked_roi['status'] == 'manual']
            
            # 3. Update the annotation metadata file content
            if not manual_rows.empty:
                print(f"Updating manual annotations for {object_name}...")
                for index, row in manual_rows.iterrows():
                    annotation_manager.set_roi(
                        object_name, 
                        row['frame_id'], 
                        row['roi_x'], 
                        row['roi_y'], 
                        row['roi_width'], 
                        row['roi_height']
                    )
            
            annotation_manager.update_status(object_name, ROIProcessingStatus.TO_BE_REVIEWED)
            annotation_file_modified = True
    
    # Save the consolidated results (existing + new)
    if ROITrackedFileHandler.is_roi_tracked_objects(results): 
        tracked_data_iohandler.save_all_data(results)

    if annotation_file_modified:
        ROIAnnotationFileHandler.save(metadata_path, annotation_manager.data)
        
    return output_path