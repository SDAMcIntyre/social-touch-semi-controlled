import os

from preprocessing.stickers_analysis.roi import (
    ROITrackedFileHandler,

    XYZMetadataModel,
    XYZMetadataFileHandler,
    XYZVisualizationHandler,
    XYZDataFileHandler,
    XYZStickerOrchestrator
)


def extract_stickers_xyz_positions(
        source_video_path: str,
        input_csv_path: str,
        output_csv_path: str,
        metadata_path: str = None,
        video_path: str = None,
        *,
        force_processing: bool = False,
        monitor: bool = False
):
    """
    Extracts 3D sticker positions and optionally shows a standard, non-distorted monitor window.
    """
    if not force_processing and os.path.exists(output_csv_path) and os.path.exists(metadata_path):
        print(f"data already processed. Skipping...")
        return 
    
    if not os.path.exists(source_video_path): raise FileNotFoundError(f"Source video not found: {source_video_path}")
    if not os.path.exists(input_csv_path): raise FileNotFoundError(f"Center CSV not found: {input_csv_path}")
    
    print("Starting sticker 3D position extraction...")

    # 1. Define the configuration parameters for the job.
    config_data = {
        "source_video_path": source_video_path,
        "input_csv_path": input_csv_path,
        "output_csv_path": output_csv_path,
        "metadata_path": metadata_path,
        "monitor": monitor,
        "video_path": video_path
    }
    
    # Create the typed config object.
    config = XYZMetadataModel(**config_data)

    # 2. Instantiate dependency.
    visualizer = XYZVisualizationHandler(config)

    # 1. Load Data
    tracked_data_iohandler = ROITrackedFileHandler(input_csv_path)
    tracked_data = tracked_data_iohandler.load_all_data()
    sticker_names = list(tracked_data.keys())

    metadata_manager = XYZMetadataFileHandler(config)
    metadata = metadata_manager.get_metadata()
    metadata.update_processing_detail("stickers_found", sticker_names)

    # 3. Instantiate the main orchestrator with its dependencies.
    orchestrator = XYZStickerOrchestrator(visualizer=visualizer)

    # 4. Execute the process! 🚀
    try:
        results_df, processed_count = orchestrator.run(source_video_path, tracked_data)
        print("✅ Processing finished successfully!")
    except Exception as e:
        print(f"❌ An error occurred during processing: {e}")
        raise
    metadata.set_status("Completed")
    metadata.update_processing_detail("frames_processed", processed_count)
    metadata.finalize()

    # 4. Save Results
    XYZDataFileHandler.save(results_df, config.output_csv_path)
    metadata_manager.save(metadata, config.metadata_path)