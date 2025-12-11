import os
from pathlib import Path

from utils.should_process_task import should_process_task
from preprocessing.stickers_analysis import (
    ConsolidatedTracksFileHandler,
    ConsolidatedTracksManager,

    XYZMetadataFileHandler,
    XYZMetadataModel,

    XYZDataFileHandler,

    XYZExtractorFactory,
    XYZStickerOrchestrator
)


def extract_stickers_xyz_positions(
        source_video_path: Path,
        input_csv_path: Path,
        method: str,  # This string will be used to select the extractor
        output_csv_path: Path,
        output_metadata_path: Path = None,
        *,
        force_processing: bool = False,
        debug: bool = True
):
    """
    Extracts 3D sticker positions using a dynamically selected method.
    """
    if not source_video_path.exists():
        raise FileNotFoundError(f"Source video not found: {source_video_path}")

    if not input_csv_path.exists():
        raise FileNotFoundError(f"Center CSV not found: {input_csv_path}")
    
    if not should_process_task(
        output_paths=[output_csv_path, output_metadata_path], 
        input_paths=[source_video_path, input_csv_path], 
        force=force_processing):
        print(f"✅ Output file '{output_csv_path}' and {output_metadata_path} already exist. Use --force to overwrite.")
        return

    print(f"Starting sticker 3D position extraction using method: '{method}'...")

    # 1. Load Data
    tracked_data: ConsolidatedTracksManager = ConsolidatedTracksFileHandler.load(input_csv_path)
    sticker_names = tracked_data.object_names

    # 2. Define the configuration parameters for the job.
    metadata = XYZMetadataModel(source_video_path, input_csv_path, output_csv_path)
    metadata.update_processing_detail("stickers_found", sticker_names)
    # 2a. Record the chosen extraction method in the metadata.
    metadata.update_processing_detail("extraction_method", method)

    try:
        # 3. Get the specific extractor instance from the factory using the method parameter.
        extractor = XYZExtractorFactory.get_extractor(method, debug=debug)
        print(f"✅ Successfully loaded extractor: {extractor.__class__.__name__}")

        # 4. Execute the process! 🚀
        # The orchestrator now receives the extractor object to use for its logic.
        # Note: The XYZStickerOrchestrator.run() method must be updated to accept this new 'extractor' argument.
        results_df, processed_count = XYZStickerOrchestrator.run(
            source_video_path=source_video_path,
            data_source=tracked_data,
            extractor=extractor
        )
        print("✅ Processing finished successfully!")

        metadata.set_status("Completed")
        metadata.update_processing_detail("frames_processed", processed_count)
        metadata.finalize()

        # 5. Save Results
        XYZDataFileHandler.save(results_df, output_csv_path)
        XYZMetadataFileHandler.save(metadata, output_metadata_path)

    except Exception as e:
        print(f"❌ An error occurred during processing: {e}")
        # Optionally, update metadata status on failure
        metadata.set_status("Failed")
        metadata.update_processing_detail("error_message", str(e))
        metadata.finalize()
        XYZMetadataFileHandler.save(metadata, output_metadata_path)
        raise