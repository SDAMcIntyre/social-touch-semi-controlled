
from preprocessing.stickers_analysis import (
    ROIAnnotationFileHandler,
    ROIAnnotationManager,
    ROIProcessingStatus,
    
    ColorSpaceFileHandler,
    ColorSpaceManager,
    ColorSpaceStatus
)

def is_2d_stickers_tracking_valid(metadata_path: str):
    """
    Wrapper function that calls tracking_valid and raises an exception on failure.

    Args:
        md_filename_abs (str): The absolute path to the metadata file.

    Raises:
        ValueError: If the tracking status is not valid or if the file/data
                    is unreadable, resulting in a non-True return from tracking_valid.
    """
    annotation_data_iohandler = ROIAnnotationFileHandler.load(metadata_path)
    annotation_manager = ROIAnnotationManager(annotation_data_iohandler)

    return annotation_manager.are_all_objects_with_status(ROIProcessingStatus.COMPLETED)


def is_correlation_videos_threshold_defined(metadata_path: str):
    manager: ColorSpaceManager = ColorSpaceFileHandler.load(metadata_path)
    
    for name in manager.colorspace_names:
        # Skip "discarded" definitions in the main loop; they are only used as helpers
        # for their parent colors.
        if "discarded" in name:
            continue
        if manager.get_status(name) != ColorSpaceStatus.REVIEW_COMPLETED.value:
            return False   
    return True
