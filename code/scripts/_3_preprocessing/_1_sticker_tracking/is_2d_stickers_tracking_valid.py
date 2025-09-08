
from preprocessing.stickers_analysis.roi import (
    ROIAnnotationFileHandler,
    ROIAnnotationManager,
    ROIProcessingStatus
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

