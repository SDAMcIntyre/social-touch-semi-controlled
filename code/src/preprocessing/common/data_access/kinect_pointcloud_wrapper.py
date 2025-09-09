import cv2
from typing import Iterator

from .kinect_mkv_manager import (
    KinectMKV,
    KinectFrame
)

from ..gui.scene_viewer import PointCloudData


class KinectPointCloudView:
    """
    A wrapper around the KinectMKV class to provide direct access to
    a data object with .color and .points attributes for each frame.
    
    This simplifies access patterns, allowing for intuitive, object-like retrieval.

    Example:
        with KinectMKV("video.mkv") as mkv:
            wrapper = KinectPointCloudView(mkv)
            
            # Directly get a PointCloudData object for frame 5
            frame_data = wrapper[5]
            color_image = frame_data.color
            point_cloud = frame_data.points
            
            # Iterate through all frames, getting the PointCloudData object on each iteration
            for frame in wrapper:
                # process frame.color and frame.points
                pass
    """
    def __init__(self, kinect_mkv: KinectMKV):
        """
        Initializes the wrapper.

        Args:
            kinect_mkv: An opened and ready-to-use instance of the KinectMKV class.
        """
        self._mkv = kinect_mkv

    def _create_frame_view(self, frame_object: KinectFrame) -> PointCloudData:
        """
        Internal helper to process a KinectFrame, convert color to RGB, and return a PointCloudData.
        """
        # Check if either the color or point cloud data is missing
        if frame_object.color is None or frame_object.transformed_depth_point_cloud is None:
            return PointCloudData(color=None, points=None)

        # Convert the color image from BGR to RGB
        color_rgb = cv2.cvtColor(frame_object.color, cv2.COLOR_BGR2RGB)

        # Reshape the data into flat vectors
        flat_colors_vector = color_rgb.reshape(-1, 3)
        flat_points_vector = frame_object.transformed_depth_point_cloud.reshape(-1, 3)
        
        # Return a PointCloudData object with the desired properties
        return PointCloudData(color=flat_colors_vector, points=flat_points_vector)

    def __getitem__(self, frame_index: int) -> PointCloudData:
        """
        Enables accessing a frame's data using square-bracket notation.

        This fetches the specified KinectFrame, converts its color data to RGB,
        and returns a PointCloudData object containing the RGB color image and point cloud.
        If data is missing for the frame, both attributes will be None.

        Args:
            frame_index: The integer index of the frame to retrieve.

        Returns:
            A PointCloudData object with .color (RGB) and .points attributes.
        """
        # Get the full KinectFrame object from the underlying reader
        frame_object = self._mkv[frame_index]
        return self._create_frame_view(frame_object)

    def __len__(self) -> int:
        """
        Returns the total number of frames by delegating the call to the
        original KinectMKV object. This allows `len(wrapper)` to work.
        """
        return len(self._mkv)

    def __iter__(self) -> Iterator[PointCloudData]:
        """
        Allows for iterating over the wrapper in a for-loop.

        Yields:
            A PointCloudData object for each frame in sequence, with color data in RGB.
        """
        for frame_object in self._mkv:
            # For each frame, yield the processed PointCloudData
            yield self._create_frame_view(frame_object)