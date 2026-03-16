from ..gui.user_interface import UserInterface
from ..models.video_processor import VideoProcessor

class ROIManager:
    def __init__(self, video_path):
        self.video_path = video_path
        # The UI handler is no longer a persistent attribute of the class.
        self.metadata = {}
        self.roi_coords = []
        self.roi_frames = []

    def set_parameters(self, metadata: dict):
        self.metadata = metadata
        self.roi_coords = {
            "x": metadata["x"],
            "y": metadata["y"],
            "width": metadata["width"],
            "height": metadata["height"]
        }


    def choose_parameters(self, window_title: str = "ROI Selection"):
        """
        Orchestrates the user interaction workflow.
        It now creates and destroys its own UserInterface instance for this specific task.
        
        Args:
            window_title (str): The title to display on the selection windows.
        """
        # ✅ STEP 1: Instantiate the UI handler at the start of the operation.
        ui = UserInterface()
        
        try:
            # ✅ STEP 2: Perform all UI-related actions within the 'try' block.
            print("Initializing video processing...")

            with VideoProcessor(self.video_path) as video:
                video_props = video.get_properties()
                self.metadata.update(video_props)
                self.metadata["video_path"] = self.video_path

                # 1. User selects a reference frame
                montage_indices = video.get_montage_frame_indices(num_frames=200)
                montage_frames = [video.get_frame(i) for i in montage_indices]
                
                # Pass the custom title to the frame selector
                selected_index = ui.select_frame_from_video(
                    montage_frames, 
                    title=f"{window_title} - Select Frame"
                )

                if selected_index is None:
                    print("No frame selected. Exiting operation.")
                    self.metadata["reference_frame_idx"] = -1
                    return # Exit the method
                
                reference_frame_idx = montage_indices[selected_index]
                self.metadata["reference_frame_idx"] = reference_frame_idx
                reference_frame = video.get_frame(reference_frame_idx)

                # 2. User draws the ROI
                # Pass the custom title to the ROI selector
                self.roi_coords = ui.select_roi_from_frame(
                    reference_frame, 
                    window_title=f"{window_title} - Draw ROI"
                )
                
                if self.roi_coords:
                    self.metadata.update(self.roi_coords)
                    print("Processing complete.")
                else:
                    print("ROI selection cancelled.")

        finally:
            # ✅ STEP 3: GUARANTEE the UI is destroyed, no matter what happens.
            # This 'finally' block will execute even if an error occurs or we return early.
            print("Cleaning up UI resources for this operation...")
            ui.destroy()

    def extract_roi_video(self):
        # The 'with' statement creates the VideoProcessor instance.
        # Its resources are guaranteed to be released when the block is exited.
        with VideoProcessor(self.video_path) as video:
            print("Extracting ROI from all frames...")
            self.roi_frames = video.extract_roi_from_video(self.roi_coords)
        # At this point, VideoProcessor.__exit__() has been called automatically.
        print("✅ Video resources have been released.")


# The __main__ block remains the same
if __name__ == '__main__':
    manager = ROIManager(
        video_path="path/to/your/video.mkv"
    )
    manager.choose_parameters(window_title="Debug Session")