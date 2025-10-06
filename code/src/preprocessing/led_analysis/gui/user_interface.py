import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox

# Assuming these imports point to your existing, correct GUI components
from preprocessing.common.gui.video_frame_selector import VideoFrameSelector
from preprocessing.common.gui.frame_roi_square import FrameROISquare

class UserInterface:
    """
    Handles user interactions for selecting video frames and defining regions of interest (ROI)
    by managing a single, persistent Tkinter root window.
    """
    def __init__(self):
        """
        Initializes the UI handler and the main, hidden Tkinter root window.
        This root will be the parent for all subsequent dialogs.
        """
        self.root = tk.Tk()
        # ❌ Instead of this:
        # self.root.withdraw()
        # ✅ Do this: Move the window to an off-screen position.
        # This keeps it active for managing dialogs without being visible.
        self.root.geometry('+10000+10000')

    def destroy(self):
        """
        Properly destroys the main Tkinter root window to clean up all resources.
        """
        if self.root:
            self.root.destroy()
            self.root = None

    def select_frame_from_video(self, video_source, start_frame: int = 0, initial_frame: int = 0) -> int | None:
        """
        Opens a modal dialog to select a specific frame from a video source.
        This method now uses the persistent root window as the parent.
        """
        print("✅ Initializing frame selector...")
        
        selected_frame = None
        try:
            # The selector GUI is managed by VideoFrameSelector, parented by our single root
            selector = VideoFrameSelector(
                parent=self.root, # Use the class-level root
                video_source=video_source,
                first_frame=start_frame,
                current_frame_num=initial_frame,
                title="Video Processor"
            )
            selected_frame = selector.select_frame()
        except Exception as e:
            print(f"❌ An error occurred with the frame selector: {e}")
            messagebox.showerror("Error", f"Could not open frame selector: {e}", parent=self.root)
            
        return selected_frame

    def select_roi_from_frame(self, frame: np.ndarray, predefined_roi: dict = None) -> dict | None:
        """
        Displays a frame to allow the user to define a Region of Interest (ROI).
        
        Note: For optimal architecture, the 'FrameROISquare' class should also be
        modified to accept a parent window, similar to 'VideoFrameSelector'.
        """
        print("✅ Initializing ROI selector...")
        if frame is None:
            print("❌ Cannot select ROI from a null frame.")
            return None

        try:
            # This component is called as-is. Ideally, it would also take `parent=self.root`.
            roi_selector = FrameROISquare(
                image_input=frame,
                is_rgb=True,
                window_title="Define Region of Interest (ROI)",
                predefined_roi=predefined_roi
            )
            roi_selector.run()
            return roi_selector.get_roi_data()
        except Exception as e:
            print(f"❌ An error occurred during ROI selection: {e}")
            return None

    def confirm_action(self, title: str = "Confirm Action", question: str = "Do you want to proceed?") -> bool:
        """
        Displays a simple confirmation dialog using the persistent root window.
        """
        # The messagebox will be properly parented to our hidden root window.
        return messagebox.askyesno(title, question, parent=self.root)

# --- CORRECTED Example Usage ---
if __name__ == '__main__':
    video_path = 'F:\\liu-onedrive-nospecial-carac\\_Teams\\Social touch Kinect MNG\\data\\semi-controlled\\1_primary\\kinect\\2022-06-14_ST13-01\\block-order-03\\2022-06-14_ST13-01_semicontrolled_block-order03_kinect.mp4'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        # Use a messagebox for GUI-based error reporting
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("File Error", f"Could not open video file at '{video_path}'")
        root.destroy()
    else:
        # --- STEP 1: Instantiate the UserInterface ONCE ---
        ui = UserInterface()
        
        try:
            # --- STEP 2: Use the UI instance to select a frame ---
            selected_frame_num = ui.select_frame_from_video(video_source=cap, initial_frame=100)
            
            # --- STEP 3: If a frame was selected, proceed to select an ROI ---
            if selected_frame_num is not None:
                print(f"\nProceeding with frame number: {selected_frame_num}")
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, selected_frame_num)
                success, frame_bgr = cap.read()
                
                if success:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    
                    # --- STEP 4: Use the UI instance to get an ROI ---
                    roi_data = ui.select_roi_from_frame(frame=frame_rgb)
                    
                    if roi_data:
                        print("\n🎉 Final Selection Complete!")
                        print(f"   - Selected Frame: {selected_frame_num}")
                        print(f"   - Selected ROI: {roi_data}")

                        if ui.confirm_action(question="Would you like to edit the ROI?"):
                            print("\nRe-opening ROI editor with previous selection...")
                            edited_roi_data = ui.select_roi_from_frame(frame=frame_rgb, predefined_roi=roi_data)
                            
                            if edited_roi_data:
                                print("\n🎉 Final Edited Selection:")
                                print(f"   - Edited ROI: {edited_roi_data}")
                            else:
                                print("\nROI editing was cancelled.")
                    else:
                        print("\nROI selection was cancelled.")
                else:
                    print(f"❌ Failed to read frame {selected_frame_num} from the video.")
            else:
                print("\nFrame selection was cancelled. Exiting.")
                
        finally:
            # --- STEP 5: Clean up ALL resources at the end ---
            cap.release()
            ui.destroy() # This destroys the single Tkinter root
            print("\nProgram finished.")