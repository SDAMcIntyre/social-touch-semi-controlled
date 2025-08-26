import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, Scale
from PIL import Image, ImageTk
import os
import json

class VideoSegmenterApp:
    """
    A GUI application for interactively segmenting a video based on pre-generated
    foreground masks. Can be controlled via the GUI or programmatically.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive Video Segmenter")

        # --- Member variables ---
        self.video_path = None
        self.mask_path = None
        self.cap_video = None
        self.cap_mask = None
        self.frame_count = 0
        self.current_frame_idx = 0
        self.threshold_value = 127 # Default threshold

        # --- GUI Widgets ---
        # Frame for video display
        self.video_frame = tk.Frame(root)
        self.video_frame.pack(padx=10, pady=10)
        self.canvas = tk.Canvas(self.video_frame, bg="black")
        self.canvas.pack()

        # Frame for controls
        self.controls_frame = tk.Frame(root)
        self.controls_frame.pack(padx=10, pady=5, fill=tk.X)
        
        # MODIFIED: Button now calls a wrapper method to prompt for a file.
        self.btn_load = tk.Button(self.controls_frame, text="Load Video", command=self._prompt_for_load)
        self.btn_load.pack(side=tk.LEFT, padx=5)

        self.threshold_slider = Scale(self.controls_frame, from_=0, to=255, orient=tk.HORIZONTAL, 
                                      label="Threshold", command=self.update_view, length=200)
        self.threshold_slider.set(self.threshold_value)
        self.threshold_slider.pack(side=tk.LEFT, padx=5)

        # MODIFIED: Button now calls a wrapper method to prompt for a save location.
        self.btn_validate = tk.Button(self.controls_frame, text="Validate & Save", command=self._prompt_for_save, state=tk.DISABLED)
        self.btn_validate.pack(side=tk.RIGHT, padx=5)

        # Frame for frame navigation
        self.nav_frame = tk.Frame(root)
        self.nav_frame.pack(padx=10, pady=5, fill=tk.X)
        self.frame_slider = Scale(self.nav_frame, from_=0, to=0, orient=tk.HORIZONTAL, 
                                  label="Frame", command=self.on_frame_slider_change, length=500, showvalue=True)
        self.frame_slider.pack(fill=tk.X, expand=True)

    def _prompt_for_load(self):
        """NEW: Wrapper method to open a file dialog for loading."""
        path = filedialog.askopenfilename(
            title="Select the ORIGINAL video file",
            filetypes=(("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("All files", "*.*"))
        )
        if path:
            self.load_video(path)

    # MODIFIED: This method now accepts a file path directly.
    def load_video(self, video_path: str):
        """Opens a video file and its corresponding mask file from a given path."""
        # Automatically find the corresponding mask file
        dir_name, file_name = os.path.split(video_path)
        base_name, _ = os.path.splitext(file_name)
        expected_mask_path = os.path.join(dir_name, f"{base_name}_fg_masks.mp4")

        if not os.path.exists(video_path):
            messagebox.showerror("Error", f"Video file not found!\nPath: {video_path}")
            return
            
        if not os.path.exists(expected_mask_path):
            messagebox.showerror("Error", f"Mask file not found!\nExpected at: {expected_mask_path}\n\nPlease run 'model_generator.py' first.")
            return

        self.video_path = video_path
        self.mask_path = expected_mask_path
        
        self.cap_video = cv2.VideoCapture(self.video_path)
        self.cap_mask = cv2.VideoCapture(self.mask_path)
        
        self.frame_count = int(self.cap_video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_slider.config(to=self.frame_count - 1)
        self.current_frame_idx = 0
        self.frame_slider.set(0)
        self.btn_validate.config(state=tk.NORMAL)
        
        self.update_view()

    def on_frame_slider_change(self, value):
        """Handles changes from the frame navigation slider."""
        self.current_frame_idx = int(value)
        self.update_view()

    def update_view(self, _=None):
        """Updates the video display based on the current frame and threshold."""
        if not self.cap_video or not self.cap_mask:
            return

        # Set the capture objects to the correct frame
        self.cap_video.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        self.cap_mask.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)

        ret_vid, frame_vid = self.cap_video.read()
        ret_mask, frame_mask = self.cap_mask.read()

        if ret_vid and ret_mask:
            self.threshold_value = self.threshold_slider.get()
            
            # Apply the selected threshold to the grayscale mask
            _, binary_mask = cv2.threshold(frame_mask, self.threshold_value, 255, cv2.THRESH_BINARY)
            
            # The mask from VideoCapture might be 3-channel, so convert to grayscale
            if len(binary_mask.shape) == 3:
                binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)

            # Apply the binary mask to the original frame
            result = cv2.bitwise_and(frame_vid, frame_vid, mask=binary_mask)

            # Convert for Tkinter display
            img = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            self.photo = ImageTk.PhotoImage(image=img)
            
            # Update canvas size and display image
            self.canvas.config(width=self.photo.width(), height=self.photo.height())
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def _prompt_for_save(self):
        """NEW: Wrapper method to open a file dialog for saving."""
        if not self.video_path:
            return
        
        save_path = filedialog.asksaveasfilename(
            title="Save processed video",
            defaultextension=".mp4",
            filetypes=(("MP4 files", "*.mp4"),)
        )
        if save_path:
            self.validate_and_save(save_path)

    # MODIFIED: This method now accepts a save path directly.
    def validate_and_save(self, save_path: str):
        """Processes the entire video with the final threshold and saves to a given path."""
        if not self.video_path:
            messagebox.showerror("Error", "No video loaded to process.")
            return

        messagebox.showinfo("Processing", "This may take a while. The application will be unresponsive during processing.")
        self.root.update_idletasks() # Update GUI before long task

        # Reset video captures to the beginning
        self.cap_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.cap_mask.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Get video properties for the output
        frame_width = int(self.cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap_video.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(save_path, fourcc, fps, (frame_width, frame_height))

        for i in range(self.frame_count):
            ret_vid, frame_vid = self.cap_video.read()
            ret_mask, frame_mask = self.cap_mask.read()
            if not ret_vid or not ret_mask:
                break
            
            # Apply the final threshold
            _, binary_mask = cv2.threshold(frame_mask, self.threshold_value, 255, cv2.THRESH_BINARY)
            if len(binary_mask.shape) == 3:
                binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)
            
            result = cv2.bitwise_and(frame_vid, frame_vid, mask=binary_mask)
            out_video.write(result)

        out_video.release()
        
        # Save computational data to a JSON file
        metadata = {
            "source_video": self.video_path,
            "mask_file": self.mask_path,
            "applied_threshold": self.threshold_value,
            "output_video": save_path,
            "frame_count": self.frame_count
        }
        meta_path = os.path.splitext(save_path)[0] + "_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        messagebox.showinfo("Success", f"Video and metadata saved successfully!\nVideo: {save_path}\nMetadata: {meta_path}")
        self.release_captures()

    def release_captures(self):
        if self.cap_video:
            self.cap_video.release()
        if self.cap_mask:
            self.cap_mask.release()

if __name__ == '__main__':
    root = tk.Tk()
    app = VideoSegmenterApp(root)
    
    # --- Example of programmatic control ---
    # To run programmatically, you could uncomment the following lines
    # and comment out root.mainloop()
    
    # video_to_process = "path/to/your/video.mp4"
    # output_path = "path/to/your/output.mp4"
    # app.load_video(video_to_process)
    # app.threshold_slider.set(150) # Set a threshold
    # app.update_view()
    # app.validate_and_save(output_path)
    
    # For interactive use, just run the main loop
    root.mainloop()
    
    app.release_captures()