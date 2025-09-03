import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import numpy as np

class VideoFrameSelector:
    """
    Manages the frame selection GUI, built with Tkinter.
    This class allows a user to scrub through frames from either an OpenCV
    video capture object or a pre-loaded array of RGB frames.
    """
    def __init__(self, parent, video_source, first_frame=0, last_frame=-1, current_frame_num=0, title=None):
        """
        Initializes the VideoFrameSelector Tkinter GUI.

        Args:
            parent: The parent tkinter window (e.g., the main tk.Tk() instance).
            video_source: Either an OpenCV video capture object or a list/NumPy
                          array of RGB frames.
            title (str): The base title for the window.
            first_frame (int): The first frame number for the slider.
            last_frame (int): The last frame number for the slider. If -1, it's
                              auto-detected from the source.
            current_frame_num (int): The initial frame to display.
        """
        self.parent = parent
        self.base_title = title
        self.selected_frame_num = current_frame_num
        self.proceed_was_clicked = False
        self.first_frame = first_frame

        # --- Determine source type and configure accordingly ---
        # This block checks if the source is a video file or an array of frames.
        self.is_video_mode = False
        if hasattr(video_source, 'get') and callable(getattr(video_source, 'get')): # Duck-typing for cv2.VideoCapture
            self.is_video_mode = True
            self.video = video_source
            if last_frame == -1:
                # Get total frames from video capture
                self.last_frame = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
            else:
                self.last_frame = last_frame
        elif isinstance(video_source, (list, np.ndarray)):
            self.is_video_mode = False
            self.frames = np.asarray(video_source) # Ensure it's a numpy array
            if last_frame == -1:
                # Get total frames from the array
                self.last_frame = len(self.frames) - 1 if len(self.frames) > 0 else 0
            else:
                self.last_frame = last_frame
        else:
            raise TypeError("video_source must be an OpenCV VideoCapture object or a list/array of RGB frames.")


        # --- Create the Toplevel window instead of a new Tk() root ---
        self.root = self.parent
        self.root.title(f"Frame Selector")

        # --- Get screen dimensions and calculate target size for the video frame ---
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        self.target_w = screen_w // 2
        self.target_h = screen_h // 2

        # --- Create variables to hold widget values ---
        self.scale_var = tk.IntVar(value=current_frame_num)
        self.entry_var = tk.StringVar(value=str(current_frame_num))

        # --- Create the main frame for video display ---
        self.frame_display = ttk.Frame(self.root, padding="10")
        self.frame_display.pack(fill="both", expand=True)

        self.image_label = ttk.Label(self.frame_display)
        self.image_label.pack(pady=10)

        # --- Create the main control frame for widgets ---
        self.control_frame = ttk.Frame(self.root, padding="10")
        self.control_frame.pack(fill="x")
        
        # --- Create and pack slider and its min/max labels ---
        self.min_label = ttk.Label(self.control_frame, text=str(self.first_frame))
        self.min_label.pack(side="left", padx=(5, 10))

        self.scale = ttk.Scale(
            self.control_frame,
            from_=self.first_frame,
            to=self.last_frame,
            orient="horizontal",
            variable=self.scale_var,
            command=self._update_frame
        )
        self.scale.pack(fill="x", expand=True, side="left")

        self.max_label = ttk.Label(self.control_frame, text=str(self.last_frame))
        self.max_label.pack(side="left", padx=(10, 5))

        # --- Create a new frame for the entry box, current value label, and button ---
        self.info_frame = ttk.Frame(self.root, padding="5")
        self.info_frame.pack(fill="x")
        
        # Configure grid layout for the info_frame
        self.info_frame.columnconfigure(0, weight=1)
        self.info_frame.columnconfigure(1, weight=1)
        self.info_frame.columnconfigure(2, weight=1)

        # --- Current frame label and entry box ---
        self.current_label = ttk.Label(self.info_frame, text=f"Current Frame: {self.selected_frame_num}")
        self.current_label.grid(row=0, column=0, sticky="e", padx=5)

        self.frame_entry = ttk.Entry(
            self.info_frame, 
            textvariable=self.entry_var, 
            width=10
        )
        self.frame_entry.grid(row=0, column=1, sticky="w")
        self.frame_entry.bind("<Return>", self._update_from_entry)
        self.frame_entry.bind("<FocusOut>", self._update_from_entry)

        # --- Proceed button ---
        self.proceed_button = ttk.Button(
            self.info_frame,
            text="Proceed",
            command=self._on_proceed
        )
        self.proceed_button.grid(row=0, column=2, sticky="e", padx=10)

        # --- Bind arrow keys for frame navigation ---
        self.root.bind('<Left>', self._handle_arrow_keys)
        self.root.bind('<Right>', self._handle_arrow_keys)
        self.root.bind('<Control-Left>', self._handle_arrow_keys)
        self.root.bind('<Control-Right>', self._handle_arrow_keys)

        self.root.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self._update_frame() # Initial frame load

    def _update_frame(self, _=None):
        """
        Callback to get, resize, and display the selected frame.
        This method now handles both video and array sources.
        """
        self.selected_frame_num = self.scale_var.get()
        
        self.entry_var.set(str(self.selected_frame_num))
        self.current_label.config(text=f"Current Frame: {self.selected_frame_num}")

        frame = None
        success = False

        # --- Get the frame from the appropriate source ---
        if self.is_video_mode:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, self.selected_frame_num)
            success, bgr_frame = self.video.read()
            if success:
                # Convert from BGR (OpenCV default) to RGB for PIL
                frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        else:  # Array mode
            if 0 <= self.selected_frame_num < len(self.frames):
                # Frames are assumed to be RGB already
                frame = self.frames[self.selected_frame_num]
                success = True

        if success and frame is not None:
            # --- Common processing for the frame (resizing and displaying) ---
            
            # Resize frame
            original_h, original_w = frame.shape[:2]
            scale = min(self.target_w / original_w, self.target_h / original_h)
            new_w = int(original_w * scale)
            new_h = int(original_h * scale)
            resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Convert for Tkinter display (frame is already RGB)
            img = Image.fromarray(resized_frame)
            self.photo = ImageTk.PhotoImage(image=img)
            self.image_label.config(image=self.photo)
        
        if self.base_title:
            self.root.title(f"{self.base_title}: Frame Selector - Frame: {self.selected_frame_num}")
        else:
            self.root.title(f"Frame Selector - Frame: {self.selected_frame_num}")

    def _update_from_entry(self, _=None):
        """Update slider and frame based on entry box input."""
        try:
            frame_num = int(self.entry_var.get())
            frame_num = max(self.first_frame, min(frame_num, self.last_frame))
            self.scale_var.set(frame_num)
            self._update_frame()
        except ValueError:
            self.entry_var.set(str(self.scale_var.get()))

    def _handle_arrow_keys(self, event):
        """Handle left and right arrow key presses to navigate frames."""
        current_frame = self.scale_var.get()
        is_ctrl_pressed = (event.state & 4) != 0
        step = 10 if is_ctrl_pressed else 1
        
        new_frame = current_frame
        if event.keysym == 'Left':
            new_frame -= step
        elif event.keysym == 'Right':
            new_frame += step
            
        clamped_frame = max(self.first_frame, min(new_frame, self.last_frame))
        self.scale_var.set(clamped_frame)
        self._update_frame()
    
    def _on_proceed(self):
        """Called when the Proceed button is clicked."""
        self.proceed_was_clicked = True
        self.root.destroy()

    def _on_cancel(self):
        """Called when the window is closed directly."""
        self.proceed_was_clicked = False
        self.root.destroy()

    def select_frame(self):
        """
        Shows the dialog, makes it modal, and waits until it is closed.
        Returns the selected frame number or None if cancelled.
        """
        print("Move slider, use arrow keys, or enter frame number. Click 'Proceed' to confirm.")
        # --- Modal Dialog Behavior ---
        self.root.transient(self.parent)
        self.root.grab_set()
        self.root.wait_window(self.root)

        if self.proceed_was_clicked:
            print(f"Frame {self.selected_frame_num} selected.")
            return self.selected_frame_num
        else:
            print("Selection cancelled.")
            return None

# Example usage: Demonstrates creating instances of the selector
if __name__ == '__main__':
    # --- Create a single, main Tk() instance for the application ---
    # We can hide this window if we only want to show the dialogs.

    # --- DEMO 1: Using a video file ---
    # IMPORTANT: Replace this with a valid path to a video file on your system.
    video_path = 'F:\\liu-onedrive-nospecial-carac\\_Teams\\Social touch Kinect MNG\\data\\semi-controlled\\1_primary\\kinect\\2022-06-14_ST13-01\\block-order-03\\2022-06-14_ST13-01_semicontrolled_block-order03_kinect.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        print("Skipping video file demo.")
    else:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print("\n--- Opening selector with video file ---")
        try:
            root1 = tk.Tk()
            selector1 = VideoFrameSelector(
                root1,
                cap,
                last_frame=total_frames - 1, 
                current_frame_num=0, 
                title="Video File Example"
            )
            root1.mainloop()

            if selector1.proceed_was_clicked:
                print(f"Proceeding with frame from video selection: {selector1.selected_frame_num}")
            else:
                print("Video selection was cancelled.")
        except Exception as e:
            print(f"An error occurred during video selector execution: {e}")
        finally:
            # Release the video capture object
            cap.release()

    # --- DEMO 2: Using an array of frames ---
    print("\n--- Opening selector with a numpy array ---")
    # Create a dummy array of 200 frames (e.g., a color gradient)
    dummy_frames = []
    for i in range(200):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Create a gradient from blue to red
        frame[:, :, 0] = int(255 * (i / 199.0))  # Red channel
        frame[:, :, 2] = int(255 * (1 - i / 199.0)) # Blue channel
        dummy_frames.append(frame)

    try:
        root2 = tk.Tk()
        selector2 = VideoFrameSelector(
            root2,
            dummy_frames,
            current_frame_num=100,
            title="Array Example"
        )
        root2.mainloop()

        if selector2.proceed_was_clicked:
            print(f"Frame {selector2.selected_frame_num} selected from the array.")
        else:
            print("Array selection was cancelled.")
    except Exception as e:
        print(f"An error occurred during array selector execution: {e}")

