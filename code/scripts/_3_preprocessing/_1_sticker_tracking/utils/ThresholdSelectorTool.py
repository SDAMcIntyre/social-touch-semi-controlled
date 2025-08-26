import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from typing import List, Optional, Tuple

class ThresholdSelectorTool:
    """
    A Tkinter-based GUI tool to interactively select a threshold value from video frames.
    
    This class provides a user interface where an original frame and its thresholded
    version are displayed side-by-side. Users can navigate through frames and adjust
    the threshold using sliders. The selection is confirmed or cancelled with buttons.
    """

    def __init__(
            self, 
            frames: List[np.ndarray], 
            video_name: str = "video", 
            threshold: int = 127,
            spot_type: str = 'dark'):
        """
        Initializes the Threshold Selector Tool.

        Args:
            frames (List[np.ndarray]): A list of video frames (BGR or Grayscale).
            video_name (str): The name of the video file, for display purposes.
            spot_type (str): The type of feature to isolate ('dark' or 'light').
        """
        if not frames:
            raise ValueError("The provided frame list cannot be empty.")
            
        self.frames = frames
        self.spot_type = spot_type
        self.total_frames = len(self.frames)
        self.video_name = video_name

        self.init_threshold = threshold

        # --- State variables ---
        self.result: Optional[int] = None

        # --- Tkinter UI elements ---
        self.root: Optional[tk.Tk] = None
        self.original_img_label: Optional[ttk.Label] = None
        self.thresholded_img_label: Optional[ttk.Label] = None
        self.image_pane: Optional[ttk.Frame] = None
        
        # Tkinter variables to link with sliders
        self.frame_var: Optional[tk.IntVar] = None
        self.thresh_var: Optional[tk.IntVar] = None
        self.title_var: Optional[tk.StringVar] = None

        # Variable to store the calculated target display size for one image
        self.target_display_size: Optional[Tuple[int, int]] = None

    def select_threshold(self) -> Optional[int]:
        """
        Creates and runs the GUI application, then returns the selected value.
        """
        self.root = tk.Tk()
        self.root.title("Interactive Threshold Selection")
        self.root.protocol("WM_DELETE_WINDOW", self._on_cancel)

        self.frame_var = tk.IntVar(value=0)
        self.thresh_var = tk.IntVar(value=self.init_threshold)
        self.title_var = tk.StringVar()

        self._setup_ui()

        # Start window maximized and in the foreground
        self.root.state('zoomed')
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after_idle(self.root.attributes, '-topmost', False)
        self.root.update_idletasks()

        # Calculate the available space for each image panel
        pane_width = self.image_pane.winfo_width()
        pane_height = self.image_pane.winfo_height()
        self.target_display_size = (pane_width // 2, pane_height)

        self._update_display()

        print("\n--- Interactive Threshold Selection ---")
        print("An interactive window has been opened.")
        print("Please adjust the sliders and use the buttons to finalize your choice.")
        print("-------------------------------------\n")

        self.root.mainloop()

        return self.result

    def _setup_ui(self):
        """Creates and arranges all the Tkinter widgets."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # Title label
        title_label = ttk.Label(main_frame, textvariable=self.title_var, font=("Helvetica", 14, "bold"))
        title_label.grid(row=0, column=0, sticky="w", pady=(0, 10))

        # Image Display Area
        self.image_pane = ttk.Frame(main_frame)
        self.image_pane.grid(row=1, column=0, sticky="nsew", padx=5, pady=5) 
        main_frame.rowconfigure(1, weight=1) 
        main_frame.columnconfigure(0, weight=1)

        self.original_img_label = ttk.Label(self.image_pane)
        self.original_img_label.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(0, 5))
        
        self.thresholded_img_label = ttk.Label(self.image_pane)
        self.thresholded_img_label.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(5, 0))

        # Controls Area
        controls_pane = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        controls_pane.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        controls_pane.columnconfigure(1, weight=1)

        # Frame Slider
        ttk.Label(controls_pane, text="Frame:").grid(row=0, column=0, sticky="w")
        frame_slider = ttk.Scale(controls_pane, from_=0, to=self.total_frames - 1,
                                  orient=tk.HORIZONTAL, variable=self.frame_var,
                                  command=self._on_slider_change)
        frame_slider.grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Label(controls_pane, textvariable=self.frame_var, width=4).grid(row=0, column=2)

        # Threshold Slider
        ttk.Label(controls_pane, text="Threshold:").grid(row=1, column=0, sticky="w")
        thresh_slider = ttk.Scale(controls_pane, from_=0, to=255,
                                  orient=tk.HORIZONTAL, variable=self.thresh_var,
                                  command=self._on_slider_change)
        thresh_slider.grid(row=1, column=1, sticky="ew", padx=5)
        ttk.Label(controls_pane, textvariable=self.thresh_var, width=4).grid(row=1, column=2)

        # Action Buttons
        button_pane = ttk.Frame(main_frame)
        button_pane.grid(row=3, column=0, sticky="e", pady=(10, 0))
        
        confirm_button = ttk.Button(button_pane, text="Confirm (Enter)", command=self._on_confirm)
        confirm_button.pack(side=tk.RIGHT, padx=5)
        
        cancel_button = ttk.Button(button_pane, text="Cancel (Esc)", command=self._on_cancel)
        cancel_button.pack(side=tk.RIGHT)
        
        # --- Key Bindings ---
        self.root.bind('<Return>', lambda e: self._on_confirm())
        self.root.bind('<Escape>', lambda e: self._on_cancel())

        # --- MODIFICATION --- Added arrow key bindings for frame navigation
        self.root.bind('<Left>', lambda e: self._increment_frame(-1))
        self.root.bind('<Right>', lambda e: self._increment_frame(1))
        self.root.bind('<Control-Left>', lambda e: self._increment_frame(-10))
        self.root.bind('<Control-Right>', lambda e: self._increment_frame(10))

    def _increment_frame(self, step: int):
        """
        Changes the current frame by the given step value.
        
        Args:
            step (int): The number of frames to move. Can be positive or negative.
        """
        current_frame = self.frame_var.get()
        # Calculate the new frame index, clamping it within the valid range
        new_frame = max(0, min(current_frame + step, self.total_frames - 1))
        
        # Avoid unnecessary updates if the frame doesn't change (e.g., at the boundaries)
        if new_frame != current_frame:
            self.frame_var.set(new_frame)
            # Manually trigger the update since .set() doesn't call the slider's command
            self._on_slider_change()
    
    def _on_slider_change(self, _=None):
        """Callback triggered when any slider value changes."""
        self._update_display()

    def _update_display(self):
        """Processes and displays the original and thresholded frames."""
        frame_idx = self.frame_var.get()
        threshold_val = self.thresh_var.get()

        self.title_var.set(f"Video: {self.video_name}  |  Frame: {frame_idx}/{self.total_frames - 1}  |  Threshold: {threshold_val}")
        
        original_frame = self.frames[frame_idx]
        if len(original_frame.shape) == 3:
            gray_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = original_frame.copy()

        thresh_type = cv2.THRESH_BINARY_INV if self.spot_type == 'dark' else cv2.THRESH_BINARY
        _, thresholded_frame = cv2.threshold(gray_frame, threshold_val, 255, thresh_type)

        display_gray = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        display_thresh = cv2.cvtColor(thresholded_frame, cv2.COLOR_GRAY2BGR)

        self._update_image_label(self.original_img_label, display_gray)
        self._update_image_label(self.thresholded_img_label, display_thresh)

    def _update_image_label(self, label: ttk.Label, frame: np.ndarray):
        """
        Resizes an OpenCV image to fit the target display area, converts it to a
        PhotoImage, and updates the given label.
        """
        if self.target_display_size:
            target_w, target_h = self.target_display_size
            h, w = frame.shape[:2]
            if w > 0 and h > 0:
                scale = min(target_w / w, target_h / h)
                if scale < 1.0 or (target_w > w and target_h > h):
                    new_w, new_h = int(w * scale), int(h * scale)
                    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        
        label.imgtk = imgtk
        label.config(image=imgtk)

    def _on_confirm(self):
        """Stores the result and closes the application."""
        self.result = self.thresh_var.get()
        print(f"✅ Threshold value {self.result} confirmed.")
        self.root.destroy()

    def _on_cancel(self):
        """Sets the result to None and closes the application."""
        self.result = None
        print("Threshold selection cancelled by user.")
        if self.root:
            self.root.destroy()