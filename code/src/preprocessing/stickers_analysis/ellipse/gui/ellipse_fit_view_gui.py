import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import pandas as pd
from typing import Dict, List

class EllipseFitViewGUI:
    """
    A Tkinter-based GUI for reviewing and visualizing multiple ellipse fitting results.

    This class displays a sequence of grayscale or color frames and overlays the
    corresponding fitted ellipses for each frame from a dictionary of pandas DataFrames.
    Each key in the dictionary represents a distinct object. It includes an option to apply a
    binary threshold and provides checkboxes to toggle the visibility of each object's
    ellipse.
    """
    # A palette of distinct BGR colors for drawing ellipses
    ELLIPSE_COLORS_BGR = [
        (0, 255, 0),    # Bright Green
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 165, 255),  # Orange
        (255, 255, 0),  # Yellow
        (128, 0, 128),  # Purple
        (0, 128, 255)   # Light Orange
    ]

    def __init__(self,
                 *,
                 ellipse_df: Dict[str, pd.DataFrame],
                 frames: np.ndarray,
                 default_threshold: int = 127,
                 title: str = "Ellipse Fit Review",
                 windowState: str = 'normal'):
        """
        Initializes the GUI View.

        Args:
            ellipse_df (Dict[str, pd.DataFrame]): Dictionary where keys are object names
                and values are DataFrames with ellipse parameters, indexed by frame_number.
            frames (np.ndarray): A numpy array of frames. Can be 3D for grayscale
                (frames, height, width) or 4D for BGR color (frames, height, width, channels).
            default_threshold (int): The initial value for the binary threshold.
            title (str): The title for the main window.
            windowState (str): Initial state of the window ('normal' or 'maximized').
        """
        self.ellipse_dfs = ellipse_df
        self.frames = frames
        self.num_frames = self.frames.shape[0]
        self.default_threshold = default_threshold
        self.title = title
        self.windowState = windowState

        # --- Object tracking and colors ---
        self.object_names = list(self.ellipse_dfs.keys())
        self.object_colors = {
            name: self.ELLIPSE_COLORS_BGR[i % len(self.ELLIPSE_COLORS_BGR)]
            for i, name in enumerate(self.object_names)
        }

        # --- Playback state ---
        self._is_paused = True
        self._update_job = None
        self._playback_delay_ms = int(1000 / 30) # Assume 30 FPS

        # --- Tkinter UI elements ---
        self.root = None
        self.image_label = None
        self.play_pause_button = None
        self.timeline_scale = None
        self.current_frame_label = None
        self.threshold_check_button = None
        self.threshold_spinbox = None
        self.object_visibility_vars: Dict[str, tk.BooleanVar] = {}

    def setup_ui(self):
        """Creates and arranges all the Tkinter widgets."""
        self.root = tk.Tk()
        self.root.title(self.title)

        # --- Initialize Tkinter variables ---
        self.scale_var = tk.IntVar(value=0)
        self.threshold_enabled_var = tk.BooleanVar(value=False)
        self.threshold_value_var = tk.IntVar(value=self.default_threshold)

        # Listeners to update the view when controls change
        redraw_on_change = lambda *_: self.seek_to_frame(self.scale_var.get())
        self.threshold_enabled_var.trace_add("write", redraw_on_change)
        self.threshold_value_var.trace_add("write", redraw_on_change)

        # --- Window Positioning ---
        if self.windowState.upper() == 'NORMAL':
            window_width, window_height = 1280, 800
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            center_x = int(screen_width / 2 - window_width / 2)
            center_y = int(screen_height / 2 - window_height / 2)
            self.root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        elif self.windowState.upper() in ['MAXIMIZED', 'MAXIMISED']:
            self.root.state('zoomed')

        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # --- Main Layout Frame ---
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)

        # --- Video Display Area ---
        self.image_label = ttk.Label(main_frame)
        self.image_label.grid(row=0, column=0, sticky="nsew")

        # --- Timeline Slider ---
        self.timeline_scale = ttk.Scale(
            main_frame, from_=0, to=self.num_frames - 1,
            orient=tk.HORIZONTAL, variable=self.scale_var,
            command=self.seek_to_frame
        )
        self.timeline_scale.grid(row=1, column=0, sticky="ew", pady=5)

        # --- Controls Container ---
        controls_container = ttk.Frame(main_frame)
        controls_container.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        controls_container.columnconfigure(3, weight=1) # Make empty space expand

        # --- Playback Controls ---
        self.play_pause_button = ttk.Button(controls_container, text="▶ Play", command=self.toggle_play_pause, width=10)
        self.play_pause_button.grid(row=0, column=0, padx=5, sticky="w")

        # --- Frame Info Label ---
        self.current_frame_label = ttk.Label(controls_container, text=f"Frame: 0 / {self.num_frames - 1}")
        self.current_frame_label.grid(row=0, column=1, padx=10, sticky="w")

        # --- Object Visibility Checkboxes ---
        visibility_frame = ttk.LabelFrame(controls_container, text="Objects")
        visibility_frame.grid(row=0, column=2, padx=10, sticky="w")
        for i, name in enumerate(self.object_names):
            self.object_visibility_vars[name] = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(
                visibility_frame,
                text=name,
                variable=self.object_visibility_vars[name]
            )
            cb.pack(side=tk.LEFT, padx=5)
            # Add a color swatch for easy identification
            color_hex = '#%02x%02x%02x' % (self.object_colors[name][2], self.object_colors[name][1], self.object_colors[name][0])
            color_label = tk.Label(visibility_frame, text="●", fg=color_hex, font=("", 12))
            color_label.pack(side=tk.LEFT, padx=(0, 5))
            self.object_visibility_vars[name].trace_add("write", redraw_on_change)

        # --- Threshold Controls ---
        threshold_frame = ttk.Frame(controls_container)
        threshold_frame.grid(row=0, column=4, sticky="e")

        self.threshold_check_button = ttk.Checkbutton(
            threshold_frame, text="Apply Binary Threshold:", variable=self.threshold_enabled_var
        )
        self.threshold_check_button.pack(side=tk.LEFT, padx=5)

        self.threshold_spinbox = ttk.Spinbox(
            threshold_frame, from_=0, to=255, width=5, textvariable=self.threshold_value_var
        )
        self.threshold_spinbox.pack(side=tk.LEFT)

        self._bind_keys()
        self.root.after(50, lambda: self.seek_to_frame(0)) # Display first frame

    def _bind_keys(self):
        """Binds keyboard shortcuts to playback methods."""
        self.root.bind('<Left>', lambda e: self.seek_to_frame(self.scale_var.get() - 1))
        self.root.bind('<Right>', lambda e: self.seek_to_frame(self.scale_var.get() + 1))
        self.root.bind('<Control-Left>', lambda e: self.seek_to_frame(self.scale_var.get() - 10))
        self.root.bind('<Control-Right>', lambda e: self.seek_to_frame(self.scale_var.get() + 10))
        self.root.bind('<space>', lambda e: self.toggle_play_pause())

    def start(self):
        """Starts the Tkinter event loop."""
        self.setup_ui()
        self.root.mainloop()

    def quit(self):
        """Stops the playback loop and destroys the Tkinter window."""
        if self._update_job:
            self.root.after_cancel(self._update_job)
        self.root.destroy()

    # --- Playback Control Methods ---

    def toggle_play_pause(self):
        """Toggles the video playback state between playing and paused."""
        self._is_paused = not self._is_paused
        self.play_pause_button.config(text="▶ Play" if self._is_paused else "❚❚ Pause")
        if not self._is_paused:
            self._update_playback_loop()

    def seek_to_frame(self, frame_val):
        """Seeks to a specific frame number."""
        try:
            frame_num = int(float(frame_val))
            if not (0 <= frame_num < self.num_frames):
                return

            self.scale_var.set(frame_num)
            self._update_ui_for_frame(frame_num)

        except (ValueError, IndexError):
            pass # Ignore invalid inputs

    def _update_playback_loop(self):
        """The core loop that drives video playback."""
        if self._is_paused:
            return

        current_frame = self.scale_var.get()
        next_frame = current_frame + 1

        if next_frame < self.num_frames:
            self.seek_to_frame(next_frame)
            self._update_job = self.root.after(self._playback_delay_ms, self._update_playback_loop)
        else:
            self.seek_to_frame(self.num_frames - 1)
            self._is_paused = True
            self.play_pause_button.config(text="▶ Play")

    # --- UI Update Methods ---

    def _update_ui_for_frame(self, frame_num: int):
        """Updates all UI elements for a given frame number."""
        self.update_frame_label(frame_num)
        self.update_video_display(frame_num)

    def update_frame_label(self, current_num: int):
        """Updates the 'Frame X / Y' text label."""
        self.current_frame_label.config(text=f"Frame: {current_num} / {self.num_frames - 1}")

    def update_video_display(self, frame_num: int):
        """Fetches a frame, draws all visible ellipses on it, and displays it."""
        # 1. Get raw frame and determine if it's color
        raw_frame = self.frames[frame_num].copy()
        is_color = raw_frame.ndim == 3 and raw_frame.shape[2] in [3, 4]

        # 2. Process frame: apply threshold if enabled, ensure BGR format for drawing
        if self.threshold_enabled_var.get():
            try:
                thresh_val = self.threshold_value_var.get()
                # For thresholding, we always work with a grayscale image
                if is_color:
                    frame_gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
                else:
                    frame_gray = raw_frame

                _, processed_frame = cv2.threshold(frame_gray, thresh_val, 255, cv2.THRESH_BINARY)
                # Convert the binary image back to BGR to draw colored ellipses
                frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
            except (ValueError, tk.TclError):
                # Fallback on error: use original frame, ensuring it's BGR
                frame_bgr = raw_frame if is_color else cv2.cvtColor(raw_frame, cv2.COLOR_GRAY2BGR)
        else:
            # No threshold: just ensure the frame is BGR for drawing
            frame_bgr = raw_frame if is_color else cv2.cvtColor(raw_frame, cv2.COLOR_GRAY2BGR)

        # 3. Iterate through each object and draw its ellipse if visible
        for object_name, df in self.ellipse_dfs.items():
            if not self.object_visibility_vars[object_name].get():
                continue # Skip if object is not visible

            color = self.object_colors[object_name]
            ellipse_data = df[df['frame_number'] == frame_num]

            if not ellipse_data.empty and not ellipse_data.isnull().values.any():
                data = ellipse_data.iloc[0]
                center = (int(data['center_x']), int(data['center_y']))
                axes = (int(data['axes_major'] / 2), int(data['axes_minor'] / 2))
                angle = int(data['angle'])

                # Draw the ellipse
                cv2.ellipse(frame_bgr, center, axes, angle, 0, 360, color, 2)

                # Draw the object name and score
                score = data['score']
                text = f"{object_name}: {score:.3f}"
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                text_origin = (center[0] - text_width // 2, center[1] - axes[1] - 10)
                cv2.putText(frame_bgr, text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 4. Resize and display the final image
        container_w = self.image_label.winfo_width()
        container_h = self.image_label.winfo_height()
        if container_w < 50 or container_h < 50: return

        original_h, original_w, _ = frame_bgr.shape
        scale = min(container_w / original_w, container_h / original_h)
        new_w, new_h = int(original_w * scale), int(original_h * scale)

        if new_w > 0 and new_h > 0:
            resized_frame = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            img_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            self.image_label.imgtk = imgtk
            self.image_label.config(image=imgtk)


# --- Example Usage ---
if __name__ == '__main__':
    # 1. Create dummy data to simulate the inputs for multiple objects

    num_frames = 200
    frame_height, frame_width = 80, 150

    # Data for the first object
    data_a = {
        'frame_number': range(num_frames),
        'center_x': [75 + np.cos(i / 15) * 40 for i in range(num_frames)],
        'center_y': [40 + np.sin(i / 15) * 15 for i in range(num_frames)],
        'axes_major': [35 + np.sin(i / 10) * 5 for i in range(num_frames)],
        'axes_minor': [20 - np.sin(i / 10) * 3 for i in range(num_frames)],
        'angle': [(i * 2) % 180 for i in range(num_frames)],
        'score': [0.95 - np.random.rand() * 0.05 for _ in range(num_frames)]
    }
    df_a = pd.DataFrame(data_a)

    # Data for the second object
    data_b = {
        'frame_number': range(num_frames),
        'center_x': [30 + i * 0.5 for i in range(num_frames)],
        'center_y': [20 for i in range(num_frames)],
        'axes_major': [25 for i in range(num_frames)],
        'axes_minor': [15 for i in range(num_frames)],
        'angle': [45 for i in range(num_frames)],
        'score': [0.88 - np.random.rand() * 0.05 for _ in range(num_frames)]
    }
    df_b = pd.DataFrame(data_b)
    # Add some NaN values to test robustness
    df_b.loc[50:60, ['center_x', 'score']] = np.nan

    # Combine into the required dictionary structure
    ellipse_data_dict = {
        "Pupil": df_a,
        "Reflection": df_b
    }

    # Create dummy BGR color frames with a moving, color-changing rectangle
    frames_bgr = np.zeros((num_frames, frame_height, frame_width, 3), dtype=np.uint8)
    for i in range(num_frames):
        start_x = int(10 + 5 * np.sin(i / 10))
        start_y = int(frame_height / 2 + 20 * np.cos(i/25))
        # Dynamically change the blue and green channels of the rectangle color
        color = (
            int(127 + 127 * np.sin(i / 15)), # Blue channel
            int(127 + 127 * np.cos(i / 20)), # Green channel
            50                               # Red channel
        )
        cv2.rectangle(frames_bgr[i], (start_x, start_y), (start_x + 25, start_y + 15), color, -1)


    # 2. Instantiate and run the GUI with color frames
    gui = EllipseFitViewGUI(
        ellipse_df=ellipse_data_dict,
        frames=frames_bgr,
        title="Multi-Object Ellipse Visualization (Color Video)",
        windowState='maximized'
    )
    gui.start()