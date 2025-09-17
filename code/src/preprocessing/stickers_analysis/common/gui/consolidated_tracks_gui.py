# tracks_review_gui.py

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np

# Import TYPE_CHECKING to create a conditional block for type hinting
from typing import TYPE_CHECKING, Dict, List

# This block is only 'True' for type checkers, not at runtime
if TYPE_CHECKING:
    from preprocessing.common import VideoMP4Manager
    from ..models.consolidated_tracks_manager import ConsolidatedTracksManager


class ConsolidatedTracksReviewGUI:
    """
    A Tkinter-based GUI for reviewing and visualizing object tracking results.

    This class displays a video and overlays bounding boxes (ROIs) and ellipses
    for tracked objects on each frame, based on data from a ConsolidatedTracksManager.
    It provides standard video playback controls like play/pause, a seekable timeline,
    and speed adjustments.
    """

    # Map color names to BGR tuples for OpenCV drawing
    BGR_COLOR_MAP = {
        "blue": (255, 0, 0), "green": (0, 255, 0), "red": (0, 0, 255),
        "yellow": (0, 255, 255), "orange": (0, 165, 255), "purple": (128, 0, 128),
        "pink": (203, 192, 255), "cyan": (255, 255, 0), "magenta": (255, 0, 255),
        "white": (255, 255, 255), "black": (0, 0, 0), "gray": (128, 128, 128),
        "grey": (128, 128, 128), "brown": (42, 42, 165), "violet": (226, 43, 138)
    }

    def __init__(self,
                 *,
                 video_manager: 'VideoMP4Manager',
                 tracks_manager: 'ConsolidatedTracksManager',
                 object_colors: Dict[str, str],
                 title: str = "Tracking Results Review",
                 windowState: str = 'normal'):
        """
        Initializes the GUI View.

        Args:
            video_manager (VideoMP4Manager): Manager for accessing video frames.
            tracks_manager (ConsolidatedTracksManager): Manager for accessing tracking data.
            object_colors (Dict[str, str]): A mapping from object names to color strings.
            title (str): The title for the main window.
            windowState (str): Initial state of the window ('normal' or 'maximized').
        """
        self.video_manager = video_manager
        self.tracks_manager = tracks_manager
        self.object_colors = {name: self.BGR_COLOR_MAP.get(color.lower(), (255, 255, 255))
                              for name, color in object_colors.items()}
        self.title = title
        self.windowState = windowState

        # --- Playback state ---
        self._is_paused = True
        self._update_job = None
        self._playback_delay_ms = int(1000 / self.video_manager.fps)

        # --- Tkinter UI elements ---
        self.root = None
        self.image_label = None
        self.play_pause_button = None
        self.timeline_scale = None
        self.current_frame_label = None
        self.speed_slider = None

        # Get a set of all frame numbers that have tracking data
        all_frames_df = self.tracks_manager.get_all_data()
        self.tracked_frame_indices = sorted(list(all_frames_df['frame_number'].unique()))


    def setup_ui(self):
        """Creates and arranges all the Tkinter widgets."""
        self.root = tk.Tk()
        self.root.title(self.title)

        # --- Initialize Tkinter variables now that root exists ---
        self.scale_var = tk.IntVar(value=0)
        self.speed_var = tk.DoubleVar(value=1.0)

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
        total_frames = len(self.video_manager)
        self.timeline_scale = ttk.Scale(
            main_frame, from_=0, to=total_frames - 1,
            orient=tk.HORIZONTAL, variable=self.scale_var,
            command=self.seek_to_frame
        )
        self.timeline_scale.grid(row=1, column=0, sticky="ew", pady=5)

        # --- Controls Container ---
        controls_container = ttk.Frame(main_frame)
        controls_container.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        controls_container.columnconfigure(2, weight=1) # Make speed slider expand

        # --- Playback Controls ---
        self.play_pause_button = ttk.Button(controls_container, text="▶ Play", command=self.toggle_play_pause, width=10)
        self.play_pause_button.grid(row=0, column=0, padx=5)

        # --- Frame Info Label ---
        self.current_frame_label = ttk.Label(controls_container, text=f"Frame: 0 / {total_frames - 1}")
        self.current_frame_label.grid(row=0, column=1, padx=10)
        
        # --- Speed Control ---
        speed_frame = ttk.Frame(controls_container)
        speed_frame.grid(row=0, column=2, sticky="ew")
        speed_frame.columnconfigure(1, weight=1)
        
        self.speed_label = ttk.Label(speed_frame, text="Speed: 1.0x")
        self.speed_label.pack(side=tk.LEFT, padx=(10, 5))

        self.speed_slider = ttk.Scale(
            speed_frame, from_=0.25, to=8.0,
            orient=tk.HORIZONTAL, variable=self.speed_var,
            command=self.change_speed
        )
        self.speed_slider.pack(side=tk.LEFT, expand=True, fill=tk.X)

        # --- Legend for Object Colors ---
        legend_frame = ttk.LabelFrame(controls_container, text="Legend")
        legend_frame.grid(row=0, column=3, padx=10, sticky="e")
        
        for i, (name, bgr_color) in enumerate(self.object_colors.items()):
            # Create a small colored square
            color_swatch = tk.Label(legend_frame, text="■", fg=f'#{bgr_color[2]:02x}{bgr_color[1]:02x}{bgr_color[0]:02x}', font=("", 12))
            color_swatch.grid(row=i, column=0, sticky="w")
            # Create the text label
            label = ttk.Label(legend_frame, text=f" {name}")
            label.grid(row=i, column=1, sticky="w", padx=(0, 5))

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
        """
        Seeks to a specific frame number.

        Args:
            frame_val (str or int): The frame number to navigate to.
        """
        try:
            frame_num = int(float(frame_val))
            if not (0 <= frame_num < len(self.video_manager)):
                return
            
            self.scale_var.set(frame_num)
            self._update_ui_for_frame(frame_num)

        except (ValueError, IndexError):
            pass # Ignore invalid inputs
            
    def change_speed(self, speed_val):
        """
        Adjusts the playback speed.
        
        Args:
            speed_val (str or float): The new speed multiplier.
        """
        speed = float(speed_val)
        base_delay = 1000 / self.video_manager.fps
        self._playback_delay_ms = int(base_delay / speed)
        self.speed_label.config(text=f"Speed: {speed:.2f}x")

    def _update_playback_loop(self):
        """The core loop that drives video playback."""
        if self._is_paused:
            return

        current_frame = self.scale_var.get()
        next_frame = current_frame + 1

        if next_frame < len(self.video_manager):
            self.seek_to_frame(next_frame)
            self._update_job = self.root.after(self._playback_delay_ms, self._update_playback_loop)
        else:
            # End of video, so pause
            self.seek_to_frame(len(self.video_manager) - 1)
            self._is_paused = True
            self.play_pause_button.config(text="▶ Play")


    # --- UI Update Methods ---

    def _update_ui_for_frame(self, frame_num: int):
        """Updates all UI elements for a given frame number."""
        self.update_frame_label(frame_num)
        self.update_video_display(frame_num)

    def update_frame_label(self, current_num: int):
        """Updates the 'Frame X / Y' text label."""
        total_frames = len(self.video_manager)
        self.current_frame_label.config(text=f"Frame: {current_num} / {total_frames - 1}")

    def update_video_display(self, frame_num: int):
        """
        Fetches a frame, draws tracking data on it, and displays it in the GUI.
        """
        # 1. Get the raw frame from the video manager (in BGR format)
        frame_bgr = self.video_manager[frame_num]

        # 2. Get all tracked items for the current frame
        tracked_items = self.tracks_manager.get_items_for_frame(frame_num)
        
        # 3. Draw annotations for each tracked item
        for object_name, data in tracked_items:
            color = self.object_colors.get(object_name, (255, 255, 255)) # Default to white

            # Draw ROI rectangle
            if all(k in data for k in ['roi_x', 'roi_y', 'roi_width', 'roi_height']):
                x1, y1 = int(data['roi_x']), int(data['roi_y'])
                x2, y2 = x1 + int(data['roi_width']), y1 + int(data['roi_height'])
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                # Add a label
                cv2.putText(frame_bgr, object_name, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw ellipse
            if all(k in data for k in ['ellipse_center_x', 'ellipse_center_y', 'axes_major', 'axes_minor', 'angle']):
                center = (int(data['ellipse_center_x']), int(data['ellipse_center_y']))
                axes = (int(data['axes_major'] / 2), int(data['axes_minor'] / 2))
                angle = int(data['angle'])
                cv2.ellipse(frame_bgr, center, axes, angle, 0, 360, color, 2)

        # 4. Resize frame to fit the GUI window
        container_w = self.image_label.winfo_width()
        container_h = self.image_label.winfo_height()
        if container_w < 50 or container_h < 50: return # Avoid division by zero on init

        original_h, original_w, _ = frame_bgr.shape
        scale = min(container_w / original_w, container_h / original_h)
        new_w, new_h = int(original_w * scale), int(original_h * scale)
        
        if new_w > 0 and new_h > 0:
            resized_frame = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # 5. Convert to format Tkinter can use and display
            img_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.image_label.imgtk = imgtk
            self.image_label.config(image=imgtk)