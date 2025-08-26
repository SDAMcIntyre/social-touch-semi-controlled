import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import sys        
from pathlib import Path
from enum import Enum


class TrackerReviewStatus(Enum):
    """Defines status constants for tracking and annotation."""
    PROCEED = "proceed"
    COMPLETED = "completed"
    UNPERFECT = "unperfect"


class ObjectTrackerReviewTool:
    """
    Handles interactive video playback using a Tkinter GUI.
    It allows for frame-by-frame review, annotation, and provides
    advanced video controls including frame seeking via slider, entry box,
    and arrow keys. Includes a panel for managing marked frames for
    multiple objects. The tracking history is optional.
    """
    STATUS_COLORS = {
        "Tracking": (0, 255, 0),      # Green for BGR
        "Out of Frame": (0, 0, 255),  # Red for BGR
        "Failure": (0, 165, 255),     # Orange for BGR
        "Re-initialized": (0, 165, 255), # Orange for BGR
        "default": (255, 0, 0)        # Blue for BGR
    }

    def __init__(self, 
                 video_source, 
                 tracking_history=None, 
                 landmarks=None, 
                 landmark_properties=None, 
                 *, 
                 as_bgr=True, 
                 title="Loaded Frames Review"):
        """
        Initializes the video player.

        Args:
            video_source (str or list or np.ndarray): Path to the video file or a list/array of pre-loaded frames.
            tracking_history (list, optional): A list where each element corresponds to a frame.
                                                 Each element is a list of dictionaries, with each
                                                 dictionary representing a tracked object. Defaults to None.
            landmarks (list[int], optional): A vector storing frame_ids to be shown as marks on the timeline.
                                               Defaults to None.
            landmark_properties (dict, optional): Properties for the landmark marks.
                                                    Keys: 'color' (str), 'thickness' (int), 'height' (int).
                                                    Defaults to {'color': 'blue', 'thickness': 2, 'height': 10}.
            as_bgr (bool): If True, frames are treated as BGR. If False, as RGB.
                           This applies to both loading from a file and receiving a list of frames.
        """
        self._is_bgr = as_bgr
        self.frames = None
        self.video_path = None

        if isinstance(video_source, (str, Path)):
            self.video_path = str(video_source)
            self.title = os.path.basename(self.video_path)
            self.load_video_frames()  # This populates self.frames and frame properties
        elif isinstance(video_source, (list, np.ndarray)):
            self.frames = list(video_source)
            if not self.frames:
                raise ValueError("The provided frame list cannot be empty.")
            self.total_frames = len(self.frames)
            self.frame_height, self.frame_width = self.frames[0].shape[:2]
            self.title = title
        else:
            raise TypeError("video_source must be a string (path) or a list/numpy array of frames.")
        
        self.tracking_history = tracking_history if tracking_history is not None else []
        self.landmarks = landmarks if landmarks is not None else []
        
        # Set default landmark properties and override with user-provided ones
        default_props = {'color': 'blue', 'thickness': 2, 'height': 10}
        if landmark_properties:
            default_props.update(landmark_properties)
        self.landmark_properties = default_props
        
        # --- State variables to be returned ---
        self.status = TrackerReviewStatus.PROCEED
        self.marked_for_labeling = []

        # --- Player control variables ---
        self.paused = True # Start paused
        self.current_frame_num = 0
        self.playback_speed = 1.0
        self.base_delay_ms = 30 # Base delay for 1x speed, corresponds to ~33fps

        # --- Tkinter UI elements (to be initialized later) ---
        self.root = None
        self.video_frame = None 
        self.image_label = None
        self.play_pause_button = None
        self.timeline_canvas = None
        self.timeline_scale = None
        self.frame_entry = None
        self.current_frame_label = None
        self.scale_var = None
        self.entry_var = None
        self.valid_button = None
        self.proceed_button = None
        self.marked_listbox = None
        self.speed_slider = None
        self.speed_label = None
        self.speed_var = None

    @property
    def is_bgr(self):
        """Property to check if frames are in BGR format."""
        return self._is_bgr

    @property
    def is_rgb(self):
        """Property to check if frames are in RGB format."""
        return not self._is_bgr

    def load_video_frames(self):
        """Wrapper for the frame loading logic."""
        self.frames = self._load_video_frames()
        if not self.frames:
            raise IOError(f"Cannot open or read video at {self.video_path}")
        self.total_frames = len(self.frames)
        self.frame_height, self.frame_width = self.frames[0].shape[:2]

    def _load_video_frames(self):
        """
        Loads all frames from the video into a list of images with progress feedback,
        respecting the color format set during initialization.
        """
        video_capture = cv2.VideoCapture(self.video_path)
        if not video_capture.isOpened():
            raise IOError(f"Error: Could not open video file at {self.video_path}")

        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            print("Warning: Video contains no frames.")
            return []

        frames = []
        processed_frames = 0
        print("Starting video processing...")

        while True:
            success, frame = video_capture.read()
            if not success:
                break
            
            # Check the instance's color format property
            if self.is_rgb:
                # Convert the frame from BGR to RGB and then append
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                # Append the frame as is (BGR)
                frames.append(frame)
            
            processed_frames += 1

            # Calculate and display the progress
            percentage = (processed_frames / total_frames) * 100
            sys.stdout.write(f"\rLoading frames: {processed_frames}/{total_frames} frames ({percentage:.2f}%)")
            sys.stdout.flush()

        video_capture.release()
        print("\nVideo processing complete.")
        return frames
    
    def play(self, windowState='normal'):
        """
        Starts the interactive video playback session.
        This will open a Tkinter window with the video and controls.
        The method blocks until the window is closed.

        Returns:
            tuple: A tuple containing the final status (str) and a list of
                   frame numbers marked for labeling (list[int]).
        """
        self.root = tk.Tk()
        self.root.title(f"Video Replay: {self.title}")

        # --- Logic to Position the Window ---
        if windowState.upper() == 'NORMAL':
            window_width = 1024
            window_height = 768

            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()

            center_x = int(screen_width/2 - window_width / 2)
            center_y = int(screen_height/2 - window_height / 2)

            self.root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

        elif windowState.upper() == 'MAXIMIZED' or windowState.upper() == 'MAXIMISED':
            self.root.state('zoomed')

        self.root.protocol("WM_DELETE_WINDOW", self._on_quit)

        self._setup_ui()
        self._update_button_states() # Set initial button states
        self.root.bind('<Configure>', lambda e: self._display_current_frame())
        self.root.after(10, self._display_current_frame) 
        self.root.mainloop()

        print("\nPlayback finished.")
        print(f"Final Status: {self.status.value}")
        print(f"Frames marked for labeling: {self.marked_for_labeling}")
        return self.status, self.marked_for_labeling

    def _setup_ui(self):
        """Creates and arranges all the Tkinter widgets."""
        style = ttk.Style(self.root)
        style.configure("Mark.TButton", foreground="orange")
        style.configure("Delete.TButton", foreground="red")
        style.configure("Finish.TButton", foreground="blue")
        style.configure("Valid.TButton", foreground="green")

        main_paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        player_pane = ttk.Frame(main_paned_window)
        main_paned_window.add(player_pane, weight=3)

        list_pane = ttk.Frame(main_paned_window)
        main_paned_window.add(list_pane, weight=1)

        player_pane.rowconfigure(0, weight=1)
        player_pane.columnconfigure(0, weight=1)

        self.scale_var = tk.IntVar(value=self.current_frame_num)
        self.entry_var = tk.StringVar(value=str(self.current_frame_num))
        self.speed_var = tk.DoubleVar(value=self.playback_speed)

        self.video_frame = ttk.Frame(player_pane)
        self.video_frame.grid(row=0, column=0, sticky="nsew", pady=5)
        self.video_frame.rowconfigure(0, weight=1)
        self.video_frame.columnconfigure(0, weight=1)
        
        self.image_label = ttk.Label(self.video_frame)
        self.image_label.grid(row=0, column=0, sticky="nsew")
        
        # --- Timeline with Landmarks ---
        # The canvas will contain both the landmarks and the slider itself
        canvas_height = self.landmark_properties['height'] + 15
        self.timeline_canvas = tk.Canvas(player_pane, height=canvas_height, bd=0, highlightthickness=0)
        self.timeline_canvas.grid(row=1, column=0, sticky="ew", pady=5)

        self.timeline_scale = ttk.Scale(
            self.timeline_canvas, from_=0, to=self.total_frames - 1,
            orient=tk.HORIZONTAL, variable=self.scale_var, command=self._on_scale_drag
        )
        
        # Embed the scale inside the canvas, below the space for landmarks
        slider_y_pos = self.landmark_properties['height'] + 2
        self.timeline_canvas.create_window(0, slider_y_pos, window=self.timeline_scale, anchor='nw', tags="scale_widget")
        self.timeline_canvas.bind("<Configure>", self._on_canvas_resize)
        # --- End of Timeline with Landmarks ---

        info_frame = ttk.Frame(player_pane)
        info_frame.grid(row=2, column=0, sticky="ew")
        info_frame.columnconfigure(1, weight=1)

        self.current_frame_label = ttk.Label(info_frame, text=f"Frame: 0 / {self.total_frames - 1}")
        self.current_frame_label.grid(row=0, column=0, sticky="w")

        self.frame_entry = ttk.Entry(info_frame, textvariable=self.entry_var, width=10)
        self.frame_entry.grid(row=0, column=2, sticky="e", padx=5)
        self.frame_entry.bind("<Return>", self._update_from_entry)
        self.frame_entry.bind("<FocusOut>", self._update_from_entry)

        # --- Controls Container ---
        controls_container = ttk.Frame(player_pane)
        controls_container.grid(row=3, column=0, pady=10, sticky="ew")

        action_controls_frame = ttk.Frame(controls_container)
        action_controls_frame.pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.play_pause_button = ttk.Button(action_controls_frame, text="Play", command=self._on_play_pause)
        self.play_pause_button.pack(side=tk.LEFT, padx=5)

        # --- Speed Control ---
        speed_frame = ttk.Frame(action_controls_frame)
        speed_frame.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)

        self.speed_label = ttk.Label(speed_frame, text=f"Speed: {self.playback_speed:.1f}x")
        self.speed_label.pack(side=tk.LEFT)

        self.speed_slider = ttk.Scale(
            speed_frame, from_=0.5, to=8.0,
            orient=tk.HORIZONTAL, variable=self.speed_var, command=self._on_speed_change
        )
        self.speed_slider.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        # --- End of Speed Control ---

        ttk.Button(action_controls_frame, text="✏️ Mark for Labeling", command=self._on_mark, style="Mark.TButton").pack(side=tk.LEFT, padx=5)

        finish_controls_frame = ttk.LabelFrame(controls_container, text="Finish")
        finish_controls_frame.pack(side=tk.RIGHT)

        self.proceed_button = ttk.Button(finish_controls_frame, text="➡️ Proceed with Marked", command=self._on_proceed, style="Finish.TButton")
        self.proceed_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.valid_button = ttk.Button(finish_controls_frame, text="✅ Finish as Valid", command=self._on_valid, style="Valid.TButton")
        self.valid_button.pack(side=tk.LEFT, padx=5, pady=5)

        # --- List Pane ---
        list_pane.rowconfigure(1, weight=1)
        list_pane.columnconfigure(0, weight=1)
        ttk.Label(list_pane, text="Marked Frames", font=("", 10, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0,5))
        self.marked_listbox = tk.Listbox(list_pane, height=10)
        self.marked_listbox.grid(row=1, column=0, columnspan=2, sticky="nsew")
        self.marked_listbox.bind('<<ListboxSelect>>', self._on_listbox_select)
        list_scrollbar = ttk.Scrollbar(list_pane, orient="vertical", command=self.marked_listbox.yview)
        list_scrollbar.grid(row=1, column=2, sticky="ns")
        self.marked_listbox['yscrollcommand'] = list_scrollbar.set
        delete_button = ttk.Button(list_pane, text="Delete Selected", command=self._delete_from_listbox, style="Delete.TButton")
        delete_button.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(5,0))

        # --- Bind Keys ---
        self.root.bind('<Left>', self._handle_arrow_keys)
        self.root.bind('<Right>', self._handle_arrow_keys)
        self.root.bind('<Control-Left>', self._handle_arrow_keys)
        self.root.bind('<Control-Right>', self._handle_arrow_keys)
        self.root.bind('<space>', lambda e: self._on_play_pause())
        self.root.bind('<Delete>', lambda e: self._delete_from_listbox())

    def _on_speed_change(self, value):
        """Updates the playback speed and label when the speed slider is moved."""
        self.playback_speed = self.speed_var.get()
        self.speed_label.config(text=f"Speed: {self.playback_speed:.1f}x")

    def _on_canvas_resize(self, event):
        """Handles resizing of the timeline canvas, updating scale and landmarks."""
        canvas_width = event.width
        # Update the width of the scale widget to fill the canvas
        self.timeline_canvas.itemconfigure("scale_widget", width=canvas_width)
        # Redraw landmarks for the new size
        self._draw_landmarks()

    def _draw_landmarks(self):
        """Draws or redraws the landmark rectangles on the timeline canvas."""
        # Delete old landmarks to prevent drawing over them
        self.timeline_canvas.delete("landmark")
        
        if not self.landmarks:
            return

        canvas_width = self.timeline_canvas.winfo_width()
        # Canvas might not be rendered yet, so width is 1
        if canvas_width <= 1:
            return

        # Get properties from the dictionary
        color = self.landmark_properties['color']
        thickness = self.landmark_properties['thickness']
        height = self.landmark_properties['height']

        for frame_id in self.landmarks:
            # Calculate the x-position as a ratio of the canvas width
            # Check for total_frames > 1 to avoid division by zero
            x_ratio = frame_id / (self.total_frames - 1) if self.total_frames > 1 else 0
            x_pos = x_ratio * canvas_width

            # Draw the rectangle using the landmark tag for easy deletion
            self.timeline_canvas.create_rectangle(
                x_pos, 0, x_pos + thickness, height,
                fill=color, outline=color, tags="landmark"
            )
            
    def _update_frame(self, force_update=False):
        """
        Handles the video playback loop. This function should only be
        called to start the loop; it perpetuates itself via `root.after`.
        """
        # If the video is paused, stop the loop.
        if self.paused and not force_update:
            return

        # --- Draw current frame and update labels ---
        self.current_frame_num = self.scale_var.get()
        self._display_current_frame()
        self.entry_var.set(str(self.current_frame_num))
        self.current_frame_label.config(text=f"Frame: {self.current_frame_num} / {self.total_frames - 1}")

        # --- Advance to the next frame ---
        next_frame_num = self.current_frame_num + 1
        if next_frame_num >= self.total_frames:
            # Reached the end. Pause the video and stop the loop.
            self.paused = True
            self.play_pause_button.config(text="Play")
        else:
            # Continue playback: update the slider and schedule the next frame.
            self.scale_var.set(next_frame_num)
            delay_ms = max(1, int(self.base_delay_ms / self.playback_speed))
            self.root.after(delay_ms, self._update_frame)


    def _display_current_frame(self):
        frame = self.frames[self.current_frame_num]
        if self.current_frame_num < len(self.tracking_history):
            objects_in_frame = self.tracking_history[self.current_frame_num]
            if isinstance(objects_in_frame, list):
                for obj_result in objects_in_frame:
                    status_text = obj_result.get("status", "No Data")
                    box = obj_result.get("box")
                    color = self.STATUS_COLORS.get(status_text.split(':')[0], self.STATUS_COLORS["default"])
                    if box:
                        x, y, w, h = [int(v) for v in box]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, status_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        self._display_image(frame)


    def _display_image(self, frame):
        container_w = self.video_frame.winfo_width()
        container_h = self.video_frame.winfo_height()
        if container_w < 50 or container_h < 50: return
        original_h, original_w, _ = frame.shape
        if original_w == 0 or original_h == 0: return
        scale = min(container_w / original_w, container_h / original_h)
        new_w, new_h = int(original_w * scale), int(original_h * scale)
        if new_w > 0 and new_h > 0:
            resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            cv2image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.image_label.imgtk = imgtk
            self.image_label.config(image=imgtk)

    def _on_play_pause(self):
        """Toggles the paused state and starts the update loop if playing."""
        # If we're at the end and press play, restart from the beginning.
        if self.paused and self.scale_var.get() >= self.total_frames - 1:
            self.scale_var.set(0)

        self.paused = not self.paused
        self.play_pause_button.config(text="Play" if self.paused else "Pause")

        # If we just switched to the "playing" state, kick off the update loop.
        # If we are pausing, the loop will terminate on its own.
        if not self.paused:
            self._update_frame()
    
    def _on_scale_drag(self, _=None):
        """
        Called when the user drags the timeline slider.
        This will pause the video and force a display update.
        """
        if not self.paused:
            self.paused = True
            self.play_pause_button.config(text="Play")
        
        # Manually update the display to give immediate feedback.
        # The playback loop is already stopped because self.paused is True.
        self.current_frame_num = self.scale_var.get()
        self._display_current_frame()
        self.entry_var.set(str(self.current_frame_num))
        self.current_frame_label.config(text=f"Frame: {self.current_frame_num} / {self.total_frames - 1}")

    def _update_from_entry(self, _=None):
        try:
            frame_num = int(self.entry_var.get())
            frame_num = max(0, min(frame_num, self.total_frames - 1))
            self.scale_var.set(frame_num)
            self._on_scale_drag()
        except ValueError:
            self.entry_var.set(str(self.scale_var.get()))

    def _handle_arrow_keys(self, event):
        current_frame = self.scale_var.get()
        is_ctrl_pressed = (event.state & 4) != 0
        step = 10 if is_ctrl_pressed else 1
        new_frame = current_frame - step if event.keysym == 'Left' else current_frame + step
        clamped_frame = max(0, min(new_frame, self.total_frames - 1))
        self.scale_var.set(clamped_frame)
        self._on_scale_drag()

    def _on_valid(self):
        self.status =  TrackerReviewStatus.COMPLETED
        self._on_quit()

    def _on_mark(self):
        self.status =  TrackerReviewStatus.UNPERFECT
        if self.current_frame_num not in self.marked_for_labeling:
            self.marked_for_labeling.append(self.current_frame_num)
            self.marked_for_labeling.sort()
            self._update_listbox()
        self._update_button_states()

    def _on_proceed(self):
        self.status =  TrackerReviewStatus.PROCEED
        self._on_quit()

    def _on_quit(self):
        if self.root:
            self.root.destroy()

    def _update_listbox(self):
        self.marked_listbox.delete(0, tk.END)
        for frame_num in self.marked_for_labeling:
            self.marked_listbox.insert(tk.END, f"Frame {frame_num}")

    def _update_button_states(self):
        is_list_empty = not self.marked_for_labeling
        if is_list_empty:
            self.valid_button.config(state=tk.NORMAL)
            self.proceed_button.config(state=tk.DISABLED)
        else:
            self.valid_button.config(state=tk.DISABLED)
            self.proceed_button.config(state=tk.NORMAL)

    def _on_listbox_select(self, event):
        selection_indices = self.marked_listbox.curselection()
        if not selection_indices: return
        selected_index = selection_indices[0]
        selected_frame = self.marked_for_labeling[selected_index]
        self.scale_var.set(selected_frame)
        self._on_scale_drag()

    def _delete_from_listbox(self):
        selection_indices = self.marked_listbox.curselection()
        if not selection_indices: return
        selected_index = selection_indices[0]
        del self.marked_for_labeling[selected_index]
        self._update_listbox()
        self._update_button_states()