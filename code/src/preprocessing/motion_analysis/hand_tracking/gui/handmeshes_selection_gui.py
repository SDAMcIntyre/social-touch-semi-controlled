import logging
import pickle
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox
from typing import Set, Optional

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

# Import modules from the same package
from ..models.hand_tracking_data_manager import HandTrackingDataManager
from .handmesh_3d_viewer import Hand3DViewer
from preprocessing.common import VideoMP4Manager, ColorFormat

class HamerCheckupSelector:
    """
    Main Application: Combines Hand Tracking Visualization with Frame Selection Interface.
    """
    def __init__(self, root: tk.Tk, video_path: str, data_path: str, output_path: str, csv_path: str):
        self.root = root
        self.root.title("Hamer Checkup & Selector")
        self.selected_frames: Set[int] = set()
        self.output_path = Path(output_path)
        
        # --- State Variable for Exit Logic ---
        # False = Just Save, True = Save and generate Success flag
        self.is_task_validated = False
        
        # Initialize placeholders for safe cleanup
        self.video_manager: Optional[VideoMP4Manager] = None
        self.viewer_3d: Optional[Hand3DViewer] = None

        # 1. Initialize Managers
        try:
            self.video_manager = VideoMP4Manager(video_path, color_format=ColorFormat.RGB)
            self.data_manager = HandTrackingDataManager(data_path)
        except Exception as e:
            messagebox.showerror("Initialization Error", str(e))
            # If init fails, we destroy immediately. Caller must handle this gracefully.
            self.cleanup()
            self.root.destroy()
            return

        # 2. State Variables
        self.total_frames = len(self.video_manager)
        self.current_frame_idx = 0
        self.is_playing = False
        self.overlay_enabled = tk.BooleanVar(value=True)
        self.trial_on_data: Optional[np.ndarray] = None

        # 3. Load CSV Data
        self._load_csv_data(csv_path)

        # 4. Setup UI (Video Frames Selector Style)
        self._setup_ui()

        # 5. Initialize 3D Viewer
        self.viewer_3d = Hand3DViewer(self.root)

        # 6. Load Existing Progress
        self._load_previous_selections()
        
        # 7. Initial Render
        self._update_display()
        
        # 8. Protocol handling for window 'X' button
        self.root.protocol("WM_DELETE_WINDOW", self._on_save_exit)

    def cleanup(self):
        """
        Explicitly releases resources held by managers and viewers.
        """
        logging.info("Cleaning up resources...")
        
        # Release Video Manager
        if self.video_manager is not None:
            # Assuming VideoMP4Manager wraps cv2 or similar and might have a release/close
            if hasattr(self.video_manager, 'release'):
                self.video_manager.release()
            elif hasattr(self.video_manager, 'close'):
                self.video_manager.close()
            self.video_manager = None

        # Cleanup Matplotlib if used by Viewer_3D
        # This closes any satellite windows created via plt.figure()
        plt.close('all')

    def _load_csv_data(self, csv_path: str):
        """Loads the trial_on column from the CSV."""
        if not csv_path or not Path(csv_path).exists():
            logging.warning("CSV path invalid or not provided. Slider highlight disabled.")
            return

        try:
            df = pd.read_csv(csv_path)
            if 'trial_on' in df.columns:
                # Fill NaNs with 0, convert to int
                raw_data = df['trial_on'].fillna(0).astype(int).values
                
                # Handle Length Mismatch: Trim or Pad
                if len(raw_data) >= self.total_frames:
                    self.trial_on_data = raw_data[:self.total_frames]
                else:
                    logging.warning(f"CSV length ({len(raw_data)}) < Video Frames ({self.total_frames}). Padding with 0.")
                    padded = np.zeros(self.total_frames, dtype=int)
                    padded[:len(raw_data)] = raw_data
                    self.trial_on_data = padded
                
                logging.info("CSV data loaded successfully for timeline highlight.")
            else:
                logging.warning("Column 'trial_on' not found in CSV.")
        except Exception as e:
            logging.error(f"Failed to load CSV: {e}")

    def _setup_ui(self):
        """Constructs the split-pane UI."""
        
        # --- Configure Styles ---
        style = ttk.Style()
        # Ensure we use a theme that supports background colors
        try:
            style.theme_use('clam') 
        except:
            pass
            
        # Style for Validate Button (Greenish)
        style.configure("Validate.TButton", 
                        background="#4CAF50", 
                        foreground="white", 
                        font=('Helvetica', 10, 'bold'))
        style.map("Validate.TButton", background=[('active', '#45a049')])

        # Style for Save Button (Blueish/Neutral)
        style.configure("Save.TButton", 
                        background="#2196F3", 
                        foreground="white",
                        font=('Helvetica', 10))
        style.map("Save.TButton", background=[('active', '#0b7dda')])

        # --- Main Layout Container ---
        main_layout = ttk.Frame(self.root, padding="10")
        main_layout.pack(fill="both", expand=True)

        # --- Left Panel: Video ---
        left_panel = ttk.Frame(main_layout)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.image_label = ttk.Label(left_panel)
        self.image_label.pack(pady=10, fill="both", expand=True)

        # --- Right Panel: Selection List ---
        right_panel = ttk.Frame(main_layout, width=250)
        right_panel.pack(side="right", fill="y")
        right_panel.pack_propagate(False)

        ttk.Label(right_panel, text="Selected Frames:").pack(anchor="w", pady=(0, 5))
        
        listbox_frame = ttk.Frame(right_panel)
        listbox_frame.pack(fill="both", expand=True)

        self.selected_listbox = tk.Listbox(listbox_frame, selectmode=tk.SINGLE)
        self.selected_listbox.pack(side="left", fill="both", expand=True)
        
        scrollbar = ttk.Scrollbar(listbox_frame, orient="vertical", command=self.selected_listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.selected_listbox.config(yscrollcommand=scrollbar.set)
        
        # Double click to remove
        self.selected_listbox.bind("<Double-1>", self._remove_from_list)

        self.delete_btn = ttk.Button(right_panel, text="Remove Selected", command=self._delete_selected_btn)
        self.delete_btn.pack(fill="x", pady=(5, 10))
        
        # --- Action Buttons ---
        
        # 1. Save Progress & Exit (No Success Flag)
        # Initialize as disabled; enabled via _update_selection_ui
        self.save_btn = ttk.Button(
            right_panel, 
            text="Save Progress & Exit", 
            command=self._on_save_exit, 
            style="Save.TButton",
            state="disabled"
        )
        self.save_btn.pack(fill="x", pady=(5, 5))

        # 2. Validate Selection & Exit (Generates Success Flag)
        # Initialize as disabled; enabled via _update_selection_ui
        self.validate_btn = ttk.Button(
            right_panel, 
            text="Validate & Complete", 
            command=self._on_validate_exit, 
            style="Validate.TButton",
            state="disabled"
        )
        self.validate_btn.pack(fill="x", pady=(5, 0))

        # --- Bottom Control Frame ---
        self.control_frame = ttk.Frame(self.root, padding="10")
        self.control_frame.pack(fill="x")

        # Slider Box Container
        slider_box = ttk.Frame(self.control_frame)
        slider_box.pack(fill="x", pady=5)
        
        self.lbl_start = ttk.Label(slider_box, text="0")
        self.lbl_start.pack(side="left")
        
        # --- Timeline Visualization Canvas (Above Slider) ---
        slider_container = ttk.Frame(slider_box)
        slider_container.pack(side="left", fill="x", expand=True, padx=5)

        # Canvas for blue highlight
        self.timeline_canvas = tk.Canvas(slider_container, height=15, bg="#e0e0e0", bd=0, highlightthickness=0)
        self.timeline_canvas.pack(fill="x", expand=True, side="top")
        self.timeline_canvas.bind("<Configure>", self._on_timeline_resize)
        self.timeline_canvas.bind("<Button-1>", self._on_timeline_click)
        self.timeline_canvas.bind("<B1-Motion>", self._on_timeline_click)

        self.slider_var = tk.IntVar(value=0)
        self.slider = ttk.Scale(
            slider_container, 
            from_=0, 
            to=self.total_frames - 1, 
            orient="horizontal", 
            variable=self.slider_var,
            command=self._on_slider_move
        )
        self.slider.pack(fill="x", expand=True, side="top")
        
        self.lbl_end = ttk.Label(slider_box, text=str(self.total_frames - 1))
        self.lbl_end.pack(side="left")

        # Buttons
        self.btn_frame = ttk.Frame(self.control_frame)
        self.btn_frame.pack(fill="x")

        self.play_btn = ttk.Button(self.btn_frame, text="Play", command=self._toggle_play)
        self.play_btn.pack(side="left", padx=5)

        self.prev_btn = ttk.Button(self.btn_frame, text="<", command=lambda: self._step_frame(-1))
        self.prev_btn.pack(side="left")

        self.next_btn = ttk.Button(self.btn_frame, text=">", command=lambda: self._step_frame(1))
        self.next_btn.pack(side="left")

        # Select Frame Button (Big and obvious)
        self.toggle_select_btn = ttk.Button(self.btn_frame, text="Select Frame", command=self._toggle_selection_current)
        self.toggle_select_btn.pack(side="left", padx=20)

        self.overlay_check = ttk.Checkbutton(self.btn_frame, text="Show Mesh", variable=self.overlay_enabled, command=self._update_display)
        self.overlay_check.pack(side="left", padx=10)

        self.info_label = ttk.Label(self.btn_frame, text=f"Frame: 0")
        self.info_label.pack(side="right", padx=5)

        # Keyboard Bindings
        self.root.bind("<Left>", lambda e: self._step_frame(-1))
        self.root.bind("<Right>", lambda e: self._step_frame(1))
        self.root.bind("<space>", lambda e: self._toggle_play())
        self.root.bind("<Return>", lambda e: self._toggle_selection_current())

    def _on_timeline_resize(self, event):
        """Redraws the timeline when the window is resized."""
        self._draw_timeline()

    def _draw_timeline(self):
        """
        Draws timeline indicators:
        1. Blue rectangles where trial_on == 1.
        2. Red rectangles for selected frames.
        """
        self.timeline_canvas.delete("all")
        
        w = self.timeline_canvas.winfo_width()
        h = self.timeline_canvas.winfo_height()
        if w <= 1: return

        scale_factor = w / self.total_frames

        # 1. Draw Blue Highlight (Trial Data)
        if self.trial_on_data is not None:
            padded = np.pad(self.trial_on_data, (1, 1), mode='constant', constant_values=0)
            diff = np.diff(padded)
            
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]

            for s, e in zip(starts, ends):
                x0 = s * scale_factor
                x1 = e * scale_factor
                if x1 - x0 < 1:
                    x1 = x0 + 1
                self.timeline_canvas.create_rectangle(x0, 0, x1, h, fill="#4a90e2", outline="")

        # 2. Draw Red Highlight (Selected Frames)
        for frame_idx in self.selected_frames:
            if 0 <= frame_idx < self.total_frames:
                x0 = frame_idx * scale_factor
                x1 = x0 + max(2, scale_factor)
                self.timeline_canvas.create_rectangle(x0, 0, x1, h, fill="red", outline="")

    def _on_timeline_click(self, event):
        """Allows clicking the timeline canvas to jump to frame."""
        w = self.timeline_canvas.winfo_width()
        if w <= 0: return
        x = max(0, min(event.x, w))
        frame_idx = int((x / w) * (self.total_frames - 1))
        self.slider_var.set(frame_idx)
        self.current_frame_idx = frame_idx
        self._update_display()

    def _load_previous_selections(self):
        """Loads previous selections from the output path if it exists."""
        if not self.output_path.exists():
            return

        try:
            logging.info(f"Output file found at {self.output_path}. Loading existing selections...")
            with open(self.output_path, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, dict) and 'meshes' in data:
                existing_keys = data['meshes'].keys()
                valid_keys = {int(k) for k in existing_keys}
                self.selected_frames = valid_keys
                logging.info(f"Restored {len(self.selected_frames)} selected frames.")
                self._update_selection_ui()
            else:
                logging.warning("Existing output file found but format was unrecognized.")
        except Exception as e:
            logging.error(f"Failed to load existing selections from file: {e}")

    # --- Interaction Methods ---

    def _toggle_play(self):
        self.is_playing = not self.is_playing
        self.play_btn.config(text="Pause" if self.is_playing else "Play")
        if self.is_playing:
            self._play_loop()

    def _play_loop(self):
        if self.is_playing and self.current_frame_idx < self.total_frames - 1:
            self.current_frame_idx += 1
            self.slider_var.set(self.current_frame_idx)
            self._update_display()
            self.root.after(33, self._play_loop)
        else:
            self.is_playing = False
            self.play_btn.config(text="Play")

    def _on_slider_move(self, val):
        self.current_frame_idx = int(float(val))
        self._update_display()

    def _step_frame(self, step):
        new_idx = self.current_frame_idx + step
        if 0 <= new_idx < self.total_frames:
            self.current_frame_idx = new_idx
            self.slider_var.set(new_idx)
            self._update_display()

    def _toggle_selection_current(self):
        if self.current_frame_idx in self.selected_frames:
            self.selected_frames.remove(self.current_frame_idx)
        else:
            self.selected_frames.add(self.current_frame_idx)
        self._update_selection_ui()

    def _delete_selected_btn(self):
        selected_indices = self.selected_listbox.curselection()
        if not selected_indices:
            return
        selected_item = self.selected_listbox.get(selected_indices[0])
        try:
            frame_num = int(selected_item.split(" ")[1])
            if frame_num in self.selected_frames:
                self.selected_frames.remove(frame_num)
                self._update_selection_ui()
        except (IndexError, ValueError):
            pass

    def _remove_from_list(self, event):
        self._delete_selected_btn()

    def _update_selection_ui(self):
        """Updates the listbox, timeline, and button states based on selection."""
        self.selected_listbox.delete(0, tk.END)
        for frame_num in sorted(list(self.selected_frames)):
            self.selected_listbox.insert(tk.END, f"Frame {frame_num}")
            
        # Update Toggle Button Text
        if self.current_frame_idx in self.selected_frames:
            self.toggle_select_btn.config(text="Deselect Frame")
        else:
            self.toggle_select_btn.config(text="Select Frame")
        
        # Update Timeline Visuals
        self._draw_timeline()
        
        # Update Save/Validate Button States
        # Only enable buttons if there is at least one selected frame.
        if self.selected_frames:
            state = "normal"
        else:
            state = "disabled"
            
        self.save_btn.config(state=state)
        self.validate_btn.config(state=state)

    # --- Button Callbacks ---

    def _on_save_exit(self):
        """Exit loop, save data, DO NOT validate. Does not destroy window, just quits loop."""
        logging.info("User requested: Save & Exit (Work In Progress)")
        self.is_task_validated = False
        self.root.quit() # Stop mainloop, handle destroy in controller

    def _on_validate_exit(self):
        """Exit loop, save data, AND validate. Does not destroy window, just quits loop."""
        logging.info("User requested: Validate & Complete")
        self.is_task_validated = True
        self.root.quit() # Stop mainloop, handle destroy in controller

    # --- Drawing Methods ---

    def _draw_overlay(self, image: np.ndarray, frame_idx: int) -> np.ndarray:
        geometry = self.data_manager.get_pixelwise_hand_geometry(frame_idx)
        if geometry is None:
            cv2.putText(image, "No Tracking Data", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 0), 2)
            return image

        vertices, faces = geometry
        pts_int = vertices.astype(np.int32)
        triangles = pts_int[faces]
        
        cv2.polylines(image, list(triangles), isClosed=True, color=(0, 255, 255), thickness=1)
        return image
    
    def _update_display(self):
        try:
            frame = self.video_manager[self.current_frame_idx]
            display_frame = frame.copy() 
        except Exception as e:
            logging.error(f"Error reading frame {self.current_frame_idx}: {e}")
            return

        if self.overlay_enabled.get():
            display_frame = self._draw_overlay(display_frame, self.current_frame_idx)

        geometry_3d = self.data_manager.get_3dspace_hand_geometry(self.current_frame_idx)
        if self.viewer_3d:
            self.viewer_3d.update_3d_plot(geometry_3d, global_max_range=self.data_manager.global_max_range)

        h, w = display_frame.shape[:2]
        target_h = 600
        scale = min(1.0, target_h / h)
        
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            display_frame = cv2.resize(display_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        img = Image.fromarray(display_frame)
        tk_img = ImageTk.PhotoImage(image=img)

        self.image_label.configure(image=tk_img)
        self.image_label.image = tk_img 
        self.info_label.config(text=f"Frame: {self.current_frame_idx} / {self.total_frames}")
        
        if self.current_frame_idx in self.selected_frames:
            self.toggle_select_btn.config(text="Deselect Frame")
        else:
            self.toggle_select_btn.config(text="Select Frame")

        if self.timeline_canvas.winfo_width() > 1:
            self._draw_timeline()
            x = (self.current_frame_idx / self.total_frames) * self.timeline_canvas.winfo_width()
            self.timeline_canvas.create_line(x, 0, x, self.timeline_canvas.winfo_height(), fill="red", width=2)