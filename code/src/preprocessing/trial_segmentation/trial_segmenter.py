import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from typing import TYPE_CHECKING, Dict, List, Tuple, Optional

if TYPE_CHECKING:
    from preprocessing.common import VideoMP4Manager
    from preprocessing.stickers_analysis.common.models.consolidated_tracks_manager import ConsolidatedTracksManager

class TrialSegmenterGUI:
    """
    A GUI for reviewing video and recording 'trial' (contiguous frame ranges).
    
    Modified from ConsolidatedTracksReviewGUI to focus on boolean labeling
    rather than object tracking visualization.
    
    Attributes:
        chunks (List[Tuple[int, int]]): The list of recorded frame ranges.
    """

    def __init__(self,
                 *,
                 video_manager: 'VideoMP4Manager',
                 tracks_manager: 'ConsolidatedTracksManager',
                 initial_chunks: Optional[List[Tuple[int, int]]] = None,
                 title: str = "Chunk Recording Interface",
                 windowState: str = 'normal'):
        """
        Initializes the Chunk Recording GUI.

        Args:
            video_manager: Manager for accessing video frames.
            tracks_manager: Included for architectural consistency.
            initial_chunks: Optional list of (start, end) tuples to preload.
            title: Window title.
            windowState: 'normal' or 'maximized'.
        """
        self.video_manager = video_manager
        self.tracks_manager = tracks_manager
        self.title = title
        self.windowState = windowState

        # --- Chunk Data State ---
        self.chunks: List[Tuple[int, int]] = initial_chunks if initial_chunks else []
        self._is_recording = False
        self._recording_start_frame: Optional[int] = None

        # --- Playback State ---
        self._is_playing = False
        self._update_job = None
        # Fixed framerate delay, speed controls removed per requirements
        self._playback_delay_ms = int(1000 / self.video_manager.fps)

        # --- UI Elements ---
        self.root = None
        self.image_label = None
        self.timeline_scale = None
        self.chunk_canvas = None
        self.record_button = None
        self.chunk_listbox = None
        self.current_frame_label = None

    def setup_ui(self):
        """Creates and arranges the Tkinter widgets."""
        self.root = tk.Tk()
        self.root.title(self.title)

        self.scale_var = tk.IntVar(value=0)

        # --- Window Positioning ---
        if self.windowState.upper() == 'NORMAL':
            window_width, window_height = 1280, 900
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
        main_frame.rowconfigure(0, weight=1) # Video expands
        main_frame.columnconfigure(0, weight=1)

        # 1. Video Display Area
        self.image_label = ttk.Label(main_frame)
        self.image_label.grid(row=0, column=0, sticky="nsew")

        # 2. Timeline Navigation Slider
        total_frames = len(self.video_manager)
        self.timeline_scale = ttk.Scale(
            main_frame, from_=0, to=total_frames - 1,
            orient=tk.HORIZONTAL, variable=self.scale_var,
            command=self.seek_to_frame
        )
        self.timeline_scale.grid(row=1, column=0, sticky="ew", pady=(5, 0))

        # 3. Chunk Visualization Canvas
        # Located physically below the navigation slider
        self.chunk_canvas = tk.Canvas(main_frame, height=30, bg="#e0e0e0", highlightthickness=0)
        self.chunk_canvas.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        self.chunk_canvas.bind("<Configure>", self._on_canvas_resize)

        # 4. Controls Container
        controls_container = ttk.Frame(main_frame)
        controls_container.grid(row=3, column=0, sticky="ew")
        controls_container.columnconfigure(1, weight=1) # Spacer

        # Left Side: Playback & Record Controls
        left_controls = ttk.Frame(controls_container)
        left_controls.grid(row=0, column=0, sticky="w")

        # Play Button (Functionality retained for navigation, separated from recording toggle)
        self.play_button = ttk.Button(left_controls, text="▶ Play Video", command=self.toggle_video_playback)
        self.play_button.pack(side=tk.LEFT, padx=5)

        # Record Toggle Button
        self.record_button = tk.Button(
            left_controls, text="● Start Recording (Space)", 
            bg="lightgrey", fg="black",
            command=self.toggle_recording_state,
            width=25, relief=tk.RAISED
        )
        self.record_button.pack(side=tk.LEFT, padx=15)

        self.current_frame_label = ttk.Label(left_controls, text=f"Frame: 0 / {total_frames - 1}")
        self.current_frame_label.pack(side=tk.LEFT, padx=10)

        # Right Side: Deletion Management
        right_controls = ttk.Frame(controls_container)
        right_controls.grid(row=0, column=2, sticky="e")

        ttk.Label(right_controls, text="Recorded Chunks:").pack(side=tk.TOP, anchor="w")
        
        # Scrollable Listbox for Chunks
        list_frame = ttk.Frame(right_controls)
        list_frame.pack(side=tk.TOP)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.chunk_listbox = tk.Listbox(list_frame, height=4, width=30, yscrollcommand=scrollbar.set)
        self.chunk_listbox.pack(side=tk.LEFT, fill=tk.BOTH)
        scrollbar.config(command=self.chunk_listbox.yview)

        delete_btn = ttk.Button(right_controls, text="Delete Selected Chunk", command=self.delete_selected_chunk)
        delete_btn.pack(side=tk.TOP, pady=5, fill=tk.X)

        # Initialize visual state
        self._bind_keys()
        self._refresh_chunk_list()
        self.root.after(50, lambda: self.seek_to_frame(0)) 

    def _bind_keys(self):
        """Binds keyboard shortcuts."""
        # Navigation
        self.root.bind('<Left>', lambda e: self.seek_to_frame(self.scale_var.get() - 1))
        self.root.bind('<Right>', lambda e: self.seek_to_frame(self.scale_var.get() + 1))
        
        # Space Bar acts as Toggle for Recording (Overrides Play/Pause)
        self.root.bind('<space>', lambda e: self.toggle_recording_state())

    def start(self):
        self.setup_ui()
        self.root.mainloop()

    def quit(self):
        if self._update_job:
            self.root.after_cancel(self._update_job)
        self.root.destroy()
        return self.chunks

    # --- Logic: Chunk Recording ---

    def toggle_recording_state(self):
        """Toggles between Idle and Recording states."""
        current_frame = self.scale_var.get()

        if not self._is_recording:
            # Start Trigger
            self._is_recording = True
            self._recording_start_frame = current_frame
            
            # UI Feedback: Button becomes Red/Rec pressed
            self.record_button.config(text="■ Stop Recording (Space)", bg="#ffcccc", fg="red", relief=tk.SUNKEN)
        else:
            # End Trigger
            start = self._recording_start_frame
            end = current_frame
            
            # Normalize range (handle reverse playback recording)
            actual_start = min(start, end)
            actual_end = max(start, end)
            
            # Commit chunk
            self.chunks.append((actual_start, actual_end))
            # Sort chunks by start frame
            self.chunks.sort(key=lambda x: x[0])
            
            # Reset State
            self._is_recording = False
            self._recording_start_frame = None
            
            # UI Feedback: Button becomes normal
            self.record_button.config(text="● Start Recording (Space)", bg="lightgrey", fg="black", relief=tk.RAISED)
            
            # Update Visuals
            self._refresh_chunk_list()
            self._draw_chunks_on_timeline()

    def delete_selected_chunk(self):
        """Deletes the selected range from the listbox and data."""
        selection = self.chunk_listbox.curselection()
        if not selection:
            return
            
        index = selection[0]
        removed = self.chunks.pop(index)
        print(f"Deleted chunk: {removed}")
        
        self._refresh_chunk_list()
        self._draw_chunks_on_timeline()

    def _refresh_chunk_list(self):
        """Updates the listbox with current chunks."""
        self.chunk_listbox.delete(0, tk.END)
        for idx, (start, end) in enumerate(self.chunks):
            duration = end - start
            self.chunk_listbox.insert(tk.END, f"{idx + 1}: Frames {start}-{end} ({duration})")

    # --- Logic: Visualization ---

    def _on_canvas_resize(self, event):
        """Redraws the timeline when window is resized."""
        self._draw_chunks_on_timeline()

    def _draw_chunks_on_timeline(self):
        """Renders visual representations of chunks on the canvas."""
        if not self.chunk_canvas:
            return
            
        self.chunk_canvas.delete("all")
        
        canvas_width = self.chunk_canvas.winfo_width()
        canvas_height = self.chunk_canvas.winfo_height()
        total_frames = len(self.video_manager)
        
        if total_frames == 0 or canvas_width == 1: return

        # Draw base line
        self.chunk_canvas.create_line(0, canvas_height/2, canvas_width, canvas_height/2, fill="gray")

        # Draw chunks
        for (start, end) in self.chunks:
            x1 = (start / total_frames) * canvas_width
            x2 = (end / total_frames) * canvas_width
            
            # Ensure visible width even for single frame chunks
            if x2 - x1 < 1: x2 = x1 + 1
            
            # Draw Rectangle
            self.chunk_canvas.create_rectangle(
                x1, 2, x2, canvas_height - 2,
                fill="#4CAF50", outline="#388E3C" # Green color scheme
            )

    # --- Logic: Playback & Video (Simplified from Original) ---

    def toggle_video_playback(self):
        """Toggles video playback (distinct from recording state)."""
        self._is_playing = not self._is_playing
        self.play_button.config(text="❚❚ Pause Video" if self._is_playing else "▶ Play Video")
        if self._is_playing:
            self._update_playback_loop()

    def seek_to_frame(self, frame_val):
        try:
            frame_num = int(float(frame_val))
            total = len(self.video_manager)
            if not (0 <= frame_num < total): return
            
            self.scale_var.set(frame_num)
            self.current_frame_label.config(text=f"Frame: {frame_num} / {total - 1}")
            
            # Display Frame
            frame_bgr = self.video_manager[frame_num]
            
            # Visual Feedback for "Active Recording" overlay
            if self._is_recording:
                cv2.circle(frame_bgr, (30, 30), 15, (0, 0, 255), -1) # Red recording dot on video
            
            # Resize and Show
            self._display_frame(frame_bgr)

        except (ValueError, IndexError):
            pass

    def _update_playback_loop(self):
        if not self._is_playing: return

        current = self.scale_var.get()
        if current + 1 < len(self.video_manager):
            self.seek_to_frame(current + 1)
            self._update_job = self.root.after(self._playback_delay_ms, self._update_playback_loop)
        else:
            self._is_playing = False
            self.play_button.config(text="▶ Play Video")

    def _display_frame(self, frame_bgr):
        container_w = self.image_label.winfo_width()
        container_h = self.image_label.winfo_height()
        
        if container_w < 10 or container_h < 10: return

        h, w, _ = frame_bgr.shape
        scale = min(container_w / w, container_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
        
        self.image_label.imgtk = img
        self.image_label.config(image=img)