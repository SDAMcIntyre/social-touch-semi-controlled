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
    
    Modified to prevent overlapping chunks, support deletion, and visualize tracking data
    (ROIs and Ellipses) identical to the ConsolidatedTracksReviewGUI.
    
    Attributes:
        chunks (List[Tuple[int, int]]): The list of recorded frame ranges.
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
                 initial_chunks: Optional[List[Tuple[int, int]]] = None,
                 title: str = "Chunk Recording Interface",
                 windowState: str = 'normal'):
        """
        Initializes the Chunk Recording GUI.

        Args:
            video_manager: Manager for accessing video frames.
            tracks_manager: Manager for accessing tracking data.
            object_colors: A mapping from object names to color strings.
            initial_chunks: Optional list of (start, end) tuples to preload.
            title: Window title.
            windowState: 'normal' or 'maximized'.
        """
        self.video_manager = video_manager
        self.tracks_manager = tracks_manager
        self.title = title
        self.windowState = windowState

        # --- Color Setup ---
        self.object_colors = {name: self.BGR_COLOR_MAP.get(color.lower(), (255, 255, 255))
                              for name, color in object_colors.items()}

        # --- Chunk Data State ---
        self.chunks: List[Tuple[int, int]] = initial_chunks if initial_chunks else []
        self._is_recording = False
        self._recording_start_frame: Optional[int] = None

        # --- Playback State ---
        self._is_playing = False
        self._update_job = None
        # Fixed framerate delay
        try:
            self._playback_delay_ms = int(1000 / self.video_manager.fps)
        except AttributeError:
            self._playback_delay_ms = 33  # Default to ~30fps if undefined

        # --- UI Elements ---
        self.root = None
        self.image_label = None
        self.timeline_scale = None
        self.chunk_canvas = None
        self.record_button = None
        self.chunk_listbox = None
        self.current_frame_label = None
        self.scale_var = None
        self.play_button = None

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

        # Play Button
        self.play_button = ttk.Button(left_controls, text="▶ Play Video", command=self.toggle_video_playback)
        self.play_button.pack(side=tk.LEFT, padx=5)

        # Record Toggle Button
        self.record_button = tk.Button(
            left_controls, text="● Start Recording (Space/Enter)", 
            bg="lightgrey", fg="black",
            command=self.toggle_recording_state,
            width=30, relief=tk.RAISED
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

        # --- NEW: OK Button to Finish and Return Data ---
        # We use a bold font or distinct styling to indicate this is the primary exit action.
        ok_btn = tk.Button(
            right_controls, 
            text="OK (Finish)", 
            command=self.quit,
            bg="#4CAF50", fg="white", 
            font=("Arial", 10, "bold"),
            relief=tk.RAISED
        )
        ok_btn.pack(side=tk.TOP, pady=(10, 5), fill=tk.X)

        # Initialize visual state
        self._bind_keys()
        self._refresh_chunk_list()
        self.root.after(50, lambda: self.seek_to_frame(0)) 

    def _bind_keys(self):
        """Binds keyboard shortcuts."""
        total_frames = len(self.video_manager)

        # Standard Navigation (1 Frame)
        # We clamp these to ensure safety, although seek_to_frame checks bounds too.
        self.root.bind('<Left>', lambda e: self.seek_to_frame(max(0, self.scale_var.get() - 1)))
        self.root.bind('<Right>', lambda e: self.seek_to_frame(min(total_frames - 1, self.scale_var.get() + 1)))
        
        # Fast Navigation (5 Frames) via Control + Arrows
        self.root.bind('<Control-Left>', lambda e: self.seek_to_frame(max(0, self.scale_var.get() - 5)))
        self.root.bind('<Control-Right>', lambda e: self.seek_to_frame(min(total_frames - 1, self.scale_var.get() + 5)))

        # Space Bar AND Enter Key act as Toggle for Recording (Overrides Play/Pause)
        self.root.bind('<space>', lambda e: self.toggle_recording_state())
        self.root.bind('<Return>', lambda e: self.toggle_recording_state())

    def start(self):
        """
        Starts the GUI loop.
        
        Returns:
            List[Tuple[int, int]]: The final list of chunks when the window is closed.
        """
        self.setup_ui()
        self.root.mainloop()
        # Only reached after root.destroy() in self.quit()
        return self.chunks

    def quit(self):
        """Closes the window and stops the loop."""
        if self._update_job:
            self.root.after_cancel(self._update_job)
        self.root.destroy()
        # Note: The return value here is used by the event handler but not the start() caller.
        # The start() method returns self.chunks after mainloop exits.
        return self.chunks

    # --- Logic: Chunk Recording & Overlap Prevention ---

    def _check_overlap(self, start_f: int, end_f: int) -> bool:
        """
        Checks if the range [start_f, end_f] overlaps with any existing chunk.
        """
        s = min(start_f, end_f)
        e = max(start_f, end_f)
        
        for (cs, ce) in self.chunks:
            # Check for intersection
            if max(s, cs) <= min(e, ce):
                return True
        return False

    def _update_record_button_state(self):
        """
        Updates the record button's state and color based on overlap logic.
        """
        if not self.root: return
        
        current_frame = self.scale_var.get()
        
        if not self._is_recording:
            # IDLE STATE: Check if current frame is inside an existing chunk
            in_existing_chunk = False
            for (cs, ce) in self.chunks:
                if cs <= current_frame <= ce:
                    in_existing_chunk = True
                    break
            
            if in_existing_chunk:
                # Cannot start recording here
                self.record_button.config(
                    state=tk.DISABLED, 
                    text="Overlap Detected", 
                    bg="#cccccc", 
                    fg="#666666"
                )
            else:
                # Ready to record
                self.record_button.config(
                    state=tk.NORMAL, 
                    text="● Start Recording (Space/Enter)", 
                    bg="lightgrey", 
                    fg="black",
                    relief=tk.RAISED
                )
        else:
            # RECORDING STATE: Check if the segment formed so far overlaps anything
            rec_start = self._recording_start_frame
            
            if self._check_overlap(rec_start, current_frame):
                # Overlap occurring - Disable saving (Force user to move back)
                self.record_button.config(
                    state=tk.DISABLED, 
                    text="Overlap Detected", 
                    bg="#cccccc", 
                    fg="#666666"
                )
            else:
                # Valid recording segment
                self.record_button.config(
                    state=tk.NORMAL, 
                    text="■ Stop Recording (Space/Enter)", 
                    bg="#ffcccc", 
                    fg="red",
                    relief=tk.SUNKEN
                )

    def toggle_recording_state(self):
        """Toggles between Idle and Recording states."""
        
        # If button is disabled due to overlap, ignore shortcut keys
        if self.record_button['state'] == tk.DISABLED:
            return

        current_frame = self.scale_var.get()

        if not self._is_recording:
            # Start Trigger
            self._is_recording = True
            self._recording_start_frame = current_frame
            self._update_record_button_state()
            
        else:
            # End Trigger
            start = self._recording_start_frame
            end = current_frame
            
            # Normalize range
            actual_start = min(start, end)
            actual_end = max(start, end)
            
            # Double check overlap before saving
            if self._check_overlap(actual_start, actual_end):
                messagebox.showwarning("Invalid Chunk", "The selected range overlaps with an existing chunk.")
                return

            # Commit chunk
            self.chunks.append((actual_start, actual_end))
            # Sort chunks by start frame
            self.chunks.sort(key=lambda x: x[0])
            
            # Reset State
            self._is_recording = False
            self._recording_start_frame = None
            
            # Update Visuals
            self._refresh_chunk_list()
            self._draw_chunks_on_timeline()
            self._update_record_button_state()

    def delete_selected_chunk(self):
        """Deletes the selected range from the listbox and data."""
        selection = self.chunk_listbox.curselection()
        if not selection:
            return
            
        index = selection[0]
        # Delete from data
        removed = self.chunks.pop(index)
        print(f"Deleted chunk: {removed}")
        
        # Refresh UI
        self._refresh_chunk_list()
        self._draw_chunks_on_timeline()
        
        # Re-evaluate button state (current frame might now be free)
        self._update_record_button_state()

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
        for idx, (start, end) in enumerate(self.chunks):
            x1 = (start / total_frames) * canvas_width
            x2 = (end / total_frames) * canvas_width
            
            # Ensure visible width even for single frame chunks
            if x2 - x1 < 1: x2 = x1 + 1
            
            # Draw Rectangle
            self.chunk_canvas.create_rectangle(
                x1, 2, x2, canvas_height - 2,
                fill="#4CAF50", outline="#388E3C" # Green color scheme
            )
            
            # Draw Chunk Number
            center_x = (x1 + x2) / 2
            center_y = canvas_height / 2
            
            self.chunk_canvas.create_text(
                center_x, center_y,
                text=str(idx + 1),
                fill="white",
                font=("Arial", 8, "bold")
            )

    # --- Logic: Playback & Video ---

    def toggle_video_playback(self):
        """Toggles video playback."""
        self._is_playing = not self._is_playing
        self.play_button.config(text="❚❚ Pause Video" if self._is_playing else "▶ Play Video")
        if self._is_playing:
            self._update_playback_loop()

    def seek_to_frame(self, frame_val):
        """
        Seeks to a specific frame and renders overlays. 
        Includes robust error handling for missing frames or corrupt tracking data.
        """
        frame_bgr = None
        tracking_failure_msg = None

        # --- 1. Frame Validation and Retrieval ---
        try:
            frame_num = int(float(frame_val))
            total = len(self.video_manager)
            if not (0 <= frame_num < total): 
                return
            
            self.scale_var.set(frame_num)
            self.current_frame_label.config(text=f"Frame: {frame_num} / {total - 1}")

            try:
                # Attempt to fetch the actual frame
                frame_bgr = self.video_manager[frame_num]
                if frame_bgr is None:
                    raise ValueError("Returned frame is None")
            except Exception as e:
                print(f"Error fetching frame {frame_num}: {e}")
                # Warn user via console or simple print as requested
                print(f"Warning: Could not load frame {frame_num}. Using black fallback.")
                
                # Create fallback black image
                # Try to deduce dimensions from video manager attributes or use defaults
                w = getattr(self.video_manager, 'width', 1280)
                h = getattr(self.video_manager, 'height', 720)
                frame_bgr = np.zeros((h, w, 3), dtype=np.uint8)
                
                # Burn error message into the black frame
                cv2.putText(frame_bgr, "FRAME LOAD FAILED", (50, h // 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            # --- 2. Tracking Data Visualization ---
            try:
                # Only attempt tracking visualization if we have a valid image container
                tracked_items = self.tracks_manager.get_items_for_frame(frame_num)
                
                for object_name, data in tracked_items:
                    try:
                        # Check for explicit 'Ignored' status
                        if str(data.get('status', '')).lower() == 'ignored':
                            tracking_failure_msg = "Status: Ignored"
                            continue

                        color = self.object_colors.get(object_name, (255, 255, 255))

                        # --- Draw ROI Rectangle ---
                        if all(k in data for k in ['roi_x', 'roi_y', 'roi_width', 'roi_height']):
                            # Check for NaN values which crash int() conversion
                            roi_vals = [data['roi_x'], data['roi_y'], data['roi_width'], data['roi_height']]
                            if any(np.isnan(v) for v in roi_vals if isinstance(v, (int, float))):
                                tracking_failure_msg = "Invalid ROI (NaN)"
                                continue

                            x1, y1 = int(data['roi_x']), int(data['roi_y'])
                            x2, y2 = x1 + int(data['roi_width']), y1 + int(data['roi_height'])
                            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame_bgr, object_name, (x1, y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        # --- Draw Ellipse ---
                        if all(k in data for k in ['ellipse_center_x', 'ellipse_center_y', 'axes_major', 'axes_minor', 'angle']):
                            # Check for NaN values
                            ell_vals = [data['ellipse_center_x'], data['ellipse_center_y'], 
                                        data['axes_major'], data['axes_minor'], data['angle']]
                            if any(np.isnan(v) for v in ell_vals if isinstance(v, (int, float))):
                                tracking_failure_msg = "Invalid Ellipse (NaN)"
                                continue

                            center = (int(data['ellipse_center_x']), int(data['ellipse_center_y']))
                            axes = (int(data['axes_major'] / 2), int(data['axes_minor'] / 2))
                            angle = int(data['angle'])
                            cv2.ellipse(frame_bgr, center, axes, angle, 0, 360, color, 2)

                    except (ValueError, OverflowError) as ve:
                        # Catch specific conversion errors per item
                        print(f"Error drawing item {object_name}: {ve}")
                        tracking_failure_msg = "Tracking Data Error"

            except Exception as e:
                # Catch errors in getting items or general tracking logic
                print(f"Tracking system failure on frame {frame_num}: {e}")
                tracking_failure_msg = "Tracking System Failed"

            # --- 3. Overlays & Feedback ---
            
            # Visual Feedback for "Active Recording"
            if self._is_recording:
                cv2.circle(frame_bgr, (30, 30), 15, (0, 0, 255), -1)

            # Visual Feedback for Tracking Failures
            if tracking_failure_msg:
                cv2.putText(frame_bgr, f"Tracking Warning: {tracking_failure_msg}", (50, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            # --- 4. Display ---
            self._display_frame(frame_bgr)
            
            # Update Button State
            self._update_record_button_state()

        except Exception as e:
            print(f"Critical error in seek_to_frame: {e}")
            # If everything failed (including the first fallback), try one last desperate fallback
            if frame_bgr is None:
                 frame_bgr = np.zeros((720, 1280, 3), dtype=np.uint8)
                 cv2.putText(frame_bgr, "CRITICAL RENDER ERROR", (50, 360), 
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            self._display_frame(frame_bgr)

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
        if frame_bgr is None: return
        
        container_w = self.image_label.winfo_width()
        container_h = self.image_label.winfo_height()
        
        if container_w < 10 or container_h < 10: return

        h, w, _ = frame_bgr.shape
        scale = min(container_w / w, container_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        try:
            resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            img = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
            
            self.image_label.imgtk = img
            self.image_label.config(image=img)
        except Exception as e:
            print(f"Error displaying frame buffer: {e}")