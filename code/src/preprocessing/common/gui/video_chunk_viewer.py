import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import multiprocessing
import time
import queue  # Import standard queue for Empty exception

# Context: Retaining the specific user import as requested
from preprocessing.common.data_access.video_mp4_manager import VideoMP4Manager, ColorFormat

class VideoChunkViewer:
    """
    A lightweight Tkinter-based viewer for a sequence of images.
    """
    def __init__(self, parent, frames: np.ndarray, title: str = "Video Chunk Viewer", fps: float = 30.0):
        self.frames = frames
        self.total_frames = len(frames)
        self.fps = fps
        self.current_frame_idx = 0
        
        # Use parent directly (assumed to be root in persistent mode)
        self.root = parent
        self.root.title(title)
        
        # Screen dimensions for resizing logic
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        self.target_w = screen_w // 2
        self.target_h = screen_h // 2
        
        # GUI Variables
        self.scale_var = tk.IntVar(value=0)
        self.lbl_var = tk.StringVar(value="Frame: 0")

        # Set a minimum size to ensure controls are never squashed significantly
        self.root.minsize(600, 500)

        self._setup_ui()
        self._bind_events()
        self._update_display()

    def update_frames(self, new_frames: np.ndarray, new_title: str = ""):
        """
        Hot-swaps the frame buffer and forces a refresh.
        """
        self.frames = new_frames
        self.total_frames = len(new_frames)
        self.current_frame_idx = 0
        self.scale_var.set(0)
        
        # Update Slider constraints
        if self.total_frames > 1:
            self.slider.configure(to=self.total_frames - 1, state='normal')
        else:
            self.slider.configure(to=0, state='disabled')
        
        if new_title:
            self.root.title(new_title)
            
        # Force immediate display update
        self._update_display()
        
        # Reset geometry to autosize based on the new content (image size + controls)
        self.root.geometry("")
        
        # Force Tkinter to redraw widgets immediately
        self.root.update_idletasks()

    def _setup_ui(self):
        layout = ttk.Frame(self.root, padding=10)
        layout.pack(fill="both", expand=True)

        self.image_label = ttk.Label(layout)
        self.image_label.pack(pady=5, fill="both", expand=True)

        controls = ttk.Frame(layout)
        controls.pack(fill="x", pady=5)

        self.slider = ttk.Scale(
            controls, 
            from_=0, 
            to=max(0, self.total_frames - 1), 
            orient="horizontal", 
            variable=self.scale_var,
            command=self._on_slider_move
        )
        self.slider.pack(side="left", fill="x", expand=True, padx=5)

        ttk.Label(controls, textvariable=self.lbl_var, width=15).pack(side="right")
        ttk.Label(self.root, text="Left/Right: Navigate | Close parent to Exit", font=("Arial", 8)).pack(side="bottom", pady=2)

    def _bind_events(self):
        self.root.bind("<Left>", lambda e: self._navigate(-1))
        self.root.bind("<Right>", lambda e: self._navigate(1))
        self.root.bind("<Control-Left>", lambda e: self._navigate(-10))
        self.root.bind("<Control-Right>", lambda e: self._navigate(10))
        self.root.focus_set()

    def _navigate(self, delta: int):
        if self.total_frames == 0: return
        new_idx = self.scale_var.get() + delta
        new_idx = max(0, min(new_idx, self.total_frames - 1))
        self.scale_var.set(new_idx)
        self._update_display()

    def _on_slider_move(self, val):
        self.current_frame_idx = int(float(val))
        self._update_display(from_slider=True)

    def _update_display(self, from_slider=False):
        if self.total_frames == 0:
            return

        idx = self.scale_var.get() if not from_slider else self.current_frame_idx
        idx = min(idx, self.total_frames - 1)

        self.lbl_var.set(f"Frame: {idx}/{self.total_frames - 1}")

        frame = self.frames[idx]
        
        # Resize only if dimensions change significantly
        h, w = frame.shape[:2]
        scale = min(self.target_w / w, self.target_h / h)
        
        # Only resize if necessary to save CPU
        if abs(scale - 1.0) > 0.05:
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        img_pil = Image.fromarray(frame)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        
        self.image_label.configure(image=img_tk)
        self.image_label.image = img_tk 

    def destroy(self):
        self.root.destroy()

# --- Optimized Persistent Runner ---

def run_persistent_viewer(command_queue: multiprocessing.Queue, video_path: str):
    """
    Runs the Tkinter loop and aggressively polls the queue for updates.
    """
    root = tk.Tk()
    
    # Fix: Set a larger initial geometry to ensure slider visibility for placeholder/loaded data.
    # 800x700 is generally safe for 1080p+ screens while providing ample vertical space for controls.
    root.geometry("800x700") 
    
    root.title("Initializing Video Viewer...")
    
    # Initialize Video Manager
    try:
        video_manager = VideoMP4Manager(video_path, color_format=ColorFormat.RGB)
    except Exception as e:
        print(f"❌ Worker Error: Could not load video {video_path}: {e}")
        return

    # Placeholder
    placeholder = np.zeros((300, 400, 3), dtype=np.uint8)
    viewer = VideoChunkViewer(root, np.array([placeholder]), "Waiting for Trial Data...")

    def poll_queue():
        try:
            # Loop to drain queue of pending updates
            while True:
                try:
                    msg = command_queue.get_nowait()
                except queue.Empty:
                    break # No more messages right now

                if msg == "EXIT":
                    root.destroy()
                    return
                
                if isinstance(msg, tuple) and len(msg) == 3:
                    start, end, title = msg
                    # Perform I/O
                    try:
                        # Ensure bounds
                        start = max(0, start)
                        chunk = np.array(video_manager[start : end + 1])
                        
                        if chunk is None or len(chunk) == 0:
                             print(f"⚠️ Warning: Empty chunk loaded for {start}-{end}")
                        else:
                             viewer.update_frames(chunk, new_title=title)
                             # Bring window to front when updated
                             root.deiconify() 
                             root.lift()
                    except Exception as io_err:
                        print(f"❌ Error reading frames: {io_err}")

        except Exception as e:
            print(f"❌ Viewer Queue Error: {e}")
        
        # Check again in 100ms
        root.after(100, poll_queue)

    # Start polling loop
    poll_queue()
    root.mainloop()