import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from typing import List, Optional, Tuple
from enum import Enum, auto

# --- Matplotlib Integration ---
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class SelectionState(Enum):
    PENDING = auto()
    CONFIRMED = auto()
    CANCELLED = auto()
    REDO_COLORSPACE = auto()

class ViewMode(Enum):
    SINGLE_FRAME_2D = auto()
    POINT_CLOUD_3D = auto()

class ThresholdSelectorTool:
    """
    Architecturally refined GUI tool for threshold selection.
    
    Modifications:
    - Variable Renaming: 'frames' -> 'frames_rgb', 'frames_rgb' -> 'frames_corr'.
    - Logic Swap: Thresholding/Volume is now calculated on 'frames_corr'. 
      Visual context is provided by 'frames_rgb'.
    - RGB Integration: Accepts and renders parallel RGB video stream.
    - Geometry Propagation: Disabled on 2D render frame to prevent infinite resize loops.
    - Dynamic Resizing: 2D view listens to <Configure> events to maximize video usage.
    - Default 3D: Initializes directly into POINT_CLOUD_3D mode.
    - Voxel Simulation: 3D scatter plots use square markers.
    - Layout Update: 2D view now supports 3-column layout (RGB | Corr | Thresh).
    """

    def __init__(
            self, 
            frames_rgb: List[np.ndarray], 
            frames_corr: List[np.ndarray], 
            video_name: str = "video", 
            threshold: int = 127,
            spot_type: str = 'dark'):
        
        if not frames_rgb:
            raise ValueError("Frame list cannot be empty.")
        
        # --- Architecture Update: Validate and Truncate Streams ---
        if len(frames_rgb) != len(frames_corr):
            print(f"Warning: RGB frames ({len(frames_rgb)}) and Correlation frames ({len(frames_corr)}) count mismatch.")
            min_len = min(len(frames_rgb), len(frames_corr))
            self.frames_rgb = frames_rgb[:min_len]
            self.frames_corr = frames_corr[:min_len]
        else:
            self.frames_rgb = frames_rgb
            self.frames_corr = frames_corr

        self.total_frames = len(self.frames_rgb)
        self.video_name = video_name
        self.init_threshold = threshold
        self.spot_type = spot_type

        # --- Data Preparation ---
        # We generate the volume from frames_corr (the analysis data)
        self._prepare_volumes()

        # --- State ---
        self.result: Optional[int] = None
        self.selection_state: SelectionState = SelectionState.PENDING
        self.view_mode: ViewMode = ViewMode.POINT_CLOUD_3D
        
        # Cache for window dimensions to prevent render loops
        self.last_2d_size: Tuple[int, int] = (0, 0)

        # --- UI Components ---
        self.root: Optional[tk.Tk] = None
        self.canvas_2d_frame: Optional[ttk.Frame] = None
        self.canvas_3d_frame: Optional[ttk.Frame] = None
        
        # Matplotlib references
        self.fig = None
        self.ax = None
        self.mpl_canvas = None

        # Tkinter Vars
        self.frame_var: Optional[tk.IntVar] = None
        self.thresh_var: Optional[tk.IntVar] = None
        self.title_var: Optional[tk.StringVar] = None
        self.mode_btn_text: Optional[tk.StringVar] = None

    def _prepare_volumes(self):
        """
        Pre-processes correlation video data into 3D arrays.
        Operates on self.frames_corr.
        """
        print("Processing volume data from correlation frames...")
        shape = self.frames_corr[0].shape
        
        # Ensure Grayscale for Volume Analysis
        if len(shape) == 3:
            # Assuming input might be BGR, convert to Gray
            gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in self.frames_corr]
        else:
            gray_frames = self.frames_corr
        
        self.volume_full = np.stack(gray_frames) # Shape: (Z, Y, X)
        self.full_h, self.full_w = self.volume_full.shape[1], self.volume_full.shape[2]

        # Downsample for 3D visualization (Performance Optimization)
        # Target roughly 100x100xZ resolution
        target_dim = 100
        h, w = self.full_h, self.full_w
        scale = target_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize all frames for the cache
        resized_frames = [cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_NEAREST) 
                          for f in gray_frames]
        
        self.volume_small = np.stack(resized_frames)
        self.small_h, self.small_w = self.volume_small.shape[1], self.volume_small.shape[2]
        print(f"Volume cached. Full: {self.volume_full.shape}, Small: {self.volume_small.shape}")

    def run(self) -> Optional[int]:
        self.root = tk.Tk()
        self.root.title("Interactive Threshold Architecture")
        self.root.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.root.state('zoomed')

        self._init_vars()
        self._setup_styles()
        self._setup_layout()

        # Force initial render logic based on default mode (3D)
        self.root.update()
        
        # Set initial button text based on default mode
        if self.view_mode == ViewMode.POINT_CLOUD_3D:
             self.mode_btn_text.set("Switch to 2D View")
             self.canvas_2d_frame.pack_forget()
             self.canvas_3d_frame.pack(fill=tk.BOTH, expand=True)
             self.scale_frame.state(['disabled'])
        
        self._update_display()

        print(f"\n--- Tool Running: {self.video_name} ---")
        self.root.mainloop()

        self._cleanup()
        return self.result

    def _init_vars(self):
        self.frame_var = tk.IntVar(value=0)
        self.thresh_var = tk.IntVar(value=self.init_threshold)
        self.title_var = tk.StringVar()
        self.mode_btn_text = tk.StringVar(value="Switch to 2D View")

    def _cleanup(self):
        if self.root:
            self.root.destroy()
            self.root = None
        if self.fig:
            matplotlib.pyplot.close(self.fig)

    def _setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        btn_font = ("Segoe UI", 10, "bold")
        
        colors = {
            "Confirm": ("#2E7D32", "#4CAF50"),
            "Redo": ("#EF6C00", "#FF9800"),
            "Cancel": ("#C62828", "#EF5350"),
            "Mode": ("#1565C0", "#42A5F5")
        }

        for name, (normal, active) in colors.items():
            style_name = f"{name}.TButton"
            self.style.configure(style_name, background=normal, foreground="white", font=btn_font)
            self.style.map(style_name, background=[("active", active)])

    def _setup_layout(self):
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Header
        ttk.Label(main, textvariable=self.title_var, font=("Segoe UI", 14, "bold")).pack(pady=(0, 10), anchor="w")

        # Central View Container
        self.view_container = ttk.Frame(main)
        self.view_container.pack(fill=tk.BOTH, expand=True)

        # 2D View Frame
        self.canvas_2d_frame = ttk.Frame(self.view_container)
        self.canvas_2d_frame.pack_propagate(False) 
        
        # Split 2D frame into three sections: RGB | Corr | Thresh
        
        # Left: RGB
        self.lbl_orig = ttk.Label(self.canvas_2d_frame, anchor="center")
        self.lbl_orig.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2)
        
        # Middle: Correlation
        self.lbl_corr = ttk.Label(self.canvas_2d_frame, anchor="center")
        self.lbl_corr.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2)

        # Right: Threshold
        self.lbl_thresh = ttk.Label(self.canvas_2d_frame, anchor="center")
        self.lbl_thresh.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2)

        # Bind Resize Event
        self.canvas_2d_frame.bind("<Configure>", self._on_2d_resize)

        # 3D View Frame
        self.canvas_3d_frame = ttk.Frame(self.view_container)
        
        # Initialize Matplotlib Figure
        self.fig = Figure(figsize=(5, 4), dpi=100, facecolor='#f0f0f0')
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#f0f0f0')
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_3d_frame)
        self.mpl_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Controls Area
        controls = ttk.LabelFrame(main, text="Parameters", padding=10)
        controls.pack(fill=tk.X, pady=(10, 0))

        controls.columnconfigure(1, weight=1)
        
        # Frame Slider
        ttk.Label(controls, text="Frame Index:").grid(row=0, column=0, sticky="e", padx=5)
        self.scale_frame = ttk.Scale(controls, from_=0, to=self.total_frames-1, variable=self.frame_var, 
                                     orient=tk.HORIZONTAL, command=lambda v: self._update_display())
        self.scale_frame.grid(row=0, column=1, sticky="ew")
        ttk.Label(controls, textvariable=self.frame_var, width=4).grid(row=0, column=2, padx=5)

        # Threshold Slider
        ttk.Label(controls, text="Threshold:").grid(row=1, column=0, sticky="e", padx=5)
        self.scale_thresh = ttk.Scale(controls, from_=0, to=255, variable=self.thresh_var, 
                                      orient=tk.HORIZONTAL, command=lambda v: self._update_display())
        self.scale_thresh.grid(row=1, column=1, sticky="ew")
        ttk.Label(controls, textvariable=self.thresh_var, width=4).grid(row=1, column=2, padx=5)

        # Buttons
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X, pady=10)

        ttk.Button(btn_frame, text="Confirm", command=self._confirm, style="Confirm.TButton").pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="Redo", command=self._redo, style="Redo.TButton").pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self._on_cancel, style="Cancel.TButton").pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, textvariable=self.mode_btn_text, command=self._toggle_mode, style="Mode.TButton").pack(side=tk.RIGHT, padx=5)

        self.root.bind('<Return>', lambda e: self._confirm())
        self.root.bind('<Escape>', lambda e: self._on_cancel())

    def _on_2d_resize(self, event):
        if self.view_mode != ViewMode.SINGLE_FRAME_2D:
            return
            
        if abs(event.width - self.last_2d_size[0]) > 10 or abs(event.height - self.last_2d_size[1]) > 10:
            self.last_2d_size = (event.width, event.height)
            self._update_display()

    def _toggle_mode(self):
        if self.view_mode == ViewMode.SINGLE_FRAME_2D:
            self.view_mode = ViewMode.POINT_CLOUD_3D
            self.mode_btn_text.set("Switch to 2D View")
            self.canvas_2d_frame.pack_forget()
            self.canvas_3d_frame.pack(fill=tk.BOTH, expand=True)
            self.scale_frame.state(['disabled'])
        else:
            self.view_mode = ViewMode.SINGLE_FRAME_2D
            self.mode_btn_text.set("Switch to 3D View")
            self.canvas_3d_frame.pack_forget()
            self.canvas_2d_frame.pack(fill=tk.BOTH, expand=True)
            self.scale_frame.state(['!disabled'])
        
        self._update_display()

    def _update_display(self):
        thresh = self.thresh_var.get()
        
        if self.view_mode == ViewMode.SINGLE_FRAME_2D:
            idx = self.frame_var.get()
            self.title_var.set(f"2D View | Frame {idx}/{self.total_frames} | Threshold: {thresh}")
            self._render_2d(idx, thresh)
        else:
            self.title_var.set(f"3D View | Voxel Cloud | Threshold: {thresh}")
            self._render_3d(thresh)

    def _render_2d(self, idx, thresh):
        # 1. Prepare Correlation Frame for Analysis (Grayscale)
        frame_corr_gray = self.volume_full[idx] 
        
        # 2. Prepare RGB Frame (Left Image - Visual Context)
        frame_rgb_bgr = self.frames_rgb[idx]
        show_orig = cv2.cvtColor(frame_rgb_bgr, cv2.COLOR_BGR2RGB)
        
        # 3. Prepare Correlation Frame (Middle Image - Source Data)
        frame_corr_raw = self.frames_corr[idx]
        # Check dimensionality to handle conversion correctly
        if len(frame_corr_raw.shape) == 2:
             show_corr = cv2.cvtColor(frame_corr_raw, cv2.COLOR_GRAY2RGB)
        else:
             show_corr = cv2.cvtColor(frame_corr_raw, cv2.COLOR_BGR2RGB)

        # 4. Create Binary Mask (Right Image - Result)
        thresh_type = cv2.THRESH_BINARY_INV if self.spot_type == 'dark' else cv2.THRESH_BINARY
        _, bin_mask = cv2.threshold(frame_corr_gray, thresh, 255, thresh_type)
        show_thresh = cv2.cvtColor(bin_mask, cv2.COLOR_GRAY2RGB)

        # 5. Dynamic size calculation
        container_w = self.canvas_2d_frame.winfo_width()
        container_h = self.canvas_2d_frame.winfo_height()
        
        if container_w < 10: 
            container_w = 800
            container_h = 600

        # Split width by 3 for RGB | Corr | Thresh
        target_w = container_w // 3
        target_h = container_h

        self._set_image(self.lbl_orig, show_orig, (target_w, target_h))
        self._set_image(self.lbl_corr, show_corr, (target_w, target_h))
        self._set_image(self.lbl_thresh, show_thresh, (target_w, target_h))

    def _render_3d(self, thresh):
        self.ax.clear()
        
        # Data Thresholding on volume_small (derived from frames_corr)
        if self.spot_type == 'dark':
            mask = self.volume_small < thresh
        else:
            mask = self.volume_small > thresh

        z_idxs, y_idxs, x_idxs = np.where(mask)

        # Downsampling for interactivity
        max_points = 5000
        total_points = len(z_idxs)
        if total_points > max_points:
            choices = np.random.choice(total_points, max_points, replace=False)
            z_idxs = z_idxs[choices]
            y_idxs = y_idxs[choices]
            x_idxs = x_idxs[choices]

        # Add Reference Plane
        d_z, d_y, d_x = self.volume_small.shape
        verts = [
            [(0, 0, 0), (d_x, 0, 0), (d_x, d_y, 0), (0, d_y, 0)]
        ]
        poly = Poly3DCollection(verts, alpha=0.3, facecolors='cyan', edgecolors='blue')
        self.ax.add_collection3d(poly)

        # Scatter Plot
        marker_size = 25 
        self.ax.scatter(x_idxs, y_idxs, -z_idxs, c='g', marker='s', s=marker_size, alpha=1.0, depthshade=False)
        
        self.ax.set_xlim(0, d_x)
        self.ax.set_ylim(d_y, 0)
        self.ax.set_zlim(-d_z, 0)
        
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Frame")
        
        self.mpl_canvas.draw()

    def _set_image(self, label, img_arr, target_dims: Tuple[int, int]):
        tw, th = target_dims
        h, w = img_arr.shape[:2]
        
        if tw <= 0 or th <= 0 or w <= 0 or h <= 0:
            return

        scale = min(tw/w, th/h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        new_w = max(1, new_w)
        new_h = max(1, new_h)

        img_arr = cv2.resize(img_arr, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        im_pil = Image.fromarray(img_arr)
        im_tk = ImageTk.PhotoImage(image=im_pil)
        label.image = im_tk 
        label.configure(image=im_tk)

    def _confirm(self):
        self.result = self.thresh_var.get()
        self.selection_state = SelectionState.CONFIRMED
        self.root.quit()

    def _redo(self):
        self.selection_state = SelectionState.REDO_COLORSPACE
        self.root.quit()

    def _on_cancel(self):
        self.selection_state = SelectionState.CANCELLED
        self.root.quit()