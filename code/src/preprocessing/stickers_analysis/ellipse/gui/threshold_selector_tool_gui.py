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
    - Geometry Propagation: Disabled on 2D render frame to prevent infinite resize loops.
    - Dynamic Resizing: 2D view listens to <Configure> events to maximize video usage.
    - Default 3D: Initializes directly into POINT_CLOUD_3D mode.
    - Voxel Simulation: 3D scatter plots use square markers scaled to approximate 
      continuous surfaces (touching pixels).
    - Contextual Geometry: Renders a transparent reference plane for video bounds.
    - Layout Optimization: Mode switch button grouped with action buttons.
    """

    def __init__(
            self, 
            frames: List[np.ndarray], 
            video_name: str = "video", 
            threshold: int = 127,
            spot_type: str = 'dark'):
        
        if not frames:
            raise ValueError("Frame list cannot be empty.")
            
        self.frames = frames
        self.total_frames = len(self.frames)
        self.video_name = video_name
        self.init_threshold = threshold
        self.spot_type = spot_type

        # --- Data Preparation ---
        self._prepare_volumes()

        # --- State ---
        self.result: Optional[int] = None
        self.selection_state: SelectionState = SelectionState.PENDING
        # Requirement: 3D mode is shown by default
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
        Pre-processes video data into 3D arrays.
        """
        print("Processing volume data...")
        shape = self.frames[0].shape
        if len(shape) == 3:
            gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in self.frames]
        else:
            gray_frames = self.frames
        
        self.volume_full = np.stack(gray_frames) # Shape: (Z, Y, X)
        self.full_h, self.full_w = self.volume_full.shape[1], self.volume_full.shape[2]

        # Downsample for 3D visualization (Performance Optimization)
        # Target roughly 100x100xZ resolution
        target_dim = 100
        h, w = shape[:2]
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
        
        # --- ARCHITECTURAL FIX: DISABLE GEOMETRY PROPAGATION ---
        # This prevents the frame from expanding when children (images) are resized,
        # breaking the infinite resize loop.
        self.canvas_2d_frame.pack_propagate(False) 
        
        # Split 2D frame into two halves
        self.lbl_orig = ttk.Label(self.canvas_2d_frame, anchor="center")
        self.lbl_orig.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2)
        
        self.lbl_thresh = ttk.Label(self.canvas_2d_frame, anchor="center")
        self.lbl_thresh.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2)

        # Bind Resize Event for Responsive 2D
        self.canvas_2d_frame.bind("<Configure>", self._on_2d_resize)

        # 3D View Frame
        self.canvas_3d_frame = ttk.Frame(self.view_container)
        
        # Initialize Matplotlib Figure
        self.fig = Figure(figsize=(5, 4), dpi=100, facecolor='#f0f0f0')
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#f0f0f0')
        
        # Adjust subplot parameters to maximize 3D view area
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

        # Modified Layout: All buttons grouped on the right side
        # Stacking order: pack(side=RIGHT) stacks from Right to Left.
        # Desired Order (Visual Right-to-Left): Confirm -> Redo -> Cancel -> Switch Mode
        
        ttk.Button(btn_frame, text="Confirm", command=self._confirm, style="Confirm.TButton").pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="Redo", command=self._redo, style="Redo.TButton").pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self._on_cancel, style="Cancel.TButton").pack(side=tk.RIGHT, padx=5)
        
        # Mode button moved here to be close to the other buttons
        ttk.Button(btn_frame, textvariable=self.mode_btn_text, command=self._toggle_mode, style="Mode.TButton").pack(side=tk.RIGHT, padx=5)

        self.root.bind('<Return>', lambda e: self._confirm())
        self.root.bind('<Escape>', lambda e: self._on_cancel())

    def _on_2d_resize(self, event):
        """
        Event handler for window resizing in 2D mode. 
        Debounces slightly to prevent massive CPU load during drag.
        """
        if self.view_mode != ViewMode.SINGLE_FRAME_2D:
            return
            
        # Basic debounce: only update if size changed significantly
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
        frame_gray = self.volume_full[idx] 
        
        thresh_type = cv2.THRESH_BINARY_INV if self.spot_type == 'dark' else cv2.THRESH_BINARY
        _, bin_mask = cv2.threshold(frame_gray, thresh, 255, thresh_type)

        show_orig = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2RGB)
        show_thresh = cv2.cvtColor(bin_mask, cv2.COLOR_GRAY2RGB)

        # Dynamic size calculation based on container size
        # We assume the container is split 50/50 for the two images
        container_w = self.canvas_2d_frame.winfo_width()
        container_h = self.canvas_2d_frame.winfo_height()
        
        # Fallback if window hasn't rendered yet
        if container_w < 10: 
            container_w = 800
            container_h = 600

        target_w = container_w // 2
        target_h = container_h

        self._set_image(self.lbl_orig, show_orig, (target_w, target_h))
        self._set_image(self.lbl_thresh, show_thresh, (target_w, target_h))

    def _render_3d(self, thresh):
        self.ax.clear()
        
        # 1. Data Thresholding
        if self.spot_type == 'dark':
            mask = self.volume_small < thresh
        else:
            mask = self.volume_small > thresh

        z_idxs, y_idxs, x_idxs = np.where(mask)

        # 2. Downsampling for interactivity
        max_points = 5000
        total_points = len(z_idxs)
        if total_points > max_points:
            choices = np.random.choice(total_points, max_points, replace=False)
            z_idxs = z_idxs[choices]
            y_idxs = y_idxs[choices]
            x_idxs = x_idxs[choices]

        # 3. Add Semi-Transparent Reference Rectangle (Video Size)
        # We place this at Z=0 (top of the stack)
        d_z, d_y, d_x = self.volume_small.shape
        
        # Vertices of the video plane (X, Y, Z)
        # Note: In matplotlib 3d, Z is up. We map image Z to negative Z.
        verts = [
            [(0, 0, 0), (d_x, 0, 0), (d_x, d_y, 0), (0, d_y, 0)]
        ]
        poly = Poly3DCollection(verts, alpha=0.3, facecolors='cyan', edgecolors='blue')
        self.ax.add_collection3d(poly)

        # 4. Scatter Plot with "Touching" Markers
        # To make points touch, we use square markers.
        # Calculation: We need the marker to cover 1 unit in data coordinates.
        # Heuristic: s is in points^2. 
        # A crude approximation for "filling" the grid in a default view.
        # Since exact data-to-pixel calculation changes with zoom, we pick a 
        # reasonably large constant relative to the small volume dimension (100).
        marker_size = 25 
        
        self.ax.scatter(x_idxs, y_idxs, -z_idxs, c='g', marker='s', s=marker_size, alpha=1.0, depthshade=False)
        
        # 5. Axes Configuration
        self.ax.set_xlim(0, d_x)
        self.ax.set_ylim(d_y, 0) # Invert Y for image coords
        self.ax.set_zlim(-d_z, 0)
        
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Frame")
        
        # Remove grid/background for cleaner look if desired, or keep for reference
        # self.ax.grid(False) 

        self.mpl_canvas.draw()

    def _set_image(self, label, img_arr, target_dims: Tuple[int, int]):
        tw, th = target_dims
        h, w = img_arr.shape[:2]
        
        # Safety check for zero dimensions
        if tw <= 0 or th <= 0 or w <= 0 or h <= 0:
            return

        # Calculate aspect-ratio preserving resize
        scale = min(tw/w, th/h)
        
        # Resize based on target dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Ensure dimensions are at least 1x1
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