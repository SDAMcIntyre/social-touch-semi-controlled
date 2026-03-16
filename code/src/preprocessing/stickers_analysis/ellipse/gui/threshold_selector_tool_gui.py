import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from typing import List, Optional, Tuple, Union
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
    
    Architectural Fixes:
    - Lifecycle Management: Decoupled Tcl interpreter lifecycle from tool execution.
      Accepts an optional 'master' (root) to allow reuse of the existing Tcl context.
    - Windowing: Uses tk.Toplevel when a master is provided, preventing interpreter crashes.
    - Garbage Collection: Explicitly clears Matplotlib figures before window destruction.
    """

    def __init__(
            self, 
            frames_rgb: List[np.ndarray], 
            frames_corr: List[np.ndarray], 
            video_name: str = "video", 
            threshold: int = 127,
            spot_type: str = 'dark',
            master: Optional[Union[tk.Tk, tk.Toplevel]] = None):
        """
        Args:
            master: Optional Tkinter root or window. If provided, tool runs as a Toplevel 
                    window, preventing Tcl interpreter crashes on repeated runs.
        """
        
        if not frames_rgb:
            raise ValueError("Frame list cannot be empty.")
        
        # Validation
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

        # Architecture: UI Dependency Injection
        self.master_ref = master
        self.owns_root = False

        self._prepare_volumes()

        # State
        self.result: Optional[int] = None
        self.selection_state: SelectionState = SelectionState.PENDING
        self.view_mode: ViewMode = ViewMode.SINGLE_FRAME_2D
        self.last_2d_size: Tuple[int, int] = (0, 0)

        # UI Components
        self.window: Optional[Union[tk.Tk, tk.Toplevel]] = None
        self.canvas_2d_frame: Optional[ttk.Frame] = None
        self.canvas_3d_frame: Optional[ttk.Frame] = None
        
        self.fig = None
        self.ax = None
        self.mpl_canvas = None

        # Tkinter Vars
        self.frame_var: Optional[tk.IntVar] = None
        self.thresh_var: Optional[tk.IntVar] = None
        self.title_var: Optional[tk.StringVar] = None
        self.mode_btn_text: Optional[tk.StringVar] = None

    def _prepare_volumes(self):
        print("Processing volume data from correlation frames...")
        shape = self.frames_corr[0].shape
        
        if len(shape) == 3:
            gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in self.frames_corr]
        else:
            gray_frames = self.frames_corr
        
        self.volume_full = np.stack(gray_frames)
        self.full_h, self.full_w = self.volume_full.shape[1], self.volume_full.shape[2]

        target_dim = 100
        h, w = self.full_h, self.full_w
        scale = target_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized_frames = [cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_NEAREST) 
                          for f in gray_frames]
        
        self.volume_small = np.stack(resized_frames)
        self.small_h, self.small_w = self.volume_small.shape[1], self.volume_small.shape[2]
        print(f"Volume cached. Full: {self.volume_full.shape}, Small: {self.volume_small.shape}")

    def run(self) -> Optional[int]:
        # Architecture: Determine Window Type
        if self.master_ref:
            self.owns_root = False
            self.window = tk.Toplevel(self.master_ref)
        else:
            self.owns_root = True
            self.window = tk.Tk()
            self.window.state('zoomed')

        self.window.title(f"Interactive Threshold: {self.video_name}")
        self.window.protocol("WM_DELETE_WINDOW", self._on_cancel)
        
        self._init_vars()
        self._setup_styles()
        self._setup_layout()

        # Initial Render
        self.window.update()
        
        if self.view_mode == ViewMode.POINT_CLOUD_3D:
             self.mode_btn_text.set("Switch to 2D View")
             self.canvas_2d_frame.pack_forget()
             self.canvas_3d_frame.pack(fill=tk.BOTH, expand=True)
             self.scale_frame.state(['disabled'])
        else:
             self.mode_btn_text.set("Switch to 3D View")
             self.canvas_3d_frame.pack_forget()
             self.canvas_2d_frame.pack(fill=tk.BOTH, expand=True)
             self.scale_frame.state(['!disabled'])
        
        self._update_display()

        print(f"\n--- Tool Running: {self.video_name} ---")
        
        # Lifecycle: Block execution
        if self.owns_root:
            self.window.mainloop()
        else:
            # If using Toplevel, we wait for this specific window to close
            # This makes the call blocking, just like mainloop
            self.window.wait_window(self.window)

        # Cleanup is handled by _close_tool called by protocols
        return self.result

    def _init_vars(self):
        self.frame_var = tk.IntVar(value=0, master=self.window)
        self.thresh_var = tk.IntVar(value=self.init_threshold, master=self.window)
        self.title_var = tk.StringVar(master=self.window)
        self.mode_btn_text = tk.StringVar(value="Switch to 3D View", master=self.window)

    def _close_tool(self):
        """
        Safely closes the tool without destroying the master root if it exists.
        """
        # Explicitly clean Matplotlib to prevent backend leakage
        if self.fig:
            self.fig.clf()
            matplotlib.pyplot.close(self.fig)
            self.fig = None
        
        if self.window:
            if self.owns_root:
                self.window.destroy()
            else:
                self.window.destroy()
                # Do not destroy master
            self.window = None

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
        main = ttk.Frame(self.window)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(main, textvariable=self.title_var, font=("Segoe UI", 14, "bold")).pack(pady=(0, 10), anchor="w")

        self.view_container = ttk.Frame(main)
        self.view_container.pack(fill=tk.BOTH, expand=True)

        # 2D Frame
        self.canvas_2d_frame = ttk.Frame(self.view_container)
        self.canvas_2d_frame.pack_propagate(False) 
        
        self.lbl_orig = ttk.Label(self.canvas_2d_frame, anchor="center")
        self.lbl_orig.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2)
        
        self.lbl_corr = ttk.Label(self.canvas_2d_frame, anchor="center")
        self.lbl_corr.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2)

        self.lbl_thresh = ttk.Label(self.canvas_2d_frame, anchor="center")
        self.lbl_thresh.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2)

        self.canvas_2d_frame.bind("<Configure>", self._on_2d_resize)

        # 3D Frame
        self.canvas_3d_frame = ttk.Frame(self.view_container)
        
        # Matplotlib Initialization
        self.fig = Figure(figsize=(5, 4), dpi=100, facecolor='#f0f0f0')
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#f0f0f0')
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_3d_frame)
        self.mpl_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Controls
        controls = ttk.LabelFrame(main, text="Parameters", padding=10)
        controls.pack(fill=tk.X, pady=(10, 0))
        controls.columnconfigure(1, weight=1)
        
        ttk.Label(controls, text="Frame Index:").grid(row=0, column=0, sticky="e", padx=5)
        self.scale_frame = ttk.Scale(controls, from_=0, to=self.total_frames-1, variable=self.frame_var, 
                                     orient=tk.HORIZONTAL, command=lambda v: self._update_display())
        self.scale_frame.grid(row=0, column=1, sticky="ew")
        ttk.Label(controls, textvariable=self.frame_var, width=4).grid(row=0, column=2, padx=5)

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

        self.window.bind('<Return>', lambda e: self._confirm())
        self.window.bind('<Escape>', lambda e: self._on_cancel())

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
        if not self.window: return # Guard against updates during destruction
        thresh = self.thresh_var.get()
        if self.view_mode == ViewMode.SINGLE_FRAME_2D:
            idx = self.frame_var.get()
            self.title_var.set(f"2D View | Frame {idx}/{self.total_frames} | Threshold: {thresh}")
            self._render_2d(idx, thresh)
        else:
            self.title_var.set(f"3D View | Voxel Cloud | Threshold: {thresh}")
            self._render_3d(thresh)

    def _render_2d(self, idx, thresh):
        frame_corr_gray = self.volume_full[idx] 
        frame_rgb_bgr = self.frames_rgb[idx]
        show_orig = cv2.cvtColor(frame_rgb_bgr, cv2.COLOR_BGR2RGB)
        
        frame_corr_raw = self.frames_corr[idx]
        if len(frame_corr_raw.shape) == 2:
             show_corr = cv2.cvtColor(frame_corr_raw, cv2.COLOR_GRAY2RGB)
        else:
             show_corr = cv2.cvtColor(frame_corr_raw, cv2.COLOR_BGR2RGB)

        thresh_type = cv2.THRESH_BINARY_INV if self.spot_type == 'dark' else cv2.THRESH_BINARY
        _, bin_mask = cv2.threshold(frame_corr_gray, thresh, 255, thresh_type)
        show_thresh = cv2.cvtColor(bin_mask, cv2.COLOR_GRAY2RGB)

        container_w = self.canvas_2d_frame.winfo_width()
        container_h = self.canvas_2d_frame.winfo_height()
        
        if container_w < 10: 
            container_w = 800
            container_h = 600

        target_w = container_w // 3
        target_h = container_h

        self._set_image(self.lbl_orig, show_orig, (target_w, target_h))
        self._set_image(self.lbl_corr, show_corr, (target_w, target_h))
        self._set_image(self.lbl_thresh, show_thresh, (target_w, target_h))

    def _render_3d(self, thresh):
        self.ax.clear()
        if self.spot_type == 'dark':
            mask = self.volume_small < thresh
        else:
            mask = self.volume_small > thresh

        z_idxs, y_idxs, x_idxs = np.where(mask)
        max_points = 5000
        total_points = len(z_idxs)
        if total_points > max_points:
            choices = np.random.choice(total_points, max_points, replace=False)
            z_idxs = z_idxs[choices]
            y_idxs = y_idxs[choices]
            x_idxs = x_idxs[choices]

        d_z, d_y, d_x = self.volume_small.shape
        verts = [[(0, 0, 0), (d_x, 0, 0), (d_x, d_y, 0), (0, d_y, 0)]]
        poly = Poly3DCollection(verts, alpha=0.3, facecolors='cyan', edgecolors='blue')
        self.ax.add_collection3d(poly)
        self.ax.scatter(x_idxs, y_idxs, -z_idxs, c='g', marker='s', s=25, alpha=1.0, depthshade=False)
        self.ax.set_xlim(0, d_x)
        self.ax.set_ylim(d_y, 0)
        self.ax.set_zlim(-d_z, 0)
        self.mpl_canvas.draw()

    def _set_image(self, label, img_arr, target_dims: Tuple[int, int]):
        tw, th = target_dims
        h, w = img_arr.shape[:2]
        if tw <= 0 or th <= 0 or w <= 0 or h <= 0: return
        scale = min(tw/w, th/h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        img_arr = cv2.resize(img_arr, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        im_pil = Image.fromarray(img_arr)
        im_tk = ImageTk.PhotoImage(image=im_pil)
        label.image = im_tk 
        label.configure(image=im_tk)

    def _confirm(self):
        self.result = self.thresh_var.get()
        self.selection_state = SelectionState.CONFIRMED
        self._close_tool()

    def _redo(self):
        self.selection_state = SelectionState.REDO_COLORSPACE
        self._close_tool()

    def _on_cancel(self):
        self.selection_state = SelectionState.CANCELLED
        self._close_tool()