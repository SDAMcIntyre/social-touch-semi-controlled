import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from typing import List, Optional, Tuple
from enum import Enum, auto

class SelectionState(Enum):
    """
    Enumeration for the possible exit states of the Threshold Selector.
    """
    PENDING = auto()
    CONFIRMED = auto()
    CANCELLED = auto()
    REDO_COLORSPACE = auto()

class ThresholdSelectorTool:
    """
    A Tkinter-based GUI tool to interactively select a threshold value from video frames.
    
    Architectural Note:
    This class implements the 'Quit-Clean-Destroy' pattern to manage Tcl/Tk resource 
    lifecycles safely, preventing RuntimeErrors during Python garbage collection.
    
    UI Update:
    Implements 'clam' theme override to allow custom button coloration and enhanced
    visual accessibility.
    """

    def __init__(
            self, 
            frames: List[np.ndarray], 
            video_name: str = "video", 
            threshold: int = 127,
            spot_type: str = 'dark'):
        
        if not frames:
            raise ValueError("The provided frame list cannot be empty.")
            
        self.frames = frames
        self.spot_type = spot_type
        self.total_frames = len(self.frames)
        self.video_name = video_name

        self.init_threshold = threshold

        # --- State variables ---
        self.result: Optional[int] = None
        self.selection_state: SelectionState = SelectionState.PENDING

        # --- Tkinter UI elements ---
        self.root: Optional[tk.Tk] = None
        self.original_img_label: Optional[ttk.Label] = None
        self.thresholded_img_label: Optional[ttk.Label] = None
        self.image_pane: Optional[ttk.Frame] = None
        self.style: Optional[ttk.Style] = None
        
        # Tkinter variables (Initialized in run to ensure binding to correct root)
        self.frame_var: Optional[tk.IntVar] = None
        self.thresh_var: Optional[tk.IntVar] = None
        self.title_var: Optional[tk.StringVar] = None

        self.target_display_size: Optional[Tuple[int, int]] = None

    def run(self) -> Optional[int]:
        """
        Creates and runs the GUI application, then returns the selected value.
        """
        self.root = tk.Tk()
        self.root.title("Interactive Threshold Selection")
        self.root.protocol("WM_DELETE_WINDOW", self._on_cancel)

        # Initialize variables bound to this specific root
        self.frame_var = tk.IntVar(value=0)
        self.thresh_var = tk.IntVar(value=self.init_threshold)
        self.title_var = tk.StringVar()

        self._setup_styles()
        self._setup_ui()

        # Window Setup
        self.root.state('zoomed')
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after_idle(self.root.attributes, '-topmost', False)
        self.root.update_idletasks()

        # Layout calculation
        pane_width = self.image_pane.winfo_width()
        pane_height = self.image_pane.winfo_height()
        self.target_display_size = (pane_width // 2, pane_height)

        self._update_display()

        print(f"\n--- Interactive Threshold Selection: {self.video_name} ---")
        
        # BLOCKING CALL: Runs until self.root.quit() is called
        self.root.mainloop()

        # --- CLEANUP PHASE (The Fix) ---
        # Explicitly delete Tkinter variables while self.root is still valid.
        # This prevents __del__ from firing on a destroyed interpreter later.
        if self.frame_var is not None: del self.frame_var
        if self.thresh_var is not None: del self.thresh_var
        if self.title_var is not None: del self.title_var
        
        # Now it is safe to destroy the interpreter
        if self.root:
            self.root.destroy()
            self.root = None

        return self.result

    def _setup_styles(self):
        """
        Configures the visual styles for the application.
        
        Architecture Note:
        To support custom background colors on Windows/Mac, we must force the 
        'clam' theme. Standard themes ignore background color changes.
        """
        self.style = ttk.Style(self.root)
        current_theme = self.style.theme_use()
        
        # Force 'clam' theme for button color support if not already active
        # This might change the look of sliders slightly, but ensures buttons work.
        self.style.theme_use('clam')

        # Define Common Font
        button_font = ("Helvetica", 11, "bold")

        # --- 1. Confirm Style (Green) ---
        self.style.configure(
            "Confirm.TButton",
            background="#2E7D32",  # Dark Green
            foreground="white",
            font=button_font,
            borderwidth=1,
            focuscolor="none"
        )
        self.style.map(
            "Confirm.TButton",
            background=[("active", "#4CAF50")],  # Lighter Green on Hover
            relief=[("pressed", "sunken")]
        )

        # --- 2. Redo Style (Orange/Amber) ---
        self.style.configure(
            "Redo.TButton",
            background="#F57C00",  # Orange
            foreground="white",
            font=button_font,
            borderwidth=1,
            focuscolor="none"
        )
        self.style.map(
            "Redo.TButton",
            background=[("active", "#FF9800")],  # Lighter Orange on Hover
            relief=[("pressed", "sunken")]
        )

        # --- 3. Cancel Style (Red) ---
        self.style.configure(
            "Cancel.TButton",
            background="#C62828",  # Dark Red
            foreground="white",
            font=button_font,
            borderwidth=1,
            focuscolor="none"
        )
        self.style.map(
            "Cancel.TButton",
            background=[("active", "#EF5350")],  # Lighter Red on Hover
            relief=[("pressed", "sunken")]
        )

        # Revert frames/labels to standard look if desired, or keep clam style
        # For consistency, we leave the rest as 'clam' default.

    def _setup_ui(self):
        """Creates and arranges all the Tkinter widgets."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # Title
        title_label = ttk.Label(main_frame, textvariable=self.title_var, font=("Helvetica", 14, "bold"))
        title_label.grid(row=0, column=0, sticky="w", pady=(0, 10))

        # Images
        self.image_pane = ttk.Frame(main_frame)
        self.image_pane.grid(row=1, column=0, sticky="nsew", padx=5, pady=5) 
        main_frame.rowconfigure(1, weight=1) 
        main_frame.columnconfigure(0, weight=1)

        self.original_img_label = ttk.Label(self.image_pane)
        self.original_img_label.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(0, 5))
        
        self.thresholded_img_label = ttk.Label(self.image_pane)
        self.thresholded_img_label.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(5, 0))

        # Controls
        controls_pane = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        controls_pane.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        controls_pane.columnconfigure(1, weight=1)

        # Sliders
        ttk.Label(controls_pane, text="Frame:").grid(row=0, column=0, sticky="w")
        frame_slider = ttk.Scale(controls_pane, from_=0, to=self.total_frames - 1,
                                 orient=tk.HORIZONTAL, variable=self.frame_var,
                                 command=self._on_slider_change)
        frame_slider.grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Label(controls_pane, textvariable=self.frame_var, width=4).grid(row=0, column=2)

        ttk.Label(controls_pane, text="Threshold:").grid(row=1, column=0, sticky="w")
        thresh_slider = ttk.Scale(controls_pane, from_=0, to=255,
                                  orient=tk.HORIZONTAL, variable=self.thresh_var,
                                  command=self._on_slider_change)
        thresh_slider.grid(row=1, column=1, sticky="ew", padx=5)
        ttk.Label(controls_pane, textvariable=self.thresh_var, width=4).grid(row=1, column=2)

        # Buttons - Modified to use custom styles and padding
        button_pane = ttk.Frame(main_frame)
        button_pane.grid(row=3, column=0, sticky="e", pady=(20, 10))
        
        # Confirm Button (Green)
        confirm_button = ttk.Button(
            button_pane, 
            text="Confirm (Enter)", 
            command=self._on_confirm,
            style="Confirm.TButton",
            width=20
        )
        confirm_button.pack(side=tk.RIGHT, padx=5, ipady=5) # ipady adds internal height
        
        # Redo Button (Orange)
        redo_button = ttk.Button(
            button_pane, 
            text="Redo (Reselect Colorspace)", 
            command=self._on_redo,
            style="Redo.TButton",
            width=25
        )
        redo_button.pack(side=tk.RIGHT, padx=5, ipady=5)

        # Cancel Button (Red)
        cancel_button = ttk.Button(
            button_pane, 
            text="Cancel (Esc)", 
            command=self._on_cancel,
            style="Cancel.TButton",
            width=15
        )
        cancel_button.pack(side=tk.RIGHT, ipady=5)
        
        # Bindings
        self.root.bind('<Return>', lambda e: self._on_confirm())
        self.root.bind('<Escape>', lambda e: self._on_cancel())
        self.root.bind('<Left>', lambda e: self._increment_frame(-1))
        self.root.bind('<Right>', lambda e: self._increment_frame(1))
        self.root.bind('<Control-Left>', lambda e: self._increment_frame(-10))
        self.root.bind('<Control-Right>', lambda e: self._increment_frame(10))

    def _increment_frame(self, step: int):
        if self.frame_var:
            current_frame = self.frame_var.get()
            new_frame = max(0, min(current_frame + step, self.total_frames - 1))
            if new_frame != current_frame:
                self.frame_var.set(new_frame)
                self._on_slider_change()
    
    def _on_slider_change(self, _=None):
        self._update_display()

    def _update_display(self):
        # Guard clause in case update is triggered during shutdown
        if not self.root or not self.frame_var:
            return

        frame_idx = self.frame_var.get()
        threshold_val = self.thresh_var.get()

        self.title_var.set(f"Video: {self.video_name}  |  Frame: {frame_idx}/{self.total_frames - 1}  |  Threshold: {threshold_val}")
        
        original_frame = self.frames[frame_idx]
        if len(original_frame.shape) == 3:
            gray_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = original_frame.copy()

        thresh_type = cv2.THRESH_BINARY_INV if self.spot_type == 'dark' else cv2.THRESH_BINARY
        _, thresholded_frame = cv2.threshold(gray_frame, threshold_val, 255, thresh_type)

        display_gray = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        display_thresh = cv2.cvtColor(thresholded_frame, cv2.COLOR_GRAY2BGR)

        self._update_image_label(self.original_img_label, display_gray)
        self._update_image_label(self.thresholded_img_label, display_thresh)

    def _update_image_label(self, label: ttk.Label, frame: np.ndarray):
        if self.target_display_size:
            target_w, target_h = self.target_display_size
            h, w = frame.shape[:2]
            if w > 0 and h > 0:
                scale = min(target_w / w, target_h / h)
                if scale < 1.0 or (target_w > w and target_h > h):
                    new_w, new_h = int(w * scale), int(h * scale)
                    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=interp)

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        
        label.imgtk = imgtk
        label.config(image=imgtk)

    def _on_confirm(self):
        """Stores the result and quits the mainloop (does not destroy root yet)."""
        self.result = self.thresh_var.get()
        self.selection_state = SelectionState.CONFIRMED
        print(f"✅ Threshold value {self.result} confirmed.")
        self.root.quit() # Exit mainloop

    def _on_redo(self):
        """Marks state as REDO and quits the mainloop."""
        self.result = None
        self.selection_state = SelectionState.REDO_COLORSPACE
        print("🔄 Redo processing requested: Reselect Colorspace.")
        self.root.quit() # Exit mainloop

    def _on_cancel(self):
        """Marks state as CANCELLED and quits the mainloop."""
        self.result = None
        self.selection_state = SelectionState.CANCELLED
        print("❌ Threshold selection cancelled by user.")
        self.root.quit() # Exit mainloop