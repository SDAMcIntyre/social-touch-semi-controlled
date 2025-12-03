import tkinter as tk
from tkinter import ttk, messagebox
import os
from pathlib import Path
import cv2
from PIL import Image, ImageTk

# Assuming these imports are in your project structure
from preprocessing.common import (
    VideoFramesSelector,
    VideoMP4Manager
)


# --- High DPI Awareness for Windows 11 ---
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

# --- Missing Class Implementation: VideoFramesSelector ---
class VideoFramesSelector:
    """
    A GUI to view a video and select specific frame indices.
    Implemented to replace the missing dependency and fix API mismatches.
    """
    def __init__(self, parent, video_manager, last_frame, title="Frame Selector", initial_selection=None):
        self.parent = parent
        self.manager = video_manager
        self.total_frames = last_frame + 1
        self.title = title
        self.selected_frames = set(initial_selection) if initial_selection else set()
        self.proceed_was_clicked = False
        self.current_frame_idx = 0
        self.photo_image = None  # Keep reference to avoid garbage collection

        # UI Setup
        self.parent.title(self.title)
        self.parent.geometry("900x700")
        
        # Main Layout
        self.main_container = ttk.Frame(self.parent)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Canvas for Video
        self.canvas_frame = ttk.Frame(self.main_container)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(self.canvas_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Controls
        self.controls_frame = ttk.Frame(self.main_container)
        self.controls_frame.pack(fill=tk.X, pady=10)

        # Slider
        self.slider = tk.Scale(
            self.controls_frame, 
            from_=0, 
            to=self.total_frames - 1, 
            orient=tk.HORIZONTAL, 
            command=self._on_slider_move,
            label="Frame Index"
        )
        self.slider.pack(fill=tk.X, expand=True, side=tk.TOP)

        # Buttons
        btn_frame = ttk.Frame(self.controls_frame)
        btn_frame.pack(fill=tk.X, pady=5)

        self.btn_toggle = ttk.Button(btn_frame, text="Select/Deselect Frame", command=self._toggle_selection)
        self.btn_toggle.pack(side=tk.LEFT, padx=5)

        self.status_lbl = ttk.Label(btn_frame, text=f"Selected: {len(self.selected_frames)}")
        self.status_lbl.pack(side=tk.LEFT, padx=15)

        ttk.Button(btn_frame, text="Save & Proceed", command=self._on_proceed).pack(side=tk.RIGHT, padx=5)
        
        # Initial Render
        self._update_image(0)
        self._update_status()

    def _on_slider_move(self, value):
        idx = int(value)
        self.current_frame_idx = idx
        self._update_image(idx)
        self._update_status()

    def _toggle_selection(self):
        if self.current_frame_idx in self.selected_frames:
            self.selected_frames.remove(self.current_frame_idx)
        else:
            self.selected_frames.add(self.current_frame_idx)
        self._update_status()

    def _update_status(self):
        is_selected = self.current_frame_idx in self.selected_frames
        status_text = f"Frame {self.current_frame_idx} | Total Selected: {len(self.selected_frames)}"
        if is_selected:
            status_text += " [SELECTED]"
            self.btn_toggle.config(text="Remove Selection")
        else:
            self.btn_toggle.config(text="Select Frame")
        self.status_lbl.config(text=status_text)

    def _update_image(self, frame_idx):
        try:
            # Use the VideoMP4Manager's __getitem__ logic
            frame = self.manager[frame_idx]
            
            # Convert BGR (OpenCV) to RGB (PIL)
            # Note: VideoMP4Manager might already return RGB if configured, 
            # but default is BGR. We assume BGR based on provided file default.
            if self.manager.color_format.name == 'BGR':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize for display performance if necessary (optional optimization)
            h, w, _ = frame.shape
            display_h = 500
            scale = display_h / h
            display_w = int(w * scale)
            frame_resized = cv2.resize(frame, (display_w, display_h))

            image = Image.fromarray(frame_resized)
            self.photo_image = ImageTk.PhotoImage(image)
            
            # Update Canvas
            self.canvas.delete("all")
            # Center image
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            # If canvas not ready, use default
            if cw < 10: cw = 800
            if ch < 10: ch = 500
            
            self.canvas.create_image(cw//2, ch//2, image=self.photo_image, anchor=tk.CENTER)
            
        except Exception as e:
            print(f"Error displaying frame {frame_idx}: {e}")

    def _on_proceed(self):
        self.proceed_was_clicked = True
        self.parent.destroy()


# --- Main Class: Selector for MULTIPLE Videos ---
class MultiVideoFramesSelector:
    """
    A GUI to manage frame selection across multiple video files.
    """
    def __init__(self, parent, video_paths, initial_selections=None):
        self.parent = parent
        self.video_paths = [str(p) for p in video_paths]
        self.video_paths_dict = {os.path.basename(p): p for p in self.video_paths}

        # --- Data Storage ---
        self.all_selected_frames = {}
        if initial_selections:
            for basename, frames in initial_selections.items():
                if basename in self.video_paths_dict:
                    full_path = self.video_paths_dict[basename]
                    self.all_selected_frames[full_path] = sorted(list(set(map(int, frames))))
                else:
                    print(f"⚠️ Warning: Basename '{basename}' from initial_selections not found.")

        self.selectors_opened = {filename: False for filename in self.video_paths_dict.keys()}
        self.validated = False

        # --- UI State ---
        self.selected_video = tk.StringVar()
        self.status_labels = {}

        # --- Window Configuration ---
        self.parent.title("Multi-Video Selector")

        # --- UI Elements ---
        main_frame = ttk.Frame(self.parent, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.parent.columnconfigure(0, weight=1)
        self.parent.rowconfigure(0, weight=1)

        ttk.Label(main_frame, text="Select a video to process:").grid(row=0, column=0, pady=(0, 10), sticky=tk.W)

        self.video_list_frame = ttk.LabelFrame(main_frame, text="Videos | Status", padding=10)
        self.video_list_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add canvas/scrollbar for list if many videos
        self._create_video_list_ui()

        self.open_selector_btn = ttk.Button(
            main_frame,
            text="Select Frames for this Video",
            command=self.open_frame_selector
        )
        self.open_selector_btn.grid(row=2, column=0, pady=15, ipady=5, sticky=tk.EW)

        self.validate_btn = ttk.Button(main_frame, text="Validate Selections and Exit", command=self.validate_and_close)
        self.validate_btn.grid(row=3, column=0, pady=(5, 0), ipady=5, sticky=tk.EW)

        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1) # List should expand
        
        self._update_status_display()

        # --- Dynamic Height Calculation ---
        self.parent.update_idletasks()
        fixed_width = 500
        # Calculate height based on list size, capped at 800
        required_height = min(self.parent.winfo_reqheight(), 800)
        self.parent.geometry(f"{fixed_width}x{required_height}")
        self.parent.minsize(fixed_width, 400)

    def _create_video_list_ui(self):
        """Creates the radio button list for video selection."""
        style = ttk.Style()
        style.configure("Status.TLabel", foreground="green", font=("Segoe UI", 10, "bold"))

        # Scrollable container
        canvas = tk.Canvas(self.video_list_frame)
        scrollbar = ttk.Scrollbar(self.video_list_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for i, filename in enumerate(self.video_paths_dict.keys()):
            item_frame = ttk.Frame(scrollable_frame)
            item_frame.grid(row=i, column=0, sticky=tk.W, pady=2)

            rb = ttk.Radiobutton(
                item_frame,
                text=filename,
                variable=self.selected_video,
                value=filename
            )
            rb.grid(row=0, column=0, sticky=tk.W)

            opened_label = ttk.Label(item_frame, text="", style="Status.TLabel", width=3)
            opened_label.grid(row=0, column=1, padx=(10, 0))

            selected_label = ttk.Label(item_frame, text="", style="Status.TLabel", width=3)
            selected_label.grid(row=0, column=2)
            
            self.status_labels[filename] = (opened_label, selected_label)

        if self.video_paths_dict:
            first_video = list(self.video_paths_dict.keys())[0]
            self.selected_video.set(first_video)

    def _update_status_display(self):
        """Updates the checkmarks for all videos based on the current state."""
        checkmark = "✓"
        for filename, (opened_label, selected_label) in self.status_labels.items():
            opened_label.config(text="[Viewed]" if self.selectors_opened.get(filename) else "")
            
            video_path = self.video_paths_dict[filename]
            count = len(self.all_selected_frames.get(video_path, []))
            selected_label.config(text=f"[{count} Frames]" if count > 0 else "")

    def open_frame_selector(self):
        """Opens the single video frame selector in a new window."""
        video_filename = self.selected_video.get()
        if not video_filename:
            messagebox.showwarning("No Video Selected", "Please select a video from the list.")
            return

        self.selectors_opened[video_filename] = True
        video_path = self.video_paths_dict[video_filename]
        
        try:
            # Initialize Manager
            video_manager = VideoMP4Manager(video_path)
        except FileNotFoundError as e:
            print(f"❌ {e}")
            messagebox.showerror("File Not Found", str(e))
            return
        except NameError:
            messagebox.showerror("Dependency Error", "VideoMP4Manager class is missing.")
            return

        initial_selection = self.all_selected_frames.get(video_path, [])
        
        selector_window = tk.Toplevel(self.parent)
        selector_window.grab_set() # Modal window

        # --- ARCHITECTURE FIX: Pass the manager instance, NOT manager.capture ---
        # The original code called video_manager.capture, which does not exist.
        selector = VideoFramesSelector(
            parent=selector_window,
            video_manager=video_manager, # Corrected Argument
            last_frame=video_manager.total_frames - 1,
            title=f"Selector: {video_filename}",
            initial_selection=initial_selection
        )

        self.parent.wait_window(selector_window)

        if selector.proceed_was_clicked:
            selected_frames = sorted(list(selector.selected_frames))
            if selected_frames:
                self.all_selected_frames[video_path] = selected_frames
            elif video_path in self.all_selected_frames:
                del self.all_selected_frames[video_path]
            print(f"Stored selection for '{video_filename}': {self.all_selected_frames.get(video_path, [])}")
        else:
            print(f"Selection cancelled for '{video_filename}'.")

        # Explicit release is not strictly necessary if Manager uses context managers, 
        # but good for cleanup if preloaded.
        # video_manager.release() # VideoMP4Manager doesn't have explicit release, it handles it via context or GC.
        self._update_status_display()

    def validate_and_close(self):
        """Closes the main window and flags that the process is complete."""
        self.validated = True
        self.parent.destroy()

# --- Example Usage ---
if __name__ == '__main__':
    # Create some dummy video files for the example to run if none exist
    temp_dir = Path("./temp_videos")
    temp_dir.mkdir(exist_ok=True)
    video_files = []
    
    # Generate a dummy video with OpenCV if it doesn't exist (so the selector actually shows something)
    dummy_vid_path = temp_dir / "test_video_1.mp4"
    if not dummy_vid_path.exists():
        height, width = 480, 640
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(dummy_vid_path), fourcc, 30.0, (width, height))
        for i in range(60): # 2 seconds
            # Create a frame with changing color
            frame = cv2.rectangle(
                img=tk.Frame().winfo_rgb("black"), # Placeholder, actually creating numpy array below
                pt1=(0,0), pt2=(width, height), color=(0,0,0), thickness=-1
            ) 
            import numpy as np
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:] = (i * 4, 255 - i*4, 100) # Changing color
            cv2.putText(frame, f"Frame {i}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            out.write(frame)
        out.release()
    
    video_files.append(str(dummy_vid_path))

    root = tk.Tk()
    app = MultiVideoFramesSelector(root, video_paths=video_files)
    root.mainloop()

    # After the window is closed, you can access the results
    if app.validated:
        print("\n✅ Selections Validated!")
        print(app.all_selected_frames)
    else:
        print("\n❌ Window was closed without validation.")

    # Cleanup (Optional)
    # import shutil
    # shutil.rmtree(temp_dir)