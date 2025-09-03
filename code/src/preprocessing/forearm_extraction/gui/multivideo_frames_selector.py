import tkinter as tk
from tkinter import ttk, messagebox
import os
from pathlib import Path

# Assuming these imports are in your project structure
from preprocessing.common import (
    VideoFramesSelector,
    VideoMP4Manager
)


# --- Main Class: Selector for MULTIPLE Videos ---
class MultiVideoFramesSelector:
    """
    A GUI to manage frame selection across multiple video files.
    It provides a bullet-point list to choose a video and then launches a
    dedicated selector window for it, tracking the status of each video.
    """
    def __init__(self, parent, video_paths, initial_selections=None):
        """
        Initializes the multi-video selector.

        Args:
            parent: The parent tkinter widget.
            video_paths (list): A list of full string paths to the video files.
            initial_selections (dict, optional): A dict where keys are video
                basenames (e.g., "video1.mp4") and values are lists of
                pre-selected frame IDs (e.g., [10, 25]). Defaults to None.
        """
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
                    print(f"⚠️ Warning: Basename '{basename}' from initial_selections not found in the provided video paths.")

        self.selectors_opened = {filename: False for filename in self.video_paths_dict.keys()}
        self.validated = False

        # --- UI State ---
        self.selected_video = tk.StringVar()
        self.status_labels = {}

        # --- Window Configuration ---
        self.parent.title("Multi-Video Selector")
        # --- REMOVED --- Hardcoded geometry is not flexible.
        # self.parent.geometry("450x300")

        # --- UI Elements ---
        main_frame = ttk.Frame(self.parent, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.parent.columnconfigure(0, weight=1)
        self.parent.rowconfigure(0, weight=1)

        ttk.Label(main_frame, text="Select a video to process:").grid(row=0, column=0, pady=(0, 10), sticky=tk.W)

        self.video_list_frame = ttk.LabelFrame(main_frame, text="Videos | Status (Opened, Selected)", padding=10)
        self.video_list_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
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
        
        self._update_status_display()

        # --- MODIFIED: Dynamic Height Calculation ---
        # This block replaces the hardcoded geometry.
        # It's placed at the end of __init__ after all widgets are created.
        
        # 1. Force tkinter to process all pending tasks, including widget sizing.
        self.parent.update_idletasks()

        # 2. Get the minimum required height to fit all the content.
        # We use a fixed width for aesthetics but a dynamic height.
        fixed_width = 450
        required_height = self.parent.winfo_reqheight()

        # 3. Set the initial size and the minimum resizable size.
        self.parent.geometry(f"{fixed_width}x{required_height}")
        self.parent.minsize(fixed_width, required_height)
        # --- END MODIFICATION ---

    def _create_video_list_ui(self):
        """Creates the radio button list for video selection with status indicators."""
        style = ttk.Style()
        style.configure("Status.TLabel", foreground="green", font=("Helvetica", 12, "bold"))

        for i, filename in enumerate(self.video_paths_dict.keys()):
            item_frame = ttk.Frame(self.video_list_frame)
            item_frame.grid(row=i, column=0, sticky=tk.W, pady=2)

            rb = ttk.Radiobutton(
                item_frame,
                text=filename,
                variable=self.selected_video,
                value=filename
            )
            rb.grid(row=0, column=0, sticky=tk.W)

            opened_label = ttk.Label(item_frame, text="", style="Status.TLabel", width=3, anchor=tk.W)
            opened_label.grid(row=0, column=1, padx=(10, 0))

            selected_label = ttk.Label(item_frame, text="", style="Status.TLabel", width=3, anchor=tk.W)
            selected_label.grid(row=0, column=2)
            
            self.status_labels[filename] = (opened_label, selected_label)

        if self.video_paths_dict:
            first_video = list(self.video_paths_dict.keys())[0]
            self.selected_video.set(first_video)

    def _update_status_display(self):
        """Updates the checkmarks for all videos based on the current state."""
        checkmark = "✓"
        for filename, (opened_label, selected_label) in self.status_labels.items():
            opened_label.config(text=checkmark if self.selectors_opened.get(filename) else "")
            
            video_path = self.video_paths_dict[filename]
            selected_label.config(text=checkmark if self.all_selected_frames.get(video_path) else "")

    def open_frame_selector(self):
        """Opens the single video frame selector in a new window."""
        video_filename = self.selected_video.get()
        if not video_filename:
            messagebox.showwarning("No Video Selected", "Please select a video from the list.")
            return

        self.selectors_opened[video_filename] = True
        video_path = self.video_paths_dict[video_filename]
        
        try:
            video_manager = VideoMP4Manager(video_path)
        except FileNotFoundError as e:
            print(f"❌ {e}")
            messagebox.showerror("File Not Found", str(e))
            return

        initial_selection = self.all_selected_frames.get(video_path, [])
        
        selector_window = tk.Toplevel(self.parent)
        selector_window.grab_set()

        selector = VideoFramesSelector(
            selector_window,
            video_manager.capture,
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

        video_manager.release()
        self._update_status_display()

    def validate_and_close(self):
        """Closes the main window and flags that the process is complete."""
        self.validated = True
        self.parent.destroy()

# --- Example Usage ---
if __name__ == '__main__':
    # Create some dummy video files for the example to run
    temp_dir = Path("./temp_videos")
    temp_dir.mkdir(exist_ok=True)
    video_files = []
    for i in range(8): # Change this number to see the window resize!
        f_path = temp_dir / f"video_{i+1}.mp4"
        f_path.touch() # Create an empty file
        video_files.append(str(f_path))

    root = tk.Tk()
    app = MultiVideoFramesSelector(root, video_paths=video_files)
    root.mainloop()

    # After the window is closed, you can access the results
    if app.validated:
        print("\n✅ Selections Validated!")
        print(app.all_selected_frames)
    else:
        print("\n❌ Window was closed without validation.")

    # Clean up dummy files
    import shutil
    shutil.rmtree(temp_dir)