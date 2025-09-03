import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import numpy as np

class VideoFramesSelector:
    """
    Manages the frame selection GUI, built with Tkinter.
    This class populates a parent Tkinter window, allowing a user to scrub 
    through frames, select multiple frames, and see the list of selected frames.
    The main application logic should call mainloop() on the parent window.
    """
    # --- MODIFIED: Added 'initial_selection' parameter ---
    def __init__(self, parent, video_source, initial_selection=None, first_frame=0, last_frame=-1, current_frame_num=0, title=None):
        """
        Initializes the VideoFramesSelector GUI components within the parent window.

        Args:
            parent: The parent tkinter window (e.g., a tk.Tk() instance).
            video_source: Either an OpenCV video capture object or a list/NumPy
                          array of RGB frames.
            initial_selection (list, optional): A list of frame numbers to be
                                                pre-selected. Defaults to None.
            title (str): The base title for the window.
            first_frame (int): The first frame number for the slider.
            last_frame (int): The last frame number for the slider. If -1, it's
                              auto-detected from the source.
            current_frame_num (int): The initial frame to display.
        """
        self.parent = parent
        self.base_title = title
        self.selected_frame_num = current_frame_num
        self.proceed_was_clicked = False
        self.first_frame = first_frame
        
        # --- MODIFIED: Initialize selected_frames from the 'initial_selection' argument ---
        if initial_selection is not None:
            try:
                # A set is used to store multiple selected frames, initialized from the provided list
                self.selected_frames = {int(f) for f in initial_selection}
            except (ValueError, TypeError):
                print("Warning: 'initial_selection' contained invalid data. Starting with an empty selection.")
                self.selected_frames = set()
        else:
            self.selected_frames = set()

        # --- Determine source type and configure accordingly ---
        self.is_video_mode = False
        if hasattr(video_source, 'get') and callable(getattr(video_source, 'get')): # Duck-typing for cv2.VideoCapture
            self.is_video_mode = True
            self.video = video_source
            if last_frame == -1:
                self.last_frame = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
            else:
                self.last_frame = last_frame
        elif isinstance(video_source, (list, np.ndarray)):
            self.is_video_mode = False
            self.frames = np.asarray(video_source)
            if last_frame == -1:
                self.last_frame = len(self.frames) - 1 if len(self.frames) > 0 else 0
            else:
                self.last_frame = last_frame
        else:
            raise TypeError("video_source must be an OpenCV VideoCapture object or a list/array of RGB frames.")

        self.parent.title(f"Frame Selector")

        # --- Get screen dimensions for appropriate window sizing ---
        screen_w = self.parent.winfo_screenwidth()
        screen_h = self.parent.winfo_screenheight()
        self.target_w = screen_w // 2
        self.target_h = screen_h // 2

        # --- Create variables ---
        self.scale_var = tk.IntVar(value=current_frame_num)
        self.entry_var = tk.StringVar(value=str(current_frame_num))
        
        # --- Main layout frame to hold video and list side-by-side ---
        main_layout_frame = ttk.Frame(self.parent, padding="10")
        main_layout_frame.pack(fill="both", expand=True)

        # Left panel for video display
        left_panel = ttk.Frame(main_layout_frame)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.image_label = ttk.Label(left_panel)
        self.image_label.pack(pady=10, fill="both", expand=True)

        # Right panel for the list of selected frames
        right_panel = ttk.Frame(main_layout_frame, width=200)
        right_panel.pack(side="right", fill="y")
        right_panel.pack_propagate(False) # Prevent panel from shrinking

        ttk.Label(right_panel, text="Selected Frames:").pack(anchor="w", pady=(0, 5))
        listbox_frame = ttk.Frame(right_panel)
        listbox_frame.pack(fill="both", expand=True)

        self.selected_listbox = tk.Listbox(listbox_frame)
        self.selected_listbox.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(listbox_frame, orient="vertical", command=self.selected_listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.selected_listbox.config(yscrollcommand=scrollbar.set)
        self.selected_listbox.bind("<Double-1>", self._remove_from_list)

        self.delete_button = ttk.Button(right_panel, text="Delete Selected", command=self._delete_selected_from_list)
        self.delete_button.pack(fill="x", pady=(5, 0))

        # --- Control frame for slider ---
        self.control_frame = ttk.Frame(self.parent, padding="10")
        self.control_frame.pack(fill="x")
        
        self.min_label = ttk.Label(self.control_frame, text=str(self.first_frame))
        self.min_label.pack(side="left", padx=(5, 10))

        self.scale = ttk.Scale(
            self.control_frame, from_=self.first_frame, to=self.last_frame,
            orient="horizontal", variable=self.scale_var, command=self._update_frame
        )
        self.scale.pack(fill="x", expand=True, side="left")

        self.max_label = ttk.Label(self.control_frame, text=str(self.last_frame))
        self.max_label.pack(side="left", padx=(10, 5))

        # --- Reworked info/button frame ---
        self.info_frame = ttk.Frame(self.parent, padding="5")
        self.info_frame.pack(fill="x")
        
        self.info_frame.columnconfigure((0, 1, 2, 3, 4), weight=1)

        self.current_label = ttk.Label(self.info_frame, text=f"Current: {self.selected_frame_num}")
        self.current_label.grid(row=0, column=0, sticky="e", padx=5)

        self.frame_entry = ttk.Entry(self.info_frame, textvariable=self.entry_var, width=10)
        self.frame_entry.grid(row=0, column=1, sticky="w")
        self.frame_entry.bind("<Return>", self._update_from_entry)
        self.frame_entry.bind("<FocusOut>", self._update_from_entry)

        self.toggle_button = ttk.Button(self.info_frame, text="Select Frame", command=self._toggle_selection)
        self.toggle_button.grid(row=0, column=2, sticky="ew", padx=5)

        self.confirm_button = ttk.Button(self.info_frame, text="Confirm Selection", command=self._on_confirm)
        self.confirm_button.grid(row=0, column=3, sticky="ew", padx=5)

        self.proceed_empty_button = ttk.Button(self.info_frame, text="Proceed Without Any", command=self._on_proceed_empty)
        self.proceed_empty_button.grid(row=0, column=4, sticky="ew", padx=10)

        # --- Bindings ---
        self.parent.bind('<Left>', self._handle_arrow_keys)
        self.parent.bind('<Right>', self._handle_arrow_keys)
        self.parent.bind('<Control-Left>', self._handle_arrow_keys)
        self.parent.bind('<Control-Right>', self._handle_arrow_keys)
        self.parent.protocol("WM_DELETE_WINDOW", self._on_cancel)
        
        # --- Initialize the first frame view and populate the listbox ---
        self._update_frame()

    def _update_frame(self, _=None):
        self.selected_frame_num = self.scale_var.get()
        self.entry_var.set(str(self.selected_frame_num))
        self.current_label.config(text=f"Current: {self.selected_frame_num}")

        frame = None
        success = False

        if self.is_video_mode:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, self.selected_frame_num)
            success, bgr_frame = self.video.read()
            if success:
                frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        else:
            if 0 <= self.selected_frame_num < len(self.frames):
                frame = self.frames[self.selected_frame_num]
                success = True

        if success and frame is not None:
            original_h, original_w = frame.shape[:2]
            scale = min(self.target_w / original_w, self.target_h / original_h)
            new_w = int(original_w * scale)
            new_h = int(original_h * scale)
            resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            img = Image.fromarray(resized_frame)
            self.photo = ImageTk.PhotoImage(image=img)
            self.image_label.config(image=self.photo)
        
        if self.base_title:
            self.parent.title(f"{self.base_title} - Frame: {self.selected_frame_num}")
        else:
            self.parent.title(f"Frame Selector - Frame: {self.selected_frame_num}")
        
        self._update_selection_ui() # Update button and list

    def _update_from_entry(self, _=None):
        try:
            frame_num = int(self.entry_var.get())
            frame_num = max(self.first_frame, min(frame_num, self.last_frame))
            self.scale_var.set(frame_num)
            self._update_frame()
        except ValueError:
            self.entry_var.set(str(self.scale_var.get()))

    def _handle_arrow_keys(self, event):
        current_frame = self.scale_var.get()
        is_ctrl_pressed = (event.state & 4) != 0
        step = 10 if is_ctrl_pressed else 1
        
        new_frame = current_frame
        if event.keysym == 'Left':
            new_frame -= step
        elif event.keysym == 'Right':
            new_frame += step
            
        clamped_frame = max(self.first_frame, min(new_frame, self.last_frame))
        self.scale_var.set(clamped_frame)
        self._update_frame()
    
    def _toggle_selection(self):
        """Adds or removes the current frame from the selection set."""
        frame_num = self.selected_frame_num
        if frame_num in self.selected_frames:
            self.selected_frames.remove(frame_num)
        else:
            self.selected_frames.add(frame_num)
        self._update_selection_ui()

    def _update_selection_ui(self):
        """Updates the listbox and the toggle button's text."""
        # Update button text
        if self.selected_frame_num in self.selected_frames:
            self.toggle_button.config(text="Deselect Frame")
        else:
            self.toggle_button.config(text="Select Frame")

        # Update listbox content
        self.selected_listbox.delete(0, tk.END)
        for frame_num in sorted(list(self.selected_frames)):
            self.selected_listbox.insert(tk.END, f"Frame {frame_num}")
    
    def _remove_from_list(self, event):
        """Removes a frame by double-clicking it in the listbox."""
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

    def _delete_selected_from_list(self):
        """Removes the highlighted frame from the listbox via the delete button."""
        selected_indices = self.selected_listbox.curselection()
        if not selected_indices:
            return # Do nothing if nothing is selected
        
        selected_item = self.selected_listbox.get(selected_indices[0])
        try:
            frame_num = int(selected_item.split(" ")[1])
            if frame_num in self.selected_frames:
                self.selected_frames.remove(frame_num)
                self._update_selection_ui()
        except (IndexError, ValueError):
            pass

    def _on_confirm(self):
        """Called when the 'Confirm Selection' button is clicked."""
        self.proceed_was_clicked = True
        self.parent.destroy()

    def _on_proceed_empty(self):
        """Called to proceed with an empty selection."""
        self.selected_frames.clear()
        self.proceed_was_clicked = True
        self.parent.destroy()

    def _on_cancel(self):
        """Called when the window is closed via the 'X' button."""
        self.proceed_was_clicked = False
        self.parent.destroy()


def create_dummy_video(filename="dummy_video.mp4", num_frames=200):
    """Generates a simple dummy video file for testing."""
    width, height = 640, 480
    fps = 30
    # Use 'mp4v' codec for good compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    if not writer.isOpened():
        raise IOError(f"Failed to create video writer for {filename}")

    print(f"Creating dummy video at '{filename}'...")
    for i in range(num_frames):
        # Create a frame with a color gradient and frame number
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = int(255 * (1 - i / (num_frames - 1))) # Blue channel
        frame[:, :, 2] = int(255 * (i / (num_frames - 1)))    # Red channel
        text = f"Frame: {i}"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        writer.write(frame)
        
    writer.release()
    print("Dummy video created successfully.")
    return filename

if __name__ == '__main__':
    # --- DEMO 1: Using a dynamically created and deleted video file ---
    video_path = 'dummy_test_video.mp4'
    cap = None  # Initialize cap to None to ensure it's defined for the finally block
    
    try:
        # 1. Create the dummy video for the test
        create_dummy_video(video_path, num_frames=200)

        # 2. Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file at {video_path}")
        
        print("\n--- Opening selector with generated video file ---")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 3. Create and run the Tkinter GUI
        root1 = tk.Tk()
        selector1 = VideoFramesSelector(
            root1,
            cap,
            last_frame=total_frames - 1, 
            current_frame_num=0, 
            title="Video File Example (Auto-Deleted)"
        )
        root1.mainloop()
        
        # 4. Access the results after the window is closed
        if selector1.proceed_was_clicked:
            selected_frames = sorted(list(selector1.selected_frames))
            print(f"Proceeding with frames from video selection: {selected_frames}")
        else:
            print("Video selection was cancelled by closing the window.")

    except (IOError, cv2.error) as e:
        print(f"Error during video demo: {e}")
    finally:
        import os
        # 5. This block ALWAYS runs, ensuring cleanup
        print("\n--- Cleaning up resources for Demo 1 ---")
        if cap is not None and cap.isOpened():
            cap.release() # Safely release the video capture object
            print("Video capture released.")
        
        if os.path.exists(video_path):
            os.remove(video_path) # Delete the dummy video file
            print(f"Dummy video '{video_path}' has been deleted. ✨")


    # --- DEMO 2: Using an array of frames (No changes needed here) ---
    pre_selected_frames = [10, 25, 88, 150, 199]
    print(f"\n--- Opening selector with a numpy array and initial selection: {pre_selected_frames} ---")
    
    # Create some dummy frames for the example
    dummy_frames = []
    for i in range(200):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 0] = int(255 * (1 - i / 199.0))
        frame[:, :, 2] = int(255 * (i / 199.0))
        dummy_frames.append(frame)

    root2 = tk.Tk()
    selector2 = VideoFramesSelector(
        root2,
        dummy_frames,
        initial_selection=pre_selected_frames,
        current_frame_num=100,
        title="Array Example"
    )
    root2.mainloop()
    
    if selector2.proceed_was_clicked:
        selected_frames_2 = sorted(list(selector2.selected_frames))
        print(f"Frames selected from the array: {selected_frames_2}")
    else:
        print("Array selection was cancelled by closing the window.")