import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2

# Import TYPE_CHECKING to create a conditional block
from typing import TYPE_CHECKING, List, Tuple
# This block is only 'True' for type checkers, not at runtime
if TYPE_CHECKING:
    from ..core.review_tracking_orchestrator import TrackerReviewOrchestrator


class TrackerReviewGUI:
    """
    Handles the Tkinter GUI for video playback (the 'View').
    
    Its sole responsibilities are to build the user interface and display data
    provided by the Controller. All user actions (e.g., button clicks, 
    slider drags) are forwarded directly to the Controller to handle the logic.
    """
    def __init__(self, 
                 *, 
                 title: str = "Loaded Frames Review", 
                 landmarks = None, 
                 landmark_properties = None,
                 controller: 'TrackerReviewOrchestrator' = None,
                 windowState: str = 'normal'):
        """
        Initializes the View.
        
        Args:
            controller (ReviewController): The controller that manages application logic.
            title (str): The title for the main window.
            landmarks (list[int], optional): Frame IDs to mark on the timeline.
            landmark_properties (dict, optional): Properties for the landmark marks.
        """
        self.controller = controller
        self.title = title
        self.windowState = windowState
        self.landmarks = landmarks if landmarks is not None else []
        
        # Set default landmark properties and override with user-provided ones
        default_props = {'color': 'blue', 'thickness': 2, 'height': 10}
        if landmark_properties:
            default_props.update(landmark_properties)
        self.landmark_properties = default_props

        # --- Tkinter UI elements ---
        self.root = None
        self.video_frame = None 
        self.image_label = None
        self.play_pause_button = None
        self.timeline_canvas = None
        self.timeline_scale = None
        self.frame_entry = None
        self.current_frame_label = None
        self.valid_button = None
        self.proceed_button = None
        self.rerun_button = None # --- NEW: Added reroll button attribute
        self.marked_listbox = None
        self.speed_slider = None
        self.speed_label = None

        # --- Tkinter variables ---
        self.scale_var = None
        self.entry_var = None
        self.speed_var = None
        # --- Dictionary to hold BooleanVar for each object checkbox ---
        self.object_vars = {}

    def setup_ui(self):
        """Creates and arranges all the Tkinter widgets."""
        self.root = tk.Tk()
        self.root.title(f"Video Replay: {self.title}")
        
        # --- Logic to Position the Window ---
        if self.windowState.upper() == 'NORMAL':
            window_width = 1024
            window_height = 768

            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()

            center_x = int(screen_width/2 - window_width / 2)
            center_y = int(screen_height/2 - window_height / 2)

            self.root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

        elif self.windowState.upper() == 'MAXIMIZED' or self.windowState.upper() == 'MAXIMISED':
            self.root.state('zoomed')


        self.root.protocol("WM_DELETE_WINDOW", self.quit)

        # --- Style Configuration ---
        style = ttk.Style(self.root)
        style.configure("Mark.TButton", foreground="orange")
        style.configure("Delete.TButton", foreground="red")
        style.configure("Finish.TButton", foreground="blue")
        style.configure("Valid.TButton", foreground="green")
        # --- NEW: Added style for the new button for consistency ---
        style.configure("Rerun.TButton", foreground="#8e44ad")


        # --- UI Variables ---
        self.scale_var = tk.IntVar(value=0)
        self.entry_var = tk.StringVar(value="0")
        self.speed_var = tk.DoubleVar(value=1.0)

        # --- Main Layout Panes ---
        main_paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        player_pane = ttk.Frame(main_paned_window)
        main_paned_window.add(player_pane, weight=3)

        list_pane = ttk.Frame(main_paned_window)
        main_paned_window.add(list_pane, weight=1)

        player_pane.rowconfigure(0, weight=1)
        player_pane.columnconfigure(0, weight=1)

        # --- Video Display Area ---
        self.video_frame = ttk.Frame(player_pane)
        self.video_frame.grid(row=0, column=0, sticky="nsew", pady=5)
        self.video_frame.rowconfigure(0, weight=1)
        self.video_frame.columnconfigure(0, weight=1)
        
        self.image_label = ttk.Label(self.video_frame)
        self.image_label.grid(row=0, column=0, sticky="nsew")
        
        # --- Timeline with Landmarks ---
        canvas_height = self.landmark_properties['height'] + 15
        self.timeline_canvas = tk.Canvas(player_pane, height=canvas_height, bd=0, highlightthickness=0)
        self.timeline_canvas.grid(row=1, column=0, sticky="ew", pady=5)

        self.timeline_scale = ttk.Scale(
            self.timeline_canvas, from_=0, to=self.controller.model.total_frames - 1,
            orient=tk.HORIZONTAL, variable=self.scale_var, 
            command=lambda val: self.controller.seek_to_frame(int(float(val)))
        )
        
        slider_y_pos = self.landmark_properties['height'] + 2
        self.timeline_canvas.create_window(0, slider_y_pos, window=self.timeline_scale, anchor='nw', tags="scale_widget")
        self.timeline_canvas.bind("<Configure>", self._on_canvas_resize)

        # --- Frame Info and Entry Box ---
        info_frame = ttk.Frame(player_pane)
        info_frame.grid(row=2, column=0, sticky="ew")
        info_frame.columnconfigure(1, weight=1)

        self.current_frame_label = ttk.Label(info_frame, text=f"Frame: 0 / {self.controller.model.total_frames - 1}")
        self.current_frame_label.grid(row=0, column=0, sticky="w")

        self.frame_entry = ttk.Entry(info_frame, textvariable=self.entry_var, width=10)
        self.frame_entry.grid(row=0, column=2, sticky="e", padx=5)
        self.frame_entry.bind("<Return>", lambda e: self.controller.seek_to_frame(self.entry_var.get()))
        self.frame_entry.bind("<FocusOut>", lambda e: self.controller.seek_to_frame(self.entry_var.get()))

        # --- Controls Container ---
        controls_container = ttk.Frame(player_pane)
        controls_container.grid(row=3, column=0, pady=10, sticky="ew")

        action_controls_frame = ttk.Frame(controls_container)
        action_controls_frame.pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.play_pause_button = ttk.Button(action_controls_frame, text="Play", command=self.controller.toggle_play_pause)
        self.play_pause_button.pack(side=tk.LEFT, padx=5)

        # --- Speed Control ---
        speed_frame = ttk.Frame(action_controls_frame)
        speed_frame.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)

        self.speed_label = ttk.Label(speed_frame, text="Speed: 1.0x")
        self.speed_label.pack(side=tk.LEFT)

        self.speed_slider = ttk.Scale(
            speed_frame, from_=0.5, to=8.0,
            orient=tk.HORIZONTAL, variable=self.speed_var, 
            command=lambda val: self.controller.change_speed(val)
        )
        self.speed_slider.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        # --- Object Checkboxes Frame ---
        checkbox_frame = ttk.LabelFrame(action_controls_frame, text="Objects to Mark")
        checkbox_frame.pack(side=tk.LEFT, padx=10, fill=tk.Y)
        
        if hasattr(self.controller, 'object_names') and self.controller.object_names:
            for name in self.controller.object_names:
                var = tk.BooleanVar(value=True)
                self.object_vars[name] = var
                
                cb = ttk.Checkbutton(checkbox_frame, text=name, variable=var, 
                                     command=self._update_valid_button_state)
                cb.pack(side=tk.TOP, anchor="w", padx=3, pady=2)
        
        ttk.Button(action_controls_frame, text="✏️ Mark for Labeling", command=self._on_mark_button_click, style="Mark.TButton").pack(side=tk.LEFT, padx=5)

        # --- Finish Controls ---
        finish_controls_frame = ttk.LabelFrame(controls_container, text="Finish")
        finish_controls_frame.pack(side=tk.RIGHT)

        self.proceed_button = ttk.Button(finish_controls_frame, text="➡️ Proceed with Marked", command=self.controller.finish_and_proceed, style="Finish.TButton")
        self.proceed_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.valid_button = ttk.Button(finish_controls_frame, text="✅ Finish as Valid", command=self.controller.finish_as_valid, style="Valid.TButton")
        self.valid_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # --- NEW: Create and place the new 'Rerun' button ---
        self.rerun_button = ttk.Button(
            finish_controls_frame, 
            text="🔄 Rerun Auto-Processing", 
            command=self._on_rerun_button_click, 
            style="Rerun.TButton"
        )
        self.rerun_button.pack(side=tk.LEFT, padx=5, pady=5)

        # --- Marked Frames List Pane ---
        list_pane.rowconfigure(1, weight=1)
        list_pane.columnconfigure(0, weight=1)
        
        ttk.Label(list_pane, text="Marked Frames", font=("", 10, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0,5))
        
        self.marked_listbox = tk.Listbox(list_pane, height=10)
        self.marked_listbox.grid(row=1, column=0, columnspan=2, sticky="nsew")
        self.marked_listbox.bind('<<ListboxSelect>>', self._on_listbox_select)
        
        list_scrollbar = ttk.Scrollbar(list_pane, orient="vertical", command=self.marked_listbox.yview)
        list_scrollbar.grid(row=1, column=2, sticky="ns")
        self.marked_listbox['yscrollcommand'] = list_scrollbar.set
        
        delete_button = ttk.Button(list_pane, text="Delete Selected", command=self._delete_from_listbox, style="Delete.TButton")
        delete_button.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(5,0))

        # --- Bind Keys to Controller ---
        self._bind_keys()
        
        # Draw initial state of landmarks
        self.root.after(50, self._draw_landmarks)


    def _bind_keys(self):
        """Binds keyboard shortcuts to controller methods."""
        self.root.bind('<Left>', lambda e: self.controller.seek_to_frame(self.scale_var.get() - 1))
        self.root.bind('<Right>', lambda e: self.controller.seek_to_frame(self.scale_var.get() + 1))
        self.root.bind('<Control-Left>', lambda e: self.controller.seek_to_frame(self.scale_var.get() - 10))
        self.root.bind('<Control-Right>', lambda e: self.controller.seek_to_frame(self.scale_var.get() + 10))
        self.root.bind('<space>', lambda e: self.controller.toggle_play_pause())
        self.root.bind('<Delete>', lambda e: self._delete_from_listbox())
        self.root.bind('m', lambda e: self._on_mark_button_click())

    def _update_valid_button_state(self):
        """
        Updates the state of the 'Finish as Valid' button.
        
        The button is enabled (NORMAL) only if the marked list is empty AND all 
        object checkboxes are checked. Otherwise, it is disabled.
        """
        if not hasattr(self, 'valid_button') or not self.valid_button:
            return

        is_list_empty = self.marked_listbox.size() == 0

        all_boxes_checked = True
        if hasattr(self, 'object_vars') and self.object_vars:
            all_boxes_checked = all(var.get() for var in self.object_vars.values())

        if is_list_empty and all_boxes_checked:
            self.valid_button.config(state=tk.NORMAL)
        else:
            self.valid_button.config(state=tk.DISABLED)
    
    # --- PRIVATE EVENT HANDLERS (Forwarding to Controller) ---
    
    # --- NEW: Handler for the 'Rerun' button click ---
    def _on_rerun_button_click(self):
        """
        Handles the click event for the 'Rerun Auto-Processing' button.
        This simply calls the controller's standard 'proceed' method, which
        is expected to handle the application logic and close the GUI.
        """
        self.controller.finish_and_proceed()
        
    def _on_mark_button_click(self):
        """
        Gathers selected objects from checkboxes and tells the controller to mark
        the current frame. Resets checkboxes to their default (checked) state.
        """
        selected_objects = [name for name, var in self.object_vars.items() if var.get()]
        self.controller.mark_current_frame(objects_to_mark=selected_objects)

        for var in self.object_vars.values():
            var.set(True)

    def _on_listbox_select(self, event):
        """When a listbox item is selected, tell the controller to seek."""
        selection_indices = self.marked_listbox.curselection()
        if not selection_indices: return
        
        selected_index = selection_indices[0]
        
        try:
            listbox_text = self.marked_listbox.get(selected_index)
            frame_part = listbox_text.split(':')[0]
            frame_num = int(frame_part.split()[-1])
            self.controller.seek_to_frame(frame_num)
        except (ValueError, IndexError):
            print(f"Error parsing listbox entry: {self.marked_listbox.get(selected_index)}")
            pass 

    def _delete_from_listbox(self):
        """Tells the controller to delete the selected item from the marked list."""
        selection_indices = self.marked_listbox.curselection()
        if not selection_indices: return
        self.controller.delete_marked_frame(selection_indices[0])

    def _on_canvas_resize(self, event):
        """Handles resizing of the timeline canvas, updating scale and landmarks."""
        canvas_width = event.width
        self.timeline_canvas.itemconfigure("scale_widget", width=canvas_width)
        self._draw_landmarks()

    def _draw_landmarks(self):
        """Draws the landmark rectangles on the timeline canvas."""
        self.timeline_canvas.delete("landmark")
        if not self.landmarks or self.controller.model.total_frames <= 1:
            return

        canvas_width = self.timeline_canvas.winfo_width()
        if canvas_width <= 1: return

        color = self.landmark_properties['color']
        thickness = self.landmark_properties['thickness']
        height = self.landmark_properties['height']

        for frame_id in self.landmarks:
            x_ratio = frame_id / (self.controller.model.total_frames - 1)
            x_pos = x_ratio * canvas_width
            self.timeline_canvas.create_rectangle(
                x_pos, 0, x_pos + thickness, height,
                fill=color, outline=color, tags="landmark"
            )

    # --- UI UPDATE METHODS (Called by the Controller) ---

    def update_video_display(self, frame):
        """Displays a single frame image on the GUI."""
        container_w = self.video_frame.winfo_width()
        container_h = self.video_frame.winfo_height()
        if container_w < 50 or container_h < 50: return
        
        original_h, original_w, _ = frame.shape
        if original_w == 0 or original_h == 0: return

        scale = min(container_w / original_w, container_h / original_h)
        new_w, new_h = int(original_w * scale), int(original_h * scale)
        if new_w > 0 and new_h > 0:
            resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            cv2image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.image_label.imgtk = imgtk
            self.image_label.config(image=imgtk)

    def update_timeline(self, frame_num):
        """Updates the slider and entry box with the current frame number."""
        self.scale_var.set(frame_num)
        self.entry_var.set(str(frame_num))

    def update_frame_label(self, current_num, total_frames):
        """Updates the 'Frame X / Y' text label."""
        self.current_frame_label.config(text=f"Frame: {current_num} / {total_frames - 1}")

    def update_play_pause_button(self, is_paused):
        """Updates the text of the play/pause button."""
        self.play_pause_button.config(text="Play" if is_paused else "Pause")

    def update_speed_label(self, speed):
        """Updates the speed label text."""
        self.speed_label.config(text=f"Speed: {speed:.1f}x")

    def update_marked_list(self, marked_list: dict[int, List[str]]):
        """
        Refreshes the listbox with the current list of marked frames and their associated objects.
        """
        # Deletes all items from the listbox to prepare for the update.
        self.marked_listbox.delete(0, tk.END)

        # Iterates over each key-value pair (frame number and its list of objects) in the dictionary.
        for frame_num, objects in marked_list.items():
            object_str = ", ".join(objects)
            self.marked_listbox.insert(tk.END, f"Frame {frame_num}: {object_str}")
            
    # --- MODIFIED: This method now also controls the state of the new rerun button ---
    def update_finish_buttons(self, marked_list):
        """Enables/disables finish buttons based on the state of the marked list."""
        is_list_empty = not marked_list
        
        # The 'Proceed' button is enabled only if the list is NOT empty.
        self.proceed_button.config(state=tk.DISABLED if is_list_empty else tk.NORMAL)
        
        # --- NEW: The 'Rerun' button is enabled only if the list IS empty. ---
        if self.rerun_button: # Check if button exists before configuring
             self.rerun_button.config(state=tk.NORMAL if is_list_empty else tk.DISABLED)
    
        # For the 'Valid' button, delegate to its dedicated state update method,
        # which checks both the list and the checkboxes.
        self._update_valid_button_state()

    # --- MAINLOOP CONTROL ---

    def start_mainloop(self):
        """Starts the Tkinter event loop."""
        self.root.after(100, self.update_finish_buttons, [])
        self.root.mainloop()

    def quit(self):
        """Destroys the Tkinter window."""
        if self.root:
            if self.controller and hasattr(self.controller, '_update_job') and self.controller._update_job:
                self.root.after_cancel(self.controller._update_job)
            self.root.destroy()