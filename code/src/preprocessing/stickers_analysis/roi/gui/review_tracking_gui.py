import tkinter as tk
from tkinter import ttk, Toplevel
from PIL import Image, ImageTk
import cv2
from enum import Enum

# Import TYPE_CHECKING to create a conditional block
from typing import TYPE_CHECKING, List, Dict

# This block is only 'True' for type checkers, not at runtime
if TYPE_CHECKING:
    from ..core.review_tracking_orchestrator import TrackerReviewOrchestrator

from ..models.roi_review_shared_types import FrameAction, FrameMark


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
        self.rerun_button = None 
        self.label_listbox = None
        self.deleting_listbox = None
        self.speed_slider = None
        self.speed_label = None
        # --- MODIFICATION: Added attributes for the mark buttons ---
        self.mark_label_button = None
        self.mark_delete_button = None

        # --- Tkinter variables ---
        self.scale_var = None
        self.entry_var = None
        self.speed_var = None
        # --- Dictionary to hold BooleanVar for each object checkbox ---
        self.object_vars = {}
        # --- NEW: Attribute to track the active listbox ---
        self.active_listbox = None

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
        
        # --- Create a frame to hold the marking buttons vertically ---
        mark_buttons_frame = ttk.Frame(action_controls_frame)
        mark_buttons_frame.pack(side=tk.LEFT, padx=5)

        # --- MODIFICATION: Store buttons as instance attributes ---
        self.mark_label_button = ttk.Button(mark_buttons_frame, text="✏️ Mark for Labeling", command=self._on_mark_labeling_button_click, style="Mark.TButton")
        self.mark_label_button.pack(side=tk.TOP, fill=tk.X)
        
        self.mark_delete_button = ttk.Button(mark_buttons_frame, text="🗑️ Mark for Deletion", command=self._on_mark_deleting_button_click, style="Delete.TButton")
        self.mark_delete_button.pack(side=tk.TOP, fill=tk.X, pady=(5,0))

        # --- Finish Controls ---
        finish_controls_frame = ttk.LabelFrame(controls_container, text="Finish")
        finish_controls_frame.pack(side=tk.RIGHT)

        self.proceed_button = ttk.Button(finish_controls_frame, text="➡️ Proceed with Marked", command=self.controller.finish_and_proceed, style="Finish.TButton")
        self.proceed_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.valid_button = ttk.Button(finish_controls_frame, text="✅ Finish as Valid", command=self.controller.finish_as_valid, style="Valid.TButton")
        self.valid_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.rerun_button = ttk.Button(
            finish_controls_frame, 
            text="🔄 Rerun Auto-Processing", 
            command=self._on_rerun_button_click, 
            style="Rerun.TButton"
        )
        self.rerun_button.pack(side=tk.LEFT, padx=5, pady=5)

        # --- Marked Frames List Pane ---
        list_pane.rowconfigure(1, weight=1)
        list_pane.rowconfigure(4, weight=1)
        list_pane.columnconfigure(0, weight=1)
        
        ttk.Label(list_pane, text="Frames Marked for Labeling", font=("", 10, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0,5))
        
        self.label_listbox = tk.Listbox(list_pane, height=10)
        self.label_listbox.grid(row=1, column=0, columnspan=2, sticky="nsew")
        self.label_listbox.bind('<<ListboxSelect>>', self._on_label_listbox_select)
        # --- NEW: Bind FocusIn to track the active listbox ---
        self.label_listbox.bind('<FocusIn>', self._set_active_listbox)
        
        list_scrollbar = ttk.Scrollbar(list_pane, orient="vertical", command=self.label_listbox.yview)
        list_scrollbar.grid(row=1, column=2, sticky="ns")
        self.label_listbox['yscrollcommand'] = list_scrollbar.set
        
        delete_button = ttk.Button(list_pane, text="Delete Selected", command=self._delete_from_label_listbox, style="Delete.TButton")
        delete_button.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(5,0))

        # --- Add the list for frames marked for deletion ---
        ttk.Label(list_pane, text="Frames Marked for Deletion", font=("", 10, "bold")).grid(row=3, column=0, columnspan=2, sticky="w", pady=(10,5))
        
        self.deleting_listbox = tk.Listbox(list_pane, height=10)
        self.deleting_listbox.grid(row=4, column=0, columnspan=2, sticky="nsew")
        self.deleting_listbox.bind('<<ListboxSelect>>', self._on_deleting_listbox_select)
        # --- NEW: Bind FocusIn to track the active listbox ---
        self.deleting_listbox.bind('<FocusIn>', self._set_active_listbox)
        
        deleting_list_scrollbar = ttk.Scrollbar(list_pane, orient="vertical", command=self.deleting_listbox.yview)
        deleting_list_scrollbar.grid(row=4, column=2, sticky="ns")
        self.deleting_listbox['yscrollcommand'] = deleting_list_scrollbar.set
        
        delete_deleting_button = ttk.Button(list_pane, text="Delete Selected", command=self._delete_from_deleting_listbox, style="Delete.TButton")
        delete_deleting_button.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(5,0))

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
        # --- MODIFIED: Bind Delete key to the new generic handler ---
        self.root.bind('<Delete>', lambda e: self._delete_from_active_listbox())
        self.root.bind('m', lambda e: self._on_mark_labeling_button_click())
        self.root.bind('d', lambda e: self._on_mark_deleting_button_click())

    def _update_valid_button_state(self):
        """
        Updates the state of the 'Finish as Valid' button.
        
        The button is enabled (NORMAL) only if the marked list is empty AND all 
        object checkboxes are checked. Otherwise, it is disabled.
        """
        if not hasattr(self, 'valid_button') or not self.valid_button:
            return

        is_list_empty = self.label_listbox.size() == 0 and self.deleting_listbox.size() == 0

        all_boxes_checked = True
        if hasattr(self, 'object_vars') and self.object_vars:
            all_boxes_checked = all(var.get() for var in self.object_vars.values())

        if is_list_empty and all_boxes_checked:
            self.valid_button.config(state=tk.NORMAL)
        else:
            self.valid_button.config(state=tk.DISABLED)
    
    def _update_mark_buttons_state(self):
        """
        Disables the 'Mark' buttons if the current frame is already in one of
        the marked lists.
        """
        if not self.mark_label_button or not self.mark_delete_button:
            return

        current_frame = self.scale_var.get()
        
        if self.controller.is_marked(current_frame) or not (current_frame in self.landmarks):
            self.mark_delete_button.config(state=tk.DISABLED)
        else:
            self.mark_delete_button.config(state=tk.NORMAL)

        if self.controller.is_marked(current_frame):
            self.mark_label_button.config(state=tk.DISABLED)
        else:
            self.mark_label_button.config(state=tk.NORMAL)
        


    # --- PRIVATE EVENT HANDLERS (Forwarding to Controller) ---
    
    def _on_rerun_button_click(self):
        """
        Handles the click event for the 'Rerun Auto-Processing' button.
        This simply calls the controller's standard 'proceed' method, which
        is expected to handle the application logic and close the GUI.
        """
        self.controller.finish_and_proceed()
        
    def _on_mark_labeling_button_click(self):
        """
        Gathers selected objects from checkboxes and tells the controller to mark
        the current frame for labeling. Resets checkboxes to their default state.
        """
        # --- MODIFICATION: Added a check to prevent action if disabled ---
        if self.mark_label_button['state'] == tk.DISABLED:
            self.pop_up_window(message="This frame is already marked.")
            return

        selected_objects = [name for name, var in self.object_vars.items() if var.get()]
        self.controller.mark_frame(FrameAction.LABEL, objects_to_mark=selected_objects)

        for var in self.object_vars.values():
            var.set(True)

    def _on_mark_deleting_button_click(self):
        """
        Tells the controller to mark the current frame for deletion.
        """
        # --- MODIFICATION: Added a check to prevent action if disabled ---
        if self.mark_delete_button['state'] == tk.DISABLED:
            self.pop_up_window(message="This frame is already marked.")
            return
            
        selected_objects = [name for name, var in self.object_vars.items() if var.get()]
        self.controller.mark_frame(FrameAction.DELETE, objects_to_mark=selected_objects)
            
        for var in self.object_vars.values():
            var.set(True)

    def _on_label_listbox_select(self, event):
        """When a listbox item is selected, tell the controller to seek."""
        selection_indices = self.label_listbox.curselection()
        if not selection_indices: return
        
        selected_index = selection_indices[0]
        
        try:
            listbox_text = self.label_listbox.get(selected_index)
            frame_part = listbox_text.split(':')[0]
            frame_num = int(frame_part.split()[-1])
            self.controller.seek_to_frame(frame_num)
        except (ValueError, IndexError):
            print(f"Error parsing listbox entry: {self.label_listbox.get(selected_index)}")
            pass 

    def _on_deleting_listbox_select(self, event):
        """When a deleting listbox item is selected, tell the controller to seek."""
        selection_indices = self.deleting_listbox.curselection()
        if not selection_indices: return
        
        selected_index = selection_indices[0]
        
        try:
            listbox_text = self.deleting_listbox.get(selected_index)
            frame_part = listbox_text.split(':')[0]
            frame_num = int(frame_part.split()[-1])
            self.controller.seek_to_frame(frame_num)
        except (ValueError, IndexError):
            print(f"Error parsing deleting listbox entry: {self.deleting_listbox.get(selected_index)}")
            pass

    # --- NEW: Method to set the active listbox based on focus ---
    def _set_active_listbox(self, event):
        """Sets the currently focused listbox as the active one."""
        self.active_listbox = event.widget
    
    # --- NEW: Generic delete method that acts on the focused listbox ---
    def _delete_from_active_listbox(self):
        """Deletes the selected item from whichever listbox currently has focus."""
        if not self.active_listbox:
            # If no listbox has been focused yet, do nothing.
            return

        if self.active_listbox == self.label_listbox:
            self._delete_from_label_listbox()
        elif self.active_listbox == self.deleting_listbox:
            self._delete_from_deleting_listbox()

    def _delete_from_label_listbox(self):
        """Tells the controller to delete the selected item from the marked list."""
        selection_indices = self.label_listbox.curselection()
        if not selection_indices: return
        
        # This assumes the controller can map the index back to the correct item.
        # See architectural critique for potential improvements.
        selected_index = selection_indices[0]
        try:
            listbox_text = self.label_listbox.get(selected_index)
            frame_part = listbox_text.split(':')[0]
            frame_num = int(frame_part.split()[-1])
            self.controller.remove_mark(frame_num)
        except (ValueError, IndexError):
            print(f"Error parsing listbox entry for deletion: {self.label_listbox.get(selected_index)}")

    def _delete_from_deleting_listbox(self):
        """Tells the controller to delete the selected item from the deleting list."""
        selection_indices = self.deleting_listbox.curselection()
        if not selection_indices: return
        
        # This assumes the controller can map the index back to the correct item.
        # See architectural critique for potential improvements.
        selected_index = selection_indices[0]
        try:
            listbox_text = self.deleting_listbox.get(selected_index)
            frame_part = listbox_text.split(':')[0]
            frame_num = int(frame_part.split()[-1])
            self.controller.remove_mark(frame_num)
        except (ValueError, IndexError):
            print(f"Error parsing listbox entry for deletion: {self.deleting_listbox.get(selected_index)}")
        
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
    
    def pop_up_window(self, *, message: str, title: str = "Info"):
        """
        Generates a small modal pop-up box with a message.

        Args:
            message (str): The message to display in the pop-up.
            title (str, optional): The title for the pop-up window. Defaults to "Info".
        """
        popup = Toplevel(self.root)
        popup.title(title)
        popup.transient(self.root)  # Keep popup on top of the main window
        popup.grab_set()        # Make the popup modal
        popup.resizable(False, False)

        # Calculate position to center on parent window
        self.root.update_idletasks()
        parent_x = self.root.winfo_x()
        parent_y = self.root.winfo_y()
        parent_width = self.root.winfo_width()
        parent_height = self.root.winfo_height()
        
        popup_width = 320
        popup_height = 120
        
        center_x = parent_x + (parent_width - popup_width) // 2
        center_y = parent_y + (parent_height - popup_height) // 2
        
        popup.geometry(f'{popup_width}x{popup_height}+{center_x}+{center_y}')
        
        # Create and pack widgets
        popup_frame = ttk.Frame(popup, padding="15")
        popup_frame.pack(expand=True, fill=tk.BOTH)
        popup_frame.rowconfigure(0, weight=1)
        popup_frame.columnconfigure(0, weight=1)

        label = ttk.Label(popup_frame, text=message, wraplength=popup_width - 30, justify=tk.CENTER)
        label.grid(row=0, column=0, sticky="nsew")

        ok_button = ttk.Button(popup_frame, text="OK", command=popup.destroy)
        ok_button.grid(row=1, column=0, pady=(10, 0))
        ok_button.focus_set()

        # Wait for the user to close the popup before continuing
        self.root.wait_window(popup)

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
        # --- MODIFICATION: Update button state when timeline changes ---
        self._update_mark_buttons_state()

    def update_frame_label(self, current_num, total_frames):
        """Updates the 'Frame X / Y' text label."""
        self.current_frame_label.config(text=f"Frame: {current_num} / {total_frames - 1}")

    def update_play_pause_button(self, is_paused):
        """Updates the text of the play/pause button."""
        self.play_pause_button.config(text="Play" if is_paused else "Pause")

    def update_speed_label(self, speed):
        """Updates the speed label text."""
        self.speed_label.config(text=f"Speed: {speed:.1f}x")

    def update_marked_list(self, marked_frames: Dict[int, FrameMark]):
            """
            Splits the main marked_frames dictionary into separate collections for
            labeling and deletion, then calls the specific UI update methods for each
            listbox.
            
            Args:
                marked_frames (Dict[int, FrameMark]): A dictionary mapping frame numbers
                    to FrameMark objects, containing the action and associated data.
            """
            # 1. Initialize empty collections to hold the separated data.
            label_list_data = {}
            deleting_list_data = {}

            # 2. Iterate through the consolidated dictionary from the controller.
            for frame_num, frame_mark in marked_frames.items():
                # 3. Check the action type and populate the appropriate collection.
                if frame_mark.action == FrameAction.LABEL:
                    label_list_data[frame_num] = frame_mark.object_ids
                elif frame_mark.action == FrameAction.DELETE:
                    deleting_list_data[frame_num] = frame_mark.object_ids

            # 4. Call the dedicated update methods with the newly created collections.
            self.update_label_list(label_list_data)
            self.update_deleting_list(deleting_list_data)
            
            # 5. Update the state of the finish buttons based on the new list state.
            self.update_finish_buttons(marked_frames)
    
    def update_label_list(self, label_list: dict[int, List[str]]):
        """
        Refreshes the listbox with the current list of marked frames and their associated objects.
        """
        self.label_listbox.delete(0, tk.END)
        for frame_num, objects in label_list.items():
            object_str = ", ".join(objects)
            self.label_listbox.insert(tk.END, f"Frame {frame_num}: {object_str}")
        self._update_mark_buttons_state()

    def update_deleting_list(self, deleting_list: dict[int, List[str]]):
        """
        Refreshes the listbox with the current list of frames marked for deletion.
        """
        self.deleting_listbox.delete(0, tk.END)
        for frame_num, objects in deleting_list.items():
            object_str = ", ".join(objects)
            self.deleting_listbox.insert(tk.END, f"Frame {frame_num}: {object_str}")
        self._update_mark_buttons_state()

    def update_finish_buttons(self, label_list):
        """Enables/disables finish buttons based on the state of the marked list."""
        is_list_empty = not label_list
        
        self.proceed_button.config(state=tk.DISABLED if is_list_empty else tk.NORMAL)
        
        if self.rerun_button:
            self.rerun_button.config(state=tk.NORMAL if is_list_empty else tk.DISABLED)
    
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