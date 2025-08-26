# --- 3. THE CONTROLLER ---
import pandas as pd
import numpy as np
import cv2
from typing import Any, Dict, List

from ColorspaceFileHandler import ColorspaceFileHandler
from ColorspaceModel import ColorspaceEntry, AnnotationSession
from ColorspaceEditorView import ColorspaceEditorView


class ColorspaceController:
    """
    The Controller. It connects the View (UI) and the Model (data),
    and contains all the application logic and state management.
    """
    def __init__(self, frames: List[np.ndarray], output_path: str):
        # Initialize Model and View
        self.session = AnnotationSession(frames=frames)
        self.view = ColorspaceEditorView(self)
        self.file_handler = ColorspaceFileHandler(output_path) # For saving data
        self.session = AnnotationSession(frames=frames)
        self.view = ColorspaceEditorView(self)
        self.file_handler = dest_file_handler
        self.object_name = object_name

        # UI Interaction State (previously in FrameROIColor)
        self.zoom = 1.0
        self.pan_offset = [0.0, 0.0]
        self.pan_start = [0.0, 0.0]
        self.is_panning = False
        self.is_drawing = False

        # ROI Data State for the current frame
        self.roi_points = []
        self.fitted_ellipse = None

    def run(self):
        """Starts the main application loop."""
        self._update_view()
        while True:
            key = cv2.waitKey(20) & 0xFF
            
            # --- Handle Keyboard Input ---
            if key == 27: # ESC: Save and Quit
                # self.save_session_data()
                print("Session finished.")
                break
            elif key == 13: # ENTER: Confirm ROI
                self.confirm_roi()
            elif key == ord('r'): # 'r': Reset ROI
                self.reset_roi()
            elif key == 81 or key == 2: # Left Arrow
                self.navigate_prev()
            elif key == 83 or key == 3: # Right Arrow
                self.navigate_next()

        self.view.close()

    def handle_mouse_event(self, event, x, y, flags):
        """Processes raw mouse events from the View and updates state."""
        # Drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_drawing = True
            self.reset_roi() # Start a new drawing
            self.roi_points.append(self._to_image_coords(x, y))
        elif event == cv2.EVENT_MOUSEMOVE and self.is_drawing:
            self.roi_points.append(self._to_image_coords(x, y))
        elif event == cv2.EVENT_LBUTTONUP and self.is_drawing:
            self.is_drawing = False
            if len(self.roi_points) > 5:
                # The view expects coordinates relative to the resized (display) image
                h, w = self.view.window_h, self.view.window_w
                points_for_fitting = np.array([(p[0]*w, p[1]*h) for p in self.roi_points], dtype=np.float32)
                self.fitted_ellipse = cv2.fitEllipse(points_for_fitting)

        # Panning
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.is_panning = True
            self.pan_start = [x, y]
        elif event == cv2.EVENT_MOUSEMOVE and self.is_panning:
            dx = x - self.pan_start[0]
            dy = y - self.pan_start[1]
            self.pan_offset[0] += dx
            self.pan_offset[1] += dy
            self.pan_start = [x, y]
        elif event == cv2.EVENT_RBUTTONUP:
            self.is_panning = False

        # Zooming
        elif event == cv2.EVENT_MOUSEWHEEL:
            old_zoom = self.zoom
            zoom_delta = 1.1 if flags > 0 else 0.9
            self.zoom = max(0.1, min(self.zoom * zoom_delta, 10.0))
            # Adjust pan to keep mouse position stable
            mouse_x_img = (x - self.pan_offset[0]) / old_zoom
            mouse_y_img = (y - self.pan_offset[1]) / old_zoom
            self.pan_offset[0] = x - mouse_x_img * self.zoom
            self.pan_offset[1] = y - mouse_y_img * self.zoom

        self._update_view()
    
    # --- Actions ---

    def navigate_next(self):
        self.session.next_frame()
        self.reset_roi() # Clear drawing when changing frames
        print(f"Navigated to frame {self.session.current_frame_index}")

    def navigate_prev(self):
        self.session.prev_frame()
        self.reset_roi()
        print(f"Navigated to frame {self.session.current_frame_index}")

    def confirm_roi(self):
        """Analyzes the current ROI and adds it to the session model."""
        if not self.fitted_ellipse:
            print("⚠️ No ellipse drawn to confirm.")
            return
        
        # This is the logic from analyze_roi_color and get_tracking_data
        analysis_data = self._analyze_current_roi()
        self.session.add_annotation(analysis_data)
        self._update_view()

    def reset_roi(self):
        self.roi_points = []
        self.fitted_ellipse = None
        self._update_view()

    # --- Helper Methods ---

    def _update_view(self):
        """Gets current state and tells the View to redraw itself."""
        frame = self.session.get_current_frame()
        is_annotated = self.session.has_annotation_for_current_frame()
        
        # If there's a saved annotation, show it instead of the live drawing
        ellipse_to_show = self.fitted_ellipse
        if is_annotated and not self.is_drawing:
             # This assumes the data is stored in a compatible format
             stored_data = self.session.annotations[self.session.current_frame_index].colorspace_data
             if "ellipse_display" in stored_data:
                ellipse_to_show = stored_data["ellipse_display"]

        self.view.update(frame, self.roi_points, ellipse_to_show, self.zoom, self.pan_offset, is_annotated)
    
    def _to_image_coords(self, x_win, y_win):
        """Converts window coordinates to normalized image coordinates."""
        img_x = (x_win - self.pan_offset[0]) / self.zoom
        img_y = (y_win - self.pan_offset[1]) / self.zoom
        return (img_x / self.view.window_w, img_y / self.view.window_h)

    def _analyze_current_roi(self) -> Dict[str, Any]:
        """
        Contains the analysis logic. This is a simplified version of the
        original class's methods. Returns a dictionary for the Model.
        """
        # Note: Analysis should be done on the ORIGINAL image, not the display one.
        # This requires scaling coordinates back, which is omitted here for brevity
        # but was present in the original get_tracking_data method.
        output = {}
        if self.fitted_ellipse:
            output["ellipse_display"] = self.fitted_ellipse
            # ... add scaled coordinates, mean color, std dev, etc. here ...
        # ... add freehand pixel extraction here ...
        print("ROI analyzed (details omitted for brevity).")
        return output

    def save_session_data(self):
        """Formats session data and uses the file handler to save it."""
        frame_ids = list(self.session.annotations.keys())
        colorspaces = [ann.colorspace_data for ann in self.session.annotations.values()]
        self.file_handler.update_object("full_session", frame_ids, colorspaces, "completed")
        self.file_handler.save()