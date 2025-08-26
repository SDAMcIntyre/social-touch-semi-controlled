# --- 2. THE VIEW ---
import cv2
import tkinter as tk
import numpy as np


class ColorspaceEditorView:
    """
    The View component. It is responsible for rendering the UI and capturing
    raw user input. It is "dumb" and holds no application state or logic.
    """
    def __init__(self, controller, window_title="Colorspace Editor"):
        self.controller = controller
        self.window_name = window_title
        self.window_h = 720  # Default window size
        self.window_w = 1280
        
        # Colors for drawing, can be configured by the controller
        self.color_live = (0, 255, 255)  # Yellow
        self.color_final = (255, 0, 0)   # Blue
        self.color_annotated = (0, 255, 0) # Green for already annotated frames

        self._setup_window()

    def _setup_window(self):
        """Sets up the OpenCV window, centers it, and binds the mouse callback."""
        cv2.namedWindow(self.window_name)
        # Centering logic can be kept here as it's purely a UI concern
        try:
            root = tk.Tk()
            root.withdraw()
            screen_w = root.winfo_screenwidth()
            screen_h = root.winfo_screenheight()
            x = max(0, int(screen_w / 2 - self.window_w / 2))
            y = max(0, int(screen_h / 2 - self.window_h / 2))
            root.destroy()
            cv2.moveWindow(self.window_name, x, y)
        except Exception:
            print("⚠️ Could not auto-center window.")
        
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, param):
        """
        Captures raw mouse events and forwards them to the controller.
        The View itself does not know what these events mean.
        """
        self.controller.handle_mouse_event(event, x, y, flags)

    def update(self, frame, roi_points, fitted_ellipse, zoom, pan_offset, is_annotated):
        """
        Redraws the entire display based on the state provided by the Controller.
        """
        display_image = cv2.resize(frame, (self.window_w, self.window_h))
        M = np.float32([[zoom, 0, pan_offset[0]], [0, zoom, pan_offset[1]]])
        display_image = cv2.warpAffine(display_image, M, (self.window_w, self.window_h))

        # Draw current user drawing
        if roi_points:
            self._draw_shape(display_image, roi_points, self.color_live, zoom, pan_offset)

        # Draw confirmed ellipse
        if fitted_ellipse:
            color = self.color_annotated if is_annotated else self.color_final
            self._draw_ellipse(display_image, fitted_ellipse, color, zoom, pan_offset)
        
        self._draw_hud(display_image)
        cv2.imshow(self.window_name, display_image)

    def _draw_shape(self, image, points, color, zoom, pan):
        """Helper to draw the free-hand shape."""
        transformed_pts = [ (px * zoom + pan[0], py * zoom + pan[1]) for px, py in points]
        cv2.polylines(image, [np.array(transformed_pts, dtype=np.int32)], isClosed=False, color=color, thickness=2)

    def _draw_ellipse(self, image, ellipse, color, zoom, pan):
        """Helper to draw the fitted ellipse."""
        center = (ellipse[0][0] * zoom + pan[0], ellipse[0][1] * zoom + pan[1])
        axes = (ellipse[1][0] * zoom, ellipse[1][1] * zoom)
        display_ellipse = (center, axes, ellipse[2])
        cv2.ellipse(image, display_ellipse, color, 2)

    def _draw_hud(self, image):
        """Draws help text and instructions on the screen."""
        commands = [
            "Pan: R-Click | Zoom: Wheel", "Draw: L-Click",
            "Confirm: ENTER | Reset: 'r'",
            "Navigate: <- / -> arrows", "Save & Quit: ESC",
        ]
        y0, dy = 25, 20
        for i, line in enumerate(commands):
            y = y0 + i * dy
            cv2.putText(image, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(image, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def close(self):
        cv2.destroyAllWindows()