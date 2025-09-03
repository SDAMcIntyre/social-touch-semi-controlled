import cv2
import json
import numpy as np
from PIL import Image
import tkinter as tk # Import tkinter to get screen dimensions

class FrameROISquare:
    """
    A class to display an image, allow the user to zoom, pan, draw a 
    rectangular region of interest (ROI), and return its dimensions. 
    Can also be initialized with a predefined ROI.

    Controls:
        - 'Proceed' Button: Confirm the ROI and exit.
        - Left Mouse Button (Drag): Draw a rectangle.
        - Right/Middle Mouse Button (Drag): Pan/move the image.
        - Mouse Wheel Scroll: Zoom in and out, centered on the mouse pointer.
        - CTRL+Z: Reset the drawn ROI.
        - Enter/Space key: Alternative keys to confirm the ROI.
        - 'ESC' key: Close the window.
    """

    def __init__(self, image_input, is_rgb: bool = True, window_title: str = None, 
                 color_live: tuple = (0, 255, 0), color_final: tuple = (0, 0, 255),
                 predefined_roi: dict = None):
        """
        Initializes the tracker with an image.

        Args:
            image_input (np.ndarray or PIL.Image.Image): The input image as a 
                                                         NumPy array or a PIL Image.
            is_rgb (bool): Flag indicating if the input NumPy array is in RGB format.
            window_title (str): Optional title for the window.
            color_live (tuple): BGR color tuple for the live drawing ROI. 
                                Defaults to green.
            color_final (tuple): BGR color tuple for the finalized ROI. 
                                 Defaults to red.
            predefined_roi (dict, optional): A dictionary with keys 'x', 'y', 
                                             'width', 'height' to pre-load an ROI.
        """
        image_array_rgb = None

        if isinstance(image_input, Image.Image):
            image_array_rgb = np.array(image_input.convert('RGB'))
        elif isinstance(image_input, np.ndarray):
            image_array_rgb = image_input
        else:
            raise TypeError("Input must be a NumPy array or a PIL.Image.Image.")
        
        if image_array_rgb.ndim != 3 or image_array_rgb.shape[2] != 3:
            raise ValueError("Input image must be in a 3-channel (RGB) format.")

        if is_rgb:
            self.image = cv2.cvtColor(image_array_rgb, cv2.COLOR_RGB2BGR)
        else:
            self.image = image_array_rgb

        if self.image is None:
            raise ValueError("The provided image array is invalid.")

        self.original_image = self.image.copy()
        if window_title:
            self.window_name = window_title
        else:
            self.window_name = "Interactive ROI Selector"
        self.window_h, self.window_w = self.original_image.shape[:2]

        # State variables
        self.zoom_factor = 1.0
        self.pan_offset_x, self.pan_offset_y = 0.0, 0.0
        self.pan_start_x, self.pan_start_y = 0.0, 0.0
        self.panning = False
        self.drawing = False
        
        # Color variables
        self.color_live = color_live
        self.color_final = color_final

        # ROI variables
        self.roi_start_point = None
        self.roi_end_point = None
        self.roi_rect = None # Stores (x, y, w, h) in original image coordinates
        self.roi_confirmed = False

        # --- New: Process predefined_roi ---
        if predefined_roi:
            # Validate that the input is a dictionary with the required keys
            if isinstance(predefined_roi, dict) and all(k in predefined_roi for k in ['x', 'y', 'width', 'height']):
                x = predefined_roi['x']
                y = predefined_roi['y']
                w = predefined_roi['width']
                h = predefined_roi['height']
                
                # Basic validation: ensure ROI is within image bounds
                img_h, img_w = self.original_image.shape[:2]
                if 0 <= x < img_w and 0 <= y < img_h and x + w <= img_w and y + h <= img_h:
                    self.roi_rect = (int(x), int(y), int(w), int(h))
                    print(f"✅ Predefined ROI loaded: {self.roi_rect}")
                else:
                    print(f"[Warning] Predefined ROI {predefined_roi} is out of image bounds ({img_w}x{img_h}). Ignoring.")
            else:
                print("[Warning] 'predefined_roi' must be a dictionary with keys 'x', 'y', 'width', 'height'. Ignoring.")

        self._setup_window()

    def _setup_window(self):
        """Sets up the OpenCV window, centers it, and adds a button."""
        cv2.namedWindow(self.window_name)

        try:
            # Center the window on the screen
            root = tk.Tk()
            root.withdraw()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()

            center_x = max(0, int(screen_width / 2 - self.window_w / 2))
            center_y = max(0, int(screen_height / 2 - self.window_h / 2))
            cv2.moveWindow(self.window_name, center_x, center_y)

            # Add a 'Proceed' button if QT backend is available
            cv2.createButton(
                "Proceed",
                lambda state, userdata: self.confirm_roi(),
                None,
                cv2.QT_PUSH_BUTTON,
                1
            )
        except cv2.error:
            print("\n[Warning] Could not set up QT features (button, window centering).")
            print("This may be because your OpenCV installation lacks QT support.")
            print("Please use the Enter or Space key to confirm the ROI instead.\n")

        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, param):
        """Handles all mouse events for drawing, panning, and zooming."""
        
        # Convert window coordinates to original image coordinates
        original_x = (x - self.pan_offset_x) / self.zoom_factor
        original_y = (y - self.pan_offset_y) / self.zoom_factor

        # --- Drawing Logic (Left Mouse Button) ---
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.reset_roi() # Reset any previous ROI
            self.roi_start_point = (original_x, original_y)
            self.roi_end_point = (original_x, original_y)
            self._update_display()

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.roi_end_point = (original_x, original_y)
            self._update_display()

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                self.roi_end_point = (original_x, original_y)
                
                # Calculate the final rectangle (x, y, width, height)
                x1, y1 = self.roi_start_point
                x2, y2 = self.roi_end_point
                
                start_x = min(x1, x2)
                start_y = min(y1, y2)
                width = abs(x1 - x2)
                height = abs(y1 - y2)
                
                self.roi_rect = (int(start_x), int(start_y), int(width), int(height))
                self._update_display()

        # --- Panning Logic (Right OR Middle Mouse Button) ---
        elif event == cv2.EVENT_RBUTTONDOWN or event == cv2.EVENT_MBUTTONDOWN:
            self.panning = True
            self.pan_start_x, self.pan_start_y = float(x), float(y)

        elif event == cv2.EVENT_MOUSEMOVE and self.panning:
            dx = x - self.pan_start_x
            dy = y - self.pan_start_y
            self.pan_offset_x += dx
            self.pan_offset_y += dy
            self.pan_start_x, self.pan_start_y = float(x), float(y)
            self._update_display()

        elif event == cv2.EVENT_RBUTTONUP or event == cv2.EVENT_MBUTTONUP:
            self.panning = False

        # --- Centered Zoom Logic (Mouse Wheel) ---
        elif event == cv2.EVENT_MOUSEWHEEL:
            old_zoom = self.zoom_factor
            
            if flags > 0:
                self.zoom_factor *= 1.1
            else:
                self.zoom_factor *= 0.9
            
            self.zoom_factor = max(0.1, min(self.zoom_factor, 10.0))

            # Adjust pan offset to keep the zoom centered on the mouse pointer
            mouse_x_on_image = (x - self.pan_offset_x) / old_zoom
            mouse_y_on_image = (y - self.pan_offset_y) / old_zoom
            self.pan_offset_x = x - mouse_x_on_image * self.zoom_factor
            self.pan_offset_y = y - mouse_y_on_image * self.zoom_factor
            
            self._update_display()

    def _update_display(self):
        """
        Redraws the image with the current zoom/pan and overlays the ROI,
        including diagonals.
        """
        # Create transformation matrix for zoom and pan
        M = np.float32([
            [self.zoom_factor, 0, self.pan_offset_x],
            [0, self.zoom_factor, self.pan_offset_y]
        ])
        display_image = cv2.warpAffine(
            self.original_image, M, (self.window_w, self.window_h)
        )

        # If currently drawing, use start/end points for a live preview
        if self.drawing and self.roi_start_point and self.roi_end_point:
            # Transform ROI points to display coordinates
            start_disp_x = self.roi_start_point[0] * self.zoom_factor + self.pan_offset_x
            start_disp_y = self.roi_start_point[1] * self.zoom_factor + self.pan_offset_y
            end_disp_x = self.roi_end_point[0] * self.zoom_factor + self.pan_offset_x
            end_disp_y = self.roi_end_point[1] * self.zoom_factor + self.pan_offset_y
            
            color = self.color_live
            thickness = 2
            
            # Draw rectangle
            cv2.rectangle(display_image, 
                          (int(start_disp_x), int(start_disp_y)), 
                          (int(end_disp_x), int(end_disp_y)), 
                          color, thickness)
            
            # Draw diagonals
            cv2.line(display_image, 
                     (int(start_disp_x), int(start_disp_y)), 
                     (int(end_disp_x), int(end_disp_y)), 
                     color, thickness)
            cv2.line(display_image, 
                     (int(start_disp_x), int(end_disp_y)), 
                     (int(end_disp_x), int(start_disp_y)), 
                     color, thickness)

        # If drawing is finished, use the finalized roi_rect for a persistent box
        elif self.roi_rect:
            x, y, w, h = self.roi_rect
            # Transform rect properties to display coordinates
            disp_x = x * self.zoom_factor + self.pan_offset_x
            disp_y = y * self.zoom_factor + self.pan_offset_y
            disp_w = w * self.zoom_factor
            disp_h = h * self.zoom_factor
            
            color = self.color_final
            thickness = 2

            # Define the four corners in display coordinates
            pt1 = (int(disp_x), int(disp_y))               # Top-left
            pt2 = (int(disp_x + disp_w), int(disp_y + disp_h)) # Bottom-right
            pt3 = (int(disp_x + disp_w), int(disp_y))      # Top-right
            pt4 = (int(disp_x), int(disp_y + disp_h))      # Bottom-left
            
            # Draw rectangle
            cv2.rectangle(display_image, pt1, pt2, color, thickness)
            
            # Draw diagonals
            cv2.line(display_image, pt1, pt2, color, thickness) # Top-left to bottom-right
            cv2.line(display_image, pt3, pt4, color, thickness) # Top-right to bottom-left

        # Add command instructions overlay
        commands = [
            "Pan: Right/Middle-Click Drag",
            "Zoom: Mouse Wheel",
            "Draw ROI: Left-Click Drag",
            "Confirm: 'Proceed' Button, Enter, or Space",
            "Reset: CTRL+Z",
            "Exit: ESC",
        ]
        
        y0, dy = 10 + 15, 20
        for i, line in enumerate(commands):
            y = y0 + i * dy
            cv2.putText(display_image, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(display_image, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(self.window_name, display_image)

    def set_color_live(self, color: tuple):
        """
        Sets the color for the live drawing ROI.

        Args:
            color (tuple): A 3-element BGR color tuple (e.g., (255, 0, 0) for blue).
        """
        if isinstance(color, tuple) and len(color) == 3:
            self.color_live = color
            self._update_display() # Update display to reflect the change
        else:
            print("Warning: Color must be a 3-element tuple (B, G, R).")

    def set_color_final(self, color: tuple):
        """
        Sets the color for the finalized ROI.

        Args:
            color (tuple): A 3-element BGR color tuple (e.g., (0, 255, 255) for yellow).
        """
        if isinstance(color, tuple) and len(color) == 3:
            self.color_final = color
            self._update_display() # Update display to reflect the change
        else:
            print("Warning: Color must be a 3-element tuple (B, G, R).")

    def reset_roi(self):
        """Clears the drawn rectangle."""
        print("ROI has been reset.")
        self.roi_start_point = None
        self.roi_end_point = None
        self.roi_rect = None
        self._update_display()

    def confirm_roi(self):
        """Confirms the selected ROI and prepares to exit the main loop."""
        if not self.roi_rect:
            print("No ROI selected. Draw a rectangle first.")
            return

        print(f"✅ ROI Confirmed:")
        x, y, w, h = self.roi_rect
        print(f"   - Top-left (x, y): ({x}, {y})")
        print(f"   - Dimensions (width, height): ({w}, {h})")
        
        self.roi_confirmed = True

    def run(self):
        """Starts the main event loop for the interactive window."""
        print("✅ Window is now active. Use the 'Proceed' button or keyboard controls.")
        self._update_display()
        
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == 27 or self.roi_confirmed: # Exit on ESC or after confirmation
                break
            
            if key == 26: # CTRL+Z for reset
                self.reset_roi()

            if key == 13 or key == 32: # 13 is Enter, 32 is Space
                self.confirm_roi()
        
        cv2.destroyAllWindows()

    def get_roi_data(self):
        """
        Returns the final ROI parameters as a dictionary.

        Returns:
            dict: A dictionary containing the ROI's 'x', 'y', 'width', and 'height',
                  or None if no ROI was confirmed.
        """
        if self.roi_rect:
            x, y, w, h = self.roi_rect
            return {
                "x": x,
                "y": y,
                "width": w,
                "height": h
            }
        return None