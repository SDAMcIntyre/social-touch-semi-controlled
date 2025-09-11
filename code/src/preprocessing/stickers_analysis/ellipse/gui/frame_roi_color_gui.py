import cv2
import json
import numpy as np
import pandas as pd
from PIL import Image
import tkinter as tk # Import tkinter to get screen dimensions

class FrameROIColor:
    """
    A class to display an image, allow the user to zoom, pan, draw a 
    free-hand region of interest (ROI), fit an ellipse to it, and 
    analyze the color properties of that elliptical region. It also extracts
    all pixels from the free-hand shape into a pandas DataFrame.

    The class can optionally resize the image for display to ensure a 
    consistent window size, while all final output data is scaled back to the 
    original image's coordinate space.

    Controls:
        - 'Proceed' Button: Confirm the ROI and analyze the fitted ellipse.
        - Left Mouse Button (Drag): Draw a free-hand shape.
        - Right/Middle Mouse Button (Drag): Pan/move the image.
        - Mouse Wheel Scroll: Zoom in and out, centered on the mouse pointer.
        - CTRL+Z: Reset the drawn ROI and fitted ellipse.
        - ENTER or 'c' key: Alternative keys to confirm the ROI.
        - 'ESC' key: Close the window.
    """

    def __init__(
            self, 
            image_input, 
            window_title=None, 
            resize_to=(1024, 768), 
            is_bgr=False, 
            color_live=(0, 255, 0), 
            color_final=(0, 0, 255)
    ):
        """
        Initializes the tracker with an image and color settings.

        Args:
            image_input (np.ndarray or PIL.Image.Image): The input image.
            window_title (str, optional): The title for the display window.
            resize_to (tuple, optional): A (width, height) tuple to resize the 
                                         image for display. If None, the original 
                                         size is used. Defaults to (1024, 768).
            is_bgr (bool, optional): Set to True if the input np.ndarray is 
                                     already in BGR format. Ignored for PIL
                                     inputs. Defaults to False (assumes RGB).
            color_live (tuple, optional): BGR color for the hand-drawn line.
                                          Defaults to green (0, 255, 0).
            color_final (tuple, optional): BGR color for the fitted ellipse.
                                           Defaults to red (0, 0, 255).
        """
        # --- 1. Validate and prepare the input image ---
        image_array = self._validate_and_convert_image(image_input, is_bgr)
        
        # 2. Convert to BGR for internal OpenCV operations if necessary
        # The _validate_and_convert_image method handles this and ensures image_array is in BGR.
        self.original_image_bgr = self._convert_to_bgr(image_array, is_bgr)

        # 3. Handle resizing for display
        self.original_h, self.original_w = self.original_image_bgr.shape[:2]
        self.was_resized = False
        self.image_for_display = self.original_image_bgr.copy()
        if resize_to and isinstance(resize_to, tuple) and len(resize_to) == 2:
            print(f"Resizing image from {self.original_w}x{self.original_h} to {resize_to[0]}x{resize_to[1]}")
            self.image_for_display = cv2.resize(self.image_for_display, resize_to, interpolation=cv2.INTER_AREA)
            self.was_resized = True

        if window_title:
            self.window_name = window_title
        else:
            self.window_name = "Interactive Color Tracker"
        self.window_h, self.window_w = self.image_for_display.shape[:2]

        # 4. Initialize state and color variables
        self.zoom_factor = 1.0
        self.pan_offset_x, self.pan_offset_y = 0.0, 0.0
        self.pan_start_x, self.pan_start_y = 0.0, 0.0
        self.panning = False
        self.drawing = False

        self.roi_points = []
        self.fitted_ellipse = None
        self.mean_color = None
        self.color_std_dev = None
        self.roi_pixels_df = None
        self.analysis_complete = False
        
        # New parameters for colors
        self.color_live = color_live
        self.color_final = color_final

        self._setup_window()

    def _validate_and_convert_image(self, image_input, is_bgr):
        """Helper to validate and convert the input image to a NumPy array."""
        image_array = None
        if isinstance(image_input, Image.Image):
            if is_bgr:
                print("[Warning] 'is_bgr=True' is ignored for PIL.Image input, which is assumed to be RGB.")
            image_array = np.array(image_input.convert('RGB'))
        elif isinstance(image_input, np.ndarray):
            image_array = image_input
        else:
            raise TypeError("Input must be a NumPy array or a PIL.Image.Image.")
        
        if image_array.ndim != 3 or image_array.shape[2] != 3:
            raise ValueError("Input image must be a 3-channel (color) format.")
            
        return image_array

    def _convert_to_bgr(self, image_array, is_bgr):
        """Helper to convert an image array to BGR if it's not already."""
        if is_bgr:
            return image_array.copy()
        else:
            return cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    def _setup_window(self):
        """Sets up the OpenCV window, centers it, and adds a button."""
        cv2.namedWindow(self.window_name)

        try:
            root = tk.Tk()
            root.withdraw()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()

            center_x = int(screen_width / 2 - self.window_w / 2)
            center_y = int(screen_height / 2 - self.window_h / 2)
            
            center_x = max(0, center_x)
            center_y = max(0, center_y)
            
            root.destroy()

            cv2.moveWindow(self.window_name, center_x, center_y)
        except cv2.error:
            print("\n[Warning] Could not set up QT features (button, window centering).")
            print("This may be because your OpenCV installation lacks QT support.")
            print("Please use the ENTER or 'c' key to confirm the ROI instead.\n")

        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, param):
        """Handles all mouse events for drawing, panning, and zooming."""
        
        # --- Drawing Logic (Left Mouse Button) ---
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.reset_roi()
            original_x = (x - self.pan_offset_x) / self.zoom_factor
            original_y = (y - self.pan_offset_y) / self.zoom_factor
            self.roi_points = [(original_x, original_y)]
            self._update_display()

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            original_x = (x - self.pan_offset_x) / self.zoom_factor
            original_y = (y - self.pan_offset_y) / self.zoom_factor
            self.roi_points.append((original_x, original_y))
            self._update_display()

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                if len(self.roi_points) > 5:
                    self.fitted_ellipse = cv2.fitEllipse(np.array(self.roi_points, dtype=np.float32))
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
            
            if flags > 0: self.zoom_factor *= 1.1
            else: self.zoom_factor *= 0.9
            
            self.zoom_factor = max(0.1, min(self.zoom_factor, 10.0))
            mouse_x_on_image = (x - self.pan_offset_x) / old_zoom
            mouse_y_on_image = (y - self.pan_offset_y) / old_zoom
            self.pan_offset_x = x - mouse_x_on_image * self.zoom_factor
            self.pan_offset_y = y - mouse_y_on_image * self.zoom_factor
            self._update_display()

    def _update_display(self):
        """Redraws the image, overlays, and instructions."""
        M = np.float32([[self.zoom_factor, 0, self.pan_offset_x], [0, self.zoom_factor, self.pan_offset_y]])
        display_image = cv2.warpAffine(self.image_for_display, M, (self.window_w, self.window_h))

        if self.roi_points:
            transformed_pts = [ (px * self.zoom_factor + self.pan_offset_x, py * self.zoom_factor + self.pan_offset_y) for px, py in self.roi_points]
            cv2.polylines(display_image, [np.array(transformed_pts, dtype=np.int32)], isClosed=False, color=self.color_live, thickness=2)

        if self.fitted_ellipse:
            center = (self.fitted_ellipse[0][0] * self.zoom_factor + self.pan_offset_x, self.fitted_ellipse[0][1] * self.zoom_factor + self.pan_offset_y)
            axes = (self.fitted_ellipse[1][0] * self.zoom_factor, self.fitted_ellipse[1][1] * self.zoom_factor)
            display_ellipse = (center, axes, self.fitted_ellipse[2])
            cv2.ellipse(display_image, display_ellipse, self.color_final, 2)
            
        # Updated command list to include ENTER key
        commands = [
            "Pan: Right/Middle-Click Drag",
            "Zoom: Mouse Wheel",
            "Draw Shape: Left-Click Drag",
            "Confirm: 'Proceed' Button, ENTER or 'c'",
            "Reset: CTRL+Z",
            "Exit: ESC",
        ]
        
        y0, dy = 25, 20
        for i, line in enumerate(commands):
            y = y0 + i * dy
            cv2.putText(display_image, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(display_image, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(self.window_name, display_image)

    def reset_roi(self):
        """Clears the drawn points, ellipse, DataFrame, and analysis results."""
        print("ROI has been reset.")
        self.roi_points = []
        self.fitted_ellipse = None
        self.mean_color = None
        self.color_std_dev = None
        self.roi_pixels_df = None # Reset the DataFrame
        self._update_display()

    def analyze_roi_color(self, show=False):
        """
        Analyzes the ROI, extracting pixels from both the fitted ellipse
        and the free-hand polygon.
        """
        if not self.roi_points:
            print("No ROI points drawn. Please draw a shape first.")
            return

        # --- 1. Analyze pixels within the FITTED ELLIPSE ---
        if self.fitted_ellipse:
            mask = np.zeros(self.image_for_display.shape[:2], dtype=np.uint8)
            cv2.ellipse(mask, self.fitted_ellipse, 255, -1)
            pixels = self.image_for_display[mask == 255]
            if len(pixels) > 0:
                self.mean_color = np.mean(pixels, axis=0)
                self.color_std_dev = np.std(pixels, axis=0)
                print(f"✅ Ellipse Analysis Complete:")
                print(f"  - Mean Color (BGR): {self.mean_color.astype(int)}")
                print(f"  - Color Std Dev (BGR): {self.color_std_dev.astype(int)}")
                if show: self._show_color_landscape(pixels)
            else:
                print("Warning: Could not extract pixels from the fitted ellipse.")
        else:
            print("No valid ellipse fitted. Draw a larger area to enable ellipse analysis.")

        # --- 2. Extract pixels from the FREE-HAND POLYGON ---
        poly_mask = np.zeros(self.image_for_display.shape[:2], dtype=np.uint8)
        poly_points = np.array(self.roi_points, dtype=np.int32)
        cv2.fillPoly(poly_mask, [poly_points], 255)

        rows, cols = np.where(poly_mask == 255) # y, x coordinates
        if rows.size > 0:
            # Get BGR colors for the selected pixels
            colors_bgr = self.image_for_display[rows, cols]
            # Combine x and y values into a list of [x, y] coordinates
            coordinates = list(zip(cols, rows))
            # Reverse the BGR columns to get RGB, then convert to a list of [r, g, b] values
            colors_rgb = colors_bgr[:, ::-1].tolist()
            # Create the dictionary for the new DataFrame structure
            data = {'coordinate': coordinates, 'rgb': colors_rgb}
            # Create the DataFrame with 'coordinate' and 'rgb' columns
            self.roi_pixels_df = pd.DataFrame(data)
            print(f"✅ Extracted {len(self.roi_pixels_df)} pixels from the free-hand drawing into a DataFrame.")
        else:
            print("Warning: Could not extract pixels from the free-hand drawing.")
            return

        self.analysis_complete = True

    def _show_color_landscape(self, pixels):
        """Creates and displays a visual representation of the color palette."""
        if self.mean_color is None: return
        sorted_pixels = sorted(pixels, key=lambda p: np.dot(p, [0.114, 0.587, 0.299])) 
        self.color_landscape = cv2.resize(np.array([sorted_pixels], dtype=np.uint8), (500, 100), interpolation=cv2.INTER_NEAREST)
        landscape_with_mean = np.zeros((150, 500, 3), dtype=np.uint8)
        landscape_with_mean[0:100, :] = self.color_landscape
        landscape_with_mean[100:150, :] = self.mean_color.astype(np.uint8)
        cv2.imshow("Color Analysis", landscape_with_mean)

    def run(self):
        """Starts the main event loop for the interactive window."""
        print("✅ Window is now active. Use the 'Proceed' button or keyboard controls.")
        self._update_display()
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == 27 or self.analysis_complete: # ESC or analysis is done
                break
            
            if key == 26: # CTRL+Z
                self.reset_roi()

            # Confirm ROI with ENTER (13) or 'c'
            if key == 13 or key == ord('c'):
                self.analyze_roi_color()
        
        self._on_quit()

    def get_tracking_data(self):
        """
        Returns all analysis results, scaled to the original image space.

        Returns:
            dict: A dictionary containing ellipse data and the free-hand
                  pixel DataFrame, or None if no analysis was completed.
        """
        output_data = {}

        # 1. Add ellipse data if available
        if self.fitted_ellipse and self.mean_color is not None:
            center, axes, angle = self.fitted_ellipse
            if self.was_resized:
                working_h, working_w = self.image_for_display.shape[:2]
                scale_x = self.original_w / working_w
                scale_y = self.original_h / working_h
                center = (center[0] * scale_x, center[1] * scale_y)
                axes = (axes[0] * scale_x, axes[1] * scale_y)
            
            if self.original_w > 0 and self.original_h > 0:
                output_data["ellipse"] = {
                    "center_original_px": center,
                    "axes_original_px": axes,
                    "center_normalized": (center[0] / self.original_w, center[1] / self.original_h),
                    "axes_normalized": (axes[0] / self.original_w, axes[1] / self.original_h),
                    "angle": angle
                }
                output_data["mean_color_bgr"] = self.mean_color.tolist()
                output_data["color_std_dev_bgr"] = self.color_std_dev.tolist()
        
        # 2. Add free-hand pixel DataFrame if available
        if self.roi_pixels_df is not None and not self.roi_pixels_df.empty:
            scaled_df = self.roi_pixels_df.copy()
            if self.was_resized:
                working_h, working_w = self.image_for_display.shape[:2]
                scale_x = self.original_w / working_w
                scale_y = self.original_h / working_h
                
                scaled_df['coordinate'] = scaled_df['coordinate'].apply(
                    lambda coord: [
                        int(round(coord[0] * scale_x)),
                        int(round(coord[1] * scale_y))
                    ]
                )
                
                scaled_df['coordinate'] = scaled_df['coordinate'].apply(tuple)

                averaged_df = scaled_df.groupby('coordinate')['rgb'].apply(
                    lambda x: np.mean(x.tolist(), axis=0).astype(int).tolist()
                ).reset_index()

            output_data["freehand_pixels"] = averaged_df.to_dict('list')
        
        return output_data if output_data else None

    def _on_quit(self):
        cv2.destroyAllWindows()


def main():
    """Main function to run the color tracker on a sample image."""
    image_path = "landscape.jpg" 
    frame_bgr = cv2.imread(image_path)

    if frame_bgr is None:
        print(f"❌ Error: Could not read the image file at '{image_path}'.")
        print("Creating a dummy 800x600 image for demonstration.")
        frame_bgr = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.rectangle(frame_bgr, (100, 100), (700, 500), (80, 50, 20), -1)
        cv2.putText(frame_bgr, "Image not found", (120, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    print(f"🖼️  Loaded frame of size: {frame_bgr.shape[1]}x{frame_bgr.shape[0]}")
    
    try:
        # Pass the new color parameters to the constructor
        tracker = FrameROIColor(frame_bgr, is_bgr=True, resize_to=(1280, 720), color_live=(0, 255, 255), color_final=(255, 0, 0))
        tracker.run()
        tracking_data = tracker.get_tracking_data()

        if tracking_data:
            print("\n--- ✅ Tracking Data Extracted (Scaled to Original Image) ---")
            
            if "freehand_pixels" in tracking_data:
                pixel_df = pd.DataFrame(tracking_data["freehand_pixels"])
                print(f"\nDataFrame with {len(pixel_df)} pixels from free-hand draw:")
                print(pixel_df.head())
                del tracking_data["freehand_pixels"]

            print("\nEllipse and Color Analysis:")
            print(json.dumps(tracking_data, indent=4))

        else:
            print("\nNo data was confirmed to be saved.")

    except (ValueError, TypeError) as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()