import open3d as o3d
import numpy as np
import os
import tkinter as tk
from tkinter import ttk
import cv2
import imageio
from typing import List, Union
from tqdm import tqdm

# Define a mapping for common color names to RGB float values (0.0 to 1.0)
_COLOR_MAP = {
    "red": [1.0, 0.0, 0.0],
    "green": [0.0, 1.0, 0.0],
    "blue": [0.0, 0.0, 1.0],
    "white": [1.0, 1.0, 1.0],
    "black": [0.0, 0.0, 0.0],
    "yellow": [1.0, 1.0, 0.0],
    "cyan": [0.0, 1.0, 1.0],
    "magenta": [1.0, 0.0, 1.0],
    "orange": [1.0, 0.5, 0.0],
    "purple": [0.5, 0.0, 0.5],
    "grey": [0.5, 0.5, 0.5],
    "gray": [0.5, 0.5, 0.5],
}

def _hex_to_rgb_float(hex_color: str) -> List[float]:
    """Converts a hex color string (e.g., '#FF0000') to RGB float values."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        return [int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
    elif len(hex_color) == 8: # Handle ARGB or RGBA if needed, but only RGB is returned
        return [int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
    return [0.0, 0.0, 0.0] # Default to black if invalid

def _convert_colors_to_rgb_array(colors_input: np.ndarray) -> np.ndarray:
    """
    Converts an input NumPy array of color strings or RGB floats to a 
    (num_points, 3) NumPy array of RGB float values (0.0-1.0).
    
    Args:
        colors_input (np.ndarray): Can be:
            - (N,) array of color strings (e.g., "red", "#FF0000", "0.5,0.2,0.8")
            - (N, 3) array of RGB float values (already 0.0-1.0)
    
    Returns:
        np.ndarray: A (N, 3) NumPy array of RGB float values.
    """
    if colors_input.ndim == 2 and colors_input.shape[1] == 3 and colors_input.dtype == np.float32:
        # Already an (N, 3) RGB array of floats, assume values are 0.0-1.0
        return colors_input
    
    # Ensure colors_input is of string type if not already (N,3) float
    if colors_input.dtype != object and not np.issubdtype(colors_input.dtype, np.str_):
        raise TypeError("Input 'colors' array must contain string representations of colors or be a float array of shape (N, 3).")

    if colors_input.ndim != 1:
        raise ValueError("Input 'colors' array for string conversion must be 1-dimensional (N,) for color strings.")

    num_points = colors_input.shape[0]
    rgb_colors = []

    for point_idx in range(num_points):
        color_str_val = colors_input[point_idx]
        # Ensure the color string is decoded if it's a bytes type (e.g., from numpy loadtxt)
        if isinstance(color_str_val, bytes):
            color_str = color_str_val.decode('utf-8')
        else:
            color_str = str(color_str_val)

        if color_str.lower() in _COLOR_MAP:
            rgb_colors.append(_COLOR_MAP[color_str.lower()])
        elif color_str.startswith("#"):
            rgb_colors.append(_hex_to_rgb_float(color_str))
        else:
            # Try to interpret as comma-separated RGB floats (e.g., "1.0,0.5,0.0")
            try:
                rgb = [float(c.strip()) for c in color_str.split(',')]
                if len(rgb) == 3 and all(0.0 <= x <= 1.0 for x in rgb):
                    rgb_colors.append(rgb)
                else:
                    print(f"Warning: Invalid RGB float string format '{color_str}' for point {point_idx}. Using default red.")
                    rgb_colors.append(_COLOR_MAP["red"])
            except ValueError:
                print(f"Warning: Unrecognized color string '{color_str}' for point {point_idx}. Using default red.")
                rgb_colors.append(_COLOR_MAP["red"])
    
    return np.array(rgb_colors, dtype=np.float32)

class StickersVideoGenerator:
    """
    Generates a synchronized video of 2D images and a 3D time series.

    This class creates a video where the left side displays a sequence of images
    (from a video file or a pre-loaded array) and the right side displays an
    animated 3D point cloud, synchronized frame by frame. The final video is
    sized to the user's screen resolution, split into two halves.
    """

    def __init__(self, time_series_xyz: np.ndarray, 
                 images: Union[List[np.ndarray], str], 
                 reference_pcd: o3d.geometry.PointCloud,
                 colors: Union[np.ndarray, List[str]], 
                 point_size: float = 10.0, 
                 save_path: str = "output_video.mp4", fps: int = 30):
        """
        Initializes the video generator.

        Args:
            time_series_xyz (np.ndarray): A time series NumPy array for the 3D points.
                                          Shape: (num_frames, num_points, 3)
            images (Union[List[np.ndarray], str]): A list of images or a path to an MP4 video.
            reference_pcd (o3d.geometry.PointCloud): A static reference point cloud.
            colors (Union[np.ndarray, List[str]]): A NumPy array or Python list representing colors for each point.
                                          If np.ndarray: Shape (num_points,) where each element
                                          is a color string (e.g., "red", "#FF0000", "0.5,0.2,0.8").
                                          Alternatively, can be (num_points, 3) for
                                          pre-converted RGB float values (0.0-1.0).
                                          If List[str]: A list of color strings.
                                          These colors will be static throughout the video.
            point_size (float, optional): The size of the points displayed in the 3D visualization.
                                          Defaults to 5.0. This size will apply to the
                                          animated points (spheres), while the reference point cloud
                                          will retain Open3D's default point rendering.
            save_path (str, optional): Path to save the output video.
            fps (int, optional): Frames per second for the output video.
        """
        if not isinstance(time_series_xyz, np.ndarray) or time_series_xyz.ndim != 3 or time_series_xyz.shape[2] != 3:
            raise ValueError("time_series_xyz must be a NumPy array of shape (num_frames, num_points, 3).")
        
        # Convert list to numpy array if colors is a list
        if isinstance(colors, list):
            colors_array = np.array(colors, dtype=object) # Use object dtype for string lists
        elif isinstance(colors, np.ndarray):
            colors_array = colors
        else:
            raise TypeError("colors must be a NumPy array or a Python list of strings.")
        
        # Validate shape for colors_array: (N,) for strings or (N, 3) for RGB floats
        if not (colors_array.ndim == 1 or (colors_array.ndim == 2 and colors_array.shape[1] == 3)):
            raise ValueError("colors must be a 1D NumPy array of strings (num_points,) or a 2D NumPy array of RGB floats (num_points, 3).")

        if not isinstance(point_size, (int, float)) or point_size <= 0:
            raise ValueError("point_size must be a positive float or integer.")

        self.time_series_xyz = time_series_xyz
        self.reference_pcd = reference_pcd
        self.save_path = save_path
        self.fps = fps
        self.view_params = None # Initialize view_params to None
        self.tk_root = None
        
        # Attributes for screen resolution
        self.screen_width = None
        self.screen_height = None

        self.images_source = images
        self.is_video_path = isinstance(images, str)
        self.video_capture = None

        # Convert input colors to a standardized RGB float array of shape (num_points, 3)
        self.colors = _convert_colors_to_rgb_array(colors_array) 
        
        # point_size will be used as a radius scale for spheres
        self.point_size = float(point_size)

        num_frames_time_series = self.time_series_xyz.shape[0]
        num_points_time_series = self.time_series_xyz.shape[1]
        
        num_points_colors = self.colors.shape[0] 

        num_frames_images = 0
        if self.is_video_path:
            print(f"Image source is a video path: {self.images_source}")
            cap = cv2.VideoCapture(self.images_source)
            if not cap.isOpened():
                raise IOError(f"Could not open video file at: {self.images_source}")
            num_frames_images = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        else:
            print("Image source is a pre-loaded frame array.")
            num_frames_images = len(self.images_source)

        # Validate matching number of frames and points across all inputs
        if False and num_frames_time_series != num_frames_images: # Original code had False, keeping it as is
            raise ValueError(
                f"The number of 3D time series frames ({num_frames_time_series}) "
                f"must match the number of image frames ({num_frames_images})."
            )
        
        # Validate that the number of points in the time series matches the number of colors
        if num_points_time_series != num_points_colors:
             raise ValueError(
                f"The number of points in the 3D time series ({num_points_time_series}) "
                f"must match the number of points in the colors array ({num_points_colors})."
            )

    def _get_screen_resolution(self):
        """Gets the screen resolution using a temporary Tkinter window."""
        if self.screen_width is None or self.screen_height is None:
            # Create a temporary root to get screen info without showing a window
            temp_root = tk.Tk()
            temp_root.withdraw() # Hide the window
            self.screen_width = temp_root.winfo_screenwidth()
            self.screen_height = temp_root.winfo_screenheight()
            temp_root.destroy()
            print(f"Detected screen resolution: {self.screen_width}x{self.screen_height}")

    def _create_temporal_mesh(self, points_xyz: np.ndarray, colors_rgb: np.ndarray) -> o3d.geometry.TriangleMesh:
        """Helper to create a combined mesh of spheres for temporal points."""
        combined_mesh = o3d.geometry.TriangleMesh()
        for j in range(points_xyz.shape[0]):
            # Create a new sphere for each point
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.point_size)
            sphere.compute_vertex_normals() # Needed for proper lighting
            sphere.translate(points_xyz[j]) # Translate to point position
            sphere.paint_uniform_color(colors_rgb[j])
            combined_mesh += sphere
        return combined_mesh

    def _capture_view_parameters_and_proceed(self):
        """Closes the Tkinter window and signals the main process to proceed."""
        if self.tk_root:
            self.tk_root.quit()
            self.tk_root.destroy()
            self.tk_root = None

    def adjust_view_parameters_interactively(self):
        """
        Opens an Open3D window for the user to adjust the camera angle.
        The captured view parameters will be stored in self.view_params
        for subsequent video generation by create_video().
        """ 
        self._get_screen_resolution()
        panel_width = self.screen_width // 2
        panel_height = self.screen_height
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Adjust 3D View (Right Half)", width=panel_width, height=panel_height)
        
        vis.add_geometry(self.reference_pcd)
        
        # Initialize temporal geometry as a mesh of spheres with first frame's data
        if self.time_series_xyz.shape[0] > 0:
            initial_points_xyz = self.time_series_xyz[0, :, :].T
            initial_colors_rgb = self.colors # static colors
            self.temporal_geometry = self._create_temporal_mesh(initial_points_xyz, initial_colors_rgb)
        else:
            self.temporal_geometry = o3d.geometry.TriangleMesh() # Empty mesh if no temporal data

        vis.add_geometry(self.temporal_geometry)
        
        # Removed vis.reset_viewpoint(True) to resolve the attribute error.
        # Open3D's Visualizer typically auto-frames the scene on initial geometry addition.

        self.tk_root = tk.Tk()
        self.tk_root.title("Controls")
        self.tk_root.geometry("300x100")
        main_frame = ttk.Frame(self.tk_root, padding="10")
        main_frame.pack(expand=True, fill='both')
        label = ttk.Label(main_frame, text="Adjust the 3D view, then click Proceed.")
        label.pack(pady=5)
        proceed_button = ttk.Button(main_frame, text="Proceed", command=self._capture_view_parameters_and_proceed)
        proceed_button.pack(pady=10, fill='x', expand=True)

        print("Please adjust the view in the 'Adjust 3D View (Right Half)' window.")
        print("Click 'Proceed' in the 'Controls' window to continue.")
        
        # Main loop for interactive adjustment
        while self.tk_root is not None:
            vis.poll_events()
            vis.update_renderer()
            try:
                self.tk_root.update()
            except tk.TclError:
                # Tkinter window was likely closed by user or system
                break
        
        view_control = vis.get_view_control()
        self.view_params = view_control.convert_to_pinhole_camera_parameters()
        vis.destroy_window()
        
        if self.view_params:
            print("View parameters captured successfully.")
        else:
            print("View adjustment cancelled or failed. Default view will be used for video generation.")



    def create_video(self, width3d=None, height3d=None):
        """
        Generates and saves the video. If adjust_view_parameters_interactively()
        was called and view parameters were captured, they will be used.
        Otherwise, a default Open3D viewpoint will be applied.
        """
        self.temporal_geometry = None
        self._get_screen_resolution() # Ensure screen resolution is obtained

        print("Starting video generation...")
        if self.is_video_path:
            self.video_capture = cv2.VideoCapture(self.images_source)
            if not self.video_capture.isOpened():
                print(f"Error: Failed to reopen video file at {self.images_source}")
                return
        
        # Define target dimensions for each panel (half screen width)
        if width3d is None or height3d is None:
            height3d = self.screen_height
            width3d = self.screen_width // 2
        
        video_height = self.screen_height
        video_width = self.screen_width * 2
        video_width_section = video_width // 2
        
        # --- Setup Visualizer for Off-screen Rendering ---
        vis = o3d.visualization.Visualizer()        
        # Set visible=False for off-screen rendering for video generation
        vis.create_window(window_name="Video Generation", width=width3d, height=height3d, visible=False)
        vis.add_geometry(self.reference_pcd)
        
        if self.view_params:
            vis.get_view_control().convert_from_pinhole_camera_parameters(self.view_params)
            print("Using previously captured view parameters.")
        else:
            print("No view parameters were captured, using default view for video generation (auto-framed by Open3D).")

        video_writer = imageio.get_writer(self.save_path, fps=self.fps, codec='libx264', quality=8)

        total_frames = len(self.time_series_xyz)
        for i in tqdm(range(total_frames), desc="Generating Video"):
            # 1. Get the corresponding 2D image
            img_2d = None
            if self.is_video_path:
                ret, frame = self.video_capture.read()
                if not ret:
                    print(f"\nWarning: Could not read frame {i+1} from video. Stopping.")
                    break
                img_2d = frame
            else:
                img_2d = self.images_source[i]

            # 2. Update the 3D scene: Recreate and update the temporal mesh
            # Remove existing temporal geometry and add new one for the current frame
            if self.temporal_geometry: 
                vis.remove_geometry(self.temporal_geometry, reset_bounding_box=False)

            current_points_xyz = self.time_series_xyz[i, :, :].T
            current_colors_rgb = self.colors # colors are static

            self.temporal_geometry = self._create_temporal_mesh(current_points_xyz, current_colors_rgb)
            vis.add_geometry(self.temporal_geometry, reset_bounding_box=False)
            
            vis.poll_events()
            vis.update_renderer()

            # 3. Capture the 3D scene as an image
            o3d_frame_np = np.asarray(vis.capture_screen_float_buffer(False))
            o3d_frame_rgb = (o3d_frame_np * 255).astype(np.uint8)
            o3d_2d_resized = cv2.resize(o3d_frame_rgb, (video_width_section, video_height))

            # 4. Resize 2D image to match the target panel dimensions
            img_2d_resized = cv2.resize(img_2d, (video_width_section, video_height))
            if len(img_2d_resized.shape) == 3 and img_2d_resized.shape[2] == 3:
                img_2d_rgb = cv2.cvtColor(img_2d_resized, cv2.COLOR_BGR2RGB)
            else:
                img_2d_rgb = cv2.cvtColor(img_2d_resized, cv2.COLOR_GRAY2RGB)

            # 5. Combine the 2D image and 3D scene side-by-side
            combined_frame = np.hstack((img_2d_rgb, o3d_2d_resized))

            # 6. Write the frame to the video file
            video_writer.append_data(combined_frame)

        # --- Cleanup ---
        video_writer.close()
        vis.destroy_window()
        if self.is_video_path and self.video_capture:
            self.video_capture.release()
        print(f"\nVideo saved successfully to {os.path.abspath(self.save_path)}")
