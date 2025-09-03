import open3d as o3d
import pygame
import numpy as np
from skimage.transform import resize as imresize

class ObjectsInteractionVisualizer:
    """
    Manages the visualization of the interaction using Open3D and Pygame.
    This is the 'View' in the MVC pattern.
    """
    def __init__(self,
                 window_title: str = "Interaction Visualizer",
                 width: int = 1920,
                 height: int = 1080,
                 pygame_width: int = 500,
                 fps: int = 30):
        """
        Initializes the visualizer windows.

        Args:
            window_title (str): The title for the Open3D window.
            width (int): Width of the Open3D window.
            height (int): Height of the Open3D window.
            pygame_width (int): Width and height of the Pygame window.
            fps (int): The target frames per second for the simulation clock.
        """
        # --- Open3D Setup ---
        self.viewer = o3d.visualization.Visualizer()
        self.viewer.create_window(window_name=window_title, width=width, height=height)
        self.geometries = {} # To store references to geometries for updating
        self.fps = fps

        # --- Pygame Setup ---
        self.pygame_width = pygame_width
        pygame.init()
        self.display = pygame.display.set_mode((pygame_width, pygame_width))
        pygame.display.set_caption("2D View")
        self.clock = pygame.time.Clock()

    def add_geometry(self, name: str, geometry: o3d.geometry.Geometry):
        """
        Adds a geometry in the Open3D scene.
        Args:
            name (str): A unique name for the geometry.
            geometry (o3d.geometry.Geometry): The geometry object to add.
        """
        # Add the new geometry to the viewer and store its reference in our dictionary.
        self.geometries[name] = geometry
        self.viewer.add_geometry(geometry, reset_bounding_box=False)
    
    def is_existing_geometry(self, name: str):
        return name in self.geometries

    def update_geometry(self, name:str, geometry: o3d.geometry.Geometry):
        # If it exists, remove the old geometry from the viewer.
        # Using reset_bounding_box=False prevents the camera from auto-resizing.
        if self.is_existing_geometry(name):
            self.viewer.remove_geometry(self.geometries[name], reset_bounding_box=False)
        self.add_geometry(name, geometry)

    def update(self, frame_data: dict):
        """
        Updates the scene with data from a single processed frame.

        Args:
            frame_data (dict): A dictionary containing visualization data for the frame.
                Expected keys: 'transformed_hand_mesh', 'contact_points', 'video_frame' (optional).
        """
        # --- Update Open3D Geometries ---
        hand_mesh = frame_data.get('transformed_hand_mesh')
        if hand_mesh and 'hand' in self.geometries:
            hand_vis = self.geometries['hand']
            hand_vis.vertices = hand_mesh.vertices
            hand_vis.triangles = hand_mesh.triangles # Ensure topology is also updated
            hand_vis.compute_vertex_normals()
            self.viewer.update_geometry(hand_vis)

        # Handle contact points visualization, including the case of an empty array.
        contact_points = frame_data.get('contact_points')
        if 'contacts' in self.geometries:
            contacts_vis = self.geometries['contacts']
            # Check if contact_points is a valid, non-empty numpy array.
            if contact_points is not None and contact_points.size > 0:
                contacts_vis.points = o3d.utility.Vector3dVector(contact_points)
            else:
                # If contact_points is None or an empty array, clear the points.
                # This prevents displaying stale contacts from the previous frame.
                contacts_vis.points = o3d.utility.Vector3dVector()
            self.viewer.update_geometry(contacts_vis)

        # Poll events to keep the window responsive
        self.viewer.poll_events()
        self.viewer.update_renderer()

        # --- Update Pygame Display (if video frame is provided) ---
        video_frame = frame_data.get('video_frame')
        if video_frame is not None:
            # This processing mimics the user's snippet
            frame_show = np.flip(video_frame, -1).copy() # BGR to RGB
            resized_frame = imresize(frame_show, (self.pygame_width, self.pygame_width), preserve_range=True).astype(np.uint8)
            # Pygame requires a specific transpose for its surface format
            surface = pygame.surfarray.make_surface(np.transpose(resized_frame, (1, 0, 2)))
            self.display.blit(surface, (0, 0))
        
        pygame.display.update()
        self.clock.tick(self.fps)

    def close(self):
        """Closes all visualization windows cleanly."""
        print("Closing visualizer windows...")
        self.viewer.destroy_window()
        pygame.quit()