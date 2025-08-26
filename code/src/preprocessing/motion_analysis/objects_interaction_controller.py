import pandas as pd
import open3d as o3d

from .objects_interaction_processor import ObjectsInteractionProcessor
from .objects_interaction_visualizer import ObjectsInteractionVisualizer


class ObjectsInteractionController:
    """
    Manages the simulation, coordinating the ObjectsInteractionProcessor (Model)
    and InteractionVisualizer (View). This is the 'Controller'.
    """
    def __init__(self,
                 hand_motion_data: dict,
                 reference_pcd: o3d.geometry.PointCloud,
                 *,
                 visualize: bool = True,
                 fps: int = 30
    ):
        
        self.processor = ObjectsInteractionProcessor(
            reference_pcd=reference_pcd,
            base_vertices=hand_motion_data['vertices'],
            fps=fps
        )
        self.trajectory_data = {
            'time_points': hand_motion_data['time_points'], 
            'translations': hand_motion_data['translations'], 
            'rotations': hand_motion_data['rotations']
        }
        self.visualize = visualize
        
        # Controller now owns the domain-specific data

        self.visualizer = None
        if self.visualize:
            hand_color = [228/255, 178/255, 148/255]
            base_mesh = o3d.geometry.TriangleMesh()
            base_mesh.vertices = o3d.utility.Vector3dVector(hand_motion_data['vertices'])
            base_mesh.triangles = o3d.utility.Vector3iVector(hand_motion_data['faces'])
            base_mesh.paint_uniform_color(hand_color)
            base_mesh.compute_vertex_normals()
            self.base_triangles = base_mesh.triangles
            
            self.visualizer = ObjectsInteractionVisualizer()
            self.visualizer.add_geometry('arm', self.processor.ref_pcd)
            self.visualizer.add_geometry('hand', base_mesh)
            contact_pcd = o3d.geometry.PointCloud()
            contact_pcd.paint_uniform_color([1.0, 0.0, 0.0]) # Red
            self.visualizer.add_geometry('contacts', contact_pcd)

    def run(self) -> pd.DataFrame:
        results = []
        num_frames = len(self.trajectory_data['time_points'])
        print(f"Starting simulation for {num_frames} frames...")

        for i, time in enumerate(self.trajectory_data['time_points']):
            frame_result = self.processor.process_single_frame(
                translation=self.trajectory_data['translations'][i],
                rotation_quat=self.trajectory_data['rotations'][i]
            )

            if self.visualize:
                vis_data = frame_result["visualization"]
                # Controller creates the mesh for the visualizer
                transformed_mesh = o3d.geometry.TriangleMesh(
                    o3d.utility.Vector3dVector(vis_data["transformed_vertices"]),
                    self.base_triangles
                )
                vis_data["transformed_hand_mesh"] = transformed_mesh
                self.visualizer.update(vis_data)

            metrics = frame_result["metrics"]
            metrics["time"] = time
            metrics["frame_index"] = i
            results.append(metrics)
            
            if i > 0 and i % self.processor.fps == 0:
                print(f"  Processed frame {i}/{num_frames}")

        if self.visualize:
            self.visualizer.close()
            
        print("Simulation finished.")
        return pd.DataFrame(results)