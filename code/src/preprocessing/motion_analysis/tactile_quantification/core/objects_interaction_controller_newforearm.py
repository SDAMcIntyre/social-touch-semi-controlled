import pandas as pd
import open3d as o3d
from typing import Dict

from .objects_interaction_processor import ObjectsInteractionProcessor
from ..gui.objects_interaction_visualizer_newforearm import ObjectsInteractionVisualizer


class ObjectsInteractionController:
    """
    Manages the simulation, coordinating the ObjectsInteractionProcessor (Model)
    and InteractionVisualizer (View). This is the 'Controller'.
    """
    def __init__(self,
                 hand_motion_data: dict,
                 references_pcd: Dict[int, o3d.geometry.PointCloud],
                 *,
                 visualize: bool = True,
                 fps: int = 30
    ):
        
        self.processor = ObjectsInteractionProcessor(
            references_pcd=references_pcd,
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

        self.view = None
        if self.visualize:
            hand_color = [228/255, 178/255, 148/255]
            base_mesh = o3d.geometry.TriangleMesh()
            base_mesh.vertices = o3d.utility.Vector3dVector(hand_motion_data['vertices'])
            base_mesh.triangles = o3d.utility.Vector3iVector(hand_motion_data['faces'])
            base_mesh.paint_uniform_color(hand_color)
            base_mesh.compute_vertex_normals()
            self.base_triangles = base_mesh.triangles
            
            self.view = ObjectsInteractionVisualizer()
            self.view.add_geometry('hand', base_mesh)
            contact_pcd = o3d.geometry.PointCloud()
            contact_pcd.paint_uniform_color([1.0, 0.0, 0.0]) # Red
            self.view.add_geometry('contacts', contact_pcd)

    def run(self) -> pd.DataFrame:
        results = []
        num_frames = len(self.trajectory_data['time_points'])
        print(f"Starting simulation for {num_frames} frames...")

        for frame_id, time in enumerate(self.trajectory_data['time_points']):
            if self.processor.is_valid_pcd_key(frame_id):
                self.processor.set_current_pcd(frame_id)
                
            frame_result = self.processor.process_single_frame(
                translation=self.trajectory_data['translations'][frame_id],
                rotation_quat=self.trajectory_data['rotations'][frame_id]
            )

            if self.visualize:
                if self.processor.is_valid_pcd_key(frame_id):
                    self.view.update_geometry('arm', self.processor.ref_pcd)

                vis_data = frame_result["visualization"]
                # Controller creates the mesh for the visualizer
                transformed_mesh = o3d.geometry.TriangleMesh(
                    o3d.utility.Vector3dVector(vis_data["transformed_vertices"]),
                    self.base_triangles
                )
                vis_data["transformed_hand_mesh"] = transformed_mesh
                self.view.update(vis_data)

            metrics = frame_result["metrics"]
            metrics["time"] = time
            metrics["frame_index"] = frame_id
            results.append(metrics)
            
            if frame_id > 0 and frame_id % self.processor.fps == 0:
                print(f"  Processed frame {frame_id}/{num_frames}")

        if self.visualize:
            self.view.close()
            
        print("Simulation finished.")
        return pd.DataFrame(results)