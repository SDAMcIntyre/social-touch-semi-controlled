
import open3d as o3d
import numpy as np

from preprocessing.common.data_access.glb_data_handler import GLBDataHandler
from preprocessing.common.data_access.pc_data_handler import PointCloudDataHandler
from preprocessing.motion_analysis.objects_interaction_controller import ObjectsInteractionController

def compute_somatosensory_characteristics(
        hand_motion_glb_path: str, 
        forearm_ply_path: str, 
        output_csv_path: str,
        *,
        monitor: bool = True,
        fps: bool = 30
) -> str:
    loader = GLBDataHandler()
    loader.load(hand_motion_glb_path)
    hand_motion_data = loader.get_data()
    if hand_motion_data:
        print(f"Successfully loaded hand motion dictionary.")

    arm_pcd = PointCloudDataHandler.load(forearm_ply_path)
    if arm_pcd:
        print(f"Successfully loaded arm point cloud with {len(arm_pcd.points)} points.")

    # 3. INITIALIZE AND RUN THE CONTROLLER
    controller_with_vis = ObjectsInteractionController(
        hand_motion_data,
        arm_pcd,
        visualize=monitor,
        fps=fps
    )
    results_df = controller_with_vis.run()
    print("Results from visualized run:")
    print(results_df.head())

    # Save the DataFrame to the CSV file
    # index=False prevents pandas from writing the row index to the file
    results_df.to_csv(output_csv_path, index=False)

    return output_csv_path
