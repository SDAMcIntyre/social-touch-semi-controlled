import sys
from pathlib import Path
from typing import Iterable, List, Dict
from PyQt5.QtWidgets import QApplication
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from preprocessing.common import (
    KinectMKV,
    KinectPointCloudView,

    SceneViewer,

    LazyPointCloudSequence,
    PersistentOpen3DPointCloudSequence,
    Trajectory,

    SceneViewerVideoMaker
)
from preprocessing.stickers_analysis import (
    XYZDataFileHandler
)

from preprocessing.forearm_extraction import (
    ForearmFrameParametersFileHandler,
    ForearmParameters,
    ForearmCatalog,
    get_forearms_with_fallback
)


@dataclass
class RigidityMetrics:
    """
    Data Transfer Object for shape consistency metrics.
    """
    is_consistent: bool
    max_std_dev: float
    max_coeff_variation: float
    edge_stats: Dict[str, Dict[str, float]]
    valid_frame_count: int

class RigidityAnalyzer:
    """
    Analyzes the geometric consistency of point constellations over time.
    """

    def __init__(self, abs_tolerance: float = 1.0, rel_tolerance: float = 0.05):
        """
        Args:
            abs_tolerance (float): Maximum allowed standard deviation for edge lengths (in units).
            rel_tolerance (float): Maximum allowed coefficient of variation (sigma/mu).
        """
        self.abs_tol = abs_tolerance
        self.rel_tol = rel_tolerance

    def validate_triangle_consistency(self, data: np.ndarray, plot_variations: bool = False) -> RigidityMetrics:
        """
        Computes the rigidity of the triangle formed by 3 points over N frames.

        Args:
            data (np.ndarray): Array of shape (N, 3, 3) representing N frames of 3 XYZ points.
            plot_variations (bool): If True, generates a plot of edge deviations from their mean.

        Returns:
            RigidityMetrics: Object containing pass/fail status and statistical details.
        """
        # Input validation
        if data.ndim != 3 or data.shape[1] != 3 or data.shape[2] != 3:
            raise ValueError(f"Input must be of shape (N, 3, 3). Received {data.shape}")

        # 1. Vectorized Edge Calculation
        # Points are P0, P1, P2.
        # Vectors: v01 = P1 - P0, v12 = P2 - P1, v20 = P0 - P2
        p0 = data[:, 0, :]
        p1 = data[:, 1, :]
        p2 = data[:, 2, :]

        # Calculate Euclidean distances (L2 norm) across the last axis (XYZ)
        # Resulting shapes: (N,)
        d01 = np.linalg.norm(p1 - p0, axis=1)
        d12 = np.linalg.norm(p2 - p1, axis=1)
        d20 = np.linalg.norm(p0 - p2, axis=1)

        # Stack into (N, 3) matrix where columns are the 3 edges
        edges = np.stack([d01, d12, d20], axis=1)

        # 2. NaN Handling
        # Identify frames where ANY edge calculation resulted in NaN (due to input NaNs)
        # mask is True if the row contains valid data
        mask = ~np.isnan(edges).any(axis=1)
        valid_edges = edges[mask]
        valid_count = valid_edges.shape[0]

        if valid_count < 2:
            # Insufficient data to calculate variance
            return RigidityMetrics(
                is_consistent=False,
                max_std_dev=float('inf'),
                max_coeff_variation=float('inf'),
                edge_stats={},
                valid_frame_count=valid_count
            )

        # 3. Statistical Analysis
        # Calculate Mean and Std Dev for each edge column
        means = np.mean(valid_edges, axis=0)
        stds = np.std(valid_edges, axis=0)

        # Avoid division by zero for Coefficient of Variation
        with np.errstate(divide='ignore', invalid='ignore'):
            cvs = stds / means
            cvs = np.nan_to_num(cvs, nan=float('inf'))

        # 4. Evaluation
        max_std = np.max(stds)
        max_cv = np.max(cvs)

        # Check thresholds
        # Rigid if standard deviation is low AND relative variation is low
        consistent = (max_std <= self.abs_tol) and (max_cv <= self.rel_tol)

        # Format stats for specific edges
        edge_names = ["P0-P1", "P1-P2", "P2-P0"]
        stats_map = {}
        for i, name in enumerate(edge_names):
            stats_map[name] = {
                "mean_length": means[i],
                "std_dev": stds[i],
                "coeff_var": cvs[i]
            }

        # 5. Visualization (Optional)
        if plot_variations:
            self._plot_edge_variations(edges, means, edge_names)

        return RigidityMetrics(
            is_consistent=consistent,
            max_std_dev=max_std,
            max_coeff_variation=max_cv,
            edge_stats=stats_map,
            valid_frame_count=valid_count
        )

    def _plot_edge_variations(self, edges: np.ndarray, means: np.ndarray, names: list[str]):
        """
        Helper method to plot edge variations from the mean.
        Uses the full 'edges' array (including NaNs) to preserve frame index.
        """
        # Broadcasting: (N, 3) - (3,) -> (N, 3)
        variations = edges - means
        
        plt.figure(figsize=(12, 6))
        colors = ['r', 'g', 'b']
        
        for i in range(3):
            # Matplotlib handles NaNs by breaking the line, which is desired for missing frames
            plt.plot(variations[:, i], 
                     label=f"{names[i]} (Mean: {means[i]:.1f}mm)", 
                     color=colors[i], 
                     alpha=0.7, 
                     linewidth=1)
            
        plt.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        plt.title("Edge Length Variation from Mean over Time")
        plt.xlabel("Frame Index")
        plt.ylabel("Deviation (mm)")
        plt.legend(loc='upper right')
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show(block=False)

def define_custom_colors(string_list: Iterable[str]) -> dict[str, str]:
    """
    Searches an iterable of strings for standard color keywords.

    Args:
        string_list: An iterable (e.g., list, dict_keys) of strings to search through.

    Returns:
        A list of unique color names found in the strings.
    """
    # Define the set of standard color keywords to search for
    STANDARD_COLORS = {
        "red", "green", "blue", "yellow", "orange", "purple", "pink",
        "black", "white", "brown", "gray", "grey", "cyan", "magenta", "violet"
    }
    
    found_colors = {}
    
    # Iterate through each string in the input list
    for item in string_list:
        # Convert the string to lowercase for case-insensitive matching
        item_lower = item.lower()
        # Check if any of the standard colors are a substring of the item
        for color in STANDARD_COLORS:
            if color in item_lower:
                found_colors[item] = color
    
    return found_colors

def view_xyz_stickers_on_depth_data(
    xyz_csv_path: Path,
    kinect_video_path: Path,
    forearm_pointcloud_dir: Path,
    forearm_metadata_path: Path,
    rgb_video_path: Path,
    *,
    record: bool = True
):
    
    try:
        stickers_df_dict = XYZDataFileHandler.load(xyz_csv_path)
    except Exception as e:
        print(f"An error occurred during XYZ stickers data loading: {e}")
        return
    custom_colors = define_custom_colors(stickers_df_dict.keys())
    stickers_xyz_dict = {
        key: df[['x_mm', 'y_mm', 'z_mm']].to_numpy()
        for key, df in stickers_df_dict.items()
    }
     
    
    coordinates_over_time = np.stack(
        [stickers_xyz_dict[key] for key in stickers_df_dict.keys()], 
        axis=1
    )
    analyzer = RigidityAnalyzer(abs_tolerance=2.0, rel_tolerance=0.05)
    # Run Analysis with Plotting enabled
    result = analyzer.validate_triangle_consistency(coordinates_over_time, plot_variations=True)

    # Output Results
    print(f"Dataset Shape: {coordinates_over_time.shape}")
    print(f"Valid Frames Processed: {result.valid_frame_count}")
    print(f"Is Consistent: {result.is_consistent}")
    print("-" * 30)
    print(f"Max Standard Deviation: {result.max_std_dev:.4f}")
    print(f"Max Coefficient of Variation: {result.max_coeff_variation:.4%}")
    print("-" * 30)
    print("Edge Statistics:")
    for edge, stats in result.edge_stats.items():
        print(f"  {edge}: Mean={stats['mean_length']:.2f}, Std={stats['std_dev']:.2f}, CV={stats['coeff_var']:.2%}")


    try:
        forearm_params: List[ForearmParameters] = ForearmFrameParametersFileHandler.load(forearm_metadata_path)
    except Exception as e:
        print(f"An error occurred during forearm data loading: {e}")
        return None
    # Collect the forearms pointclouds for the specific video
    catalog = ForearmCatalog(forearm_params, forearm_pointcloud_dir)
    forearms_dict = get_forearms_with_fallback(catalog, rgb_video_path)

    app = QApplication.instance() or QApplication(sys.argv)
    
    with KinectMKV(kinect_video_path, seek_strategy='sequential') as mkv:
        # 2. Instantiate the SceneViewer
        if record:
            viewer = SceneViewerVideoMaker()
        else:
            viewer = SceneViewer()

        point_cloud_view = KinectPointCloudView(mkv)
        # 3. Create the LazyPointCloudSequence and pass the wrapper directly.
        #    No pre-loading loop is needed!
        main_cloud = LazyPointCloudSequence(
            name="kinect_point_cloud",
            data_source=point_cloud_view 
        )
        viewer.add_object(main_cloud)

        forearms_cloud = PersistentOpen3DPointCloudSequence(name="forearms", frame_data=forearms_dict, point_size=10)
        viewer.add_object(forearms_cloud)

        # 4. Prepare the sticker trajectory data
        #    Iterate through each sticker and convert its data into the
        #    Dict[int, np.ndarray] format required by Trajectory.
        for sticker_name, all_positions in stickers_xyz_dict.items():
            # The all_positions array is (num_frames, 3). We need to create a
            # dictionary mapping each frame index to its corresponding (3,) position.
            trajectory_frame_data = {
                frame_index: position
                for frame_index, position in enumerate(all_positions)
            }
            
            # Get the color for this sticker, defaulting to a visible color like 'magenta'
            sticker_color = custom_colors.get(sticker_name, 'magenta')
            
            # Create a Trajectory SceneObject for the sticker and add it
            sticker_object = Trajectory(
                name=sticker_name, 
                frame_data=trajectory_frame_data, 
                color=sticker_color, 
                radius=4.0  # Adjust radius as needed
            )
            viewer.add_object(sticker_object)

        # 5. Show the viewer and run the application
        #    This replaces the old 'window = PointCloudViewer(...)' call.
        viewer.show()
        app.exec_()

    return # This line will not be reached, which is expected for a GUI app.


if __name__ == "__main__":
    from pathlib import Path

    # Define the file and directory paths
    xyz_csv = Path('F:/liu-onedrive-nospecial-carac/_Teams/Social touch Kinect MNG/data/semi-controlled/2_processed/kinect/2022-06-15_ST14-01/block-order-07/handstickers/2022-06-15_ST14-01_semicontrolled_block-order07_kinect_handstickers_xyz_tracked.csv')
    
    # NOTE: Inferred the metadata path from the xyz_csv_path
    xyz_md = xyz_csv.with_suffix('.json')
    
    # The rgb_video_path is full path object
    kinect_video_path = Path('F:/liu-onedrive-nospecial-carac/_Teams/Social touch Kinect MNG/data/semi-controlled/1_primary/kinect/2022-06-15_ST14-01/block-order-07/2022-06-15_ST14-01_semicontrolled_block-order07_kinect.mkv')
    
    forearm_pointcloud_dir = Path('F:/liu-onedrive-nospecial-carac/_Teams/Social touch Kinect MNG/data/semi-controlled/2_processed/kinect/2022-06-15_ST14-01/forearm_pointclouds')
    rgb_video_path = Path('2022-06-15_ST14-01_semicontrolled_block-order07_kinect.mp4')
    forearm_metadata_path = Path('F:/liu-onedrive-nospecial-carac/_Teams/Social touch Kinect MNG/data/semi-controlled/2_processed/kinect/2022-06-15_ST14-01/forearm_pointclouds/2022-06-15_ST14-01_arm_roi_metadata.json')

    # Run the review process
    view_xyz_stickers_on_depth_data(
        xyz_csv_path=xyz_csv,
        kinect_video_path=kinect_video_path,
        forearm_pointcloud_dir=forearm_pointcloud_dir,
        forearm_metadata_path=forearm_metadata_path,
        rgb_video_path=rgb_video_path
    )