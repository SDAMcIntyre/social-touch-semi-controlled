import os
import sys
import cv2
import numpy as np
import yaml
from pathlib import Path

from utils.should_process_task import should_process_task, refresh_output_mtimes
from preprocessing.common import (
    KinectMKV,
    KinectFrame,
    PointCloudDataHandler
)

from preprocessing.forearm_extraction import (
    ForearmFrameParametersFileHandler,
    ForearmParameters,
    RegionOfInterest,

    ArmSegmentation,

    ForearmSegmentationParamsFileHandler
)
from preprocessing.forearm_extraction.depth_averaging import FrameDepthAverager


DEFAULT_CONFIG_PATH = 'config.yaml'

# -----------------------------------------------------------------
# 1. Custom Exceptions for Better Error Handling
# -----------------------------------------------------------------
class ConfigError(Exception):
    """Exception raised for errors in the config file."""
    pass

# -----------------------------------------------------------------
# 3. Helper Functions (Separation of Concerns)
# -----------------------------------------------------------------
def load_config(config_path: str) -> dict:
    """Loads the YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise ConfigError(f"Configuration file not found at: {config_path}")
    except yaml.YAMLError as e:
        raise ConfigError(f"Error parsing YAML configuration: {e}")

def get_corners_from_roi(roi: RegionOfInterest) -> np.ndarray:
    p1 = (roi.top_left_corner.x, roi.top_left_corner.y)
    p2 = (roi.bottom_right_corner.x, roi.bottom_right_corner.y)
    return np.array([p1, p2])

def get_3d_cuboid_from_roi(frame: KinectFrame, roi: RegionOfInterest) -> np.ndarray:
    """Converts the 2D ROI into 3D corner points for the box filter."""
    top_left_corner, bottom_right_corner = get_corners_from_roi(roi)
    p1 = frame.convert_xy_to_xyz(top_left_corner)
    p2 = frame.convert_xy_to_xyz(bottom_right_corner)
    return np.array([p1, p2])

def show_annotated_frames(
        roi: RegionOfInterest, 
        frame: KinectFrame):
    """Displays the depth and color frames with the ROI rectangle."""
    pt1, pt2 = get_corners_from_roi(roi)
    
    depth_mat = frame.get_depth_for_viewing()
    cv2.rectangle(depth_mat, pt1, pt2, (0, 0, 255), 2)
    cv2.imshow("Kinect Depth Frame with ROI", depth_mat)

    color_mat = frame.color
    cv2.rectangle(color_mat, pt1, pt2, (0, 0, 255), 2)
    cv2.imshow("Kinect Color Frame with ROI", color_mat)
    
    print("Press any key in an image window to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# -----------------------------------------------------------------
# 4. Main Orchestrator (uses Dependency Injection)
# -----------------------------------------------------------------
def extract_forearm(
        video_path: str,
        video_config: ForearmParameters,
        output_ply_path: str,
        output_params_path: str,
        *,
        monitor: str = False,
        interactive: str = False,
        force_processing: bool = False,
):
    """
    Orchestrates the entire processing pipeline for a single file.
    
    Args:
        config (dict): The loaded configuration dictionary.
    """
    
    if not should_process_task(
        output_paths=[output_ply_path, output_params_path],
        input_paths=[video_path],
        force=force_processing,
    ):
        return output_ply_path

    # Load configuration
    if os.path.exists(output_params_path):
        segmentation_params = ForearmSegmentationParamsFileHandler.load(output_params_path)
    else:
        config = load_config(os.path.join(os.path.dirname(__file__), DEFAULT_CONFIG_PATH))
        segmentation_params = config['segmentation_params']

    try:
        # 1. Load Data
        print(f"--- Processing {os.path.basename(video_path)} ---")
        
        # 2. Setup Dependencies
        # Dependencies are created here and "injected" into the functions that need them.
        with KinectMKV(video_path) as mkv:
            # Always load the representative frame for ROI cuboid and monitoring.
            frame: KinectFrame = mkv[video_config.frame_id]

            if video_config.is_averaged:
                print(
                    f"   Averaging {len(video_config.frame_ids)} frames "
                    f"(representative: {video_config.representative_frame_id})..."
                )
                point_cloud = FrameDepthAverager.average(
                    mkv,
                    video_config.frame_ids,
                )
            else:
                point_cloud = frame.generate_o3d_point_cloud()

            # Record whether outputs already exist before running the segmenter.
            # Used below to skip re-saving when the operator closes the GUI
            # without applying any changes (interactive mode only).
            outputs_existed = all(
                os.path.exists(p) for p in [output_ply_path, output_params_path]
            )

            segmenter = ArmSegmentation(segmentation_params, interactive=interactive)

            cuboid_oppposed_corners = get_3d_cuboid_from_roi(frame, video_config.region_of_interest)
            pcd = segmenter.preprocess(
                point_cloud,
                cuboid_oppposed_corners,
                monitor #  config['visualization']['show_intermediate_steps']
            )

            pcd = segmenter.extract_arm(
                pcd,
                monitor #  config['visualization']['show_intermediate_steps']
            )

            # 4. Finalize
            if monitor: #  config['visualization']['show_intermediate_steps']
                show_annotated_frames(video_config.region_of_interest, frame)

            if outputs_existed and not segmenter.was_modified:
                # The operator closed the GUI without editing — outputs are still
                # valid. Touch their mtimes so the pipeline considers them fresh.
                refresh_output_mtimes([output_ply_path, output_params_path])
            else:
                PointCloudDataHandler.save(pcd, output_path=output_ply_path)
                # save the parameters
                ForearmSegmentationParamsFileHandler.save(segmenter.params, output_params_path)
            
        return output_ply_path

    except (ConfigError) as e:
        print(f"❌ ERROR: A pipeline failure occurred: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)




if __name__ == "__main__":
    from preprocessing.forearm_extraction.models.forearm_parameters import Point

    database_path  = r"F:/liu-onedrive-nospecial-carac/_Teams/Social touch Kinect MNG/02_data/semi-controlled/"
    video_block = "kinect/2022-06-17_ST16-05/"

    # ── Configuration ────────────────────────────────────────────
    # Option A: Load from an existing metadata JSON (recommended)
    metadata_json = Path(database_path + "2_processed/" + video_block + "forearm_pointclouds/2022-06-17_ST16-05_arm_roi_metadata.json")
    # Option B: Set to None to use manual parameters below
    # metadata_json = None

    session_primary_dir = Path(database_path + "1_primary/" + video_block + "block-order-01")
    output_dir = Path(database_path + "2_processed/" + video_block + "forearm_pointclouds/")

    # ── Resolve video_config ─────────────────────────────────────
    if metadata_json is not None and metadata_json.exists():
        all_params = ForearmFrameParametersFileHandler.load(str(metadata_json))
        if not all_params:
            raise RuntimeError(f"No parameters found in {metadata_json}")
        video_config = all_params[0]  # pick the first entry (change index to debug others)
        print(f"Loaded config: {video_config.video_filename}, frame {video_config.frame_id}")
    else:
        # Manual fallback — fill in real values for your session
        video_config = ForearmParameters(
            video_filename="depth_video.mp4",
            frame_ids=[542],
            representative_frame_id=542,
            region_of_interest=RegionOfInterest(
                top_left_corner=Point(x=400, y=300),
                bottom_right_corner=Point(x=1200, y=900),
                angle_deg=0.0,
            ),
            frame_width=1920,
            frame_height=1080,
            fps=30.0,
            nframes=6000,
            fourcc_str="mp4v",
        )

    # ── Resolve paths ────────────────────────────────────────────
    video_path = session_primary_dir / video_config.video_filename.replace(".mp4", ".mkv")
    output_stem = video_config.build_output_stem(video_path.stem)
    output_ply_path = output_dir / f"{output_stem}.ply"
    output_params_path = output_dir / f"{output_stem}_extraction_params.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Video:  {video_path}")
    print(f"Output: {output_ply_path}")

    # ── Run ──────────────────────────────────────────────────────
    extract_forearm(
        video_path=str(video_path),
        video_config=video_config,
        output_ply_path=str(output_ply_path),
        output_params_path=str(output_params_path),
        monitor=True,
        interactive=True,
        force_processing=True,
    )

