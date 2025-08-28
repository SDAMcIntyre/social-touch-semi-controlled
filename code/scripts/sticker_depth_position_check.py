from pathlib import Path

from pydantic import BaseModel

from _3_preprocessing._1_sticker_tracking import extract_stickers_xyz_positions
import utils.path_tools as path_tools

class StickersConfig(BaseModel):
    """Configuration model for sticker processing paths"""
    tiff_folder: Path
    stickers_roi_csv_path: Path
    result_csv_path: Path
    result_md_path: Path
    result_video_path: Path

def load_config(session_id: str, block_order: str) -> StickersConfig:
    """Load configuration for a specific session and block order"""
    project_data_root = path_tools.get_project_data_root()
    
    # The project_data_root already points to the semi-controlled folder
    base_path = project_data_root
    
    # Find the actual session folder by scanning for folders ending with the session_id
    kinect_processed_path = base_path / "2_processed/kinect"
    full_session_id = None
    
    if kinect_processed_path.exists():
        for folder in kinect_processed_path.iterdir():
            if folder.is_dir() and folder.name.endswith(f"_{session_id}"):
                full_session_id = folder.name
                break
    
    if not full_session_id:
        raise FileNotFoundError(f"Could not find session folder for {session_id} in {kinect_processed_path}")
    
    # Extract session date from the full session ID
    session_date = full_session_id.replace(f"_{session_id}", "")
    
    # Convert block_order to integer for formatting
    block_order_int = int(block_order)
    
    config = {
        "tiff_folder": base_path / "1_primary/kinect" / full_session_id / f"block-order-{block_order}" / 
                       f"{session_date}_{session_id}_semicontrolled_block-order{block_order_int:02d}_kinect_depth",
        
        "stickers_roi_csv_path": base_path / "2_processed/kinect" / full_session_id / f"block-order-{block_order}/handstickers" /
                                f"{session_date}_{session_id}_semicontrolled_block-order{block_order_int:02d}_kinect_handstickers_roi_tracking.csv",
        
        "result_csv_path": base_path / "2_processed/kinect" / full_session_id / f"block-order-{block_order}/handstickers" /
                          f"{session_date}_{session_id}_semicontrolled_block-order{block_order_int:02d}_kinect_xyz_tracked.csv",
        
        "result_md_path": base_path / "2_processed/kinect" / full_session_id / f"block-order-{block_order}/handstickers" /
                         f"{session_date}_{session_id}_semicontrolled_block-order{block_order_int:02d}_kinect_xyz_tracked_metadata.json",
        
        "result_video_path": base_path / "2_processed/kinect" / full_session_id / f"block-order-{block_order}/handstickers" /
                           f"{session_date}_{session_id}_semicontrolled_block-order{block_order_int:02d}_kinect_xyz_tracked.mp4"
    }
    
    return StickersConfig(**config)

def process_stickers(config: StickersConfig, monitor: bool = True) -> None:
    """Process stickers using the provided configuration"""
    extract_stickers_xyz_positions(
        source=config.tiff_folder,
        input_csv_path=config.stickers_roi_csv_path,
        output_csv_path=config.result_csv_path,
        metadata_path=config.result_md_path,
        monitor=monitor,
        video_path=config.result_video_path,
        input_type='tiff'
    )

def main():
    # Example usage
    session_id = "ST14-01"
    block_order = "01"
    
    try:
        config = load_config(session_id, block_order)
        print(f"Processing session {session_id}, block {block_order}")
        process_stickers(config)
        print("✅ Processing completed successfully")
    except Exception as e:
        print(f"❌ Error processing stickers: {str(e)}")

if __name__ == "__main__":
    main()
