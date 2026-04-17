from pathlib import Path

from preprocessing.common import VideoMP4Manager
from preprocessing.stickers_analysis import (
    ConsolidatedTracksFileHandler,
    DepthAggregationDiagnosticsGUI,
)


def view_xyz_depth_aggregation(
    xy_csv_path: Path,
    mkv_path: Path,
    rgb_video_path: Path,
    sticker_diameter_mm: float = 10.0,
    depth_weight_sigma: float = 0.3,
) -> None:
    """Launch the depth-aggregation diagnostics viewer for one session block.

    Loads the consolidated 2D tracking data (which carries ellipse parameters)
    and opens an interactive Tkinter GUI that lets you scrub through frames and
    inspect every intermediate variable of the depth-weighted XYZ aggregation
    pipeline.

    Args:
        xy_csv_path: Path to the ``*_handstickers_summary_2d_coordinates.csv``
            produced by the consolidation step.
        mkv_path: Path to the raw Kinect MKV file (source of depth point
            clouds).
        rgb_video_path: Path to the RGB MP4 video (used for FPS and total
            frame count).
        sticker_diameter_mm: Sticker-size guard threshold forwarded to the
            extractor diagnostics (must match the extractor configuration).
        depth_weight_sigma: Exponential-decay sigma forwarded to the extractor
            diagnostics (must match the extractor configuration).
    """
    print("Loading tracking data...")
    try:
        tracks_manager = ConsolidatedTracksFileHandler.load(xy_csv_path)
        print(f"Loaded stickers: {tracks_manager.object_names}")
    except Exception as exc:
        print(f"Failed to load tracking data: {exc}")
        return

    print("Loading video manager...")
    try:
        video_manager = VideoMP4Manager(rgb_video_path)
        print(f"Loaded video: {rgb_video_path.name}  ({len(video_manager)} frames @ {video_manager.fps:.1f} fps)")
    except Exception as exc:
        print(f"Failed to load video: {exc}")
        return

    gui = DepthAggregationDiagnosticsGUI(
        video_manager=video_manager,
        mkv_path=mkv_path,
        tracks_manager=tracks_manager,
        sticker_diameter_mm=sticker_diameter_mm,
        depth_weight_sigma=depth_weight_sigma,
        title=f"Depth Aggregation Diagnostics — {rgb_video_path.name}",
        windowState="maximized",
    )
    gui.start()


if __name__ == "__main__":
    # --- Example Usage ---
    # Replace with the actual paths to your files.
    _xy_csv = Path(
        "F:/liu-onedrive-nospecial-carac/_Teams/Social touch Kinect MNG/data/"
        "semi-controlled/2_processed/kinect/2022-06-15_ST14-01/block-order-07/"
        "handstickers/2022-06-15_ST14-01_semicontrolled_block-order07_kinect"
        "_handstickers_summary_2d_coordinates.csv"
    )
    _mkv = Path(
        "F:/liu-onedrive-nospecial-carac/_Teams/Social touch Kinect MNG/data/"
        "semi-controlled/1_primary/kinect/2022-06-15_ST14-01/block-order-07/"
        "2022-06-15_ST14-01_semicontrolled_block-order07_kinect.mkv"
    )
    _rgb = Path(
        "F:/liu-onedrive-nospecial-carac/_Teams/Social touch Kinect MNG/data/"
        "semi-controlled/2_processed/kinect/2022-06-15_ST14-01/block-order-07/"
        "2022-06-15_ST14-01_semicontrolled_block-order07_kinect.mp4"
    )

    if not _xy_csv.exists() or not _mkv.exists() or not _rgb.exists():
        print("=" * 60)
        print("!! UPDATE THE FILE PATHS IN THE __main__ BLOCK !!")
        print(f"CSV:   {'OK' if _xy_csv.exists() else 'NOT FOUND: ' + str(_xy_csv)}")
        print(f"MKV:   {'OK' if _mkv.exists() else 'NOT FOUND: ' + str(_mkv)}")
        print(f"Video: {'OK' if _rgb.exists() else 'NOT FOUND: ' + str(_rgb)}")
        print("=" * 60)
    else:
        view_xyz_depth_aggregation(
            xy_csv_path=_xy_csv,
            mkv_path=_mkv,
            rgb_video_path=_rgb,
        )
