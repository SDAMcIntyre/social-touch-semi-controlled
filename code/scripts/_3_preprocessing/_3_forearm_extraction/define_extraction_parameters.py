import cv2
import numpy as np
import tkinter as tk
import os
from collections import defaultdict
from typing import Optional

# Imports from the provided file structure
from preprocessing.common import (
    VideoMP4Manager,
    FrameROISquare
)

from preprocessing.forearm_extraction import (
    MultiVideoFramesSelector,
    ForearmParameters,
    RegionOfInterest,
    Point,
    ForearmFrameParametersFileHandler,
    sort_forearm_parameters_by_video_and_frame
)

# Global variable to hold the root Tkinter instance, ensuring it's a singleton.
_tk_root_instance = None


# ----------------------------------------------------------------------------
# TKINTER LIFECYCLE
# ----------------------------------------------------------------------------

def _get_or_create_tk_root() -> tk.Tk:
    """
    Returns the singleton Tkinter root, creating and hiding it on the first call.

    A single root is reused across calls to avoid re-initialisation errors.
    """
    global _tk_root_instance
    if _tk_root_instance is None:
        _tk_root_instance = tk.Tk()
        _tk_root_instance.withdraw()
    return _tk_root_instance


# ----------------------------------------------------------------------------
# VIDEO HELPERS
# ----------------------------------------------------------------------------

def _retrieve_fourcc(video_path: str) -> str:
    """
    Returns the FourCC codec string (e.g. 'mp4v') for the given video file.

    Falls back to 'unknown' if the file cannot be opened or the codec cannot
    be decoded. This is needed because VideoMP4Manager does not expose FourCC.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "unknown"

    try:
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        return "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).lower()
    except Exception:
        return "unknown"
    finally:
        cap.release()


# ----------------------------------------------------------------------------
# FRAME GROUP SELECTION (STEP 1)
# ----------------------------------------------------------------------------

def _build_groups_by_video_from_saved_parameters(
    parameters_list: list[ForearmParameters],
) -> dict:
    """
    Reconstructs the ``groups_by_video`` dict expected by MultiVideoFramesSelector
    from a previously saved list of ForearmParameters.

    This allows the selector UI to pre-populate groups from a prior session.
    """
    groups_by_video = defaultdict(lambda: {"groups": [], "representatives": []})
    for params in parameters_list:
        groups_by_video[params.video_filename]["groups"].append(params.frame_ids)
        groups_by_video[params.video_filename]["representatives"].append(
            params.representative_frame_id
        )
    return groups_by_video


def _open_frame_group_selector(
    rgb_video_paths: list[str],
    prefilled_groups: Optional[dict] = None,
) -> MultiVideoFramesSelector:
    """
    Opens the interactive frame-group selector window and blocks until the user
    closes it.

    Args:
        rgb_video_paths: Paths to all RGB video files to display.
        prefilled_groups: Optional pre-existing group data to populate the UI
            (output of ``_build_groups_by_video_from_saved_parameters``).

    Returns:
        The selector widget, from which ``validated`` and ``all_selected_groups``
        can be read.
    """
    root = _get_or_create_tk_root()
    app_window = tk.Toplevel(root)
    app_window.title("Multi-Video Frame Group Selector")

    if prefilled_groups:
        selector = MultiVideoFramesSelector(app_window, rgb_video_paths, prefilled_groups)
    else:
        selector = MultiVideoFramesSelector(app_window, rgb_video_paths)

    root.wait_window(app_window)
    return selector


def _log_selected_groups(selected_groups: dict) -> None:
    """Prints a summary of selected frame groups to stdout."""
    print("\n--- Frame Groups Selected for Processing ---")
    for video_path, group_data in selected_groups.items():
        n = len(group_data["groups"])
        print(f"  - {os.path.basename(video_path)}: {n} group(s)")
        for i, (group, rep) in enumerate(
            zip(group_data["groups"], group_data["representatives"])
        ):
            print(f"      Group {i + 1}: {group} (representative: {rep})")


# ----------------------------------------------------------------------------
# ROI SELECTION (STEP 2)
# ----------------------------------------------------------------------------

def _find_existing_roi_for_group(
    parameters_list: list[ForearmParameters],
    video_filename: str,
    representative_frame_id: int,
) -> Optional[dict]:
    """
    Searches a list of saved ForearmParameters for a previously defined ROI
    that matches the given video and representative frame.

    Returns a ``predefined_roi`` dict compatible with FrameROISquare, or None
    if no match is found.
    """
    matching = next(
        (
            p for p in parameters_list
            if p.video_filename == video_filename
            and p.representative_frame_id == representative_frame_id
        ),
        None,
    )

    if matching is None:
        return None

    roi = matching.region_of_interest
    return {
        "x": roi.top_left_corner.x,
        "y": roi.top_left_corner.y,
        "width":  roi.bottom_right_corner.x - roi.top_left_corner.x,
        "height": roi.bottom_right_corner.y - roi.top_left_corner.y,
    }


def _build_roi_window_title(video_filename: str, representative: int, group_size: int) -> str:
    """Returns a descriptive window title for the ROI selection dialog."""
    if group_size > 1:
        return (
            f"ROI for {video_filename} — rep. frame {representative:04d} "
            f"(group of {group_size} frames)"
        )
    return f"ROI for {video_filename} — frame {representative:04d}"


def _select_roi_for_group(
    frame: np.ndarray,
    video_filename: str,
    representative: int,
    group_size: int,
    predefined_roi: Optional[dict],
) -> Optional[dict]:
    """
    Displays ``frame`` in an interactive ROI selection window and returns the
    user's selection.

    Args:
        frame: The representative video frame (RGB numpy array) to annotate.
        video_filename: Used only for the window title.
        representative: Representative frame index, used for the window title.
        group_size: Number of frames in the group; affects the window title.
        predefined_roi: Optional prior ROI to pre-draw in the UI.

    Returns:
        A dict with keys ``x``, ``y``, ``width``, ``height``, or None if the
        user cancelled the selection.
    """
    window_title = _build_roi_window_title(video_filename, representative, group_size)
    roi_ui = FrameROISquare(
        frame,
        is_rgb=True,
        window_title=window_title,
        predefined_roi=predefined_roi,
    )
    roi_ui.run()
    return roi_ui.get_roi_data()


def _build_forearm_parameters(
    roi_data: dict,
    video_filename: str,
    group: list[int],
    representative: int,
    video_manager: VideoMP4Manager,
    fourcc_str: str,
) -> ForearmParameters:
    """
    Constructs a ForearmParameters object from raw ROI data and video metadata.
    """
    x, y, w, h = roi_data["x"], roi_data["y"], roi_data["width"], roi_data["height"]
    roi = RegionOfInterest(
        top_left_corner=Point(x=x, y=y),
        bottom_right_corner=Point(x=x + w, y=y + h),
    )
    return ForearmParameters(
        video_filename=video_filename,
        frame_ids=group,
        representative_frame_id=representative,
        region_of_interest=roi,
        frame_width=video_manager.width,
        frame_height=video_manager.height,
        fps=video_manager.fps,
        nframes=video_manager.total_frames,
        fourcc_str=fourcc_str,
    )


def _collect_parameters_for_video(
    video_path: str,
    groups: list[list[int]],
    representatives: list[int],
    saved_parameters: Optional[list[ForearmParameters]],
) -> list[ForearmParameters]:
    """
    Iterates over each frame group for a single video and collects the user's
    ROI selection for each one.

    For each group:
      - The representative frame is rendered.
      - Any previously saved ROI is pre-filled in the UI.
      - The resulting ROI is converted into a ForearmParameters object.

    Groups whose ROI selection is cancelled by the user are skipped.

    Args:
        video_path: Absolute path to the RGB video file.
        groups: List of frame-index lists, one per group.
        representatives: Representative frame index for each group.
        saved_parameters: Previously saved parameters for pre-populating ROIs.

    Returns:
        A list of ForearmParameters, one per successfully defined group.
    """
    video_filename = os.path.basename(video_path)
    print(f"\n▶️  Processing video: '{video_filename}'")

    try:
        video_manager = VideoMP4Manager(video_path)
    except FileNotFoundError as e:
        print(f"  ❌ Could not open video: {e}. Skipping.")
        return []

    fourcc_str = _retrieve_fourcc(video_path)
    collected: list[ForearmParameters] = []

    for i, (group, representative) in enumerate(zip(groups, representatives)):
        n = len(group)
        group_label = (
            f"frames {group} (representative: {representative})" if n > 1
            else f"frame {representative}"
        )
        print(f"  - Group {i + 1}: {group_label}")

        predefined_roi = (
            _find_existing_roi_for_group(saved_parameters, video_filename, representative)
            if saved_parameters else None
        )

        frame = video_manager[representative]
        roi_data = _select_roi_for_group(frame, video_filename, representative, n, predefined_roi)

        if not roi_data:
            print(f"    🟡 ROI selection cancelled — skipping group {i + 1}.")
            continue

        x, y, w, h = roi_data["x"], roi_data["y"], roi_data["width"], roi_data["height"]
        print(f"    ✅ ROI at (x={x}, y={y}), size (w={w}, h={h}).")

        params = _build_forearm_parameters(
            roi_data, video_filename, group, representative, video_manager, fourcc_str
        )
        collected.append(params)

    return collected


def _collect_all_parameters(
    selected_groups: dict,
    saved_parameters: Optional[list[ForearmParameters]],
) -> list[ForearmParameters]:
    """
    Iterates over every video and group returned by the frame selector and
    collects ForearmParameters for each one.

    Args:
        selected_groups: Mapping of video path → group data, as returned by
            MultiVideoFramesSelector.
        saved_parameters: Previously saved parameters for pre-populating ROIs,
            or None if this is a fresh session.

    Returns:
        All successfully collected ForearmParameters across all videos.
    """
    all_parameters: list[ForearmParameters] = []

    for video_path, group_data in selected_groups.items():
        parameters = _collect_parameters_for_video(
            video_path=video_path,
            groups=group_data["groups"],
            representatives=group_data["representatives"],
            saved_parameters=saved_parameters,
        )
        all_parameters.extend(parameters)

    return all_parameters


# ----------------------------------------------------------------------------
# PUBLIC PIPELINE STAGES
# ----------------------------------------------------------------------------

def load_saved_parameters(metadata_path: str) -> Optional[list[ForearmParameters]]:
    """
    Loads a previously saved metadata file if it exists and has a valid structure.

    Returns the list of ForearmParameters, or None if no valid file is found.
    This is used to pre-populate the frame-group selector and ROI UI in
    subsequent sessions, avoiding redundant re-annotation.
    """
    if not ForearmFrameParametersFileHandler.is_valid_structure(metadata_path):
        return None

    print(f"⚠️  Existing metadata found at '{metadata_path}'. Loading for pre-fill...")
    return ForearmFrameParametersFileHandler.load(metadata_path)


def select_frame_groups(
    rgb_video_paths: list[str],
    saved_parameters: Optional[list[ForearmParameters]] = None,
) -> Optional[dict]:
    """
    Opens the interactive frame-group selector and returns the user's selection.

    If ``saved_parameters`` are provided, the UI is pre-populated with the
    previously defined groups so the user can review or adjust them.

    Args:
        rgb_video_paths: Paths to the RGB video files to display in the UI.
        saved_parameters: Previously saved parameters used to pre-fill the UI,
            or None for a blank session.

    Returns:
        The ``all_selected_groups`` dict from the selector (mapping video path
        → group data), or None if the user closed the window without confirming.
    """
    print("\n🖱️  Select frame groups for all videos. Close the window to continue...")
    prefilled_groups = (
        _build_groups_by_video_from_saved_parameters(saved_parameters)
        if saved_parameters else None
    )
    selector = _open_frame_group_selector(rgb_video_paths, prefilled_groups)

    if not selector.validated or not selector.all_selected_groups:
        print("\n🟡 No groups confirmed.")
        return None

    _log_selected_groups(selector.all_selected_groups)
    return selector.all_selected_groups


def define_rois_for_frame_groups(
    selected_groups: dict,
    saved_parameters: Optional[list[ForearmParameters]] = None,
) -> list[ForearmParameters]:
    """
    Prompts the user to draw an ROI for each frame group and returns the
    resulting ForearmParameters.

    For each group, the representative frame is displayed. If a matching ROI
    exists in ``saved_parameters``, it is pre-drawn so the user can accept or
    adjust it. Groups whose ROI selection is cancelled are silently skipped.

    Args:
        selected_groups: Mapping of video path → group data, as returned by
            ``select_frame_groups``.
        saved_parameters: Previously saved parameters for pre-populating ROIs,
            or None if this is a fresh session.

    Returns:
        All successfully collected ForearmParameters, across all videos and groups.
    """
    print("\n🖱️  Draw a Region of Interest (ROI) for each frame group.")
    return _collect_all_parameters(selected_groups, saved_parameters)


def save_forearm_parameters(
    parameters: list[ForearmParameters],
    metadata_path: str,
) -> None:
    """
    Sorts and persists a list of ForearmParameters to a JSON file.

    Args:
        parameters: The parameters to save. Must be non-empty.
        metadata_path: Destination path for the JSON file.

    Raises:
        ValueError: If ``parameters`` is empty, since writing an empty file
            would silently discard all annotation work.
    """
    if not parameters:
        raise ValueError("No parameters to save — metadata file will not be written.")

    print("\n💾 Saving forearm extraction parameters...")
    sorted_parameters = sort_forearm_parameters_by_video_and_frame(parameters)
    ForearmFrameParametersFileHandler.save(sorted_parameters, metadata_path)
    print(f"   ✅ Saved {len(sorted_parameters)} parameter set(s) to '{metadata_path}'.")


# ----------------------------------------------------------------------------
# CONVENIENCE WRAPPER
# ----------------------------------------------------------------------------

def define_forearm_extraction_parameters(rgb_video_paths: list[str], metadata_path: str) -> None:
    """
    Convenience wrapper that runs all three annotation stages in sequence:
    load existing parameters → select frame groups → draw ROIs → save.

    Prefer calling the individual stage functions directly when you need finer
    control over the pipeline (e.g. to interleave logging, validation, or
    conditional branching between steps).

    Args:
        rgb_video_paths: Paths to the input RGB video files.
        metadata_path: Destination path for the output JSON metadata file.
    """
    saved_parameters = load_saved_parameters(metadata_path)
    selected_groups  = select_frame_groups(rgb_video_paths, saved_parameters)

    if selected_groups is None:
        print("Aborting — no frame groups were selected.")
        return

    parameters = define_rois_for_frame_groups(selected_groups, saved_parameters)

    if not parameters:
        print("\n🟡 No ROIs were defined. Metadata file will not be written.")
        return

    save_forearm_parameters(parameters, metadata_path)


# ----------------------------------------------------------------------------
# EXAMPLE USAGE
# ----------------------------------------------------------------------------

def create_dummy_video(filename: str, width: int, height: int, num_frames: int, color: tuple):
    """Helper function to create a sample video file for testing."""
    if os.path.exists(filename):
        print(f"Found existing video: '{filename}'")
        return

    print(f"'{filename}' not found. Creating a dummy video...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))

    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        x_pos = int((width / 3) + (width / 4) * np.sin(i * 0.1))
        y_pos = int((height / 3) + (height / 4) * np.cos(i * 0.15))
        cv2.rectangle(frame, (x_pos, y_pos), (x_pos + 50, y_pos + 50), color, -1)
        cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)

    out.release()
    print(f"Dummy video '{filename}' created.")


if __name__ == "__main__":
    VIDEO_FILENAMES = ["sample_video_1.mp4", "sample_video_2.mp4"]
    METADATA_FILENAME = "forearm_extraction_parameters.json"

    create_dummy_video(VIDEO_FILENAMES[0], 640, 480, 80, color=(255, 0, 0))
    create_dummy_video(VIDEO_FILENAMES[1], 800, 600, 60, color=(0, 255, 0))

    print("\nStarting metadata generation process...")
    try:
        define_forearm_extraction_parameters(
            rgb_video_paths=VIDEO_FILENAMES,
            metadata_path=METADATA_FILENAME,
        )
    finally:
        if _tk_root_instance:
            _tk_root_instance.destroy()
            print("\nTkinter instance destroyed.")