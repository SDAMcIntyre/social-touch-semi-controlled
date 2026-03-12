from pathlib import Path

from utils.should_process_task import should_process_task
from preprocessing.forearm_extraction import (
    ForearmRegistrator,
    transform_unified_csv,
)
from preprocessing.forearm_extraction.registration.csv_spatial_transformer import (
    find_applicable_transform_key as _find_applicable_transform_key,
)


def apply_registration_transform(
    somatosensory_chars_path: Path,
    session_processed_dir: Path,
    session_id: str,
    current_video_filename: str,
    output_dir: Path,
    *,
    force_processing: bool = False,
) -> Path:
    """Apply a pre-computed registration transform to a somatosensory contact CSV.

    For multi-forearm sessions, loads the per-snapshot 4x4 rigid transform and
    applies it to the spatial columns so that all blocks share a common coordinate
    frame.  Single-forearm sessions (no transforms file) pass through unchanged.

    Transform selection follows "applies forward" semantics: the transform from
    snapshot ``block-orderXX:FRAME`` is used for the current video if that video
    is block-orderXX, or for any subsequent block that has no snapshot of its own,
    up until the next snapshot in the session timeline.

    Returns the path to the (possibly new) registered CSV.
    """
    # 1. Load registration transforms
    forearm_pc_dir = session_processed_dir / "forearm_pointclouds"
    transforms_path = forearm_pc_dir / f"{session_id}_registration_transforms.json"
    transforms_data = ForearmRegistrator.load_transforms(transforms_path)

    if transforms_data is None:
        print(f"  No registration transforms found ({transforms_path.name}). "
              "Single-forearm session — passing through original CSV.")
        return somatosensory_chars_path

    # 2. Determine which snapshot key applies to this video
    current_video_stem = Path(current_video_filename).stem
    snapshot_key = _find_applicable_transform_key(
        transforms_data["transforms"], current_video_stem
    )

    if snapshot_key is None:
        print(f"  WARNING: No applicable transform found for '{current_video_filename}' "
              "(no preceding snapshot in session timeline). "
              "Passing through original CSV.")
        return somatosensory_chars_path

    # Log when falling back to a preceding block's snapshot
    if not snapshot_key.startswith(current_video_stem + ":"):
        preceding_stem, _, preceding_frame = snapshot_key.rpartition(":")
        print(f"  No snapshot for '{current_video_stem}'. "
              f"Using most recent preceding transform: '{preceding_stem}' "
              f"(frame {preceding_frame}).")

    # 3. Look up the 4x4 transform
    transform_entry = transforms_data["transforms"][snapshot_key]
    transform_4x4 = transform_entry["matrix_4x4"]
    fitness = transform_entry["fitness"]
    print(f"  Using transform for '{snapshot_key}' (fitness={fitness:.4f}).")

    # 4. Build output path and apply transform
    output_path = output_dir / somatosensory_chars_path.name.replace(
        "_contact_and_kinematic_data.csv",
        "_contact_and_kinematic_data_with_reference-transformation.csv"
    )

    if not should_process_task(
        output_paths=[output_path],
        input_paths=[somatosensory_chars_path],
        force=force_processing,
    ):
        return output_path

    transform_unified_csv(somatosensory_chars_path, output_path, transform_4x4)
    print(f"  Registered CSV written to: {output_path.name}")
    return output_path
