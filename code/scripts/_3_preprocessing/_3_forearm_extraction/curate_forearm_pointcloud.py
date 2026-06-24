import sys
from pathlib import Path
from typing import Union

import open3d as o3d
from PyQt5.QtWidgets import QApplication

from utils.should_process_task import should_process_task, refresh_output_mtimes
from preprocessing.forearm_extraction.curation import (
    CurationMetadataFileHandler,
    ForearmCurationGUI,
)


def curate_forearm_pointcloud(
    input_ply_path: Union[str, Path],
    output_ply_path: Union[str, Path],
    output_metadata_path: Union[str, Path],
    *,
    force_processing: bool = False,
) -> None:
    """
    Opens an interactive GUI for manual curation of a raw forearm point cloud.

    Allows the operator to box-select and remove unwanted points (artifacts,
    table edges, clothing fragments) before automated cleaning. Both the curated
    PLY and a JSON metadata file (containing removed indices) are written on
    successful validation.

    Skip logic: if both output files exist and ``force_processing`` is False,
    the GUI is not opened.

    Re-editing: when outputs already exist and ``force_processing`` is True,
    previously removed indices are pre-highlighted in the GUI.

    Args:
        input_ply_path: Path to the raw extracted point cloud (.ply).
        output_ply_path: Path where the curated point cloud will be saved (.ply).
        output_metadata_path: Path where curation metadata will be saved (.json).
        force_processing: If True, re-opens the GUI even when outputs already exist.
    """
    input_path = Path(input_ply_path)
    output_path = Path(output_ply_path)
    meta_path = Path(output_metadata_path)

    if not should_process_task(
        output_paths=[output_path, meta_path],
        input_paths=[input_path],
        force=force_processing,
    ):
        return

    outputs_existed = output_path.exists() and meta_path.exists()

    pcd = o3d.io.read_point_cloud(str(input_path))

    if not pcd.has_points():
        print(f"Warning: raw point cloud at {input_path} is empty — writing empty curated PLY.")
        o3d.io.write_point_cloud(str(output_path), pcd)
        CurationMetadataFileHandler.save([], 0, meta_path)
        return

    # Load existing removed indices so the operator can review prior work
    existing_meta = CurationMetadataFileHandler.load(meta_path)
    existing_removed = existing_meta.get("removed_point_indices", []) if existing_meta else []

    app = QApplication.instance() or QApplication(sys.argv)

    validated = [False]
    removed_indices: list = []

    gui = ForearmCurationGUI(pcd, existing_removed_indices=existing_removed)

    def _on_validated(indices: list) -> None:
        removed_indices.extend(indices)
        validated[0] = True

    gui.curation_validated.connect(_on_validated)
    gui.show()
    app.exec_()

    if not validated[0]:
        print("Warning: curation GUI closed without validation — outputs not saved.")
        if outputs_existed:
            refresh_output_mtimes([output_path, meta_path])
        return

    total_original = len(pcd.points)
    removed_set = set(removed_indices)
    kept_indices = [i for i in range(total_original) if i not in removed_set]
    curated_pcd = pcd.select_by_index(kept_indices)

    o3d.io.write_point_cloud(str(output_path), curated_pcd)
    CurationMetadataFileHandler.save(removed_indices, total_original, meta_path)

    print(f"Curated point cloud saved: {output_path}")
    print(f"  {len(removed_indices)} points removed, {len(kept_indices)} remaining.")
