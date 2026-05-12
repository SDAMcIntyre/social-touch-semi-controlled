from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from utils.should_process_task import should_process_task, clean_task_outputs
from preprocessing.motion_analysis.hand_tracking import PoseStabilisation

_DEFAULT_FILTER_METHOD = "butterworth"
_DEFAULT_FILTER_PARAMS = {"butterworth": {"order": 2, "cutoff_hz": 5.0}}


def stabilise_hand_motion(
    input_npz_path: Path,
    output_npz_path: Path,
    *,
    filter_method: Union[str, Any] = _DEFAULT_FILTER_METHOD,
    filter_params: Optional[Dict[str, Any]] = None,
    force_processing: bool = False,
):
    """Load the raw hand-motion NPZ, apply pose stabilisation, and write the corrected NPZ.

    The raw NPZ is never modified.  Only ``translations``, ``rotations``, and ``scales``
    are replaced; all other arrays (``vertices``, ``faces``, ``timestamps``, ``fps``,
    ``sticker_vertex_indices``) are written unchanged.

    Args:
        input_npz_path: Path to ``*_handmodel_motion.npz`` produced by
            ``generate_3d_hand_in_motion``.
        output_npz_path: Destination path for ``*_handmodel_motion_stabilised.npz``.
        filter_method: Filter name forwarded to ``PoseStabilisation.stabilise``.
        filter_params: Filter parameters forwarded to ``PoseStabilisation.stabilise``.
        force_processing: When ``False`` (default), skip if the output already exists
            and is newer than the input.

    Raises:
        FileNotFoundError: If ``input_npz_path`` does not exist (raised by
            ``should_process_task`` before any work is done).
        KeyError: If the raw NPZ is missing ``sticker_vertex_indices``; re-run
            ``generate_3d_hand_in_motion`` to regenerate the NPZ with that key.
        ValueError: If the session is too short for the chosen filter, or no valid
            scales remain after range filtering.
    """
    if filter_params is None:
        filter_params = _DEFAULT_FILTER_PARAMS

    if not should_process_task(
        input_paths=[input_npz_path],
        output_paths=[output_npz_path],
        force=force_processing,
    ):
        print("Output already up to date. Use force_processing=True to overwrite.")
        return
    clean_task_outputs([output_npz_path])

    if not input_npz_path.exists():
        raise FileNotFoundError(
            f"Raw hand-motion NPZ not found: {input_npz_path}\n"
            "Run 'generate_3d_hand_in_motion' first."
        )

    print(f"Loading {input_npz_path.name} ...")
    with np.load(input_npz_path) as raw:
        vertices = raw["vertices"]
        translations = raw["translations"]
        rotations = raw["rotations"]
        scales = raw["scales"]
        timestamps = raw["timestamps"]
        faces = raw["faces"] if "faces" in raw else None
        fps = float(raw["fps"])

        if "sticker_vertex_indices" not in raw:
            raise KeyError(
                f"NPZ file '{input_npz_path}' is missing the 'sticker_vertex_indices' key. "
                "Re-run 'generate_3d_hand_in_motion' to regenerate the NPZ with this key."
            )
        sticker_vertex_indices = raw["sticker_vertex_indices"]

    anchor_idx = int(sticker_vertex_indices[0])

    print(
        f"Stabilising {len(translations)} frames "
        f"(filter={filter_method}, anchor_idx={anchor_idx}) ..."
    )
    translations_new, rotations_new, scales_new = PoseStabilisation.stabilise(
        vertices=vertices,
        translations=translations,
        rotations_xyzw=rotations,
        scales=scales,
        anchor_idx=anchor_idx,
        fps=fps,
        filter_method=filter_method,
        filter_params=filter_params,
    )

    print(f"Saving stabilised NPZ to {output_npz_path.name} ...")
    save_dict = {
        "vertices": vertices,
        "translations": translations_new,
        "rotations": rotations_new,
        "scales": scales_new,
        "timestamps": timestamps,
        "fps": fps,
        "sticker_vertex_indices": sticker_vertex_indices,
    }
    if faces is not None:
        save_dict["faces"] = faces

    np.savez_compressed(output_npz_path, **save_dict)
    print("Stabilised NPZ saved.")


if __name__ == "__main__":
    dataset_path = r"F:/liu-onedrive-nospecial-carac/_Teams/Social touch Kinect MNG/02_data/semi-controlled/"
    dataset_block_path = dataset_path + r"2_processed/kinect/2022-06-17_ST16-05/block-order-01/"
    kinematics_dir = dataset_block_path + r"kinematics_analysis/"
    prefix = "2022-06-17_ST16-05_semicontrolled_block-order01_kinect_handmodel"

    input_npz = Path(kinematics_dir + prefix + "_motion.npz")
    output_npz = Path(kinematics_dir + prefix + "_motion_stabilised.npz")

    stabilise_hand_motion(
        input_npz_path=input_npz,
        output_npz_path=output_npz,
        force_processing=True,
    )
