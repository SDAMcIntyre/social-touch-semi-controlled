import os
from pathlib import Path

from utils.should_process_task import should_process_task
from preprocessing.stickers_analysis import (
    MotionCorrectionOrchestrator,
    OutlierConfig,
)


def correct_xyz_stickers_motion(
    input_csv_path: Path,
    output_csv_path: Path,
    diagnostics_dir: Path,
    *,
    mode: str = "correct",
    filter_method: str = "butterworth",
    filter_params: dict | None = None,
    outlier_detection: dict | None = None,
    sampling_rate_hz: float = 30.0,
    force_processing: bool = False,
):
    """Correct raw XYZ sticker trajectories by removing outliers and smoothing.

    Operates in two modes:

    - ``"correct"`` — apply the chosen filter, save ``_xyz_corrected.csv`` and
      per-sticker diagnostic plots.
    - ``"compare"`` — run all registered filters, produce overlay plots only
      (no CSV is written; ``output_csv_path`` is ignored).

    Args:
        input_csv_path: Path to the raw ``_xyz_tracked.csv`` produced by
                        ``generate_xyz_stickers``.
        output_csv_path: Destination for the corrected CSV (``"correct"`` mode
                         only).
        diagnostics_dir: Directory for PNG diagnostic plots.
        mode: ``"correct"`` or ``"compare"``.
        filter_method: ``"butterworth"`` or ``"savgol"`` (used in correct mode).
        filter_params: Dict with per-filter constructor kwargs, e.g.::

                {
                    "butterworth": {"order": 2, "cutoff_hz": 6.0},
                    "savgol":      {"window_length": 11, "polyorder": 3},
                }

        outlier_detection: Dict matching :class:`OutlierConfig` field names.
        sampling_rate_hz: Recording frame rate (default 30.0 for Azure Kinect).
        force_processing: If ``False``, skip if ``output_csv_path`` already
                          exists.
    """
    if mode == "correct":
        output_paths = [output_csv_path]
    else:
        output_paths = []

    if not force_processing and mode == "correct" and output_csv_path.exists():
        print(
            f"✅ Corrected CSV already exists: {output_csv_path}. "
            "Use force_processing=True to overwrite."
        )
        return

    print(f"Starting XYZ motion correction (mode='{mode}')...")

    # Build outlier config from dict or use defaults
    outlier_cfg = None
    if outlier_detection:
        outlier_cfg = OutlierConfig(**outlier_detection)

    orchestrator = MotionCorrectionOrchestrator(
        outlier_config=outlier_cfg,
        sampling_rate_hz=sampling_rate_hz,
    )

    if mode == "correct":
        orchestrator.run_correct(
            input_csv=input_csv_path,
            output_csv=output_csv_path,
            diagnostics_dir=diagnostics_dir,
            filter_method=filter_method,
            filter_params=filter_params,
        )
        print(f"✅ Corrected CSV saved to: {output_csv_path}")
    elif mode == "compare":
        orchestrator.run_compare(
            input_csv=input_csv_path,
            diagnostics_dir=diagnostics_dir,
            filter_params=filter_params,
        )
        print(f"✅ Comparison plots saved to: {diagnostics_dir}")
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'correct' or 'compare'.")
