import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[4] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.should_process_task import should_process_task, clean_task_outputs


def adjust_nerve_conduction_velocity(
    input_csv_path: Path,
    metadata_csv_path: Path,
    output_csv_path: Path,
    *,
    force_processing: bool = False
) -> dict | None:
    if not should_process_task(
        output_paths=output_csv_path,
        input_paths=input_csv_path,
        force=force_processing
    ):
        return None

    clean_task_outputs(output_csv_path)

    nerve = pd.read_csv(input_csv_path)

    for col in ("Nervespike1", "Freq"):
        if col not in nerve.columns:
            raise ValueError(f"Column '{col}' missing from nerve CSV: {input_csv_path.name}")

    metadata = pd.read_csv(metadata_csv_path)

    neuron_id = input_csv_path.stem.split("_")[1]

    mask = metadata["Unit_name"] == neuron_id
    if not mask.any():
        raise ValueError(f"Unit '{neuron_id}' not found in metadata CSV")

    row = metadata[mask]
    cond_vel_m_s = row["conduction_velocity (m/s)"].values[0]
    distance_cm = row["electrode_endorgan_distance (cm)"].values[0]

    if cond_vel_m_s == 0:
        raise ValueError(f"Conduction velocity is zero for unit '{neuron_id}', cannot compute lag")

    lag_sec = (distance_cm / 100) / cond_vel_m_s
    sampling_frequency = 1 / np.mean(np.diff(nerve["Sec_FromStart"].values))
    lag_nsample = int(sampling_frequency * lag_sec)

    df_output = nerve.copy()
    df_output["Nervespike1"] = nerve["Nervespike1"].shift(-lag_nsample, fill_value=0)
    df_output["Freq"] = nerve["Freq"].shift(-lag_nsample, fill_value=0)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(output_csv_path, index=False)

    return {
        "filename": input_csv_path.name,
        "lag_sec": lag_sec,
        "lag_nsample": lag_nsample,
    }
