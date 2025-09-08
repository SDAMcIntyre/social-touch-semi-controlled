import os
from pathlib import Path

def is_forearm_valid(
        session_output_dir: Path
):
    pointclouds_output_dir = session_output_dir
    flag_file = pointclouds_output_dir / ".SUCCESS"
    return os.path.exists(flag_file)
