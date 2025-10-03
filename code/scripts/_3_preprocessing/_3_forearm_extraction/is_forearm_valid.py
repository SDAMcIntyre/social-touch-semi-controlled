import os
from pathlib import Path

def is_forearm_valid(
        session_output_dir: Path,
        verbose: bool = False
):
    pointclouds_output_dir = session_output_dir
    flag_file = pointclouds_output_dir / ".SUCCESS"
    is_valid = os.path.exists(flag_file)
    # Handle verbose output
    if verbose:
        if is_valid:
            print(f"✅ Forearm data has been manually validated for: {session_output_dir.name}")
        else:
            print(f"❌ ValiForearm data has not been manuallys validated yet for: {session_output_dir.name}")

    return is_valid
