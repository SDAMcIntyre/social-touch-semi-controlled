from pathlib import Path
from typing import List, Union

def should_process_task(
    output_paths: Union[Path, List[Path]],
    input_paths: Union[Path, List[Path]],
    *,
    force: bool = False
) -> bool:
    """
    Determines if a task should run based on file existence and timestamps,
    supporting single or multiple inputs/outputs.

    Returns:
        bool: True if processing is required, False otherwise.
    """
    # 1. Normalize arguments to always be lists for consistent handling
    outputs = [output_paths] if isinstance(output_paths, Path) else output_paths
    inputs = [input_paths] if isinstance(input_paths, Path) else input_paths

    if force:
        print(f"➡️ Forced processing for task generating: {[str(p) for p in outputs]}.")
        return True

    # 2. Check if any output or input files are missing
    for path in outputs:
        if not path.exists():
            print(f"➡️ Output file '{path}' does not exist. Processing required.")
            return True
            
    for path in inputs:
        if not path.exists():
            print(f"➡️ Input file '{path}' is missing. Processing required.")
            return True

    # 3. The Staleness Check: Compare the newest input to the OLDEST output
    # If even the oldest output is newer than all inputs, the task is valid.
    oldest_output_mod_time = min(p.stat().st_mtime for p in outputs)
    latest_input_mod_time = max(p.stat().st_mtime for p in inputs)

    if latest_input_mod_time > oldest_output_mod_time:
        print(f"⚠️ Task is stale. An input has been updated more recently than output.")
        return True

    print(f"✅ Task outputs are up-to-date.")
    return False