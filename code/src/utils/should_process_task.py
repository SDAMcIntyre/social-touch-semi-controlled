from pathlib import Path
from typing import List, Union

# Define a type alias for clarity and reusability
PathLike = Union[str, Path]
PathInput = Union[PathLike, List[PathLike]]

def should_process_task(
    *,
    output_paths: PathInput,
    input_paths: PathInput,
    force: bool = False
) -> bool:
    """
    Determines if a task should run based on file existence and timestamps,
    supporting single or multiple inputs/outputs provided as Strings or Paths.

    Args:
        output_paths: Single path or list of paths (str or Path) representing generated artifacts.
        input_paths: Single path or list of paths (str or Path) representing source files.
        force: If True, bypasses checks and forces processing.

    Returns:
        bool: True if processing is required, False otherwise.
    """
    
    # 1. Helper function to normalize arguments to List[Path]
    def _normalize_to_paths(targets: PathInput) -> List[Path]:
        if isinstance(targets, (str, Path)):
            return [Path(targets)]
        return [Path(p) for p in targets]

    outputs: List[Path] = _normalize_to_paths(output_paths)
    inputs: List[Path] = _normalize_to_paths(input_paths)

    # 2. Check if any input files are missing
    for path in inputs:
        if not path.exists():
            # If an input file is missing, the dependency graph is broken.
            raise FileNotFoundError(f"❌ Input file '{path}' is missing. Cannot process task.")
    
    # 3. Check if force processing is requested
    if force:
        # Convert paths to strings for logging readability
        print(f"➡️ Forced processing for task generating: {[str(p) for p in outputs]}.")
        return True

    # 4. Check if any output files are missing
    for path in outputs:
        if not path.exists():
            print(f"➡️ Output file '{path}' does not exist. Processing required.")
            return True
            
    # 5. The Staleness Check: Compare the newest input to the OLDEST output
    # If the most recently modified input is newer than the oldest output, the output is stale.
    try:
        oldest_output_mod_time = min(p.stat().st_mtime for p in outputs)
        latest_input_mod_time = max(p.stat().st_mtime for p in inputs)
    except ValueError:
        # Handles cases where inputs or outputs lists might be empty, though unlikely given logic above
        print("⚠️ Empty input or output list detected during timestamp check.")
        return True

    if latest_input_mod_time > oldest_output_mod_time:
        print(f"⚠️ Task is stale. An input has been updated more recently than output.")
        return True

    print(f"✅ Task outputs are up-to-date.")
    return False