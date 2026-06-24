from pathlib import Path
from typing import List, Union

# Define a type alias for clarity and reusability
PathLike = Union[str, Path]
PathInput = Union[PathLike, List[PathLike]]


def _normalize_to_paths(targets: PathInput) -> List[Path]:
    if isinstance(targets, (str, Path)):
        return [Path(targets)]
    return [Path(p) for p in targets]


def refresh_output_mtimes(output_paths: PathInput) -> None:
    """Refresh the modification timestamps of existing output files without changing their contents.

    Use this when processing is skipped (e.g. operator closes a GUI without edits) but
    output mtimes must be brought forward so the staleness check in should_process_task()
    does not refire on a subsequent non-forced run.

    Args:
        output_paths: Single path or list of paths (str or Path) to touch.
                      Missing paths are silently skipped.
                      Read-only paths are logged and skipped — no exception is raised.
    """
    paths = _normalize_to_paths(output_paths)
    for p in paths:
        if not p.exists():
            continue
        try:
            p.touch()
        except PermissionError:
            print(f"  Could not touch '{p}' (permission denied). Skipping mtime refresh.")


def clean_task_outputs(output_paths: PathInput) -> None:
    """Delete output files before processing to prevent stale artifacts.

    Call immediately after should_process_task() returns True,
    before actual processing begins.
    """
    paths = _normalize_to_paths(output_paths)
    for p in paths:
        if p is None:
            continue
        if p.exists() and p.is_file():
            try:
                p.unlink()
                print(f"  Cleaned stale output: '{p.name}'")
            except PermissionError:
                print(f"  Could not delete '{p}' (file locked). Proceeding anyway.")


def should_process_task(
    *,
    output_paths: PathInput,
    input_paths: PathInput,
    force: bool = False,
    keep_stale: bool = False
) -> bool:
    """
    Determines if a task should run based on file existence and timestamps,
    supporting single or multiple inputs/outputs provided as Strings or Paths.

    Args:
        output_paths: Single path or list of paths (str or Path) representing generated artifacts.
        input_paths: Single path or list of paths (str or Path) representing source files.
        force: If True, bypasses checks and forces processing. Overrides keep_stale.
        keep_stale: If True, refreshes timestamps of stale outputs instead of reprocessing.

    Returns:
        bool: True if processing is required, False otherwise.
    """

    outputs: List[Path] = _normalize_to_paths(output_paths)
    inputs:  List[Path] = _normalize_to_paths(input_paths)

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
        if keep_stale:
            print("⚠️ Task is stale but keep_stale=True — refreshing output timestamps.")
            for p in outputs:
                try:
                    p.touch()
                except PermissionError:
                    print(f"  Could not touch '{p}' (permission denied). Proceeding with processing.")
                    return True
            return False
        print(f"⚠️ Task is stale. An input has been updated more recently than output.")
        return True

    print(f"✅ Task outputs are up-to-date.")
    return False