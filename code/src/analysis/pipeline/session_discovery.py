"""Session-discovery helpers shared by both analysis entry scripts.

``collect_unique_session_dirs`` maps each block config file to the
corresponding session merged-output directory.  ``discover_input_items``
scans those directories for the aggregated session CSV that every analysis
flow expects as its primary input.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

from primary_processing import KinectConfigFileHandler, KinectConfig


def collect_unique_session_dirs(
    block_files: List[Path],
    project_data_root: Path,
) -> Dict[Path, Path]:
    """Return a mapping of session-merged-output dirs to their database paths.

    Iterates *block_files*, loads each kinect config, and records the
    ``session_merged_output_dir → database_path`` pair.  Config files that
    cannot be loaded are skipped with a DEBUG-level log entry — they may
    belong to an incomplete recording or a different pipeline branch.

    Returns a dict with one entry per unique session directory.
    """
    session_dir_map: Dict[Path, Path] = {}
    for block_file in block_files:
        try:
            config_data = KinectConfigFileHandler.load_and_resolve_config(block_file)
            config = KinectConfig(config_data=config_data, database_path=project_data_root)

            if config.session_merged_output_dir and config.database_path:
                session_dir_map[config.session_merged_output_dir] = config.database_path
        except Exception as e:
            logging.debug(f"Skipping {block_file.name}: {e}")

    logging.info(f"Scanned {len(block_files)} config files.")
    logging.info(f"Identified {len(session_dir_map)} unique session contexts.")

    return session_dir_map


def discover_input_items(
    session_map: Dict[Path, Path],
) -> List[Tuple[Path, Path]]:
    """Scan each session directory for its aggregated session CSV.

    For every ``search_dir`` in *session_map*, globs for
    ``*_semicontrolled_aggregated_session.csv``.  The first match (if any)
    is paired with the corresponding ``database_path`` and appended to the
    result list.  Sessions with no matching CSV are silently skipped —
    they have not yet been merged and are not ready for analysis.

    Returns a list of ``(aggregated_csv_path, database_path)`` tuples,
    one per session that has a valid input file.
    """
    items_to_process: List[Tuple[Path, Path]] = []

    logging.info(f"Scanning {len(session_map)} sessions for data files...")
    for search_dir in sorted(session_map.keys()):
        database_path_context = session_map[search_dir]
        candidates = list(search_dir.glob("*_semicontrolled_aggregated_session.csv"))

        if candidates:
            target_file = candidates[0]
            items_to_process.append((target_file, database_path_context))

    return items_to_process
