"""Rename existing on-disk output directories under ``4_analysed/`` from legacy
names to the canonical names defined in ``analysis.pipeline.output_dirs``.

Usage::

    python code/scripts/migrate_output_dirs.py --database-path /path/to/data
    python code/scripts/migrate_output_dirs.py --database-path /path/to/data --dry-run
"""

import argparse
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make ``code/src`` importable so ``analysis.pipeline.output_dirs`` resolves
# when this script is executed standalone.
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC_DIR))

from analysis.pipeline.output_dirs import RENAME_MAPPING  # noqa: E402


def _delete_sentinels(directory: Path) -> int:
    """Delete ``*_done.json`` and ``*_summary.json`` sentinel files inside
    *directory* (recursive).  Returns the number of deleted files."""
    deleted = 0
    for pattern in ("**/*_done.json", "**/*_summary.json"):
        for sentinel in directory.glob(pattern):
            sentinel.unlink()
            print(f"  deleted sentinel: {sentinel}")
            deleted += 1
    return deleted


def migrate(database_path: Path, *, dry_run: bool) -> None:
    analysed_root = database_path / "4_analysed"
    if not analysed_root.is_dir():
        raise FileNotFoundError(
            f"Expected directory does not exist: {analysed_root}"
        )

    renamed = 0
    skipped = 0
    sentinels_deleted = 0

    for old_name, new_name in RENAME_MAPPING.items():
        old_dir = analysed_root / old_name
        new_dir = analysed_root / new_name

        old_exists = old_dir.is_dir()
        new_exists = new_dir.is_dir()

        if old_exists and new_exists:
            raise RuntimeError(
                f"Ambiguous state: both old and new directories exist.\n"
                f"  old: {old_dir}\n"
                f"  new: {new_dir}\n"
                f"Resolve manually before re-running the migration."
            )

        if not old_exists:
            # Already migrated or never created — skip silently.
            skipped += 1
            continue

        # old_exists is True, new_exists is False → rename.
        if dry_run:
            print(f"[dry-run] would rename: {old_dir} -> {new_dir}")
        else:
            shutil.move(str(old_dir), str(new_dir))
            print(f"renamed: {old_dir} -> {new_dir}")
            sentinels_deleted += _delete_sentinels(new_dir)
        renamed += 1

    # Summary
    mode = "[dry-run] " if dry_run else ""
    print(
        f"\n{mode}Migration complete: "
        f"{renamed} renamed, {skipped} skipped, "
        f"{sentinels_deleted} sentinels deleted."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rename legacy output directories under 4_analysed/ to their "
            "canonical names defined in analysis.pipeline.output_dirs."
        ),
    )
    parser.add_argument(
        "--database-path",
        type=Path,
        required=True,
        help="Path to the parent directory that contains 4_analysed/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print what would be renamed without touching disk.",
    )
    args = parser.parse_args()

    database_path = args.database_path.resolve()
    if not database_path.is_dir():
        raise FileNotFoundError(
            f"Database path does not exist: {database_path}"
        )

    migrate(database_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
