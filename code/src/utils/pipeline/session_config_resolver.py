"""Utility for resolving session config entries to a flat list of YAML paths.

Each entry can be a directory name (resolved to all *.yaml files inside) or
a relative file path (resolved directly). Paths are relative to a given
``config_root`` directory (e.g. ``configs/kinect_configs/``).
"""

from __future__ import annotations

from pathlib import Path


def resolve_session_configs(
    entries: str | list[str],
    config_root: Path,
) -> list[Path]:
    """Resolve a mix of directory and file entries into a sorted flat list of YAML paths.

    Args:
        entries: A single string or a list of strings. Each entry is either:
            - A subdirectory name → all ``*.yaml`` files inside are included.
            - A relative file path (e.g. ``"subdir/file.yaml"``) → included directly.
        config_root: The base directory against which ``entries`` are resolved
            (e.g. ``Path("configs") / "kinect_configs"``).

    Returns:
        Deduplicated, order-preserving list of resolved ``Path`` objects.

    Raises:
        FileNotFoundError: If an entry resolves to a path that does not exist.
        ValueError: If ``entries`` is empty or ``None``.
    """
    if not entries:
        return []

    if isinstance(entries, str):
        entries = [entries]

    result: list[Path] = []
    seen: set[Path] = set()

    for entry in entries:
        path = config_root / entry
        if path.is_dir():
            for yaml_file in sorted(path.glob("*.yaml")):
                if yaml_file not in seen:
                    result.append(yaml_file)
                    seen.add(yaml_file)
        elif path.exists():
            if path not in seen:
                result.append(path)
                seen.add(path)
        else:
            raise FileNotFoundError(
                f"Config entry not found: '{entry}' (resolved to '{path}')"
            )

    return result
