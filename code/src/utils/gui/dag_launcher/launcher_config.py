"""Launcher configuration — WorkflowEntry dataclass and YAML parser."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class WorkflowEntry:
    """A single workflow declared in launcher.yaml.

    Attributes:
        name: Display name shown on the workflow button.
        script: Absolute path to the Python script.
        dag_config: Absolute path to the DAG YAML, or None for scripts that
            don't use a DAG config (task panel and kinect selector are hidden).
        category: Category header under which this workflow is grouped.
    """

    name: str
    script: Path
    dag_config: Path | None
    category: str


def parse_launcher_config(yaml_path: Path, project_root: Path) -> list[WorkflowEntry]:
    """Load launcher.yaml and return validated WorkflowEntry instances.

    Entries whose ``script`` or ``dag_config`` paths don't exist on disk are
    skipped with a :mod:`warnings` warning.  The GUI still starts with the
    remaining valid entries.

    Args:
        yaml_path: Path to the ``launcher.yaml`` file.
        project_root: Project root directory; relative paths in the YAML are
            resolved relative to this directory.

    Returns:
        List of valid :class:`WorkflowEntry` instances in declaration order.

    Raises:
        FileNotFoundError: If *yaml_path* doesn't exist.
        yaml.YAMLError: If *yaml_path* contains invalid YAML.
    """
    with yaml_path.open() as fh:
        data = yaml.safe_load(fh)

    entries: list[WorkflowEntry] = []
    for category_block in data.get("categories", []):
        category: str = category_block["name"]
        for wf in category_block.get("workflows", []):
            name: str = wf["name"]
            script = project_root / wf["script"]
            dag_config_rel = wf.get("dag_config")
            dag_config = project_root / dag_config_rel if dag_config_rel else None

            if not script.exists():
                warnings.warn(
                    f"Launcher: skipping '{name}' — script not found: {script}",
                    stacklevel=2,
                )
                continue
            if dag_config is not None and not dag_config.exists():
                warnings.warn(
                    f"Launcher: skipping '{name}' — dag_config not found: {dag_config}",
                    stacklevel=2,
                )
                continue

            entries.append(
                WorkflowEntry(
                    name=name,
                    script=script,
                    dag_config=dag_config,
                    category=category,
                )
            )

    return entries
