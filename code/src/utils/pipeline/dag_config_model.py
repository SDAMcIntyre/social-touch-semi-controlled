"""Round-trip YAML model for GUI editing of DAG configuration files.

Uses ruamel.yaml to preserve comments, ordering, and formatting when
loading, modifying, and saving DAG workflow YAML files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq


class DagConfigModel:
    """In-memory representation of a DAG YAML config with round-trip fidelity.

    Unlike :class:`DagConfigHandler` (which uses ``yaml.safe_load`` and
    discards comments), this class uses ``ruamel.yaml`` in round-trip mode
    so that save → reload cycles preserve section headers, inline comments,
    and key ordering.
    """

    def __init__(self, config_path: Path) -> None:
        self._path = config_path
        self._yaml = YAML()
        self._yaml.preserve_quotes = True
        with open(config_path, "r") as fh:
            self._data = self._yaml.load(fh)
        self._dirty = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    @property
    def dirty(self) -> bool:
        return self._dirty

    # ------------------------------------------------------------------
    # Parameters — session config entries (kinect or forearm)
    # ------------------------------------------------------------------

    def get_config_type(self) -> str:
        """Return the config key type for this workflow.

        Returns ``"forearm_configs"`` for workflows that use ``forearm_configs``,
        and ``"kinect_configs"`` for all others.
        """
        params = self._data.get("parameters", {}) or {}
        if "forearm_configs" in params:
            return "forearm_configs"
        return "kinect_configs"

    def get_config_entries(self) -> list[str]:
        """Return the session config entries as a list of strings.

        Each entry is either a directory name or a relative file path, both
        relative to the implied config root (``configs/kinect_configs/`` or
        ``configs/forearm_configs/``).
        """
        params = self._data.get("parameters", {}) or {}
        config_type = self.get_config_type()
        val = params.get(config_type)
        if val is None:
            return []
        if isinstance(val, str):
            return [val] if val else []
        return list(val)

    def set_config_entries(self, entries: list[str]) -> None:
        """Persist session config entries, using flow-style list when multiple.

        A single entry is written as a plain string. Multiple entries are
        written as a flow-style YAML sequence (``[item1, item2]``).
        """
        params = self._data.get("parameters")
        if params is None:
            return
        config_type = self.get_config_type()
        if not entries:
            params[config_type] = ""
        elif len(entries) == 1:
            params[config_type] = entries[0]
        else:
            seq = CommentedSeq(entries)
            seq.fa.set_flow_style()
            params[config_type] = seq
        self._dirty = True

    # ------------------------------------------------------------------
    # Parameters — generic
    # ------------------------------------------------------------------

    def get_parameter(self, name: str, default: Any = None) -> Any:
        params = self._data.get("parameters", {}) or {}
        return params.get(name, default)

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------

    def get_task_names(self) -> list[str]:
        """Return task names in YAML-defined order."""
        tasks = self._data.get("tasks", {}) or {}
        return list(tasks.keys())

    def _get_task(self, task_name: str) -> dict:
        tasks = self._data.get("tasks", {}) or {}
        task = tasks.get(task_name)
        if task is None:
            raise KeyError(f"Task '{task_name}' not found in config")
        return task

    def is_task_enabled(self, task_name: str) -> bool:
        return bool(self._get_task(task_name).get("enabled", False))

    def set_task_enabled(self, task_name: str, enabled: bool) -> None:
        self._get_task(task_name)["enabled"] = enabled
        self._dirty = True

    def get_task_options(self, task_name: str) -> dict[str, Any]:
        return dict(self._get_task(task_name).get("options", {}) or {})

    def get_task_option(self, task_name: str, option: str) -> Any:
        opts = self._get_task(task_name).get("options", {}) or {}
        return opts.get(option)

    def set_task_option(self, task_name: str, option: str, value: Any) -> None:
        task = self._get_task(task_name)
        if "options" not in task or task["options"] is None:
            task["options"] = {}
        task["options"][option] = value
        self._dirty = True

    def get_task_dependencies(self, task_name: str) -> list[str]:
        deps = self._get_task(task_name).get("depends_on", [])
        return list(deps) if deps else []

    def get_task_description(self, task_name: str) -> str | None:
        return self._get_task(task_name).get("description")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write back to the original file, preserving comments."""
        self.save_as(self._path)
        self._dirty = False

    def save_as(self, path: Path) -> None:
        """Write the current state to *path*."""
        with open(path, "w") as fh:
            self._yaml.dump(self._data, fh)
        if path == self._path:
            self._dirty = False

    def reload(self) -> None:
        """Re-read the file from disk, discarding in-memory changes."""
        with open(self._path, "r") as fh:
            self._data = self._yaml.load(fh)
        self._dirty = False
