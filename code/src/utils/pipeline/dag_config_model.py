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

    def has_session_configs(self) -> bool:
        """Return True if this workflow uses session configs (kinect or forearm)."""
        params = self._data.get("parameters", {}) or {}
        return "kinect_configs" in params or "forearm_configs" in params

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

    def get_profile_names(self, task_name: str, option_key: str) -> list[str]:
        """Return profile names for a profile-container option."""
        opts = self._get_task(task_name).get("options", {}) or {}
        container = opts.get(option_key)
        if not isinstance(container, dict):
            return []
        return list(container.keys())

    def get_profile_enabled(self, task_name: str, option_key: str, profile_name: str) -> bool:
        """Return True if the profile is enabled (absent ``enabled`` key defaults to True)."""
        opts = self._get_task(task_name).get("options", {}) or {}
        container = opts.get(option_key, {}) or {}
        profile = container.get(profile_name, {}) or {}
        return bool(profile.get("enabled", True))

    def set_profile_enabled(self, task_name: str, option_key: str, profile_name: str, enabled: bool) -> None:
        """Set or remove the ``enabled`` key on a profile dict.

        When *enabled* is False, writes ``enabled: false``.
        When *enabled* is True, removes the key so the YAML stays clean.
        """
        task = self._get_task(task_name)
        opts = task.get("options") or {}
        container = opts.get(option_key) or {}
        profile = container.get(profile_name)
        if profile is None:
            return
        if enabled:
            profile.pop("enabled", None)
        else:
            profile["enabled"] = False
        self._dirty = True

    # ------------------------------------------------------------------
    # Combinations — CRUD for dict-of-dicts option entries
    # ------------------------------------------------------------------

    def add_combination(
        self, task_name: str, option_key: str, combo_name: str, combo_config: dict
    ) -> None:
        """Insert a new combination entry into a dict-of-dicts option."""
        task = self._get_task(task_name)
        opts = task.get("options") or {}
        container = opts.get(option_key)
        if not isinstance(container, dict):
            return
        container[combo_name] = combo_config
        self._dirty = True

    def remove_combination(self, task_name: str, option_key: str, combo_name: str) -> None:
        """Delete a combination entry from a dict-of-dicts option."""
        task = self._get_task(task_name)
        opts = task.get("options") or {}
        container = opts.get(option_key)
        if not isinstance(container, dict):
            return
        container.pop(combo_name, None)
        self._dirty = True

    def get_combination_features(
        self, task_name: str, option_key: str, combo_name: str
    ) -> list[str]:
        """Return the features list for a combination entry."""
        opts = self._get_task(task_name).get("options", {}) or {}
        container = opts.get(option_key, {}) or {}
        combo = container.get(combo_name, {}) or {}
        features = combo.get("features", [])
        return list(features) if features else []

    def set_combination_features(
        self, task_name: str, option_key: str, combo_name: str, features: list[str]
    ) -> None:
        """Write a flow-style features list into a combination entry."""
        task = self._get_task(task_name)
        opts = task.get("options") or {}
        container = opts.get(option_key) or {}
        combo = container.get(combo_name)
        if combo is None:
            return
        seq = CommentedSeq(features)
        seq.fa.set_flow_style()
        combo["features"] = seq
        self._dirty = True

    def get_task_dependencies(self, task_name: str) -> list[str]:
        deps = self._get_task(task_name).get("depends_on", [])
        return list(deps) if deps else []

    def get_task_description(self, task_name: str) -> str | None:
        return self._get_task(task_name).get("description")

    # ------------------------------------------------------------------
    # Cluster groups — CRUD for touch_clustering groups and downstream refs
    # ------------------------------------------------------------------
    #
    # get_cluster_group_names / set_profile_enabled reuse existing profile helpers
    # with option_key="cluster_groups".

    def get_cluster_group_spec(self, task_name: str, group_name: str) -> dict:
        """Return a copy of the full spec dict for *group_name* in *task_name*."""
        opts = self._get_task(task_name).get("options", {}) or {}
        groups = opts.get("cluster_groups", {}) or {}
        spec = groups.get(group_name)
        if spec is None:
            raise KeyError(f"Cluster group '{group_name}' not found in task '{task_name}'")
        return dict(spec)

    def set_cluster_group_spec(self, task_name: str, group_name: str, spec: dict) -> None:
        """Write (or overwrite) the full spec dict for *group_name* in *task_name*."""
        task = self._get_task(task_name)
        opts = task.get("options")
        if opts is None:
            task["options"] = {}
            opts = task["options"]
        if "cluster_groups" not in opts or opts["cluster_groups"] is None:
            opts["cluster_groups"] = {}
        opts["cluster_groups"][group_name] = spec
        self._dirty = True

    def get_downstream_cluster_group_names(self, task_name: str) -> list[str]:
        """Return the list of group names referenced by a downstream task.

        Reads the flow-style ``cluster_groups: [name, ...]`` value.
        Returns an empty list if the option is absent or not a sequence.
        """
        opts = self._get_task(task_name).get("options", {}) or {}
        val = opts.get("cluster_groups")
        if isinstance(val, (list, CommentedSeq)):
            return list(val)
        return []

    def set_downstream_cluster_group_names(
        self, task_name: str, names: list[str]
    ) -> None:
        """Write *names* as a flow-style ``cluster_groups: [...]`` list."""
        task = self._get_task(task_name)
        opts = task.get("options")
        if opts is None:
            task["options"] = {}
            opts = task["options"]
        seq = CommentedSeq(names)
        seq.fa.set_flow_style()
        opts["cluster_groups"] = seq
        self._dirty = True

    def get_group_clustering_methods(self, task_name: str, group_name: str) -> dict:
        """Return the clustering_methods dict for *group_name* in *task_name*."""
        spec = self.get_cluster_group_spec(task_name, group_name)
        methods = spec.get("clustering_methods", {})
        return dict(methods) if isinstance(methods, dict) else {}

    def set_group_clustering_profile_enabled(
        self,
        task_name: str,
        group_name: str,
        profile_name: str,
        enabled: bool,
    ) -> None:
        """Enable or disable a clustering profile inside a cluster group."""
        task = self._get_task(task_name)
        opts = task.get("options", {}) or {}
        groups = opts.get("cluster_groups", {}) or {}
        group = groups.get(group_name)
        if group is None:
            raise KeyError(f"Cluster group '{group_name}' not found in task '{task_name}'")
        methods = group.get("clustering_methods", {}) or {}
        profile = methods.get(profile_name)
        if profile is None:
            raise KeyError(
                f"Clustering profile '{profile_name}' not found in group '{group_name}'"
            )
        if enabled:
            profile.pop("enabled", None)
        else:
            profile["enabled"] = False
        self._dirty = True

    def set_group_clustering_profile_spec(
        self,
        task_name: str,
        group_name: str,
        profile_name: str,
        spec: dict,
    ) -> None:
        """Write (or overwrite) a clustering profile spec inside a cluster group."""
        task = self._get_task(task_name)
        opts = task.get("options")
        if opts is None:
            task["options"] = {}
            opts = task["options"]
        if "cluster_groups" not in opts or opts["cluster_groups"] is None:
            opts["cluster_groups"] = {}
        group = opts["cluster_groups"].get(group_name)
        if group is None:
            raise KeyError(f"Cluster group '{group_name}' not found in task '{task_name}'")
        if "clustering_methods" not in group or group["clustering_methods"] is None:
            group["clustering_methods"] = {}
        group["clustering_methods"][profile_name] = spec
        self._dirty = True

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
