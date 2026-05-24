"""Per-session SLIM UV config: dataclass + round-trip YAML I/O.

Reads and writes ``slim_uv_config.yaml`` files placed under
``<output_dir>/<session_id>/`` so that each forearm session can override the
DAG-level defaults (mesh method, max edge length, mesh-cleaning toggles, SLIM
iteration count, diagnostic saving).

All YAML I/O uses ``ruamel.yaml`` round-trip mode so comments and key order
are preserved across save → reload cycles.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap


# ---------------------------------------------------------------------------
# Module-level defaults — sourced from precompute_forearm_slim_uv signature and
# the documented "all clean steps default to True" contract of clean_mesh().
# ---------------------------------------------------------------------------

_VALID_MESH_METHODS: tuple[str, ...] = ("bpa", "delaunay")

DEFAULT_MESH_METHOD: str = "bpa"
# 0.0 sentinel means "auto" (3.0 × mean nearest-neighbour distance) per the
# precompute_forearm_slim_uv docstring; encoded as a float in YAML for clean
# round-trip rather than null.
DEFAULT_MAX_EDGE_MM: float = 0.0
DEFAULT_N_ITER: int = 40
DEFAULT_SAVE_DIAGNOSTICS: bool = True


@dataclass
class SlimUvCleanSteps:
    """Boolean toggles for the 5 toggleable mesh-cleaning sub-steps.

    Steps 1-3 of ``clean_mesh`` (largest component, fix winding, remove
    orphans) are mandatory and not represented here. Field names match the
    keys recognised by ``clean_mesh`` so the dataclass can be converted to
    that dict directly via :meth:`to_dict`.
    """

    remove_non_manifold: bool = True
    repair_pinch_vertices: bool = True
    remove_slivers: bool = True
    stitch_boundary_gaps: bool = True
    fill_interior_holes: bool = True

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SlimUvCleanSteps":
        if not isinstance(data, dict):
            raise ValueError(
                f"clean_steps must be a mapping, got {type(data).__name__}"
            )
        expected = {f.name for f in fields(cls)}
        missing = expected - set(data.keys())
        if missing:
            raise ValueError(
                f"clean_steps is missing required key(s): {sorted(missing)}"
            )
        extra = set(data.keys()) - expected
        if extra:
            raise ValueError(
                f"clean_steps contains unknown key(s): {sorted(extra)}. "
                f"Valid keys: {sorted(expected)}"
            )
        return cls(**{k: bool(data[k]) for k in expected})


DEFAULT_CLEAN_STEPS: SlimUvCleanSteps = SlimUvCleanSteps()


@dataclass
class SlimUvConfig:
    """Per-session SLIM UV configuration persisted as ``slim_uv_config.yaml``."""

    session_id: str
    mesh_method: str = DEFAULT_MESH_METHOD
    max_edge_mm: float = DEFAULT_MAX_EDGE_MM
    n_iter: int = DEFAULT_N_ITER
    save_diagnostics: bool = DEFAULT_SAVE_DIAGNOSTICS
    clean_steps: SlimUvCleanSteps = field(default_factory=SlimUvCleanSteps)
    created_at: str = ""
    modified_at: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError(
                f"session_id must be a non-empty string, got {self.session_id!r}"
            )
        if self.mesh_method not in _VALID_MESH_METHODS:
            raise ValueError(
                f"Unknown mesh_method {self.mesh_method!r}. "
                f"Valid options are: {list(_VALID_MESH_METHODS)}"
            )
        self.max_edge_mm = float(self.max_edge_mm)
        if self.max_edge_mm < 0.0:
            raise ValueError(
                f"max_edge_mm must be >= 0.0 (0.0 means auto), "
                f"got {self.max_edge_mm}"
            )
        self.n_iter = int(self.n_iter)
        if not (1 <= self.n_iter <= 200):
            raise ValueError(
                f"n_iter must be in [1, 200], got {self.n_iter}"
            )
        self.save_diagnostics = bool(self.save_diagnostics)
        if isinstance(self.clean_steps, dict):
            self.clean_steps = SlimUvCleanSteps.from_dict(self.clean_steps)
        elif not isinstance(self.clean_steps, SlimUvCleanSteps):
            raise ValueError(
                f"clean_steps must be SlimUvCleanSteps or dict, "
                f"got {type(self.clean_steps).__name__}"
            )


# ---------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------

_REQUIRED_TOP_LEVEL_KEYS: tuple[str, ...] = (
    "session_id",
    "mesh_method",
    "max_edge_mm",
    "n_iter",
    "save_diagnostics",
    "clean_steps",
    "created_at",
    "modified_at",
)


def _make_yaml() -> YAML:
    yaml = YAML(typ="rt")
    yaml.preserve_quotes = True
    yaml.default_flow_style = False
    yaml.indent(mapping=2, sequence=4, offset=2)
    return yaml


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def config_path_for_session(output_dir: Path, session_id: str) -> Path:
    """Return ``<output_dir>/<session_id>/slim_uv_config.yaml``."""
    return Path(output_dir) / session_id / "slim_uv_config.yaml"


def load_slim_uv_config(path: Path) -> SlimUvConfig:
    """Load a per-session SLIM UV config YAML.

    Raises ``FileNotFoundError`` if the file does not exist, and ``ValueError``
    if any required field is missing or invalid (no silent defaults).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SLIM UV config not found: {path}")

    yaml = _make_yaml()
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.load(fh)
    if data is None:
        raise ValueError(f"SLIM UV config is empty: {path}")
    if not isinstance(data, dict):
        raise ValueError(
            f"SLIM UV config root must be a mapping, got "
            f"{type(data).__name__} in {path}"
        )

    missing = [k for k in _REQUIRED_TOP_LEVEL_KEYS if k not in data]
    if missing:
        raise ValueError(
            f"SLIM UV config is missing required key(s) {missing} in {path}"
        )

    clean_steps = SlimUvCleanSteps.from_dict(dict(data["clean_steps"]))

    return SlimUvConfig(
        session_id=str(data["session_id"]),
        mesh_method=str(data["mesh_method"]),
        max_edge_mm=float(data["max_edge_mm"]),
        n_iter=int(data["n_iter"]),
        save_diagnostics=bool(data["save_diagnostics"]),
        clean_steps=clean_steps,
        created_at=str(data["created_at"]),
        modified_at=str(data["modified_at"]),
    )


def save_slim_uv_config(path: Path, config: SlimUvConfig) -> None:
    """Write the SLIM UV config to ``path`` as round-trip YAML.

    Creates the parent directory if needed, stamps ``modified_at`` to now (UTC),
    and stamps ``created_at`` to now when it is empty on the input config.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    now = _utc_now_iso()
    if not config.created_at:
        config.created_at = now
    config.modified_at = now

    doc = CommentedMap()
    doc["session_id"] = config.session_id
    doc["mesh_method"] = config.mesh_method
    doc["max_edge_mm"] = float(config.max_edge_mm)
    doc["n_iter"] = int(config.n_iter)
    doc["save_diagnostics"] = bool(config.save_diagnostics)

    clean_steps_map = CommentedMap()
    for key, value in config.clean_steps.to_dict().items():
        clean_steps_map[key] = bool(value)
    doc["clean_steps"] = clean_steps_map

    doc["created_at"] = config.created_at
    doc["modified_at"] = config.modified_at

    yaml = _make_yaml()
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(doc, fh)


# ---------------------------------------------------------------------------
# Hashing and defaults
# ---------------------------------------------------------------------------

def config_hash(config: SlimUvConfig) -> str:
    """Deterministic SHA-256 hex digest of the config parameters.

    Excludes ``created_at`` and ``modified_at`` (timestamps are not part of
    the parameter identity). Includes ``session_id`` only insofar as it is a
    parameter field — for staleness checks two sessions with identical
    parameters and different ``session_id`` will produce different hashes.
    """
    payload = {
        "mesh_method": config.mesh_method,
        "max_edge_mm": float(config.max_edge_mm),
        "n_iter": int(config.n_iter),
        "save_diagnostics": bool(config.save_diagnostics),
        "clean_steps": config.clean_steps.to_dict(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def make_default_config(session_id: str, **dag_overrides: Any) -> SlimUvConfig:
    """Build a ``SlimUvConfig`` from DAG-level defaults plus optional overrides.

    Recognised kwargs (any subset): ``mesh_method``, ``max_edge_mm``, ``n_iter``,
    ``save_diagnostics``, ``clean_steps`` (dict or :class:`SlimUvCleanSteps`).
    Unknown kwargs raise ``ValueError`` — fail-fast rather than silently
    ignoring typos.
    """
    valid_kwargs = {
        "mesh_method",
        "max_edge_mm",
        "n_iter",
        "save_diagnostics",
        "clean_steps",
    }
    unknown = set(dag_overrides) - valid_kwargs
    if unknown:
        raise ValueError(
            f"make_default_config received unknown kwarg(s): {sorted(unknown)}. "
            f"Valid kwargs: {sorted(valid_kwargs)}"
        )

    mesh_method = dag_overrides.get("mesh_method", DEFAULT_MESH_METHOD)
    max_edge_mm = dag_overrides.get("max_edge_mm", DEFAULT_MAX_EDGE_MM)
    n_iter = dag_overrides.get("n_iter", DEFAULT_N_ITER)
    save_diagnostics = dag_overrides.get(
        "save_diagnostics", DEFAULT_SAVE_DIAGNOSTICS
    )

    if "clean_steps" in dag_overrides:
        steps = dag_overrides["clean_steps"]
        if isinstance(steps, SlimUvCleanSteps):
            clean_steps = SlimUvCleanSteps(**asdict(steps))
        else:
            clean_steps = SlimUvCleanSteps.from_dict(dict(steps))
    else:
        clean_steps = SlimUvCleanSteps(**asdict(DEFAULT_CLEAN_STEPS))

    now = _utc_now_iso()
    return SlimUvConfig(
        session_id=session_id,
        mesh_method=mesh_method,
        max_edge_mm=max_edge_mm,
        n_iter=n_iter,
        save_diagnostics=save_diagnostics,
        clean_steps=clean_steps,
        created_at=now,
        modified_at=now,
    )
