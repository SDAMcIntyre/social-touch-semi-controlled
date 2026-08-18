"""The provenance sidecar that identifies a deduplicated forearm PLY.

A vertex index into the deduplicated forearm PLY is only meaningful relative to
the epsilon that produced that PLY: change epsilon and every vertex is
renumbered.  The DAG config is not part of the mtime-based staleness check
(``utils/should_process_task.py``), and under ``monitor: true`` the epsilon is
chosen interactively and never reaches the config at all.  This sidecar is the
only place the *effective* epsilon and the resulting vertex count survive, so a
downstream consumer can prove which PLY a vertex index belongs to.

Why this is a leaf and not part of the dedup stage script
---------------------------------------------------------
Two stage scripts need it and they are forbidden to import one another:
``deduplicate_xy_points`` **writes** it beside the PLY it just produced, and
``project_contacts_onto_forearm`` **reads** it to stamp ``reference_ply``,
``reference_ply_vertex_count`` and ``dedup_epsilon`` onto the depth field it
assigns ``vertex_id`` to.  Putting the format in one of them would make the
other depend on it; putting it here makes both depend on a leaf.  The module is
deliberately free of Open3D, pandas and pipeline knowledge — it is a JSON file
format and nothing else.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

__all__ = [
    "FOREARM_DEDUP_METADATA_SUFFIX",
    "FOREARM_DEDUP_METADATA_SCHEMA_VERSION",
    "EPSILON_SOURCE_DAG_CONFIG",
    "EPSILON_SOURCE_INTERACTIVE_MONITOR",
    "EPSILON_SOURCES",
    "ForearmDedupMetadata",
    "forearm_dedup_metadata_path",
    "write_forearm_dedup_metadata",
    "read_forearm_dedup_metadata",
]

FOREARM_DEDUP_METADATA_SUFFIX = "_dedup_metadata.json"
FOREARM_DEDUP_METADATA_SCHEMA_VERSION = "1"

#: Epsilon came from the DAG config's ``deduplicate_xy.epsilon`` option.
EPSILON_SOURCE_DAG_CONFIG = "dag_config"
#: Epsilon was chosen by the operator in the interactive monitor viewer.
EPSILON_SOURCE_INTERACTIVE_MONITOR = "interactive_monitor"

EPSILON_SOURCES = frozenset({
    EPSILON_SOURCE_DAG_CONFIG,
    EPSILON_SOURCE_INTERACTIVE_MONITOR,
})


@dataclass(frozen=True)
class ForearmDedupMetadata:
    """The parsed contents of one deduplicated-forearm provenance sidecar.

    Attributes:
        path: Where it was read from, so a failure downstream can name the file.
        schema_version: The sidecar format version it declares.
        source_ply: Filename of the PLY that was deduplicated.
        deduplicated_ply: Filename of the PLY this sidecar describes.
        dedup_epsilon: The epsilon actually applied — not the configured
            default, and not necessarily what the DAG config says today.
        epsilon_source: One of :data:`EPSILON_SOURCES`.
        n_vertices_original: Vertex count before deduplication.
        n_vertices_deduped: Vertex count after — the number a ``vertex_id`` must
            fall below.
        n_vertices_removed: How many collapsed away.
    """

    path: Path
    schema_version: str
    source_ply: str
    deduplicated_ply: str
    dedup_epsilon: float
    epsilon_source: str
    n_vertices_original: int
    n_vertices_deduped: int
    n_vertices_removed: int


def forearm_dedup_metadata_path(deduped_ply: Path) -> Path:
    """Return the path of the provenance sidecar for a deduplicated forearm PLY.

    Producers and consumers must both go through this helper so the sidecar is
    always found next to the PLY it describes.

    Args:
        deduped_ply: Path of the deduplicated forearm PLY.

    Returns:
        Path of the JSON sidecar (same directory, same stem).
    """
    deduped_ply = Path(deduped_ply)
    return deduped_ply.with_name(deduped_ply.stem + FOREARM_DEDUP_METADATA_SUFFIX)


def write_forearm_dedup_metadata(
    deduped_ply: Path,
    *,
    source_ply: Path,
    epsilon: float,
    epsilon_source: str,
    stats: Mapping[str, Any],
) -> Path:
    """Record the effective dedup epsilon and vertex counts beside the deduped PLY.

    Written as a deterministic JSON sidecar (sorted keys, no timestamps) so an
    unchanged re-run produces a byte-identical file.

    Args:
        deduped_ply: Path of the deduplicated forearm PLY this describes.
        source_ply: Path of the PLY that was deduplicated.
        epsilon: The epsilon actually applied — not the configured default.
        epsilon_source: One of EPSILON_SOURCES, saying where that value came from.
        stats: The dict returned by deduplicate_forearm_ply.

    Returns:
        Path of the written sidecar.

    Raises:
        ValueError: If epsilon is not a positive finite value, epsilon_source is
            unknown, stats is missing a required key, or the vertex counts do
            not satisfy n_deduped + n_removed == n_original.
    """
    deduped_ply = Path(deduped_ply)
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(
            f"dedup epsilon must be a positive finite value, got {epsilon!r} "
            f"(writing metadata for {deduped_ply})"
        )

    if epsilon_source not in EPSILON_SOURCES:
        raise ValueError(
            f"Unknown epsilon_source {epsilon_source!r}; expected one of "
            f"{sorted(EPSILON_SOURCES)}"
        )

    required = ("n_original", "n_deduped", "n_removed")
    missing = [key for key in required if key not in stats]
    if missing:
        raise ValueError(
            f"dedup stats is missing required key(s) {missing} "
            f"(writing metadata for {deduped_ply})"
        )

    n_original = int(stats["n_original"])
    n_deduped = int(stats["n_deduped"])
    n_removed = int(stats["n_removed"])
    if n_deduped + n_removed != n_original:
        raise ValueError(
            f"Inconsistent dedup vertex counts: {n_deduped} + {n_removed} "
            f"!= {n_original} (writing metadata for {deduped_ply})"
        )

    metadata = {
        "schema_version": FOREARM_DEDUP_METADATA_SCHEMA_VERSION,
        "source_ply": Path(source_ply).name,
        "deduplicated_ply": deduped_ply.name,
        "dedup_epsilon": epsilon,
        "epsilon_source": epsilon_source,
        "n_vertices_original": n_original,
        "n_vertices_deduped": n_deduped,
        "n_vertices_removed": n_removed,
    }

    metadata_path = forearm_dedup_metadata_path(deduped_ply)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata_path


def read_forearm_dedup_metadata(deduped_ply: Path) -> ForearmDedupMetadata:
    """Read and validate the provenance sidecar beside *deduped_ply*.

    Every field is parsed, not merely fetched: a sidecar whose epsilon is a
    string, or whose counts do not add up, describes a PLY nobody can prove
    anything about, and stamping an unverified value onto a ``vertex_id``
    artifact is worse than not producing the artifact at all.

    Args:
        deduped_ply: Path of the deduplicated forearm PLY.  The sidecar is
            located beside it by :func:`forearm_dedup_metadata_path`.

    Returns:
        The parsed :class:`ForearmDedupMetadata`.

    Raises:
        FileNotFoundError: If the sidecar does not exist.  It is written by the
            dedup stage; its absence means the PLY predates that stage or was
            produced outside the pipeline, and in either case the epsilon that
            numbered its vertices is unknown.
        ValueError: If the JSON is malformed, declares an unknown
            ``schema_version``, omits a key, or carries a value that is not the
            number or the vocabulary member it claims to be.
    """
    deduped_ply = Path(deduped_ply)
    metadata_path = forearm_dedup_metadata_path(deduped_ply)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Deduplicated-forearm provenance sidecar not found: {metadata_path}. "
            f"It is written beside {deduped_ply.name} by the deduplicate_xy stage "
            "and records the epsilon that numbered the PLY's vertices. Without it "
            "a vertex_id cannot be attributed to a specific mesh; re-run the "
            "deduplicate_xy task for this session rather than guessing the epsilon."
        )

    try:
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{metadata_path} is not valid JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(
            f"{metadata_path} holds a {type(raw).__name__}, not a JSON object."
        )

    version = raw.get("schema_version")
    if version != FOREARM_DEDUP_METADATA_SCHEMA_VERSION:
        raise ValueError(
            f"{metadata_path} declares schema_version={version!r}; this module "
            f"reads {FOREARM_DEDUP_METADATA_SCHEMA_VERSION!r} only."
        )

    required = (
        "source_ply",
        "deduplicated_ply",
        "dedup_epsilon",
        "epsilon_source",
        "n_vertices_original",
        "n_vertices_deduped",
        "n_vertices_removed",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"{metadata_path} is missing key(s) {missing}.")

    epsilon = _require_positive_finite_float(raw, "dedup_epsilon", metadata_path)
    epsilon_source = raw["epsilon_source"]
    if epsilon_source not in EPSILON_SOURCES:
        raise ValueError(
            f"{metadata_path} declares epsilon_source={epsilon_source!r}; "
            f"expected one of {sorted(EPSILON_SOURCES)}."
        )

    counts: Dict[str, int] = {
        key: _require_non_negative_int(raw, key, metadata_path)
        for key in ("n_vertices_original", "n_vertices_deduped", "n_vertices_removed")
    }
    if counts["n_vertices_deduped"] + counts["n_vertices_removed"] != counts[
        "n_vertices_original"
    ]:
        raise ValueError(
            f"{metadata_path} has inconsistent vertex counts: "
            f"{counts['n_vertices_deduped']} + {counts['n_vertices_removed']} "
            f"!= {counts['n_vertices_original']}."
        )

    return ForearmDedupMetadata(
        path=metadata_path,
        schema_version=str(version),
        source_ply=str(raw["source_ply"]),
        deduplicated_ply=str(raw["deduplicated_ply"]),
        dedup_epsilon=epsilon,
        epsilon_source=str(epsilon_source),
        n_vertices_original=counts["n_vertices_original"],
        n_vertices_deduped=counts["n_vertices_deduped"],
        n_vertices_removed=counts["n_vertices_removed"],
    )


def _require_positive_finite_float(raw: Mapping[str, Any], key: str, path: Path) -> float:
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{path} key {key!r} is {value!r}, not a number."
        )
    as_float = float(value)
    if not np.isfinite(as_float) or as_float <= 0.0:
        raise ValueError(
            f"{path} key {key!r} is {value!r}; a positive finite value is required."
        )
    return as_float


def _require_non_negative_int(raw: Mapping[str, Any], key: str, path: Path) -> int:
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{path} key {key!r} is {value!r}, not an integer."
        )
    if value < 0:
        raise ValueError(
            f"{path} key {key!r} is {value!r}; a non-negative count is required."
        )
    return int(value)
