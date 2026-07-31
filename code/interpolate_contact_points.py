#!/usr/bin/env python
"""Geodesic interpolation of contact points across kinect->neural merge gaps.

When the kinect contact data is merged with the higher-rate neural recording,
each kinect frame lands on one row and the rows in between are empty (a gap of
~33 rows). This script fills those gaps by morphing the contact set from one
kinect frame to the next *along the forearm surface*.

Every interpolated point is an actual vertex of the forearm point cloud, so no
interpolated contact can ever fall outside the point-cloud domain (no cutting
through the arm, no floating points).

Method
------
1. Build a kNN graph over the forearm cloud S (edge weight = euclidean distance
   to the neighbour). Shortest paths through this graph approximate geodesic
   distance along the surface.
2. Snap the contacts of both endpoints onto S.
3. Dijkstra from every endpoint-1 contact -> geodesic cost matrix to the
   endpoint-2 contacts (predecessors kept so the actual paths are recoverable).
4. Match contacts with the Hungarian algorithm; leftovers on the larger side are
   attached to their geodesically nearest counterpart (splits/merges).
5. Walk each matched pair a fraction t along its path by arc length and snap to
   the vertex there. That vertex is the interpolated contact for that row.

Gap rule
--------
Interpolation only happens between two *consecutive* kinect samples that both
have contacts. Consecutive non-empty rows further apart than ``--max-gap``
imply that a kinect sample in between was genuinely empty (contact ended), so
that span is left empty rather than interpolated through.
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

CONTACT_COL = "contact_points"
DEFAULT_MAX_GAP = 33  # kinect->neural upsampling factor
DEFAULT_K = 10
_NO_PATH = -9999

Point = Tuple[float, float, float]


# --------------------------------------------------------------------- format
# Matches the pipeline's own serialisation: "[[x y z] [x y z] ...]" / "[]".

def parse_contact_points(cell) -> List[Point]:
    """Parse one ``contact_points`` cell into a list of (x, y, z) tuples."""
    if not isinstance(cell, str):
        return []
    stripped = cell.strip()
    if not stripped or stripped == "[]":
        return []

    points: List[Point] = []
    for match in re.findall(r"\[([^\]]+)\]", stripped):
        # lstrip("[") drops the outer bracket captured with the first triplet.
        parts = match.strip().lstrip("[").replace(",", " ").split()
        if len(parts) == 3:
            try:
                points.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                continue
    return points


def serialize_contact_points(points: Sequence[Point]) -> str:
    """Re-serialise (x, y, z) tuples back into the pipeline's cell format."""
    if len(points) == 0:
        return "[]"
    inner = " ".join(f"[{x} {y} {z}]" for x, y, z in points)
    return f"[{inner}]"


def load_forearm_vertices(ply_path: Path) -> np.ndarray:
    """Load the forearm point cloud S as an (N, 3) array."""
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(ply_path))
    vertices = np.asarray(pcd.points, dtype=np.float64)
    if vertices.size == 0:
        raise ValueError(f"Forearm PLY contains no points: {ply_path}")
    logger.info("Loaded forearm cloud: %d points from %s", len(vertices), ply_path.name)
    return vertices


# ---------------------------------------------------------------- surface graph

def build_surface_graph(vertices: np.ndarray, k: int = DEFAULT_K) -> csr_matrix:
    """Symmetric kNN graph over S, edge weight = euclidean distance."""
    n_points = len(vertices)
    k_eff = min(k, n_points - 1)
    if k_eff < 1:
        raise ValueError("Forearm cloud needs at least 2 points.")

    distances, indices = cKDTree(vertices).query(vertices, k=k_eff + 1)  # col 0 == self
    rows = np.repeat(np.arange(n_points), k_eff)
    cols = indices[:, 1:].ravel()
    weights = distances[:, 1:].ravel()

    graph = csr_matrix((weights, (rows, cols)), shape=(n_points, n_points))
    return graph.maximum(graph.T)  # undirected


def snap_to_surface(points: Sequence[Point], tree: cKDTree) -> np.ndarray:
    """Indices of the nearest surface vertex for each contact point."""
    if len(points) == 0:
        return np.empty(0, dtype=int)
    _, indices = tree.query(np.asarray(points, dtype=np.float64))
    return np.atleast_1d(indices).astype(int)


# ------------------------------------------------------------------ geodesics

def _reconstruct_path(predecessors: np.ndarray, source: int, target: int) -> Optional[List[int]]:
    """Vertex path source -> target from a Dijkstra predecessor row."""
    if target == source:
        return [source]

    path = [target]
    limit = len(predecessors) + 1
    while path[-1] != source:
        previous = predecessors[path[-1]]
        if previous < 0:  # unreachable (includes _NO_PATH)
            return None
        path.append(int(previous))
        if len(path) > limit:  # defensive: malformed predecessor chain
            return None
    path.reverse()
    return path


def _walk_fraction(path: Sequence[int], vertices: np.ndarray, t: float) -> int:
    """Vertex a fraction ``t`` along ``path``, measured by arc length."""
    if len(path) == 1:
        return path[0]

    points = vertices[list(path)]
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segments)])
    total = cumulative[-1]
    if total <= 0:
        return path[0]

    position = int(np.searchsorted(cumulative, t * total, side="left"))
    return path[min(max(position, 0), len(path) - 1)]


def _match_contacts(cost: np.ndarray) -> List[Tuple[int, int]]:
    """Pair endpoint-1 contacts to endpoint-2 contacts, covering every contact.

    Hungarian gives the optimal one-to-one core; whatever is left over on the
    larger side is attached to its cheapest counterpart, which reproduces the
    split/merge behaviour of optimal transport without the extra dependency.
    """
    n_rows, n_cols = cost.shape
    finite = np.isfinite(cost)
    fallback = cost[finite].max() * 10.0 if finite.any() else 1.0
    workable = np.where(finite, cost, fallback)

    rows, cols = linear_sum_assignment(workable)
    pairs = list(zip(rows.tolist(), cols.tolist()))

    for i in set(range(n_rows)) - set(rows.tolist()):
        pairs.append((i, int(np.argmin(workable[i, :]))))
    for j in set(range(n_cols)) - set(cols.tolist()):
        pairs.append((int(np.argmin(workable[:, j])), j))
    return pairs


def interpolate_between(
    vertices: np.ndarray,
    graph: csr_matrix,
    tree: cKDTree,
    start_points: Sequence[Point],
    end_points: Sequence[Point],
    n_inner: int,
) -> List[List[Point]]:
    """Morph ``start_points`` into ``end_points`` over ``n_inner`` rows."""
    if n_inner <= 0:
        return []

    sources = snap_to_surface(start_points, tree)
    targets = snap_to_surface(end_points, tree)
    if len(sources) == 0 or len(targets) == 0:
        return [[] for _ in range(n_inner)]

    distances, predecessors = dijkstra(
        graph, directed=False, indices=sources, return_predecessors=True
    )
    pairs = _match_contacts(distances[:, targets])

    paths: List[List[int]] = []
    for i, j in pairs:
        path = _reconstruct_path(predecessors[i], int(sources[i]), int(targets[j]))
        paths.append(path if path is not None else [int(sources[i])])

    frames: List[List[Point]] = []
    for step in range(1, n_inner + 1):
        t = step / (n_inner + 1)
        reached = {_walk_fraction(path, vertices, t) for path in paths}
        frames.append([tuple(vertices[i]) for i in sorted(reached)])
    return frames


# --------------------------------------------------------------------- driver

def interpolate_contact_column(
    csv_path: Path,
    vertices: np.ndarray,
    graph: csr_matrix,
    tree: cKDTree,
    column: str = CONTACT_COL,
    max_gap: int = DEFAULT_MAX_GAP,
) -> pd.DataFrame:
    """Return a single-column frame of aligned, interpolated contact points."""
    frame = pd.read_csv(csv_path)
    if column not in frame.columns:
        raise KeyError(f"'{column}' not found in {csv_path.name}. Columns: {list(frame.columns)}")

    parsed = [parse_contact_points(cell) for cell in frame[column]]
    output = ["[]"] * len(parsed)

    # Anchors are the rows that actually carry a kinect measurement.
    anchors = [i for i, points in enumerate(parsed) if points]
    for i in anchors:
        snapped = sorted(set(snap_to_surface(parsed[i], tree).tolist()))
        output[i] = serialize_contact_points([tuple(vertices[v]) for v in snapped])

    n_filled = n_skipped = 0
    for start, end in zip(anchors, anchors[1:]):
        n_inner = end - start - 1
        if n_inner <= 0:
            continue
        if (end - start) > max_gap:
            # A kinect sample in between was genuinely empty -> contact ended.
            n_skipped += 1
            continue

        for offset, points in enumerate(
            interpolate_between(vertices, graph, tree, parsed[start], parsed[end], n_inner),
            start=1,
        ):
            output[start + offset] = serialize_contact_points(points)
        n_filled += n_inner

    logger.info(
        "%s: %d anchors, %d rows interpolated, %d spans skipped (empty endpoint)",
        csv_path.name, len(anchors), n_filled, n_skipped,
    )
    return pd.DataFrame({f"{column}_interpolated": output})


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ply", required=True, type=Path,
                        help="Forearm point cloud (e.g. <session>_unified_registered.ply)")
    parser.add_argument("--contacts", required=True, nargs="+", type=Path,
                        help="One or more block CSVs containing the contact column")
    parser.add_argument("--outdir", required=True, type=Path, help="Output directory")
    parser.add_argument("--column", default=CONTACT_COL, help=f"Contact column (default: {CONTACT_COL})")
    parser.add_argument("--max-gap", type=int, default=DEFAULT_MAX_GAP,
                        help=f"Max row distance between consecutive kinect samples (default: {DEFAULT_MAX_GAP})")
    parser.add_argument("-k", type=int, default=DEFAULT_K, help=f"kNN neighbours (default: {DEFAULT_K})")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    vertices = load_forearm_vertices(args.ply)
    graph = build_surface_graph(vertices, k=args.k)
    tree = cKDTree(vertices)

    args.outdir.mkdir(parents=True, exist_ok=True)
    for csv_path in args.contacts:
        result = interpolate_contact_column(
            csv_path, vertices, graph, tree, column=args.column, max_gap=args.max_gap
        )
        destination = args.outdir / f"{csv_path.stem}_contact-interpolated.csv"
        result.to_csv(destination, index=False)
        logger.info("Wrote %s (%d rows)", destination.name, len(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
