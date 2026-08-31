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
3. Split each endpoint's contacts into contiguous patches (see below), and pair
   the patches of endpoint 1 with those of endpoint 2.
4. Dijkstra from every endpoint-1 contact -> geodesic cost matrix to the
   endpoint-2 contacts (predecessors kept so the actual paths are recoverable).
5. Match contacts with the Hungarian algorithm *within each paired patch*;
   leftovers on the larger side are attached to their geodesically nearest
   counterpart (splits/merges).
6. Walk each matched pair a fraction t along its path by arc length and snap to
   the vertex there. That vertex is the interpolated contact for that row.

Contact mode (one finger vs whole hand)
---------------------------------------
A one-finger touch puts a single blob on the arm; a whole hand puts down 2-4
separate finger patches with gaps between them. Matching contacts globally would
let points drift from one finger into the next, smearing material across those
gaps. So contacts are first clustered (single linkage at ``--cluster-eps``) and
the morph is confined to *corresponding* clusters.

Which regime a block is in is decided **once per block** from its early contact
frames (``--contact-mode auto``) and then held fixed:

* ``single`` - the whole patch is treated as one group (a lone finger patch can
  briefly break in two from sensor noise; clustering it would only add jitter).
* ``multi``  - every frame is clustered and clusters are paired before matching.
  When two clusters merge into one (or one splits in two) the shared cluster is
  divided between its claimants along a geodesic nearest-cluster boundary, so a
  merge closes the gap from both sides instead of teleporting points across it.

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
from collections import Counter
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

CONTACT_COL = "contact_points"
AREA_COL = "contact_area_metadata"  # optional; logged as a cross-check on the detected mode
DEFAULT_MAX_GAP = 33  # kinect->neural upsampling factor
DEFAULT_K = 10
DEFAULT_PRECISION = 1  # decimals; matches the input's ~0.1 mm precision
DEFAULT_TIME_COL = "time_nerve"  # dense (neural-rate) time column carried into the output
_NO_PATH = -9999

# Clustering. Contact points sit ~1 mm apart within a patch and the cloud's own
# vertices ~0.6 mm apart, while the gaps between fingers are several mm — 4 mm
# splits fingers reliably without fragmenting a single patch.
DEFAULT_CLUSTER_EPS = 4.0  # in the coordinate units of the data (mm here)
DEFAULT_MIN_CLUSTER = 3  # smaller specks are absorbed into their nearest neighbour cluster
DETECT_SAMPLE = 200  # early anchor frames inspected when auto-detecting the block's mode
DETECT_MIN_SHARE = 0.05  # a cluster must hold this share of a frame's points to count
DETECT_MIN_FRAC = 0.25  # ...in this fraction of the inspected frames -> "multi"

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


def serialize_contact_points(points: Sequence[Point], precision: Optional[int] = None) -> str:
    """Re-serialise (x, y, z) tuples back into the pipeline's cell format.

    If ``precision`` is given, coordinates are rounded to that many decimals —
    keeps the output small and matches the input's ~0.1 mm precision.
    """
    if len(points) == 0:
        return "[]"
    if precision is None:
        inner = " ".join(f"[{x} {y} {z}]" for x, y, z in points)
    else:
        inner = " ".join(
            f"[{round(x, precision)} {round(y, precision)} {round(z, precision)}]"
            for x, y, z in points
        )
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


def precompute_geodesics(graph: csr_matrix) -> Tuple[np.ndarray, np.ndarray]:
    """All-pairs geodesic distances + predecessors over S, computed once.

    For an N-vertex cloud this is an N x N float64 distance matrix and an N x N
    predecessor matrix (~165 MB at N=3718). Doing it once is far cheaper than a
    fresh Dijkstra per frame-pair, since contact patches carry tens-to-hundreds
    of source vertices each.
    """
    distances, predecessors = dijkstra(graph, directed=False, return_predecessors=True)
    logger.info("Precomputed all-pairs geodesics over %d vertices.", graph.shape[0])
    return distances, predecessors


def snap_to_surface(points: Sequence[Point], tree: cKDTree) -> np.ndarray:
    """Indices of the nearest surface vertex for each contact point."""
    if len(points) == 0:
        return np.empty(0, dtype=int)
    _, indices = tree.query(np.asarray(points, dtype=np.float64))
    return np.atleast_1d(indices).astype(int)


# ------------------------------------------------------------------- clusters
# One finger leaves a single patch; a whole hand leaves several with gaps in
# between. Clustering is done on the *raw* contact points (finer than the cloud)
# and the labels are carried over to the snapped vertices.

def _compact(labels: np.ndarray) -> np.ndarray:
    """Relabel to a contiguous 0..k-1 range."""
    _, compacted = np.unique(labels, return_inverse=True)
    return compacted.astype(int)


def cluster_points(
    points: Sequence[Point],
    eps: float = DEFAULT_CLUSTER_EPS,
    min_size: int = DEFAULT_MIN_CLUSTER,
) -> np.ndarray:
    """Label contact points into contiguous patches (single linkage at ``eps``).

    Two points join the same patch when they are within ``eps`` of each other,
    transitively — i.e. connected components of the eps-neighbourhood graph.
    Patches holding fewer than ``min_size`` points are absorbed into the nearest
    surviving patch so a stray speck never becomes a group of its own.
    """
    n_points = len(points)
    if n_points == 0:
        return np.empty(0, dtype=int)

    coords = np.asarray(points, dtype=np.float64)
    pairs = np.asarray(list(cKDTree(coords).query_pairs(eps)), dtype=int).reshape(-1, 2)
    graph = coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n_points, n_points)
    )
    _, labels = connected_components(graph, directed=False)

    sizes = np.bincount(labels)
    keep = np.flatnonzero(sizes >= min_size)
    if len(keep) == 0:
        return np.zeros(n_points, dtype=int)  # nothing substantial -> one group
    if len(keep) == len(sizes):
        return _compact(labels)

    stray = ~np.isin(labels, keep)
    solid = np.flatnonzero(~stray)
    _, nearest = cKDTree(coords[solid]).query(coords[stray])
    labels = labels.copy()
    labels[stray] = labels[solid[np.atleast_1d(nearest)]]
    return _compact(labels)


def cluster_vertex_sets(
    points: Sequence[Point],
    tree: cKDTree,
    eps: Optional[float],
    min_size: int = DEFAULT_MIN_CLUSTER,
) -> List[np.ndarray]:
    """Snap contacts onto S, grouped by patch: one unique-vertex array per patch.

    ``eps=None`` disables clustering — the whole contact set becomes one group,
    which is the single-finger behaviour.
    """
    if len(points) == 0:
        return []
    snapped = snap_to_surface(points, tree)
    if eps is None:
        return [np.unique(snapped)]
    labels = cluster_points(points, eps, min_size)
    # Distinct patches can share a vertex where the cloud is coarser than the
    # gap; harmless, the vertex is simply walked along both patches' paths.
    return [np.unique(snapped[labels == g]) for g in range(int(labels.max()) + 1)]


def detect_contact_mode(
    frames: Sequence[Sequence[Point]],
    eps: float = DEFAULT_CLUSTER_EPS,
    min_size: int = DEFAULT_MIN_CLUSTER,
    sample: int = DETECT_SAMPLE,
) -> str:
    """Decide ``'single'`` vs ``'multi'`` for a whole block from its early frames.

    Deliberately *not* re-decided per frame: the regime is a property of the
    block (one finger tip / whole hand), and a whole-hand frame legitimately
    collapses to one cluster at the start and end of a tap, when only part of
    the hand is down. Sampling many early frames rides over that.
    """
    inspected = [points for points in frames if points][:sample]
    if not inspected:
        return "single"

    n_multi = 0
    for points in inspected:
        sizes = np.bincount(cluster_points(points, eps, min_size))
        floor = max(min_size, DETECT_MIN_SHARE * len(points))
        if np.count_nonzero(sizes >= floor) >= 2:
            n_multi += 1

    share = n_multi / len(inspected)
    mode = "multi" if share >= DETECT_MIN_FRAC else "single"
    logger.info(
        "  cluster detection: %d/%d early frames hold >=2 patches (%.0f%%) -> %s",
        n_multi, len(inspected), 100 * share, mode,
    )
    return mode


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


def _nearest_share(
    members: np.ndarray,
    claimants: Sequence[np.ndarray],
    which: int,
    geo_dist: np.ndarray,
) -> np.ndarray:
    """The part of ``members`` geodesically closest to ``claimants[which]``.

    Used when several clusters are paired with the same counterpart (two fingers
    merging into one patch, or one splitting in two): the shared patch is divided
    along a geodesic nearest-cluster boundary so each claimant morphs only into
    its own side of it.
    """
    # Distance from every member to each claimant cluster = min over that cluster.
    distances = np.stack(
        [geo_dist[np.ix_(cluster, members)].min(axis=0) for cluster in claimants]
    )
    part = members[np.argmin(distances, axis=0) == which]
    if len(part) == 0:  # every claimant must still morph somewhere
        part = members[[int(np.argmin(distances[which]))]]
    return part


def _pair_clusters(
    sources: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    geo_dist: np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Pair start clusters with end clusters, splitting any that are shared."""
    if len(sources) == 1 and len(targets) == 1:
        return [(sources[0], targets[0])]

    # Cluster-level cost = mean geodesic distance between the two vertex sets.
    # Mean (rather than min) tracks patch *identity*: it stays large for two
    # fingers that happen to touch, so they are not confused with each other.
    cost = np.empty((len(sources), len(targets)), dtype=np.float64)
    for a, source in enumerate(sources):
        for b, target in enumerate(targets):
            block = geo_dist[np.ix_(source, target)]
            finite = block[np.isfinite(block)]
            cost[a, b] = finite.mean() if finite.size else np.inf

    pairs = _match_contacts(cost)
    source_claims = Counter(a for a, _ in pairs)
    target_claims = Counter(b for _, b in pairs)

    paired: List[Tuple[np.ndarray, np.ndarray]] = []
    for a, b in pairs:
        source, target = sources[a], targets[b]
        if target_claims[b] > 1:  # merge: several fingers -> one patch
            claimants = [x for x, y in pairs if y == b]
            target = _nearest_share(
                target, [sources[x] for x in claimants], claimants.index(a), geo_dist
            )
        if source_claims[a] > 1:  # split: one patch -> several fingers
            claimants = [y for x, y in pairs if x == a]
            source = _nearest_share(
                source, [targets[y] for y in claimants], claimants.index(b), geo_dist
            )
        paired.append((source, target))
    return paired


def interpolate_between(
    vertices: np.ndarray,
    geo_dist: np.ndarray,
    geo_pred: np.ndarray,
    tree: cKDTree,
    start_points: Sequence[Point],
    end_points: Sequence[Point],
    n_inner: int,
    cluster_eps: Optional[float] = None,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER,
) -> List[List[Point]]:
    """Morph ``start_points`` into ``end_points`` over ``n_inner`` rows.

    ``geo_dist`` / ``geo_pred`` are the precomputed all-pairs geodesic distance
    and predecessor matrices from :func:`precompute_geodesics`.

    With ``cluster_eps`` set (whole-hand blocks) the contacts of each endpoint
    are split into patches and the morph is confined to paired patches, so no
    point crosses the gap between two fingers. ``None`` treats the contacts as a
    single patch, which is the single-finger behaviour.
    """
    if n_inner <= 0:
        return []

    # A contact patch is a *set* of surface vertices — dedupe before matching.
    sources = cluster_vertex_sets(start_points, tree, cluster_eps, min_cluster_size)
    targets = cluster_vertex_sets(end_points, tree, cluster_eps, min_cluster_size)
    sources = [cluster for cluster in sources if len(cluster)]
    targets = [cluster for cluster in targets if len(cluster)]
    if not sources or not targets:
        return [[] for _ in range(n_inner)]

    paths: List[List[int]] = []
    for source, target in _pair_clusters(sources, targets, geo_dist):
        for i, j in _match_contacts(geo_dist[np.ix_(source, target)]):
            s, t = int(source[i]), int(target[j])
            path = _reconstruct_path(geo_pred[s], s, t)
            paths.append(path if path is not None else [s])

    frames: List[List[Point]] = []
    for step in range(1, n_inner + 1):
        fraction = step / (n_inner + 1)
        reached = {_walk_fraction(path, vertices, fraction) for path in paths}
        frames.append([tuple(vertices[i]) for i in sorted(reached)])
    return frames


# --------------------------------------------------------------------- driver

def interpolate_contact_column(
    csv_path: Path,
    vertices: np.ndarray,
    geo_dist: np.ndarray,
    geo_pred: np.ndarray,
    tree: cKDTree,
    column: str = CONTACT_COL,
    max_gap: int = DEFAULT_MAX_GAP,
    precision: Optional[int] = DEFAULT_PRECISION,
    time_col: Optional[str] = DEFAULT_TIME_COL,
    contact_mode: str = "auto",
    cluster_eps: float = DEFAULT_CLUSTER_EPS,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER,
) -> pd.DataFrame:
    """Return an aligned frame: a dense time column plus the interpolated contacts."""
    frame = pd.read_csv(csv_path, low_memory=False)
    if column not in frame.columns:
        raise KeyError(f"'{column}' not found in {csv_path.name}. Columns: {list(frame.columns)}")

    parsed = [parse_contact_points(cell) for cell in frame[column]]
    output = ["[]"] * len(parsed)

    # Anchors are the rows that actually carry a kinect measurement. Keep the
    # measured contacts as-is (only reformatted / rounded) — do NOT snap them;
    # snapping is applied only to the interpolated in-between rows.
    anchors = [i for i, points in enumerate(parsed) if points]
    for i in anchors:
        output[i] = serialize_contact_points(parsed[i], precision=precision)

    # One decision per block, held for the whole block.
    logger.info("%s: %d anchor frames", csv_path.name, len(anchors))
    mode = contact_mode
    if mode == "auto":
        mode = detect_contact_mode(parsed, cluster_eps, min_cluster_size)
    else:
        logger.info("  contact mode forced to '%s'", mode)
    if AREA_COL in frame.columns:  # cross-check only; never drives the algorithm
        recorded = frame[AREA_COL].dropna().unique()
        logger.info("  (%s says: %s)", AREA_COL, ", ".join(map(str, recorded[:4])) or "-")
    eps = cluster_eps if mode == "multi" else None

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
            interpolate_between(
                vertices, geo_dist, geo_pred, tree, parsed[start], parsed[end], n_inner,
                cluster_eps=eps, min_cluster_size=min_cluster_size,
            ),
            start=1,
        ):
            output[start + offset] = serialize_contact_points(points, precision=precision)
        n_filled += n_inner

    logger.info(
        "  %d rows interpolated, %d spans skipped (empty endpoint)", n_filled, n_skipped,
    )

    columns = {}
    if time_col is not None:
        if time_col in frame.columns:
            columns[time_col] = frame[time_col].to_numpy()  # dense timestamp per row
        else:
            logger.warning(
                "Time column '%s' not in %s; output will have no time column.",
                time_col, csv_path.name,
            )
    columns[f"{column}_interpolated"] = output
    return pd.DataFrame(columns)


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
    parser.add_argument("--precision", type=int, default=DEFAULT_PRECISION,
                        help=f"Decimals to round output coords (default: {DEFAULT_PRECISION}; -1 = full precision)")
    parser.add_argument("--time-col", default=DEFAULT_TIME_COL,
                        help=f"Dense time column to include as the first output column "
                             f"(default: {DEFAULT_TIME_COL}; 'none' to omit)")
    parser.add_argument("--contact-mode", choices=("auto", "single", "multi"), default="auto",
                        help="'single' = one contact patch per frame (one finger tip); "
                             "'multi' = several finger patches, morphed cluster-by-cluster "
                             "(whole hand); 'auto' (default) decides once per block")
    parser.add_argument("--cluster-eps", type=float, default=DEFAULT_CLUSTER_EPS,
                        help=f"Distance below which contact points join the same patch, in the "
                             f"data's units (default: {DEFAULT_CLUSTER_EPS} mm)")
    parser.add_argument("--min-cluster-size", type=int, default=DEFAULT_MIN_CLUSTER,
                        help=f"Patches smaller than this are absorbed into their nearest "
                             f"neighbour (default: {DEFAULT_MIN_CLUSTER} points)")
    args = parser.parse_args(argv)
    precision = None if args.precision is not None and args.precision < 0 else args.precision
    time_col = None if args.time_col.lower() == "none" else args.time_col

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    vertices = load_forearm_vertices(args.ply)
    graph = build_surface_graph(vertices, k=args.k)
    geo_dist, geo_pred = precompute_geodesics(graph)
    tree = cKDTree(vertices)

    args.outdir.mkdir(parents=True, exist_ok=True)
    for csv_path in args.contacts:
        result = interpolate_contact_column(
            csv_path, vertices, geo_dist, geo_pred, tree,
            column=args.column, max_gap=args.max_gap, precision=precision,
            time_col=time_col, contact_mode=args.contact_mode,
            cluster_eps=args.cluster_eps, min_cluster_size=args.min_cluster_size,
        )
        destination = args.outdir / f"{csv_path.stem}_contact-interpolated.csv"
        result.to_csv(destination, index=False)
        logger.info("Wrote %s (%d rows)", destination.name, len(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
