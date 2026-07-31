# contact_point_interpolation

Standalone tool for filling the empty rows that appear in `contact_points` after the
kinect contact data is merged with the higher-rate neural recording (one kinect frame
lands on a row, the next ~32 rows are empty). Self-contained — independent of the main
pipeline.

**Guarantee:** every interpolated point is an actual vertex of the forearm point cloud,
so no interpolated contact can fall outside the point-cloud domain or cut through the arm.

## Layout

```
code/   interpolate_contact_points.py   the tool
data/   2022-06-14_ST13-01/             example session (forearm PLYs)
plot/   (visualisations, added later)
```

## Method

Contacts are morphed from one kinect frame to the next *along the forearm surface*:

1. kNN graph over the forearm cloud `S` (edge weight = euclidean distance to the
   neighbour) — shortest paths through it approximate geodesic distance on the surface.
2. Contacts of both endpoints are snapped onto `S`.
3. Dijkstra from every endpoint-1 contact gives the geodesic cost matrix to the
   endpoint-2 contacts, keeping predecessors so the paths can be rebuilt.
4. Contacts are paired with the Hungarian algorithm; leftovers on the larger side attach
   to their cheapest counterpart (contact splits / merges).
5. Each pair is walked a fraction `t` along its path by arc length, snapping to the
   vertex there.

Using geodesic rather than straight-line distance is what keeps points on the surface —
a contact never tunnels through the forearm.

## Gap rule

Interpolation happens **only between two consecutive kinect samples that both have
contacts**. If consecutive non-empty rows are further apart than `--max-gap`, a kinect
sample in between was genuinely empty (contact ended), so that span is left empty
instead of being interpolated through.

## Which PLY

Use the session-level cloud, e.g. `data/2022-06-14_ST13-01/forearm_pointclouds/2022-06-14_ST13-01_unified_registered.ply`
(one per experiment, ~3.7k points, with normals). The per-block files
(`..._block-orderNN_kinect_frames_*.ply`) are individual per-frame extractions; the
unified file is the registered forearm-of-reference for the whole session.

> The PLY and the contact CSV must be in the **same coordinate frame**. `_unified_registered.ply`
> is in the registered (pre-PCA) frame — if your contacts are PCA-calibrated, use the
> matching `forearm_pca_calibrated` PLY instead.

## Usage

```bash
python code/interpolate_contact_points.py \
    --ply      data/2022-06-14_ST13-01/forearm_pointclouds/2022-06-14_ST13-01_unified_registered.ply \
    --contacts path/to/block-order01_merged.csv path/to/block-order02_merged.csv \
    --outdir   out/
```

Options: `--column` (default `contact_points`), `--max-gap` (default `33`), `-k` (default `10`).

## Output

One CSV per input block, `<input-stem>_contact-interpolated.csv`, containing a **single
column** `contact_points_interpolated` — same row count and same cell format as the input
(`[[x y z] [x y z] ...]`, or `[]`).

## Requirements

`numpy`, `pandas`, `scipy`, `open3d`
