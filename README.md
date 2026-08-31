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
3. Each endpoint's contacts are split into contiguous patches, and the patches of one
   endpoint are paired with those of the other (see **Contact mode** below).
4. Dijkstra from every endpoint-1 contact gives the geodesic cost matrix to the
   endpoint-2 contacts, keeping predecessors so the paths can be rebuilt.
5. Contacts are paired with the Hungarian algorithm *within each paired patch*; leftovers
   on the larger side attach to their cheapest counterpart (contact splits / merges).
6. Each pair is walked a fraction `t` along its path by arc length, snapping to the
   vertex there.

Using geodesic rather than straight-line distance is what keeps points on the surface —
a contact never tunnels through the forearm.

## Contact mode (one finger vs whole hand)

A one-finger touch leaves a single blob on the arm. A whole hand leaves **2–4 separate
finger patches** with gaps between them, and those patches merge and split from frame to
frame. Matching all contacts in one pool lets points drift out of one finger and into the
next, smearing material across the gaps — on this dataset **7.9 %** (tap) / **3.6 %**
(stroke) of all matched pairs jumped between fingers, travelling a median of 10–12 mm and
up to 76 mm across the arm.

So contacts are clustered (single linkage at `--cluster-eps`, default 4 mm) and the morph
is confined to *corresponding* clusters. Which regime a block is in is decided **once per
block** from its early contact frames and then held fixed:

| mode | behaviour |
|---|---|
| `single` | the whole patch is one group. A lone finger patch can briefly break in two from sensor noise; clustering it would only add jitter. |
| `multi` | every frame is clustered, and clusters are paired before contacts are matched. |

When two clusters **merge** into one (or one **splits** into two), the shared cluster is
divided between its claimants along a geodesic nearest-cluster boundary — a merge closes
the gap from both sides instead of teleporting points across it.

Detection uses the first 200 anchor frames and asks how often a frame holds ≥ 2 substantial
patches (≥ 5 % of the frame's points): ≥ 25 % of frames → `multi`. It reads the regime off
many frames rather than the first one because a whole-hand frame legitimately collapses to
one patch at the very start and end of a tap, when only part of the hand is down. Override
with `--contact-mode single|multi`.

On this session the detector agrees with the recorded `contact_area_metadata` on all four
blocks; that column is logged next to the decision as a cross-check but never drives it
(other datasets may not carry it).

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

For the merged ST13-01 blocks, use `--max-gap 50` (the single-frame gap is 33 **or** 34,
so 33 is slightly too strict). Larger gaps mean an intervening kinect frame was empty.

Options: `--column` (default `contact_points`), `--max-gap` (default `33`), `-k` (default
`10`), `--precision` (default `1` decimal; `-1` for full precision), `--time-col` (default
`time_nerve`; `none` to omit), `--contact-mode` (`auto` / `single` / `multi`, default
`auto`), `--cluster-eps` (default `4.0`, in the data's units), `--min-cluster-size`
(default `3` points).

## Output

One CSV per input block, `<input-stem>_contact-interpolated.csv`, with two columns:

| column | |
|---|---|
| `time_nerve` | the dense neural-rate timestamp for each row (for easy alignment with the nerve signal) |
| `contact_points_interpolated` | same row count and cell format as the input (`[[x y z] ...]` or `[]`) |

Same row count as the input; the time column is copied straight from the merged CSV so the
output lines up row-for-row with the neural data. Use `--time-col none` for contacts only.

- **Anchor rows** (kinect frames that carry a measurement) keep the **original measured
  contacts** (only rounded), unchanged.
- **Interpolated rows** are snapped to the forearm cloud, so they lie on the surface.
- Coordinates are rounded to `--precision` decimals (default 1 = 0.1 mm, matching the
  input) to keep files reasonable — full-precision output is several × larger.

Outputs are large (hundreds of MB per block) and are **regenerated by running the tool**,
so they are git-ignored rather than committed.

## Verification

`code/interpolate_test.py` renders a quick visual check: for each block it picks a few
random single touches (`single_touch_id`) and plots the xy contact cloud during each touch,
**original (left) vs interpolated (right)**, coloured by relative time within the touch.
One figure per block in `plot/`.

```bash
python code/interpolate_test.py            # uses data/, output/, writes to plot/
```

The interpolated panel should sit in the same footprint as the original (on-surface), be
denser, and show a continuous time sweep. Committed examples are in `plot/`.

`code/interpolate_video.py` additionally writes small 3D **videos** — for each block it
picks 5 random touches and, per touch, two mp4s (`_1_original`, `_2_interpolated`) showing
the contact patch evolving inside a fixed 3D grid box, with an accumulating trail. The
original stair-steps between the few kinect frames; the interpolated glides continuously.

```bash
python code/interpolate_video.py           # writes plot/videos/*.mp4 (git-ignored)
```

> Both scripts avoid matplotlib and numpy `@`/`dot`: in this conda env both the matplotlib
> rasteriser **and** numpy's BLAS matmul segfault (a machine-level native DLL conflict,
> unrelated to this code). Plots render via Pillow, videos via OpenCV (mp4v), and all 3D
> projection is done with element-wise math. The interpolation tool itself is unaffected
> (it uses no matmul). Videos are regenerable, so they are git-ignored.

## Requirements

`numpy`, `pandas`, `scipy`, `open3d` (tool); plus `Pillow` (plots) and `opencv-python` (videos)
