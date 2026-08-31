# CLAUDE.md — contact_point_interpolation

Context for Claude Code sessions opened in this folder. Read this first.

## What this repo is
Standalone tool that fills the ~33-row gaps in `contact_points` created when the
kinect contact data is merged with the higher-rate neural recording. It morphs the
contact patch from one kinect frame to the next **along the forearm point-cloud
surface** (kNN-graph geodesics), so no interpolated point leaves the forearm.
`code/interpolate_contact_points.py` is the tool; `interpolate_test.py` /
`interpolate_video.py` are verification.

## Git setup (important — non-obvious)
- This is its **own git repo**, but `origin` points at a branch of the lab repo:
  - remote `origin` = `https://github.com/LHTMR/social-touch-semi-controlled.git`
  - branch = **`dev-haonan-interpolate-contact-point`** — an **orphan branch** that
    contains *only this repo* (unrelated history to the lab pipeline). This is
    intentional: the developer applies it to their workflow.
- **Never touch `main` or `dev`** of that lab repo. Only push this one branch.
- Commit + push from here:
  ```bash
  git add <files>
  git commit -m "..."
  git push origin dev-haonan-interpolate-contact-point
  ```
- Local git identity is set for this repo: `qsm9za <qsm9za@virginia.edu>`.
- `core.autocrlf=true` on this machine, so binaries are protected by
  `.gitattributes` (`*.ply *.obj *.pkl *.mp4 *.png` = `binary`). Keep that — without
  it, git line-ending conversion **corrupts** the point-cloud `.ply` files.
- The lab repo moved from `SDAMcIntyre/…` to `LHTMR/…`; the old URL still redirects.

## Layout & what's committed
```
code/    the tool + the two verification scripts        (committed)
data/    2022-06-14_ST13-01/ example session            (committed, ~54 MB incl. inputs)
plot/    verification PNGs                               (committed, small)
```
Git-ignored (regenerable — do NOT commit, they are large): `output/` (interpolated
CSVs, ~1.5 GB) and `plot/videos/` (~11 MB of mp4s).

## Environment — READ before plotting/rendering
Python: `D:/conda_envs/social-touch/python.exe` (numpy 2.0, opencv 4.12, Pillow ok).

**This env has native DLL conflicts. Two things segfault (exit 127, no traceback):**
1. **numpy `@` / `np.dot` (BLAS matmul)** — use element-wise math instead.
2. **matplotlib rendering** (`canvas.draw` / `savefig`, any backend) — the transforms
   use matmul. Do **not** use matplotlib for output. Its *colormap data* (LUT lookup)
   is safe.

So: plots are rendered with **Pillow**, videos with **OpenCV** (`mp4v` codec — `avc1`
/ H.264 fails, no OpenH264 dll), and all 3D projection is hand-rolled element-wise.
The interpolation tool itself is fine (it never uses matmul).
Also: interpreter launches can intermittently fail with 127 under heavy concurrent
CPU/RAM load — run one heavy job at a time.

## Run
```bash
PY="D:/conda_envs/social-touch/python.exe"
# interpolate (all 4 example blocks) — ~9 min, all-pairs geodesics + dense patches
$PY code/interpolate_contact_points.py \
    --ply data/2022-06-14_ST13-01/forearm_pointclouds/2022-06-14_ST13-01_unified_registered.ply \
    --contacts data/2022-06-14_ST13-01/*_merged_data.csv --outdir output --max-gap 50
$PY code/interpolate_test.py     # plot/*.png  (3x2 per block: original vs interpolated)
$PY code/interpolate_video.py    # plot/videos/*.mp4  (3D evolution, original vs interp)
```

## Key facts (verified)
- **Scaffold PLY** = `…_unified_registered.ply` (per-experiment, 3718 pts, has normals).
  Contacts in the `*_merged_data.csv` are in the **same registered frame** — they snap
  to it at ~0.4–0.6 mm median (checked). The per-block `…_frames_*.ply` are per-frame
  extractions, not the scaffold.
- **`--max-gap 50`**: the single-kinect-frame gap is 33 **or** 34 rows (jitter), so the
  default 33 is too strict. A gap ≥ ~66 means an intervening kinect frame was empty →
  not interpolated (correct).
- **Contact mode**: blocks are either `one finger tip` (1 contact patch/frame) or
  `whole hand` (2–4 finger patches with gaps). The tool clusters contacts (single
  linkage, 4 mm) and confines the morph to *corresponding* patches, otherwise points
  drift between fingers (was 7.9% of matches on block-02, median 12 mm). Mode is
  detected **once per block** from the first 200 anchor frames — not per frame, since
  a whole-hand frame collapses to 1 patch at tap onset/offset. `single` mode is the
  pre-cluster code path, bit-for-bit (verified on 12.6k rows). Detection agrees with
  `contact_area_metadata` on all 4 example blocks; that column is logged as a
  cross-check but never drives the algorithm. Override: `--contact-mode single|multi`.
- **Output** = 2 columns: `time_nerve` (dense, for aligning to the nerve signal) +
  `contact_points_interpolated`. Same row count as input. Anchor (measured) rows keep
  the **original** contacts; only interpolated in-between rows are snapped to the cloud.
  Coords rounded to 1 decimal (matches input; source is genuinely 0.1 mm, not integer).
