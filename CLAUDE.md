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

## Environment
Python: `D:/conda_envs/social-touch/python.exe` (python 3.10, numpy 2.0, scipy 1.15,
matplotlib 3.10, pyarrow 23, open3d 0.19, opencv 4.13, Pillow 11).
Package set is pinned in `environment.yml`.

**numpy must stay a pip wheel — do not `conda install numpy` here.** Everything in
this env except numpy is a pip wheel with its own bundled OpenBLAS. The conda-forge
numpy instead links the env's shared `libblas` shim, and that pairing had drifted:
numpy 2.0.0 (built against `libblas >=3.9.0`) was running against `libblas 3.11.0` →
`mkl 2025.3.0`. The version constraint still matched so conda allowed it, but the
binary segfaulted (exit 127, no traceback) the moment it called BLAS — `a @ b`,
2-D `np.dot`, `np.linalg.*`, and therefore **every** matplotlib transform, 2-D and
3-D alike. Telltale: `scipy.linalg.inv` worked while `np.linalg.inv` crashed.

Fixed 2026-09-04 with `pip install --force-reinstall --no-deps numpy==2.0.0`, which
brings numpy's own OpenBLAS. Verified afterwards: matmul, `np.linalg`, matplotlib 2-D
and 3-D all fine, and the interpolation output is unchanged (12,123 rows re-checked
cell-for-cell, both single and multi mode). To roll back:
`conda install -n social-touch numpy=2.0.0=py310h1ec8c79_0`.

**matplotlib now works**, but the verification scripts still render with **Pillow**
(plots) and **OpenCV** (videos, `mp4v` — `avc1`/H.264 fails, no OpenH264 dll), with 3D
projection hand-rolled element-wise. That code predates the fix and works; it is not
worth rewriting. New scripts may use matplotlib freely.

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
