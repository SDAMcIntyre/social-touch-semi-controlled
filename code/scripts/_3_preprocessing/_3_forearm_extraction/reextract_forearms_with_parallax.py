"""
Re-extract forearm point clouds for every snapshot listed in a forearm
DAG config, using the parallax-corrected ``KinectFrame``.

Why this script exists
----------------------
The parallax correction (see
``docs/development/plans/active/kinect-frame-parallax-correction.md``
and the KB note
``docs/development/knowledge-base/note-azure-kinect-rgb-depth-parallax.md``)
is now applied by default inside ``KinectFrame``. Every forearm point
cloud extracted **before** that change paired XYZ points with the wrong
RGB pixel, so the HSV skin-colour filter in ``ArmSegmentation`` selected
a subset of points that is not the same as the subset the corrected
pairing would select. A post-hoc transform of the saved ``.ply`` cannot
reproduce the corrected result — the skin-filter decision is frozen into
the file. The only correct fix is to re-run ``extract_forearm`` against
the MKV with the saved ROI / HSV parameters.

Scope
-----
This is a one-shot migration tool. It re-runs **only** stage 1
(``extract_forearm``) non-interactively, using the already-saved
``{session_id}_arm_roi_metadata.json`` and per-snapshot
``*_extraction_params.json``. Downstream artefacts (``*_curated.ply``,
``*_cleaned.ply``, ``*_with_normals.ply``, ``*_mesh.obj``) become stale;
they are listed at the end of the run so the user can decide whether to
re-run the manual pipeline. Curation is interactive-only and is
therefore not automated here.

Usage
-----
    python reextract_forearms_with_parallax.py \\
        --dag-config configs/preprocess_pipeline_extract_forearm_manual_dag.yaml

    # Preview without writing anything:
    python ... --dry-run

Backup
------
Raw ``.ply`` files in ``forearm_pointclouds/`` are overwritten. Before a
large batch, copy the folders you want to preserve, e.g.::

    robocopy <pointclouds_dir> <pointclouds_dir>_preparallax /E /XO
"""

# CuPy must precede preprocessing imports — see CLAUDE.md.
try:
    import cupy  # noqa: F401
except Exception:
    pass

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure code/src and code/scripts are importable when launched directly
# (e.g. from a debugger without PYTHONPATH set).
_CODE_ROOT = Path(__file__).resolve().parents[3]
for _p in (_CODE_ROOT / "src", _CODE_ROOT / "scripts"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import utils.path_tools as path_tools
from utils import DagConfigHandler
from utils.pipeline.session_config_resolver import resolve_session_configs
from primary_processing import (
    ForearmConfigFileHandler,
    ForearmConfig,
    KinectConfigFileHandler,
    KinectConfig,
)
from preprocessing.forearm_extraction import (
    ForearmFrameParametersFileHandler,
    ForearmParameters,
)
from _3_preprocessing._3_forearm_extraction import extract_forearm


DOWNSTREAM_SUFFIXES = (
    "_curated.ply",
    "_curation_metadata.json",
    "_cleaned.ply",
    "_cleaning_stats.json",
    "_with_normals.ply",
    "_with_normals_metadata.json",
    "_mesh.obj",
)


def _resolve_rgb_video_paths(
    config_links: List[str], project_data_root: Path
) -> List[Path]:
    rgb_paths: List[Path] = []
    for link in config_links:
        try:
            config_data = KinectConfigFileHandler.load_and_resolve_config(link)
            kc = KinectConfig(config_data=config_data, database_path=project_data_root)
            rgb = kc.source_video.with_suffix(".mp4")
            if not rgb.exists():
                print(f"  ⚠️  RGB video not found, skipping: {rgb}")
                continue
            rgb_paths.append(rgb)
        except Exception as exc:
            print(f"  ❌ Failed to load Kinect config '{link}': {exc}")
    return rgb_paths


def _process_session(
    session_file: Path,
    project_data_root: Path,
    dry_run: bool,
    summary: dict,
    stale_files: List[Path],
) -> None:
    cfg: ForearmConfig = ForearmConfigFileHandler.load(session_file)
    pc_dir = cfg.session_processed_path / "forearm_pointclouds"
    meta = pc_dir / f"{cfg.session_id}_arm_roi_metadata.json"

    if not meta.exists():
        print(f"  ⚠️  No metadata at {meta} — nothing to re-extract; skipping session.")
        summary["sessions_skipped_no_meta"] += 1
        return

    rgb_paths = _resolve_rgb_video_paths(cfg.config_file_links, project_data_root)
    if not rgb_paths:
        print("  ⚠️  No RGB videos resolved — skipping session.")
        summary["sessions_skipped_no_rgb"] += 1
        return

    frame_params: List[ForearmParameters] = ForearmFrameParametersFileHandler.load(str(meta))
    if not frame_params:
        print(f"  ⚠️  Metadata at {meta} has no entries; skipping session.")
        summary["sessions_skipped_no_meta"] += 1
        return

    summary["sessions_processed"] += 1
    print(f"  📋 {len(frame_params)} snapshot(s) to re-extract.")

    for params in frame_params:
        rgb = next((p for p in rgb_paths if p.name == params.video_filename), None)
        if rgb is None:
            print(f"  ⚠️  No RGB match for {params.video_filename}; skipping frame.")
            summary["frames_skipped_no_video"] += 1
            continue

        mkv = rgb.with_suffix(".mkv")
        if not mkv.exists():
            print(f"  ⚠️  MKV not found: {mkv}; skipping frame.")
            summary["frames_skipped_no_video"] += 1
            continue

        stem = params.build_output_stem(mkv.stem)
        ply = pc_dir / f"{stem}.ply"
        params_json = pc_dir / f"{stem}_extraction_params.json"

        for suffix in DOWNSTREAM_SUFFIXES:
            p = pc_dir / f"{stem}{suffix}"
            if p.exists():
                stale_files.append(p)

        if dry_run:
            print(f"  [dry-run] would re-extract {mkv.name} → {ply.name}")
            summary["frames_planned"] += 1
            continue

        print(f"  🔁 {mkv.name} {params.representative_frame_id} → {ply.name}")
        extract_forearm(
            video_path=str(mkv),
            video_config=params,
            output_ply_path=str(ply),
            output_params_path=str(params_json),
            interactive=False,
            monitor=False,
            force_processing=True,
        )
        summary["frames_reextracted"] += 1


def _resolve_session_files(config_path: Path, project_root: Path) -> List[Path]:
    """Accept either a forearm DAG config or a direct ForearmConfig session yaml."""
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}

    if isinstance(doc, dict) and "session_id" in doc and "config_file_links" in doc:
        return [config_path]

    if isinstance(doc, dict) and "parameters" in doc:
        dag = DagConfigHandler(config_path)
        entries = dag.get_parameter("forearm_configs")
        return resolve_session_configs(
            entries, project_root / "configs" / "forearm_configs"
        )

    raise ValueError(
        f"Unrecognised config: {config_path}. Expected either a forearm DAG "
        "yaml (with a 'parameters.forearm_configs' list) or a ForearmConfig "
        "session yaml (with 'session_id' + 'config_file_links')."
    )


def _prompt_options_gui(default_dir: Path) -> Optional[Tuple[Path, bool]]:
    """Popup when no --dag-config was passed. Returns (dag_path, dry_run) or None if cancelled."""
    import tkinter as tk
    from tkinter import filedialog, messagebox

    result: dict = {"dag_path": None, "dry_run": True, "confirmed": False}

    root = tk.Tk()
    root.title("Re-extract forearms with parallax correction")
    root.geometry("640x280")

    tk.Label(
        root,
        text="Re-extract forearm point clouds using the parallax-corrected KinectFrame.",
        wraplength=600, justify="left", font=("", 10, "bold"),
    ).pack(padx=12, pady=(12, 6), anchor="w")

    path_var = tk.StringVar()
    row = tk.Frame(root)
    row.pack(fill="x", padx=12, pady=4)
    tk.Label(row, text="Config (DAG or session):").pack(side="left")
    entry = tk.Entry(row, textvariable=path_var)
    entry.pack(side="left", fill="x", expand=True, padx=(6, 6))

    def _browse() -> None:
        chosen = filedialog.askopenfilename(
            title="Select forearm DAG config",
            initialdir=str(default_dir) if default_dir.exists() else str(Path.cwd()),
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if chosen:
            path_var.set(chosen)

    tk.Button(row, text="Browse…", command=_browse).pack(side="left")

    dry_var = tk.BooleanVar(value=True)
    tk.Checkbutton(root, text="Dry run (checked by default)", variable=dry_var).pack(
        anchor="w", padx=12, pady=(10, 0)
    )
    tk.Label(
        root,
        text=(
            "Dry run: plan the work and list which .ply files would be re-extracted "
            "and which downstream files would become stale — without writing anything. "
            "Uncheck to actually overwrite the raw .ply files."
        ),
        wraplength=600, justify="left", fg="#444",
    ).pack(anchor="w", padx=30, pady=(2, 8))

    def _ok() -> None:
        raw = path_var.get().strip()
        if not raw:
            messagebox.showwarning("Missing config", "Please select a DAG config file.")
            return
        p = Path(raw)
        if not p.is_file():
            messagebox.showerror("Not found", f"File does not exist:\n{p}")
            return
        result["dag_path"] = p
        result["dry_run"] = bool(dry_var.get())
        result["confirmed"] = True
        root.destroy()

    def _cancel() -> None:
        root.destroy()

    btns = tk.Frame(root)
    btns.pack(pady=10)
    tk.Button(btns, text="Run", width=12, command=_ok).pack(side="left", padx=6)
    tk.Button(btns, text="Cancel", width=12, command=_cancel).pack(side="left", padx=6)

    root.mainloop()

    if not result["confirmed"]:
        return None
    return result["dag_path"], result["dry_run"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run extract_forearm on every snapshot in a forearm DAG config, "
            "using the parallax-corrected KinectFrame. Non-interactive. "
            "If --dag-config is omitted, a GUI picker is shown."
        )
    )
    parser.add_argument(
        "--dag-config",
        type=Path,
        default=None,
        help="Path to either a forearm DAG yaml "
             "(e.g. configs/preprocess_pipeline_extract_forearm_manual_dag.yaml) "
             "or a single ForearmConfig session yaml "
             "(configs/forearm_configs/session_*.yaml). "
             "If omitted, a file-picker GUI is shown.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the work and list stale downstream files without writing anything.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[4]

    if args.dag_config is None:
        chosen = _prompt_options_gui(project_root / "configs")
        if chosen is None:
            print("Cancelled.")
            sys.exit(0)
        args.dag_config, args.dry_run = chosen

    project_data_root = path_tools.get_project_data_root()

    session_files = _resolve_session_files(args.dag_config, project_root)

    print(f"Project root : {project_root}")
    print(f"Data root    : {project_data_root}")
    print(f"DAG config   : {args.dag_config}")
    print(f"Sessions     : {len(session_files)}")
    print(f"Mode         : {'DRY-RUN' if args.dry_run else 'EXECUTE'}")

    summary = {
        "sessions_processed": 0,
        "sessions_skipped_no_meta": 0,
        "sessions_skipped_no_rgb": 0,
        "frames_reextracted": 0,
        "frames_planned": 0,
        "frames_skipped_no_video": 0,
    }
    stale_files: List[Path] = []

    for sf in session_files:
        print(f"\n── Session: {sf.name} ──")
        try:
            _process_session(sf, project_data_root, args.dry_run, summary, stale_files)
        except Exception as exc:
            print(f"  ❌ FATAL — skipping {sf.name}: {exc}")

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if stale_files:
        print(
            f"\n=== STALE DOWNSTREAM FILES ({len(stale_files)}) ===\n"
            "These were derived from the OLD raw .ply and are now out of date.\n"
            "Re-run via preprocess_pipeline_extract_forearm_manual.py if you need\n"
            "them refreshed (curation remains interactive).\n"
        )
        for p in stale_files:
            print(f"  {p}")


if __name__ == "__main__":
    main()
