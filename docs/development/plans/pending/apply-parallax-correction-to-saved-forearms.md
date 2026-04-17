# Plan: Re-extract forearms to propagate parallax correction

**Created:** 2026-04-14
**Approved:** —
**Completed:** —
**Author:** Basil
**Status:** Pending
**Branch:** `feature/kinect-parallax-tolerance-band` (continuation)

> **Note:** On approval, this content should be moved into
> `docs/development/plans/pending/apply-parallax-correction-to-saved-forearms.md`
> (project convention; see CLAUDE.md). The sandbox plan file is only used
> here because plan mode restricts edits to it.

---

## Context

The parallax correction (plan
[`kinect-frame-parallax-correction.md`](../active/kinect-frame-parallax-correction.md))
is now applied by default inside `KinectFrame`. Every forearm
point-cloud extracted **before** that change has its XYZ points paired
with RGB colors from the wrong pixel — which means the HSV skin-colour
filter in `ArmSegmentation` selected a subset of points that is not
the same as the subset it would select under the corrected pairing.

The user does not want to re-annotate ROIs or re-tune HSV parameters;
both are already persisted in
`{session_id}_arm_roi_metadata.json` and per-snapshot
`*_extraction_params.json`. The correction can therefore be propagated
by re-running **only** the `extract_forearm` step non-interactively,
with those saved parameters, against the same MKV — now read through
the corrected `KinectMKV`.

Downstream stages (`curate`, `clean`, `define_normals`, `build_mesh`,
`register`) depend on the raw `.ply` and will become stale after
re-extraction. The script will list them; re-running them is
deferred to the existing manual/registration pipelines.

## Problem Statement

A post-hoc transform of an already-saved `.ply` cannot reproduce the
parallax-corrected result — the skin-filter decisions are frozen into
the saved point set. Re-running `extract_forearm()` against the MKV is
the only correct path. The existing
`preprocess_pipeline_extract_forearm_manual.py` does this but is
bundled with interactive ROI annotation, curation GUIs, and the full
5-stage batch. A focused script is needed that skips all interaction
and re-runs just stage 1 for every frame group of every session in a
DAG config.

## Goals

### In Scope
1. New script
   `code/scripts/_3_preprocessing/_3_forearm_extraction/reextract_forearms_with_parallax.py`
   that takes `--dag-config <yaml>` and iterates every session listed
   under `forearm_configs`.
2. For each session, load the existing
   `{session_id}_arm_roi_metadata.json`; if missing, skip with a
   warning (nothing to correct).
3. For each `ForearmParameters`, resolve the depth `.mkv` path and
   call `extract_forearm(...)` with `force_processing=True`,
   `interactive=False`, `monitor=False`. Saved
   `*_extraction_params.json` is auto-loaded by `extract_forearm`, so
   HSV + DBSCAN params are preserved.
4. After re-extraction of each frame, list the stale downstream files
   (`*_curated.ply`, `*_cleaned.ply`, `*_with_normals.ply`,
   `*_mesh.obj`) so the user can decide whether to re-run the manual
   pipeline afterwards.
5. Print a final summary: sessions processed, frames re-extracted,
   sessions skipped (no metadata), frames skipped (no MKV).

### Out of Scope
- Automatically re-running `curate`, `clean`, `define_normals`,
  `build_mesh`, or `register_forearms`. Curate is interactive-only; the
  rest are cleanly re-runnable via the existing manual pipeline.
- Any change to the parallax correction itself.
- Backwards-compatibility with pre-correction PLYs — overwriting is
  the whole point.
- A version tag inside the extraction params JSON (could be a
  follow-up idea: persist `PARALLAX_CORRECTION_VERSION`).

## Success Criteria

- [ ] Running
      `python code/scripts/_3_preprocessing/_3_forearm_extraction/reextract_forearms_with_parallax.py --dag-config configs/preprocess_pipeline_extract_forearm_manual_dag.yaml`
      re-writes every `*.ply` (raw) listed in each session's
      `arm_roi_metadata.json`.
- [ ] No interactive window opens at any point during the batch.
- [ ] A session with no metadata JSON is skipped with a clear warning,
      not an exception.
- [ ] A frame whose MKV is missing on disk is skipped with a clear
      warning.
- [ ] Final stdout summary lists the stale downstream files (if any).
- [ ] Running the script twice in a row is idempotent (second run
      simply overwrites again).

## Technical Design

### Approach

The script is a thin orchestration layer over existing building
blocks. **No new core logic is introduced.**

Reused modules / functions:

| Concern | Module | What we reuse |
|---|---|---|
| DAG config | `utils.DagConfigHandler` | load + read `forearm_configs` param |
| Session list resolution | `utils.pipeline.session_config_resolver.resolve_session_configs` | entries → flat `list[Path]` |
| Session config | `primary_processing.ForearmConfigFileHandler`, `ForearmConfig` | `session_id`, `session_processed_path`, `config_file_links` |
| RGB → MKV resolution | `preprocess_pipeline_extract_forearm_manual.resolve_rgb_video_paths` | kinect config links → RGB `.mp4` paths |
| Frame params | `preprocessing.forearm_extraction.ForearmFrameParametersFileHandler.load` | load list of `ForearmParameters` |
| Output path layout | `preprocessing.forearm_extraction.models.forearm_parameters.ForearmParameters.build_output_stem` | base filename for each frame group |
| Extraction | `_3_preprocessing._3_forearm_extraction.extract_forearm` | the stage-1 function; already accepts `interactive=False`, `force_processing=True`, auto-loads saved params |

Control flow (pseudocode):

```python
dag = DagConfigHandler(args.dag_config)
session_files = resolve_session_configs(
    dag.get_parameter('forearm_configs'),
    project_root / "configs" / "forearm_configs",
)
for session_file in session_files:
    cfg = ForearmConfigFileHandler.load(session_file)
    pc_dir = cfg.session_processed_path / "forearm_pointclouds"
    meta = pc_dir / f"{cfg.session_id}_arm_roi_metadata.json"
    if not meta.exists(): warn+skip
    rgb_paths = resolve_rgb_video_paths(cfg.config_file_links, data_root)
    for params in ForearmFrameParametersFileHandler.load(str(meta)):
        rgb = match(rgb_paths, params.video_filename)
        mkv = rgb.with_suffix(".mkv")
        if not mkv.exists(): warn+skip
        stem = params.build_output_stem(mkv.stem)
        ply    = pc_dir / f"{stem}.ply"
        params_json = pc_dir / f"{stem}_extraction_params.json"
        extract_forearm(
            video_path=str(mkv), video_config=params,
            output_ply_path=str(ply), output_params_path=str(params_json),
            interactive=False, monitor=False, force_processing=True,
        )
        list_stale([pc_dir / f"{stem}_curated.ply", ...])
print_summary()
```

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|---|---|---|---|
| Post-hoc XYZ translation on each saved `.ply` | No MKV access; very cheap | Not the same correction — skin-filter result is frozen; scientifically wrong | Rejected |
| Reuse `preprocess_pipeline_extract_forearm_manual.py` with `--dag-config` and all non-`extract_forearm` tasks `enabled: false` | Zero new code | Still calls `_annotate_forearm_roi`, which opens an interactive frame-group selector even with pre-filled data; not true "no GUI"; hard-coded `interactive=True` inside `execute_frame_batch` | Rejected |
| **New thin wrapper script** that calls `extract_forearm` directly per frame params | Minimal surface; no interaction; reuses every downstream helper; respects existing file naming | Small amount of new code (~80–120 lines) | **Chosen** |
| Add a `parallax_correction_version` field to `*_extraction_params.json` to enable a "re-run only stale ones" mode | Idempotent resumption on partial runs | Scope creep; `force_processing=True` + manual skipping is enough | Deferred (idea-file follow-up) |

### Architecture Constraints

- **CuPy import order** (CLAUDE.md): the new script imports
  `preprocessing.*`. Add the standard `try: import cupy` guard
  before any `preprocessing.*` import. Existing forearm scripts
  already follow this — mirror `extract_participant_forearm.py`.
- The parallax correction is applied inside `KinectMKV` /
  `KinectFrame` by default; `extract_forearm` constructs
  `KinectMKV(video_path)` with no flag, so correction is active
  automatically. The script does **not** need to pass any
  parallax-related argument.

### Architecture Changes

```
code/scripts/_3_preprocessing/_3_forearm_extraction/
├── extract_participant_forearm.py               (unchanged)
├── preprocess_pipeline_extract_forearm_manual.py (unchanged)
└── reextract_forearms_with_parallax.py           — NEW
```

No changes to `code/src/`.

## Implementation Plan

### Phase 1: Script + smoke-test on one session
**Goal:** Working one-shot re-extraction driven by the existing DAG config.

**Tasks:**
- [ ] Create `reextract_forearms_with_parallax.py` per the Approach above.
- [ ] `argparse`: `--dag-config <Path>` required; `--dry-run` optional
      (prints the plan without running `extract_forearm`).
- [ ] Smoke-test on `configs/preprocess_pipeline_extract_forearm_manual_dag.yaml`
      (currently points to `session_2022-06-17_ST16-05.yaml`):
  - confirm every raw `.ply` under that session's
    `forearm_pointclouds/` has a newer mtime after the run
  - confirm no window opens
  - spot-check one pair of before/after `.ply` files in a viewer —
    the corrected cloud should look visually similar but shifted
    slightly; top row + right 10 px of the source area should have
    dropped out (NaN edge from the correction)

**Files Modified:**
- `code/scripts/_3_preprocessing/_3_forearm_extraction/reextract_forearms_with_parallax.py` — new

**Dependencies:** None (all helpers already exist).

### Phase 2: Run across the full dataset
**Goal:** Propagate the correction to every forearm snapshot the user
cares about.

**Tasks:**
- [ ] Edit the DAG config (or a copy of it) to list every forearm
      session yaml under `configs/forearm_configs/` the user wants
      re-processed.
- [ ] Run the script; archive the stdout summary.
- [ ] Decide per session whether to re-run the downstream stages
      (`curate` manually, then `clean`/`normals`/`mesh`/`register`
      non-interactively via the existing manual pipeline).

**Files Modified:**
- `configs/preprocess_pipeline_extract_forearm_manual_dag.yaml` —
  (optional) expand `forearm_configs` list

**Dependencies:** Phase 1.

## Testing Plan

### Unit / Smoke
- [ ] `--dry-run` produces the expected list of `(session_id, frame,
      mkv_path, output_ply_path)` tuples without touching the
      filesystem.
- [ ] Missing metadata JSON → warning + non-zero continuation.
- [ ] Missing MKV → warning + non-zero continuation.

### Integration
- [ ] On `session_2022-06-17_ST16-05`: number of output `.ply` files
      matches number of `ForearmParameters` in the metadata JSON.
- [ ] Each output `.ply` is readable by `o3d.io.read_point_cloud` and
      has > 0 points.
- [ ] Comparing old vs new `.ply` bounding boxes: roughly the same
      center, minor shift in the +x / +y directions consistent with
      the 10 px / 1 px pixel shift projected through the intrinsics.

### Manual Verification
- [ ] Open a pre- and post-correction `.ply` side by side in MeshLab;
      the participant's skin cluster should look cleaner (fewer stray
      background points that were picked up by mis-paired colour).

## Documentation Plan

- [ ] Module docstring on `reextract_forearms_with_parallax.py`
      explaining what it is, why it exists (link to the parallax plan
      and the KB note), and that it is a one-shot migration tool.
- [ ] No `CLAUDE.md` change required.
- [ ] When complete, add a one-line entry in the parallax KB note
      under a new "Historical data migration" section.

## Rollback Plan

1. The raw `.ply` files are in the session's `forearm_pointclouds/`
   directory; **there is no built-in backup**. Before running in
   anger, the user should `git`-ignore or manually copy the
   `forearm_pointclouds/` folders they want to preserve, e.g.:
   `robocopy <src> <src>_preparallax /E /XO`.
2. If the re-extraction produces a worse result (e.g. the correction
   direction turns out to be wrong for some session outside the
   500–800 mm operating band), restore from that copy.
3. No on-disk format or metadata change: the `.ply` and
   `*_extraction_params.json` contents keep the same schema.

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Overwriting a pre-parallax PLY the user still needs for audit | Med | High | `--dry-run` flag; document the robocopy backup step in the module docstring and in the Rollback section |
| Session outside the 500–800 mm operating range → correction is wrong-direction | Low | Med | Scope note in script docstring; user visually inspects one frame per session post-run |
| Downstream stale artefacts (mesh, normals) silently used by a later analysis | Med | Med | Final summary prints every stale downstream path; optional future enhancement: delete them |
| CuPy import-order regression | Low | Low | Copy the guard pattern verbatim from `extract_participant_forearm.py` |

## Timeline

| Phase | Estimated Effort | Dependencies |
|---|---|---|
| Phase 1 | ~45 min | None |
| Phase 2 | variable (mostly batch runtime) | Phase 1 |

## References

- Active: [`docs/development/plans/active/kinect-frame-parallax-correction.md`](../active/kinect-frame-parallax-correction.md)
- KB: [`docs/development/knowledge-base/note-azure-kinect-rgb-depth-parallax.md`](../../knowledge-base/note-azure-kinect-rgb-depth-parallax.md)
- Reused script:
  `code/scripts/_3_preprocessing/_3_forearm_extraction/extract_participant_forearm.py`
- Reused pipeline (for helpers, not invoked directly):
  `code/scripts/preprocess_pipeline_extract_forearm_manual.py`
- DAG config used by default:
  `configs/preprocess_pipeline_extract_forearm_manual_dag.yaml`
