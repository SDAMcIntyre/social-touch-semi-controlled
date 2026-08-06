# Plan: Remove the Analysis Package

**Date:** 2026-08-06
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `dev`
**Branch:** `refactor/remove-analysis-package`

---

## Overview

Delete `code/src/analysis/` (140 files) and every analysis-stage coupling from this
repository, so that `social-touch-semi-controlled` ends at the postprocessing stage.
The analysis code already lives in a standalone repository
(`social-touch-semi-controlled-analysis`), so this is a *verify-then-delete*, not an
extraction. The one live consumer outside the package — the postprocessing DAG task
`center_on_receptive_field` — keeps working by relocating ~164 lines of generic DBSCAN
clustering helpers into `postprocessing/`.

## Problem Statement

The repository currently owns two pipelines with different lifecycles. Analysis-stage
code (`receptive_field_mapping`, `touch_analytics`) has been extracted to its own
repository and has since diverged — the analysis repo is ahead on every file that
differs, including a bug fix this repo never received. Keeping a stale duplicate here:

1. **Creates a false source of truth.** 34 of 140 files differ; all are newer in the
   analysis repo. Anyone editing the copy here is editing dead code.
2. **Violates the repo's own layering.** `utils/gui/dag_launcher/feature_combination_dialog.py`
   imports `analysis.touch_analytics.feature_extraction` — an upward import from a
   shared helper into a downstream stage. It is the only `analysis.` import anywhere in
   `code/src/` outside the package itself.
3. **Violates the fail-fast rule.** That same import is wrapped in
   `try/except ImportError` with empty-collection fallbacks, so a broken analysis tree
   degrades silently into a dialog with zero checkboxes rather than raising.
4. **Imposes a 4.5 s import cost on postprocessing.** `center_on_receptive_field.py`
   imports a leaf config module, but `analysis/receptive_field_mapping/__init__.py` is
   fully eager, pulling 120 modules including pyvista, open3d, igl, trimesh and seaborn.

## Goals

### In Scope

1. Relocate the DBSCAN receptive-field clustering helpers out of `analysis/` into
   `postprocessing/`, preserving the `center_on_receptive_field` DAG task's behaviour
   exactly.
2. Sever the `utils` → `analysis` GUI coupling by removing `FeatureCombinationDialog`
   and its three call sites.
3. Delete `code/src/analysis/**`, its 12 test files, the `analysis.pipeline` stub in
   `code/tests/conftest.py`, the 2 workflow entry scripts, the 8 sandbox/diagnostic
   scripts, the 3 `analyse_*_dag.yaml` configs and their 2 `.layout.json` siblings, the
   Analysis category in `configs/launcher.yaml`, and the 2 rows in `configs/README.md`.
4. Update the root `CLAUDE.md` package table and the lazy-load pointer to the deleted
   `code/src/analysis/CLAUDE.md`.
5. Reconcile the one piece of unmerged analysis work (`8ed6e2b`) before deleting it.
6. Establish a documented grep audit that keeps the boundary enforced.

### Out of Scope

- **All of `docs/`.** `docs/development/knowledge-base/` (44 notes) and
  `docs/development/plans/completed/` (~15 analysis plans) stay untouched by explicit
  decision. Roughly 25 KB notes will describe code that no longer exists here; this is
  an accepted, recorded consequence (see Risks).
- **The ~12 analysis-scoped plans in `docs/development/plans/pending/`.** They become
  un-actionable in this repo but are left in place, consistent with the docs decision.
- **`origin/sarah_sandbox`.** Its 4 commits reference `rf_cluster_visualizer.py`, deleted
  by the May reorganization — already orphaned. Left alone.
- **Any change to the analysis repository other than the `8ed6e2b` port** in Phase 0.
- **Re-filing `rf-center-top-spike-filter.md`**, which is already implemented but still
  marked `Draft` in `pending/`. Flagged, not fixed.
- **Renaming or restructuring `postprocessing/`** beyond adding the one new module.

## Success Criteria

- [x] `grep -rn "from analysis\|import analysis" code/` returns **zero** hits.
- [x] `code/src/analysis/` does not exist; `git ls-tree -r HEAD -- code/src/analysis` is empty.
- [x] `python code/scripts/launch_pipeline_gui.py` starts, and the launcher lists five
      categories (Setup, Primary, Preprocess, Merging, Postprocess) with no Analysis entry.
      *(Verified headlessly — `QT_QPA_PLATFORM=offscreen`, `parse_launcher_config` +
      `LauncherWindow(...).show()`; not a windowed desktop launch.)*
- [x] Opening a task's detail panel for every remaining DAG config raises no exception
      and shows no orphaned feature-combination control. *(62 tasks × 11 configs rendered
      offscreen; `FeatureCombinationDialog` / `_is_feature_combinations_dict` /
      `_make_combination_section` grep to zero hits.)*
- [x] `pytest` passes with zero collection errors.
- [ ] The `center_on_receptive_field` DAG task runs end-to-end on one session and produces
      byte-identical `blocks_rf_centered/*.csv`, `forearm_rf_centered/*.ply`, and
      `rf_center_origin.json` compared to a pre-change run on the same input.
      **Left unticked deliberately:** verified in Phase 1 Task 1.4 (two sessions, both C5
      branches, 17 files SHA-256-identical), but not re-verified in Phase 5 — it needs the
      real session data and a pre-change baseline that no longer exists on this branch.
- [x] `python -c "import postprocessing.receptive_field.rf_clustering"` completes without
      importing pyvista, open3d, igl, trimesh or seaborn (verify via `sys.modules`).
- [x] Root `CLAUDE.md` has no `analysis/` row and no dangling pointer to
      `code/src/analysis/CLAUDE.md`.

## Definitions

- **Agnostic (of post-postprocessing stages):** `grep -rn "from analysis\|import analysis" code/`
  returns zero hits, AND no file under `code/src/`, `code/scripts/`, or `configs/`
  references a receptive-field-mapping or touch-analytics *pipeline stage* by name. It
  does **not** mean the repo contains no receptive-field concepts at all — the
  postprocessing stage legitimately computes an RF center in order to define its output
  coordinate origin.
- **Generic helper (eligible for relocation):** a symbol whose transitive import closure
  contains no `analysis.*` module and whose logic is expressed over plain numeric types
  (`dict[(x,y,z)] -> float`, `ndarray`) rather than over neural/spike/CSV domain schemas.
  All five relocated symbols meet this test; `RFMappingColumnConfig` does not, and stays
  behind (deleted).
- **Behaviour preserved (for the relocated task):** the four documented C5 semantics hold
  unchanged — (1) per-point selectivity = `spike_count / total_count`; (2) DBSCAN over
  points above threshold; (3) RF center = spike-count-weighted centroid of the top 10 %
  highest-spike points of the **largest** cluster; (4) on DBSCAN failure, inputs are
  copied through unchanged and `rf_center_origin.json` records `status: "no_cluster_found"`.
  Criterion (4) is a contract `aggregate_session` depends on and must **not** become an
  exception.
- **Reconciled (for `8ed6e2b`):** either the `centroid_stroke` semantic change exists in
  the analysis repo's `rf_proximal_distal_comparison_pipeline.py`, or this plan records an
  explicit written decision to drop it. Silent loss does not count.

---

## Technical Design

### Approach

Three severances in dependency order, then the bulk delete.

The ordering matters: the GUI coupling and the postprocessing coupling must both be cut
*before* the package is deleted, so that every intermediate commit leaves the repository
in a working state and the branch can be reviewed one phase at a time.

The relocation is a **hard move**, not a copy. `RFMappingEngine` and
`SelectivityDBSCANConfig` have no other consumer in this repo, so no duplication is
created. `GroupedSpatialData` and `RFMapResult` do have other consumers
(`rf_data_loader.py`, `rf_visualizer.py`) — but those are inside the package being
deleted, so the constraint evaporates. The analysis repo keeps its own copies; the two
repos diverge deliberately from here.

The transitive closure terminates immediately: `engine.py` imports only from `.config`,
and the four dataclasses needed depend on nothing but `dataclasses`, `numpy` and
`collections.Counter`. Critically, `config.py:8` imports `NERVE_SPIKE_COL` and
`CONTACT_POINTS_COL` from `analysis.pipeline.shared_constants`, but those serve only
`RFMappingColumnConfig` — which the script does not use. Leaving that class behind drops
`analysis.pipeline` from the closure entirely, so the relocated module needs no constant
inlining and no postprocessing-side constant.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Relocate the 5 helpers into `postprocessing/`, then delete | Keeps a live postprocessing task working; removes the 4.5 s eager-import cost; no cross-repo dependency | ~164 lines now maintained in two repos | **Chosen** |
| Keep `analysis/` as a pip dependency on the analysis repo | No code moves | Reintroduces the coupling this plan exists to remove; makes postprocessing unrunnable without the analysis repo installed | Rejected |
| Delete `center_on_receptive_field.py` along with the package | Simplest | Deletes a live, `enabled: true` DAG task that `aggregate_session` depends on | Rejected |
| Vendor the helpers under `code/src/_vendor/` mirroring the analysis repo | Symmetric with how the analysis repo solved its side | `_vendor` signals "foreign code, do not edit"; these helpers are now first-class postprocessing logic | Rejected |
| Keep `FeatureCombinationDialog`, stub out its feature list | Smaller GUI diff | Leaves a dialog that cannot function; entrenches the silent fallback | Rejected |

### Architecture & Module Contracts

| Module / layer | Responsibility | Inputs → Outputs | Must NOT know about |
|----------------|----------------|------------------|---------------------|
| `postprocessing/receptive_field/rf_clustering.py` (new) | Threshold-and-cluster an arbitrary spatial score field; return clusters ranked by size | `dict[(x,y,z)] -> count` pairs + `SelectivityDBSCANConfig` → `RFMapResult` | Spikes, neurons, CSV column names, file paths, Prefect, Qt, the DAG, any `analysis.*` module |
| `code/scripts/_5_postprocessing/center_on_receptive_field.py` (modified) | Orchestrate stage C5: read blocks + PLY, call the clusterer, build and apply the translation | block CSVs + forearm PLY → `blocks_rf_centered/`, `forearm_rf_centered/`, `rf_center_origin.json` | The internals of DBSCAN parameter selection; any `analysis.*` module |
| `utils/gui/dag_launcher/task_detail_panel.py` (modified) | Render editors for DAG task options | task option dict → Qt widgets | Any downstream pipeline stage's domain vocabulary |

Dependency direction after the change is strictly one-way — `scripts` → `postprocessing`
→ `numpy/sklearn` — with no upward import. This restores the layering the pipeline
guides specify (orchestration → stages → helpers, nothing depending back upward) and
removes the repo's only violation of it.

```
code/src/postprocessing/
└── receptive_field/
    ├── __init__.py            # re-exports the five symbols
    └── rf_clustering.py       # ~164 lines, moved verbatim

    SelectivityDBSCANConfig    #   7 lines — 4 scalars (threshold 0.3, eps 5.0 mm,
                               #             min_samples 3, min_cluster_points 5)
    RFCluster                  #   9 lines — id, points (N,3), scores (N,), mean, count
    RFMapResult                #  10 lines — label, clusters, scores, counts
    GroupedSpatialData         #   7 lines — two Counters
    RFMappingEngine            # 123 lines — compute_selectivity(), cluster_receptive_field()
```

### Knowledge Base Constraints

Derived from a full relevance sweep of `docs/development/knowledge-base/`:

- **`note-cupy-import-order.md`** — the guarded `import cupy` at
  `code/scripts/launch_pipeline_gui.py:13-17` must remain the first substantive import.
  Do **not** reorder or tidy that import block while editing GUI imports. Do not add a
  lazy `import cupy` to the relocated module (a rejected approach on record).
- **`note-qt-itemchanged-signal-recursion.md`** — the dialog call sites at
  `task_detail_panel.py:1060` and `:1096` sit inside item-changed flows. Preserve any
  surrounding `blockSignals(True/False)` bracketing and the explicit post-unblock
  `emit()`; do not delete a guard that merely looks empty after the dialog call is removed.
- **`note-kinect-depth-access-single-path.md`** — the repo's precedent for a
  convention-enforced module boundary: a blanket rule without an allowlist was found
  unworkable; the accepted form is a documented grep audit plus a review-checklist item.
  Phase 5 follows that precedent.
- **`note-spatial-alignment-pipeline.md` §4 C5** — the four behavioural semantics listed
  under Definitions. Note the existing note text has drifted (it says
  "selectivity-weighted centroid", and lists the input as `blocks_pca_calibrated/`); the
  code weights by spike count over the top 10 % and reads `blocks_contact_projected/`.
  Do not propagate the note's wording into the relocated docstring — copy the code's
  actual behaviour.
- **`note-analysis-pipeline-coordinate-spaces.md` / `note-somatosensory-units-and-calculations.md`**
  — the relocated helpers operate in millimetres, Kinect-origin. `dbscan_eps = 5.0` is
  5 mm. No rescaling or unit normalisation during the move.

---

## Implementation Plan

### Phase 0: Reconcile unmerged work
**Goal:** No analysis work is lost by the deletion.

**Started:** 2026-08-06 · **Completed:** 2026-08-06

- [x] Task 0.1 — **DECIDED: port.** Port the `centroid_stroke` change from `8ed6e2b` into the
      analysis repo's `rf_proximal_distal_comparison_pipeline.py`. Port *only* this part:

      - `_REQUIRED_GTYPES`: `('all', 'stroke_proximal', 'stroke_distal')` → `('stroke', …)`
      - read `npz['boundary_centroid_uv_stroke']` instead of `boundary_centroid_uv_all`
      - `offset_proximal/offset_distal` referenced to `centroid_stroke`, not `centroid_all`
      - summary-CSV columns `centroid_u_all`/`centroid_v_all` → `centroid_u_stroke`/`centroid_v_stroke`
      - `rf_center_comparison_renderer.py` axis labels → "ΔU/ΔV from stroke center"

      Rationale: it re-references proximal-vs-distal RF center offsets to a stroke-only
      baseline instead of an all-gesture baseline contaminated by taps — a semantic change,
      not a refactor. The enabling half (the `'stroke'` virtual subset that creates the
      `boundary_centroid_uv_stroke` key) already exists in the analysis repo, so only the
      consumer side is missing.

      Do **not** port the commit's `cmap` threading: the analysis repo already has it, with a
      `"inferno"` default. `8ed6e2b` defaults to `"jet"`, which
      `investigation-jet-colormap-perceptual-problems.md` argues against.

      Do not attempt a git merge — the branch edits `rf_proximal_distal_center_pipeline.py`,
      renamed to `rf_proximal_distal_comparison_pipeline.py` on `dev` (confirmed via
      `git ls-tree dev`). Apply the change by hand.
      **Done.** Ported by hand onto branch `feature/port-stroke-centroid-baseline` in the
      analysis repo (off `feature/rf-boundary-plugin-architecture`), commit `e2d65f3`, not
      pushed. Two deviations from the task text, both forced by divergence and both benign:
      (a) the renderer had been renamed `rf_center_comparison_renderer.py` →
      `rf_proximal_distal_comparison_renderer.py`, and its labels carry a `(mm)` suffix that
      reflects a `uv_to_mm` scaling the analysis repo added — the suffix is preserved, giving
      `'ΔU from stroke center (mm)'`; (b) the analysis repo has a line the branch never had,
      `centroid_hotspot_distance_mm_stroke = ‖centroid_all − hotspot_stroke‖`, which the
      single-variable rename necessarily re-baselines to `centroid_stroke`. That makes the
      metric internally consistent with its own name; there is no way to preserve the old
      value without a second `'all'` read that `_REQUIRED_GTYPES` no longer guarantees.
- [x] Task 0.1b — The branch also carries its own 171-line plan document at
      `docs/development/plans/active/rf-stroke-virtual-gesture-subset.md`, which exists only
      on that branch. Move it to the analysis repo alongside the ported change, or it is lost
      with the branch.

      **Done.** Written to the analysis repo at
      `docs/development/plans/completed/rf-stroke-virtual-gesture-subset.md` (that repo has
      only `active/` and `completed/`), status flipped to `Completed` and a provenance note
      added. Same commit as Task 0.1.
- [ ] Task 0.2 — Confirm the port landed in the analysis repo (or the drop is recorded),
      then delete the local branch `feature/rf-stroke-virtual-gesture-subset`.

      **Confirmation half done; deletion half deferred to the user.** The port is verified:
      zero `centroid_all` / `centroid_u_all` / `centroid_v_all` / `boundary_centroid_uv_all`
      references remain in either edited file, both compile, and the analysis repo's
      `tests/test_rf_boundary_dag_wiring.py` (17 tests, the only suite touching this pipeline)
      passes. The branch `feature/rf-stroke-virtual-gesture-subset` is therefore ready to
      delete, but `git branch -D` was **not** run — that call is reserved for the user.
- [x] Task 0.3 — Commit or discard the unrelated working-tree changes first: the
      duplicated `/graphify-out` line in `.gitignore` and the untracked
      `code/scripts/_sandbox_gradient_ridge_out/` PNGs (the latter should be gitignored,
      not committed).

      **Done.** The four working-tree lines collapsed to a single `/graphify-out` (the two
      `cache` sub-paths were already covered by the parent rule), a trailing newline added,
      and `/code/scripts/_sandbox_gradient_ridge_out/` ignored. The 5 PNGs are left on disk,
      untracked and no longer reported by `git status`.

**Files Modified:** `.gitignore` — de-duplicate `/graphify-out`, ignore the sandbox output dir

**Dependencies:** None

### Phase 1: Relocate RF clustering into postprocessing
**Goal:** `center_on_receptive_field` runs with no `analysis` import.

**Started:** 2026-08-06 · **Completed:** 2026-08-06

- [x] Task 1.1 — Create `code/src/postprocessing/receptive_field/rf_clustering.py`; move
      `SelectivityDBSCANConfig`, `RFCluster`, `RFMapResult`, `GroupedSpatialData` (from
      `analysis/receptive_field_mapping/config.py`) and `RFMappingEngine` (from
      `.../engine.py`) verbatim. Do not move `RFMappingColumnConfig` or `RFMappingConfig`.

      **Done.** 210 lines (37 of them the module docstring). Bodies and per-symbol
      docstrings are byte-identical to the originals; only the module docstring is new, and
      it describes the code's actual behaviour rather than
      `note-spatial-alignment-pipeline.md`'s drifted "selectivity-weighted centroid" wording
      — the module computes no centroid at all, and it records that `dbscan_eps = 5.0` is a
      5 mm radius. `Optional` was dropped from the `typing` import (it was unused in
      `config.py`). Imports are exactly stdlib + `numpy` + `sklearn.cluster.DBSCAN`; no
      analysis-side constant is inlined.
- [x] Task 1.2 — Add `code/src/postprocessing/receptive_field/__init__.py` re-exporting
      the five symbols.
- [x] Task 1.3 — Repoint `center_on_receptive_field.py` lines 22–26 to the new module.

      **Done.** Two import statements collapsed into one; nothing else in the file touched.
- [x] Task 1.4 — Capture a baseline run of the DAG task on one session **before** the
      change, to diff outputs against afterwards.

      **Done — two sessions, both C5 branches, byte-identical.** `center_on_receptive_field()`
      was driven directly with `force_processing=True` on `2022-06-14_ST13-03` (recorded
      status `ok` → the translation path) and `2022-06-15_ST14-04` (recorded status
      `no_cluster_found` → the pass-through path), writing into a scratch tree. All 17
      produced files (15 `blocks_rf_centered/*.csv`, 2 `forearm_rf_centered/*.ply`,
      2 `rf_center_origin.json`) have identical SHA-256 before and after. ST13-03's RF center
      is `[-377.075, 388.775, 482.6375]` mm from cluster 1 (30 points, mean selectivity
      0.4983) in both runs; ST14-04 records `status: "no_cluster_found"` in both, confirming
      the pass-through contract survived the move.

      A second, synthetic oracle covers the algorithm in isolation and the C5 edge cases
      (`_compute_rf_center` on fabricated block CSVs): known-answer selectivity 4/10 = 0.4
      exactly; two blobs 50 mm apart at `eps = 5.0` → exactly 2 clusters of 8 and 6 points
      with a 3-point micro-cluster discarded; nothing-above-threshold, `no_contact_points`,
      missing-file and missing-column paths. All outputs identical before and after.

      Note: the resolved input directory is `blocks_pca_calibrated/`, not
      `blocks_contact_projected/` as the C5 docstrings and the knowledge-base note claim.
      Left alone — out of Phase 1 scope, but worth a follow-up.
- [x] Task 1.5 — Verify the new module's import closure excludes pyvista/open3d/igl/
      trimesh/seaborn.

      **Done.** A fresh interpreter importing `postprocessing.receptive_field.rf_clustering`
      adds 1574 `sys.modules` entries in 1.72 s, with zero `analysis.*` modules and none of
      pyvista/open3d/igl/trimesh/seaborn. Importing `center_on_receptive_field.py` itself
      went from 4070 modules / 4.80 s to 3137 modules / 3.60 s, and from 121 `analysis.*`
      modules to 0; igl, trimesh and seaborn are gone. open3d and pyvista remain, but they
      now arrive through the script's own `import open3d` and the `preprocessing.*` import
      chain, not through `analysis` — removing them is not in this plan's scope.

**Verification:** `pytest -q` → 437 passed, 31 failed, 1 collection error. Every failure is
pre-existing and unrelated: stashing the Phase 1 changes and re-running the same files at
`2e82642` reproduces the identical 31 failures (`test_dag_config_model.py`,
`test_rf_grid_cell_metrics.py`, `test_rf_tap_stroke_comparison.py`) plus the same
`test_gmm_clusterer.py` collection error. None of those files references `postprocessing`
or the relocated module. `grep -rn "from analysis\|import analysis"` over
`code/scripts/_5_postprocessing/` and `code/src/postprocessing/` returns zero hits.

**Files Modified:**
- `code/src/postprocessing/receptive_field/rf_clustering.py` — new, ~164 lines
- `code/src/postprocessing/receptive_field/__init__.py` — new
- `code/scripts/_5_postprocessing/center_on_receptive_field.py` — import block only

**Dependencies:** Phase 0

### Phase 2: Sever the GUI coupling
**Goal:** `utils` no longer references `analysis`; the launcher still starts.

**Started:** 2026-08-06 · **Completed:** 2026-08-06

- [x] Task 2.1 — Delete `code/src/utils/gui/dag_launcher/feature_combination_dialog.py`.

      **Done.** 137 lines removed, including the `try/except ImportError` +
      `frozenset()` / `{}` fallback that was the repo's only `analysis.` import outside
      the package.
- [x] Task 2.2 — Remove the import at `task_detail_panel.py:30` and the two call sites at
      `:1060` (edit mode) and `:1096` (create mode), preserving surrounding `blockSignals`
      bracketing.

      **Done.** The two call sites were the whole bodies of
      `_make_combination_edit_handler` and `_make_combination_add_handler`, so both
      methods were removed; `_make_combination_context_handler` (right-click delete) went
      with them, since `_make_combination_section` was its only caller. `task_detail_panel.py`
      contains no `blockSignals` call at all — the documented Qt recursion guard lives in
      `task_panel.py::_on_item_changed`, which this phase does not touch, so no bracketing
      was disturbed. Removing the context handler orphaned the `QMenu` and `QCursor`
      imports; both were dropped. `QMessageBox` is still used by four surviving handlers
      and stays.
- [x] Task 2.3 — Remove `_is_feature_combinations_dict` (`task_detail_panel.py:148`), its
      dispatch branch at `:345`, and `_make_combination_section`. **Resolved by inspection:**
      the only `features:` option keys in `configs/` are in the two `analyse_*_dag.yaml`
      files deleted in Phase 3 (the `preprocess_workflow_kinect_auto_dag.yaml` match is the
      *task name* `refine_sticker_features`, not an option), and the string "combination"
      appears zero times anywhere in `configs/` including `_dag_templates/`. The dialog is
      the sole producer of that dict shape and writes it back into analysis DAG configs, so
      producer and consumer are removed together and the `:345` branch becomes unreachable.
      Leave the sibling predicates (`_is_feature_dict`, `_is_cluster_groups_dict`,
      `_is_radar_groups_dict`, `_is_grid_groups_dict`, `_is_profile_dict`) alone — verify
      each still has a live config key before touching it.

      **Done, and the decision was re-checked empirically rather than trusted.** Replaying
      the removed predicate over all 14 `configs/*_dag.yaml` files shows it matched five
      task options, every one of them in an `analyse_*` config deleted by Phase 3, and four
      of the five (`cluster_groups` ×2, `grid_groups`, `radar_groups`) were already claimed
      by an earlier dispatch branch and never reached `:345`. The single option that did
      reach it — `stimulus_compare_sessions / comparison_groups` in
      `analyse_workflow_processing_dag.yaml` — now falls through to `_make_scalar_section`
      and renders as a clickable YAML-editor label. No surviving config is affected. The
      five sibling predicates were left untouched.
- [x] Task 2.4 — Remove the stale Sphinx cross-reference to
      `analysis.receptive_field_mapping...PopulationRFGridConfig` in the
      `grid_group_dialog.py:8` docstring (prose only; that module imports no analysis).

      **Done.** Reworded to say the spec is written back into the `grid_groups` DAG option
      and consumed by the downstream population-grid workflow, without naming a module
      that will not exist in this repo.

**Verification:** `grep -rn "FeatureCombinationDialog\|_is_feature_combinations_dict\|_make_combination_section" code/` → zero hits.
`grep -rn "from analysis\|import analysis" code/src/utils/` → zero hits. With
`QT_QPA_PLATFORM=offscreen`, a fresh interpreter imports `task_detail_panel` and
`launcher_window` with zero `analysis.*` entries in `sys.modules`, constructs a
`QApplication`, and renders `TaskDetailPanel.show_task()` for all 114 tasks across all 14
`configs/*_dag.yaml` files (the three `analyse_*` configs included) with no exception.
`pytest -q --continue-on-collection-errors` → 437 passed, 31 failed, 1 collection error —
byte-for-byte the same pre-existing failure set recorded for Phase 1
(`test_dag_config_model.py`, `test_rf_grid_cell_metrics.py`, `test_rf_tap_stroke_comparison.py`,
plus the `test_gmm_clusterer.py` collection error). Zero new failures. Note that plain
`pytest` aborts on the collection error; `--continue-on-collection-errors` is required to
see the full result.

**Files Modified:**
- `code/src/utils/gui/dag_launcher/feature_combination_dialog.py` — deleted
- `code/src/utils/gui/dag_launcher/task_detail_panel.py` — 3 references removed
- `code/src/utils/gui/dag_launcher/grid_group_dialog.py` — docstring only

**Follow-up (not done, out of Phase 2 scope):** `DagConfigModel.get_combination_features`
and `set_combination_features` now have zero callers — the combination section was their
only consumer. `add_combination` and `remove_combination` are still live (the cluster-,
radar- and grid-group handlers use them). `dag_config_model.py` is not in this phase's file
list and `test_dag_config_model.py` exercises the model API, so the two dead methods were
left in place.

**Dependencies:** Phase 1

### Phase 3: Delete the package and its assets
**Goal:** Analysis-stage code and configuration are gone.

**Started:** 2026-08-06 · **Completed:** 2026-08-06

- [x] Task 3.1 — `git rm -r code/src/analysis/` (140 files, incl. `code/src/analysis/CLAUDE.md`).

      **Done, with one correction to the task text.** 140 *tracked* files staged as deleted.
      `code/src/analysis/CLAUDE.md` is **not** among them — `.gitignore:147` ignores every
      `CLAUDE.md`, so it was never tracked. Nine files under `code/src/analysis/` were
      untracked-because-ignored (that `CLAUDE.md` plus the eight modules in
      `receptive_field_mapping/data/`, ignored by `.gitignore:117 data/`) and therefore
      survived `git rm` on disk. They had to go too: they are importable as implicit
      namespace packages and they contain `from analysis...` imports, so leaving them would
      have broken the plan's primary grep criterion. The eight `data/` modules were first
      confirmed tracked in the analysis repo at
      `src/analysis/receptive_field_mapping/data/` (no loss). The 116-line
      `code/src/analysis/CLAUDE.md` is **not** reproduced there — the analysis repo's root
      `CLAUDE.md` is a different, generic 61-line document — so it was copied to the session
      scratchpad before `rm -rf code/src/analysis`. **Follow-up for the user:** decide
      whether that guide should be carried into the analysis repo; it is otherwise gone.
- [x] Task 3.2 — Delete the 12 test files: `test_rf_session_comparison_renderer.py`,
      `test_rf_tap_stroke_comparison.py`, `test_rf_population_heatmap.py`,
      `test_rf_inflection_boundary.py`, `test_rf_population_grid_pipeline.py`,
      `test_rf_extraction_io.py`, `test_rf_grid_cell_metrics.py`, `test_iff_windowed_mean.py`,
      `test_forearm_slim_uv.py`, `test_gmm_clusterer.py`, `test_feature_space_renderer.py`,
      `test_fit_models.py`.

      **Done.** Each was checked before deletion: every non-stdlib, non-numeric import in all
      twelve resolves to `analysis.*` (the only other third-party imports are `igl`,
      `trimesh` and `matplotlib.path`, all used solely to exercise analysis surface code).
      None covers any surviving package.
- [x] Task 3.3 — Remove the `analysis.pipeline` stub from `code/tests/conftest.py`.

      **Done.** Only that stub and its comment block were removed. The
      `preprocessing.stickers_analysis`, `preprocessing.forearm_extraction` and `utils`
      stubs are untouched — `stickers_analysis` merely contains the substring "analysis"
      and is unrelated.
- [x] Task 3.4 — Delete the 2 entry scripts (`analysis_workflow_processing.py`,
      `analysis_workflow_viewers.py`) and the 8 sandbox/diagnostic scripts
      (`sandbox_gradient_ridge_boundary.py`, `migrate_output_dirs.py`,
      `inspect_touch_feature_ranges.py`, `diagnose_stroke_direction.py`,
      `flatten_forearm_sandbox.py`, `diagnose_contact_points.py`,
      `diagnose_rf_explorer_distance.py`, `compare_flattening_methods.py`).

      **Done.** A pre-deletion grep for all ten script names across `code/`, `configs/` and
      `pyproject.toml` found references only in files also being removed
      (`analysis/pipeline/__init__.py` prose) or edited in this phase
      (`configs/launcher.yaml`), plus two rows in `code/scripts/README.md` — see Task 3.7.
      No surviving `__init__.py`, console-script entry point or workflow references any of
      them.
- [x] Task 3.5 — Delete `configs/analyse_workflow_dag.yaml`,
      `configs/analyse_workflow_processing_dag.yaml` + `.layout.json`,
      `configs/analyse_workflow_viewers_dag.yaml` + `.layout.json`.

      **Done.** The three `.yaml` files were tracked and are staged as deleted. The two
      `.layout.json` files are **untracked** — `.gitignore:168` ignores `*.layout.json`
      repo-wide (they are per-workstation GUI node-layout state) — so they were removed from
      disk directly; they cannot appear in the commit.
- [x] Task 3.6 — Remove the Analysis category from `configs/launcher.yaml` (lines 70–77)
      using `ruamel.yaml` round-trip mode, **not** PyYAML.

      **Done, but ruamel's defaults were not safe here and had to be constrained.** A plain
      `YAML(typ='rt')` load/dump rewrote the whole file: it re-indented every block sequence
      (`sequence=2, offset=0` instead of the file's `sequence=4, offset=2`) and, dumping to a
      `Path`, encoded with the Windows locale codec, corrupting the two em dashes in the
      header comment into `?`. That first attempt was reverted with `git checkout --`. The
      final script pins `yaml.indent(mapping=2, sequence=4, offset=2)` and `yaml.width=4096`,
      loads from a UTF-8-decoded `StringIO`, and writes through
      `open(..., encoding='utf-8', newline='\n')` to preserve the file's UTF-8/LF form. It
      also fails fast if the category list is not exactly the six expected names or the
      Analysis block does not hold exactly the two expected workflows. The resulting
      `git diff` is a pure 9-line deletion with no other hunk — comments, key order and
      indentation are byte-identical elsewhere.
- [x] Task 3.7 — Remove the two `analyse_workflow_*` rows from `configs/README.md` (lines 29–30).

      **Done.** The surrounding prose ("They map directly to the top-level workflow scripts
      in `code/scripts/`") remains accurate and was left as-is.

      **Plan gap closed:** `code/scripts/README.md:20-21` carried the same two rows for
      `analysis_workflow_processing.py` / `analysis_workflow_viewers.py`. That file appears
      in no phase of this plan, so the rows would have been left advertising scripts deleted
      by Task 3.4. They were removed on the same rationale as this task. The other two
      "analysis" matches in that file (line 32 "tracking and analysis substages", line 51
      "LED ROI analysis") are ordinary English about preprocessing and were left alone.

**Verification:** `grep -rn "from analysis\|import analysis" code/` → **zero hits** (the
plan's primary success criterion). `git ls-files code/src/analysis` → 0. `pytest -q
--continue-on-collection-errors` → **195 passed, 2 skipped, 0 failed, 0 collection errors**,
down from the 437/31/1 baseline at `2c4f936`. The delta was reconciled exactly against a
throwaway `git worktree` at `2c4f936` (with the gitignored `data/` modules restored), which
reproduced 437/31/1 precisely: the 12 deleted test files accounted for 232 passed + 30 failed
+ the `test_gmm_clusterer.py` collection error; deleting the 3 `analyse_*_dag.yaml` configs
removed 9 more passing tests, because `test_dag_config_model.py:14` builds
`DAG_FILES = sorted(CONFIGS_DIR.glob("*_dag.yaml"))` and parametrises three tests over it
(3 configs × 3 tests); and the 2 new skips are the two `test_dag_config_model.py` tests that
name the deleted configs behind pre-existing `pytest.skip("config not found")` guards. One of
those two was the single remaining baseline failure, so `test_dag_config_model.py` now
reports **0** failures — not because anything was fixed, but because the test skips.

**Follow-up (not fixed — `test_dag_config_model.py` is out of this plan's scope):**
`test_get_config_entries_multi` existed to assert that a DAG config can declare more than one
config entry, and `analyse_workflow_dag.yaml` was the only root config that did. It now skips
unconditionally and asserts nothing. It needs either a surviving multi-entry fixture or
deletion; leaving a permanently-skipped test is a silent coverage hole.

With `QT_QPA_PLATFORM=offscreen`, `parse_launcher_config` + `LauncherWindow(...).show()`
succeed on 11 workflow entries across exactly five categories in declaration order —
Setup, Primary, Preprocess, Merging, Postprocess — every `dag_config` path resolves to a file
on disk, and `sys.modules` contains zero `analysis*` entries.

**Files Modified:** as enumerated above (~165 deletions, 4 edits)

**Dependencies:** Phase 2

### Phase 4: Documentation and metadata
**Goal:** The repo describes itself accurately.

**Started:** 2026-08-06 · **Completed:** 2026-08-06

> **Plan correction — the root `CLAUDE.md` is gitignored.** This phase as written routes
> most of its output through the root `CLAUDE.md`, but `.gitignore:147` ignores every
> `CLAUDE.md`, so that file is untracked and its edits **cannot be committed**. They are
> still worth making — the file loads into every Claude Code session in this repo — but
> they will never reach a collaborator. The same discovery was already recorded in Task
> 3.1 for `code/src/analysis/CLAUDE.md`. Consequence: **Task 4.5's changelog is the only
> tracked record of the split**, and it was written to carry the full weight accordingly
> rather than as a one-line pointer. No `git add -f` was used. The same caveat applies to
> Phase 5, whose entire output is a `CLAUDE.md` section.

- [x] Task 4.1 — Root `CLAUDE.md`: remove the `analysis/` row from the package table and
      the closing paragraph pointing at `code/src/analysis/CLAUDE.md`.

      **Done.** The `analysis/` row is gone and the `postprocessing/` row was widened from
      "XYZ reference refinement from manual gesture data" to also name ICP registration,
      contact projection and RF centring — it now terminates the pipeline and its old
      one-clause description covered one of six DAG tasks. The dangling pointer paragraph
      was replaced by a two-line statement that the repo ends at postprocessing, naming the
      analysis repo and linking the new changelog. The "six categories … Analysis" sentence
      under Pipeline execution was corrected to five (not in the task text, but it was the
      same stale fact).

      **Scope toggle section removed entirely.** It described
      `.claude/settings.local.json` restricting `Read`/`Glob` to "analysis-pipeline files
      only" and told the reader to check `permissions.deny`. That file currently has **no**
      `deny` key at all (only `permissions.allow`), and the directory the toggle existed to
      isolate no longer exists — so the section was doubly inoperative. Rewording was not
      possible: there is no surviving mechanism to describe.
- [x] Task 4.2 — Root `CLAUDE.md`: update the "Data flow" diagram so it terminates at
      postprocessing.

      **Done.** `→ Merged CSV → Analysis (preparation → … → RF mapping)` became
      `→ Merged CSV → Postprocessing (reference forearm, ICP registration, contact
      projection, PCA calibration, RF centring) → Aggregated session`. The stage names were
      read off the seven tasks in `configs/postprocess_workflow_kinect_auto_dag.yaml`, not
      invented.
- [x] Task 4.3 — Root `CLAUDE.md`: update the Test-environment paragraph, which names
      `analysis.pipeline` as a stubbed package root.

      **Done, and the list was wrong in a second way.** Reading `code/tests/conftest.py`
      shows five surviving `_stub_package` calls: `preprocessing.stickers_analysis`,
      `preprocessing.forearm_extraction`, `utils`, `utils.pipeline` and
      `utils.pipeline.monitoring`. The paragraph named only the first three plus
      `analysis.pipeline`, so dropping the dead entry alone would have left the list
      incomplete. All five are now listed.
- [x] Task 4.4 — `pyproject.toml`: the description reads "A package for motion
      preprocessing and analysis" — narrow it.

      **Done.** → `"Acquisition-to-postprocessing pipeline for semi-controlled social touch
      recordings."` Nothing else in the file was touched.
- [x] Task 4.5 — Add a changelog entry recording the split and naming the analysis repo
      as the new home, so the provenance is discoverable from this repo.

      **Done.** `docs/changelogs/remove-analysis-package.md` (new directory). Written as
      the *primary* artifact of this phase, per the gitignore finding above: it records the
      commit range, the removal counts, the analysis repo's identity (373 commits, full
      history, self-contained via `src/_vendor/`), why `rf_clustering.py` was kept, the
      untouched-docs decision with measured counts (13 of 44 knowledge-base notes and 17 of
      26 pending plans name a now-absent path), the five recorded-not-fixed discrepancies,
      the test-baseline reconciliation, and the grep audit. The stale-note section tells a
      reader explicitly that a note pointing at missing code means the code moved, not that
      the note rotted.

**Verification:** `grep -rniE "analysis|analyse"` over `CLAUDE.md`, `README.md` and
`pyproject.toml` → the only surviving hits are ordinary English about the analysis repo or
unrelated prose (`README.md:101`, "removed for analysis" re: the IR track). Zero references
to the deleted package, scripts or configs remain in any of the three.
`python -c "import tomllib; tomllib.load(...)"` parses `pyproject.toml`.
`pytest -q --continue-on-collection-errors` → **195 passed, 2 skipped, 0 failed, 0
collection errors** — unchanged from Phase 3.

**Files Modified:**
- `CLAUDE.md` — **local-only, gitignored, not committable**
- `pyproject.toml` — description line only
- `docs/changelogs/remove-analysis-package.md` — new

**Dependencies:** Phase 3

### Phase 5: Enforce the boundary
**Goal:** The separation does not silently regress.

**Started:** 2026-08-06 · **Completed:** 2026-08-06

> **Scope adjustment — `README.md` added as the tracked home of the rule.** This phase as
> written routes its entire output through the root `CLAUDE.md`, which `.gitignore:147`
> ignores (the same finding already recorded in Task 3.1 and the Phase 4 correction). A
> rule written only there is uncommittable and never reaches a collaborator, which defeats
> the phase's stated goal. The section was therefore written **twice**: into `CLAUDE.md`
> (local-only, but loaded into every Claude Code session in this repo) and into
> `README.md` (tracked, contributor-facing — the durable home). No `git add -f` was used.

- [x] Task 5.1 — Add a short "Stage boundary" section to root `CLAUDE.md` stating that
      this repo ends at postprocessing, naming the analysis repo, and giving the audit
      command `grep -rn "from analysis\|import analysis" code/` (expected: zero hits) —
      following the `note-kinect-depth-access-single-path` precedent of rule + documented
      grep rather than an unenforced convention.

      **Done, in both files.** In `CLAUDE.md` the section is
      `### Stage boundary — this repo ends at postprocessing`, placed under
      `## Project Conventions` immediately after `### Kinect depth access — single path` —
      the rule it is modelled on, so the two convention-enforced boundaries sit together.
      In `README.md` it is `## 🧭 Repository Scope`, placed after `## ✨ Features` and
      before `## 📋 Prerequisites`, matching that file's emoji-heading and `-----` rule
      conventions; a contributor meets it while still reading what the repo is, before
      installation. Both carry the rule, the verbatim audit command, the
      stages-not-concepts nuance naming `postprocessing/receptive_field/rf_clustering.py`
      explicitly so it is not "cleaned up", and a relative link to
      `docs/changelogs/remove-analysis-package.md`. The narrative is not duplicated — both
      sections point at the changelog for it.

      Per the knowledge-base precedent, no allowlist is defined: unlike the `pyk4a` rule,
      this boundary has **zero** sanctioned exceptions, so the rule + grep + review-check
      form is complete without one.

**Verification:** `grep -rn "from analysis\|import analysis" code/` → **zero hits**.
`pytest -q --continue-on-collection-errors` → **195 passed, 2 skipped, 0 failed, 0
collection errors** — unchanged from Phases 3 and 4. `docs/changelogs/remove-analysis-package.md`
exists at the linked relative path from the repo root, so both links resolve.
A fresh interpreter importing `postprocessing.receptive_field.rf_clustering` loads 1644
modules with zero `analysis.*` entries and none of pyvista/open3d/igl/trimesh/seaborn.
With `QT_QPA_PLATFORM=offscreen`, `parse_launcher_config` yields 11 entries across exactly
five categories in declaration order (Setup, Primary, Preprocess, Merging, Postprocess),
every `dag_config` resolves to a file on disk, `LauncherWindow(...).show()` succeeds, and
`TaskDetailPanel.show_task()` renders all 62 tasks across the 11 surviving
`configs/*_dag.yaml` files without exception and with zero `analysis*` entries in
`sys.modules`.

**Files Modified:**
- `README.md` — new `## 🧭 Repository Scope` section (**tracked** — the durable home)
- `CLAUDE.md` — new `### Stage boundary` section (**local-only, gitignored, not committable**)

**Dependencies:** Phase 4

---

## Testing Plan

### Unit Tests
- [ ] `pytest` collects and passes with zero errors after Phase 3 (the 12 deleted test
      files must not leave dangling imports in `conftest.py`).
- [ ] New test for `RFMappingEngine.compute_selectivity()` — known-answer: a `Counter`
      with 4 spikes / 10 total at one point yields exactly 0.4.
- [ ] New test for `RFMappingEngine.cluster_receptive_field()` — two synthetic point
      blobs 50 mm apart with `eps=5.0` produce exactly 2 clusters, ranked by point count.
- [ ] New test asserting the relocated module imports no analysis symbol and no heavy 3D
      library (inspect `sys.modules` after a fresh import).

### Integration Tests
- [ ] Run the full `postprocess_workflow_kinect_auto` DAG on one session; confirm
      `center_on_receptive_field` completes and `aggregate_session` (which depends on it)
      still resolves its inputs.
- [ ] Launch `launch_pipeline_gui.py`; open every remaining DAG config and every task
      detail panel; confirm no exception and no orphaned feature-combination control.

### Manual Verification
- [ ] Byte-compare `blocks_rf_centered/*.csv`, `forearm_rf_centered/*.ply` and
      `rf_center_origin.json` against the Phase 1.4 baseline — must be identical.
- [ ] Time the import of `center_on_receptive_field.py` before and after; confirm the
      ~4.5 s eager-import cost is gone.
- [ ] Confirm the launcher shows five categories with no Analysis entry.

### Edge Cases
- [ ] DBSCAN finds no cluster → inputs still pass through unchanged and
      `rf_center_origin.json` records `status: "no_cluster_found"` (contract preserved, not
      converted to an exception).
- [ ] Dominant cluster has fewer than 10 points → `max(1, ceil(N * 0.1))` still yields ≥ 1 point.
- [ ] A DAG config whose task options contain a feature-combinations-shaped dict is opened
      after Phase 2 — confirm Task 2.3's decision holds and nothing crashes.

---

## Documentation Plan

- [x] Update root `CLAUDE.md`: package table, data-flow diagram, test-environment
      paragraph (Phase 4) and the stage-boundary section (Phase 5). **These edits are
      local-only — `CLAUDE.md` is gitignored.**
- [x] Add the stage-boundary rule to `README.md` (Phase 5 scope adjustment) — the
      **tracked**, contributor-facing home of the rule
- [x] Update `configs/README.md`: remove the two `analyse_workflow_*` rows (done in Phase 3)
- [x] Update `pyproject.toml` description
- [x] Add changelog entry: `docs/changelogs/remove-analysis-package.md`
- [x] **No changes to `docs/development/knowledge-base/` or `docs/development/plans/`** —
      explicitly out of scope by decision (upheld; only this plan document was edited)

---

## Rollback Plan

1. **Before merge:** the work is a single feature branch off `dev`. Abandon it —
   `git checkout dev && git branch -D refactor/remove-analysis-package`. Nothing is lost;
   the analysis repo is unaffected.
2. **After merge:** `git revert -m 1 <merge-commit>` restores all 165 files. Because the
   phases are separately committed, a partial revert is also possible — e.g. reverting
   Phase 2 alone restores the GUI dialog without resurrecting the package.
3. **Data considerations:** none. No migrations, no schema changes, no stored state. The
   only pipeline output affected is `center_on_receptive_field`'s, and Phase 1 requires
   byte-identical outputs, so no regeneration is needed.
4. **If the relocation proves wrong after merge:** the analysis repo retains its own copies
   of all five symbols, so the reference implementation remains available for comparison.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| The `8ed6e2b` `centroid_stroke` change is silently lost | Med | Med | Phase 0 blocks the deletion on an explicit port-or-drop decision |
| Deleting the dialog breaks the pipeline GUI at startup | Med | High | The 3 call sites in `task_detail_panel.py` are edited in the same commit; Phase 2 ends with a launcher smoke test |
| Relocation subtly changes RF center coordinates | Low | High | Verbatim move, no logic edits; byte-comparison against a pre-change baseline is a success criterion |
| The pass-through-on-failure contract is "cleaned up" into an exception during the move | Med | High | Called out in Definitions and in Edge Cases; it is a documented downstream contract, and one of the repo's few sanctioned fallbacks |
| ~25 knowledge-base notes describe deleted code and mislead a future reader | High | Low | Accepted by explicit decision. The Phase 4.5 changelog entry names the analysis repo so a reader can trace where the code went |
| `configs/launcher.yaml` loses comments or key order when edited | Low | Med | Use `ruamel.yaml` round-trip mode per the repo YAML rule |
| The CuPy import guard is disturbed while editing GUI imports | Low | High | Explicit constraint in Technical Design; `launch_pipeline_gui.py` is not in any phase's file list |
| A future contributor re-adds an `analysis` import | Med | Low | Phase 5 documents the rule plus the grep audit command |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 0 — Reconcile | 1–2 h (mostly the cross-repo port) | None |
| Phase 1 — Relocate | 1–2 h | Phase 0 |
| Phase 2 — GUI severance | 1 h | Phase 1 |
| Phase 3 — Delete | 1 h | Phase 2 |
| Phase 4 — Docs | 1 h | Phase 3 |
| Phase 5 — Enforce | 0.5 h | Phase 4 |

---

## Resolved Decisions

Both decisions that were open at drafting are now closed; no decision blocks implementation.

1. **Phase 0.1 — `8ed6e2b`: PORT** (not drop). The `centroid_stroke` change is a semantic
   change to how proximal/distal offsets are baselined, and the analysis repo's copy still
   reads `centroid_all`. Its `cmap` half is explicitly excluded. See Task 0.1.
2. **Phase 2.3 — `_is_feature_combinations_dict`: REMOVE.** Resolved by inspecting
   `configs/`: no surviving config uses the dict shape, and the deleted dialog was its only
   producer. See Task 2.3.

---

## References

- Sibling repository: `F:\GitHub\touch_projects\social-touch-semi-controlled-analysis`
  (372 commits, history preserved; standalone via `src/_vendor/`)
- Knowledge base: `note-cupy-import-order.md`, `note-qt-itemchanged-signal-recursion.md`,
  `note-kinect-depth-access-single-path.md`, `note-spatial-alignment-pipeline.md`,
  `note-analysis-pipeline-coordinate-spaces.md`
- Related plan (already implemented, mis-filed in `pending/`):
  `docs/development/plans/pending/rf-center-top-spike-filter.md`

---
