# The analysis package moved out of this repository

**Date:** 2026-08-06
**Branch:** `refactor/remove-analysis-package`
**Commit range:** `e990c47..` (the branch's four commits: `2e82642`, `82e5083`, `2c4f936`, `a1dbf8f`)

`social-touch-semi-controlled` now ends at the postprocessing stage. Everything
downstream of that — touch analytics and receptive-field mapping — was deleted
here and lives in its own repository.

## Where the code went

**`social-touch-semi-controlled-analysis`**
(`git@github.com:LHTMR/social-touch-semi-controlled-analysis.git`), 373 commits
with **full history preserved**. It is not a snapshot: the extraction happened
earlier, the two trees then diverged, and by the time of this deletion the
analysis repo was ahead on every file that differed — 34 of the 140 files, all
newer there, including a bug fix this repo never received. So the copy deleted
here was already stale. Nothing was lost by deleting it.

That repo is **self-contained**: it does not depend on this one. The shared
helpers it needs (`kinect_config`, `session_config_resolver`, `path_tools`,
`should_process_task`, `task_executor`, the monitoring package) are vendored
under `src/_vendor/`. Clone it and it runs.

If you are looking for a stage, a script or a config that used to be here, look
there first — the name almost certainly survived the move unchanged.

## What was removed here

| Thing | Count | Notes |
|---|---|---|
| `code/src/analysis/**` | 140 tracked files | plus 9 that were gitignored-on-disk (`code/src/analysis/CLAUDE.md`, and 8 modules under `receptive_field_mapping/data/` hidden by the `data/` ignore rule) — removed from the filesystem directly |
| Test files | 12 | `test_rf_*` (7), `test_fit_models`, `test_gmm_clusterer`, `test_feature_space_renderer`, `test_iff_windowed_mean`, `test_forearm_slim_uv` |
| `conftest.py` stub | 1 | the `analysis.pipeline` stub |
| Scripts in `code/scripts/` | 10 | 2 workflow entry points (`analysis_workflow_processing.py`, `analysis_workflow_viewers.py`) + 8 sandbox/diagnostic scripts |
| Config files | 5 | 3 `analyse_workflow*_dag.yaml` (tracked) + their 2 `.layout.json` siblings (gitignored per-workstation GUI state) |
| Launcher category | 1 | the `Analysis` block in `configs/launcher.yaml`; the GUI now shows five categories |
| GUI dialog | 1 | `utils/gui/dag_launcher/feature_combination_dialog.py` (137 lines) and its three call sites in `task_detail_panel.py` (153 lines) |

Net: ~54,400 deleted lines across the branch.

The dialog is worth calling out separately. It was the only `analysis.` import
anywhere in `code/src/` outside the package itself — a shared GUI helper
reaching *upward* into a downstream stage — and the import was wrapped in
`try/except ImportError` with empty-collection fallbacks, so a broken analysis
tree silently produced a dialog with zero checkboxes instead of raising. That
was the repo's only violation of both its layering and its fail-fast rule.
Both are gone.

Deleting the package also removed a **4.5 s eager-import cost** from
postprocessing: `center_on_receptive_field.py` imported one leaf config module,
but `analysis/receptive_field_mapping/__init__.py` was fully eager and dragged
in 120 modules including pyvista, open3d, igl, trimesh and seaborn. That script
now imports 3137 modules in 3.6 s instead of 4070 in 4.8 s, with zero
`analysis.*` entries.

## What was deliberately kept — and why

`code/src/postprocessing/receptive_field/rf_clustering.py` (210 lines) holds
five generic DBSCAN clustering helpers, moved verbatim out of
`analysis/receptive_field_mapping/{config,engine}.py`:

```
SelectivityDBSCANConfig   threshold 0.3, eps 5.0 (millimetres), min_samples 3,
                          min_cluster_points 5
RFCluster                 id, points (N,3), scores (N,), mean, count
RFMapResult               label, clusters, scores, counts
GroupedSpatialData        two Counters
RFMappingEngine           compute_selectivity(), cluster_receptive_field()
```

This is **not** a leftover. The postprocessing DAG task
`center_on_receptive_field` (stage C5) legitimately computes an RF center in
order to define the output coordinate origin for `blocks_rf_centered/`,
`forearm_rf_centered/` and `rf_center_origin.json` — and `aggregate_session`
depends on it. The distinction the split draws is:

> This repo is agnostic of downstream analysis **stages**, not of
> receptive-field **concepts**.

The relocation was a hard move, not a copy — nothing in this repo duplicates
the analysis repo's version, and the two are free to diverge from here. The
helpers qualified for relocation because their transitive import closure
contains no `analysis.*` module and their logic is expressed over plain numeric
types (`dict[(x,y,z)] -> float`, `ndarray`) rather than over neural/spike/CSV
schemas. `RFMappingColumnConfig` failed that test — it imports
`NERVE_SPIKE_COL` / `CONTACT_POINTS_COL` from `analysis.pipeline` — so it
stayed behind and was deleted, which is what kept the moved module free of any
inlined analysis constant.

Behaviour is byte-identical. Two sessions were run before and after the move,
covering both C5 branches (`2022-06-14_ST13-03` → `status: ok`, the translation
path; `2022-06-15_ST14-04` → `status: no_cluster_found`, the pass-through
path). All 17 output files match by SHA-256. The pass-through-on-failure
behaviour is a **deliberate, documented exception** to the repo's no-silent-
fallback rule: `aggregate_session` depends on that contract, so it must not be
"cleaned up" into an exception.

## Docs were left alone — read this before trusting a note

`docs/development/knowledge-base/` and `docs/development/plans/` were
**intentionally not touched**, by explicit decision. The consequence was
accepted knowingly rather than overlooked:

- **13 of 44 knowledge-base notes** name a path, module or config that no
  longer exists in this repo (`grep -rlE "code/src/analysis|receptive_field_mapping|touch_analytics|analyse_workflow" docs/development/knowledge-base/*.md`).
- **17 of 26 pending plans** are analysis-scoped and are un-actionable here.
- Most of `docs/development/plans/completed/` is analysis history.

**If you land on a note or plan describing code you cannot find, it is not
rotted documentation — the code moved.** Look in
`social-touch-semi-controlled-analysis`. The notes remain accurate about the
*behaviour* they describe; only their paths are wrong, and only relative to
this repo.

Notes that still apply here unchanged: `note-cupy-import-order.md`,
`note-qt-itemchanged-signal-recursion.md`,
`note-kinect-depth-access-single-path.md`, and the C5 section of
`note-spatial-alignment-pipeline.md` (with the caveat below).

## Known discrepancies, recorded not fixed

1. **`note-spatial-alignment-pipeline.md` §4 C5 has drifted from the code.** It
   says "selectivity-weighted centroid" and names `blocks_pca_calibrated/` as
   the input. The code weights by *spike count* over the top 10 % of the
   largest cluster, and its docstrings claim `blocks_contact_projected/` while
   the resolved input directory is in fact `blocks_pca_calibrated/`. The
   relocated module's docstring documents the code's real behaviour, not the
   note's. Neither the note nor the docstrings were corrected.
2. **`test_dag_config_model.py::test_get_config_entries_multi` now skips
   unconditionally.** It existed to assert that a DAG config can declare more
   than one config entry, and `analyse_workflow_dag.yaml` was the only root
   config that did. It sits behind a pre-existing `pytest.skip("config not
   found")` guard, so it passes while asserting nothing — a silent coverage
   hole. It needs a surviving multi-entry fixture or deletion.
3. **`DagConfigModel.get_combination_features` / `set_combination_features`
   have zero callers.** The deleted combination section was their only
   consumer. `add_combination` / `remove_combination` are still live.
4. **`code/src/analysis/CLAUDE.md` (116 lines) has no home.** It was gitignored
   here, so it was never tracked and is not in the deletion commit; the
   analysis repo's root `CLAUDE.md` is a different, generic document. If that
   guide is wanted, it must be re-created in the analysis repo.
5. **The root `CLAUDE.md` edits accompanying this change are local-only.**
   `.gitignore` ignores every `CLAUDE.md`, so this changelog is the only
   tracked record of the split.

## Test baseline

`pytest -q --continue-on-collection-errors` goes from **437 passed / 31 failed
/ 1 collection error** to **195 passed / 2 skipped / 0 failed / 0 collection
errors**. The drop is fully accounted for: the 12 deleted test files carried
232 passed + 30 failed + the collection error, and deleting the 3
`analyse_*_dag.yaml` configs removed 9 more passing tests because
`test_dag_config_model.py` parametrises three tests over
`CONFIGS_DIR.glob("*_dag.yaml")`. **No test regressed.** Note that plain
`pytest` aborted on the old collection error — `--continue-on-collection-errors`
was required to see the full result, and is no longer needed.

## Keeping the boundary

```bash
grep -rn "from analysis\|import analysis" code/    # expected: zero hits
```

This repo does not depend on the analysis repo, and must not start to. If you
need an analysis stage, work in the other repository.
