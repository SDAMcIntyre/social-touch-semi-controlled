# Plan: Persistent Pipeline Progress Reporting

**Date:** 2026-03-17
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-03-18 11:29
**Branch:** `feature/unified-touch-analysis-pipeline`

---

## Overview

Add permanent, scrollable milestone lines to the unified touch analysis pipeline so the user can review what was processed after a run completes. Currently the tqdm progress bar overwrites itself on every update, leaving no history in the GUI console. Additionally, heatmap generation is not tracked in the progress bar at all.

## Problem Statement

The unified pipeline (`run_unified_touch_analysis`) has a tqdm progress bar that covers extraction and clustering steps. Two issues remain:

1. **Heatmap generation is invisible** — `_generate_session_heatmaps()` is called inside the clustering loop but is not counted in `n_steps` and does not increment the progress bar. The user gets no feedback during heatmap rendering.
2. **All progress is ephemeral in the GUI** — tqdm outputs carriage-return (`\r`) lines. The `ProcessOutputReader` routes these through `cr_line_received` → `ConsoleWidget.replace_last_line()`, so every update overwrites the previous one. When a phase finishes and the next begins, the completed phase's bar vanishes. The user cannot scroll back to see what was processed.

## Goals

### In Scope
1. Include heatmap generation in the progress bar step count and increment after each heatmap phase
2. Emit permanent milestone lines (via `print()`) after each extraction, clustering, and heatmap step so they persist in the GUI console
3. Add start/finish banner lines to bookend a pipeline run
4. Handle all edge cases (skipped steps, errors, empty input) so the bar always reaches 100%

### Out of Scope
- GUI widget changes (no modifications to `ConsoleWidget` or `ProcessOutputReader`)
- Progress reporting for other analysis workflows (AP efficacy, RF mapping)
- Per-session inner progress within heatmap generation (one milestone per clusterer is sufficient)
- Persistent progress logging to file

## Success Criteria

- [ ] `n_steps` includes heatmap generation; bar reaches exactly 100% on completion
- [ ] After each extraction step, a permanent line like `[extract] ST13-01 / max — 45 touches` appears in the console
- [ ] After each clustering step, a permanent line like `[cluster] max / kmeans — 3 clusters, 97 samples` appears
- [ ] After each heatmap step, a permanent line like `[heatmap] max / binning — 4 sessions` appears
- [ ] Skipped steps show `— up to date`; error/skip steps show appropriate status
- [ ] In the GUI, all milestone lines are scrollable — user can scroll up after completion to see full history
- [ ] The live tqdm bar continues to render on the last line without disrupting milestone lines
- [ ] Pipeline output files are identical before and after this change (no functional changes)

---

## Technical Design

### Approach

Use `print(..., flush=True)` to emit permanent milestone lines to stdout. The key insight: `print()` output has no `\r`, so `ProcessOutputReader` routes it through `line_received` → `ConsoleWidget.append_line()`, making it **permanent and scrollable**. The live tqdm bar continues to use `\r` and overwrites only itself on the last line. All changes are confined to `unified_pipeline.py`.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| `print()` milestone lines alongside tqdm | Simple, no GUI changes, robust | Two output mechanisms (print + tqdm) in same function | **Chosen** — simplicity and zero GUI risk outweigh minor style concern |
| Modify `ConsoleWidget` to detect completed bars | Keeps all output in tqdm | Fragile tqdm output parsing; couples GUI to tqdm internals | Rejected |
| `tqdm.write()` for milestone lines | Uses tqdm's own API | In a pipe context, `tqdm.write()` output still contains `\r` from bar management — routed to `replace_last_line`, not permanent | Rejected |
| Replace tqdm with custom progress protocol | Full control over output format | Over-engineered; tqdm already works for the live bar | Rejected |

### Architecture Changes

No new modules or classes. Changes are internal to `unified_pipeline.py`:

- `run_unified_touch_analysis()` — updated `n_steps` formula, start/finish banners
- `_extract_session()` — milestone `print()` after each profile extraction
- `_cluster_profile()` — milestone `print()` after each clustering step; separate progress increments for clustering vs heatmaps; pass context to `_generate_session_heatmaps`
- `_generate_session_heatmaps()` — new `profile_name` and `clusterer_name` parameters for milestone identification

### Console Output Example

```
=== unified pipeline: 2 sessions, 2 extractors, 2 clusterers ===
  [extract] ST13-01 / max — 45 touches
  [extract] ST13-01 / velocity — 45 touches
  [extract] ST13-02 / max — 52 touches
  [extract] ST13-02 / velocity — 52 touches
  [cluster] max / kmeans — 3 clusters, 97 samples
  [heatmap] max / kmeans — 2 sessions
  [cluster] max / binning — 20 bins, 97 samples
  [heatmap] max / binning — 2 sessions
  [cluster] velocity / kmeans — 3 clusters, 97 samples
  [heatmap] velocity / kmeans — 2 sessions
  [cluster] velocity / binning — 20 bins, 97 samples
  [heatmap] velocity / binning — 2 sessions
=== unified pipeline complete: 12 outputs ===
unified pipeline: 100%|██████████| 16/16 [01:00<00:00]
```

---

## Implementation Plan

### Phase 1: Update step count and add banners
**Goal:** Include heatmaps in progress tracking and add run bookends

- [ ] Update `n_steps` formula to `N*E + 2*E*C` (extraction + clustering + heatmaps)
- [ ] Add start banner `print()` before the `tqdm` context
- [ ] Add finish banner `print()` after the `tqdm` context
- [ ] Update empty-clustering-input skip to advance by `2 * len(clustering_profiles)` instead of `len(clustering_profiles)`

**Files Modified:**
- `code/src/analysis/touch_analytics/unified_pipeline.py` — `run_unified_touch_analysis()` function (lines 97-133)

**Dependencies:** None

### Phase 2: Add extraction milestones
**Goal:** Emit permanent lines after each extraction step

- [ ] After successful extraction (line 242): `print(f"  [extract] {session_id} / {profile_name} — {len(summary_df)} touches", flush=True)`
- [ ] After idempotency skip (line 206): `print(f"  [extract] {session_id} / {profile_name} — up to date", flush=True)`
- [ ] After extractor KeyError (line 218): `print(f"  [extract] {session_id} / {profile_name} — error: unknown extractor", flush=True)`
- [ ] After save failure (line 236): `print(f"  [extract] {session_id} / {profile_name} — error: save failed", flush=True)`

**Files Modified:**
- `code/src/analysis/touch_analytics/unified_pipeline.py` — `_extract_session()` function (lines 155-247)

**Dependencies:** None

### Phase 3: Split clustering/heatmap progress and add milestones
**Goal:** Separate progress increments for clustering and heatmaps; emit milestone lines

- [ ] Replace single `_advance_progress()` with explicit `progress.update(1)` calls at two points: after clustering, after heatmaps
- [ ] After successful clustering (line 427): `print(f"  [cluster] {profile_name} / {clusterer_name} — {n_clusters} clusters, {len(result_df)} samples", flush=True)` then `progress.update(1)`
- [ ] After idempotency skip (line 369): `print()` milestone + advance by 2 (cluster + heatmap)
- [ ] After clusterer KeyError / no features / clustering exception: `print()` milestone + advance by 2
- [ ] Add `profile_name` and `clusterer_name` parameters to `_generate_session_heatmaps()` signature
- [ ] After `_generate_session_heatmaps()` returns (line 433): `print(f"  [heatmap] {profile_name} / {clusterer_name} — {n_sessions} sessions", flush=True)` then `progress.update(1)`
- [ ] When heatmaps are skipped (< 2 features, no session_id): milestone with "skipped" + `progress.update(1)` in caller

**Files Modified:**
- `code/src/analysis/touch_analytics/unified_pipeline.py` — `_cluster_profile()` (lines 330-436), `_generate_session_heatmaps()` (lines 456-533)

**Dependencies:** Phase 1

---

## Testing Plan

### Manual Verification
- [ ] Run from CLI: `python code/scripts/analysis_workflow.py --dag-config configs/analyse_workflow_dag.yaml`
  - Milestone lines appear between tqdm bar updates
  - Bar reaches exactly 100% (no total mismatch)
  - Start/finish banners bookend the output
- [ ] Run from CLI with `force=False` when outputs exist — confirm "up to date" milestones appear
- [ ] Run from GUI launcher — milestone lines are permanent; scroll up after completion to see full history
- [ ] Run from GUI launcher — tqdm bar renders on last line, does not overwrite milestone lines

### Edge Cases
- [ ] Session with CSV load failure — progress advances by `len(extraction_profiles)`, no milestone lines for that session
- [ ] Session skipped by idempotency — milestone shows "up to date", bar advances
- [ ] Clusterer with no numeric features — `[heatmap]` milestone shows "skipped", progress advances by 2
- [ ] Empty clustering input (no session CSVs) — progress advances by `2 * len(clustering_profiles)`
- [ ] Heatmap generation exception for one session — other sessions still render; milestone still printed

### Regression
- [ ] Compare output files (CSVs, heatmap PNGs, metadata JSON) before and after — must be identical

---

## Documentation Plan

- [ ] Update existing plan `docs/development/plans/active/unified-pipeline-progress-reporting.md` to reference this plan as a follow-up
- [ ] No other documentation updates needed (internal UX improvement)

---

## Rollback Plan

1. Revert the commit(s) touching `unified_pipeline.py`
2. No data migrations or breaking changes — purely additive console output
3. No GUI code is modified, so no GUI rollback needed

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `print()` output interleaves with tqdm bar in terminal | Low | Low | tqdm handles stdout/stderr coordination; `flush=True` ensures timely delivery |
| `print()` buffering in subprocess pipe delays milestone display | Low | Low | `flush=True` forces immediate write; pipe reads line-by-line |
| Step count mismatch causes bar to not reach 100% | Medium | Medium | Careful accounting in all code paths (skip, error, success); verified in edge case testing |
| Milestone lines clutter output for large runs | Low | Low | Lines are concise (one per step); `QPlainTextEdit` has 10,000 line cap |

---

## References

- Predecessor plan: `docs/development/plans/active/unified-pipeline-progress-reporting.md`
- GUI console plan: `docs/development/plans/active/gui-embedded-console-output.md`
- Key files:
  - `code/src/analysis/touch_analytics/unified_pipeline.py` — all production changes
  - `code/src/utils/gui/dag_launcher/process_output_reader.py` — `\r` routing logic (no changes)
  - `code/src/utils/gui/dag_launcher/console_widget.py` — `append_line` vs `replace_last_line` (no changes)
