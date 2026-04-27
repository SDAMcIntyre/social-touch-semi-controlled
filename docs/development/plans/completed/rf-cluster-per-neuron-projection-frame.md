# Plan: Per-neuron 2D projection frame for cluster RF renders

**Date:** 2026-04-24
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `feature/rf-cluster-visualization-improvements` (existing — extends the active plan `rf-cluster-visualization-improvements.md`)

---

## Overview

Fix the 2D RF cluster heatmap so convex-hull perimeters enclose the heatmap, and make per-cluster PNGs of the same neuron share one 2D coordinate frame. Replace the current per-cluster spike-centroid projection origin with a single per-neuron projection origin computed from all of the neuron's contact points. Applied consistently in both the 2D and 3D render branches so everything lives in one frame per neuron.

## Problem Statement

Manual verification of the active plan `rf-cluster-visualization-improvements.md` (Phase 4) surfaced two linked defects in `rf_cluster_visualizer.py`:

1. **Perimeters do not enclose the heatmap in 2D.** The convex hulls (neuron and neuron ∩ cluster) are projected with a different `contact_centroid` than the spike-contact points, so they land in a different 2D coordinate frame despite enclosing the spike contacts in 3D.
2. **Per-cluster images of the same neuron are not visually comparable.** Each cluster's render seeds its projection from *that cluster's* spike centroid, so the (u, v) axes, tangent plane orientation, and 3D camera angle shift between clusters of the same neuron.

Both stem from the same choice: the projection frame is derived per (cluster, session) from the spike centroid rather than per-session from the full neuron's contact cloud.

Set containment in 3D is already correct (`heatmap_spike_contacts ⊆ neuron∩cluster_contacts ⊆ neuron_contacts`), and the 3D render path's rotation `R` is consistent within a single cluster. The defect is purely in how the 2D projection origin is chosen, and in re-choosing it for every cluster.

## Goals

### In Scope
1. Single `projection_centroid` per (session) = mean of `neuron_contacts_xyz`, reused for every cluster render of that session.
2. In the 2D render branch: project spike points, forearm background, and both hull sets with this same centroid.
3. In the 3D render branch: derive the tangent-plane rotation `R` from this same centroid so spike points, forearm mesh, and both hulls live in one rotated frame.
4. Fail-fast guard: `render_forearm_heatmap` raises `ValueError` when `render_context` is missing or `neuron_contacts_xyz` is empty (pipeline contract violation per `CLAUDE.md`).
5. Small refactor: hoist `neuron_contacts_xyz` computation above the per-cluster loop in `rf_cluster_pipeline.py` (cluster-independent, currently recomputed every iteration).

### Out of Scope
- Changing `compute_rf_metrics` centroid logic (not a render concern).
- Alpha shapes / concave hulls (the original plan already rejected these).
- Storing the per-neuron projection frame to disk for downstream tooling.
- Cross-session comparability (different sessions still project independently — this is deliberate).

## Success Criteria

- [ ] In every rendered 2D heatmap, perimeter (b) encloses every finite-colour cell and perimeter (a) encloses perimeter (b).
- [ ] For a given `session_id` with multiple clusters, the `_count.png` / `_ratio.png` figures share numerically identical (u, v) axis limits and forearm outline position across clusters.
- [ ] 3D renders for the same session share the same camera angle across clusters (minor visual shift vs. current output, not a regression).
- [ ] `rf_cluster_summary.json` and `cluster_description.json` are byte-for-byte unchanged (rendering-only fix).
- [ ] `render_forearm_heatmap` raises `ValueError` when invoked without a usable `render_context` — no silent fallback.
- [ ] Both `projection_method="tangent_plane"` and `projection_method="cylindrical_unwrap"` verified on at least one multi-cluster session.

---

## Technical Design

### Approach

`project_to_2d` is strongly centroid-dependent (see `rf_projection.py:20` for `project_tangent_plane` and `:80` for `project_cylindrical_unwrap` — the centroid drives the tangent-plane normal, the cylinder axis fit, and the seam placement). Same 3D point → different 2D coordinates under different centroids. So the fix is to use one centroid consistently, and the obvious choice is the one cloud that is a superset of every renderable point: the neuron's full contact cloud.

```python
projection_centroid = render_context.neuron_contacts_xyz.mean(axis=0)
```

This single value replaces every `contact_centroid` use in both branches of `render_forearm_heatmap()`. It is identical across every cluster render for a given session because `render_context.neuron_contacts_xyz` depends only on `session_id`.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Per-neuron centroid from all contacts (this plan) | One frame per neuron → hulls enclose heatmap AND cluster PNGs comparable across clusters | Camera/frame shifts slightly vs. today for neurons whose cluster spike centroid differs from their overall contact centroid | **Chosen** |
| Minimal fix: project hulls with the spike-contact centroid | 5-line change; fixes the reported bug | Doesn't address cross-cluster comparability; each cluster still has its own frame | Rejected — fixes the symptom, not the class |
| Per-session centroid from *spike* contacts pooled across clusters | Also stable per-neuron | Excludes non-firing contacts, so the frame shifts when new clusters are added / cluster assignments change | Rejected — neuron contact cloud is the more stable anchor |
| Store projection frame on disk for cross-run consistency | Reproducibility across re-runs | Adds a persistence concern for a purely visual fix | Out of scope |

### Architecture Changes

No new modules. Minor signature tightening in `render_forearm_heatmap`: `render_context` becomes effectively required (raise on None or empty `neuron_contacts_xyz`) rather than an optional default. The pipeline already always constructs one (see `rf_cluster_pipeline.py:500–533`), so no call-site changes beyond the one module.

Files touched:
- `code/src/analysis/receptive_field_mapping/rf_cluster_visualizer.py` — replace `contact_centroid` sources; remove `hull_centroid`; add fail-fast guard; one rotation `R` shared across branches.
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` — hoist `neuron_contacts_xyz` dict construction above the per-cluster loop.

---

## Implementation Plan

### Phase 1: Renderer — single projection frame per session
**Goal:** One centroid, one rotation, one frame — used for heatmap, forearm, and both hulls in both 2D and 3D branches.

**Started:** 2026-04-24
**Completed:** 2026-04-24

- [x] 1.1 — Near the top of `render_forearm_heatmap` (after the `display_metric` validation block at `rf_cluster_visualizer.py:209`), add:
  ```python
  if render_context is None or len(render_context.neuron_contacts_xyz) == 0:
      raise ValueError(
          "render_forearm_heatmap: render_context with non-empty neuron_contacts_xyz "
          f"is required (session={session_id}, cluster={cluster_label}). "
          "Pipeline contract violation — see rf_cluster_pipeline.py."
      )
  projection_centroid = render_context.neuron_contacts_xyz.mean(axis=0)
  ```
- [x] 1.2 — 2D branch (`rf_cluster_visualizer.py:230-315`): replace `contact_centroid = spike_xyz.mean(axis=0)` with `projection_centroid`. Replace the four `project_to_2d(..., contact_centroid, ...)` / `project_to_2d(..., hull_centroid, ...)` calls (spike points, forearm, perimeter a, perimeter b) with `projection_centroid`. Delete the `hull_centroid = ...` block at lines 268–272.
- [x] 1.3 — 3D branch (`rf_cluster_visualizer.py:334-540`): replace the early `contact_centroid = np.array([xs.mean(), ys.mean(), zs.mean()])` (line 338) with `projection_centroid`. The existing `R = compute_tangent_plane_rotation(forearm_vertices, contact_centroid)` at line 358 now reads `projection_centroid`. Hull alignment (`:528`, `:533`) and spike alignment (`:385`, `:491`) are unchanged — they already share `R`.
- [x] 1.4 — Audit `_compute_surface_normal` / `_normal_to_view_angles` usage for any remaining `contact_centroid` reference and switch to `projection_centroid`.

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_cluster_visualizer.py`

**Dependencies:** None.

### Phase 2: Pipeline cleanup — hoist neuron_contacts_xyz
**Goal:** Stop recomputing `neuron_contacts_xyz[sid]` once per cluster when it is cluster-independent.

**Started:** 2026-04-24
**Completed:** 2026-04-24

- [x] 2.1 — In `rf_cluster_pipeline.py`, move the `neuron_contacts_xyz` dict construction (currently inside the cluster loop at lines 518–525) to immediately after the `neuron_touches` dict at line 489, above `for cluster_idx, ...` at line 500.
- [x] 2.2 — Keep `neuron_cluster_contacts_xyz` inside the cluster loop — it genuinely depends on `cluster_label`.
- [x] 2.3 — Verify no other consumer reads `neuron_contacts_xyz` inside the cluster loop with a cluster-specific assumption (it should not — already cluster-independent today, just redundantly recomputed).

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py`

**Dependencies:** None (independent of Phase 1).

### Phase 3: Manual verification
**Goal:** Confirm the fix on the session that surfaced the bug, plus a cross-method sanity check.

- [ ] 3.1 — Re-run `map_receptive_fields_clustered` end-to-end via the GUI launcher on the multi-cluster session that revealed the defect.
- [ ] 3.2 — Per-cluster, confirm: perimeter (b) visibly encloses every finite-colour cell of the 2D heatmap; perimeter (a) encloses (b).
- [ ] 3.3 — Open two PNGs for the same session (different clusters) and overlay in an image viewer — forearm outline and (u, v) axes should be numerically identical.
- [ ] 3.4 — Re-run with `projection_method="cylindrical_unwrap"` on the same session — hulls must still enclose.
- [ ] 3.5 — Confirm 3D renders still show hulls enclosing the heatmap; camera may shift slightly vs. before but must be stable across clusters of the same session.

**Dependencies:** Phase 1.

---

## Testing Plan

### Unit Tests
- [ ] `test_render_forearm_heatmap_requires_render_context` — invoking without a context raises `ValueError` with a clear message naming the pipeline contract.
- [ ] `test_render_forearm_heatmap_requires_non_empty_neuron_contacts` — context with empty `neuron_contacts_xyz` raises `ValueError`.
- [ ] `test_projection_centroid_matches_neuron_contacts_mean` — given a synthetic `RFRenderContext`, the centroid the renderer computes equals `neuron_contacts_xyz.mean(axis=0)` (can be asserted by monkeypatching `project_to_2d` to capture its third argument across all calls within one render — all four calls must receive the same value).

### Integration Tests
- [ ] `test_two_clusters_same_session_share_2d_frame` — on a tiny fixture with one session and two clusters, render both and assert that the `uv_points` produced by the captured `project_to_2d` calls share identical axis bounds and forearm projection.
- [ ] `test_neuron_hull_encloses_spike_uv` — after projection with the shared centroid, every spike `uv` point lies inside the 2D convex hull of the projected `neuron_cluster_contacts_xyz`. (Mathematically trivial with the fix; fails on the pre-fix code.)

### Manual Verification
(covered in Phase 3 above)

### Edge Cases
- [ ] Neuron with a single contact — `neuron_contacts_xyz.mean()` is the contact itself; projection still well-defined; hull skipped per existing `_draw_hull_2d`/`_draw_hull_3d` guards.
- [ ] Neuron whose spike centroid differs markedly from its contact-cloud centroid — this is precisely the pathological case the fix addresses; rendered hulls must still enclose the heatmap.
- [ ] `projection_method="cylindrical_unwrap"` — the seam is now anchored to the neuron centroid rather than the cluster spike centroid; confirm no wraparound artefacts for off-centre clusters.

---

## Documentation Plan

- [ ] Update the active plan `docs/development/plans/active/rf-cluster-visualization-improvements.md` with a brief "Known issue — fixed by rf-cluster-per-neuron-projection-frame" note under Phase 4, so reviewers tracing the original plan understand the follow-up.
- [ ] Short docstring update on `render_forearm_heatmap` stating that `render_context` is required and that the projection frame is neuron-scoped.
- [ ] No CLAUDE.md update (local render fix, no architectural convention change).
- [ ] Memory update after merge: amend `project_postprocessing_pipeline_state.md` to note the per-neuron projection frame.

---

## Rollback Plan

1. **Before deployment:** Work continues on `feature/rf-cluster-visualization-improvements`. Merge is `--no-ff` per project convention; a single revert commit restores previous behaviour.
2. **Data considerations:** None. Rendering-only fix — no CSV schema changes, no JSON schema changes.
3. **Rollback procedure:** `git revert -m 1 <merge-sha>` on dev. Old PNGs remain on disk but are trivially regenerated by re-running the stage.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `_compute_surface_normal` / camera helpers use `contact_centroid` implicitly and break when renamed | Med | Low | Explicit audit in task 1.4; fall back to a local search for the name across the file during implementation. |
| Minor visual shift (3D camera angle, 2D axis bounds) triggers a re-review of existing cluster outputs | Med | Low | Expected and documented; call it out in the PR description. |
| `render_context` absent in an obscure call path (e.g., a smoke test) | Low | Med | Fail-fast raise surfaces the call site immediately; fix at the call site rather than re-adding a silent fallback. |
| `neuron_contacts_xyz` all-NaN after `.dropna()` for some session | Low | Med | Already handled by `.dropna()` in `rf_cluster_pipeline.py:523`; empty-array fail-fast in task 1.1 catches the residual case with a clear message. |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 (renderer — single projection frame) | 0.25 day | None |
| Phase 2 (pipeline hoist refactor) | 0.1 day | None |
| Phase 3 (manual verification) | 0.15 day | Phase 1 |

---

## References

- Parent plan: `docs/development/plans/active/rf-cluster-visualization-improvements.md` (this plan completes its Phase 4 verification)
- 3D→2D projection catalogue: `docs/development/knowledge-base/note-3d-to-2d-surface-projection-algorithms.md`
- Fail-fast convention: `CLAUDE.md` ("Fail-fast pipeline — no silent fallbacks")
- Scratch analysis (internal): `C:\Users\basil\.claude\plans\analyse-the-plan-docs-development-plans-floofy-bentley.md`
