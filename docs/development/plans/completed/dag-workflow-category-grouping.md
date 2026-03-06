# Plan: DAG Workflow Category Grouping

**Date:** 2026-03-06
**Author:** Basil Duvernoy
**Status:** Completed
**Completion Date:** 2026-03-06
**Branch:** `feature/multi-snapshot-forearm-registration` (or new branch TBD)

---

## Overview

Group the DAG launcher's workflow selector buttons by category, derived from
each config filename's prefix (the part before the first underscore). This makes
it easier for the user to see which workflows belong together at a glance.

## Problem Statement

The workflow selector currently shows all `*_dag.yaml` configs as a flat list of
toggle buttons. With 9 workflows spanning 6 categories (analyse, merging,
postprocess, preprocess, primary, view), the list is hard to scan and offers no
visual hierarchy.

## Goals

### In Scope
1. Group workflow buttons under labelled category headers derived from the
   filename prefix (text before the first `_`).
2. Add a visual separator between categories (label + optional divider line).
3. Preserve the existing display order defined in `_ORDERED_STEMS`.

### Out of Scope
- Collapsible/expandable category sections (future enhancement).
- Changes to the DAG YAML filenames or directory structure.
- Any functional changes to workflow selection behaviour or signal emission.

## Success Criteria

- [x] Buttons are visually grouped by prefix category with a readable label.
- [x] Categories appear in the same order as the existing `_ORDERED_STEMS` list.
- [x] Workflows not listed in `_ORDERED_STEMS` are appended in a catch-all group.
- [x] Selecting a workflow still emits `workflow_changed` with the correct path.
- [x] `current_path()` and `select_path()` public API unchanged.

---

## Technical Design

### Approach

After sorting workflows by the existing `_order_key`, iterate through the sorted
list and detect category boundaries (prefix changes). Before the first button of
each new category, insert a `QLabel` styled as a section header and optionally a
`QFrame` horizontal line into the `_btn_layout`. The prefix is title-cased for
display (e.g. "preprocess" becomes "Preprocess").

No new classes or files are needed — the change is contained within
`WorkflowSelector._scan()` and its imports.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| QLabel headers between groups | Simple, no new widgets, preserves flat layout | Categories not collapsible | **Chosen** |
| Nested QGroupBox per category | Strong visual boundary, native Qt grouping | Heavier nesting, more padding overhead, changes widget hierarchy | Rejected |
| QTreeWidget with categories | Collapsible, standard tree pattern | Overkill for ~9 items, different selection UX | Rejected |

### Architecture Changes

- **Modified file:** `code/src/utils/gui/dag_launcher/workflow_selector.py`
  - Add `QLabel` and `QFrame` to imports.
  - Update `_scan()` to insert category headers when the prefix changes.
  - Add a small helper `_category_prefix(stem: str) -> str` to extract the
    first word before `_`.

---

## Implementation Plan

### Phase 1: Category headers in workflow selector
**Goal:** Visually separate workflow buttons by category.

**Tasks:**
- [x] Task 1.1 — Add `QLabel`, `QFrame` to PyQt5 imports.
- [x] Task 1.2 — Add `_category_prefix()` helper (extracts text before first `_`).
- [x] Task 1.3 — In `_scan()`, track the current category; when it changes, insert
      a styled `QLabel` header (and optionally a `QFrame` separator line) into
      `_btn_layout` before the next button.
- [x] Task 1.4 — Ensure `_ORDERED_STEMS` ordering is preserved across categories.

**Files Modified:**
- `code/src/utils/gui/dag_launcher/workflow_selector.py` — imports, `_scan()`,
  new `_category_prefix()`.

**Dependencies:** None

---

## Testing Plan

### Manual Verification
- [x] Launch the DAG config GUI and confirm buttons are grouped under labelled
      category headers (Primary, Preprocess, Merging, Postprocess, Analyse, View).
- [x] Confirm clicking any button still loads the correct workflow config.
- [x] Add a new `*_dag.yaml` file with an unknown prefix and confirm it appears
      in a catch-all group at the bottom.
- [x] Confirm `select_path()` still programmatically selects the correct button.

### Edge Cases
- [x] Single-workflow category (e.g. "analyse") still gets a header.
- [x] A config file with no underscore in its stem is handled gracefully.

---

## Documentation Plan

- [x] No external documentation changes needed (internal GUI enhancement).

---

## Rollback Plan

Revert the single modified file to restore the flat button list:
```
git checkout HEAD -- code/src/utils/gui/dag_launcher/workflow_selector.py
```

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Category labels consume too much vertical space | Low | Low | Use compact styling (small font, minimal margin) |
| Prefix extraction gives poor labels for some files | Low | Low | Can add a display-name mapping dict if needed |
