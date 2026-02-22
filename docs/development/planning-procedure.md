# Custom Planning Procedure

This document describes the standard planning workflow and document structure for feature development in this project.

## Quick Start

Creating a new plan:
1. Create file in `docs/development/plans/active/[feature-name].md`
2. Use the [planning template](#template)
3. Complete all required sections
4. Submit for review
5. Begin implementation after approval

---

## Planning Document Lifecycle

```
Draft  →  Approved  →  In Progress  →  Completed  →  Archived
  ↓         ↓            ↓               ↓            ↓
Review    Ready to    Building      Shipped      Historical
pending   build       feature       feature      reference
```

### Directory Structure

```
docs/development/plans/
├── active/                            # Plans currently in progress
│   ├── feature-name.md               # Primary plan document
│   └── feature-name/                 # Optional: phase breakdowns
│       ├── 01-foundation.md
│       ├── 02-core.md
│       └── 03-integration.md
├── completed/                         # Successfully shipped features
│   └── [shipped feature plans]
└── archived/                          # Old versions, superseded plans
    └── [old plans]
```

**completed/ vs archived/**

- **`completed/`** — Plans for features that were **shipped as designed**. The
  plan accurately reflects what was built. These are the canonical design
  records and should be kept indefinitely.
- **`archived/`** — Plans that were **superseded, abandoned, or substantially
  rewritten** before completion. They preserve decision history (why an
  approach was rejected) but do not describe the current system.

---

## Required Plan Sections

Every planning document must include these sections:

### 1. Overview (2-3 sentences)
**What:** High-level description of what is being built
**Why:** Problem being solved or improvement being made
**How:** Concise summary of the approach

### 2. Problem Statement
- Current limitation or issue
- Why it matters for the project
- User impact or technical debt implications

### 3. Goals

#### In Scope
1. Specific goal 1
2. Specific goal 2
3. Specific goal 3

#### Out of Scope
- Explicitly excluded feature or responsibility
- Future work that won't be included now

### 4. Success Criteria
Measurable checkboxes that define "done":
- [ ] Specific, measurable criterion 1
- [ ] Specific, measurable criterion 2
- [ ] Specific, measurable criterion 3

### 5. Technical Design

#### Approach
- High-level description of chosen solution
- Why this approach was selected

#### Alternatives Considered
Table comparing approaches:
| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Option A | [Pro] | [Con] | Chosen |
| Option B | [Pro] | [Con] | Rejected |

#### Architecture Changes
- New modules or classes being created
- Existing files that will be significantly modified
- Integration points with existing code
- Any architectural patterns being introduced

Include diagrams or file structure if helpful:
```
src/new_module/
├── __init__.py
├── core.py        — Main logic
├── models.py      — Data structures
└── gui/
    └── widget.py  — UI component
```

### 6. Implementation Plan

Break into logical phases with clear dependencies:

#### Phase 1: [Foundation]
**Goal:** What this phase accomplishes

**Tasks:**
- [ ] Task 1.1 — Description
- [ ] Task 1.2 — Description
- [ ] Task 1.3 — Description

**Files Modified:**
- `path/to/file.py` — Brief description of changes
- `path/to/other.py` — Brief description of changes

**Dependencies:** None

#### Phase 2: [Core Feature]
**Goal:** What this phase accomplishes

**Tasks:**
- [ ] Task 2.1 — Description
- [ ] Task 2.2 — Description

**Files Modified:**
- `path/to/file.py` — Brief description of changes

**Dependencies:** Phase 1

#### Phase 3: [Integration]
**Goal:** What this phase accomplishes

**Tasks:**
- [ ] Task 3.1 — Description
- [ ] Task 3.2 — Description

**Files Modified:**
- `path/to/file.py` — Brief description of changes

**Dependencies:** Phase 2

### 7. Testing Plan

#### Unit Tests
- [ ] Test case 1 — What is being tested and expected behavior
- [ ] Test case 2 — What is being tested and expected behavior
- [ ] Test case 3 — What is being tested and expected behavior

#### Integration Tests
- [ ] Test scenario 1 — How components work together
- [ ] Test scenario 2 — How components work together

#### Manual Verification
- [ ] Verification step 1 — Manual steps to confirm feature works
- [ ] Verification step 2 — Manual steps to confirm feature works

#### Edge Cases
- [ ] Edge case 1 — Unusual but valid input/state
- [ ] Edge case 2 — Boundary condition testing

### 8. Documentation Plan

- [ ] Update README.md with new commands/features
- [ ] Update CLAUDE.md with architecture changes
- [ ] Create/update user guide: `docs/guides/[feature].md`
- [ ] Create/update API documentation (if applicable)
- [ ] Add changelog entry: `docs/changelogs/[feature].md`
- [ ] Update inline code comments for complex logic

### 9. Rollback Plan

How to safely revert if something goes wrong:

1. **Before deployment:**
   - [Rollback step 1]
   - [Rollback step 2]

2. **Data considerations:**
   - Are there migrations? How to reverse them?
   - Are there breaking changes? How to handle?

3. **Rollback procedure:**
   - Which commits to revert
   - Which files to restore
   - Database/state reset steps

### 10. Risks and Mitigations

Identify potential blockers and how to handle them:

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| [Risk description] | Low/Med/High | Low/Med/High | [How to prevent or handle] |
| [Risk description] | Low/Med/High | Low/Med/High | [How to prevent or handle] |

---

## <a id="template"></a>Planning Template

Use this template when creating a new plan:

```markdown
# Plan: [Feature Name]

**Date:** YYYY-MM-DD
**Author:** [Your Name]
**Status:** Draft | Approved | In Progress | Completed
**Branch:** `feature/[branch-name]`

---

## Overview

[What is being built and why. 2-3 sentences.]

## Problem Statement

[What problem does this solve? What is the current limitation?]

## Goals

### In Scope
1. [Goal 1]
2. [Goal 2]
3. [Goal 3]

### Out of Scope
- [Explicitly excluded item 1]
- [Explicitly excluded item 2]

## Success Criteria

- [ ] [Measurable criterion 1]
- [ ] [Measurable criterion 2]
- [ ] [Measurable criterion 3]

---

## Technical Design

### Approach

[High-level description of the chosen approach and why it was selected over alternatives]

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| [Option A] | [Pro] | [Con] | Chosen |
| [Option B] | [Pro] | [Con] | Rejected |

### Architecture Changes

[Describe any architectural changes, new modules, or modified interfaces]

```
[Diagram or directory structure if helpful]
```

---

## Implementation Plan

### Phase 1: [Foundation]
**Goal:** [What this phase achieves]

- [ ] [Task 1.1]
- [ ] [Task 1.2]
- [ ] [Task 1.3]

**Files Modified:**
- `path/to/file.ext` — [What changes]

**Dependencies:** None

### Phase 2: [Core Feature]
**Goal:** [What this phase achieves]

- [ ] [Task 2.1]
- [ ] [Task 2.2]

**Files Modified:**
- `path/to/file.ext` — [What changes]

**Dependencies:** Phase 1

### Phase 3: [Integration]
**Goal:** [What this phase achieves]

- [ ] [Task 3.1]
- [ ] [Task 3.2]

**Files Modified:**
- `path/to/file.ext` — [What changes]

**Dependencies:** Phase 2

---

## Testing Plan

### Unit Tests
- [ ] [Test case 1]
- [ ] [Test case 2]

### Integration Tests
- [ ] [Test scenario 1]
- [ ] [Test scenario 2]

### Manual Verification
- [ ] [Verification step 1]
- [ ] [Verification step 2]

---

## Documentation Plan

- [ ] Update README.md with new commands/features
- [ ] Update CLAUDE.md with architecture changes
- [ ] Create/update user guide: `docs/guides/[feature].md`
- [ ] Add changelog entry: `docs/changelogs/[feature].md`

---

## Rollback Plan

[How to revert if something goes wrong]

1. [Rollback step 1]
2. [Rollback step 2]

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| [Risk 1] | Low/Med/High | Low/Med/High | [Mitigation strategy] |
| [Risk 2] | Low/Med/High | Low/Med/High | [Mitigation strategy] |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 | [estimate] | None |
| Phase 2 | [estimate] | Phase 1 |
| Phase 3 | [estimate] | Phase 2 |

---

## References

- Related Issue: #[number]
- Related Plans: `docs/development/plans/[related-plan].md`

---
```

---

## Best Practices

### Before Writing the Plan

- **Check knowledge base for relevant patterns** — As one of the first parallel
  research tasks, spawn an Explore subagent with the following prompt:

  > Read `docs/development/knowledge-base/README.md`. For each note listed in
  > the index, assess whether its problem class overlaps with the feature being
  > planned. Read the full content of any relevant notes and return a concise
  > summary of what applies: which constraints to respect, which approaches
  > were rejected and why, and which code patterns to reuse.

  Incorporate the findings into the Technical Design section — especially
  **Alternatives Considered** and **Architecture Constraints** — before the
  plan is submitted for review.

- **Explore the codebase** — Understand existing patterns, similar features, and constraints
- **Identify dependencies** — What other systems does this integrate with?
- **Consider alternatives** — Why is your chosen approach best? Document trade-offs
- **Ask questions** — Clarify requirements before planning

### During Planning

- **Be specific** — "Add validation" is vague; "Add `validate_references()` method to `ConfigLoader` that checks all topology node references" is actionable
- **List files explicitly** — Don't just say "update core modules", name them
- **Estimate scope** — How many lines of code? How many files?
- **Identify risks** — What could go wrong? Unknown dependencies? Complex integrations?
- **Consider testing** — How will this be verified? What are edge cases?

### Naming and Organization

- **File naming:** `feature-name.md` in `docs/development/plans/active/` directory
- **Branch naming:** `feature/[feature-name]` (use hyphens, lowercase)
- **Phase breakdown:** Only use phase files if plan is large (3+ phases with significant detail)

---

## Review Process

### Plan Review Checklist

Before approving a plan:

- [ ] **Clarity** — Goals are clear and measurable
- [ ] **Scope** — Well-defined boundaries (what's in/out of scope)
- [ ] **Technical soundness** — Approach is solid and considers alternatives
- [ ] **Testing** — Plan covers success and failure paths
- [ ] **Documentation** — Docs plan is included
- [ ] **Risk management** — Risks identified with mitigation strategies
- [ ] **No over-engineering** — Solution matches problem scope, not over-designed

### During Implementation

- [ ] Following the approved plan
- [ ] Each phase tested before proceeding
- [ ] Documentation updated alongside code
- [ ] Plan updated if approach changes
- [ ] No unplanned scope additions

---

## Common Mistakes to Avoid

❌ **Too vague**
- "Add features to improve user experience"
- "Refactor the system for better performance"

✅ **Specific and actionable**
- "Add LED state validation filter: moving average over 5 frames, 3σ threshold"
- "Extract centroid calculation into separate `CentroidCalculator` class for testability"

---

❌ **Missing out-of-scope**
- Reader doesn't know what's NOT being done
- Creates scope creep during implementation

✅ **Clear boundaries**
- Out of Scope: "Real-time video streaming" (future enhancement)
- Out of Scope: "GPU acceleration" (not in this phase)

---

❌ **Isolated design**
- No mention of integration with existing code
- Surprise incompatibilities discovered during implementation

✅ **Integrated thinking**
- "Extends `DataHandler` abstract class (used by 3 other modules)"
- "Adds optional parameter to `process()` (backward compatible)"

---

## Post-Implementation

### Completing a Plan

1. **Mark as completed:** Update `Status: Completed` and add completion date
2. **Move to completed:** `docs/development/plans/completed/[feature-name].md`
3. **Reference from code:** Link to plan from relevant module docs
4. **Lessons learned:** Optional note on what went well/differently

### Post-Mortem (Optional)

If the implementation revealed important learnings:
1. Document what changed from the plan
2. Why the change was necessary
3. What the team learned for future plans
4. Link from both plan and related code

---

## Related Documentation

- [Commit Procedure](../git/commit-procedure.md) — Commit message conventions
- [Git Workflow](../git/git-workflow.md) — Branch strategy
- [skeleton/topics/04-planning-process.md](../skeleton/topics/04-planning-process.md) — Full reference material
