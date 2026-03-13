# Plan: Agent Team Workflow — Orchestrated Development Pipeline

**Created:** 2026-03-12 22:00
**Approved:** —
**Completed:** —
**Author:** Basil Duvernoy
**Status:** Draft
**Branch:** `feature/agent-team-workflow`

---

## Overview

**What:** Restructure the Claude Code development workflow into a role-based
agent team driven by a single orchestrator skill.
**Why:** The current flat 4-step workflow (plan → implement → test → commit)
lacks separation of concerns, has no quality gate before committing, and
suffers from context loss between steps.
**How:** One `/develop` skill acts as a dispatcher and implementation guide —
it never generates code, but it maintains awareness of the full plan and serves
as the user's point of contact throughout the process. The user interacts with
the orchestrator in two steps: generate a plan, then implement it. During
implementation, the orchestrator can answer questions about the plan, report
progress, explain upcoming phases, and spawn the appropriate subagents.

## Problem Statement

The current workflow relies on four loosely defined steps with no structured
handoffs between them:

1. `/planning-procedure` on Opus — requirements gathering AND design are mixed
2. Implementation on Sonnet — reads the plan file but receives no role-specific priming
3. Testing with back-and-forth on Sonnet — unstructured, no review step
4. `/commit-procedure` on Sonnet — works well but has no quality gate before it

**Issues:**
- No separation between understanding user intent (PM work) and technical design
  (architecture work) — the Architect shouldn't need to negotiate requirements
- No code review or quality gate between implementation and commit
- Implementation agents receive no domain-specific knowledge injection
- Model selection is ad-hoc rather than role-driven
- Multiple separate skills force the user to manually orchestrate the pipeline
- Testing may require changes to both code and the original plan, but there is
  no protocol for tracking plan amendments

## Goals

### In Scope

1. Create a single orchestrator skill (`/develop`) that drives the full pipeline
2. Define 4 subagent roles (PM, Architect, Builder, Reviewer) spawned by the orchestrator
3. Implement a two-step user interaction: plan generation, then implementation
4. Support two implementation modes: phase-by-phase or automatic all-phases
5. Create domain-specialist knowledge files for Builder subagent injection
6. Establish a Requirements Brief artifact and directory (`briefs/`)
7. Define a plan amendment protocol for changes during testing

### Out of Scope

- Automated testing infrastructure (tests remain manual/user-driven)
- Model enforcement (model selection remains manual via `/model`)
- Persistent agent state across conversations (Claude Code limitation)
- CI/CD integration
- Replacing `/commit-procedure` (it works well and remains a separate lifecycle step)

## Success Criteria

- [ ] Single `/develop` skill handles both planning and implementation
- [ ] Orchestrator never generates code — only dispatches subagents and communicates with the user
- [ ] Orchestrator presents progress overview at start of implementation and between phases
- [ ] Orchestrator answers user questions about the plan without spawning subagents
- [ ] "Plan" step produces a Requirements Brief and a Plan document via subagents
- [ ] "Implement" step offers phase-by-phase or auto-all mode
- [ ] Each phase runs a build→review cycle (Builder then Reviewer subagent)
- [ ] Must-fix review items trigger a re-build before proceeding
- [ ] Builder spawns domain-specialist subagents when touching domain-specific code
- [ ] Plan amendment section is appended when changes occur during testing
- [ ] One real feature is run through the full pipeline end-to-end

---

## Technical Design

### Approach

Model the development workflow as a **software engineering team** with 4 roles,
all spawned by a single orchestrator. The orchestrator is a **dispatcher and
implementation guide**: it reads artifacts, decides which subagent to spawn
next, passes context, relays results, and — crucially — remains available to
the user as an interactive point of contact during implementation. It **must
never generate code** — that is the subagents' job — but it **does** communicate
about the plan, progress, and upcoming work.

The user interacts with the orchestrator through one skill (`/develop`) in two
steps:
1. **Plan** — orchestrator spawns PM subagent (Requirements Brief), then
   Architect subagent (Plan document), presents result for approval
2. **Implement** — orchestrator reads the approved plan, presents a progress
   overview (phases, status, what's next), asks the user for execution mode
   (one phase at a time, or all phases), and then manages the build→review
   cycle. Between phases (or at any time in one-at-a-time mode), the user can
   ask the orchestrator questions about the plan, remaining work, or phase
   details — the orchestrator answers directly from the plan document without
   spawning a subagent.

### The Agent Team

| # | Role | Spawned by | Model | Input | Output |
|---|------|-----------|-------|-------|--------|
| 0 | **Orchestrator** | User via `/develop` | Opus | User intent / plan reference | Dispatches subagents, routes artifacts, answers plan questions |
| 1 | PM | Orchestrator | Opus (subagent) | User intent | Requirements Brief |
| 2 | Architect | Orchestrator | Opus (subagent) | Requirements Brief | Plan document |
| 3 | Builder | Orchestrator | Sonnet (subagent) | Plan phase + specialist knowledge | Working code |
| 4 | Reviewer | Orchestrator | Sonnet (subagent) | Plan + git diff | Review report |

The **Tester** role remains user-driven (manual testing between phases or at the
end). The **Release Engineer** role is covered by the existing `/commit-procedure`.

### Orchestrator Design

The orchestrator's SKILL.md defines it as a **dispatcher and implementation
guide**:

> You are a dispatcher and implementation guide. You MUST NOT write code or
> make implementation decisions. Your job is to:
> 1. Determine which step the user is at (planning or implementing)
> 2. Spawn the right subagent with the right context
> 3. Relay results back to the user
> 4. Drive the build→review loop
> 5. Answer user questions about the plan, progress, and upcoming phases
>    directly — without spawning a subagent
> 6. Present a progress overview at the start of implementation and between
>    phases (completed phases, current phase, remaining phases)

The orchestrator **may** produce text that summarizes, explains, or navigates
the plan. It **must not** produce code, architectural decisions, or plan
content — those are the subagents' responsibility.

#### Orchestrator Flow

```
User invokes /develop
  │
  ├─ No approved plan exists (or user says "plan")
  │   │
  │   ├─ Spawn PM subagent
  │   │   Context: user intent, knowledge base, existing plans
  │   │   Output: Requirements Brief → briefs/[feature].md
  │   │
  │   ├─ Spawn Architect subagent
  │   │   Context: Requirements Brief, codebase exploration
  │   │   Output: Plan document → pending/[feature].md
  │   │
  │   └─ Present plan summary to user for approval
  │
  └─ Approved plan exists (or user says "implement")
      │
      ├─ Present progress overview:
      │   "Plan: [name] — [N] phases"
      │   "✓ Phase 1: [done]    ◆ Phase 2: [current]    ○ Phase 3: [upcoming]"
      │   Brief description of current/next phase goal
      │
      ├─ Ask: "One phase at a time, or all phases automatically?"
      │
      ├─ User asks a question about the plan ←──────────────────┐
      │   └─ Orchestrator answers directly from plan context    │
      │      (no subagent spawned — this is plan navigation,    │
      │       not code generation)                              │
      │      └─ Return to waiting for user instruction ─────────┘
      │
      └─ For each phase:
          │
          ├─ Present: "Starting Phase N: [goal] — [task count] tasks"
          │
          ├─ Spawn Builder subagent
          │   Context: plan phase, specialist knowledge files
          │   (spawns its own specialist sub-subagents if needed)
          │
          ├─ Spawn Reviewer subagent
          │   Context: plan phase + git diff since phase start
          │   Output: categorized report (must-fix / should-fix / nice-to-have)
          │
          ├─ If must-fix items exist:
          │   └─ Re-spawn Builder with review report → re-spawn Reviewer
          │      (loop until clean or max 2 iterations)
          │
          ├─ If one-at-a-time mode:
          │   └─ Present phase summary + updated progress overview
          │      Pause for user approval/testing/questions ─────┐
          │      User can ask questions (answered by             │
          │      orchestrator) or say "continue" ───────────────┘
          │
          └─ If auto mode:
              └─ Continue to next phase
          │
          After all phases complete:
          └─ Present final progress overview + suggest /commit-procedure
```

### Builder + Specialist Subagents

The Builder subagent delegates to domain specialists via the Agent tool when
changes touch domain-specific code:

| Specialist | Triggers when files touch... | Knowledge source |
|-----------|------------------------------|------------------|
| GUI | `code/src/utils/gui/`, viewers | `.claude/specialists/gui.md` |
| Pipeline | `code/scripts/`, DAG configs | `.claude/specialists/pipeline.md` |
| Neural/Kinect | `code/src/analysis/`, neural data | `.claude/specialists/neural-kinect.md` |
| Analysis | `code/src/analysis/receptive_field*` | `.claude/specialists/analysis.md` |

**Delegation rules:**
- Simple changes (config tweaks, small utilities) → Builder handles directly
- Domain-specific changes → Builder spawns the relevant specialist(s)
- Cross-domain changes → Builder spawns multiple specialists, coordinates

### Reviewer Subagent

The Reviewer produces a categorized report:

- **Must-fix:** Bugs, security issues, deviations from plan, broken contracts
- **Should-fix:** Code quality issues, missing error handling, unclear naming
- **Nice-to-have:** Style suggestions, minor improvements

Only **must-fix** items block the phase. The orchestrator only re-spawns the
Builder if must-fix items exist.

### Plan Amendment Protocol

When testing reveals the plan needs to change, the Builder appends a
`## Changes During Implementation` section to the plan doc, documenting:
- What changed
- Why it changed
- When it was changed (date)

The original design sections are preserved for reference. Git history
provides the authoritative diff.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Flat 4-step (current) | Simple, low overhead | No review, no role isolation, context loss | Replaced |
| Multiple user-facing skills (5 skills) | Each role has its own entry point | Too many commands, user must orchestrate manually | Rejected |
| All roles in one conversation | No handoff friction | Context pollution, model can't switch | Rejected |
| Separate conversations per role | Clean isolation | Context loss at every boundary, heavy manual handoff | Rejected |
| Single orchestrator + subagents | One entry point, context isolation, automatic build→review loop | More complex skill design | **Chosen** |

### Architecture Changes

**New modules/files:**

```
.claude/
├── skills/
│   └── develop/SKILL.md                 (new — Orchestrator)
├── specialists/
│   ├── gui.md                           (new — PyQt5, Open3D, widget patterns)
│   ├── pipeline.md                      (new — Prefect DAGs, config resolution)
│   ├── neural-kinect.md                 (new — Neural data, kinect alignment)
│   └── analysis.md                      (new — Receptive fields, statistics)

docs/development/plans/
└── briefs/                              (new directory)
    └── _template.md                     (new — Requirements Brief template)
```

**Modified files:**
- `CLAUDE.md` — Add agent team workflow section

**Unchanged files:**
- `.claude/skills/planning-procedure/SKILL.md` — Remains as-is (the Architect
  subagent prompt within the orchestrator references its logic, but the skill
  file itself is not modified)
- `.claude/skills/commit-procedure/SKILL.md` — Remains as-is

---

## Implementation Plan

### Phase 1: Foundation — Briefs Directory and Template
**Goal:** Establish the Requirements Brief artifact structure.
**Started:** —
**Completed:** —

- [ ] Task 1.1 — Create `docs/development/plans/briefs/` directory
- [ ] Task 1.2 — Create `docs/development/plans/briefs/_template.md` with
  Requirements Brief template (functional requirements, motivation, constraints,
  acceptance criteria)

**Files Modified:**
- `docs/development/plans/briefs/_template.md` — New file

**Dependencies:** None

### Phase 2: Specialist Knowledge Files
**Goal:** Create domain knowledge files for specialist subagent injection.
**Started:** —
**Completed:** —

- [ ] Task 2.1 — Create `.claude/specialists/gui.md` (PyQt5 patterns, Open3D
  SceneWidget constraints, widget conventions, key files/classes)
- [ ] Task 2.2 — Create `.claude/specialists/pipeline.md` (Prefect flow/task
  patterns, DAG config model, session config resolution, key files)
- [ ] Task 2.3 — Create `.claude/specialists/neural-kinect.md` (neural-kinect
  alignment, forearm extraction, coordinate transforms, key files)
- [ ] Task 2.4 — Create `.claude/specialists/analysis.md` (receptive field
  mapping, somatosensory calculations, statistical methods, key files)

**Files Modified:**
- `.claude/specialists/*.md` — 4 new files

**Dependencies:** None (can run in parallel with Phase 1)

### Phase 3: Orchestrator Skill — `/develop`
**Goal:** Create the single orchestrator skill that drives the full pipeline.
**Started:** —
**Completed:** —

- [ ] Task 3.1 — Create `.claude/skills/develop/SKILL.md` with:
  - Orchestrator role definition (dispatcher + implementation guide, never
    generates code but does communicate about plan/progress)
  - Step detection logic (plan vs implement vs user question)
  - Progress overview format and when to present it
  - Plan Q&A behaviour (answer from plan context, no subagent needed)
  - PM subagent spawning instructions and context injection
  - Architect subagent spawning instructions and context injection
  - Builder subagent spawning with specialist delegation rules
  - Reviewer subagent spawning and must-fix gating logic
  - Phase execution modes (one-at-a-time vs auto-all)
  - Artifact routing (briefs, plans, review reports)
- [ ] Task 3.2 — Include inline reference to Requirements Brief template
- [ ] Task 3.3 — Include inline reference to Plan template (from planning-procedure)
- [ ] Task 3.4 — Define PM subagent prompt (clarify intent, check knowledge base,
  write Requirements Brief)
- [ ] Task 3.5 — Define Architect subagent prompt (read brief, explore codebase,
  write plan following template)
- [ ] Task 3.6 — Define Builder subagent prompt (read plan phase, spawn specialists,
  flag deviations, amend plan when needed)
- [ ] Task 3.7 — Define Reviewer subagent prompt (check plan conformance, code quality,
  produce categorized report)

**Files Modified:**
- `.claude/skills/develop/SKILL.md` — New file

**Dependencies:** Phases 1 and 2

### Phase 4: Documentation
**Goal:** Update CLAUDE.md and planning procedure docs.
**Started:** —
**Completed:** —

- [ ] Task 4.1 — Add agent team workflow section to `CLAUDE.md`
- [ ] Task 4.2 — Update `docs/development/planning-procedure.md` to reference
  the new Requirements Brief step and the `/develop` workflow

**Files Modified:**
- `CLAUDE.md` — Add workflow section
- `docs/development/planning-procedure.md` — Update lifecycle

**Dependencies:** Phase 3

---

## Testing Plan

### Integration Tests
- [ ] Run `/develop` with "plan" intent → verify PM subagent produces brief
  in `docs/development/plans/briefs/`, then Architect produces plan in `pending/`
- [ ] Run `/develop` with "implement" on an approved plan → verify it presents
  progress overview, asks for execution mode, and spawns Builder for the first phase
- [ ] Verify Builder spawns GUI specialist when phase touches `code/src/utils/gui/`
- [ ] Verify Reviewer produces categorized report after Builder completes
- [ ] Verify must-fix items trigger a re-build cycle
- [ ] Verify one-at-a-time mode pauses between phases with updated progress overview
- [ ] Ask orchestrator a question about the plan mid-implementation → verify it
  answers from plan context without spawning a subagent

### Manual Verification
- [ ] Run one real feature through the full pipeline end-to-end
- [ ] Verify orchestrator never generates code (only dispatches and communicates)
- [ ] Verify orchestrator presents progress overview at implementation start and
  between phases
- [ ] Verify orchestrator can explain what a specific phase will do, what files
  it will touch, and what remains after it
- [ ] Verify plan amendment section is correctly appended during testing
- [ ] Verify auto-all mode runs through multiple phases without pausing
- [ ] Compare experience to old 4-step workflow

### Edge Cases
- [ ] Feature that touches multiple domains (cross-specialist coordination)
- [ ] Feature where testing requires a plan change (amendment protocol)
- [ ] Simple feature where specialist delegation is unnecessary
- [ ] User switches from auto mode to one-at-a-time mid-implementation
- [ ] Reviewer finds must-fix items on second review (nested re-build loop)
- [ ] User asks about a phase that has already been completed (orchestrator
  should summarize what was done and reference the review report)

---

## Documentation Plan

- [ ] Update `CLAUDE.md` with agent team workflow section
- [ ] Update `docs/development/planning-procedure.md` with Requirements Brief step
  and `/develop` workflow reference
- [ ] Requirements Brief template in `docs/development/plans/briefs/_template.md`

---

## Rollback Plan

All changes are to Claude Code configuration files (`.claude/skills/`,
`.claude/specialists/`) and documentation. Rollback is straightforward:

1. Revert the skill files to their previous state (git revert)
2. Delete new files (specialists, briefs template, develop skill)
3. The old workflow still works — `/planning-procedure` and `/commit-procedure`
   are unchanged

No data migrations, no breaking changes to code.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Orchestrator SKILL.md is too complex for the model to follow reliably | Medium | High | Keep instructions minimal and declarative; test with real features; iterate |
| Subagent context isolation is imperfect (Agent tool limitations) | Medium | Medium | Test with a real feature; fall back to separate conversations if isolation is insufficient |
| Specialist knowledge files become stale | Medium | Low | Review specialist files when touching those domains; keep them minimal |
| Too much overhead for small features | Medium | Medium | Document that the orchestrator should detect simple features and skip PM subagent, going directly to Architect |
| Must-fix re-build loop doesn't converge | Low | Medium | Cap re-build iterations at 2; after that, pause and ask user |
| Model selection remains manual | Low | Low | Document recommended models per role; accept this as a Claude Code limitation |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Briefs directory + template | Small | None |
| Phase 2: Specialist knowledge files | Medium | None (parallel with Phase 1) |
| Phase 3: Orchestrator skill | Large | Phases 1-2 |
| Phase 4: Documentation | Small | Phase 3 |

---

## References

- Working draft: `.claude/plans/iridescent-tumbling-dusk.md`
- Current planning procedure: `.claude/skills/planning-procedure/SKILL.md`
- Current commit procedure: `.claude/skills/commit-procedure/SKILL.md`
- Planning lifecycle: `docs/development/planning-procedure.md`
