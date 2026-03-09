# NOTE: Autonomous fast-forward merge caused loss of uncommitted changes

| Field    | Value |
|----------|-------|
| Date     | 2026-03-09 |
| Severity | **High** — uncommitted working-tree changes permanently destroyed |
| Trigger  | AI assistant autonomously chose `--ff-only` when user said "merge branch A into branch B" |

---

## What happened

1. User said: *"merge this branch to feature/launcher-yaml-driven-config"*
2. AI ran `git merge --ff-only feature/flexible-session-config-sources` without being asked
   to use fast-forward — the choice was made silently and autonomously.
3. User noticed provenance was lost and asked to reverse it.
4. AI ran `git reset --hard a76c101` to undo the fast-forward.
5. **`git reset --hard` wiped uncommitted working-tree changes** to several tracked files
   that had been modified before the session started:
   - `code/scripts/_3_preprocessing/_3_forearm_extraction/define_extraction_parameters.py`
   - `code/src/preprocessing/common/gui/frame_roi_square.py`
   - `code/src/preprocessing/forearm_extraction/arm_segmentation.py`
   - `code/src/preprocessing/forearm_extraction/registration/registration_workbench.py`
   - `code/src/utils/pipeline/monitoring/pipeline_monitor.py`
   - `code/src/utils/pipeline/monitoring/pipeline_monitor_live_plotter.py`
6. Those changes were not recoverable from git history because they had never been committed.

---

## Root cause

Two compounding failures, both from unsolicited autonomous decisions:

### Failure 1 — Autonomous merge strategy
"Merge branch A into branch B" is a neutral instruction. It says nothing about
fast-forward. The AI silently chose `--ff-only`, which:
- Erased branch provenance (commits from A look like they were made on B)
- Made recovery require a destructive operation

**Rule:** When the user says "merge", always use `--no-ff` to produce a merge commit
and preserve provenance, unless the user explicitly asks for a fast-forward.

### Failure 2 — `git reset --hard` without checking for uncommitted changes
Before running `git reset --hard`, the AI did not:
- Run `git status` to check for uncommitted changes
- Warn the user that `--hard` would destroy them
- Ask for confirmation

**Rule:** Before any `git reset --hard` (or any other working-tree-destructive command),
always run `git status` first. If uncommitted changes are present, stop and warn the
user before proceeding.

---

## Prevention rules (for AI assistants)

1. **Merge strategy is never implicit.** "Merge X into Y" → always use `--no-ff`.
   Only use `--ff-only` if the user explicitly says "fast-forward" or "ff".

2. **`git reset --hard` is a last resort, not a reflex.** Before running it:
   - Run `git status`
   - If any uncommitted changes exist (modified tracked files OR untracked files),
     stop and present the situation to the user: what will be lost, what alternatives
     exist (`git stash`, committing first, `--soft` or `--mixed` reset, etc.)
   - Only proceed after explicit user approval.

3. **Destructive git operations require explicit user confirmation**, regardless of
   whether the user said "just do it" in a general sense earlier. Each destructive
   action must be individually approved at the time it is taken.

---

## Safe alternatives that would have avoided the damage

| Goal | Safe command |
|------|-------------|
| Undo a fast-forward while preserving working tree | `git reset --mixed <sha>` |
| Undo a fast-forward while preserving staged changes | `git reset --soft <sha>` |
| Save uncommitted changes before any reset | `git stash` first, then reset |
| Merge with provenance from the start | `git merge --no-ff` |
