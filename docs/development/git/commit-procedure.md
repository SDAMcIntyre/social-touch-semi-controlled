# GitHub Commit Procedure

This document describes the standard commit, branching, and merging workflow
for this repository.

---

## ⚠️ Working Tree Safety Protocol

**Run `git status` before any operation that touches the working tree.**

This includes: `git checkout`, `git reset`, `git merge`, `git rebase`, `git stash`.

If uncommitted changes are present:
1. **Stop.** Do not proceed automatically.
2. Report what files are modified/staged and what will happen to them.
3. Ask the user how to handle them (commit, stash, or accept the risk).

```bash
git status   # always run this first
```

> **Why this matters:** `git reset --hard` and branch switches can silently destroy
> uncommitted changes on tracked files. These changes are **unrecoverable** from git
> history because they were never committed.
> See: `docs/development/knowledge-base/note-git-merge-autonomous-fast-forward.md`

---

## Quick Checklist

Before committing:
- [ ] Run `git status` — identify any unrelated uncommitted changes
- [ ] You're on a feature branch (not `main` or `dev`)
- [ ] You've staged only the files you intend to commit
- [ ] Your changes are atomic (related to one concern)
- [ ] Your commit message follows the format below

---

## Commit Message Format

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <subject>

<body - explain what and why>

<footer - references, breaking changes>
```

### Commit Types

| Type | When to Use | Example |
|------|-------------|---------|
| `feat` | New feature | `feat(auth): add JWT token refresh endpoint` |
| `fix` | Bug fix | `fix(config): handle missing optional fields gracefully` |
| `refactor` | Code restructuring (no behavior change) | `refactor(loader): extract validation into separate module` |
| `docs` | Documentation only | `docs(api): update endpoint examples` |
| `test` | Adding or updating tests | `test(auth): add coverage for edge cases` |
| `chore` | Build, CI, tooling changes | `chore(deps): upgrade pytest to v7.0` |
| `style` | Formatting (no logic change) | `style(formatting): remove trailing whitespace` |
| `perf` | Performance improvement | `perf(search): optimize query caching` |

### Scope Guidelines

The scope should specify what part of the codebase is affected:
- Use kebab-case (lowercase with hyphens): `hand-tracking`, `led-analysis`
- Keep it concise (1-3 words)
- Examples: `forearm-extraction`, `neural-kinect`, `sticker-tracking`, `dag-launcher`

### Subject Line Rules

- Start with lowercase
- Be imperative: "add", "fix", "update", NOT "added", "fixed", "updated"
- Don't end with a period
- Keep under 50 characters when possible
- Be specific and descriptive

### Bad Examples ❌

```bash
git commit -m "update stuff"
git commit -m "fix bug and add feature and update docs"
git commit -m "Fixed the thing"  # Not imperative
git commit -m "feat: work completed."  # Ends with period
```

### Good Examples ✅

```bash
git commit -m "feat(hand-tracking): add 3D hand visualization support"
git commit -m "fix(forearm-extraction): handle missing point cloud data"
git commit -m "refactor(sticker-tracking): extract centroid calculation"
git commit -m "docs(setup): add GPU configuration instructions"
git commit -m "test(neural-kinect): increase merger test coverage to 90%"
```

---

## Commit Body

The body should explain:
1. **What** — What changed
2. **Why** — Why this change is necessary
3. **How** — Technical approach (if non-obvious)

Keep lines under 72 characters. Leave a blank line between subject and body.

### Example with Body

```
feat(led-analysis): add LED state validation

The LED blinking detection was unreliable when camera noise
occurred. Added a moving average filter over 5 frames and
increased the threshold multiplier from 2x to 3x standard
deviation.

This resolves the false positives in low-light conditions
(Issue #234).
```

---

## Breaking Changes

Use the `BREAKING CHANGE:` footer when your change breaks backward compatibility:

```
feat(api): change authentication to OAuth2

BREAKING CHANGE: /login endpoint now requires client_id parameter.
Previous token-based auth is removed.

Migration: See docs/migration/v2-auth.md
```

---

## Pre-Commit Workflow

### Step 1: Check the working tree

```bash
git status
```

Identify any modified files unrelated to the current task. Do not stage them.
If they belong to another in-progress concern, consider stashing them first.

### Step 2: Stage Specific Files

```bash
# Stage individual files (preferred)
git add path/to/file1.py path/to/file2.py

# Or stage hunks interactively
git add -p
```

⚠️ **Avoid `git add .` or `git add -A`** — too easy to accidentally commit `.env`,
build artifacts, or unrelated in-progress changes.

### Step 3: Review Changes Before Committing

```bash
# Full diff of staged changes
git diff --cached

# Summary of staged files
git status
```

### Step 4: Create the Commit

```bash
# Simple commit (single line)
git commit -m "fix(core): resolve null pointer in data handler"

# Commit with body (use HEREDOC to avoid quoting issues)
git commit -m "$(cat <<'EOF'
feat(visualization): add 3D point cloud renderer

Implemented WebGL-based renderer for real-time visualization
of Kinect point cloud data. Supports rotation, zoom, and color
mapping.

Closes #456
EOF
)"
```

### Step 5: Verify the Commit

```bash
git log -1 --stat
```

---

## Branch Operations

### Creating a Feature Branch

Always check the working tree is clean before switching branches:

```bash
git status                          # check first
git checkout -b feature/my-feature  # create and switch
```

Branch from the correct base:
- `dev` — for new features
- `main` — for hotfixes only

### Merging a Branch

**Always use `--no-ff` (no fast-forward).** This is a project-wide rule.
Never choose a merge strategy autonomously — if the user says "merge branch X into Y",
execute `git merge --no-ff`. Only use fast-forward if the user explicitly requests it.

```bash
# Standard — always:
git status                        # confirm working tree is clean
git checkout target-branch
git merge --no-ff feature/my-feature -m "Merge branch 'feature/my-feature' into target-branch"

# Only if user explicitly says "fast-forward":
git merge --ff-only feature/my-feature
```

**Why `--no-ff`:** A fast-forward erases branch provenance — commits from the
merged branch become indistinguishable from commits made directly on the target.
A merge commit permanently records which branch each group of commits came from,
making history auditable and reversible without destructive operations.

---

## Destructive Operations Checklist

Before running any of the following:
`git reset --hard`, `git checkout -- .`, `git restore .`, `git clean -f`

**Protocol:**
1. Run `git status` and present the output
2. If **any** uncommitted changes exist — stop and warn the user:
   - Explicitly list what files will be affected
   - State that the changes are unrecoverable from git history
   - Suggest safe alternatives
3. Only proceed after the user gives explicit confirmation for that specific operation

### Reset mode reference

| Mode | Working tree | Index | Safe to run without warning? |
|------|-------------|-------|------------------------------|
| `--soft` | unchanged | unchanged | Yes — nothing is lost |
| `--mixed` | unchanged | reset | Yes — file edits preserved |
| `--hard` | reset ⚠️ | reset | **No — always warn first** |

### Safe alternatives to `git reset --hard`

```bash
# Save uncommitted changes before resetting
git stash
git reset --hard <sha>
git stash pop   # restore after reset

# Undo a commit but keep working tree intact
git reset --mixed HEAD~1

# Undo a commit and keep changes staged
git reset --soft HEAD~1
```

---

## Common Scenarios

### Scenario 1: Forgot to Add a File

```bash
# If you haven't pushed yet, you can amend
git add forgotten-file.py
git commit --amend --no-edit
# Don't amend after pushing!
```

### Scenario 2: Committed to the Wrong Branch

```bash
git log --oneline | head              # find your commit SHA
git checkout -b feature/correct-name
git cherry-pick <commit-sha>
git checkout wrong-branch
git reset --soft HEAD~1               # undo commit, keep changes staged
```

### Scenario 3: Pre-existing Uncommitted Changes During Branch Work

When starting work on a new feature but other files have uncommitted changes:

```bash
git stash push -m "wip: <description>"   # save unrelated changes
git checkout -b feature/new-feature
# ... do your work ...
git stash pop                            # restore unrelated changes
```

---

## Integration with Hooks

The project includes pre-commit hooks that:
- Prevent commits directly to `main` and `dev` branches
- Validate commit messages follow the format above
- Run linters and formatters

If a hook prevents your commit:
1. Read the error message
2. Fix the issue (switch branches, reformat)
3. Try committing again — never use `--no-verify` to bypass

**Emergency bypass** (genuine emergencies only — document why in the message):

```bash
git commit --no-verify -m "hotfix: critical security patch"
```

---

## Branch Strategy

- **`main`** — Production-ready, deployable at any time (protected)
- **`dev`** — Integration branch for completed features (protected)
- **`feature/*`**, **`fix/*`**, etc. — Your working branches

**Never commit directly to `main` or `dev`.** Use feature branches with pull requests.

---

## Authorship

All commits are authored by **Basil Duvernoy <basil.duvernoy@gmail.com>**.

- **Never** add a `Co-Authored-By: Claude` (or any AI assistant) trailer.
- Git operations must reflect only the human author.

---

## Related Documentation

- [Git Workflow](git-workflow.md) — Branch strategy and protection rules
- [KB: Autonomous fast-forward merge incident](../knowledge-base/note-git-merge-autonomous-fast-forward.md) — Why `--no-ff` and working tree checks are mandatory
