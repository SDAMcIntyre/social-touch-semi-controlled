# GitHub Commit Procedure

This document describes the standard commit workflow and conventions for this project.

## Quick Checklist

Before committing:
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
- Examples: `forearm-extraction`, `neural-kinect`, `sticker-tracking`

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
# Atomic, focused changes
git commit -m "feat(hand-tracking): add 3D hand visualization support"
git commit -m "fix(forearm-extraction): handle missing point cloud data"
git commit -m "refactor(sticker-tracking): extract centroid calculation"
git commit -m "docs(setup): add GPU configuration instructions"
git commit -m "test(neural-kinect): increase merger test coverage to 90%"
```

---

## Body Content

The body should explain:
1. **What** — What changed
2. **Why** — Why this change is necessary
3. **How** — Technical approach (if non-obvious)

Keep lines under 72 characters for readability. Leave a blank line between subject and body.

### Example with Body

```
feat(led-analysis): add LED state validation

The LED blinking detection was unreliable when camera noise
occurred. Added a moving average filter over 5 frames and
increased the threshold multiplier from 2x to 3x standard deviation.

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

### Step 1: Stage Specific Files

```bash
# Stage individual files (preferred)
git add path/to/file1.py path/to/file2.py

# Or use git add -p for interactive staging of hunks
git add -p
```

⚠️ **Avoid `git add .` or `git add -A`** — Too easy to accidentally commit `.env`, build artifacts, or unrelated changes.

### Step 2: Review Changes Before Committing

```bash
# See what you're about to commit
git diff --cached

# See summary of staged files
git status
```

### Step 3: Create the Commit

```bash
# Simple commit (single line)
git commit -m "fix(core): resolve null pointer in data handler"

# Commit with body and footer
git commit -m "feat(visualization): add 3D point cloud renderer

Implemented WebGL-based renderer for real-time visualization
of Kinect point cloud data. Supports rotation, zoom, and color
mapping.

Closes #456"
```

### Step 4: Verify the Commit

```bash
# See your commit and message
git log -1

# Or with more details
git log -1 --stat
```

---

## Common Scenarios

### Scenario 1: Forgot to Add a File

```bash
# If you haven't pushed yet, you can amend
git add forgotten-file.py
git commit --amend --no-edit

# This updates the previous commit (don't do this after pushing!)
```

### Scenario 2: Committed to the Wrong Branch

```bash
# Find your commit SHA
git log --oneline | head

# Create a new feature branch and cherry-pick it
git checkout -b feature/correct-name
git cherry-pick <commit-sha>

# Go back to the wrong branch and undo
git checkout wrong-branch
git reset --soft HEAD~1  # Keep changes, undo commit
```

### Scenario 3: Need to Uncommit Before Pushing

```bash
# Undo the commit but keep changes staged
git reset --soft HEAD~1

# Undo the commit and unstage changes
git reset HEAD~1
```

---

## Integration with Hooks

The project includes pre-commit hooks that:
- Prevent commits directly to `main` and `dev` branches
- Validate commit messages follow the format above
- Run linters and formatters

If a hook prevents your commit:
1. Read the error message
2. Fix the issue (e.g., switch branches, reformat)
3. Try committing again

**Emergency bypass** (use sparingly):

```bash
git commit --no-verify -m "hotfix: critical security patch"
```

Only use `--no-verify` for genuine emergencies and document why in the message.

---

## Related Documentation

- [Git Workflow](../git-workflow.md) — Branch strategy and protection
- [skeleton/topics/01-git-workflow.md](../../../../skeleton/topics/01-git-workflow.md) — Full reference material
