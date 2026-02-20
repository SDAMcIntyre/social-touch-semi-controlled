# Git Workflow Overview

This is the reference guide for the project's git workflow. For quick start instructions, see [Commit Procedure](commit-procedure.md).

## Protected Branch Strategy

### Branch Hierarchy

```
main          ← Production-ready code, deployable at any time
  └── dev     ← Integration branch for completed features
       └── feature/...  ← Active development branches
```

**Rules:**
- `main` and `dev` are **protected** — never commit directly
- All changes go through feature branches → pull request → review → merge
- Feature branches are created from `dev`

### Branch Naming Convention

| Prefix | Purpose | Example |
|--------|---------|---------|
| `feature/` | New features | `feature/user-authentication` |
| `fix/` | Bug fixes | `fix/login-redirect-loop` |
| `refactor/` | Code restructuring | `refactor/extract-config-loader` |
| `docs/` | Documentation only | `docs/api-reference-update` |
| `test/` | Test additions/improvements | `test/auth-service-coverage` |

---

## Pre-Work Checklist

**Execute these commands BEFORE reading code or making ANY changes:**

```bash
# Step 1: ALWAYS verify current branch (FIRST COMMAND)
git branch --show-current

# Step 2: Check for uncommitted changes
git status

# Step 3: If on main or dev → create feature branch
current=$(git branch --show-current)
if [ "$current" = "main" ] || [ "$current" = "dev" ]; then
    echo "WARNING: On protected branch '$current'! Creating feature branch..."
    git checkout dev 2>/dev/null || true
    git pull origin dev
    git checkout -b feature/descriptive-name
fi

# Step 4: Verify you're on the correct feature branch
git branch --show-current  # Should show "feature/descriptive-name"

# Step 5: ONLY NOW is it safe to start reading/editing code
```

**Why this matters:**
- Commits to `main` can't be easily undone without force-pushing
- Force-pushing to protected branches disrupts all collaborators
- Feature branches enable code review and experimentation without risk

---

## Commit Message Format

See [Commit Procedure](commit-procedure.md) for detailed instructions.

Quick reference:

```
<type>(<scope>): <subject>

<body - explain what and why>

<footer - references, breaking changes>
```

### Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `style`, `perf`

---

## SSH Setup

SSH key-based authentication eliminates password prompts for all git operations.

### Quick Setup

```bash
# Generate key
ssh-keygen -t ed25519 -C "your_email@example.com"

# Copy public key to clipboard/screen
cat ~/.ssh/id_ed25519.pub

# Add to GitHub: https://github.com/settings/keys

# Test connection
ssh -T git@github.com

# Switch remote to SSH
git remote set-url origin git@github.com:user/repo.git
```

### Verification Checklist

- [ ] SSH key exists: `ls ~/.ssh/id_ed25519`
- [ ] Public key on GitHub: Check https://github.com/settings/keys
- [ ] Connection works: `ssh -T git@github.com`
- [ ] Remote uses SSH: `git remote -v` shows `git@github.com:...`
- [ ] Can fetch: `git fetch origin` (no password prompt)

---

## Pre-Commit Hooks

### Branch Protection Hook

Prevent accidental commits to protected branches:

```bash
#!/bin/bash
# .git/hooks/pre-commit

branch="$(git rev-parse --abbrev-ref HEAD)"
protected_branches=("main" "dev")

for protected in "${protected_branches[@]}"; do
    if [ "$branch" = "$protected" ]; then
        echo "ERROR: Direct commits to '$branch' are forbidden!"
        echo ""
        echo "To fix:"
        echo "  1. git stash"
        echo "  2. git checkout -b feature/your-feature-name"
        echo "  3. git stash pop"
        echo "  4. git commit"
        exit 1
    fi
done
exit 0
```

### Installing Hooks

Use the provided installer script:

```bash
bash skeleton/templates/install-hooks.sh.template
```

### Emergency Bypass

In rare cases where you need to bypass hooks (e.g., hotfix to main):

```bash
git commit --no-verify -m "hotfix: critical security patch"
```

**Only use `--no-verify` when:**
- Applying an emergency hotfix
- You understand and accept the risk
- You document why in the commit message

---

## Attribution Policy

### No AI Co-Authoring

Git history should reflect human authorship and accountability:

- Do **not** add `Co-Authored-By` for AI assistants (Claude, Copilot, etc.)
- AI tools are development assistants, like IDEs and linters
- Humans review, approve, and are accountable for all commits
- Focus commit messages on technical content, not tool attribution

```bash
# Correct
git commit -m "feat(search): add fuzzy matching algorithm"

# Incorrect
git commit -m "feat(search): add fuzzy matching algorithm

Co-Authored-By: AI-Tool <noreply@example.com>"
```

---

## Quick Reference

```bash
# Daily workflow
git branch --show-current          # Verify branch (FIRST!)
git checkout -b feature/my-work    # Create feature branch
# ... make changes ...
git add specific-files.py          # Stage specific files
git commit -m "feat(scope): description"
git push -u origin feature/my-work # Push and set upstream
# Create PR via GitHub or gh CLI
```

---

## See Also

- [Commit Procedure](commit-procedure.md) — Step-by-step commit workflow
- [skeleton/topics/01-git-workflow.md](../../../../skeleton/topics/01-git-workflow.md) — Original source material
