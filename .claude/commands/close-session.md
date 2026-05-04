# Close Work Session

Run this routine at the end of every work session to keep project knowledge current.

## Steps

### 1. Summarise what happened
- Briefly list: what stages ran, what bugs were fixed, what was learned, what was left incomplete.

### 2. Update the stage status table in CLAUDE.md
- Check `4_analysed/` on disk to verify which stage outputs actually exist.
- Update the Status column for any stages that changed (✅ complete / in progress / not started).
- Update the "Last checked" date above the table.

### 3. Update Common Gotchas in CLAUDE.md
- Add any new non-obvious behaviour discovered this session.
- Remove or correct any gotchas that turned out to be wrong.

### 4. Update the active session in CLAUDE.md if it changed.

### 5. Update memory files
- Check `/Users/sarmc72/.claude/projects/-Users-sarmc72-Documents-GitHub-social-touch-semi-controlled/memory/`
- Update `project_context.md` with current stage focus and any new findings.
- Add a `feedback_*.md` entry for any new working patterns or corrections from this session.

### 6. Check for uncommitted changes
- Run `git status` and `git diff`.
- If there are meaningful changes, offer to commit them with a descriptive message.
- Do not commit unless the user confirms.

### 7. Delete the session reference note

- Delete `.claude/SESSION_NOTE.md` if it exists (`rm -f .claude/SESSION_NOTE.md`).
- This file is gitignored and is intended to be ephemeral.

### 8. Flag anything incomplete
- List any tasks started but not finished, open questions, or known broken things.
- Note the natural next step (e.g. which pipeline stage to run next).
