# Open Work Session

Run this routine at the start of every work session to orient quickly and prepare a session reference note.

## Steps

### 1. Confirm current pipeline state

- Read `CLAUDE.md` to get the stage status table and the active session ID.
- Check `4_analysed/` on disk (using `find` or `ls`) to verify which stage output directories actually exist, and reconcile against the table.
- Note any discrepancies (e.g. a stage marked "not started" that has output on disk, or vice versa).

### 2. Identify the next stage to run

- From the confirmed state, determine which stage is next in sequence and is unblocked (i.e. its `depends_on` stages are complete).
- If multiple stages are eligible (e.g. `touch_preparation` and `map_receptive_fields_simple` both have no dependencies), note all of them and ask the user which to focus on if it is not obvious from context.

### 3. Research the next stage

Read the implementation of the next stage thoroughly:

- Find the `@flow` function and the main pipeline module it calls (check `code/scripts/analysis_workflow.py` for the flow, then follow imports to the implementation).
- Read the implementation file(s) — understand what transforms or computations are applied, what the inputs and outputs are, how touches are grouped, what config options exist.
- Check `code/src/analysis/CLAUDE.md` for any relevant notes on that stage.
- Scan for issues: look for `fillna`, `iloc[0]`, unguarded column access, places where NaN could silently become 0, or any pattern that looks fragile given the known data characteristics (Kinect dropout, 30 Hz → 1 kHz forward-fill, boundary NaN rows, SAI sustained-firing behaviour).
- Cross-reference any known bugs in `docs/development/knowledge-base/` that touch the same code.

### 4. Write the session reference note

Write a markdown file to `.claude/SESSION_NOTE.md` (this file is gitignored and will be deleted at close-session). Structure it as follows:

```
# Session note — <date> — <stage name>

## Next stage: <task key>
**Status:** <confirmed status from disk>
**Run command:**
\`\`\`
conda run -n pcl_env python code/scripts/analysis_workflow.py \
  --dag-config configs/analyse_workflow_dag.yaml \
  --tasks <task_key>
\`\`\`

## What it does
<2–4 sentence summary of the transforms or computations applied>

### Transforms / outputs
<table or bullet list of: transform name | output columns | method>

### Key implementation files
<list with relative paths>

### Config options
<relevant YAML config block>

## Known issues to watch for
<numbered list of issues, each with: what the symptom is, which file/line causes it, and impact on results>

## Cross-references
<links to relevant knowledge-base docs if any>
```

### 5. Brief the user

Tell the user:
- Which stage is next and its confirmed status.
- The run command to use.
- How many issues were found and what the most significant one is.
- That the full reference note is at `.claude/SESSION_NOTE.md`.
