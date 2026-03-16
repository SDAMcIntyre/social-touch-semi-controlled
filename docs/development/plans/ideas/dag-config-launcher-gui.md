# Idea: DAG Config Launcher GUI

**Date:** 2026-02-23
**Status:** Idea

## Summary

A GUI that acts as a unified launcher for any DAG workflow config in the project.
Instead of manually editing YAML files to switch kinect config directories, toggle
tasks, or change processing options, the user opens the GUI, makes selections via
checkboxes and toggles, and the changes are written back to the DAG YAML on disk.

## Rough Approach

- **Workflow picker** — scan `configs/*.yaml` for all root-level DAG files and
  present them in a dropdown or list. Loading one populates the rest of the GUI
  from its contents.
- **Kinect config directory selector** — list every subdirectory under
  `configs/kinect_configs/` as checkboxes. Selecting one or more updates the
  `kinect_configs_directory` parameter (or equivalent) in the loaded DAG config.
- **Per-file granularity** — once a directory is selected, show its individual
  YAML files as a nested checklist so the user can exclude specific recording
  blocks without removing them from the directory.
- **Task parameter panel** — render all tasks from the DAG config with toggles
  for `enabled`, `force_processing`, `monitor`, and any other option fields.
  Changes are reflected immediately in the YAML structure.
- **Save** — write the modified config back to the original YAML file on disk
  (in-place update).

Likely GUI framework: PyQt5 (already a project dependency for the hand model
selector) or a lightweight alternative like tkinter.

## Notes

- Applies to all 7 DAG workflow configs, not just the preprocess workflow.
- The per-file exclusion feature implies either a new DAG parameter (e.g.
  `exclude_files`) or a filter mechanism in the batch-processing loop — needs
  design during full planning.
- Existing interactive GUIs in the project (hand model selector, arm segmentation)
  use PyQt5 and Open3D respectively; PyQt5 is the more natural fit for a
  form-based config editor.
- Related code: `DagConfigHandler` in
  `code/src/utils/pipeline/pipeline_config_manager.py`,
  workflow entry points in `code/scripts/`.
