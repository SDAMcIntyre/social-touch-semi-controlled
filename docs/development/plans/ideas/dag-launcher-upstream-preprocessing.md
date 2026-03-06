# Idea: Upstream Preprocessing Integration into DAG Launcher

**Date:** 2026-02-24
**Status:** Idea

## Summary

The DAG config launcher GUI currently allows users to configure and select existing kinect configs and workflow tasks. However, two upstream preprocessing steps are required *before* any DAG workflow can run: (1) creation of kinect YAML config files from session metadata, and (2) form extraction from raw sensor data. Integrating these into the launcher GUI as optional pre-workflow steps would create a seamless, unified entrypoint for the entire processing pipeline.

## Rough Approach

- **YAML generation step** — expose a "Create Configs" button or panel in the launcher that lets users point at a session directory and generate kinect YAML files automatically. This uses existing config-generation logic but makes it discoverable from the GUI.
- **Form extraction step** — add a "Extract Forms" option that runs the form extraction process on selected sessions before launching the main DAG workflow.
- **Workflow sequencing** — wire these as optional preparatory steps in a preprocessing panel (left sidebar or bottom panel), so the user sees the full pipeline: Create → Extract → Configure → Run.

## Notes

- The current DAG launcher focuses on editing *existing* configs; this idea extends it to support config *creation* and data preparation.
- Form extraction may depend on sticker tracking or other kinect preprocessing; ensure dependencies are respected.
- Consider whether these steps should be blocking (must complete before main workflow) or optional (user can skip if already done).
- Related to current workflows: Kinect-based workflows and merging with neurodata both depend on these upstream steps being complete.
