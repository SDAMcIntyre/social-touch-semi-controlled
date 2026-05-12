# Idea: Contact Point Heatmap Export

## Summary

Export the per-session contact frequency heatmap (forearm mesh coloured by contact
count) as a static image, using the saved camera orientation from `set_rf_camera_settings`
as the viewpoint. Where outputs live and exactly how they are triggered is TBD.

## Rough Approach

The `RFCameraSettingsViewer` already builds the scene (`_build_scene`) and restores the
saved camera (`_restore_camera`). The same two steps can be run headlessly:

1. Load `PopulationData` for each session (same as the viewer).
2. Build the PyVista point cloud with the `contact_count` scalar (bincount heatmap,
   zeros → NaN, colourmap `hot`).
3. Load `rf_camera_settings.json` for the session; apply position / focal_point /
   up_vector / view_angle to the off-screen plotter.
4. Call `plotter.screenshot(output_path)` (PyVista off-screen rendering, no display
   needed).

## Notes

- **Where outputs go**: options are alongside existing RF artefacts
  (`4_analysed/receptive_field_maps_*/`) or in a dedicated
  `4_analysed/contact_point_heatmaps/<session>/` folder — TBD.
- **When it runs**: could be a standalone DAG task after `set_rf_camera_settings`, or
  a flag on the camera settings viewer itself ("Export on save").
- **Off-screen rendering**: PyVista supports `pv.start_xvfb()` on Linux and
  `pv.Plotter(off_screen=True)` on all platforms — no GUI needed for batch export.
- **Scalar range**: clamp to a fixed percentile (e.g. 1st–99th) across all sessions
  for a consistent colourbar, or per-session auto-range.
- **Format**: PNG at a configurable DPI; SVG not feasible for point clouds.
