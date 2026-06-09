"""Canonical output subdirectory names under ``4_analysed/``."""

# -- Foundation ---------------------------------------------------------------
TOUCH_PREPARE_SESSIONS = "touch_prepare_sessions"
TOUCH_COMPUTE_SERIES = "touch_compute_series"
TOUCH_SUMMARIZE_BLOCKS = "touch_summarize_blocks"

# -- Spatial sensitivity ------------------------------------------------------
SPATIAL_MAP_SINGLE_TOUCH = "spatial_map_single_touch"
SPATIAL_SET_CAMERA = "spatial_set_camera"
SPATIAL_MAP_BASELINE = "spatial_map_baseline"
SPATIAL_SLIM_UV = "spatial_slim_uv"
SPATIAL_EXTRACT_BOUNDARIES = "spatial_extract_boundaries"
SPATIAL_COMPARE_BOUNDARIES = "spatial_compare_boundaries"
SPATIAL_COMPARE_RF_CENTERS = "spatial_compare_rf_centers"

# -- Stimulus sensitivity -----------------------------------------------------
STIMULUS_EXTRACT_FEATURES = "stimulus_extract_features"
STIMULUS_CLUSTER_TOUCHES = "stimulus_cluster_touches"
STIMULUS_COMPARE_CLUSTERS = "stimulus_compare_clusters"
STIMULUS_RENDER_RADAR = "stimulus_render_radar"
STIMULUS_COMPARE_SESSIONS = "stimulus_compare_sessions"
STIMULUS_RESPONSE_TUNING = "stimulus_response_tuning"
STIMULUS_RESPONSE_INSTRUCTION_TUNING = "stimulus_response_instruction_tuning"
STIMULUS_ANALYSE_EFFICACY = "stimulus_analyse_efficacy"

# -- Cross-domain -------------------------------------------------------------
CROSS_MAP_FEATURE_GRID = "cross_map_feature_grid"
CROSS_EXTRACT_GRID_METRICS = "cross_extract_grid_metrics"
CROSS_RENDER_GRID_METRICS = "cross_render_grid_metrics"
CROSS_RENDER_SESSIONS = "cross_render_sessions"
CROSS_CLUSTER_RF = "cross_cluster_rf"

# -- Migration mapping (old name → new name) ----------------------------------
RENAME_MAPPING: dict[str, str] = {
    "preparation": TOUCH_PREPARE_SESSIONS,
    "series_transforms": TOUCH_COMPUTE_SERIES,
    "session_summary": TOUCH_SUMMARIZE_BLOCKS,
    "single_touch_rf_maps": SPATIAL_MAP_SINGLE_TOUCH,
    "rf_camera_settings": SPATIAL_SET_CAMERA,
    "receptive_field_maps_simple": SPATIAL_MAP_BASELINE,
    "forearm_slim_uv": SPATIAL_SLIM_UV,
    "population_response_fields": SPATIAL_EXTRACT_BOUNDARIES,
    "session_rf_boundary_comparison": SPATIAL_COMPARE_BOUNDARIES,
    "rf_center_proximal_distal": SPATIAL_COMPARE_RF_CENTERS,
    "touch_features": STIMULUS_EXTRACT_FEATURES,
    "touch_clusters": STIMULUS_CLUSTER_TOUCHES,
    "touch_comparisons": STIMULUS_COMPARE_CLUSTERS,
    "touch_feature_radar": STIMULUS_RENDER_RADAR,
    "stimulus_compare_sessions": STIMULUS_COMPARE_SESSIONS,
    "stimulus_iff_tuning_curves": STIMULUS_RESPONSE_TUNING,
    "stimulus_iff_instruction_tuning": STIMULUS_RESPONSE_INSTRUCTION_TUNING,
    "ap_efficacy": STIMULUS_ANALYSE_EFFICACY,
    "population_rf_grid": CROSS_MAP_FEATURE_GRID,
    "population_rf_grid_metrics": CROSS_EXTRACT_GRID_METRICS,
    "population_rf_grid_metrics_heatmaps": CROSS_RENDER_GRID_METRICS,
    "session_comparison": CROSS_RENDER_SESSIONS,
    "receptive_field_maps_clustered": CROSS_CLUSTER_RF,
}
