"""Pipelines sub-package: DAG-invoked orchestrators for RF mapping workflows.

Modules are NOT eagerly imported here to avoid circular imports between
pipelines/ and rendering/ (rf_session_comparison_renderer imports pipeline modules).
Import pipeline modules directly:
    from analysis.receptive_field_mapping.pipelines.rf_simple_pipeline import run_simple_rf_mapping
"""
