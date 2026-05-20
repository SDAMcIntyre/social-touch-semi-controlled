"""Shared bootstrap helpers for the analysis workflow entry scripts.

Provides session-discovery utilities and a generic stage-runner dispatcher
so ``analysis_workflow_processing.py`` and ``analysis_workflow_viewers.py``
can each be self-contained without duplicating infrastructure code.
"""

from .session_discovery import collect_unique_session_dirs, discover_input_items
from .stage_runner import run_pipeline_stages

__all__ = [
    "collect_unique_session_dirs",
    "discover_input_items",
    "run_pipeline_stages",
]
