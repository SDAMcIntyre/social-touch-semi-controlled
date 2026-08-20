"""
postprocessing.gui
------------------
GUI components for visualising postprocessed (PCA-calibrated) data.
"""
from .postprocessed_scene_viewer import PostprocessedSceneViewer
from .before_after_step_viewer import BeforeAfterStepViewer
from .forearm_stage_inspector import ForearmStageInspector
from .stage_depth_field import (
    ACCEPTED_SPACES_BY_STAGE,
    CANONICAL_SPACE_BY_STAGE,
    PASSTHROUGH_SPACE_BY_STAGE,
    STAGE_LABELS,
    StageDepthField,
    resolve_stage_depth_field,
)
from .postprocessing_stage_viewer import PostprocessingStageViewer, StagePaths

__all__ = [
    "PostprocessedSceneViewer",
    "BeforeAfterStepViewer",
    "ForearmStageInspector",
    "PostprocessingStageViewer",
    "StagePaths",
    "STAGE_LABELS",
    "StageDepthField",
    "ACCEPTED_SPACES_BY_STAGE",
    "CANONICAL_SPACE_BY_STAGE",
    "PASSTHROUGH_SPACE_BY_STAGE",
    "resolve_stage_depth_field",
]
