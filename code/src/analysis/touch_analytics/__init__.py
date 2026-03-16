# __init__.py

from .touch_config import DISCRETIZATION_CONFIG
from .reporting import (
    VisualReportingStrategy # Implicitly exported by usage, but not explicitly in original __all__
)
from .touch_analysis import (
    generate_unified_summary
)
from .matrix_generation import (
    generate_touch_summary_matrix,
    generate_ap_efficacy_matrix
)
from .session_summary import (
    generate_session_summary
)

# Define the public API of the package
__all__ = [
    "DISCRETIZATION_CONFIG",
    "generate_unified_summary",
    "generate_touch_summary_matrix",
    "generate_ap_efficacy_matrix",
    "generate_session_summary",
]