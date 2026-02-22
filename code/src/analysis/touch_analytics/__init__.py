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

# Define the public API of the package
__all__ = [
    "DISCRETIZATION_CONFIG",
    "analyse_number_single_touches", # Kept if it exists externally, though not in provided files
    "analyse_ap_generation_efficacy", # Kept if it exists externally
    "generate_touch_summary_matrix",
    "generate_ap_efficacy_matrix",
]