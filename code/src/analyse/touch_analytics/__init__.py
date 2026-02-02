# __init__.py

from .touch_config import DISCRETIZATION_CONFIG
from .reporting import (
    TableRenderer, 
    GreatTablesStrategy, 
    TableContext
)
from .touch_analysis import analyse_number_single_touches
from .matrix_generation import generate_touch_summary_matrix

# Define the public API of the package
__all__ = [
    "DISCRETIZATION_CONFIG",
    "TableRenderer",
    "GreatTablesStrategy",
    "TableContext",
    "analyse_number_single_touches",
    "generate_touch_summary_matrix",
]