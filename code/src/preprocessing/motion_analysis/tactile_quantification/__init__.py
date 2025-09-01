# preprocessing/motion_analysis/tactile_quantification/__init__.py

"""
This file exposes the public API for the tactile quantification analysis sub-package.

By importing the key classes here, we allow users to access them directly
from the 'tactile_quantification' namespace, decoupling their code from our internal file structure.
"""

# Import from sub-modules to "lift" them into this package's namespace
from .core.objects_interaction_orchestrator import ObjectsInteractionOrchestrator


# Define what gets imported with 'from . import *'
__all__ = [
    "ObjectsInteractionOrchestrator",
]
