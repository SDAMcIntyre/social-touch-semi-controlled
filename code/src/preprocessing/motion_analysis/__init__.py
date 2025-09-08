# preprocessing/motion_analysis/__init__.py

"""
This file exposes the public API for the tactile quantification analysis sub-package.

By importing the key classes here, we allow users to access them directly
from the 'tactile_quantification' namespace, decoupling their code from our internal file structure.
"""

# Import from sub-modules to "lift" them into this package's namespace

from .tactile_quantification.core.objects_interaction_controller import ObjectsInteractionController

__all__ = [
    "ObjectsInteractionController",

]
