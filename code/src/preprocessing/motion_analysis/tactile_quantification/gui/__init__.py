"""Presentation layer for tactile quantification.

Exports are resolved **lazily** (PEP 562).  ``preprocessing.motion_analysis``
imports ``objects_interaction_visualizer`` from this package, which executes
this file; a normal eager re-export would therefore drag PyQt5, pyvistaqt and
VTK into every compute-only import of ``preprocessing.motion_analysis``.  The
lazy accessor keeps the name discoverable here while the Qt/VTK cost is paid
only by a caller that actually asks for the viewer.
"""

from typing import TYPE_CHECKING, Any

__all__ = [
    "CONTACT_SCALAR_NAME",
    "ContactDepthFieldViewer",
    "ContactDepthFrameView",
    "polydata_from_triangle_arrays",
]

if TYPE_CHECKING:  # pragma: no cover - import-time typing only.
    from .contact_depth_field_viewer import (
        CONTACT_SCALAR_NAME,
        ContactDepthFieldViewer,
        ContactDepthFrameView,
        polydata_from_triangle_arrays,
    )


def __getattr__(name: str) -> Any:
    """Import the viewer module on first attribute access."""
    if name in __all__:
        from . import contact_depth_field_viewer

        return getattr(contact_depth_field_viewer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list:
    return sorted(set(globals()) | set(__all__))
