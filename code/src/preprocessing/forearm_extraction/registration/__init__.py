"""Registration subpackage for aligning multiple forearm point cloud snapshots.

Provides ICP-based registration (:class:`ForearmRegistrator`), a CSV
spatial transformer (:func:`transform_unified_csv`) that applies pre-computed
rigid transforms to somatosensory contact data, and an interactive parameter
exploration GUI (:class:`RegistrationWorkbench`).
"""

from .forearm_registrator import ForearmRegistrator
from .csv_spatial_transformer import transform_unified_csv
from .register_session_forearms import register_session_forearms
from .registration_workbench import RegistrationWorkbench, RegistrationResult

__all__ = [
    "ForearmRegistrator",
    "register_session_forearms",
    "transform_unified_csv",
    "RegistrationWorkbench",
    "RegistrationResult",
]
