"""Registration subpackage for aligning multiple forearm point cloud snapshots.

Provides ICP-based registration (:class:`ForearmRegistrator`), spatial
transformer utilities (:func:`transform_spatial_columns_in_place`) that apply
pre-computed rigid transforms to somatosensory contact data, and an interactive
parameter exploration GUI (:class:`RegistrationWorkbench`).
"""

from .forearm_registrator import ForearmRegistrator
from .csv_spatial_transformer import (
    apply_rigid_transform,
    find_applicable_transform_key,
    get_transform_schedule,
    parse_contact_points,
    serialize_contact_points,
    transform_spatial_columns_in_place,
    transform_spatial_columns_scheduled,
)
from .register_session_forearms import register_session_forearms
from .registration_workbench import RegistrationWorkbench, RegistrationResult

__all__ = [
    "apply_rigid_transform",
    "find_applicable_transform_key",
    "ForearmRegistrator",
    "get_transform_schedule",
    "parse_contact_points",
    "register_session_forearms",
    "serialize_contact_points",
    "transform_spatial_columns_in_place",
    "transform_spatial_columns_scheduled",
    "RegistrationWorkbench",
    "RegistrationResult",
]
