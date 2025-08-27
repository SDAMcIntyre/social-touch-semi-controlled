"""
shared_types.py

This module contains simple, shared data structures like Enums and data classes
used across different parts of the application to avoid circular dependencies.
"""
from enum import Enum
from dataclasses import dataclass
from typing import List

class FrameAction(Enum):
    """Defines the types of actions that can be marked on a frame."""
    LABEL = "Re-label"
    DELETE = "Delete"


@dataclass
class FrameMark:
    """Represents a marked frame and the action to be taken."""
    action: FrameAction
    object_ids: List[str]
