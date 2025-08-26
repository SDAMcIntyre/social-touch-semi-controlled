# --- 1. THE MODEL ---
from dataclasses import dataclass, field
from typing import Dict, Any, List
import numpy as np

@dataclass
class ColorspaceEntry:
    """A dataclass to hold the analyzed data for a single frame."""
    frame_id: int
    colorspace_data: Dict[str, Any]

@dataclass
class AnnotationSession:
    """Manages the state of a labeling session. This is our core Model."""
    frames: List[np.ndarray]
    annotations: Dict[int, ColorspaceEntry] = field(default_factory=dict)
    current_frame_index: int = 0

    def add_annotation(self, colorspace_data: Dict[str, Any]):
        """Adds or updates an annotation for the current frame."""
        if colorspace_data:
            entry = ColorspaceEntry(self.current_frame_index, colorspace_data)
            self.annotations[self.current_frame_index] = entry
            print(f"✅ Annotation added for frame {self.current_frame_index}")

    def next_frame(self):
        if self.current_frame_index < len(self.frames) - 1:
            self.current_frame_index += 1

    def prev_frame(self):
        if self.current_frame_index > 0:
            self.current_frame_index -= 1
    
    def get_current_frame(self) -> np.ndarray:
        return self.frames[self.current_frame_index]

    def has_annotation_for_current_frame(self) -> bool:
        return self.current_frame_index in self.annotations