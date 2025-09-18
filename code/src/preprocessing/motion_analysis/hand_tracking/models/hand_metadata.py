from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class HandMetadataManager:
    """
    Manages the state and formatting of the selection metadata.

    This class holds all the data selected in the GUI and provides a method
    to generate the final, structured metadata dictionary.
    """
    source_video_path: Path
    selected_hand_model_path: Path
    selected_frame_number: int = 0
    is_left_hand: bool = False
    selected_points: dict[str, int | None] = field(default_factory=dict)

    def generate_output(self) -> dict:
        """
        Formats the stored data into the final dictionary for serialization.

        Returns:
            dict: A dictionary containing the structured metadata.

        Raises:
            ValueError: If essential data like video or model paths are missing.
        """
        if not all([self.source_video_path, self.selected_hand_model_path]):
            raise ValueError("Incomplete metadata: Source video or hand model path is missing.")

        # Convert the selected points from a dictionary to the required list format
        selected_points_list = [
            {"label": label, "vertex_id": vertex_id}
            for label, vertex_id in self.selected_points.items()
            if vertex_id is not None
        ]

        metadata = {
            "source_video_name": self.source_video_path.name,
            "selected_hand_model_name": self.selected_hand_model_path.name,
            "selected_frame_number": self.selected_frame_number,
            "hand_orientation": "left" if self.is_left_hand else "right",
            "selected_points": selected_points_list
        }
        return metadata