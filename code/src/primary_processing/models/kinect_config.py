from pathlib import Path
from pydantic import BaseModel, DirectoryPath, FilePath, ValidationError
from typing import Any, Dict, List

class SessionInputs(BaseModel):
    """The Pydantic data model defining the schema for session inputs."""
    session_id: str
    objects_to_track: List[str] # Added list of objects to track
    source_video: FilePath
    stimulus_metadata: FilePath
    hand_models_dir: DirectoryPath
    video_primary_output_dir: Path
    video_processed_output_dir: Path
    session_primary_output_dir: Path
    session_processed_output_dir: Path


class KinectConfig:
    """
    Manages and provides validated access to session configuration data.
    """
    def __init__(self, config_data: Dict[str, Any], database_path: Path):
        """
        Initializes the SessionManager by validating the configuration data.

        Args:
            config_data (Dict[str, Any]): The raw, resolved configuration dictionary.

        Raises:
            ValueError: If the configuration data is invalid or validation fails.
        """
        try:
            config_data_abs = {}
            # Define keys that are not file paths to prevent incorrect path joining
            non_path_keys = {"session_id", "objects_to_track"}

            for key, value in config_data.items():
                # Ignore any keys in the YAML that aren't defined in our model
                if key not in SessionInputs.model_fields:
                    continue

                if key in non_path_keys:
                    # Assign the original value without path concatenation
                    config_data_abs[key] = value
                else:
                    # Concatenate the database path for all path-like keys
                    config_data_abs[key] = database_path / value

            if not config_data:
                raise ValueError("Configuration data is empty.")
            
            # Validate and load the data into the Pydantic model
            self.settings = SessionInputs(**config_data_abs)

        except ValidationError as e:
            # Re-raise Pydantic's detailed error for easier debugging
            raise ValueError(f"Session configuration validation failed: \n{e}") from e

    @property
    def session_id(self) -> str:
        return self.settings.session_id
    
    @property
    def objects_to_track(self) -> List[str]:
        return self.settings.objects_to_track
    
    @property
    def source_video(self) -> Path:
        return self.settings.source_video

    @property
    def stimulus_metadata(self) -> Path:
        return self.settings.stimulus_metadata

    @property
    def hand_models_dir(self) -> Path:
        return self.settings.hand_models_dir

    @property
    def video_primary_output_dir(self) -> Path:
        return self.settings.video_primary_output_dir

    @property
    def video_processed_output_dir(self) -> Path:
        return self.settings.video_processed_output_dir
    
    @property
    def session_primary_output_dir(self) -> Path:
        return self.settings.session_primary_output_dir

    @property
    def session_processed_output_dir(self) -> Path:
        return self.settings.session_processed_output_dir

    def __repr__(self) -> str:
        return f"<SessionManager source_video='{self.settings.source_video.name}'>"