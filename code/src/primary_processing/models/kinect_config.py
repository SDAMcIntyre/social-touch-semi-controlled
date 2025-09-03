from pathlib import Path
from pydantic import BaseModel, DirectoryPath, FilePath, ValidationError
from typing import Any, Dict

class SessionInputs(BaseModel):
    """The Pydantic data model defining the schema for session inputs."""
    session_id: str
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
            ValueError: If the 'session_inputs' key is missing or validation fails.
        """
        try:
            config_data_abs = {}
            for key, value in config_data.items():
                if key == 'session_id':
                    # Assign the original value without path concatenation
                    config_data_abs[key] = value
                else:
                    # Concatenate path for all other keys
                    config_data_abs[key] = database_path / value

            if not config_data_abs:
                raise ValueError("Configuration file is missing the 'session_inputs' section.")
            
            # Validate and load the data into the Pydantic model
            self.settings = SessionInputs(**config_data_abs)

        except ValidationError as e:
            # Re-raise Pydantic's detailed error for easier debugging
            raise ValueError(f"Session configuration validation failed: \n{e}") from e

    @property
    def session_id(self) -> Path:
        return self.settings.session_id
    
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