from pathlib import Path
from pydantic import BaseModel, DirectoryPath, FilePath, ValidationError, ConfigDict
from typing import Any, Dict, List, Optional

class SessionInputs(BaseModel):
    """
    The Pydantic data model defining the schema for session inputs.
    Configured to allow extra fields to support dynamic templates from project_config.yaml.
    """
    model_config = ConfigDict(extra='allow')

    session_id: str
    block_id: str
    objects_to_track: List[str]
    source_video: str  # Kept as str here for relative path validation before resolution
    stimulus_metadata: str
    hand_models_dir: str
    
    # Existing dynamic/implied fields
    nerve_primary_dir: Optional[str] = None 
    
    # New fields derived from the provided YAML example
    nerve_processed_dir: Optional[str] = None
    session_merged_output_dir: Optional[str] = None

    # Output directories
    video_primary_output_dir: str
    video_processed_output_dir: str
    session_primary_output_dir: str
    session_processed_output_dir: str


class KinectConfig:
    """
    Manages and provides validated access to session configuration data.
    Automatically resolves relative paths against a provided database root.
    """
    def __init__(self, config_data: Dict[str, Any], database_path: Path):
        """
        Initializes the SessionManager by validating the configuration data.

        Args:
            config_data (Dict[str, Any]): The raw, resolved configuration dictionary.
            database_path (Path): The root path to resolve relative paths against.

        Raises:
            ValueError: If the configuration data is invalid or validation fails.
        """
        try:
            if not config_data:
                raise ValueError("Configuration data is empty.")
            
            # 1. Validate structure using Pydantic (handles types and missing required fields)
            self.settings = SessionInputs(**config_data)
            self._database_path = database_path
            
            # 2. Define metadata keys that should NOT be resolved as paths
            self._non_path_keys = {"session_id", "block_id", "objects_to_track"}

        except ValidationError as e:
            raise ValueError(f"Session configuration validation failed: \n{e}") from e

    def _resolve_path(self, value: Any) -> Any:
        """Helper to resolve a value to an absolute path if it's a path string."""
        if isinstance(value, str) and value:
            # We assume strings in the config (excluding non_path_keys) are relative paths
            return self._database_path / value
        return value

    def __getattr__(self, name: str) -> Any:
        """
        Dynamic attribute access. 
        Retrieves fields from the settings model and resolves paths on the fly.
        """
        # Check if the attribute exists in the Pydantic model
        if hasattr(self.settings, name):
            value = getattr(self.settings, name)
            
            # If it's a metadata field, return raw value
            if name in self._non_path_keys:
                return value
            
            # Otherwise, assume it is a path and resolve it
            return self._resolve_path(value)
        
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    # Explicit properties for IDE autocompletion on standard and dynamically added fields
    
    @property
    def session_id(self) -> str:
        return self.settings.session_id

    @property
    def block_id(self) -> str:
        return self.settings.block_id
    
    @property
    def objects_to_track(self) -> List[str]:
        return self.settings.objects_to_track
    
    @property
    def source_video(self) -> Path:
        return self._resolve_path(self.settings.source_video)

    @property
    def stimulus_metadata(self) -> Path:
        return self._resolve_path(self.settings.stimulus_metadata)

    @property
    def hand_models_dir(self) -> Path:
        return self._resolve_path(self.settings.hand_models_dir)

    @property
    def nerve_primary_dir(self) -> Optional[Path]:
        if self.settings.nerve_primary_dir:
            return self._resolve_path(self.settings.nerve_primary_dir)
        return None
    
    @property
    def nerve_processed_dir(self) -> Optional[Path]:
        if self.settings.nerve_processed_dir:
            return self._resolve_path(self.settings.nerve_processed_dir)
        return None

    @property
    def session_merged_output_dir(self) -> Optional[Path]:
        if self.settings.session_merged_output_dir:
            return self._resolve_path(self.settings.session_merged_output_dir)
        return None

    @property
    def video_primary_output_dir(self) -> Path:
        return self._resolve_path(self.settings.video_primary_output_dir)

    @property
    def video_processed_output_dir(self) -> Path:
        return self._resolve_path(self.settings.video_processed_output_dir)
    
    @property
    def session_primary_output_dir(self) -> Path:
        return self._resolve_path(self.settings.session_primary_output_dir)

    @property
    def session_processed_output_dir(self) -> Path:
        return self._resolve_path(self.settings.session_processed_output_dir)

    def __repr__(self) -> str:
        return f"<KinectConfig session_id='{self.session_id}' block_id='{self.block_id}' source_video='{self.settings.source_video}'>"