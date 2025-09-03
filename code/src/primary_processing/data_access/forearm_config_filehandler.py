import yaml
from pathlib import Path
from pydantic import ValidationError

from ..models.forearm_config import ForearmConfig


class ForearmConfigFileHandler:
    """
    A utility class to handle the saving and loading of the new 
    ForearmConfig files. It acts as the dedicated "file handler".
    """
    @staticmethod
    def save(session_config: ForearmConfig, output_path: Path):
        """
        Saves a ForearmConfig object to a YAML file.

        Args:
            session_config (ForearmConfig): The session data to save.
            output_path (Path): The destination file path for the new config.
        """
        print(f"   -> Saving session config to: {output_path}")
        # Ensure the parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # model_dump(mode='json') is a robust way to serialize Pydantic models,
        # correctly converting types like Path to strings for YAML compatibility.
        data_to_dump = session_config.model_dump(mode='json')
        
        with open(output_path, 'w') as f:
            yaml.dump(data_to_dump, f, default_flow_style=False, sort_keys=False, indent=2)

    @staticmethod
    def load(config_path: Path) -> ForearmConfig:
        """
        Loads and validates a session config YAML file into a ForearmConfig object.

        Args:
            config_path (Path): The path to the session config file to load.

        Returns:
            ForearmConfig: A validated instance of the session data.
            
        Raises:
            ValidationError: If the file content does not match the ForearmConfig schema.
            FileNotFoundError: If the file does not exist.
        """
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
            
        try:
            return ForearmConfig(**data)
        except ValidationError as e:
            print(f"❌ Validation error loading {config_path}: {e}")
            raise
