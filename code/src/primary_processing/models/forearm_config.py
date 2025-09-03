from typing import List
from pydantic import BaseModel, FilePath
from pathlib import Path

# --- 1. New Classes for Session Management ---

class ForearmConfig(BaseModel):
    """
    The Pydantic data model that holds the configuration for a single session.
    This class acts as the "manager" for the session data, ensuring it is
    well-structured and valid.
    
    It contains the unique session ID and a list of links (file paths) to
    all the original YAML configuration files belonging to that session.
    """
    session_id: str
    session_primary_path: Path
    session_processed_path: Path
    config_file_links: List[FilePath]
