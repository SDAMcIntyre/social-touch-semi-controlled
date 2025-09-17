# file: xyz_extractor_interface.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

class XYZExtractorInterface(ABC):
    """
    Abstract Base Class (Interface) for all 3D position extractors.
    
    This defines the "contract" that any extraction strategy must follow.
    """
    
    @classmethod
    @abstractmethod
    def can_process(cls, tracked_data: Any) -> bool:
        """
        Check if this extractor is compatible with the given tracking data format.
        """
        pass
    
    @classmethod
    @abstractmethod
    def should_process_row(cls, tracked_obj_row: pd.Series) -> bool:
        """
        Determines if a given row of tracking data should be processed.
        For example, it might check a 'status' column.
        """
        pass

    @abstractmethod
    def extract(
        self, 
        tracked_obj_row: pd.Series, 
        point_cloud: np.ndarray
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Extracts the 3D coordinates for a single object in a single frame.
        """
        pass

    @abstractmethod
    def get_empty_result(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Returns the default empty result when tracking fails or is skipped.
        """
        pass