# a new file: data_source_interface.py
from abc import ABC, abstractmethod
from typing import Iterator, Tuple
import pandas as pd

class TrackedDataInterface(ABC):
    """
    Interface for data sources that provide tracked object data.
    """
    @abstractmethod
    def get_items_for_frame(self, frame_index: int) -> Iterator[Tuple[str, pd.Series]]:
        """
        Yields the object name and its corresponding data row for a specific frame.
        """
        pass
    
    @property
    @abstractmethod
    def object_names(self) -> list[str]:
        """Returns a list of all unique object names in the data source."""
        pass