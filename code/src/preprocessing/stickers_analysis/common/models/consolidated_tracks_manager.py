import pandas as pd
from typing import Dict, Optional, List, Iterator, Tuple

# Assuming 'tracked_data_interface.py' is in the same directory or a reachable path
from .tracked_data_interface import TrackedDataInterface

class ConsolidatedTracksManager(TrackedDataInterface):
    """
    Manages the calculated center coordinate data for multiple tracked objects.

    This class encapsulates the final coordinate data within a dictionary, where
    keys are object names and values are DataFrames containing the data for that
    specific object. This provides fast, direct access to data for any given object.

    This class implements the TrackedDataInterface, allowing it to serve as a
    standardized data source for tracked items.

    Attributes:
        data (Dict[str, pd.DataFrame]): A dictionary mapping object names to
                                        DataFrames. Each DataFrame has columns
                                        ['frame_number', 'center_x', 'center_y', 'score'].
    """
    REQUIRED_COLUMNS = {'frame_number', 'object_name', 'center_x', 'center_y', 'score'}

    def __init__(self, coordinates_df: pd.DataFrame):
        """
        Initializes the manager with a DataFrame of coordinate data, which is
        then converted into a dictionary for internal storage.

        Args:
            coordinates_df (pd.DataFrame): The single, long-format DataFrame to manage.

        Raises:
            ValueError: If the DataFrame is missing required columns.
        """
        if not self.REQUIRED_COLUMNS.issubset(coordinates_df.columns):
            raise ValueError(
                f"DataFrame is missing one or more required columns. "
                f"Required: {self.REQUIRED_COLUMNS}"
            )
        
        # Group the initial DataFrame by 'object_name' and create the dictionary
        self.data: Dict[str, pd.DataFrame] = {
            name: group.sort_values(by='frame_number').reset_index(drop=True)
            for name, group in coordinates_df.groupby('object_name')
        }

    def get_items_for_frame(self, frame_index: int) -> Iterator[Tuple[str, pd.Series]]:
        """
        Yields the object name and its corresponding data row for a specific frame.
        
        This method fulfills the TrackedDataInterface requirement.
        """
        for object_name, df in self.data.items():
            # Filter the DataFrame for the specific frame_index
            frame_data = df[df['frame_number'] == frame_index]
            
            # If an entry exists for that frame, yield the object name and the data row
            if not frame_data.empty:
                # .iloc[0] is safe as there's only one entry per object per frame
                yield object_name, frame_data.iloc[0]

    def get_all_data(self) -> pd.DataFrame:
        """
        Reconstructs and returns a single DataFrame from the internal dictionary.
        
        The DataFrame is sorted by object name and frame number for consistency.
        """
        if not self.data:
            return pd.DataFrame() # Return an empty DataFrame if there's no data
            
        all_dfs = list(self.data.values())
        combined_df = pd.concat(all_dfs, ignore_index=True)
        return combined_df.sort_values(by=['object_name', 'frame_number']).reset_index(drop=True)

    def get_coordinates_for_object(self, object_name: str) -> Optional[pd.DataFrame]:
        """
        Retrieves the coordinate data for a single object via dictionary lookup.

        Args:
            object_name (str): The name of the object to retrieve data for.

        Returns:
            Optional[pd.DataFrame]: A DataFrame for the specified object,
                                    or None if the object is not found.
        """
        return self.data.get(object_name)

    @property
    def object_names(self) -> List[str]:
        """
        Returns a unique list of object names managed by this instance.
        
        This method fulfills the TrackedDataInterface requirement.
        """
        return list(self.data.keys())

    def __len__(self) -> int:
        """Returns the total number of coordinate entries (rows) across all objects."""
        if not self.data:
            return 0
        return sum(len(df) for df in self.data.values())

    def __repr__(self) -> str:
        """Provides a concise, developer-friendly representation of the object."""
        num_objects = len(self.object_names)
        if num_objects == 0:
            return "<ConsolidatedTracksManager(objects=0, total_entries=0, unique_frames=0)>"
            
        # To get unique_frames, we need to check across all objects
        all_frames = pd.concat([df['frame_number'] for df in self.data.values()])
        num_frames = all_frames.nunique()
        
        return (
            f"<ConsolidatedTracksManager(objects={num_objects}, "
            f"total_entries={len(self)}, unique_frames={num_frames})>"
        )
    
    def to_dict_by_object(self) -> Dict[str, pd.DataFrame]:
        """
        Returns the internal dictionary mapping object names to their
        respective coordinate DataFrames.
        """
        return self.data