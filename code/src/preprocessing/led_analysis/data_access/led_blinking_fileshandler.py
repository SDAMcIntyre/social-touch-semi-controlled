import csv
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union, Optional, Literal
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LEDBlinkingFilesHandler:
    """
    Handles the serialization and deserialization of analysis results to/from files.
    
    This class can save time-series data to CSV and metadata to JSON. It also provides
    static methods to load this data back from files into different data structures.
    """
    def __init__(
        self, 
        result_dir: Optional[Union[str, Path]] = None, 
        file_basename: Optional[str] = None
    ):
        """
        Initializes the file handler.

        Args:
            result_dir (Optional[Union[str, Path]]): The default directory where results will be saved.
            file_basename (Optional[str]): The default base name for the output files.
        """
        if result_dir:
            self.result_dir = Path(result_dir)
            self.result_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.result_dir = None
        
        self.file_basename = file_basename

    def _get_output_path(self, output_path: Optional[Union[str, Path]], extension: str) -> Path:
        """Helper method to determine the final output path."""
        if output_path:
            return Path(output_path)
        
        if self.result_dir and self.file_basename:
            return self.result_dir / f"{self.file_basename}{extension}"
        
        raise ValueError(
            "An output path must be provided to the save method if the handler "
            "was instantiated without a 'result_dir' and 'file_basename'."
        )

    def save(self, results: Dict[str, Any], overwrite: bool = True):
        """
        Saves the provided results dictionary to CSV and JSON files using the 
        default directory and basename configured during initialization.

        This method will fail if the handler was not initialized with a default path.

        Args:
            results (Dict[str, Any]): A dictionary containing the analysis results.
                                       Expected keys include 'time_series_data' and 'metadata'.
            overwrite (bool): If False, will not overwrite existing files. Defaults to True.
        """
        if 'time_series_data' not in results or 'metadata' not in results:
            logging.error("Results dictionary is missing required 'time_series_data' or 'metadata' keys.")
            raise ValueError("Results dictionary is missing required 'time_series_data' or 'metadata' keys.")
            
        # This method relies on the instance having a default path configured.
        if not self.result_dir or not self.file_basename:
             raise ValueError(
                 "The 'save' method requires the handler to be instantiated with "
                 "'result_dir' and 'file_basename'. Alternatively, call the specific save methods "
                 "like 'save_timeseries_to_csv' with an explicit 'output_path'."
             )
            
        self.save_timeseries_to_csv(results['time_series_data'], overwrite=overwrite)
        self.save_metadata_to_json(results['metadata'], overwrite=overwrite)

    def save_timeseries_to_csv(
        self, 
        time_series_data: Dict[str, np.ndarray], 
        output_path: Optional[Union[str, Path]] = None,
        overwrite: bool = True
    ):
        """
        Saves the core time-series data to a CSV file.

        Args:
            time_series_data (Dict[str, np.ndarray]): Dictionary with time-series data.
            output_path (Optional[Union[str, Path]]): Full path for the output CSV file. 
                                                     If None, constructs path from instance attributes.
            overwrite (bool): If False, will not overwrite an existing file. Defaults to True.
        """
        try:
            csv_path = self._get_output_path(output_path, extension='.csv')
        except ValueError as e:
            logging.error(e)
            raise
        
        # Check for overwrite condition
        if not overwrite and csv_path.exists():
            logging.warning(f"File '{csv_path}' already exists and overwrite is False. Skipping save.")
            return

        # Ensure the target directory exists
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        time = time_series_data.get('time', np.array([]))
        green_levels = time_series_data.get('green_levels', np.array([]))
        led_on = time_series_data.get('led_on', np.array([]))

        # Prepare data for writing, handling potential NaNs
        led_on_str = [str(int(v)) if not np.isnan(v) else '' for v in led_on]
        green_levels_str = [f'{v:.4f}' if not np.isnan(v) else '' for v in green_levels]

        try:
            with open(csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["time (second)", "green_level_normalized", "led_on"])
                writer.writerows(zip(time, green_levels_str, led_on_str))
            logging.info(f"Time-series results saved to {csv_path}")
        except IOError as e:
            logging.error(f"Failed to write CSV file to {csv_path}: {e}")
            raise

    def save_metadata_to_json(
        self, 
        metadata: Dict[str, Any],
        output_path: Optional[Union[str, Path]] = None,
        overwrite: bool = True
    ):
        """
        Saves the analysis metadata to a JSON file.

        Args:
            metadata (Dict[str, Any]): Dictionary with metadata.
            output_path (Optional[Union[str, Path]]): Full path for the output JSON file. 
                                                     If None, constructs path from instance attributes.
            overwrite (bool): If False, will not overwrite an existing file. Defaults to True.
        """
        try:
            metadata_path = self._get_output_path(output_path, extension='_metadata.json')
        except ValueError as e:
            logging.error(e)
            raise

        # Check for overwrite condition
        if not overwrite and metadata_path.exists():
            logging.warning(f"File '{metadata_path}' already exists and overwrite is False. Skipping save.")
            return

        # Ensure the target directory exists
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            logging.info(f"Metadata saved to {metadata_path}")
        except IOError as e:
            logging.error(f"Failed to write JSON file to {metadata_path}: {e}")
            raise

    # --- MODIFIED/ADDED METHODS ---

    @staticmethod
    def load_timeseries_from_csv(
        file_path: Union[str, Path],
        output_format: Literal['dict', 'dataframe'] = 'dict'
    ) -> Union[Dict[str, np.ndarray], pd.DataFrame]:
        """
        Loads time-series data from a CSV file.

        This method can return the data as a dictionary of NumPy arrays or as a 
        pandas DataFrame. It correctly handles empty fields, converting them to np.nan.

        Args:
            file_path (Union[str, Path]): The path to the input CSV file.
            output_format (Literal['dict', 'dataframe']): The desired output format. 
                'dict' (default): returns a dictionary of NumPy arrays.
                'dataframe': returns a pandas DataFrame.

        Returns:
            Union[Dict[str, np.ndarray], pd.DataFrame]: The time-series data in the specified format.
        """
        load_path = Path(file_path)
        if not load_path.exists():
            logging.error(f"File not found: {load_path}")
            raise FileNotFoundError(f"File not found: {load_path}")

        if output_format == 'dict':
            time, green_levels, led_on = [], [], []
            try:
                with open(load_path, mode='r', newline='') as file:
                    reader = csv.reader(file)
                    next(reader)  # Skip header row
                    
                    for i, row in enumerate(reader):
                        # Expecting 3 columns per row
                        if len(row) != 3:
                            logging.warning(f"Skipping malformed row {i+2} in {load_path}: {row}")
                            continue
                        
                        time.append(float(row[0]))
                        green_levels.append(float(row[1]) if row[1] else np.nan)
                        led_on.append(float(row[2]) if row[2] else np.nan)

                logging.info(f"Time-series data successfully loaded as a dict from {load_path}")
                # NumPy automatically promotes arrays with np.nan to float dtype
                return {
                    'time': np.array(time),
                    'green_levels': np.array(green_levels),
                    'led_on': np.array(led_on)
                }
            except (IOError, ValueError, IndexError) as e:
                logging.error(f"Failed to read or parse CSV into dict from {load_path}: {e}")
                raise
        
        elif output_format == 'dataframe':
            try:
                df = pd.read_csv(load_path)
                # Rename columns for consistency and ease of use
                df.rename(columns={
                    "time (second)": "time",
                    "green_level_normalized": "green_levels",
                }, inplace=True)
                logging.info(f"Time-series data successfully loaded as a DataFrame from {load_path}")
                return df
            except Exception as e:
                logging.error(f"Failed to read CSV into DataFrame from {load_path}: {e}")
                raise
        
        else:
            raise ValueError(f"Invalid output_format: '{output_format}'. Must be 'dict' or 'dataframe'.")

    @staticmethod
    def load_metadata_from_json(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Loads metadata from a JSON file.

        Args:
            file_path (Union[str, Path]): The path to the input JSON file.

        Returns:
            Dict[str, Any]: A dictionary containing the metadata.
        """
        load_path = Path(file_path)
        if not load_path.exists():
            logging.error(f"File not found: {load_path}")
            raise FileNotFoundError(f"File not found: {load_path}")
        
        try:
            with open(load_path, 'r') as f:
                metadata = json.load(f)
            logging.info(f"Metadata successfully loaded from {load_path}")
            return metadata
        except (IOError, json.JSONDecodeError) as e:
            logging.error(f"Failed to read or parse JSON file {load_path}: {e}")
            raise