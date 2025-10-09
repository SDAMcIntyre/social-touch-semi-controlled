# src/data_manager.py

import pandas as pd
from pathlib import Path
from filelock import FileLock
from typing import List, Optional

class DataManager:
    """
    Handles concurrency-safe reading and writing of the pipeline status to a CSV file.
    """
    def __init__(self, report_file_path: Path):
        self.report_file = report_file_path
        self.lock_file = Path(f"{self.report_file}.lock")
        self.lock = FileLock(self.lock_file)
        self.report_file.parent.mkdir(parents=True, exist_ok=True)

    def initialize(self, columns: List[str]):
        header = ['dataset'] + columns
        with self.lock:
            if not self.report_file.exists():
                df = pd.DataFrame(columns=header)
                df.to_csv(self.report_file, index=False)
                print(f"📊 Report initialized at: {self.report_file}")

    def update(self, dataset: str, event: str, status: str, message: str = "") -> Optional[pd.DataFrame]:
        """
        Updates the status and returns the entire updated DataFrame upon success.

        Returns:
            Optional[pd.DataFrame]: The newly updated DataFrame, or None on failure.
        """
        value = f"{status}: {message}" if message else status
        
        try:
            with self.lock:
                df = pd.read_csv(self.report_file)

                if event not in df.columns:
                    print(f"⚠️ Warning: Event '{event}' not in initialized columns. Ignoring.")
                    return None

                if dataset in df['dataset'].values:
                    df.loc[df['dataset'] == dataset, event] = value
                else:
                    new_row = {'dataset': dataset, event: value}
                    new_row_df = pd.DataFrame([new_row])
                    df = pd.concat([df, new_row_df], ignore_index=True)

                df.to_csv(self.report_file, index=False)
                # Return the updated dataframe on successful write
                return df
        except Exception as e:
            print(f"🚨 CRITICAL: Failed to update report file {self.report_file}. Error: {e}")
            return None
    
    def get_dataframe(self) -> Optional[pd.DataFrame]:
        with self.lock:
            if self.report_file.exists():
                return pd.read_csv(self.report_file)
        return None