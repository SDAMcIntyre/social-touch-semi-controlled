# src/pipeline_monitor.py
from pathlib import Path
from typing import List
from multiprocessing import Queue, Manager

try:
    from .pipeline_monitor_data_manager import DataManager
    from .pipeline_monitor_live_plotter import LivePlotter
except:
    from pipeline_monitor_data_manager import DataManager
    from pipeline_monitor_live_plotter import LivePlotter


class PipelineMonitor:
    """
    The main user-facing interface that mediates between data management and visualization.
    """
    def __init__(self, report_path: str, stages: List[str]):
        """
        Initializes the monitoring system.

        Args:
            report_path (str): Path to the CSV file where the report will be saved.
            stages (List[str]): The defined stages of the pipeline (report columns).
        """
        report_file = Path(report_path)
        self._data_manager = DataManager(report_file)
        
        # The queue is the communication channel to the plotter process
        self._data_queue = Queue()
        self._plotter = LivePlotter(self._data_queue)
        
        self._data_manager.initialize(stages)

    def update(self, dataset: str, stage: str, status: str, message: str = ""):
        """
        Logs a status update, saves it to disk, and pushes the new state to the dashboard.
        """
        print(f"[{dataset}] -> Stage '{stage}': {status}")
        
        # 1. Update the data file and get the new state
        updated_df = self._data_manager.update(dataset, stage, status, message)
        
        # 2. If the update was successful and the dashboard is running, push the new data
        if updated_df is not None and self._plotter.is_running():
            self._data_queue.put(updated_df)

    def show_dashboard(self):
        """Display the live, graphical dashboard in a new window."""
        # Prime the dashboard with the current state if the file already exists
        initial_df = self._data_manager.get_dataframe()
        if initial_df is not None:
             self._data_queue.put(initial_df)
        self._plotter.start()

    def close_dashboard(self):
        """Close the live dashboard window."""
        self._plotter.stop()




# main.py
# --- Configuration ---
PIPELINE_STAGES = [
    "Data Ingestion", 
    "Preprocessing", 
    "Feature Engineering", 
    "Model Training", 
    "Evaluation"
]
DATASETS = [f"Dataset_{i:02d}" for i in range(1, 8)]
REPORTS_DIR = Path("reports")
REPORT_FILE = REPORTS_DIR / "pipeline_status.csv"

# --- Worker Simulation ---

def worker_process(dataset_name: str, monitor: PipelineMonitor):
    """
    Simulates a pipeline processing a single dataset.
    
    Each worker function updates the central report via the monitor instance,
    mimicking a distributed data processing task.
    """
    import time
    import random

    print(f"⚙️  Worker started for {dataset_name}...")
    
    for stage in PIPELINE_STAGES:
        # 1. Mark the stage as RUNNING
        monitor.update(dataset_name, stage, "RUNNING")
        time.sleep(random.uniform(1, 3))  # Simulate work

        # 2. Randomly determine if the stage succeeded or failed
        if random.random() > 0.15:  # 85% chance of success
            monitor.update(dataset_name, stage, "SUCCESS", "Completed successfully.")
        else:
            monitor.update(dataset_name, stage, "FAILURE", "A critical error occurred.")
            print(f"❌ Worker for {dataset_name} failed at stage '{stage}'.")
            return  # Stop processing this dataset on failure

    print(f"✅ Worker finished for {dataset_name}.")


# --- Main Execution ---

if __name__ == "__main__":
    from multiprocessing import Process, freeze_support
    import shutil
    import time

    # `freeze_support()` is required for multiprocessing to work correctly 
    # when the script is frozen into an executable (e.g., with PyInstaller on Windows).
    freeze_support()
    
    # Clean up previous run's report for a fresh start
    if REPORTS_DIR.exists():
        shutil.rmtree(REPORTS_DIR)
    print(f"🧹 Cleaned up old reports.")

    # 1. Initialize the main monitor object.
    # This creates the report CSV file with the correct headers.
    monitor = PipelineMonitor(report_path=str(REPORT_FILE), stages=PIPELINE_STAGES)

    # 2. Launch the live dashboard in a separate process.
    monitor.show_dashboard()

    # 3. Start concurrent worker processes to simulate the pipeline.
    processes = []
    print("\n🚀 Starting pipeline workers...")
    for dataset in DATASETS:
        # Each process gets the same monitor instance, which is handled safely
        # by the multiprocessing library.
        process = Process(target=worker_process, args=(dataset, monitor))
        processes.append(process)
        process.start()
        time.sleep(0.5)  # Stagger the start times slightly for a better visual effect.

    # 4. Wait for all worker processes to complete their tasks.
    for p in processes:
        p.join()

    print("\n🏁 All pipeline workers have completed their tasks.")
    print("✨ Dashboard will remain open. Close the plot window to exit the script.")
    
    # The main script will wait here until the plot window is closed.
    # You could also programmatically close it after a delay:
    # time.sleep(10)
    # monitor.close_dashboard()