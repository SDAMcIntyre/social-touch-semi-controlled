# src/pipeline_monitor.py (Refactored for explicit queue passing and plotting control)

from pathlib import Path
from typing import List, Optional
from multiprocessing import Manager
from multiprocessing.queues import Queue
import pandas as pd

try:
    from .pipeline_monitor_data_manager import DataManager
    from .pipeline_monitor_live_plotter import LivePlotter
except ImportError:
    from pipeline_monitor_data_manager import DataManager
    from pipeline_monitor_live_plotter import LivePlotter

class PipelineMonitor:
    """
    A unified class to monitor a pipeline.
    - If initialized without a 'data_queue', it becomes the main controller,
      creating and managing shared resources (Manager, Queue, Plotter).
    - If initialized with an existing 'data_queue', it acts as a lightweight
      client for a worker process, sending updates to that queue.
    """

    def __init__(self,
                 report_path: str,
                 stages: List[str],
                 data_queue: Optional[Queue] = None,
                 live_plotting: bool = True):
        """
        Initializes either the main monitor or a worker client.

        Args:
            report_path (str): Path to the Excel report file.
            stages (List[str]): List of pipeline stage names.
            data_queue (Optional[Queue]): If provided, configures this instance
              as a client. If None, configures as the main controller.
            live_plotting (bool): If True, enables the live plotting dashboard.
              Only effective for the main controller instance. Defaults to True.
        """
        self._data_manager = DataManager(Path(report_path))
        self._is_client = (data_queue is not None)

        if self._is_client:
            # --- Client Configuration (for worker processes) ---
            self._shared_queue = data_queue
            self._manager = None
            self._plotter = None
            print("📦 Monitor client configured for worker.")
        else:
            # --- Main Controller Configuration ---
            print("✨ Initializing main monitor and shared resources...")
            self._manager = Manager()
            self._shared_queue = self._manager.Queue()
            
            if live_plotting:
                print("📊 Live dashboard is ENABLED.")
                self._plotter = LivePlotter(self._shared_queue)
            else:
                print("📉 Live dashboard is DISABLED.")
                self._plotter = None
                
            self._data_manager.initialize(stages)

    @property
    def queue(self) -> Queue:
        """Returns the shared queue. Only available on the main instance."""
        if self._is_client or self._shared_queue is None:
            raise AttributeError("Worker clients do not own the queue. Access it from the main instance.")
        return self._shared_queue

    def update(self, dataset: str, stage: str, status: str, message: str = ""):
        """
        Logs a status update, saves it, and pushes the new state to the queue.
        This method works for both the main instance and worker clients.
        """
        print(f"[{dataset}] -> Stage '{stage}': {status}")
        updated_df = self._data_manager.update(dataset, stage, status, message)
        
        if updated_df is not None and self._shared_queue:
            try:
                self._shared_queue.put(updated_df)
            except Exception as e:
                print(f"🚨 Could not put data into queue: {e}")

    def show_dashboard(self):
        """Displays the live dashboard. Only callable from the main instance."""
        if self._is_client:
            raise RuntimeError("Dashboard can only be shown from the main monitor instance.")
        
        if not self._plotter:
            print("📉 Live dashboard is disabled. Cannot show.")
            return
        
        initial_df = self._data_manager.get_dataframe()
        if initial_df is not None and self._shared_queue is not None:
            self._shared_queue.put(initial_df)
        self._plotter.start()

    def close_dashboard(self, block: bool = False):
        """
        Closes the live dashboard. Only callable from the main instance.

        Args:
            block (bool): If True, the call will block until the plot window
                          is manually closed by the user. Defaults to False.
        """
        if self._is_client or not self._plotter:
            return  # Silently ignore for clients or when plotting is disabled

        print("👋 Closing dashboard...")
        # Assumes the LivePlotter's stop method is updated to accept a 'block' argument.
        self._plotter.stop(block=block)