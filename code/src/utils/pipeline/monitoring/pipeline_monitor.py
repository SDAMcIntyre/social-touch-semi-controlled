# src/pipeline_monitor.py

import time
import threading
from pathlib import Path
from typing import List, Optional, Tuple
from multiprocessing import Manager, Process
from multiprocessing.queues import Queue as MPQueue
import pandas as pd

try:
    from .pipeline_monitor_data_manager import DataManager
    from .pipeline_monitor_live_plotter import LivePlotter
except ImportError:
    from pipeline_monitor_data_manager import DataManager
    from pipeline_monitor_live_plotter import LivePlotter

class PipelineMonitor:
    """
    High-Performance Asynchronous Monitor.
    
    Architecture:
    [Workers] --(msg)--> [InputQueue] --(Coordinator Thread)--> [DataManager (Memory)]
                                               |
                                               +--(periodic)--> [Excel File]
                                               +--(continuous)--> [PlotQueue] -> [LivePlotter]
    """

    def __init__(self,
                 report_path: str,
                 stages: List[str],
                 data_queue: Optional[MPQueue] = None,
                 live_plotting: bool = True,
                 save_interval: float = 2.0):
        """
        Args:
            save_interval: Seconds to wait between disk writes (throttling).
        """
        self._is_client = (data_queue is not None)
        self._input_queue = data_queue
        
        # --- Client Mode ---
        if self._is_client:
            return

        # --- Server/Controller Mode ---
        self._manager = Manager()
        self._input_queue = self._manager.Queue() # Workers write here
        self._plot_queue = self._manager.Queue()  # Plotter reads here
        
        self._data_manager = DataManager(Path(report_path))
        self._data_manager.initialize(stages)
        
        # Configuration
        self._save_interval = save_interval
        self._running = True
        
        # Start the Coordinator Thread (Handles logic & persistence)
        self._coordinator_thread = threading.Thread(target=self._coordinator_loop, daemon=True)
        self._coordinator_thread.start()

        # Start Live Plotter Process
        if live_plotting:
            print("投 Live dashboard is ENABLED.")
            self._plotter = LivePlotter(self._plot_queue)
            # Push initial state
            self._plot_queue.put(self._data_manager.get_dataframe())
        else:
            self._plotter = None

    @property
    def queue(self) -> MPQueue:
        """Returns the input queue for workers."""
        if self._input_queue is None:
             raise RuntimeError("Queue not initialized.")
        return self._input_queue

    def _coordinator_loop(self):
        """
        Runs in the main process (background thread).
        Consumes messages, updates memory, and throttles disk writes.
        """
        last_save_time = time.time()
        needs_save = False

        while self._running:
            try:
                # 1. Process all pending messages (Batch processing)
                # We drain the queue to avoid lag if many updates come at once.
                msg_count = 0
                while not self._input_queue.empty():
                    try:
                        # Expected format: (dataset, stage, status, message)
                        data = self._input_queue.get_nowait()
                        
                        if data == "STOP":
                            self._running = False
                            break
                        
                        dataset, stage, status, msg = data
                        self._data_manager.update_memory(dataset, stage, status, msg)
                        needs_save = True
                        msg_count += 1
                    except Exception:
                        break # Queue empty or error
                
                if not self._running:
                    break

                # 2. Update Plotter (if state changed)
                if msg_count > 0:
                    current_df = self._data_manager.get_dataframe()
                    # Push copy to plotter (fire and forget)
                    if self._plot_queue:
                         self._plot_queue.put(current_df)

                # 3. Throttled Disk Write
                now = time.time()
                if needs_save and (now - last_save_time > self._save_interval):
                    self._data_manager.save_report()
                    last_save_time = now
                    needs_save = False
                
                # Sleep briefly to prevent CPU spinning
                time.sleep(0.05)

            except Exception as e:
                print(f"🚨 Coordinator Error: {e}")
                time.sleep(1)

        # Final Save on exit
        print("💾 Saving final report...")
        self._data_manager.save_report()

    def update(self, dataset: str, stage: str, status: str, message: str = ""):
        """
        Fast, non-blocking update.
        """
        # Create message tuple
        msg = (dataset, stage, status, message)
        
        try:
            self._input_queue.put(msg)
            # Print log for feedback (optional, can be removed for extreme speed)
            print(f"[{dataset}] -> {stage}: {status}") 
        except Exception as e:
            print(f"圷 Failed to send update: {e}")

    def show_dashboard(self):
        if self._is_client or not self._plotter:
            return
        self._plotter.start()

    def close_dashboard(self, block: bool = False):
        if self._is_client:
            return
            
        # Stop coordinator
        if hasattr(self, '_input_queue'):
            self._input_queue.put("STOP")
        
        if hasattr(self, '_coordinator_thread'):
            self._coordinator_thread.join(timeout=3.0)

        if self._plotter:
            self._plotter.stop(block=block)