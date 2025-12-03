# src/live_plotter.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Optional

import multiprocessing
from multiprocessing import Queue
from queue import Empty as QueueEmpty # To handle timeout exceptions

# Define a clear color scheme for different statuses. The order of integer values is:
# PENDING (-1) < FAILURE (0) < RUNNING (1) < SUCCESS (2).
STATUS_MAPPING = {'SUCCESS': 2, 'RUNNING': 1, 'FAILURE': 0, 'PENDING': -1}

# The colormap list is ordered to match the integer values above.
# PENDING: grey, FAILURE: red, RUNNING: orange, SUCCESS: green.
cmap = ListedColormap(['#c7c7c7', '#d62728', '#ffbf00', '#2ca02c'])

def _plot_process_target(queue: Queue):
    """Target function run in a separate process, listening for data on a queue."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Initial message
    ax.text(0.5, 0.5, "Waiting for data...", ha='center', va='center', fontsize=16)
    plt.title("Live Pipeline Status Dashboard")

    # Define heatmap normalization range. This ensures that the integer status values
    # map to the correct discrete colors in our custom colormap.
    vmin, vmax = -1, 3
    
    is_running = True
    while is_running:
        try:
            # Wait for a new DataFrame to arrive. Timeout allows the loop to check
            # if the window was manually closed.
            df = queue.get(timeout=0.2)
            
            if df is None:  # Sentinel value to terminate the process
                is_running = False
                continue

            # If the dataframe is empty, skip the plotting logic for this loop iteration.
            if df.empty:
                continue

            # --- Plotting Logic ---
            ax.clear()
            df_plot = df.set_index('dataset')

            # Map status strings to their corresponding integer values for plotting.
            df_numeric = df_plot.map(
                lambda x: STATUS_MAPPING.get(str(x).split(':')[0], -1) if pd.notna(x) else -1
            )

            # Extract the status string for use as cell annotations.
            df_status_annotations = df_plot.map(
                lambda x: str(x).split(':')[0] if pd.notna(x) else ''
            )

            sns.heatmap(
                df_numeric, ax=ax, annot=df_status_annotations, fmt='s', cmap=cmap,
                linewidths=.5, linecolor='black', cbar=False, vmin=vmin, vmax=vmax
            )

            # --- Formatting ---
            # Use the same normalization as the heatmap to ensure the legend colors are correct.
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            
            # Create legend items and sort them logically (by status value) for a clean look.
            status_color_map = {status: cmap(norm(value)) for status, value in STATUS_MAPPING.items()}
            sorted_statuses = sorted(status_color_map.keys(), key=lambda status: STATUS_MAPPING[status])
            
            patches = [plt.Rectangle((0,0),1,1, color=status_color_map[status]) for status in sorted_statuses]
            ax.legend(patches, sorted_statuses, bbox_to_anchor=(1.02, 1), loc='upper left')
            
            ax.set_title("Live Pipeline Status Dashboard", fontsize=16)
            ax.set_ylabel("Dataset", fontsize=12); ax.set_xlabel("Pipeline Stage", fontsize=12)
            plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
            fig.tight_layout(rect=[0, 0, 0.9, 1])

        except QueueEmpty:
            # No new data, just continue. This allows the process to stay alive.
            pass
        
        # Check if the plot window has been closed by the user
        if not plt.fignum_exists(fig.number):
            is_running = False
        
        # Process GUI events to keep the window responsive
        plt.pause(0.01)

    plt.close(fig)

class LivePlotter:
    """Manages the non-blocking, real-time plotting process."""
    def __init__(self, data_queue: Queue):
        self._queue = data_queue
        self._plot_process: Optional[multiprocessing.Process] = None

    def start(self):
        if self._plot_process and self._plot_process.is_alive():
            print("📈 Plotter is already running.")
            return

        print("🚀 Launching dashboard in a separate window...")
        self._plot_process = multiprocessing.Process(target=_plot_process_target, args=(self._queue,))
        self._plot_process.start()


    def stop(self, block: bool = False):
        """
        Stops the plotting process.

        Args:
            block (bool, optional): If True, the method will send the final plot state
                                    and wait until the user manually closes the plot window.
                                    If False (default), it closes the window immediately.
        """
        if not self.is_running():
            return

        if block:
            print("✋ Dashboard is waiting for you to close it manually.")
            self._plot_process.join() # Wait indefinitely until the process terminates
        else:
            self._queue.put(None)  # Send sentinel value for immediate termination
            self._plot_process.join(timeout=2)
            if self._plot_process.is_alive():
                self._plot_process.terminate() # Force terminate if it doesn't close gracefully

    def is_running(self) -> bool:
        return self._plot_process is not None and self._plot_process.is_alive()