# src/live_plotter.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Optional 

import multiprocessing
from multiprocessing import Queue
from queue import Empty as QueueEmpty # To handle timeout exceptions

# Define a clear color scheme for different statuses
STATUS_MAPPING = {'SUCCESS': 2, 'RUNNING': 1, 'FAILURE': 0, 'PENDING': -1}
cmap = ListedColormap(['#d62728', '#ffbf00', '#2ca02c', '#c7c7c7'])

def _plot_process_target(queue: Queue):
    """Target function run in a separate process, listening for data on a queue."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Initial message
    ax.text(0.5, 0.5, "Waiting for data...", ha='center', va='center', fontsize=16)
    plt.title("Live Pipeline Status Dashboard")
    
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

            # This replaces both of your original .applymap() calls
            df_numeric = df_plot.map(
                lambda x: STATUS_MAPPING.get(str(x).split(':')[0], -1) if pd.notna(x) else -1
            )

            # You'll still need a way to get the string annotations
            df_status_annotations = df_plot.map(
                lambda x: str(x).split(':')[0] if pd.notna(x) else ''
            )

            sns.heatmap(
                df_numeric, ax=ax, annot=df_status_annotations, fmt='s', cmap=cmap,
                linewidths=.5, linecolor='black', cbar=False, vmin=-1, vmax=3
            )

            # --- Formatting ---
            legend_labels = {k: cmap(v / (len(STATUS_MAPPING)-0.5)) for k, v in STATUS_MAPPING.items()}
            patches = [plt.Rectangle((0,0),1,1, color=color) for color in legend_labels.values()]
            ax.legend(patches, legend_labels.keys(), bbox_to_anchor=(1.02, 1), loc='upper left')
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
        a=1

    def stop(self):
        if self._plot_process and self._plot_process.is_alive():
            self._queue.put(None)  # Send sentinel value to terminate
            self._plot_process.join(timeout=2)
            if self._plot_process.is_alive():
                 self._plot_process.terminate() # Force terminate if it doesn't close gracefully
            print("🛑 Dashboard has been closed.")

    def is_running(self) -> bool:
        return self._plot_process is not None and self._plot_process.is_alive()