# src/pipeline_monitor_live_plotter.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Optional
import multiprocessing
from multiprocessing import Queue
from queue import Empty as QueueEmpty

STATUS_MAPPING = {'SUCCESS': 2, 'RUNNING': 1, 'FAILURE': 0, 'PENDING': -1}
cmap = ListedColormap(['#c7c7c7', '#d62728', '#ffbf00', '#2ca02c'])

def _plot_process_target(queue: Queue):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.text(0.5, 0.5, "Waiting for data...", ha='center', va='center', fontsize=16)
    plt.title("Live Pipeline Status Dashboard")
    
    # Use 0.1 for faster UI refresh rate
    plt_pause_interval = 0.1 
    vmin, vmax = -1, 3
    
    is_running = True
    last_df = None

    while is_running:
        try:
            # Drain queue to get the LATEST dataframe, skipping intermediate states
            # This prevents the plotter from falling behind if updates are too fast.
            df = None
            while not queue.empty():
                df = queue.get_nowait()
            
            if df is None and last_df is None:
                # If we drained nothing and have no history, wait a bit
                try:
                    df = queue.get(timeout=plt_pause_interval)
                except QueueEmpty:
                    pass
            
            if df is None and queue.empty():
                # Still nothing? Use the last known dataframe to keep window responsive
                df = last_df

            if df is None: # Sentinel for stop
                 # We only check for explicit None sentinel if we implemented it that way, 
                 # currently we rely on process termination or implicit empty checks.
                 pass
            elif isinstance(df, pd.DataFrame):
                 last_df = df

            # Check for close
            if not plt.fignum_exists(fig.number):
                is_running = False
                continue

            # Only redraw if we have data
            if last_df is not None and not last_df.empty:
                ax.clear()
                df_plot = last_df.set_index('dataset')

                df_numeric = df_plot.map(
                    lambda x: STATUS_MAPPING.get(str(x).split(':')[0], -1) if pd.notna(x) else -1
                )
                df_status_annotations = df_plot.map(
                    lambda x: str(x).split(':')[0] if pd.notna(x) else ''
                )

                sns.heatmap(
                    df_numeric, ax=ax, annot=df_status_annotations, fmt='s', cmap=cmap,
                    linewidths=.5, linecolor='black', cbar=False, vmin=vmin, vmax=vmax
                )

                norm = plt.Normalize(vmin=vmin, vmax=vmax)
                status_color_map = {status: cmap(norm(value)) for status, value in STATUS_MAPPING.items()}
                sorted_statuses = sorted(status_color_map.keys(), key=lambda status: STATUS_MAPPING[status])
                patches = [plt.Rectangle((0,0),1,1, color=status_color_map[status]) for status in sorted_statuses]
                ax.legend(patches, sorted_statuses, bbox_to_anchor=(1.02, 1), loc='upper left')
                
                ax.set_title("Live Pipeline Status Dashboard", fontsize=16)
                plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
                fig.tight_layout(rect=[0, 0, 0.9, 1])

        except Exception:
            pass
        
        plt.pause(plt_pause_interval)

    plt.close(fig)

class LivePlotter:
    def __init__(self, data_queue: Queue):
        self._queue = data_queue
        self._plot_process: Optional[multiprocessing.Process] = None

    def start(self):
        if self._plot_process and self._plot_process.is_alive():
            return
        self._plot_process = multiprocessing.Process(target=_plot_process_target, args=(self._queue,))
        self._plot_process.start()

    def stop(self, block: bool = False):
        if not self.is_running():
            return
        if block:
            self._plot_process.join()
        else:
            self._plot_process.terminate()
            self._plot_process.join(timeout=1)

    def is_running(self) -> bool:
        return self._plot_process is not None and self._plot_process.is_alive()