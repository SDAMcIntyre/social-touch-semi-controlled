# src/pipeline_monitor_live_plotter.py

import multiprocessing

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Optional
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
        
        backend = plt.get_backend()
        if backend.lower() != "agg":
            plt.pause(plt_pause_interval)
        else:
            import time
            time.sleep(plt_pause_interval)

    plt.close(fig)

class LivePlotter:
    def __init__(self, data_queue: Queue):
        self._queue = data_queue
        self._plot_process: Optional[multiprocessing.Process] = None
        self._stopped = False
        self._reported_dead = False

    def start(self):
        if self._plot_process and self._plot_process.is_alive():
            return
        self._stopped = False
        self._reported_dead = False
        self._plot_process = multiprocessing.Process(target=_plot_process_target, args=(self._queue,))
        try:
            self._plot_process.start()
        except (PermissionError, OSError) as e:
            print(f"Live dashboard unavailable (process spawn failed: {e}). "
                  "Pipeline will continue — status is saved to the Excel report.")
            self._plot_process = None

    def report_if_dead(self) -> None:
        """Announce a dashboard child that died *after* a successful ``start()``.

        ``start()``'s own ``except`` only covers failures raised in this process
        while spawning.  The child can instead die during its own bootstrap, well
        after ``start()`` has returned: on Windows, unpickling the queue's pipe
        handle raises ``PermissionError: [WinError 5] Access is denied`` from
        ``DuplicateHandle`` whenever a debugger has patched ``multiprocessing``
        (debugpy does this by default via its ``subProcess`` option).  That path
        leaves a raw child traceback on the console, no dashboard, and nothing
        here any the wiser — including the message above, which names exactly
        this situation but cannot fire in it.

        Reported once, then the plotter stands itself down so the pipeline runs
        on without it.  Deliberately not raising: the dashboard is an optional
        convenience and the authoritative status is the Excel report.  A
        terminate() from :meth:`stop` is not a death and is not reported.
        """
        if self._plot_process is None or self._stopped or self._reported_dead:
            return
        if self._plot_process.is_alive():
            return
        exitcode = self._plot_process.exitcode
        self._reported_dead = True
        self._plot_process = None
        print(f"Live dashboard stopped unexpectedly (child exit code {exitcode}). "
              "Pipeline will continue — status is saved to the Excel report.")

    def stop(self, block: bool = False):
        self._stopped = True
        if not self.is_running():
            return
        self._plot_process.terminate()
        self._plot_process.join(timeout=3 if block else 1)

    def is_running(self) -> bool:
        return self._plot_process is not None and self._plot_process.is_alive()