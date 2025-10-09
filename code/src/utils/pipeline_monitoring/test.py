# test.py (Modified to test all PipelineMonitor modes)

import time
import random
from multiprocessing import Process, freeze_support, Queue
from pathlib import Path
import shutil

try:
    from .pipeline_monitor import PipelineMonitor
except ImportError:
    from pipeline_monitor import PipelineMonitor

# --- Configuration (Unchanged) ---
PIPELINE_STAGES = [
    "Data Ingestion", 
    "Preprocessing", 
    "Feature Engineering", 
    "Model Training", 
    "Evaluation"
]
DATASETS = [f"Dataset_{i:02d}" for i in range(1, 8)]
REPORTS_DIR = Path("reports")
REPORT_FILE = REPORTS_DIR / "pipeline_status.xlsx"

# --- Worker Simulation (Unchanged) ---

def worker_process(dataset_name: str, report_path: str, stages: list, data_queue: Queue):
    """
    Simulates a pipeline for a single dataset. It now receives the shared
    queue and uses it to initialize its own 'client' monitor instance.
    """
    # This call creates a lightweight client instance because we pass the queue.
    monitor = PipelineMonitor(report_path=report_path, stages=stages, data_queue=data_queue, live_plotting=False)
    
    print(f"⚙️  Worker started for {dataset_name}...")
    
    for stage in stages:
        monitor.update(dataset_name, stage, "RUNNING")
        time.sleep(random.uniform(1, 3))

        if random.random() > 0.15:
            monitor.update(dataset_name, stage, "SUCCESS", "Completed successfully.")
        else:
            monitor.update(dataset_name, stage, "FAILURE", "A critical error occurred.")
            print(f"❌ Worker for {dataset_name} failed at stage '{stage}'.")
            return

    print(f"✅ Worker finished for {dataset_name}.")

# --- Main Execution Logic (Refactored) ---

def run_pipeline_simulation(live_plotting: bool, blocking_close: bool):
    """
    Runs a full simulation of the pipeline monitoring with specified settings.
    
    Args:
        live_plotting (bool): Whether to enable the live GUI dashboard.
        blocking_close (bool): If plotting is enabled, whether to wait for the user
                               to close the plot window.
    """
    # 0. Setup
    if REPORTS_DIR.exists():
        shutil.rmtree(REPORTS_DIR)
    print("🧹 Cleaned up old reports.")
    print("-" * 60)
    print(f"▶️  Running Test: Live Plotting = {live_plotting}, Blocking Close = {blocking_close}")
    print("-" * 60)

    # 1. Initialize the main monitor with the desired plotting mode.
    main_monitor = PipelineMonitor(
        report_path=str(REPORT_FILE), 
        stages=PIPELINE_STAGES,
        live_plotting=live_plotting
    )

    # 2. Attempt to launch the live dashboard.
    #    If live_plotting is False, this will just print a message.
    main_monitor.show_dashboard()

    # 3. Start concurrent worker processes.
    processes = []
    print("\n🚀 Starting pipeline workers...")
    for dataset in DATASETS:
        # We explicitly pass the shared queue to each worker process.
        process = Process(
            target=worker_process, 
            args=(dataset, str(REPORT_FILE), PIPELINE_STAGES, main_monitor.queue)
        )
        processes.append(process)
        process.start()
        time.sleep(0.5)

    # 4. Wait for all worker processes to complete.
    for p in processes:
        p.join()

    print("\n🏁 All pipeline workers have completed their tasks.")
    
    # 5. Close the dashboard with the specified blocking mode.
    if live_plotting:
        if blocking_close:
            print("✨ Dashboard will remain open. Close the plot window to exit.")
        else:
            print("✨ Dashboard will close automatically in 3 seconds...")
            time.sleep(3) # Give user a moment to see the final state
    
    main_monitor.close_dashboard(block=blocking_close)
    print("👋 Simulation finished.")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    freeze_support()

    while True:
        print("Select a PipelineMonitor test scenario to run:")
        print("  1. Live Plotting Enabled, Blocking Close (Default interactive mode)")
        print("  2. Live Plotting Enabled, Non-Blocking Close (Automated mode)")
        print("  3. Live Plotting Disabled (Headless/CI mode)")
        print("  q. Quit")
        
        choice = input("Enter your choice (1/2/3/q): ")

        if choice == '1':
            run_pipeline_simulation(live_plotting=True, blocking_close=True)
        elif choice == '2':
            run_pipeline_simulation(live_plotting=True, blocking_close=False)
        elif choice == '3':
            # blocking_close is irrelevant when plotting is off, but we pass False.
            run_pipeline_simulation(live_plotting=False, blocking_close=False)
        elif choice.lower() == 'q':
            print("Exiting.")
            break
        else:
            print("\n❌ Invalid choice. Please try again.\n")