# main_corrected.py

import time
import random
from multiprocessing import Process, freeze_support
from pathlib import Path
import os
import shutil

try:
    from .pipeline_monitor import PipelineMonitor
except:
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
REPORT_FILE = REPORTS_DIR / "pipeline_status.csv"

# --- Worker Simulation (MODIFIED) ---

def worker_process(dataset_name: str, report_path: str, stages: list):
    """
    Simulates a pipeline processing a single dataset.
    
    💡 **CHANGE:** This worker now creates its OWN PipelineMonitor instance
    instead of receiving one from the parent process.
    """
    # Each worker creates its own monitor. This is safe and correct.
    monitor = PipelineMonitor(report_path=report_path, stages=stages)
    
    print(f"⚙️  Worker started for {dataset_name}...")
    
    for stage in stages:
        # 1. Mark the stage as RUNNING
        monitor.update(dataset_name, stage, "RUNNING")
        time.sleep(random.uniform(1, 3))  # Simulate work

        # 2. Randomly determine if the stage succeeded or failed
        if random.random() > 0.15:  # 85% chance of success
            monitor.update(dataset_name, stage, "SUCCESS", "Completed successfully.")
        else:
            monitor.update(dataset_name, stage, "FAILURE", "A critical error occurred.")
            print(f"❌ Worker for {dataset_name} failed at stage '{stage}'.")
            return

    print(f"✅ Worker finished for {dataset_name}.")


# --- Main Execution (MODIFIED) ---

if __name__ == "__main__":
    freeze_support()
    
    if REPORTS_DIR.exists():
        shutil.rmtree(REPORTS_DIR)
    print(f"🧹 Cleaned up old reports.")

    # 1. The MAIN process creates a monitor for managing the DASHBOARD.
    main_monitor = PipelineMonitor(report_path=str(REPORT_FILE), stages=PIPELINE_STAGES)

    # 2. Launch the live dashboard.
    main_monitor.show_dashboard()

    # 3. Start concurrent worker processes.
    processes = []
    print("\n🚀 Starting pipeline workers...")
    for dataset in DATASETS:
        # 💡 **CHANGE:** Pass the raw arguments (string path and list) to the worker,
        # not the main_monitor object itself.
        process = Process(
            target=worker_process, 
            args=(dataset, str(REPORT_FILE), PIPELINE_STAGES)
        )
        processes.append(process)
        process.start()
        time.sleep(0.5)

    # 4. Wait for all worker processes to complete their tasks.
    for p in processes:
        p.join()

    print("\n🏁 All pipeline workers have completed their tasks.")
    print("✨ Dashboard will remain open. Close the plot window to exit the script.")
    
    # We now need to explicitly close the dashboard from the main process
    # when we are done, for a clean exit.
    main_monitor.close_dashboard()
    print("👋 Script finished.")