import os
import logging
from pathlib import Path
from datetime import datetime
import shutil
import time
import traceback
from multiprocessing import Queue, freeze_support

from prefect import flow

# Setup a basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import utils.path_tools as path_tools

from utils import (
    DagConfigHandler,
    PipelineMonitor,
    TaskExecutor
)

from primary_processing import (
    KinectConfigFileHandler,
    KinectConfig,
    get_block_files
)

from _2_primary_processing._2_generate_rgb_depth_video import (
    generate_mkv_stream_analysis,
    extract_depth_to_tiff,
    extract_color_to_mp4
)

# -----------------------------------------------------------------------------
# PRIMARY PIPELINE TASKS
# -----------------------------------------------------------------------------

@flow(name="0. Analyse MKV video")
def validate_mkv_video(source_video: Path, output_dir: Path, *, force_processing: bool = False) -> Path:
    print(f"[{output_dir.name}] Analysing MKV video...")
    analysis_csv_path = output_dir / "mkv_analysis_report.csv"
    return generate_mkv_stream_analysis(source_video, analysis_csv_path, force_processing=force_processing)

@flow(name="1. Generate RGB Video")
def generate_rgb_video(source_video: Path, output_dir: Path, *, force_processing: bool = False) -> Path:
    print(f"[{output_dir.name}] Generating RGB video...")
    base_filename = os.path.splitext(os.path.basename(source_video))[0]
    rgb_path = output_dir / f"{base_filename}.mp4"
    rgb_video_path = extract_color_to_mp4(source_video, rgb_path, force_processing=force_processing)
    return Path(rgb_video_path) if not isinstance(rgb_video_path, Path) else rgb_video_path

@flow(name="2. Generate Depth Images")
def generate_depth_images(source_video: Path, output_dir: Path, *, force_processing: bool = False) -> Path:
    print(f"[{output_dir.name}] Generating depth images...")
    depth_dir = output_dir / source_video.name.replace(".mkv", "_depth")
    extract_depth_to_tiff(source_video, depth_dir, force_processing=force_processing)
    return depth_dir

# -----------------------------------------------------------------------------
# EXECUTION LOGIC
# -----------------------------------------------------------------------------

def run_primary_pipeline_session(
    config: KinectConfig,
    dag_handler: DagConfigHandler,
    monitor_queue: Queue = None,
    report_file_path: Path = None
):
    block_name = config.source_video.stem
    print(f"🚀 Starting PRIMARY pipeline for: {block_name}")

    if monitor_queue is not None:
        monitor = PipelineMonitor(
            report_path=report_file_path, stages=list(dag_handler.tasks.keys()), data_queue=monitor_queue
        )
    else:
        monitor = None

    # Primary Stages
    pipeline_stages = [
        {"name": "validate_mkv_video", 
         "func": validate_mkv_video, 
         "params": lambda: {"source_video": config.source_video, 
                            "output_dir": config.video_primary_output_dir}},
        {"name": "generate_rgb_video", 
         "func": generate_rgb_video, 
         "params": lambda: {"source_video": config.source_video, 
                            "output_dir": config.video_primary_output_dir}},
        {"name": "generate_depth_images", 
         "func": generate_depth_images, 
         "params": lambda: {"source_video": config.source_video, 
                            "output_dir": config.video_primary_output_dir}},
    ]

    for stage in pipeline_stages:
        task_name = stage["name"]
        executor = TaskExecutor(task_name, block_name, dag_handler, monitor)

        with executor:
            if not executor.can_run:
                continue
            
            options = dag_handler.get_task_options(task_name)
            params = stage["params"]()
            force = options.get('force_processing')
            if force is not None:
                params['force_processing'] = force

            stage["func"](**params)

        if executor.error_msg:
            print(f"🛑 Failure in {task_name}. Aborting session.")
            return {"status": "failed", "error": executor.error_msg}

    print(f"✅ Primary pipeline finished: {block_name}")
    return {"status": "success"}

def run_batch_primary(
    kinect_configs_dir: Path,
    project_data_root: Path,
    dag_config_path: Path,
    monitor_queue: Queue,
    report_file_path: Path,
    parallel: bool,
):
    dag_template = DagConfigHandler(dag_config_path)
    block_files = get_block_files(kinect_configs_dir)
    
    submitted_runs = []
    for block_file in block_files:
        config_data = KinectConfigFileHandler.load_and_resolve_config(block_file)
        validated_config = KinectConfig(config_data=config_data, database_path=project_data_root)
        dag_instance = dag_template.copy()

        if parallel:
            run = run_primary_pipeline_session.submit(
                config=validated_config,
                dag_handler=dag_instance,
                monitor_queue=monitor_queue,
                report_file_path=report_file_path,
            )
            submitted_runs.append(run)
        else:
            run_primary_pipeline_session(
                config=validated_config,
                dag_handler=dag_instance,
                monitor_queue=monitor_queue,
                report_file_path=report_file_path
            )

    if parallel:
        for run in submitted_runs:
            run.wait()

def setup_environment():
    project_data_root = path_tools.get_project_data_root()
    configs_dir = Path("configs")
    dag_config_path = Path(configs_dir / "primary_dag.yaml") # Pointing to primary yaml
    
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    report_file_path = reports_dir / f"{timestamp}_PRIMARY_status.xlsx"
    
    return project_data_root, configs_dir, dag_config_path, report_file_path

def main():
    freeze_support() 
    project_data_root, configs_dir, dag_config_path, report_file_path = setup_environment()

    if not dag_config_path.exists():
         print(f"❌ Config {dag_config_path} missing.")
         exit(1)

    main_dag_handler = DagConfigHandler(dag_config_path)
    is_parallel = main_dag_handler.get_parameter('parallel_execution', False)
    kinect_configs_dir = configs_dir / main_dag_handler.get_parameter('kinect_configs_directory')

    main_monitor = PipelineMonitor(
        report_path=str(report_file_path), 
        stages=list(main_dag_handler.tasks.keys()), 
        live_plotting=True
    )
    main_monitor.show_dashboard()

    run_batch_primary(
        kinect_configs_dir=kinect_configs_dir,
        project_data_root=project_data_root,
        dag_config_path=dag_config_path,
        monitor_queue=main_monitor.queue,
        report_file_path=report_file_path,
        parallel=is_parallel,
    )

    time.sleep(5)
    main_monitor.close_dashboard(block=True)

if __name__ == "__main__":
    main()