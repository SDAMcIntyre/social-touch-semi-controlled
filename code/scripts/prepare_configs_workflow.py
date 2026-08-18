import argparse
import logging
from multiprocessing import freeze_support
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import utils.path_tools as path_tools

from utils import DagConfigHandler, TaskExecutor

from primary_processing import KinectConfigFileHandler

from _2_primary_processing._1_prepare_configs.create_kinect_configs import generate_yaml_config
from _2_primary_processing._1_prepare_configs.create_forearm_configs import create_session_configs

# -----------------------------------------------------------------------------
# PREPARE CONFIGS TASKS
# -----------------------------------------------------------------------------

def create_kinect_configs_flow(
    project_data_root: Path,
    project_code_root: Path,
    *,
    force_processing: bool = False,
) -> None:
    project_config_path = project_data_root / "project_config.yaml"
    project_config = KinectConfigFileHandler.load_and_resolve_config(project_config_path)

    raw_root_name = project_config['path_roots']['raw_root']
    search_dir = project_data_root / raw_root_name / 'kinect'

    print(f"Scanning for videos in: {search_dir}")
    videos = list(search_dir.rglob('*_kinect.mkv'))
    print(f"Found {len(videos)} video(s).")

    for video_path in videos:
        generate_yaml_config(video_path, project_config, project_data_root, project_code_root)


def create_forearm_configs_flow(
    project_code_root: Path,
    project_data_root: Path,
    *,
    force_processing: bool = False,
) -> None:
    source_dir = project_code_root / 'configs' / 'kinect_configs'
    output_dir = project_code_root / 'configs' / 'forearm_configs'
    output_dir.mkdir(parents=True, exist_ok=True)

    create_session_configs(
        source_dir=source_dir,
        output_dir=output_dir,
        database_path=project_data_root,
    )


# -----------------------------------------------------------------------------
# EXECUTION LOGIC
# -----------------------------------------------------------------------------

def run_prepare_configs(
    dag_handler: DagConfigHandler,
    project_data_root: Path,
    project_code_root: Path,
) -> None:
    pipeline_stages = [
        {
            "name": "create_kinect_configs",
            "func": create_kinect_configs_flow,
            "params": lambda: {
                "project_data_root": project_data_root,
                "project_code_root": project_code_root,
            },
        },
        {
            "name": "create_forearm_configs",
            "func": create_forearm_configs_flow,
            "params": lambda: {
                "project_code_root": project_code_root,
                "project_data_root": project_data_root,
            },
        },
    ]

    for stage in pipeline_stages:
        task_name = stage["name"]
        executor = TaskExecutor(task_name, "prepare_configs", dag_handler, monitor=None)

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
            print(f"🛑 Failure in {task_name}. Aborting.")
            return


def main():
    freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dag-config", type=Path, required=True)
    args = parser.parse_args()

    project_data_root = path_tools.get_project_data_root()
    if not project_data_root:
        print("Operation cancelled by user.")
        return

    # Resolve project code root relative to this script's location:
    # code/scripts/prepare_configs_workflow.py → parents[2] = project root
    project_code_root = Path(__file__).resolve().parents[2]

    dag_handler = DagConfigHandler(args.dag_config)
    run_prepare_configs(dag_handler, project_data_root, project_code_root)


if __name__ == "__main__":
    main()
