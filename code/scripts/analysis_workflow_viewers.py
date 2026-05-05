# analysis_workflow_viewers.py
# Thin entry point for viewer / viewer_support analysis tasks.
# All flow logic lives in analysis_workflow.py; this script imports and calls it
# with mode="viewers" so that only tasks with category "viewer" or
# "viewer_support" run.
import sys
import argparse
import logging
from pathlib import Path
from multiprocessing import freeze_support

# Make sure sibling scripts (analysis_workflow.py) are importable.
sys.path.insert(0, str(Path(__file__).parent))

from utils import path_tools, DagConfigHandler
from utils.pipeline.session_config_resolver import resolve_session_configs
from analysis_workflow import run_batch_analysis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    freeze_support()
    parser = argparse.ArgumentParser(
        description=(
            "Analysis workflow — viewer tasks only. "
            "Runs tasks with category 'viewer' or 'viewer_support' from the given "
            "DAG config. Processing outputs (CSVs, heatmaps, etc.) are assumed to "
            "exist on disk; use analysis_workflow_processing.py to generate them."
        )
    )
    parser.add_argument(
        "--dag-config",
        type=Path,
        default=Path("configs/analyse_workflow_viewers_dag.yaml"),
        help="Path to the DAG config YAML (default: configs/analyse_workflow_viewers_dag.yaml)",
    )
    args = parser.parse_args()

    dag_config_path: Path = args.dag_config
    project_data_root = path_tools.get_project_data_root()
    configs_dir = Path("configs")
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    report_file_path = reports_dir / "analysis_viewers_status.xlsx"

    if not dag_config_path.exists():
        raise FileNotFoundError(
            f"DAG config not found at {dag_config_path}"
        )

    dag_handler = DagConfigHandler(dag_config_path)
    entries = dag_handler.get_parameter('kinect_configs')
    block_files = resolve_session_configs(entries, configs_dir / "kinect_configs")

    run_batch_analysis(
        block_files=block_files,
        project_data_root=project_data_root,
        dag_handler=dag_handler,
        report_file_path=report_file_path,
        mode="viewers",
    )


if __name__ == "__main__":
    main()
