"""Generic pipeline stage dispatcher for the analysis entry scripts.

``run_pipeline_stages`` iterates a list of stage descriptors, opens a
``TaskExecutor`` for each, invokes the stage's ``params`` lambda to build
kwargs, injects the common ``input_items`` and ``force_processing`` kwargs,
and calls the flow function.  Exceptions are caught by the executor and
recorded as ``error_msg``; subsequent stages continue to run.

Stage descriptor schema (dict):
    name   : str          — DAG task name; must match the key in the DAG YAML.
    func   : callable     — the Prefect ``@flow`` function to invoke.
    params : callable     — zero-argument lambda that returns a dict of
                            task-specific kwargs.  Evaluated lazily so DAG
                            option reads only happen for tasks that actually
                            run.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from utils import DagConfigHandler, PipelineMonitor, TaskExecutor


def run_pipeline_stages(
    pipeline_stages: List[Dict[str, Any]],
    dag_handler: DagConfigHandler,
    monitor: PipelineMonitor,
    items_to_process: List[Tuple[Path, Path]],
    block_id_prefix: str,
) -> None:
    """Dispatch each stage in *pipeline_stages* via a ``TaskExecutor``.

    For every stage descriptor the runner:

    1. Skips the stage with a WARNING if its ``name`` is not present in
       the DAG handler's task registry (the DAG YAML has no entry for it).
    2. Opens a ``TaskExecutor`` using ``block_name = f"{block_id_prefix}_{name}"``.
    3. Bails out of the stage body early when ``executor.can_run`` is
       ``False`` (task disabled or dependency not met).
    4. Calls ``stage["params"]()`` to obtain task-specific kwargs, then
       merges common kwargs: ``input_items`` is always injected;
       ``force_processing`` is taken from the task's DAG options.
    5. Calls ``stage["func"](**kwargs)``.  Any exception is caught by the
       executor's ``__exit__`` and stored as ``executor.error_msg``;
       execution continues with the next stage.

    Parameters
    ----------
    pipeline_stages:
        Ordered list of stage descriptors.  See module docstring for the
        required keys.
    dag_handler:
        ``DagConfigHandler`` instance already loaded with the workflow's
        DAG YAML.
    monitor:
        ``PipelineMonitor`` used to report per-task status updates.
    items_to_process:
        List of ``(aggregated_csv_path, database_path)`` tuples produced
        by ``discover_input_items``.  Passed verbatim to every flow as
        ``input_items``.
    block_id_prefix:
        Prefix for the ``block_name`` string forwarded to each
        ``TaskExecutor``.  Use ``"batch_run_processing"`` for the
        processing slice and ``"batch_run_viewers"`` for the viewer slice.
    """
    for stage in pipeline_stages:
        task_name: str = stage["name"]
        flow_func = stage["func"]

        if task_name not in dag_handler.tasks:
            logging.warning(
                f"Task '{task_name}' is not in the DAG registry — skipping."
            )
            continue

        block_name = f"{block_id_prefix}_{task_name}"
        executor = TaskExecutor(task_name, block_name, dag_handler, monitor)

        with executor:
            if not executor.can_run:
                continue

            options = dag_handler.get_task_options(task_name)
            kwargs: Dict[str, Any] = stage["params"]()
            kwargs["input_items"] = items_to_process
            kwargs["force_processing"] = options.get("force_processing", False)

            flow_func(**kwargs)
