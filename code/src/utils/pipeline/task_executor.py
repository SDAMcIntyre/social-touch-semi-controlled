import traceback

from .pipeline_config_manager import DagConfigHandler
from .monitoring.pipeline_monitor import PipelineMonitor

# -----------------------------------------------------------------------------
# SHARED INFRASTRUCTURE (Required for standalone execution)
# -----------------------------------------------------------------------------
class TaskExecutor:
    """A context manager to handle the boilerplate of running a pipeline task."""
    def __init__(self, task_name, block_name, dag_handler, monitor):
        self.task_name: str = task_name
        self.block_name: str = block_name
        self.dag_handler: DagConfigHandler = dag_handler
        self.monitor: PipelineMonitor = monitor
        self.can_run: bool = False
        self.error_msg: str = None

    def __enter__(self):
        if self.dag_handler.can_run(self.task_name):
            self.can_run = True
            print(f"[{self.block_name}] ==> Running task: {self.task_name}")
            if self.monitor is not None:
                self.monitor.update(self.block_name, self.task_name, "RUNNING")
        return self 

    def __exit__(self, exc_type, exc_value, tb):
        if not self.can_run:
            return 

        if exc_type: 
            self.error_msg = f"Task '{self.task_name}' failed: {exc_value}"
            print(f"❌ {self.error_msg}\n{traceback.format_exc()}")
            if self.monitor is not None:
                self.monitor.update(self.block_name, self.task_name, "FAILURE", self.error_msg)
            return True
        else: 
            self.dag_handler.mark_completed(self.task_name)
            if self.monitor is not None:
                self.monitor.update(self.block_name, self.task_name, "SUCCESS")
        return False
