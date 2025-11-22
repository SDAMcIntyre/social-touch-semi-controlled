import copy
from typing import Dict, Any, List, Set
from pathlib import Path
import yaml

# --- DAG Configuration Handler ---
class DagConfigHandler:
    """
    Handles loading and interpreting a DAG configuration file to manage task execution.

    This class reads a YAML file that defines a series of tasks, their dependencies,
    and whether they are enabled. It provides methods to check if a task is ready
    to run based on these rules and tracks the completion of tasks for a single pipeline run.
    """
    def __init__(self, config_path: Path):
        """Initializes the handler by loading and parsing the YAML config file."""
        if not config_path.exists():
            raise FileNotFoundError(f"DAG configuration file not found at: {config_path}")
        with open(config_path, 'r') as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)

        self.tasks: Dict[str, Any] = self.config.get('tasks', {})
        self.parameters: Dict[str, Any] = self.config.get('parameters', {})
        self.completed_tasks: Set[str] = set()

    def get_parameter(self, param_name: str, default: Any = None) -> Any:
        """Retrieves a global parameter from the configuration."""
        return self.parameters.get(param_name, default)

    def get_task_options(self, task_name: str) -> Dict[str, Any]:
        """
        Retrieves the 'options' dictionary for a specific task.

        This allows you to access task-specific parameters, such as 'monitor'.
        
        Args:
            task_name: The name of the task for which to retrieve options.

        Returns:
            A dictionary of options for the specified task. Returns an empty
            dictionary if the task or its 'options' key are not found.
        """
        task_config = self.tasks.get(task_name, {})
        return task_config.get('options', {})

    def can_run(self, task_name: str) -> bool:
        """
        Determines if a task can be executed based on its configuration and dependencies.
        A task can run if it's enabled and all its dependencies have been completed.
        """
        task_config = self.tasks.get(task_name)
        if not task_config:
            print(f"⚠️ Warning: Task '{task_name}' not found in DAG configuration. Skipping.")
            return False
        if not task_config.get('enabled', False):
            return False
        dependencies: List[str] = task_config.get('depends_on', [])
        if not set(dependencies).issubset(self.completed_tasks):
            return False
        return True

    def mark_completed(self, task_name: str) -> None:
        """Marks a task as completed, allowing dependent tasks to run."""
        if task_name in self.tasks:
            self.completed_tasks.add(task_name)
            print(f"✅ Task '{task_name}' marked as completed.")
        else:
            print(f"⚠️ Warning: Attempted to mark non-existent task '{task_name}' as completed.")

    def copy(self):
        """Creates a deep copy of the handler for independent processing."""
        return copy.deepcopy(self)