def __getattr__(name):
    if name == "DagConfigHandler":
        from .pipeline.pipeline_config_manager import DagConfigHandler
        return DagConfigHandler
    if name == "PipelineMonitor":
        from .pipeline.monitoring.pipeline_monitor import PipelineMonitor
        return PipelineMonitor
    if name == "TaskExecutor":
        from .pipeline.task_executor import TaskExecutor
        return TaskExecutor
    if name == "get_pca1_signal":
        from .generic_signal_processing import get_pca1_signal
        return get_pca1_signal
    if name == "get_pca1_signal_configurable":
        from .generic_signal_processing import get_pca1_signal_configurable
        return get_pca1_signal_configurable
    raise AttributeError(f"module 'utils' has no attribute {name!r}")
