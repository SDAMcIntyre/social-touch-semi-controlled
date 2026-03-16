class PipelineDependencyError(Exception):
    """Raised when an auto pipeline task requires a manual pipeline step that hasn't been run."""

    def __init__(self, message: str, required_pipeline: str):
        super().__init__(message)
        self.required_pipeline = required_pipeline
