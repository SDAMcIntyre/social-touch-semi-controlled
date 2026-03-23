class PipelineDependencyError(Exception):
    """Raised when an auto pipeline task requires a manual pipeline step that hasn't been run."""

    def __init__(self, message: str, required_pipeline: str, session_name: str = None):
        super().__init__(message)
        self.required_pipeline = required_pipeline
        self.session_name = session_name
