class EngineDeadError(RuntimeError):
    """The engine raised or exited. Nothing can be served until the process restarts."""
