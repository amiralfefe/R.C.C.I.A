"""User-facing errors shared by MultiCancer adapters and the Streamlit hub."""


class AdapterError(RuntimeError):
    """Base error raised by a specialized adapter."""


class InvalidImageError(AdapterError):
    """Raised when uploaded content cannot be used as an image."""


class CheckpointMissingError(AdapterError):
    """Raised when a required local checkpoint is absent."""


class CheckpointIncompatibleError(AdapterError):
    """Raised when a checkpoint cannot be loaded by its specialized pipeline."""


class ModelNotLoadedError(AdapterError):
    """Raised when prediction is requested before lazy model loading."""


class ExplanationUnavailableError(AdapterError):
    """Raised when a visual explanation cannot be generated."""
