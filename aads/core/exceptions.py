"""
AADS custom exceptions.

Provides a hierarchy of exceptions for clear error handling across agents
and tools. All AADS exceptions inherit from AADSError so callers can catch
the entire family when appropriate.
"""


class AADSError(Exception):
    """Base exception for all AADS errors."""


class ConfigError(AADSError):
    """Raised when configuration is invalid or missing."""


class DataLoadError(AADSError):
    """Raised when a dataset cannot be loaded or validated."""


class ArtifactError(AADSError):
    """Raised when an artifact operation fails (path conflict, write error, etc.)."""


class LeakageError(AADSError):
    """Raised when a data-leakage violation is detected."""


class SchemaValidationError(AADSError):
    """Raised when input data does not match the expected schema."""


class AgentError(AADSError):
    """Raised when an agent encounters an unrecoverable error during execution."""


class ToolExecutionError(AADSError):
    """Raised when a tool (loader, viz, ML) fails during execution."""
