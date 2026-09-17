class IngestionError(Exception):
    """Base ingestion error."""


class UnsupportedFileTypeError(IngestionError):
    """No loader is registered for the file type."""


class OptionalDependencyError(IngestionError):
    """The selected loader needs an optional third-party dependency."""
