"""Exception hierarchy. Every error carries a stable machine-readable ``code``."""

from __future__ import annotations


class FraudMLError(Exception):
    """Base class for all project errors."""

    code: str = "INTERNAL_ERROR"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code


class DataValidationError(FraudMLError):
    """Input data violates the transaction contract."""

    code = "INVALID_INPUT"

    def __init__(self, message: str, *, errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.errors = errors or []


class SchemaVersionMismatchError(FraudMLError):
    code = "SCHEMA_VERSION_MISMATCH"


class ModelVersionMismatchError(FraudMLError):
    code = "MODEL_VERSION_MISMATCH"


class ModelNotReadyError(FraudMLError):
    code = "MODEL_NOT_READY"


class UnsupportedCategoryError(DataValidationError):
    code = "UNSUPPORTED_CATEGORY"


class ScoringError(FraudMLError):
    code = "SCORING_FAILED"


class ArtifactIntegrityError(FraudMLError):
    """A model artifact failed its integrity or trust checks and was not loaded."""

    code = "ARTIFACT_INTEGRITY"


class LeakageError(FraudMLError):
    """A forbidden column reached the feature matrix."""

    code = "LEAKAGE_GUARD"


class ConfigurationError(FraudMLError):
    code = "CONFIGURATION_ERROR"
