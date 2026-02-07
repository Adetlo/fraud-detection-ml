"""Structured error envelope: ``{"error": {"code", "message", "correlation_id", "details"}}``.

Stack traces never leave the process; unexpected exceptions are logged with the correlation
id and returned as ``SCORING_FAILED`` / ``INTERNAL_ERROR`` with a generic message.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from fraud_ml.contracts.score import ErrorDetail, ErrorResponse
from fraud_ml.exceptions import (
    ArtifactIntegrityError,
    DataValidationError,
    FraudMLError,
    ModelNotReadyError,
    ModelVersionMismatchError,
    SchemaVersionMismatchError,
    ScoringError,
    UnsupportedCategoryError,
)
from fraud_ml.logging import get_logger

log = get_logger(__name__)

STATUS_BY_CODE: dict[str, int] = {
    "INVALID_INPUT": 422,
    "UNSUPPORTED_CATEGORY": 422,
    "SCHEMA_VERSION_MISMATCH": 422,
    "MODEL_VERSION_MISMATCH": 503,
    "MODEL_NOT_READY": 503,
    "ARTIFACT_INTEGRITY": 503,
    "SCORING_FAILED": 500,
    "REQUEST_TOO_LARGE": 413,
    "BATCH_TOO_LARGE": 413,
    "INTERNAL_ERROR": 500,
}


def correlation_id_of(request: Request) -> str:
    return str(getattr(request.state, "correlation_id", "unknown"))


def error_response(
    code: str, message: str, correlation_id: str, details: list[Any] | None = None
) -> JSONResponse:
    payload = ErrorResponse(
        error=ErrorDetail(
            code=code, message=message, correlation_id=correlation_id, details=details
        )
    )
    return JSONResponse(
        status_code=STATUS_BY_CODE.get(code, 500), content=payload.model_dump(mode="json")
    )


def _classify_validation(errors: list[dict[str, Any]]) -> tuple[str, str]:
    for item in errors:
        location = ".".join(str(part) for part in item.get("loc", ()))
        message = str(item.get("msg", ""))
        if "schema_version" in location:
            return "SCHEMA_VERSION_MISMATCH", "Unsupported schema_version in request."
        if item.get("type") in {"enum", "literal_error"}:
            return "UNSUPPORTED_CATEGORY", f"Unsupported category value at {location}: {message}"
    return "INVALID_INPUT", "Request body violates the transaction contract."


def _sanitise(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Keep location/message/type; drop echoed input values so payload data is not reflected.
    return [
        {
            "loc": list(e.get("loc", ())),
            "msg": str(e.get("msg", "")),
            "type": str(e.get("type", "")),
        }
        for e in errors[:25]
    ]


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def _request_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [dict(e) for e in exc.errors()]
        code, message = _classify_validation(errors)
        return error_response(code, message, correlation_id_of(request), _sanitise(errors))

    @app.exception_handler(ModelNotReadyError)
    @app.exception_handler(ModelVersionMismatchError)
    @app.exception_handler(ArtifactIntegrityError)
    async def _model_errors(request: Request, exc: FraudMLError) -> JSONResponse:
        log.warning("api.model_error", code=exc.code, correlation_id=correlation_id_of(request))
        return error_response(exc.code, exc.message, correlation_id_of(request))

    @app.exception_handler(SchemaVersionMismatchError)
    @app.exception_handler(UnsupportedCategoryError)
    @app.exception_handler(DataValidationError)
    async def _input_errors(request: Request, exc: FraudMLError) -> JSONResponse:
        details = getattr(exc, "errors", None) or None
        return error_response(exc.code, exc.message, correlation_id_of(request), details)

    @app.exception_handler(ScoringError)
    async def _scoring_error(request: Request, exc: ScoringError) -> JSONResponse:
        log.error(
            "api.scoring_failed", correlation_id=correlation_id_of(request), reason=exc.message
        )
        return error_response(
            "SCORING_FAILED",
            "Scoring failed; see server logs with the correlation id.",
            correlation_id_of(request),
        )

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        log.exception(
            "api.unhandled",
            correlation_id=correlation_id_of(request),
            error_type=type(exc).__name__,
        )
        return error_response(
            "INTERNAL_ERROR", "An unexpected error occurred.", correlation_id_of(request)
        )
