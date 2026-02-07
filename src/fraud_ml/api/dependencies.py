"""Application state: settings, the loaded scorer, correlation ids and request-size limits."""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated

from fastapi import Depends, Request, Response
from fastapi.responses import JSONResponse

from fraud_ml.api.errors import error_response
from fraud_ml.config import Settings, get_settings
from fraud_ml.exceptions import FraudMLError, ModelNotReadyError
from fraud_ml.logging import bind_context, clear_context, get_logger
from fraud_ml.models.registry import LATEST
from fraud_ml.scoring.scorer import Scorer

log = get_logger(__name__)
CORRELATION_HEADER = "X-Correlation-ID"


class ModelHolder:
    """Holds the scorer for the process; ``load`` is safe to call repeatedly."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.scorer: Scorer | None = None
        self.load_error: str | None = None

    @property
    def artifact_path(self) -> Path:
        return self.settings.model_dir / LATEST

    def load(self) -> None:
        try:
            self.scorer = Scorer.from_path(self.artifact_path)
            self.load_error = None
            log.info("model.loaded", model_version=self.scorer.model_version)
        except FraudMLError as exc:
            self.scorer = None
            self.load_error = f"{exc.code}: {exc.message}"
            log.warning("model.load_failed", code=exc.code, reason=exc.message)

    @property
    def ready(self) -> bool:
        return self.scorer is not None

    def require(self) -> Scorer:
        if self.scorer is None:
            raise ModelNotReadyError("No valid fraud risk model is loaded.")
        return self.scorer


def get_holder(request: Request) -> ModelHolder:
    holder: ModelHolder = request.app.state.model_holder
    return holder


def get_scorer(holder: Annotated[ModelHolder, Depends(get_holder)]) -> Scorer:
    return holder.require()


def get_app_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


ScorerDep = Annotated[Scorer, Depends(get_scorer)]
HolderDep = Annotated[ModelHolder, Depends(get_holder)]
SettingsDep = Annotated[Settings, Depends(get_app_settings)]


def build_holder(settings: Settings | None = None) -> ModelHolder:
    return ModelHolder(settings or get_settings())


async def correlation_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    correlation_id = request.headers.get(CORRELATION_HEADER) or uuid.uuid4().hex
    request.state.correlation_id = correlation_id[:64]
    started = time.perf_counter()
    bind_context(correlation_id=request.state.correlation_id)
    try:
        length = request.headers.get("content-length")
        limit: int = request.app.state.settings.api_max_request_bytes
        if length is not None and length.isdigit() and int(length) > limit:
            response: Response = error_response(
                "REQUEST_TOO_LARGE",
                f"Request body exceeds {limit} bytes.",
                request.state.correlation_id,
            )
        else:
            response = await call_next(request)
        response.headers[CORRELATION_HEADER] = request.state.correlation_id
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        if not request.url.path.startswith("/health"):
            log.info(
                "api.request",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=duration_ms,
            )
        return response
    finally:
        clear_context()


__all__ = [
    "CORRELATION_HEADER",
    "HolderDep",
    "JSONResponse",
    "ModelHolder",
    "ScorerDep",
    "SettingsDep",
    "build_holder",
    "correlation_middleware",
    "get_holder",
    "get_scorer",
]
