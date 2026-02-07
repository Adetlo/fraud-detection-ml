"""HTTP routes: health, model info, single and batch scoring."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Response, status
from pydantic import BaseModel

from fraud_ml import __version__
from fraud_ml.api.dependencies import HolderDep, ScorerDep, SettingsDep
from fraud_ml.api.errors import error_response
from fraud_ml.contracts.score import (
    BatchScoreRequest,
    BatchScoreResponse,
    ErrorResponse,
    ModelInfo,
    ScoreResponse,
)
from fraud_ml.contracts.transaction import TransactionFeatures
from fraud_ml.logging import get_logger

log = get_logger(__name__)
health = APIRouter(tags=["health"])
api = APIRouter(prefix="/api/v1", tags=["scoring"])

ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    422: {"model": ErrorResponse},
    413: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
}


class LiveResponse(BaseModel):
    status: str
    service: str
    version: str


class ReadyResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None


@health.get("/health/live", response_model=LiveResponse)
async def live() -> LiveResponse:
    return LiveResponse(status="ok", service="fraud-detection-ml", version=__version__)


@health.get(
    "/health/ready", response_model=ReadyResponse, responses={503: {"model": ReadyResponse}}
)
async def ready(holder: HolderDep, response: Response) -> ReadyResponse:
    if not holder.ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadyResponse(status="not_ready", model_loaded=False, model_version=None)
    scorer = holder.require()
    return ReadyResponse(status="ready", model_loaded=True, model_version=scorer.model_version)


@api.get("/model", response_model=ModelInfo, responses=ERROR_RESPONSES)
async def model_info(scorer: ScorerDep) -> ModelInfo:
    return scorer.info()


@api.post("/score", response_model=ScoreResponse, responses=ERROR_RESPONSES)
async def score(transaction: TransactionFeatures, scorer: ScorerDep) -> ScoreResponse:
    started = time.perf_counter()
    result = scorer.score_one(transaction)
    log.info(
        "api.score",
        model_version=scorer.model_version,
        transaction_id=transaction.transaction_id,
        risk_band=result.risk_band.value,
        decision_support=result.decision_support.value,
        duration_ms=round((time.perf_counter() - started) * 1000, 2),
    )
    return result


@api.post("/score/batch", response_model=BatchScoreResponse, responses=ERROR_RESPONSES)
async def score_batch(
    request: BatchScoreRequest, scorer: ScorerDep, settings: SettingsDep
) -> BatchScoreResponse | Response:
    if len(request.transactions) > settings.api_max_batch_size:
        return error_response(
            "BATCH_TOO_LARGE",
            f"Batch exceeds api_max_batch_size={settings.api_max_batch_size} transactions.",
            "see-header",
        )
    started = time.perf_counter()
    scores = scorer.score_transactions(request.transactions)
    log.info(
        "api.score_batch",
        model_version=scorer.model_version,
        count=len(scores),
        duration_ms=round((time.perf_counter() - started) * 1000, 2),
    )
    return BatchScoreResponse(model_version=scorer.model_version, count=len(scores), scores=scores)
