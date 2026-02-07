"""ASGI application factory. ``uvicorn fraud_ml.main:app`` serves the scoring API."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from fraud_ml import __version__
from fraud_ml.api.dependencies import build_holder, correlation_middleware
from fraud_ml.api.errors import register_error_handlers
from fraud_ml.api.routes import api, health
from fraud_ml.config import Settings, get_settings
from fraud_ml.logging import configure_logging

DESCRIPTION = (
    "Fraud risk scoring and decision support for synthetic financial transactions. "
    "Educational implementation: no authentication, not for production fraud decisioning."
)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_format)
    holder = build_holder(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        holder.load()
        yield

    app = FastAPI(
        title="Fraud Detection ML - risk scoring API",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )
    app.state.settings = settings
    app.state.model_holder = holder
    app.middleware("http")(correlation_middleware)
    register_error_handlers(app)
    app.include_router(health)
    app.include_router(api)
    return app


app = create_app()
