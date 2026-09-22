"""Local REST interface. Model absence is explicit in readiness and scoring."""

import logging
import os
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException

from risklab.service import CreditRequest, RiskService

logger = logging.getLogger(__name__)


def create_app(service=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.service = service
        if service is None:
            try:
                app.state.service = RiskService(os.getenv("MODEL_DIR", "artifacts"))
            except (OSError, ValueError, KeyError):
                logger.exception("Model could not be loaded")
        yield

    app = FastAPI(title="Credit risk research API", version="1.0.0", lifespan=lifespan)

    def current():
        value = getattr(app.state, "service", None)
        if value is None:
            raise HTTPException(503, "Model unavailable. Run python -m risklab.train first.")
        return value

    @app.get("/health")
    def health():
        return {"alive": True}

    @app.get("/ready")
    def ready():
        return {"ready": True, "model_version": current().version}

    @app.post("/predict")
    def predict(request: CreditRequest):
        result = current().predict(request)
        result["request_id"] = str(uuid4())
        logger.info(
            "score request_id=%s model=%s action=%s",
            result["request_id"],
            result["model_version"],
            result["action"],
        )
        return result

    return app


app = create_app()
