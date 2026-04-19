"""
FastAPI application for the NBA Prediction API.
Includes routes for generating predictions and updating game results.
"""

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from predict_nba.backend.limiter import limiter
from predict_nba.backend.routes.predict import router as predict_router
from predict_nba.backend.routes.update import router as update_router
from predict_nba.pipeline.model_predictor import ModelPredictor
from predict_nba.utils.wait_for_model import wait_for_required_files

wait_for_required_files()


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor = ModelPredictor()
    lock = threading.Lock()
    with lock:
        predictor.load_bundle()
    app.state.predictor = predictor
    yield


def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please try again later."},
    )


app = FastAPI(title="NBA Prediction API", version="1.0", lifespan=lifespan)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://markusmuilu.page"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(predict_router)
app.include_router(update_router)
