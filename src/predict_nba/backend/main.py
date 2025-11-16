"""
FastAPI application for the NBA Prediction API.
Includes routes for generating predictions and updating game results.
"""

from fastapi import FastAPI

from predict_nba.backend.routes.predict import router as predict_router
from predict_nba.backend.routes.update import router as update_router
from predict_nba.utils.wait_for_model import wait_for_required_files


wait_for_required_files()

app = FastAPI(
    title="NBA Prediction API",
    version="1.0"
)

# Register route groups
app.include_router(predict_router)
app.include_router(update_router)

