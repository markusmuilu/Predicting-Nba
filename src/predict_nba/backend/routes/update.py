"""
Update routes for triggering daily prediction updates.
Handles:
- Resolving completed games
- Creating predictions for today's matchups
"""

import os
from typing import Optional

from fastapi import APIRouter, HTTPException, Header, Request

from predict_nba.automation import DailyPredictor
from predict_nba.backend.limiter import limiter
from predict_nba.utils.logger import logger

router = APIRouter(prefix="/update", tags=["Updates"])


@router.post("")
@limiter.limit("5/minute")
def update_daily_stats(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """
    Runs the daily update workflow:
      1. Resolves finished games and moves them to prediction_history.
      2. Generates new predictions for today's games.

    Requires Authorization: Bearer <UPDATE_SECRET_TOKEN> header.
    """
    expected_token = os.getenv("UPDATE_SECRET_TOKEN")
    if not expected_token:
        logger.error("UPDATE_SECRET_TOKEN env var is not set.")
        raise HTTPException(status_code=500, detail="Server misconfiguration.")

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=403, detail="Missing or invalid Authorization header.")

    token = authorization[len("Bearer "):]
    if token != expected_token:
        raise HTTPException(status_code=403, detail="Invalid token.")

    try:
        logger.info("Starting daily update job...")
        dp = DailyPredictor()
        dp.run_all()
        logger.info("Daily update job completed successfully.")
        return {"message": "Daily update complete."}

    except Exception as e:
        logger.error(f"Daily update failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Daily update job failed. See logs for details.",
        )
