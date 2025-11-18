"""
Update routes for triggering daily prediction updates.
Handles:
- Resolving completed games
- Creating predictions for today's matchups
"""

from fastapi import APIRouter, HTTPException

from predict_nba.automation import DailyPredictor
from predict_nba.utils.logger import logger

router = APIRouter(prefix="/update", tags=["Updates"])


@router.post("")
def update_daily_stats():
    """
    Runs the daily update workflow:
      1. Resolves finished games and moves them to prediction_history.
      2. Generates new predictions for today's games.
    """
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
            detail="Daily update job failed. See logs for details."
        )

