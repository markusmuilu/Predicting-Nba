"""
Prediction API routes for generating matchup predictions.
"""

import re

from fastapi import APIRouter, Query, HTTPException, Request

from predict_nba.backend.limiter import limiter
from predict_nba.pipeline.data_cleaner import DataCleaner
from predict_nba.pipeline.data_collector import DataCollector
from predict_nba.utils.logger import logger

router = APIRouter(prefix="/predict", tags=["Predictions"])

_TEAM_RE = re.compile(r"^[A-Z]{2,4}$")


@router.get("")
@limiter.limit("1/minute")
def get_prediction(
    request: Request,
    team1: str = Query(..., description="Home team abbreviation (e.g., CLE)"),
    team2: str = Query(..., description="Away team abbreviation (e.g., ATL)"),
):
    """
    Returns an ML prediction for the result of a matchup between two teams.
    Example: /predict?team1=CLE&team2=ATL
    """
    if not _TEAM_RE.match(team1) or not _TEAM_RE.match(team2):
        raise HTTPException(
            status_code=400,
            detail="Team abbreviations must be 2-4 uppercase letters (e.g. CLE, ATL).",
        )
    if team1 == team2:
        raise HTTPException(status_code=400, detail="team1 and team2 must be different.")

    try:
        logger.info(f"API prediction request: {team1} vs {team2}")

        collector = DataCollector()
        collector.get_current_season(team1)
        collector.get_current_season(team2)
        DataCleaner().clean_prediction_data(team1, team2)

        predictor = request.app.state.predictor
        result = predictor.predict_matchup_with_bundle(team1, team2)

        if result is None:
            raise HTTPException(
                status_code=500,
                detail="Prediction failed. See logs for details.",
            )

        return {
            "winner": str(result["winner"]),
            "confidence": float(result["confidence"]),
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Prediction route failed: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction error.")
