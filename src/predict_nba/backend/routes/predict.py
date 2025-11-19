"""
Prediction API routes for generating matchup predictions.
"""

from fastapi import APIRouter, Query, HTTPException

from predict_nba.pipeline.make_prediction import MakePrediction
from predict_nba.utils.logger import logger

router = APIRouter(prefix="/predict", tags=["Predictions"])


@router.get("")
def get_prediction(
    team1: str = Query(..., description="Home team abbreviation (e.g., CLE)"),
    team2: str = Query(..., description="Away team abbreviation (e.g., ATL)")
):
    """
    Returns an ML prediction for the result of a matchup between two teams.
    Example: /predict?team1=CLE&team2=ATL
    """
    try:
        logger.info(f"API prediction request: {team1} vs {team2}")

        predictor = MakePrediction()
        result = predictor.predict(team1, team2)

        if result is None:
            raise HTTPException(
                status_code=500,
                detail="Prediction failed. See logs for details."
            )

        return {
            "winner": str(result["winner"]),
            "confidence": float(result["confidence"])
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Prediction route failed: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction error.")


