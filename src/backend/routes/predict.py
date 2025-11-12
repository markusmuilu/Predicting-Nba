from fastapi import APIRouter, Query, HTTPException
from src.features.make_prediction import MakePrediction

router = APIRouter(prefix="/predict", tags=["Predictions"])

@router.get("/")
def get_prediction(team1: str = Query(...), team2: str = Query(...)):
    """
    Returns the prediction of a game between two teams.
    Example: /predict?team1=CLE&team2=ATL
    """
    try:
        predictor = MakePrediction()
        result = predictor.predict(team1, team2)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
