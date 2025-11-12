from fastapi import APIRouter, HTTPException
from src.features.daily_predictor import DailyPredictor
from src.features.model_trainer import ModelTrainer
from src.features.data_collector import DataCollector
from src.features.data_cleaner import DataCleaner

router = APIRouter(prefix="/update", tags=["Updates"])

@router.post("/")
def update_daily_stats(train: bool = False, seasons: list[str] = ["2024-25"]):
    """
    Daily job:
      1. Optionally retrains model using given seasons.
      2. Updates yesterday's predictions in Supabase.
      3. Creates new predictions for today's games.
    """
    try:
        dp = DailyPredictor()
        dp.update_predictions()
        dp.new_predictions()
        return {"message": "Daily update complete."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
