# src/features/prediction_pipeline.py

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from supabase import create_client
from nba_api.stats.endpoints import scoreboardv2
from src.utils.exception import CustomException
from src.features.MakePrediction import MakePrediction
import sys

load_dotenv()

class DailyPredictor:
    """
    Handles automatic daily NBA predictions:
    - Updates yesterday's predictions with real results.
    - Creates new predictions for today's games.
    """

    def __init__(self, bucket_name="modelData"):
        try:
            self.url = os.getenv("SUPABASE_URL")
            self.key = os.getenv("SUPABASE_KEY")
            self.supabase = create_client(self.url, self.key)
        except Exception as e:
            raise CustomException(f"Failed to initialize Supabase client: {e}", sys)

    def update_predictions(self):
        """
        Updates prediction accuracy for yesterday's games.
        Moves data from 'Current Predictions' to 'Prediction History'.
        """
        try:
            current_preds = self.supabase.table("Current Predictions").select("*").execute().data
            teams = self.supabase.table("Teams").select("id, name").execute().data
            team_map = {t["name"]: str(t["id"]) for t in teams}

            updated_rows = []
            for row in current_preds:
                team_id = team_map[row["team"]]
                res = requests.get(
                    "https://api.pbpstats.com/get-games/nba",
                    params={
                        "Season": "2025-26",
                        "SeasonType": "Regular Season",
                        "EntityType": "Team",
                        "EntityId": team_id,
                    },
                )
                games = pd.DataFrame(res.json()["multi_row_table_data"])
                latest = games.sort_values("Date", ascending=False).iloc[0]
                row["date"] = latest["Date"]
                row["winner"] = latest["Points"] > latest["OpponentPoints"]
                row["prediction_correct"] = row["prediction"] == row["winner"]
                updated_rows.append(row)

            self.supabase.table("Prediction History").insert(updated_rows).execute()
            self.supabase.table("Current Predictions").delete().neq("team", "").execute()
            print(f"✅ Updated {len(updated_rows)} predictions in history.")
        except Exception as e:
            raise CustomException(f"update_predictions failed: {e}", sys)

    def new_predictions(self):
        """
        Fetches today's games from NBA API, predicts winners, and stores them.
        """
        try:
            teams = self.supabase.table("Teams").select("*").execute().data
            team_map = {str(t["id"]): t["name"] for t in teams}

            date_str = (datetime.utcnow() + timedelta(days=1)).strftime("%m/%d/%Y")
            games = scoreboardv2.ScoreboardV2(game_date=date_str).get_normalized_dict()["GameHeader"]

            predictor = MakePrediction()
            new_rows = []
            for g in games:
                home = team_map[str(g["HOME_TEAM_ID"])]
                away = team_map[str(g["VISITOR_TEAM_ID"])]
                result = predictor.predict(home, away)
                row = {
                    "team": home,
                    "opponent": away,
                    "prediction": 1 if result["winner"] == home else 0,
                    "confidence": result["confidence"],
                }
                new_rows.append(row)
                print(f"🧠 Predicted {home} vs {away}: {row}")
                time.sleep(5)

            self.supabase.table("Current Predictions").insert(new_rows).execute()
            print(f"✅ Inserted {len(new_rows)} new predictions.")
        except Exception as e:
            raise CustomException(f"new_predictions failed: {e}", sys)



