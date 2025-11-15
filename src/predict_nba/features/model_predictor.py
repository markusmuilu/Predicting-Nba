"""
Model prediction module for the NBA prediction project.
Loads the trained model and scaler and predicts the outcome of a matchup.
"""

import io
import os
import sys

import pandas as pd
import skops.io as sio
from dotenv import load_dotenv
from supabase import create_client

from predict_nba.utils.exception import CustomException
from predict_nba.utils.logger import logger


class ModelPredictor:
    """
    Loads the trained model and predicts the winner for a given matchup
    using Supabase as the model and data source.
    """

    def __init__(self, bucket_name="modelData"):
        load_dotenv()
        self.bucket_name = bucket_name

        try:
            self.supabase = create_client(
                os.getenv("SUPABASE_URL"),
                os.getenv("SUPABASE_KEY"),
            )
        except Exception as e:
            CustomException(f"Failed to initialize Supabase client: {e}", sys)
            self.supabase = None

    def predict_matchup(self, team1, team2):
        """
        Downloads model + scaler + cleaned matchup data from Supabase
        and returns the predicted winner with confidence.
        """
        if self.supabase is None:
            CustomException("Supabase client is not initialized.", sys)
            return None

        try:
            model_key = "models/prediction_model.skops"
            data_key = f"predict/clean/{team1}vs{team2}.csv"

            logger.info(f"Downloading model: {model_key}")
            model_bytes = self.supabase.storage.from_(self.bucket_name).download(model_key)

            untrusted = sio.get_untrusted_types(data=model_bytes)
            if untrusted:
                logger.warning(f"Untrusted types found in model: {untrusted}")

            bundle = sio.loads(model_bytes, trusted=untrusted)

            model = bundle.get("model")
            scaler = bundle.get("scaler")
            if model is None or scaler is None:
                CustomException("Model file missing 'model' or 'scaler'.", sys)
                return None

            logger.info(f"Downloading cleaned matchup data: {data_key}")
            data_bytes = self.supabase.storage.from_(self.bucket_name).download(data_key)
            df = pd.read_csv(io.BytesIO(data_bytes))

            feature_cols = [
                col for col in df.columns
                if col.endswith("_avg") or col.endswith("_diff") or col in ["IsHome", "HomeAdvantage"]
            ]

            if not feature_cols:
                CustomException("No valid feature columns in prediction data.", sys)
                return None

            X = df[feature_cols]
            X_scaled = scaler.transform(X)

            y_pred = model.predict(X_scaled)[0]
            prob = model.predict_proba(X_scaled)[0][1]

            winner = team1 if y_pred == 1 else team2
            confidence = round(prob * 100 if y_pred == 1 else (1 - prob) * 100, 2)

            logger.info(f"Predicted winner: {winner} (confidence: {confidence}%)")

            return {"winner": winner, "confidence": confidence}

        except Exception as e:
            CustomException(f"predict_matchup failed: {e}", sys)
            return None
