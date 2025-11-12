# model_predict.py

import io
import os
import sys
import pandas as pd
import skops.io as sio
from src.utils.exception import CustomException
from supabase import create_client
from dotenv import load_dotenv


class ModelPredictor:
    """
    Loads the trained model and predicts the winner for a given matchup
    using Supabase as the data source.
    """

    def __init__(self, bucket_name="modelData"):
        load_dotenv()
        try:
            self.supabase = create_client(
                os.getenv("SUPABASE_URL"),
                os.getenv("SUPABASE_KEY"),
            )
            self.bucket_name = bucket_name
        except Exception as e:
            raise CustomException(f"Failed to initialize Supabase client: {e}", sys)

    def predict_matchup(self, team1, team2):
        """
        Downloads model + scaler + cleaned prediction data from Supabase,
        predicts the winner, and prints the confidence.
        """
        try:
            model_key = "models/prediction_model.skops"
            data_key = f"predict/clean/{team1}vs{team2}.csv"

            print(f"⬇️ Downloading model: {model_key}")
            model_bytes = self.supabase.storage.from_(self.bucket_name).download(model_key)

            # ✅ Correct skops API for bytes
            untrusted = sio.get_untrusted_types(data=model_bytes)
            if untrusted:
                print("Untrusted types found in model:", untrusted)

            bundle = sio.loads(model_bytes, trusted=untrusted)

            model = bundle.get("model")
            scaler = bundle.get("scaler")
            if model is None or scaler is None:
                raise CustomException("Invalid model file: missing model/scaler", sys)

            print(f"⬇️ Downloading cleaned matchup data: {data_key}")
            data_bytes = self.supabase.storage.from_(self.bucket_name).download(data_key)
            df = pd.read_csv(io.BytesIO(data_bytes))

            feature_cols = [
                col for col in df.columns
                if col.endswith("_avg") or col.endswith("_diff") or col in ["IsHome", "HomeAdvantage"]
            ]
            if not feature_cols:
                raise CustomException("No valid feature columns found in prediction data.", sys)

            X = df[feature_cols]
            X_scaled = scaler.transform(X)

            y_pred = model.predict(X_scaled)[0]
            prob = model.predict_proba(X_scaled)[0][1]
            winner = team1 if y_pred == 1 else team2
            confidence = round(prob * 100 if y_pred == 1 else (1 - prob) * 100, 2)

            print(f"🏀 Predicted winner: {winner} (confidence: {confidence}%)")
            return {"winner": winner, "confidence": confidence}

        except Exception as e:
            raise CustomException(f"predict_matchup failed: {e}", sys)
