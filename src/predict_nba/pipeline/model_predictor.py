"""
Model prediction module for the NBA prediction project.

Responsibilities:
- Load the trained model + scaler bundle
- Load the cleaned matchup CSV from S3
- Prepare features for inference
- Predict the winner and confidence percentage
"""

import io
import sys

import pandas as pd
import skops.io as sio
from dotenv import load_dotenv

from predict_nba.utils.exception import CustomException
from predict_nba.utils.logger import logger
from predict_nba.utils.s3_client import S3Client




class ModelPredictor:
    """Loads the trained model and performs predictions for team matchups."""

    def __init__(self):
        load_dotenv()

        self.model = None
        self.scaler = None

        self.features = [
            "OffPoss_avg", "DefPoss_avg", "Pace_avg", "Fg3Pct_avg", "Fg2Pct_avg", "TsPct_avg",
            "OffRtg_avg", "DefRtg_avg", "NetRtg_avg", "EfgDiff_avg", "TsDiff_avg",
            "Rebounds_avg", "Steals_avg", "Blocks_avg", "SeasonWins", "SeasonLosses", 
            "SeasonWinPct", "IsBackToBack",

            "Opp_OffPoss_avg", "Opp_DefPoss_avg", "Opp_Pace_avg", "Opp_Fg3Pct_avg",
            "Opp_Fg2Pct_avg", "Opp_TsPct_avg", "Opp_OffRtg_avg", "Opp_DefRtg_avg",
            "Opp_NetRtg_avg", "Opp_EfgDiff_avg", "Opp_TsDiff_avg",
            "Opp_Rebounds_avg", "Opp_Steals_avg", "Opp_Blocks_avg",
            "Opp_SeasonWins", "Opp_SeasonLosses", "Opp_SeasonWinPct", "Opp_IsBackToBack",

            "OffPoss_diff", "DefPoss_diff", "Pace_diff", "Fg3Pct_diff", "Fg2Pct_diff",
            "TsPct_diff", "OffRtg_diff", "DefRtg_diff", "NetRtg_diff", "EfgDiff_diff",
            "TsDiff_diff", "Rebounds_diff", "Steals_diff", "Blocks_diff",

            "IsHome", "HomeAdvantage",
        ]

        try:
            self.s3 = S3Client()
        except Exception as e:
            CustomException(f"Failed to initialize S3 client: {e}", sys)
            self.s3 = None

    def load_bundle(self):
        """Downloads the model bundle once and caches model+scaler on the instance."""
        if self.s3 is None:
            raise RuntimeError("S3 client not initialized.")

        model_key = "models/prediction_model.skops"
        logger.info(f"Loading model bundle at startup: {model_key}")
        model_bytes = self.s3.download(model_key)
        if model_bytes is None:
            raise RuntimeError("Failed to download model bundle.")

        untrusted = sio.get_untrusted_types(data=model_bytes)
        bundle = sio.loads(model_bytes, trusted=untrusted)

        self.model = bundle.get("model")
        self.scaler = bundle.get("scaler")

        if self.model is None or self.scaler is None:
            raise RuntimeError("Model bundle is missing 'model' or 'scaler'.")

        logger.info("Model bundle cached successfully.")

    def predict_matchup_with_bundle(self, team1, team2):
        """Predicts using the pre-loaded model/scaler — skips the R2 model download."""
        if self.model is None or self.scaler is None:
            logger.error("predict_matchup_with_bundle called before load_bundle.")
            return None

        if self.s3 is None:
            logger.error("S3 client not initialized.")
            return None

        try:
            data_key = f"predict/clean/{team1}vs{team2}.csv"
            logger.info(f"Downloading matchup data: {data_key}")
            data_bytes = self.s3.download(data_key)
            if data_bytes is None:
                return None

            df = pd.read_csv(io.BytesIO(data_bytes))
            feature_cols = [f for f in self.features if f in df.columns]

            if not feature_cols:
                logger.error("Prediction data contains no valid features.")
                return None

            X = df[feature_cols]
            X_scaled = self.scaler.transform(X)

            pred = self.model.predict(X_scaled)[0]
            prob = self.model.predict_proba(X_scaled)[0][1]

            winner = team1 if pred == 1 else team2
            confidence = round(prob * 100 if pred == 1 else (1 - prob) * 100, 2)

            logger.info(f"Predicted: {winner} ({confidence}%)")
            return {"winner": winner, "confidence": confidence}

        except Exception as e:
            CustomException(f"predict_matchup_with_bundle failed: {e}", sys)
            return None

    def predict_matchup(self, team1, team2):
        """
        Predicts the winner between two teams using the trained model.

        Steps:
        - Download Bundle
        - Extract model + scaler
        - Download cleaned matchup data
        - Select relevant features
        - Run prediction and compute confidence
        """
        if self.s3 is None:
            CustomException("S3 client not initialized.", sys)
            return None

        try:
            model_key = "models/prediction_model.skops"
            data_key = f"predict/clean/{team1}vs{team2}.csv"

            # Load model bundle
            logger.info(f"Downloading model: {model_key}")
            model_bytes = self.s3.download(model_key)
            if model_bytes is None:
                return None

            untrusted = sio.get_untrusted_types(data=model_bytes)
            bundle = sio.loads(model_bytes, trusted=untrusted)

            model = bundle.get("model")
            scaler = bundle.get("scaler")

            if model is None or scaler is None:
                CustomException("Model bundle missing 'model' or 'scaler'.", sys)
                return None

            # Load matchup data
            logger.info(f"Downloading cleaned matchup data: {data_key}")
            data_bytes = self.s3.download(data_key)
            if data_bytes is None:
                return None

            df = pd.read_csv(io.BytesIO(data_bytes))

            # Select inference features
            feature_cols = [f for f in self.features if f in df.columns]



            if not feature_cols:
                CustomException("Prediction data contains no valid features.", sys)
                return None

            X = df[feature_cols]
            X_scaled = scaler.transform(X)

            # Predict outcome
            pred = model.predict(X_scaled)[0]
            prob = model.predict_proba(X_scaled)[0][1]

            winner = team1 if pred == 1 else team2
            confidence = round(prob * 100 if pred == 1 else (1 - prob) * 100, 2)

            logger.info(f"Predicted: {winner} ({confidence}%)")

            return {"winner": winner, "confidence": confidence}

        except Exception as e:
            CustomException(f"predict_matchup failed: {e}", sys)
            return None

if __name__ == "__main__":
    team1 = "BKN"
    team2 = "NOP"

    predictor = ModelPredictor()
    predictor.predict_matchup(team1,team2)