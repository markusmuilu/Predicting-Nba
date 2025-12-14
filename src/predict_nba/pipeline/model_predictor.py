"""
Model prediction module for the NBA prediction project.

Responsibilities:
- Load the trained model + scaler from S3 (.npz & .skops)
- Load the cleaned matchup CSV from S3
- Prepare features for inference
- Predict the winner and confidence percentage
"""

import io
import os
import sys

import boto3
import pandas as pd
import numpy as np
import skops.io as sio
from dotenv import load_dotenv
from nn import NeuralNetwork, load_model

from predict_nba.utils.exception import CustomException
from predict_nba.utils.logger import logger


class S3Client:
    """Utility class for downloading binary files from S3."""

    def __init__(self):
        load_dotenv()
        self.bucket = os.getenv("AWS_S3_BUCKET_NAME")
        region = os.getenv("AWS_REGION")

        try:
            self.s3 = boto3.client(
                "s3",
                region_name=region,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            )
        except Exception as e:
            raise CustomException(f"S3 initialization failed: {e}", sys)

    def download_bytes(self, key: str):
        """Download raw bytes from S3."""
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=key)
            return resp["Body"].read()
        except Exception as e:
            CustomException(f"S3 download failed for {key}: {e}", sys)
            return None


class ModelPredictor:
    """Loads the trained model and performs predictions for team matchups."""

    def __init__(self):
        load_dotenv()

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

    def predict_matchup(self, team1, team2):
        """
        Predicts the winner between two teams using the trained model.

        Steps:
        - Download model and scaler
        - Extract model + scaler
        - Download cleaned matchup data
        - Select relevant features
        - Run prediction and compute confidence
        """
        if self.s3 is None:
            CustomException("S3 client not initialized.", sys)
            return None

        try:
            model_key = "models/model.npz"
            scaler_key = "models/scaler.skops"
            data_key = f"predict/clean/{team1}vs{team2}.csv"

            # Load model 
            logger.info(f"Downloading model: {model_key}")
            model_bytes = self.s3.download_bytes(model_key)
            if model_bytes is None:
                return None

            with open("tmp_model.npz", "wb") as f:
                f.write(model_bytes)

            model = load_model("tmp_model.npz")

            # Load scaler
            logger.info(f"Downloading scaler: {scaler_key}")
            scaler_bytes = self.s3.download_bytes(scaler_key)
            if scaler_bytes is None:
                return None

            untrusted = sio.get_untrusted_types(data=scaler_bytes)
            bundle = sio.loads(scaler_bytes, trusted=untrusted)
            scaler = bundle["scaler"]


            if model is None or scaler is None:
                CustomException("Model or scaler missing.", sys)
                return None

            # Load matchup data
            logger.info(f"Downloading cleaned matchup data: {data_key}")
            data_bytes = self.s3.download_bytes(data_key)
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
            row = X_scaled[0]                
            out = model.predict(row)         

            result = out["result"]
            confidence = out["confidence"]

            winner = team1 if result == 1 else team2
            confidence = round(confidence * 100, 2)

            logger.info(f"Predicted: {winner} ({confidence}%)")

            return {
                "winner": winner,
                "confidence": confidence
            }

        except Exception as e:
            CustomException(f"predict_matchup failed: {e}", sys)
            return None

if __name__ == "__main__":
    team1 = "BKN"
    team2 = "NOP"

    predictor = ModelPredictor()
    predictor.predict_matchup(team1,team2)