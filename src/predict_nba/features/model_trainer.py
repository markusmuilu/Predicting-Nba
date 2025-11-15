"""
Model training module for the NBA prediction project.
Supports Logistic Regression and includes data loading,
feature preparation, scaling, evaluation, and optional upload to Supabase.
"""

import io
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import skops.io as sio
from supabase import create_client

from predict_nba.utils.exception import CustomException
from predict_nba.utils.logger import logger


class ModelTrainer:
    """
    Trains and stores a general-purpose prediction model (currently Logistic Regression)
    using advanced NBA statistics.
    """

    def __init__(
        self,
        model_type="logistic_regression",
        C=1.0,
        max_iter=1000,
        test_size=0.2,
        random_state=42,
        bucket_name="modelData",
    ):
        self.model_type = model_type
        self.C = C
        self.max_iter = max_iter
        self.test_size = test_size
        self.random_state = random_state
        self.bucket_name = bucket_name

        # Full feature set (team, opponent, differentials, context)
        self.features = [
            "OffPoss_avg", "DefPoss_avg", "Pace_avg", "Fg3Pct_avg", "Fg2Pct_avg", "TsPct_avg",
            "OffRtg_avg", "DefRtg_avg", "NetRtg_avg", "EfgDiff_avg", "TsDiff_avg",
            "Rebounds_avg", "Steals_avg", "Blocks_avg",
            "Opp_OffPoss_avg", "Opp_DefPoss_avg", "Opp_Pace_avg", "Opp_Fg3Pct_avg",
            "Opp_Fg2Pct_avg", "Opp_TsPct_avg", "Opp_OffRtg_avg", "Opp_DefRtg_avg",
            "Opp_NetRtg_avg", "Opp_EfgDiff_avg", "Opp_TsDiff_avg",
            "Opp_Rebounds_avg", "Opp_Steals_avg", "Opp_Blocks_avg",
            "OffPoss_diff", "DefPoss_diff", "Pace_diff", "Fg3Pct_diff", "Fg2Pct_diff",
            "TsPct_diff", "OffRtg_diff", "DefRtg_diff", "NetRtg_diff", "EfgDiff_diff",
            "TsDiff_diff", "Rebounds_diff", "Steals_diff", "Blocks_diff",
            "IsHome", "HomeAdvantage",
        ]

        load_dotenv()
        try:
            self.supabase = create_client(
                os.getenv("SUPABASE_URL"),
                os.getenv("SUPABASE_KEY")
            )
        except Exception as e:
            CustomException(f"Failed to initialize Supabase client: {e}", sys)
            self.supabase = None

    def _initialize_model(self):
        """Initializes the selected model type."""
        if self.model_type == "logistic_regression":
            return LogisticRegression(
                C=self.C,
                max_iter=self.max_iter,
                solver="lbfgs",
                n_jobs=-1,
            )

        CustomException(f"Unsupported model type: {self.model_type}", sys)
        return None

    def train_model(self, data_key="clean/training_data_clean.csv", save=True):
        """
        Trains the ML model on cleaned training data from Supabase.
        Optionally uploads the model + scaler back to storage.
        """
        if self.supabase is None:
            CustomException("Supabase client is not initialized.", sys)
            return None, None

        try:
            logger.info(f"Downloading training data from Supabase: {data_key}")
            res = self.supabase.storage.from_(self.bucket_name).download(data_key)
            df = pd.read_csv(io.BytesIO(res))

            if "TeamWin" not in df.columns:
                CustomException("Training data missing 'TeamWin' target column.", sys)
                return None, None

            available = [f for f in self.features if f in df.columns]
            missing = [f for f in self.features if f not in df.columns]

            if missing:
                logger.warning(f"{len(missing)} missing features skipped: {missing}")

            X = df[available]
            y = df["TeamWin"]

            logger.info(f"Loaded dataset with {X.shape[0]} samples, {X.shape[1]} features.")

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                stratify=y,
                random_state=self.random_state
            )

            # Standardization
            scaler = StandardScaler().fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Model initialization
            model = self._initialize_model()
            if model is None:
                return None, None

            logger.info(f"Training model: {self.model_type}")
            model.fit(X_train_scaled, y_train)

            # Evaluation
            y_pred = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred)

            logger.info(f"Model accuracy: {acc:.4f}")
            logger.info(f"Model ROC-AUC: {auc:.4f}")
            logger.info("Classification report:\n" + classification_report(y_test, y_pred))

            # Upload model + scaler
            if save:
                model_obj = {"model": model, "scaler": scaler}
                tmp_path = "prediction_model.skops"
                sio.dump(model_obj, tmp_path)

                with open(tmp_path, "rb") as f:
                    file_bytes = f.read()

                model_key = "models/prediction_model.skops"
                logger.info(f"Uploading trained model to Supabase: {model_key}")

                upload_result = self.supabase.storage.from_(self.bucket_name).upload(
                    path=model_key,
                    file=file_bytes,
                    file_options={"content_type": "application/octet-stream", "upsert": "true"},
                )

                if hasattr(upload_result, "error") and upload_result.error:
                    CustomException(f"Supabase upload failed: {upload_result.error}", sys)
                else:
                    logger.info("Model uploaded successfully.")

            return model, scaler

        except Exception as e:
            CustomException(f"train_model failed: {e}", sys)
            return None, None
