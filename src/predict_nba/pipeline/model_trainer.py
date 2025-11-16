"""
Model training module for the NBA prediction project.

Responsibilities:
- Load cleaned training data from S3
- Train the ML model (Logistic Regression)
- Evaluate accuracy and ROC-AUC
- Save the trained model + scaler back to S3 as a .skops bundle
"""

import io
import os
import sys

import boto3
import pandas as pd
import skops.io as sio
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from predict_nba.utils.exception import CustomException
from predict_nba.utils.logger import logger


class S3Client:
    """Handles downloading training data and uploading model files."""

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

    def download_csv(self, key: str):
        """Download a CSV file from S3 as raw bytes."""
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=key)
            return resp["Body"].read()
        except Exception as e:
            CustomException(f"S3 download failed for {key}: {e}", sys)
            return None

    def upload_bytes(self, data: bytes, key: str, content_type="application/octet-stream"):
        """Upload raw bytes to S3."""
        try:
            self.s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
            )
            logger.info(f"Uploaded {key} to S3")
        except Exception as e:
            CustomException(f"S3 upload failed for {key}: {e}", sys)


class ModelTrainer:
    """Trains a logistic regression prediction model using advanced NBA statistics."""

    def __init__(
        self,
        model_type="logistic_regression",
        C=1.0,
        max_iter=1000,
        test_size=0.2,
        random_state=42,
    ):
        self.model_type = model_type
        self.C = C
        self.max_iter = max_iter
        self.test_size = test_size
        self.random_state = random_state

        load_dotenv()
        try:
            self.s3 = S3Client()
        except Exception as e:
            CustomException(f"Failed to initialize S3 client: {e}", sys)
            self.s3 = None

        # Full feature list used for training
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

    def _initialize_model(self):
        """Create the ML model instance based on configuration."""
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
        Train the model using training data loaded from S3.
        If 'save=True', the model + scaler are uploaded back to S3.
        """
        if self.s3 is None:
            CustomException("S3 client not initialized.", sys)
            return None, None

        try:
            raw = self.s3.download_csv(data_key)
            if raw is None:
                return None, None

            df = pd.read_csv(io.BytesIO(raw))

            if "TeamWin" not in df.columns:
                CustomException("Training data missing 'TeamWin' column.", sys)
                return None, None

            available = [f for f in self.features if f in df.columns]
            missing = [f for f in self.features if f not in df.columns]

            if missing:
                logger.warning(f"{len(missing)} missing features skipped: {missing}")

            X = df[available]
            y = df["TeamWin"]

            logger.info(f"Training samples: {X.shape[0]} | features: {X.shape[1]}")

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                stratify=y,
                random_state=self.random_state,
            )

            # Standardization
            scaler = StandardScaler().fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = self._initialize_model()
            if model is None:
                return None, None

            logger.info("Training model...")
            model.fit(X_train_scaled, y_train)

            # Evaluation
            y_pred = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred)

            logger.info(f"Accuracy: {acc:.4f}")
            logger.info(f"ROC-AUC: {auc:.4f}")
            logger.info("Classification report:\n" + classification_report(y_test, y_pred))

            # Save model bundle to S3
            if save:
                bundle = {"model": model, "scaler": scaler}
                tmp_path = "prediction_model.skops"
                sio.dump(bundle, tmp_path)

                with open(tmp_path, "rb") as f:
                    raw_bytes = f.read()

                self.s3.upload_bytes(
                    raw_bytes,
                    "models/prediction_model.skops",
                    content_type="application/octet-stream",
                )

            return model, scaler

        except Exception as e:
            CustomException(f"train_model failed: {e}", sys)
            return None, None
