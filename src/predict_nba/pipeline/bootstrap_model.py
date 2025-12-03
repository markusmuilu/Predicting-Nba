"""
Bootstrap script that ensures:
- teams.json exists in S3
- model file exists in S3 (otherwise it trains one)
"""

import sys
import requests
from dotenv import load_dotenv
from botocore.exceptions import ClientError

from predict_nba.pipeline.data_collector import DataCollector
from predict_nba.pipeline.data_cleaner import DataCleaner
from predict_nba.pipeline.model_trainer import ModelTrainer
from predict_nba.utils.exception import CustomException
from predict_nba.utils.logger import logger
from predict_nba.utils.s3_client import S3Client


MODEL_KEY = "models/prediction_model.skops"
TEAMS_KEY = "teams/teams.json"
TEAMS_API_URL = "https://api.pbpstats.com/get-teams/nba"


def ensure_teams_json():
    """
    Ensures teams/teams.json exists in S3.
    Downloads from pbpstats API if missing.
    """
    s3 = S3Client()
    bucket = s3.bucket

    # Check if file exists
    try:
        s3.s3.head_object(Bucket=bucket, Key=TEAMS_KEY)
        return
    except Exception:
        pass

    # Fetch team list
    try:
        resp = requests.get(TEAMS_API_URL)
        resp.raise_for_status()
        raw = resp.json()
    except Exception as e:
        raise CustomException(f"Failed to fetch teams from pbpstats: {e}", sys)

    teams_raw = raw.get("teams", [])
    if not teams_raw:
        raise CustomException("pbpstats returned no teams list", sys)

    # Clean structure
    teams_clean = [{"id": int(t["id"]), "name": t["text"]} for t in teams_raw]

    # Upload to S3
    try:
        s3.upload_json(TEAMS_KEY, {"teams": teams_clean})
        logger.info("Uploaded teams.json to S3")
    except Exception as e:
        raise CustomException(f"Failed to upload teams.json: {e}", sys)


def model_exists():
    """
    Checks whether the trained model file exists in S3.
    """
    s3 = S3Client().s3
    bucket = S3Client().bucket

    try:
        s3.head_object(Bucket=bucket, Key=MODEL_KEY)
        return True
    except s3.exceptions.ClientError as e:  # type: ignore
        code = e.response.get("Error", {}).get("Code")
        if code in ("404", "NoSuchKey"):
            return False
        raise


def bootstrap_model():
    """
    Runs the bootstrap process:
    - ensures teams.json exists
    - trains a model if missing
    """
    try:
        load_dotenv()

        # Skip if model exists
        if model_exists():
            logger.info("Model exists — bootstrap skipped.")
            return

        # Ensure teams.json exists first
        ensure_teams_json()

        # Collect and prepare data
        collector = DataCollector()
        seasons = ["2015-16", "2016-2017", "2017-18", "2018-19", "2019-20", "2020-21","2021-22", "2022-23", "2023-24", "2024-25"]
        collector.collect_training_data(seasons)

        cleaner = DataCleaner()
        cleaner.clean_training_data()

        # Train and upload model
        trainer = ModelTrainer()
        trainer.train_model()


        logger.info("Bootstrap completed successfully.")

    except Exception as e:
        raise CustomException(f"Bootstrap model training failed: {e}", sys)


if __name__ == "__main__":
    bootstrap_model()
