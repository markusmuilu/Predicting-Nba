"""
Initial S3 setup script for NBA prediction project.

Responsibilities:
- Ensure S3 folder structure exists
- Ensure teams.json exists (via bootstrap function)
- Ensure model exists (bootstrap trains it if missing)
"""

import os
import sys
import boto3
from dotenv import load_dotenv

from predict_nba.utils.logger import logger
from predict_nba.utils.exception import CustomException
from predict_nba.pipeline.bootstrap_model import ensure_teams_json, bootstrap_model


REQUIRED_PREFIXES = [
    "teams/",
    "training/",
    "clean/",
    "models/",
    "current/",
    "history/",
    "predict/",
    "predict/clean/"
]


def ensure_s3_structure(s3, bucket: str):
    """Create required S3 prefixes."""
    logger.info("Ensuring S3 folder structure...")

    for prefix in REQUIRED_PREFIXES:
        try:
            s3.put_object(Bucket=bucket, Key=prefix)
            logger.info(f"Verified folder: {prefix}")
        except Exception as e:
            raise CustomException(f"Failed to create S3 prefix {prefix}: {e}", sys)


def initialize_project():
    """Setup S3 and delegate the rest to bootstrap_model()."""
    try:
        load_dotenv()

        bucket = os.getenv("AWS_S3_BUCKET_NAME")
        region = os.getenv("AWS_REGION")

        if not bucket:
            raise CustomException("AWS_S3_BUCKET_NAME missing in environment", sys)

        s3 = boto3.client(
            "s3",
            region_name=region,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )

        logger.info("======== Starting S3 Project Setup ========")
        logger.info(f"Using bucket: {bucket}")

        # Ensure folder structure
        ensure_s3_structure(s3, bucket)

        # This will:
        # - ensure teams.json exists
        # - train model if missing
        logger.info("Running bootstrap process…")
        bootstrap_model()

        logger.info("======== Project setup complete ========")

    except Exception as e:
        raise CustomException(f"Project setup failed: {e}", sys)


if __name__ == "__main__":
    initialize_project()
