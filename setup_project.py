"""
Initial project setup script.

This script:
- Ensures required Supabase tables exist
- Ensures the modelData storage bucket exists
- Collects and cleans training data
- Trains the initial prediction model
"""

import os
import sys
from dotenv import load_dotenv
from supabase import create_client, Client

from predict_nba.utils.logger import logger
from predict_nba.utils.exception import CustomException
from predict_nba.features.data_collector import DataCollector
from predict_nba.features.data_cleaner import DataCleaner
from predict_nba.features.model_trainer import ModelTrainer



REQUIRED_TABLES = {
    "teams": """
        CREATE TABLE IF NOT EXISTS teams (
            id INT PRIMARY KEY,
            name TEXT NOT NULL
        );
    """,
    "current_predictions": """
        CREATE TABLE IF NOT EXISTS current_predictions (
            id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            team TEXT NOT NULL,
            opponent TEXT NOT NULL,
            date TEXT NOT NULL,
            prediction BOOLEAN NOT NULL,
            confidence DOUBLE PRECISION NOT NULL,
            gameId TEXT NOT NULL UNIQUE
        );
    """,
    "prediction_history": """
        CREATE TABLE IF NOT EXISTS prediction_history (
            id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            team TEXT NOT NULL,
            opponent TEXT NOT NULL,
            date TEXT NOT NULL,
            prediction BOOLEAN NOT NULL,
            confidence DOUBLE PRECISION NOT NULL,
            winner BOOLEAN NOT NULL,
            prediction_correct BOOLEAN NOT NULL,
            gameId TEXT NOT NULL UNIQUE
        );
    """
}

REQUIRED_BUCKET = "modelData"


def ensure_bucket(supabase: Client, bucket_name: str):
    """Create the bucket if it does not exist."""
    try:
        buckets = supabase.storage.list_buckets()
        exists = any(b["name"] == bucket_name for b in buckets)

        if not exists:
            supabase.storage.create_bucket(bucket_name, {"public": False})
            logger.info(f"Created bucket: {bucket_name}")
        else:
            logger.info(f"Bucket exists: {bucket_name}")

    except Exception as e:
        raise CustomException(f"Bucket creation failed: {e}", sys)


def ensure_tables(supabase: Client):
    """Ensure all required tables exist in Supabase."""
    try:
        for table_name, ddl in REQUIRED_TABLES.items():
            logger.info(f"Checking table: {table_name}")
            supabase.rpc("exec_sql", {"sql": ddl}).execute()
            logger.info(f"Table ready: {table_name}")

    except Exception as e:
        raise CustomException(f"Table creation failed: {e}", sys)


def initialize_project():
    """Runs full setup workflow: create tables, create bucket, train model."""
    try:
        load_dotenv()

        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        supabase = create_client(supabase_url, supabase_key)

        logger.info("Starting project setup")

        ensure_tables(supabase)
        ensure_bucket(supabase, REQUIRED_BUCKET)

        logger.info("Collecting training data")
        collector = DataCollector()
        collector.collect_training_data(["2023-24", "2024-25"])

        logger.info("Cleaning training data")
        cleaner = DataCleaner()
        cleaner.clean_training_data()

        logger.info("Training model")
        trainer = ModelTrainer()
        trainer.train_model()

        logger.info("Project setup complete")

    except Exception as e:
        raise CustomException(f"Project setup failed: {e}", sys)


if __name__ == "__main__":
    initialize_project()
