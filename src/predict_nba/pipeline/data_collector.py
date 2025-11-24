"""
Data collection module for NBA training and prediction datasets.

Responsibilities:
- Load team metadata from S3 (teams.json)
- Fetch raw game logs from the PBPStats API
- Build per-team CSVs and combined multi-season training datasets
- Upload outputs to S3 for downstream cleaning and modeling
"""

import io
import os
import sys
import time
import json

import boto3
import pandas as pd
import requests
from dotenv import load_dotenv

from predict_nba.utils.exception import CustomException
from predict_nba.utils.logger import logger


class S3Client:
    """Handles uploads and downloads between the application and S3."""

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

    def upload_csv(self, df: pd.DataFrame, key: str):
        """Upload a DataFrame to S3 under the given key."""
        try:
            buffer = io.StringIO()
            df.to_csv(buffer, index=False)
            payload = buffer.getvalue().encode("utf-8")

            self.s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=payload,
                ContentType="text/csv",
            )

            logger.info(f"Uploaded {key} ({len(df)} rows) to S3")

        except Exception as e:
            CustomException(f"S3 upload failed for {key}: {e}", sys)

    def download_json(self, key: str):
        """Download a JSON file from S3 and return a Python object."""
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=key)
            return json.loads(resp["Body"].read().decode("utf-8"))
        except Exception as e:
            CustomException(f"S3 JSON download failed for {key}: {e}", sys)
            return None


class ConfigCollection:
    """Loads team metadata from S3 (teams.json)."""

    TEAMS_KEY = "teams/teams.json"

    def __init__(self):
        try:
            self.s3 = S3Client()
            logger.info("Loading team list from S3...")

            teams = self.s3.download_json(self.TEAMS_KEY)
            if not teams:
                CustomException("Team list in S3 is empty or missing.", sys)
                self.teams = []
                return

            teams_json = teams.get("teams", [])
            self.teams = teams_json
            logger.info(f"Loaded {len(self.teams)} teams from S3")

        except Exception as e:
            CustomException(f"Failed to initialize ConfigCollection: {e}", sys)
            self.teams = []


class DataCollector:
    """
    Fetches team logs from PBPStats and writes the raw data to S3.
    Used for both multi-season training datasets and current-season predictions.
    """

    BASE_URL = "https://api.pbpstats.com"

    def __init__(self):
        try:
            self.config = ConfigCollection()
            self.teams = self.config.teams
            self.s3 = S3Client()
        except Exception as e:
            CustomException(f"Failed to initialize DataCollector: {e}", sys)
            self.s3 = None

    def collect_training_data(self, seasons=["2024-25"], upload=True):
        """
        Download logs for every team for the selected seasons.
        Produces a unified training CSV and uploads it to:
            training/training_data.csv
        """
        if self.s3 is None or not self.teams:
            CustomException("DataCollector not initialized correctly.", sys)
            return None

        all_data = pd.DataFrame()

        try:
            for season in seasons:
                logger.info(f"Processing season: {season}")

                for team in self.teams:
                    team_name = team["name"]
                    team_id = team["id"]

                    try:
                        logger.info(f"Fetching logs for {team_name} ({team_id})")

                        resp = requests.get(
                            f"{self.BASE_URL}/get-game-logs/nba",
                            params={
                                "Season": season,
                                "SeasonType": "Regular Season",
                                "EntityType": "Team",
                                "EntityId": team_id,
                            },
                            timeout=20,
                        )
                        resp.raise_for_status()

                        logs = resp.json().get("multi_row_table_data", [])
                        if not logs:
                            logger.warning(f"No logs found for {team_name} in {season}")
                            continue

                        df = pd.DataFrame(logs)
                        if "Date" not in df.columns:
                            logger.warning(f"Missing Date column for {team_name}, skipping")
                            continue

                        df = df.sort_values("Date")
                        df["GamesPlayed"] = range(1, len(df) + 1)
                        df["team"] = team_name
                        df["season"] = season

                        all_data = pd.concat([all_data, df], ignore_index=True)
                        time.sleep(5)

                    except Exception as e:
                        CustomException(f"Failed to fetch logs for {team_name}: {e}", sys)
                        continue

            logger.info(f"Final training dataset contains {len(all_data)} rows")

            if upload and not all_data.empty:
                self.s3.upload_csv(all_data, "training/training_data.csv")

            return all_data

        except Exception as e:
            CustomException(f"collect_training_data failed: {e}", sys)
            return None

    def get_current_season(self, team_abbrev, season="2025-26", upload=True):
        """
        Fetch logs for a single team for the current season.
        Uploads results to:
            predict/<TEAM>.csv
        """
        try:
            team = next(
                (t for t in self.teams if t["name"].lower() == team_abbrev.lower()),
                None,
            )

            if not team:
                CustomException(f"Team '{team_abbrev}' not found in teams.json.", sys)
                return None

            team_id = team["id"]
            team_name = team["name"]

            logger.info(f"Fetching {season} logs for {team_name} ({team_id})")

            resp = requests.get(
                f"{self.BASE_URL}/get-game-logs/nba",
                params={
                    "Season": season,
                    "SeasonType": "Regular Season",
                    "EntityType": "Team",
                    "EntityId": team_id,
                },
                timeout=20,
            )
            resp.raise_for_status()

            logs = resp.json().get("multi_row_table_data", [])
            if not logs:
                CustomException(f"No logs found for {team_name} in {season}", sys)
                return None

            df = pd.DataFrame(logs).sort_values("Date")
            df["team"] = team_name
            df["season"] = season

            if upload:
                key = f"predict/{team_name}.csv"
                self.s3.upload_csv(df, key)
                logger.info(f"Uploaded {team_name}.csv to S3 as {key}")

            time.sleep(2)
            return df

        except Exception as e:
            CustomException(f"get_current_season failed: {e}", sys)
            return None
