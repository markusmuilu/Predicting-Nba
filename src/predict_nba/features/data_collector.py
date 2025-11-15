"""
Data collection module for NBA training and prediction datasets.
Fetches team logs from PBPStats API and stores them in Supabase.
"""

import io
import os
import sys
import time

import pandas as pd
import requests
from dotenv import load_dotenv
from supabase import create_client

from predict_nba.utils.exception import CustomException
from predict_nba.utils.logger import logger


class ConfigCollection:
    """
    Loads NBA team information from Supabase instead of the PBPStats API.
    Expected columns in the 'teams' table:
        - id   (team ID used in API calls)
        - name (team abbreviation)
    """

    def __init__(self):
        try:
            load_dotenv()
            supabase = create_client(
                os.getenv("SUPABASE_URL"),
                os.getenv("SUPABASE_KEY")
            )

            logger.info("Loading team list from Supabase table 'teams'...")
            response = supabase.table("teams").select("*").execute()

            if not response.data:
                CustomException("Team list is empty in Supabase table 'teams'.", sys)
                self.teams = []
                return

            self.teams = response.data
            logger.info(f"Loaded {len(self.teams)} teams from Supabase.")

        except Exception as e:
            CustomException(f"Failed to initialize ConfigCollection: {e}", sys)
            self.teams = []


class DataCollector:
    """
    Downloads NBA team game logs from PBPStats API and uploads them to Supabase.
    """

    BASE_URL = "https://api.pbpstats.com"

    def __init__(self):
        try:
            self.config = ConfigCollection()
            self.teams = self.config.teams

            load_dotenv()
            self.supabase = create_client(
                os.getenv("SUPABASE_URL"),
                os.getenv("SUPABASE_KEY")
            )
            self.bucket_name = "modelData"

        except Exception as e:
            CustomException(f"Failed to initialize DataCollector: {e}", sys)
            self.supabase = None

    def collect_training_data(self, seasons=["2024-25"], upload=True):
        """
        Collects all team logs for given seasons and optionally uploads the
        combined dataset as 'training_data.csv' to Supabase.
        """
        if self.supabase is None or not self.teams:
            CustomException("DataCollector not initialized correctly.", sys)
            return None

        try:
            all_data = pd.DataFrame()

            for season in seasons:
                logger.info(f"Processing season: {season}")

                for team in self.teams:
                    team_name = team["name"]
                    team_id = team["id"]

                    try:
                        logger.info(f"Fetching logs for {team_name} ({team_id})...")

                        response = requests.get(
                            f"{self.BASE_URL}/get-game-logs/nba",
                            params={
                                "Season": season,
                                "SeasonType": "Regular Season",
                                "EntityType": "Team",
                                "EntityId": team_id,
                            },
                            timeout=20,
                        )
                        response.raise_for_status()

                        logs = response.json().get("multi_row_table_data", [])
                        if not logs:
                            logger.warning(f"No logs found for {team_name} in {season}")
                            continue

                        df = pd.DataFrame(logs)
                        if "Date" not in df.columns:
                            logger.warning(f"No 'Date' column for {team_name}, skipping.")
                            continue

                        df = df.sort_values(by="Date")
                        df["GamesPlayed"] = range(1, len(df) + 1)
                        df["team"] = team_name
                        df["season"] = season

                        all_data = pd.concat([all_data, df], ignore_index=True)
                        time.sleep(1)

                    except Exception as e:
                        CustomException(f"Error fetching {team_name}: {e}", sys)
                        continue

            logger.info(f"Combined dataset contains {len(all_data)} rows.")

            if upload and not all_data.empty:
                self._upload_to_supabase(all_data, "training_data.csv")
                logger.info("Uploaded training_data.csv to Supabase.")

            return all_data

        except Exception as e:
            CustomException(f"collect_training_data failed: {e}", sys)
            return None

    def get_current_season(self, team_abbrev, season="2025-26", upload=True):
        """
        Downloads logs for a single team for the current season.
        Uses Supabase 'teams' table for team ID.
        """
        try:
            team = next(
                (t for t in self.teams if t["name"].lower() == team_abbrev.lower()),
                None
            )

            if not team:
                CustomException(f"Team '{team_abbrev}' not found in Supabase teams table.", sys)
                return None

            team_id = team["id"]
            team_name = team["name"]

            logger.info(f"Fetching {season} logs for {team_name} ({team_id})...")

            response = requests.get(
                f"{self.BASE_URL}/get-game-logs/nba",
                params={
                    "Season": season,
                    "SeasonType": "Regular Season",
                    "EntityType": "Team",
                    "EntityId": team_id,
                },
                timeout=20,
            )
            response.raise_for_status()

            logs = response.json().get("multi_row_table_data", [])
            if not logs:
                CustomException(f"No logs found for {team_name} in {season}", sys)
                return None

            df = pd.DataFrame(logs).sort_values(by="Date")
            df["team"] = team_name
            df["season"] = season

            if upload:
                key = f"predict/{team_name}.csv"
                self._upload_to_supabase(df, key)
                logger.info(f"Uploaded {team_name}.csv to Supabase → {key}")

            time.sleep(1)
            return df

        except Exception as e:
            CustomException(f"get_current_season failed: {e}", sys)
            return None

    def _upload_to_supabase(self, df: pd.DataFrame, key: str):
        """
        Uploads a DataFrame as a CSV to Supabase storage.
        """
        try:
            buffer = io.StringIO()
            df.to_csv(buffer, index=False)
            csv_bytes = buffer.getvalue().encode("utf-8")

            result = self.supabase.storage.from_(self.bucket_name).upload(
                path=key,
                file=csv_bytes,
                file_options={
                    "content_type": "text/csv",
                    "upsert": "true"
                },
            )

            if hasattr(result, "error") and result.error:
                CustomException(f"Supabase upload failed: {result.error}", sys)
            else:
                logger.info(f"Uploaded {key} ({len(df)} rows) to Supabase.")

        except Exception as e:
            CustomException(f"_upload_to_supabase failed: {e}", sys)

