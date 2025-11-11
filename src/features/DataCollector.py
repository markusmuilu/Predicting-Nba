import os
import sys
import io
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client
from src.utils.exception import CustomException


class ConfigCollection:
    """Fetches basic API configuration and NBA team data."""

    BASE_URL = "https://api.pbpstats.com"

    def __init__(self):
        try:
            response = requests.get(f"{self.BASE_URL}/get-teams/nba", timeout=15)
            response.raise_for_status()
            self.teams = response.json().get("teams", [])
            if not self.teams:
                raise CustomException("No teams returned from API", sys)
        except Exception as e:
            raise CustomException(f"Failed to initialize ConfigCollection: {e}", sys)


class DataCollector:
    """Downloads team game logs and uploads them to Supabase."""

    def __init__(self):
        try:
            self.config = ConfigCollection()
            self.teams = self.config.teams

            load_dotenv()
            self.supabase = create_client(
                os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY")
            )
            self.bucket_name = "modelData"

        except Exception as e:
            raise CustomException(f"Failed to initialize DataCollector: {e}", sys)

    def collect_training_data(self, seasons=["2024-25"], upload=True):
        """
        Collects all team logs across the given seasons and optionally uploads
        them as a single CSV file ('training_data.csv') to Supabase.
        """
        try:
            all_data = pd.DataFrame()

            for season in seasons:
                print(f"\nProcessing season: {season}")

                for team in self.teams:
                    team_name = team["text"]
                    try:
                        print(f"Fetching {team_name} ({team['id']}) for {season}...")

                        response = requests.get(
                            f"{self.config.BASE_URL}/get-game-logs/nba",
                            params={
                                "Season": season,
                                "SeasonType": "Regular Season",
                                "EntityType": "Team",
                                "EntityId": team["id"],
                            },
                            timeout=20,
                        )
                        response.raise_for_status()
                        logs = response.json().get("multi_row_table_data", [])
                        if not logs:
                            continue

                        team_df = pd.DataFrame(logs)
                        if "Date" not in team_df.columns:
                            continue

                        team_df = team_df.sort_values(by="Date")
                        team_df["GamesPlayed"] = range(1, len(team_df) + 1)
                        team_df["team"] = team_name
                        team_df["season"] = season

                        all_data = pd.concat([all_data, team_df], ignore_index=True)
                        time.sleep(1)

                    except Exception as e:
                        print(f"Error fetching {team_name}: {e}")
                        continue

            print(f"\nCombined dataset size: {len(all_data)} rows")

            if upload and not all_data.empty:
                self._upload_to_supabase(all_data, "training_data.csv")
                print("Uploaded training_data.csv to Supabase.")

            return all_data

        except Exception as e:
            raise CustomException(f"collect_training_data failed: {e}", sys)

    def get_current_season(self, team_name, season="2025-26", upload=True):
        """
        Downloads logs for a single team and optionally uploads them to Supabase.
        """
        try:
            team = next(
                (t for t in self.teams if t["text"].lower() == team_name.lower()), None
            )
            if not team:
                raise CustomException(f"Team '{team_name}' not found in API list", sys)

            print(f"Fetching {team_name} ({team['id']}) for {season}...")
            response = requests.get(
                f"{self.config.BASE_URL}/get-game-logs/nba",
                params={
                    "Season": season,
                    "SeasonType": "Regular Season",
                    "EntityType": "Team",
                    "EntityId": team["id"],
                },
                timeout=20,
            )
            response.raise_for_status()
            logs = response.json().get("multi_row_table_data", [])
            if not logs:
                raise CustomException(f"No logs found for {team_name} {season}", sys)

            team_df = pd.DataFrame(logs).sort_values(by="Date")
            team_df["team"] = team["text"]
            team_df["season"] = season

            if upload:
                key = f"predict/{team_name.replace(' ', '_')}.csv"
                self._upload_to_supabase(team_df, key)
                print(f"Uploaded {team_name}.csv to Supabase → {key}")

            time.sleep(1)
            return team_df

        except Exception as e:
            raise CustomException(f"get_current_season failed: {e}", sys)

    def _upload_to_supabase(self, df: pd.DataFrame, key: str):
        """Uploads a DataFrame as a CSV file to Supabase storage."""
        try:
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_bytes = csv_buffer.getvalue().encode("utf-8")

            res = self.supabase.storage.from_(self.bucket_name).upload(
                path=key,
                file=csv_bytes,
                file_options={"content_type": "text/csv", "upsert": "true"},
            )

            if hasattr(res, "error") and res.error:
                raise CustomException(f"Supabase upload failed: {res.error}", sys)

            print(f"Uploaded {key} ({len(df)} rows) to Supabase bucket '{self.bucket_name}'")

        except Exception as e:
            raise CustomException(f"_upload_to_supabase failed: {e}", sys)
