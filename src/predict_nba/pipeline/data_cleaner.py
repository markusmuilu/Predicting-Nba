"""
Cleaning module for NBA training and prediction datasets.

Responsibilities:
- Load raw team logs from S3
- Compute rolling averages and efficiency metrics
- Align team and opponent features
- Prepare model-ready training and prediction datasets
- Write cleaned outputs back to S3
"""

import io
import os
import sys
import json
import time 

import boto3
import pandas as pd
import requests
from dotenv import load_dotenv

from predict_nba.utils.exception import CustomException
from predict_nba.utils.logger import logger


class S3Client:
    """Simple S3 helper for downloading and uploading CSV files."""

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
            # Fail fast: raising to ensure caller knows initialization failed
            raise CustomException(f"S3 initialization failed: {e}", sys)

    def download_csv(self, key: str):
        """Return raw CSV bytes from S3."""
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=key)
            return resp["Body"].read()
        except Exception as e:
            # raise instead of silently creating an exception object
            raise CustomException(f"S3 download failed for {key}: {e}", sys)

    def upload_csv_bytes(self, csv_bytes: bytes, key: str):
        """Upload CSV bytes to S3."""
        try:
            self.s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=csv_bytes,
                ContentType="text/csv",
            )
            logger.info(f"Uploaded {key} to S3")
        except Exception as e:
            # raise instead of silently creating an exception object
            raise CustomException(f"S3 upload failed for {key}: {e}", sys)


class DataCleaner:
    """
    Converts raw team statistics into standardized datasets used by the model.

    Main tasks:
    - Compute offensive/defensive ratings
    - Generate rolling averages
    - Build team and opponent feature pairs
    - Compute feature differentials
    - Export cleaned training and prediction datasets
    """

    BASE_URL = "https://api.pbpstats.com"

    def __init__(self, window_size: int = 10):
        self.window_size = window_size

        # Base stats from PBPStats logs
        self.base_features = ["OffPoss", "DefPoss", "Pace", "Fg3Pct", "Fg2Pct", "TsPct"]

        # Advanced and derived metrics
        self.advanced_features = [
            "OffRtg", "DefRtg", "NetRtg",
            "EfgDiff", "TsDiff",
            "Rebounds", "Steals", "Blocks",
        ]

        # initialize s3 client; fail if S3 client cannot be created
        self.s3 = S3Client()

    def _fetch_home_away_map(self, seasons):
        """
        Fetches GameId → home/away info for given seasons.
        Includes retry logic to avoid pbpstats timeout issues.
        """

        mapping = {}

        for season in seasons:
            retries = 5

            for attempt in range(retries):
                try:
                    resp = requests.get(
                        f"{self.BASE_URL}/get-games/nba",
                        params={"Season": season, "SeasonType": "Regular Season"},
                        timeout=10,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    # fill mapping
                    for g in data.get("results", []):
                        mapping[g["GameId"]] = {
                            "HomeTeam": g.get("HomeTeamAbbreviation"),
                            "AwayTeam": g.get("AwayTeamAbbreviation"),
                            "HomePoints": g.get("HomePoints"),
                            "AwayPoints": g.get("AwayPoints"),
                        }

                    break  # success → exit retry-loop

                except Exception as e:
                    logger.warning(
                        f"Home/away fetch failed ({season}) attempt {attempt+1}/{retries}: {e}"
                    )
                    time.sleep(2)  # wait before retry

                    if attempt == retries - 1:
                        raise CustomException(
                            f"Home/away fetch failed after retries for {season}: {e}", sys
                        )

        logger.info(f"Home/away map contains {len(mapping)} games")
        return mapping


    def clean_training_data(self, key="training/training_data.csv", upload=True):
        """
        Clean raw multi-season logs from S3 and generate the model's training dataset.

        Outputs:
            clean/training_data_clean.csv
        """
        if self.s3 is None:
            raise CustomException("S3 client not initialized.", sys)

        raw = self.s3.download_csv(key)
        if raw is None:
            return None

        data = pd.read_csv(io.BytesIO(raw))

        # Standardize column formatting
        data.columns = (
            data.columns.str.strip()
            .str.replace("\ufeff", "", regex=False)
            .str.replace(" ", "", regex=False)
        )

        data = data.sort_values(["team", "Date"])
        data["GameId"] = data["GameId"].astype(str).str.zfill(10)

        # Ensure required raw fields exist
        required = [
            "team", "GameId", "Date", "Opponent", "Points",
            "GamesPlayed", "DefPoss", "OffPoss",
            "Fg3Pct", "Fg2Pct", "TsPct", "EfgPct",
            "Rebounds", "Steals", "Blocks", "season",
        ]
        missing = [c for c in required if c not in data.columns]
        if missing:
            raise CustomException(f"Missing required columns: {missing}", sys)

        # Map GameId → home/away info
        seasons = data["season"].unique().tolist()
        home_away_map = self._fetch_home_away_map(seasons)

        # Determine home/away for each log row
        data["IsHome"] = data.apply(
            lambda r: 1 if home_away_map.get(r["GameId"], {}).get("HomeTeam") == r["team"]
            else 0 if home_away_map.get(r["GameId"], {}).get("AwayTeam") == r["team"]
            else None,
            axis=1,
        )
        data = data.dropna(subset=["IsHome"])
        data["IsHome"] = data["IsHome"].astype(int)

        # Basic rating metrics
        data["OffRtg"] = (data["Points"] / data["OffPoss"]) * 100
        # DefRtg must use the opponent points allowed in the previous team game -> shift per team
        data["DefRtg"] = (data.groupby("team")["Points"].shift(1) / data["DefPoss"]) * 100
        data["NetRtg"] = data["OffRtg"] - data["DefRtg"]
        data["EfgDiff"] = data.groupby("team")["EfgPct"].diff().fillna(0)
        data["TsDiff"] = data.groupby("team")["TsPct"].diff().fillna(0)

        # Rolling averages
        all_features = self.base_features + self.advanced_features
        for col in all_features:
            if col in data.columns:
                data[f"{col}_avg"] = (
                    data.groupby("team")[col]
                    .transform(lambda x: x.shift(1).rolling(self.window_size, min_periods=self.window_size).mean())
                )

        # Only keep rows where all averages are available
        avg_cols = [f"{c}_avg" for c in all_features if f"{c}_avg" in data.columns]
        data = data.dropna(subset=avg_cols)

        # Build opponent feature set
        opp_cols = avg_cols
        opp_df = data[["team", "GameId", "Points"] + opp_cols + ["IsHome"]].copy()
        opp_df.columns = (
            ["OpponentTeam", "GameId", "Opp_Points"]
            + [f"Opp_{c}" for c in opp_cols]
            + ["Opp_IsHome"]
        )

        merged = data.merge(
            opp_df,
            left_on=["Opponent", "GameId"],
            right_on=["OpponentTeam", "GameId"],
            how="left",
        )

        # Labels and additional metrics
        merged["PointDifferential"] = merged["Points"] - merged["Opp_Points"]
        merged["TeamWin"] = (merged["PointDifferential"] > 0).astype(int)

        # Compute DefRtg_avg and NetRtg_avg based on averages where available
        if "Opp_Points" in merged and "DefPoss_avg" in merged:
            merged["DefRtg_avg"] = (merged["Opp_Points"] / merged["DefPoss_avg"]) * 100
        if "OffRtg_avg" in merged and "DefRtg_avg" in merged:
            merged["NetRtg_avg"] = merged["OffRtg_avg"] - merged["DefRtg_avg"]

        # Differences between teams and opponents
        if "EfgPct_avg" in merged and "Opp_EfgPct_avg" in merged:
            merged["EfgDiff_avg"] = merged["EfgPct_avg"] - merged["Opp_EfgPct_avg"]

        if "TsPct_avg" in merged and "Opp_TsPct_avg" in merged:
            merged["TsDiff_avg"] = merged["TsPct_avg"] - merged["Opp_TsPct_avg"]

        for col in all_features:
            left = f"{col}_avg"
            right = f"Opp_{col}_avg"
            if left in merged.columns and right in merged.columns:
                merged[f"{col}_diff"] = merged[left] - merged[right]

        merged["HomeAdvantage"] = merged["IsHome"] - merged["Opp_IsHome"]
        merged = merged[merged["IsHome"] == 1]

        # Final training columns
        final_cols = (
            ["team", "Opponent", "IsHome", "HomeAdvantage", "PointDifferential", "TeamWin"]
            + [f"{c}_avg" for c in all_features if f"{c}_avg" in merged.columns]
            + [f"Opp_{c}_avg" for c in all_features if f"Opp_{c}_avg" in merged.columns]
            + [f"{c}_diff" for c in all_features if f"{c}_diff" in merged.columns]
        )

        final = merged[final_cols].dropna()

        if upload:
            csv_bytes = final.to_csv(index=False).encode("utf-8")
            self.s3.upload_csv_bytes(csv_bytes, "clean/training_data_clean.csv")

        return final

    def clean_prediction_data(self, team1, team2, upload=True):
        """
        Produce a single-row matchup dataset for prediction inference.
        Reads latest per-team CSVs from S3 and computes aligned features.
        """
        key1 = f"predict/{team1}.csv"
        key2 = f"predict/{team2}.csv"

        t1_bytes = self.s3.download_csv(key1)
        t2_bytes = self.s3.download_csv(key2)
        if t1_bytes is None or t2_bytes is None:
            return None

        t1 = pd.read_csv(io.BytesIO(t1_bytes))
        t2 = pd.read_csv(io.BytesIO(t2_bytes))

        # Standardize basic formatting (match training cleaning)
        for df in (t1, t2):
            df.columns = (
                df.columns.str.strip()
                .str.replace("\ufeff", "", regex=False)
                .str.replace(" ", "", regex=False)
            )
            df["GameId"] = df["GameId"].astype(str).str.zfill(10)
            df.sort_values("Date", inplace=True)

        # Compute metrics needed for prediction cleaning
        for df in (t1, t2):
            df["OffRtg"] = (df["Points"] / df["OffPoss"]) * 100
            # DefRtg should use previous team game points (shift per team)
            df["DefRtg"] = (df.groupby("team")["Points"].shift(1) / df["DefPoss"]) * 100
            df["NetRtg"] = df["OffRtg"] - df["DefRtg"]
            df["EfgDiff"] = df.groupby("team")["EfgPct"].diff().fillna(0)
            df["TsDiff"] = df.groupby("team")["TsPct"].diff().fillna(0)

            all_cols = self.base_features + self.advanced_features
            for col in all_cols:
                if col in df.columns:
                    df[f"{col}_avg"] = (
                        df.groupby("team")[col]
                        .transform(lambda x: x.shift(1).rolling(self.window_size, min_periods=1).mean())
                    )

        # Use only the most recent row for each team
        t1_latest = t1.tail(1)
        t2_latest = t2.tail(1)

        # Combine rows and align opponents
        merged = pd.concat(
            [
                t1_latest.assign(Opponent=t2_latest["team"].values[0]),
                t2_latest.assign(Opponent=t1_latest["team"].values[0]),
            ],
            ignore_index=True,
        )

        opp_cols = [f"{c}_avg" for c in self.base_features + self.advanced_features]
        opp_df = merged[["team"] + opp_cols].copy()
        opp_df.columns = ["Opponent"] + [f"Opp_{c}" for c in opp_cols]

        merged = merged.merge(opp_df, on="Opponent", how="left")

        # Compute feature differences
        for col in self.base_features + self.advanced_features:
            left = f"{col}_avg"
            right = f"Opp_{col}_avg"
            if left in merged.columns and right in merged.columns:
                merged[f"{col}_diff"] = merged[left] - merged[right]

        merged["IsHome"] = 1
        merged["HomeAdvantage"] = 1

        out_key = f"predict/clean/{team1}vs{team2}.csv"
        csv_bytes = merged.to_csv(index=False).encode("utf-8")

        if upload:
            self.s3.upload_csv_bytes(csv_bytes, out_key)

        return merged
