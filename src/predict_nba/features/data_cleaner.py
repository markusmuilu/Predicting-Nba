"""
Cleans NBA training and prediction datasets for the model pipeline.

Includes:
- Efficiency metrics (OffRtg, DefRtg, NetRtg, EfgDiff, TsDiff)
- Rolling averages per team
- Opponent feature alignment
- Feature differentials
- Home/away detection using PBPStats API
"""

import io
import os
import sys

import pandas as pd
import requests
from dotenv import load_dotenv
from supabase import create_client

from predict_nba.utils.exception import CustomException
from predict_nba.utils.logger import logger


class DataCleaner:
    """
    Cleans NBA training and prediction data using rolling averages and derived features.
    """

    def __init__(self, window_size: int = 10):
        self.window_size = window_size

        self.base_features = ["OffPoss", "DefPoss", "Pace", "Fg3Pct", "Fg2Pct", "TsPct"]
        self.advanced_features = [
            "OffRtg",
            "DefRtg",
            "NetRtg",
            "EfgDiff",
            "TsDiff",
            "Rebounds",
            "Steals",
            "Blocks",
        ]

        load_dotenv()
        try:
            self.supabase = create_client(
                os.getenv("SUPABASE_URL"),
                os.getenv("SUPABASE_KEY"),
            )
            self.bucket_name = "modelData"
        except Exception as e:
            CustomException(f"DataCleaner initialization failed: {e}", sys)
            self.supabase = None

    def _fetch_home_away_map(self, seasons):
        """
        Fetches home/away game data from PBPStats for all specified seasons.
        """
        mapping = {}
        logger.info(f"Fetching home/away info for seasons: {seasons}")

        for season in seasons:
            try:
                url = "https://api.pbpstats.com/get-games/nba"
                params = {"Season": season, "SeasonType": "Regular Season"}

                resp = requests.get(url, params=params, timeout=10).json()
                for g in resp.get("results", []):
                    mapping[g["GameId"]] = {
                        "HomeTeam": g.get("HomeTeamAbbreviation"),
                        "AwayTeam": g.get("AwayTeamAbbreviation"),
                        "HomePoints": g.get("HomePoints"),
                        "AwayPoints": g.get("AwayPoints"),
                    }

            except Exception as e:
                CustomException(f"Failed to fetch home/away data for {season}: {e}", sys)
                continue

        logger.info(f"Retrieved home/away data for {len(mapping)} games.")
        return mapping

    def clean_training_data(self, key: str = "training_data.csv", upload: bool = True):
        """
        Cleans the combined raw training dataset:
        - Loads from Supabase
        - Cleans columns
        - Computes efficiency metrics
        - Computes rolling averages
        - Builds full team vs opponent feature set
        """
        if self.supabase is None:
            CustomException("Supabase client not initialized.", sys)
            return None

        try:
            logger.info(f"Downloading {key} from Supabase bucket '{self.bucket_name}'...")
            res = self.supabase.storage.from_(self.bucket_name).download(key)
            data = pd.read_csv(io.BytesIO(res))

            data.columns = (
                data.columns.str.strip()
                .str.replace("\ufeff", "", regex=False)
                .str.replace(" ", "", regex=False)
            )
            data = data.sort_values(["team", "Date"]).copy()
            data["GameId"] = data["GameId"].astype(str).str.zfill(10)

            required_cols = [
                "team", "GameId", "Date", "Opponent", "Points",
                "GamesPlayed", "DefPoss", "OffPoss",
                "Fg3Pct", "Fg2Pct", "TsPct", "EfgPct",
                "Rebounds", "Steals", "Blocks",
            ]
            missing = [c for c in required_cols if c not in data.columns]
            if missing:
                CustomException(f"Missing required columns: {missing}", sys)
                return None

            seasons = data["season"].unique().tolist()
            home_away_map = self._fetch_home_away_map(seasons)

            data["IsHome"] = data.apply(
                lambda row: 1
                if home_away_map.get(row["GameId"], {}).get("HomeTeam") == row["team"]
                else (
                    0
                    if home_away_map.get(row["GameId"], {}).get("AwayTeam") == row["team"]
                    else None
                ),
                axis=1,
            )

            data = data.dropna(subset=["IsHome"])
            data["IsHome"] = data["IsHome"].astype(int)

            data["OffRtg"] = (data["Points"] / data["OffPoss"]) * 100
            data["DefRtg"] = (data["Points"].shift(1) / data["DefPoss"]) * 100
            data["NetRtg"] = data["OffRtg"] - data["DefRtg"]

            data["EfgDiff"] = data.groupby("team")["EfgPct"].diff().fillna(0)
            data["TsDiff"] = data.groupby("team")["TsPct"].diff().fillna(0)

            all_features = self.base_features + self.advanced_features
            for col in all_features:
                if col not in data.columns:
                    logger.warning(f"Skipping missing column: {col}")
                    continue

                data[f"{col}_avg"] = (
                    data.groupby("team")[col]
                    .transform(lambda x: x.shift(1).rolling(self.window_size, min_periods=self.window_size).mean())
                )

            data = data.dropna(subset=[f"{c}_avg" for c in all_features if f"{c}_avg" in data.columns])

            opp_cols = [f"{c}_avg" for c in all_features]
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

            merged["PointDifferential"] = merged["Points"] - merged["Opp_Points"]
            merged["TeamWin"] = (merged["PointDifferential"] > 0).astype(int)

            merged["DefRtg_avg"] = (merged["Opp_Points"] / merged["DefPoss_avg"]) * 100
            merged["NetRtg_avg"] = merged["OffRtg_avg"] - merged["DefRtg_avg"]

            if "EfgPct_avg" in merged.columns and "Opp_EfgPct_avg" in merged.columns:
                merged["EfgDiff_avg"] = merged["EfgPct_avg"] - merged["Opp_EfgPct_avg"]
            if "TsPct_avg" in merged.columns and "Opp_TsPct_avg" in merged.columns:
                merged["TsDiff_avg"] = merged["TsPct_avg"] - merged["Opp_TsPct_avg"]

            for col in all_features:
                if f"{col}_avg" in merged.columns and f"Opp_{col}_avg" in merged.columns:
                    merged[f"{col}_diff"] = merged[f"{col}_avg"] - merged[f"Opp_{col}_avg"]

            merged["HomeAdvantage"] = merged["IsHome"] - merged["Opp_IsHome"]

            final_cols = (
                ["team", "Opponent", "IsHome", "HomeAdvantage", "PointDifferential", "TeamWin"]
                + [f"{c}_avg" for c in all_features if f"{c}_avg" in merged.columns]
                + [f"Opp_{c}_avg" for c in all_features if f"Opp_{c}_avg" in merged.columns]
                + [f"{c}_diff" for c in all_features if f"{c}_diff" in merged.columns]
            )

            merged = merged[final_cols].dropna()

            if upload:
                csv_bytes = merged.to_csv(index=False).encode("utf-8")
                dest_key = "clean/training_data_clean.csv"

                result = self.supabase.storage.from_(self.bucket_name).upload(
                    path=dest_key,
                    file=csv_bytes,
                    file_options={"content_type": "text/csv", "upsert": "true"},
                )

                if hasattr(result, "error") and result.error:
                    CustomException(f"Supabase upload failed: {result.error}", sys)
                else:
                    logger.info(f"Uploaded cleaned training data → {dest_key} ({len(merged)} rows)")

            return merged

        except Exception as e:
            CustomException(f"clean_training_data failed: {e}", sys)
            return None

    def clean_prediction_data(self, team1_key, team2_key, output_key=None, upload=True):
        """
        Cleans and merges prediction datasets for two teams.
        Produces a single-row matchup file for model inference.
        """
        try:
            logger.info(f"Downloading prediction CSVs for {team1_key} and {team2_key}...")

            t1_bytes = self.supabase.storage.from_(self.bucket_name).download(f"predict/{team1_key}.csv")
            t2_bytes = self.supabase.storage.from_(self.bucket_name).download(f"predict/{team2_key}.csv")

            t1 = pd.read_csv(io.BytesIO(t1_bytes))
            t2 = pd.read_csv(io.BytesIO(t2_bytes))

            for df in (t1, t2):
                df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)
                df["GameId"] = df["GameId"].astype(str).str.zfill(10)
                df.sort_values("Date", inplace=True)

            for df in (t1, t2):
                df["OffRtg"] = (df["Points"] / df["OffPoss"]) * 100
                df["DefRtg"] = (df["Points"].shift(1) / df["DefPoss"]) * 100
                df["NetRtg"] = df["OffRtg"] - df["DefRtg"]
                df["EfgDiff"] = df["EfgPct"].diff().fillna(0)
                df["TsDiff"] = df["TsPct"].diff().fillna(0)

                for col in self.base_features + self.advanced_features:
                    if col not in df.columns:
                        continue
                    df[f"{col}_avg"] = (
                        df.groupby("team")[col]
                        .transform(lambda x: x.shift(1).rolling(self.window_size, min_periods=1).mean())
                    )

            t1_latest = t1.tail(1)
            t2_latest = t2.tail(1)

            merged = pd.concat(
                [
                    t1_latest.assign(Opponent=t2_latest["team"].values[0]),
                    t2_latest.assign(Opponent=t1_latest["team"].values[0]),
                ],
                ignore_index=True
            )

            opp_cols = [f"{c}_avg" for c in self.base_features + self.advanced_features]
            opp_df = merged[["team"] + opp_cols].copy()
            opp_df.columns = ["Opponent"] + [f"Opp_{c}" for c in opp_cols]

            merged = merged.merge(opp_df, on="Opponent", how="left")

            for col in self.base_features + self.advanced_features:
                if f"{col}_avg" in merged.columns and f"Opp_{col}_avg" in merged.columns:
                    merged[f"{col}_diff"] = merged[f"{col}_avg"] - merged[f"Opp_{col}_avg"]

            merged["IsHome"] = 1
            merged["HomeAdvantage"] = 1

            try:
                if output_key is None:
                    name1 = str(t1_latest["team"].iloc[0]).replace(" ", "_")
                    name2 = str(t2_latest["team"].iloc[0]).replace(" ", "_")
                    output_key = f"predict/clean/{name1}vs{name2}.csv"

                if upload:
                    csv_bytes = merged.to_csv(index=False).encode("utf-8")
                    result = self.supabase.storage.from_(self.bucket_name).upload(
                        path=str(output_key),
                        file=csv_bytes,
                        file_options={"content_type": "text/csv", "upsert": "true"},
                    )

                    if hasattr(result, "error") and result.error:
                        CustomException(f"Prediction upload failed: {result.error}", sys)
                    else:
                        logger.info(f"Uploaded cleaned prediction data → {output_key}")

                return merged

            except Exception as e:
                CustomException(f"clean_prediction_data upload failed: {e}", sys)
                return None

        except Exception as e:
            CustomException(f"clean_prediction_data failed: {e}", sys)
            return None
