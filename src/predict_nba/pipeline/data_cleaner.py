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
from predict_nba.utils.s3_client import S3Client

pd.set_option("future.no_silent_downcasting", True)


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
                    headers = {
                        "User-Agent": (
                            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/121.0 Safari/537.36"
                        )
                    }
                    resp = requests.get(
                        f"{self.BASE_URL}/get-games/nba", 
                        headers=headers,
                        params={"Season": season, "SeasonType": "Regular Season"},
                        timeout=30,
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
                    time.sleep(attempt*60 + 60)  # wait before retry

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

        raw = self.s3.download(key)
        if raw is None:
            return None

        data = pd.read_csv(io.BytesIO(raw))

        # Standardize column formatting
        data.columns = (
            data.columns.str.strip()
            .str.replace("\ufeff", "", regex=False)
            .str.replace(" ", "", regex=False)
        )

        data = data.sort_values(["team", "season", "Date"])
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


        # Add the home and away points
        data["HomePoints"] = data["GameId"].map(lambda gid: home_away_map.get(gid, {}).get("HomePoints"))
        data["AwayPoints"] = data["GameId"].map(lambda gid: home_away_map.get(gid, {}).get("AwayPoints"))

        #Add opponent points as we already have team points as points
        data["OppPoints"] = data.apply(
            lambda r: r["AwayPoints"] if r["IsHome"] == 1 else r["HomePoints"],
            axis=1
        )

        data["PointDifferential"] = data["Points"] - data["OppPoints"]
        data["TeamWin"] = (data["PointDifferential"] > 0).astype(int)


        # Count season wins, lossses win % and is it back to back game and how many games in the last 10 days
        data = data.sort_values(["team", "season", "Date"])

        # Wins can be counted by for a row, take cumulative sum of teamwin column for that team during this season before this game 
        data["SeasonWins"] = data.groupby(["team", "season"])["TeamWin"].cumsum().shift(1).fillna(0)

        # Games played is just counts all the rows before this one for that team during this season, Pct by deviding wins by games played, losses by subtracting wins from games played
        data["SeasonGames"] = data.groupby(["team", "season"])["TeamWin"].cumcount().fillna(0)
        data["SeasonWinPct"] = data["SeasonWins"] / data["SeasonGames"].fillna(0)
        data["SeasonLosses"] = data["SeasonGames"] - data["SeasonWins"].fillna(0)

        # Back to back games can be calculated by checking the date difference between this game and previous game for a team in season
        data["Date"] = pd.to_datetime(data["Date"])
        data["Prevdate"] = data.groupby(["team", "season"])["Date"].shift(1)
        data["IsBackToBack"] = (data["Date"] - data["Prevdate"]).dt.days.eq(1).astype(int)



        # Basic rating metrics
        data["OffRtg"] = (data["Points"] / data["OffPoss"]) * 100
        data["DefRtg"] = (data["Points"] / data["DefPoss"]) * 100
        data["NetRtg"] = data["OffRtg"] - data["DefRtg"]
        data["EfgDiff"] = data.groupby("team")["EfgPct"].diff().fillna(0)
        data["TsDiff"] = data.groupby("team")["TsPct"].diff().fillna(0)

        # Rolling averages
        all_features = self.base_features + self.advanced_features
        for col in all_features:
            if col in data.columns:
                data[f"{col}_avg"] = (
                    data.groupby(["team", "season"])[col]
                    .transform(lambda x: x.shift(1).rolling(self.window_size, min_periods=1).mean())
                )
        # Only keep rows where all averages are available
        avg_cols = [f"{c}_avg" for c in all_features if f"{c}_avg" in data.columns]
        data = data.dropna(subset=avg_cols)

        # Build opponent feature set
        season_cols = ["SeasonWins", "SeasonLosses", "SeasonWinPct", "IsBackToBack"]
        opp_cols = avg_cols + season_cols
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
        
        for col in all_features:
            left = f"{col}_avg"
            right = f"Opp_{col}_avg"
            if left in merged.columns and right in merged.columns:
                merged[f"{col}_diff"] = merged[left] - merged[right]

        merged["HomeAdvantage"] = merged["IsHome"] - merged["Opp_IsHome"]
        merged = merged[merged["IsHome"] == 1]

        # Final training columns
        final_cols = (
            ["team", "Date", "Opponent", "IsHome", "HomeAdvantage", "PointDifferential", "TeamWin"]
            + season_cols
            + [f"{c}_avg" for c in all_features if f"{c}_avg" in merged.columns]
            + [f"Opp_{c}_avg" for c in all_features if f"Opp_{c}_avg" in merged.columns]
            + [f"Opp_{c}" for c in season_cols if f"Opp_{c}" in merged.columns]
            + [f"{c}_diff" for c in all_features if f"{c}_diff" in merged.columns]
        )

        final = merged[final_cols].dropna().sort_values("Date")

        if upload:
            csv_bytes = final.to_csv(index=False).encode("utf-8")
            self.s3.upload("clean/training_data_clean.csv", csv_bytes, "text/csv")

        return final

    def clean_prediction_data(self, team1, team2, upload=True):
        """
        Produce a single-row matchup dataset for prediction inference.
        Reads latest per-team CSVs from S3 and computes aligned features.
        """
        logger.info(f"Cleaning prediction data for {team1} vs {team2}")
        key1 = f"predict/{team1}.csv"
        key2 = f"predict/{team2}.csv"

        t1_bytes = self.s3.download(key1)
        t2_bytes = self.s3.download(key2)
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

        seasons = t1["season"].unique()
        home_away_map = self._fetch_home_away_map(seasons)


        for df in (t1, t2):
            # Add the ishome and home and away points
            df["IsHome"] = df["GameId"].map(lambda gid: home_away_map.get(gid, {}).get("HomeTeam") == df["team"].iloc[0]).astype(int)
            df["HomePoints"] = df["GameId"].map(lambda gid: home_away_map.get(gid, {}).get("HomePoints"))
            df["AwayPoints"] = df["GameId"].map(lambda gid: home_away_map.get(gid, {}).get("AwayPoints"))

            #Add opponent points as we already have team points as points
            df["OppPoints"] = df.apply(
                lambda r: r["AwayPoints"] if r["IsHome"] == 1 else r["HomePoints"],
                axis=1
            )

            df["PointDifferential"] = df["Points"] - df["OppPoints"]
            df["TeamWin"] = (df["PointDifferential"] > 0).astype(int)
        # Compute metrics needed for prediction cleaning
        for df in (t1, t2):
            df["OffRtg"] = (df["Points"] / df["OffPoss"]) * 100
            # DefRtg should use previous team game points (shift per team)
            df["DefRtg"] = (df["OppPoints"] / df["DefPoss"]) * 100
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

        # Season wins, losses, pct, back to back
        t1 = t1.sort_values("Date")
        t2 = t2.sort_values("Date")
        for df in (t1, t2):

            df["SeasonWins"] = df.groupby(["team", "season"])["TeamWin"].cumsum().shift(1).fillna(0)
            df["SeasonGames"] = df.groupby(["team", "season"])["TeamWin"].cumcount()
            df["SeasonWinPct"] = (df["SeasonWins"] / df["SeasonGames"].replace(0, pd.NA)).fillna(0)
            df["SeasonLosses"] = df["SeasonGames"] - df["SeasonWins"]

            df["Date"] = pd.to_datetime(df["Date"])
            df["PrevDate"] = df.groupby(["team", "season"])["Date"].shift(1)
            df["IsBackToBack"] = (df["Date"] - df["PrevDate"]).dt.days.eq(1).astype(int)


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

        season_cols = ["SeasonWins", "SeasonLosses", "SeasonWinPct", "IsBackToBack"]

        for col in season_cols:
            if col not in merged.columns:
                merged[col] = merged.apply(
                    lambda r: t1_latest[col].iloc[0] if r["team"] == t1_latest["team"].iloc[0]
                    else t2_latest[col].iloc[0],
                    axis=1
                )

        opp_cols = (
            [f"{c}_avg" for c in self.base_features + self.advanced_features]
            + season_cols
        )

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
            self.s3.upload(out_key, csv_bytes, "text/csv")

        return merged

