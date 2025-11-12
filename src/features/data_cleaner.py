import os
import sys
import io
import pandas as pd
import requests
from src.utils.exception import CustomException
from supabase import create_client
from dotenv import load_dotenv


class DataCleaner:
    """
    Cleans NBA training data for the prediction pipeline.

    - Adds metrics such as OffRtg, DefRtg, NetRtg, EfgDiff, TsDiff, Rebounds, Steals, Blocks
    - Optionally adds home/away information using the pbpstats API
    - Computes rolling averages, opponent stats, and feature differentials
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
        self.supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY"),
        )
        self.bucket_name = "modelData"

    def _fetch_home_away_map(self, seasons):
        """
        Fetch home/away information for all games in the given seasons
        using the pbpstats API.
        """
        mapping = {}
        print(f"Fetching home/away info from pbpstats for {len(seasons)} season(s)...")

        for season in seasons:
            try:
                url = "https://api.pbpstats.com/get-games/nba"
                params = {"Season": season, "SeasonType": "Regular Season"}
                resp = requests.get(url, params=params, timeout=10).json()

                for g in resp.get("results", []):
                    mapping[g["GameId"]] = {
                        "HomeTeam": g["HomeTeamAbbreviation"],
                        "AwayTeam": g["AwayTeamAbbreviation"],
                        "HomePoints": g["HomePoints"],
                        "AwayPoints": g["AwayPoints"],
                    }

            except Exception as e:
                print(f"Failed to fetch games for {season}: {e}")
                continue

        print(f"Retrieved home/away data for {len(mapping)} total games.")
        return mapping

    def clean_training_data(self, key: str = "training_data.csv", upload: bool = True):
        """
        Cleans combined training data and optionally uploads the result.

        Steps:
        - Load raw training data from Supabase
        - Normalize columns and basic structure
        - Attach home/away flags
        - Compute advanced efficiency metrics
        - Compute rolling averages and opponent stats
        - Build final feature set for model training
        """
        try:
            print(f"Downloading {key} from Supabase bucket '{self.bucket_name}'...")
            res = self.supabase.storage.from_(self.bucket_name).download(key)
            data = pd.read_csv(io.BytesIO(res))

            # Normalize column names and sort
            data.columns = (
                data.columns.str.strip()
                .str.replace("\ufeff", "", regex=False)
                .str.replace(" ", "", regex=False)
            )
            data = data.sort_values(["team", "Date"]).copy()

            # Normalize GameId format (needed for home/away matching)
            data["GameId"] = data["GameId"].astype(str).str.zfill(10)

            required_cols = [
                "team",
                "GameId",
                "Date",
                "Opponent",
                "Points",
                "GamesPlayed",
                "DefPoss",
                "OffPoss",
                "Fg3Pct",
                "Fg2Pct",
                "TsPct",
                "EfgPct",
                "Rebounds",
                "Steals",
                "Blocks",
            ]
            missing = [c for c in required_cols if c not in data.columns]
            if missing:
                raise CustomException(f"Missing required columns: {missing}", sys)

            # Home/away detection
            seasons = data["season"].unique().tolist()
            home_away_map = self._fetch_home_away_map(seasons)

            data["IsHome"] = data.apply(
                lambda row: 1
                if home_away_map.get(row["GameId"], {}).get("HomeTeam") == row["team"]
                else (
                    0
                    if home_away_map.get(row["GameId"], {}).get("AwayTeam")
                    == row["team"]
                    else None
                ),
                axis=1,
            )

            data = data.dropna(subset=["IsHome"])
            data["IsHome"] = data["IsHome"].astype(int)

            # Basic offensive/defensive ratings
            data["OffRtg"] = (data["Points"] / data["OffPoss"]) * 100
            # Initial placeholder; proper defensive rating is recalculated after the merge
            data["DefRtg"] = (data["Points"].shift(1) / data["DefPoss"]) * 100
            data["NetRtg"] = data["OffRtg"] - data["DefRtg"]

            # Shooting differential stats (within team over time)
            data["EfgDiff"] = data.groupby("team")["EfgPct"].diff().fillna(0)
            data["TsDiff"] = data.groupby("team")["TsPct"].diff().fillna(0)

            # Rolling averages for team-level trends
            all_features = self.base_features + self.advanced_features
            for col in all_features:
                if col not in data.columns:
                    print(f"Skipping missing column: {col}")
                    continue
                data[f"{col}_avg"] = (
                    data.groupby("team")[col]
                    .transform(
                        lambda x: x.shift(1)
                        .rolling(window=self.window_size, min_periods=self.window_size)
                        .mean()
                    )
                )

            data = data.dropna(
                subset=[f"{c}_avg" for c in all_features if f"{c}_avg" in data.columns]
            )

            # Opponent statistics by matching on GameId and opponent name
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

            # Game outcome and point differential
            merged["PointDifferential"] = merged["Points"] - merged["Opp_Points"]
            merged["TeamWin"] = (merged["PointDifferential"] > 0).astype(int)

            # Defensive and net rating using opponent points
            merged["DefRtg_avg"] = (merged["Opp_Points"] / merged["DefPoss_avg"]) * 100
            merged["NetRtg_avg"] = merged["OffRtg_avg"] - merged["DefRtg_avg"]

            # Shooting efficiency differentials
            if "EfgPct_avg" in merged.columns and "Opp_EfgPct_avg" in merged.columns:
                merged["EfgDiff_avg"] = (
                    merged["EfgPct_avg"] - merged["Opp_EfgPct_avg"]
                )
            if "TsPct_avg" in merged.columns and "Opp_TsPct_avg" in merged.columns:
                merged["TsDiff_avg"] = merged["TsPct_avg"] - merged["Opp_TsPct_avg"]

            # Column-wise feature differentials
            for col in all_features:
                if (
                    f"{col}_avg" in merged.columns
                    and f"Opp_{col}_avg" in merged.columns
                ):
                    merged[f"{col}_diff"] = (
                        merged[f"{col}_avg"] - merged[f"Opp_{col}_avg"]
                    )

            # Home-court advantage indicator
            merged["HomeAdvantage"] = merged["IsHome"] - merged["Opp_IsHome"]

            # Final feature set
            final_cols = (
                [
                    "team",
                    "Opponent",
                    "IsHome",
                    "HomeAdvantage",
                    "PointDifferential",
                    "TeamWin",
                ]
                + [f"{c}_avg" for c in all_features if f"{c}_avg" in merged.columns]
                + [
                    f"Opp_{c}_avg"
                    for c in all_features
                    if f"Opp_{c}_avg" in merged.columns
                ]
                + [
                    f"{c}_diff"
                    for c in all_features
                    if f"{c}_diff" in merged.columns
                ]
            )

            merged = merged[final_cols].dropna()

            # Upload cleaned dataset to Supabase
            if upload:
                csv_bytes = merged.to_csv(index=False).encode("utf-8")
                dest_key = "clean/training_data_clean.csv"
                res = self.supabase.storage.from_(self.bucket_name).upload(
                    path=dest_key,
                    file=csv_bytes,
                    file_options={"content_type": "text/csv", "upsert": "true"},
                )
                if hasattr(res, "error") and res.error:
                    raise CustomException(
                        f"Supabase upload failed: {res.error}", sys
                    )
                print(
                    f"Uploaded cleaned training data to {dest_key} "
                    f"({len(merged)} rows)"
                )

            return merged

        except Exception as e:
            raise CustomException(f"clean_training_data failed: {e}", sys)
        

    def clean_prediction_data(self, team1_key, team2_key, output_key=None, upload=True):
        """
        Cleans and merges two team prediction datasets stored in Supabase (predict/ folder).
        Unlike training, this allows predictions even if <10 games are available.
        """
        try:
            print(f"⬇️ Downloading prediction CSVs for {team1_key} and {team2_key}...")

            t1 = pd.read_csv(io.BytesIO(self.supabase.storage.from_(self.bucket_name).download(f"predict/{team1_key}.csv")))
            t2 = pd.read_csv(io.BytesIO(self.supabase.storage.from_(self.bucket_name).download(f"predict/{team2_key}.csv")))

            for df in [t1, t2]:
                df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)
                df["GameId"] = df["GameId"].astype(str).str.zfill(10)
                df.sort_values("Date", inplace=True)

            # Basic feature engineering for both teams
            for df in [t1, t2]:
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
                        .transform(lambda x: x.shift(1).rolling(window=self.window_size, min_periods=1).mean())
                    )

            # Keep the most recent row (latest game)
            t1_latest = t1.tail(1)
            t2_latest = t2.tail(1)

            # Merge for matchup prediction
            merged = pd.concat([t1_latest.assign(Opponent=t2_latest["team"].values[0]),
                                t2_latest.assign(Opponent=t1_latest["team"].values[0])],
                               ignore_index=True)

            # Add opponent-prefixed averages
            opp_cols = [f"{c}_avg" for c in self.base_features + self.advanced_features]
            opp_df = merged[["team"] + opp_cols].copy()
            opp_df.columns = ["Opponent"] + [f"Opp_{c}" for c in opp_cols]
            merged = merged.merge(opp_df, on="Opponent", how="left")

            # Add differentials
            for col in self.base_features + self.advanced_features:
                if f"{col}_avg" in merged.columns and f"Opp_{col}_avg" in merged.columns:
                    merged[f"{col}_diff"] = merged[f"{col}_avg"] - merged[f"Opp_{col}_avg"]

            merged["IsHome"] = 1  # Placeholder if needed
            merged["HomeAdvantage"] = 1

            # Save cleaned prediction matchup
            try:
                if output_key is None:
                    team1_name = str(t1_latest["team"].iloc[0]) if "team" in t1_latest.columns else str(team1_key)
                    team2_name = str(t2_latest["team"].iloc[0]) if "team" in t2_latest.columns else str(team2_key)

                    # Replace problematic characters
                    team1_name = team1_name.replace(" ", "_").replace("/", "-")
                    team2_name = team2_name.replace(" ", "_").replace("/", "-")

                    output_key = f"predict/clean/{team1_name}vs{team2_name}.csv"

                # ✅ Force cast to string to avoid numpy.str_ issues
                output_key = str(output_key)

                if upload:
                    csv_bytes = merged.to_csv(index=False).encode("utf-8")
                    res = self.supabase.storage.from_(self.bucket_name).upload(
                        path=output_key,
                        file=csv_bytes,
                        file_options={"content_type": "text/csv", "upsert": "true"},
                    )
                    print(f"✅ Uploaded cleaned prediction data → {output_key} ({len(merged)} rows)")

                return merged

            except Exception as e:
                raise CustomException(f"clean_prediction_data upload failed: {e}", sys)


        except Exception as e:
            raise CustomException(f"clean_prediction_data failed: {e}", sys)
