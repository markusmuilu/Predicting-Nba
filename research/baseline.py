"""
Step 1: the current logistic regression, measured on the fixed split.

The point is to measure the model that is actually in production, not a
re-implementation of it. So this imports the production DataCleaner and the
production feature list unchanged, and only swaps out the two things that
reach the network:
- R2 storage -> an in-memory store fed from the local cache
- the PBPStats home/away lookup -> the cached games file

What does change is the split. Production trains on everything before
2024-08-01 and checks one hold-out season. Here it is train 2020-21..2023-24,
validate 2024-25, test 2025-26, the same split every other model uses. The
model settings (StandardScaler, LogisticRegression C=1, lbfgs) are production's.

Output: data/processed/team_features.csv (one row per game, home team view)
and research/results/pred_logreg.csv (home win probability per game).
"""

import io
import sys
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from research.config import PROCESSED_DIR, RESULTS_DIR, ROOT, SEASONS, TEST_SEASON, TRAIN_SEASONS, VAL_SEASON
from research.fetch import pbpstats_games, pbpstats_team_logs

sys.path.insert(0, str(ROOT / "src"))


class LocalStore:
    """Stands in for S3Client: download() returns bytes, upload() is ignored."""

    def __init__(self, files=None):
        self.files = files or {}

    def download(self, key):
        return self.files.get(key)

    def upload(self, key, content, content_type=None):
        self.files[key] = content


def season_of(date):
    """'2025-01-15' -> '2024-25'. NBA seasons start in autumn."""
    d = pd.Timestamp(date)
    start = d.year if d.month >= 7 else d.year - 1
    return f"{start}-{str(start + 1)[2:]}"


def build_team_features():
    """Run the production cleaner over all six seasons of cached raw logs."""
    with mock.patch("predict_nba.pipeline.data_cleaner.S3Client", LocalStore):
        from predict_nba.pipeline.data_cleaner import DataCleaner

        raw = pd.concat([pbpstats_team_logs(s) for s in SEASONS], ignore_index=True)
        cleaner = DataCleaner()
        cleaner.s3 = LocalStore({"training/training_data.csv": raw.to_csv(index=False).encode()})

        def cached_home_away_map(seasons):
            mapping = {}
            for s in seasons:
                for g in pbpstats_games(s):
                    mapping[g["GameId"]] = {
                        "HomeTeam": g.get("HomeTeamAbbreviation"),
                        "AwayTeam": g.get("AwayTeamAbbreviation"),
                        "HomePoints": g.get("HomePoints"),
                        "AwayPoints": g.get("AwayPoints"),
                    }
            return mapping

        cleaner._fetch_home_away_map = cached_home_away_map
        clean = cleaner.clean_training_data(upload=False)

    clean["Date"] = pd.to_datetime(clean["Date"]).dt.strftime("%Y-%m-%d")
    clean["season"] = clean["Date"].map(season_of)
    clean = clean.rename(columns={"team": "home", "Opponent": "away", "TeamWin": "home_win"})
    clean.to_csv(PROCESSED_DIR / "team_features.csv", index=False)
    return clean


def production_features():
    """The exact feature list ModelTrainer trains on."""
    with mock.patch("predict_nba.pipeline.model_trainer.S3Client", LocalStore), \
         mock.patch("predict_nba.pipeline.model_trainer.load_dotenv", lambda: None):
        from predict_nba.pipeline.model_trainer import ModelTrainer
        return ModelTrainer().features


def run():
    df = build_team_features()
    features = [f for f in production_features() if f in df.columns]

    train = df[df.season.isin(TRAIN_SEASONS)]
    scaler = StandardScaler().fit(train[features])
    model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    model.fit(scaler.transform(train[features]), train["home_win"])

    df["p_home"] = model.predict_proba(scaler.transform(df[features]))[:, 1]
    out = df[df.season.isin([VAL_SEASON, TEST_SEASON])][["Date", "season", "home", "away", "home_win", "p_home"]]
    out.to_csv(RESULTS_DIR / "pred_logreg.csv", index=False)

    print(f"Features: {len(features)}  train rows: {len(train)}")
    for s in (VAL_SEASON, TEST_SEASON):
        part = out[out.season == s]
        acc = np.mean((part.p_home >= 0.5) == (part.home_win == 1))
        print(f"{s}: {len(part)} games, accuracy {acc:.3f}")
    return out


if __name__ == "__main__":
    run()
