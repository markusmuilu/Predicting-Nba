"""
Daily prediction automation:
- Updates completed games based on ESPN scoreboard data.
- Generates new predictions for today's games.
"""

import os
import time
import sys
from datetime import datetime

import numpy as np
import pytz
import requests
from dotenv import load_dotenv
from supabase import create_client

from predict_nba.features.make_prediction import MakePrediction
from predict_nba.utils.exception import CustomException
from predict_nba.utils.logger import logger


load_dotenv()

ESPN_ABBR_FIX = {
    "UTAH": "UTA",
    "GS": "GSW",
    "NY": "NYK",
    "SA": "SAS",
    "NO": "NOP",
}


def normalize_abbr(raw_abbr: str) -> str:
    """Normalize ESPN abbreviations into expected NBA abbreviations."""
    if not raw_abbr:
        return None
    return ESPN_ABBR_FIX.get(raw_abbr.upper(), raw_abbr.upper())


def espn_to_est_date(espn_date_str: str) -> str:
    """Convert ESPN UTC datetime string to EST date."""
    utc = pytz.utc
    est = pytz.timezone("America/New_York")

    dt_utc = datetime.fromisoformat(espn_date_str.replace("Z", "+00:00"))
    dt_est = dt_utc.astimezone(est)
    return dt_est.strftime("%Y-%m-%d")


class DailyPredictor:
    """
    Handles daily prediction routines:
    - Resolves completed games and stores them in prediction_history.
    - Creates predictions for unstarted games today.
    """

    def __init__(self, bucket_name="modelData"):
        try:
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_KEY")
            self.supabase = create_client(url, key)
            self.bucket_name = bucket_name
        except Exception as e:
            CustomException(f"Failed to initialize Supabase client: {e}", sys)
            self.supabase = None

    def _clean(self, obj):
        """Convert numpy datatypes into normal Python types."""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return obj

    def update_predictions(self):
        """
        Resolves finished games stored in current_predictions by checking ESPN.
        Moves resolved games → prediction_history.
        """
        if self.supabase is None:
            CustomException("Supabase client not initialized.", sys)
            return None

        try:
            rows = self.supabase.table("current_predictions").select("*").execute().data
            if not rows:
                logger.info("No current predictions to update.")
                return None

            dates = sorted({r["date"] for r in rows})
            logger.info(f"Checking ESPN results for dates: {dates}")

            updated_rows = []

            for date_str in dates:
                espn_date = date_str.replace("-", "")
                url = (
                    "https://site.api.espn.com/apis/site/v2/sports/"
                    f"basketball/nba/scoreboard?dates={espn_date}"
                )

                logger.info(f"Fetching ESPN scoreboard for {date_str}...")
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                events = resp.json().get("events", [])

                if not events:
                    logger.warning(f"No ESPN events found for {date_str}")
                    continue

                game_results = {}

                for ev in events:
                    event_id = ev["id"]
                    comp = ev.get("competitions", [{}])[0]
                    status = comp.get("status", {}).get("type", {})
                    state = status.get("state")
                    completed = status.get("completed", False)

                    if state != "post" and not completed:
                        continue

                    competitors = comp.get("competitors", [])
                    if len(competitors) != 2:
                        continue

                    home = [c for c in competitors if c["homeAway"] == "home"][0]
                    away = [c for c in competitors if c["homeAway"] == "away"][0]

                    game_results[event_id] = {
                        "home_score": int(home.get("score", 0)),
                        "away_score": int(away.get("score", 0)),
                    }

                for row in rows:
                    if row["date"] != date_str:
                        continue

                    game_id = row["gameId"]
                    if game_id not in game_results:
                        continue

                    scores = game_results[game_id]
                    home_win = scores["home_score"] > scores["away_score"]
                    predicted_home_win = bool(row["prediction"])
                    correct = home_win == predicted_home_win

                    logger.info(
                        f"Resolved game {game_id}: {scores['home_score']}–{scores['away_score']} "
                        f"(correct: {correct})"
                    )

                    entry = {
                        "date": row["date"],
                        "team": row["team"],
                        "opponent": row["opponent"],
                        "prediction": predicted_home_win,
                        "confidence": float(row["confidence"]),
                        "winner": home_win,
                        "prediction_correct": correct,
                        "gameId": game_id,
                    }

                    updated_rows.append({k: self._clean(v) for k, v in entry.items()})

            if updated_rows:
                self.supabase.table("prediction_history").insert(updated_rows).execute()

                for entry in updated_rows:
                    self.supabase.table("current_predictions") \
                        .delete() \
                        .eq("gameId", entry["gameId"]) \
                        .execute()

                logger.info(f"Updated {len(updated_rows)} finished games.")
            else:
                logger.info("No finished games found.")

            return updated_rows

        except Exception as e:
            CustomException(f"update_predictions failed: {e}", sys)
            return None

    def new_predictions(self):
        """
        Fetches today's ESPN schedule and creates predictions
        for games not yet started.
        """
        if self.supabase is None:
            CustomException("Supabase client not initialized.", sys)
            return None

        try:
            today = datetime.utcnow().strftime("%Y%m%d")
            url = (
                "https://site.api.espn.com/apis/site/v2/sports/"
                f"basketball/nba/scoreboard?dates={today}"
            )

            resp = requests.get(url, timeout=10)
            resp.raise_for_status()

            events = resp.json().get("events", [])
            predictor = MakePrediction()
            new_rows = []

            for ev in events:
                comp = ev.get("competitions", [{}])[0]
                status = comp.get("status", {}).get("type", {})
                state = status.get("state")
                completed = status.get("completed", False)

                if completed or state not in (None, "pre"):
                    continue

                competitors = comp.get("competitors", [])
                if len(competitors) != 2:
                    continue

                home = [c for c in competitors if c["homeAway"] == "home"][0]
                away = [c for c in competitors if c["homeAway"] == "away"][0]

                home_abbr = normalize_abbr(home["team"].get("abbreviation"))
                away_abbr = normalize_abbr(away["team"].get("abbreviation"))

                if not home_abbr or not away_abbr:
                    logger.warning("Could not normalize abbreviations for a matchup.")
                    continue

                event_id = ev["id"]
                date_str = espn_to_est_date(ev["date"])

                result = predictor.predict(home_abbr, away_abbr)
                if result is None:
                    continue

                entry = {
                    "team": home_abbr,
                    "opponent": away_abbr,
                    "date": date_str,
                    "prediction": (result["winner"] == home_abbr),
                    "confidence": float(result["confidence"]),
                    "gameId": event_id,
                }

                entry = {k: self._clean(v) for k, v in entry.items()}
                new_rows.append(entry)
                time.sleep(0.3)

            if new_rows:
                self.supabase.table("current_predictions").insert(new_rows).execute()
                logger.info(f"Inserted {len(new_rows)} new predictions.")
            else:
                logger.info("No new predictions for today.")

            return new_rows

        except Exception as e:
            CustomException(f"new_predictions failed: {e}", sys)
            return None

