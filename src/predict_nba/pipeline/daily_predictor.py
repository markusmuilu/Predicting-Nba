"""
Daily prediction automation.

Responsibilities:
- Resolve completed games from current_predictions using ESPN scoreboard data
  and append them to prediction_history.
- Generate new predictions for today's games and store them in current_predictions.

All data is stored in S3 as JSON:
- history/prediction_history.json
- current/current_predictions.json
"""

import json
import os
import sys
import time
from datetime import datetime

import boto3
import numpy as np
import pytz
import requests
from dotenv import load_dotenv

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
    "WSH": "WAS",  
}


def normalize_abbr(raw_abbr: str) -> str:
    """Normalize ESPN team abbreviations into the project's abbreviations."""
    if not raw_abbr:
        return None
    return ESPN_ABBR_FIX.get(raw_abbr.upper(), raw_abbr.upper())


def espn_to_est_date(espn_date_str: str) -> str:
    """Convert ESPN UTC datetime string to EST date (YYYY-MM-DD)."""
    utc = pytz.utc
    est = pytz.timezone("America/New_York")

    dt_utc = datetime.fromisoformat(espn_date_str.replace("Z", "+00:00"))
    dt_est = dt_utc.astimezone(est)
    return dt_est.strftime("%Y-%m-%d")


class S3Client:
    """Minimal JSON helper for S3."""

    def __init__(self):
        bucket = os.getenv("AWS_S3_BUCKET_NAME")
        region = os.getenv("AWS_REGION")

        if not bucket:
            raise CustomException("AWS_S3_BUCKET_NAME must be set in environment.", sys)

        try:
            self.bucket = bucket
            self.s3 = boto3.client(
                "s3",
                region_name=region,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            )
        except Exception as e:
            raise CustomException(f"S3 initialization failed: {e}", sys)

    def load_json_list(self, key: str):
        """
        Load a JSON file from S3 that should contain a list.
        Returns [] if the file does not exist or is empty/invalid.
        """
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=key)
            raw = resp["Body"].read().decode("utf-8").strip()
            if not raw:
                return []
            data = json.loads(raw)
            if isinstance(data, list):
                return data
            logger.warning(f"S3 key {key} did not contain a list, returning empty list.")
            return []
        except self.s3.exceptions.NoSuchKey:
            logger.info(f"S3 key {key} not found, starting with empty list.")
            return []
        except Exception as e:
            CustomException(f"Failed to load JSON from {key}: {e}", sys)
            return []

    def save_json_list(self, data, key: str):
        """Save a list as JSON to S3."""
        try:
            payload = json.dumps(data, indent=2).encode("utf-8")
            self.s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=payload,
                ContentType="application/json",
            )
            logger.info(f"Saved {len(data)} entries to s3://{self.bucket}/{key}")
        except Exception as e:
            CustomException(f"Failed to save JSON to {key}: {e}", sys)


class DailyPredictor:
    """
    Handles daily prediction workflow:
    - update_predictions(): resolve finished games and move them from current_predictions
      to prediction_history.
    - new_predictions(): generate new predictions for today's unstarted games.
    """

    HISTORY_KEY = "history/prediction_history.json"
    CURRENT_KEY = "current/current_predictions.json"

    def __init__(self):
        try:
            self.s3 = S3Client()
        except Exception as e:
            CustomException(f"Failed to initialize S3 client for DailyPredictor: {e}", sys)
            self.s3 = None

    def _clean(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return obj

    def _load_current_predictions(self):
        """Load current_predictions JSON from S3."""
        if self.s3 is None:
            return []
        return self.s3.load_json_list(self.CURRENT_KEY)

    def _save_current_predictions(self, rows):
        """Overwrite current_predictions JSON in S3 with the given rows."""
        if self.s3 is None:
            return
        self.s3.save_json_list(rows, self.CURRENT_KEY)

    def _append_history(self, new_entries):
        """Append new entries to prediction_history JSON in S3, avoiding duplicates by gameId."""
        if self.s3 is None or not new_entries:
            return

        history = self.s3.load_json_list(self.HISTORY_KEY)
        existing_ids = {h.get("gameId") for h in history if "gameId" in h}

        to_add = [e for e in new_entries if e.get("gameId") not in existing_ids]
        history.extend(to_add)

        self.s3.save_json_list(history, self.HISTORY_KEY)

    def update_predictions(self):
        """
        Resolve finished games stored in current_predictions by checking ESPN.
        For each resolved game:
        - append to prediction_history
        - remove from current_predictions

        Safely handles:
        - empty current_predictions
        - days with no ESPN events
        - games not yet completed
        """
        if self.s3 is None:
            CustomException("S3 client not initialized.", sys)
            return None

        try:
            rows = self._load_current_predictions()
            if not rows:
                logger.info("No current_predictions found, nothing to update.")
                return []

            dates = sorted({r["date"] for r in rows if "date" in r})
            if not dates:
                logger.info("current_predictions has rows but no valid dates, skipping update.")
                return []

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

                # Build a map of ESPN event id → final scores for completed games
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

                    home = [c for c in competitors if c.get("homeAway") == "home"]
                    away = [c for c in competitors if c.get("homeAway") == "away"]
                    if not home or not away:
                        continue

                    home = home[0]
                    away = away[0]

                    try:
                        game_results[event_id] = {
                            "home_score": int(home.get("score", 0)),
                            "away_score": int(away.get("score", 0)),
                        }
                    except (TypeError, ValueError):
                        continue

                # Match current_predictions rows to finished ESPN games
                for row in rows:
                    if row.get("date") != date_str:
                        continue

                    game_id = row.get("gameId")
                    if not game_id or game_id not in game_results:
                        continue

                    scores = game_results[game_id]
                    home_win = scores["home_score"] > scores["away_score"]

                    # current_predictions stores prediction from home team's perspective
                    predicted_home_win = bool(row.get("prediction"))
                    correct = home_win == predicted_home_win

                    logger.info(
                        f"Resolved game {game_id}: "
                        f"{scores['home_score']}–{scores['away_score']} "
                        f"(correct: {correct})"
                    )

                    entry = {
                        "date": row.get("date"),
                        "team": row.get("team"),
                        "opponent": row.get("opponent"),
                        "prediction": predicted_home_win,
                        "confidence": float(row.get("confidence", 0.0)),
                        "winner": home_win,  # True if home team actually won
                        "prediction_correct": correct,
                        "gameId": game_id,
                    }

                    updated_rows.append({k: self._clean(v) for k, v in entry.items()})

            if not updated_rows:
                logger.info("No finished games found to update.")
                return []

            # Append to history and remove from current_predictions
            self._append_history(updated_rows)

            resolved_ids = {e["gameId"] for e in updated_rows if "gameId" in e}
            remaining_rows = [r for r in rows if r.get("gameId") not in resolved_ids]
            self._save_current_predictions(remaining_rows)

            logger.info(f"Updated {len(updated_rows)} finished games. "
                        f"{len(remaining_rows)} games remain in current_predictions.")
            return updated_rows

        except Exception as e:
            CustomException(f"update_predictions failed: {e}", sys)
            return None

    def new_predictions(self):
        """
        Fetch today's ESPN schedule and create predictions
        for games that have not started yet.

        Safely handles:
        - no ESPN events for today
        - all games already started or finished
        - empty or missing current_predictions.json
        """
        if self.s3 is None:
            CustomException("S3 client not initialized.", sys)
            return None

        try:
            today_str = datetime.utcnow().strftime("%Y%m%d")
            url = (
                "https://site.api.espn.com/apis/site/v2/sports/"
                f"basketball/nba/scoreboard?dates={today_str}"
            )

            logger.info(f"Fetching ESPN schedule for today ({today_str})...")
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()

            events = resp.json().get("events", [])
            if not events:
                logger.info("No ESPN events found for today. No predictions generated.")
                return []

            predictor = MakePrediction()
            existing = self._load_current_predictions()
            existing_ids = {r.get("gameId") for r in existing if "gameId" in r}

            new_rows = []

            for ev in events:
                comp = ev.get("competitions", [{}])[0]
                status = comp.get("status", {}).get("type", {})
                state = status.get("state")
                completed = status.get("completed", False)

                # Only predict pre-game matchups
                if completed or state not in (None, "pre"):
                    continue

                competitors = comp.get("competitors", [])
                if len(competitors) != 2:
                    continue

                home = [c for c in competitors if c.get("homeAway") == "home"]
                away = [c for c in competitors if c.get("homeAway") == "away"]
                if not home or not away:
                    continue

                home = home[0]
                away = away[0]

                home_abbr = normalize_abbr(home["team"].get("abbreviation"))
                away_abbr = normalize_abbr(away["team"].get("abbreviation"))

                if not home_abbr or not away_abbr:
                    logger.warning("Could not normalize abbreviations for a matchup.")
                    continue

                event_id = ev.get("id")
                if not event_id:
                    continue

                # Avoid duplicating predictions for the same game
                if event_id in existing_ids:
                    logger.info(f"Skipping event {event_id}, already in current_predictions.")
                    continue

                date_str = espn_to_est_date(ev["date"])

                result = predictor.predict(home_abbr, away_abbr)
                if result is None:
                    continue

                entry = {
                    "team": home_abbr,
                    "opponent": away_abbr,
                    "date": date_str,
                    "prediction": (result.get("winner") == home_abbr),
                    "confidence": float(result.get("confidence", 0.0)),
                    "gameId": event_id,
                }

                entry = {k: self._clean(v) for k, v in entry.items()}
                new_rows.append(entry)
                time.sleep(0.3)

            if not new_rows:
                logger.info("No new predictions generated for today.")
                return []

            # Append new predictions to any existing ones
            combined = existing + new_rows
            self._save_current_predictions(combined)

            logger.info(f"Inserted {len(new_rows)} new predictions.")
            return new_rows

        except Exception as e:
            CustomException(f"new_predictions failed: {e}", sys)
            return None
