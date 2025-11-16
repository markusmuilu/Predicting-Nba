"""
Daily prediction generation.

Uses:
- ESPN schedule for today's games
- MakePrediction ML interface
- S3 JSON files for current_predictions
"""

import sys
import time
from datetime import datetime

import requests

from predict_nba.automation.espn_utils import espn_to_est_date, normalize_abbr
from predict_nba.automation.history_manager import HistoryManager
from predict_nba.pipeline.make_prediction import MakePrediction
from predict_nba.utils.exception import CustomException
from predict_nba.utils.logger import logger


def generate_new_predictions():
    """
    Fetch today's ESPN schedule and create predictions
    for games that have not started yet.

    Returns:
        list[dict]: newly created predictions (or [] if none).
    """
    history = HistoryManager()
    if history.s3 is None:
        CustomException("S3 client not initialized in HistoryManager.", sys)
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
        existing = history.load_current_predictions()
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

            new_rows.append(entry)
            time.sleep(0.3)

        if not new_rows:
            logger.info("No new predictions generated for today.")
            return []

        combined = existing + new_rows
        history.save_current_predictions(combined)

        logger.info(f"Inserted {len(new_rows)} new predictions.")
        return new_rows

    except Exception as e:
        CustomException(f"generate_new_predictions failed: {e}", sys)
        return None
