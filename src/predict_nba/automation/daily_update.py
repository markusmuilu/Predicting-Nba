"""
Daily update logic for finished games.

Uses:
- ESPN scoreboard to check which games have finished.
- current_predictions JSON from S3 to know which games were predicted.
- prediction_history JSON to record resolved results.
"""

import sys
from datetime import datetime

import requests

from predict_nba.automation.history_manager import HistoryManager
from predict_nba.utils.exception import CustomException
from predict_nba.utils.logger import logger


def update_predictions():
    """
    Resolve finished games stored in current_predictions using ESPN.

    For each finished game:
    - append to prediction_history
    - remove from current_predictions

    Returns:
        list[dict]: resolved game entries (or [] if nothing was updated).
    """
    history = HistoryManager()
    if history.s3 is None:
        CustomException("S3 client not initialized in HistoryManager.", sys)
        return None

    try:
        rows = history.load_current_predictions()
        if not rows:
            logger.info("No current_predictions found, nothing to update.")
            return []

        #Check for NO_GAMES_TODAY placeholder
        if rows and rows[0].get("team") == "NO_GAMES_TODAY":
            logger.info("Skipping update — previous day had NO_GAMES_TODAY entry.")

            # Remove placeholder from current predictions:
            history.save_current_predictions([])
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

            # Build map of ESPN event id → final scores for completed games
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

                # prediction is from home team's perspective
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

                updated_rows.append(entry)

        if not updated_rows:
            logger.info("No finished games found to update.")
            return []

        # Append to history and remove from current_predictions
        history.append_history(updated_rows)

        resolved_ids = {e["gameId"] for e in updated_rows if "gameId" in e}
        remaining_rows = [r for r in rows if r.get("gameId") not in resolved_ids]
        history.save_current_predictions(remaining_rows)

        logger.info(
            f"Updated {len(updated_rows)} finished games. "
            f"{len(remaining_rows)} games remain in current_predictions."
        )
        return updated_rows

    except Exception as e:
        CustomException(f"update_predictions failed: {e}", sys)
        return None
