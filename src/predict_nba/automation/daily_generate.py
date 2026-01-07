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

from predict_nba.utils.espn import espn_to_est_date, normalize_abbr
from predict_nba.automation.history_manager import HistoryManager
from predict_nba.pipeline.make_prediction import MakePrediction
from predict_nba.utils.exception import CustomException
from predict_nba.utils.logger import logger
from predict_nba.utils.oddsfetcher import OddsFetcher



TEAMS = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}



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
        try:
            existing = history.load_current_predictions()
            existing_ids = {r.get("gameId") for r in existing if "gameId" in r}
        except:
            existing = []
            existing_ids = []

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
        if new_rows != []:
            odds = OddsFetcher.fetch_odds()
            for row in new_rows:
                home_team = TEAMS.get(row["team"])
                away_team = TEAMS.get(row["opponent"])
                match = odds[
                    (odds["home_team"] == home_team) &
                    (odds["away_team"] == away_team)
                ]
                if not match.empty:
                    row["home_odds"] = match.iloc[0]["home_odds"]
                    row["away_odds"] = match.iloc[0]["away_odds"]
                else:
                    row["home_odds"] = None
                    row["away_odds"] = None

        if not new_rows and existing == []:
            logger.info("No NBA games today, creating placeholder prediction.")

            today = datetime.utcnow().strftime("%Y-%m-%d")
            #Place holder needed, as powerbi update fails with empty current_predictions
            placeholder = {
                "team": "NO_GAMES_TODAY",
                "opponent": None,
                "date": today,
                "prediction": False,
                "confidence": 100.0,
                "gameId": f"NO_GAMES_{today.replace('-', '')}",
            }

            existing = history.load_current_predictions() or []
            history.save_current_predictions(existing + [placeholder])

            return [placeholder]

        combined = existing + new_rows
        history.save_current_predictions(combined)

        logger.info(f"Inserted {len(new_rows)} new predictions.")
        return new_rows

    except Exception as e:
        CustomException(f"generate_new_predictions failed: {e}", sys)
        return None

if __name__ == "__main__":
    generate_new_predictions()