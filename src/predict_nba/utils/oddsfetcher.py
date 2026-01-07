import requests
import os
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv

BASE_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"

load_dotenv()

API_KEY = os.getenv("ODDS_API_KEY")

class OddsFetcher:
    def fetch_odds():
        """Fetches NBA odds from the Odds API."""
        params = {
            "apiKey": API_KEY,
            "regions": "eu",   
            "markets": "h2h",
            "oddsFormat": "decimal",
            "dateFormat": "iso"
        }
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()

        events = response.json()

        local_tz = datetime.now().astimezone().tzinfo
        now_local = datetime.now().astimezone(local_tz)
        next_24h_local = now_local + timedelta(hours=24)

        rows = []

        for event in events:
            start_utc = datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00'))
            start_local = start_utc.astimezone(local_tz)

            # Filter games within the next 24 hours
            if not (now_local <= start_local <= next_24h_local):
                continue

            home = event['home_team']
            away = event['away_team']

            for bookmaker in event.get('bookmakers', []):
                if bookmaker.get('key') == 'pinnacle':
                    for market in bookmaker.get('markets', []):
                        if market['key'] == 'h2h':
                            outcomes = market['outcomes']
                            odds_dict = {outcome['name']: outcome['price'] for outcome in outcomes}
                            game_data = {
                                'home_team': home,
                                'away_team': away,
                                'home_odds': odds_dict.get(home),
                                'away_odds': odds_dict.get(away)
                            }
                            rows.append(game_data)
                            break  

        return pd.DataFrame(rows)

if __name__ == "__main__":
    odds_df = OddsFetcher.fetch_odds()
    print(odds_df)