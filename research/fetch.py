"""
Cached downloads for the research code.

Two sources:
- PBPStats, the same API the production pipeline uses, for the team game logs
  that the logistic regression is built from.
- nba_api (stats.nba.com) LeagueGameLog, for one row per player per game.

Every response is written to data/raw/ the first time and read from there after,
so re-running an experiment never hits the network. Both APIs throttle or block
aggressive clients, so there is a pause after every real request.

Run as a script to fill the cache:  python -m research.fetch
"""

import json
import time

import pandas as pd
import requests
from nba_api.stats.endpoints import leaguegamelog
from nba_api.stats.static import teams as nba_teams

from research.config import EXTRA_SEASONS, RAW_DIR, SEASONS

PBP_URL = "https://api.pbpstats.com"
PBP_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/121.0 Safari/537.36"}
PBP_PAUSE = 2.0      # seconds after each PBPStats request
NBA_API_PAUSE = 3.0  # stats.nba.com is stricter


def _get_json(url, params, retries=5):
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=PBP_HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            wait = 20 * (attempt + 1)
            print(f"  request failed ({e}), retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Giving up on {url} {params}")


def team_list():
    """The 30 teams as (abbreviation, NBA team id). PBPStats uses the same ids."""
    return sorted((t["abbreviation"], t["id"]) for t in nba_teams.get_teams())


def pbpstats_team_logs(season):
    """All team game logs for a season, in the raw format production collects."""
    path = RAW_DIR / f"pbpstats_team_logs_{season}.csv"
    if path.exists():
        return pd.read_csv(path)

    frames = []
    for abbr, team_id in team_list():
        print(f"PBPStats team logs {season} {abbr}")
        data = _get_json(
            f"{PBP_URL}/get-game-logs/nba",
            {"Season": season, "SeasonType": "Regular Season", "EntityType": "Team", "EntityId": team_id},
        )
        df = pd.DataFrame(data.get("multi_row_table_data", []))
        # Same post-processing as DataCollector.collect_training_data
        df = df.sort_values("Date")
        df["GamesPlayed"] = range(1, len(df) + 1)
        df["team"] = abbr
        df["season"] = season
        frames.append(df)
        time.sleep(PBP_PAUSE)

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(path, index=False)
    return out


def pbpstats_games(season):
    """GameId -> home/away teams and points, as DataCleaner._fetch_home_away_map uses it."""
    path = RAW_DIR / f"pbpstats_games_{season}.json"
    if path.exists():
        return json.loads(path.read_text())

    print(f"PBPStats games {season}")
    data = _get_json(f"{PBP_URL}/get-games/nba", {"Season": season, "SeasonType": "Regular Season"})
    results = data.get("results", [])
    path.write_text(json.dumps(results))
    time.sleep(PBP_PAUSE)
    return results


def nba_game_log(season, mode):
    """
    LeagueGameLog for a whole regular season in one call.
    mode "P": one row per player per game (only players who got on the floor).
    mode "T": one row per team per game.
    """
    path = RAW_DIR / f"nba_gamelog_{mode}_{season}.csv"
    if path.exists():
        return pd.read_csv(path, dtype={"GAME_ID": str})

    print(f"nba_api LeagueGameLog {mode} {season}")
    for attempt in range(5):
        try:
            df = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star="Regular Season",
                player_or_team_abbreviation=mode,
                timeout=60,
            ).get_data_frames()[0]
            break
        except Exception as e:
            wait = 30 * (attempt + 1)
            print(f"  nba_api failed ({e}), retrying in {wait}s")
            time.sleep(wait)
    else:
        raise RuntimeError(f"nba_api LeagueGameLog {mode} {season} failed")

    df.to_csv(path, index=False)
    time.sleep(NBA_API_PAUSE)
    return df


if __name__ == "__main__":
    import sys
    seasons = EXTRA_SEASONS if "--extra" in sys.argv else SEASONS
    for s in seasons:
        pbpstats_games(s)
        pbpstats_team_logs(s)
        nba_game_log(s, "T")
        nba_game_log(s, "P")
    print("Cache complete:", sorted(p.name for p in RAW_DIR.iterdir()))
