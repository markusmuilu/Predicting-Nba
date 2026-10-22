"""
Older national-league seasons, 2015-16 to 2021-22, from basket.fi's statistics service (the API behind
basket.fi/sarjat/tilastopalvelu, wp-json/basket/v1/pena-api). The results service the rest of the pipeline uses
(tulospalvelu.basket.fi) starts at 2022-23.

What it gives per game: date, home team ("team_a"), away team, final score, attendance, regular season or playoffs.
No quarter scores, so the margin modes that use quarters fall back to the final margin for these seasons.

Team identity: the statistics service's team id "g:team:<id>:<season>" carries the results service's team id in the
middle, so a club keeps one id across both sources and across renames (Kauhajoen Karhu and Kauhajoki are both 25).
Checked on 2022-23, which both sources cover: all 584 statistics-service games match a results-service game on date
and score, with team_a as the home team.

The II divisioona is not in this service, so its history stays at the results service's five seasons.

Run: python -m predict_nba.finland.history        (fetches once into .cache/finland/history.csv; the old seasons do not change)
"""

import time

import pandas as pd
import requests

from predict_nba.finland import CACHE_DIR

API = "https://basket.fi/wp-json/basket/v1/pena-api/api/v1"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"}
COMPETITIONS = {"KL": "f874dc00-e92f-5090-beb3-e47826cda185",
                "I-A": "4de55694-6a4d-557d-becd-0a22659d1a63",
                "I-B": "b04224ac-b474-5011-998a-119aaa628942"}
SEASONS = [f"{y}-{y + 1}" for y in range(2015, 2022)]
OUT = CACHE_DIR / "history.csv"


def get(path, **params):
    for attempt in range(4):
        try:
            r = requests.get(f"{API}{path}", params=params, headers=UA, timeout=60)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"statistics service: {path} failed")


def team_id(canonical, name):
    parts = (canonical or "").split(":")
    return parts[2] if len(parts) >= 3 and parts[2] else f"name:{name}"


def fetch():
    rows = []
    for league, cid in COMPETITIONS.items():
        seasons = {s["season_name"]: s["season_public_id"] for s in get(f"/competitions/{cid}/seasons")["seasons"]}
        for season in SEASONS:
            if season not in seasons:
                continue
            games = get(f"/competitions/{cid}/seasons/{seasons[season]}/games", limit=1000)["games"]
            for g in games:
                if g.get("phase") not in ("regular", "playoff") or g.get("score_a") is None or g.get("score_b") is None:
                    continue
                if not g.get("time"):
                    continue
                when = pd.Timestamp(g["time"]).tz_convert("Europe/Helsinki")
                rows.append({
                    "season": season, "league": league, "region": "", "group": g.get("group_name_fi") or g.get("phase"),
                    "match_id": f"pena:{g['game_public_id']}", "date": when.strftime("%Y-%m-%d"), "time": when.strftime("%H:%M"),
                    "home_id": team_id(g.get("team_a_canonical_id"), g.get("team_a_name_fi")), "home": g.get("team_a_name_fi"),
                    "away_id": team_id(g.get("team_b_canonical_id"), g.get("team_b_name_fi")), "away": g.get("team_b_name_fi"),
                    "home_pts": g["score_a"], "away_pts": g["score_b"], "attendance": g.get("attendance"),
                    "finished": True, "source": "statistics service",
                })
            print(season, league, sum(1 for r in rows if r["season"] == season and r["league"] == league), "games", flush=True)
            time.sleep(1)
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(df)} games")
    return df


def load():
    """The older seasons, fetched once and cached."""
    if not OUT.exists():
        fetch()
    return pd.read_csv(OUT, dtype={"home_id": str, "away_id": str, "match_id": str})


if __name__ == "__main__":
    fetch()
