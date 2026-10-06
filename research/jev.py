"""
Part 2: Jev (TypeSafe AI) as a second predictor.

Jev is not trained here. It gets a description of the game as it stood before
tip-off and one yes/no question, "does the home team win?", and returns a
probability (a "noul" answer, see docs.typesafe.ai/primitives/noul).

What goes in the state is the same pre-game information the models get, and
nothing after tip-off: season record and last-10 team ratings going into the
game (from the production cleaner's columns), rest and back-to-backs, and each
team's eight players with the most expected minutes, with their pre-game
rolling averages. Rosters come from "played in that game", the same optimistic
availability the player model uses.

Two runs per game:
- named: real team abbreviations and player names
- anonymised: "Home team"/"Away team" and "Home player 1".. instead
A model that has read last season's results in pretraining can recall them
from names. If named beats anonymised clearly, that gap is memory, not skill,
and the anonymised number is the honest one.

Budget: stops before total input tokens pass JEV_TOKEN_BUDGET (50M). Every
response is cached in data/raw/jev/, so a rerun costs nothing for games
already answered. The API key is read from JEV_API_KEY and never printed,
logged or written anywhere.

Run: python -m research.jev [previous|actual] [limit]   (does nothing without a key)
"""

import json
import os
import time

import pandas as pd
import requests

from concurrent.futures import ThreadPoolExecutor

from dotenv import dotenv_values

from research.config import PROCESSED_DIR, RAW_DIR, ROOT, RESULTS_DIR, TEST_SEASON, VAL_SEASON
from research.player_features import load_games, player_rows, previous_game_ids, team_context

API_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
JEV_TOKEN_BUDGET = 50_000_000
TOP_PLAYERS = 8
CACHE = RAW_DIR / "jev"
QUESTION = {
    "home_wins": {
        "type": "noul",
        "instructions": "Using only the pre-game information in the state, will the home team win this NBA game?",
        "criteria": {"true": "The home team wins", "false": "The away team wins"},
    }
}


def player_lines(rows, anonymise, side):
    out = []
    for k, r in enumerate(rows.itertuples(), start=1):
        out.append({
            "player": f"{side} player {k}" if anonymise else r.PLAYER_NAME,
            "expected_minutes": round(r.exp_min, 1),
            "last10_points": round(r.r10_PTS, 1),
            "last10_rebounds": round(r.r10_REB, 1),
            "last10_assists": round(r.r10_AST, 1),
            "last10_plus_minus": round(r.r10_PLUS_MINUS, 1),
            "season_games_before": int(r.season_games_before),
        })
    return out


def team_block(tf_row, prefix, ctx, players, anonymise, side, abbr):
    stats = {}
    if tf_row is not None:
        stats = {
            "season_wins": int(tf_row[f"{prefix}SeasonWins"]),
            "season_losses": int(tf_row[f"{prefix}SeasonLosses"]),
            "last10_offensive_rating": round(tf_row[f"{prefix}OffRtg_avg"], 1),
            "last10_defensive_rating": round(tf_row[f"{prefix}DefRtg_avg"], 1),
            "last10_pace": round(tf_row[f"{prefix}Pace_avg"], 1),
        }
    return {
        "team": f"{side} team" if anonymise else abbr,
        **stats,
        "days_rest": int(round(ctx["rest_days"] * 7)),
        "back_to_back": bool(ctx["back_to_back"]),
        "players": player_lines(players, anonymise, side),
    }


def find_key():
    """
    The key comes from the environment (JEV_API_KEY or JEVS_API_KEY) or from the repo's
    .env file. dotenv_values reads the file without touching os.environ. The value is
    only ever passed to the Authorization header.
    """
    env = dotenv_values(ROOT / ".env")
    for name in ("JEV_API_KEY", "JEVS_API_KEY"):
        key = os.environ.get(name) or env.get(name)
        if key:
            return key
    return None


def build_states(games, anonymise, roster_mode):
    """
    Yield (game_id, state) for every game given.
    roster_mode "previous": the players who played in the team's previous game (known before
    tip-off, the fair setting). "actual": the players who played in this game (optimistic).
    """
    rows = player_rows()
    rows["exp_min"] = rows["exp_min"].fillna(rows.loc[rows.no_history == 1, "MIN"].median())
    by_team_game = {k: v.nlargest(TOP_PLAYERS, "exp_min") for k, v in rows.groupby(["GAME_ID", "TEAM_ABBREVIATION"])}
    prev_ids = previous_game_ids(load_games())
    ctx = team_context(games)

    def roster(game_id, team):
        source = game_id
        if roster_mode == "previous" and prev_ids[(game_id, team)] is not None:
            source = prev_ids[(game_id, team)]
        return by_team_game[(source, team)]
    tf = pd.read_csv(PROCESSED_DIR / "team_features.csv")
    tf = {(r["Date"], r["home"], r["away"]): r for _, r in tf.iterrows()}

    for g in games.itertuples():
        tf_row = tf.get((g.date, g.home, g.away))
        state = {
            "sport": "NBA regular season game",
            "date": g.date if not anonymise else None,
            "home": team_block(tf_row, "", ctx.loc[(g.game_id, g.home)], roster(g.game_id, g.home), anonymise, "Home", g.home),
            "away": team_block(tf_row, "Opp_", ctx.loc[(g.game_id, g.away)], roster(g.game_id, g.away), anonymise, "Away", g.away),
        }
        if anonymise:
            state.pop("date")
        yield g.game_id, state


def ask(session, key, state):
    for attempt in range(6):
        resp = session.post(API_URL, json={"state": state, "model": MODEL, "questions": QUESTION},
                            headers={"Authorization": f"Bearer {key}"}, timeout=30)
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
        body = resp.json()
        return float(body["answers"]["home_wins"]["noul"]), body.get("usage", {}), body.get("model")
    raise RuntimeError(f"Jev request kept failing (last status {resp.status_code})")


def run(roster_mode="previous", limit=None, workers=8):
    """limit: only the first N games per variant (for a small test before the full run)."""
    key = find_key()
    if not key:
        print("No Jev API key (JEV_API_KEY / JEVS_API_KEY in the environment or .env); skipping Part 2.")
        return None

    CACHE.mkdir(exist_ok=True)
    games = load_games()
    games = games[games.season.isin([VAL_SEASON, TEST_SEASON])]
    if limit:
        games = games.head(limit)
    # Totals include cached responses, so the log is the total spend across every run.
    usage = {"input_tokens": 0, "output_tokens": 0, "requests": 0, "cached": 0, "failed": 0, "models": set()}
    session = requests.Session()

    def fetch(item):
        game_id, state, path = item
        if path.exists():
            return game_id, json.loads(path.read_text()), True
        if usage["input_tokens"] >= JEV_TOKEN_BUDGET:
            return game_id, None, False
        try:
            p, u, model = ask(session, key, state)
        except Exception as e:                      # never include request headers in the message
            print(f"  game {game_id} failed: {type(e).__name__}")
            return game_id, None, False
        rec = {"p_home": p, "usage": u, "model": model}
        path.write_text(json.dumps(rec))
        return game_id, rec, False

    for variant in ("named", "anonymised"):
        items = [(gid, state, CACHE / f"{variant}_{roster_mode}_{gid}.json")
                 for gid, state in build_states(games, anonymise=(variant == "anonymised"), roster_mode=roster_mode)]
        out = []
        with ThreadPoolExecutor(workers) as pool:
            for game_id, rec, was_cached in pool.map(fetch, items):
                if rec is None:
                    usage["failed"] += 1
                    continue
                usage["cached" if was_cached else "requests"] += 1
                usage["input_tokens"] += int(rec["usage"].get("input_tokens", 0))
                usage["output_tokens"] += int(rec["usage"].get("output_tokens", 0))
                usage["models"].add(rec.get("model"))
                out.append({"game_id": game_id, "p_home": rec["p_home"]})
        print(f"{variant}: {len(out)}/{len(items)} games answered")

        pred = games.merge(pd.DataFrame(out), on="game_id")
        suffix = "" if limit is None else "_partial"
        pred[["game_id", "date", "season", "home", "away", "home_win", "p_home"]]             .to_csv(RESULTS_DIR / f"pred_jev_{variant}_{roster_mode}{suffix}.csv", index=False)

    usage["models"] = sorted(m for m in usage["models"] if m)
    usage["roster_mode"] = roster_mode
    if limit is None:
        (RESULTS_DIR / f"jev_usage_{roster_mode}.json").write_text(json.dumps(usage, indent=2))
    print(f"Jev: {usage['requests']} new requests, {usage['cached']} cached, {usage['failed']} failed, "
          f"{usage['input_tokens']:,} input tokens in total (cached included)")
    return usage


if __name__ == "__main__":
    import sys
    run(roster_mode=sys.argv[1] if len(sys.argv) > 1 else "previous",
        limit=int(sys.argv[2]) if len(sys.argv) > 2 else None)
