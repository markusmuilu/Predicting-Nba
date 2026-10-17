"""
The player model's inputs, computed exactly as in training (research/player_features.py on the next-season branch),
with one change: a player is identified by his normalised name, which both stats.nba.com and ESPN provide.

A team's roster for an upcoming game is the players who played in its previous game, and each player's features are
his row from that game, i.e. computed from his games before it. That is the "previous" roster mode the model was
trained on. Players listed Out right before the prediction are removed from that roster.
"""

import numpy as np
import pandas as pd

ROLL_N = 10
MAX_PLAYERS = 15
REST_CAP = 7
BOX = ["MIN", "PTS", "REB", "OREB", "AST", "STL", "BLK", "TOV", "PF",
       "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "PLUS_MINUS"]
PLAYER_FEATURES = (
    [f"r10_{c}" for c in ["MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "PLUS_MINUS"]]
    + ["sea_MIN", "sea_PTS", "sea_PLUS_MINUS"]
    + ["ts_pct", "fg3_rate", "ft_rate", "usage_per36", "pts_per36", "reb_per36", "ast_per36", "pm_per36"]
    + ["log_games_before", "season_games_before", "days_since_last", "no_history"]
)


def player_rows(p):
    """Every player-game in the box-score table with its pre-game features. p: the stored history."""
    p = p.copy()
    p["date"] = pd.to_datetime(p["GAME_DATE"])
    p = p.sort_values(["key", "date", "GAME_ID"]).reset_index(drop=True)
    p[BOX] = p[BOX].fillna(0)
    by_player = p.groupby("key", sort=False)
    by_player_season = p.groupby(["key", "season"], sort=False)

    roll = {c: by_player[c].transform(lambda x: x.shift(1).rolling(ROLL_N, min_periods=1).sum()) for c in BOX}
    n_recent = by_player["MIN"].transform(lambda x: x.shift(1).rolling(ROLL_N, min_periods=1).count())

    f = pd.DataFrame(index=p.index)
    for c in ["MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "PLUS_MINUS"]:
        f[f"r10_{c}"] = roll[c] / n_recent
    for c in ["MIN", "PTS", "PLUS_MINUS"]:
        f[f"sea_{c}"] = by_player_season[c].transform(lambda x: x.shift(1).expanding().mean())
    mins = roll["MIN"].clip(lower=1)
    shots = roll["FGA"] + 0.44 * roll["FTA"]
    f["ts_pct"] = np.where(shots > 0, roll["PTS"] / (2 * shots.clip(lower=1e-9)), 0.0)
    f["fg3_rate"] = np.where(roll["FGA"] > 0, roll["FG3A"] / roll["FGA"].clip(lower=1e-9), 0.0)
    f["ft_rate"] = np.where(roll["FGA"] > 0, roll["FTA"] / roll["FGA"].clip(lower=1e-9), 0.0)
    f["usage_per36"] = 36 * (shots + roll["TOV"]) / mins
    f["pts_per36"] = 36 * roll["PTS"] / mins
    f["reb_per36"] = 36 * roll["REB"] / mins
    f["ast_per36"] = 36 * roll["AST"] / mins
    f["pm_per36"] = 36 * roll["PLUS_MINUS"] / mins
    f["games_before"] = by_player.cumcount()
    f["log_games_before"] = np.log1p(f["games_before"])
    f["season_games_before"] = by_player_season.cumcount()
    prev_date = by_player["date"].shift(1)
    f["days_since_last"] = (p["date"] - prev_date).dt.days.clip(upper=60).fillna(60) / 60
    f["no_history"] = (f["games_before"] == 0).astype(float)
    f = f.fillna(0.0)
    f["exp_min"] = f["r10_MIN"].where(f["games_before"] > 0)
    keep = ["key", "PLAYER_NAME", "TEAM_ABBREVIATION", "GAME_ID", "season", "date", "MIN"]
    return pd.concat([p[keep], f], axis=1)


def team_inputs(rows, team, today, live_season, meta, out_keys):
    """
    Inputs for one team's next game. rows: player_rows() output, already standardised.
    Returns (x [15, F], w [15], ctx [3], info).
    """
    t = rows[rows.TEAM_ABBREVIATION == team]
    games = t.groupby("GAME_ID")["date"].first().sort_values()
    last_game = games.index[-1]
    roster = t[t.GAME_ID == last_game].copy()
    removed = sorted(roster.loc[roster.key.isin(out_keys), "PLAYER_NAME"])
    roster = roster[~roster.key.isin(out_keys)].sort_values("exp_min", ascending=False).head(MAX_PLAYERS)
    x = np.zeros((MAX_PLAYERS, len(PLAYER_FEATURES)), np.float32)
    w = np.zeros(MAX_PLAYERS, np.float32)
    x[: len(roster)] = roster[PLAYER_FEATURES].to_numpy()
    w[: len(roster)] = roster["exp_min"].clip(lower=0.5).to_numpy()

    season_games = t[t.season == live_season].groupby("GAME_ID")["date"].first()
    rest = (pd.Timestamp(today) - season_games.max()).days if len(season_games) else None
    ctx = np.array([min(rest, REST_CAP) / REST_CAP if rest is not None else 1.0,
                    1.0 if rest == 1 else 0.0, len(season_games) / 82], np.float32)
    info = {"roster_from": str(games.iloc[-1].date()), "players": int(len(roster)), "removed_out": removed}
    return x, w, ctx, info


def standardise(rows, meta):
    rows = rows.copy()
    rows["exp_min"] = rows["exp_min"].fillna(meta["prior_minutes"])
    mu, sd = pd.Series(meta["mu"]), pd.Series(meta["sd"])
    rows[PLAYER_FEATURES] = (rows[PLAYER_FEATURES] - mu[PLAYER_FEATURES]) / sd[PLAYER_FEATURES]
    return rows
