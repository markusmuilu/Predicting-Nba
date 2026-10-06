"""
Steps 3 and 4: point-in-time player features and rosters.

For every player in every game, describe the player using only games he played
*before* that one. The rule that keeps this honest is the shift(1) in every
rolling statistic: the row for game g is computed from games g-1, g-2, ...
and never from game g itself.

Rosters ("who is available") are the players who actually got on the floor in
that game. That is optimistic: before tip-off you do not know for certain who
plays. It is acceptable for learning how strong a set of players is, but it
means the backtest is easier than live prediction. Say so wherever the numbers
are quoted.

The other leak this file avoids: weighting players by the minutes they actually
played in the game. Actual minutes carry the result (blowouts send the starters
to the bench early), so they are replaced by *expected* minutes, the player's
pre-game rolling average.

Output: data/processed/player_dataset.npz (arrays the model trains on) and
data/processed/games.csv (one row per game, aligned with the arrays).
"""

import numpy as np
import pandas as pd

from research.config import PROCESSED_DIR, SEASONS, TRAIN_SEASONS
from research.fetch import nba_game_log

ROLL_N = 10        # "recent form" window, in player games
MAX_PLAYERS = 15   # roster slots per team; NBA active lists are at most 15
REST_CAP = 7       # days; longer breaks all count as fully rested

BOX = ["MIN", "PTS", "REB", "OREB", "AST", "STL", "BLK", "TOV", "PF",
       "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "PLUS_MINUS"]

# The per-player vector that goes into the network, in this order.
PLAYER_FEATURES = (
    [f"r10_{c}" for c in ["MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "PLUS_MINUS"]]
    + ["sea_MIN", "sea_PTS", "sea_PLUS_MINUS"]
    + ["ts_pct", "fg3_rate", "ft_rate", "usage_per36", "pts_per36", "reb_per36", "ast_per36", "pm_per36"]
    + ["log_games_before", "season_games_before", "days_since_last", "no_history"]
)
CONTEXT_FEATURES = ["rest_days", "back_to_back", "games_into_season"]


def load_games(seasons=None):
    """One row per game from the team game logs: date, home, away, result."""
    seasons = seasons or SEASONS
    t = pd.concat([nba_game_log(s, "T").assign(season=s) for s in seasons], ignore_index=True)
    t["is_home"] = t["MATCHUP"].str.contains("vs.")
    home = t[t.is_home][["GAME_ID", "GAME_DATE", "season", "TEAM_ABBREVIATION", "WL", "PTS"]]
    away = t[~t.is_home][["GAME_ID", "TEAM_ABBREVIATION", "PTS"]]
    g = home.merge(away, on="GAME_ID", suffixes=("_home", "_away"))
    g = g.rename(columns={"GAME_ID": "game_id", "GAME_DATE": "date",
                          "TEAM_ABBREVIATION_home": "home", "TEAM_ABBREVIATION_away": "away"})
    g["home_win"] = (g["WL"] == "W").astype(int)
    return g[["game_id", "date", "season", "home", "away", "home_win", "PTS_home", "PTS_away"]] \
        .sort_values(["date", "game_id"]).reset_index(drop=True)


def player_rows(seasons=None):
    """Every player-game with pre-game features attached."""
    seasons = seasons or SEASONS
    p = pd.concat([nba_game_log(s, "P").assign(season=s) for s in seasons], ignore_index=True)
    p["date"] = pd.to_datetime(p["GAME_DATE"])
    p = p.sort_values(["PLAYER_ID", "date", "GAME_ID"]).reset_index(drop=True)
    p[BOX] = p[BOX].fillna(0)

    by_player = p.groupby("PLAYER_ID", sort=False)
    by_player_season = p.groupby(["PLAYER_ID", "season"], sort=False)

    # Sums over the previous ROLL_N games. shift(1) first, so game g is excluded.
    roll = {}
    for c in BOX:
        roll[c] = by_player[c].transform(lambda x: x.shift(1).rolling(ROLL_N, min_periods=1).sum())
    n_recent = by_player["MIN"].transform(lambda x: x.shift(1).rolling(ROLL_N, min_periods=1).count())

    f = pd.DataFrame(index=p.index)
    for c in ["MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "PLUS_MINUS"]:
        f[f"r10_{c}"] = roll[c] / n_recent

    # Season-to-date means: reset every season, also shifted.
    for c in ["MIN", "PTS", "PLUS_MINUS"]:
        f[f"sea_{c}"] = by_player_season[c].transform(lambda x: x.shift(1).expanding().mean())

    # Rates from the same 10-game sums. Ratios of sums, not means of ratios,
    # so one 2-minute cameo does not swing a percentage.
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

    # Rookies and first appearances: no history at all. Zeros plus a flag, as the plan says.
    f["no_history"] = (f["games_before"] == 0).astype(float)
    f = f.fillna(0.0)

    # Expected minutes, the pooling weight: recent average, season average as a
    # fallback, and for a player with no history a fixed prior.
    f["exp_min"] = f["r10_MIN"].where(f["games_before"] > 0)

    keep = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GAME_ID", "season", "MIN"]
    return pd.concat([p[keep], f], axis=1)


def team_context(games):
    """Rest days and back-to-backs per team per game, from the schedule only."""
    long = pd.concat([
        games[["game_id", "date", "season", "home"]].rename(columns={"home": "team"}),
        games[["game_id", "date", "season", "away"]].rename(columns={"away": "team"}),
    ])
    long["date"] = pd.to_datetime(long["date"])
    long = long.sort_values(["team", "date"])
    prev = long.groupby(["team", "season"])["date"].shift(1)
    rest = (long["date"] - prev).dt.days
    long["rest_days"] = rest.clip(upper=REST_CAP).fillna(REST_CAP) / REST_CAP
    long["back_to_back"] = (rest == 1).astype(float)
    long["games_into_season"] = long.groupby(["team", "season"]).cumcount() / 82
    return long.set_index(["game_id", "team"])[CONTEXT_FEATURES]


def previous_game_ids(games):
    """For each (game_id, team): the id of that team's previous game in the data, or None."""
    prev = {}
    last = {}
    for g in games.sort_values(["date", "game_id"]).itertuples():
        for team in (g.home, g.away):
            prev[(g.game_id, team)] = last.get(team)
            last[team] = g.game_id
    return prev


def build(roster_mode="actual"):
    """
    roster_mode "actual":   the players who played in this game (optimistic, see module docstring).
    roster_mode "previous": the players who played in the team's previous game, which is
                            known before tip-off. Their features are the rows from that game,
                            so they are one game stale. Closer to what live prediction can see.
    """
    games = load_games()
    players = player_rows()

    # Prior for players with no history: the median minutes that debuting
    # players got in the *training* seasons only.
    debut = players[(players.no_history == 1) & players.season.isin(TRAIN_SEASONS)]
    prior = float(debut["MIN"].median())
    players["exp_min"] = players["exp_min"].fillna(prior)

    # Standardise player features with training-season statistics only.
    train_rows = players.season.isin(TRAIN_SEASONS)
    mu = players.loc[train_rows, PLAYER_FEATURES].mean()
    sd = players.loc[train_rows, PLAYER_FEATURES].std().replace(0, 1)
    players[PLAYER_FEATURES] = (players[PLAYER_FEATURES] - mu) / sd

    ctx = team_context(games)
    grouped = {k: v for k, v in players.groupby(["GAME_ID", "TEAM_ABBREVIATION"])}

    n, F, C = len(games), len(PLAYER_FEATURES), len(CONTEXT_FEATURES)
    X = np.zeros((n, 2, MAX_PLAYERS, F), np.float32)   # [game, home/away, slot, feature]
    W = np.zeros((n, 2, MAX_PLAYERS), np.float32)      # expected minutes, 0 = empty slot
    CTX = np.zeros((n, 2, C), np.float32)
    prev_ids = previous_game_ids(games)
    truncated = 0
    for i, g in enumerate(games.itertuples()):
        for side, team in enumerate((g.home, g.away)):
            source = g.game_id
            if roster_mode == "previous" and prev_ids[(g.game_id, team)] is not None:
                source = prev_ids[(g.game_id, team)]
            roster = grouped[(source, team)].sort_values("exp_min", ascending=False)
            truncated += max(0, len(roster) - MAX_PLAYERS)
            roster = roster.head(MAX_PLAYERS)
            k = len(roster)
            X[i, side, :k] = roster[PLAYER_FEATURES].to_numpy()
            W[i, side, :k] = roster["exp_min"].clip(lower=0.5).to_numpy()
            CTX[i, side] = ctx.loc[(g.game_id, team)].to_numpy()

    suffix = "" if roster_mode == "actual" else f"_{roster_mode}"
    np.savez_compressed(PROCESSED_DIR / f"player_dataset{suffix}.npz", X=X, W=W, CTX=CTX,
                        y=games["home_win"].to_numpy(np.float32))
    games.to_csv(PROCESSED_DIR / "games.csv", index=False)
    print(f"[{roster_mode} roster] {n} games, {F} player features, {C} context features, "
          f"no-history minutes prior {prior:.1f}, players cut by the 15 cap: {truncated}")
    return games


if __name__ == "__main__":
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else "actual")
