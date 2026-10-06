"""
Player-level rating systems, in the spirit of how public NBA models do it.

Public systems (RAPTOR, EPM, DARKO and the "CARM-Elo" family) share one structure:
rate each player, then build a team's strength as a minutes-weighted sum of its players'
ratings, using *projected* minutes so that a trade or an injury shows up immediately.
Two ways of rating players are built here, both strictly point-in-time:

P1  Player Elo (additive). A team's strength is sum_i w_i * r_i with w_i the player's share
    of expected minutes (scaled so a full lineup sums to 5). The home win probability is the
    usual Elo curve of (home strength - away strength + home bonus). After the game every
    player who played moves by K * (result - expected) * (his actual minute share), scaled
    by the margin of victory, so players on the floor for more of a surprise get more of the
    credit or blame. Players regress towards 0 each season; a player seen for the first time
    starts at a "replacement level" below 0. This is online logistic regression on player
    indicators, trained one game at a time.

P3  Regularised plus-minus (a RAPM-style ridge regression). Regress the game margin on
    (home minute shares - away minute shares). The ridge penalty shrinks every player towards
    0, which is what makes the regression stable when players are almost always on the court
    with the same teammates. Older games are down-weighted exponentially (half-life in
    games), and the coefficients are re-solved every 10 games from games played so far.

Two rosters are produced for every game, matching the earlier experiments:
- "previous": the players who played in the team's previous game. Known before tip-off.
- "actual":   the players who played in this game. Optimistic, as before.
Both use *expected* minutes (the mean of the player's last 10 games before this one), never
the minutes he played in the game being predicted. The update after a game does use that
game's actual minutes, which is fine: it happens after the result is known.

Hyperparameters are chosen by validation-season (2024-25) log loss using the "previous" roster.
The test season is never used for tuning.
"""

import json
from collections import defaultdict

import numpy as np
import pandas as pd

from research.config import EXTRA_SEASONS, PROCESSED_DIR, RAW_DIR, RESULTS_DIR, SEASONS, TEST_SEASON, TRAIN_SEASONS, VAL_SEASON
from research.evaluate import classification_metrics
from research.fetch import nba_game_log
from research.player_features import load_games

PRIOR_MINUTES = 12.0     # expected minutes for a player with no games on record (as in the Deep Sets data)
LOG10 = np.log(10.0)


def usable_seasons():
    """Extra warm-up seasons that are downloaded (both logs), then the six main seasons."""
    extra = [s for s in EXTRA_SEASONS
             if (RAW_DIR / f"nba_gamelog_T_{s}.csv").exists() and (RAW_DIR / f"nba_gamelog_P_{s}.csv").exists()]
    return extra + SEASONS


def prepare(seasons=None):
    """
    Games in date order plus, per game and team, the arrays of player ids and minutes.
    Returns (games, rosters, n_players) where rosters[(game_id, team)] = (player_index[], minutes[]).
    """
    seasons = seasons or usable_seasons()
    games = load_games(seasons).sort_values(["date", "game_id"]).reset_index(drop=True)
    plog = pd.concat([nba_game_log(s, "P") for s in seasons], ignore_index=True)
    plog["MIN"] = plog["MIN"].fillna(0)
    plog = plog[plog["MIN"] > 0]
    index = {pid: k for k, pid in enumerate(plog["PLAYER_ID"].unique())}
    rosters = {}
    for (gid, team), grp in plog.groupby(["GAME_ID", "TEAM_ABBREVIATION"]):
        rosters[(gid, team)] = (grp["PLAYER_ID"].map(index).to_numpy(), grp["MIN"].to_numpy(float))
    # A game with no player rows for either side (one 2012-13 game) cannot be rated.
    has_both = [(g.game_id, g.home) in rosters and (g.game_id, g.away) in rosters for g in games.itertuples()]
    games = games[has_both].reset_index(drop=True)
    return games, rosters, len(index)


class MinutesTracker:
    """Expected minutes: the mean of a player's last 10 games' minutes, vectorised over players."""

    def __init__(self, n):
        self.buf = np.zeros((n, 10))
        self.pos = np.zeros(n, int)
        self.cnt = np.zeros(n, int)

    def expected(self, ids):
        c = self.cnt[ids]
        return np.where(c > 0, self.buf[ids].sum(1) / np.maximum(c, 1), PRIOR_MINUTES)

    def update(self, ids, minutes):
        self.buf[ids, self.pos[ids]] = minutes
        self.pos[ids] = (self.pos[ids] + 1) % 10
        self.cnt[ids] = np.minimum(self.cnt[ids] + 1, 10)


def weights(ids, tracker):
    """Share of expected minutes, scaled so a full lineup (5 players on court) sums to 5."""
    e = tracker.expected(ids)
    return 5.0 * e / e.sum()


def run_player_elo(games, rosters, n_players, K=20.0, home=40.0, carry=0.7, r_new=-20.0, mov_exp=1.0, snapshots=None):
    """
    P1. Returns arrays aligned with games: team strength difference (home - away) in Elo points
    for the previous-game roster and for the actual roster, and the final ratings.
    snapshots: if a list is given, a copy of every player's rating at the start of each game is
    appended (after the season carryover and the replacement-level start of new players).
    """
    r = np.zeros(n_players)
    seen = np.zeros(n_players, bool)
    tracker = MinutesTracker(n_players)
    last_roster = {}
    n = len(games)
    diff_prev, diff_act = np.zeros(n), np.zeros(n)
    last_season = None

    for i, g in enumerate(games.itertuples(index=False)):
        if g.season != last_season:
            r *= carry                                   # regress every player towards 0
            last_season = g.season
        ids_h, min_h = rosters[(g.game_id, g.home)]
        ids_a, min_a = rosters[(g.game_id, g.away)]
        for ids in (ids_h, ids_a):                       # first appearance: replacement-level start
            fresh = ids[~seen[ids]]
            r[fresh] = r_new
            seen[fresh] = True

        if snapshots is not None:
            snapshots.append(r.astype(np.float32).copy())
        # Pre-game strengths with the two roster definitions (expected minutes only).
        w_h, w_a = weights(ids_h, tracker), weights(ids_a, tracker)
        diff_act[i] = (w_h * r[ids_h]).sum() - (w_a * r[ids_a]).sum()
        ph, pa = last_roster.get(g.home), last_roster.get(g.away)
        if ph is None or pa is None:
            diff_prev[i] = diff_act[i]
        else:
            diff_prev[i] = (weights(ph, tracker) * r[ph]).sum() - (weights(pa, tracker) * r[pa]).sum()

        # Update from the result, using the actual roster's actual minute shares.
        d = diff_act[i] + home
        p = 1.0 / (1.0 + 10.0 ** (-np.clip(d, -1500, 1500) / 400.0))
        result = float(g.home_win)
        margin = abs(g.PTS_home - g.PTS_away)
        if mov_exp > 0:
            winner_d = d if result == 1 else -d
            mult = ((margin + 3.0) ** mov_exp) / (7.5 + 0.006 * winner_d)
        else:
            mult = 1.0
        shift = K * mult * (result - p)
        r[ids_h] += shift * (min_h / 48.0)
        r[ids_a] -= shift * (min_a / 48.0)

        tracker.update(ids_h, min_h)
        tracker.update(ids_a, min_a)
        last_roster[g.home], last_roster[g.away] = ids_h, ids_a
    return diff_prev, diff_act, r


def run_ridge_margin(games, rosters, n_players, lam=50.0, half_life=800.0, refit_every=10, snapshots=None):
    """
    P3. Online ridge regression of the home margin on minute-share differences.
    Returns arrays of predicted home margin (points) for the previous and the actual roster.
    """
    tracker = MinutesTracker(n_players)
    A = np.zeros((n_players + 1, n_players + 1))        # column 0 = home-court constant
    b = np.zeros(n_players + 1)
    beta = np.zeros(n_players + 1)
    gamma = 0.5 ** (1.0 / half_life)
    last_roster = {}
    last_active = np.full(n_players, -10**9)            # game index of each player's last appearance
    n = len(games)
    pred_prev, pred_act = np.zeros(n), np.zeros(n)
    batch = []

    def solve(i):
        active = np.where(last_active > i - 3000)[0]    # players seen in roughly the last two seasons
        idx = np.concatenate([[0], active + 1])
        Asub = A[np.ix_(idx, idx)] + lam * np.eye(len(idx))
        Asub[0, 0] -= lam - 1e-3                        # the home constant is barely penalised
        sol = np.linalg.solve(Asub, b[idx])
        beta[:] = 0.0
        beta[idx] = sol

    for i, g in enumerate(games.itertuples(index=False)):
        ids_h, min_h = rosters[(g.game_id, g.home)]
        ids_a, min_a = rosters[(g.game_id, g.away)]
        if snapshots is not None:
            snapshots.append(beta[1:].astype(np.float32).copy())
        w_h, w_a = weights(ids_h, tracker), weights(ids_a, tracker)
        pred_act[i] = beta[0] + (w_h * beta[ids_h + 1]).sum() - (w_a * beta[ids_a + 1]).sum()
        ph, pa = last_roster.get(g.home), last_roster.get(g.away)
        if ph is None or pa is None:
            pred_prev[i] = pred_act[i]
        else:
            pred_prev[i] = beta[0] + (weights(ph, tracker) * beta[ph + 1]).sum() - (weights(pa, tracker) * beta[pa + 1]).sum()

        # Learn from the game: sparse design row (home constant, + home shares, - away shares).
        cols = np.concatenate([[0], ids_h + 1, ids_a + 1])
        vals = np.concatenate([[1.0], min_h / 48.0, -(min_a / 48.0)])
        batch.append((cols, vals, g.PTS_home - g.PTS_away))
        tracker.update(ids_h, min_h)
        tracker.update(ids_a, min_a)
        last_roster[g.home], last_roster[g.away] = ids_h, ids_a
        last_active[ids_h] = i
        last_active[ids_a] = i

        if (i + 1) % refit_every == 0:
            A *= gamma ** refit_every                   # decay everything once per batch
            b *= gamma ** refit_every
            for age, (cols, vals, y) in enumerate(reversed(batch)):
                wgt = gamma ** age
                A[np.ix_(cols, cols)] += wgt * np.outer(vals, vals)
                b[cols] += wgt * y * vals
            batch = []
            solve(i)
    return pred_prev, pred_act
