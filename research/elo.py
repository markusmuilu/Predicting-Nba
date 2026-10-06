"""
Team Elo, the way FiveThirtyEight ran it, plus a validation-tuned version.

Elo in one paragraph: every team has a rating. Before a game the home team's
win probability is 1 / (1 + 10^(-d/400)), where d is the rating difference plus a
home-court bonus. After the game both ratings move by K * (result - expected), so a
surprise moves them more than an expected result. Two refinements from FiveThirtyEight's
NBA version are included because they are well documented and cheap:
- a margin-of-victory multiplier, so a 25-point win moves ratings more than a 2-point
  win, with diminishing returns, damped when the winner was already far ahead (a big
  favourite winning big is not much news);
- season carryover: at the start of a season each rating is pulled part-way back to the
  league average, because rosters change.
FiveThirtyEight's published values (K=20, home bonus 100 points, carryover 0.75, margin
exponent 0.8) are the starting point. Here the four numbers are *tuned on the 2024-25
validation season*, never on the test season.

Everything is point-in-time: the rating used for a game is the one before that game, and
the update after the game uses only that game's result. Ratings run through all seasons
in order, including validation and test, because updating on past results is legitimate;
only the four numbers are fixed in advance.

Elo can start earlier than the model training window. Extra older seasons only serve
as warm-up so the ratings are not still at their starting values in 2020-21.
"""

import itertools
import json

import numpy as np
import pandas as pd

from research.config import EXTRA_SEASONS, PROCESSED_DIR, RAW_DIR, RESULTS_DIR, SEASONS, TEST_SEASON, TRAIN_SEASONS, VAL_SEASON
from research.evaluate import classification_metrics
from research.player_features import load_games

MEAN_RATING = 1505.0   # what ratings regress towards between seasons (538's choice)
START_RATING = 1500.0  # a team seen for the first time


def run_elo(games, K=20.0, home=100.0, carry=0.75, mov_exp=0.8):
    """
    games: DataFrame ordered by date with home, away, home_win, PTS_home, PTS_away, season.
    mov_exp = 0 turns the margin-of-victory multiplier off (plain win/loss Elo).
    Returns arrays aligned to games: home rating, away rating before the game, and the
    home win probability.
    """
    rating = {}
    last_season = None
    n = len(games)
    r_home, r_away, prob = np.zeros(n), np.zeros(n), np.zeros(n)

    for i, g in enumerate(games.itertuples(index=False)):
        if g.season != last_season:
            for team in rating:
                rating[team] = carry * rating[team] + (1 - carry) * MEAN_RATING
            last_season = g.season
        rh = rating.setdefault(g.home, START_RATING)
        ra = rating.setdefault(g.away, START_RATING)
        r_home[i], r_away[i] = rh, ra

        d = rh + home - ra
        p = 1.0 / (1.0 + 10.0 ** (-max(-1500.0, min(1500.0, d)) / 400.0))  # clip: diverging settings must score badly, not crash
        prob[i] = p

        result = float(g.home_win)
        margin = abs(g.PTS_home - g.PTS_away)
        if mov_exp > 0:
            winner_d = d if result == 1 else -d          # winner's rating edge incl. home bonus
            mult = ((margin + 3.0) ** mov_exp) / (7.5 + 0.006 * winner_d)
        else:
            mult = 1.0
        shift = K * mult * (result - p)
        rating[g.home] = rh + shift
        rating[g.away] = ra - shift
    return r_home, r_away, prob


def all_games():
    """Extra warm-up seasons (if downloaded) followed by the six main seasons, in date order."""
    seasons = [s for s in EXTRA_SEASONS if (RAW_DIR / f"nba_gamelog_T_{s}.csv").exists()]
    return load_games(seasons + SEASONS), seasons


def tune(games, extra_used):
    """Grid over the four numbers, scored by validation log loss of the raw Elo probability."""
    # The first grid (K 8-40, home 40-120, carry 0.5-0.9, margin exponent 0-1) put its best
    # point on the edge in three of four dimensions, so the grid was widened downwards.
    grid = {"K": [6, 8, 10, 12, 16, 20, 30], "home": [0, 20, 30, 40, 50, 60, 80], "carry": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "mov_exp": [0.0, 0.8, 1.0, 1.2, 1.5]}
    val = (games.season == VAL_SEASON).to_numpy()
    y = games["home_win"].to_numpy()
    rows = []
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), values))
        _, _, p = run_elo(games, **params)
        rows.append({**params, **{f"val_{k}": v for k, v in classification_metrics(y[val], p[val]).items()}})
    return pd.DataFrame(rows).sort_values("val_log_loss").reset_index(drop=True)


def run():
    games, extra_used = all_games()
    games = games.sort_values(["date", "game_id"]).reset_index(drop=True)
    y = games["home_win"].to_numpy()
    val = (games.season == VAL_SEASON).to_numpy()
    test = (games.season == TEST_SEASON).to_numpy()
    print(f"Elo over {len(games)} games, warm-up seasons used: {extra_used}")

    # 538's published settings, untouched, as the reference point.
    _, _, p538 = run_elo(games)
    ref = {"params": {"K": 20, "home": 100, "carry": 0.75, "mov_exp": 0.8},
           "val": classification_metrics(y[val], p538[val])}
    print("538 settings, validation:", {k: round(v, 4) for k, v in ref["val"].items() if k != "n"})

    grid = tune(games, extra_used)
    grid.to_csv(RESULTS_DIR / "elo_grid.csv", index=False)
    best = grid.iloc[0]
    params = {"K": float(best.K), "home": float(best.home), "carry": float(best.carry), "mov_exp": float(best.mov_exp)}
    print("Best on validation:", params, round(best.val_log_loss, 4))
    print("Top 5 val log loss:", [round(x, 4) for x in grid.val_log_loss.head(5)], " worst:", round(grid.val_log_loss.max(), 4))

    rh, ra, p = run_elo(games, **params)
    feats = games[["game_id", "date", "season", "home", "away", "home_win"]].assign(elo_home=rh, elo_away=ra, p_elo_raw=p)
    feats.to_csv(PROCESSED_DIR / "team_elo.csv", index=False)

    # Probability used as a model: the Elo probability is on the right scale in theory but the
    # tuned numbers can leave it a little over- or under-confident. One scale and one intercept,
    # fitted on the TRAINING seasons (not validation, not test).
    from sklearn.linear_model import LogisticRegression
    tr = games.season.isin(TRAIN_SEASONS).to_numpy()
    d = (rh - ra).reshape(-1, 1)
    cal = LogisticRegression(C=1e6).fit(d[tr], y[tr])
    p_cal = cal.predict_proba(d)[:, 1]
    keep = val | test
    out = feats[keep].assign(p_home=p_cal[keep])
    out[["date", "season", "home", "away", "home_win", "p_home"]].to_csv(RESULTS_DIR / "pred_team_elo.csv", index=False)

    summary = {"warmup_seasons": extra_used, "reference_538": ref, "chosen": params,
               "val": classification_metrics(y[val], p_cal[val]), "n_grid": int(len(grid)),
               "scale_per_point": float(cal.coef_[0][0]), "intercept": float(cal.intercept_[0])}
    (RESULTS_DIR / "team_elo_summary.json").write_text(json.dumps(summary, indent=2))
    print("Calibrated Elo validation:", {k: round(v, 4) for k, v in summary["val"].items() if k != "n"})
    return summary


if __name__ == "__main__":
    run()
