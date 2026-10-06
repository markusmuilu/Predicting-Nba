"""
Do the Elo and player-rating scalars add anything to a plain logistic regression?

Every variant is a standardised logistic regression with C chosen on the 2024-25 validation
season, trained on 2020-21..2023-24. Variants differ only in which columns go in:

  prod            the 52 production team features (the baseline, re-fitted here for a like-for-like C search)
  prod+elo        + team Elo difference
  prod+ratings    + player Elo and ridge-margin differences (previous-game roster, the fair setting)
  prod+all        + all three
  all_actual      the same three scalars with the *actual* roster (optimistic), plus prod
  scalars         only the three scalars and the rest/back-to-back context (a 7-number model)
  scalars_w{4,8,12}  the 7-number model trained on the last 4, 8 or 12 seasons: more history for a linear model

All rating columns come from sequential, point-in-time runs (research/elo.py,
research/player_ratings.py), so a training row never sees its own result.
"""

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from research.baseline import production_features
from research.config import PROCESSED_DIR, RESULTS_DIR, TEST_SEASON, TRAIN_SEASONS, VAL_SEASON
from research.evaluate import classification_metrics

C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


def load():
    tf = pd.read_csv(PROCESSED_DIR / "team_features.csv")
    tf["date"] = tf["Date"]
    elo = pd.read_csv(PROCESSED_DIR / "team_elo.csv", dtype={"game_id": str})
    elo["elo_diff"] = (elo.elo_home - elo.elo_away) / 100.0
    rat = pd.read_csv(PROCESSED_DIR / "player_ratings.csv", dtype={"game_id": str})
    ctx = pd.read_csv(PROCESSED_DIR / "dl_games.csv", dtype={"game_id": str})[["game_id"]]
    df = elo[["game_id", "date", "home", "away", "elo_diff"]].merge(rat[["game_id", "season", "home_win", "p1_prev", "p1_act", "p3_prev", "p3_act"]], on="game_id")
    for c in ("p1_prev", "p1_act"):
        df[c] = df[c] / 100.0
    for c in ("p3_prev", "p3_act"):
        df[c] = df[c] / 5.0
    # team context from the schedule (rest, back-to-back), via the DL builder's arrays
    ctxs = np.load(PROCESSED_DIR / "dl_previous_ctx.npy")
    games = pd.read_csv(PROCESSED_DIR / "dl_games.csv", dtype={"game_id": str})
    cdf = pd.DataFrame({"game_id": games.game_id, "rest_h": ctxs[:, 0, 0], "b2b_h": ctxs[:, 0, 1], "rest_a": ctxs[:, 1, 0], "b2b_a": ctxs[:, 1, 1]})
    df = df.merge(cdf, on="game_id", how="left")
    return df.merge(tf.drop(columns=["season", "home_win", "Date"], errors="ignore"), on=["date", "home", "away"], how="left"), tf


def fit_eval(df, cols, train_seasons, name, results):
    tr = df.season.isin(train_seasons) & df[cols].notna().all(axis=1)
    val, test = (df.season == VAL_SEASON), (df.season == TEST_SEASON)
    scaler = StandardScaler().fit(df.loc[tr, cols])
    X = scaler.transform(df[cols].fillna(0))
    best = None
    for C in C_GRID:
        m = LogisticRegression(C=C, max_iter=3000).fit(X[tr.to_numpy()], df.home_win[tr])
        ll = classification_metrics(df.home_win[val & df[cols].notna().all(axis=1)], m.predict_proba(X)[val.to_numpy() & df[cols].notna().all(axis=1).to_numpy(), 1])["log_loss"]
        if best is None or ll < best[0]:
            best = (ll, C, m)
    ll, C, m = best
    p = m.predict_proba(X)[:, 1]
    keep = (val | test) & df[cols].notna().all(axis=1)
    out = df.loc[keep, ["date", "season", "home", "away", "home_win"]].assign(p_home=p[keep.to_numpy()])
    out.to_csv(RESULTS_DIR / f"pred_comb_{name}.csv", index=False)
    results[name] = {"C": C, "n_features": len(cols), "n_train": int(tr.sum()), "val_log_loss": ll,
                     "val": classification_metrics(out.home_win[out.season == VAL_SEASON], out.p_home[out.season == VAL_SEASON])}
    print(f"{name:16s} features {len(cols):3d}  train {int(tr.sum()):5d}  C={C:<6}  val log loss {ll:.4f}  acc {results[name]['val']['accuracy']:.3f}", flush=True)


def main():
    df, tf = load()
    prod = [c for c in production_features() if c in df.columns]
    ctx = ["rest_h", "b2b_h", "rest_a", "b2b_a"]
    fair = ["elo_diff", "p1_prev", "p3_prev"]
    actual = ["elo_diff", "p1_act", "p3_act"]
    results = {}
    fit_eval(df, prod, TRAIN_SEASONS, "prod", results)
    fit_eval(df, prod + ["elo_diff"], TRAIN_SEASONS, "prod+elo", results)
    fit_eval(df, prod + ["p1_prev", "p3_prev"], TRAIN_SEASONS, "prod+ratings", results)
    fit_eval(df, prod + fair, TRAIN_SEASONS, "prod+all", results)
    fit_eval(df, prod + actual, TRAIN_SEASONS, "all_actual", results)
    fit_eval(df, fair + ctx, TRAIN_SEASONS, "scalars", results)
    seasons = sorted(df.season.unique())
    pre_val = [s for s in seasons if int(s[:4]) < int(VAL_SEASON[:4])]
    for w in (8, 12):
        fit_eval(df, fair + ctx, pre_val[-w:], f"scalars_w{w}", results)
    (RESULTS_DIR / "combine_summary.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
