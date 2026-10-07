"""
Walk-forward version of research/availability_eval.py: five test seasons instead of one.

For each test season S from 2021-22 to 2025-26: fit on every earlier season from 2018-19 on except the one just
before S, choose the regularisation on that one (S-1), and score S once. The same procedure for every model, so the
pooled result covers about 6,000 games and no season is judged by a model that has seen it.

Models and games are the same as in availability_eval.py (games with a report at all three cutoffs).

Run: python -m research.availability_walkforward
"""

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from research.availability_eval import C_GRID, load, paired
from research.config import RESULTS_DIR
from research.evaluate import classification_metrics

SEASONS = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
FOLDS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]


def fold_predictions(df, cols, test_season):
    i = SEASONS.index(test_season)
    val_season, train_seasons = SEASONS[i - 1], SEASONS[:i - 1]
    tr, va, te = df.season.isin(train_seasons).to_numpy(), (df.season == val_season).to_numpy(), (df.season == test_season).to_numpy()
    sc = StandardScaler().fit(df.loc[tr, cols])
    X = sc.transform(df[cols])
    best = None
    for C in C_GRID:
        m = LogisticRegression(C=C, max_iter=3000).fit(X[tr], df.home_win[tr])
        ll = classification_metrics(df.home_win[va], m.predict_proba(X[va])[:, 1])["log_loss"]
        if best is None or ll < best[0]:
            best = (ll, m)
    return best[1].predict_proba(X[te])[:, 1], te


def main():
    df = load()
    need = ["a1_60", "a1_30", "a1_15"]
    df = df[df.season.isin(SEASONS) & df[need].notna().all(axis=1)].reset_index(drop=True)
    ctx = ["rest_h", "b2b_h", "rest_a", "b2b_a"]
    specs = {"previous": ["elo_diff", "p1_prev", "p3_prev"] + ctx,
             "avail@T-60": ["elo_diff", "a1_60", "a3_60", "lost_60"] + ctx,
             "avail@T-30": ["elo_diff", "a1_30", "a3_30", "lost_30"] + ctx,
             "avail@T-15": ["elo_diff", "a1_15", "a3_15", "lost_15"] + ctx,
             "actual": ["elo_diff", "p1_act", "p3_act"] + ctx}
    pooled = {k: [] for k in specs}
    y_all, season_all = [], []
    per_fold = {}
    for s in FOLDS:
        res = {}
        for name, cols in specs.items():
            p, te = fold_predictions(df, cols, s)
            pooled[name].append(p)
            res[name] = classification_metrics(df.home_win[te], p)
        y_all.append(df.home_win[te].to_numpy())
        season_all += [s] * int(te.sum())
        per_fold[s] = {"games": int(te.sum()), **{k: round(v["log_loss"], 4) for k, v in res.items()}}
        print(f"{s}  n={te.sum():4d}  " + "  ".join(f"{k} {v['log_loss']:.4f}" for k, v in res.items()), flush=True)
    y = np.concatenate(y_all)
    P = {k: np.concatenate(v) for k, v in pooled.items()}
    print(f"\nPooled over {len(y)} games:")
    out = {"folds": per_fold, "pooled": {}, "gain_over_previous": {}, "cutoff_comparison": {}}
    for k in specs:
        m = classification_metrics(y, P[k])
        out["pooled"][k] = m
        print(f"  {k:12s} log loss {m['log_loss']:.4f}  accuracy {m['accuracy']:.3f}  brier {m['brier']:.4f}")
    print("\nGain in log loss over 'previous' (positive = better), pooled, 95% interval; and in each season:")
    seasons = np.array(season_all)
    for k in specs:
        if k == "previous":
            continue
        g = paired(y, P["previous"], P[k])
        out["gain_over_previous"][k] = {"pooled": g, "by_season": {s: paired(y[seasons == s], P["previous"][seasons == s], P[k][seasons == s])["mean"] for s in FOLDS}}
        print(f"  {k:12s} {g['mean']:+.4f} [{g['lo']:+.4f}, {g['hi']:+.4f}]   by season: " + ", ".join(f"{out['gain_over_previous'][k]['by_season'][s]:+.4f}" for s in FOLDS))
    for a, b in (("avail@T-30", "avail@T-60"), ("avail@T-15", "avail@T-30"), ("avail@T-15", "avail@T-60")):
        g = paired(y, P[b], P[a])
        out["cutoff_comparison"][f"{a} over {b}"] = g
        print(f"  {a} over {b}: {g['mean']:+.4f} [{g['lo']:+.4f}, {g['hi']:+.4f}]")
    (RESULTS_DIR / "availability_walkforward.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
