"""
Step 6 of the brief (first version): does knowing the injury report help, and does the cutoff matter?

All models are logistic regressions on a handful of numbers (so differences come from the information, not the
model), standardised, C chosen on 2024-25, fitted on 2019-20 to 2023-24 (the seasons with injury reports and
ratings), and scored once on 2024-25 and 2025-26:

  previous    team Elo, player Elo and ridge plus-minus from the previous-game roster, rest and back-to-back
              (the "7-number model": no availability information)
  avail@T-60  the same, with the player Elo and ridge strengths replaced by the availability-aware strengths
              built from the report as it stood at T-60 (+ minutes lost to absences)
  avail@T-30, avail@T-15   likewise
  actual      the strengths from the players who really played (optimistic, an upper bound; not available live)

Games are the ones with a report at all three cutoffs, so every row of the table is scored on the same games.
The paired bootstrap resamples games. The odds comparison uses the 680 games with stored prices (test season only).

Run: python -m research.availability_eval
"""

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from research.availability import TRAIN
from research.config import PROCESSED_DIR, RESULTS_DIR, TEST_SEASON, VAL_SEASON
from research.evaluate import classification_metrics, devig
from research.report import KEY, load_live_history

C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


def paired(y, p_base, p_new, n=2000, seed=0):
    y = np.asarray(y, float)
    ll = lambda p: -(y * np.log(np.clip(p, 1e-9, 1 - 1e-9)) + (1 - y) * np.log(np.clip(1 - p, 1e-9, 1 - 1e-9)))
    d = ll(np.asarray(p_base)) - ll(np.asarray(p_new))
    rng = np.random.default_rng(seed)
    means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return {"mean": float(d.mean()), "lo": float(np.percentile(means, 2.5)), "hi": float(np.percentile(means, 97.5))}


def load():
    games = pd.read_csv(PROCESSED_DIR / "dl_games.csv", dtype={"game_id": str}).reset_index().rename(columns={"index": "g"})
    elo = pd.read_csv(PROCESSED_DIR / "team_elo.csv", dtype={"game_id": str})
    elo["elo_diff"] = (elo.elo_home - elo.elo_away) / 100.0
    rat = pd.read_csv(PROCESSED_DIR / "player_ratings.csv", dtype={"game_id": str})[["game_id", "p1_prev", "p3_prev", "p1_act", "p3_act"]]
    ctx = np.load(PROCESSED_DIR / "dl_previous_ctx.npy")
    games["rest_h"], games["b2b_h"], games["rest_a"], games["b2b_a"] = ctx[:, 0, 0], ctx[:, 0, 1], ctx[:, 1, 0], ctx[:, 1, 1]
    df = games.merge(elo[["game_id", "elo_diff"]], on="game_id").merge(rat, on="game_id")
    for c in ("p1_prev", "p1_act"):
        df[c] = df[c] / 100.0
    for c in ("p3_prev", "p3_act"):
        df[c] = df[c] / 5.0
    av = pd.read_csv(PROCESSED_DIR / "avail_features.csv.gz", dtype={"game_id": str})
    for cut in (60, 30, 15):
        a = av[av.cutoff == cut].copy()
        a[f"a1_{cut}"] = (a.a1_home - a.a1_away) / 100.0
        a[f"a3_{cut}"] = (a.a3_home - a.a3_away) / 5.0
        a[f"lost_{cut}"] = (a.min_lost_home - a.min_lost_away) / 48.0
        df = df.merge(a[["game_id", f"a1_{cut}", f"a3_{cut}", f"lost_{cut}"]], on="game_id", how="left")
    return df


def fit(df, cols, name, results, store):
    tr = df.season.isin(TRAIN).to_numpy()
    val, test = (df.season == VAL_SEASON).to_numpy(), (df.season == TEST_SEASON).to_numpy()
    sc = StandardScaler().fit(df.loc[tr, cols])
    X = sc.transform(df[cols])
    best = None
    for C in C_GRID:
        m = LogisticRegression(C=C, max_iter=3000).fit(X[tr], df.home_win[tr])
        ll = classification_metrics(df.home_win[val], m.predict_proba(X[val])[:, 1])["log_loss"]
        if best is None or ll < best[0]:
            best = (ll, C, m)
    _, C, m = best
    p = m.predict_proba(X)[:, 1]
    store[name] = p
    results[name] = {"C": C, "n_features": len(cols),
                     "validation": classification_metrics(df.home_win[val], p[val]), "test": classification_metrics(df.home_win[test], p[test])}
    print(f"{name:12s} val log loss {results[name]['validation']['log_loss']:.4f} acc {results[name]['validation']['accuracy']:.3f} | "
          f"test {results[name]['test']['log_loss']:.4f} acc {results[name]['test']['accuracy']:.3f}", flush=True)


def main():
    df = load()
    need = ["a1_60", "a1_30", "a1_15", "a3_60", "a3_30", "a3_15"]
    df = df[df[need].notna().all(axis=1) | ~df.season.isin(TRAIN + [VAL_SEASON, TEST_SEASON])]
    df = df[df.season.isin(TRAIN + [VAL_SEASON, TEST_SEASON]) & df[need].notna().all(axis=1)].reset_index(drop=True)
    print(f"{len(df)} games with a report at all three cutoffs; "
          f"train {df.season.isin(TRAIN).sum()}, val {(df.season == VAL_SEASON).sum()}, test {(df.season == TEST_SEASON).sum()}", flush=True)
    ctx = ["rest_h", "b2b_h", "rest_a", "b2b_a"]
    results, store = {}, {}
    fit(df, ["elo_diff", "p1_prev", "p3_prev"] + ctx, "previous", results, store)
    for cut in (60, 30, 15):
        fit(df, ["elo_diff", f"a1_{cut}", f"a3_{cut}", f"lost_{cut}"] + ctx, f"avail@T-{cut}", results, store)
    fit(df, ["elo_diff", "p1_act", "p3_act"] + ctx, "actual", results, store)

    test = (df.season == TEST_SEASON).to_numpy()
    val = (df.season == VAL_SEASON).to_numpy()
    y = df.home_win.to_numpy()
    gains = {}
    for name in store:
        if name == "previous":
            continue
        gains[name] = {"validation": paired(y[val], store["previous"][val], store[name][val]), "test": paired(y[test], store["previous"][test], store[name][test])}
    print("\nGain in log loss over 'previous' (positive = better), paired bootstrap 95% interval:")
    for name, g in gains.items():
        print(f"  {name:12s} val {g['validation']['mean']:+.4f} [{g['validation']['lo']:+.4f}, {g['validation']['hi']:+.4f}]   "
              f"test {g['test']['mean']:+.4f} [{g['test']['lo']:+.4f}, {g['test']['hi']:+.4f}]")
    for a, b in (("avail@T-30", "avail@T-60"), ("avail@T-15", "avail@T-30")):
        gains[f"{a} vs {b}"] = {"validation": paired(y[val], store[b][val], store[a][val]), "test": paired(y[test], store[b][test], store[a][test])}
        g = gains[f"{a} vs {b}"]
        print(f"  {a} over {b}: val {g['validation']['mean']:+.4f} [{g['validation']['lo']:+.4f}, {g['validation']['hi']:+.4f}]   test {g['test']['mean']:+.4f} [{g['test']['lo']:+.4f}, {g['test']['hi']:+.4f}]")

    # Against the market on the games with stored prices
    live = load_live_history()
    d = df.assign(p_prev=store["previous"], p60=store["avail@T-60"], p30=store["avail@T-30"], p15=store["avail@T-15"], p_act=store["actual"])
    d = d[d.season == TEST_SEASON].merge(live.rename(columns={"home": "home", "away": "away"}), on=["date", "home", "away"], how="inner")
    d = d[d.home_odds.notna() & d.away_odds.notna()]
    if len(d):
        market = devig(d.home_odds, d.away_odds)
        mk = classification_metrics(d.home_win, market)
        print(f"\nGames with prices: {len(d)}; market log loss {mk['log_loss']:.4f}")
        odds = {"n": int(len(d)), "market": mk}
        for k in ("p_prev", "p60", "p30", "p15", "p_act"):
            m = classification_metrics(d.home_win, d[k])
            odds[k] = m
            print(f"  {k:7s} log loss {m['log_loss']:.4f}  (market {mk['log_loss']:.4f}, gap {m['log_loss'] - mk['log_loss']:+.4f})")
        results["odds"] = odds
    results["gains"] = gains
    (RESULTS_DIR / "availability_eval.json").write_text(json.dumps(results, indent=1, default=float))
    out = df[["game_id", "date", "season", "home", "away", "home_win"]].assign(
        **{f"p_{k.replace('@', '_').replace('-', '')}": v for k, v in store.items()})
    out.to_csv(RESULTS_DIR / "availability_predictions.csv", index=False)


if __name__ == "__main__":
    main()
