"""
Tunes, scores and writes out the Finnish rating models.

For each pool (national = Korisliiga + I divisioona A + B; m2d = Miesten II divisioona) and for every
combination of family (elo, margin) and margin definition (final, q3, half, mix):
  1. random search of 1,500 settings on the log loss of 2023-24 and 2024-25 (2022-23 is warm-up);
  2. the best setting is scored on 2025-26 (never used for tuning) and on the finished part of 2026-27.
All combinations are reported. The model that is published is the one with the best *tuning* log loss.

Outputs under research/results/finland/:
  <pool>_results.json     every combination: settings and metrics
  <pool>_predictions.csv  the chosen model's probability and predicted margin for every game
  <pool>_ratings.csv      current ratings of the teams in the live season

Run: python -m research.finland.run_models
"""

import json

import numpy as np
import pandas as pd

from research.finland import model as M
from research.evaluate import calibration_table, classification_metrics

OUT = M.OUT
N_SEARCH = 1500


def paired_gain(y, p_base, p_new, n_boot=2000, seed=0):
    y = np.asarray(y, float)
    ll = lambda p: -(y * np.log(np.clip(p, 1e-9, 1 - 1e-9)) + (1 - y) * np.log(np.clip(1 - p, 1e-9, 1 - 1e-9)))
    d = ll(np.asarray(p_base)) - ll(np.asarray(p_new))
    rng = np.random.default_rng(seed)
    means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return {"mean": float(d.mean()), "lo": float(np.percentile(means, 2.5)), "hi": float(np.percentile(means, 97.5))}


def margin_fit(games, family, pm):
    """Predicted margin in points. The margin family already predicts points; for Elo, regress margin on the rating gap."""
    if family == "margin":
        return pm, (1.0, 0.0)
    m = games.season.isin(M.TUNE_SEASONS) & games.finished
    a, b = np.polyfit(pm[m], games.m_final[m], 1)
    return a * pm + b, (float(a), float(b))


def run_pool(pool):
    games = M.load_games(pool)
    const, pick = M.baselines(games)
    fin = games.finished
    split = {"tune": games.season.isin(M.TUNE_SEASONS) & fin, "test": (games.season == M.TEST_SEASON) & fin,
             "live": (games.season == M.LIVE_SEASON) & fin}
    results, store = [], {}
    for family in ("elo", "margin"):
        for mode in M.MARGIN_MODES:
            params, tune_ll, rows = M.tune(games, family, pool, mode, n=N_SEARCH)
            prob, pm, rating = M.run(games, family, params, mode)
            res = {"family": family, "margin_mode": mode, "params": params, "tune_log_loss_best": tune_ll,
                   "metrics": {k: classification_metrics(games.home_win[m], prob[m]) for k, m in split.items() if m.sum() > 5}}
            res["search_spread"] = {"best": float(min(r["tune_log_loss"] for r in rows)), "median": float(np.median([r["tune_log_loss"] for r in rows]))}
            results.append(res)
            store[(family, mode)] = (prob, pm, rating, params)
            t = res["metrics"]
            print(f"{pool:8s} {family:7s} {mode:6s} tune {t['tune']['log_loss']:.4f}  test {t['test']['log_loss']:.4f} acc {t['test']['accuracy']:.3f}"
                  f"  live {t['live']['log_loss'] if 'live' in t else float('nan'):.4f}", flush=True)

    best = min(results, key=lambda r: r["metrics"]["tune"]["log_loss"])
    family, mode = best["family"], best["margin_mode"]
    prob, pm, rating, params = store[(family, mode)]
    margin_pts, margin_coef = margin_fit(games, family, pm)
    base = {k: {"constant": classification_metrics(games.home_win[m], const[m]),
                "record_pick_accuracy": float((pick[m] == games.home_win[m]).mean())} for k, m in split.items() if m.sum() > 5}
    best["baselines"] = base
    best["gain_vs_constant"] = {k: paired_gain(games.home_win[m], const[m], prob[m]) for k, m in split.items() if m.sum() > 5}
    best["calibration_test"] = calibration_table(games.home_win[split["test"]], prob[split["test"]], bins=6).to_dict("records")
    best["chosen"] = True
    best["margin_coef"] = margin_coef

    out = games[["season", "league", "region", "group", "round", "match_id", "when", "home_id", "home", "away_id", "away", "home_pts", "away_pts",
                 "finished", "home_win", "m_final"]].copy()
    out["p_home"], out["pred_margin"] = prob, margin_pts
    out["p_home_const"] = const
    out.to_csv(OUT / f"{pool}_predictions.csv", index=False)

    live_teams = set(games[games.season == M.LIVE_SEASON].home_id) | set(games[games.season == M.LIVE_SEASON].away_id)
    names = pd.concat([games[["home_id", "home", "league", "when"]].rename(columns={"home_id": "id", "home": "name"}),
                       games[["away_id", "away", "league", "when"]].rename(columns={"away_id": "id", "away": "name"})]).sort_values("when")
    last = names.groupby("id").last()
    played = pd.concat([games[games.finished & (games.season == M.LIVE_SEASON)].home_id, games[games.finished & (games.season == M.LIVE_SEASON)].away_id]).value_counts()
    table = pd.DataFrame([{"team_id": t, "team": last.loc[t, "name"], "league": last.loc[t, "league"],
                           "rating": rating.get(t, 0.0), "games_this_season": int(played.get(t, 0))} for t in live_teams])
    table = table.sort_values("rating", ascending=False)
    table.to_csv(OUT / f"{pool}_ratings.csv", index=False)
    (OUT / f"{pool}_results.json").write_text(json.dumps({"pool": pool, "combinations": results, "chosen": {"family": family, "margin_mode": mode}}, indent=1, default=str))
    print(f"\n{pool}: chosen {family}/{mode}; baselines tune/test constant log loss:",
          round(base['tune']['constant']['log_loss'], 4), round(base['test']['constant']['log_loss'], 4),
          "| record-pick accuracy test:", round(base['test']['record_pick_accuracy'], 3), "\n", flush=True)
    return best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for pool in ("national", "m2d"):
        run_pool(pool)


if __name__ == "__main__":
    main()
