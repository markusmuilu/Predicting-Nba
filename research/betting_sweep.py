"""
Does any model have a betting chance? Kelly staking with different fractions and edge thresholds,
and the question that sits underneath all staking rules: does the model know anything the market does not?

Data: the 680 games of the 2025-26 test season with stored Pinnacle prices (7 Jan to 12 Apr 2026).
Those prices were taken around midday, not at closing, and they are the only odds that exist, so there
is no earlier season on which to choose a fraction or a threshold. Everything below is therefore a
*description* of what each setting would have done, not a tuned strategy: with 680 games and a grid of
settings, a profitable-looking cell is what chance produces, and is read that way.

1. Kelly sweep: stake k * (p*odds - 1) / (odds - 1) of the bankroll on the side the model rates as having
   positive expected value, only when that edge (p*odds - 1) exceeds a threshold. k is the Kelly fraction.
2. Flat stakes on the same bets, with a bootstrap interval on the ROI (resampling games).
3. Edge calibration: among bets grouped by how big the model's edge was, did the winners show up as often
   as the model said? An edge that is real shows up as a win rate above the odds-implied rate.
4. Market blend: logit(final) = logit(market) + w * (logit(model) - logit(market)). w = 0 means the model adds
   nothing to the market, w = 1 means the model replaces it. w is fitted on one half of the odds games (by date)
   and the log loss is measured on the other half, in both directions. If the blend beats the market on the
   held-out half, the model carries information the market lacks.

Run: python -m research.betting_sweep
"""

import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from research.config import RESULTS_DIR, TEST_SEASON
from research.evaluate import devig
from research.report import KEY, common_games, load_live_history, load_predictions

MODELS = ["logreg", "team_elo", "comb_scalars", "player_model_prev", "dl_both_w12_pr_wide", "dl_both_w12_pr_wide_league",
          "jev_named_cal", "dl_both_w12_wide_actual"]
FRACTIONS = [0.05, 0.1, 0.25, 0.5, 1.0]
THRESHOLDS = [0.0, 0.02, 0.05, 0.10]


def odds_games():
    preds = load_predictions()
    keep = {k: v for k, v in preds.items() if k in MODELS}
    games = common_games(keep, TEST_SEASON).merge(load_live_history(), on=KEY, how="left")
    g = games[games.home_odds.notna() & games.away_odds.notna()].sort_values(KEY).reset_index(drop=True)
    g["p_market"] = devig(g.home_odds, g.away_odds)
    return g, [k for k in MODELS if k in keep]


def bets(g, p):
    """Side, price and edge of the best positive-EV side for each game (edge = p*odds - 1)."""
    oh, oa = g.home_odds.to_numpy(float), g.away_odds.to_numpy(float)
    ev_h, ev_a = p * oh - 1, (1 - p) * oa - 1
    home = ev_h >= ev_a
    edge = np.where(home, ev_h, ev_a)
    price = np.where(home, oh, oa)
    p_side = np.where(home, p, 1 - p)
    won = np.where(home, g.home_win.to_numpy() == 1, g.home_win.to_numpy() == 0)
    return edge, price, p_side, won


def kelly_run(edge, price, p_side, won, k, threshold):
    bankroll, peak, max_dd, n = 1.0, 1.0, 0.0, 0
    for e, o, ps, w in zip(edge, price, p_side, won):
        if e <= threshold:
            continue
        stake = bankroll * k * (ps * o - 1) / (o - 1)
        bankroll += stake * (o - 1) if w else -stake
        peak = max(peak, bankroll)
        max_dd = max(max_dd, 1 - bankroll / peak)
        n += 1
    return bankroll, max_dd, n


def flat_roi(edge, price, won, threshold, rng, n_boot=2000):
    m = edge > threshold
    if m.sum() < 10:
        return None
    profit = np.where(won[m], price[m] - 1, -1.0)
    boots = [profit[rng.integers(0, len(profit), len(profit))].mean() for _ in range(n_boot)]
    return {"n": int(m.sum()), "roi": float(profit.mean()), "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5))}


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def log_loss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def blend_test(g, key):
    """Fit w on one half of the games (by date), score on the other, both directions."""
    y = g.home_win.to_numpy(float)
    zm, zp = logit(g.p_market.to_numpy()), logit(g[f"p_{key}"].to_numpy())
    order = np.argsort(g.date.to_numpy(), kind="stable")
    half = len(order) // 2
    out = {}
    for name, (fit, test) in {"fit_early_test_late": (order[:half], order[half:]), "fit_late_test_early": (order[half:], order[:half])}.items():
        f = lambda w: log_loss(y[fit], 1 / (1 + np.exp(-(zm[fit] + w * (zp[fit] - zm[fit])))))
        w = minimize_scalar(f, bounds=(-1, 2), method="bounded").x
        p_blend = 1 / (1 + np.exp(-(zm[test] + w * (zp[test] - zm[test]))))
        out[name] = {"w": float(w), "market": log_loss(y[test], 1 / (1 + np.exp(-zm[test]))), "blend": log_loss(y[test], p_blend),
                     "model_alone": log_loss(y[test], 1 / (1 + np.exp(-zp[test])))}
    return out


def edge_calibration(g, key):
    """Bets with positive edge, bucketed by the model's edge: actual win rate vs the model's and the market's."""
    edge, price, p_side, won = bets(g, g[f"p_{key}"].to_numpy())
    home = g[f"p_{key}"].to_numpy() * g.home_odds.to_numpy() - 1 >= (1 - g[f"p_{key}"].to_numpy()) * g.away_odds.to_numpy() - 1
    market_side = np.where(home, g.p_market.to_numpy(), 1 - g.p_market.to_numpy())
    rows = []
    for lo, hi in [(0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 9)]:
        m = (edge > lo) & (edge <= hi)
        if m.sum():
            rows.append({"edge_bucket": f"{lo:.0%}-{'+' if hi > 1 else format(hi, '.0%')}", "bets": int(m.sum()),
                         "model_said": float(p_side[m].mean()), "market_said": float(market_side[m].mean()), "actual": float(won[m].mean())})
    return rows


def main():
    g, models = odds_games()
    rng = np.random.default_rng(0)
    out = {"n_games": len(g), "first": g.date.min(), "last": g.date.max(), "kelly": {}, "flat": {}, "blend": {}, "edge_calibration": {}}
    print(f"{len(g)} odds games, {g.date.min()} to {g.date.max()}\n")

    print("Final bankroll (start 1.00) by Kelly fraction k and minimum edge; (max drawdown, bets)")
    for key in models:
        edge, price, p_side, won = bets(g, g[f"p_{key}"].to_numpy())
        out["kelly"][key] = {}
        print(f"\n{key}")
        print("  edge>   " + "".join(f"k={k:<14}" for k in FRACTIONS))
        for t in THRESHOLDS:
            cells = []
            for k in FRACTIONS:
                b, dd, n = kelly_run(edge, price, p_side, won, k, t)
                out["kelly"][key][f"k{k}_t{t}"] = {"final": b, "max_drawdown": dd, "bets": n}
                cells.append(f"{b:5.2f} ({dd:.0%},{n:3d}) ")
            print(f"  {t:>5.0%}   " + "".join(f"{c:<15}" for c in cells))
        out["flat"][key] = {f"t{t}": flat_roi(edge, price, won, t, rng) for t in THRESHOLDS}

    # market as a betting model: bets only when the model's edge over the vig-laden price is positive
    print("\nFlat-stake ROI with 95% bootstrap interval, by minimum edge")
    for key in models:
        cells = []
        for t in THRESHOLDS:
            r = out["flat"][key][f"t{t}"]
            cells.append("n/a" if r is None else f"{100 * r['roi']:+5.1f}% [{100 * r['lo']:+.0f},{100 * r['hi']:+.0f}] n={r['n']}")
        print(f"  {key:28s} " + " | ".join(cells))

    print("\nMarket blend: does the model add anything to the market price? (log loss on the held-out half)")
    for key in models:
        out["blend"][key] = blend_test(g, key)
        for name, r in out["blend"][key].items():
            print(f"  {key:28s} {name:20s} w={r['w']:+.2f}  market {r['market']:.4f}  blend {r['blend']:.4f}  model alone {r['model_alone']:.4f}")

    for key in ("logreg", "dl_both_w12_pr_wide"):
        if key in models:
            out["edge_calibration"][key] = edge_calibration(g, key)
            print(f"\nEdge calibration, {key}:")
            for r in out["edge_calibration"][key]:
                print(f"  edge {r['edge_bucket']:>8s}  bets {r['bets']:3d}  model said {r['model_said']:.3f}  market said {r['market_said']:.3f}  actual {r['actual']:.3f}")
    (RESULTS_DIR / "betting_sweep.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
