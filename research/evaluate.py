"""
The evaluation harness every model goes through.

A model hands over one number per game: the probability that the home team
wins. Everything here is computed from that number and the result, so the
logistic regression, the player model and Jev are scored identically.

Metrics:
- accuracy: share of games where the side with p >= 0.5 won. Crude, but the
  number people ask for first.
- Brier score: mean squared error of the probability. 0.25 is what a constant
  50% earns; lower is better.
- log loss: mean negative log-likelihood. Punishes confident mistakes much
  harder than Brier. This is what the models are trained and early-stopped on.
- ECE (expected calibration error): average gap between predicted probability
  and observed win rate across 10 probability bins, weighted by bin size.

Betting (only on games where odds were recorded):
- The market's own probability, de-vigged, is scored with the same metrics.
  This is the real benchmark: a model that cannot match the market's log loss
  has no information the market lacks.
- Flat-stake ROI for two fixed rules, set before looking at any results:
  "back the predicted winner" and "back any side with positive expected value".
- Quarter- and half-Kelly bankroll on the positive-EV rule, with max drawdown.
No confidence threshold is used anywhere: the only season with odds is the
test season, so there is no clean data to choose a threshold on.
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 1e-12


def classification_metrics(y, p):
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return {
        "n": int(len(y)),
        "accuracy": float(np.mean((p >= 0.5) == (y == 1))),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "ece": expected_calibration_error(y, p),
    }


def expected_calibration_error(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def calibration_table(y, p, bins=10):
    """Per-bin mean prediction, observed win rate and count, for plotting."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        if mask.sum() > 0:
            rows.append({"bin": b, "mean_pred": p[mask].mean(), "observed": y[mask].mean(), "count": int(mask.sum())})
    return pd.DataFrame(rows)


def plot_calibration(curves, path, title):
    """
    curves: {label: (y, p)}. Reliability diagram plus a histogram of predictions,
    so a well-calibrated but timid model (everything near 0.5) is visible as such.
    """
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(6.4, 7.2), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax.plot([0, 1], [0, 1], color="#9ca3af", lw=1, ls="--", label="perfect calibration")
    for label, (y, p) in curves.items():
        t = calibration_table(y, p)
        ax.plot(t.mean_pred, t.observed, marker="o", lw=1.8, label=label)
        axh.hist(p, bins=20, range=(0, 1), histtype="step", lw=1.6, label=label)
    ax.set_ylabel("Observed home win rate")
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    axh.set_xlabel("Predicted home win probability")
    axh.set_ylabel("Games")
    axh.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── Odds ──────────────────────────────────────────────────────────────────────

def devig(home_odds, away_odds):
    """
    Decimal odds -> the market's home win probability with the bookmaker margin
    removed. 1/odds overstates both sides (they sum to ~1.02-1.04); dividing by
    the sum spreads the margin proportionally. Simple, and fine for Pinnacle's
    low margins; Shin or power de-vigging would differ by tenths of a percent.
    """
    ih, ia = 1 / np.asarray(home_odds, float), 1 / np.asarray(away_odds, float)
    return ih / (ih + ia)


def betting_metrics(y, p, home_odds, away_odds):
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    oh, oa = np.asarray(home_odds, float), np.asarray(away_odds, float)

    # Rule 1: one unit on whichever side the model makes the favourite.
    back_home = p >= 0.5
    won = np.where(back_home, y == 1, y == 0)
    price = np.where(back_home, oh, oa)
    profit_winner = np.where(won, price - 1, -1.0)

    # Rule 2: one unit on a side only if model_p * odds > 1 (positive expected value).
    ev_home = p * oh - 1
    ev_away = (1 - p) * oa - 1
    side = np.where(ev_home >= ev_away, 1, 0)
    best_ev = np.maximum(ev_home, ev_away)
    bet = best_ev > 0
    won_ev = np.where(side == 1, y == 1, y == 0)
    price_ev = np.where(side == 1, oh, oa)
    profit_ev = np.where(bet, np.where(won_ev, price_ev - 1, -1.0), 0.0)

    out = {
        "n_odds_games": int(len(y)),
        "flat_winner_bets": int(len(y)),
        "flat_winner_roi": float(profit_winner.mean()),
        "flat_ev_bets": int(bet.sum()),
        "flat_ev_roi": float(profit_ev[bet].mean()) if bet.any() else 0.0,
    }

    # Kelly: stake the fraction (p*o - 1)/(o - 1) of the current bankroll, scaled down.
    p_side = np.where(side == 1, p, 1 - p)
    kelly_full = np.where(bet, (p_side * price_ev - 1) / (price_ev - 1), 0.0)
    for frac, name in [(0.25, "quarter_kelly"), (0.5, "half_kelly")]:
        bankroll, peak, max_dd, path = 1.0, 1.0, 0.0, []
        for f, w, o in zip(kelly_full * frac, won_ev, price_ev):
            stake = bankroll * f
            bankroll += stake * (o - 1) if w else -stake
            peak = max(peak, bankroll)
            max_dd = max(max_dd, 1 - bankroll / peak)
            path.append(bankroll)
        out[f"{name}_final"] = float(bankroll)
        out[f"{name}_max_drawdown"] = float(max_dd)
        out[f"{name}_path"] = path
    out["flat_winner_cum"] = np.cumsum(profit_winner).tolist()
    out["flat_ev_cum"] = np.cumsum(profit_ev).tolist()
    return out


def confidence_buckets(y, p, edges=(0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.0)):
    """Accuracy by how confident the model was in its pick. Fixed buckets, not tuned."""
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    conf = np.maximum(p, 1 - p)
    correct = (p >= 0.5) == (y == 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        rows.append({"bucket": f"{lo:.0%}-{hi:.0%}", "games": int(mask.sum()),
                     "accuracy": float(correct[mask].mean()) if mask.any() else None})
    return rows
