"""
Scores every model's predictions with the shared harness and writes the
results table, the plots, and a JSON file the dashboard reads.

Fairness rule: all models are compared on the *same games*. The production
cleaner drops each team's first game of a season (no rolling average exists
yet), so the logistic regression has slightly fewer games than the player
model; the comparison uses the games every model has.

Two reference rows that are not models trained here:
- market: Pinnacle's price de-vigged, on the games where the production
  pipeline recorded odds (from 7 Jan 2026 on). Prices were captured at the
  daily 12:00 Helsinki run, i.e. hours before tip-off, so they are not closing
  lines; closing lines would be sharper still.
- production (live): what the deployed service actually predicted on the day,
  from history/prediction_history.json. Four model versions over the season.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research.config import RAW_DIR, RESULTS_DIR, TEST_SEASON, VAL_SEASON
from research.evaluate import (betting_metrics, classification_metrics, confidence_buckets, devig,
                               plot_calibration)

MODELS = {
    "logreg": ("Logistic regression (production features)", "pred_logreg.csv"),
    "player_model": ("Player model, actual roster (optimistic)", "pred_player_model.csv"),
    "player_logreg": ("Ablation: logreg on player features, actual roster", "pred_player_logreg.csv"),
    "player_model_prev": ("Player model, previous-game roster", "pred_player_model_previous.csv"),
    "player_logreg_prev": ("Ablation: logreg on player features, previous-game roster", "pred_player_logreg_previous.csv"),
    "jev_named": ("Jev, real names, previous-game roster", "pred_jev_named_previous.csv"),
    "jev_anonymised": ("Jev, anonymised, previous-game roster", "pred_jev_anonymised_previous.csv"),
    "jev_named_cal": ("Jev, real names, recalibrated on validation", None),
    "jev_anonymised_cal": ("Jev, anonymised, recalibrated on validation", None),
    # Elo and rating systems (research/elo.py, player_ratings.py, neural_elo.py, combine.py)
    "team_elo": ("Team Elo (settings tuned on validation)", "pred_team_elo.csv"),
    "neural_elo_plain": ("Neural Elo: K, home bonus, carryover learned by gradient", "pred_neural_elo_plain.csv"),
    "neural_elo": ("Neural Elo with a learned update network", "pred_neural_elo.csv"),
    "player_elo_prev": ("Player Elo, previous-game roster", "pred_player_elo_prev.csv"),
    "ridge_prev": ("Ridge plus-minus, previous-game roster", "pred_ridge_prev.csv"),
    "player_elo_act": ("Player Elo, actual roster (optimistic)", "pred_player_elo_act.csv"),
    "ridge_act": ("Ridge plus-minus, actual roster (optimistic)", "pred_ridge_act.csv"),
    "comb_scalars": ("Logistic regression on 7 rating scalars (team Elo, player Elo, ridge, rest)", "pred_comb_scalars.csv"),
    "comb_prod_elo": ("Production features + team Elo", "pred_comb_prod+elo.csv"),
    "comb_prod_ratings": ("Production features + player Elo and ridge", "pred_comb_prod+ratings.csv"),
    "comb_prod_all": ("Production features + all rating scalars", "pred_comb_prod+all.csv"),
    "comb_all_actual": ("Production features + rating scalars, actual roster (optimistic)", "pred_comb_all_actual.csv"),
}
KEY = ["date", "home", "away"]

# Shown in the plots and the headline dashboard table. Everything else is in the full tables.
FEATURED = ["logreg", "team_elo", "ridge_prev", "comb_scalars", "player_model_prev", "dl_hf_w12", "dl_seq_w12",
            "dl_both_w12_pr_wide_league", "jev_named_cal"]


def dl_label(name):
    """pred_dl_<name>.csv -> readable label, e.g. both_w12_wide -> 'Deep Sets, hand features + raw sequences, 12 seasons, + rating scalars'."""
    actual = name.endswith("_actual")
    parts = name.replace("_actual", "").split("_")
    enc = {"hf": "hand-built features", "seq": "raw sequences (GRU)", "both": "hand features + raw sequences"}[parts[0]]
    label = f"Deep Sets, {enc}, {parts[1][1:]} seasons"
    extras = {"wide": "+ rating scalars", "year": "+ season year", "league": "+ league context", "recency": "+ recency weights", "pr": "+ per-player Elo/ridge ratings",
              "margin": "trained on margin only", "wm": "trained on win + margin", "wmt": "trained on win + margin + total points"}
    for tag in parts[2:]:
        label += f", {extras[tag]}"
    return label + (", actual roster (optimistic)" if actual else "")


def discover_dl_models():
    out = {}
    for path in sorted(RESULTS_DIR.glob("pred_dl_*.csv")):
        name = path.stem[len("pred_dl_"):]
        out[f"dl_{name}"] = (dl_label(name), path.name)
    return out


def platt_recalibrate(df):
    """
    Jev's raw probabilities are overconfident. Fit p' = sigmoid(a * logit(p) + b) on the
    validation season only (two numbers) and apply it unchanged to the test season. The
    validation rows are scored in-sample, so validation numbers for these rows are slightly
    optimistic; test numbers are not.
    """
    from sklearn.linear_model import LogisticRegression
    d = df.copy()
    p = d["p_home"].clip(0.01, 0.99)
    z = np.log(p / (1 - p)).to_numpy().reshape(-1, 1)
    val = (d["season"] == VAL_SEASON).to_numpy()
    lr = LogisticRegression(C=1e6).fit(z[val], d.loc[val, "home_win"])
    d["p_home"] = lr.predict_proba(z)[:, 1]
    return d


def all_models():
    return {**MODELS, **discover_dl_models()}


def load_predictions():
    preds = {}
    for key, (_, fname) in all_models().items():
        if fname is None:
            continue
        path = RESULTS_DIR / fname
        if not path.exists():
            continue
        df = pd.read_csv(path).rename(columns={"Date": "date"})
        preds[key] = df[KEY + ["season", "home_win", "p_home"]]
    for key in ("jev_named", "jev_anonymised"):
        if key in preds:
            preds[key + "_cal"] = platt_recalibrate(preds[key])
    return preds


def load_live_history():
    """Production's own day-of predictions and the odds stored with them."""
    h = pd.DataFrame(json.loads((RAW_DIR / "history_prediction_history.json").read_text()))
    h = h[~h.team.isin(["STARS", "STRIPES", "WORLD", "NO_GAMES_TODAY"])]
    h = h[h["confidence"].notna()]  # early-season rows hold a result but no stored prediction
    conf = h["confidence"].astype(float) / 100
    h["p_live"] = np.where(h["prediction"].astype(bool), conf, 1 - conf)
    h = h.rename(columns={"team": "home", "opponent": "away"})
    return h[KEY + ["p_live", "home_odds", "away_odds"]].drop_duplicates(KEY)


def common_games(preds, season):
    """Inner join of every model's games for one season, one p_<model> column each."""
    base = None
    for key, df in preds.items():
        part = df[df.season == season][KEY + ["home_win", "p_home"]].rename(columns={"p_home": f"p_{key}"})
        base = part if base is None else base.merge(part.drop(columns="home_win"), on=KEY)
    return base.sort_values(KEY).reset_index(drop=True)


def paired_bootstrap(y, p_base, p_new, n_boot=2000, seed=0):
    """
    Per-game log loss of the baseline minus the new model, resampled over games.
    Positive = new model is better. Returns the mean difference and a 95% interval.
    Games are resampled, not seasons, so this captures sampling noise on ~1,200 games
    and nothing else (not seed noise, not the choice of season).
    """
    y = np.asarray(y, float)
    def ll(p):
        p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    d = ll(p_base) - ll(p_new)
    rng = np.random.default_rng(seed)
    means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean_diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi)}


def plot_bankroll(curves, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, series in curves.items():
        ax.plot(range(1, len(series) + 1), series, lw=1.6, label=label)
    ax.axhline(1.0, color="#9ca3af", lw=1, ls="--")
    ax.set_xlabel("Games with odds, in date order (test season)")
    ax.set_ylabel("Bankroll (start = 1)")
    ax.set_title("Quarter-Kelly on positive-EV bets")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run():
    preds = load_predictions()
    live = load_live_history()
    names = {k: all_models()[k][0] for k in preds}
    out = {"models": names, "splits": {}}

    for season, label in [(VAL_SEASON, "validation"), (TEST_SEASON, "test")]:
        games = common_games(preds, season)
        y = games.home_win.to_numpy()
        split = {"season": season, "n_games": len(games),
                 "home_win_rate": float(y.mean()), "metrics": {}, "buckets": {}}
        for k in preds:
            split["metrics"][k] = classification_metrics(y, games[f"p_{k}"])
            split["buckets"][k] = confidence_buckets(y, games[f"p_{k}"])
        split["paired_vs_logreg"] = {k: paired_bootstrap(y, games["p_logreg"], games[f"p_{k}"])
                                     for k in preds if k != "logreg"}
        # The production logistic regression turned out to be a weak baseline (a 7-number model and
        # plain team Elo both beat it), so every model is also compared with the 7-number model.
        if "comb_scalars" in preds:
            split["paired_vs_scalars"] = {k: paired_bootstrap(y, games["p_comb_scalars"], games[f"p_{k}"])
                                          for k in preds if k != "comb_scalars"}
        if "jev_named" in preds and "jev_anonymised" in preds:
            # Memorisation check: positive = real names help. Raw and recalibrated.
            split["jev_names_vs_anonymised"] = {
                "raw": paired_bootstrap(y, games["p_jev_anonymised"], games["p_jev_named"]),
                "recalibrated": paired_bootstrap(y, games["p_jev_anonymised_cal"], games["p_jev_named_cal"]),
            }
        split["metrics"]["always_home"] = classification_metrics(y, np.full(len(y), y.mean()))

        plot_calibration({names[k]: (y, games[f"p_{k}"].to_numpy()) for k in FEATURED if k in preds},
                         RESULTS_DIR / f"calibration_{label}.png",
                         f"Calibration, {label} season {season} ({len(games)} games)")

        if season == TEST_SEASON:
            g = games.merge(live, on=KEY, how="left")
            has_live = g.p_live.notna()
            split["live_production"] = {
                "n_matched": int(has_live.sum()),
                "production_live": classification_metrics(g.home_win[has_live], g.p_live[has_live]),
                **{k: classification_metrics(g.home_win[has_live], g[f"p_{k}"][has_live]) for k in preds},
            }

            odds = g[g.home_odds.notna() & g.away_odds.notna()].reset_index(drop=True)
            yo = odds.home_win.to_numpy()
            market = devig(odds.home_odds, odds.away_odds)
            odds_block = {"n_games": len(odds), "first_date": odds.date.min(), "last_date": odds.date.max(),
                          "market": classification_metrics(yo, market), "models": {}, "betting": {}}
            curves = {}
            for k in preds:
                odds_block["models"][k] = classification_metrics(yo, odds[f"p_{k}"])
                bm = betting_metrics(yo, odds[f"p_{k}"], odds.home_odds, odds.away_odds)
                if k in FEATURED:
                    curves[names[k]] = bm["quarter_kelly_path"]
                odds_block["betting"][k] = {kk: v for kk, v in bm.items() if not isinstance(v, list)}
            split["odds"] = odds_block
            plot_bankroll(curves, RESULTS_DIR / "bankroll_test.png")

            # Per-game test predictions for the dashboard's comparison view.
            g["p_market"] = np.where(g.home_odds.notna(), devig(g.home_odds.fillna(2), g.away_odds.fillna(2)), np.nan)
            cols = KEY + ["home_win"] + [f"p_{k}" for k in preds] + ["p_live", "p_market"]
            out["test_games"] = json.loads(g[cols].round(4).to_json(orient="records"))

        out["splits"][label] = split

    (RESULTS_DIR / "metrics.json").write_text(json.dumps(out, indent=2, default=float))
    write_table(out)
    return out


def write_table(out):
    lines = []
    for label, split in out["splits"].items():
        lines.append(f"\n### {label.capitalize()}: {split['season']}, {split['n_games']} games "
                     f"(home win rate {split['home_win_rate']:.3f})\n")
        lines.append("| Model | Accuracy | Brier | Log loss | ECE |")
        lines.append("|---|---|---|---|---|")
        for k, m in split["metrics"].items():
            name = out["models"].get(k, "Constant: home win rate")
            lines.append(f"| {name} | {m['accuracy']:.3f} | {m['brier']:.4f} | {m['log_loss']:.4f} | {m['ece']:.3f} |")
        if "odds" in split:
            o = split["odds"]
            lines.append(f"\n**Games with recorded Pinnacle odds:** {o['n_games']} ({o['first_date']} to {o['last_date']})\n")
            lines.append("| Model | Accuracy | Brier | Log loss | Flat ROI, back pick | +EV bets | Flat ROI, +EV | 1/4 Kelly final | 1/4 Kelly max DD | 1/2 Kelly final |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|")
            m = o["market"]
            lines.append(f"| Market (de-vigged) | {m['accuracy']:.3f} | {m['brier']:.4f} | {m['log_loss']:.4f} | | | | | | |")
            for k, m in o["models"].items():
                b = o["betting"][k]
                lines.append(f"| {out['models'][k]} | {m['accuracy']:.3f} | {m['brier']:.4f} | {m['log_loss']:.4f} | "
                             f"{b['flat_winner_roi']:+.3f} | {b['flat_ev_bets']} | {b['flat_ev_roi']:+.3f} | "
                             f"{b['quarter_kelly_final']:.3f} | {b['quarter_kelly_max_drawdown']:.3f} | {b['half_kelly_final']:.3f} |")
    for label, split in out["splits"].items():
        lines.append(f"\n**{label.capitalize()}: log loss improvement over production logreg, paired bootstrap 95% CI** (positive = better)\n")
        lines.append("| Model | Mean | 95% CI |")
        lines.append("|---|---|---|")
        for k, b in split["paired_vs_logreg"].items():
            lines.append(f"| {out['models'][k]} | {b['mean_diff']:+.4f} | [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}] |")
    for label, split in out["splits"].items():
        if "paired_vs_scalars" in split:
            lines.append(f"\n**{label.capitalize()}: log loss improvement over the 7-number logistic regression, paired bootstrap 95% CI** (positive = better)\n")
            lines.append("| Model | Mean | 95% CI |")
            lines.append("|---|---|---|")
            for k, b in split["paired_vs_scalars"].items():
                lines.append(f"| {out['models'][k]} | {b['mean_diff']:+.4f} | [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}] |")
    for label, split in out["splits"].items():
        if "jev_names_vs_anonymised" in split:
            lines.append(f"\n**{label.capitalize()}: Jev with real names vs anonymised, log loss gain from names, paired bootstrap 95% CI**\n")
            lines.append("| Version | Mean | 95% CI |")
            lines.append("|---|---|---|")
            for k, b in split["jev_names_vs_anonymised"].items():
                lines.append(f"| {k} | {b['mean_diff']:+.4f} | [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}] |")
    (RESULTS_DIR / "results_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    run()
