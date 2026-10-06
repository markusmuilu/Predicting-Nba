"""
Linear check of the margin question: for the same seven rating scalars, is it better to fit a win/loss
logistic regression or a regression on the point margin (turned into a probability afterwards)?

Margin route: ridge regression of the home margin on the scalars (alpha chosen on 2024-25), then
P(home win) = Phi(predicted margin / sigma) with sigma the spread of the training residuals. No fitting on
validation beyond alpha. Win route: the logistic regression of research/combine.py ("scalars").
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from research.combine import load
from research.config import TEST_SEASON, TRAIN_SEASONS, VAL_SEASON
from research.evaluate import classification_metrics


def main():
    df, _ = load()
    games = pd.read_csv("data/processed/dl_games.csv", dtype={"game_id": str})[["game_id", "PTS_home", "PTS_away"]]
    df = df.merge(games, on="game_id")
    df["margin"] = df.PTS_home - df.PTS_away
    cols = ["elo_diff", "p1_prev", "p3_prev", "rest_h", "b2b_h", "rest_a", "b2b_a"]
    tr = df.season.isin(TRAIN_SEASONS).to_numpy()
    scaler = StandardScaler().fit(df.loc[tr, cols])
    X = scaler.transform(df[cols].fillna(0))
    val, test = (df.season == VAL_SEASON).to_numpy(), (df.season == TEST_SEASON).to_numpy()
    best = None
    for alpha in (1, 10, 100, 1000, 10000):
        m = Ridge(alpha=alpha).fit(X[tr], df.margin[tr])
        sigma = float(np.std(df.margin[tr] - m.predict(X[tr])))
        p = norm.cdf(m.predict(X) / sigma)
        ll = classification_metrics(df.home_win[val], p[val])["log_loss"]
        if best is None or ll < best[0]:
            best = (ll, alpha, sigma, p)
    ll, alpha, sigma, p = best
    print(f"margin route: alpha {alpha}, residual sigma {sigma:.2f} points")
    for name, mask in (("validation", val), ("test", test)):
        mm = classification_metrics(df.home_win[mask], p[mask])
        print(f"  {name:10s} log loss {mm['log_loss']:.4f}  accuracy {mm['accuracy']:.3f}  brier {mm['brier']:.4f}")
    wins = pd.read_csv("research/results/pred_comb_scalars.csv")
    for name, s in (("validation", VAL_SEASON), ("test", TEST_SEASON)):
        w = wins[wins.season == s]
        mm = classification_metrics(w.home_win, w.p_home)
        print(f"win route ({name}): log loss {mm['log_loss']:.4f}  accuracy {mm['accuracy']:.3f}  (logistic regression, 7 scalars)")


if __name__ == "__main__":
    main()
