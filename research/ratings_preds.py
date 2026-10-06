"""
Turns each rating scalar into a standalone probability so it can be scored by the shared harness.

p1_prev / p1_act   player Elo team-strength difference (previous-game / actual roster)
p3_prev / p3_act   ridge plus-minus predicted margin (previous-game / actual roster)

Each is mapped to a home win probability by a one-variable logistic regression (a slope and an
intercept) fitted on the TRAINING seasons only, so validation and test are untouched.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression

from research.config import PROCESSED_DIR, RESULTS_DIR, TEST_SEASON, TRAIN_SEASONS, VAL_SEASON
from research.evaluate import classification_metrics


def main():
    rat = pd.read_csv(PROCESSED_DIR / "player_ratings.csv", dtype={"game_id": str})
    tr = rat.season.isin(TRAIN_SEASONS)
    for col, name in [("p1_prev", "player_elo_prev"), ("p1_act", "player_elo_act"),
                      ("p3_prev", "ridge_prev"), ("p3_act", "ridge_act")]:
        lr = LogisticRegression(C=1e6).fit(rat.loc[tr, [col]], rat.home_win[tr])
        rat["p_home"] = lr.predict_proba(rat[[col]])[:, 1]
        keep = rat.season.isin([VAL_SEASON, TEST_SEASON])
        rat.loc[keep, ["date", "season", "home", "away", "home_win", "p_home"]].to_csv(RESULTS_DIR / f"pred_{name}.csv", index=False)
        v = rat[rat.season == VAL_SEASON]
        m = classification_metrics(v.home_win, v.p_home)
        print(f"{name:18s} val log loss {m['log_loss']:.4f}  acc {m['accuracy']:.3f}")


if __name__ == "__main__":
    main()
