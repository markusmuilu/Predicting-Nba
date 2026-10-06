"""
Per-player rating snapshots, so the deep model can use each player's own Elo-style rating.

Re-runs the two player-rating systems with the settings already chosen on validation
(research/results/ratings_chosen.json) and records, for every game, every player's rating
as it stood at the start of that game:

  data/processed/snap_p1.npy   player Elo rating, Elo points            [games, players]
  data/processed/snap_p3.npy   ridge plus-minus coefficient, points      [games, players]

The rows follow the same game order and the same player indexing as the deep-learning
dataset (research/dl_data.py); a check below confirms the indexing agrees.
"""

import json

import numpy as np

from research.config import PROCESSED_DIR, RESULTS_DIR
from research.dl_data import load_all
from research.player_ratings import prepare, run_player_elo, run_ridge_margin, usable_seasons
from research.fetch import nba_game_log
import pandas as pd


def main():
    chosen = json.loads((RESULTS_DIR / "ratings_chosen.json").read_text())
    games, rosters, n = prepare()

    # The player index in player_ratings.prepare and in dl_data.load_all must be the same mapping.
    seasons = usable_seasons()
    plog = pd.concat([nba_game_log(s, "P") for s in seasons], ignore_index=True)
    plog = plog[plog["MIN"].fillna(0) > 0]
    index_a = {pid: k for k, pid in enumerate(plog["PLAYER_ID"].unique())}
    _, plog_b, _ = load_all(seasons)
    index_b = dict(zip(plog_b["PLAYER_ID"], plog_b["pidx"]))
    assert index_a == index_b, "player indexing differs between the rating engines and the DL dataset"
    dl_games = pd.read_csv(PROCESSED_DIR / "dl_games.csv", dtype={"game_id": str})
    assert (dl_games.game_id.values == games.game_id.values).all(), "game order differs"

    snap1, snap3 = [], []
    run_player_elo(games, rosters, n, snapshots=snap1, **chosen["p1_player_elo"])
    run_ridge_margin(games, rosters, n, snapshots=snap3, **chosen["p3_ridge_margin"])
    np.save(PROCESSED_DIR / "snap_p1.npy", np.stack(snap1))
    np.save(PROCESSED_DIR / "snap_p3.npy", np.stack(snap3))
    print("snapshots saved:", len(snap1), "games x", n, "players")


if __name__ == "__main__":
    main()
