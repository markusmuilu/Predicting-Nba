"""
Tunes the two player-rating systems on the validation season and exports their
pre-game ratings for the combination and deep-learning experiments.

Every configuration tried is written to research/results/ratings_iterations.json,
not only the winner, so the amount of searching is visible. Choice rule, fixed in
advance: lowest 2024-25 log loss with the *previous-game* roster (the fair setting).
The test season plays no part in any choice here.

Run: python -m research.ratings_experiment
"""

import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from research.config import PROCESSED_DIR, RESULTS_DIR, SEED, TEST_SEASON, TRAIN_SEASONS, VAL_SEASON
from research.evaluate import classification_metrics
from research.player_ratings import prepare, run_player_elo, run_ridge_margin

_STATE = {}


def _init():
    games, rosters, n = prepare()
    _STATE.update(games=games, rosters=rosters, n=n,
                  y=games["home_win"].to_numpy(), val=(games.season == VAL_SEASON).to_numpy())


def _score_p1(params):
    g, r, n, y, val = (_STATE[k] for k in ("games", "rosters", "n", "y", "val"))
    dp, _, _ = run_player_elo(g, r, n, **params)
    p = 1.0 / (1.0 + 10.0 ** (-np.clip(dp + params["home"], -1500, 1500) / 400.0))
    return {**params, **{f"val_{k}": v for k, v in classification_metrics(y[val], p[val]).items()}}


def _score_p3(params):
    g, r, n, y = (_STATE[k] for k in ("games", "rosters", "n", "y"))
    pp, _ = run_ridge_margin(g, r, n, **params)
    tr = g.season.isin(TRAIN_SEASONS).to_numpy()
    lr = LogisticRegression(C=1e6).fit(pp[tr].reshape(-1, 1), y[tr])
    p = lr.predict_proba(pp.reshape(-1, 1))[:, 1]
    val = _STATE["val"]
    return {**params, **{f"val_{k}": v for k, v in classification_metrics(y[val], p[val]).items()}}


def sample_p1(n, seed):
    rng = np.random.default_rng(seed)
    return [{"K": float(np.exp(rng.uniform(np.log(1), np.log(60)))), "home": float(rng.uniform(0, 80)),
             "carry": float(rng.uniform(0.3, 0.95)), "r_new": float(rng.uniform(-60, 0)),
             "mov_exp": float(rng.choice([0.0, 0.8, 1.0, 1.2]))} for _ in range(n)]


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    workers = max(2, min(6, (os.cpu_count() or 4) // 2))
    p1_grid = sample_p1(240, SEED)
    p3_grid = [{"lam": lam, "half_life": hl} for lam, hl in itertools.product([10, 30, 100, 300, 1000], [400, 1000, 3000])]

    with ProcessPoolExecutor(workers, initializer=_init) as pool:
        print(f"P1: {len(p1_grid)} random configurations, {workers} workers", flush=True)
        p1 = list(pool.map(_score_p1, p1_grid, chunksize=4))
        best1 = min(p1, key=lambda r: r["val_log_loss"])
        print("P1 best:", {k: round(v, 3) for k, v in best1.items()}, flush=True)
        print(f"P3: {len(p3_grid)} grid points", flush=True)
        p3 = list(pool.map(_score_p3, p3_grid))
        best3 = min(p3, key=lambda r: r["val_log_loss"])
        print("P3 best:", {k: round(v, 4) for k, v in best3.items()}, flush=True)

    (RESULTS_DIR / "ratings_iterations.json").write_text(json.dumps({"p1_player_elo": p1, "p3_ridge_margin": p3}, indent=1))

    # Final runs with the chosen settings, exported for the next stage.
    games, rosters, n = prepare()
    keys = ["K", "home", "carry", "r_new", "mov_exp"]
    dp, da, _ = run_player_elo(games, rosters, n, **{k: best1[k] for k in keys})
    pp, pa = run_ridge_margin(games, rosters, n, lam=best3["lam"], half_life=best3["half_life"])
    out = games[["game_id", "date", "season", "home", "away", "home_win"]].assign(
        p1_prev=dp, p1_act=da, p3_prev=pp, p3_act=pa)
    out.to_csv(PROCESSED_DIR / "player_ratings.csv", index=False)
    (RESULTS_DIR / "ratings_chosen.json").write_text(json.dumps({"p1_player_elo": {k: best1[k] for k in keys},
                                                                  "p3_ridge_margin": {"lam": best3["lam"], "half_life": best3["half_life"]}}, indent=1))
    print("exported data/processed/player_ratings.csv", len(out), "games")


def refine_p3():
    """
    The first P3 grid (lam 10-1000) put its best point at lam=10, the smallest value tried.
    This extends the grid downwards, keeps the earlier results, and re-exports the ratings
    with the overall best setting by validation log loss.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    path = RESULTS_DIR / "ratings_iterations.json"
    log = json.loads(path.read_text())
    grid = [{"lam": lam, "half_life": hl} for lam, hl in itertools.product([1, 3, 10, 30], [600, 1000, 1600, 2500])]
    done = {(r["lam"], r["half_life"]) for r in log["p3_ridge_margin"]}
    grid = [g for g in grid if (g["lam"], g["half_life"]) not in done]
    with ProcessPoolExecutor(max(2, min(6, (os.cpu_count() or 4) // 2)), initializer=_init) as pool:
        new = list(pool.map(_score_p3, grid))
    log["p3_ridge_margin"] += new
    path.write_text(json.dumps(log, indent=1))
    best = min(log["p3_ridge_margin"], key=lambda r: r["val_log_loss"])
    print("P3 best after refinement:", {k: round(v, 4) for k, v in best.items()}, flush=True)

    games, rosters, n = prepare()
    pp, pa = run_ridge_margin(games, rosters, n, lam=best["lam"], half_life=best["half_life"])
    out = pd.read_csv(PROCESSED_DIR / "player_ratings.csv", dtype={"game_id": str})
    assert (out.game_id.values == games.game_id.values).all()
    out["p3_prev"], out["p3_act"] = pp, pa
    out.to_csv(PROCESSED_DIR / "player_ratings.csv", index=False)
    chosen = json.loads((RESULTS_DIR / "ratings_chosen.json").read_text())
    chosen["p3_ridge_margin"] = {"lam": best["lam"], "half_life": best["half_life"]}
    (RESULTS_DIR / "ratings_chosen.json").write_text(json.dumps(chosen, indent=1))


if __name__ == "__main__":
    import sys
    refine_p3() if "refine" in sys.argv else main()
