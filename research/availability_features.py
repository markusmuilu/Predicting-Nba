"""
Steps 4 and 5 of the brief, second half: availability-aware team strength for every game and cutoff, and how well
the minutes forecast works.

Input: data/processed/avail_listed.pkl (research/availability.py: every listed player with P(plays)), the previous-game
rosters and expected minutes of the deep-learning dataset (dl_previous_*.npy), and the per-player rating snapshots
(snap_p1.npy player Elo, snap_p3.npy ridge plus-minus), all point-in-time.

For each game, side and cutoff:
  candidates  = the previous game's ten main players + every player listed on the report (with an id)
  P(plays)    = from the report for listed players; for unlisted candidates a logistic regression on expected minutes
                (fitted on the training seasons, 2019-20 to 2023-24)
  minutes     = lambda * P * expected minutes, with lambda = previous-roster minutes / sum of P * expected minutes,
                clipped to [0.6, 1.5], so that absent players' minutes pass to the rest and a returning star's minutes
                come from the rest
  strength    = sum of (5 * minutes / total minutes) * rating, for player Elo (A1) and ridge plus-minus (A3)

The minutes forecast is scored on validation and test: mean absolute error of each candidate's minutes against what he
played (0 when he did not play), for three forecasts: previous-game roster with expected minutes (the baseline),
the same with P(plays) as a multiplier, and the full forecast above.

Output: data/processed/avail_features.csv.gz (one row per game and cutoff), research/results/availability_minutes.json.

Run: python -m research.availability_features
"""

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from research.availability import LAMBDA_CAP, TRAIN
from research.config import PROCESSED_DIR, RESULTS_DIR, TEST_SEASON, VAL_SEASON
from research.dl_data import load_all
from research.player_ratings import usable_seasons

CUTOFFS = (60, 30, 15)


def main():
    seasons = usable_seasons()
    games, plog, _ = load_all(seasons)
    dl_games = pd.read_csv(PROCESSED_DIR / "dl_games.csv", dtype={"game_id": str})
    assert (dl_games.game_id.values == games.game_id.values).all(), "game order differs from the deep-learning dataset"
    PIDX = np.load(PROCESSED_DIR / "dl_previous_pidx.npy")
    EXP = np.load(PROCESSED_DIR / "dl_previous_exp.npy")
    S1, S3 = np.load(PROCESSED_DIR / "snap_p1.npy", mmap_mode="r"), np.load(PROCESSED_DIR / "snap_p3.npy", mmap_mode="r")
    listed = pd.read_pickle(PROCESSED_DIR / "avail_listed.pkl")
    minutes = {(g, p): m for g, p, m in zip(plog.GAME_ID, plog.pidx, plog.MIN)}

    # Unlisted candidates: P(plays) from expected minutes, fitted on training seasons (cutoff T-30 listings)
    key30 = set(zip(listed[listed.cutoff == 30].g, listed[listed.cutoff == 30].pidx))
    rows = []
    for g in np.where(games.season.isin(TRAIN).to_numpy())[0]:
        gid = games.game_id.iloc[g]
        for side in (0, 1):
            for k in range(10):
                p = PIDX[g, side, k]
                if p >= 0 and (g, p) not in key30:
                    rows.append((EXP[g, side, k], (gid, p) in minutes))
    un = pd.DataFrame(rows, columns=["exp", "played"])
    unlisted_model = LogisticRegression(C=1e6).fit(np.log1p(un[["exp"]]), un.played)
    print(f"unlisted candidates (training): {len(un):,}, share who played {un.played.mean():.3f}", flush=True)

    by_game_cut = {k: g for k, g in listed.groupby(["g", "cutoff"])}
    out, minute_rows = [], []
    for g in sorted(listed.g.unique()):
        gid, season = games.game_id.iloc[g], games.season.iloc[g]
        for cut in CUTOFFS:
            snap = by_game_cut.get((g, cut))
            if snap is None:
                continue
            row = {"g": g, "game_id": gid, "season": season, "cutoff": cut}
            for side in (0, 1):
                prev = {int(PIDX[g, side, k]): float(EXP[g, side, k]) for k in range(10) if PIDX[g, side, k] >= 0}
                mine = snap[snap.side == side]
                lst = {int(r.pidx): (float(r.exp_min), float(r.p_play)) for r in mine.itertuples()}
                cand = {}
                for p, e in prev.items():
                    cand[p] = (e, lst[p][1] if p in lst else float(unlisted_model.predict_proba(np.log1p([[e]]))[0, 1]))
                for p, (e, pp) in lst.items():
                    if p not in cand and e >= 5.0:                      # a listed player not in the previous roster (returning, or traded in)
                        cand[p] = (e, pp)
                if not cand:
                    continue
                ids = np.array(list(cand))
                e = np.array([cand[p][0] for p in ids])
                pp = np.array([cand[p][1] for p in ids])
                base_total = sum(prev.values()) or e.sum()
                lam = float(np.clip(base_total / max((pp * e).sum(), 1e-6), 0.6, LAMBDA_CAP))
                m = lam * pp * e
                w = 5.0 * m / max(m.sum(), 1e-6)
                r1, r3 = np.asarray(S1[g, ids]), np.asarray(S3[g, ids])
                tag = "home" if side == 0 else "away"
                row[f"a1_{tag}"], row[f"a3_{tag}"] = float((w * r1).sum()), float((w * r3).sum())
                row[f"min_lost_{tag}"] = float(sum(e_ * (1 - p_) for p, (e_, p_) in cand.items() if p in prev))
                row[f"n_out_{tag}"] = int(sum(1 for p, (e_, p_) in cand.items() if p_ < 0.2 and e_ >= 12))
                row[f"lambda_{tag}"] = lam
                # minutes forecast accuracy (validation and test only)
                if season in (VAL_SEASON, TEST_SEASON) and cut == 30:
                    actual = np.array([minutes.get((gid, int(p)), 0.0) for p in ids])
                    pe = np.array([prev.get(int(p), 0.0) for p in ids])             # previous roster, expected minutes, nobody absent
                    minute_rows.append({"season": season, "mae_baseline": np.abs(pe - actual).mean(),
                                        "mae_with_p": np.abs(pp * e - actual).mean(), "mae_full": np.abs(m - actual).mean(),
                                        "n": len(ids), "total_pred": m.sum(), "total_actual": actual.sum()})
            out.append(row)
    feats = pd.DataFrame(out)
    feats.to_csv(PROCESSED_DIR / "avail_features.csv.gz", index=False)
    mr = pd.DataFrame(minute_rows)
    summary = {s: {"mae_baseline": float(g.mae_baseline.mean()), "mae_with_p": float(g.mae_with_p.mean()), "mae_full": float(g.mae_full.mean()),
                   "games": int(len(g) // 1)} for s, g in mr.groupby("season")}
    (RESULTS_DIR / "availability_minutes.json").write_text(json.dumps(summary, indent=1))
    print(f"{len(feats):,} game-cutoff rows; minutes forecast error (mean absolute error per candidate, minutes):")
    print(pd.DataFrame(summary).T.round(3).to_string())


if __name__ == "__main__":
    main()
