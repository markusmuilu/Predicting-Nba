"""
Neural Elo: an Elo system whose settings are learned by gradient descent.

Elo is a recursion: ratings -> win probability -> result -> new ratings. Every step is
differentiable, so the numbers a person normally tunes on a grid (K, home bonus, season
carryover) can be learned by back-propagating the log loss through the sequence of games.
Going one step further, the *size of the update* can be a small network instead of a
constant K:

    shift = K * softplus(net(features)) * (result - expected)

where the features are things Elo ignores: how big the margin was, how lopsided the matchup
was, how far into the season it is, both teams' rest. The network can learn, for example, that
early-season games should move ratings more, or that blowouts carry less extra information
than the standard margin multiplier assumes.

Training protocol, matching everything else here:
- ratings are re-simulated from scratch each epoch over the training seasons
  (gradients flow within a season; ratings carry across seasons but gradients are cut, a
  truncated back-propagation that keeps memory flat);
- the first training season is burn-in and is not scored;
- the loss is log loss on the training seasons, early stopping on 2024-25 log loss;
- validation and test games are simulated by continuing the same recursion with fixed settings.

Games on one date are processed together (a team plays at most once per day), which makes an
epoch about 160 steps per season rather than 1,230.
"""

import json

import numpy as np
import pandas as pd
import torch
from torch import nn

from research.config import PROCESSED_DIR, RESULTS_DIR, SEED, TEST_SEASON, VAL_SEASON
from research.evaluate import classification_metrics

MEAN_RATING = 1505.0


class NeuralElo(nn.Module):
    def __init__(self, hidden=16, use_net=True):
        super().__init__()
        self.log_k = nn.Parameter(torch.tensor(np.log(12.0), dtype=torch.float32))
        self.home = nn.Parameter(torch.tensor(30.0))
        self.carry_logit = nn.Parameter(torch.tensor(0.0))           # sigmoid(0) = 0.5, as the grid found
        self.home_trend = nn.Parameter(torch.zeros(1))               # home bonus shift with league home win rate
        self.use_net = use_net
        if use_net:
            self.net = nn.Sequential(nn.Linear(6, hidden), nn.Tanh(), nn.Linear(hidden, 1))
            nn.init.zeros_(self.net[2].weight)                        # starts as plain constant-K Elo
            nn.init.zeros_(self.net[2].bias)

    def simulate(self, day_batches, n_teams, grad_until_season):
        """
        day_batches: list of dicts of tensors per date. Returns probabilities for all games in
        the order they were processed (a permutation recorded in `order`).
        """
        R = torch.full((n_teams,), 1500.0)
        seen = torch.zeros(n_teams, dtype=torch.bool)
        probs, order, last_season = [], [], None
        carry = torch.sigmoid(self.carry_logit)
        for b in day_batches:
            if b["season"] != last_season:
                # Cut the gradient to the previous season's ratings (truncated back-propagation) but
                # keep it for the carry multiplier itself, so carryover is learned from this season on.
                R = carry * R.detach() + (1 - carry) * MEAN_RATING
                last_season = b["season"]
            h, a = b["h"], b["a"]
            d = R[h] + self.home + self.home_trend * b["league_hw"] - R[a]
            p = torch.sigmoid(d * (np.log(10.0) / 400.0))
            result = b["y"]
            sign = 2 * result - 1
            if self.use_net:
                f = torch.stack([b["margin"] / 10.0, (sign * d).clamp(-600, 600) / 200.0, b["progress"], b["rest_h"], b["rest_a"], b["b2b"]], 1)
                scale = nn.functional.softplus(self.net(f).squeeze(-1) + 0.5413)   # softplus(0.5413) = 1
            else:
                scale = torch.ones_like(p)
            # Fixed margin multiplier of the 538 form with exponent 1 gives the net something to correct.
            mult = (b["margin"] + 3.0) / (7.5 + 0.006 * (sign * d).detach())
            shift = self.log_k.exp() * scale * mult * (result - p)
            R = R.index_add(0, h, shift).index_add(0, a, -shift)
            probs.append(p)
            order.append(b["row"])
        return torch.cat(probs), torch.cat(order)


def make_batches(games, ctx, n_teams_map):
    team_idx = n_teams_map
    seasons = list(dict.fromkeys(games.season))
    batches = []
    league_hw = games["league_home_win"].to_numpy()
    margin = (games.PTS_home - games.PTS_away).abs().to_numpy(np.float32)
    start_of_season = games.groupby("season")["date"].transform("min")
    day_in_season = (pd.to_datetime(games["date"]) - pd.to_datetime(start_of_season)).dt.days.to_numpy()
    for date, grp in games.groupby("date", sort=True):
        rows = grp.index.to_numpy()
        season = grp.season.iloc[0]
        batches.append({
            "season": season, "season_idx": seasons.index(season), "row": torch.tensor(rows),
            "h": torch.tensor([team_idx[t] for t in grp.home]), "a": torch.tensor([team_idx[t] for t in grp.away]),
            "y": torch.tensor(grp.home_win.to_numpy(np.float32)), "margin": torch.tensor(margin[rows]),
            "league_hw": torch.tensor(((league_hw[rows] - 0.58) * 10).astype(np.float32)),
            "progress": torch.tensor((day_in_season[rows] / 200.0).astype(np.float32)),
            "rest_h": torch.tensor(ctx[rows, 0, 0]), "rest_a": torch.tensor(ctx[rows, 1, 0]),
            "b2b": torch.tensor(ctx[rows, 0, 1] - ctx[rows, 1, 1]),
        })
    return batches, seasons


def run(epochs=120, patience=15, use_net=True, name="neural_elo"):
    torch.manual_seed(SEED)
    games = pd.read_csv(PROCESSED_DIR / "dl_games.csv", dtype={"game_id": str})
    ctx = np.load(PROCESSED_DIR / "dl_previous_ctx.npy")
    team_idx = {t: k for k, t in enumerate(sorted(set(games.home) | set(games.away)))}
    batches, seasons = make_batches(games, ctx, team_idx)
    train_seasons = [s for s in seasons if int(s[:4]) < int(VAL_SEASON[:4])]
    burn_in = train_seasons[0]
    val_idx, grad_until = seasons.index(VAL_SEASON), seasons.index(VAL_SEASON) - 1

    y_all = torch.tensor(games.home_win.to_numpy(np.float32))
    season_of = games.season.to_numpy()
    train_rows = torch.tensor(np.where(np.isin(season_of, train_seasons) & (season_of != burn_in))[0])
    val_rows = torch.tensor(np.where(season_of == VAL_SEASON)[0])

    model = NeuralElo(use_net=use_net)
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    best, best_state, waited, history = 1e9, None, 0, []
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        p, order = model.simulate(batches, len(team_idx), grad_until)
        full = torch.empty(len(games)); full[order] = p
        loss = nn.functional.binary_cross_entropy(full[train_rows].clamp(1e-6, 1 - 1e-6), y_all[train_rows])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        with torch.no_grad():
            vl = nn.functional.binary_cross_entropy(full[val_rows].detach().clamp(1e-6, 1 - 1e-6), y_all[val_rows]).item()
        history.append({"epoch": epoch, "train": float(loss), "val": vl})
        if vl < best - 1e-5:
            best, waited = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break
    model.load_state_dict(best_state)
    with torch.no_grad():
        p, order = model.simulate(batches, len(team_idx), -1)
    full = np.empty(len(games)); full[order.numpy()] = p.numpy()
    games["p_home"] = full
    keep = games.season.isin([VAL_SEASON, TEST_SEASON])
    games.loc[keep, ["game_id", "date", "season", "home", "away", "home_win", "p_home"]].to_csv(RESULTS_DIR / f"pred_{name}.csv", index=False)
    v = games[games.season == VAL_SEASON]
    learned = {"K": float(model.log_k.exp()), "home": float(model.home), "carry": float(torch.sigmoid(model.carry_logit)),
               "home_trend": float(model.home_trend), "use_net": use_net}
    summary = {"learned": learned, "val": classification_metrics(v.home_win, v.p_home), "epochs_run": len(history),
               "best_epoch": int(np.argmin([h["val"] for h in history]))}
    print(name, learned, {k: round(x, 4) for k, x in summary["val"].items() if k != "n"}, "epochs", len(history), flush=True)
    return summary


def main():
    out = {"constant_k_learned": run(use_net=False, name="neural_elo_plain"), "with_update_net": run(use_net=True, name="neural_elo")}
    (RESULTS_DIR / "neural_elo_summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
