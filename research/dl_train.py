"""
Deep-learning experiments on the raw-sequence dataset (dl_data.py).

One network family, several switches, so that each experiment changes one thing:

  encoder   "hf"   MLP over the 22 hand-built features
            "seq"  GRU over the raw 20-game window (no hand-built features at all)
            "both" the two embeddings concatenated
  wide      extra game-level scalars (team Elo difference, player-rating differences) added
            straight to the logit through a linear layer ("wide and deep"), so the network
            does not have to rediscover them
  pr        each player's own pre-game ratings (player Elo and ridge coefficient) appended to his
            hand-built features, so the network can learn how far to trust a rating given the rest of the
            player's recent form
  trend     what lets the model see *time*, only through the home-court term:
            "none" | "year" (season start year, standardised) | "league" (league points per game
            and home win rate so far this season) | recency weights on the training games
  window    how many training seasons before 2024-25 are used (4, 8, 12)

Shared with the earlier Deep Sets model: minutes-weighted mean pooling over the roster,
the antisymmetric home/away head, BCE-with-logits, Adam, early stopping on validation log
loss. Selection is by validation log loss only. Every trained configuration is logged.
"""

import json
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from research.config import PROCESSED_DIR, RESULTS_DIR, TEST_SEASON, VAL_SEASON
from research.evaluate import classification_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_EPOCHS, PATIENCE, BATCH = 200, 15, 256


class SetsNet(nn.Module):
    def __init__(self, encoder, n_hf, n_ch, n_ctx, n_wide, n_trend, embed=16, hidden=32, gru_hidden=32, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        d = 0
        if encoder in ("hf", "both"):
            self.hf = nn.Sequential(nn.Linear(n_hf, hidden), nn.ReLU(), nn.Linear(hidden, embed), nn.ReLU())
            d += embed
        if encoder in ("seq", "both"):
            self.gru = nn.GRU(n_ch, gru_hidden, batch_first=True)
            self.gru_out = nn.Sequential(nn.Linear(gru_hidden, embed), nn.ReLU())
            d += embed
        self.head = nn.Sequential(nn.Linear(3 * d + 2 * n_ctx, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.wide = nn.Linear(n_wide, 1, bias=False) if n_wide else None
        self.trend = nn.Linear(n_trend, 1) if n_trend else None
        self.home = nn.Parameter(torch.zeros(1))

    def players(self, hf, seq):
        B, S, P = hf.shape[:3]
        parts = []
        if self.encoder in ("hf", "both"):
            parts.append(self.hf(hf))
        if self.encoder in ("seq", "both"):
            _, h = self.gru(seq.reshape(B * S * P, seq.shape[3], seq.shape[4]))
            parts.append(self.gru_out(h[-1]).reshape(B, S, P, -1))
        return torch.cat(parts, dim=-1)

    def forward(self, hf, seq, exp, ctx, wide, trend):
        z = self.players(hf, seq)                                  # [B, 2, P, d]
        w = exp / exp.sum(2, keepdim=True).clamp(min=1e-6)         # expected-minutes weights, 0 for empty slots
        team = (z * w.unsqueeze(-1)).sum(2)                        # [B, 2, d]
        a, b = team[:, 0], team[:, 1]
        s_ab = self.head(torch.cat([a, b, a - b, ctx[:, 0], ctx[:, 1]], -1)).squeeze(-1)
        s_ba = self.head(torch.cat([b, a, b - a, ctx[:, 1], ctx[:, 0]], -1)).squeeze(-1)
        logit = s_ab - s_ba + self.home
        if self.wide is not None:
            logit = logit + self.wide(wide).squeeze(-1)
        if self.trend is not None:
            logit = logit + self.trend(trend).squeeze(-1)
        return logit


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Data:
    """All arrays on the GPU, plus masks for the split and the game-level scalar inputs."""

    def __init__(self, mode):
        self.mode = mode
        g = pd.read_csv(PROCESSED_DIR / "dl_games.csv", dtype={"game_id": str})
        self.games = g
        t = lambda name, dtype=torch.float32: torch.tensor(np.load(PROCESSED_DIR / f"dl_{mode}_{name}.npy")).to(DEVICE, dtype)
        self.hf, self.exp, self.ctx = t("hf"), t("exp"), t("ctx")
        self.seq = torch.tensor(np.load(PROCESSED_DIR / f"dl_{mode}_seq.npy")).to(DEVICE)       # float16 on device
        self.y = torch.tensor(g["home_win"].to_numpy(np.float32)).to(DEVICE)

        elo = pd.read_csv(PROCESSED_DIR / "team_elo.csv", dtype={"game_id": str})[["game_id", "elo_home", "elo_away"]]
        rat = pd.read_csv(PROCESSED_DIR / "player_ratings.csv", dtype={"game_id": str})
        p1col, p3col = ("p1_prev", "p3_prev") if mode == "previous" else ("p1_act", "p3_act")
        m = g[["game_id"]].merge(elo, on="game_id", how="left").merge(rat[["game_id", p1col, p3col]], on="game_id", how="left")
        self.wide_all = torch.tensor(np.stack([(m.elo_home - m.elo_away).fillna(0) / 100.0,
                                               m[p1col].fillna(0) / 100.0, m[p3col].fillna(0) / 5.0], 1).astype(np.float32)).to(DEVICE)
        sstart = g["season_start"].to_numpy()
        train_ref = sstart < int(VAL_SEASON[:4])
        self.pr = None
        snap1, snap3 = PROCESSED_DIR / "snap_p1.npy", PROCESSED_DIR / "snap_p3.npy"
        pidx_path = PROCESSED_DIR / f"dl_{mode}_pidx.npy"
        if snap1.exists() and snap3.exists() and pidx_path.exists():
            pidx = np.load(pidx_path)
            rows, pi = np.arange(len(g))[:, None, None], np.clip(pidx, 0, None)
            slots = []
            for snap in (np.load(snap1), np.load(snap3)):
                r = np.where(pidx >= 0, snap[rows, pi], 0.0)
                use = (pidx >= 0) & train_ref[:, None, None]       # standardise with pre-validation slots only
                r = np.where(pidx >= 0, (r - r[use].mean()) / r[use].std(), 0.0)
                slots.append(r)
            self.pr = torch.tensor(np.stack(slots, -1).astype(np.float32)).to(DEVICE)
        yr = (sstart - sstart[train_ref].mean()) / sstart[train_ref].std()
        ppg = (g.league_ppg - g.league_ppg[train_ref].mean()) / g.league_ppg[train_ref].std()
        hw = (g.league_home_win - g.league_home_win[train_ref].mean()) / g.league_home_win[train_ref].std()
        self.trend_all = {"none": None, "year": torch.tensor(yr[:, None].astype(np.float32)).to(DEVICE),
                          "league": torch.tensor(np.stack([ppg, hw], 1).astype(np.float32)).to(DEVICE)}
        self.sstart = sstart
        self.val = (g.season == VAL_SEASON).to_numpy()
        self.test = (g.season == TEST_SEASON).to_numpy()

    def train_mask(self, n_seasons):
        first = int(VAL_SEASON[:4]) - n_seasons
        return (self.sstart >= first) & (self.sstart < int(VAL_SEASON[:4]))


def batch(D, idx, wide_on, trend, pr_on=False):
    seq = D.seq[idx].float()
    hf = torch.cat([D.hf[idx], D.pr[idx]], -1) if pr_on else D.hf[idx]
    return (hf, seq, D.exp[idx], D.ctx[idx],
            D.wide_all[idx] if wide_on else None, D.trend_all[trend][idx] if trend != "none" else None)


def predict(model, D, idx):
    model.eval()
    out = []
    with torch.no_grad():
        for part in torch.split(torch.as_tensor(idx, device=DEVICE), 1024):
            out.append(model(*batch(D, part, model.wide is not None, model.trend_name, model.pr_on)))
    return torch.cat(out)


def train_one(D, cfg, seed):
    """Returns (model, best validation log loss, best epoch)."""
    set_seed(seed)
    wide_on, trend, pr_on = cfg.get("wide", False), cfg.get("trend", "none"), cfg.get("pr", False)
    n_trend = 0 if trend == "none" else D.trend_all[trend].shape[1]
    model = SetsNet(cfg["encoder"], D.hf.shape[-1] + (D.pr.shape[-1] if pr_on else 0), D.seq.shape[-1], D.ctx.shape[-1], 3 if wide_on else 0, n_trend,
                    cfg.get("embed", 16), cfg.get("hidden", 32), cfg.get("gru_hidden", 32), cfg.get("dropout", 0.1)).to(DEVICE)
    model.trend_name, model.pr_on = trend, pr_on
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("wd", 1e-3))
    train_idx = np.where(D.train_mask(cfg.get("window", 4)))[0]
    val_idx = np.where(D.val)[0]

    # Recency weights: older seasons count less. tau in seasons; None = all equal.
    tau = cfg.get("recency_tau")
    weights = torch.ones(len(D.y), device=DEVICE)
    if tau:
        age = (int(VAL_SEASON[:4]) - 1) - D.sstart
        weights = torch.tensor(np.exp(-np.clip(age, 0, None) / tau).astype(np.float32)).to(DEVICE)

    gen = torch.Generator().manual_seed(seed)
    best, best_state, best_epoch, waited = 1e9, None, 0, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.as_tensor(train_idx)[torch.randperm(len(train_idx), generator=gen)]
        for part in torch.split(perm, BATCH):
            part = part.to(DEVICE)
            opt.zero_grad()
            logit = model(*batch(D, part, wide_on, trend, pr_on))
            loss = (nn.functional.binary_cross_entropy_with_logits(logit, D.y[part], reduction="none") * weights[part]).sum() / weights[part].sum()
            loss.backward()
            opt.step()
        with torch.no_grad():
            vl = nn.functional.binary_cross_entropy_with_logits(predict(model, D, val_idx), D.y[val_idx]).item()
        if vl < best - 1e-5:
            best, best_epoch, waited = vl, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best, best_epoch


def fit_temperature(logits, y):
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
    logits, y = logits.detach().cpu(), y.detach().cpu()

    def closure():
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(logits / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp())


def run_experiment(D, name, base_cfg, search, seeds=(42, 43, 44, 45, 46), log=None, log_path=None):
    """
    search: list of hyperparameter overrides tried with seed 42; the best (validation log loss)
    is retrained with all seeds, logits averaged, temperature fitted on validation.
    Writes research/results/pred_dl_<name>.csv for validation and test games.
    """
    t0 = time.time()
    tried = []
    for extra in search:
        cfg = {**base_cfg, **extra}
        _, vl, ep = train_one(D, cfg, seeds[0])
        tried.append({"cfg": cfg, "val_log_loss": vl, "epoch": ep})
    best = min(tried, key=lambda r: r["val_log_loss"])
    cfg = best["cfg"]
    models = [train_one(D, cfg, s)[0] for s in seeds]
    idx = np.where(D.val | D.test)[0]
    with torch.no_grad():
        logits = torch.stack([predict(m, D, idx) for m in models]).mean(0)
    val_pos = torch.as_tensor(D.val[idx], device=DEVICE)
    T = fit_temperature(logits[val_pos], D.y[torch.as_tensor(idx, device=DEVICE)][val_pos])
    p = torch.sigmoid(logits / T).cpu().numpy()
    games = D.games.iloc[idx][["game_id", "date", "season", "home", "away", "home_win"]].assign(p_home=p)
    games.to_csv(RESULTS_DIR / f"pred_dl_{name}.csv", index=False)
    vm = classification_metrics(games.home_win[games.season == VAL_SEASON], games.p_home[games.season == VAL_SEASON])
    rec = {"name": name, "cfg": cfg, "temperature": T, "val": vm, "n_tried": len(tried), "tried": tried,
           "seconds": round(time.time() - t0)}
    print(f"{name:34s} val log loss {vm['log_loss']:.4f}  acc {vm['accuracy']:.3f}  (T={T:.2f}, {len(tried)} configs, {rec['seconds']}s)", flush=True)
    if log is not None:
        log.append(rec)
        (log_path or RESULTS_DIR / "dl_iterations.json").write_text(json.dumps(log, indent=1))
    return rec


SEARCH_HF = [{"embed": 16, "hidden": 32, "dropout": 0.1}, {"embed": 16, "hidden": 32, "dropout": 0.3},
             {"embed": 32, "hidden": 64, "dropout": 0.2, "wd": 1e-2}, {"embed": 8, "hidden": 16, "dropout": 0.1}]
SEARCH_SEQ = [{"gru_hidden": 32, "embed": 16, "dropout": 0.1}, {"gru_hidden": 32, "embed": 16, "dropout": 0.3},
              {"gru_hidden": 64, "embed": 32, "dropout": 0.2, "wd": 1e-2}, {"gru_hidden": 16, "embed": 8, "dropout": 0.1}]


def main(mode="previous", only=None):
    D = Data(mode)
    suffix = "" if mode == "previous" else f"_{mode}"
    log_path = RESULTS_DIR / f"dl_iterations{suffix}.json"
    # Append to an existing record rather than replacing it (an earlier run overwrote its own log this way).
    log = json.loads(log_path.read_text()) if log_path.exists() else []
    R = lambda name, cfg, search: run_experiment(D, name + suffix, cfg, search, log=log, log_path=log_path)
    print("device", DEVICE, "| games", len(D.y), "| roster mode", mode)

    if only == "pr":
        # Each player's own Elo and ridge rating as an input, alone and with the game-level scalars.
        R("hf_w12_pr", {"encoder": "hf", "window": 12, "pr": True}, SEARCH_HF)
        R("hf_w12_pr_wide", {"encoder": "hf", "window": 12, "pr": True, "wide": True}, SEARCH_HF)
        R("both_w12_pr_wide", {"encoder": "both", "window": 12, "pr": True, "wide": True}, SEARCH_SEQ)
        R("both_w12_pr_wide_league", {"encoder": "both", "window": 12, "pr": True, "wide": True, "trend": "league"}, SEARCH_SEQ)
        return
    # 1. How much history helps, with hand-built features (the earlier model's setup).
    for w in (4, 8, 12):
        R(f"hf_w{w}", {"encoder": "hf", "window": w}, SEARCH_HF)
    # 2. Does the network need the features built for it? Raw sequences, no hand features.
    for w in (4, 12):
        R(f"seq_w{w}", {"encoder": "seq", "window": w}, SEARCH_SEQ)
        R(f"both_w{w}", {"encoder": "both", "window": w}, SEARCH_SEQ)
    # 3. Elo-type scalars fed in directly.
    R("hf_w12_wide", {"encoder": "hf", "window": 12, "wide": True}, SEARCH_HF)
    R("both_w12_wide", {"encoder": "both", "window": 12, "wide": True}, SEARCH_SEQ)
    # 4. Letting the model see time.
    base = {"encoder": "hf", "window": 12, "wide": True}
    R("hf_w12_wide_year", {**base, "trend": "year"}, SEARCH_HF)
    R("hf_w12_wide_league", {**base, "trend": "league"}, SEARCH_HF)
    R("hf_w12_wide_recency", {**base, "recency_tau": 3.0}, SEARCH_HF)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "previous", sys.argv[2] if len(sys.argv) > 2 else None)
