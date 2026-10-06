"""
Steps 5 and 6: train the Deep Sets model, then calibrate it.

Procedure, in the order it runs:
1. A small grid of settings, one seed each, trained on 2020-21..2023-24 and
   early-stopped on 2024-25 log loss. The best setting is picked by 2024-25
   log loss. The test season is not loaded into the model until step 4.
2. The chosen setting is retrained with five seeds; their logits are averaged.
   Averaging seeds removes most of the run-to-run noise a network this size
   has on ~5,000 games, so the result does not depend on a lucky seed.
3. Temperature scaling on 2024-25: one number T, logits divided by T, fitted
   by minimising validation log loss. It cannot change which side is favoured
   (accuracy is unchanged); it only stretches or shrinks the probabilities.
4. Predict 2025-26 once and write the probabilities out.

Also trains an ablation: plain logistic regression on the minutes-weighted
average of the same player features. If that does as well as the network, the
gain (or loss) comes from the player data, not from the deep model.

Validation is used twice, for early stopping and for temperature. That is
standard and the leak is small (one scalar and a stopping epoch), but it means
the validation numbers are slightly optimistic. Test numbers are not.
"""

import itertools
import json
import random

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import nn

from research.config import PROCESSED_DIR, RESULTS_DIR, SEED, TEST_SEASON, TRAIN_SEASONS, VAL_SEASON
from research.evaluate import classification_metrics
from research.model import DeepSetsMatchup

GRID = {
    "embed_dim": [16, 32],
    "hidden": [32, 64],
    "dropout": [0.1, 0.3],
    "weight_decay": [1e-4, 1e-3],
    "antisymmetric": [True, False],
}
LR = 1e-3
BATCH = 128
MAX_EPOCHS = 300
PATIENCE = 25
FINAL_SEEDS = [42, 43, 44, 45, 46]


def pick_device():
    """CUDA if it exists *and* runs a kernel; new GPUs can have a build mismatch."""
    if torch.cuda.is_available():
        try:
            torch.ones(1, device="cuda").add_(1)
            return torch.device("cuda")
        except Exception as e:
            print(f"CUDA present but unusable ({e}); using CPU")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load(dataset="player_dataset"):
    d = np.load(PROCESSED_DIR / f"{dataset}.npz")
    games = pd.read_csv(PROCESSED_DIR / "games.csv", dtype={"game_id": str})
    split = {
        "train": games.season.isin(TRAIN_SEASONS).to_numpy(),
        "val": (games.season == VAL_SEASON).to_numpy(),
        "test": (games.season == TEST_SEASON).to_numpy(),
    }
    return d, games, split


def tensors(d, mask, device):
    return [torch.tensor(d[k][mask], device=device) for k in ("X", "W", "CTX", "y")]


def val_log_loss(model, val):
    model.eval()
    with torch.no_grad():
        logits = model(*val[:3])
        return nn.functional.binary_cross_entropy_with_logits(logits, val[3]).item()


def train_one(cfg, seed, train, val, n_feat, n_ctx, device):
    set_seed(seed)
    model = DeepSetsMatchup(n_feat, n_ctx, cfg["embed_dim"], cfg["hidden"], cfg["dropout"], cfg["antisymmetric"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=cfg["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(seed)

    best, best_state, best_epoch, waited = float("inf"), None, 0, 0
    n = len(train[3])
    for epoch in range(MAX_EPOCHS):
        model.train()
        for idx in torch.randperm(n, generator=gen).split(BATCH):
            idx = idx.to(device)
            opt.zero_grad()
            loss = loss_fn(model(train[0][idx], train[1][idx], train[2][idx]), train[3][idx])
            loss.backward()
            opt.step()

        current = val_log_loss(model, val)
        if current < best - 1e-5:
            best, best_epoch, waited = current, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, best, best_epoch


def fit_temperature(logits, y):
    """One scalar T > 0 minimising log loss of logits / T. Optimised in log space."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)
    logits, y = logits.detach().cpu(), y.detach().cpu()

    def closure():
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(logits / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp())


def player_logreg(d, games, split, tag):
    """Ablation: same player features, minutes-weighted average per team, linear model."""
    w = d["W"] / np.clip(d["W"].sum(axis=2, keepdims=True), 1e-6, None)
    team_mean = (d["X"] * w[..., None]).sum(axis=2)                     # [games, 2, F]
    feats = np.concatenate([team_mean[:, 0] - team_mean[:, 1], d["CTX"][:, 0], d["CTX"][:, 1]], axis=1)
    y = d["y"]

    scaler = StandardScaler().fit(feats[split["train"]])
    Xs = scaler.transform(feats)
    best = None
    for C in [0.001, 0.01, 0.1, 1.0]:
        m = LogisticRegression(C=C, max_iter=2000).fit(Xs[split["train"]], y[split["train"]])
        ll = classification_metrics(y[split["val"]], m.predict_proba(Xs[split["val"]])[:, 1])["log_loss"]
        if best is None or ll < best[0]:
            best = (ll, C, m)
    _, C, m = best
    p = m.predict_proba(Xs)[:, 1]
    keep = split["val"] | split["test"]
    out = games.loc[keep, ["game_id", "date", "season", "home", "away", "home_win"]].assign(p_home=p[keep])
    out.to_csv(RESULTS_DIR / f"pred_player_logreg{tag}.csv", index=False)
    print(f"Player-feature logistic regression: C={C}, val log loss {best[0]:.4f}")
    return {"C": C, "val_log_loss": best[0]}


def run(roster_mode="actual"):
    """roster_mode "actual" or "previous": which dataset from player_features.build to use."""
    tag = "" if roster_mode == "actual" else f"_{roster_mode}"
    device = pick_device()
    print("Device:", device, "| roster:", roster_mode)
    d, games, split = load(f"player_dataset{tag}")
    n_feat, n_ctx = d["X"].shape[-1], d["CTX"].shape[-1]
    train, val = tensors(d, split["train"], device), tensors(d, split["val"], device)

    # 1. Grid search, one seed, scored on validation only.
    grid_results = []
    for values in itertools.product(*GRID.values()):
        cfg = dict(zip(GRID.keys(), values))
        _, ll, epoch = train_one(cfg, SEED, train, val, n_feat, n_ctx, device)
        grid_results.append({**cfg, "val_log_loss": ll, "best_epoch": epoch})
        print(f"{cfg} -> val log loss {ll:.4f} (epoch {epoch})")
    grid_results.sort(key=lambda r: r["val_log_loss"])
    best_cfg = {k: grid_results[0][k] for k in GRID}
    print("Chosen:", best_cfg)

    # 2. Five seeds of the chosen setting, logits averaged.
    models, seed_val = [], []
    for s in FINAL_SEEDS:
        m, ll, epoch = train_one(best_cfg, s, train, val, n_feat, n_ctx, device)
        models.append(m)
        seed_val.append({"seed": s, "val_log_loss": ll, "best_epoch": epoch})

    def ensemble_logits(mask):
        x, w, ctx, _ = tensors(d, mask, device)
        with torch.no_grad():
            return torch.stack([m.eval()(x, w, ctx) for m in models]).mean(0).cpu()

    # 3. Temperature on validation.
    val_logits = ensemble_logits(split["val"])
    T = fit_temperature(val_logits, torch.tensor(d["y"][split["val"]]))
    print(f"Temperature: {T:.3f}")

    # 4. Test season, once.
    keep = split["val"] | split["test"]
    logits = ensemble_logits(keep)
    out = games.loc[keep, ["game_id", "date", "season", "home", "away", "home_win"]].copy()
    out["p_home_raw"] = torch.sigmoid(logits).numpy()
    out["p_home"] = torch.sigmoid(logits / T).numpy()
    out.to_csv(RESULTS_DIR / f"pred_player_model{tag}.csv", index=False)

    ablation = player_logreg(d, games, split, tag)
    summary = {"device": str(device), "chosen": best_cfg, "temperature": T, "seed_runs": seed_val,
               "grid": grid_results, "player_logreg": ablation,
               "n_parameters": sum(p.numel() for p in models[0].parameters())}
    (RESULTS_DIR / f"player_model_training{tag}.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "actual")
