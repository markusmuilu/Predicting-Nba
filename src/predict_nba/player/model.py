"""
The Deep Sets player model's forward pass in NumPy, so the app needs no PyTorch.

Weights are the five trained seeds exported from PyTorch (research/live_model.py on the next-season branch). In
evaluation mode dropout does nothing, so the forward pass is: phi on every player, a minutes-weighted mean per team,
an antisymmetric head on the pair, the mean of the five seeds' logits, divided by the fitted temperature.
"""

import io
import json

import numpy as np


def relu(x):
    return np.maximum(x, 0)


class PlayerModel:
    def __init__(self, weights_npz: bytes, meta_json: bytes):
        self.meta = json.loads(meta_json)
        w = np.load(io.BytesIO(weights_npz))
        self.seeds = []
        for s in self.meta["seeds"]:
            g = lambda k: w[f"s{s}.{k}"]
            self.seeds.append({k: g(k) for k in ("phi.0.weight", "phi.0.bias", "phi.2.weight", "phi.2.bias",
                                                 "head.0.weight", "head.0.bias", "head.3.weight", "head.3.bias",
                                                 "home_advantage")})
        self.antisymmetric = self.meta["config"]["antisymmetric"]
        self.temperature = self.meta["temperature"]

    @staticmethod
    def _team(p, x, w):
        h = relu(x @ p["phi.0.weight"].T + p["phi.0.bias"])
        h = relu(h @ p["phi.2.weight"].T + p["phi.2.bias"])
        w = w / np.clip(w.sum(axis=1, keepdims=True), 1e-6, None)
        return (h * w[..., None]).sum(axis=1)

    @staticmethod
    def _score(p, a, b, ca, cb):
        z = np.concatenate([a, b, a - b, ca, cb], axis=-1)
        z = relu(z @ p["head.0.weight"].T + p["head.0.bias"])
        return (z @ p["head.3.weight"].T + p["head.3.bias"])[..., 0]

    def logit(self, x, w, ctx):
        """x: [B, 2, P, F], w: [B, 2, P], ctx: [B, 2, C]. Home is index 0."""
        out = []
        for p in self.seeds:
            home, away = self._team(p, x[:, 0], w[:, 0]), self._team(p, x[:, 1], w[:, 1])
            s = self._score(p, home, away, ctx[:, 0], ctx[:, 1])
            if self.antisymmetric:
                s = s - self._score(p, away, home, ctx[:, 1], ctx[:, 0])
            out.append(s + p["home_advantage"][0])
        return np.mean(out, axis=0) / self.temperature

    def p_home(self, x, w, ctx):
        return 1 / (1 + np.exp(-self.logit(x, w, ctx)))
