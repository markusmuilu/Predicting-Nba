"""
Step 5: the player-level model, a Deep Sets network in PyTorch.

A team is a *set* of players: there is no natural order to a roster, and the
size varies from game to game. Deep Sets (Zaheer et al., 2017) is the simplest
architecture that respects that: run every player through the same small
network (phi), then pool the results with an order-independent operation. Any
function of a set can be written as rho(pool(phi(x_i))), so nothing is lost in
principle by building it this way.

    player vector --phi--> 32-d embedding        (shared weights, every player)
    weighted mean over the roster                 (weights = expected minutes)
    -> team vector

Pooling is a weighted *mean*, so a team vector describes the minutes-weighted
typical player. The weights come from expected minutes, never the minutes
actually played (see player_features.py).

The match-up head is antisymmetric:

    logit = s(home, away) - s(away, home) + home_advantage

where s is one MLP that scores "team A against team B". Swapping the two
teams flips the sign of everything except the home-court term, so the model
cannot learn that, say, being listed first is good. It halves what the network
has to learn from ~5,000 training games, which matters at this size. The
alternative, a plain MLP over [home, away, home - away], is what the plan
suggested; it is kept as `antisymmetric=False` and compared on validation.
"""

import torch
from torch import nn


class DeepSetsMatchup(nn.Module):
    def __init__(self, n_player_features, n_context, embed_dim=32, hidden=64, dropout=0.2, antisymmetric=True):
        super().__init__()
        self.antisymmetric = antisymmetric

        # phi: applied to each player independently
        self.phi = nn.Sequential(
            nn.Linear(n_player_features, hidden), nn.ReLU(),
            nn.Linear(hidden, embed_dim), nn.ReLU(),
        )

        # Team "A vs B" input: A's vector, B's vector, their difference, both contexts.
        pair_in = 3 * embed_dim + 2 * n_context
        self.head = nn.Sequential(
            nn.Linear(pair_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.home_advantage = nn.Parameter(torch.zeros(1))

    def team_vector(self, x, w):
        """
        x: [batch, players, features], w: [batch, players] expected minutes (0 = empty slot).
        Returns [batch, embed_dim]: minutes-weighted mean of the player embeddings.
        """
        h = self.phi(x)                                   # [B, P, E]
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return (h * w.unsqueeze(-1)).sum(dim=1)           # empty slots have weight 0

    def score(self, a, b, ctx_a, ctx_b):
        return self.head(torch.cat([a, b, a - b, ctx_a, ctx_b], dim=-1)).squeeze(-1)

    def forward(self, x, w, ctx):
        """
        x: [B, 2, P, F] (index 0 = home), w: [B, 2, P], ctx: [B, 2, C].
        Returns the logit of a home win, [B].
        """
        home = self.team_vector(x[:, 0], w[:, 0])
        away = self.team_vector(x[:, 1], w[:, 1])
        s_home = self.score(home, away, ctx[:, 0], ctx[:, 1])
        if not self.antisymmetric:
            return s_home + self.home_advantage
        s_away = self.score(away, home, ctx[:, 1], ctx[:, 0])
        return s_home - s_away + self.home_advantage
