"""
Leakage test for the Finnish rating models: a game's prediction may only depend on EARLIER games.

Changing the results of the last 150 games must leave every earlier prediction unchanged, for both rating
families. If this fails, the model has look-ahead.
"""

import numpy as np
import pytest

from research.finland import model as M

pytestmark = pytest.mark.skipif(not M.DATA.exists(), reason="needs data/finland/matches.csv (run research.finland.scrape)")

PARAMS = {"elo": {"K": 12.0, "home": 40.0, "carry": 0.5, "scale": 400.0, "mov_exp": 0.8, "init_ia": -50.0, "init_ib": -60.0},
          "margin": {"K": 0.1, "home": 3.0, "carry": 0.5, "scale": 13.0, "init_ia": -5.0, "init_ib": -6.0}}


@pytest.mark.parametrize("family", ["elo", "margin"])
def test_predictions_do_not_depend_on_later_results(family):
    games = M.load_games("national")
    base, _, _ = M.run(games, family, PARAMS[family], "mix")
    changed = games.copy()
    last = changed.index[changed.finished][-150:]
    changed.loc[last, ["home_pts", "away_pts", "m_final", "m_q3", "m_half", "m_mix", "m_final_reg"]] = changed.loc[last, ["away_pts", "home_pts", "m_final", "m_q3", "m_half", "m_mix", "m_final_reg"]].to_numpy() * np.array([1, 1, -1, -1, -1, -1, -1])
    changed.loc[last, "home_win"] = 1 - changed.loc[last, "home_win"]
    again, _, _ = M.run(changed, family, PARAMS[family], "mix")
    first_changed = changed.index.get_loc(last[0])
    assert np.allclose(base[:first_changed + 1], again[:first_changed + 1])      # the first altered game itself is predicted before it is played


def test_probabilities_are_valid_and_home_advantage_helps():
    games = M.load_games("m2d")
    prob, _, _ = M.run(games, "margin", PARAMS["margin"], "final")
    assert ((prob > 0) & (prob < 1)).all()
    assert prob[games.finished.to_numpy()].mean() > 0.5
