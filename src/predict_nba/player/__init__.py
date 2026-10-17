"""
The player model in production: a Deep Sets network over each team's players, run in NumPy.

Data comes from ESPN, which answers from the cloud (stats.nba.com does not): box scores for each player's recent form,
and the injury list for who is out. The history it starts from (2020-21 to 2025-26) was built from stats.nba.com and
checked against ESPN's box scores for the same games; the two agree stat for stat.

Modules: espn (fetching), features (the same features the model was trained on), model (NumPy forward pass),
store (R2), runner (the 15-minute loop), export (the data the portfolio reads).
"""
