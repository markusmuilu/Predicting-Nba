# Player-level model, evaluation and dashboard

Written 2026-10-06 for Markus, to be read after the fact. It explains what was built and why, what the numbers say and
do not say, what broke, and what you would need to understand to defend this in an interview.
Branch: `player-model` in this repo, `static-site` in `nba-dashboard`. Nothing was pushed to `main` and the production
pipeline and deploy workflows were not touched.

## 1. The short version

- **The production logistic regression, re-measured on a fixed split** (train 2020-21 to 2023-24, validate 2024-25,
  test 2025-26): 66.7% accuracy, Brier 0.2093, log loss 0.6057 on the test season.
- **A PyTorch player-level model (Deep Sets) beats it, but how much depends on one assumption.**
  Told which players actually played in the game (the plan's step 4, optimistic): log loss 0.5782, accuracy 69.3%.
  Using only the roster of the team's *previous* game, which is knowable before tip-off: log loss 0.5895, accuracy 67.6%.
  The second is the honest number for live use. It is about 0.016 log loss better than the baseline on test
  (95% interval +0.004 to +0.029), but only +0.005 on validation (interval -0.007 to +0.018, includes zero).
- **The first Deep Sets model was not what helped (superseded in part by section 9).** A plain logistic regression on the same player features does as well
  (test log loss 0.5936 vs 0.5895, validation 0.6019 vs 0.6024). The gain comes from the player data, not from the deep model.
- **The optimistic version looks like it matches the betting market and it does not mean that.** On the 680 games with stored
  odds, the actual-roster model has log loss 0.5427 against the market's 0.5432. That is because it knows who played and
  the market, priced hours earlier, did not. The same model with a previous-game roster scores 0.5573, clearly worse than the market.
  Its +6.8x quarter-Kelly bankroll is an artefact, not an edge.
- **Jev (Part 2) was run in a second session**, once the key was available (it had not been in the environment the first time).
  4,894 requests, 8.9 million input tokens, no failures. Its raw probabilities are overconfident and score worse than the logistic
  regression (test log loss 0.6416 with real names, 0.6583 anonymised, against 0.6057). Recalibrated with two numbers fitted on the
  validation season only, it scores 0.5964 (names) and 0.5985 (anonymised): level with the logistic regression (+0.009, interval
  -0.002 to +0.020), a little behind the player model (0.5895), and clearly behind the market. Real names help by about 0.017 raw but
  only 0.002 after recalibration, so most of that raw gap is calibration, not memorised results (section 8).
- **Second session: Elo, player ratings and deep learning (section 9).** Plain team Elo, one number per team, beats the production regression on test (log loss 0.5925 vs 0.6057).
  A 7-number logistic regression on team Elo, player Elo, a ridge plus-minus rating and rest scores 0.5897, and is as good as anything simple. The best deep model, fed each player's own Elo and ridge ratings plus raw
  20-game sequences, scores 0.5821 to 0.5853 (two near-tied variants), which is about 0.005 to 0.008 better than the 7-number model, with intervals that
  straddle zero or just clear it. Still behind the market on the 680 odds games (about 0.555 vs 0.543).
  Raw sequences matched hand-built features; more seasons helped deep models and not linear ones; a season-year input did not help; a learned-parameter "neural Elo" confirmed the hand-tuned grid.
- **The Fly.io service is alive** but has stopped saving predictions (section 11).
- **The dashboard is now a static site** rebuilt daily by a GitHub Action (section 10). It needs three manual steps from you to go live.

## 2. What is where

Predicting-Nba, branch `player-model`, everything new under `research/` (outside `src/`, so the Docker image does not install it):

| File | What it does |
|---|---|
| `research/config.py` | The seasons, the split, the seed. One place, so scripts cannot disagree. |
| `research/fetch.py` | Cached downloads: PBPStats team logs and games (what production uses), `nba_api` team and player game logs. Pauses between requests. |
| `research/baseline.py` | Step 1. Runs the production `DataCleaner` and production feature list unchanged, trains the production model type on the fixed split. |
| `research/player_features.py` | Steps 3 and 4. Point-in-time player features and rosters. |
| `research/model.py` | Step 5. The Deep Sets network. |
| `research/train.py` | Steps 5 and 6. Grid, five seeds, temperature scaling, the linear ablation. |
| `research/evaluate.py` | Step 7. Metrics, calibration, de-vigging, ROI and Kelly. Shared by every model. |
| `research/report.py` | Scores every model on the same games, writes tables, plots and `metrics.json`. |
| `research/jev.py` | Part 2. Asks Jev for a home win probability per game, named and anonymised. Cached responses in `data/raw/jev/`. |
| `research/elo.py` | Team Elo (538 style) and its validation grid. |
| `research/player_ratings.py` | Player Elo (P1) and ridge plus-minus (P3) engines. |
| `research/ratings_experiment.py` | Tunes P1 and P3 on validation, exports ratings; `refine` widens the P3 grid. |
| `research/ratings_preds.py`, `rating_snapshots.py` | Standalone predictions from each rating; per-player rating snapshots for the deep model. |
| `research/combine.py` | Logistic regressions on production features and rating scalars. |
| `research/neural_elo.py` | Differentiable Elo with learned parameters and an optional update network. |
| `research/dl_data.py`, `dl_train.py` | Raw-sequence dataset; the deep-learning experiments. |
| `research/results/` | Committed outputs: per-game predictions per model, metrics, plots. |
| `data/` | The raw download cache. Git-ignored, about 75 MB. |

To reproduce (from the repo root; `pip install -r research/requirements.txt`; on Windows use `.venv\Scripts\python`):

```
python -m research.fetch            # fills data/raw, about 20 minutes, rate limited
python -m research.baseline
python -m research.player_features actual && python -m research.train actual
python -m research.player_features previous && python -m research.train previous
python -m research.jev previous      # needs a Jev key; about 5 minutes, 8.9M input tokens
python -m research.fetch --extra     # eight older seasons, nba_api part is quick, PBPStats part is slow and optional
python -m research.elo
python -m research.ratings_experiment && python -m research.ratings_experiment refine
python -m research.ratings_preds && python -m research.combine && python -m research.neural_elo
python -m research.dl_data previous && python -m research.dl_data actual && python -m research.rating_snapshots
python -m research.dl_train previous && python -m research.dl_train previous pr && python -m research.dl_train actual
python -m research.report
```

## 3. How the data was built

**Two sources, for two reasons.**
- The baseline uses **PBPStats** team game logs, because that is what production uses. Going through the production `DataCleaner`
  rather than re-implementing it means the baseline is the model that actually runs, not my reading of it.
  Only two things were swapped: R2 storage became an in-memory store fed from the local cache, and the home/away lookup
  reads the cached games file.
- The player model uses **`nba_api` `LeagueGameLog`**. The plan said to verify that player mode returns a whole season in one call
  before building on it. It does: one call gave 26,306 player-game rows covering all 1,230 games of 2024-25. One call per season
  per mode keeps well inside rate limits.

**Player features** (`player_features.py`), 23 numbers per player per game, computed from games *before* that game:
rolling-10 means of minutes, points, rebounds, assists, steals, blocks, turnovers and plus/minus; season-to-date minutes, points
and plus/minus; five per-36-minute rates and shooting shape (true shooting, three-point rate, free-throw rate, a usage proxy);
games played so far, days since the last game, and a "no history" flag. Three context numbers per team: rest days, back-to-back,
games into the season. Everything is standardised with mean and standard deviation from the training seasons only.

**Rosters.** Each team-game gets up to 15 slots (nobody was ever cut by the cap), sorted by expected minutes, padded with zeros.

**The split.** Train 2020-21 to 2023-24 (4,705 baseline games), validate 2024-25, test 2025-26. Chronological, fixed, in `config.py`.

**Games compared.** Each season has 1,230 games. The production cleaner drops each team's first game of a season (no rolling average
yet), leaving 1,214. Five more per season fall out when the two sources are joined: they appear in PBPStats's regular-season list
but not in `nba_api`'s, and their dates (Dec 13, Jan 15 and so on) look like NBA Cup knockout games. That leaves **1,209 games per
season** that every model has, and all comparisons use exactly those.

## 4. The models and the choices behind them

### 4.1 The baseline

Logistic regression: 52 team-level features (rolling-10 offence and defence ratings, pace, shooting efficiency, season record,
back-to-back, for both teams and their differences), standardised, `C=1`. It outputs one probability per game.
It is a good baseline for a reason beyond being the incumbent: it is hard to overfit with 4,705 games, and it is calibrated by
construction because it is trained on log loss.

### 4.2 The player model, step by step

A team is a **set of players**: there is no meaningful order to a roster and its size changes. **Deep Sets** is the simplest
architecture that respects that.

1. A small network `phi` (23 → 32 → 16, ReLU) turns each player's feature vector into a 16-number embedding. The *same* weights are
   used for every player. This is the key idea: the network learns what makes a player valuable once, not once per roster slot.
2. The embeddings are combined with a **minutes-weighted mean**, with weights = expected minutes. Empty slots have weight 0.
   Out comes one 16-number vector per team.
3. A head network scores "team A against team B" from A's vector, B's vector, their difference and both teams' context.
4. The final logit is `score(home, away) - score(away, home) + home_advantage`.

Parameters: 3,090. That is deliberately small. There are about 4,700 training games and a game result is mostly noise;
a bigger network only memorises faster (early stopping fired after 3 to 19 epochs in every run).

**Choices and the alternatives I did not take:**

| Choice | Why | Alternative |
|---|---|---|
| Minutes-weighted mean pooling | Order-independent, size-independent, and high-minute players dominate as they do on court | Plain sum (depends on roster size), max pooling (loses depth), attention pooling (plan's v2) |
| Antisymmetric head | Swapping the teams flips the sign, so the model cannot learn "listed first is better". Halves what must be learned from few games. | A plain MLP on [home, away, difference], which the plan suggested. Kept as `antisymmetric=False` and compared in the grid: antisymmetric won on validation in all 16 matched pairs, but by only 0.00002 to 0.004 log loss, small next to the seed-to-seed spread of about 0.005. |
| Expected minutes as the weight | Actual minutes carry the result: blowouts bench starters. See section 5. | Actual minutes (leaks), equal weights (ignores that a 4-minute player matters less) |
| BCE-with-logits, Adam, early stopping on validation log loss | Training and stopping on the same quantity that is reported | Accuracy as stopping criterion (noisy, coarse) |
| Five seeds, logits averaged | A network this size varies run to run; averaging removes most of that so the result does not hinge on a lucky seed | A single seed (the five individual runs span 0.589 to 0.594 on validation) |
| 32-setting grid, one seed each | To see whether architecture matters | Hand-picking. The grid result is the useful finding: it is flat (0.589 to 0.593 for the actual roster), so the architecture choice barely matters. |

**Traded players and rookies.** Features follow the player, because they are computed per player id across teams. A player with no
prior games gets zeros for his rolling features, a "no history" flag, and the median minutes that debuting players got in the
training seasons (12.0) as expected minutes.

### 4.3 Calibration (step 6)

**Temperature scaling**: divide the logits by one number `T` fitted on the validation season by minimising log loss.
It cannot change which team is favoured, so accuracy is untouched. It came out at 0.973 and 1.012: essentially 1, meaning the networks
were already close to calibrated and the step changed almost nothing. That is a result, not a failure of the step: it is cheap
insurance and the measurement says it was not needed here.

### 4.4 The ablation

A logistic regression on the same player features: the minutes-weighted average of each feature per team, home minus away, plus
context. Regularisation chosen on validation (`C=0.01`). It exists to answer "is the deep model earning its keep?" The answer so far is no.

## 5. Leakage: what was prevented and what remains

**Prevented, and how it was checked:**
- Every rolling and season-to-date player statistic is `shift(1)` before the rolling window, so a row never sees its own game.
  The production cleaner does the same for team statistics (checked by reading it: `x.shift(1).rolling(...)`).
- Pooling weights use expected minutes (rolling-10 average before the game), not minutes played.
- Standardisation constants and the no-history prior come from training seasons only.
- Hyperparameters, early stopping and the temperature use validation only. The test season is predicted once per model, and
  nothing was chosen from it. No confidence threshold exists anywhere.

**One thing in production code that looks like a leak and is not:** `SeasonWins` is `cumsum().shift(1)` over the whole frame, not per team,
so a team's first game of a season inherits the previous group's total. Those first rows are then dropped by the missing-rolling-average
filter, so it never reaches the model. Harmless, but it only works by accident of ordering.

**What remains optimistic, and I want this to be the first thing you remember about the results:** "who played" is information you
only have after the game. A star rested on the second night of a back-to-back is a strong signal about the result *and* is invisible to
anyone pricing the game at midday. This is the plan's step 4 ("acceptable for learning player strength; say so in any result") and it is
the whole difference between the two player rows in every table below.

**How I measured how big the effect is.** After the first results looked too good (the optimistic player model matching the market's log
loss), I added a variant whose roster is the players who played in the team's *previous* game: `python -m research.player_features previous`.
Those players' features are one game stale. It is still imperfect (it misses a star who returns, or is out, tonight) but it uses only
information available before tip-off. That variant cuts the improvement over the baseline by roughly 40% on test and 70% on validation.

**A forking-path disclosure:** I added that variant *after* seeing the test numbers of the first one. No hyperparameter was changed in
response, and the previous-roster model's settings were chosen on validation like the others, but the decision to look was triggered by
a test result. The clean fix is to treat the previous-game-roster rows as the headline and the actual-roster rows as an upper bound.

## 6. The evaluation harness (`evaluate.py`)

Every model hands over one number per game, the home win probability, and everything is computed from it:

- **Accuracy**: share of games where the side with p ≥ 0.5 won. Intuitive, and nearly useless for comparing probability models.
- **Brier score**: mean squared error of the probability. A constant 50% scores 0.25.
- **Log loss**: mean negative log-likelihood. Punishes a confident wrong call much harder than Brier. What the networks are trained on.
- **ECE**: the average gap between predicted probability and observed win rate across ten probability bins.
- **De-vigged market probability**: the bookmaker's prices overstate both sides (the implied probabilities sum to above 1). Dividing each by
  the sum removes that margin proportionally. Simple and adequate for Pinnacle; Shin or power methods differ by tenths of a percent.
- **Flat ROI**: one unit on a rule. Two rules, fixed in advance: "back the model's favourite" and "back any side with positive expected
  value (`p * odds > 1`)".
- **Kelly**: stake `(p*odds - 1)/(odds - 1)` of the bankroll, scaled by 1/4 or 1/2, on the positive-EV rule, with max drawdown.
- **Paired bootstrap** (added in `report.py`): resample games, take per-game log loss of baseline minus new model, report the 95% interval.
  It answers "is this gap bigger than game-sampling noise?" It does *not* capture seed noise or the fact that this is one test season.

## 7. Results

All on the same 1,209 games per season. Higher accuracy is better, lower Brier, log loss and ECE are better.
(Full tables: `research/results/results_table.md`. Plots: `research/results/*.png`.)

*The tables in this section are the first session's. Elo, player ratings and deep learning, and the combined table, are in section 9 and in `research/results/results_table.md`.*

### Test season 2025-26

| Model | Accuracy | Brier | Log loss | ECE |
|---|---|---|---|---|
| Logistic regression (production features) | 0.667 | 0.2093 | 0.6057 | 0.025 |
| Player model, previous-game roster | 0.676 | 0.2027 | 0.5895 | 0.026 |
| Linear on player features, previous-game roster | 0.678 | 0.2045 | 0.5936 | 0.027 |
| Jev, real names, recalibrated on validation | 0.682 | 0.2048 | 0.5964 | 0.024 |
| Jev, anonymised, recalibrated on validation | 0.674 | 0.2058 | 0.5985 | 0.023 |
| Jev, real names, raw | 0.663 | 0.2217 | 0.6416 | 0.114 |
| Jev, anonymised, raw | 0.653 | 0.2277 | 0.6583 | 0.127 |
| Player model, **actual roster (optimistic)** | 0.693 | 0.1976 | 0.5782 | 0.029 |
| Linear on player features, actual roster (optimistic) | 0.693 | 0.1988 | 0.5808 | 0.021 |
| Constant (home win rate 55.3%) | 0.553 | 0.2472 | 0.6874 | n/a |

### Validation season 2024-25

| Model | Accuracy | Brier | Log loss | ECE |
|---|---|---|---|---|
| Logistic regression (production features) | 0.679 | 0.2104 | 0.6077 | 0.035 |
| Player model, previous-game roster | 0.684 | 0.2077 | 0.6024 | 0.029 |
| Linear on player features, previous-game roster | 0.689 | 0.2073 | 0.6019 | 0.041 |
| Jev, real names / anonymised, raw | 0.647 / 0.637 | 0.2288 / 0.2355 | 0.6550 / 0.6744 | 0.109 / 0.128 |
| Jev, real names / anonymised, recalibrated (fitted on this season, so slightly optimistic) | 0.663 / 0.661 | 0.2110 / 0.2117 | 0.6095 / 0.6110 | 0.045 / 0.041 |
| Player model, actual roster (optimistic) | 0.690 | 0.2022 | 0.5892 | 0.033 |
| Linear on player features, actual roster (optimistic) | 0.698 | 0.2013 | 0.5874 | 0.024 |

### Improvement in log loss over the baseline, 95% bootstrap interval (positive = better)

| Model | Validation | Test |
|---|---|---|
| Player model, previous-game roster | +0.005 [-0.007, +0.018] | +0.016 [+0.004, +0.029] |
| Linear on player features, previous-game roster | +0.006 [-0.007, +0.018] | +0.012 [-0.001, +0.025] |
| Jev, real names, recalibrated | -0.002 [-0.012, +0.008] | +0.009 [-0.002, +0.020] |
| Jev, anonymised, recalibrated | -0.003 [-0.013, +0.007] | +0.007 [-0.003, +0.017] |
| Jev, real names, raw | -0.047 [-0.071, -0.025] | -0.036 [-0.060, -0.012] |
| Jev, anonymised, raw | -0.067 [-0.094, -0.041] | -0.053 [-0.080, -0.026] |
| Player model, actual roster (optimistic) | +0.019 [+0.002, +0.034] | +0.028 [+0.012, +0.043] |

**How to read this honestly.** With a fair roster the player model is slightly better than the baseline on both seasons, but the validation
gap is inside the noise, the test gap just clears it, and a linear model on the same inputs is within noise of the network. Jev, recalibrated, lands in the same place as the linear models. The defensible
sentence is: *"Adding player-level features improved log loss by about 0.005 to 0.016 over a team-level logistic regression on a fixed
chronological split; a deep set model was no better than a linear model on the same features."* The sentence you cannot defend is any
claim about beating the market.

![Calibration on the test season](../research/results/calibration_test.png)

The reliability curves sit close to the diagonal for every model and the temperature step found nothing to fix. The lower panel shows
how confident each model is; the optimistic roster model is bolder: 253 test games at 80% or more confidence, against 190 for the previous-roster model and 107 for the baseline.

### Against the market (680 games, 2026-01-07 to 2026-04-12, the only games with stored odds)

| Source | Accuracy | Brier | Log loss | Flat ROI (back pick) | ¼ Kelly final | Max drawdown |
|---|---|---|---|---|---|---|
| Market (de-vigged Pinnacle) | 0.710 | 0.1829 | 0.5432 | | | |
| Logistic regression | 0.696 | 0.1980 | 0.5805 | -0.2% | 0.04x | 98% |
| Player model, previous-game roster | 0.701 | 0.1888 | 0.5573 | -0.9% | 0.57x | 85% |
| Linear, previous-game roster | 0.699 | 0.1926 | 0.5663 | -0.4% | 0.24x | 92% |
| Jev, real names, recalibrated | 0.699 | 0.1960 | 0.5772 | -0.1% | 0.04x | 97% |
| Jev, anonymised, recalibrated | 0.691 | 0.1978 | 0.5813 | -1.4% | 0.02x | 99% |
| Jev, real names, raw | 0.679 | 0.2097 | 0.6096 | -1.7% | 0.02x | 98% |
| Player model, actual roster (optimistic) | 0.722 | 0.1825 | 0.5427 | +3.2% | 6.79x | 59% |
| Linear, actual roster (optimistic) | 0.724 | 0.1869 | 0.5532 | +4.5% | 1.65x | 66% |

![Quarter-Kelly bankroll over the 680 games with odds](../research/results/bankroll_test.png)

- Every fair model is **worse than the market** on log loss. The market is the benchmark that matters, and nobody here has beaten it.
- The fair models, Jev included, lose money under the quarter-Kelly rule, and lose most of the bankroll (drawdowns of 85% to 98%). That is what betting on
  an edge that is not there looks like. The production baseline ends at 0.04x.
- The optimistic rows reaching 6.8x are the roster effect, not skill. Kelly sizing amplifies any overconfidence relative to the market.
- **The odds are not closing lines.** `daily_generate.py` fetches Pinnacle prices when the daily job runs, about 12:00 Helsinki, for games
  in the next 24 hours, so for an evening NBA game the price is many hours old. Real closing prices are sharper, so the market's true
  benchmark is stronger than the 0.5432 shown, and the vault note's phrase "closing lines" is not accurate for this data.
- No confidence threshold was used for the betting numbers: the only season with odds is the test season, so there was no clean place to
  fix a threshold. Confidence buckets (`metrics.json`) are reported as plain descriptive tables.

### The production service's own live record, for context

On the same regular-season games, what the deployed service actually predicted on the day (four model versions over the season): 64.0% accuracy
over 1,079 games. Broken down: the current version (V2.2) scored 68.2% over its 674 regular-season games, matching the README's figure,
and then 61.5% over 91 play-in and playoff games. Versions V1 and the custom NumPy network, which ran earlier in the season, scored 60.1% and
53.1%. So the README number is true and is the best of four versions over the easier part of the season.

## 8. Jev (Part 2)

**Setup.** `research/jev.py` calls `POST https://api.typesafe.ai/v1/systemone` (`jev-latest`, which answered as `jev-1.13.0`) with a `state`
and one `noul` question, "using only the pre-game information in the state, will the home team win?", and takes the returned probability.
The state is the same pre-game information the models get: season record and last-10 ratings from the production cleaner's columns, rest days,
back-to-back, and each team's eight players with most expected minutes with their pre-game rolling points, rebounds, assists and plus/minus.
No results, no scores, no post-tip-off information. The roster is the **previous-game roster**, the fair setting; an actual-roster run was
not done for Jev because it would only reproduce the optimism seen for the other models.

**Key handling.** The task named `JEV_API_KEY`; the key on this machine is `JEVS_API_KEY` in the repo's `.env`. The script reads either name from the
environment or from `.env`, using `dotenv_values`, which does not put the value in the process environment. The value goes only into the
Authorization header. It is not printed, logged, cached (the cache files hold only the probability, token counts and model name) or committed. `.env` was never displayed.

**Run.** Every validation and test game (2,450 games, of which 1,209 per season are in the common comparison set), twice: **real names** (team abbreviations,
player names, the date) and **anonymised** ("Home team", "Home player 3", no date). 4,894 requests, **8,903,729 input tokens** and 107,800 output
tokens, 0 failures, against the 50 million budget. Usage totals: `research/results/jev_usage_previous.json`. One prompt was written and used;
it was not iterated on any season.

**What came out.**
- **Raw Jev is overconfident.** Its answers sit nearer 0 and 1 than the outcomes justify (calibration error 0.11 to 0.13 against 0.02 to 0.03
  for the other models), and its raw log loss is worse than the logistic regression by 0.036 (names) and 0.053 (anonymised) on test, both clearly outside noise.
- **Recalibrated, it is competitive.** A two-parameter Platt correction, `sigmoid(a * logit(p) + b)`, fitted on 2024-25 only and applied unchanged to
  2025-26, brings test log loss to 0.5964 (names) and 0.5985 (anonymised). That is +0.009 and +0.007 over the logistic regression with intervals that
  include zero, i.e. level with the baseline, and slightly behind the player model's 0.5895. On accuracy it is ahead of the logistic regression (68.2% and 67.4% against 66.7%).
  Against the market on the 680 odds games it is at 0.5772 (names), behind the previous-roster player model at 0.5573 and the market at 0.5432.
- **The memorisation check.** Real names improve raw log loss by 0.0167 on test (interval +0.012 to +0.022) and 0.0193 on validation. After recalibration the
  gain is 0.0021 on test (interval -0.0002 to +0.0044) and 0.0015 on validation (-0.0008 to +0.0037). So the raw gap is mostly the named run being less overconfident
  rather than recalling scores. Two cautions: this does not prove there is no memorisation (names also carry legitimate general knowledge, such as that
  a franchise has been strong for years, and the check only detects what survives a two-number recalibration); and the two seasons may differ in whether they
  are inside the model's pretraining data. They look alike here, but I cannot see its training cutoff.
- **A forking path.** The recalibration rows were added after I saw the raw Jev numbers on both seasons. The correction uses validation only, so test was
  not used to fit anything, but the decision to add it was prompted by seeing the raw result.

**What this does and does not show.** Jev, with no training on NBA data and given a short text table of stats, performs about as well as a
trained logistic regression once its probabilities are put on the right scale. It does not beat the market, and a combination of Jev with the player model
(the "hybrid" idea in the vault note) has not been tried. Other limits: one prompt version, only the top eight players shown, one test season.
Cost was tiny: about $0.37 of input at the quoted $0.042 per million tokens (a company figure, not independently checked).

## 9. Elo, player ratings and deep learning (second session)

You asked for team Elo, some way of using player Elo, several iterations, deep learning on top (including letting it set the Elo parameters),
as much data as possible, and an answer to whether deep learning removes the need for feature engineering. This section is all of that.
Code: `research/elo.py`, `player_ratings.py`, `ratings_experiment.py`, `ratings_preds.py`, `rating_snapshots.py`, `combine.py`, `neural_elo.py`,
`dl_data.py`, `dl_train.py`. Everything uses the same fixed split, the same harness (`evaluate.py`, `report.py`) and the same rule:
**choices on 2024-25 only, test scored once, and every variant reported, not only the winners.**

### 9.1 What other people do (read online, not verified by running anything)

- **Team Elo, the FiveThirtyEight way.** Rating difference plus a home-court bonus gives the win probability; after the game both ratings move by
  K times (result minus expected), scaled by a margin-of-victory multiplier; between seasons ratings are pulled back towards the average. The published values
  I could confirm were a home bonus of 100 Elo points and a carryover of 0.75; I used K = 20 and the margin formula `(margin + 3)^0.8 / (7.5 + 0.006 * winner's edge)`
  from memory, because FiveThirtyEight's own page now redirects elsewhere. Sources:
  [archived Elo repository](https://github.com/Neil-Paine-1/NBA-elo), [Harvard Sports Analysis on improving it](https://harvardsportsanalysis.org/2019/01/a-simple-improvement-to-fivethirtyeights-nba-elo-model/).
- **Player-based systems** (RAPTOR, EPM, DARKO and the "CARM-Elo" family) rate every player and take a team's strength as a *projected-minutes-weighted sum* of its players'
  ratings, which makes trades and injuries show up immediately. RAPTOR's team adjustment multiplies the minute-weighted average by 4.5 so five players on the court add up to a team rating
  ([Estimated RAPTOR notes](https://github.com/Neil-Paine-1/NBA-elo)). One open-source project stacks a player Glicko-2 rating, a lineup Elo, a ridge plus-minus regression and a team Elo,
  then calibrates the result ([BasketballElo](https://github.com/ShaneHurley/BasketballElo)). That is the pattern copied here, in smaller form.
- **Tuning Elo's few numbers on validation log loss** is standard practice ([SCOPE](https://ojs.aaai.org/index.php/AIIDE/article/view/5233)).
- **Trees versus neural networks on tabular data of this size** (about 10,000 rows): in a large benchmark tree models stayed ahead, with the reasons given as neural networks being
  biased towards smooth functions and sensitive to uninformative features ([Grinsztajn et al., NeurIPS 2022](https://arxiv.org/abs/2207.08815)). Relevant to your question; see 9.8.

### 9.2 Team Elo

Run through all 14 seasons in order (2012-13 to 2025-26, 16,888 games; the eight older seasons are only warm-up so ratings are not still at their starting value in 2020-21).
FiveThirtyEight's published settings, untouched, score a validation log loss of 0.6232, worse than the production regression (0.6077). Tuning four numbers on 2024-25
(a grid of 1,470 settings; the first, smaller grid had its best point on the edge, so it was widened and re-run) gives **K = 12, home bonus 30, carryover 0.5, margin exponent 1.0**.
The best point is now interior and the neighbours are almost as good (top five all within 0.0003), which means the choice is not delicate.
The home bonus of 30 points (about one point of margin) against 538's 100 matches the weaker home-court advantage of recent seasons; the carryover of 0.5 against 0.75 says rosters change more now.

Result: validation log loss 0.6065, same as the 52-feature production model; **test 0.5925 against 0.6057**, a gain of +0.013 (interval +0.003 to +0.024).
One number per team does the work of 52 features. The likely reason is that the production features are rolling 10-game averages that do not adjust for opponent strength,
while Elo does, and Elo remembers further back with sensible decay.

### 9.3 Two ways to rate players

Both are strictly point-in-time and share the roster rule from section 5 (previous-game roster is the fair one). Team strength is the sum over a roster of
`weight x rating`, with weight = share of *expected* minutes (the player's last-10-game average before this game), scaled so a full lineup sums to 5.

- **P1, player Elo.** The same recursion as team Elo, but the "rating" being updated belongs to the players on the floor: after each game every player moves by
  `K x margin multiplier x (result - expected) x his share of the minutes actually played`. Players regress to 0 each season; a player seen for the first time starts at a replacement level.
  A random search of 240 settings on validation chose K = 9.3, home 39, carryover 0.71, replacement level -40, margin exponent 0.8.
- **P3, ridge plus-minus** (the idea behind RAPM). Regress the game's point margin on the minute-share differences between the two rosters. Ridge shrinkage is essential: players are almost always on
  court with the same teammates, so an unpenalised fit is wildly unstable. Older games are down-weighted with a half-life, and the coefficients are re-solved every 10 games from games played so far.
  A grid of 29 settings (the first 15 put the best point at the smallest penalty tried, so the grid was extended downwards, and it came back to the same answer) chose lambda = 10 and a half-life of 1,000 games.

Results with the previous-game roster: P1 validation 0.6039, test 0.5930; P3 validation 0.6003, test 0.5897 (accuracy 70.0%). Neither separates from team Elo on test (player Elo is 0.0005 behind it, the ridge model 0.003 ahead; intervals include zero).
With the actual roster (optimistic) P3 reaches 0.5825 on test, which again shows how much of the apparent edge is knowing who plays.

### 9.4 Stacking: do the rating numbers add anything to a logistic regression?

All logistic regressions, `C` chosen on validation, trained on 2020-21 to 2023-24:

| Model | Val log loss | Test log loss | Test accuracy |
|---|---|---|---|
| Production logistic regression (baseline) | 0.6077 | 0.6057 | 66.7% |
| Team Elo (settings tuned on validation) | 0.6065 | 0.5925 | 69.1% |
| Neural Elo: K, home bonus, carryover learned by gradient | 0.6063 | 0.5918 | 68.7% |
| Neural Elo with a learned update network | 0.6065 | 0.5926 | 69.0% |
| Player Elo, previous-game roster | 0.6039 | 0.5930 | 68.7% |
| Ridge plus-minus, previous-game roster | 0.6003 | 0.5897 | 70.0% |
| Production features + team Elo | 0.6035 | 0.5999 | 68.2% |
| Production features + player Elo and ridge | 0.5978 | 0.5960 | 69.2% |
| Production features + all rating scalars | 0.5978 | 0.5946 | 69.1% |
| Logistic regression on 7 rating scalars (team Elo, player Elo, ridge, rest) | 0.5962 | 0.5897 | 68.7% |

- Adding team Elo to the 52 production features helps a little; adding the player ratings helps more.
- **The 7-number model** (team Elo difference, player Elo difference, ridge margin, rest and back-to-back for both teams) is **better than the 55-feature stack on validation and test**.
  More columns are not helping; the information is in the three rating differences.
- Training that 7-number model on 4, 8 or 12 seasons gives validation log loss 0.5962, 0.5965 and 0.5962: **a linear model gets nothing from extra history.** This matters for 9.7.

### 9.5 Neural Elo: letting gradient descent set the parameters

`research/neural_elo.py` makes the Elo recursion differentiable and learns K, home bonus and carryover by back-propagating the log loss through whole seasons (gradients cut at season boundaries).
A second version replaces the constant K with a small network that sees the margin, the lopsidedness of the matchup, how far into the season it is and both teams' rest.

- **Learned values:** K = 11.8, home bonus 30.0, carryover 0.505. They started at the grid's answer (12, 30, 0.5) and barely moved; early stopping ended training after 16 epochs.
  The two methods agree, which is reassuring about both and means gradients found nothing the grid missed.
- **The update network does not help**: validation log loss 0.6065 against 0.6063 for the constant-K version, test 0.5926 against 0.5918.
  Four numbers is about all this recursion has to tune, and on ~16,000 games an extra network has nothing left to learn.

### 9.6 The deep-learning dataset

`dl_data.py` builds, for each game and each team, the ten players with the most expected minutes, each described by his **last 20 box-score rows before the game date** (16 stats, days since his
previous game, days before the target game, same-season flag, valid flag). Two views come from that identical window:
`SEQ`, the raw sequence, and `HF`, 22 hand-built numbers (last-10 and last-20 means, per-36 rates, shooting shape, days since last game, how much history exists). Comparing a model that reads one with a model that reads the other
isolates a single question: does the network need the features built for it?
Leakage checks run: for sampled players the window equals his real history, every row is strictly earlier than the game, and the game's own stats are not in it.
A side benefit: in previous-roster mode the players' windows now include their previous game, so the features are no longer stale by one game as in the first player model.
Then `snap_p1.npy` and `snap_p3.npy` record every player's Elo and ridge rating at the start of every game, so the network can be given each *player's own* rating.

### 9.7 The deep-learning experiments

Same network family as before (shared per-player encoder, minutes-weighted pooling, antisymmetric home/away head) with switches: the player encoder (hand features, a GRU over raw sequences, or both), game-level rating scalars added to the logit
("wide"), each player's own ratings as input ("pr"), a time input, and how many training seasons are used. Each experiment tried 4 hyperparameter settings with one seed (chosen on validation), then 5 seeds were trained on the best and averaged,
and a temperature fitted on validation. 28 experiments, 252 trainings in all (console logs of every sweep are in `research/results/dl_sweep_*_console.log`). Previous-game roster unless stated.

| Model | Val log loss | Test log loss | Test accuracy |
|---|---|---|---|
| Hand features, 4 seasons | 0.6018 | 0.6019 | 67.4% |
| Hand features, 8 seasons | 0.5999 | 0.5976 | 67.7% |
| Hand features, 12 seasons | 0.5980 | 0.5963 | 67.2% |
| Raw sequences (GRU), 4 seasons | 0.5996 | 0.5955 | 68.0% |
| Raw sequences (GRU), 12 seasons | 0.5971 | 0.5913 | 68.5% |
| Both, 4 seasons | 0.6040 | 0.5973 | 67.7% |
| Both, 12 seasons | 0.5977 | 0.5910 | 68.2% |
| Hand features, 12 seasons, + game-level rating scalars | 0.5934 | 0.5868 | 69.1% |
| Both, 12 seasons, + game-level rating scalars | 0.5910 | 0.5829 | 68.8% |
| ...+ scalars, + season year | 0.5938 | 0.5877 | 68.2% |
| ...+ scalars, + league context | 0.5924 | 0.5866 | 67.5% |
| ...+ scalars, + recency weights | 0.5931 | 0.5890 | 68.4% |
| Hand features, 12 seasons, + per-player Elo and ridge ratings | 0.5909 | 0.5872 | 69.1% |
| Hand features, + per-player ratings + scalars | 0.5921 | 0.5865 | 68.5% |
| Both, + per-player ratings + scalars | 0.5892 | 0.5821 | 69.2% |
| Both, + per-player ratings + scalars + league context | 0.5891 | 0.5853 | 69.5% |

**How much does more history help?** Hand features: 0.6018 (4 seasons), 0.5999 (8), 0.5980 (12) on validation, and 0.6019, 0.5976, 0.5963 on test. Raw sequences: 0.5996 to 0.5971 validation, 0.5955 to 0.5913 test.
A steady gain of about 0.003 to 0.006 from tripling the data, in the deep models, and none for the linear 7-number model. That is what the textbooks say should happen: flexible models have a use for more data, simple ones do not.
(The 2019-20 bubble and the no-fans 2020-21 season are in the training window; I did not test removing them.)

**Do raw sequences replace hand-built features?** Roughly yes: the GRU on raw windows scored 0.5971 validation and 0.5913 test, against 0.5980 and 0.5963 for hand features built from the same windows. A difference of 0.001 on validation and 0.005 on test, with test
intervals against each other not computed and seed noise of up to about 0.005 (the five seeds of one setting spanned that much in the first model), so call it a tie with a slight lean to the raw sequences, and not worse. Giving the network both was no better than either.

**Where the real gain came from** was not the encoder. It was information the network cannot easily build for itself:

- adding the game-level rating scalars (an Elo-type input) moved hand-feature models from 0.5980 to 0.5934 on validation and 0.5963 to 0.5868 on test;
- giving each player's own Elo and ridge ratings as inputs did slightly better still (0.5909 / 0.5872 for hand features, 0.5892 / 0.5821 for both encoders).
  An Elo rating is a hand-built feature with a recursive memory and opponent adjustment. A network handed 20 box-score rows has no way to discover opponent-adjusted strength, because opponents are not in the window.

**Letting the model see time.** Season year as an input did nothing (0.5938 vs 0.5934 without; test 0.5877 vs 0.5868): it is a number the network has never seen at test time (2025 beyond the training years)
and the model was already handed ratings that carry the era's information. Recency weighting also did nothing (0.5931). League context (league scoring and home win rate so far this season) helped slightly on validation (0.5924) and on test (0.5866), well inside noise.
So of the four ways to express trend, none was worth the added complication on this evidence.

**The headline deep model.** The validation-selected best fair model is *both encoders, per-player ratings, rating scalars and league context*: validation 0.5891, test 0.5853. The same model without league context scored 0.5892 validation and 0.5821 test.
Those two are a tie on the selection criterion (0.0001 apart); I am reporting both because picking the better test number would be exactly the mistake the rules forbid.

| Model | Gain over production logreg, test, 95% interval | Gain over the 7-number model, test, 95% interval |
|---|---|---|
| Team Elo (settings tuned on validation) | +0.0132 [+0.003, +0.024] | -0.0027 [-0.010, +0.004] |
| Ridge plus-minus, previous-game roster | +0.0160 [+0.003, +0.029] | +0.0000 [-0.005, +0.005] |
| Logistic regression on 7 rating scalars (team Elo, player Elo, ridge, rest) | +0.0160 [+0.005, +0.027] | (the reference) |
| Deep, raw sequences, 12 seasons | +0.0145 [-0.001, +0.029] | -0.0015 [-0.011, +0.008] |
| Deep, hand features + scalars, 12 seasons | +0.0189 [+0.005, +0.032] | +0.0029 [-0.003, +0.008] |
| Deep, both + scalars, 12 seasons | +0.0228 [+0.009, +0.036] | +0.0068 [-0.000, +0.013] |
| Deep, both + per-player ratings + scalars | +0.0236 [+0.010, +0.037] | +0.0076 [+0.001, +0.014] |
| Deep, both + per-player ratings + scalars + league context (validation-selected) | +0.0205 [+0.006, +0.035] | +0.0045 [-0.003, +0.012] |

**What that table says.**
- Everything with rating information beats the production regression with intervals above zero.
- Against the *7-number model*, the best deep models are ahead by +0.005 to +0.008, with intervals that straddle zero or only just clear it. I would describe it as "probably a small gain, not proven".
- Team Elo, the ridge model, the 7-number regression, the earlier Deep Sets and recalibrated Jev are all within noise of each other. The structure of the problem (strength of the five on the floor) is captured by about three numbers;
  everything after that fights for the last 0.005.

**With the actual roster (optimistic)**, for the record:

| Model | Val log loss | Test log loss | Test accuracy |
|---|---|---|---|
| Hand features, 12 seasons | 0.5901 | 0.5833 | 68.6% |
| Raw sequences, 12 seasons | 0.5878 | 0.5829 | 69.6% |
| Hand features + scalars | 0.5825 | 0.5754 | 69.1% |
| Both + scalars | 0.5813 | 0.5751 | 70.5% |
| Ridge plus-minus (no network) | 0.5933 | 0.5825 | 69.8% |
| Production features + scalars (linear) | 0.5914 | 0.5905 | 69.1% |

The best of these, 0.5751 on test, is not a result to quote. It uses information not available before the game.

**Against the market** (the 680 games with stored odds; log loss and bankroll):

| Source | Log loss | Accuracy | Quarter-Kelly final bankroll |
|---|---|---|---|
| Market (de-vigged Pinnacle) | 0.5432 | | |
| Production logistic regression | 0.5805 | 69.6% | 0.04x, max drawdown 98% |
| Team Elo (settings tuned on validation) | 0.5693 | 71.8% | 0.12x, max drawdown 96% |
| Logistic regression on 7 rating scalars (team Elo, player Elo, ridge, rest) | 0.5664 | 71.0% | 0.12x, max drawdown 95% |
| Earlier Deep Sets player model, previous-game roster | 0.5573 | 70.1% | 0.57x, max drawdown 85% |
| Deep, both + per-player ratings + scalars | 0.5549 | 71.3% | 0.48x, max drawdown 87% |
| Deep, ... + league context | 0.5570 | 72.8% | 0.53x, max drawdown 89% |
| Deep, both + scalars, actual roster (optimistic) | 0.5435 | 71.9% | 2.14x, max drawdown 68% |

No fair model reaches the market's 0.5432; the best deep models are about 0.012 to 0.014 behind. Every fair bankroll ends below its start. As before, the optimistic row reaching further is the roster effect.

### 9.8 Your question: with deep learning is feature engineering less important?

Partly yes, but not in the way "deep learning learns the features" suggests. What the experiments show:

1. **Raw per-player sequences versus hand-built summaries of them: a tie.** The network did not need rolling means built for it. That part of your intuition is right, and it needed the extra seasons (with 4 seasons the raw GRU was ahead of hand features by 0.002 on validation and 0.006 on test, which is at the edge of the noise).
2. **Structure that is not in the input cannot be learned from it.** Opponent-adjusted, long-memory team strength is the thing Elo supplies. Handing the network that scalar was worth more than any change of encoder. That is feature engineering, in the form of an algorithm instead of a column.
3. **Small, noisy data favours simple structure.** About 10,000 usable games, outcomes mostly luck: the 7-number logistic regression sits within 0.005 of the best network. This is the regime where the benchmark in [Grinsztajn et al.](https://arxiv.org/abs/2207.08815)
   found trees and simpler models strong. I did not run a tree model here (gradient boosting on the same scalars is an obvious missing comparison).
4. **Fourier and time-series transforms.** They earn their place on signals with real periodicity at the resolution you sample (vibration, a gait cycle, sensor streams). A player's 20 box scores have no such rhythm worth extracting;
   the repeating structure in this problem is the weekly schedule and the season, and rest days and season progress are already inputs. **I did not test spectral features**, so this is reasoning, not a result. If you want it tested, it is a cheap experiment: FFT magnitudes of the minutes and points windows as extra hand features.
5. **Season as a number.** Covered in 9.7: not helpful, and for a reason worth knowing: models cannot extrapolate a counter. Express *what changed* (league scoring, home advantage) rather than *when*.

### 9.9 What was not done, and cautions

- **"All of the data":** I used every box-score row for 14 seasons from `nba_api` (358,520 player-games) plus the schedule. I did not pull play-by-play or player-tracking data, injury reports or lineup data. Those exist but are one request per game (about 17,000), and some do not go back that far.
- **Not tried:** a differentiable *player* Elo; transformers over a whole roster; gradient-boosted trees; blending models; spectral features; removing the bubble seasons; a 14-season (all) window rather than 12.
- **Search size and selection.** Across this section about 1,470 team-Elo settings, 240 plus 29 rating settings, 28 deep experiments with 4 settings each, and a handful of linear variants were scored on the same validation season.
  Selecting the best of that many on one season of ~1,200 games overstates the best validation number somewhat. Test numbers are unselected, but differences of about 0.005 between top models are inside the noise.
- **The 2025-26 season looks more predictable than 2024-25 for rating models.** Team Elo is level with the production model on validation (+0.001) and clearly ahead on test (+0.013); the logistic regression barely moved between seasons while Elo improved by 0.014. I do not know why
  (more lopsided teams, perhaps). Do not extrapolate the size of any gain to another season.
- **The production regression is a weak baseline.** Beating it is easy now; the 7-number model is the fairer yardstick.
- **Ratings were tuned on validation** and then used as inputs for the deep models, so their validation numbers carry a small optimism; test does not.
- **Process slip:** a later run overwrote the per-configuration log of the first deep-learning sweep (`dl_iterations.json`); the console logs of all sweeps are saved instead, and the code now appends.

## 10. The dashboard

**Choice: a static site on GitHub Pages**, rebuilt by a scheduled GitHub Action. Repository `nba-dashboard`, branch `static-site`.

Why, and what I weighed:

| Option | Verdict |
|---|---|
| Keep Streamlit Community Cloud | The problem is the platform's sleep policy, not the code. No code change fixes it. |
| Streamlit on Fly.io or Render | Always-on costs money or, on free tiers, also sleeps. A Python server for a page that changes once a day is the wrong tool. |
| Static page, built daily (chosen) | Plain files cannot sleep. Free. R2 credentials live only inside the Action, never in a browser. The data changes once a day, so a daily build loses nothing. |
| Cloudflare Pages | Equivalent, and fine. GitHub Pages was chosen because the repo and the Action are already on GitHub, so one fewer account and token. |
| Page reads R2 directly from the browser | Would need public bucket access or a key in the page. No. |

What it gives up: the old sidebar filters (no live querying), and data up to a day old.

**What it contains:**
1. *2026-27 season*: the production model's daily predictions against results, running accuracy (cumulative and last 20), calibration, latest games.
   **It is empty right now and says so**, because no 2026-27 regular-season game has been scored yet (section 11 explains why preseason games are missing).
2. *Model comparison*: the test-season table, the paired-bootstrap table, calibration of the models, the market table, a visible warning about the
   optimistic roster rows, and the Jev rows (raw and recalibrated) with a note explaining the recalibration.
3. *2025-26 archive*: from the five old tabs I kept what answers a question: accuracy by model version, running accuracy, calibration, per-team
   accuracy, favourite versus underdog accuracy, and the flat-stake result across all 773 games with odds (-1.2% ROI, the unselected overall figure,
   not a slice). Dropped: the confusion matrix, five betting strategies, daily and monthly P&L tables and the head-to-head grid, which need filters to be useful.

**How it is deployed** (`.github/workflows/pages.yml`): daily at 10:30 UTC and on every push to master, it installs three packages, runs
`build_site.py` (reads `history/prediction_history.json` and `current/current_predictions.json` from R2, read only), and publishes `site/`.
Chart library is vendored into the repo so a CDN outage cannot break the page.

**What I could not do, and what you need to do** (no way to log in to GitHub settings or R2 from here, and I was told not to push to main):
1. Merge `static-site` into `master`.
2. Settings > Pages > Source = GitHub Actions.
3. Add four repository secrets: `R2_ENDPOINT`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET_NAME`. Use a read-only R2 token.
4. Run the workflow once from the Actions tab.
It builds and renders locally (screenshots below are from a local server). I have not run the Action itself, so its first run is untested.
Known limit: GitHub turns off scheduled workflows in a public repo after 60 days without activity.

**Portfolio embed, not changed.** In the portfolio repo, `src/pages/NbaPrediction.js` has three references to the Streamlit URL: the iframe `src`
(`https://nba-ml-dashboard.streamlit.app/?embed=true`, line 222), an "Open Dashboard" link (line 94) and a second link (line 205), plus the
fake browser-bar label text (line 219). After the Pages site is live they would point to `https://markusmuilu.github.io/nba-dashboard/`, and `?embed=true` goes.
GitHub Pages sets no frame-blocking headers, so the iframe works.

![2025-26 archive tab](img/dashboard_arc.png)

![Model comparison tab](img/dashboard_cmp.png)

![2026-27 season tab, empty state](img/dashboard_now.png)

## 11. The Fly.io service (read only, nothing redeployed)

Checked with GET requests only on 2026-10-05 and 06:
- `GET /` returns `{"detail":"Not Found"}`. That is simply because the app defines no `/` route. It does **not** mean the service is down.
- `GET /docs` and `GET /openapi.json` respond; `GET /predict?team1=BOS&team2=NYK` returned `{"winner":"BOS","confidence":79.75}` in 9 seconds.
  (That call also made the service refresh its per-team files in R2, as every `/predict` call does. No predictions were written.)
- The automation is running: the per-team CSVs in R2 were refreshed at 09:00 UTC on 3, 4 and 5 October.
- **But `current/current_predictions.json` has been empty since 2026-06-14 and `history` ends 2026-06-13, even though preseason games
  were played on 3 and 4 October (and more were scheduled for 5 October).** So the daily job runs and produces nothing.
  *Likely cause, from reading the code, not confirmed in the service logs (I had no log access):* `OddsFetcher.fetch_odds()` filters to Pinnacle
  prices for games in the next 24 hours; in preseason there are none, so it returns an empty DataFrame with no `home_team` column. In
  `daily_generate.py` the next line indexes `odds["home_team"]`, which raises `KeyError('home_team')` (reproduced in isolation). The enclosing
  `try/except` swallows it and returns `None`, so the predictions made a moment earlier are discarded.
  If that is right, the first regular-season day with Pinnacle prices will work and the failure matters only on days with no priced games. It would also
  drop predictions on any regular-season day the odds API returns nothing.
- **The season is hardcoded**: `get_current_season(..., season="2025-26")` in `data_collector.py`. At the start of 2026-27 the service will keep
  predicting from last season's data until that default changes.
- **Serving features are one game stale** *(from reading `clean_prediction_data`, not tested)*: it takes each team's most recent logged row, whose
  rolling averages exclude that game by `shift(1)`. Training rows use "everything before this game"; the served row uses "everything before the last game played".
  The back-to-back flag is similarly taken from the last game, not the upcoming one. Probably a small effect on accuracy; worth fixing before shadow mode.
- A smaller one: `model_trainer.py` computes ROC-AUC from hard 0/1 predictions, which understates it. Not used by anything.

## 12. Bugs, wrong turns and how they were found

| What | How found | Fix |
|---|---|---|
| R2 listing failed with `NoneType` bucket | Traceback from `boto3` in a scratch script | `load_dotenv()` searches from the script's folder; pass the path to the repo `.env` explicitly |
| `plotly-basic.min.js` 404 on cdnjs | `curl -I` on the URL before relying on it | Vendored the file from jsDelivr into `site/vendor/` so there is no runtime CDN at all |
| Brier and log loss `NaN` in the dashboard data | The build printed `nan` | 151 history rows (21 Oct 2025 to 25 Jan 2026) hold a result and no prediction; they are excluded from scoring and counted. The same filter was added to `report.py`, where they would have become NaN silently |
| Charts drawn over the text below them | Screenshot of the page | Plotly does not size its own container reliably inside a hidden tab; the card now gets an explicit height |
| Bar labels printed inside every team bar | Same screenshot | `textposition: "none"` |
| Syntax error in `report.py` | Python refused to import it | A `\n` in my edit script became a literal newline inside an f-string |
| Player-model results looked too good | Log loss equal to the market's on odds games, which is implausible for a free-data model | The "previous-game roster" variant (section 5) |
| First Elo grid had its best point on the edge | The best setting was the lowest home bonus, lowest carryover and highest margin exponent tried | Widened the grid; the new best is interior and the neighbours are nearly as good |
| Elo grid crashed on an extreme setting | `OverflowError` from `10 ** x` when ratings diverged | Clipped the exponent so diverging settings just score badly |
| One 2012-13 game had no player rows | `KeyError` in the rating engine | Dropped games without both rosters |
| Ridge grid best point on the edge (smallest penalty) | Same check as for Elo | Extended the grid; the answer did not move |
| Neural Elo detached the carry parameter | Reading the code: the carryover step ran before the gradient cut, so its gradient would have been lost | Detach the old ratings, keep the multiplier in the graph |
| A later run overwrote the earlier deep-learning iteration log | The file held only 4 entries instead of 12 | Kept the console logs of every sweep in `research/results/`, and the code now appends |
| 5 baseline games per season had no counterpart | Counted 1,214 vs 1,225 games when comparing sets | They exist in PBPStats only (apparently NBA Cup knockouts); the comparison uses the intersection |

## 13. What you would need to understand to defend this in an interview

1. **Why log loss, not accuracy.** Accuracy throws away the probability. Two models at 67% accuracy can differ a lot in whether "70%" means 70%. The
   market comparison, Kelly sizing and calibration all need honest probabilities.
2. **What "no leakage" meant concretely**: `shift(1)`, expected not actual minutes, scaling on train only, validation-only tuning, test scored once.
   And the one leak you *chose* to keep (the roster) and how you measured its size.
3. **Why the headline gain is small and why that is the interesting part.** A roughly 0.016 log loss gain, interval barely above zero, a linear model matching the network.
   An interviewer who probes will find all of that; having found it yourself first is the point.
4. **Why the market is the right benchmark and why you did not beat it.** And why your odds are not closing lines.
5. **Deep Sets in two sentences:** shared per-player network, order-independent pooling, so the model handles any roster size and order. The antisymmetric head
   makes "home vs away" symmetric by construction.
6. **Why a 3,090-parameter network on 4,700 games**, why early stopping fires in a handful of epochs, and what that says about the signal-to-noise ratio of game results.
7. **Why a bootstrap over games is not the whole story**: one test season, five seeds, 1,200 games. The interval says nothing about a different season.
8. **The static dashboard decision**: the failure was the platform's sleep policy, so the fix was to remove the process, not to tune the app.
   And what it costs (no live filters, a 60-day scheduled-workflow limit, daily staleness).
9. **The forking path** in section 5. Say it before they ask.
10. **Why team Elo beat a 52-feature regression**: it adjusts for opponent strength and remembers further back with decay. Be able to write the update rule and say what K, the home bonus and the carryover each do.
11. **Why a 7-number model is as good as the networks**, and what that says about signal in game results. The honest summary of section 9 is "ratings carry the signal; the deep model adds a little".
12. **Why you cannot feed a model the year**, and what to give it instead.
13. **What an interviewer might push on**: that you tuned about 1,800 settings on one validation season, that the baseline was weak, that 2025-26 was an easier season for rating models, and that the optimistic roster is still not available before tip-off.

Do not claim: that this beats the market, that the 6.8x bankroll is meaningful, that Jev beat the baseline (it matched it after recalibration), or that the player model is deployed.

## 14. Out of scope, and what it would take

Not done, as instructed. Rough list, in the order I would do it:

1. **Fix the production job first**: handle an empty odds table so a day without priced games still saves predictions; make the season default come from the date;
   decide whether to fix the one-game-stale serving features. Without the first, forward predictions cannot be logged at all.
2. **Serve the model**: export weights (`state_dict` plus the training-season standardisation constants and the 12.0-minute prior), load them in the FastAPI
   container (torch CPU is large; ONNX Runtime or a plain NumPy forward pass of a 3,090-parameter network avoids shipping torch to a 512 MB machine).
3. **Live feature pipeline**: per-player rolling features from `nba_api` at prediction time, an injury/availability source (ESPN statuses), and a rule for the roster when
   status is unknown. This is where the roster optimism gets resolved properly. Plan step 8 (snapshotting ESPN statuses daily) is the data that does not exist retroactively.
4. **Shadow mode**: run next to the logistic regression, store both probabilities, the model version and the feature timestamp per prediction in R2, never show the new one as primary.
5. **Monitoring**: calibration drift over a rolling window, data freshness (age of the newest player row), prediction volume per day, input drift against the training distribution,
   model version per prediction.
6. **Evaluation on forward data only** for 2026-27 (Jev included: its pretraining may contain 2025-26, so only games after setup are clean for it), with a threshold fixed before the first game. The Jev plus player-model hybrid from the vault note is untested.

## 15. Decisions made without asking

- Two roster variants, with the previous-game one as the fair comparison (section 5).
- Antisymmetric head as the default, the plan's plain MLP kept as an option and compared.
- A linear ablation was added to separate "player data" from "deep model".
- Paired bootstrap added so that small gaps are not over-read.
- Jev recalibration (two numbers, validation only) added after seeing its raw result, and reported next to the raw rows.
- `research/` outside `src/` and a separate requirements file, so the production image is unchanged.
- Static site on GitHub Pages rather than Cloudflare Pages; Plotly vendored.
- The legacy Streamlit code was left in place and its README moved to `docs/streamlit-legacy.md`.
- Second session: 8 older seasons as warm-up and as extra training data; team Elo tuned by grid, player Elo by random search, ridge by grid, all on validation with the previous-game roster.
- Added the 7-number model as a second baseline after seeing that the production regression was weak.
- Deep models built so the only difference between the "raw" and "hand-built" arms is who computes the features.
- Reported both near-tied top deep models rather than picking by test result.
- Did not test spectral features, play-by-play data, tree models or a differentiable player Elo (listed in 9.9).
- Commits have plain messages with no attribution trailer, as asked.
