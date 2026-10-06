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
- **The neural network is not what helps.** A plain logistic regression on the same player features does as well
  (test log loss 0.5936 vs 0.5895, validation 0.6019 vs 0.6024). The gain comes from the player data, not from the deep model.
- **The optimistic version looks like it matches the betting market and it does not mean that.** On the 680 games with stored
  odds, the actual-roster model has log loss 0.5427 against the market's 0.5432. That is because it knows who played and
  the market, priced hours earlier, did not. The same model with a previous-game roster scores 0.5573, clearly worse than the market.
  Its +6.8x quarter-Kelly bankroll is an artefact, not an edge.
- **Jev (Part 2) was not run.** `JEV_API_KEY` was not set. The code is written and tested only to the point of exiting
  cleanly without a key. No Jev number exists anywhere in this work.
- **The Fly.io service is alive** but has stopped saving predictions (section 10).
- **The dashboard is now a static site** rebuilt daily by a GitHub Action (section 9). It needs three manual steps from you to go live.

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
| `research/jev.py` | Part 2. Written, never run with a key. |
| `research/results/` | Committed outputs: per-game predictions per model, metrics, plots. |
| `data/` | The raw download cache. Git-ignored, about 75 MB. |

To reproduce (from the repo root; `pip install -r research/requirements.txt`; on Windows use `.venv\Scripts\python`):

```
python -m research.fetch            # fills data/raw, about 20 minutes, rate limited
python -m research.baseline
python -m research.player_features actual && python -m research.train actual
python -m research.player_features previous && python -m research.train previous
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

### Test season 2025-26

| Model | Accuracy | Brier | Log loss | ECE |
|---|---|---|---|---|
| Logistic regression (production features) | 0.667 | 0.2093 | 0.6057 | 0.025 |
| Player model, previous-game roster | 0.676 | 0.2027 | 0.5895 | 0.026 |
| Linear on player features, previous-game roster | 0.678 | 0.2045 | 0.5936 | 0.027 |
| Player model, **actual roster (optimistic)** | 0.693 | 0.1976 | 0.5782 | 0.029 |
| Linear on player features, actual roster (optimistic) | 0.693 | 0.1988 | 0.5808 | 0.021 |
| Constant (home win rate 55.3%) | 0.553 | 0.2472 | 0.6874 | n/a |

### Validation season 2024-25

| Model | Accuracy | Brier | Log loss | ECE |
|---|---|---|---|---|
| Logistic regression (production features) | 0.679 | 0.2104 | 0.6077 | 0.035 |
| Player model, previous-game roster | 0.684 | 0.2077 | 0.6024 | 0.029 |
| Linear on player features, previous-game roster | 0.689 | 0.2073 | 0.6019 | 0.041 |
| Player model, actual roster (optimistic) | 0.690 | 0.2022 | 0.5892 | 0.033 |
| Linear on player features, actual roster (optimistic) | 0.698 | 0.2013 | 0.5874 | 0.024 |

### Improvement in log loss over the baseline, 95% bootstrap interval (positive = better)

| Model | Validation | Test |
|---|---|---|
| Player model, previous-game roster | +0.005 [-0.007, +0.018] | +0.016 [+0.004, +0.029] |
| Linear on player features, previous-game roster | +0.006 [-0.007, +0.018] | +0.012 [-0.001, +0.025] |
| Player model, actual roster (optimistic) | +0.019 [+0.002, +0.034] | +0.028 [+0.012, +0.043] |

**How to read this honestly.** With a fair roster the player model is slightly better than the baseline on both seasons, but the validation
gap is inside the noise, the test gap just clears it, and a linear model on the same inputs is within noise of the network. The defensible
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
| Player model, actual roster (optimistic) | 0.722 | 0.1825 | 0.5427 | +3.2% | 6.79x | 59% |
| Linear, actual roster (optimistic) | 0.724 | 0.1869 | 0.5532 | +4.5% | 1.65x | 66% |

![Quarter-Kelly bankroll over the 680 games with odds](../research/results/bankroll_test.png)

- Every fair model is **worse than the market** on log loss. The market is the benchmark that matters, and nobody here has beaten it.
- The three fair models lose money under the quarter-Kelly rule, and lose most of the bankroll (drawdowns of 85% to 98%). That is what betting on
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

## 8. Jev (Part 2): not run

The task said to run it only if `JEV_API_KEY` is set. It was not set (checked as a process, user and machine environment variable) so nothing was
sent anywhere. What exists:

- `research/jev.py` follows `docs.typesafe.ai/api`: `POST https://api.typesafe.ai/v1/systemone`, bearer auth, a `state` object, one `noul`
  (yes/no) question whose answer is a probability, `jev-latest`. The response carries `usage.input_tokens`, which the script sums.
- It builds the same pre-game information the models get (season record, last-10 ratings from the production cleaner's columns, rest, back-to-back,
  and each team's eight players with most expected minutes and their pre-game rolling averages), with no results.
- It runs each validation and test game twice, **named** and **anonymised** ("Home team", "Home player 3", no date). A model that has memorised
  results can recall them from names. A large named-minus-anonymised gap would be memory, not skill; the anonymised score would be the honest one.
- It caches every response in `data/raw/jev/`, stops before 50 million input tokens, writes the token total to `research/results/jev_usage.json`,
  reads the key only from the environment, and never prints or writes it.
- **Untested against the live API**, because there was no key. The first thing to check on a real run is that `answers.home_wins.noul` is the
  field name (taken from the docs' examples), then to run it on a handful of games and look at the usage numbers before the full run.
  The roster here uses who played, so the same optimism applies and the fair Jev run should use the previous-game roster too (not yet wired in).
- The dashboard's comparison tab shows a "not run" notice and fills in automatically once `pred_jev_*.csv` files exist and `report.py` is re-run.
- Access note from the vault: direct signup was paused on 22 Sep; I did not check whether that has changed.

## 9. The dashboard

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
   **It is empty right now and says so**, because no 2026-27 regular-season game has been scored yet (section 10 explains why preseason games are missing).
2. *Model comparison*: the test-season table, the paired-bootstrap table, calibration of the models, the market table, and a visible warning about the
   optimistic roster rows. The Jev rows are absent with a "not run" note.
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

## 10. The Fly.io service (read only, nothing redeployed)

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

## 11. Bugs, wrong turns and how they were found

| What | How found | Fix |
|---|---|---|
| R2 listing failed with `NoneType` bucket | Traceback from `boto3` in a scratch script | `load_dotenv()` searches from the script's folder; pass the path to the repo `.env` explicitly |
| `plotly-basic.min.js` 404 on cdnjs | `curl -I` on the URL before relying on it | Vendored the file from jsDelivr into `site/vendor/` so there is no runtime CDN at all |
| Brier and log loss `NaN` in the dashboard data | The build printed `nan` | 151 history rows (21 Oct 2025 to 25 Jan 2026) hold a result and no prediction; they are excluded from scoring and counted. The same filter was added to `report.py`, where they would have become NaN silently |
| Charts drawn over the text below them | Screenshot of the page | Plotly does not size its own container reliably inside a hidden tab; the card now gets an explicit height |
| Bar labels printed inside every team bar | Same screenshot | `textposition: "none"` |
| Syntax error in `report.py` | Python refused to import it | A `\n` in my edit script became a literal newline inside an f-string |
| Player-model results looked too good | Log loss equal to the market's on odds games, which is implausible for a free-data model | The "previous-game roster" variant (section 5) |
| 5 baseline games per season had no counterpart | Counted 1,214 vs 1,225 games when comparing sets | They exist in PBPStats only (apparently NBA Cup knockouts); the comparison uses the intersection |

## 12. What you would need to understand to defend this in an interview

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

Do not claim: that this beats the market, that the 6.8x bankroll is meaningful, that Jev was evaluated, or that the player model is deployed.

## 13. Out of scope, and what it would take

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
6. **Evaluation on forward data only** for 2026-27 (including Jev if access exists), with a threshold fixed before the first game.

## 14. Decisions made without asking

- Two roster variants, with the previous-game one as the fair comparison (section 5).
- Antisymmetric head as the default, the plan's plain MLP kept as an option and compared.
- A linear ablation was added to separate "player data" from "deep model".
- Paired bootstrap added so that small gaps are not over-read.
- `research/` outside `src/` and a separate requirements file, so the production image is unchanged.
- Static site on GitHub Pages rather than Cloudflare Pages; Plotly vendored.
- The legacy Streamlit code was left in place and its README moved to `docs/streamlit-legacy.md`.
- Commits have plain messages with no attribution trailer, as asked.
