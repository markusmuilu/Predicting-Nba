# 📊 NBA Game Prediction System

> **TL;DR:**  
A fully automated, containerized NBA win-probability prediction system built with FastAPI, scikit-learn, Docker, and AWS S3, using my own built neural network utilizing only numpy.  
It collects multi-season NBA data, trains an ML model stored on S3, runs daily predictions inside a Docker automation container, and serves results through a FastAPI API — all without a traditional database.  
Originally planned for Supabase, but switched entirely to S3 due to better compatibility with Power BI.  
Everything is stateless and S3-driven: training data, cleaned datasets, models, current predictions, and history all live in S3.



Machine-learning powered NBA win probability prediction pipeline with FastAPI, Docker, and AWS S3 storage.

This project fully automates:

- 🏀 Fetching NBA game logs from PBPStats API
- 🧹 Cleaning + transforming multi-season data
- 🤖 Training a ML model using own built neural network (stored in S3)
- 🔄 Running daily prediction automation (scheduled inside Docker)
- 🌐 Serving new predictions over a FastAPI REST API
- ☁️ Managing training data, models, and predictions in S3 only
- No Supabase, no external DB, fully S3-based and stateless.

# 🚀 Features
Data Pipeline

- Multi-season training data collection (2015-16 → 2024–25)
- Automatic team metadata loading from teams/teams.json in S3
- Per-team game logs fetched via PBPStats
- Cleaned + feature engineered dataset uploaded back to S3
- Model training with scikit-learn MLP(currently using custom nn) and model stored in S3

Automated Daily System

- Detects finished games from ESPN scoreboar
- Moves completed predictions → history in S3
- Fetches today’s matchups
- Generates fresh predictions and uploads:
- current/current_predictions.json
- history/prediction_history.json

FastAPI Backend

- /predict – Predict outcome for a given matchup
- /update – Manually trigger daily update (same logic as automation)
- Loads the trained model directly from S3 via model_predictor.py

Container-Oriented Architecture

- bootstrap container:
  - Ensures teams/teams.json exists in S3 (from PBPStats /get-teams/nba)
  - Trains model if models/prediction_model.skops is missing

- api container:

  - FastAPI app, waits until model exists in S3

- automation container:
  - Periodic job runner, waits until model exists in S3


# 🧱 Project Structure
```
src/predict_nba/
│   __init__.py
│
├── automation/                    # Automation & scheduling layer
│   │   automation_runner.py       # Main scheduler: runs daily jobs at 12:00 Helsinki
│   │   daily_generate.py          # Generates today's predictions from ESPN schedule
│   │   daily_update.py            # Resolves finished games & updates history
│   │   history_manager.py         # Read/write current/history JSON in S3
│   │   predictor_runner.py        # Legacy/compatibility runner wrapper
│   │   __init__.py
│
├── backend/                       # FastAPI backend
│   │   main.py                    # FastAPI app instance & router registration
│   │   __init__.py
│   │
│   └── routes/
│       │   predict.py             # POST /predict (single matchup prediction)
│       │   update.py              # POST /update (manual daily run)
│       │   __init__.py
│
├── pipeline/                      # Core ML & data pipeline
│   │   bootstrap_model.py         # Ensures teams.json + trains & uploads model if missing
│   │   data_cleaner.py            # Cleans training data and engineer features
│   │   data_collector.py          # Collects raw logs from PBPStats into training CSV
│   │   make_prediction.py         # High-level "predict this matchup" function
│   │   model_predictor.py         # Loads model from S3, runs predict_proba
│   │   model_trainer.py           # Trains scikit-learn MLP and uploads .skops bundle
│   │   __init__.py
│
├── utils/                         # Shared utilities
│   │   espn_utils.py              # ESPN helpers (dates, IDs, mapping, etc.)
│   │   exception.py               # CustomException with improved trace formatting
│   │   logger.py                  # Project-wide structured logger
│   │   s3_client.py               # Generic JSON & bytes S3 helper (upload/download)
│   │   wait_for_model.py          # Blocks until model exists in S3 (used by automation/api)
│   │   __init__.py
│
└── __pycache__/                   # Python bytecode (ignored in Git)
```

Root-level files:
```
.github/workflows/deploy     # CI/CD or deployment workflow (if configured)
.env                         # Local env vars (not committed)
.dockerignore                # Files ignored by Docker build context
.gitignore                   # Git ignore rules
docker-compose.yml           # Multi-container dev/prod stack
Dockerfile                   # Base image for all three services
pyproject.toml               # Package metadata + dependencies (installable via pip)
requirements.txt             # Resolved dependency versions (optional helper)
setup_project.py             # Legacy initial setup script (superseded by bootstrap_model.py)
README.md                    # This file
```
# ⚙️ Environment & S3 Layout
Required env variables (.env)
```
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_S3_BUCKET_NAME=nbatraindata
AWS_REGION=eu-north-1
```
S3 keys used by the system

- Config
  - teams/teams.json – auto-created by bootstrap_model.py from PBPStats /get-teams/nba

- Training data
  - training/training_data.csv – raw multi-season logs
  - clean/training_data_clean.csv – cleaned, feature-engineered dataset

- Model
  - models/prediction_model.skops – trained scikit-learn MLP bundle

- Predictions
  - current/current_predictions.json – active (unresolved) predictions
  - history/prediction_history.json – full prediction/result history

# 🛠 How Bootstrapping Works

pipeline/bootstrap_model.py:

1. Check for model:

- If models/prediction_model.skops exists in S3 → log and exit cleanly.

2. Ensure team metadata:

- Check for teams/teams.json in S3.
- If missing:
  - Fetch from https://api.pbpstats.com/get-teams/nba
  - Normalize to:
```
{
  "teams": [
    { "id": 1610612737, "name": "ATL" },
    ...
  ]
}
```
  - Upload via S3Client.upload_json(...).

3. Collect multi-season training data:

- Uses DataCollector.collect_training_data(seasons)
- Default seasons:
```
seasons = ["2021-22", "2022-23", "2023-24", "2024-25"]
```

- Loops all teams in teams.json for each season.
- Each team-season fetches:
  - https://api.pbpstats.com/get-game-logs/nba
- Combines all logs into a single training/training_data.csv in S3.

4. Clean the training data:

- DataCleaner.clean_training_data()
  - Downloads training/training_data.csv
  - Fetches home/away mapping via PBPStats /get-games/nba
  - Merges, imputes, engineers advanced features
  - Uploads clean/training_data_clean.csv to S3

5. Train the ML model:

- ModelTrainer.train_model():
  - Loads clean/training_data_clean.csv from S3
  - Builds a feature matrix with ~44 feature
  - Trains an MLPClassifier (scikit-learn)
  - Logs performance (accuracy, ROC-AUC, classification report)
  - Serializes with skops and uploads models/prediction_model.skops to S3

# 🐳 Docker Architecture
docker-compose.yml (conceptual)

- bootstrap (one-shot)
  - Runs python3 -m predict_nba.pipeline.bootstrap_model

- api


  - Runs:
```
uvicorn predict_nba.backend.main:app --host 0.0.0.0 --port 8000
```
  - Waits for S3 model and teams files to appear before starting
- automation
  - Also waits for S3 model and teams files to appear before starting
  - Runs:
```
python3 -m predict_nba.automation.automation_runner
```

This ensures:

1. Model gets trained at first run (if missing).

2. Only after that:
  - API starts and can successfully load model from S3.
  - Automation starts and can call model_predictor.py safely.

# 🔄 Daily Automation Flow

Main entry: automation/automation_runner.py

- On startup:
  - Logs “Running FIRST automation job immediately…”
  - Calls DailyPredictor (in pipeline/daily_predictor.py) to:

    1. Resolve finished games using ESPN:
      - Loads current/current_predictions.json from S
      - Checks real results from ESPN scoreboard
      - Moves them to history/prediction_history.json

    2. Generate today’s predictions:
      - Calls PBPStats to pull latest logs for teams playing toda
      - Cleans them using data_cleaner
      - Uses model_predictor + make_prediction to produce probabilitie
      - Writes to current/current_predictions.json in S3

- After first run:

  - Computes next 12:00 Helsinki time using TZ-aware logic
  - Sleeps until then
  - Repeats the same cycle every day

# 🌐 FastAPI Endpoints
backend/main.py

Sets up the FastAPI app, adds routers from backend.routes.predict and backend.routes.update.

POST /predict

Predict the outcome of a specific matchup:

Request body example:
```
{
  "home": "BOS",
  "away": "LAL"
}
```

Response example:
```
{
  "home": "BOS",
  "away": "LAL",
  "prob_home_win": 0.72
}
```

Internally uses:

- make_prediction.py → which calls
- model_predictor.py (to load model from S3)
- And uses feature engineering consistent with training.

# POST /update

Manual trigger for the same logic automation runs:

- Resolve finished games
- Generate predictions for today

Useful for debugging or on-demand refreshes.

# 🧠 Model Details

- Algorithm: MLPClassifier from scikit-learn
- Hidden layers: (256, 128, 64)
- Metrics logged:
  - Accuracy
  - ROC-AUC
  - Full classification report

- Training data:
  - Multi-season dataset 2021–22 → 2024–25
  - Cleaned and engineered features stored in clean/training_data_clean.csv

- Storage:
  - Binary model bundle in S3: models/prediction_model.skops

#🧾 File-by-File Overview
### `automation/`

| File                  | Role |
|-----------------------|------|
| `automation_runner.py`| Scheduler + main loop for daily automation |
| `daily_generate.py`   | Generates predictions for today’s games |
| `daily_update.py`     | Resolves finished games and updates history JSON |
| `espn_utils.py`       | Helper functions for ESPN APIs, IDs, and dates |
| `history_manager.py`  | Loads/saves current + history JSON from S3 |
| `predictor_runner.py` | Legacy runner |
| `__init__.py`         | Package initialization |

### `backend/`

| File                 | Role                               |
|----------------------|------------------------------------|
| `main.py`            | FastAPI app, mounts all routes     |
| `routes/predict.py`  | Implements `/predict` endpoint     |
| `routes/update.py`   | Implements `/update` endpoint      |
| `routes/__init__.py` | Router package setup               |


### `pipeline/`

| File                 | Role                                                                      |
|----------------------|---------------------------------------------------------------------------|
| `bootstrap_model.py` | Creates `teams.json` (if missing) and trains model (if missing)          |
| `data_collector.py`  | Downloads multi-season training game logs from PBPStats                  |
| `data_cleaner.py`    | Cleans data, merges home/away mapping, and engineers features            |
| `model_trainer.py`   | Trains MLP model and uploads `models/prediction_model.skops` to S3       |
| `model_predictor.py` | Downloads model from S3 and runs predictions                             |
| `make_prediction.py` | Single-game prediction helper                                            |
| `daily_predictor.py` | Orchestrates daily update + prediction generation workflow               |
| `__init__.py`        | Package initialization                                                   |


### `utils/`

| File               | Role                                                     |
|--------------------|----------------------------------------------------------|
| `s3_client.py`     | Generic JSON + bytes upload/download helper for S3       |
| `logger.py`        | Centralized logger used across modules                   |
| `exception.py`     | `CustomException` with formatted stack traces            |
| `wait_for_model.py`| Blocks process until model file exists in S3             |
| `__init__.py`      | Package initialization                                   |
# 🏁 Summary

This project gives you:

- A complete end-to-end NBA prediction system
- A clean, modular Python package (predict_nba) installable via pip
- Stateless architecture: everything persisted in S3
- A reusable infrastructure pattern:
- bootstrap job → automation jobs → API sitting on top
  
Perfect for:

- Portfolio projects demonstrating:
  - ML
  - Cloud (S3)
  - Containers (Docker)
  - Backend (FastAPI)
  - Automation / scheduling
  - Real-world use where you just hook a front-end or BI tool (like Power BI) to S3 outputs.


# ⭐ If you like this project, consider giving it a GitHub star!








