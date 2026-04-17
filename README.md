# NBA Win Probability Prediction System

A fully automated, containerized NBA game win-probability prediction system built with FastAPI, Docker, and Cloudflare R2.

The system implements a complete production-style machine learning pipeline: multi-season data ingestion, feature engineering, model training, scheduled daily inference, and API-based access — all without a traditional database.

All system state (training data, cleaned datasets, trained models, current predictions, and historical results) is persisted exclusively in object storage.

---

## Project Overview

This repository demonstrates how to design and operate a real-world machine learning system rather than a standalone model or notebook.

Key characteristics:

- End-to-end ML pipeline (collection → training → inference)
- Stateless, cloud-native architecture
- Fully automated daily prediction workflow
- Container-oriented design with clear separation of responsibilities
- API-first interface for downstream consumers (dashboards, BI tools, etc.)

---

## Important Disclaimer

This project is **not** a betting system.

- Predictions are produced for technical, analytical, and educational purposes only
- Betting odds are fetched strictly for comparison and visualization
- Odds data does **not** influence model predictions
- The author does **not** support or encourage betting based on these outputs

---

## Key Features

### Data Pipeline

- Multi-season NBA game data ingestion (2020–21 through 2024–25)
- Per-team game log collection from PBPStats with automatic retries
- 6-hour cache on per-team data to avoid redundant PBPStats requests
- Centralized team metadata stored in object storage
- Corrected defensive rating formula (`OppPoints / DefPoss * 100`) for accurate feature engineering
- Cleaned datasets persisted for reproducibility

### Machine Learning

- Model: Logistic Regression (scikit-learn)
- Rationale: stronger real-world stability and better calibration than the experimental custom neural network
- Consistent feature pipeline between training and inference
- Trained model serialized as a skops bundle (model + scaler together) and stored in object storage

### Odds Integration

- Game odds fetched from The Odds API
- Odds stored alongside predictions for comparison
- Odds do **not** affect predictions in any way
- Used strictly for analysis and visualization

### Automation

- Fully automated daily prediction lifecycle:
  - Resolve finished games against ESPN results
  - Archive resolved predictions to history
  - Generate predictions for today's matchups
- Time-zone aware scheduling (Helsinki time, 12:00 daily)
- Safe startup coordination via storage readiness checks

---

## Project Evolution

This project has gone through two major infrastructure generations.

### V1 — Original Stack (Nov 2025 – Apr 2026)

| Component | Technology |
|-----------|------------|
| Backend hosting | AWS EC2 |
| Object storage | AWS S3 |
| Analytics layer | Power BI (embedded in portfolio) |

The original system ran on an AWS EC2 instance with an S3 bucket for model and prediction storage. A Power BI dashboard was embedded in the portfolio site as the analytics layer.

### V2 — Current Stack (Apr 2026–present)

| Component | Technology |
|-----------|------------|
| Backend hosting | Fly.io (arn region) |
| Object storage | Cloudflare R2 |
| Analytics layer | Streamlit Community Cloud |

**Reasons for migration:**
- AWS EC2 and S3 free trials ended — Fly.io (256MB shared machine) and Cloudflare R2 are free at the project's current scale
- Power BI free trial ended — replaced with a custom Streamlit dashboard that is permanently free
- Fly.io simplifies container orchestration and HTTPS termination vs managing an EC2 instance manually
- Cloudflare R2 has no egress fees, which matters when the model bundle is loaded from storage on every prediction request

The S3 client abstraction (`s3_client.py`) was kept S3-compatible so migration required only setting an `endpoint_url` — no application code changes.

---

## Infrastructure

### Hosting — Fly.io

The application runs on [Fly.io](https://fly.io) (`arn` region, 1 GB / 1 CPU).

Previously hosted on AWS EC2; migrated to Fly.io to reduce costs and simplify deployment. Fly handles container orchestration, HTTPS termination, and machine lifecycle automatically.

### Storage — Cloudflare R2

Object storage uses [Cloudflare R2](https://developers.cloudflare.com/r2/).

Previously used AWS S3; migrated to Cloudflare R2 after the AWS free trial ended. R2 is S3-compatible (boto3 works unchanged via `endpoint_url`) and has no egress fees, which significantly reduces costs for a project that reads storage on every prediction request.

The S3Client supports both AWS S3 and R2 via the `R2_ENDPOINT` environment variable — set it for R2, leave blank for AWS.

---

## Architecture Summary

### Bootstrap Container

- Ensures team metadata exists in storage
- Trains the model if the bundle is missing
- Exits after completing setup

### Automation Container

- Runs daily prediction workflow
- Updates current and historical prediction JSON files

### API Container

- Serves prediction endpoints
- Loads the trained model directly from storage at request time

All containers block on storage readiness instead of relying on fragile startup ordering.

---

## Project Structure

```
src/predict_nba/
│
├── automation/
│   ├── automation_runner.py   # Daily scheduler (12:00 Helsinki)
│   ├── daily_generate.py      # ESPN schedule → predictions + odds
│   ├── daily_update.py        # Resolve finished games via ESPN
│   ├── history_manager.py     # Read/write current + history JSON
│   └── predictor_runner.py    # DailyPredictor wrapper
│
├── backend/
│   ├── main.py                # FastAPI app + CORS
│   └── routes/
│       ├── predict.py         # GET /predict
│       └── update.py          # POST /update
│
├── pipeline/
│   ├── bootstrap_model.py     # One-shot setup: teams + model
│   ├── data_collector.py      # PBPStats ingestion + 6h cache
│   ├── data_cleaner.py        # Feature engineering, rolling avgs
│   ├── make_prediction.py     # Orchestrates collect → clean → predict
│   ├── model_predictor.py     # Loads bundle, runs inference
│   └── model_trainer.py       # Trains LogisticRegression, saves bundle
│
└── utils/
    ├── s3_client.py           # S3/R2 upload + download helper
    ├── espn.py                # Abbreviation normalization, date conversion
    ├── oddsfetcher.py         # The Odds API integration
    ├── logger.py              # Rotating file + console logger
    ├── exception.py           # CustomException (non-halting, logs trace)
    └── wait_for_model.py      # Polls storage until model bundle is ready
```

---

## Storage Layout

```
<bucket>/
├── teams/
│   └── teams.json                  # All 30 NBA team IDs and names
├── training/
│   └── training_data.csv           # Raw multi-season game logs
├── clean/
│   └── training_data_clean.csv     # Engineered feature dataset
├── models/
│   └── prediction_model.skops      # skops bundle: {model, scaler}
├── current/
│   └── current_predictions.json    # Unresolved predictions for today
├── history/
│   └── prediction_history.json     # Full prediction + outcome history
└── predict/
    ├── <TEAM>.csv                   # Latest season logs (6h cached)
    └── clean/
        └── <TEAM1>vs<TEAM2>.csv     # Cleaned matchup row for inference
```

Storage acts as configuration store, feature store, model registry, prediction output, and historical audit log.

---

## Environment Configuration

```
STORAGE_BUCKET=
STORAGE_REGION=
STORAGE_ACCESS_KEY=
STORAGE_SECRET_KEY=
# For Cloudflare R2: https://<account_id>.r2.cloudflarestorage.com
# For AWS S3: leave blank
R2_ENDPOINT=
ODDS_API_KEY=
```

No credentials are committed to the repository.

---

## Bootstrapping Workflow

On first deployment:

1. Check if a trained model bundle exists in storage
2. If missing:
   - Fetch and normalize team metadata from PBPStats
   - Collect multi-season training data (5 seasons)
   - Clean and engineer features
   - Train logistic regression model
   - Upload bundle (`{model, scaler}`) to storage
3. Exit cleanly

Subsequent deployments skip all steps and reuse existing artifacts.

---

## Daily Automation Flow

Executed by the automation container at 12:00 Helsinki time:

1. **Resolve finished games**
   - Load unresolved predictions from storage
   - Check real results from ESPN scoreboard
   - Move resolved games (with correctness flag) to history

2. **Generate new predictions**
   - Fetch today's schedule from ESPN
   - Collect latest team statistics (with 6-hour cache to avoid redundant API calls)
   - Run inference using the trained model
   - Fetch betting odds for context
   - Upload results to current predictions

3. **Sleep** until next scheduled run

---

## API Endpoints

### GET /predict?team1=&team2=

Predict a single matchup using the trained model.

- `team1`: home team abbreviation (e.g. `CLE`)
- `team2`: away team abbreviation (e.g. `ATL`)

Response:
```json
{"winner": "CLE", "confidence": 72.5}
```

### POST /update

Manually triggers the same workflow as the daily automation:
- Resolve finished games
- Generate today's predictions

---

## Design Goals

This project prioritizes:

- System reliability over theoretical model complexity
- Production realism over notebook experimentation
- Reproducibility and automation
- Clear separation of responsibilities
- Cost-efficient, cloud-native architecture

---

## Summary

This repository demonstrates:

- End-to-end machine learning engineering
- Practical cloud architecture using Cloudflare R2 and Fly.io
- Containerized automation and APIs
- Robust, production-quality data pipelines
- Thoughtful infrastructure decisions driven by real cost and operational constraints

It is intended as a portfolio-quality example of how to build, operate, and reason about a real ML system.

If you like this project, consider giving it a GitHub star.
