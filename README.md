# NBA Win Probability Prediction System

A fully automated, containerized NBA game win-probability prediction system built with FastAPI, Docker, and AWS S3.

The system implements a complete production-style machine learning pipeline: multi-season data ingestion, feature engineering, model training, scheduled daily inference, and API-based access, all without a traditional database.

All system state (training data, cleaned datasets, trained models, current predictions, and historical results) is persisted exclusively in Amazon S3.

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

- Multi-season NBA game data ingestion (currently 2020–21 through 2024–25)
- Per-team game log collection from PBPStats
- Centralized team metadata stored in S3
- Robust cleaning and feature engineering
- Cleaned datasets persisted for reproducibility

### Machine Learning

- Model: Logistic Regression (current production model)
- Rationale:
  - Stronger real-world stability than the experimental neural network
  - Better calibration and interpretability
- Consistent feature pipeline between training and inference
- Trained model serialized and stored in S3

### Odds Integration

- Game odds fetched from The Odds API
- Odds stored alongside predictions for comparison
- Odds do **not** affect predictions in any way
- Used strictly for analysis and visualization

### Automation

- Fully automated daily prediction lifecycle:
  - Resolve finished games
  - Archive resolved predictions
  - Generate predictions for today’s matchups
- Time-zone aware scheduling (Helsinki time)
- Safe startup coordination via S3 readiness checks

---

## Infrastructure

- Dockerized services with strict separation of responsibilities
- No external database
- No local persistence
- Entire system reproducible from S3 + containers

---

## Architecture Summary

### Bootstrap Container

- Ensures team metadata exists in S3
- Trains the model if missing

### Automation Container

- Runs daily prediction workflow
- Updates current and historical predictions

### API Container

- Serves prediction endpoints
- Loads the trained model directly from S3

All containers block on S3 readiness instead of relying on fragile startup ordering.

---

## Project Structure

```
src/predict_nba/
│
├── automation/
│   ├── automation_runner.py
│   ├── daily_generate.py
│   ├── daily_update.py
│   ├── history_manager.py
│   └── predictor_runner.py
│
├── backend/
│   ├── main.py
│   └── routes/
│       ├── predict.py
│       └── update.py
│
├── pipeline/
│   ├── bootstrap_model.py
│   ├── data_collector.py
│   ├── data_cleaner.py
│   ├── make_prediction.py
│   ├── model_predictor.py
│   └── model_trainer.py
│
├── utils/
│   ├── s3_client.py
│   ├── logger.py
│   ├── exception.py
│   └── wait_for_model.py
│
└── __init__.py
```

---

## S3 Storage Layout

```
teams/
  teams.json

training/
  training_data.csv

clean/
  training_data_clean.csv

models/
  prediction_model.skops

current/
  current_predictions.json

history/
  prediction_history.json
```

S3 acts as configuration store, feature store, model registry, prediction output store, and historical audit log.

---

## Environment Configuration

Example `.env.example`:

```
AWS_S3_BUCKET_NAME=
AWS_REGION=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
ODDS_API_KEY=
```

No credentials are committed to the repository.

---

## Bootstrapping Workflow

On first deployment:

1. Check if a trained model exists in S3
2. If missing:
   - Fetch and normalize team metadata
   - Collect multi-season training data
   - Clean and engineer features
   - Train logistic regression model
   - Upload trained model to S3
3. Exit cleanly

Subsequent deployments reuse existing artifacts.

---

## Daily Automation Flow

Executed by the automation container:

1. Resolve finished games
   - Compare active predictions against real results
   - Move resolved games to history

2. Generate new predictions
   - Fetch today’s matchups
   - Fetch latest team statistics
   - Fetch game odds (reference only)
   - Run inference using trained model
   - Upload results to current predictions

3. Sleep until next scheduled run (12:00 Helsinki)

---

## API Endpoints

### POST /predict

Predict a single matchup using the trained model.

Input:
- Home team abbreviation
- Away team abbreviation

Output:
- Home win probability
- Optional odds context (if available)

### POST /update

Manually triggers the same workflow as the daily automation:
- Resolve finished games
- Generate today’s predictions

---

## Design Goals

This project prioritizes:

- System reliability over theoretical model complexity
- Production realism over notebook experimentation
- Reproducibility and automation
- Clear separation of responsibilities
- Cloud-native, stateless architecture

---

## Summary

This repository demonstrates:

- End-to-end machine learning engineering
- Practical cloud architecture using Amazon S3
- Containerized automation and APIs
- Robust data pipelines
- Thoughtful model selection based on real-world performance

It is intended as a portfolio-quality example of how to build, operate, and reason about a real ML system.

If you like this project, consider giving it a GitHub star.
