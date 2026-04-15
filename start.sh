#!/bin/bash
set -e

echo "=== Running bootstrap ==="
python -m predict_nba.pipeline.bootstrap_model

echo "=== Starting API ==="
uvicorn predict_nba.backend.main:app --host 0.0.0.0 --port 8000 &

echo "=== Starting automation ==="
python -m predict_nba.automation.automation_runner &

echo "=== All services started ==="
wait