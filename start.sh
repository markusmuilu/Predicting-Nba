#!/bin/bash
set -e

echo "=== Running bootstrap ==="
python -m predict_nba.pipeline.bootstrap_model
set +e

# Each service restarts if it exits. The machine has 512 MB, and when memory runs out the kernel kills the largest
# process; on 20 Oct 2026 that was the API, which then stayed down while the prediction loop kept running.
keep() {
  while true; do
    "$@"
    echo "=== $1 exited with $?, restarting in 5 s ==="
    sleep 5
  done
}

echo "=== Starting API ==="
keep uvicorn predict_nba.backend.main:app --host 0.0.0.0 --port 8000 &

echo "=== Starting automation ==="
keep python -m predict_nba.automation.automation_runner &

echo "=== Starting player model loop ==="
keep python -m predict_nba.player.runner &

echo "=== All services started ==="
wait
