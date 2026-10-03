#!/bin/bash
# Change to repository root directory
cd "$(dirname "$0")/.." || exit 1

while true; do
    echo "Starting Uvicorn..."
    source .venv/bin/activate && uvicorn exotel_server:app --host 0.0.0.0 --port 8000
    echo "Server crashed! Restarting in 3 seconds..."
    sleep 3
done
