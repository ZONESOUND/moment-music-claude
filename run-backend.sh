#!/bin/bash

# Run the backend server only
# API server: http://localhost:8100

cd "$(dirname "$0")/backend"

echo "Starting backend on http://localhost:8100..."
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8100 --loop asyncio
