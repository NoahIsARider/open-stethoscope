#!/bin/bash
# Open Stethoscope QC companion — start both services
# Backend (FastAPI, port 3001) + Frontend dev server (Vite, exposed port).
cd "$(dirname "$0")"

echo "[start] backend on :3001"
python3 -m uvicorn main:app --host 0.0.0.0 --port 3001 --app-dir app &
BACKEND_PID=$!

trap "kill $BACKEND_PID 2>/dev/null" EXIT

echo "[start] frontend on :5173"
cd web && npm run dev
