#!/bin/bash

# Production startup script for Render with 4 background workers
echo "Starting RFP Backend with 4 Background Workers on Render..."

# Check if required environment variables are set
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "ERROR: GOOGLE_API_KEY environment variable is not set"
    exit 1
fi

if [ -z "$SUPABASE_URL" ]; then
    echo "ERROR: SUPABASE_URL environment variable is not set"
    exit 1
fi

if [ -z "$SUPABASE_KEY" ]; then
    echo "ERROR: SUPABASE_KEY environment variable is not set"
    exit 1
fi

echo "Environment variables validated successfully"

# Array to store worker PIDs
WORKER_PIDS=()

# Function to cleanup all worker processes on exit
cleanup() {
    echo "Shutting down all worker processes..."
    for pid in "${WORKER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping worker process (PID: $pid)..."
            kill "$pid" 2>/dev/null
            wait "$pid" 2>/dev/null
            echo "Worker process $pid stopped"
        fi
    done
    echo "All worker processes stopped"
}

# Set up signal handlers for graceful shutdown
trap cleanup SIGTERM SIGINT

# Start 4 background workers
echo "Starting 4 background worker processes..."
for i in {1..4}; do
    echo "Starting worker $i..."
    python worker.py &
    WORKER_PID=$!
    WORKER_PIDS+=($WORKER_PID)
    echo "Worker $i started with PID: $WORKER_PID"
    sleep 2  # Small delay between worker starts
done

echo "All 4 workers started successfully"
echo "Starting Gunicorn server with Uvicorn workers..."

# Start the application with Gunicorn for production
exec gunicorn app:app --config gunicorn.conf.py
