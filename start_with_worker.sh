#!/bin/bash

# Production startup script for Render with background worker
echo "Starting RFP Backend with Background Worker on Render..."

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

# Start the background worker in the background
echo "Starting background worker process..."
python worker.py &
WORKER_PID=$!

# Function to cleanup worker process on exit
cleanup() {
    echo "Shutting down worker process (PID: $WORKER_PID)..."
    kill $WORKER_PID 2>/dev/null
    wait $WORKER_PID 2>/dev/null
    echo "Worker process stopped"
}

# Set up signal handlers for graceful shutdown
trap cleanup SIGTERM SIGINT

echo "Starting Gunicorn server with Uvicorn workers..."

# Start the application with Gunicorn for production
exec gunicorn app:app --config gunicorn.conf.py
