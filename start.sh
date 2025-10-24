#!/bin/bash

# Production startup script for Render
echo "Starting RFP Backend on Render..."

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
echo "Starting uvicorn server..."

# Start the application
exec uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1
