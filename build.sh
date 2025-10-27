#!/bin/bash

# Build script for Render deployment with pandas compatibility fixes
echo "Starting build process for RFP Backend..."

# Upgrade pip first
echo "Upgrading pip..."
pip install --upgrade pip

# Install system dependencies if needed
echo "Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq \
    build-essential \
    libffi-dev \
    libssl-dev \
    python3-dev \
    pkg-config

# Install Python dependencies with specific pandas handling
echo "Installing Python dependencies..."

# Try to install pandas with pre-compiled wheels first
pip install --only-binary=all pandas>=2.1.0,<2.3.0 || {
    echo "Pre-compiled pandas failed, trying with compilation..."
    pip install pandas>=2.1.0,<2.3.0
}

# Install other dependencies
pip install --no-cache-dir -r requirements.txt

# Verify installation
echo "Verifying installation..."
python -c "import fastapi, uvicorn, pandas, supabase, google.generativeai; print('All dependencies installed successfully')"

echo "Build completed successfully!"
