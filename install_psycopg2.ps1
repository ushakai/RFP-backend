# PowerShell script to install psycopg2-binary on Windows
# This script tries multiple methods to install psycopg2-binary

Write-Host "Attempting to install psycopg2-binary..." -ForegroundColor Cyan

# Method 1: Try installing from pre-built wheels (recommended)
Write-Host "`nMethod 1: Installing from pre-built wheels..." -ForegroundColor Yellow
try {
    pip install psycopg2-binary --only-binary :all: --no-cache-dir
    Write-Host "✓ Successfully installed psycopg2-binary!" -ForegroundColor Green
    exit 0
} catch {
    Write-Host "✗ Method 1 failed: $_" -ForegroundColor Red
}

# Method 2: Try specific version with pre-built wheel
Write-Host "`nMethod 2: Installing specific version (2.9.9)..." -ForegroundColor Yellow
try {
    pip install psycopg2-binary==2.9.9 --only-binary :all: --no-cache-dir
    Write-Host "✓ Successfully installed psycopg2-binary!" -ForegroundColor Green
    exit 0
} catch {
    Write-Host "✗ Method 2 failed: $_" -ForegroundColor Red
}

# Method 3: Try without --only-binary (may download wheel automatically)
Write-Host "`nMethod 3: Installing without binary restriction..." -ForegroundColor Yellow
try {
    pip install psycopg2-binary --no-cache-dir
    Write-Host "✓ Successfully installed psycopg2-binary!" -ForegroundColor Green
    exit 0
} catch {
    Write-Host "✗ Method 3 failed: $_" -ForegroundColor Red
}

# Method 4: Try upgrading pip first, then install
Write-Host "`nMethod 4: Upgrading pip and retrying..." -ForegroundColor Yellow
try {
    python -m pip install --upgrade pip
    pip install psycopg2-binary --only-binary :all: --no-cache-dir
    Write-Host "✓ Successfully installed psycopg2-binary!" -ForegroundColor Green
    exit 0
} catch {
    Write-Host "✗ Method 4 failed: $_" -ForegroundColor Red
}

Write-Host "`n❌ All installation methods failed!" -ForegroundColor Red
Write-Host "`nAlternative solutions:" -ForegroundColor Yellow
Write-Host "1. Install PostgreSQL from https://www.postgresql.org/download/windows/"
Write-Host "2. Use WSL (Windows Subsystem for Linux) and install there"
Write-Host "3. Use Docker to run your backend"
Write-Host "4. Continue using Supabase REST API (no psycopg2 needed)"
exit 1


