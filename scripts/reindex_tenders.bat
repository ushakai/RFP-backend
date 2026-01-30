@echo off
REM Batch script to re-index existing tenders with search data
REM This extracts keywords, locations, and industries from all existing tenders

echo ========================================
echo Tender Reindexing Script
echo ========================================
echo.

cd /d "%~dp0\.."

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo Starting tender reindexing...
echo.

REM Run the reindexing script
REM Default: Process all tenders in batches of 100
python scripts/reindex_existing_tenders.py --batch-size 100

if errorlevel 1 (
    echo.
    echo ERROR: Reindexing failed with errors
    pause
    exit /b 1
) else (
    echo.
    echo SUCCESS: Reindexing completed successfully
    pause
    exit /b 0
)


