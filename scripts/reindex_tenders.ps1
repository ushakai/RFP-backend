# PowerShell script to re-index existing tenders with search data
# This extracts keywords, locations, and industries from all existing tenders

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Tender Reindexing Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Change to script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $scriptDir "..")

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Starting tender reindexing..." -ForegroundColor Yellow
Write-Host ""

# Parse command line arguments
$batchSize = 100
$maxTenders = $null
$dryRun = $false
$force = $false

# Simple argument parsing
foreach ($arg in $args) {
    if ($arg -match "--batch-size=(\d+)") {
        $batchSize = [int]$matches[1]
    } elseif ($arg -match "--max-tenders=(\d+)") {
        $maxTenders = [int]$matches[1]
    } elseif ($arg -eq "--dry-run") {
        $dryRun = $true
    } elseif ($arg -eq "--force") {
        $force = $true
    }
}

# Build command
$cmd = "python scripts/reindex_existing_tenders.py --batch-size $batchSize"
if ($maxTenders) {
    $cmd += " --max-tenders $maxTenders"
}
if ($dryRun) {
    $cmd += " --dry-run"
}
if ($force) {
    $cmd += " --force"
}

Write-Host "Command: $cmd" -ForegroundColor Gray
Write-Host ""

# Run the reindexing script
Invoke-Expression $cmd

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Reindexing failed with errors" -ForegroundColor Red
    exit 1
} else {
    Write-Host ""
    Write-Host "SUCCESS: Reindexing completed successfully" -ForegroundColor Green
    exit 0
}


