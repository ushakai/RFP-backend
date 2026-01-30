# PowerShell script to run expired tenders cleanup
# Run from RFP-backend directory

Write-Host "Moving expired tenders..." -ForegroundColor Cyan
cd $PSScriptRoot\..
python scripts/move_expired_tenders.py

