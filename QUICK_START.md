# 🚀 Quick Start Guide

## Start Development Environment

```bash
# Terminal 1: Start API server
cd RFP-backend
uvicorn app:app --reload --port 8000

# Terminal 2: Start background worker (optional)
cd RFP-backend
python worker.py

# Terminal 3: Monitor logs (optional)
cd RFP-backend
Get-Content logs\app_*.log -Wait -Tail 50
```

## Verify Everything Works

```bash
# Test imports
python -c "import app; import worker; import tender_ingestion; print('✓ All working!')"

# Check health
curl http://localhost:8000/health

# View API docs
# Open browser: http://localhost:8000/docs
```

## View Logs

```powershell
# Application logs
Get-Content logs\app_*.log -Tail 50

# Error logs only
Get-Content logs\error_*.log -Tail 50

# Worker logs
Get-Content logs\worker_*.log -Tail 50

# Tender ingestion logs
Get-Content logs\tender_*.log -Tail 50

# Search for errors
Select-String -Path logs\*.log -Pattern "ERROR"
```

## Environment Variables

Create `.env` file (if not exists):

```bash
# Required
GOOGLE_API_KEY=your_gemini_api_key
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=your_supabase_key

# Optional
GEMINI_MODEL=gemini-2.5-flash
FRONTEND_ORIGIN=http://localhost:5173
TENDER_INGESTION_INTERVAL_MINUTES=360
DISABLE_TENDER_INGESTION_LOOP=0
```

## Common Commands

```bash
# Check if app loads
python -c "import app"

# Run tender ingestion manually
python tender_ingestion.py

# Clean old logs (30+ days)
Get-ChildItem logs\*.log | Where-Object {$_.LastWriteTime -lt (Get-Date).AddDays(-30)} | Remove-Item
```

## Troubleshooting

**Problem**: Import errors  
**Solution**: Ensure you're in `RFP-backend` directory

**Problem**: Worker not running  
**Solution**: Check `logs/worker_*.log` for errors

**Problem**: No logs appearing  
**Solution**: `logs/` folder auto-created on first run

**Problem**: Tender ingestion not working  
**Solution**: Check `DISABLE_TENDER_INGESTION_LOOP` not set to 1

## Documentation

- 📖 **Full Documentation**: `MODULAR_STRUCTURE.md`
- ✅ **Refactoring Summary**: `REFACTORING_COMPLETE.md`
- 📝 **Log Guide**: `logs/README.md`
- 🌐 **API Docs**: http://localhost:8000/docs

## Status

✅ All modules working  
✅ Logging configured  
✅ Worker updated  
✅ Tender ingestion fixed  
✅ Zero breaking changes  
✅ **Production ready!**

