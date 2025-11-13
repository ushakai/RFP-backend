# RFP Backend Logs

This directory contains application logs organized by date and type.

## Log Files

### Application Logs (`app_YYYYMMDD.log`)
- Main application logs
- API requests and responses
- General application flow
- Level: INFO and above

### Error Logs (`error_YYYYMMDD.log`)
- Error-level logs only
- Exceptions and stack traces
- Critical issues
- Level: ERROR and above

### Worker Logs (`worker_YYYYMMDD.log`)
- Background job processing
- Job status updates
- RFP processing and QA extraction
- Level: DEBUG and above

### Tender Logs (`tender_YYYYMMDD.log`)
- Tender ingestion cycles
- API calls to tender sources
- Matching and notifications
- Level: DEBUG and above

## Log Rotation

Logs are automatically rotated daily. Each log file includes the date in its filename (YYYYMMDD format).

## Log Levels

- **DEBUG**: Detailed information for debugging
- **INFO**: General informational messages
- **WARNING**: Warning messages for potential issues
- **ERROR**: Error messages with exceptions
- **CRITICAL**: Critical errors that may cause shutdown

## Viewing Logs

### Real-time monitoring (PowerShell)
```powershell
# Watch application logs
Get-Content logs\app_*.log -Wait -Tail 50

# Watch error logs
Get-Content logs\error_*.log -Wait -Tail 50

# Watch worker logs
Get-Content logs\worker_*.log -Wait -Tail 50

# Watch tender logs
Get-Content logs\tender_*.log -Wait -Tail 50
```

### Search logs (PowerShell)
```powershell
# Search for specific term
Select-String -Path logs\*.log -Pattern "ERROR"

# Search with context (5 lines before/after)
Select-String -Path logs\*.log -Pattern "exception" -Context 5
```

## Cleanup

Logs older than 30 days should be archived or deleted to save disk space.

```powershell
# Delete logs older than 30 days
Get-ChildItem logs\*.log | Where-Object {$_.LastWriteTime -lt (Get-Date).AddDays(-30)} | Remove-Item
```

