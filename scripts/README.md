# Tender Reindexing Scripts

This directory contains scripts to extract and store search data (keywords, locations, industries) for existing tenders in the database.

## Quick Start

### Windows (PowerShell)
```powershell
.\scripts\reindex_tenders.ps1
```

### Windows (Command Prompt)
```cmd
scripts\reindex_tenders.bat
```

### Python (Direct)
```bash
python scripts/reindex_existing_tenders.py
```

## Script Options

The `reindex_existing_tenders.py` script supports several options:

```bash
python scripts/reindex_existing_tenders.py [OPTIONS]

Options:
  --batch-size N        Number of tenders to process per batch (default: 100)
  --max-tenders N       Maximum number of tenders to process (default: all)
  --dry-run             Don't update database, just show what would be done
  --no-skip-existing    Re-index all tenders, even those already indexed
  --force               Force re-indexing (same as --no-skip-existing)
```

## Examples

### Process all tenders (default)
```bash
python scripts/reindex_existing_tenders.py
```

### Process first 1000 tenders only
```bash
python scripts/reindex_existing_tenders.py --max-tenders 1000
```

### Test run (see what would be done without updating)
```bash
python scripts/reindex_existing_tenders.py --dry-run
```

### Process with larger batches (faster but more memory)
```bash
python scripts/reindex_existing_tenders.py --batch-size 500
```

### Force re-index all tenders (even if already indexed)
```bash
python scripts/reindex_existing_tenders.py --force
```

## What the Script Does

1. **Fetches tenders** that don't have indexed search data (or all if `--force` is used)
2. **Extracts search data** from each tender:
   - **Keywords**: From title, description, and summary
   - **Locations**: From location fields, metadata, and delivery addresses
   - **Industries**: CPV codes from category and sector fields
   - **Search text**: Combined text for full-text search
3. **Updates tender records** with the extracted data
4. **Updates search terms lookup table** with counts for dropdown suggestions
5. **Rebuilds search term counts** after all updates are complete

## Performance

- **Batch processing**: Processes tenders in batches to avoid memory issues
- **Progress logging**: Shows progress every 10 batches
- **Error handling**: Continues processing even if individual tenders fail
- **Rate**: Typically processes 10-50 tenders per second (depends on database performance)

## Output

The script provides detailed logging:
- Progress updates every 10 batches
- Success/error counts per batch
- Final summary with total counts and processing time

Example output:
```
================================================================================
Starting tender reindexing process
Batch size: 100
Max tenders: ALL
Dry run: False
Skip existing: True
================================================================================

Processing batch: 0 to 99 (100 tenders)
Batch complete: 100 success, 0 errors, 0 skipped in 12.3s

Progress: 1000 processed, 995 success, 5 errors, 0 skipped (45.2 tenders/sec)

================================================================================
Reindexing complete!
Total processed: 5000
Success: 4985
Errors: 15
Skipped: 0
Time elapsed: 110.5s
Average rate: 45.2 tenders/sec
================================================================================
```

## Troubleshooting

### "No module named 'config'"
Make sure you're running the script from the `RFP-backend` directory:
```bash
cd RFP-backend
python scripts/reindex_existing_tenders.py
```

### Database connection errors
- Check your `.env` file has correct database credentials
- Ensure your database is accessible
- Check if you're using direct connection or REST API

### Memory issues with large batches
- Reduce `--batch-size` (e.g., `--batch-size 50`)
- Process in smaller chunks using `--max-tenders`

### Slow processing
- Increase `--batch-size` for faster processing (if memory allows)
- Check database connection (direct connection is faster than REST API)
- Ensure database indexes are created (run the migration first)

## When to Run

Run this script:
- **After migration**: After running the `2026-01-29_tender_search_indexes.sql` migration
- **After bulk import**: After importing a large number of tenders
- **Periodically**: To ensure all tenders have indexed data
- **After schema changes**: If search extraction logic is updated

## Notes

- The script automatically skips tenders that already have indexed data (unless `--force` is used)
- Search term counts are rebuilt at the end for accurate dropdown suggestions
- The script is safe to run multiple times (idempotent)
- Use `--dry-run` first to see what would be processed


