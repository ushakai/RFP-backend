"""
Script to extract and store search data for all existing tenders in the database.

This script:
1. Fetches all tenders that don't have indexed search data yet
2. Extracts keywords, locations, and industries from each tender
3. Updates the tender records with the extracted data
4. Updates the search terms lookup table with counts

Usage:
    python scripts/reindex_existing_tenders.py
    
    Or with options:
    python scripts/reindex_existing_tenders.py --batch-size 100 --max-tenders 10000 --dry-run
"""

import os
import sys
import argparse
import time
from datetime import datetime, timezone
from typing import Dict, Any, List

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_supabase_client
from services.tender_search_service import (
    extract_tender_search_data,
    _update_search_terms_counts,
    logger
)


def reindex_tenders(
    batch_size: int = 100,
    max_tenders: int = None,
    dry_run: bool = False,
    skip_existing: bool = True,
    skip_term_rebuild: bool = False
) -> Dict[str, int]:
    """
    Re-index all existing tenders with search data.
    
    Args:
        batch_size: Number of tenders to process in each batch
        max_tenders: Maximum number of tenders to process (None = all)
        dry_run: If True, don't actually update the database
        skip_existing: If True, skip tenders that already have indexed data
    
    Returns:
        Dictionary with success_count, error_count, skipped_count
    """
    logger.info("=" * 80)
    logger.info("Starting tender reindexing process")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Max tenders: {max_tenders or 'ALL'}")
    logger.info(f"Dry run: {dry_run}")
    logger.info(f"Skip existing: {skip_existing}")
    logger.info("=" * 80)
    
    supabase = get_supabase_client()
    success_count = 0
    error_count = 0
    skipped_count = 0
    offset = 0
    start_time = time.time()
    
            # Build query to fetch tenders
    # Note: We don't fetch full_data to avoid loading large JSONB fields that slow down processing
    # The extraction functions can work without full_data (it's optional)
    query = supabase.table("tenders").select(
        "id, title, description, summary, location, category, sector, metadata, "
        "indexed_keywords, indexed_locations, indexed_industries"
    )
    
    # Skip tenders that already have indexed data if requested
    if skip_existing:
        query = query.or_("indexed_keywords.is.null,indexed_locations.is.null,indexed_industries.is.null")
    
    # Order by created_at to process oldest first
    query = query.order("created_at", desc=False)
    
    total_processed = 0
    
    while True:
        try:
            # Fetch batch
            fetch_start = time.time()
            logger.info(f"Fetching batch at offset {offset}...")
            result = query.limit(batch_size).offset(offset).execute()
            tenders = result.data or []
            fetch_time = time.time() - fetch_start
            logger.info(f"Fetched {len(tenders)} tenders in {fetch_time:.2f}s")
            
            if not tenders:
                logger.info(f"No more tenders to process at offset {offset}")
                break
            
            logger.info(f"\nProcessing batch: {offset} to {offset + len(tenders) - 1} ({len(tenders)} tenders)")
            
            batch_start = time.time()
            batch_success = 0
            batch_errors = 0
            batch_skipped = 0
            
            for idx, tender in enumerate(tenders):
                tender_id = tender.get("id")
                
                # Log progress every 10 tenders in batch
                if (idx + 1) % 10 == 0:
                    logger.info(f"  Processing tender {idx + 1}/{len(tenders)} in batch...")
                
                # Skip if already indexed (double-check)
                if skip_existing:
                    has_indexed = (
                        tender.get("indexed_keywords") or 
                        tender.get("indexed_locations") or 
                        tender.get("indexed_industries")
                    )
                    if has_indexed:
                        batch_skipped += 1
                        skipped_count += 1
                        continue
                
                try:
                    # Extract search data (optimized - doesn't process full_data)
                    extraction_start = time.time()
                    try:
                        search_data = extract_tender_search_data(tender)
                    except Exception as extract_error:
                        logger.warning(f"  Extraction error for {tender_id[:8]}...: {extract_error}")
                        # Create minimal search data to avoid failing completely
                        search_data = {
                            "indexed_industries": None,
                            "indexed_locations": None,
                            "indexed_keywords": None,
                            "search_text": (tender.get("title") or "")[:1000] if tender.get("title") else None,
                        }
                    
                    extraction_time = time.time() - extraction_start
                    
                    # Log if extraction took more than 0.5 seconds
                    if extraction_time > 0.5:
                        logger.debug(f"  Tender {tender_id[:8]}... extraction took {extraction_time:.2f}s")
                    
                    if dry_run:
                        # Just log what would be updated
                        logger.debug(
                            f"[DRY RUN] Tender {tender_id[:8]}... "
                            f"Keywords: {len(search_data.get('indexed_keywords') or [])}, "
                            f"Locations: {len(search_data.get('indexed_locations') or [])}, "
                            f"Industries: {len(search_data.get('indexed_industries') or [])}"
                        )
                        batch_success += 1
                        success_count += 1
                    else:
                        # Update tender with search data
                        update_start = time.time()
                        try:
                            update_result = supabase.table("tenders").update(search_data).eq("id", tender_id).execute()
                            update_time = time.time() - update_start
                            
                            if update_result.data:
                                batch_success += 1
                                success_count += 1
                                
                                # Store search data for batch update of search terms (more efficient)
                                # We'll update search terms in batches at the end
                                
                                # Log if update took more than 2 seconds
                                if update_time > 2.0:
                                    logger.debug(f"  Tender {tender_id[:8]}... update took {update_time:.2f}s")
                            else:
                                batch_errors += 1
                                error_count += 1
                                logger.warning(f"Failed to update tender {tender_id}: No data returned")
                        except Exception as update_error:
                            batch_errors += 1
                            error_count += 1
                            logger.error(f"Error updating tender {tender_id}: {update_error}")
                            # Continue with next tender
                            continue
                    
                except Exception as e:
                    batch_errors += 1
                    error_count += 1
                    logger.error(f"Error processing tender {tender_id}: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    # Continue with next tender
                    continue
                
                total_processed += 1
                
                # Check max_tenders limit
                if max_tenders and total_processed >= max_tenders:
                    logger.info(f"Reached max_tenders limit ({max_tenders})")
                    break
            
            batch_time = time.time() - batch_start
            logger.info(
                f"Batch complete: {batch_success} success, {batch_errors} errors, "
                f"{batch_skipped} skipped in {batch_time:.1f}s"
            )
            
            # Update search terms in batches (every 10 batches or at the end) for efficiency
            # This avoids updating search terms for every single tender
            # Skip during processing to avoid timeouts - we'll do it at the end
            # if not dry_run and batch_success > 0 and (offset // batch_size) % 10 == 0:
            #     try:
            #         logger.info("Updating search term counts...")
            #         _rebuild_search_term_counts(supabase)
            #         logger.info("✓ Search term counts updated")
            #     except Exception as term_error:
            #         logger.warning(f"Could not update search terms: {term_error}")
            #         # Non-critical - continue
            
            # Check if we've reached the limit
            if max_tenders and total_processed >= max_tenders:
                break
            
            # Check if we got fewer results than batch_size (end of data)
            if len(tenders) < batch_size:
                logger.info("Reached end of tender list")
                break
            
            offset += batch_size
            
            # Progress update every 10 batches
            if (offset // batch_size) % 10 == 0:
                elapsed = time.time() - start_time
                rate = total_processed / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Progress: {total_processed} processed, {success_count} success, "
                    f"{error_count} errors, {skipped_count} skipped "
                    f"({rate:.1f} tenders/sec)"
                )
            
            # Small delay to avoid overwhelming the database
            # Only delay if batch took less than 5 seconds (fast batches need throttling)
            if not dry_run and batch_time < 5.0:
                time.sleep(0.2)
        
        except Exception as e:
            logger.error(f"Error fetching batch at offset {offset}: {e}")
            error_count += len(tenders) if 'tenders' in locals() else 0
            break
    
    # Rebuild search term counts after all updates (final update)
    # Note: This can be slow for large databases - you can skip it and run separately
    if not dry_run and success_count > 0 and not skip_term_rebuild:
        logger.info("\nRebuilding search term counts (final update)...")
        logger.info("Note: This may take several minutes for large databases.")
        try:
            _rebuild_search_term_counts(supabase)
            logger.info("✓ Search term counts rebuilt")
        except Exception as e:
            logger.warning(f"Error rebuilding search term counts: {e}")
            logger.warning("You can rebuild search terms later by running the script with --rebuild-terms-only")
            # Non-critical - log but don't fail
    elif skip_term_rebuild:
        logger.info("\nSkipping search term count rebuild (use --rebuild-terms-only to run separately)")
    
    elapsed = time.time() - start_time
    
    logger.info("=" * 80)
    logger.info("Reindexing complete!")
    logger.info(f"Total processed: {total_processed}")
    logger.info(f"Success: {success_count}")
    logger.info(f"Errors: {error_count}")
    logger.info(f"Skipped: {skipped_count}")
    logger.info(f"Time elapsed: {elapsed:.1f}s")
    if total_processed > 0:
        logger.info(f"Average rate: {total_processed / elapsed:.1f} tenders/sec")
    logger.info("=" * 80)
    
    return {
        "success_count": success_count,
        "error_count": error_count,
        "skipped_count": skipped_count,
        "total_processed": total_processed,
        "elapsed_time": elapsed
    }


def _rebuild_search_term_counts(supabase):
    """Rebuild the tender_search_terms counts from indexed tenders."""
    from services.tender_search_service import CPV_SECTIONS
    
    logger.info("Rebuilding search term counts (this may take a while)...")
    
    try:
        # Get counts for each term type
        for term_type, column in [
            ("industry", "indexed_industries"),
            ("location", "indexed_locations"),
            ("keyword", "indexed_keywords"),
        ]:
            try:
                logger.info(f"  Processing {term_type} terms...")
                
                # Process in batches to avoid timeouts
                term_counts: Dict[str, int] = {}
                batch_offset = 0
                batch_size = 500  # Smaller batches
                max_batches = 50  # Limit total batches to avoid very long operations
                batch_count = 0
                
                while batch_count < max_batches:
                    try:
                        # Fetch batch
                        result = supabase.table("tenders").select(column).not_.is_(column, "null").limit(batch_size).offset(batch_offset).execute()
                        rows = result.data or []
                        
                        if not rows:
                            break
                        
                        # Count occurrences in this batch
                        for row in rows:
                            values = row.get(column) or []
                            if isinstance(values, list):
                                for val in values:
                                    if val:
                                        term_counts[val] = term_counts.get(val, 0) + 1
                        
                        batch_count += 1
                        batch_offset += batch_size
                        
                        if len(rows) < batch_size:
                            break
                            
                        # Log progress every 10 batches
                        if batch_count % 10 == 0:
                            logger.info(f"    Processed {batch_count} batches, found {len(term_counts)} unique {term_type} terms...")
                            
                    except Exception as batch_error:
                        logger.warning(f"    Error fetching batch {batch_count} for {term_type}: {batch_error}")
                        break
                
                # Upsert terms in very small batches (10 at a time) to avoid statement timeouts
                if term_counts:
                    terms_to_upsert = []
                    for term_value, count in term_counts.items():
                        display = term_value
                        if term_type == "industry" and term_value in CPV_SECTIONS:
                            display = f"{term_value} - {CPV_SECTIONS[term_value]}"
                        else:
                            display = term_value.replace("_", " ").title()
                        
                        terms_to_upsert.append({
                            "term_type": term_type,
                            "term_value": term_value,
                            "term_display": display,
                            "tender_count": count,
                        })
                    
                    # Upsert in very small batches (10 at a time) to avoid timeouts
                    upsert_batch_size = 10
                    upserted = 0
                    total_batches = (len(terms_to_upsert) + upsert_batch_size - 1) // upsert_batch_size
                    
                    for i in range(0, len(terms_to_upsert), upsert_batch_size):
                        batch = terms_to_upsert[i:i + upsert_batch_size]
                        batch_num = i // upsert_batch_size + 1
                        
                        try:
                            supabase.table("tender_search_terms").upsert(
                                batch,
                                on_conflict="term_type,term_value"
                            ).execute()
                            upserted += len(batch)
                            
                            # Log progress every 10 batches
                            if batch_num % 10 == 0:
                                logger.info(f"    Upserted {batch_num}/{total_batches} batches ({upserted}/{len(terms_to_upsert)} terms)...")
                                
                        except Exception as batch_error:
                            logger.warning(f"    Error upserting {term_type} batch {batch_num}: {batch_error}")
                            # Try individual inserts for this batch as fallback
                            for item in batch:
                                try:
                                    supabase.table("tender_search_terms").upsert(
                                        item,
                                        on_conflict="term_type,term_value"
                                    ).execute()
                                    upserted += 1
                                except Exception:
                                    pass  # Skip failed items
                    
                    logger.info(f"  ✓ Updated {upserted}/{len(term_counts)} {term_type} term counts")
                else:
                    logger.info(f"  No {term_type} terms found to update")
                
            except Exception as e:
                logger.warning(f"Error rebuilding {term_type} counts: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                
    except Exception as e:
        logger.error(f"Error rebuilding search term counts: {e}")
        import traceback
        logger.debug(traceback.format_exc())


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Re-index existing tenders with search data (keywords, locations, industries)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of tenders to process in each batch (default: 100)"
    )
    parser.add_argument(
        "--max-tenders",
        type=int,
        default=None,
        help="Maximum number of tenders to process (default: all)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually update the database, just show what would be done"
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-index all tenders, even those that already have indexed data"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-indexing even if data exists (same as --no-skip-existing)"
    )
    parser.add_argument(
        "--skip-term-rebuild",
        action="store_true",
        help="Skip rebuilding search term counts at the end (faster, but counts won't be updated)"
    )
    parser.add_argument(
        "--rebuild-terms-only",
        action="store_true",
        help="Only rebuild search term counts, don't re-index tenders"
    )
    
    args = parser.parse_args()
    
    # Handle force flag
    skip_existing = not (args.no_skip_existing or args.force)
    
    # Handle rebuild terms only
    if args.rebuild_terms_only:
        try:
            from config.settings import get_supabase_client
            supabase = get_supabase_client()
            logger.info("Rebuilding search term counts only...")
            _rebuild_search_term_counts(supabase)
            logger.info("✓ Search term counts rebuilt successfully")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error rebuilding search terms: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    try:
        results = reindex_tenders(
            batch_size=args.batch_size,
            max_tenders=args.max_tenders,
            dry_run=args.dry_run,
            skip_existing=skip_existing,
            skip_term_rebuild=args.skip_term_rebuild
        )
        
        # Exit with error code if there were errors
        if results["error_count"] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
    
    except KeyboardInterrupt:
        logger.info("\n\nReindexing interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

