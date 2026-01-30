"""
Cleanup script to remove orphaned tender_matches that reference tenders
that have been moved to expired_tenders table.

Usage:
    python scripts/cleanup_orphaned_matches.py
"""
import sys
import os

# Add parent directory to path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_supabase_client
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the script."""
    logger.info("=" * 80)
    logger.info("Orphaned Tender Matches Cleanup Script")
    logger.info("=" * 80)
    
    try:
        supabase = get_supabase_client()
        
        # Get all match tender IDs
        logger.info("Fetching all tender matches...")
        all_matches = []
        page_size = 1000
        offset = 0
        
        while True:
            result = supabase.table("tender_matches").select("id, tender_id").range(offset, offset + page_size - 1).execute()
            matches = result.data or []
            if not matches:
                break
            all_matches.extend(matches)
            offset += page_size
            if len(matches) < page_size:
                break
        
        logger.info(f"Found {len(all_matches)} total matches")
        
        if not all_matches:
            logger.info("No matches to check. Exiting.")
            return
        
        # Get unique tender IDs
        tender_ids = list(set(m.get("tender_id") for m in all_matches if m.get("tender_id")))
        logger.info(f"Checking {len(tender_ids)} unique tender IDs...")
        
        # Check which tenders exist
        existing_tenders = set()
        batch_size = 100
        for i in range(0, len(tender_ids), batch_size):
            batch = tender_ids[i:i + batch_size]
            result = supabase.table("tenders").select("id").in_("id", batch).execute()
            existing_tenders.update(t["id"] for t in (result.data or []))
        
        # Find orphaned matches (tenders that don't exist in active tenders table)
        orphaned_matches = [
            m for m in all_matches
            if m.get("tender_id") and m.get("tender_id") not in existing_tenders
        ]
        
        logger.info(f"Found {len(orphaned_matches)} orphaned matches")
        
        if not orphaned_matches:
            logger.info("No orphaned matches found. All matches are valid.")
            return
        
        # Delete orphaned matches
        logger.info("Deleting orphaned matches...")
        deleted_count = 0
        batch_size = 100
        for i in range(0, len(orphaned_matches), batch_size):
            batch = orphaned_matches[i:i + batch_size]
            match_ids = [m["id"] for m in batch if m.get("id")]
            
            for match_id in match_ids:
                try:
                    supabase.table("tender_matches").delete().eq("id", match_id).execute()
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Error deleting match {match_id}: {e}")
        
        logger.info("=" * 80)
        logger.info(f"Successfully deleted {deleted_count} orphaned matches")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.info("\n\nScript interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

