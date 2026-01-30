"""
Standalone script to manually move expired tenders to expired_tenders table.
Can be run from the RFP-backend directory.

Usage:
    python scripts/move_expired_tenders.py
"""
import sys
import os

# Add parent directory to path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.expired_tenders_service import move_expired_tenders, count_expired_tenders
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
    logger.info("Expired Tenders Cleanup Script")
    logger.info("=" * 80)
    
    try:
        # Count expired tenders before moving
        logger.info("Counting expired tenders...")
        count_before = count_expired_tenders(use_direct_db=True)
        logger.info(f"Found {count_before} expired tenders to move")
        
        if count_before == 0:
            logger.info("No expired tenders to move. Exiting.")
            return
        
        # Move expired tenders
        logger.info("Moving expired tenders to expired_tenders table...")
        result = move_expired_tenders(use_direct_db=True)
        
        if result.get("error"):
            logger.error(f"Error moving expired tenders: {result['error']}")
            sys.exit(1)
        
        moved_count = result.get("moved_count", 0)
        logger.info("=" * 80)
        logger.info(f"Successfully moved {moved_count} expired tenders to expired_tenders table")
        logger.info("=" * 80)
        
        # Verify by counting again
        count_after = count_expired_tenders(use_direct_db=True)
        logger.info(f"Remaining expired tenders in active table: {count_after}")
        
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

