"""
Service for managing expired tenders - moving them to expired_tenders table
"""
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from config.settings import get_supabase_client
from config.db import _db_config

logger = logging.getLogger(__name__)


def move_expired_tenders(use_direct_db: bool = True) -> Dict[str, Any]:
    """
    Move tenders with passed deadlines to expired_tenders table.
    
    Args:
        use_direct_db: If True, use direct PostgreSQL connection. If False, use Supabase RPC.
    
    Returns:
        Dict with 'moved_count' and 'error' keys
    """
    try:
        if use_direct_db:
            # Use direct PostgreSQL connection for better performance
            try:
                # Ensure database is initialized
                _db_config.initialize()
                conn = _db_config.get_connection()
                try:
                    cursor = conn.cursor()
                    # Call the PostgreSQL function
                    cursor.execute("SELECT * FROM public.move_expired_tenders()")
                    result = cursor.fetchone()
                    moved_count = result[0] if result else 0
                    conn.commit()
                    cursor.close()
                    
                    logger.info(f"Moved {moved_count} expired tenders to expired_tenders table")
                    return {"moved_count": moved_count, "error": None}
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Error moving expired tenders via direct DB: {e}")
                    return {"moved_count": 0, "error": str(e)}
                finally:
                    _db_config.release_connection(conn)
            except Exception as e:
                logger.warning(f"Direct DB connection failed, falling back to RPC: {e}")
                # Fall through to RPC method
        
        # Fallback to Supabase RPC
        supabase = get_supabase_client()
        result = supabase.rpc("move_expired_tenders").execute()
        
        if result.data:
            moved_count = result.data[0].get("moved_count", 0) if isinstance(result.data, list) else result.data.get("moved_count", 0)
        else:
            moved_count = 0
        
        logger.info(f"Moved {moved_count} expired tenders to expired_tenders table")
        return {"moved_count": moved_count, "error": None}
        
    except Exception as e:
        logger.error(f"Error moving expired tenders: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return {"moved_count": 0, "error": str(e)}


def count_expired_tenders(use_direct_db: bool = True) -> int:
    """
    Count how many tenders have expired deadlines.
    
    Args:
        use_direct_db: If True, use direct PostgreSQL connection. If False, use Supabase RPC.
    
    Returns:
        Number of expired tenders
    """
    try:
        if use_direct_db:
            try:
                # Ensure database is initialized
                _db_config.initialize()
                conn = _db_config.get_connection()
                try:
                    cursor = conn.cursor()
                    cursor.execute("SELECT public.count_expired_tenders()")
                    result = cursor.fetchone()
                    count = result[0] if result else 0
                    cursor.close()
                    return count
                except Exception as e:
                    logger.warning(f"Error counting expired tenders via direct DB: {e}")
                finally:
                    _db_config.release_connection(conn)
            except Exception as e:
                logger.warning(f"Direct DB connection failed, falling back to RPC: {e}")
                # Fall through to RPC method
        
        # Fallback to Supabase RPC
        supabase = get_supabase_client()
        result = supabase.rpc("count_expired_tenders").execute()
        
        if result.data:
            return result.data if isinstance(result.data, int) else 0
        return 0
        
    except Exception as e:
        logger.error(f"Error counting expired tenders: {e}")
        return 0


def get_expired_tenders(
    limit: int = 100,
    offset: int = 0,
    source: Optional[str] = None
) -> list[Dict[str, Any]]:
    """
    Retrieve expired tenders (for admin/history purposes).
    
    Args:
        limit: Maximum number of tenders to return
        offset: Offset for pagination
        source: Optional source filter
    
    Returns:
        List of expired tender records
    """
    try:
        supabase = get_supabase_client()
        query = supabase.table("expired_tenders").select("*").order("expired_at", desc=True).limit(limit).offset(offset)
        
        if source:
            query = query.eq("source", source)
        
        result = query.execute()
        return result.data or []
        
    except Exception as e:
        logger.error(f"Error fetching expired tenders: {e}")
        return []

