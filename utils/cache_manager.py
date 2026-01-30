"""
Centralized cache management with event-driven invalidation
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from utils.logging_config import get_logger

logger = get_logger(__name__, "app")

# Cache storage
_TENDERS_CACHE: Dict[str, Dict[str, Any]] = {}
_MATCHED_CACHE: Dict[str, Dict[str, Any]] = {}
_PURCHASED_CACHE: Dict[str, Dict[str, Any]] = {}
_KEYWORDS_CACHE: Dict[str, Dict[str, Any]] = {}
_MATCHING_STATUS: Dict[str, str] = {}  # client_id -> "idle" | "matching"

# Cache TTL
CACHE_TTL_SECONDS = 1800  # 30 minutes

# Track which clients need cache invalidation
_invalidation_queue: Set[str] = set()


def get_cached_tenders(client_id: str) -> Optional[List[dict]]:
    """Get cached all tenders for a client"""
    entry = _TENDERS_CACHE.get(client_id)
    if not entry:
        return None
    ts = entry.get("timestamp")
    if not ts:
        return None
    if (datetime.now(timezone.utc) - ts).total_seconds() > CACHE_TTL_SECONDS:
        return None
    data = entry.get("data")
    if isinstance(data, list):
        return data
    return None


def set_cached_tenders(client_id: str, data: List[dict]) -> None:
    """Set cached all tenders for a client"""
    _TENDERS_CACHE[client_id] = {
        "timestamp": datetime.now(timezone.utc),
        "data": data,
    }


def get_cached_matched(client_id: str) -> Optional[List[dict]]:
    """Get cached matched tenders for a client"""
    entry = _MATCHED_CACHE.get(client_id)
    if not entry:
        return None
    ts = entry.get("timestamp")
    if not ts:
        return None
    if (datetime.now(timezone.utc) - ts).total_seconds() > CACHE_TTL_SECONDS:
        return None
    data = entry.get("data")
    if isinstance(data, list):
        return data
    return None


def set_cached_matched(client_id: str, data: List[dict]) -> None:
    """Set cached matched tenders for a client"""
    _MATCHED_CACHE[client_id] = {
        "timestamp": datetime.now(timezone.utc),
        "data": data,
    }


def get_cached_purchased(client_id: str) -> Optional[List[dict]]:
    """Get cached purchased tenders for a client"""
    entry = _PURCHASED_CACHE.get(client_id)
    if not entry:
        return None
    ts = entry.get("timestamp")
    if not ts:
        return None
    if (datetime.now(timezone.utc) - ts).total_seconds() > CACHE_TTL_SECONDS:
        return None
    data = entry.get("data")
    if isinstance(data, list):
        return data
    return None


def set_cached_purchased(client_id: str, data: List[dict]) -> None:
    """Set cached purchased tenders for a client"""
    _PURCHASED_CACHE[client_id] = {
        "timestamp": datetime.now(timezone.utc),
        "data": data,
    }


def get_cached_keywords(client_id: str) -> Optional[List[dict]]:
    """Get cached keywords for a client"""
    entry = _KEYWORDS_CACHE.get(client_id)
    if not entry:
        return None
    ts = entry.get("timestamp")
    if not ts:
        return None
    if (datetime.now(timezone.utc) - ts).total_seconds() > CACHE_TTL_SECONDS:
        return None
    data = entry.get("data")
    if isinstance(data, list):
        return data
    return None


def set_cached_keywords(client_id: str, data: List[dict]) -> None:
    """Set cached keywords for a client"""
    _KEYWORDS_CACHE[client_id] = {
        "timestamp": datetime.now(timezone.utc),
        "data": data,
    }


def invalidate_client_caches(client_id: str) -> None:
    """Invalidate all caches for a specific client"""
    logger.info(f"Invalidating all caches for client {client_id[:8]}...")
    
    if client_id in _TENDERS_CACHE:
        del _TENDERS_CACHE[client_id]
        logger.debug(f"Cleared tenders cache for client {client_id[:8]}...")
    
    if client_id in _MATCHED_CACHE:
        del _MATCHED_CACHE[client_id]
        logger.debug(f"Cleared matched cache for client {client_id[:8]}...")
    
    if client_id in _PURCHASED_CACHE:
        del _PURCHASED_CACHE[client_id]
        logger.debug(f"Cleared purchased cache for client {client_id[:8]}...")
    
    if client_id in _KEYWORDS_CACHE:
        del _KEYWORDS_CACHE[client_id]
        logger.debug(f"Cleared keywords cache for client {client_id[:8]}...")


def invalidate_all_tenders_cache() -> None:
    """Invalidate all tenders cache for ALL clients (used when new tenders are ingested)"""
    logger.info(f"Invalidating all tenders cache for all clients (count: {len(_TENDERS_CACHE)})")
    _TENDERS_CACHE.clear()


def invalidate_all_matched_cache() -> None:
    """Invalidate matched tenders cache for ALL clients (used when new tenders are ingested)"""
    logger.info(f"Invalidating all matched cache for all clients (count: {len(_MATCHED_CACHE)})")
    _MATCHED_CACHE.clear()


def invalidate_matched_cache(client_id: str) -> None:
    """Invalidate only matched tenders cache for a specific client"""
    if client_id in _MATCHED_CACHE:
        del _MATCHED_CACHE[client_id]
        logger.debug(f"Cleared matched cache for client {client_id[:8]}...")


def invalidate_keywords_cache(client_id: str) -> None:
    """Invalidate only keywords cache for a specific client"""
    if client_id in _KEYWORDS_CACHE:
        del _KEYWORDS_CACHE[client_id]
        logger.debug(f"Cleared keywords cache for client {client_id[:8]}...")


def invalidate_purchased_cache(client_id: str) -> None:
    """Invalidate only purchased tenders cache for a specific client"""
    if client_id in _PURCHASED_CACHE:
        del _PURCHASED_CACHE[client_id]
        logger.debug(f"Cleared purchased cache for client {client_id[:8]}...")


def mark_client_for_invalidation(client_id: str) -> None:
    """Mark a client for cache invalidation (used in async contexts)"""
    _invalidation_queue.add(client_id)
    logger.debug(f"Marked client {client_id[:8]}... for cache invalidation")


def process_invalidation_queue() -> List[str]:
    """Process and clear the invalidation queue, returning the list of invalidated clients"""
    if not _invalidation_queue:
        return []
    
    clients = list(_invalidation_queue)
    _invalidation_queue.clear()
    
    for client_id in clients:
        invalidate_client_caches(client_id)
    
    logger.info(f"Processed cache invalidation for {len(clients)} clients")
    return clients


def get_cache_stats() -> Dict[str, int]:
    """Get current cache statistics"""
    return {
        "tenders_cached_clients": len(_TENDERS_CACHE),
        "matched_cached_clients": len(_MATCHED_CACHE),
        "purchased_cached_clients": len(_PURCHASED_CACHE),
        "keywords_cached_clients": len(_KEYWORDS_CACHE),
        "pending_invalidations": len(_invalidation_queue),
    }


def get_matching_status(client_id: str) -> str:
    """Get the matching status for a client"""
    return _MATCHING_STATUS.get(client_id, "idle")


def set_matching_status(client_id: str, status: str) -> None:
    """Set the matching status for a client"""
    _MATCHING_STATUS[client_id] = status

