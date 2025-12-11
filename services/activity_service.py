"""Activity, session, and analytics helpers for admin surfaces."""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from io import StringIO
from typing import Any, Dict, Iterable, List, Optional

from config.settings import (
    ADMIN_ACTIVITY_RETENTION_DAYS,
    ADMIN_ANALYTICS_CACHE_SECONDS,
    ADMIN_SESSION_MAX_AGE_DAYS,
    get_supabase_client,
)
from utils.logging_config import get_logger


logger = get_logger(__name__, "app")

EVENT_TYPES = {"auth", "bid", "file", "system"}


def record_event(
    event_type: str,
    action: str,
    actor_client_id: str | None = None,
    actor_email: str | None = None,
    subject_id: str | None = None,
    subject_type: str | None = None,
    metadata: Dict[str, Any] | None = None,
) -> None:
    """Persist a single activity event."""
    normalized_type = (event_type or "").strip().lower()
    if normalized_type not in EVENT_TYPES:
        normalized_type = "system"

    supabase = get_supabase_client()
    payload = {
        "event_type": normalized_type,
        "action": action,
        "actor_client_id": actor_client_id,
        "actor_email": actor_email,
        "subject_id": subject_id,
        "subject_type": subject_type,
        "metadata": metadata or {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        supabase.table("activity_events").insert(payload).execute()
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Failed to record activity event: %s", exc)


def fetch_events(
    limit: int = 200,
    offset: int = 0,
    event_type: str | None = None,
    client_id: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
):
    """Return filtered events for admin UI with actor names enriched."""
    supabase = get_supabase_client()
    query = supabase.table("activity_events").select("*", count="exact").order("created_at", desc=True).range(offset, offset + limit - 1)

    if event_type:
        query = query.eq("event_type", event_type)
    if client_id:
        query = query.eq("actor_client_id", client_id)
    if since:
        query = query.gte("created_at", since.isoformat())
    if until:
        query = query.lte("created_at", until.isoformat())

    resp = query.execute()
    events = resp.data or []
    
    # Enrich events with actor names
    actor_ids = {e.get("actor_client_id") for e in events if e.get("actor_client_id")}
    if actor_ids:
        clients_resp = supabase.table("clients").select("id, name").in_("id", list(actor_ids)).execute()
        actor_map = {c["id"]: c.get("name") for c in (clients_resp.data or [])}
        for event in events:
            actor_id = event.get("actor_client_id")
            if actor_id and actor_id in actor_map:
                event["actor_name"] = actor_map[actor_id]
    
    return {"items": events, "total": resp.count or 0}


def export_events_csv(events: Iterable[dict]) -> bytes:
    """Export events to CSV."""
    buffer = StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=[
            "created_at",
            "event_type",
            "action",
            "actor_client_id",
            "actor_email",
            "subject_id",
            "subject_type",
            "metadata",
        ],
    )
    writer.writeheader()
    for event in events:
        writer.writerow(
            {
                "created_at": event.get("created_at"),
                "event_type": event.get("event_type"),
                "action": event.get("action"),
                "actor_client_id": event.get("actor_client_id"),
                "actor_email": event.get("actor_email"),
                "subject_id": event.get("subject_id"),
                "subject_type": event.get("subject_type"),
                "metadata": event.get("metadata"),
            }
        )
    return buffer.getvalue().encode("utf-8")


def record_session(
    client_id: str,
    api_key: str,
    user_agent: str | None = None,
    ip_address: str | None = None,
) -> str | None:
    """Insert or upsert a client session row for visibility and revocation."""
    supabase = get_supabase_client()
    now_iso = datetime.now(timezone.utc).isoformat()
    payload = {
        "client_id": client_id,
        "api_key": api_key,
        "user_agent": user_agent,
        "ip_address": ip_address,
        "last_seen_at": now_iso,
        "created_at": now_iso,
        "revoked": False,
    }
    try:
        resp = (
            supabase.table("client_sessions")
            .upsert(payload, on_conflict="api_key", ignore_duplicates=False)
            .execute()
        )
        rows = resp.data or []
        if rows:
            return rows[0].get("id")
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Failed to record session: %s", exc)
    return None


def touch_session(
    client_id: str,
    api_key: str,
    user_agent: str | None = None,
    ip_address: str | None = None,
) -> None:
    """Refresh last_seen_at for an existing session."""
    supabase = get_supabase_client()
    now_iso = datetime.now(timezone.utc).isoformat()
    try:
        supabase.table("client_sessions").update(
            {
                "last_seen_at": now_iso,
                "user_agent": user_agent,
                "ip_address": ip_address,
            }
        ).eq("client_id", client_id).eq("api_key", api_key).execute()
    except Exception:
        # Non-blocking
        pass


def list_sessions(include_revoked: bool = False, limit: int = 200):
    supabase = get_supabase_client()
    query = supabase.table("client_sessions").select("*").order("last_seen_at", desc=True).limit(limit)
    if not include_revoked:
        query = query.eq("revoked", False)
    resp = query.execute()
    return resp.data or []


def revoke_session(session_id: str, revoked_by: str) -> dict:
    supabase = get_supabase_client()
    payload = {
        "revoked": True,
        "revoked_by": revoked_by,
        "revoked_at": datetime.now(timezone.utc).isoformat(),
    }
    resp = supabase.table("client_sessions").update(payload).eq("id", session_id).execute()
    rows = resp.data or []
    if not rows:
        raise ValueError("Session not found")
    return rows[0]


def revoke_sessions_for_client(client_id: str) -> None:
    """Mark all sessions for a client as revoked."""
    supabase = get_supabase_client()
    try:
        supabase.table("client_sessions").update(
            {"revoked": True, "revoked_at": datetime.now(timezone.utc).isoformat()}
        ).eq("client_id", client_id).execute()
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Failed to revoke sessions for client %s: %s", client_id, exc)


def get_admin_analytics(hours: int = 168) -> dict:
    """Aggregate admin analytics with time-series buckets."""
    supabase = get_supabase_client()
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=hours)

    def _count(table: str, **filters) -> int:
        query = supabase.table(table).select("id", count="exact")
        for key, value in filters.items():
            query = query.eq(key, value)
        query = query.gte("created_at", since.isoformat())
        resp = query.execute()
        return resp.count or 0

    def _bucket_time_series(rows: list[dict], hours_range: int) -> list[dict]:
        """Bucket data by hour (for <=24h) or by day (for >24h)."""
        buckets: dict[str, int] = {}
        
        # Use hourly buckets for 24 hours or less, daily for longer periods
        use_hourly = hours_range <= 24
        
        for row in rows:
            ts = row.get("created_at") or row.get("ingested_at")
            if not ts:
                continue
            
            if use_hourly:
                # Extract hour: "2025-12-11T15:30:45" -> "2025-12-11T15"
                hour_key = ts[:13]
                buckets[hour_key] = buckets.get(hour_key, 0) + 1
            else:
                # Extract day: "2025-12-11T15:30:45" -> "2025-12-11"
                day_key = ts[:10]
                buckets[day_key] = buckets.get(day_key, 0) + 1
        
        # Fill in missing periods with zeros
        result = []
        current = since
        end = now
        
        if use_hourly:
            # Generate hourly buckets
            while current <= end:
                key = current.strftime("%Y-%m-%dT%H")
                result.append({"date": key, "count": buckets.get(key, 0)})
                current += timedelta(hours=1)
        else:
            # Generate daily buckets
            while current <= end:
                key = current.strftime("%Y-%m-%d")
                result.append({"date": key, "count": buckets.get(key, 0)})
                current += timedelta(days=1)
        
        return result

    total_clients = supabase.table("clients").select("id", count="exact").execute().count or 0
    active_clients = (
        supabase.table("clients")
        .select("id", count="exact")
        .eq("status", "active")
        .execute()
        .count
        or 0
    )
    suspended_clients = (
        supabase.table("clients")
        .select("id", count="exact")
        .eq("status", "suspended")
        .execute()
        .count
        or 0
    )

    rfps_created = _count("client_rfps")
    jobs_submitted = _count("client_jobs")
    events_recorded = (
        supabase.table("activity_events")
        .select("id", count="exact")
        .gte("created_at", since.isoformat())
        .execute()
        .count
        or 0
    )

    activity_rows = (
        supabase.table("activity_events")
        .select("created_at, actor_client_id")
        .gte("created_at", since.isoformat())
        .execute()
        .data
        or []
    )
    dau_buckets = _bucket_time_series(activity_rows, hours)

    rfps_rows = supabase.table("client_rfps").select("created_at").gte("created_at", since.isoformat()).execute().data or []
    jobs_rows = supabase.table("client_jobs").select("created_at").gte("created_at", since.isoformat()).execute().data or []
    tender_rows = supabase.table("tender_matches").select("created_at").gte("created_at", since.isoformat()).execute().data or []
    signup_rows = supabase.table("clients").select("created_at").gte("created_at", since.isoformat()).execute().data or []

    return {
        "clients": {
            "total": total_clients,
            "active": active_clients,
            "suspended": suspended_clients,
        },
        "rfps_created": rfps_created,
        "jobs_submitted": jobs_submitted,
        "events_recorded": events_recorded,
        "range_hours": hours,
        "generated_at": now.isoformat(),
        "series": {
            "dau": dau_buckets,
            "rfps": _bucket_time_series(rfps_rows, hours),
            "jobs": _bucket_time_series(jobs_rows, hours),
            "tenders": _bucket_time_series(tender_rows, hours),
            "signups": _bucket_time_series(signup_rows, hours),
        },
    }

