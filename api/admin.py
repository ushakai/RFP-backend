from __future__ import annotations

import importlib
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Header, HTTPException

from config.settings import UK_TIMEZONE, ENABLE_TENDER_INGESTION
from services.digest_service import (
    send_daily_digest_since,
    record_ingestion,
    record_digest,
)
from utils.auth import get_client_id_from_key, is_admin_client
from api.tenders import notify_client as _notify_client
import api.tenders as tenders_api

router = APIRouter(prefix="/admin")


@router.get("/status")
def admin_status(x_client_key: str | None = Header(default=None, alias="X-Client-Key")):
    client_id = get_client_id_from_key(x_client_key)
    return {"is_admin": is_admin_client(client_id)}


@router.post("/run-daily-cycle")
def run_daily_cycle(x_client_key: str | None = Header(default=None, alias="X-Client-Key")):
    """
    Manually trigger the daily ingestion and digest email process.
    """
    client_id = get_client_id_from_key(x_client_key)
    if not is_admin_client(client_id):
        raise HTTPException(status_code=403, detail="Admin access required")

    ingestion_module = importlib.import_module("tender_ingestion")
    stored = matched = 0
    new_ids: list[str] = []
    if ENABLE_TENDER_INGESTION:
        stored, matched, new_ids = ingestion_module.ingest_all_tenders(force=True)
        record_ingestion(datetime.now(timezone.utc))
        for cid in list(tenders_api._client_streams.keys()):
            _notify_client(cid, "matches-updated", {"reason": "ingestion"})
    else:
        print("ENABLE_TENDER_INGESTION disabled; skipping manual ingestion run.")

    # Send digest for the last 24 hours by default
    since_utc = datetime.now(timezone.utc) - timedelta(hours=24)
    digest_summary = send_daily_digest_since(since_utc)
    record_digest(datetime.now(UK_TIMEZONE).date())

    return {
        "stored": stored,
        "matched": matched,
        "new_tenders": len(new_ids),
        "emails_attempted": digest_summary["attempted"],
        "emails_sent": digest_summary["sent"],
    }

