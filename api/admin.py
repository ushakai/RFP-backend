from __future__ import annotations

import importlib
from datetime import datetime, timedelta, timezone
from io import BytesIO

from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import StreamingResponse

from config.settings import UK_TIMEZONE, ENABLE_TENDER_INGESTION
from services.digest_service import (
    send_daily_digest_since,
    record_ingestion,
    record_digest,
    collect_digest_payloads,
    generate_digest_preview_pdf,
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


def _parse_since_param(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        else:
            parsed = parsed.astimezone(timezone.utc)
        return parsed
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid 'since' timestamp. Use ISO 8601.") from exc


@router.get("/daily-digest-report", response_class=StreamingResponse)
def download_daily_digest_report(
    x_client_key: str | None = Header(default=None, alias="X-Client-Key"),
    since: str | None = Query(default=None, description="ISO8601 UTC timestamp to filter matches from"),
):
    """
    Generate a PDF report showing the pending daily digest emails per client.
    """
    client_id = get_client_id_from_key(x_client_key)
    if not is_admin_client(client_id):
        raise HTTPException(status_code=403, detail="Admin access required")

    since_dt = _parse_since_param(since)
    digest_bundle = collect_digest_payloads(since_dt)
    pdf_bytes = generate_digest_preview_pdf(
        digest_bundle["payloads"],
        digest_bundle["prepared_at"],
        digest_bundle["since_utc"],
    )

    filename = f"digest-preview-{digest_bundle['prepared_at'].strftime('%Y%m%d-%H%M%S')}.pdf"
    return StreamingResponse(
        BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

