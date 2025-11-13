"""
Daily tender digest processing and email delivery.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone, time as dt_time
from typing import Dict, List

from config.settings import FRONTEND_ORIGIN, UK_TIMEZONE, get_supabase_client
from services.email_service import send_email

FRONTEND_BASE = FRONTEND_ORIGIN.rstrip("/")
_LAST_INGESTION_UTC: datetime | None = None
_LAST_DIGEST_DATE: date | None = None


def _parse_datetime(value) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _format_currency(amount, currency) -> str:
    if amount is None:
        return "Not disclosed"
    try:
        amount = float(amount)
    except Exception:
        return str(amount)
    code = (currency or "GBP").upper()
    symbol = "£" if code == "GBP" else ""
    return f"{symbol}{amount:,.0f} {code if symbol == '' else ''}".strip()


def _build_email_html(client_name: str, entries: List[Dict[str, str]], uk_date: datetime) -> str:
    header = f"""
    <h2 style="font-family: Arial, sans-serif;">Daily Tender Digest — {uk_date.strftime('%d %B %Y')}</h2>
    <p style="font-family: Arial, sans-serif; color:#555;">Hello {client_name or 'there'}, here are the new tenders that matched your keywords in the last 24 hours.</p>
    """

    body_sections = []
    for entry in entries:
        section = f"""
        <div style="margin-bottom:18px; padding:16px; border:1px solid #eee; border-radius:8px;">
            <h3 style="margin:0 0 8px 0; font-family: Arial, sans-serif;">
                <a href="{entry['link']}" style="color:#2563eb; text-decoration:none;" target="_blank">{entry['title']}</a>
            </h3>
            <p style="margin:0 0 12px 0; color:#444; font-family: Arial, sans-serif;">{entry['summary']}</p>
            <ul style="margin:0; padding-left:18px; color:#555; font-family: Arial, sans-serif; font-size:14px;">
                <li>Source: <strong>{entry['source']}</strong></li>
                <li>Deadline: <strong>{entry['deadline']}</strong></li>
                <li>Value: <strong>{entry['value']}</strong></li>
                <li>Matching keywords: <strong>{entry['keywords']}</strong></li>
            </ul>
        </div>
        """
        body_sections.append(section)

    footer = """
    <p style="font-family: Arial, sans-serif; color:#888; font-size:12px;">
        You are receiving this email because you subscribed to daily tender updates.
    </p>
    """

    if not body_sections:
        body_sections.append(
            "<p style='font-family: Arial, sans-serif; color:#555;'>No new tenders matched your criteria today.</p>"
        )

    return header + "".join(body_sections) + footer


def send_daily_digest_since(since_utc: datetime | None = None) -> Dict[str, int]:
    """
    Gather tender matches created since `since_utc` and send per-client digests.

    Args:
        since_utc: Only include matches created after this UTC timestamp. Defaults to start of the current UK day.

    Returns:
        Dictionary with counts of attempted and successfully sent emails.
    """
    supabase = get_supabase_client()

    uk_now = datetime.now(UK_TIMEZONE)
    if since_utc is None:
        start_of_day_uk = datetime.combine(uk_now.date(), dt_time.min, tzinfo=UK_TIMEZONE)
        since_utc = start_of_day_uk.astimezone(timezone.utc)
    since_iso = since_utc.isoformat()

    summary = {"attempted": 0, "sent": 0}

    keyword_resp = supabase.table("user_tender_keywords").select("client_id, keywords, is_active").eq("is_active", True).execute()
    eligible_clients: set[str] = set()
    for row in keyword_resp.data or []:
        client_id = row.get("client_id")
        keywords = row.get("keywords")
        if client_id and isinstance(keywords, list):
            if any(isinstance(kw, str) and kw.strip() for kw in keywords):
                eligible_clients.add(client_id)

    if not eligible_clients:
        return summary

    clients_resp = supabase.table("clients").select("id, name, contact_email").in_("id", list(eligible_clients)).execute()
    clients = clients_resp.data or []

    for client in clients:
        client_id = client.get("id")
        email = (client.get("contact_email") or "").strip()
        if not client_id or not email:
            continue

        matches_resp = supabase.table("tender_matches").select(
            "match_score, matched_keywords, created_at, "
            "tenders(id, title, summary, source, deadline, value_amount, value_currency, created_at)"
        ).eq("client_id", client_id).gte("created_at", since_iso).order("match_score", desc=True).limit(50).execute()

        matches = matches_resp.data or []
        entries: List[Dict[str, str]] = []
        for match in matches:
            tender = match.get("tenders")
            if not isinstance(tender, dict):
                continue

            tender_created = _parse_datetime(tender.get("created_at"))
            if tender_created and tender_created < since_utc:
                continue

            title = (tender.get("title") or "").strip()
            if not title:
                continue

            summary_text = (tender.get("summary") or "").strip()
            if not summary_text:
                continue

            deadline = _parse_datetime(tender.get("deadline"))
            deadline_text = deadline.astimezone(UK_TIMEZONE).strftime("%d %b %Y") if deadline else "Not specified"

            entries.append(
                {
                    "title": title,
                    "summary": summary_text,
                    "source": tender.get("source") or "Unknown",
                    "deadline": deadline_text,
                    "value": _format_currency(tender.get("value_amount"), tender.get("value_currency")),
                    "keywords": ", ".join(match.get("matched_keywords") or []) or "Not specified",
                    "link": f"{FRONTEND_BASE}/tenders/{tender.get('id')}",
                }
            )
            if len(entries) >= 10:
                break

        if not entries:
            continue

        summary["attempted"] += 1
        html = _build_email_html(client.get("name") or "", entries, uk_now)
        subject = f"Daily Tender Digest — {uk_now.strftime('%d %b %Y')}"
        if send_email([email], subject, html):
            summary["sent"] += 1

    return summary


def record_ingestion(timestamp: datetime | None = None) -> None:
    global _LAST_INGESTION_UTC
    _LAST_INGESTION_UTC = timestamp or datetime.now(timezone.utc)


def get_last_ingestion() -> datetime | None:
    return _LAST_INGESTION_UTC


def record_digest(date_value: date | None = None) -> None:
    global _LAST_DIGEST_DATE
    _LAST_DIGEST_DATE = date_value or datetime.now(UK_TIMEZONE).date()


def get_last_digest_date() -> date | None:
    return _LAST_DIGEST_DATE

