"""
Daily tender digest processing and email delivery.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone, time as dt_time
from typing import Any, Dict, List

from fpdf import FPDF

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


def collect_digest_payloads(since_utc: datetime | None = None) -> Dict[str, Any]:
    """
    Build the per-client digest payloads (without dispatching emails).

    Returns a dict containing:
        - payloads: list of per-client payload dictionaries
        - prepared_at: timezone-aware datetime in UK timezone
        - since_utc: UTC datetime used for filtering matches
    """
    supabase = get_supabase_client()

    uk_now = datetime.now(UK_TIMEZONE)
    if since_utc is None:
        start_of_day_uk = datetime.combine(uk_now.date(), dt_time.min, tzinfo=UK_TIMEZONE)
        since_utc = start_of_day_uk.astimezone(timezone.utc)
    since_iso = since_utc.isoformat()

    payloads: List[Dict[str, Any]] = []

    keyword_resp = (
        supabase.table("user_tender_keywords")
        .select("client_id, keywords, is_active")
        .eq("is_active", True)
        .execute()
    )
    eligible_clients: set[str] = set()
    for row in keyword_resp.data or []:
        client_id = row.get("client_id")
        keywords = row.get("keywords")
        if client_id and isinstance(keywords, list):
            if any(isinstance(kw, str) and kw.strip() for kw in keywords):
                eligible_clients.add(client_id)

    if not eligible_clients:
        return {"payloads": payloads, "prepared_at": uk_now, "since_utc": since_utc}

    clients_resp = (
        supabase.table("clients")
        .select("id, name, contact_email")
        .in_("id", list(eligible_clients))
        .execute()
    )
    clients = clients_resp.data or []

    for client in clients:
        client_id = client.get("id")
        email = (client.get("contact_email") or "").strip()
        if not client_id or not email:
            continue

        matches_resp = (
            supabase.table("tender_matches")
            .select(
                "match_score, matched_keywords, created_at, "
                "tenders(id, title, summary, source, deadline, value_amount, value_currency, created_at)"
            )
            .eq("client_id", client_id)
            .gte("created_at", since_iso)
            .order("match_score", desc=True)
            .limit(50)
            .execute()
        )

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
            # Skip tenders with passed deadlines
            if deadline:
                now_utc = datetime.now(timezone.utc)
                if deadline < now_utc:
                    continue
                deadline_text = deadline.astimezone(UK_TIMEZONE).strftime("%d %b %Y")
            else:
                deadline_text = "Not specified"

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

        payloads.append(
            {
                "client_id": client_id,
                "client_name": client.get("name") or "",
                "email": email,
                "subject": f"Daily Tender Digest — {uk_now.strftime('%d %b %Y')}",
                "entries": entries,
            }
        )

    return {"payloads": payloads, "prepared_at": uk_now, "since_utc": since_utc}


def generate_digest_preview_pdf(
    payloads: List[Dict[str, Any]], prepared_at: datetime, since_utc: datetime
) -> bytes:
    """
    Build a PDF report summarising all pending digest emails for admin review.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_title("Daily Tender Digest Preview")

    if not payloads:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.multi_cell(0, 10, "Daily Tender Digest Preview")
        pdf.set_font("Helvetica", "", 12)
        pdf.ln(4)
        pdf.multi_cell(
            0,
            8,
            "No client digests are ready for the selected window. "
            "Ensure keyword sets are active and matches exist.",
        )
        return pdf.output(dest="S").encode("latin1")

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Daily Tender Digest Preview", ln=1)
    pdf.set_font("Helvetica", "", 12)
    pdf.multi_cell(
        0,
        8,
        f"Prepared at: {prepared_at.strftime('%d %b %Y %H:%M %Z')}",
    )
    pdf.multi_cell(
        0,
        8,
        f"Includes matches created since: {since_utc.astimezone(UK_TIMEZONE).strftime('%d %b %Y %H:%M %Z')}",
    )
    pdf.multi_cell(0, 8, f"Total client emails prepared: {len(payloads)}")
    pdf.ln(4)
    pdf.multi_cell(0, 8, "Detailed per-client breakdown:")

    for payload in payloads:
        pdf.add_page()
        client_name = payload.get("client_name") or "Unnamed client"
        email = payload.get("email") or "Not Provided"
        subject = payload.get("subject") or "Daily Tender Digest"
        entries = payload.get("entries") or []

        pdf.set_font("Helvetica", "B", 14)
        pdf.multi_cell(0, 8, client_name)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, f"Email: {email}")
        pdf.multi_cell(0, 6, f"Subject: {subject}")
        pdf.multi_cell(0, 6, f"Tenders included: {len(entries)}")
        pdf.ln(2)

        for idx, entry in enumerate(entries, start=1):
            pdf.set_font("Helvetica", "B", 12)
            pdf.multi_cell(0, 6, f"{idx}. {entry.get('title', 'Untitled tender')}")
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 5, f"Summary: {entry.get('summary', 'N/A')}")
            pdf.multi_cell(
                0,
                5,
                "Source: {source} | Deadline: {deadline} | Value: {value}".format(
                    source=entry.get("source", "Unknown"),
                    deadline=entry.get("deadline", "Not specified"),
                    value=entry.get("value", "Not disclosed"),
                ),
            )
            pdf.multi_cell(0, 5, f"Keywords: {entry.get('keywords', 'Not specified')}")
            pdf.multi_cell(0, 5, f"Link: {entry.get('link', '-')}")
            pdf.ln(3)

    return pdf.output(dest="S").encode("latin1")


def send_daily_digest_since(since_utc: datetime | None = None) -> Dict[str, int]:
    """
    Gather tender matches created since `since_utc` and send per-client digests.

    Returns a dictionary with counts of attempted and successfully sent emails.
    """
    digest_bundle = collect_digest_payloads(since_utc)
    payloads: List[Dict[str, Any]] = digest_bundle["payloads"]
    prepared_at: datetime = digest_bundle["prepared_at"]

    summary = {"attempted": 0, "sent": 0}

    for payload in payloads:
        email = payload.get("email")
        if not email:
            continue

        summary["attempted"] += 1
        html = _build_email_html(payload.get("client_name") or "", payload.get("entries") or [], prepared_at)
        if send_email([email], payload.get("subject") or "Daily Tender Digest", html):
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

