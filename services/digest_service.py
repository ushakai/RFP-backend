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
    <p style="font-family: Arial, sans-serif; color:#666; font-size:14px; margin-top:8px;">Click on any tender to view full details and purchase access for £5.</p>
    """

    body_sections = []
    for entry in entries:
        location_info = f"<li>Location: <strong>{entry.get('location', 'Not specified')}</strong></li>" if entry.get('location') else ""
        category_info = f"<li>Category: <strong>{entry.get('category', 'Not specified')}</strong></li>" if entry.get('category') else ""
        match_score_info = f"<li>Match Score: <strong>{entry.get('match_score', 'N/A')}</strong></li>" if entry.get('match_score') else ""
        
        # Status badge styling
        status = entry.get('status', 'New')
        status_color = "#10b981" if status == "New" else "#f59e0b"
        status_badge = f'<span style="display:inline-block; padding:4px 10px; background-color:{status_color}; color:#fff; font-size:12px; font-weight:600; border-radius:4px; margin-left:8px;">{status}</span>'
        
        section = f"""
        <div style="margin-bottom:24px; padding:20px; border:1px solid #ddd; border-radius:8px; background-color:#fafafa;">
            <h3 style="margin:0 0 12px 0; font-family: Arial, sans-serif; font-size:18px; line-height:1.4;">
                <a href="{entry['link']}" style="color:#2563eb; text-decoration:none;" target="_blank">{entry['title']}</a>
                {status_badge}
            </h3>
            <div style="margin:0 0 16px 0; padding:12px; background-color:#fff; border-left:3px solid #2563eb; border-radius:4px;">
                <p style="margin:0; color:#444; font-family: Arial, sans-serif; line-height:1.6; font-size:14px;">{entry['summary']}</p>
            </div>
            <ul style="margin:0; padding-left:20px; color:#555; font-family: Arial, sans-serif; font-size:13px; line-height:1.8;">
                <li>Source: <strong>{entry['source']}</strong></li>
                <li>Published: <strong>{entry.get('published_date', 'Not specified')}</strong></li>
                <li>Deadline: <strong>{entry['deadline']}</strong></li>
                <li>Matching keywords: <strong style="color:#059669;">{entry['keywords']}</strong></li>
                {match_score_info}
            </ul>
            <div style="margin-top:16px; padding-top:16px; border-top:1px solid #eee;">
                <a href="{entry['link']}" style="display:inline-block; padding:10px 20px; background-color:#2563eb; color:#fff; text-decoration:none; border-radius:6px; font-weight:600; font-size:14px;" target="_blank">View Full Details & Purchase (£5)</a>
            </div>
        </div>
        """
        body_sections.append(section)

    footer = """
    <p style="font-family: Arial, sans-serif; color:#888; font-size:12px;">
        You are receiving this email because you subscribed to daily tender updates.
    </p>
    """

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
    now_utc = datetime.now(timezone.utc)
    if since_utc is None:
        start_of_day_uk = datetime.combine(uk_now.date(), dt_time.min, tzinfo=UK_TIMEZONE)
        since_utc = start_of_day_uk.astimezone(timezone.utc)
    since_iso = since_utc.isoformat()

    payloads: List[Dict[str, Any]] = []

    keyword_resp = (
        supabase.table("user_tender_keywords")
        .select("client_id, keywords, is_active, created_at, updated_at")
        .eq("is_active", True)
        .execute()
    )
    eligible_clients: set[str] = set()
    new_keyword_clients: set[str] = set()  # Clients who added keywords recently (last 7 days)
    for row in keyword_resp.data or []:
        client_id = row.get("client_id")
        keywords = row.get("keywords")
        if client_id and isinstance(keywords, list):
            if any(isinstance(kw, str) and kw.strip() for kw in keywords):
                eligible_clients.add(client_id)
                # Check if keywords were added/updated recently (within last 7 days)
                keyword_created = _parse_datetime(row.get("created_at") or row.get("updated_at"))
                if keyword_created:
                    days_ago = (now_utc - keyword_created).days
                    if days_ago <= 7:
                        new_keyword_clients.add(client_id)

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

        # Get matches where either:
        # 1. The match was created recently (new match for existing tender), OR
        # 2. The tender was created recently (new tender that matches existing keywords)
        # We use OR logic: match.created_at >= since_utc OR tender.created_at >= since_utc
        matches_resp = (
            supabase.table("tender_matches")
            .select(
                "match_score, matched_keywords, created_at, "
                "tenders(id, title, summary, description, source, deadline, value_amount, value_currency, created_at, location, category, metadata, full_data)"
            )
            .eq("client_id", client_id)
            .order("match_score", desc=True)
            .limit(100)
            .execute()
        )

        matches = matches_resp.data or []
        entries: List[Dict[str, Any]] = []
        
        for match in matches:
            tender = match.get("tenders")
            if not isinstance(tender, dict):
                continue

            # Check if this match/tender should be included:
            # - For new keyword clients (added keywords in last 7 days): include all matches
            # - For existing clients: include if match was created recently OR tender was created recently
            if client_id not in new_keyword_clients:
                match_created = _parse_datetime(match.get("created_at"))
                tender_created = _parse_datetime(tender.get("created_at"))
                
                match_is_recent = match_created and match_created >= since_utc
                tender_is_recent = tender_created and tender_created >= since_utc
                
                if not match_is_recent and not tender_is_recent:
                    continue

            # Get enriched title from metadata or use title
            metadata = tender.get("metadata") or {}
            if isinstance(metadata, dict):
                # Try to get enriched title from metadata (processed title)
                enriched_title = metadata.get("processed_details", {}).get("title") if isinstance(metadata.get("processed_details"), dict) else None
                if not enriched_title:
                    # Fallback to title in database (which may have been enriched during ingestion)
                    enriched_title = tender.get("title")
            else:
                enriched_title = tender.get("title")
            
            title = (enriched_title or tender.get("title") or "").strip()
            if not title:
                continue

            # Get location and category from metadata or direct fields
            location_name = metadata.get("location_name") if isinstance(metadata, dict) else None
            if not location_name:
                location_name = tender.get("location")
            
            category_label = metadata.get("category_label") if isinstance(metadata, dict) else None
            if not category_label:
                category_label = tender.get("category")

            # Build a rich summary for email (4-5 lines with key details)
            summary_parts = []
            
            # Start with existing summary/description
            base_summary = (tender.get("summary") or "").strip()
            if not base_summary:
                base_summary = (tender.get("description") or "").strip()
            
            if base_summary:
                # Limit to first 200 chars for base summary
                if len(base_summary) > 200:
                    base_summary = base_summary[:200].rsplit(' ', 1)[0] + "..."
                summary_parts.append(base_summary)
            
            # Add scope/requirements if available from full_data
            full_data = tender.get("full_data") or {}
            if isinstance(full_data, dict):
                # Try to extract items/deliverables
                tender_info = full_data.get("tender") or {}
                items = tender_info.get("items") or []
                if items and isinstance(items, list) and len(items) > 0:
                    item_descriptions = []
                    for item in items[:3]:  # Max 3 items
                        if isinstance(item, dict):
                            desc = item.get("description") or ""
                            if desc and len(desc) < 80:
                                item_descriptions.append(desc)
                    if item_descriptions:
                        summary_parts.append("Scope includes: " + "; ".join(item_descriptions))
            
            # Add key metadata context
            context_parts = []
            if category_label and category_label != "Not specified":
                context_parts.append(f"Category: {category_label}")
            if location_name and location_name != "Not specified":
                context_parts.append(f"Location: {location_name}")
            
            value_amount_raw = tender.get("value_amount")
            if value_amount_raw:
                try:
                    value_formatted = _format_currency(value_amount_raw, tender.get("value_currency"))
                    context_parts.append(f"Est. Value: {value_formatted}")
                except:
                    pass
            
            if context_parts:
                summary_parts.append(" | ".join(context_parts))
            
            # Add call-to-action hint
            summary_parts.append("View full tender details including documents, requirements, and buyer information.")
            
            summary_text = " ".join(summary_parts)

            deadline = _parse_datetime(tender.get("deadline"))
            # Skip tenders with passed deadlines
            if deadline:
                if deadline < now_utc:
                    continue
                deadline_text = deadline.astimezone(UK_TIMEZONE).strftime("%d %b %Y")
            else:
                deadline_text = "Not specified"

            # Format published date
            published_date = _parse_datetime(tender.get("published_date"))
            if published_date:
                published_text = published_date.astimezone(UK_TIMEZONE).strftime("%d %b %Y")
            else:
                published_text = "Not specified"

            # Determine if this is a new tender or updated match
            match_created = _parse_datetime(match.get("created_at"))
            tender_created = _parse_datetime(tender.get("created_at"))
            is_new_tender = tender_created and tender_created >= since_utc
            status_label = "New" if is_new_tender else "Updated Match"

            # Format match score as percentage
            match_score = match.get("match_score")
            match_score_text = ""
            numeric_score: float = 0.0
            if match_score is not None:
                try:
                    numeric_score = float(match_score)
                    score_pct = numeric_score * 100
                    match_score_text = f"{score_pct:.0f}%"
                except (TypeError, ValueError):
                    numeric_score = 0.0

            entry_payload = {
                "title": title,
                "summary": summary_text,
                "source": tender.get("source") or "Unknown",
                "location": location_name or "Not specified",
                "category": category_label or "Not specified",
                "deadline": deadline_text,
                "published_date": published_text,
                "status": status_label,
                "value": _format_currency(tender.get("value_amount"), tender.get("value_currency")),
                "keywords": ", ".join(match.get("matched_keywords") or []) or "Not specified",
                "match_score": match_score_text,
                "link": f"{FRONTEND_BASE}/tenders/{tender.get('id')}",
                "_score": numeric_score,
            }
            entries.append(entry_payload)

        if not entries:
            continue

        # Keep only the top 10 entries by match score
        entries.sort(key=lambda item: item.get("_score", 0.0), reverse=True)
        top_entries = []
        for entry in entries[:10]:
            entry = dict(entry)
            entry.pop("_score", None)
            top_entries.append(entry)

        payloads.append(
            {
                "client_id": client_id,
                "client_name": client.get("name") or "",
                "email": email,
                "subject": f"Daily Tender Digest — {uk_now.strftime('%d %b %Y')}",
                "entries": top_entries,
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

