"""
Daily tender digest processing and email delivery.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone, time as dt_time
from typing import Any, Dict, List

from fpdf import FPDF

import json
from config.settings import FRONTEND_ORIGIN, UK_TIMEZONE, get_supabase_client, FILTER_UK_ONLY
from services.email_service import send_email
from services.tender_service import is_uk_tender, is_uk_specific_source

FRONTEND_BASE = FRONTEND_ORIGIN.rstrip("/")


def _is_ted_source(value: str | None) -> bool:
    """Check if the source is TED."""
    return isinstance(value, str) and "ted" in value.lower()
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


def _generate_compelling_email_summary(tender_data: Dict[str, Any], full_data: Dict[str, Any] | None) -> str:
    """
    Generate a summary for email digest from tender data.
    Uses the description and metadata - no AI calls.
    """
    description = tender_data.get("description", "") or tender_data.get("summary", "")
    category = tender_data.get("category_label") or tender_data.get("category", "")
    location = tender_data.get("location_name") or tender_data.get("location", "")
    
    # Build a summary from available data
    parts = []
    
    if description:
        # Take first 300 chars of description for email summary
        desc_short = description[:300].strip()
        if len(description) > 300:
            # Truncate at last space
            last_space = desc_short.rfind(' ')
            if last_space > 150:
                desc_short = desc_short[:last_space] + "..."
        parts.append(desc_short)
    
    if category and location:
        parts.append(f"This {category.lower()} opportunity is based in {location}.")
    elif category:
        parts.append(f"This is a {category.lower()} procurement opportunity.")
    elif location:
        parts.append(f"This opportunity is based in {location}.")
    
    if parts:
        return " ".join(parts)
    
    return "This opportunity requires detailed review. View full tender details for complete information."


def _build_email_html(client_name: str, entries: List[Dict[str, str]], uk_date: datetime) -> str:
    # Modern email template with gradient header and card design
    header = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body style="margin:0; padding:0; background-color:#f5f7fa; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;">
        <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#f5f7fa; padding:40px 20px;">
            <tr>
                <td align="center">
                    <table width="600" cellpadding="0" cellspacing="0" style="max-width:600px; background-color:#ffffff; border-radius:12px; overflow:hidden; box-shadow:0 4px 6px rgba(0,0,0,0.1);">
                        <!-- Header with gradient -->
                        <tr>
                            <td style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding:40px 30px; text-align:center;">
                                <h1 style="margin:0; color:#ffffff; font-size:28px; font-weight:700; letter-spacing:-0.5px;">Daily Tender Digest</h1>
                                <p style="margin:12px 0 0 0; color:#ffffff; font-size:16px; opacity:0.95;">{uk_date.strftime('%d %B %Y')}</p>
                            </td>
                        </tr>
                        <!-- Greeting -->
                        <tr>
                            <td style="padding:30px 30px 20px 30px;">
                                <p style="margin:0; color:#1a202c; font-size:16px; line-height:1.6;">Hello <strong>{client_name or 'there'}</strong>,</p>
                                <p style="margin:12px 0 0 0; color:#4a5568; font-size:15px; line-height:1.6;">Here are the top opportunities that matched your keywords today.</p>
                            </td>
                        </tr>
    """

    body_sections = []
    for idx, entry in enumerate(entries, 1):
        # Status badge styling
        status = entry.get('status', 'New')
        status_color = "#10b981" if status == "New" else "#f59e0b"
        status_bg = "#d1fae5" if status == "New" else "#fef3c7"
        
        section = f"""
                        <!-- Tender Card {idx} -->
                        <tr>
                            <td style="padding:0 30px 30px 30px;">
                                <div style="background-color:#ffffff; border:1px solid #e2e8f0; border-radius:10px; overflow:hidden; transition:all 0.3s;">
                                    <!-- Card Header -->
                                    <div style="background:linear-gradient(to right, #f8fafc 0%, #ffffff 100%); padding:20px 24px; border-bottom:1px solid #e2e8f0;">
                                        <div style="display:flex; align-items:flex-start; justify-content:space-between; flex-wrap:wrap; gap:12px;">
                                            <div style="flex:1; min-width:200px;">
                                                <h2 style="margin:0 0 8px 0; color:#1a202c; font-size:20px; font-weight:600; line-height:1.3;">
                                                    <a href="{entry['link']}" style="color:#2563eb; text-decoration:none; transition:color 0.2s;" target="_blank">{entry['title']}</a>
                                                </h2>
                                                <span style="display:inline-block; padding:4px 12px; background-color:{status_bg}; color:{status_color}; font-size:11px; font-weight:600; border-radius:12px; text-transform:uppercase; letter-spacing:0.5px;">{status}</span>
                                            </div>
                                        </div>
                                    </div>
                                    
                                    <!-- Summary Section -->
                                    <div style="padding:24px;">
                                        <p style="margin:0; color:#2d3748; font-size:15px; line-height:1.7; font-weight:400;">{entry['summary']}</p>
                                    </div>
                                    
                                    <!-- Metadata Grid -->
                                    <div style="padding:0 24px 20px 24px;">
                                        <table width="100%" cellpadding="0" cellspacing="0">
                                            <tr>
                                                <td style="padding:8px 0; color:#718096; font-size:13px; width:50%;">
                                                    <strong style="color:#4a5568;">Published:</strong><br>
                                                    <span style="color:#2d3748;">{entry.get('published_date', 'Not specified')}</span>
                                                </td>
                                                <td style="padding:8px 0; color:#718096; font-size:13px; width:50%;">
                                                    <strong style="color:#4a5568;">Deadline:</strong><br>
                                                    <span style="color:#2d3748;">{entry['deadline']}</span>
                                                </td>
                                            </tr>
                                            <tr>
                                                <td colspan="2" style="padding:8px 0; color:#718096; font-size:13px;">
                                                    <strong style="color:#4a5568;">Source:</strong> {entry['source']}
                                                </td>
                                            </tr>
                                            <tr>
                                                <td colspan="2" style="padding:12px 0 0 0; border-top:1px solid #e2e8f0;">
                                                    <div style="display:flex; flex-wrap:wrap; gap:6px; align-items:center;">
                                                        <span style="color:#718096; font-size:12px; font-weight:600;">Keywords:</span>
                                                        <div style="display:flex; flex-wrap:wrap; gap:6px;">
                                                            {''.join([f'<span style="display:inline-block; padding:4px 10px; background-color:#edf2f7; color:#2d3748; font-size:12px; border-radius:6px; font-weight:500;">{kw.strip()}</span>' for kw in entry['keywords'].split(',')[:5] if kw.strip()])}
                                                        </div>
                                                    </div>
                                                </td>
                                            </tr>
                                        </table>
                                    </div>
                                    
                                    <!-- CTA Button -->
                                    <div style="padding:0 24px 24px 24px;">
                                        <a href="{entry['link']}" style="display:inline-block; width:100%; padding:14px 24px; background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); color:#ffffff; text-decoration:none; border-radius:8px; font-weight:600; font-size:15px; text-align:center; transition:opacity 0.2s; box-shadow:0 2px 4px rgba(102, 126, 234, 0.3);" target="_blank">View Full Details</a>
            </div>
        </div>
                            </td>
                        </tr>
        """
        body_sections.append(section)

    footer = f"""
                        <!-- Footer -->
                        <tr>
                            <td style="padding:30px; background-color:#f8fafc; border-top:1px solid #e2e8f0; text-align:center;">
                                <p style="margin:0; color:#718096; font-size:13px; line-height:1.6;">
                                    You are receiving this email because you subscribed to daily tender updates.<br>
                                    <a href="#" style="color:#667eea; text-decoration:none;">Manage preferences</a> | <a href="#" style="color:#667eea; text-decoration:none;">Unsubscribe</a>
                                </p>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
    </body>
    </html>
    """

    return header + "".join(body_sections) + footer


def _build_no_matches_email_html(client_name: str, keywords: List[str], uk_date: datetime) -> str:
    """Build an email for users whose keywords matched no tenders."""
    keywords_html = ""
    if keywords:
        keywords_html = "".join([
            f'<span style="display:inline-block; padding:6px 14px; background-color:#fee2e2; color:#991b1b; font-size:13px; border-radius:8px; font-weight:500; margin:4px;">{kw}</span>'
            for kw in keywords[:10]
        ])
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body style="margin:0; padding:0; background-color:#f5f7fa; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;">
        <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#f5f7fa; padding:40px 20px;">
            <tr>
                <td align="center">
                    <table width="600" cellpadding="0" cellspacing="0" style="max-width:600px; background-color:#ffffff; border-radius:12px; overflow:hidden; box-shadow:0 4px 6px rgba(0,0,0,0.1);">
                        <!-- Header with gradient -->
                        <tr>
                            <td style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding:40px 30px; text-align:center;">
                                <h1 style="margin:0; color:#ffffff; font-size:28px; font-weight:700; letter-spacing:-0.5px;">Daily Tender Digest</h1>
                                <p style="margin:12px 0 0 0; color:#ffffff; font-size:16px; opacity:0.95;">{uk_date.strftime('%d %B %Y')}</p>
                            </td>
                        </tr>
                        <!-- Content -->
                        <tr>
                            <td style="padding:40px 30px;">
                                <p style="margin:0; color:#1a202c; font-size:16px; line-height:1.6;">Hello <strong>{client_name or 'there'}</strong>,</p>
                                
                                <div style="margin:24px 0; padding:24px; background-color:#fef2f2; border-radius:10px; border-left:4px solid #dc2626;">
                                    <h2 style="margin:0 0 12px 0; color:#991b1b; font-size:18px; font-weight:600;">No Matches Found Today</h2>
                                    <p style="margin:0; color:#7f1d1d; font-size:15px; line-height:1.6;">
                                        Unfortunately, no new tenders matched your current keywords today. This could mean:
                                    </p>
                                    <ul style="margin:12px 0 0 0; padding-left:20px; color:#7f1d1d; font-size:14px; line-height:1.8;">
                                        <li>Your keywords are too specific</li>
                                        <li>No relevant tenders were published today</li>
                                        <li>Your keywords might need updating</li>
                                    </ul>
                                </div>
                                
                                <div style="margin:24px 0;">
                                    <p style="margin:0 0 12px 0; color:#4a5568; font-size:14px; font-weight:600;">Your current keywords:</p>
                                    <div style="display:flex; flex-wrap:wrap; gap:8px;">
                                        {keywords_html if keywords_html else '<span style="color:#718096; font-style:italic;">No keywords set</span>'}
                                    </div>
                                </div>
                                
                                <div style="margin:32px 0 0 0; text-align:center;">
                                    <a href="{FRONTEND_BASE}/tenders" style="display:inline-block; padding:14px 32px; background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); color:#ffffff; text-decoration:none; border-radius:8px; font-weight:600; font-size:15px; box-shadow:0 2px 4px rgba(102, 126, 234, 0.3);">Update Your Keywords</a>
                                </div>
                                
                                <p style="margin:24px 0 0 0; color:#718096; font-size:14px; line-height:1.6;">
                                    <strong>Tip:</strong> Try using broader keywords or industry terms to increase your matches. You can also add multiple keyword variations.
                                </p>
                            </td>
                        </tr>
                        <!-- Footer -->
                        <tr>
                            <td style="padding:30px; background-color:#f8fafc; border-top:1px solid #e2e8f0; text-align:center;">
                                <p style="margin:0; color:#718096; font-size:13px; line-height:1.6;">
                                    You are receiving this email because you subscribed to daily tender updates.<br>
                                    <a href="#" style="color:#667eea; text-decoration:none;">Manage preferences</a> | <a href="#" style="color:#667eea; text-decoration:none;">Unsubscribe</a>
                                </p>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
    </body>
    </html>
    """


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
        .select("id, name, contact_email, subscription_tier, subscription_status")
        .in_("id", list(eligible_clients))
        .execute()
    )
    clients = clients_resp.data or []

    for client in clients:
        client_id = client.get("id")
        email = (client.get("contact_email") or "").strip()
        
        # Check subscription status and tier
        sub_status = (client.get("subscription_status") or "").lower()
        sub_tier = (client.get("subscription_tier") or "").lower()
        
        # Only send to active/trialing users with 'tenders' or 'both' tier
        if sub_status not in ("active", "trialing"):
            continue
            
        if sub_tier not in ("tenders", "both"):
            continue

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

            # Filter UK-only tenders if flag is enabled
            # When FILTER_UK_ONLY=1, exclude ALL TED source tenders
            if FILTER_UK_ONLY:
                source = tender.get("source") or ""
                if _is_ted_source(source):
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

            # Check if we have a good email summary already generated
            tender_id = tender.get("id")
            email_summary = None
            if isinstance(metadata, dict):
                email_summary = metadata.get("email_summary")
            
            # If no email summary exists or it's too short, generate one with AI
            full_data = tender.get("full_data") or {}
            if not email_summary or len(email_summary.strip()) < 100:
                print(f"Generating AI email summary for tender {tender_id}")
                tender_data_for_ai = {
                    "title": title,
                    "description": tender.get("description") or tender.get("summary") or "",
                    "category": category_label,
                    "category_label": category_label,
                    "location": location_name,
                    "location_name": location_name,
                    "value_amount": tender.get("value_amount"),
                    "value_currency": tender.get("value_currency"),
                    "deadline": tender.get("deadline"),
                    "source": tender.get("source"),
                }
                email_summary = _generate_compelling_email_summary(tender_data_for_ai, full_data)
                
                # Save the generated summary to Supabase metadata
                if tender_id and email_summary:
                    try:
                        current_metadata = metadata.copy() if isinstance(metadata, dict) else {}
                        current_metadata["email_summary"] = email_summary
                        current_metadata["email_summary_generated_at"] = datetime.now(timezone.utc).isoformat()
                        supabase.table("tenders").update({"metadata": current_metadata}).eq("id", tender_id).execute()
                        print(f"Saved AI-generated email summary for tender {tender_id}")
                    except Exception as e:
                        print(f"Warning: Failed to save email summary to database: {e}")
            
            summary_text = email_summary or "This opportunity requires detailed review. View full tender details for complete information."

            deadline = _parse_datetime(tender.get("deadline"))
            # Skip tenders with passed deadlines
            if deadline:
                if deadline < now_utc:
                    continue
                deadline_text = deadline.astimezone(UK_TIMEZONE).strftime("%d %b %Y")
            else:
                deadline_text = "Not specified"

            # Format published date
            # For TED tenders, use ingested_at or created_at if no published_date
            published_date = _parse_datetime(tender.get("published_date"))
            if not published_date:
                # Check metadata for ingested_at (TED extraction date)
                if isinstance(metadata, dict):
                    ingested_at = metadata.get("ingested_at")
                    if ingested_at:
                        published_date = _parse_datetime(ingested_at)
                # Fallback to tender created_at
                if not published_date:
                    published_date = _parse_datetime(tender.get("created_at"))
            
            if published_date:
                published_text = published_date.astimezone(UK_TIMEZONE).strftime("%d %b %Y")
            else:
                published_text = "Not specified"

            # Determine if this is a new tender or updated match
            match_created = _parse_datetime(match.get("created_at"))
            tender_created = _parse_datetime(tender.get("created_at"))
            is_new_tender = tender_created and tender_created >= since_utc
            status_label = "New" if is_new_tender else "Updated Match"

            # Get match score for sorting (but don't include in email)
            match_score = match.get("match_score")
            numeric_score: float = 0.0
            if match_score is not None:
                try:
                    numeric_score = float(match_score)
                except (TypeError, ValueError):
                    numeric_score = 0.0

            entry_payload = {
                    "title": title,
                    "summary": summary_text,
                    "source": tender.get("source") or "Unknown",
                    "deadline": deadline_text,
                "published_date": published_text,
                "status": status_label,
                    "keywords": ", ".join(match.get("matched_keywords") or []) or "Not specified",
                    "link": f"{FRONTEND_BASE}/tenders/{tender.get('id')}",
                "_score": numeric_score,
                }
            entries.append(entry_payload)

        if not entries:
            # No matches found - get the user's keywords to include in the "no matches" email
            keyword_data = next(
                (row for row in (keyword_resp.data or []) if row.get("client_id") == client_id),
                None
            )
            user_keywords = []
            if keyword_data and isinstance(keyword_data.get("keywords"), list):
                user_keywords = [kw for kw in keyword_data["keywords"] if isinstance(kw, str) and kw.strip()]
            
            payloads.append(
                {
                    "client_id": client_id,
                    "client_name": client.get("name") or "",
                    "email": email,
                    "subject": f"Daily Tender Digest — {uk_now.strftime('%d %b %Y')}",
                    "entries": [],
                    "no_matches": True,
                    "keywords": user_keywords,
                }
            )
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
                "no_matches": False,
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

    summary = {"attempted": 0, "sent": 0, "no_matches": 0}

    for payload in payloads:
        email = payload.get("email")
        if not email:
            continue

        summary["attempted"] += 1
        
        # Check if this is a "no matches" email
        if payload.get("no_matches"):
            summary["no_matches"] += 1
            html = _build_no_matches_email_html(
                payload.get("client_name") or "",
                payload.get("keywords") or [],
                prepared_at
            )
        else:
            html = _build_email_html(
                payload.get("client_name") or "",
                payload.get("entries") or [],
                prepared_at
            )
        
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

