"""
Tender monitoring endpoints
"""
import json
import asyncio
import traceback
import importlib
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID
from fastapi import APIRouter, Header, HTTPException, Body
from sse_starlette.sse import EventSourceResponse
from postgrest.exceptions import APIError
import httpx
import httpcore
from utils.auth import get_client_id_from_key
from config.settings import get_supabase_client, reinitialize_supabase, ENABLE_PAYMENT, FILTER_UK_ONLY
from services.tender_service import rematch_for_client
from services.gemini_service import summarize_tender_for_paid_view
from utils.logging_config import get_logger
from utils.cache_manager import (
    get_cached_tenders,
    set_cached_tenders,
    get_cached_matched,
    set_cached_matched,
    get_cached_purchased,
    set_cached_purchased,
    get_cached_keywords,
    set_cached_keywords,
    invalidate_client_caches,
    invalidate_matched_cache,
    invalidate_keywords_cache,
    invalidate_purchased_cache,
)

router = APIRouter()
logger = get_logger(__name__, "app")

# Client notification streams
_client_streams: dict[str, list[asyncio.Queue]] = {}

def notify_client(client_id: str, event_type: str, payload: dict | None = None):
    """Notify connected SSE clients for a specific client_id."""
    queues = _client_streams.get(client_id) or []
    message = {"type": event_type, "data": payload or {}}
    for q in queues:
        try:
            q.put_nowait(message)
        except Exception:
            pass


@router.get("/tenders/stream")
async def tenders_stream(client_key: str):
    """SSE stream for tender match updates. Pass client_key as query param."""
    client_id = get_client_id_from_key(client_key)
    q: asyncio.Queue = asyncio.Queue()
    if client_id not in _client_streams:
        _client_streams[client_id] = []
    _client_streams[client_id].append(q)

    async def gen():
        try:
            # Initial ping
            yield {"event": "ping", "data": "ok"}
            while True:
                msg = await q.get()
                yield {"event": msg.get("type", "message"), "data": json.dumps(msg.get("data", {}))}
        except asyncio.CancelledError:
            pass
        finally:
            try:
                _client_streams[client_id].remove(q)
            except Exception:
                pass

    return EventSourceResponse(gen())


def _is_schema_cache_error(exc: Exception) -> bool:
    if not isinstance(exc, APIError):
        return False
    message = ""
    if exc.args:
        first = exc.args[0]
        if isinstance(first, dict):
            message = first.get("message") or ""
    if not message:
        message = getattr(exc, "message", "") or str(exc)
    message = (message or "").lower()
    return "schema cache" in message or "pgrst002" in message


# Note: Cache functions are imported from utils.cache_manager


def _fetch_client_keyword_set(supabase, client_id: str):
    res = supabase.table("user_tender_keywords")\
        .select("id, client_id, keywords, categories, sectors, locations, min_value, max_value, is_active, created_at, updated_at")\
        .eq("client_id", client_id)\
        .eq("is_active", True)\
        .order("created_at", desc=True)\
        .limit(1)\
        .execute()
    data = res.data or []
    return data[0] if data else None


def _with_supabase_retry(operation, attempts: int = 5, delay: float = 0.2):
    """
    Retry Supabase operations with exponential backoff.
    Handles Windows socket errors (WinError 10035) with longer delays.
    """
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:
            last_exc = exc
            error_msg = str(exc)
            error_type = type(exc).__name__
            
            # Check if this is a network/connection error that should be retried
            # Include HTTP/2 deque mutation errors and protocol errors
            is_retryable = (
                isinstance(exc, (httpx.HTTPError, httpx.ReadError, httpx.ConnectError, httpx.TimeoutException,
                                httpx.RemoteProtocolError, httpx.ProtocolError, httpx.NetworkError,
                                APIError, ConnectionError, OSError, RuntimeError,
                                httpcore.RemoteProtocolError, httpcore.ProtocolError, httpcore.NetworkError)) or
                "ReadError" in error_type or
                "ConnectError" in error_type or
                "RemoteProtocolError" in error_type or
                "ProtocolError" in error_type or
                "NetworkError" in error_type or
                "deque mutated" in error_msg.lower() or
                "deque" in error_msg.lower() or
                "WinError 10035" in error_msg or
                "non-blocking socket" in error_msg.lower() or
                "connection" in error_msg.lower() or
                "connection terminated" in error_msg.lower() or
                "stream" in error_msg.lower()
            )
            
            # Windows socket error - use longer delay
            is_windows_socket_error = "WinError 10035" in error_msg or "non-blocking socket" in error_msg.lower()
            
            if is_retryable and attempt < attempts - 1:
                # Longer delay for Windows socket errors and protocol errors
                base_delay = delay * (2 ** attempt)  # Exponential backoff
                wait_time = base_delay * (3.0 if is_windows_socket_error else 1.5)
                wait_time = min(wait_time, 5.0)  # Cap at 5 seconds
                
                # Only log on first retry to reduce noise
                if attempt == 0:
                    if is_windows_socket_error:
                        print(f"WARNING: Windows socket error detected, retrying in {wait_time:.2f}s...")
                    elif "deque" in error_msg.lower() or "protocol" in error_type.lower():
                        print(f"WARNING: Connection protocol error detected, retrying in {wait_time:.2f}s...")
                    else:
                        print(f"WARNING: Supabase operation failed, retrying in {wait_time:.2f}s...")
                
                time.sleep(wait_time)
                
                # Reinitialize Supabase connection on retry to get fresh connection
                if attempt >= 1:
                    try:
                        reinitialize_supabase()
                        time.sleep(0.1)  # Brief pause after reinit
                    except Exception:
                        pass  # Ignore reinit errors
                continue
            elif not is_retryable:
                # Non-retryable error - raise immediately
                raise exc
            else:
                # Last attempt failed
                if is_windows_socket_error:
                    print(f"ERROR: Supabase operation failed after {attempts} attempts due to Windows socket error")
                else:
                    print(f"ERROR: Supabase operation failed after {attempts} attempts: {error_type}")
                    traceback.print_exc()
                break
    
    if last_exc:
        raise last_exc


def _is_ted_source(value: str | None) -> bool:
    return isinstance(value, str) and "ted" in value.lower()


def _ensure_tender_visible(source: str | None):
    if FILTER_UK_ONLY and _is_ted_source(source):
        raise HTTPException(status_code=404, detail="Tender not available")

def _ingest_recent_tenders():
    """
    DISABLED: Previously triggered full ingestion on keyword updates.
    This caused repeated ingestion cycles and connection pool exhaustion.
    
    Tenders are now ingested on a scheduled basis (daily at 07:00 UK time).
    When keywords are updated, we only run rematch_for_client() which matches
    existing tenders against the new keywords without fetching new data.
    """
    # Intentionally disabled - ingestion should only run on schedule
    # Keyword updates now only trigger rematch_for_client() in background
    pass


def _schedule_tender_refresh():
    """
    DISABLED: No longer triggers ingestion on keyword updates.
    Rematching is handled separately via rematch_for_client().
    """
    # Intentionally disabled - see _ingest_recent_tenders() docstring
    pass


@router.get("/tenders/keywords")
def get_tender_keywords(x_client_key: str | None = Header(default=None, alias="X-Client-Key")):
    """Get the keyword set for the client (single set)."""
    client_id = get_client_id_from_key(x_client_key)
    
    # Check cache first
    cached = get_cached_keywords(client_id)
    if cached is not None:
        return cached
    
    try:
        def fetch_keywords():
            supabase = get_supabase_client()
            return _fetch_client_keyword_set(supabase, client_id)

        record = _with_supabase_retry(fetch_keywords)
        if not record:
            result = {
                "client_id": client_id,
                "keywords": [],
                "categories": [],
                "sectors": [],
                "locations": [],
                "min_value": None,
                "max_value": None,
                "is_active": True,
            }
        else:
            result = record
        
        set_cached_keywords(client_id, result)
        return result
    except Exception as e:
        print(f"Error getting tender keywords: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to get tender keywords")


def _extract_original_url(full_data: Any, source: str, external_id: str) -> str | None:
    """
    Extract the original tender URL from the full_data based on source.
    """
    if not isinstance(full_data, dict):
        return None
    
    # Handle OCDS format (ContractsFinder, Find a Tender)
    if "tender" in full_data or "releases" in full_data or "ocid" in full_data:
        # Try to get URL from OCDS release
        ocid = full_data.get("ocid")
        if ocid:
            if source == "FindATender":
                # Find a Tender URL format: https://www.find-tender.service.gov.uk/Notice/Details/{ocid}
                return f"https://www.find-tender.service.gov.uk/Notice/Details/{ocid}"
            elif source == "ContractsFinder":
                # ContractsFinder URL format: https://www.contractsfinder.service.gov.uk/Notice/{ocid}
                return f"https://www.contractsfinder.service.gov.uk/Notice/{ocid}"
        
        # Try to get URL from tender documents
        tender = full_data.get("tender") or {}
        documents = tender.get("documents") or []
        if documents and isinstance(documents, list):
            for doc in documents:
                if isinstance(doc, dict):
                    url = doc.get("url") or doc.get("uri")
                    if url and isinstance(url, str) and url.startswith("http"):
                        return url
    
    # Handle TED eForms format
    elif full_data.get("format") == "ted_eforms" or "notice_number" in full_data:
        notice_number = full_data.get("notice_number") or full_data.get("publication_id")
        if notice_number:
            # TED URL format: https://ted.europa.eu/udl?uri=TED:NOTICE:{notice_number}:TEXT:EN:HTML
            return f"https://ted.europa.eu/udl?uri=TED:NOTICE:{notice_number}:TEXT:EN:HTML"
        
        # Try documents
        tender_info = full_data.get("tender") or {}
        documents = tender_info.get("documents") or []
        if documents and isinstance(documents, list):
            for doc in documents:
                if isinstance(doc, dict):
                    url = doc.get("url") or doc.get("uri")
                    if url and isinstance(url, str) and url.startswith("http"):
                        return url
    
    # Handle legacy TED format
    elif "Form_Section" in full_data:
        form_section = full_data.get("Form_Section", {})
        for key in form_section.keys():
            if key.startswith("F"):
                ted_data = form_section[key]
                if isinstance(ted_data, dict):
                    # Try to get URL from documents
                    object_contract = ted_data.get("Object_Contract", {})
                    if isinstance(object_contract, list):
                        object_contract = object_contract[0] if object_contract else {}
                    
                    doc_fields = ["Document_Full", "URL_Document", "URL_Participation"]
                    for field_name in doc_fields:
                        doc_ref = object_contract.get(field_name) or {}
                        if isinstance(doc_ref, dict):
                            url = doc_ref.get("URL") or doc_ref.get("Value")
                            if url and isinstance(url, str) and url.startswith("http"):
                                return url
                        elif isinstance(doc_ref, str) and doc_ref.startswith("http"):
                            return doc_ref
    
    # Handle SAM.gov format
    elif "noticeId" in full_data or "solicitationNumber" in full_data:
        notice_id = full_data.get("noticeId") or full_data.get("solicitationNumber")
        if notice_id:
            return f"https://sam.gov/opp/{notice_id}/view"
    
    # Try to find any URL in the data structure
    def find_url_recursive(obj, depth=0):
        if depth > 5:  # Limit recursion depth
            return None
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(key, str) and ("url" in key.lower() or "uri" in key.lower() or "link" in key.lower()):
                    if isinstance(value, str) and value.startswith("http"):
                        return value
                result = find_url_recursive(value, depth + 1)
                if result:
                    return result
        elif isinstance(obj, list):
            for item in obj:
                result = find_url_recursive(item, depth + 1)
                if result:
                    return result
        return None
    
    return find_url_recursive(full_data)


def _normalize_full_data_for_display(full_data: Any, source: str) -> dict:
    """
    Normalize full_data from different sources (OCDS, TED, SAM.gov) into a consistent structure
    for frontend display.
    """
    if not isinstance(full_data, dict):
        return {"raw": full_data, "source": source}
    
    normalized = {
        "source": source,
        "raw": full_data,  # Keep original for reference
    }
    
    # Handle OCDS format (ContractsFinder, Find a Tender)
    if "tender" in full_data or "releases" in full_data or "ocid" in full_data:
        tender = full_data.get("tender") or {}
        buyer = full_data.get("buyer") or {}
        
        normalized["tender"] = {
            "title": tender.get("title"),
            "description": tender.get("description"),
            "items": tender.get("items") or [],
            "documents": tender.get("documents") or [],
            "value": tender.get("value"),
            "tenderPeriod": tender.get("tenderPeriod"),
        }
        normalized["buyer"] = {
            "name": buyer.get("name"),
            "address": buyer.get("address"),
            "contactPoint": buyer.get("contactPoint"),
        }
        normalized["format"] = "ocds"
    
    # Handle TED eForms format (ContractNotice UBL 2.3)
    elif full_data.get("format") == "ted_eforms":
        tender_info = full_data.get("tender") or {}
        buyer_info = full_data.get("buyer") or {}
        location_info = full_data.get("location") or {}

        cpv_codes = tender_info.get("cpv_codes") or full_data.get("cpv_codes") or []
        items = []
        for code in cpv_codes:
            if not code:
                continue
            items.append(
                {
                    "description": None,
                    "classification": {
                        "id": code,
                        "scheme": "CPV",
                    },
                }
            )

        normalized["tender"] = {
            "title": tender_info.get("title") or full_data.get("notice_number"),
            "description": tender_info.get("description"),
            "documents": tender_info.get("documents") or [],
            "items": items,
            "deadline": tender_info.get("deadline"),
            "cpv_codes": cpv_codes,
        }
        normalized["buyer"] = {
            "name": buyer_info.get("name"),
            "address": buyer_info.get("address"),
            "contactPoint": buyer_info.get("contactPoint"),
        }
        normalized["location"] = location_info
        normalized["language"] = full_data.get("language")
        normalized["format"] = "ted_eforms"

    # Handle legacy TED format (Sell2Wales, PCS Scotland legacy)
    elif "Form_Section" in full_data:
        form_section = full_data.get("Form_Section", {})
        ted_data = None
        for key in form_section.keys():
            if key.startswith("F"):
                ted_data = form_section[key]
                break
        
        if ted_data:
            contracting_body = ted_data.get("Contracting_Body", {})
            address_cb = contracting_body.get("Address_Contracting_Body", {})
            object_contract = ted_data.get("Object_Contract", {})
            if isinstance(object_contract, list):
                object_contract = object_contract[0] if object_contract else {}
            
            # Extract documents from TED format
            documents = []
            # Try multiple document fields
            doc_fields = [
                ("Document_Full", "Full Tender Document"),
                ("URL_Document", "Tender Document"),
                ("URL_Participation", "Participation Link"),
            ]
            for field_name, default_title in doc_fields:
                doc_ref = object_contract.get(field_name) or contracting_body.get(field_name) or {}
                if isinstance(doc_ref, dict):
                    url = doc_ref.get("URL") or doc_ref.get("Value")
                    if url:
                        documents.append({
                            "title": doc_ref.get("Title") or default_title,
                            "url": url,
                        })
                elif isinstance(doc_ref, str) and doc_ref.startswith("http"):
                    documents.append({
                        "title": default_title,
                        "url": doc_ref,
                    })
            
            # Extract items/CPV codes
            items = []
            # Main CPV code
            cpv_main = object_contract.get("CPV_Main", {})
            if cpv_main:
                cpv_code = cpv_main.get("CPV_Code", {})
                if isinstance(cpv_code, dict):
                    items.append({
                        "description": cpv_code.get("Description") or cpv_code.get("Code") or "",
                        "classification": {
                            "id": cpv_code.get("Code"),
                            "description": cpv_code.get("Description"),
                        }
                    })
            
            # Additional CPV codes
            cpv_additional = object_contract.get("CPV_Additional") or []
            if isinstance(cpv_additional, list):
                for cpv_add in cpv_additional:
                    if isinstance(cpv_add, dict):
                        cpv_code_add = cpv_add.get("CPV_Code", {})
                        if isinstance(cpv_code_add, dict):
                            items.append({
                                "description": cpv_code_add.get("Description") or cpv_code_add.get("Code") or "",
                                "classification": {
                                    "id": cpv_code_add.get("Code"),
                                    "description": cpv_code_add.get("Description"),
                                }
                            })
            
            normalized["tender"] = {
                "title": object_contract.get("Title", {}).get("P") if isinstance(object_contract.get("Title"), dict) else object_contract.get("Title"),
                "description": object_contract.get("Short_Descr", {}).get("P") if isinstance(object_contract.get("Short_Descr"), dict) else object_contract.get("Short_Descr"),
                "items": items,
                "documents": documents,
                "value": object_contract.get("Val_Total") or object_contract.get("Val_Estimated_Total"),
            }
            normalized["buyer"] = {
                "name": address_cb.get("OfficialName"),
                "address": {
                    "streetAddress": address_cb.get("Address"),
                    "locality": address_cb.get("Town"),
                    "postalCode": address_cb.get("Postal_Code"),
                    "countryName": address_cb.get("Country", {}).get("Value") if isinstance(address_cb.get("Country"), dict) else None,
                },
                "contactPoint": {
                    "name": address_cb.get("Contact_Point"),
                    "email": address_cb.get("E_Mail"),
                    "telephone": address_cb.get("Phone"),
                },
            }
            normalized["format"] = "ted"
    
    # Handle SAM.gov format
    elif "noticeId" in full_data or "solicitationNumber" in full_data:
        normalized["tender"] = {
            "title": full_data.get("title"),
            "description": full_data.get("description") or full_data.get("fulldescription"),
            "items": [],
            "documents": [],
            "value": {
                "amount": full_data.get("baseAndAllOptionsValue") or full_data.get("baseValue"),
                "currency": full_data.get("currency") or "USD",
            },
        }
        normalized["buyer"] = {
            "name": full_data.get("organizationType"),
            "address": {
                "locality": (full_data.get("placeOfPerformance") or {}).get("city") if isinstance(full_data.get("placeOfPerformance"), dict) else None,
                "region": (full_data.get("placeOfPerformance") or {}).get("state") if isinstance(full_data.get("placeOfPerformance"), dict) else None,
            },
        }
        normalized["format"] = "sam_gov"
    
    else:
        # Unknown format - return as-is with metadata
        normalized["format"] = "unknown"
        normalized["raw"] = full_data
    
    return normalized


def _normalize_keyword_payload(client_id: str, payload: dict) -> dict:
    keywords = payload.get("keywords") or []
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",") if k.strip()]
    if not isinstance(keywords, list):
        raise HTTPException(status_code=400, detail="Invalid keywords payload")
    
    # Normalize locations
    locations = payload.get("locations") or []
    if isinstance(locations, str):
        locations = [l.strip() for l in locations.split(",") if l.strip()]
    if not isinstance(locations, list):
        locations = []
    
    # Normalize industries (using sectors field)
    industries = payload.get("industries") or payload.get("sectors") or []
    if isinstance(industries, str):
        industries = [i.strip() for i in industries.split(",") if i.strip()]
    if not isinstance(industries, list):
        industries = []
    
    return {
        "client_id": client_id,
        "keywords": keywords,
        "match_type": "any",  # legacy column retained but unused
        "categories": payload.get("categories") or None,
        "sectors": industries if industries else None,  # Using sectors field for industries
        "min_value": payload.get("min_value"),
        "max_value": payload.get("max_value"),
        "locations": locations if locations else None,
        "is_active": True,
    }
    

def _enrich_title(title: str | None, category_label: str | None, category: str | None, location_name: str | None, location: str | None) -> str:
    base = (title or "").strip()
    if base and base.lower() != "ted tender notice":
        return base
    descriptor = category_label or (category and f"CPV {category}") or "Public sector"
    geo = location_name or location or "Europe"
    return f"{descriptor.capitalize()} opportunity in {geo}"


def _format_compact_currency(amount: Any, currency: str | None) -> str | None:
    try:
        if amount is None:
            return None
        value = float(amount)
    except (TypeError, ValueError):
        return None
    iso_currency = (currency or "GBP").upper()
    if value >= 1_000_000:
        number = f"{value / 1_000_000:.1f}M"
    elif value >= 1_000:
        number = f"{value / 1_000:.1f}K"
    else:
        number = f"{value:,.0f}"
    number = number.rstrip("0").rstrip(".") if "." in number else number
    return f"{iso_currency} {number}"


def _truncate_text(text: str | None, limit: int = 320) -> str:
    if not text:
        return ""
    clean = text.strip()
    if len(clean) <= limit:
        return clean
    truncated = clean[:limit].rsplit(" ", 1)[0]
    return truncated.rstrip(",.; ") + "..."


def _derive_summary_preview(tender_row: dict) -> str:
    if not tender_row:
        return ""
    metadata = tender_row.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    base_summary = (tender_row.get("summary") or "").strip()
    detail_sources = [
        tender_row.get("description"),
        metadata.get("processed_details", {}).get("executive_summary")
        if isinstance(metadata.get("processed_details"), dict)
        else None,
    ]
    detail_text = ""
    for candidate in detail_sources:
        if isinstance(candidate, str) and candidate.strip():
            detail_text = _truncate_text(candidate, 360)
            break

    if base_summary:
        intro = base_summary
    else:
        location_name = metadata.get("location_name") or tender_row.get("location")
        category_label = metadata.get("category_label") or tender_row.get("category")
        clauses: list[str] = []
        if category_label and location_name:
            clauses.append(f"{category_label} opportunity in {location_name}")
        elif category_label:
            clauses.append(f"{category_label} opportunity")
        elif location_name:
            clauses.append(f"Public sector opportunity in {location_name}")
        else:
            clauses.append("Public sector opportunity")

        deadline_text = None
        deadline_value = tender_row.get("deadline")
        if deadline_value:
            try:
                parsed = datetime.fromisoformat(str(deadline_value).replace("Z", "+00:00"))
                deadline_text = parsed.strftime("%d %b %Y")
            except Exception:
                deadline_text = str(deadline_value)
        if deadline_text:
            clauses.append(f"Deadline {deadline_text}")

        value_text = _format_compact_currency(tender_row.get("value_amount"), tender_row.get("value_currency"))
        if value_text:
            clauses.append(f"Value approx {value_text}")

        intro = ". ".join(clauses).strip()
        if intro and not intro.endswith("."):
            intro += "."

    if detail_text:
        return f"{intro} {detail_text}".strip()
    return intro.strip()


@router.post("/tenders/keywords")
def upsert_tender_keyword(
    payload: dict = Body(...),
    x_client_key: str | None = Header(default=None, alias="X-Client-Key")
):
    """Create or replace the client's keyword set."""
    client_id = get_client_id_from_key(x_client_key)
    keyword_data = _normalize_keyword_payload(client_id, payload)
    
    # Log all filter criteria for debugging
    logger.info(
        f"POST /tenders/keywords - client_id={client_id[:8]}... "
        f"keywords={keyword_data.get('keywords')}, "
        f"locations={keyword_data.get('locations')}, "
        f"sectors/industries={keyword_data.get('sectors')}"
    )

    # Require at least one keyword (location and industry are optional filters)
    has_keywords = keyword_data.get("keywords") and len(keyword_data["keywords"]) > 0
    
    if not has_keywords:
        raise HTTPException(status_code=400, detail="At least one keyword is required")
    
    try:
        existing = _with_supabase_retry(
            lambda: _fetch_client_keyword_set(get_supabase_client(), client_id)
        )

        if existing:
            res = _with_supabase_retry(
                lambda: get_supabase_client()
                .table("user_tender_keywords")
                .update(keyword_data)
                .eq("id", existing["id"])
                .execute()
            )
        else:
            res = _with_supabase_retry(
                lambda: get_supabase_client().table("user_tender_keywords").insert(keyword_data).execute()
            )

        if res.data:
            _schedule_tender_refresh()
            # Invalidate caches
            invalidate_client_caches(client_id)
            
            # Run rematching SYNCHRONOUSLY so the frontend can show loading state
            # This is faster than before because AI enhancement is disabled during rematch
            try:
                logger.info(f"Starting synchronous rematch for client {client_id[:8]}...")
                created = rematch_for_client(client_id)
                logger.info(f"Rematch complete: {created} matches created")
                # Notify connected clients about the update
                notify_client(client_id, "matches-updated", {"reason": "keywords-upserted"})
            except Exception as e:
                logger.error(f"Error in rematch for client {client_id}: {e}")
                traceback.print_exc()
            
            return res.data[0]
        raise HTTPException(status_code=500, detail="Failed to store tender keywords")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error storing tender keyword set: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to save tender keywords")


@router.put("/tenders/keywords/{keyword_id}")
def update_tender_keyword(
    keyword_id: str,
    payload: dict = Body(...),
    x_client_key: str | None = Header(default=None, alias="X-Client-Key")
):
    """Update the client's keyword set (legacy endpoint, maintains single-set guarantee)."""
    client_id = get_client_id_from_key(x_client_key)
    payload = dict(payload or {})
    payload.setdefault("client_id", client_id)

    keyword_data = _normalize_keyword_payload(client_id, payload)
    keyword_data["updated_at"] = datetime.now().isoformat()

    # Require at least one keyword (location and industry are optional filters)
    has_keywords = keyword_data.get("keywords") and len(keyword_data["keywords"]) > 0
    if not has_keywords:
        raise HTTPException(status_code=400, detail="At least one keyword is required")

    try:
        existing = _with_supabase_retry(
            lambda: _fetch_client_keyword_set(get_supabase_client(), client_id)
        )
        if not existing:
            raise HTTPException(status_code=404, detail="Keyword set not found")

        res = _with_supabase_retry(
            lambda: get_supabase_client()
            .table("user_tender_keywords")
            .update(keyword_data)
            .eq("id", existing["id"])
            .execute()
        )

        if not res.data:
            raise HTTPException(status_code=404, detail="Keyword set not found")

        _schedule_tender_refresh()

        # Invalidate caches
        invalidate_client_caches(client_id)
        
        # Run rematching SYNCHRONOUSLY so the frontend can show loading state
        try:
            logger.info(f"Starting synchronous rematch for client {client_id[:8]}... (PUT)")
            created = rematch_for_client(client_id)
            logger.info(f"Rematch complete (PUT): {created} matches created")
            notify_client(client_id, "matches-updated", {"reason": "keywords-updated"})
        except Exception as e:
            logger.error(f"Error in rematch for client {client_id}: {e}")
            traceback.print_exc()
        
        return res.data[0]
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error updating tender keyword: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to update tender keyword")


@router.delete("/tenders/keywords/{keyword_id}")
def delete_tender_keyword(
    keyword_id: str,
    x_client_key: str | None = Header(default=None, alias="X-Client-Key")
):
    """Delete the client's keyword set (ignores keyword_id for compatibility)."""
    client_id = get_client_id_from_key(x_client_key)
    try:
        existing = _with_supabase_retry(
            lambda: _fetch_client_keyword_set(get_supabase_client(), client_id)
        )
        if existing:
            _with_supabase_retry(
                lambda: get_supabase_client().table("user_tender_keywords").delete().eq("id", existing["id"]).execute()
            )
        
        # Invalidate caches
        invalidate_client_caches(client_id)
        # Run rematching in background thread to avoid blocking the response
        def _rematch_in_background():
            try:
                rematch_for_client(client_id)
                notify_client(client_id, "matches-updated", {"reason": "keywords-deleted"})
            except Exception as e:
                print(f"Error in background rematch for client {client_id}: {e}")
                traceback.print_exc()
        
        thread = threading.Thread(target=_rematch_in_background, daemon=True)
        thread.start()
        return {"ok": True}
    except Exception as e:
        print(f"Error deleting tender keyword: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to delete tender keyword")


@router.get("/tenders")
def get_all_tenders(x_client_key: str | None = Header(default=None, alias="X-Client-Key")):
    """Get all active tenders (includes matched status for the client)."""
    client_id = get_client_id_from_key(x_client_key)
    now_utc = datetime.now(timezone.utc)
    lookback = (now_utc - timedelta(days=30)).isoformat()
    now_iso = now_utc.isoformat()

    def build_response():
        def fetch_tenders():
            supabase = get_supabase_client()
            query = supabase.table("tenders").select(
                "id, title, summary, description, source, deadline, published_date, value_amount, value_currency, location, category, metadata"
            ).gte("published_date", lookback).eq("is_duplicate", False).order("published_date", desc=True)
            query = query.or_(f"deadline.is.null,deadline.gte.{now_iso}")
            if FILTER_UK_ONLY:
                query = query.not_.ilike("source", "%ted%")
            return query.execute()

        tenders_res = _with_supabase_retry(fetch_tenders)
        tenders = tenders_res.data or []

        def fetch_matches():
            supabase = get_supabase_client()
            return supabase.table("tender_matches").select(
                "id, tender_id, match_score, matched_keywords, created_at"
            ).eq("client_id", client_id).execute()

        matches_res = _with_supabase_retry(fetch_matches)
        matches = matches_res.data or []

        def fetch_access():
            supabase = get_supabase_client()
            return supabase.table("tender_access").select(
                "tender_id, payment_status, access_granted_at"
            ).eq("client_id", client_id).eq("payment_status", "completed").execute()

        access_res = _with_supabase_retry(fetch_access)
        access_map = {row["tender_id"]: True for row in (access_res.data or [])}

        match_map: dict[str, dict] = {}
        for match in matches:
            tender_id = match.get("tender_id")
            if not tender_id:
                continue
            match_map[tender_id] = match

        def build_cards(enforce_uk_filter: bool) -> list[dict]:
            cards: list[dict] = []
            seen_tenders: set[str] = set()
            for tender in tenders:
                tender_id = tender.get("id")
                if not tender_id:
                    continue
                if tender_id in seen_tenders:
                    continue
                seen_tenders.add(tender_id)

                if enforce_uk_filter and FILTER_UK_ONLY and _is_ted_source(tender.get("source")):
                    continue

                metadata = tender.get("metadata") or {}
                if not isinstance(metadata, dict):
                    metadata = {}
                location_name = metadata.get("location_name") or tender.get("location")
                category_label = metadata.get("category_label") or tender.get("category")
                title = _enrich_title(
                    tender.get("title"),
                    category_label,
                    tender.get("category"),
                    location_name,
                    tender.get("location"),
                )

                match = match_map.get(tender_id)
                match_score = 0.0
                matched_keywords: list[str] = []
                match_id = None

                if match:
                    try:
                        match_score = float(match.get("match_score") or 0.0)
                    except (TypeError, ValueError):
                        match_score = 0.0
                    mk = match.get("matched_keywords")
                    if isinstance(mk, list):
                        matched_keywords = mk
                    match_id = match.get("id")

                summary_preview = _derive_summary_preview({**tender, "metadata": metadata})

                cards.append(
                    {
                        "id": tender_id,
                        "match_id": match_id,
                        "title": title,
                        "summary": tender.get("summary", ""),
                        "summary_preview": summary_preview,
                        "description": tender.get("description", ""),
                        "source": tender.get("source", ""),
                        "deadline": tender.get("deadline"),
                        "published_date": tender.get("published_date"),
                        "value_amount": tender.get("value_amount"),
                        "value_currency": tender.get("value_currency"),
                        "location": tender.get("location"),
                        "location_name": location_name,
                        "category": tender.get("category"),
                        "category_label": category_label,
                        "match_score": match_score,
                        "matched_keywords": matched_keywords,
                        "is_matched": bool(match),
                        "has_access": access_map.get(tender_id, False),
                    }
                )
            return cards

        return build_cards(enforce_uk_filter=True)

    try:
        # Check cache first
        cached = get_cached_tenders(client_id)
        if cached is not None:
            return cached
        
        results = build_response()
        set_cached_tenders(client_id, results)
        return results
    except HTTPException:
        raise
    except APIError as e:
        if _is_schema_cache_error(e):
            cached = get_cached_tenders(client_id)
            if cached is not None:
                print("WARNING: Returning cached tender data due to Supabase schema cache error.")
                return cached
            time.sleep(1.0)
            try:
                reinitialize_supabase()
            except Exception:
                pass
            try:
                results = build_response()
                set_cached_tenders(client_id, results)
                return results
            except APIError:
                pass
        detail = getattr(e, "message", None) or "Failed to fetch tenders"
        raise HTTPException(status_code=503 if _is_schema_cache_error(e) else 500, detail=detail)
    except Exception as e:
        print(f"Error getting all tenders: {e}")
        traceback.print_exc()
        # Try to return cached data if available, otherwise return empty list
        cached = get_cached_tenders(client_id)
        if cached is not None:
            print("WARNING: Returning cached tender data due to error.")
            return cached
        # Return empty list instead of crashing - graceful degradation for production
        return []


@router.get("/tenders/purchased")
def get_purchased_tenders(x_client_key: str | None = Header(default=None, alias="X-Client-Key")):
    client_id = get_client_id_from_key(x_client_key)

    # Check cache first
    cached = get_cached_purchased(client_id)
    if cached is not None:
        return cached

    try:
        def fetch_access():
            supabase = get_supabase_client()
            return (
                supabase.table("tender_access")
                .select("tender_id")
                .eq("client_id", client_id)
                .eq("payment_status", "completed")
                .execute()
            )

        access_res = _with_supabase_retry(fetch_access)
        tender_ids = [row.get("tender_id") for row in (access_res.data or []) if row.get("tender_id")]

        if not tender_ids:
            return []

        def fetch_tenders():
            supabase = get_supabase_client()
            query = (
                supabase.table("tenders")
                .select(
                    "id, title, summary, description, source, deadline, published_date, value_amount, value_currency, location, category, metadata"
                )
                .in_("id", tender_ids)
                .eq("is_duplicate", False)
                .order("published_date", desc=True)
            )
            if FILTER_UK_ONLY:
                query = query.not_.ilike("source", "%ted%")
            return query.execute()

        tender_res = _with_supabase_retry(fetch_tenders)
        tenders = tender_res.data or []

        def fetch_matches():
            supabase = get_supabase_client()
            return (
                supabase.table("tender_matches")
                .select("tender_id, match_score, matched_keywords")
                .eq("client_id", client_id)
                .in_("tender_id", tender_ids)
                .execute()
            )

        matches_res = _with_supabase_retry(fetch_matches)
        match_map: dict[str, dict] = {row["tender_id"]: row for row in (matches_res.data or []) if row.get("tender_id")}

        results = []
        now_utc = datetime.now(timezone.utc)
        for tender in tenders:
            try:
                tender_id = tender.get("id")
                if not tender_id:
                    continue
                if FILTER_UK_ONLY and _is_ted_source(tender.get("source")):
                    continue
                
                # Filter expired tenders
                deadline = tender.get("deadline")
                if deadline:
                    try:
                        deadline_dt = datetime.fromisoformat(str(deadline).replace("Z", "+00:00"))
                        if deadline_dt < now_utc:
                            continue
                    except Exception:
                        pass
                
                metadata = tender.get("metadata") or {}
                if not isinstance(metadata, dict):
                    metadata = {}
                location_name = metadata.get("location_name") or tender.get("location")
                category_label = metadata.get("category_label") or tender.get("category")
                title = _enrich_title(
                    tender.get("title"),
                    category_label,
                    tender.get("category"),
                    location_name,
                    tender.get("location"),
                )

                match = match_map.get(tender_id)
                if match:
                    try:
                        match_score = float(match.get("match_score") or 0.0)
                    except (TypeError, ValueError):
                        match_score = 0.0
                    matched_keywords = match.get("matched_keywords") if isinstance(match.get("matched_keywords"), list) else []
                else:
                    match_score = 0.0
                    matched_keywords = []

                summary_preview = _derive_summary_preview({
                    **tender,
                    "metadata": metadata,
                })

                results.append(
                    {
                        "id": tender_id,
                        "title": title,
                        "summary": tender.get("summary", ""),
                        "summary_preview": summary_preview,
                        "description": tender.get("description", ""),
                        "source": tender.get("source", ""),
                        "deadline": tender.get("deadline"),
                        "published_date": tender.get("published_date"),
                        "value_amount": tender.get("value_amount"),
                        "value_currency": tender.get("value_currency"),
                        "location": tender.get("location"),
                        "location_name": location_name,
                        "category": tender.get("category"),
                        "category_label": category_label,
                        "match_score": match_score,
                        "matched_keywords": matched_keywords,
                        "is_matched": bool(match),
                        "has_access": True,
                    }
                )
            except Exception as e:
                # Skip individual tender errors - don't break entire request
                print(f"Warning: Error processing purchased tender {tender.get('id', 'unknown')}: {e}")
                continue

        results.sort(key=lambda item: (item.get("published_date") or ""), reverse=True)
        set_cached_purchased(client_id, results)
        return results
    except HTTPException:
        raise
    except APIError as e:
        detail = getattr(e, "message", None) or "Failed to fetch purchased tenders"
        raise HTTPException(status_code=500, detail=detail)
    except Exception as e:
        print(f"Error getting purchased tenders: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to get purchased tenders")


@router.get("/tenders/matches")
def get_matched_tenders(x_client_key: str | None = Header(default=None, alias="X-Client-Key")):
    """Get all matched tenders for the client (excludes tenders with passed deadlines)
    
    NOTE: Cache is disabled for matched tenders to ensure fresh results after keyword changes.
    """
    client_id = get_client_id_from_key(x_client_key)
    logger.info(f"GET /tenders/matches - client_id={client_id[:8]}...")
    
    # DISABLED: Always fetch fresh matched tenders (no cache) to ensure keywords changes are reflected immediately
    # cached = get_cached_matched(client_id)
    # if cached is not None:
    #     return cached
    
    try:
        # Fetch all matches with pagination (Supabase limits to 1000 rows per request)
        all_matches = []
        page_size = 1000
        offset = 0
        max_matches = 50000  # Safety limit
        
        while offset < max_matches:
            def fetch_matches_page(off=offset):
                supabase = get_supabase_client()
                return supabase.table("tender_matches").select(
                    "id, tender_id, keyword_set_id, match_score, matched_keywords, created_at, "
                    "tenders(id, title, summary, description, source, deadline, published_date, value_amount, value_currency, location, category, metadata)"
                ).eq("client_id", client_id).order("created_at", desc=True).range(off, off + page_size - 1).execute()

            res = _with_supabase_retry(fetch_matches_page)
            matches = res.data or []
            
            if not matches:
                break
                
            all_matches.extend(matches)
            offset += page_size
            
            if len(matches) < page_size:
                break
        
        if not all_matches:
            return []  # Early return if no matches
        
        def fetch_access():
            supabase = get_supabase_client()
            return supabase.table("tender_access").select("tender_id, payment_status").eq("client_id", client_id).eq("payment_status", "completed").execute()

        access_res = _with_supabase_retry(fetch_access)
        access_map = {acc["tender_id"]: True for acc in (access_res.data or []) if acc.get("tender_id")}
        
        matches = all_matches  # Use all fetched matches
        
        # Filter out tenders with passed deadlines (but keep them in DB)
        now_utc = datetime.now(timezone.utc)
        result = []
        seen_matches: set[str] = set()
        for match in matches:
            tender = match.get("tenders")
            if not tender:
                continue
            tender_id = match.get("tender_id")
            if not tender_id or tender_id in seen_matches:
                continue
            seen_matches.add(tender_id)
            
            if FILTER_UK_ONLY and _is_ted_source(tender.get("source")):
                continue
            
            metadata = tender.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {}
            location_name = metadata.get("location_name") or tender.get("location")
            category_label = metadata.get("category_label") or tender.get("category")
            title = _enrich_title(tender.get("title"), category_label, tender.get("category"), location_name, tender.get("location"))
            
            deadline_value = tender.get("deadline")
            deadline_passed = False
            if deadline_value:
                try:
                    deadline_dt = datetime.fromisoformat(str(deadline_value).replace("Z", "+00:00"))
                    deadline_passed = deadline_dt < now_utc
                except Exception:
                    deadline_passed = False

            summary_preview = _derive_summary_preview({
                **tender,
                "metadata": metadata,
            })

            result.append({
                "id": match["id"],
                "tender_id": tender_id,
                "title": title,
                "summary": tender.get("summary", ""),
                "summary_preview": summary_preview,
                "description": tender.get("description", ""),
                "source": tender.get("source", ""),
                "deadline": tender.get("deadline"),
                "published_date": tender.get("published_date"),
                "value_amount": tender.get("value_amount"),
                "value_currency": tender.get("value_currency"),
                "location": tender.get("location"),
                "location_name": location_name,
                "category": tender.get("category"),
                "category_label": category_label,
                "match_score": match.get("match_score", 0.0),
                "matched_keywords": match.get("matched_keywords", []),
                "has_access": access_map.get(match["tender_id"], False),
                "deadline_passed": deadline_passed,
            })
        
        logger.info(f"GET /tenders/matches - returning {len(result)} matched tenders")
        set_cached_matched(client_id, result)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting matched tenders: {e}")
        traceback.print_exc()
        # Return empty list instead of crashing - graceful degradation for production
        return []


@router.get("/tenders/{tender_id}")
def get_tender_details(
    tender_id: str,
    x_client_key: str | None = Header(default=None, alias="X-Client-Key")
):
    """Get full tender details (requires access)"""
    print(f"=== GET TENDER DETAILS: {tender_id} ===")
    print(f"Received x_client_key: {x_client_key[:8]}..." if x_client_key else "No API key provided")
    
    try:
        client_id = get_client_id_from_key(x_client_key)
        print(f"✓ Authenticated client_id: {client_id}")
    except HTTPException as he:
        print(f"✗ Auth failed: {he.detail}")
        raise
    except Exception as e:
        print(f"✗ Unexpected auth error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Authentication error: {str(e)}")
    
    tender_id = (tender_id or "").strip()
    try:
        UUID(tender_id)
    except ValueError:
        print(f"✗ Invalid UUID: {tender_id}")
        raise HTTPException(status_code=400, detail="Invalid tender identifier")
    
    try:
        print(f"Fetching tender details from database...")
        supabase = get_supabase_client()
        # Check if user has access (only if payment is enabled)
        if ENABLE_PAYMENT:
            access_res = supabase.table("tender_access").select("id, tender_id, payment_status, access_granted_at")\
                .eq("client_id", client_id)\
                .eq("tender_id", tender_id)\
                .eq("payment_status", "completed")\
                .limit(1)\
                .execute()
            access_data = (access_res.data or [])
            if not access_data:
                raise HTTPException(status_code=403, detail="Access denied. Please purchase access to view this tender.")
        
        # Get tender details - include full_data only for detail view
        tender_res = supabase.table("tenders").select(
            "id, title, summary, description, source, deadline, published_date, value_amount, value_currency, "
            "location, category, sector, metadata, full_data, external_id"
        ).eq("id", tender_id).limit(1).execute()
        tender_rows = tender_res.data or []
        if not tender_rows:
            raise HTTPException(status_code=404, detail="Tender not found")
        tender_row = tender_rows[0]
        _ensure_tender_visible(tender_row.get("source"))

        full_payload = tender_row.get("full_data")
        source = tender_row.get("source", "")
        external_id = tender_row.get("external_id", "")
        
        # Extract original URL
        original_url = _extract_original_url(full_payload, source, external_id)
        
        # Normalize full_data structure for consistent frontend access
        normalized_full_data = _normalize_full_data_for_display(full_payload, source)
        
        metadata = tender_row.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        metadata = metadata.copy()
        metadata_updated = False
        update_payload: dict = {}

        if isinstance(full_payload, dict) and "full_payload_keys" not in metadata:
            metadata["full_payload_keys"] = sorted(full_payload.keys())
            metadata_updated = True

        location_name = metadata.get("location_name") or tender_row.get("location")
        category_label = metadata.get("category_label") or tender_row.get("category")

        # Get matched keywords for this tender and client
        matched_keywords = []
        try:
            match_res = supabase.table("tender_matches").select("matched_keywords, match_score").eq("tender_id", tender_id).eq("client_id", client_id).limit(1).execute()
            if match_res.data and len(match_res.data) > 0:
                mk = match_res.data[0].get("matched_keywords")
                if isinstance(mk, list):
                    matched_keywords = mk
        except Exception as e:
            print(f"Warning: Could not fetch matched keywords: {e}")

        processed_details = metadata.get("processed_details")
        if not processed_details:
            tender_core = {
                "id": tender_row.get("id"),
                "title": tender_row.get("title"),
                "summary": tender_row.get("summary"),
                "description": tender_row.get("description"),
                "source": tender_row.get("source"),
                "deadline": tender_row.get("deadline"),
                "published_date": tender_row.get("published_date"),
                "value_amount": tender_row.get("value_amount"),
                "value_currency": tender_row.get("value_currency"),
                "location": location_name or tender_row.get("location"),
                "category": category_label or tender_row.get("category"),
                "sector": tender_row.get("sector"),
            }
            processed_details = summarize_tender_for_paid_view(tender_core, full_payload, matched_keywords=matched_keywords)
            if processed_details:
                metadata["processed_details"] = processed_details
                metadata["processed_details_generated_at"] = datetime.now().isoformat()
                metadata_updated = True

        if metadata_updated:
            update_payload["metadata"] = metadata

        display_title = _enrich_title(tender_row.get("title"), category_label, tender_row.get("category"), location_name, tender_row.get("location"))
        if display_title != tender_row.get("title"):
            tender_row["title"] = display_title
            update_payload["title"] = display_title

        if update_payload:
            supabase.table("tenders").update(update_payload).eq("id", tender_id).execute()

        return {
            "id": tender_row["id"],
            "title": tender_row["title"],
            "description": tender_row["description"],
            "summary": tender_row["summary"],
            "source": tender_row["source"],
            "deadline": tender_row["deadline"],
            "published_date": tender_row["published_date"],
            "value_amount": tender_row["value_amount"],
            "value_currency": tender_row["value_currency"],
            "location": tender_row["location"],
            "location_name": location_name,
            "category": tender_row["category"],
            "category_label": category_label,
            "sector": tender_row["sector"],
            "original_url": original_url,
            "full_data": normalized_full_data,
            "processed_details": processed_details,
            "metadata": metadata,
        }
    except HTTPException:
        raise
    except HTTPException:
        raise
    except Exception as e:
        print(f"✗ Error getting tender details: {e}")
        print(f"Error type: {type(e).__name__}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to get tender details: {str(e)}")


@router.post("/tenders/{tender_id}/access")
def request_tender_access(
    tender_id: str,
    x_client_key: str | None = Header(default=None, alias="X-Client-Key")
):
    """Request access to a tender (creates payment record)"""
    client_id = get_client_id_from_key(x_client_key)
    
    try:
        supabase = get_supabase_client()
        tender_res = supabase.table("tenders").select("id, source").eq("id", tender_id).limit(1).execute()
        tender_rows = tender_res.data or []
        if not tender_rows:
            raise HTTPException(status_code=404, detail="Tender not found")
        _ensure_tender_visible(tender_rows[0].get("source"))

        # Check if access already exists
        existing = supabase.table("tender_access").select("id, tender_id, payment_status").eq("client_id", client_id).eq("tender_id", tender_id).execute()
        
        if existing.data and existing.data[0].get("payment_status") == "completed":
            return {"message": "Access already granted", "access_id": existing.data[0]["id"]}
        
        # Create or update access record
        access_data = {
            "client_id": client_id,
            "tender_id": tender_id,
            "payment_status": "pending",
            "payment_amount": 5.00,
        }
        
        if existing.data:
            # Update existing record
            res = supabase.table("tender_access").update(access_data).eq("id", existing.data[0]["id"]).execute()
            access_id = existing.data[0]["id"]
        else:
            # Create new record
            res = supabase.table("tender_access").insert(access_data).execute()
            access_id = res.data[0]["id"] if res.data else None
        
        return {
            "message": "Access request created. Payment integration required.",
            "access_id": access_id,
            "payment_amount": 5.00,
            "note": "This is a placeholder. Payment processing integration is required."
        }
    except Exception as e:
        print(f"Error requesting tender access: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to request tender access")


@router.post("/tenders/{tender_id}/access/complete")
def complete_tender_access(
    tender_id: str,
    x_client_key: str | None = Header(default=None, alias="X-Client-Key")
):
    """Mark tender access as paid and grant access."""
    client_id = get_client_id_from_key(x_client_key)
    try:
        supabase = get_supabase_client()
        tender_res = supabase.table("tenders").select("id, source").eq("id", tender_id).limit(1).execute()
        tender_rows = tender_res.data or []
        if not tender_rows:
            raise HTTPException(status_code=404, detail="Tender not found")
        _ensure_tender_visible(tender_rows[0].get("source"))

        # Ensure access record exists
        existing = supabase.table("tender_access").select("id, tender_id, payment_status, client_id")\
            .eq("client_id", client_id).eq("tender_id", tender_id).execute()
        if not existing.data:
            # Create pending then complete
            res = supabase.table("tender_access").insert({
                "client_id": client_id,
                "tender_id": tender_id,
                "payment_status": "pending",
                "payment_amount": 5.00,
            }).execute()
            access_id = res.data[0]["id"]
        else:
            access_id = existing.data[0]["id"]
        # Mark as completed
        supabase.table("tender_access").update({
            "payment_status": "completed",
            "payment_date": datetime.now().isoformat(),
            "access_granted_at": datetime.now().isoformat(),
        }).eq("id", access_id).execute()
        
        # Invalidate caches when access is granted
        invalidate_client_caches(client_id)
        
        # Notify client about the update
        notify_client(client_id, "tender-purchased", {"tender_id": tender_id})
        
        return {"ok": True, "access_id": access_id}
    except Exception as e:
        print(f"Error completing tender access: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to complete tender access")


@router.post("/tenders/rematch")
def rematch_client(x_client_key: str | None = Header(default=None, alias="X-Client-Key")):
    """Manual trigger to recompute matches for the calling client."""
    client_id = get_client_id_from_key(x_client_key)
    created = rematch_for_client(client_id)
    notify_client(client_id, "matches-updated", {"reason": "manual-rematch"})
    return {"ok": True, "created": created}

