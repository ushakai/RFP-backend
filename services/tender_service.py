"""
Tender Service - Tender monitoring and matching
"""
import traceback
import time
from decimal import Decimal
from typing import Optional

import httpx
from postgrest.exceptions import APIError

from config.settings import get_supabase_client, reinitialize_supabase


def _coerce_float(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        cleaned = cleaned.replace(",", "")
        try:
            return float(cleaned)
        except Exception:
            return None
    return None


def _normalize_str_list(value):
    if value is None:
        return []
    result: list[str] = []
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.startswith("{") and cleaned.endswith("}"):
            cleaned = cleaned[1:-1]
        if not cleaned:
            return []
        parts = [part.strip().strip('"').strip("'") for part in cleaned.replace(";", ",").split(",")]
        result = [part.lower() for part in parts if part]
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                result.append(text.lower())
    else:
        text = str(value).strip()
        if text:
            result.append(text.lower())
    return result


def match_tender_against_keywords(tender_data: dict, keyword_set: dict) -> tuple[bool, float, list]:
    """
    Match a tender against a keyword set.
    Returns: (is_match, match_score, matched_keywords)
    """
    title = str(tender_data.get("title") or "").lower()
    description = str(tender_data.get("description") or "").lower()
    summary = str(tender_data.get("summary") or "").lower()
    category_text = str(tender_data.get("category") or "").lower()
    sector_text = str(tender_data.get("sector") or "").lower()
    location_text = str(tender_data.get("location") or "").lower()

    searchable_text = " ".join(filter(None, [title, description, summary, category_text, sector_text, location_text]))

    keywords = _normalize_str_list(keyword_set.get("keywords"))
    match_type = str(keyword_set.get("match_type") or "any").strip().lower()
    if match_type not in {"any", "all"}:
        match_type = "any"

    matched_keywords = [kw for kw in keywords if kw and kw in searchable_text]
    seen = set()
    matched_keywords = [kw for kw in matched_keywords if not (kw in seen or seen.add(kw))]

    if not keywords:
        is_match = False
    elif match_type == "all":
        is_match = len(matched_keywords) == len(keywords)
    else:
        is_match = len(matched_keywords) > 0

    match_score = len(matched_keywords) / len(keywords) if keywords else 0.0

    if is_match:
        keyword_cats = _normalize_str_list(keyword_set.get("categories"))
        if keyword_cats:
            tender_cats = _normalize_str_list(tender_data.get("category"))
            if not set(keyword_cats).intersection(tender_cats):
                is_match = False

        keyword_sectors = _normalize_str_list(keyword_set.get("sectors"))
        if is_match and keyword_sectors:
            tender_sectors = _normalize_str_list(tender_data.get("sector"))
            if not set(keyword_sectors).intersection(tender_sectors):
                is_match = False

        keyword_locations = _normalize_str_list(keyword_set.get("locations"))
        if is_match and keyword_locations:
            if not location_text or not any(loc in location_text for loc in keyword_locations):
                is_match = False

        value_amount = _coerce_float(tender_data.get("value_amount"))
        min_v = _coerce_float(keyword_set.get("min_value"))
        max_v = _coerce_float(keyword_set.get("max_value"))

        if min_v is not None and (value_amount is None or value_amount < min_v):
            is_match = False
        if max_v is not None and (value_amount is None or value_amount > max_v):
            is_match = False

    if not is_match:
        match_score = 0.0

    return is_match, match_score, matched_keywords


def _rematch_for_client_once(client_id: str) -> int:
    supabase = get_supabase_client()

    kw_res = supabase.table("user_tender_keywords").select("*").eq("client_id", client_id).eq("is_active", True).execute()
    keyword_sets = kw_res.data or []
    if not keyword_sets:
        supabase.table("tender_matches").delete().eq("client_id", client_id).execute()
        return 0

    supabase.table("tender_matches").delete().eq("client_id", client_id).execute()

    tenders_res = supabase.table("tenders").select("*").execute()
    tenders = tenders_res.data or []

    to_insert = []
    created = 0

    for tender in tenders:
        tender_data = {
            "title": tender.get("title"),
            "description": tender.get("description"),
            "summary": tender.get("summary"),
            "category": tender.get("category"),
            "sector": tender.get("sector"),
            "location": tender.get("location"),
            "value_amount": tender.get("value_amount"),
        }
        for kw in keyword_sets:
            is_match, score, matched_keywords = match_tender_against_keywords(tender_data, kw)
            if is_match:
                to_insert.append({
                    "tender_id": tender["id"],
                    "client_id": client_id,
                    "keyword_set_id": kw["id"],
                    "match_score": score,
                    "matched_keywords": matched_keywords,
                })
    if not to_insert:
        return 0

    chunk_size = 200
    for idx in range(0, len(to_insert), chunk_size):
        batch = to_insert[idx:idx + chunk_size]
        supabase.table("tender_matches").insert(batch).execute()
        created += len(batch)
    return created


def rematch_for_client(client_id: str) -> int:
    """Recompute matches for all tenders for a specific client with retry logic."""
    last_exc: Optional[Exception] = None
    for attempt in range(3):
        try:
            return _rematch_for_client_once(client_id)
        except (httpx.HTTPError, APIError) as exc:
            last_exc = exc
            print(f"WARNING: rematch_for_client attempt {attempt + 1} failed for {client_id}: {exc}")
            traceback.print_exc()
            reinitialize_supabase()
            time.sleep(0.2 * (attempt + 1))
        except Exception as exc:
            last_exc = exc
            print(f"ERROR: rematch_for_client unexpected failure for {client_id}: {exc}")
            traceback.print_exc()
            break
    if last_exc:
        raise last_exc
    return 0

