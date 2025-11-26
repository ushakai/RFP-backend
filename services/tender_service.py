"""
Tender Service - Tender monitoring and matching
"""
import traceback
import time
from decimal import Decimal
from typing import Optional, Any, Dict
from datetime import datetime, timezone

import httpx
from postgrest.exceptions import APIError

from config.settings import get_supabase_client, reinitialize_supabase


def is_uk_specific_source(source: str) -> bool:
    """
    Check if the source is a UK-specific source that always passes UK filter.
    Returns True for: ContractsFinder, Sell2Wales, PCS Scotland, FindATender
    """
    source_lower = (source or "").lower()
    uk_specific_sources = ["contractsfinder", "sell2wales", "pcs scotland", "find a tender", "findatender"]
    return any(uk_source in source_lower for uk_source in uk_specific_sources)


def is_uk_tender(tender_data: Dict[str, Any]) -> bool:
    """
    Check if a tender is from the UK.
    Returns True if the tender is from UK, False otherwise.
    This is a shared utility function used across the application.
    """
    source = (tender_data.get("source") or "").lower()
    
    # UK-specific sources are automatically UK
    uk_sources = [
        "contractsfinder",
        "find a tender",
        "findatender",
        "pcs scotland",
        "sell2wales",
    ]
    if any(uk_source in source for uk_source in uk_sources):
        return True
    
    # Non-UK sources are automatically not UK
    non_uk_sources = [
        "sam.gov",
        "austender",
    ]
    if any(non_uk_source in source for non_uk_source in non_uk_sources):
        return False
    
    # Check metadata for country and NUTS code (from TED parsing)
    metadata = tender_data.get("metadata") or {}
    nuts_code = None
    country = None
    if isinstance(metadata, dict):
        nuts_code = metadata.get("nuts_code")
        country = metadata.get("country")
    
    # Check if NUTS code indicates UK (starts with "UK")
    if nuts_code:
        nuts_str = str(nuts_code).upper()
        if nuts_str.startswith("UK"):
            return True
    
    # Check country from metadata
    if country:
        country_lower = str(country).lower()
        uk_country_names = ["united kingdom", "uk", "gb", "great britain"]
        if any(uk_name in country_lower for uk_name in uk_country_names):
            return True
    
    # Also check in full_data for TED format if country not found in metadata
    if not country:
        full_data = tender_data.get("full_data")
        if isinstance(full_data, dict):
            # Check Form_Section for TED format
            form_section = full_data.get("Form_Section") or {}
            for key in form_section.keys():
                if key.startswith("F"):
                    ted_data = form_section[key]
                    if isinstance(ted_data, dict):
                        contracting_body = ted_data.get("Contracting_Body") or {}
                        address_cb = contracting_body.get("Address_Contracting_Body") or {}
                        country_dict = address_cb.get("Country") or {}
                        if isinstance(country_dict, dict):
                            country = country_dict.get("Value") or country_dict.get("Code") or country_dict.get("P")
                        elif isinstance(country_dict, str):
                            country = country_dict
                        if country:
                            country_lower = str(country).lower()
                            uk_country_names = ["united kingdom", "uk", "gb", "great britain"]
                            if any(uk_name in country_lower for uk_name in uk_country_names):
                                return True
                        break
    
    # For TED and other sources, check location fields
    location = (tender_data.get("location") or "").lower()
    location_name = ""
    if isinstance(metadata, dict):
        location_name = (metadata.get("location_name") or "").lower()
    
    # Check full_data for country information (OCDS format)
    full_data = tender_data.get("full_data")
    country_indicators = []
    if country:
        country_indicators.append(str(country).lower())
    
    if isinstance(full_data, dict):
        # Check buyer address (OCDS format)
        buyer = full_data.get("buyer") or {}
        if isinstance(buyer, dict):
            address = buyer.get("address") or {}
            if isinstance(address, dict):
                ocds_country = (address.get("countryName") or address.get("country") or "").lower()
                if ocds_country:
                    country_indicators.append(ocds_country)
        
        # Check tender items delivery locations
        tender_info = full_data.get("tender") or {}
        items = tender_info.get("items") or []
        if isinstance(items, list):
            for item in items[:3]:  # Check first 3 items
                if isinstance(item, dict):
                    delivery = item.get("deliveryLocation") or {}
                    if isinstance(delivery, dict):
                        item_country = (delivery.get("address", {}).get("countryName") or "").lower()
                        if item_country:
                            country_indicators.append(item_country)
    
    # UK country indicators
    uk_indicators = [
        "united kingdom",
        "uk",
        "gb",
        "great britain",
        "england",
        "scotland",
        "wales",
        "northern ireland",
        "gb-eng",
        "gb-sct",
        "gb-wls",
        "gb-nir",
    ]
    
    # UK city/region indicators (common UK locations that might appear without country)
    uk_cities_regions = [
        "london", "manchester", "birmingham", "glasgow", "edinburgh", "cardiff", "belfast",
        "liverpool", "leeds", "sheffield", "bristol", "newcastle", "nottingham", "southampton",
        "yorkshire", "lancashire", "kent", "essex", "surrey", "hertfordshire", "devon",
        "cornwall", "norfolk", "suffolk", "cumbria", "northumberland", "dorset", "somerset",
    ]
    
    # Check all location fields
    all_location_text = f"{location} {location_name} {' '.join(country_indicators)}".lower()
    
    # Check if any UK indicator is present
    for indicator in uk_indicators:
        if indicator in all_location_text:
            return True
    
    # For TED tenders, also check for UK cities/regions as a fallback
    if "ted" in source:
        # Check for UK cities/regions in location text
        for uk_city in uk_cities_regions:
            if uk_city in all_location_text:
                return True
        # If no UK indicators found at all, exclude
        return False
    
    # Default: if we can't determine, return False (safer to exclude)
    return False


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
    Matching logic: (one location OR) AND (one industry OR) AND (one keyword OR)
    Returns: (is_match, match_score, matched_keywords)
    """
    title = str(tender_data.get("title") or "").lower()
    description = str(tender_data.get("description") or "").lower()
    summary = str(tender_data.get("summary") or "").lower()
    category_text = str(tender_data.get("category") or "").lower()
    sector_text = str(tender_data.get("sector") or "").lower()
    location_text = str(tender_data.get("location") or "").lower()
    location_name = str(tender_data.get("location_name") or "").lower()

    # Combine all searchable text for keyword matching
    searchable_text = " ".join(filter(None, [title, description, summary, category_text, sector_text, location_text, location_name]))

    # Get filter criteria
    keywords = _normalize_str_list(keyword_set.get("keywords"))
    keyword_locations = _normalize_str_list(keyword_set.get("locations"))
    keyword_sectors = _normalize_str_list(keyword_set.get("sectors"))  # Using sectors as industries

    # Check keyword match (OR - at least one keyword must match)
    matched_keywords = []
    if keywords:
        matched_keywords = [kw for kw in keywords if kw and kw in searchable_text]
        seen = set()
        matched_keywords = [kw for kw in matched_keywords if not (kw in seen or seen.add(kw))]
    keyword_match = len(matched_keywords) > 0 if keywords else False

    # Check location match (OR - at least one location must match if locations are provided)
    location_match = True  # Default to True if no locations specified
    if keyword_locations:
        location_match = False
        combined_location = f"{location_text} {location_name}".strip()
        if combined_location:
            for loc in keyword_locations:
                if loc in combined_location:
                    location_match = True
                    break

    # Check industry/sector match (OR - at least one industry must match if industries are provided)
    industry_match = True  # Default to True if no industries specified
    if keyword_sectors:
        industry_match = False
        tender_sectors = _normalize_str_list(tender_data.get("sector"))
        tender_categories = _normalize_str_list(tender_data.get("category"))
        # Check both sector and category fields
        all_tender_industries = set(tender_sectors + tender_categories)
        if all_tender_industries:
            if set(keyword_sectors).intersection(all_tender_industries):
                industry_match = True

    # All three conditions must be true (AND)
    is_match = keyword_match and location_match and industry_match

    # Calculate match score based on keyword matches
    match_score = len(matched_keywords) / len(keywords) if keywords else 0.0

    # Apply value filters if specified
    if is_match:
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

    # Only fetch active tenders (not past deadline) to improve performance
    now_utc = datetime.now(timezone.utc).isoformat()
    tenders_res = (
        supabase.table("tenders")
        .select("id, title, description, summary, category, sector, location, value_amount")
        .or_(f"deadline.is.null,deadline.gte.{now_utc}")
        .order("published_date", desc=True)
        .limit(10000)  # Reasonable limit to prevent excessive processing
        .execute()
    )
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

