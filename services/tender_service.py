"""
Tender Service - Tender monitoring and matching
"""
import re
import traceback
import time
from decimal import Decimal
from typing import Optional, Any, Dict
from datetime import datetime, timezone

import httpx
from postgrest.exceptions import APIError

from config.settings import get_supabase_client, reinitialize_supabase, FILTER_UK_ONLY


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


def _extract_location_fields(tender_data: dict) -> str:
    """
    Extract all location-related text from tender data for better location matching.
    Checks multiple fields including metadata and full_data.
    """
    location_parts = []
    
    # Direct location fields
    location = str(tender_data.get("location") or "").strip()
    location_name = str(tender_data.get("location_name") or "").strip()
    
    if location:
        location_parts.append(location.lower())
    if location_name:
        location_parts.append(location_name.lower())
    
    # Check metadata for location information
    metadata = tender_data.get("metadata") or {}
    if isinstance(metadata, dict):
        meta_location = metadata.get("location_name") or metadata.get("location")
        if meta_location:
            location_parts.append(str(meta_location).strip().lower())
    
    # Check full_data for location information (OCDS/TED format)
    full_data = tender_data.get("full_data")
    if isinstance(full_data, dict):
        # Check buyer address
        buyer = full_data.get("buyer") or {}
        if isinstance(buyer, dict):
            address = buyer.get("address") or {}
            if isinstance(address, dict):
                # Extract city, region, locality
                for field in ["locality", "city", "cityName", "region", "regionName", "postalCode"]:
                    value = address.get(field)
                    if value:
                        location_parts.append(str(value).strip().lower())
        
        # Check tender items delivery locations
        tender_info = full_data.get("tender") or {}
        items = tender_info.get("items") or []
        if isinstance(items, list):
            for item in items[:3]:  # Check first 3 items
                if isinstance(item, dict):
                    delivery = item.get("deliveryLocation") or {}
                    if isinstance(delivery, dict):
                        desc = delivery.get("description")
                        if desc:
                            location_parts.append(str(desc).strip().lower())
                        addr = delivery.get("address") or {}
                        if isinstance(addr, dict):
                            for field in ["locality", "city", "cityName", "region", "regionName"]:
                                value = addr.get(field)
                                if value:
                                    location_parts.append(str(value).strip().lower())
        
        # Check TED format location
        location_info = full_data.get("location") or {}
        if isinstance(location_info, dict):
            desc = location_info.get("description")
            if desc:
                location_parts.append(str(desc).strip().lower())
    
    # Combine and deduplicate
    all_locations = " ".join(set(location_parts))
    return all_locations


def match_tender_against_keywords(tender_data: dict, keyword_set: dict, enable_ai: bool = True) -> tuple[bool, float, list]:
    """
    Match a tender against a keyword set.
    Matching logic: (one location OR) AND (one industry OR) AND (one keyword OR)
    Returns: (is_match, match_score, matched_keywords)
    
    Args:
        tender_data: Tender information
        keyword_set: Keywords, locations, industries to match
        enable_ai: If True, use AI to enhance keywords (slower but better recall)
    """
    title = str(tender_data.get("title") or "").lower()
    description = str(tender_data.get("description") or "").lower()  # Include full description for search
    summary = str(tender_data.get("summary") or "").lower()
    category_text = str(tender_data.get("category") or "").lower()
    sector_text = str(tender_data.get("sector") or "").lower()

    # Prefer structured/curated metadata over raw description text to reduce noise.
    metadata = tender_data.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    meta_category_label = str(metadata.get("category_label") or "").lower()

    processed_details = metadata.get("processed_details") or {}
    if not isinstance(processed_details, dict):
        processed_details = {}
    exec_summary = str(processed_details.get("executive_summary") or "").lower()
    scope_summary = str(processed_details.get("scope_summary") or "").lower()

    # Extract comprehensive location information
    location_text = _extract_location_fields(tender_data)

    # Combine curated/searchable text for keyword matching.
    # We include the full raw description because "website" and other key terms might be buried there.
    # We trust the user's specific keywords to filter out noise.
    searchable_text = " ".join(
        filter(
            None,
            [
                title,
                description,  # Added back description
                summary,
                category_text,
                sector_text,
                meta_category_label,
                exec_summary,
                scope_summary,
            ],
        )
    )

    # Get filter criteria
    keywords = _normalize_str_list(keyword_set.get("keywords"))
    keyword_locations = _normalize_str_list(keyword_set.get("locations"))
    keyword_sectors = _normalize_str_list(keyword_set.get("sectors"))  # Using sectors as industries

    # Enhance keywords with AI for better matching (non-blocking, optional)
    # If AI fails, we gracefully fall back to original keywords
    # During bulk rematch operations, AI is disabled for speed
    original_keywords_lower = set(kw.lower() for kw in keywords)
    enhanced_keywords = keywords.copy() if keywords else []
    if keywords and enable_ai:
        try:
            from services.gemini_service import enhance_keywords_with_ai
            # Use AI enhancement with timeout protection
            ai_enhanced = enhance_keywords_with_ai(keywords)
            if ai_enhanced and len(ai_enhanced) > 0:
                # Add AI-enhanced keywords that aren't already in original list
                for ekw in ai_enhanced:
                    if ekw and ekw.lower() not in original_keywords_lower:
                        enhanced_keywords.append(ekw)
        except Exception:
            # Silently fall back to original keywords - AI enhancement is optional
            # Don't log errors here as they're expected and handled gracefully
            pass

    # Check keyword match (OR - at least one keyword must match)
    # Use word boundary matching for better accuracy (match whole words or as substring)
    matched_keywords = []
    if enhanced_keywords:
        for kw in enhanced_keywords:
            if not kw:
                continue
            kw_lower = kw.lower()
            # Try word boundary match first (whole word), then substring match
            # This ensures "website" matches "website" but also "website development"
            # We use a more lenient regex that allows matches even if surrounded by non-alphanumeric chars (except whitespace)
            # or if it's part of a larger word if word boundary fails.
            
            # 1. Exact word match (most reliable)
            pattern = r'\b' + re.escape(kw_lower) + r'\b'
            if re.search(pattern, searchable_text, re.IGNORECASE):
                if kw_lower in original_keywords_lower:
                    matched_keywords.append(kw_lower)
            # 2. Substring match (fallback, necessary for some terms)
            # e.g. "cleaning" matching "cleaning" in "housecleaning" or plurals if simple enough
            elif kw_lower in searchable_text:
                if kw_lower in original_keywords_lower:
                    matched_keywords.append(kw_lower)
        # Remove duplicates while preserving order
        seen = set()
        matched_keywords = [kw for kw in matched_keywords if not (kw in seen or seen.add(kw))]
    keyword_match = len(matched_keywords) > 0 if keywords else False

    # Check location match (OR - at least one location must match if locations are provided)
    # Use word boundary matching for better city/location detection
    # NOTE: If tender has no location data, we still allow it to match (lenient mode)
    # This prevents filtering out tenders that simply don't have location metadata
    location_match = True  # Default to True if no locations specified
    if keyword_locations:
        # If tender has no location data at all, still allow it to match
        # (many tenders don't have location info but are still relevant)
        if not location_text or not location_text.strip():
            location_match = True  # Lenient: allow match when tender has no location data
        else:
            location_match = False
            for loc in keyword_locations:
                if not loc:
                    continue
                # Use word boundary matching for location to avoid partial matches
                # e.g., "manchester" should match "Manchester" but not "manchesterfield"
                pattern = r'\b' + re.escape(loc) + r'\b'
                if re.search(pattern, location_text, re.IGNORECASE):
                    location_match = True
                    break
                # Also try substring match as fallback for compound locations
                elif loc in location_text:
                    location_match = True
                    break

    # Check industry/sector match (OR - at least one industry must match if industries are provided)
    # NOTE: If tender has no sector/category data, we still allow it to match (lenient mode)
    industry_match = True  # Default to True if no industries specified
    if keyword_sectors:
        tender_sectors = _normalize_str_list(tender_data.get("sector"))
        tender_categories = _normalize_str_list(tender_data.get("category"))
        # Check both sector and category fields
        all_tender_industries = set(tender_sectors + tender_categories)
        
        # If tender has no industry data at all, still allow it to match
        # (many tenders don't have sector/category info but are still relevant)
        if not all_tender_industries:
            industry_match = True  # Lenient: allow match when tender has no industry data
        else:
            industry_match = False
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

    # Fetch active tenders with pagination (Supabase limits to 1000 rows per request)
    # Also exclude duplicates and fetch metadata for location_name
    now_utc = datetime.now(timezone.utc).isoformat()
    
    to_insert = []
    created = 0
    page_size = 1000
    offset = 0
    max_tenders = 50000  # Safety limit to prevent infinite loops
    
    while offset < max_tenders:
        tenders_res = (
            supabase.table("tenders")
            .select("id, title, description, summary, category, sector, location, value_amount, metadata, is_duplicate, source")
            .or_(f"deadline.is.null,deadline.gte.{now_utc}")
            .eq("is_duplicate", False)
            .order("published_date", desc=True)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        tenders = tenders_res.data or []
        
        if not tenders:
            break  # No more tenders to process
            
        for tender in tenders:
            # Skip duplicates
            if tender.get("is_duplicate"):
                continue
            
            # Skip TED tenders when FILTER_UK_ONLY is enabled (they'll be filtered in API anyway)
            source = (tender.get("source") or "").lower()
            if FILTER_UK_ONLY and "ted" in source:
                continue
                
            # Extract location_name from metadata
            metadata = tender.get("metadata") or {}
            location_name = None
            if isinstance(metadata, dict):
                location_name = metadata.get("location_name")
            
            # Include metadata in tender_data for keyword matching against structured fields
            # (executive_summary, scope_summary, category_label, etc.)
            tender_data = {
                "title": tender.get("title"),
                "description": tender.get("description"),
                "summary": tender.get("summary"),
                "category": tender.get("category"),
                "sector": tender.get("sector"),
                "location": tender.get("location"),
                "location_name": location_name,
                "value_amount": tender.get("value_amount"),
                "metadata": metadata,  # Include metadata for structured field matching
            }
            for kw in keyword_sets:
                # Disable AI enhancement during bulk rematch for speed (enable_ai=False)
                is_match, score, matched_keywords = match_tender_against_keywords(tender_data, kw, enable_ai=False)
                if is_match:
                    to_insert.append({
                        "tender_id": tender["id"],
                        "client_id": client_id,
                        "keyword_set_id": kw["id"],
                        "match_score": score,
                        "matched_keywords": matched_keywords,
                    })
        
        offset += page_size
        
        # If we got fewer results than page_size, we've reached the end
        if len(tenders) < page_size:
            break
    
    if not to_insert:
        return 0

    # Upsert matches in batches (handles duplicates gracefully)
    chunk_size = 200
    for idx in range(0, len(to_insert), chunk_size):
        batch = to_insert[idx:idx + chunk_size]
        try:
            supabase.table("tender_matches").upsert(
                batch, 
                on_conflict="tender_id,client_id,keyword_set_id"
            ).execute()
            created += len(batch)
        except Exception as e:
            # If upsert fails, try individual inserts with ignore on conflict
            for item in batch:
                try:
                    supabase.table("tender_matches").upsert(
                        item,
                        on_conflict="tender_id,client_id,keyword_set_id"
                    ).execute()
                    created += 1
                except Exception:
                    pass  # Skip duplicates
    return created


def rematch_for_client(client_id: str) -> int:
    """Recompute matches for all tenders for a specific client with retry logic."""
    last_exc: Optional[Exception] = None
    for attempt in range(5):
        try:
            return _rematch_for_client_once(client_id)
        except (httpx.HTTPError, APIError) as exc:
            last_exc = exc
            print(f"WARNING: rematch_for_client attempt {attempt + 1} failed for {client_id}: {exc}")
            reinitialize_supabase()
            # Exponential backoff with cap to avoid hammering Supabase on protocol errors
            delay = min(0.5 * (attempt + 1), 5.0)
            time.sleep(delay)
        except Exception as exc:
            last_exc = exc
            print(f"ERROR: rematch_for_client unexpected failure for {client_id}: {exc}")
            traceback.print_exc()
            break
    if last_exc:
        # For background workers we don't want to crash the process on transient errors.
        # Log a concise error and return 0 so the API can continue serving existing data.
        print(f"ERROR: rematch_for_client giving up for {client_id} after retries: {last_exc}")
        return 0
    return 0

