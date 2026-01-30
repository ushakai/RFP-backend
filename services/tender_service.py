"""
Tender Service - Tender monitoring and matching
"""
import os
import re
import traceback
import time
from decimal import Decimal
from typing import Optional, Any, Dict
from datetime import datetime, timezone, timedelta

import httpx
from postgrest.exceptions import APIError

from config.settings import get_supabase_client, reinitialize_supabase, FILTER_UK_ONLY

CPV_INDUSTRY_MAP = {
    "03": "Agricultural and Farming Goods",
    "09": "Petroleum and Fuel Products",
    "14": "Mining, Basic Metals and Related Products",
    "15": "Food, Beverages and Tobacco",
    "18": "Clothing and Accessories",
    "30": "Office and Computing Equipment",
    "33": "Medical Equipment",
    "34": "Transport Equipment and Auxiliary Products",
    "35": "Security, Defence and Safety Services",
    "45": "Construction Work",
    "50": "Repair and Maintenance Services",
    "60": "Transport Services",
    "71": "Architecture and Engineering Services",
    "72": "IT Services Software Systems",
    "73": "Research and Consultancy Services",
    "79": "Business Services",
    "80": "Education and Training Services",
    "85": "Health and Social Work Services",
    "90": "Cleaning Environmental and Waste Services",
    "92": "Recreational Cultural and Sporting Services",
    "98": "Miscellaneous Services",
}


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
    # FAST PATH: Use indexed data if available for quick matching
    # This can dramatically speed up bulk matching operations
    indexed_keywords = tender_data.get("indexed_keywords") or []
    indexed_locations = tender_data.get("indexed_locations") or []
    indexed_industries = tender_data.get("indexed_industries") or []
    
    # Get filter criteria (normalized)
    filter_keywords = _normalize_str_list(keyword_set.get("keywords"))
    filter_locations = _normalize_str_list(keyword_set.get("locations"))
    filter_industries = _normalize_str_list(keyword_set.get("sectors"))
    
    # Quick indexed match check (if indexed data available)
    # This is an optimization - we still do full matching below
    has_indexed_data = bool(indexed_keywords or indexed_locations or indexed_industries)
    
    if has_indexed_data and not enable_ai:
        # Fast path: check indexed data directly
        has_keyword_filter = len(filter_keywords) > 0
        has_location_filter = len(filter_locations) > 0
        has_industry_filter = len(filter_industries) > 0
        
        if not has_keyword_filter and not has_location_filter and not has_industry_filter:
            return False, 0.0, []
        
        # Check indexed matches
        indexed_kw_set = set(indexed_keywords)
        indexed_loc_set = set(indexed_locations)
        indexed_ind_set = set(indexed_industries)
        
        keyword_match = False
        matched_kws = []
        if has_keyword_filter:
            for kw in filter_keywords:
                if kw in indexed_kw_set:
                    keyword_match = True
                    matched_kws.append(kw)
            if not keyword_match:
                # Also check if keyword appears as substring in any indexed keyword
                for kw in filter_keywords:
                    for idx_kw in indexed_kw_set:
                        if kw in idx_kw or idx_kw in kw:
                            keyword_match = True
                            matched_kws.append(kw)
                            break
                    if keyword_match:
                        break
        
        location_match = True if not has_location_filter else False
        if has_location_filter:
            for loc in filter_locations:
                if loc in indexed_loc_set:
                    location_match = True
                    break
                # Also check substring match for locations
                for idx_loc in indexed_loc_set:
                    if loc in idx_loc or idx_loc in loc:
                        location_match = True
                        break
                if location_match:
                    break
        
        industry_match = True if not has_industry_filter else False
        if has_industry_filter:
            for ind in filter_industries:
                if ind in indexed_ind_set:
                    industry_match = True
                    break
                # Check CPV prefix match (e.g., "72" matches "72222300")
                if ind.isdigit() and len(ind) <= 3:
                    for idx_ind in indexed_ind_set:
                        if idx_ind.startswith(ind) or ind.startswith(idx_ind):
                            industry_match = True
                            break
                if industry_match:
                    break
        
        # All specified filters must match
        keyword_ok = keyword_match if has_keyword_filter else True
        location_ok = location_match if has_location_filter else True
        industry_ok = industry_match if has_industry_filter else True
        
        is_match = keyword_ok and location_ok and industry_ok
        
        if is_match or (has_indexed_data and (keyword_match or location_match or industry_match)):
            # Calculate score
            match_score = 1.0
            if has_keyword_filter and len(filter_keywords) > 0:
                match_score = len(matched_kws) / len(filter_keywords)
            return is_match, match_score, matched_kws
        
        # If indexed data exists but didn't match, still fall through to full matching
        # in case indexing missed something
    
    # FULL MATCHING: Standard matching logic for comprehensive results
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

    # Include all string values from metadata in searchable text
    metadata_values = []
    if isinstance(metadata, dict):
        def _collect_strings(d):
            vals = []
            if isinstance(d, dict):
                for v in d.values(): vals.extend(_collect_strings(v))
            elif isinstance(d, list):
                for v in d: vals.extend(_collect_strings(v))
            elif isinstance(d, (str, int, float)):
                vals.append(str(d))
            return vals
        metadata_values = _collect_strings(metadata)

    # Combine curated/searchable text for keyword matching.
    searchable_text = " ".join(
        filter(
            None,
            [
                title,
                description,
                summary,
                category_text,
                sector_text,
                meta_category_label,
                exec_summary,
                scope_summary,
                " ".join(metadata_values)
            ],
        )
    ).lower()

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
            
            # 1. Exact word match (most reliable) - Case insensitive
            # We use \b to ensure we match whole words like "website" and not "anywhere" in "anywhere"
            pattern = r'(?i)\b' + re.escape(kw_lower) + r'\b'
            if re.search(pattern, searchable_text):
                if kw_lower in original_keywords_lower:
                    matched_keywords.append(kw_lower)
            # 2. Substring match (fallback for longer terms)
            # Only allow substring match for terms >= 4 chars to avoid false positives with short keywords
            elif len(kw_lower) >= 4 and kw_lower in searchable_text:
                if kw_lower in original_keywords_lower:
                    matched_keywords.append(kw_lower)
        # Remove duplicates while preserving order
        seen = set()
        matched_keywords = [kw for kw in matched_keywords if not (kw in seen or seen.add(kw))]
    keyword_match = len(matched_keywords) > 0 if keywords else False

    # Check location match (OR - at least one location must match if locations are provided)
    # Use word boundary matching for better city/location detection
    debug_matching = os.getenv("DEBUG_LOCATION_MATCHING") == "1"
    location_match = True  # Default to True if no locations specified
    if keyword_locations:
        location_match = False  # User specified locations - require a match
        matched_location = None
        match_source = None  # Track where we found the match
        
        # We search for the location in the structured location info + title/description
        # But we MUST use word boundaries for locations to avoid "UK" matching "Buckinghamshire"
        for loc in keyword_locations:
            if not loc:
                continue
            loc_lower = loc.lower()
            pattern = r'(?i)\b' + re.escape(loc_lower) + r'\b'
            
            # Check location fields first
            if location_text and re.search(pattern, location_text):
                location_match = True
                matched_location = loc
                match_source = "location_fields"
                break
                
            # Fallback to title/description search (also with word boundaries)
            search_fields = (title or "") + " " + (description or "") + " " + (summary or "")
            if re.search(pattern, search_fields):
                location_match = True
                matched_location = loc
                match_source = "title/description"
                break
        
        # Debug logging
        if debug_matching:
            tender_title = tender_data.get("title", "Unknown")[:60]
            status = "✓ MATCHED" if location_match else "✗ REJECTED"
            if location_match:
                print(f"[Location Match] {status}: {tender_title}")
                print(f"  └─ Found '{matched_location}' in {match_source}")
                if match_source == "location_fields" or match_source == "location_fields (substring)":
                    print(f"  └─ Location text: '{location_text[:100]}'")
            else:
                print(f"[Location Match] {status}: {tender_title}")
                print(f"  └─ Required: {keyword_locations}")
                print(f"  └─ Tender location: '{location_text[:100] if location_text else '(empty)'}'")

    # Check industry/sector match (OR - at least one industry must match if industries are provided)
    industry_match = True  # Default to True if no industries specified
    if keyword_sectors:
        industry_match = False  # User specified industries - require a match
        
        tender_sectors = _normalize_str_list(tender_data.get("sector"))
        tender_categories = _normalize_str_list(tender_data.get("category"))
        # Check both sector and category fields
        all_tender_industries = set(tender_sectors + tender_categories)
        
        # Normalize user sectors for comparison (ensure lowercase)
        normalized_user_sectors = [s.lower().strip() if s else "" for s in keyword_sectors if s]
        
        # Check if any of the user's specified industries match
        # Handle CPV codes: user might specify "72", tender might have "72222300"
        for user_sector in normalized_user_sectors:
            if not user_sector:
                continue
            
            # 1. EXACT MATCH: Check if user sector exactly matches any tender industry
            if user_sector in all_tender_industries:
                industry_match = True
                break
            
            # 2. PREFIX MATCH: Handle CPV codes where user specifies "72" and tender has "72222300"
            # Only do prefix match if user_sector is a short CPV code (2-3 digits)
            if len(user_sector) <= 3 and user_sector.isdigit():
                for t_ind in all_tender_industries:
                    if t_ind and t_ind.startswith(user_sector):
                        industry_match = True
                        break
                if industry_match:
                    break
            
            # 3. REVERSE PREFIX MATCH: Handle cases where tender has "72" and user specifies "72222300"
            if len(user_sector) > 3 and user_sector.isdigit():
                for t_ind in all_tender_industries:
                    if t_ind and len(t_ind) <= 3 and t_ind.isdigit() and user_sector.startswith(t_ind):
                        industry_match = True
                        break
                if industry_match:
                    break
            
            # 4. TEXT SEARCH FALLBACK: Search for industry label in title/description
            # This helps when tender doesn't have proper CPV codes but mentions the industry
            # IMPORTANT: Only use this as a last resort and be very precise to avoid false matches
            industry_label = CPV_INDUSTRY_MAP.get(user_sector)
            if industry_label:
                label_lower = industry_label.lower()
                # First, try exact phrase match (most reliable)
                if label_lower in title or label_lower in category_text or label_lower in meta_category_label:
                    industry_match = True
                    break
                
                # For multi-word labels, require the PRIMARY keyword (first significant word, not common words)
                # Common words to skip: "and", "or", "the", "of", "for", "services", "work", "goods", "products"
                common_words = {"and", "or", "the", "of", "for", "services", "work", "goods", "products", "equipment"}
                label_terms = [t for t in label_lower.split() if t not in common_words and len(t) >= 4]
                
                # Only match if we have a primary keyword AND it appears in context
                if label_terms:
                    primary_keyword = label_terms[0]  # First significant word
                    # Require the primary keyword to appear, but be more strict
                    # Check if it appears as a whole word (not just substring)
                    import re
                    primary_pattern = r'\b' + re.escape(primary_keyword) + r'\b'
                    if re.search(primary_pattern, searchable_text, re.IGNORECASE):
                        # Additional check: if label has multiple significant terms, at least 2 should match
                        if len(label_terms) == 1 or sum(1 for term in label_terms if re.search(r'\b' + re.escape(term) + r'\b', searchable_text, re.IGNORECASE)) >= 2:
                            industry_match = True
                            break

        # Debug logging
        if os.getenv("DEBUG_LOCATION_MATCHING") == "1":
            tender_title = tender_data.get("title", "Unknown")[:50]
            status = "MATCHED" if industry_match else "REJECTED"
            print(f"[Industry Match] {status}: {tender_title} | Required: {normalized_user_sectors} | Tender has: {all_tender_industries}")

    # Matching logic: All SPECIFIED filters must match (AND between specified filters)
    # If keywords are specified, at least one must match. Same for locations and industries.
    # If a filter is NOT specified, it's considered matching (no restriction).
    
    # Check if each filter type is specified
    has_keyword_filter = len(keywords) > 0
    has_location_filter = len(keyword_locations) > 0
    has_industry_filter = len(keyword_sectors) > 0
    
    # At least one filter must be specified
    if not has_keyword_filter and not has_location_filter and not has_industry_filter:
        return False, 0.0, []
    
    # For each SPECIFIED filter, check if it matches
    # Unspecified filters default to True (no restriction)
    keyword_match_ok = keyword_match if has_keyword_filter else True
    location_match_ok = location_match if has_location_filter else True
    industry_match_ok = industry_match if has_industry_filter else True
    
    # All specified filters must match (AND)
    is_match = keyword_match_ok and location_match_ok and industry_match_ok

    # Debug logging - final decision
    if os.getenv("DEBUG_LOCATION_MATCHING") == "1":
        tender_title = tender_data.get("title", "Unknown")[:60]
        status = "✓ MATCH" if is_match else "✗ NO MATCH"
        print(f"\n[FINAL DECISION] {status}: {tender_title}")
        print(f"  └─ Keyword Filter: {'(not set)' if not has_keyword_filter else ('✓' if keyword_match else '✗')} (matched: {matched_keywords})")
        print(f"  └─ Location Filter: {'(not set)' if not has_location_filter else ('✓' if location_match else '✗')} (required: {keyword_locations})")
        print(f"  └─ Industry Filter: {'(not set)' if not has_industry_filter else ('✓' if industry_match else '✗')} (required: {keyword_sectors})")
        print()

    # Calculate match score - if no keywords, use location/industry match as full score
    if has_keyword_filter and len(keywords) > 0:
        match_score = len(matched_keywords) / len(keywords)
    elif is_match:
        match_score = 1.0  # Full score for location/industry-only matches
    else:
        match_score = 0.0

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

    # Log keyword set details for debugging
    debug_matching = os.getenv("DEBUG_LOCATION_MATCHING") == "1"
    if debug_matching:
        for kw in keyword_sets:
            print(f"[Rematch] Keyword set: keywords={kw.get('keywords')}, locations={kw.get('locations')}, sectors={kw.get('sectors')}")

    supabase.table("tender_matches").delete().eq("client_id", client_id).execute()

    # Fetch active tenders with pagination
    # Search recent tenders (last 90 days) for performance - balance between speed and coverage
    now_utc = datetime.now(timezone.utc)
    now_iso = now_utc.isoformat()
    lookback_90_days = (now_utc - timedelta(days=90)).isoformat()
    
    to_insert = []
    created = 0
    page_size = 1000  # Larger page size for fewer round trips
    offset = 0
    max_tenders = 30000  # Reasonable limit for fast matching (30 seconds target)
    total_processed = 0
    max_matches = 5000  # Early termination if we find enough matches
    
    print(f"[Rematch] Starting optimized rematch for client {client_id[:8]}...")
    start_time = time.time()
    
    # Check if tenders have indexed data for fast matching
    use_indexed_matching = os.getenv("USE_INDEXED_MATCHING", "1") == "1"
    
    while offset < max_tenders and len(to_insert) < max_matches:
        # Select indexed columns if available for faster matching
        select_cols = "id, title, description, summary, category, sector, location, value_amount, metadata, is_duplicate, source"
        if use_indexed_matching:
            select_cols += ", indexed_keywords, indexed_locations, indexed_industries"
        
        tenders_res = (
            supabase.table("tenders")
            .select(select_cols)
            .gte("published_date", lookback_90_days)
            .or_(f"deadline.is.null,deadline.gte.{now_iso}")
            .eq("is_duplicate", False)
            .order("published_date", desc=True)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        tenders = tenders_res.data or []
        
        if not tenders:
            break  # No more tenders to process
            
        for tender in tenders:
            total_processed += 1
            
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
            
            # Include metadata and indexed data in tender_data for keyword matching
            tender_data = {
                "title": tender.get("title"),
                "description": tender.get("description"),
                "summary": tender.get("summary"),
                "category": tender.get("category"),
                "sector": tender.get("sector"),
                "location": tender.get("location"),
                "location_name": location_name,
                "value_amount": tender.get("value_amount"),
                "metadata": metadata,
                # Include indexed data for faster matching
                "indexed_keywords": tender.get("indexed_keywords"),
                "indexed_locations": tender.get("indexed_locations"),
                "indexed_industries": tender.get("indexed_industries"),
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
        
        # Log progress every ~5000 tenders
        if total_processed % 5000 < page_size:
            elapsed = time.time() - start_time
            print(f"[Rematch] Processed {total_processed} tenders in {elapsed:.1f}s, found {len(to_insert)} matches so far...")
        
        # Early termination if we have enough matches
        if len(to_insert) >= max_matches:
            print(f"[Rematch] Found {len(to_insert)} matches - early termination for speed")
            break
        
        # If we got fewer results than page_size, we've reached the end
        if len(tenders) < page_size:
            break
    
    elapsed = time.time() - start_time
    print(f"[Rematch] Completed: processed {total_processed} tenders in {elapsed:.1f}s, found {len(to_insert)} matches")
    
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

