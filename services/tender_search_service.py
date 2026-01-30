"""
Tender Search Indexing Service

This service extracts and indexes keywords, locations, and industries from tenders
for fast dropdown suggestions and direct matching.
"""

import re
import os
from typing import Dict, Any, List, Set, Optional, Tuple
from datetime import datetime, timezone
from config.settings import get_supabase_client, reinitialize_supabase
from utils.logging_config import get_logger

logger = get_logger(__name__, "app")

# CPV code sections (first 2 digits) - maps to industry categories
CPV_SECTIONS = {
    "03": "Agricultural and farming goods",
    "09": "Petroleum and fuel products",
    "14": "Mining, basic metals and related products",
    "15": "Food, beverages and tobacco",
    "16": "Agricultural machinery",
    "18": "Clothing and accessories",
    "19": "Leather and textile fabrics",
    "22": "Printed matter and related products",
    "24": "Chemical products",
    "30": "Office and computing equipment",
    "31": "Electrical machinery, apparatus and equipment",
    "32": "Radio, television, communication and related equipment",
    "33": "Medical equipment and pharmaceuticals",
    "34": "Transport equipment and auxiliary products",
    "35": "Security, defence and safety equipment",
    "37": "Musical instruments, sporting goods and games",
    "38": "Laboratory, optical and precision equipment",
    "39": "Furniture and furnishings",
    "41": "Collected and purified water",
    "42": "Industrial machinery",
    "43": "Machinery for mining, quarrying and construction",
    "44": "Construction structures and materials",
    "45": "Construction work",
    "48": "Software packages and information systems",
    "50": "Repair and maintenance services",
    "51": "Installation services",
    "55": "Hotel, restaurant and retail trade services",
    "60": "Transport services",
    "63": "Supporting transport services",
    "64": "Postal and telecommunications services",
    "65": "Public utilities",
    "66": "Financial and insurance services",
    "70": "Real estate services",
    "71": "Architecture and engineering services",
    "72": "IT services",
    "73": "Research and development services",
    "75": "Administration, defence and social security",
    "76": "Oil and gas industry services",
    "77": "Agricultural, forestry and horticultural services",
    "79": "Business services",
    "80": "Education and training services",
    "85": "Health and social work services",
    "90": "Sewage, refuse, cleaning and environmental services",
    "92": "Recreational, cultural and sporting services",
    "98": "Other community, social and personal services",
}

# UK regions and locations - for dropdown suggestions
UK_LOCATIONS = [
    # Countries
    "england", "scotland", "wales", "northern ireland",
    # Major regions
    "east midlands", "east of england", "greater london", "london", 
    "north east", "north west", "south east", "south west",
    "west midlands", "yorkshire", "yorkshire and the humber",
    # Major cities
    "birmingham", "manchester", "leeds", "liverpool", "sheffield",
    "bristol", "newcastle", "nottingham", "southampton", "leicester",
    "coventry", "bradford", "cardiff", "belfast", "edinburgh",
    "glasgow", "aberdeen", "dundee", "swansea", "newport",
    "oxford", "cambridge", "brighton", "plymouth", "reading",
    "luton", "wolverhampton", "derby", "stoke-on-trent", "bolton",
    "hull", "middlesbrough", "sunderland", "portsmouth", "peterborough",
    # Common location terms
    "nationwide", "uk wide", "uk-wide", "national", "regional", "local",
    "multiple locations", "remote", "hybrid",
]

# Common procurement keywords for extraction
STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "this", "that",
    "these", "those", "it", "its", "they", "their", "them", "we", "our", "us",
    "you", "your", "he", "she", "his", "her", "which", "who", "whom", "whose",
    "what", "where", "when", "why", "how", "all", "each", "every", "both",
    "few", "more", "most", "other", "some", "such", "no", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "also", "now", "any",
    "contract", "tender", "notice", "opportunity", "procurement", "services",
    "supply", "works", "goods", "provision", "framework", "agreement",
    "required", "requirements", "provide", "providing", "delivery", "deliver",
}


def _normalize_str_list(value: Any) -> List[str]:
    """Normalize a value to a list of lowercase strings."""
    if value is None:
        return []
    result = []
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
            text = str(item).strip().lower()
            if text:
                result.append(text)
    else:
        text = str(value).strip().lower()
        if text:
            result.append(text)
    return result


def extract_cpv_codes(tender_data: Dict[str, Any]) -> Set[str]:
    """Extract 2-digit CPV code prefixes from tender data."""
    cpv_codes = set()
    
    # Extract from category field
    categories = _normalize_str_list(tender_data.get("category"))
    for cat in categories:
        if not cat:
            continue
        digits = ''.join(c for c in cat if c.isdigit())
        if len(digits) >= 2:
            cpv_codes.add(digits[:2])
    
    # Extract from sector field
    sectors = _normalize_str_list(tender_data.get("sector"))
    for sector in sectors:
        if not sector:
            continue
        digits = ''.join(c for c in sector if c.isdigit())
        if len(digits) >= 2:
            cpv_codes.add(digits[:2])
    
    # Extract from metadata if available
    metadata = tender_data.get("metadata") or {}
    if isinstance(metadata, dict):
        meta_cpv = metadata.get("cpv_codes") or metadata.get("cpv")
        if meta_cpv:
            for code in _normalize_str_list(meta_cpv):
                digits = ''.join(c for c in code if c.isdigit())
                if len(digits) >= 2:
                    cpv_codes.add(digits[:2])
    
    # Extract from full_data if available (but don't fail if it's missing)
    # Note: full_data can be very large, so we limit processing
    full_data = tender_data.get("full_data")
    if full_data and isinstance(full_data, dict):
        try:
            # Check various CPV field locations
            for key in ["cpvCodes", "cpv", "cpv_codes", "classifications"]:
                if key in full_data:
                    for code in _normalize_str_list(full_data[key]):
                        digits = ''.join(c for c in code if c.isdigit())
                        if len(digits) >= 2:
                            cpv_codes.add(digits[:2])
            
            # Check items/lots for CPV codes (limit to first 5 for performance)
            items = full_data.get("items") or full_data.get("lots") or []
            if isinstance(items, list):
                for item in items[:5]:  # Reduced from 10 to 5 for performance
                    if isinstance(item, dict):
                        item_cpv = item.get("cpv") or item.get("cpvCode") or item.get("classification")
                        if item_cpv:
                            for code in _normalize_str_list(item_cpv):
                                digits = ''.join(c for c in code if c.isdigit())
                                if len(digits) >= 2:
                                    cpv_codes.add(digits[:2])
        except Exception:
            # If full_data processing fails, continue without it
            pass
    
    return cpv_codes


def extract_locations(tender_data: Dict[str, Any]) -> Set[str]:
    """Extract normalized location values from tender data."""
    locations = set()
    
    # Direct location field
    location = (tender_data.get("location") or "").strip().lower()
    if location:
        locations.add(location)
        # Also check if it matches a known UK location
        for uk_loc in UK_LOCATIONS:
            if uk_loc in location:
                locations.add(uk_loc)
    
    # Check metadata
    metadata = tender_data.get("metadata") or {}
    if isinstance(metadata, dict):
        loc_name = (metadata.get("location_name") or "").strip().lower()
        if loc_name:
            locations.add(loc_name)
            for uk_loc in UK_LOCATIONS:
                if uk_loc in loc_name:
                    locations.add(uk_loc)
        
        region = (metadata.get("region") or "").strip().lower()
        if region:
            locations.add(region)
    
    # Check full_data for buyer location (optional, can be slow for large JSON)
    full_data = tender_data.get("full_data")
    if full_data and isinstance(full_data, dict):
        try:
            # Buyer address
            buyer = full_data.get("buyer") or {}
            if isinstance(buyer, dict):
                address = buyer.get("address") or {}
                if isinstance(address, dict):
                    for key in ["region", "locality", "town", "city", "countryName"]:
                        addr_val = (address.get(key) or "").strip().lower()
                        if addr_val:
                            locations.add(addr_val)
                            for uk_loc in UK_LOCATIONS:
                                if uk_loc in addr_val:
                                    locations.add(uk_loc)
            
            # Delivery locations (limit to first 3 for performance)
            tender_info = full_data.get("tender") or {}
            items = tender_info.get("items") or full_data.get("items") or []
            if isinstance(items, list):
                for item in items[:3]:  # Reduced from 5 to 3 for performance
                    if isinstance(item, dict):
                        delivery = item.get("deliveryLocation") or item.get("placeOfPerformance") or {}
                        if isinstance(delivery, dict):
                            for key in ["region", "locality", "description"]:
                                del_val = (delivery.get(key) or "").strip().lower()
                                if del_val:
                                    locations.add(del_val)
                                    for uk_loc in UK_LOCATIONS:
                                        if uk_loc in del_val:
                                            locations.add(uk_loc)
        except Exception:
            # If full_data processing fails, continue without it
            pass
    
    # Filter out empty and very short values
    return {loc for loc in locations if loc and len(loc) >= 2}


def extract_keywords(tender_data: Dict[str, Any], max_keywords: int = 20) -> Set[str]:
    """Extract meaningful keywords from tender title and description."""
    keywords = set()
    
    # Build text to analyze
    title = (tender_data.get("title") or "").strip()
    description = (tender_data.get("description") or "").strip()
    summary = (tender_data.get("summary") or "").strip()
    
    full_text = f"{title} {description} {summary}".lower()
    
    # Extract words (alphanumeric, 3+ chars)
    words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9]{2,}\b', full_text)
    
    # Filter stop words and count frequency
    word_freq: Dict[str, int] = {}
    for word in words:
        word = word.lower()
        if word not in STOP_WORDS and len(word) >= 3:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and take top keywords
    sorted_words = sorted(word_freq.items(), key=lambda x: (-x[1], x[0]))
    for word, freq in sorted_words[:max_keywords]:
        if freq >= 1:  # At least appears once
            keywords.add(word)
    
    # Also extract common multi-word phrases
    phrases = [
        "software development", "project management", "data analysis",
        "health care", "social care", "mental health", "primary care",
        "cloud computing", "cyber security", "information technology",
        "building maintenance", "facilities management", "waste management",
        "public transport", "road maintenance", "highways",
        "legal services", "financial services", "consultancy services",
        "training services", "recruitment services",
    ]
    for phrase in phrases:
        if phrase in full_text:
            keywords.add(phrase.replace(" ", "_"))  # Store as single token
    
    return keywords


def build_search_text(tender_data: Dict[str, Any]) -> str:
    """Build combined search text for full-text search."""
    parts = []
    
    # Title (most important)
    title = (tender_data.get("title") or "").strip()
    if title:
        parts.append(title)
    
    # Description
    description = (tender_data.get("description") or "").strip()
    if description:
        parts.append(description[:5000])  # Limit description length
    
    # Location
    location = (tender_data.get("location") or "").strip()
    if location:
        parts.append(location)
    
    # Category labels
    category = tender_data.get("category")
    if category:
        if isinstance(category, list):
            parts.extend([str(c) for c in category if c])
        else:
            parts.append(str(category))
    
    # Sector
    sector = tender_data.get("sector")
    if sector:
        if isinstance(sector, list):
            parts.extend([str(s) for s in sector if s])
        else:
            parts.append(str(sector))
    
    return " ".join(parts)


def extract_tender_search_data(tender_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract all indexed search data from a tender.
    Returns a dict with indexed_keywords, indexed_locations, indexed_industries, and search_text.
    """
    industries = extract_cpv_codes(tender_data)
    locations = extract_locations(tender_data)
    keywords = extract_keywords(tender_data)
    search_text = build_search_text(tender_data)
    
    return {
        "indexed_industries": list(industries) if industries else None,
        "indexed_locations": list(locations) if locations else None,
        "indexed_keywords": list(keywords) if keywords else None,
        "search_text": search_text if search_text else None,
    }


def update_tender_with_search_data(tender_id: str, tender_data: Dict[str, Any]) -> bool:
    """
    Update a tender with extracted search data.
    Returns True if successful.
    """
    try:
        search_data = extract_tender_search_data(tender_data)
        
        supabase = get_supabase_client()
        result = supabase.table("tenders").update(search_data).eq("id", tender_id).execute()
        
        if result.data:
            # Also update the search terms lookup table
            _update_search_terms_counts(search_data)
            return True
        return False
    except Exception as e:
        logger.error(f"Error updating tender {tender_id} with search data: {e}")
        return False


def _update_search_terms_counts(search_data: Dict[str, Any]):
    """Update the tender_search_terms lookup table with extracted terms."""
    try:
        supabase = get_supabase_client()
        now = datetime.now(timezone.utc).isoformat()
        
        terms_to_upsert = []
        
        # Industries
        industries = search_data.get("indexed_industries") or []
        for ind in industries:
            if ind and ind in CPV_SECTIONS:
                terms_to_upsert.append({
                    "term_type": "industry",
                    "term_value": ind,
                    "term_display": f"{ind} - {CPV_SECTIONS[ind]}",
                    "last_seen_at": now,
                })
        
        # Locations
        locations = search_data.get("indexed_locations") or []
        for loc in locations:
            if loc and len(loc) >= 2:
                # Capitalize for display
                display = loc.replace("_", " ").title()
                terms_to_upsert.append({
                    "term_type": "location",
                    "term_value": loc,
                    "term_display": display,
                    "last_seen_at": now,
                })
        
        # Keywords
        keywords = search_data.get("indexed_keywords") or []
        for kw in keywords:
            if kw and len(kw) >= 3:
                display = kw.replace("_", " ").title()
                terms_to_upsert.append({
                    "term_type": "keyword",
                    "term_value": kw,
                    "term_display": display,
                    "last_seen_at": now,
                })
        
        # Batch upsert terms (ignore conflicts)
        if terms_to_upsert:
            # Use on_conflict to update last_seen_at on existing terms
            supabase.table("tender_search_terms").upsert(
                terms_to_upsert,
                on_conflict="term_type,term_value"
            ).execute()
            
    except Exception as e:
        # Non-critical error - log and continue
        logger.warning(f"Error updating search terms: {e}")


def reindex_all_tenders(batch_size: int = 100, max_tenders: int = 5000) -> Tuple[int, int]:
    """
    Re-index all existing tenders with search data.
    Returns (success_count, error_count).
    """
    logger.info(f"Starting tender reindexing (batch_size={batch_size}, max={max_tenders})")
    
    supabase = get_supabase_client()
    success_count = 0
    error_count = 0
    offset = 0
    
    while offset < max_tenders:
        try:
            # Fetch batch of tenders that need indexing
            result = supabase.table("tenders").select(
                "id, title, description, summary, location, category, sector, metadata, full_data"
            ).is_("indexed_industries", "null").limit(batch_size).execute()
            
            tenders = result.data or []
            if not tenders:
                logger.info(f"No more tenders to index at offset {offset}")
                break
            
            for tender in tenders:
                try:
                    tender_id = tender["id"]
                    search_data = extract_tender_search_data(tender)
                    
                    supabase.table("tenders").update(search_data).eq("id", tender_id).execute()
                    success_count += 1
                    
                except Exception as e:
                    logger.warning(f"Error indexing tender {tender.get('id')}: {e}")
                    error_count += 1
            
            offset += batch_size
            logger.info(f"Indexed {success_count} tenders, {error_count} errors...")
            
        except Exception as e:
            logger.error(f"Error fetching tenders batch: {e}")
            break
    
    # Rebuild search term counts
    _rebuild_search_term_counts()
    
    logger.info(f"Tender reindexing complete: {success_count} success, {error_count} errors")
    return success_count, error_count


def _rebuild_search_term_counts():
    """Rebuild the tender_search_terms counts from indexed tenders."""
    logger.info("Rebuilding search term counts...")
    
    try:
        supabase = get_supabase_client()
        
        # Get counts for each term type
        for term_type, column in [
            ("industry", "indexed_industries"),
            ("location", "indexed_locations"),
            ("keyword", "indexed_keywords"),
        ]:
            try:
                # Get distinct values and counts using aggregation
                result = supabase.table("tenders").select(column).not_.is_(column, "null").execute()
                
                # Count occurrences
                term_counts: Dict[str, int] = {}
                for row in result.data or []:
                    values = row.get(column) or []
                    if isinstance(values, list):
                        for val in values:
                            if val:
                                term_counts[val] = term_counts.get(val, 0) + 1
                
                # Update counts in lookup table
                for term_value, count in term_counts.items():
                    display = term_value
                    if term_type == "industry" and term_value in CPV_SECTIONS:
                        display = f"{term_value} - {CPV_SECTIONS[term_value]}"
                    else:
                        display = term_value.replace("_", " ").title()
                    
                    supabase.table("tender_search_terms").upsert({
                        "term_type": term_type,
                        "term_value": term_value,
                        "term_display": display,
                        "tender_count": count,
                    }, on_conflict="term_type,term_value").execute()
                
                logger.info(f"Updated {len(term_counts)} {term_type} term counts")
                
            except Exception as e:
                logger.warning(f"Error rebuilding {term_type} counts: {e}")
                
    except Exception as e:
        logger.error(f"Error rebuilding search term counts: {e}")


def get_search_suggestions(term_type: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Get search term suggestions for dropdowns.
    Returns list of {value, label, count} dicts.
    """
    try:
        supabase = get_supabase_client()
        
        result = supabase.table("tender_search_terms").select(
            "term_value, term_display, tender_count"
        ).eq("term_type", term_type).gt("tender_count", 0).order(
            "tender_count", desc=True
        ).limit(limit).execute()
        
        suggestions = []
        for row in result.data or []:
            suggestions.append({
                "value": row["term_value"],
                "label": row["term_display"] or row["term_value"],
                "count": row["tender_count"] or 0,
            })
        
        return suggestions
        
    except Exception as e:
        logger.error(f"Error getting {term_type} suggestions: {e}")
        return []


def match_tenders_by_indexed_values(
    keywords: Optional[List[str]] = None,
    locations: Optional[List[str]] = None,
    industries: Optional[List[str]] = None,
    limit: int = 100,
    days_lookback: int = 30,
) -> List[Dict[str, Any]]:
    """
    Fast tender matching using indexed values.
    Uses database array overlap operators for efficient matching.
    """
    try:
        supabase = get_supabase_client()
        
        # Normalize inputs
        keywords = [k.lower().strip() for k in (keywords or []) if k]
        locations = [l.lower().strip() for l in (locations or []) if l]
        industries = [i.lower().strip() for i in (industries or []) if i]
        
        # Build query
        from datetime import timedelta
        now = datetime.now(timezone.utc)
        cutoff = (now - timedelta(days=days_lookback)).isoformat()
        
        query = supabase.table("tenders").select(
            "id, source, external_id, title, description, summary, deadline, "
            "published_date, value_amount, value_currency, location, category, sector, "
            "indexed_keywords, indexed_locations, indexed_industries"
        ).eq("is_duplicate", False).gte("published_date", cutoff).or_(
            f"deadline.is.null,deadline.gte.{now.isoformat()}"
        )
        
        # Apply array overlap filters (using `ov` for @> overlap operator isn't directly supported,
        # so we use RPC or build conditions)
        # For now, fetch and filter in Python (still fast with indexed columns)
        
        result = query.order("published_date", desc=True).limit(limit * 3).execute()
        
        matched = []
        for tender in result.data or []:
            score = 0
            total_filters = 0
            
            tender_keywords = set(tender.get("indexed_keywords") or [])
            tender_locations = set(tender.get("indexed_locations") or [])
            tender_industries = set(tender.get("indexed_industries") or [])
            
            # Check keyword match
            if keywords:
                total_filters += 1
                keyword_matches = len(set(keywords) & tender_keywords)
                if keyword_matches > 0:
                    score += keyword_matches / len(keywords)
            
            # Check location match
            if locations:
                total_filters += 1
                location_matches = len(set(locations) & tender_locations)
                if location_matches > 0:
                    score += location_matches / len(locations)
            
            # Check industry match
            if industries:
                total_filters += 1
                industry_matches = len(set(industries) & tender_industries)
                if industry_matches > 0:
                    score += industry_matches / len(industries)
            
            # Match if any filter matches (OR logic)
            if score > 0:
                tender["match_score"] = score / max(total_filters, 1)
                matched.append(tender)
        
        # Sort by match score and limit
        matched.sort(key=lambda x: (-x.get("match_score", 0), x.get("published_date", "")))
        return matched[:limit]
        
    except Exception as e:
        logger.error(f"Error matching tenders by indexed values: {e}")
        return []

