"""
Tender Ingestion Service

This service retrieves tender opportunities from multiple public procurement APIs,
processes them, and matches them against user-defined keywords.

This should be run as a scheduled job (e.g., daily via cron or scheduled task).
"""

import os
import json
import requests
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any
from supabase import create_client, Client
from dotenv import load_dotenv
import traceback
from urllib.parse import urlencode

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().strip('"').strip("'")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip().strip('"').strip("'")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def generate_anonymised_summary(tender_data: Dict[str, Any]) -> str:
    """
    Generate an anonymised summary of the tender for notifications.
    This excludes sensitive information like exact values, specific locations, etc.
    """
    title = tender_data.get("title", "")
    category = tender_data.get("category", "")
    sector = tender_data.get("sector", "")
    
    # Create a summary without revealing too much detail
    summary_parts = []
    if title:
        # Truncate title if too long
        title_summary = title[:100] + "..." if len(title) > 100 else title
        summary_parts.append(title_summary)
    if category:
        summary_parts.append(f"Category: {category}")
    if sector:
        summary_parts.append(f"Sector: {sector}")
    
    return " | ".join(summary_parts) if summary_parts else "Tender opportunity available"


def _parse_ocds_release(source: str, release: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse an OCDS release into our internal raw_data shape expected by normalize_tender_data().
    """
    tender = release.get("tender") or {}
    tender_period = tender.get("tenderPeriod") or {}
    value = (tender.get("value") or {}) if isinstance(tender.get("value"), dict) else {}
    # Location best-effort: try deliveryLocations or items
    location = None
    if isinstance(tender.get("items"), list) and tender["items"]:
        # Try first item's delivery location description
        delivery = tender["items"][0].get("deliveryLocation") or {}
        location = delivery.get("description")
    if not location:
        # Fall back to buyer address if present
        buyer = release.get("buyer") or {}
        address = (buyer.get("address") or {}) if isinstance(buyer.get("address"), dict) else {}
        parts = [address.get("locality"), address.get("region"), address.get("countryName") or address.get("country")]
        location = ", ".join([p for p in parts if p])

    category = None
    # Try CPV from classification
    if isinstance(tender.get("classification"), dict):
        category = tender["classification"].get("id") or tender["classification"].get("description")
    if not category and isinstance(tender.get("items"), list) and tender["items"]:
        classification = tender["items"][0].get("classification") or {}
        category = classification.get("id") or classification.get("description")

    return {
        "id": release.get("ocid") or release.get("id"),
        "title": tender.get("title") or "",
        "description": tender.get("description") or "",
        "deadline": tender_period.get("endDate"),
        "published_date": release.get("date") or release.get("publishedDate") or release.get("publicationDate"),
        "value": value.get("amount"),
        "value_currency": value.get("currency") or "GBP",
        "location": location,
        "category": category,
        "sector": None,
        "full_data": release,
        "api_version": release.get("ocid") and "ocds" or "unknown",
    }


def ingest_ted_tenders() -> List[Dict[str, Any]]:
    """
    Ingest tenders from TED (Tenders Electronic Daily - EU)
    API: https://docs.ted.europa.eu/api/latest/index.html
    """
    tenders = []
    try:
        # TED API endpoint (example - actual implementation depends on API structure)
        # This is a placeholder - actual TED API integration would go here
        url = "https://ted.europa.eu/api/v2.1/notices/search"
        
        # For now, return empty list - actual implementation needed
        # params = {
        #     "q": "*",
        #     "publishedFrom": (datetime.now() - timedelta(days=1)).isoformat(),
        #     "publishedTo": datetime.now().isoformat(),
        # }
        # response = requests.get(url, params=params, timeout=30)
        # if response.status_code == 200:
        #     data = response.json()
        #     # Process and normalize TED data
        #     ...
        
        print("TED ingestion: Placeholder - API integration required")
    except Exception as e:
        print(f"Error ingesting TED tenders: {e}")
        traceback.print_exc()
    
    return tenders


def ingest_find_a_tender() -> List[Dict[str, Any]]:
    """
    Ingest tenders from Find a Tender (UK)
    API: https://www.find-tender.service.gov.uk/apidocumentation
    """
    tenders: List[Dict[str, Any]] = []
    try:
        # OCDS search endpoint for Find a Tender
        # Example params: publishedFrom, publishedTo in ISO8601, pageSize
        base_url = "https://www.find-tender.service.gov.uk/Published/Notices/OCDS/Search"
        published_to = datetime.now(timezone.utc)
        published_from = published_to - timedelta(days=7)
        params = {
            "publishedFrom": published_from.strftime("%Y-%m-%dT00:00:00Z"),
            "publishedTo": published_to.strftime("%Y-%m-%dT23:59:59Z"),
            "pageSize": 100,
            "order": "desc",
        }
        url = f"{base_url}?{urlencode(params)}"
        resp = requests.get(url, timeout=30, headers={
            "Accept": "application/json",
            "User-Agent": "BidWell/1.0 (+contact@bidwell.app)"
        })
        resp.raise_for_status()
        data = resp.json()
        releases = data.get("releases") or []
        for rel in releases:
            tenders.append({"source": "FindATender", **_parse_ocds_release("FindATender", rel)})
    except Exception as e:
        print(f"Error ingesting Find a Tender: {e}")
        traceback.print_exc()
    return tenders


def ingest_contracts_finder() -> List[Dict[str, Any]]:
    """
    Ingest tenders from ContractsFinder (UK)
    API: https://www.contractsfinder.service.gov.uk/apidocumentation/V2
    """
    tenders: List[Dict[str, Any]] = []
    try:
        # OCDS search endpoint for Contracts Finder
        base_url = "https://www.contractsfinder.service.gov.uk/Published/Notices/OCDS/Search"
        published_to = datetime.now(timezone.utc)
        published_from = published_to - timedelta(days=7)
        params = {
            "publishedFrom": published_from.strftime("%Y-%m-%dT00:00:00Z"),
            "publishedTo": published_to.strftime("%Y-%m-%dT23:59:59Z"),
            "pageSize": 100,
            "order": "desc",
        }
        url = f"{base_url}?{urlencode(params)}"
        resp = requests.get(url, timeout=30, headers={
            "Accept": "application/json",
            "User-Agent": "BidWell/1.0 (+contact@bidwell.app)"
        })
        resp.raise_for_status()
        data = resp.json()
        releases = data.get("releases") or []
        for rel in releases:
            tenders.append({"source": "ContractsFinder", **_parse_ocds_release("ContractsFinder", rel)})
    except Exception as e:
        print(f"Error ingesting Contracts Finder: {e}")
        traceback.print_exc()
    return tenders


def ingest_pcs_scotland() -> List[Dict[str, Any]]:
    """
    Ingest tenders from PCS Scotland (Public Contracts Scotland)
    API: https://api.publiccontractsscotland.gov.uk
    """
    tenders = []
    try:
        # PCS Scotland API endpoint (placeholder)
        # url = "https://api.publiccontractsscotland.gov.uk/v1/notices"
        # Actual implementation would go here
        print("PCS Scotland ingestion: Placeholder - API integration required")
    except Exception as e:
        print(f"Error ingesting PCS Scotland: {e}")
        traceback.print_exc()
    
    return tenders


def ingest_sell2wales() -> List[Dict[str, Any]]:
    """
    Ingest tenders from Sell2Wales
    API: https://api.sell2wales.gov.wales/v1
    """
    tenders: List[Dict[str, Any]] = []
    try:
        # Fetch current and previous two months' contract notices in OCDS format (English)
        base_url = "https://api.sell2wales.gov.wales/v1/Notices"
        now = datetime.now(timezone.utc)
        for m in range(0, 3):
            dt = (now.replace(day=15) - timedelta(days=30*m))
            date_from = dt.strftime("%m-%Y")  # mm-yyyy
            params = {
                "dateFrom": date_from,
                "noticeType": 2,     # Contract Notice
                "outputType": 0,     # OCDS output
                "locale": 2057,      # English
            }
            url = f"{base_url}?{urlencode(params)}"
            resp = requests.get(url, timeout=60, headers={
                "Accept": "application/json",
                "User-Agent": "BidWell/1.0 (+contact@bidwell.app)",
                "Referer": "https://sell2wales.gov.wales"
            })
            if resp.status_code == 403:
                # Try Welsh locale fallback
                params["locale"] = 1106
                url = f"{base_url}?{urlencode(params)}"
                resp = requests.get(url, timeout=60, headers={
                    "Accept": "application/json",
                    "User-Agent": "BidWell/1.0 (+contact@bidwell.app)",
                    "Referer": "https://sell2wales.gov.wales"
                })
            resp.raise_for_status()
            data = resp.json()
            releases = data.get("releases") or []
            for rel in releases:
                tenders.append({"source": "Sell2Wales", **_parse_ocds_release("Sell2Wales", rel)})
    except Exception as e:
        print(f"Error ingesting Sell2Wales: {e}")
        traceback.print_exc()
    return tenders


def ingest_sam_gov() -> List[Dict[str, Any]]:
    """
    Ingest tenders from SAM.gov (US Federal)
    API: https://open.gsa.gov/api/get-opportunities-public-api/
    Note: Registration required
    """
    tenders = []
    try:
        # SAM.gov API endpoint (placeholder)
        # Requires API key registration
        # url = "https://api.sam.gov/opportunities/v2/search"
        # headers = {"X-API-Key": os.getenv("SAM_GOV_API_KEY")}
        # Actual implementation would go here
        print("SAM.gov ingestion: Placeholder - API integration and registration required")
    except Exception as e:
        print(f"Error ingesting SAM.gov: {e}")
        traceback.print_exc()
    
    return tenders


def ingest_austender() -> List[Dict[str, Any]]:
    """
    Ingest tenders from AusTender (Australia)
    API: https://api.tenders.gov.au/
    Note: Registration required
    """
    tenders = []
    try:
        # AusTender API endpoint (placeholder)
        # Requires API key registration
        # url = "https://api.tenders.gov.au/opportunities"
        # headers = {"X-API-Key": os.getenv("AUSTENDER_API_KEY")}
        # Actual implementation would go here
        print("AusTender ingestion: Placeholder - API integration and registration required")
    except Exception as e:
        print(f"Error ingesting AusTender: {e}")
        traceback.print_exc()
    
    return tenders


def normalize_tender_data(raw_data: Dict[str, Any], source: str) -> Dict[str, Any]:
    """
    Normalize tender data from different sources into a common format.
    """
    # Extract common fields (structure depends on actual API responses)
    def _to_number(x):
        try:
            if x is None or x == "":
                return None
            return float(x)
        except Exception:
            return None
    normalized = {
        "source": source,
        "external_id": raw_data.get("id") or raw_data.get("notice_id") or raw_data.get("reference"),
        "title": raw_data.get("title") or raw_data.get("name") or raw_data.get("subject", ""),
        "description": raw_data.get("description") or raw_data.get("summary") or "",
        "deadline": raw_data.get("deadline") or raw_data.get("closing_date") or raw_data.get("submission_deadline"),
        "published_date": raw_data.get("published_date") or raw_data.get("publication_date") or datetime.now().isoformat(),
        "value_amount": _to_number(raw_data.get("value") or raw_data.get("estimated_value") or raw_data.get("contract_value")),
        "value_currency": raw_data.get("currency") or raw_data.get("value_currency") or "GBP",
        "location": raw_data.get("location") or raw_data.get("region") or raw_data.get("country"),
        "category": raw_data.get("category") or raw_data.get("cpv_code") or "",
        "sector": raw_data.get("sector") or raw_data.get("industry") or "",
        "full_data": raw_data,  # Store complete raw data
        "metadata": {
            "ingested_at": datetime.now().isoformat(),
            "source_api_version": raw_data.get("api_version", "unknown"),
        },
    }
    
    # Generate anonymised summary
    normalized["summary"] = generate_anonymised_summary(normalized)
    
    return normalized


def store_tender(tender_data: Dict[str, Any]) -> str | None:
    """
    Store a tender in the database, checking for duplicates.
    Returns the tender ID if stored, None if duplicate.
    """
    try:
        # Check if tender already exists
        existing = supabase.table("tenders").select("id").eq("source", tender_data["source"]).eq("external_id", tender_data["external_id"]).execute()
        
        if existing.data:
            # Tender already exists, return existing ID
            return existing.data[0]["id"]
        
        # Insert new tender
        res = supabase.table("tenders").insert(tender_data).execute()
        
        if res.data:
            return res.data[0]["id"]
        return None
    except Exception as e:
        print(f"Error storing tender: {e}")
        traceback.print_exc()
        return None


def match_tender_against_keywords(tender_data: Dict[str, Any], keyword_set: Dict[str, Any]) -> tuple[bool, float, list]:
    """
    Match a tender against a keyword set.
    Returns: (is_match, match_score, matched_keywords)
    """
    title = (tender_data.get("title") or "").lower()
    description = (tender_data.get("description") or "").lower()
    summary = (tender_data.get("summary") or "").lower()
    category = (tender_data.get("category") or "").lower()
    sector = (tender_data.get("sector") or "").lower()
    location = (tender_data.get("location") or "").lower()
    
    # Combine all searchable text
    searchable_text = f"{title} {description} {summary} {category} {sector} {location}"
    
    keywords = [k.lower() for k in (keyword_set.get("keywords") or [])]
    match_type = keyword_set.get("match_type", "any")
    
    matched_keywords = []
    for keyword in keywords:
        if keyword in searchable_text:
            matched_keywords.append(keyword)
    
    # Check match type
    if match_type == "all":
        is_match = len(matched_keywords) == len(keywords) and len(keywords) > 0
    else:  # "any"
        is_match = len(matched_keywords) > 0
    
    # Calculate match score (percentage of keywords matched)
    match_score = len(matched_keywords) / len(keywords) if keywords else 0.0
    
    # Apply additional filters
    if is_match:
        # Category filter
        if keyword_set.get("categories"):
            tender_cats = [c.lower() for c in (tender_data.get("category") or "").split(",")]
            keyword_cats = [c.lower() for c in keyword_set.get("categories", [])]
            if not any(cat in tender_cats for cat in keyword_cats):
                is_match = False
        
        # Sector filter
        if keyword_set.get("sectors"):
            tender_sectors = [s.lower() for s in (tender_data.get("sector") or "").split(",")]
            keyword_sectors = [s.lower() for s in keyword_set.get("sectors", [])]
            if not any(sec in tender_sectors for sec in keyword_sectors):
                is_match = False
        
        # Location filter
        if keyword_set.get("locations"):
            keyword_locations = [l.lower() for l in keyword_set.get("locations", [])]
            if not any(loc in location for loc in keyword_locations):
                is_match = False
        
        # Value filters
        value_amount = tender_data.get("value_amount")
        if value_amount is not None:
            try:
                min_v = float(keyword_set.get("min_value")) if keyword_set.get("min_value") is not None else None
                max_v = float(keyword_set.get("max_value")) if keyword_set.get("max_value") is not None else None
            except Exception:
                min_v = keyword_set.get("min_value")
                max_v = keyword_set.get("max_value")
            if min_v is not None and value_amount < min_v:
                is_match = False
            if max_v is not None and value_amount > max_v:
                is_match = False
    
    return is_match, match_score, matched_keywords


def process_tender_matches(tender_id: str, tender_data: Dict[str, Any]):
    """
    Match a tender against all active keyword sets and create match records.
    """
    try:
        # Get all active keyword sets
        keyword_sets_res = supabase.table("user_tender_keywords").select("*").eq("is_active", True).execute()
        keyword_sets = keyword_sets_res.data or []
        
        for keyword_set in keyword_sets:
            is_match, match_score, matched_keywords = match_tender_against_keywords(tender_data, keyword_set)
            
            if is_match:
                # Create match record
                match_data = {
                    "tender_id": tender_id,
                    "client_id": keyword_set["client_id"],
                    "keyword_set_id": keyword_set["id"],
                    "match_score": match_score,
                    "matched_keywords": matched_keywords,
                }
                
                # Check if match already exists
                existing = supabase.table("tender_matches").select("id").eq("tender_id", tender_id).eq("client_id", keyword_set["client_id"]).eq("keyword_set_id", keyword_set["id"]).execute()
                
                if not existing.data:
                    supabase.table("tender_matches").insert(match_data).execute()
                    print(f"Created match for tender {tender_id} and client {keyword_set['client_id']}")
    except Exception as e:
        print(f"Error processing tender matches: {e}")
        traceback.print_exc()


def ingest_all_tenders():
    """
    Main function to ingest tenders from all sources.
    """
    print(f"Starting tender ingestion at {datetime.now().isoformat()}")
    
    all_tenders = []
    
    # Ingest from all sources
    print("Ingesting from TED...")
    all_tenders.extend(ingest_ted_tenders())
    
    print("Ingesting from Find a Tender...")
    all_tenders.extend(ingest_find_a_tender())
    
    print("Ingesting from ContractsFinder...")
    all_tenders.extend(ingest_contracts_finder())
    
    print("Ingesting from PCS Scotland...")
    all_tenders.extend(ingest_pcs_scotland())
    
    print("Ingesting from Sell2Wales...")
    all_tenders.extend(ingest_sell2wales())
    
    print("Ingesting from SAM.gov...")
    all_tenders.extend(ingest_sam_gov())
    
    print("Ingesting from AusTender...")
    all_tenders.extend(ingest_austender())
    
    # Process and store tenders
    stored_count = 0
    matched_count = 0
    
    for raw_tender in all_tenders:
        source = raw_tender.get("source", "unknown")
        normalized = normalize_tender_data(raw_tender, source)
        tender_id = store_tender(normalized)
        
        if tender_id:
            stored_count += 1
            process_tender_matches(tender_id, normalized)
            matched_count += 1
    
    print(f"Tender ingestion completed: {stored_count} new tenders stored, {matched_count} matches processed")
    return stored_count, matched_count


if __name__ == "__main__":
    ingest_all_tenders()

