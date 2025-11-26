"""
Tender Ingestion Service

This service retrieves tender opportunities from multiple public procurement APIs,
processes them, and matches them against user-defined keywords.

This should be run as a scheduled job (e.g., daily via cron or scheduled task).
"""

import os
import json
import time
import requests
from datetime import datetime, timedelta, timezone, date
from typing import List, Dict, Any, Optional
from supabase import create_client, Client
from dotenv import load_dotenv
import traceback
from urllib.parse import urlencode
from decimal import Decimal
import zipfile
import xml.etree.ElementTree as ET
from io import BytesIO
import gzip
import tarfile
from postgrest.exceptions import APIError as PostgrestAPIError
from config.settings import UK_TIMEZONE, ENABLE_TENDER_INGESTION, FILTER_UK_ONLY

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().strip('"').strip("'")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip().strip('"').strip("'")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

LOOKBACK_DAYS = int(os.getenv("TENDER_LOOKBACK_DAYS", "7"))
SAM_GOV_API_KEY = os.getenv("SAM_GOV_API_KEY", "").strip().strip('"').strip("'")
ANON_SUMMARY_MODE = os.getenv("ANON_SUMMARY_MODE", "matched").strip().lower()
if ANON_SUMMARY_MODE not in {"none", "matched", "all"}:
    ANON_SUMMARY_MODE = "matched"
FIND_TENDER_STAGES = os.getenv("FIND_TENDER_STAGES", "tender").strip()

# Lightweight CPV section descriptors (first two digits) for anonymised summaries
CPV_SECTIONS = {
    "03": "agricultural and farming goods",
    "09": "petroleum and fuel products",
    "14": "mining, basic metals and related products",
    "15": "food, beverages and tobacco",
    "18": "clothing and accessories",
    "30": "office and computing equipment",
    "33": "medical equipment",
    "34": "transport equipment and auxiliary products",
    "35": "security, defence and safety services",
    "45": "construction work",
    "50": "repair and maintenance services",
    "60": "transport services",
    "71": "architecture and engineering services",
    "72": "IT services",
    "73": "research and consultancy services",
    "79": "business services",
    "80": "education and training services",
    "85": "health and social work services",
    "90": "cleaning, environmental and waste services",
    "92": "recreational, cultural and sporting services",
    "98": "miscellaneous services",
}

LANGUAGE_NAMES = {
    "EN": "English",
    "ENG": "English",
    "EN-GB": "English",
    "PL": "Polish",
    "POL": "Polish",
    "FR": "French",
    "FRA": "French",
    "DE": "German",
    "GER": "German",
    "DEU": "German",
    "ES": "Spanish",
    "SPA": "Spanish",
    "IT": "Italian",
    "ITA": "Italian",
}

COUNTRY_NAMES = {
    "GBR": "United Kingdom",
    "UK": "United Kingdom",
    "GB": "United Kingdom",
    "POL": "Poland",
    "DEU": "Germany",
    "DE": "Germany",
    "FRA": "France",
    "FR": "France",
    "ESP": "Spain",
    "ES": "Spain",
    "IRL": "Ireland",
    "ITA": "Italy",
    "LUX": "Luxembourg",
    "NLD": "Netherlands",
    "BEL": "Belgium",
    "USA": "United States",
    "AUS": "Australia",
}


def _generate_fallback_title(data: Dict[str, Any]) -> str:
    current = (data.get("title") or "").strip()
    if current and current.lower() != "ted tender notice":
        return current

    descriptor = (
        data.get("category_label")
        or data.get("category")
        or data.get("sector")
        or "Public sector"
    )
    if isinstance(descriptor, str):
        descriptor = descriptor.strip()
    else:
        descriptor = str(descriptor)

    location = (
        data.get("location_name")
        or data.get("location")
        or (data.get("metadata") or {}).get("location_name")
        or "Europe"
    )
    if isinstance(location, str):
        location = location.strip()
    else:
        location = str(location)

    descriptor_clean = descriptor.capitalize() if descriptor else "Public sector"
    location_clean = location or "Europe"
    fallback = f"{descriptor_clean} opportunity in {location_clean}"
    data["title"] = fallback
    return fallback


def _safe_language_name(code: str | None) -> str | None:
    if not code:
        return None
    norm = code.strip().upper()
    if not norm:
        return None
    if norm in LANGUAGE_NAMES:
        return LANGUAGE_NAMES[norm]
    try:
        import pycountry  # type: ignore
        lang = (
            pycountry.languages.get(alpha_3=norm.lower())
            or pycountry.languages.get(alpha_2=norm.lower())
        )
        if lang and getattr(lang, "name", None):
            return lang.name
    except Exception:
        pass
    return norm


def _safe_country_name(code: str | None) -> str | None:
    if not code:
        return None
    norm = code.strip().upper()
    if not norm:
        return None
    if norm in COUNTRY_NAMES:
        return COUNTRY_NAMES[norm]
    try:
        import pycountry  # type: ignore
        country = (
            pycountry.countries.get(alpha_3=norm)
            or pycountry.countries.get(alpha_2=norm)
        )
        if country and getattr(country, "name", None):
            return country.name
    except Exception:
        pass
    return norm


def _cpv_section_label(code: str | None) -> str | None:
    if not code:
        return None
    digits = "".join(ch for ch in str(code) if ch.isdigit())
    if len(digits) < 2:
        return None
    return CPV_SECTIONS.get(digits[:2])


def _coerce_float(value: Any) -> float | None:
    """Convert Supabase/JSON numeric values (including Decimal or strings) to float."""
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


def _normalize_str_list(value: Any) -> list[str]:
    """Ensure keyword/category/location values are iterable lower-case lists."""
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


def generate_anonymised_summary(tender_data: Dict[str, Any]) -> str:
    """
    Generate an anonymised summary of the tender for notifications.
    The summary should tease the opportunity without revealing sensitive or paid details.
    """
    metadata = tender_data.get("metadata") or {}
    category_code = tender_data.get("category") or metadata.get("category")
    category_label = (
        tender_data.get("category_label")
        or metadata.get("category_label")
        or _cpv_section_label(category_code)
    )
    location_name = (
        tender_data.get("location_name")
        or metadata.get("location_name")
        or tender_data.get("location")
    )
    deadline = tender_data.get("deadline")
    value_amount = _coerce_float(tender_data.get("value_amount"))
    value_currency = (tender_data.get("value_currency") or "EUR").upper()

    sentences: list[str] = []

    opening = "Public sector opportunity"
    if category_label:
        opening += f" focused on {category_label}"
    elif category_code:
        opening += f" (CPV {category_code})"

    if location_name:
        opening += f" in {location_name}"

    sentences.append(opening + ".")

    if deadline:
        try:
            deadline_dt = datetime.fromisoformat(str(deadline).replace("Z", "+00:00"))
            sentences.append(f"Responses are expected by {deadline_dt.strftime('%d %b %Y')}.")
        except Exception:
            sentences.append(f"Responses are expected by {deadline}.")

    if value_amount:
        rounded_value = round(value_amount, -3) if value_amount > 1000 else value_amount
        sentences.append(f"Estimated contract value around {value_currency} {rounded_value:,.0f}.")

    if not category_label and not category_code:
        sentences.append("Further scope details are provided after purchase.")

    return " ".join(sentences)


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


def _parse_ted_notice(source: str, notice: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse a TED format notice into our internal raw_data shape expected by normalize_tender_data().
    TED format is used by Sell2Wales and PCS Scotland when outputType=1.
    """
    form_section = notice.get("Form_Section") or {}
    
    # TED notices can have different form types (F02, F03, F05, etc.)
    # F02 is Contract Notice, F03 is Contract Award, etc.
    ted_data = None
    form_type = None
    
    for key in form_section.keys():
        if key.startswith("F"):
            ted_data = form_section[key]
            form_type = key
            break
    
    if not ted_data:
        return {
            "id": None,
            "title": "",
            "description": "",
            "deadline": None,
            "published_date": None,
            "value": None,
            "value_currency": "GBP",
            "location": None,
            "category": None,
            "sector": None,
            "full_data": notice,
            "api_version": "ted",
        }
    
    # Extract contracting body info
    contracting_body = ted_data.get("Contracting_Body") or {}
    address_cb = contracting_body.get("Address_Contracting_Body") or {}
    
    # Extract object of contract (tender details)
    # Object_Contract can be a dict or a list of dicts
    object_contract_raw = ted_data.get("Object_Contract")
    if isinstance(object_contract_raw, list):
        object_contract = object_contract_raw[0] if object_contract_raw else {}
    else:
        object_contract = object_contract_raw or {}
    
    # Extract procedure info
    procedure = ted_data.get("Procedure") or {}
    
    # Extract complementary info
    complementary = ted_data.get("Complementary_Info") or {}
    
    # Build location from contracting body address
    location_parts = []
    if address_cb.get("Town"):
        location_parts.append(address_cb["Town"])
    if address_cb.get("Postal_Code"):
        location_parts.append(address_cb["Postal_Code"])
    nuts = address_cb.get("Nuts") or {}
    nuts_code = None
    if isinstance(nuts, dict):
        nuts_code = nuts.get("Code")
        if nuts_code:
            location_parts.append(nuts_code)
    elif isinstance(nuts, str):
        nuts_code = nuts
        location_parts.append(nuts_code)
    
    # Extract country from address
    country = None
    country_dict = address_cb.get("Country") or {}
    if isinstance(country_dict, dict):
        country = country_dict.get("Value") or country_dict.get("Code") or country_dict.get("P")
    elif isinstance(country_dict, str):
        country = country_dict
    
    # Add country to location if available
    if country:
        location_parts.append(country)
    
    location = ", ".join(location_parts) if location_parts else None
    
    # Store country and NUTS code in raw_data for UK detection
    raw_data_country = country
    raw_data_nuts = nuts_code
    
    # Extract title and description
    title = object_contract.get("Title") or ""
    if isinstance(title, dict):
        title = title.get("P") or ""
    
    description = object_contract.get("Short_Descr") or ""
    if isinstance(description, dict):
        description = description.get("P") or ""
    
    # Extract value
    val_total = object_contract.get("Val_Total") or object_contract.get("Val_Estimated_Total") or {}
    value = None
    value_currency = "GBP"
    if isinstance(val_total, dict):
        value = val_total.get("Value") or val_total.get("Text")
        if value and isinstance(value, str):
            # Try to extract numeric value from string
            try:
                import re
                match = re.search(r'[\d,]+\.?\d*', value)
                if match:
                    value = float(match.group().replace(',', ''))
            except:
                value = None
        currency = val_total.get("Currency") or {}
        if isinstance(currency, dict):
            value_currency = currency.get("Value") or "GBP"
    
    # Extract CPV codes (category)
    cpv_main = object_contract.get("CPV_Main") or {}
    category = None
    if isinstance(cpv_main, dict):
        cpv_code = cpv_main.get("CPV_Code") or {}
        if isinstance(cpv_code, dict):
            category = cpv_code.get("Code")
    
    # Extract deadline
    deadline = None
    time_receipt = procedure.get("Time_Receipt_Tenders") or procedure.get("Date_Receipt_Tenders")
    if time_receipt:
        if isinstance(time_receipt, str):
            deadline = time_receipt
        elif isinstance(time_receipt, dict):
            deadline = time_receipt.get("Text") or time_receipt.get("Value")
    
    # Extract published date
    published_date = complementary.get("Date_Dispatch_Notice")
    
    # Generate a unique ID from the notice
    notice_id = None
    if complementary.get("Notice_Number_OJ"):
        notice_id = f"{source}-{complementary['Notice_Number_OJ']}"
    else:
        # Use hash of title + org name as fallback
        import hashlib
        org_name = address_cb.get("OfficialName") or ""
        id_str = f"{source}-{org_name}-{title}-{published_date}"
        notice_id = hashlib.md5(id_str.encode()).hexdigest()[:16]
    
    return {
        "id": notice_id,
        "title": title,
        "description": description,
        "deadline": deadline,
        "published_date": published_date,
        "value": value,
        "value_currency": value_currency,
        "location": location,
        "category": category,
        "sector": contracting_body.get("CA_Activity", {}).get("Value") if isinstance(contracting_body.get("CA_Activity"), dict) else None,
        "full_data": notice,
        "api_version": "ted",
        "form_type": form_type,
        "country": raw_data_country,  # Store country for UK detection
        "nuts_code": raw_data_nuts,  # Store NUTS code for UK detection
    }


def ingest_ted_tenders() -> List[Dict[str, Any]]:
    """
    Ingest tenders from TED (Tenders Electronic Daily - EU)
    Uses bulk XML download for daily packages
    API: https://ted.europa.eu/api/documentation/index.html
    Bulk download: https://ted.europa.eu/packages/daily/{yyyynnnnn}
    """
    tenders = []
    try:
        # Get today's OJ S number (format: yyyynnnnn, e.g., 202500219 for 13/11/2025)
        # Format: {year}{sequential_number} where sequential_number is 3 digits
        # The sequential number increases by 1 each day, but doesn't match day of year
        # For 13/11/2025: OJ S = 202500219
        # For 14/11/2025: OJ S = 202500220
        # Calculation: Use a reference date to calculate the offset
        # Reference: 13/11/2025 = 219, day_of_year = 317
        # So: OJ_S_number = day_of_year - 98 (for 2025)
        now = datetime.now(timezone.utc)
        
        # Try last 7 days to catch any missed days
        for days_ago in range(7):
            target_date = now - timedelta(days=days_ago)
            year = target_date.year
            day_of_year = target_date.timetuple().tm_yday  # 1-366
            
            # Calculate OJ S number based on reference date
            # Reference: 13/11/2025 has day_of_year=317 and OJ_S=219
            # So offset = 219 - 317 = -98 for 2025
            # Format: {year}{3-digit-number} e.g., 202500219
            if year == 2025:
                offset = -98  # Calculated from reference date: 13/11/2025 = day 317, OJ_S = 219
            else:
                # For other years, estimate based on similar calculation
                # This is approximate and may need adjustment per year
                offset = -98
            
            oj_s_sequential = day_of_year + offset
            # Ensure it's positive and format as 3 digits with leading zeros
            if oj_s_sequential < 1:
                oj_s_sequential = 1
            # Format: year + 5-digit number (e.g., 202500219 for 13/11/2025)
            oj_s_number = f"{year}{oj_s_sequential:05d}"
            
            bulk_url = f"https://ted.europa.eu/packages/daily/{oj_s_number}"
            
            try:
                print(f"Trying TED bulk download: {bulk_url}")
                resp = requests.get(bulk_url, timeout=120, headers={
                    "User-Agent": "BidWell/1.0 (+contact@bidwell.app)"
                })
                
                if resp.status_code == 404:
                    # This OJ S number doesn't exist, try next day
                    continue
                
                if resp.status_code == 406:
                    # Not Acceptable - might be wrong format, try next day
                    continue
                
                resp.raise_for_status()
                
                content_type = resp.headers.get("Content-Type", "").lower()
                print(f"  Content-Type: {content_type}, Size: {len(resp.content)} bytes")
                
                # Handle gzip/tar.gz format (TED uses application/gzip)
                if "gzip" in content_type or "application/gzip" in content_type:
                    # Try to decompress and extract
                    try:
                        # First, try as tar.gz (most common format)
                        with tarfile.open(fileobj=BytesIO(resp.content), mode='r:gz') as tar:
                            members = tar.getmembers()
                            print(f"  Found {len(members)} files in tar.gz archive")
                            xml_count = 0
                            for member in members:
                                if member.name.endswith(".xml"):
                                    xml_count += 1
                                    try:
                                        xml_content = tar.extractfile(member).read()
                                        root = ET.fromstring(xml_content)
                                        
                                        tender_data = _parse_ted_xml(root)
                                        if tender_data:
                                            tenders.append({"source": "TED", **tender_data})
                                    except Exception as e:
                                        print(f"  Error parsing TED XML file {member.name}: {e}")
                                        continue
                            print(f"  Processed {xml_count} XML files, extracted {len([t for t in tenders if t.get('source') == 'TED'])} tenders")
                    except Exception as tar_err:
                        # If tar.gz fails, try as plain gzip
                        print(f"  tar.gz extraction failed: {tar_err}, trying plain gzip...")
                        try:
                            decompressed = gzip.decompress(resp.content)
                            root = ET.fromstring(decompressed)
                            tender_data = _parse_ted_xml(root)
                            if tender_data:
                                tenders.append({"source": "TED", **tender_data})
                        except Exception as e:
                            print(f"  Error decompressing gzip: {e}")
                            continue
                
                # Check if it's a ZIP file
                elif content_type.startswith("application/zip"):
                    # Extract ZIP and parse XML files
                    with zipfile.ZipFile(BytesIO(resp.content)) as zip_file:
                        for xml_file_name in zip_file.namelist():
                            if xml_file_name.endswith(".xml"):
                                try:
                                    xml_content = zip_file.read(xml_file_name)
                                    root = ET.fromstring(xml_content)
                                    
                                    # Parse TED XML (simplified - TED has complex schema)
                                    tender_data = _parse_ted_xml(root)
                                    if tender_data:
                                        tenders.append({"source": "TED", **tender_data})
                                except Exception as e:
                                    print(f"Error parsing TED XML file {xml_file_name}: {e}")
                                    continue
                else:
                    # Direct XML response
                    try:
                        root = ET.fromstring(resp.content)
                        tender_data = _parse_ted_xml(root)
                        if tender_data:
                            tenders.append({"source": "TED", **tender_data})
                    except Exception as e:
                        print(f"Error parsing direct XML: {e}")
                        continue
                
                print(f"TED ingestion: Fetched {len(tenders)} tenders from {target_date.date()}")
                # If we got data, we can stop trying older dates
                if tenders:
                    break
                    
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    continue
                raise
            except Exception as e:
                print(f"Error downloading TED bulk package {oj_s_number}: {e}")
                continue
        
        if not tenders:
            print("TED ingestion: No tenders found in last 7 days (may need to use search API instead)")
    
    except Exception as e:
        print(f"Error ingesting TED tenders: {e}")
        traceback.print_exc()
    
    return tenders


def _parse_ted_xml(root) -> Dict[str, Any] | None:
    """
    Parse TED XML notice into our internal format.
    Handles both legacy TED schema and modern eForms (UBL 2.3 ContractNotice).
    """

    def _clean_text(node: ET.Element | None) -> str | None:
        if node is None:
            return None
        text = " ".join(t.strip() for t in node.itertext() if t and t.strip())
        return text or None

    def _collect_lang_map(nodes: List[ET.Element]) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for elem in nodes:
            text = _clean_text(elem)
            if not text:
                continue
            lang = (
                elem.attrib.get("languageID")
                or elem.attrib.get("lang")
                or elem.attrib.get("{http://www.w3.org/XML/1998/namespace}lang")
                or ""
            )
            lang = lang.strip().upper() or "UNSPECIFIED"
            result[lang] = text
        return result

    def _prefer_language(lang_map: Dict[str, str], preferred_code: str | None = None) -> tuple[str | None, str | None]:
        if not lang_map:
            return None, None
        if preferred_code:
            preferred_code = preferred_code.strip().upper()
            for code in (preferred_code, "EN", "ENG", "EN-GB"):
                if code in lang_map:
                    return lang_map[code], code
        for code_candidate in ("EN", "ENG", "EN-GB"):
            if code_candidate in lang_map:
                return lang_map[code_candidate], code_candidate
        # Fallback to first entry
        code, text = next(iter(lang_map.items()))
        return text, code

    try:
        tag_root = root.tag.split("}")[-1].lower()
        if tag_root == "contractnotice":
            ns = {
                "cn": "urn:oasis:names:specification:ubl:schema:xsd:ContractNotice-2",
                "cac": "urn:oasis:names:specification:ubl:schema:xsd:CommonAggregateComponents-2",
                "cbc": "urn:oasis:names:specification:ubl:schema:xsd:CommonBasicComponents-2",
                "efac": "http://data.europa.eu/p27/eforms-ubl-extension-aggregate-components/1",
                "efbc": "http://data.europa.eu/p27/eforms-ubl-extension-basic-components/1",
            }

            notice_number = (
                _clean_text(root.find("cbc:ID[@schemeName='notice-id']", ns))
                or _clean_text(root.find("cbc:ID", ns))
            )
            publication_id = _clean_text(root.find(".//efbc:NoticePublicationID", ns))
            notice_language_code = _clean_text(root.find("cbc:NoticeLanguageCode", ns))
            notice_language_name = _safe_language_name(notice_language_code)

            publication_date = (
                _clean_text(root.find(".//efbc:PublicationDate", ns))
                or _clean_text(root.find("cbc:IssueDate", ns))
            )

            deadline_date = _clean_text(root.find(".//cac:TenderSubmissionDeadlinePeriod/cbc:EndDate", ns))
            deadline_time = _clean_text(root.find(".//cac:TenderSubmissionDeadlinePeriod/cbc:EndTime", ns))
            deadline_iso = None
            if deadline_date:
                deadline_iso = deadline_date
                if deadline_time:
                    deadline_iso = f"{deadline_date}T{deadline_time}"

            procurement_project = root.find(".//cac:ProcurementProject", ns)
            title_map = _collect_lang_map(list(procurement_project.findall("cbc:Name", ns)) if procurement_project is not None else [])
            description_map = _collect_lang_map(list(procurement_project.findall("cbc:Description", ns)) if procurement_project is not None else [])
            title_text, title_lang = _prefer_language(title_map, notice_language_code)
            description_text, _ = _prefer_language(description_map, notice_language_code)

            cpv_codes = [
                _clean_text(elem)
                for elem in root.findall(".//cac:ProcurementProject//cbc:ItemClassificationCode[@listName='cpv']", ns)
            ]
            cpv_codes = [code for code in cpv_codes if code]
            main_cpv = cpv_codes[0] if cpv_codes else None
            category_label = _cpv_section_label(main_cpv)

            location_desc_map = _collect_lang_map(
                list(root.findall(".//cac:ProcurementProject//cac:RealizedLocation/cbc:Description", ns))
            )
            location_desc, _ = _prefer_language(location_desc_map, notice_language_code)
            country_code = _clean_text(
                root.find(".//cac:ProcurementProject//cac:RealizedLocation/cac:Address/cbc:IdentificationCode", ns)
            )
            country_name = _safe_country_name(country_code)
            nuts_code = _clean_text(
                root.find(".//cac:ProcurementProject//cac:RealizedLocation/cac:Address/cbc:CountrySubentityCode", ns)
            )

            # Buyer information: resolve contracting party organisation
            organisations: Dict[str, Dict[str, Any]] = {}
            for org in root.findall(".//efac:Organizations/efac:Organization", ns):
                company = org.find("efac:Company", ns)
                if company is None:
                    continue
                org_id = _clean_text(company.find("cac:PartyIdentification/cbc:ID", ns))
                if not org_id:
                    continue
                org_name_map = _collect_lang_map(list(company.findall("cac:PartyName/cbc:Name", ns)))
                org_name, org_lang = _prefer_language(org_name_map, notice_language_code)
                address = company.find("cac:PostalAddress", ns)
                contact = company.find("cac:Contact", ns)
                organisations[org_id] = {
                    "id": org_id,
                    "name": org_name,
                    "language": _safe_language_name(org_lang),
                    "address": {
                        "streetAddress": _clean_text(address.find("cbc:StreetName", ns)) if address is not None else None,
                        "locality": _clean_text(address.find("cbc:CityName", ns)) if address is not None else None,
                        "postalCode": _clean_text(address.find("cbc:PostalZone", ns)) if address is not None else None,
                        "region": _clean_text(address.find("cbc:CountrySubentity", ns)) if address is not None else None,
                        "country": _safe_country_name(
                            _clean_text(address.find("cac:Country/cbc:IdentificationCode", ns)) if address is not None else None
                        ),
                        "nutsCode": _clean_text(address.find("cbc:CountrySubentityCode", ns)) if address is not None else None,
                    },
                    "contactPoint": {
                        "name": _clean_text(contact.find("cbc:Name", ns)) if contact is not None else None,
                        "email": _clean_text(contact.find("cbc:ElectronicMail", ns)) if contact is not None else None,
                        "telephone": _clean_text(contact.find("cbc:Telephone", ns)) if contact is not None else None,
                        "fax": _clean_text(contact.find("cbc:Telefax", ns)) if contact is not None else None,
                    },
                }

            contracting_org_id = _clean_text(
                root.find(".//cac:ContractingParty/cac:Party/cac:PartyIdentification/cbc:ID", ns)
            )
            buyer_info = organisations.get(contracting_org_id) or next(iter(organisations.values()), {})

            documents = []
            for doc_node in root.findall(".//cac:CallForTendersDocumentReference", ns):
                uri = _clean_text(doc_node.find("cac:Attachment/cac:ExternalReference/cbc:URI", ns))
                title_candidate = _clean_text(doc_node.find("cbc:DocumentDescription", ns)) or _clean_text(doc_node.find("cbc:DocumentType", ns))
                if uri:
                    documents.append(
                        {
                            "title": title_candidate or "Procurement documents",
                            "url": uri,
                        }
                    )

            if not notice_number:
                xml_str = ET.tostring(root, encoding="unicode")[:500]
                notice_number = f"ted-{abs(hash(xml_str))}"

            original_title_language = title_lang
            if title_lang and title_lang.upper() not in {"EN", "ENG", "EN-GB"}:
                descriptor = category_label or (main_cpv and f"CPV {main_cpv}") or "Public sector"
                geo = country_name or location_desc or "Europe"
                title_text = f"{descriptor.capitalize()} opportunity in {geo}"

            return {
                "id": notice_number,
                "title": title_text or (category_label and f"{category_label} opportunity") or "TED Tender Notice",
                "description": description_text or "",
                "description_language": notice_language_name,
                "deadline": deadline_iso,
                "published_date": publication_date or datetime.now(timezone.utc).isoformat(),
                "value": None,
                "value_currency": "EUR",
                "location": location_desc or country_name,
                "location_name": country_name or location_desc,
                "category": main_cpv,
                "category_label": category_label,
                "sector": None,
                "full_data": {
                    "format": "ted_eforms",
                    "notice_number": notice_number,
                    "publication_id": publication_id,
                    "language_code": notice_language_code,
                    "language": notice_language_name,
                    "original_title_language": original_title_language,
                    "original_titles": title_map,
                    "titles": title_map,
                    "descriptions": description_map,
                    "cpv_codes": cpv_codes,
                    "buyer": buyer_info,
                    "tender": {
                        "title": title_text or (category_label and f"{category_label} opportunity"),
                        "description": description_text,
                        "documents": documents,
                        "deadline": deadline_iso,
                        "cpv_codes": cpv_codes,
                        "language": notice_language_name,
                    },
                    "location": {
                        "description": location_desc,
                        "country_code": country_code,
                        "country": country_name,
                        "nuts": nuts_code,
                    },
                },
                "api_version": "ted_eforms",
            }

        # Fallback: legacy TED XML (Form_Section)
        notice_number = None
        title = None
        description = None
        deadline = None
        value = None
        value_currency = "EUR"
        published_date = None
        location = None
        category = None

        for elem in root.iter():
            tag_upper = elem.tag.upper().split("}")[-1]
            text = (elem.text or "").strip() if elem.text else ""
            if not notice_number and "NOTICE" in tag_upper and "ID" in tag_upper and text:
                notice_number = text
            if not title and "TITLE" in tag_upper and text:
                title = text
            if "DESCR" in tag_upper and text:
                description = f"{description} {text}".strip() if description else text
            if not deadline and "DEADLINE" in tag_upper and text:
                deadline = text
            if not published_date and "PUBLICATION" in tag_upper and text:
                published_date = text
            if not value and "VAL" in tag_upper and text:
                try:
                    cleaned = "".join(ch for ch in text if (ch.isdigit() or ch == "." or ch == ","))
                    cleaned = cleaned.replace(",", "")
                    if cleaned:
                        value = float(cleaned)
                except Exception:
                    pass
            if "CURRENCY" in tag_upper and text:
                value_currency = text[:3].upper()
            if not location and ("COUNTRY" in tag_upper or "LOCATION" in tag_upper) and text:
                location = text
            if not category and "CPV" in tag_upper and text:
                category = text

        if not notice_number:
            xml_str = ET.tostring(root, encoding="unicode")[:500]
            notice_number = f"ted-{abs(hash(xml_str))}"

        category_label = _cpv_section_label(category)
        location_name = _safe_country_name(location) or location
        if not title or title.strip().lower() in {"ted tender notice", ""}:
            descriptor = category_label or (category and f"CPV {category}") or "Public sector"
            geo = location_name or "Europe"
            title = f"{descriptor.capitalize()} opportunity in {geo}"

        return {
            "id": notice_number,
            "title": title or "TED Tender Notice",
            "description": description or "",
            "deadline": deadline,
            "published_date": published_date or datetime.now(timezone.utc).isoformat(),
            "value": value,
            "value_currency": value_currency,
            "location": location,
            "category": category,
            "category_label": category_label,
            "location_name": location_name,
            "sector": None,
            "full_data": {
                "format": "ted_legacy",
                "raw_xml": ET.tostring(root, encoding="unicode"),
            },
            "api_version": "ted_legacy",
        }
    except Exception as e:
        print(f"  Error parsing TED XML: {e}")
        traceback.print_exc()
        return None


def ingest_find_a_tender() -> List[Dict[str, Any]]:
    """
    Ingest tenders from Find a Tender (UK)
    API: https://www.find-tender.service.gov.uk/apidocumentation
    """
    tenders: List[Dict[str, Any]] = []
    try:
        base_url = "https://www.find-tender.service.gov.uk/api/1.0/ocdsReleasePackages"
        headers = {
            "Accept": "application/json",
            "User-Agent": "BidWell/1.0 (+contact@bidwell.app)",
        }
        now = datetime.now(timezone.utc)

        def _daily_windows(days: int) -> List[tuple[datetime, datetime]]:
            windows: List[tuple[datetime, datetime]] = []
            for offset in range(days):
                target_day = (now - timedelta(days=offset)).date()
                start_dt = datetime(
                    target_day.year,
                    target_day.month,
                    target_day.day,
                    0,
                    0,
                    0,
                    tzinfo=timezone.utc,
                )
                end_dt = datetime(
                    target_day.year,
                    target_day.month,
                    target_day.day,
                    23,
                    59,
                    59,
                    tzinfo=timezone.utc,
                )
                if target_day == now.date():
                    end_dt = now
                windows.append((start_dt, end_dt))
            windows.sort(key=lambda win: win[0])
            return windows

        stages = [stage.strip() for stage in FIND_TENDER_STAGES.split(",") if stage.strip()]

        for start_dt, end_dt in _daily_windows(max(LOOKBACK_DAYS, 1)):
            cursor = None
            day_total = 0
            while True:
                params = {
                    "updatedFrom": start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                    "updatedTo": end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                    "limit": 100,
                }
                if cursor:
                    params["cursor"] = cursor
                if stages:
                    params["stages"] = ",".join(stages)

                resp = requests.get(base_url, params=params, headers=headers, timeout=60)
                if resp.status_code == 429:
                    retry_after_header = resp.headers.get("Retry-After")
                    try:
                        wait_seconds = int(retry_after_header) if retry_after_header else 30
                    except ValueError:
                        wait_seconds = 30
                    wait_seconds = max(5, min(wait_seconds, 300))
                    print(f"Find a Tender API rate limit hit, waiting {wait_seconds}s before retry...")
                    time.sleep(wait_seconds)
                    continue

                resp.raise_for_status()
                data = resp.json()

                releases: List[Dict[str, Any]] = []
                data_dict = data if isinstance(data, dict) else {}
                release_packages = data_dict.get("releasePackages")
                if isinstance(release_packages, list):
                    for package in release_packages:
                        releases.extend(package.get("releases") or [])
                elif isinstance(data, list):
                    for package in data:
                        if isinstance(package, dict):
                            releases.extend(package.get("releases") or [])
                else:
                    releases.extend(data_dict.get("releases") or [])

                if not releases:
                    # No data for this window/page
                    pass
                else:
                    for rel in releases:
                        try:
                            parsed = _parse_ocds_release("FindATender", rel)
                            tenders.append({"source": "FindATender", **parsed})
                        except Exception as parse_err:
                            print(f"Failed parsing Find a Tender release: {parse_err}")
                            continue
                    day_total += len(releases)

                pagination_block = data_dict.get("pagination") if isinstance(data_dict.get("pagination"), dict) else {}
                next_cursor = (
                    data_dict.get("nextCursor")
                    or data_dict.get("next_cursor")
                    or data_dict.get("cursor")
                    or pagination_block.get("nextCursor")
                )

                if not next_cursor:
                    break

                cursor = next_cursor
                # Gentle pause between paginated requests
                time.sleep(0.2)

            if day_total:
                print(
                    f"Find a Tender ingestion: fetched {day_total} releases between "
                    f"{start_dt.strftime('%Y-%m-%d %H:%M:%S')} and {end_dt.strftime('%Y-%m-%d %H:%M:%S')}"
                )
    except Exception as e:
        print(f"Error ingesting Find a Tender: {e}")
        traceback.print_exc()
    return tenders


def ingest_contracts_finder() -> List[Dict[str, Any]]:
    """
    Ingest tenders from ContractsFinder (UK)
    API: https://www.contractsfinder.service.gov.uk/apidocumentation/V2
    Note: ContractsFinder only provides OCDS format, not TED format like Sell2Wales/PCS Scotland
    """
    tenders: List[Dict[str, Any]] = []
    try:
        # ContractsFinder only provides OCDS format via their API
        # There is no TED format endpoint available (unlike Sell2Wales/PCS Scotland)
        base_url = "https://www.contractsfinder.service.gov.uk/Published/Notices/OCDS/Search"
        published_to = datetime.now(timezone.utc)
        published_from = published_to - timedelta(days=LOOKBACK_DAYS)
        
        # Pagination support
        page = 1
        page_size = 100
        
        while True:
            params = {
                "publishedFrom": published_from.strftime("%Y-%m-%dT00:00:00Z"),
                "publishedTo": published_to.strftime("%Y-%m-%dT23:59:59Z"),
                "pageSize": page_size,
                "order": "desc",
                "page": page,
            }
            url = f"{base_url}?{urlencode(params)}"
            resp = requests.get(url, timeout=60, headers={
                "Accept": "application/json",
                "User-Agent": "BidWell/1.0 (+contact@bidwell.app)"
            })
            resp.raise_for_status()
            data = resp.json()
            releases = data.get("releases") or []
            
            if not releases:
                break
            
            for rel in releases:
                tenders.append({"source": "ContractsFinder", **_parse_ocds_release("ContractsFinder", rel)})
            
            # Check if there are more pages
            total_pages = data.get("totalPages") or 1
            if page >= total_pages or len(releases) < page_size:
                break
            
            page += 1
        
        print(f"ContractsFinder ingestion: Fetched {len(tenders)} tenders (OCDS format)")
    except Exception as e:
        print(f"Error ingesting Contracts Finder: {e}")
        traceback.print_exc()
    return tenders


def ingest_pcs_scotland() -> List[Dict[str, Any]]:
    """
    Ingest tenders from PCS Scotland (Public Contracts Scotland)
    API: https://api.publiccontractsscotland.gov.uk
    Uses TED output format (outputType=1) for richer data
    """
    tenders: List[Dict[str, Any]] = []
    try:
        import warnings
        from urllib3.exceptions import InsecureRequestWarning
        warnings.simplefilter('ignore', InsecureRequestWarning)
        
        base_url = "https://api.publiccontractsscotland.gov.uk/v1/Notices"
        now = datetime.now(timezone.utc)
        
        # Fetch current and previous 2 months in TED format
        for m in range(0, 3):
            dt = (now.replace(day=15) - timedelta(days=30 * m))
            date_from = dt.strftime("%m-%Y")  # mm-yyyy
            params = {
                "dateFrom": date_from,
                "noticeType": 2,     # Contract Notice
                "outputType": 1,     # TED format (not OCDS)
                "locale": 2057,      # English
            }
            url = f"{base_url}?{urlencode(params)}"
            
            # PCS Scotland has SSL certificate issues, so we disable verification
            resp = requests.get(url, timeout=60, verify=False, headers={
                "Accept": "application/json",
                "User-Agent": "BidWell/1.0 (+contact@bidwell.app)",
                "Referer": "https://www.publiccontractsscotland.gov.uk"
            })
            resp.raise_for_status()
            data = resp.json()
            
            notices = data.get("notices") or []
            for notice in notices:
                tenders.append({"source": "PCSScotland", **_parse_ted_notice("PCSScotland", notice)})
                
        print(f"PCS Scotland ingestion: Fetched {len(tenders)} tenders")
    except Exception as e:
        print(f"Error ingesting PCS Scotland: {e}")
        traceback.print_exc()
    
    return tenders


def ingest_sell2wales() -> List[Dict[str, Any]]:
    """
    Ingest tenders from Sell2Wales
    API: https://api.sell2wales.gov.wales/v1
    Uses TED output format (outputType=1) for richer data
    """
    tenders: List[Dict[str, Any]] = []
    try:
        base_url = "https://api.sell2wales.gov.wales/v1/Notices"
        now = datetime.now(timezone.utc)
        
        # Fetch current and previous 2 months in TED format
        for m in range(0, 3):
            dt = (now.replace(day=15) - timedelta(days=30*m))
            date_from = dt.strftime("%m-%Y")  # mm-yyyy
            params = {
                "dateFrom": date_from,
                "noticeType": 2,     # Contract Notice
                "outputType": 1,     # TED format (not OCDS)
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
            
            notices = data.get("notices") or []
            for notice in notices:
                tenders.append({"source": "Sell2Wales", **_parse_ted_notice("Sell2Wales", notice)})
                
        print(f"Sell2Wales ingestion: Fetched {len(tenders)} tenders")
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
    tenders: List[Dict[str, Any]] = []
    if not SAM_GOV_API_KEY:
        print("SAM.gov ingestion skipped: SAM_GOV_API_KEY not set")
        return tenders

    try:
        base_url = "https://api.sam.gov/opportunities/v2/search"
        now = datetime.now(timezone.utc)
        posted_to = now.strftime("%m/%d/%Y")
        posted_from = (now - timedelta(days=LOOKBACK_DAYS)).strftime("%m/%d/%Y")

        # Try API key in header first (preferred method)
        params = {
            "postedFrom": posted_from,
            "postedTo": posted_to,
            "ptype": "o",  # 'o' for opportunities
            "limit": 100,
            "offset": 0,
        }
        
        headers = {
            "Accept": "application/json",
            "User-Agent": "BidWell/1.0 (+contact@bidwell.app)",
            "X-API-Key": SAM_GOV_API_KEY,
        }

        while True:
            resp = requests.get(base_url, params=params, headers=headers, timeout=60)
            
            if resp.status_code == 403:
                # Try with API key in params as fallback
                if "api_key" not in params:
                    params["api_key"] = SAM_GOV_API_KEY
                    headers.pop("X-API-Key", None)
                    resp = requests.get(base_url, params=params, headers=headers, timeout=60)
                
                if resp.status_code == 403:
                    error_detail = "Unknown"
                    try:
                        error_json = resp.json()
                        error_detail = error_json.get("error", {}).get("message", "API key may be invalid or expired")
                    except:
                        pass
                    print(f"SAM.gov ingestion failed: Access forbidden - {error_detail}")
                    print("Please verify:")
                    print("  1. API key is activated (check email from SAM.gov)")
                    print("  2. API key has 'opportunities' endpoint permissions")
                    print("  3. Generate a new key at: https://open.gsa.gov/api/get-opportunities-public-api/")
                    break

            resp.raise_for_status()
            data = resp.json()
            opportunities = data.get("opportunitiesData") or data.get("opportunities") or []

            for opp in opportunities:
                raw = {
                    "id": opp.get("noticeId") or opp.get("solicitationNumber"),
                    "title": opp.get("title") or "",
                    "description": opp.get("description") or opp.get("fulldescription") or "",
                    "deadline": opp.get("archivedDate") or opp.get("responseDueDate"),
                    "published_date": opp.get("postedDate"),
                    "value": opp.get("baseAndAllOptionsValue") or opp.get("baseValue"),
                    "value_currency": opp.get("currency") or "USD",
                    "location": (
                        (opp.get("placeOfPerformance") or {}).get("city") or
                        opp.get("placeOfPerformanceAddress") or
                        opp.get("placeOfPerformanceCity")
                    ),
                    "category": opp.get("naics") or opp.get("type"),
                    "sector": opp.get("typeOfSetAsideDescription"),
                    "full_data": opp,
                    "api_version": data.get("version") or "v2",
                }
                tenders.append({"source": "SAM.gov", **raw})

            total = data.get("opportunitiesCount") or data.get("totalRecords") or len(opportunities)
            offset = params.get("offset", 0)
            if len(opportunities) == 0 or len(opportunities) + offset >= total:
                break

            params["offset"] = offset + len(opportunities)
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
    def _parse_datetime(value: Any) -> Optional[datetime]:
        if not value:
            return None
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=timezone.utc)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            candidates = [
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d",
            ]
            for fmt in candidates:
                try:
                    dt = datetime.strptime(text.replace("Z", "+0000") if fmt.endswith("%z") else text, fmt)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    else:
                        dt = dt.astimezone(timezone.utc)
                    return dt
                except ValueError:
                    continue
            try:
                return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
            except Exception:
                return None
        return None

    # Determine full payload: prefer embedded `full_data` if present (e.g. OCDS release)
    full_payload = raw_data.get("full_data")
    if isinstance(full_payload, dict):
        original_payload_type = "embedded"
    else:
        full_payload = raw_data
        original_payload_type = "direct"

    deadline_dt = _parse_datetime(
        raw_data.get("deadline")
        or raw_data.get("closing_date")
        or raw_data.get("submission_deadline")
        or raw_data.get("deadline_date")
    )
    published_dt = _parse_datetime(
        raw_data.get("published_date")
        or raw_data.get("publication_date")
        or raw_data.get("published_at")
        or raw_data.get("date")
    )

    normalized = {
        "source": source,
        "external_id": raw_data.get("id") or raw_data.get("notice_id") or raw_data.get("reference"),
        "title": raw_data.get("title") or raw_data.get("name") or raw_data.get("subject", ""),
        "description": raw_data.get("description") or raw_data.get("summary") or "",
        "deadline": deadline_dt.isoformat() if deadline_dt else None,
        "published_date": (published_dt or datetime.now(timezone.utc)).isoformat(),
        "value_amount": _coerce_float(
            raw_data.get("value")
            or raw_data.get("value_amount")
            or raw_data.get("estimated_value")
            or raw_data.get("contract_value")
        ),
        "value_currency": raw_data.get("currency") or raw_data.get("value_currency") or "GBP",
        "location": raw_data.get("location") or raw_data.get("region") or raw_data.get("country"),
        "category": raw_data.get("category") or raw_data.get("cpv_code") or "",
        "sector": raw_data.get("sector") or raw_data.get("industry") or "",
        "full_data": full_payload,  # Store complete raw data for downstream consumers
        "metadata": {
            "ingested_at": datetime.now().isoformat(),
            "source_api_version": raw_data.get("api_version", "unknown"),
            "raw_top_level_keys": sorted(raw_data.keys()),
            "full_payload_type": original_payload_type,
            "full_payload_keys": sorted(full_payload.keys()) if isinstance(full_payload, dict) else [],
            "deadline_parsed": deadline_dt.isoformat() if deadline_dt else None,
            "published_parsed": (published_dt.isoformat() if published_dt else None),
        },
    }

    # Optional enriched metadata
    if raw_data.get("language") or raw_data.get("language_code"):
        normalized["metadata"]["language_code"] = (raw_data.get("language_code") or raw_data.get("language"))
        normalized["metadata"]["language"] = raw_data.get("language") or raw_data.get("description_language")
    if raw_data.get("cpv_codes"):
        normalized["metadata"]["cpv_codes"] = raw_data.get("cpv_codes")
    # Store country and NUTS code for UK detection (from TED parsing)
    if raw_data.get("country"):
        normalized["metadata"]["country"] = raw_data.get("country")
    if raw_data.get("nuts_code"):
        normalized["metadata"]["nuts_code"] = raw_data.get("nuts_code")

    # Additional friendly fields for summaries/display
    location_name = raw_data.get("location_name")
    if not location_name and normalized.get("location"):
        location_name = _safe_country_name(normalized["location"])
    category_label = raw_data.get("category_label") or _cpv_section_label(normalized.get("category"))
    if category_label:
        normalized["metadata"]["category_label"] = category_label
    if location_name:
        normalized["metadata"]["location_name"] = location_name

    normalized["title"] = _generate_fallback_title({
        "title": normalized.get("title"),
        "category_label": category_label,
        "category": normalized.get("category"),
        "location_name": location_name,
        "location": normalized.get("location"),
    })

    summary = raw_data.get("summary")
    summary_context = {
        "title": normalized.get("title"),
        "category": normalized.get("category"),
        "category_label": category_label,
        "location": normalized.get("location"),
        "location_name": location_name,
        "value_amount": normalized.get("value_amount"),
        "value_currency": normalized.get("value_currency"),
        "deadline": normalized.get("deadline"),
    }

    if ANON_SUMMARY_MODE == "all" and not summary:
        summary = generate_anonymised_summary(summary_context)
    normalized["summary"] = summary

    return normalized


# Global cache for duplicate detection during bulk ingestion
_duplicate_cache: Dict[str, Any] = {}
_cache_loaded = False

def _load_duplicate_cache(days: int = 30):
    """
    Load recent tenders into memory cache for fast duplicate detection.
    This is called once at the start of bulk ingestion.
    """
    global _duplicate_cache, _cache_loaded
    try:
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        print(f"Loading duplicate detection cache (last {days} days)...")
        
        # Fetch recent tenders with only essential fields
        recent_tenders = supabase.table("tenders").select(
            "id, source, external_id, title, deadline, value_amount, published_date"
        ).gte("published_date", cutoff_date).execute()
        
        # Organize by source+external_id for O(1) lookup
        _duplicate_cache = {
            "by_source_id": {},  # {(source, external_id): tender}
            "by_date": [],  # List of tenders for fuzzy matching
        }
        
        for tender in (recent_tenders.data or []):
            source = tender.get("source", "")
            external_id = tender.get("external_id", "")
            if source and external_id:
                _duplicate_cache["by_source_id"][(source, external_id)] = tender
            _duplicate_cache["by_date"].append(tender)
        
        _cache_loaded = True
        print(f"Loaded {len(_duplicate_cache['by_date'])} tenders into duplicate cache")
    except Exception as e:
        print(f"Error loading duplicate cache: {e}")
        _duplicate_cache = {"by_source_id": {}, "by_date": []}
        _cache_loaded = True  # Mark as loaded even if empty to avoid retries

def _clear_duplicate_cache():
    """Clear the duplicate cache after bulk ingestion is complete."""
    global _duplicate_cache, _cache_loaded
    _duplicate_cache = {}
    _cache_loaded = False


def _already_ingested_today(force: bool) -> tuple[bool, date]:
    today_uk = datetime.now(UK_TIMEZONE).date()
    if force:
        return False, today_uk
    try:
        res = supabase.table("tender_ingestion_log").select("id").eq("ingested_date", str(today_uk)).limit(1).execute()
        if res.data:
            return True, today_uk
    except Exception as exc:
        print(f"Warning: failed checking ingestion log: {exc}")
    return False, today_uk


def _record_ingestion_date(ingested_date: date):
    try:
        supabase.table("tender_ingestion_log").upsert(
            {
                "ingested_date": str(ingested_date),
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            },
            on_conflict="ingested_date",
        ).execute()
    except Exception as exc:
        print(f"Warning: failed recording ingestion log: {exc}")

def _find_duplicate_tender(tender_data: Dict[str, Any]) -> str | None:
    """
    Find duplicate tender across all platforms using multiple strategies:
    1. Same source + external_id (exact match)
    2. Similar title + same deadline + similar value (cross-platform duplicate)
    3. Same title + same published date (likely duplicate)
    
    Returns the ID of the duplicate tender if found, None otherwise.
    """
    try:
        # Ensure cache is loaded
        if not _cache_loaded:
            _load_duplicate_cache()
        
        # Strategy 1: Exact match (same source + external_id) - O(1) lookup from cache
        source = tender_data.get("source", "")
        external_id = tender_data.get("external_id", "")
        if source and external_id:
            cache_key = (source, external_id)
            if cache_key in _duplicate_cache.get("by_source_id", {}):
                return _duplicate_cache["by_source_id"][cache_key]["id"]
        
        # Strategy 2: Cross-platform duplicate detection (only if we have good title)
        title = (tender_data.get("title") or "").strip().lower()
        if not title or len(title) < 10:
            return None
        
        deadline = tender_data.get("deadline")
        value_amount = tender_data.get("value_amount")
        published_date = tender_data.get("published_date")
        
        # Extract meaningful words from title (longer than 3 chars)
        title_words = set([w for w in title.split() if len(w) > 3])
        if not title_words:
            return None
        
        # Parse dates once
        deadline_dt = None
        if deadline:
            try:
                deadline_dt = datetime.fromisoformat(str(deadline).replace("Z", "+00:00"))
            except:
                pass
        
        published_dt = None
        if published_date:
            try:
                published_dt = datetime.fromisoformat(str(published_date).replace("Z", "+00:00"))
            except:
                pass
        
        # Only check tenders published on the same day (narrow down candidates)
        candidates = []
        if published_dt:
            published_date_str = published_dt.date().isoformat()
            for existing_tender in _duplicate_cache.get("by_date", []):
                existing_published = existing_tender.get("published_date")
                if existing_published and str(existing_published)[:10] == published_date_str:
                    candidates.append(existing_tender)
        else:
            # If no published date, check last 7 days only
            cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).date()
            for existing_tender in _duplicate_cache.get("by_date", []):
                existing_published = existing_tender.get("published_date")
                if existing_published:
                    try:
                        existing_dt = datetime.fromisoformat(str(existing_published).replace("Z", "+00:00"))
                        if existing_dt.date() >= cutoff:
                            candidates.append(existing_tender)
                    except:
                        pass
        
        # Limit candidates to avoid excessive processing
        if len(candidates) > 100:
            candidates = candidates[:100]  # Only check first 100 candidates
        
        # Fuzzy match on candidates
        for existing_tender in candidates:
            existing_title = (existing_tender.get("title") or "").strip().lower()
            if not existing_title:
                continue
            
            existing_title_words = set([w for w in existing_title.split() if len(w) > 3])
            if not existing_title_words:
                continue
            
            # Quick word overlap check (70% threshold)
            overlap = len(title_words & existing_title_words) / max(len(title_words), len(existing_title_words))
            if overlap < 0.7:
                continue
            
            # Additional checks for stronger match
            match_score = 0
            
            # Same deadline (exact or within 1 day)
            if deadline_dt and existing_tender.get("deadline"):
                try:
                    existing_deadline_dt = datetime.fromisoformat(str(existing_tender["deadline"]).replace("Z", "+00:00"))
                    if abs((deadline_dt - existing_deadline_dt).days) <= 1:
                        match_score += 0.3
                except:
                    pass
            
            # Similar value (within 10%)
            if value_amount and existing_tender.get("value_amount"):
                try:
                    val_diff = abs(float(value_amount) - float(existing_tender["value_amount"]))
                    val_avg = (float(value_amount) + float(existing_tender["value_amount"])) / 2
                    if val_avg > 0 and (val_diff / val_avg) < 0.1:
                        match_score += 0.2
                except:
                    pass
            
            # Same published date
            if published_dt and existing_tender.get("published_date"):
                if str(published_date)[:10] == str(existing_tender["published_date"])[:10]:
                    match_score += 0.2
            
            # If total match score is high enough, consider it a duplicate
            if match_score >= 0.3:  # At least deadline or value match
                print(f"Found cross-platform duplicate: {tender_data['source']} tender '{title[:50]}...' matches {existing_tender['source']} tender (score: {overlap:.2f}, match: {match_score:.2f})")
                return existing_tender["id"]
        
        return None
    except Exception as e:
        print(f"Error finding duplicate tender: {e}")
        traceback.print_exc()
        return None


def _is_uk_tender(tender_data: Dict[str, Any]) -> bool:
    """
    Check if a tender is from the UK.
    Returns True if the tender is from UK, False otherwise.
    """
    source = tender_data.get("source", "").lower()
    
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
    
    # For TED and other sources, check location fields
    location = (tender_data.get("location") or "").lower()
    location_name = ""
    metadata = tender_data.get("metadata") or {}
    if isinstance(metadata, dict):
        location_name = (metadata.get("location_name") or "").lower()
    
    # Check full_data for country information
    full_data = tender_data.get("full_data")
    country_indicators = []
    if isinstance(full_data, dict):
        # Check buyer address
        buyer = full_data.get("buyer") or {}
        if isinstance(buyer, dict):
            address = buyer.get("address") or {}
            if isinstance(address, dict):
                country = (address.get("countryName") or address.get("country") or "").lower()
                if country:
                    country_indicators.append(country)
        
        # Check tender items delivery locations
        tender_info = full_data.get("tender") or {}
        items = tender_info.get("items") or []
        if isinstance(items, list):
            for item in items[:3]:  # Check first 3 items
                if isinstance(item, dict):
                    delivery = item.get("deliveryLocation") or {}
                    if isinstance(delivery, dict):
                        country = (delivery.get("address", {}).get("countryName") or "").lower()
                        if country:
                            country_indicators.append(country)
    
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
    
    # Check all location fields
    all_location_text = f"{location} {location_name} {' '.join(country_indicators)}".lower()
    
    # Check if any UK indicator is present
    for indicator in uk_indicators:
        if indicator in all_location_text:
            return True
    
    # If source is TED and no UK indicator found, it's not UK
    if "ted" in source:
        return False
    
    # Default: if we can't determine, return False (safer to exclude)
    return False


def store_tender(tender_data: Dict[str, Any]) -> str | None:
    """
    Store a tender in the database.
    Returns the tender ID if stored, None otherwise.
    """
    try:
        # Insert new tender
        res = supabase.table("tenders").insert(tender_data).execute()
        
        if res.data:
            tender_id = res.data[0]["id"]
            return tender_id
        return None
    except PostgrestAPIError as e:
        payload = e.args[0] if e.args else {}
        if isinstance(payload, dict) and payload.get("code") == "23505":
            # Duplicate key - ignore for now
            return None
        print(f"Error storing tender: {payload or e}")
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"Error storing tender: {e}")
        traceback.print_exc()
        return None


def match_tender_against_keywords(tender_data: Dict[str, Any], keyword_set: Dict[str, Any]) -> tuple[bool, float, list]:
    """
    Match a tender against a keyword set.
    Matching logic: (one location OR) AND (one industry OR) AND (one keyword OR)
    Returns: (is_match, match_score, matched_keywords)
    """
    # Import from services to avoid duplication
    from services.tender_service import match_tender_against_keywords as service_match
    return service_match(tender_data, keyword_set)


def process_tender_matches(tender_id: str, tender_data: Dict[str, Any]):
    """
    Match a tender against all active keyword sets and create match records.
    """
    try:
        # Get all active keyword sets
        keyword_sets_res = supabase.table("user_tender_keywords").select("*").eq("is_active", True).execute()
        keyword_sets = keyword_sets_res.data or []
        
        summary_updated = False

        for keyword_set in keyword_sets:
            is_match, match_score, matched_keywords = match_tender_against_keywords(tender_data, keyword_set)
            
            if is_match:
                if ANON_SUMMARY_MODE in {"matched", "all"} and not tender_data.get("summary") and not summary_updated:
                    try:
                        summary_text = generate_anonymised_summary(tender_data)
                        if summary_text:
                            supabase.table("tenders").update({"summary": summary_text}).eq("id", tender_id).execute()
                            tender_data["summary"] = summary_text
                            summary_updated = True
                    except Exception as summary_error:
                        print(f"Warning: failed to persist summary for tender {tender_id}: {summary_error}")

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


def ingest_all_tenders(force: bool = False):
    """
    Main function to ingest tenders from all sources (daily bulk fetch).
    After ingestion, matches all new tenders against all user keywords.
    """
    if not ENABLE_TENDER_INGESTION and not force:
        print("Tender ingestion skipped: ENABLE_TENDER_INGESTION flag disabled.")
        return 0, 0, []

    already, today_uk = _already_ingested_today(force)
    if already:
        print(f"Tender ingestion skipped: already ingested on {today_uk.isoformat()}")
        return 0, 0, []

    print(f"Starting daily bulk tender ingestion at {datetime.now().isoformat()}")
    
    all_tenders = []
    
    # Ingest from all sources (bulk fetch)
    if not FILTER_UK_ONLY:
        print("Ingesting from TED...")
        all_tenders.extend(ingest_ted_tenders())
    else:
        print("Skipping TED ingestion because FILTER_UK_ONLY=1")
    
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
    
    print(f"Total tenders fetched from all sources: {len(all_tenders)}")
    
    # Process and store tenders
    stored_count = 0
    new_tender_ids = []
    
    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(days=LOOKBACK_DAYS)
    
    for raw_tender in all_tenders:
        source = raw_tender.get("source", "unknown")
        normalized = normalize_tender_data(raw_tender, source)

        published_dt = normalized.get("published_date")
        deadline_dt = normalized.get("deadline")

        def _to_dt(value: Any) -> Optional[datetime]:
            if not value:
                return None
            try:
                return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except Exception:
                return None

        published_parsed = _to_dt(published_dt)
        deadline_parsed = _to_dt(deadline_dt)

        # Only filter by published date (keep tenders even if deadline passed)
        # Expired tenders will be filtered in the API endpoint, not during ingestion
        if published_parsed and published_parsed < cutoff:
            continue

        # Optional filter to drop TED tenders completely when enabled
        if FILTER_UK_ONLY:
            source_name = (normalized.get("source") or "").lower()
            if "ted" in source_name:
                continue

        # Store tender (checks for duplicates automatically)
        tender_id = store_tender(normalized)
        
        if tender_id:
            # Check if this is a new tender (not a duplicate we already had)
            existing_check = supabase.table("tenders").select("created_at").eq("id", tender_id).execute()
            if existing_check.data:
                created_at = existing_check.data[0].get("created_at")
                if created_at:
                    created_dt = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
                    # If created within last minute, it's new
                    if (now_utc - created_dt).total_seconds() < 60:
                        new_tender_ids.append(tender_id)
            stored_count += 1
    
    print(f"Tender ingestion completed: {stored_count} new tenders stored (after duplicate detection)")
    
    # Match all new tenders against all user keywords
    if new_tender_ids:
        print(f"Matching {len(new_tender_ids)} new tenders against all user keywords...")
        matched_count = 0
        for tender_id in new_tender_ids:
            # Get tender data for matching
            tender_res = supabase.table("tenders").select("*").eq("id", tender_id).execute()
            if tender_res.data:
                tender_data = tender_res.data[0]
                process_tender_matches(tender_id, tender_data)
            matched_count += 1
        print(f"Keyword matching completed: {matched_count} tenders matched against user keywords")
    else:
        print("No new tenders to match (all were duplicates)")
        matched_count = 0
    
    _record_ingestion_date(today_uk)
    
    return stored_count, matched_count, new_tender_ids


if __name__ == "__main__":
    stored, matched, new_ids = ingest_all_tenders(force=True)
    print(f"Stored: {stored}, matched: {matched}, new tender IDs: {len(new_ids)}")

