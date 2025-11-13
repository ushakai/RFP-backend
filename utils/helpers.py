"""
General helper functions
"""
import re
from datetime import datetime, timedelta

def clean_markdown(text: str) -> str:
    """Remove common Markdown formatting such as **bold**, _italic_, `code`, headings, and links.
    Keeps only readable plain text with proper wrapping.
    """
    if not text:
        return ""
    cleaned = text
    
    cleaned = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", cleaned)
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", cleaned)
    cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
    
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"__([^_]+)__", r"\1", cleaned)
    cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
    cleaned = re.sub(r"_([^_]+)_", r"\1", cleaned)
    
    cleaned = re.sub(r"^\s{0,3}#{1,6}\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s{0,3}>\s?", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*[-*+]\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*\d+\.\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", "", cleaned, flags=re.MULTILINE)
    
    cleaned = re.sub(r"\s+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r'\s*([.,!?;:])', r'\1', cleaned)
    cleaned = re.sub(r'^\s+|\s+$', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r' {2,}', ' ', cleaned)
    
    return cleaned.strip()


def format_ai_answer(raw_text: str, max_lines: int = 4) -> str:
    """Normalize AI answers to plain text with limited lines and punctuation."""
    cleaned = clean_markdown(raw_text)
    if not cleaned:
        return ""

    # Collapse excessive whitespace
    cleaned = re.sub(r"[^\S\r\n]+", " ", cleaned)
    cleaned = re.sub(r"\s*\n\s*", "\n", cleaned)

    # Remove repeated punctuation and ellipses
    cleaned = re.sub(r'([.,!?;:])\1+', r'\1', cleaned)
    cleaned = re.sub(r'\.{2,}', '.', cleaned)

    # Split into lines or sentences while respecting the max_lines limit
    segments = []
    for block in cleaned.splitlines():
        block = block.strip()
        if not block:
            continue
        sentences = re.split(r'(?<=[.!?])\s+', block)
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                segments.append(sentence)
            if len(segments) >= max_lines:
                break
        if len(segments) >= max_lines:
            break

    if not segments:
        segments = [cleaned.strip()]

    formatted = "\n".join(segments[:max_lines])
    return formatted.strip()


def estimate_processing_time(file_size: int, job_type: str) -> int:
    """Estimate processing time in minutes based on file size and job type"""
    if job_type == "process_rfp":
        # Rough estimate: 1 minute per 100KB for RFP processing
        return max(5, min(30, file_size // 102400))
    elif job_type == "extract_qa":
        # Rough estimate: 1 minute per 200KB for QA extraction
        return max(3, min(20, file_size // 204800))
    return 10


def create_rfp_from_filename(client_id: str, filename: str, supabase) -> str:
    """Auto-create RFP record from filename"""
    import traceback
    from fastapi import HTTPException
    
    rfp_name = filename.rsplit('.', 1)[0]
    rfp_name = rfp_name.replace('_', ' ').replace('-', ' ')
    rfp_name = ' '.join(word.capitalize() for word in rfp_name.split())
    
    try:
        existing_rfps_res = supabase.table("client_rfps").select("id, name").eq("client_id", client_id).execute()
        existing_rfps = existing_rfps_res.data or []
    except Exception as e:
        print(f"Error checking for existing RFPs: {e}")
        traceback.print_exc()
        existing_rfps = []
    
    rfp_id = None
    for rfp in existing_rfps:
        if rfp["name"].lower() == rfp_name.lower():
            rfp_id = rfp["id"]
            print(f"DEBUG: Found existing RFP '{rfp_name}' with ID: {rfp_id}")
            break
    
    if not rfp_id:
        rfp_data = {
            "client_id": client_id,
            "name": rfp_name,
            "description": f"Auto-created from uploaded file: {filename}",
            "created_at": datetime.now().isoformat()
        }
        try:
            rfp_result = supabase.table("client_rfps").insert(rfp_data).execute()
            rfp_id = rfp_result.data[0]["id"]
            print(f"DEBUG: Created new RFP '{rfp_name}' with ID: {rfp_id}")
        except Exception as e:
            print(f"Error creating new RFP: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to auto-create RFP from filename: {str(e)}")
    
    return rfp_id

