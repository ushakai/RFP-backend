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


def get_tags_for_rfp(rfp_id: str, supabase) -> list:
    """
    Get all tag names associated with an RFP.
    
    Args:
        rfp_id: The RFP UUID
        supabase: Supabase client
        
    Returns:
        List of tag name strings
    """
    try:
        # Query rfp_tags joined with tags table
        result = supabase.table("rfp_tags").select("tag_id, tags(name)").eq("rfp_id", rfp_id).execute()
        
        tag_names = []
        if result.data:
            for row in result.data:
                if row.get("tags") and row["tags"].get("name"):
                    tag_names.append(row["tags"]["name"])
        
        return tag_names
    except Exception as e:
        print(f"Error fetching tags for RFP {rfp_id}: {e}")
        return []


def get_tags_for_rfps_batch(rfp_ids: list, supabase) -> dict:
    """
    Batch fetch tags for multiple RFPs in a single query to avoid connection exhaustion.
    
    Args:
        rfp_ids: List of RFP UUIDs
        supabase: Supabase client
        
    Returns:
        Dict mapping rfp_id to list of tag names
    """
    if not rfp_ids:
        return {}
    
    try:
        # Single query to fetch all tags for all RFPs
        # Use a fresh connection for this query to avoid connection issues
        result = supabase.table("rfp_tags").select("rfp_id, tag_id, tags(name)").in_("rfp_id", rfp_ids).execute()
        
        # Group tags by RFP ID
        tags_by_rfp = {}
        if result.data:
            for row in result.data:
                rfp_id = row.get("rfp_id")
                if rfp_id and row.get("tags") and row["tags"].get("name"):
                    if rfp_id not in tags_by_rfp:
                        tags_by_rfp[rfp_id] = []
                    tags_by_rfp[rfp_id].append(row["tags"]["name"])
        
        # Ensure all RFP IDs are in the result (even if they have no tags)
        for rfp_id in rfp_ids:
            if rfp_id not in tags_by_rfp:
                tags_by_rfp[rfp_id] = []
        
        return tags_by_rfp
    except Exception as e:
        print(f"Error batch fetching tags: {e}")
        import traceback
        traceback.print_exc()
        # Return empty tags for all RFPs on error
        return {rfp_id: [] for rfp_id in rfp_ids}


def upsert_tags_for_rfp(client_id: str, rfp_id: str, tag_names: list, supabase):
    """
    Upsert tags for an RFP with normalized tag management.
    
    Args:
        client_id: The client UUID
        rfp_id: The RFP UUID
        tag_names: List of tag name strings
        supabase: Supabase client
        
    Returns:
        List of tag IDs that were associated with the RFP
    """
    if not tag_names:
        return []
    
    import traceback
    from fastapi import HTTPException
    
    tag_ids = []
    
    try:
        # 1. Upsert each tag (get existing or create new)
        for tag_name in tag_names:
            tag_name = tag_name.strip()
            if not tag_name:
                continue
            
            # Check if tag exists for this client
            existing_tag = supabase.table("tags").select("id").eq("client_id", client_id).eq("name", tag_name).execute()
            
            if existing_tag.data and len(existing_tag.data) > 0:
                tag_id = existing_tag.data[0]["id"]
            else:
                # Create new tag
                new_tag = supabase.table("tags").insert({
                    "client_id": client_id,
                    "name": tag_name
                }).execute()
                tag_id = new_tag.data[0]["id"]
            
            tag_ids.append(tag_id)
        
        # 2. Delete existing rfp_tags for this RFP (to avoid duplicates)
        supabase.table("rfp_tags").delete().eq("rfp_id", rfp_id).execute()
        
        # 3. Insert new rfp_tags entries
        if tag_ids:
            rfp_tags_data = [{"rfp_id": rfp_id, "tag_id": tag_id} for tag_id in tag_ids]
            supabase.table("rfp_tags").insert(rfp_tags_data).execute()
        
        return tag_ids
    
    except Exception as e:
        print(f"Error upserting tags for RFP {rfp_id}: {e}")
        traceback.print_exc()
        # Don't fail the whole operation if tags fail, just log
        return []


def create_rfp_from_filename(client_id: str, filename: str, supabase) -> str:
    """
    Auto-create RFP record from filename with automatic retry on database errors.
    Prevents [Errno 11] Resource temporarily unavailable errors.
    Always creates a new RFP for each job submission.
    """
    import traceback
    from fastapi import HTTPException
    import time
    
    rfp_name = filename.rsplit('.', 1)[0]
    rfp_name = rfp_name.replace('_', ' ').replace('-', ' ')
    rfp_name = ' '.join(word.capitalize() for word in rfp_name.split())
    
    # Always create a new RFP for each job (no reuse of existing RFPs)
    rfp_data = {
        "client_id": client_id,
        "name": rfp_name,
        "description": f"Auto-created from uploaded file: {filename}",
        "created_at": datetime.now().isoformat()
    }
    
    # Retry logic for creating RFP
    rfp_id = None
    last_error = None
    for attempt in range(3):
        try:
            rfp_result = supabase.table("client_rfps").insert(rfp_data).execute()
            rfp_id = rfp_result.data[0]["id"]
            print(f"✓ Created new RFP '{rfp_name}' with ID: {rfp_id}")
            break
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            if "resource temporarily unavailable" in error_str or "connection" in error_str:
                if attempt < 2:
                    print(f"Database error creating RFP (attempt {attempt + 1}/3): {e}")
                    time.sleep(0.5 * (attempt + 1))
                    # Try to reinitialize connection
                    try:
                        from config.settings import reinitialize_supabase
                        reinitialize_supabase()
                    except Exception:
                        pass
                    continue
            print(f"✗ Error creating new RFP: {e}")
            traceback.print_exc()
            raise HTTPException(
                status_code=500, 
                detail=f"Database connection error while creating RFP. Please try again in a moment."
            )
    
    if not rfp_id and last_error:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to create RFP after multiple attempts. Please try again later."
        )
    
    return rfp_id

