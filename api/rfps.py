"""
RFP management endpoints
"""
import io
import traceback
from fastapi import APIRouter, UploadFile, Header, HTTPException, Body
from fastapi.responses import StreamingResponse
from utils.auth import get_client_id_from_key
from config.settings import get_supabase_client
from services.excel_service import process_excel_file_obj
from utils.db_utils import safe_db_operation

router = APIRouter()

@router.post("/process")
async def process_excel(
    file: UploadFile, 
    x_client_key: str | None = Header(default=None, alias="X-Client-Key"), 
    rfp_id: str | None = Header(default=None, alias="X-RFP-ID")
):
    """Process an Excel file with RFP questions and generate AI answers"""
    client_id = get_client_id_from_key(x_client_key)
    
    file_content = await file.read()
    file_obj = io.BytesIO(file_content)
    
    processed_file_io, processed_sheets_count, total_questions_processed = process_excel_file_obj(
        file_obj, file.filename, client_id, rfp_id
    )
    
    return StreamingResponse(
        processed_file_io,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=processed_{file.filename}"}
    )

@router.get("/rfps")
@safe_db_operation("list RFPs")
def list_rfps(x_client_key: str | None = Header(default=None, alias="X-Client-Key")):
    """List all RFPs for a client with automatic retry on connection errors"""
    print(f"=== /rfps ENDPOINT CALLED ===")
    print(f"Received x_client_key: {x_client_key[:8]}..." if x_client_key else "No API key provided")
    
    try:
        client_id = get_client_id_from_key(x_client_key)
        print(f"✓ Resolved client_id: {client_id}")
    except HTTPException as he:
        print(f"✗ Auth failed: {he.detail}")
        raise
    except Exception as e:
        print(f"✗ Unexpected auth error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Authentication error: {str(e)}")
    
    supabase = get_supabase_client()
    print(f"✓ Got Supabase client")
    
    try:
        res = supabase.table("client_rfps").select(
            "id, name, description, created_at, updated_at, original_rfp_date, "
            "original_file_name, original_file_size, "
            "last_job_id, last_job_type, last_job_status, last_processed_at"
        ).eq("client_id", client_id).order("created_at", desc=True).execute()
        
        rfps = res.data or []
        rfp_count = len(rfps)
        print(f"✓ Found {rfp_count} RFPs")
        
        # Batch fetch all tags for all RFPs in one query to avoid connection exhaustion
        if rfps:
            from utils.helpers import get_tags_for_rfps_batch
            try:
                tags_map = get_tags_for_rfps_batch([rfp["id"] for rfp in rfps], supabase)
                for rfp in rfps:
                    rfp["tags"] = tags_map.get(rfp["id"], [])
            except Exception as tag_error:
                print(f"Warning: Failed to fetch tags: {tag_error}")
                # Set empty tags for all RFPs if tag fetch fails
                for rfp in rfps:
                    rfp["tags"] = []
        
        return {"rfps": rfps}
    except Exception as e:
        print(f"Error in list_rfps: {e}")
        import traceback
        traceback.print_exc()
        # Reinitialize connection on error
        from config.settings import reinitialize_supabase
        try:
            reinitialize_supabase()
        except:
            pass
        raise

@router.post("/rfps")
@safe_db_operation("create RFP")
def create_rfp(
    payload: dict = Body(...), 
    x_client_key: str | None = Header(default=None, alias="X-Client-Key")
):
    """Create a new RFP"""
    client_id = get_client_id_from_key(x_client_key)
    rfp_data = {
        "client_id": client_id,
        "name": payload.get("name", "").strip(),
        "description": payload.get("description", "").strip(),
    }
    if not rfp_data["name"]:
        raise HTTPException(status_code=400, detail="RFP name is required")
    try:
        supabase = get_supabase_client()
        res = supabase.table("client_rfps").insert(rfp_data).execute()
        return {"rfp": res.data[0] if res.data else None}
    except Exception as e:
        print(f"Error creating RFP: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to create RFP")

@router.put("/rfps/{rfp_id}")
@safe_db_operation("update RFP")
def update_rfp(
    rfp_id: str, 
    payload: dict = Body(...), 
    x_client_key: str | None = Header(default=None, alias="X-Client-Key")
):
    """Update an RFP"""
    client_id = get_client_id_from_key(x_client_key)
    updates = {k: v for k, v in payload.items() if k in ("name", "description")}
    if "name" in updates:
        updates["name"] = updates["name"].strip()
    if "description" in updates:
        updates["description"] = updates["description"].strip()
    try:
        supabase = get_supabase_client()
        res = supabase.table("client_rfps").update(updates).eq("id", rfp_id).eq("client_id", client_id).execute()
        return {"ok": True}
    except Exception as e:
        print(f"Error updating RFP {rfp_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to update RFP")

@router.delete("/rfps/{rfp_id}")
@safe_db_operation("delete RFP")
def delete_rfp(
    rfp_id: str, 
    x_client_key: str | None = Header(default=None, alias="X-Client-Key")
):
    """Delete an RFP"""
    client_id = get_client_id_from_key(x_client_key)
    try:
        supabase = get_supabase_client()
        supabase.table("client_rfps").delete().eq("id", rfp_id).eq("client_id", client_id).execute()
        return {"ok": True}
    except Exception as e:
        print(f"Error deleting RFP {rfp_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to delete RFP")

@router.post("/rfps/{rfp_id}/reprocess")
@safe_db_operation("reprocess RFP")
def reprocess_rfp(
    rfp_id: str,
    x_client_key: str | None = Header(default=None, alias="X-Client-Key")
):
    """Reprocess an RFP using its stored original file"""
    client_id = get_client_id_from_key(x_client_key)
    try:
        supabase = get_supabase_client()
        # Get RFP with original file data
        rfp_res = supabase.table("client_rfps").select(
            "id, name, original_file_data, original_file_name, original_file_size, last_job_type"
        ).eq("id", rfp_id).eq("client_id", client_id).limit(1).execute()
        
        if not rfp_res.data:
            raise HTTPException(status_code=404, detail="RFP not found")
        
        rfp = rfp_res.data[0]
        if not rfp.get("original_file_data"):
            raise HTTPException(status_code=400, detail="No original file stored for this RFP")
        
        # Import here to avoid circular dependency
        from api.jobs import submit_job_internal
        
        # Get job type and normalize it
        last_job_type = rfp.get("last_job_type")
        if not last_job_type:
            job_type = "process_rfp"
        else:
            # Basic normalization for common variations
            job_type = str(last_job_type).strip().lower().replace(" ", "_").replace("-", "_")
            if job_type not in ["process_rfp", "extract_qa", "ingest_text"]:
                # Default to process_rfp if unknown
                job_type = "process_rfp"
        
        # Decode file content from base64 if it's a string
        raw_file_data = rfp["original_file_data"]
        if isinstance(raw_file_data, str):
            import base64
            # Handle potential header in base64 like "data:...;base64,"
            if "," in raw_file_data:
                raw_file_data = raw_file_data.split(",")[1]
            file_bytes = base64.b64decode(raw_file_data)
        else:
            file_bytes = bytes(raw_file_data)
        
        # Resubmit the job with original file
        job_id = submit_job_internal(
            file_content=file_bytes,
            file_name=rfp["original_file_name"],
            file_size=rfp["original_file_size"],
            job_type=job_type,
            client_id=client_id,
            rfp_id=rfp_id
        )
        
        return {"job_id": job_id, "rfp_id": rfp_id, "message": f"Reprocessing started for {job_type}"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error reprocessing RFP {rfp_id}: {e}")
        traceback.print_exc()

