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
def list_rfps(x_client_key: str | None = Header(default=None, alias="X-Client-Key")):
    """List all RFPs for a client"""
    print(f"=== /rfps ENDPOINT CALLED ===")
    print("RFPs endpoint called")
    client_id = get_client_id_from_key(x_client_key)
    print(f"Resolved client_id: {client_id}")
    try:
        supabase = get_supabase_client()
        res = supabase.table("client_rfps").select("id, name, description, created_at, updated_at").eq("client_id", client_id).order("created_at", desc=True).execute()
        print(f"Found {len(res.data or [])} RFPs")
        return {"rfps": res.data or []}
    except Exception as e:
        print(f"Error listing RFPs: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to list RFPs")

@router.post("/rfps")
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

