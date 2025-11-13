"""
Google Drive integration endpoints
"""
import base64
import traceback
from fastapi import APIRouter, Header, HTTPException, Body
from utils.auth import get_client_id_from_key
from services.drive_service import setup_drive_folders, get_drive_service, upload_file_to_drive

router = APIRouter()

@router.post("/drive/setup")
def setup_drive_folders_endpoint(
    payload: dict = Body(...),
    x_client_key: str | None = Header(default=None, alias="X-Client-Key")
):
    """Setup Google Drive folders for client"""
    client_id = get_client_id_from_key(x_client_key)
    access_token = payload.get("access_token")
    client_name = payload.get("client_name", "Client")
    
    if not access_token:
        raise HTTPException(status_code=400, detail="Access token required")
    
    try:
        folder_structure = setup_drive_folders(access_token, client_name)
        if folder_structure:
            return {"success": True, "folders": folder_structure}
        else:
            raise HTTPException(status_code=500, detail="Failed to setup Drive folders")
    except Exception as e:
        print(f"Drive setup error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Drive setup failed: {str(e)}")


@router.post("/drive/upload")
def upload_to_drive_endpoint(
    payload: dict = Body(...),
    x_client_key: str | None = Header(default=None, alias="X-Client-Key")
):
    """Upload file to Google Drive"""
    client_id = get_client_id_from_key(x_client_key)
    access_token = payload.get("access_token")
    file_content = payload.get("file_content")
    filename = payload.get("filename")
    folder_type = payload.get("folder_type", "processed")
    
    if not all([access_token, file_content, filename]):
        raise HTTPException(status_code=400, detail="Missing required parameters")
    
    try:
        file_bytes = base64.b64decode(file_content)
        
        service = get_drive_service(access_token)
        
        if folder_type == "processed":
            folder_id = payload.get("processed_folder_id")
        else:
            folder_id = payload.get("unprocessed_folder_id")
        
        if not folder_id:
            raise HTTPException(status_code=400, detail="Folder ID required")
        
        mime_type = "application/octet-stream"
        if filename.lower().endswith('.xlsx'):
            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        elif filename.lower().endswith('.xls'):
            mime_type = "application/vnd.ms-excel"
        elif filename.lower().endswith('.pdf'):
            mime_type = "application/pdf"
        
        file_id = upload_file_to_drive(service, file_bytes, filename, folder_id, mime_type)
        
        if file_id:
            return {"success": True, "file_id": file_id, "filename": filename}
        else:
            raise HTTPException(status_code=500, detail="Failed to upload file")
            
    except Exception as e:
        print(f"Drive upload error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

