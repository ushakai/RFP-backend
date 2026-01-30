"""
Job management endpoints
"""
import base64
import time
import traceback
from datetime import datetime, timedelta
from fastapi import APIRouter, UploadFile, Header, HTTPException, File, Form, Depends
from utils.auth import get_client_id_from_key, require_subscription
from services.activity_service import record_event
from utils.helpers import create_rfp_from_filename
from config.settings import get_supabase_client, reinitialize_supabase
from services.excel_service import estimate_minutes_from_chars
from services.supabase_service import fetch_paginated_rows, execute_with_retry
from utils.db_utils import safe_db_operation, retry_on_db_error

router = APIRouter()

@retry_on_db_error(max_retries=3, delay=0.5, backoff=2.0)
def submit_job_internal(file_content: bytes, file_name: str, file_size: int, job_type: str, client_id: str, rfp_id: str) -> str:
    """Internal function to submit a job (used by reprocess endpoint) with automatic retry"""
    supabase = get_supabase_client()
    
    estimated_minutes = estimate_minutes_from_chars(file_content, job_type)
    estimated_completion = datetime.now() + timedelta(minutes=estimated_minutes)
    
    file_content_b64 = base64.b64encode(file_content).decode('utf-8')
    
    # Use 'queued' instead of 'pending' to avoid ghost workers
    initial_status = "queued"
    
    job_data = {
        "client_id": client_id,
        "rfp_id": rfp_id,
        "job_type": job_type,
        "status": initial_status,
        "file_name": file_name,
        "file_size": file_size,
        "progress_percent": 0,
        "current_step": f"Job {initial_status} for reprocessing",
        "estimated_completion": estimated_completion.isoformat(),
        "created_at": datetime.now().isoformat(),
        "job_data": {
            "file_content": file_content_b64,
            "file_name": file_name,
            "client_id": client_id,
            "rfp_id": rfp_id
        }
    }
    
    job_result = supabase.table("client_jobs").insert(job_data).execute()
    job_id = job_result.data[0]["id"]
    
    # Update RFP with job info
    supabase.table("client_rfps").update({
        "last_job_id": job_id,
        "last_job_type": job_type,
        "last_job_status": initial_status
    }).eq("id", rfp_id).execute()
    
    # Log activity
    try:
        record_event(
            "bid" if job_type == "process_rfp" else "file",
            "job_resubmitted",
            actor_client_id=client_id,
            subject_id=job_id,
            subject_type="job",
            metadata={"job_type": job_type, "rfp_id": rfp_id, "file_name": file_name},
        )
    except Exception as e:
        print(f"Warning: Failed to log activity event: {e}")
    
    return job_id

@router.post("/jobs/submit", dependencies=[Depends(require_subscription(["processing", "both"]))])
async def submit_job(
    file: UploadFile = File(...),
    job_type: str = Form(...),
    original_rfp_date: str | None = Form(default=None),
    tags: str | None = Form(default=None),
    x_client_key: str | None = Header(default=None, alias="X-Client-Key")
):
    """Submit a job for background processing - uses worker process for long-running tasks.
    
    Args:
        file: The file to process
        job_type: Type of job (process_rfp or extract_qa)
        original_rfp_date: Optional original date of the RFP (ISO format YYYY-MM-DD)
        tags: Optional comma-separated list of tags
        x_client_key: Client API key
    """
    try:
        print(f"=== /jobs/submit ENDPOINT CALLED ===")
        print(f"Received job submission: file={file.filename}, job_type={job_type}")
        
        if not file.filename:
            print("ERROR: No file filename provided")
            raise HTTPException(status_code=400, detail="No file provided")
        
        if not job_type:
            print("ERROR: No job type provided")
            raise HTTPException(status_code=400, detail="No job type provided")
        
        # Normalize job type
        job_type = str(job_type).strip().lower().replace(" ", "_").replace("-", "_")
        
        client_id = get_client_id_from_key(x_client_key)
        print(f"Resolved client_id: {client_id}")
        
        if job_type not in ["process_rfp", "extract_qa", "ingest_text"]:
            print(f"ERROR: Invalid job type: {job_type}")
            raise HTTPException(status_code=400, detail=f"Invalid job type: {job_type}. Must be 'process_rfp', 'extract_qa', or 'ingest_text'")
        
        print("Validation passed, proceeding with job creation...")
    except HTTPException as he:
        print(f"HTTPException in submit_job: {he.detail}")
        traceback.print_exc()
        raise
    except Exception as e:
        print(f"ERROR: Error in submit_job before background task: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    supabase = get_supabase_client()
    
    print("Reading file content...")
    file_content = await file.read()
    file_size = len(file_content)
    print(f"File size: {file_size} bytes")
    
    # Create RFP and store original file for reprocessing
    rfp_id = create_rfp_from_filename(client_id, file.filename, supabase)
    print(f"Created RFP with ID: {rfp_id}")
    
    # Store original file, date, and tags in RFP
    try:
        rfp_update_data = {
            "original_file_data": file_content,
            "original_file_name": file.filename,
            "original_file_size": file_size
        }
        
        # Add original_rfp_date if provided
        if original_rfp_date:
            rfp_update_data["original_rfp_date"] = original_rfp_date
        
        supabase.table("client_rfps").update(rfp_update_data).eq("id", rfp_id).execute()
        print(f"Stored original file in RFP {rfp_id}")
        
        # Process and store tags if provided
        if tags:
            from utils.helpers import upsert_tags_for_rfp
            tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
            if tag_list:
                upsert_tags_for_rfp(client_id, rfp_id, tag_list, supabase)
                print(f"Associated {len(tag_list)} tags with RFP {rfp_id}")
        
    except Exception as e:
        print(f"Warning: Failed to store original file/tags/date in RFP: {e}")
    
    file_size_mb = file_size / (1024 * 1024)
    if file_size_mb > 50:
        raise HTTPException(status_code=400, detail=f"File too large: {file_size_mb:.1f}MB. Maximum allowed: 50MB")
    
    # Validate Excel file before processing (only for Excel-based jobs)
    if job_type in ["process_rfp", "extract_qa"]:
        from services.excel_service import validate_excel_file
        is_valid, error_msg = validate_excel_file(file_content, file.filename)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
    
    estimated_minutes = estimate_minutes_from_chars(file_content, job_type)
    estimated_completion = datetime.now() + timedelta(minutes=estimated_minutes)
    
    file_content_b64 = base64.b64encode(file_content).decode('utf-8')
    
    # Use 'queued' instead of 'pending' to avoid ghost workers
    initial_status = "queued"
    
    # Create job record
    job_data = {
        "client_id": client_id,
        "rfp_id": rfp_id,
        "job_type": job_type,
        "status": initial_status,
        "file_name": file.filename,
        "file_size": file_size,
        "progress_percent": 0,
        "current_step": f"Job {initial_status} for processing",
        "estimated_completion": estimated_completion.isoformat(),
        "created_at": datetime.now().isoformat(),
        "job_data": {
            "file_content": file_content_b64,
            "file_name": file.filename,
            "client_id": client_id,
            "rfp_id": rfp_id
        }
    }
    print(f"Job data created, file content encoded")
    
    job_id = None
    try:
        job_result = supabase.table("client_jobs").insert(job_data).execute()
        job_id = job_result.data[0]["id"]
        print(f"Created job with ID: {job_id}")
        
        # Update RFP with job info
        try:
            supabase.table("client_rfps").update({
                "last_job_id": job_id,
                "last_job_type": job_type,
                "last_job_status": initial_status
            }).eq("id", rfp_id).execute()
            print(f"Updated RFP {rfp_id} with job {job_id}")
        except Exception as e:
            print(f"Warning: Failed to update RFP with job info: {e}")
            
    except Exception as e:
        print(f"Error inserting job data: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to create job record: {str(e)}")
    
    # Log activity after successful job creation
    if job_id:
        try:
            record_event(
                "bid" if job_type == "process_rfp" else "file",
                "job_submitted",
                actor_client_id=client_id,
                subject_id=job_id,
                subject_type="job",
                metadata={"job_type": job_type, "rfp_id": rfp_id, "file_name": file.filename},
            )
        except Exception as e:
            # Don't fail the request if activity logging fails
            print(f"Warning: Failed to log activity event: {e}")
    
    result = {
        "job_id": job_id, 
        "rfp_id": rfp_id, 
        "estimated_minutes": estimated_minutes, 
        "status": "submitted",
        "message": "Job submitted successfully. Processing will begin shortly."
    }
    print(f"Returning job submission result for job_id={result.get('job_id')}")
    
    return result


@router.get("/jobs", dependencies=[Depends(require_subscription(["processing", "both"]))])
@safe_db_operation("list jobs")
def list_jobs(x_client_key: str | None = Header(default=None, alias="X-Client-Key")):
    """List all jobs for a client with built-in pagination and automatic retry
    
    Uses a simpler direct query instead of pagination to avoid hanging.
    Limits to recent 500 jobs to prevent timeout issues.
    """
    print(f"=== /jobs ENDPOINT CALLED ===")
    print("Jobs endpoint called")
    client_id = get_client_id_from_key(x_client_key)

    # Retry logic with connection reinitialization
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Use direct query with limit instead of pagination to avoid hanging
            # This is faster and more reliable for job listings
            supabase = get_supabase_client()
            res = supabase.table("client_jobs").select(
                "id, job_type, status, progress_percent, current_step, file_name, "
                "created_at, completed_at, estimated_completion, rfp_id, client_id"
            ).eq("client_id", client_id).order("created_at", desc=True).limit(500).execute()
            
            jobs = res.data or []
            print(f"DEBUG: Fetched {len(jobs)} jobs")
            
            # Log first 2 jobs for debugging
            for job in jobs[:2]:
                print(f"DEBUG: Job {job.get('id', 'unknown')[:8]}... - status: {job.get('status')} progress: {job.get('progress_percent')}")
            
            return {"jobs": jobs}
        except Exception as e:
            error_str = str(e).lower()
            is_connection_error = any([
                "winerror 10054" in error_str,
                "winerror 10035" in error_str,
                "non-blocking socket" in error_str,
                "connection" in error_str and ("closed" in error_str or "forcibly" in error_str or "reset" in error_str),
                "readerror" in error_str,
            ])
            
            if is_connection_error and attempt < max_retries - 1:
                print(f"ERROR: Connection error fetching jobs (attempt {attempt + 1}/{max_retries}): {e}")
                # Reinitialize connection and retry
                try:
                    reinitialize_supabase()
                    print("✓ Reinitialized Supabase client, retrying...")
                    import time
                    time.sleep(0.3 * (attempt + 1))  # Small delay before retry
                    continue
                except Exception as reinit_err:
                    print(f"Supabase re-init failed: {reinit_err}")
                    if attempt < max_retries - 1:
                        continue
            
            # If not a connection error or all retries exhausted, log and return error
            print(f"ERROR: Error fetching jobs: {e}")
            traceback.print_exc()
            return {"jobs": [], "error": "Database connection failed"}
    
    # Fallback if all retries failed
    return {"jobs": [], "error": "Database connection failed after retries"}


@router.get("/jobs/{job_id}", dependencies=[Depends(require_subscription(["processing", "both"]))])
def get_job(job_id: str, x_client_key: str | None = Header(default=None, alias="X-Client-Key")):
    """Get specific job details with server-side retry logic"""
    client_id = get_client_id_from_key(x_client_key)
    
    max_retries = 3
    last_error = None
    
    for attempt in range(max_retries):
        try:
            supabase = get_supabase_client()
            res = supabase.table("client_jobs").select("*").eq("id", job_id).eq("client_id", client_id).single().execute()
            
            if hasattr(res, 'error') and res.error:
                raise Exception(res.error)
            
            job_data = res.data
            
            if job_data:
                # Add calculated fields for the UI
                try:
                    created_at_str = job_data.get("created_at")
                    if created_at_str:
                        created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                        elapsed_minutes = (datetime.now(created_at.tzinfo) - created_at).total_seconds() / 60
                        job_data["elapsed_minutes"] = round(elapsed_minutes, 1)
                        
                        if job_data["status"] == "pending":
                            job_data["estimated_remaining_minutes"] = job_data.get("estimated_minutes", 10)
                        elif job_data["status"] == "processing":
                            estimated_total = job_data.get("estimated_minutes", 10)
                            remaining = max(0, estimated_total - elapsed_minutes)
                            job_data["estimated_remaining_minutes"] = round(remaining, 1)
                        else:
                            job_data["estimated_remaining_minutes"] = 0
                except Exception as calc_err:
                    print(f"Warning: Calculation error for job {job_id}: {calc_err}")
                
                # Remove large fields for the status check to save bandwidth
                if "job_data" in job_data:
                    del job_data["job_data"]
            
            return job_data
            
        except Exception as e:
            last_error = e
            print(f"ERROR: Fetching job {job_id} (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))
                try:
                    from config.settings import reinitialize_supabase
                    reinitialize_supabase()
                except:
                    pass
            else:
                print(f"CRITICAL: Failed to fetch job {job_id} after {max_retries} attempts")
                traceback.print_exc()
                
    # If we reached here, it failed after all retries
    error_msg = str(last_error) if last_error else "Unknown database error"
    raise HTTPException(status_code=500, detail=f"Database connection failed: {error_msg}")


@router.get("/jobs/{job_id}/status", dependencies=[Depends(require_subscription(["processing", "both"]))])
def get_job_status(job_id: str, x_client_key: str | None = Header(default=None, alias="X-Client-Key")):
    """Get job status for polling - lightweight endpoint"""
    client_id = get_client_id_from_key(x_client_key)
    
    def _fetch_status():
        supabase = get_supabase_client()
        return supabase.table("client_jobs").select("id, status, progress_percent, current_step, created_at, completed_at").eq("id", job_id).eq("client_id", client_id).single().execute()
    
    try:
        res = execute_with_retry(_fetch_status)
        job = res.data
        
        if job:
            created_at = datetime.fromisoformat(job["created_at"].replace('Z', '+00:00'))
            elapsed_minutes = (datetime.now(created_at.tzinfo) - created_at).total_seconds() / 60
            
            return {
                "job_id": job["id"],
                "status": job["status"],
                "progress_percent": job["progress_percent"],
                "current_step": job["current_step"],
                "elapsed_minutes": round(elapsed_minutes, 1),
                "completed_at": job.get("completed_at")
            }
        else:
            raise HTTPException(status_code=404, detail="Job not found")
    except Exception as e:
        print(f"ERROR: Error fetching job status {job_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Database connection failed")


@router.delete("/jobs/{job_id}", dependencies=[Depends(require_subscription(["processing", "both"]))])
def cancel_job(job_id: str, x_client_key: str | None = Header(default=None, alias="X-Client-Key")):
    """Cancel a pending job"""
    client_id = get_client_id_from_key(x_client_key)
    try:
        supabase = get_supabase_client()
        res = supabase.table("client_jobs").select("status").eq("id", job_id).eq("client_id", client_id).single().execute()
        job_status = res.data.get("status") if res.data else None
    except Exception as e:
        print(f"Error fetching job status for cancellation {job_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to fetch job status for cancellation")

    if job_status in ["pending", "processing"]:
        try:
            supabase.table("client_jobs").update({"status": "cancelled", "current_step": "Job cancelled by user", "completed_at": datetime.now().isoformat()}).eq("id", job_id).eq("client_id", client_id).execute()
            print(f"DEBUG: Job {job_id} cancelled by user.")
            return {"ok": True, "message": f"Job {job_id} cancelled successfully."}
        except Exception as e:
            print(f"Error cancelling job {job_id}: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to cancel job {job_id}")
    else:
        print(f"WARN: Attempted to cancel job {job_id} which is in status: {job_status}. Only pending/processing jobs can be cancelled.")
        raise HTTPException(status_code=400, detail=f"Cannot cancel job with status '{job_status}'. Only pending or processing jobs can be cancelled.")


@router.post("/jobs/cleanup", dependencies=[Depends(require_subscription(["processing", "both"]))])
def cleanup_old_jobs(x_client_key: str | None = Header(default=None, alias="X-Client-Key")):
    """Clean up old completed/failed jobs to free up database space"""
    client_id = get_client_id_from_key(x_client_key)
    
    cutoff_date = datetime.now() - timedelta(days=7)
    
    try:
        supabase = get_supabase_client()
        res = supabase.table("client_jobs").delete().eq("client_id", client_id).in_("status", ["completed", "failed", "cancelled"]).lt("created_at", cutoff_date.isoformat()).execute()
        
        deleted_count = len(res.data) if res.data else 0
        print(f"DEBUG: Cleaned up {deleted_count} old jobs for client {client_id}")
        
        return {"ok": True, "deleted_count": deleted_count, "message": f"Cleaned up {deleted_count} old jobs"}
    except Exception as e:
        print(f"ERROR: Error cleaning up jobs for client {client_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to cleanup jobs")


@router.get("/jobs/stats", dependencies=[Depends(require_subscription(["processing", "both"]))])
def get_job_stats(x_client_key: str | None = Header(default=None, alias="X-Client-Key")):
    """Get job statistics for the client"""
    client_id = get_client_id_from_key(x_client_key)
    
    try:
        supabase = get_supabase_client()
        res = supabase.table("client_jobs").select("status").eq("client_id", client_id).execute()
        jobs = res.data or []
        
        stats = {
            "total": len(jobs),
            "pending": len([j for j in jobs if j["status"] == "pending"]),
            "processing": len([j for j in jobs if j["status"] == "processing"]),
            "completed": len([j for j in jobs if j["status"] == "completed"]),
            "failed": len([j for j in jobs if j["status"] == "failed"]),
            "cancelled": len([j for j in jobs if j["status"] == "cancelled"])
        }
        
        return stats
    except Exception as e:
        print(f"ERROR: Error getting job stats for client {client_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to get job statistics")

