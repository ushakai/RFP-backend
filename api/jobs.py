"""
Job management endpoints
"""
import base64
import time
import traceback
from datetime import datetime, timedelta
from fastapi import APIRouter, UploadFile, Header, HTTPException, File, Form
from utils.auth import get_client_id_from_key
from utils.helpers import create_rfp_from_filename
from config.settings import get_supabase_client, reinitialize_supabase
from services.excel_service import estimate_minutes_from_chars
from services.supabase_service import fetch_paginated_rows

router = APIRouter()

@router.post("/jobs/submit")
async def submit_job(
    file: UploadFile = File(...),
    job_type: str = Form(...),
    x_client_key: str | None = Header(default=None, alias="X-Client-Key")
):
    """Submit a job for background processing - uses worker process for long-running tasks"""
    try:
        print(f"=== /jobs/submit ENDPOINT CALLED ===")
        print(f"Received job submission: file={file.filename}, job_type={job_type}")
        
        if not file.filename:
            print("ERROR: No file filename provided")
            raise HTTPException(status_code=400, detail="No file provided")
        
        if not job_type:
            print("ERROR: No job type provided")
            raise HTTPException(status_code=400, detail="No job type provided")
        
        client_id = get_client_id_from_key(x_client_key)
        print(f"Resolved client_id: {client_id}")
        
        if job_type not in ["process_rfp", "extract_qa"]:
            print(f"ERROR: Invalid job type: {job_type}")
            raise HTTPException(status_code=400, detail=f"Invalid job type: {job_type}. Must be 'process_rfp' or 'extract_qa'")
        
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
    rfp_id = create_rfp_from_filename(client_id, file.filename, supabase)
    print(f"Created RFP with ID: {rfp_id}")
    
    print("Reading file content...")
    file_content = await file.read()
    file_size = len(file_content)
    print(f"File size: {file_size} bytes")
    
    file_size_mb = file_size / (1024 * 1024)
    if file_size_mb > 50:
        raise HTTPException(status_code=400, detail=f"File too large: {file_size_mb:.1f}MB. Maximum allowed: 50MB")
    
    estimated_minutes = estimate_minutes_from_chars(file_content, job_type)
    estimated_completion = datetime.now() + timedelta(minutes=estimated_minutes)
    
    file_content_b64 = base64.b64encode(file_content).decode('utf-8')
    
    print("Creating job record...")
    job_data = {
        "client_id": client_id,
        "rfp_id": rfp_id,
        "job_type": job_type,
        "status": "pending",
        "file_name": file.filename,
        "file_size": file_size,
        "progress_percent": 0,
        "current_step": "Job queued for processing",
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
    
    try:
        job_result = supabase.table("client_jobs").insert(job_data).execute()
        job_id = job_result.data[0]["id"]
        print(f"Created job with ID: {job_id}")
    except Exception as e:
        print(f"Error inserting job data: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to create job record: {str(e)}")
    
    result = {
        "job_id": job_id, 
        "rfp_id": rfp_id, 
        "estimated_minutes": estimated_minutes, 
        "status": "submitted",
        "message": "Job submitted successfully. Processing will begin shortly."
    }
    print(f"Returning job submission result for job_id={result.get('job_id')}")
    return result


@router.get("/jobs")
def list_jobs(x_client_key: str | None = Header(default=None, alias="X-Client-Key")):
    """List all jobs for a client with built-in pagination and retry handling"""
    print(f"=== /jobs ENDPOINT CALLED ===")
    print("Jobs endpoint called")
    client_id = get_client_id_from_key(x_client_key)

    def _build_jobs_query():
        query = get_supabase_client().table("client_jobs").select("*").eq("client_id", client_id)
        return query.order("created_at", desc=True)

    try:
        jobs = fetch_paginated_rows(_build_jobs_query, page_size=200, max_rows=500)
        for job in jobs[:2]:
            print(f"DEBUG: Job {job.get('id', 'unknown')} - status: {job.get('status')} progress: {job.get('progress_percent')}")
        return {"jobs": jobs}
    except Exception as e:
        print(f"ERROR: Error fetching jobs: {e}")
        traceback.print_exc()
        try:
            reinitialize_supabase()
        except Exception as reinit_err:
            print(f"Supabase re-init failed in /jobs: {reinit_err}")
            traceback.print_exc()
        return {"jobs": [], "error": "Database connection failed"}


@router.get("/jobs/{job_id}")
def get_job(job_id: str, x_client_key: str | None = Header(default=None, alias="X-Client-Key")):
    """Get specific job details with retry logic"""
    client_id = get_client_id_from_key(x_client_key)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            supabase = get_supabase_client()
            res = supabase.table("client_jobs").select("*").eq("id", job_id).eq("client_id", client_id).single().execute()
            job_data = res.data
            
            if job_data:
                created_at = datetime.fromisoformat(job_data["created_at"].replace('Z', '+00:00'))
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
                
                if "job_data" in job_data:
                    del job_data["job_data"]
            
            return job_data
        except Exception as e:
            print(f"ERROR: Error fetching job {job_id} (attempt {attempt + 1}/{max_retries}): {e}")
            traceback.print_exc()
            try:
                reinitialize_supabase()
            except Exception as reinit_err:
                print(f"Supabase re-init failed in /jobs/{job_id}: {reinit_err}")
                traceback.print_exc()
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                print(f"Failed to fetch job {job_id} after {max_retries} attempts")
                raise HTTPException(status_code=500, detail="Database connection failed")


@router.get("/jobs/{job_id}/status")
def get_job_status(job_id: str, x_client_key: str | None = Header(default=None, alias="X-Client-Key")):
    """Get job status for polling - lightweight endpoint"""
    client_id = get_client_id_from_key(x_client_key)
    try:
        supabase = get_supabase_client()
        res = supabase.table("client_jobs").select("id, status, progress_percent, current_step, created_at, completed_at").eq("id", job_id).eq("client_id", client_id).single().execute()
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


@router.delete("/jobs/{job_id}")
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


@router.post("/jobs/cleanup")
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


@router.get("/jobs/stats")
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

