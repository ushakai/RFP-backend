#!/usr/bin/env python3
"""
Background Job Worker for RFP Processing
This script runs as a separate process to handle long-running jobs (15-20+ minutes)
"""

import os
import sys
import time
import json
import io
import tempfile
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from supabase import create_client, Client

# Add the parent directory to the path so we can import from app.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def update_job_progress(job_id: str, progress: int, current_step: str, result_data: dict = None):
    """Update job progress in database with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            updates = {
                "progress_percent": progress,
                "current_step": current_step,
                "last_updated": datetime.now().isoformat()
            }
            if result_data:
                updates["result_data"] = result_data
            if progress == 100:
                updates["status"] = "completed"
                updates["completed_at"] = datetime.now().isoformat()
            elif progress == -1:  # Use -1 for explicit failure
                updates["status"] = "failed"
                updates["completed_at"] = datetime.now().isoformat()
            
            print(f"DEBUG: Updating job {job_id} - Progress: {progress}%, Step: {current_step}")
            supabase.table("client_jobs").update(updates).eq("id", job_id).execute()
            return  # Success, exit retry loop
        except Exception as e:
            print(f"ERROR: Error updating job progress {job_id} (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait 1 second before retry
            else:
                print(f"CRITICAL ERROR: Failed to update job progress for {job_id} after {max_retries} attempts.")

def process_rfp_job(job_id: str, file_content: bytes, file_name: str, client_id: str, rfp_id: str):
    """Process RFP job - this will be imported from app.py"""
    try:
        # Import the processing function from app.py
        from app import process_excel_file_obj
        
        print(f"DEBUG: Starting RFP processing for job {job_id}")
        start_time = time.time()
        
        update_job_progress(job_id, 10, "Starting RFP processing: Loading file...")
        
        # Check file size to prevent memory issues
        file_size_mb = len(file_content) / (1024 * 1024)
        if file_size_mb > 50:  # 50MB limit
            raise Exception(f"File too large: {file_size_mb:.1f}MB. Maximum allowed: 50MB")
        
        print(f"DEBUG: Processing file {file_name} ({file_size_mb:.1f}MB)")
        
        file_obj = io.BytesIO(file_content)
        processed_output = process_excel_file_obj(file_obj, file_name, client_id, rfp_id, job_id=job_id)
        
        processed_content = processed_output.getvalue()
        
        update_job_progress(job_id, 95, "Finalizing processed file...")
        
        result_data = {
            "file_name": f"processed_{file_name}",
            "file_size": len(processed_content),
            "processing_completed": True,
            "processing_time_seconds": int(time.time() - start_time)
        }
        
        update_job_progress(job_id, 100, "RFP processing completed successfully!", result_data)
        print(f"DEBUG: RFP processing completed for job {job_id} in {time.time() - start_time:.1f}s")
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Processing failed after {processing_time:.1f}s: {str(e)}"
        print(f"ERROR: RFP processing error for job {job_id}: {error_msg}")
        update_job_progress(job_id, -1, error_msg)

def extract_qa_job(job_id: str, file_content: bytes, file_name: str, client_id: str, rfp_id: str):
    """Extract QA job - this will be imported from app.py"""
    try:
        # Import the processing function from app.py
        from app import extract_qa_background
        
        print(f"DEBUG: Starting QA extraction for job {job_id}")
        extract_qa_background(job_id, file_content, file_name, client_id, rfp_id)
        
    except Exception as e:
        error_msg = f"QA extraction failed: {str(e)}"
        print(f"ERROR: QA extraction error for job {job_id}: {error_msg}")
        update_job_progress(job_id, -1, error_msg)

def get_pending_jobs():
    """Get all pending jobs from the database"""
    try:
        res = supabase.table("client_jobs").select("*").eq("status", "pending").order("created_at", desc=False).execute()
        return res.data or []
    except Exception as e:
        print(f"ERROR: Failed to get pending jobs: {e}")
        return []

def main():
    """Main worker loop"""
    print("Starting RFP Background Worker...")
    print(f"Worker PID: {os.getpid()}")
    
    while True:
        try:
            # Get pending jobs
            pending_jobs = get_pending_jobs()
            
            if not pending_jobs:
                print("No pending jobs, sleeping for 10 seconds...")
                time.sleep(10)
                continue
            
            # Process the oldest pending job
            job = pending_jobs[0]
            job_id = job["id"]
            job_type = job["job_type"]
            file_name = job["file_name"]
            client_id = job["client_id"]
            rfp_id = job["rfp_id"]
            
            print(f"Processing job {job_id} of type {job_type}")
            
            # Update job status to processing
            supabase.table("client_jobs").update({
                "status": "processing",
                "current_step": "Job started by worker",
                "last_updated": datetime.now().isoformat()
            }).eq("id", job_id).execute()
            
            # Get file content from job data
            job_data = job.get("job_data", {})
            file_content = job_data.get("file_content")
            
            if not file_content:
                update_job_progress(job_id, -1, "No file content found in job data")
                continue
            
            # Decode base64 file content
            import base64
            file_bytes = base64.b64decode(file_content)
            
            # Process based on job type
            if job_type == "process_rfp":
                process_rfp_job(job_id, file_bytes, file_name, client_id, rfp_id)
            elif job_type == "extract_qa":
                extract_qa_job(job_id, file_bytes, file_name, client_id, rfp_id)
            else:
                update_job_progress(job_id, -1, f"Unknown job type: {job_type}")
            
        except KeyboardInterrupt:
            print("Worker shutting down...")
            break
        except Exception as e:
            print(f"ERROR: Worker error: {e}")
            time.sleep(5)  # Wait before retrying

if __name__ == "__main__":
    main()
