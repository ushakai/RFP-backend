#!/usr/bin/env python3
"""
Background Job Worker for RFP Processing
This script runs as a separate process to handle long-running jobs (15-20+ minutes)

Uses direct PostgreSQL connection for reliable, persistent database connections.
"""

import os
import sys
import time
import base64
import traceback
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

# Import from modular structure
from config.settings import get_supabase_client, reinitialize_supabase, is_direct_db_configured
from services.job_service import process_rfp_background, extract_qa_background, ingest_text_background, update_job_progress
from utils.logging_config import get_logger

# Setup logger
logger = get_logger(__name__, "worker")


def get_db_with_retry(max_retries=3):
    """Get database client with retry for connection issues"""
    for attempt in range(max_retries):
        try:
            client = get_supabase_client()
            # Test connection with a simple query
            result = client.table("client_jobs").select("id").limit(1).execute()
            if hasattr(result, 'error') and result.error:
                raise Exception(result.error)
            logger.debug(f"Database connection successful (attempt {attempt + 1}/{max_retries})")
            return client
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                logger.warning(f"Database connection failed (attempt {attempt + 1}/{max_retries}): {e}")
                logger.warning(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                try:
                    reinitialize_supabase()
                except:
                    pass
            else:
                logger.error(f"Database connection failed after {max_retries} attempts: {e}")
                logger.error("Please check: 1) DATABASE_URL in .env, 2) Network connectivity")
                raise Exception(f"Failed to connect to database after {max_retries} attempts") from e
    return None


# Alias for backwards compatibility
get_supabase_with_retry = get_db_with_retry


def get_pending_jobs():
    """Get all jobs waiting for processing (pending or queued)"""
    try:
        db = get_db_with_retry()
        # Fetch both pending and queued jobs
        res = db.table("client_jobs").select("*").in_("status", ["pending", "queued"]).order("created_at", desc=False).execute()
        if hasattr(res, 'error') and res.error:
            logger.error(f"Database error getting pending jobs: {res.error}")
            return []
        return res.data or []
    except Exception as e:
        logger.error(f"Failed to get pending jobs: {e}")
        traceback.print_exc()
        return []


def reset_stuck_jobs():
    """Reset jobs that have been stuck in processing for too long (30+ minutes)"""
    try:
        db = get_db_with_retry()
        cutoff_time = (datetime.now() - timedelta(minutes=30)).isoformat()
        
        res = db.table("client_jobs").select("*").eq("status", "processing").lt("last_updated", cutoff_time).execute()
        stuck_jobs = res.data or []
        
        if stuck_jobs:
            logger.warning(f"Found {len(stuck_jobs)} stuck jobs, resetting to pending...")
            for job in stuck_jobs:
                db.table("client_jobs").update({
                    "status": "pending",
                    "current_step": "Job was stuck and has been reset for retry",
                    "progress_percent": 0,
                    "last_updated": datetime.now().isoformat()
                }).eq("id", job["id"]).execute()
                logger.info(f"Reset stuck job {job['id']}")
        else:
            logger.debug("No stuck jobs found")
    except Exception as e:
        logger.error(f"Error resetting stuck jobs: {e}")
        traceback.print_exc()


def process_job(job: dict):
    """Process a single job"""
    job_id = job.get("id", "unknown")
    
    # Normalize job type with alias mapping
    raw_job_type = str(job.get("job_type", "")).strip()
    job_type = raw_job_type.lower()
    
    # Alias mapping for robustness
    ALIASES = {
        "rfp processing": "process_rfp",
        "rfp_processing": "process_rfp",
        "process-rfp": "process_rfp",
        "qa extraction": "extract_qa",
        "qa_extraction": "extract_qa",
        "extract-qa": "extract_qa",
        "qa_extract": "extract_qa",
        "text ingestion": "ingest_text",
        "text_ingestion": "ingest_text",
        "ingest-text": "ingest_text",
        "text_ingest": "ingest_text"
    }
    
    if job_type in ALIASES:
        # Only log if it's actually an alias (not already canonical)
        if job_type != ALIASES[job_type]:
            logger.info(f"Mapping job type alias '{job_type}' to canonical type '{ALIASES[job_type]}'")
        job_type = ALIASES[job_type]
    
    job_data = job.get("job_data", {})
    
    logger.info(f"Starting processing for job {job_id} (canonical type: {job_type}, raw: {raw_job_type})")
    
    try:
        # Mark job as processing - use atomic update to claim the job
        # Only claim if it's still in its original state (pending or queued)
        # Store worker PID to prevent ghost workers from marking job as failed
        original_status = job.get("status", "pending")
        worker_pid = str(os.getpid())
        db = get_db_with_retry()
        update_res = db.table("client_jobs").update({
            "status": "processing",
            "last_updated": datetime.now().isoformat(),
            "started_at": datetime.now().isoformat(),
            "current_step": f"Processing started (type: {job_type})",
            "worker_pid": worker_pid  # Track which worker owns this job
        }).eq("id", job_id).eq("status", original_status).execute()
        
        # If no row was updated, it means another worker already claimed it
        if not update_res.data:
            logger.info(f"Job {job_id} already claimed by another worker. Skipping.")
            return
        
        logger.info(f"Successfully claimed job {job_id}")

        
        # Extract job data
        file_content_b64 = job_data.get("file_content")
        file_name = job_data.get("file_name")
        client_id = job_data.get("client_id")
        rfp_id = job_data.get("rfp_id")
        
        if not all([file_content_b64, file_name, client_id]):
            missing = []
            if not file_content_b64: missing.append("file_content")
            if not file_name: missing.append("file_name")
            if not client_id: missing.append("client_id")
            raise ValueError(f"Missing required job data fields: {', '.join(missing)}")
        
        # Decode file content
        try:
            file_content = base64.b64decode(file_content_b64)
        except Exception as b64_err:
            raise ValueError(f"Failed to decode file content: {b64_err}")
        
        # Process based on job type
        if job_type == "process_rfp":
            logger.info(f"Starting RFP processing for job {job_id}")
            process_rfp_background(job_id, file_content, file_name, client_id, rfp_id)
            logger.info(f"Completed RFP processing for job {job_id}")
            
        elif job_type == "extract_qa":
            logger.info(f"Starting QA extraction for job {job_id}")
            extract_qa_background(job_id, file_content, file_name, client_id, rfp_id)
            logger.info(f"Completed QA extraction for job {job_id}")
            
        elif job_type == "ingest_text":
            logger.info(f"Starting text ingestion for job {job_id}")
            try:
                ingest_text_background(job_id, file_content, file_name, client_id, rfp_id)
                logger.info(f"Completed text ingestion for job {job_id}")
            except Exception as ingest_error:
                # Check if the job was actually completed (status might be updated even if exception raised)
                # This prevents marking successful jobs as failed due to status update errors
                try:
                    db = get_db_with_retry()
                    job_check = db.table("client_jobs").select("status, progress_percent, result_data").eq("id", job_id).limit(1).execute()
                    if job_check.data:
                        job_status = job_check.data[0].get("status")
                        job_progress = job_check.data[0].get("progress_percent", 0)
                        result_data = job_check.data[0].get("result_data", {})
                        
                        # If job is already marked as completed or has result_data with chunks_stored, don't mark as failed
                        if job_status == "completed" or (job_progress == 100 and result_data.get("chunks_stored", 0) > 0):
                            logger.info(f"Job {job_id} already marked as completed despite exception - ingestion succeeded")
                            return  # Don't mark as failed
                except:
                    pass
                
                # Only mark as failed if job wasn't already completed
                error_msg = f"Text ingestion failed: {str(ingest_error)}"
                logger.error(f"Error in text ingestion for job {job_id}: {error_msg}")
                traceback.print_exc()
                try:
                    update_job_progress(job_id, -1, error_msg)
                except Exception as update_error:
                    logger.error(f"Failed to update job progress: {update_error}")
            
        else:
            logger.error(f"Unknown job type: {job_type} (raw: {raw_job_type})")
            # List supported types for clarity in logs
            logger.info("Supported job types: process_rfp, extract_qa, ingest_text")
            raise ValueError(f"Unknown job type: {job_type}. Please use 'process_rfp', 'extract_qa', or 'ingest_text'.")
            
    except Exception as e:
        error_msg = f"Job processing failed: {str(e)}"
        logger.error(f"Error processing job {job_id}: {error_msg}")
        traceback.print_exc()
        try:
            update_job_progress(job_id, -1, error_msg)
        except Exception as update_error:
            logger.error(f"Failed to update job progress: {update_error}")
            # Try direct update as fallback
            try:
                db = get_db_with_retry()
                job_res = db.table("client_jobs").select("rfp_id").eq("id", job_id).limit(1).execute()
                if job_res.data and job_res.data[0].get("rfp_id"):
                    rfp_id = job_res.data[0]["rfp_id"]
                    db.table("client_rfps").update({
                        "last_job_status": "failed"
                    }).eq("id", rfp_id).execute()
            except Exception as fallback_error:
                logger.error(f"Fallback RFP status update also failed: {fallback_error}")


def worker_loop():
    """Main worker loop"""
    # Log connection type
    if is_direct_db_configured():
        logger.info("Worker started with DIRECT PostgreSQL connection (reliable, persistent)")
    else:
        logger.info("Worker started with Supabase HTTP client (fallback mode)")
    logger.info("Polling for jobs every 5 seconds")
    
    consecutive_errors = 0
    max_consecutive_errors = 10
    
    while True:
        try:
            # Reset stuck jobs periodically
            reset_stuck_jobs()
            
            # Get pending jobs
            pending_jobs = get_pending_jobs()
            
            if pending_jobs:
                logger.info(f"Found {len(pending_jobs)} pending job(s)")
                
                for job in pending_jobs:
                    try:
                        process_job(job)
                        consecutive_errors = 0  # Reset error counter on success
                    except Exception as e:
                        logger.error(f"Error processing job {job['id']}: {e}")
                        traceback.print_exc()
                        consecutive_errors += 1
                        
                        if consecutive_errors >= max_consecutive_errors:
                            logger.critical(f"Too many consecutive errors ({consecutive_errors}). Stopping worker.")
                            return
            else:
                logger.debug("No pending jobs found")
                consecutive_errors = 0  # Reset counter when no jobs
            
            # Sleep before next check
            time.sleep(5)
            
        except KeyboardInterrupt:
            logger.info("Worker stopped by user (KeyboardInterrupt)")
            break
        except Exception as e:
            logger.error(f"Worker loop error: {e}")
            traceback.print_exc()
            consecutive_errors += 1
            
            if consecutive_errors >= max_consecutive_errors:
                logger.critical(f"Too many consecutive errors ({consecutive_errors}). Stopping worker.")
                break
            
            time.sleep(10)  # Sleep longer on error


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("RFP Background Worker Starting")
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info("=" * 80)
    
    try:
        worker_loop()
    except Exception as e:
        logger.critical(f"Worker crashed: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        logger.info("Worker shutting down")
