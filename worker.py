#!/usr/bin/env python3
"""
Background Job Worker for RFP Processing
This script runs as a separate process to handle long-running jobs (15-20+ minutes)
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
from config.settings import get_supabase_client
from services.job_service import process_rfp_background, extract_qa_background, update_job_progress
from utils.logging_config import get_logger

# Setup logger
logger = get_logger(__name__, "worker")

def get_pending_jobs():
    """Get all pending jobs from the database"""
    try:
        supabase = get_supabase_client()
        res = supabase.table("client_jobs").select("*").eq("status", "pending").order("created_at", desc=False).execute()
        return res.data or []
    except Exception as e:
        logger.error(f"Failed to get pending jobs: {e}")
        traceback.print_exc()
        return []


def reset_stuck_jobs():
    """Reset jobs that have been stuck in processing for too long (30+ minutes)"""
    try:
        supabase = get_supabase_client()
        cutoff_time = (datetime.now() - timedelta(minutes=30)).isoformat()
        
        res = supabase.table("client_jobs").select("*").eq("status", "processing").lt("last_updated", cutoff_time).execute()
        stuck_jobs = res.data or []
        
        if stuck_jobs:
            logger.warning(f"Found {len(stuck_jobs)} stuck jobs, resetting to pending...")
            for job in stuck_jobs:
                supabase.table("client_jobs").update({
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
    job_id = job["id"]
    job_type = job["job_type"]
    job_data = job.get("job_data", {})
    
    logger.info(f"Processing job {job_id} (type: {job_type})")
    
    try:
        # Mark job as processing
        supabase = get_supabase_client()
        supabase.table("client_jobs").update({
            "status": "processing",
            "last_updated": datetime.now().isoformat()
        }).eq("id", job_id).execute()
        
        # Extract job data
        file_content_b64 = job_data.get("file_content")
        file_name = job_data.get("file_name")
        client_id = job_data.get("client_id")
        rfp_id = job_data.get("rfp_id")
        
        if not all([file_content_b64, file_name, client_id]):
            raise ValueError("Missing required job data fields")
        
        # Decode file content
        file_content = base64.b64decode(file_content_b64)
        
        # Process based on job type
        if job_type == "process_rfp":
            logger.info(f"Starting RFP processing for job {job_id}")
            process_rfp_background(job_id, file_content, file_name, client_id, rfp_id)
            logger.info(f"Completed RFP processing for job {job_id}")
            
        elif job_type == "extract_qa":
            logger.info(f"Starting QA extraction for job {job_id}")
            extract_qa_background(job_id, file_content, file_name, client_id, rfp_id)
            logger.info(f"Completed QA extraction for job {job_id}")
            
        else:
            raise ValueError(f"Unknown job type: {job_type}")
            
    except Exception as e:
        error_msg = f"Job processing failed: {str(e)}"
        logger.error(f"Error processing job {job_id}: {error_msg}")
        traceback.print_exc()
        update_job_progress(job_id, -1, error_msg)


def worker_loop():
    """Main worker loop"""
    logger.info("Worker started - polling for jobs every 5 seconds")
    
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
