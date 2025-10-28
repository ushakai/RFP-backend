#!/usr/bin/env python3
"""
Debug script to check job status in the database
"""

import os
from dotenv import load_dotenv
from supabase import create_client, Client
from datetime import datetime

# Load environment variables
load_dotenv()

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def check_jobs():
    """Check all jobs in the database"""
    try:
        # Get all jobs
        res = supabase.table("client_jobs").select("*").order("created_at", desc=True).execute()
        jobs = res.data or []
        
        print(f"Total jobs found: {len(jobs)}")
        print("=" * 80)
        
        for job in jobs:
            print(f"Job ID: {job['id']}")
            print(f"Type: {job['job_type']}")
            print(f"Status: {job['status']}")
            print(f"File: {job['file_name']}")
            print(f"Progress: {job['progress_percent']}%")
            print(f"Current Step: {job['current_step']}")
            print(f"Created: {job['created_at']}")
            print(f"Last Updated: {job.get('last_updated', 'N/A')}")
            if job.get('error_message'):
                print(f"Error: {job['error_message']}")
            print("-" * 40)
        
        # Check specifically for pending jobs
        pending_res = supabase.table("client_jobs").select("*").eq("status", "pending").execute()
        pending_jobs = pending_res.data or []
        
        print(f"\nPending jobs: {len(pending_jobs)}")
        for job in pending_jobs:
            print(f"  - {job['id']} ({job['job_type']}) - {job['file_name']}")
        
        # Check for processing jobs
        processing_res = supabase.table("client_jobs").select("*").eq("status", "processing").execute()
        processing_jobs = processing_res.data or []
        
        print(f"\nProcessing jobs: {len(processing_jobs)}")
        for job in processing_jobs:
            print(f"  - {job['id']} ({job['job_type']}) - {job['file_name']}")
            print(f"    Last updated: {job.get('last_updated', 'N/A')}")
        
    except Exception as e:
        print(f"Error checking jobs: {e}")

if __name__ == "__main__":
    check_jobs()
