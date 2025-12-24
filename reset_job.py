#!/usr/bin/env python3
"""Quick script to reset a failed job to pending status"""

import sys
from config.settings import get_supabase_client
from datetime import datetime

job_id = "1e167b30-4a88-4804-a1f2-14cbc07d8d2e"

print(f"Resetting job {job_id} to pending status...")

try:
    supabase = get_supabase_client()
    
    # Update job status
    result = supabase.table("client_jobs").update({
        "status": "pending",
        "progress_percent": 0,
        "current_step": "Waiting for processing...",
        "error_message": None,
        "started_at": None,
        "last_updated": datetime.now().isoformat()
    }).eq("id", job_id).execute()
    
    if result.data:
        print("✅ Job reset successfully!")
        print(f"   Job ID: {job_id}")
        print(f"   Status: pending")
        print(f"   The worker should pick it up within 5 seconds")
    else:
        print("❌ Failed to reset job - job not found")
        
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

