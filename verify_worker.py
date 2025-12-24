#!/usr/bin/env python3
"""Verify worker can process ingest_text jobs"""

import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

from config.settings import get_supabase_client

print("=" * 60)
print("Worker Verification - Checking for ingest_text Support")
print("=" * 60)

# Check if worker code has the handler
print("\n[1] Checking worker.py code...")
with open("worker.py", "r") as f:
    code = f.read()
    if 'elif job_type == "ingest_text":' in code:
        print("   ✅ worker.py has ingest_text handler")
    else:
        print("   ❌ worker.py MISSING ingest_text handler!")
        sys.exit(1)

# Check if function is importable
print("\n[2] Checking imports...")
try:
    from services.job_service import ingest_text_background
    print("   ✅ ingest_text_background is importable")
except ImportError as e:
    print(f"   ❌ Cannot import ingest_text_background: {e}")
    sys.exit(1)

# Check for pending ingest_text jobs
print("\n[3] Checking for pending ingest_text jobs...")
try:
    supabase = get_supabase_client()
    jobs = supabase.table("client_jobs").select("*").eq("status", "pending").eq("job_type", "ingest_text").execute()
    
    if jobs.data:
        print(f"   Found {len(jobs.data)} pending ingest_text job(s):")
        for job in jobs.data[:3]:  # Show first 3
            print(f"      - Job {job['id'][:8]}... ({job['file_name']})")
        if len(jobs.data) > 3:
            print(f"      ... and {len(jobs.data) - 3} more")
    else:
        print("   ℹ️  No pending ingest_text jobs (this is OK)")
except Exception as e:
    print(f"   ⚠️  Could not check jobs: {e}")

# Check recent failed ingest_text jobs
print("\n[4] Checking for recent failed ingest_text jobs...")
try:
    cutoff = (datetime.now().timestamp() - 3600)  # Last hour
    failed = supabase.table("client_jobs").select("*").eq("status", "failed").eq("job_type", "ingest_text").gte("created_at", datetime.fromtimestamp(cutoff).isoformat()).execute()
    
    if failed.data:
        print(f"   ⚠️  Found {len(failed.data)} failed ingest_text job(s) in last hour:")
        for job in failed.data[:3]:
            error = job.get('current_step', 'Unknown error')
            print(f"      - Job {job['id'][:8]}...")
            print(f"        Error: {error[:80]}")
    else:
        print("   ✅ No recent failed ingest_text jobs")
except Exception as e:
    print(f"   ⚠️  Could not check failed jobs: {e}")

print("\n" + "=" * 60)
print("✅ VERIFICATION COMPLETE")
print("=" * 60)
print("\nIf you're still getting 'Unknown job type: ingest_text':")
print("1. The worker process (PID 6276) might be running OLD code")
print("2. Kill it: taskkill /F /PID 6276")
print("3. Restart: python worker.py")
print("=" * 60)


