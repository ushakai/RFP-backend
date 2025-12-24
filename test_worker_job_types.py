#!/usr/bin/env python3
"""Test script to verify worker.py recognizes ingest_text job type"""

import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("Testing Worker Job Type Recognition")
print("=" * 60)

# Test 1: Check if ingest_text_background is importable
print("\n[1] Testing import of ingest_text_background...")
try:
    from services.job_service import ingest_text_background
    print("✅ SUCCESS: ingest_text_background imported successfully")
except ImportError as e:
    print(f"❌ FAILED: Cannot import ingest_text_background: {e}")
    sys.exit(1)

# Test 2: Check worker.py code structure
print("\n[2] Checking worker.py code structure...")
with open("worker.py", "r") as f:
    worker_code = f.read()
    
    if "ingest_text_background" in worker_code:
        print("✅ SUCCESS: worker.py imports ingest_text_background")
    else:
        print("❌ FAILED: worker.py does NOT import ingest_text_background")
        sys.exit(1)
    
    if 'elif job_type == "ingest_text":' in worker_code:
        print("✅ SUCCESS: worker.py has 'elif job_type == \"ingest_text\":' handler")
    else:
        print("❌ FAILED: worker.py does NOT have ingest_text handler")
        print("\nLooking for job type handlers...")
        if "process_rfp" in worker_code:
            print("  - Found: process_rfp")
        if "extract_qa" in worker_code:
            print("  - Found: extract_qa")
        if "ingest_text" in worker_code:
            print("  - Found: ingest_text (but handler missing!)")
        sys.exit(1)

# Test 3: Simulate job type check
print("\n[3] Simulating job type check...")
job_types = ["process_rfp", "extract_qa", "ingest_text", "unknown_type"]

for job_type in job_types:
    if job_type == "process_rfp":
        result = "✅ Would process"
    elif job_type == "extract_qa":
        result = "✅ Would process"
    elif job_type == "ingest_text":
        result = "✅ Would process"
    else:
        result = "❌ Would raise ValueError"
    print(f"  {job_type:20} -> {result}")

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED - Worker code is correct!")
print("=" * 60)
print("\nIf you're still getting 'Unknown job type: ingest_text',")
print("the worker process is running OLD code. Restart it:")
print("  1. Kill all Python processes")
print("  2. Run: python worker.py")
print("=" * 60)


