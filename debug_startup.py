"""
Debug script to identify startup issues
"""
import sys
import traceback

print("=" * 80)
print("DEBUGGING BACKEND STARTUP")
print("=" * 80)

try:
    print("\n1. Testing basic imports...")
    import os
    import time
    print("   ✓ Standard library imports OK")
    
    print("\n2. Testing FastAPI import...")
    from fastapi import FastAPI
    print("   ✓ FastAPI import OK")
    
    print("\n3. Testing config imports...")
    from config.settings import get_supabase_client
    print("   ✓ Config imports OK")
    
    print("\n4. Testing cache_manager import...")
    from utils.cache_manager import invalidate_client_caches
    print("   ✓ Cache manager import OK")
    
    print("\n5. Testing services imports...")
    from services.tender_service import rematch_for_client
    print("   ✓ Tender service import OK")
    
    print("\n6. Testing API routers import...")
    from api import health, tenders
    print("   ✓ API routers import OK")
    
    print("\n7. Testing app import...")
    from app import app
    print("   ✓ App import OK")
    
    print("\n8. Testing Supabase connection...")
    supabase = get_supabase_client()
    result = supabase.table("clients").select("id").limit(1).execute()
    print(f"   ✓ Supabase connection OK (found {len(result.data)} records)")
    
    print("\n" + "=" * 80)
    print("✓ ALL CHECKS PASSED - SERVER SHOULD START NORMALLY")
    print("=" * 80)
    
except Exception as e:
    print(f"\n✗ ERROR FOUND: {type(e).__name__}: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    print("\n" + "=" * 80)
    print("FIX THE ERROR ABOVE BEFORE STARTING THE SERVER")
    print("=" * 80)
    sys.exit(1)

