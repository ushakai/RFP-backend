"""
Health check and test endpoints
"""
import traceback
from datetime import datetime
from fastapi import APIRouter
from config.settings import get_supabase_client

router = APIRouter()

@router.get("/")
def root():
    return {
        "message": "RFP Backend API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@router.get("/test")
def test_endpoint():
    print("=== TEST ENDPOINT CALLED ===")
    return {"message": "Backend is working!", "timestamp": datetime.now().isoformat()}

@router.get("/health")
def health_check():
    """Health check endpoint for monitoring"""
    try:
        supabase = get_supabase_client()
        # Test database connectivity
        supabase.table("clients").select("id").limit(1).execute()
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": "connected"
        }
    except Exception as e:
        print(f"ERROR: Health check failed: {e}")
        traceback.print_exc()
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "database": "disconnected",
            "error": str(e)
        }

