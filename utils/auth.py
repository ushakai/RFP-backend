"""
Authentication utilities
"""
import time
import traceback
from fastapi import HTTPException
from typing import Optional

from config.settings import (
    ADMIN_CLIENT_EMAILS,
    ADMIN_CLIENT_IDS,
    get_supabase_client,
    reinitialize_supabase,
)

def get_client_id_from_key(client_key: str | None) -> str:
    """
    Validate client API key and return client ID
    
    Args:
        client_key: API key from X-Client-Key header
        
    Returns:
        Client ID string
        
    Raises:
        HTTPException: If key is missing or invalid
    """
    if not client_key:
        raise HTTPException(status_code=401, detail="Missing X-Client-Key")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            supabase = get_supabase_client()
            resp = supabase.table("clients").select("id").eq("api_key", client_key).limit(1).execute()
            rows = resp.data or []
            if not rows:
                raise HTTPException(status_code=401, detail="Invalid X-Client-Key")
            return rows[0]["id"]
        except HTTPException:
            raise
        except Exception as e:
            print(f"Client lookup error: {e}")
            traceback.print_exc()
            # Recreate supabase client and retry on transient errors
            try:
                reinitialize_supabase()
            except Exception as reinit_err:
                print(f"Supabase re-init failed: {reinit_err}")
                traceback.print_exc()
            if attempt < max_retries - 1:
                try:
                    time.sleep(0.5)
                except Exception:
                    pass
                continue
    raise HTTPException(status_code=500, detail="Client lookup failed")


def is_admin_client(client_id: str) -> bool:
    """
    Determine whether the given client ID has administrator privileges.
    Admins can be configured via ADMIN_CLIENT_IDS or ADMIN_CLIENT_EMAILS environment variables.
    """
    if client_id in ADMIN_CLIENT_IDS:
        return True

    if not ADMIN_CLIENT_EMAILS:
        return False

    try:
        supabase = get_supabase_client()
        resp = supabase.table("clients").select("contact_email").eq("id", client_id).limit(1).execute()
        rows = resp.data or []
        if not rows:
            return False
        email = (rows[0].get("contact_email") or "").strip().lower()
        return email in ADMIN_CLIENT_EMAILS
    except Exception as exc:
        print(f"Failed to determine admin status for {client_id}: {exc}")
        return False

