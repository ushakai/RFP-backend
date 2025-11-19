"""
Authentication utilities
"""
import time
import traceback
from fastapi import HTTPException
from typing import Optional
import httpx
import httpcore

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
    
    attempts = 5
    delay = 0.2
    last_exc: Exception | None = None
    
    for attempt in range(attempts):
        try:
            supabase = get_supabase_client()
            resp = supabase.table("clients").select("id").eq("api_key", client_key).limit(1).execute()
            rows = resp.data or []
            if not rows:
                raise HTTPException(status_code=401, detail="Invalid X-Client-Key")
            return rows[0]["id"]
        except HTTPException:
            raise
        except Exception as exc:
            last_exc = exc
            error_msg = str(exc)
            error_type = type(exc).__name__
            
            # Check if this is a network/connection error that should be retried
            is_retryable = (
                isinstance(exc, (httpx.HTTPError, httpx.ReadError, httpx.ConnectError, httpx.TimeoutException,
                                ConnectionError, OSError)) or
                "ReadError" in error_type or
                "ConnectError" in error_type or
                "WinError 10035" in error_msg or
                "non-blocking socket" in error_msg.lower() or
                "connection" in error_msg.lower()
            )
            
            # Windows socket error - use longer delay
            is_windows_socket_error = "WinError 10035" in error_msg or "non-blocking socket" in error_msg.lower()
            
            if is_retryable and attempt < attempts - 1:
                # Longer delay for Windows socket errors
                base_delay = delay * (attempt + 1)
                wait_time = base_delay * (3.0 if is_windows_socket_error else 1.0)
                
                if is_windows_socket_error:
                    print(f"WARNING: Windows socket error detected in client lookup (attempt {attempt + 1}/{attempts}), retrying in {wait_time:.2f}s...")
                else:
                    print(f"WARNING: Client lookup failed (attempt {attempt + 1}/{attempts}): {error_type}")
                
                time.sleep(wait_time)
                
                # Reinitialize Supabase connection on retry (but not on first retry to avoid overhead)
                if attempt >= 1:
                    try:
                        reinitialize_supabase()
                    except Exception:
                        pass  # Ignore reinit errors
                continue
            elif not is_retryable:
                # Non-retryable error - raise immediately
                raise exc
            else:
                # Last attempt failed
                if is_windows_socket_error:
                    print(f"ERROR: Client lookup failed after {attempts} attempts due to Windows socket error")
                else:
                    print(f"ERROR: Client lookup failed after {attempts} attempts: {error_type}")
                    traceback.print_exc()
                break
    
    if last_exc:
        raise HTTPException(status_code=500, detail="Client lookup failed after retries")
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

    attempts = 3
    delay = 0.2
    
    for attempt in range(attempts):
        try:
            supabase = get_supabase_client()
            resp = supabase.table("clients").select("contact_email").eq("id", client_id).limit(1).execute()
            rows = resp.data or []
            if not rows:
                return False
            email = (rows[0].get("contact_email") or "").strip().lower()
            return email in ADMIN_CLIENT_EMAILS
        except Exception as exc:
            error_msg = str(exc)
            is_windows_socket_error = "WinError 10035" in error_msg or "non-blocking socket" in error_msg.lower()
            
            if attempt < attempts - 1:
                wait_time = delay * (attempt + 1) * (3.0 if is_windows_socket_error else 1.0)
                if is_windows_socket_error:
                    print(f"WARNING: Windows socket error in admin check (attempt {attempt + 1}/{attempts}), retrying...")
                time.sleep(wait_time)
                if attempt >= 1:
                    try:
                        reinitialize_supabase()
                    except Exception:
                        pass
                continue
            else:
                print(f"Failed to determine admin status for {client_id}: {exc}")
                return False
    return False

