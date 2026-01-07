"""Authentication and authorization utilities."""
import hashlib
import os
import time
import traceback
from datetime import datetime, timedelta, timezone

import httpx
import httpcore
import jwt
from fastapi import HTTPException, Header, Depends

from config.settings import (
    ADMIN_CLIENT_EMAILS,
    ADMIN_CLIENT_IDS,
    ADMIN_JWT_EXPIRES_MINUTES,
    ADMIN_JWT_SECRET,
    ADMIN_LOGIN_MAX_ATTEMPTS,
    ADMIN_ROLE_NAME,
    SUPER_ADMIN_EMAIL,
    PROTECTED_ADMIN_EMAILS,
    get_supabase_client,
    reinitialize_supabase,
)


def hash_password(password: str) -> str:
    """Return a deterministic SHA-256 hash for the supplied password."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


# Simple in-memory cache to reduce DB hits and socket exhaustion
# Format: {api_key: (client_id, expiry_timestamp)}
CLIENT_CACHE = {}
CACHE_TTL = 60  # 1 minute cache

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
        
    # Check cache first
    now = time.time()
    if client_key in CLIENT_CACHE:
        client_id, expiry = CLIENT_CACHE[client_key]
        if now < expiry:
            return client_id
        else:
            del CLIENT_CACHE[client_key]
    
    # Use fewer retries in production to avoid timeouts
    attempts = 3 if os.getenv("RENDER") or os.getenv("VERCEL") else 5
    delay = 0.3
    last_exc: Exception | None = None
    
    for attempt in range(attempts):
        try:
            supabase = get_supabase_client()
            resp = (
                supabase.table("clients")
                .select("id, status, api_key_revoked, role, last_active_at, name, contact_email")
                .eq("api_key", client_key)
                .limit(1)
                .execute()
            )
            rows = resp.data or []
            if not rows:
                # Don't retry for invalid API keys - fail fast
                raise HTTPException(status_code=401, detail="Invalid X-Client-Key")

            row = rows[0]
            status = (row.get("status") or "active").lower()
            if status != "active":
                raise HTTPException(status_code=403, detail="Account is suspended")

            if row.get("api_key_revoked"):
                raise HTTPException(status_code=403, detail="API key revoked")

            client_id = row["id"]

            # Check session blacklist
            try:
                sess = (
                    supabase.table("client_sessions")
                    .select("revoked")
                    .eq("api_key", client_key)
                    .limit(1)
                    .execute()
                )
                sess_rows = sess.data or []
                if sess_rows and sess_rows[0].get("revoked"):
                    raise HTTPException(status_code=403, detail="Session revoked")
            except HTTPException:
                raise
            except Exception:
                # Non-blocking
                pass

            # Opportunistically bump last_active_at; failures are non-blocking
            try:
                supabase.table("clients").update(
                    {"last_active_at": datetime.now(timezone.utc).isoformat()}
                ).eq("id", client_id).execute()
            except Exception:
                pass

            # Cache the result
            CLIENT_CACHE[client_key] = (client_id, time.time() + CACHE_TTL)

            return client_id
        except HTTPException as http_exc:
            # Don't retry for authentication/authorization errors - fail fast
            print(f"Authentication error: {http_exc.detail}")
            raise
        except Exception as exc:
            last_exc = exc
            error_msg = str(exc)
            error_type = type(exc).__name__
            
            # Check if this is a network/connection error that should be retried
            is_retryable = (
                isinstance(exc, (httpx.HTTPError, httpx.ReadError, httpx.ConnectError, httpx.TimeoutException,
                                httpcore.ReadError, httpcore.RemoteProtocolError,
                                ConnectionError, OSError, RuntimeError)) or
                "ReadError" in error_type or
                "ConnectError" in error_type or
                "RemoteProtocolError" in error_type or
                "WinError 10035" in error_msg or
                "non-blocking socket" in error_msg.lower() or
                "connection" in error_msg.lower() or
                "client has been closed" in error_msg.lower() or
                "Server disconnected" in error_msg
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


def get_client_id(x_client_key: str | None = Header(None, alias="X-Client-Key")) -> str:
    """FastAPI dependency to get validated client ID from header."""
    return get_client_id_from_key(x_client_key)

def is_admin_client(client_id: str) -> bool:
    """
    Determine whether the given client ID has administrator privileges.
    Admins can be configured via ADMIN_CLIENT_IDS or ADMIN_CLIENT_EMAILS environment variables.
    """
    try:
        supabase = get_supabase_client()
        resp = (
            supabase.table("clients")
            .select("contact_email, role")
            .eq("id", client_id)
            .limit(1)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            return False
        role = (rows[0].get("role") or "").strip().lower()
        if role == ADMIN_ROLE_NAME:
            return True
    except Exception:
        # fall back to legacy checks if the role column is unavailable
        pass

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
            is_connection_error = (
                "WinError 10035" in error_msg or 
                "non-blocking socket" in error_msg.lower() or
                "Server disconnected" in error_msg or
                "client has been closed" in error_msg.lower()
            )
            
            if attempt < attempts - 1:
                wait_time = delay * (attempt + 1) * (2.0 if is_connection_error else 1.0)
                time.sleep(wait_time)
                try:
                    reinitialize_supabase()
                except Exception:
                    pass
                continue
            else:
                print(f"Failed to determine admin status for {client_id}: {exc}")
                return False
    return False


def is_super_admin_client(client_id: str) -> bool:
    """Return True if the client is the configured super admin."""
    try:
        supabase = get_supabase_client()
        resp = (
            supabase.table("clients")
            .select("contact_email")
            .eq("id", client_id)
            .limit(1)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            return False
        email = (rows[0].get("contact_email") or "").strip().lower()
        return email == SUPER_ADMIN_EMAIL
    except Exception:
        return False


def is_protected_admin(client_id: str) -> bool:
    """Return True if the client is a protected admin (cannot be modified)."""
    try:
        supabase = get_supabase_client()
        resp = (
            supabase.table("clients")
            .select("contact_email")
            .eq("id", client_id)
            .limit(1)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            return False
        email = (rows[0].get("contact_email") or "").strip().lower()
        return email in PROTECTED_ADMIN_EMAILS
    except Exception:
        return False


def verify_admin_credentials(email: str, password: str) -> str | None:
    """
    Validate admin credentials against the clients table (role == ADMIN_ROLE_NAME).
    Returns the admin client ID when valid, otherwise None.
    """
    normalized_email = (email or "").strip().lower()
    password_hash = hash_password(password)

    attempts = max(1, ADMIN_LOGIN_MAX_ATTEMPTS)
    delay = 0.2

    for attempt in range(attempts):
        try:
            supabase = get_supabase_client()
            resp = (
                supabase.table("clients")
                .select("id, password_hash, role, status, api_key_revoked, contact_email")
                .eq("contact_email", normalized_email)
                .limit(1)
                .execute()
            )
            rows = resp.data or []
            if not rows:
                return None

            row = rows[0]
            status = (row.get("status") or "active").lower()
            role = (row.get("role") or "").strip().lower()
            
            # Use logger if available or print
            print(f"DEBUG: Admin login attempt for {normalized_email}. Status={status}, Role={role}")

            if status != "active" or row.get("api_key_revoked"):
                # super admin is always allowed even if mis-set
                if (row.get("contact_email") or "").strip().lower() != SUPER_ADMIN_EMAIL:
                    print(f"DEBUG: Admin {normalized_email} rejected due to status/revocation")
                    return None

            if role != ADMIN_ROLE_NAME:
                print(f"DEBUG: Admin {normalized_email} rejected due to role mismatch (expected {ADMIN_ROLE_NAME})")
                return None

            stored_hash = (row.get("password_hash") or "").strip()
            if stored_hash and stored_hash == password_hash:
                return row["id"]
            
            print(f"DEBUG: Admin {normalized_email} rejected due to password mismatch")
            return None
        except Exception:
            if attempt < attempts - 1:
                time.sleep(delay * (attempt + 1))
                if attempt >= 1:
                    try:
                        reinitialize_supabase()
                    except Exception:
                        pass
                continue
            raise
    return None


def create_admin_token(admin_id: str, admin_email: str) -> str:
    """Create a signed JWT for admin sessions."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": admin_id,
        "email": admin_email,
        "role": ADMIN_ROLE_NAME,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=ADMIN_JWT_EXPIRES_MINUTES)).timestamp()),
    }
    return jwt.encode(payload, ADMIN_JWT_SECRET, algorithm="HS256")


def decode_admin_token(token: str) -> dict:
    """Decode and validate an admin JWT."""
    try:
        return jwt.decode(token, ADMIN_JWT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status_code=401, detail="Admin token expired") from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail="Invalid admin token") from exc


def require_admin(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
    x_client_key: str | None = Header(default=None, alias="X-Client-Key"),
) -> dict:
    """
    FastAPI dependency to require admin authentication.
    Accepts either Bearer JWT token, x-api-key, or X-Client-Key for admin users.
    Returns a dict with 'sub' (client_id) and 'email'.
    """
    # Try JWT first
    if authorization and authorization.lower().startswith("bearer "):
        try:
            token = authorization.split(" ", 1)[1].strip()
            claims = decode_admin_token(token)
            if claims.get("role") != ADMIN_ROLE_NAME:
                raise HTTPException(status_code=403, detail="Admin role required")
            return claims
        except HTTPException as e:
            # If it's a 403 (role mismatch), we should probably keep it
            if e.status_code == 403:
                raise
            # For 401 (expired/invalid), fall through to API key if provided
            if not (x_api_key or x_client_key):
                raise
        except Exception:
            if not (x_api_key or x_client_key):
                raise HTTPException(status_code=401, detail="Invalid admin token")
    
    # Try API key (accept both x-api-key and X-Client-Key for compatibility)
    api_key = x_api_key or x_client_key
    if api_key:
        client_id = get_client_id_from_key(api_key)
        if not is_admin_client(client_id):
            raise HTTPException(status_code=403, detail="Admin role required")
        
        # Fetch email for consistency
        try:
            supabase = get_supabase_client()
            resp = supabase.table("clients").select("contact_email").eq("id", client_id).limit(1).execute()
            rows = resp.data or []
            email = rows[0].get("contact_email") if rows else ""
        except Exception:
            email = ""
        
        return {"sub": client_id, "email": email, "role": ADMIN_ROLE_NAME}
    
    raise HTTPException(status_code=401, detail="Missing authentication")


def require_subscription(required_tiers: list[str] = ["tenders", "processing", "both"]):
    """
    FastAPI dependency factory to require a valid subscription.
    """
    def _require_subscription(client_id: str = Depends(get_client_id)) -> dict:
        try:
            supabase = get_supabase_client()
            resp = (
                supabase.table("clients")
                .select("subscription_status, subscription_tier, trial_end, subscription_period_end, role")
                .eq("id", client_id)
                .limit(1)
                .execute()
            )
            rows = resp.data or []
            if not rows:
                raise HTTPException(status_code=404, detail="Client not found")
            
            row = rows[0]
            status = row.get("subscription_status", "inactive")
            tier = row.get("subscription_tier", "free")
            role = row.get("role", "client")
            
            # Admins have full access
            if role == "admin":
                return row
                
            # Check if status is active or trialing
            is_active = status in ["active", "trialing"]
            
            if not is_active:
                # Check if it was trialing but trial ended
                raise HTTPException(
                    status_code=402, 
                    detail="Subscription required. Please upgrade your plan."
                )
                
            # Check if tier matches
            if tier not in required_tiers and tier != "both":
                raise HTTPException(
                    status_code=403, 
                    detail=f"Subscription tier '{tier}' does not have access to this feature. Required tiers: {required_tiers}"
                )
                
            return row
        except HTTPException:
            raise
        except Exception as exc:
            print(f"Error checking subscription for {client_id}: {exc}")
            # In case of DB error, we might want to fail closed or open depending on policy
            # For now, fail closed (secure)
            raise HTTPException(status_code=500, detail="Failed to verify subscription status")
            
    return _require_subscription
