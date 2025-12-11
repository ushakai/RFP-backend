"""
Authentication endpoints for client login and registration.
"""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, EmailStr, Field

from config.settings import get_supabase_client
from services.activity_service import record_event, record_session
from utils.auth import hash_password
from utils.logging_config import get_logger


router = APIRouter(prefix="/auth")
logger = get_logger(__name__, "auth")


class RegisterRequest(BaseModel):
    org_name: str = Field(..., min_length=1, description="Organization name")
    email: EmailStr
    password: str = Field(..., min_length=6)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1)


class AuthResponse(BaseModel):
    api_key: str
    email: EmailStr
    org_name: str | None = None
    role: str | None = None


@router.post("/register", response_model=AuthResponse)
def register(payload: RegisterRequest) -> AuthResponse:
    """Create a client row and return the generated API key."""
    supabase = get_supabase_client()

    org_name = payload.org_name.strip()
    email = payload.email.strip().lower()
    if not org_name:
        raise HTTPException(status_code=400, detail="Organization name is required")

    try:
        # Check if email already exists
        existing = (
            supabase.table("clients")
            .select("id, contact_email, status")
            .eq("contact_email", email)
            .limit(1)
            .execute()
        )
        if existing.data:
            raise HTTPException(status_code=409, detail="An account with this email already exists")

        api_key = str(uuid4())
        response = (
            supabase.table("clients")
            .insert(
                {
                    "name": org_name,
                    "contact_email": email,
                    "api_key": api_key,
                    "password_hash": hash_password(payload.password),
                    "status": "active",
                }
            )
            .execute()
        )
        if getattr(response, "error", None):
            logger.error("Supabase registration error: %s", response.error)
            raise HTTPException(status_code=500, detail="Failed to register organization")

        return AuthResponse(api_key=api_key, email=email, org_name=org_name)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Registration failed for %s", email)
        raise HTTPException(status_code=500, detail="Failed to register organization") from exc


@router.post("/login", response_model=AuthResponse)
def login(payload: LoginRequest, request: Request) -> AuthResponse:
    """Validate credentials and return the API key."""
    supabase = get_supabase_client()
    email = payload.email.strip().lower()
    password_hash = hash_password(payload.password)

    try:
        response = (
            supabase.table("clients")
            .select("id, api_key, password_hash, name, status, api_key_revoked, role")
            .eq("contact_email", email)
            .limit(1)
            .execute()
        )

        rows = response.data or []
        if not rows:
            raise HTTPException(status_code=401, detail="Invalid email or password")

        row = rows[0]
        
        # Check password first to avoid timing attacks
        stored_hash = (row.get("password_hash") or "").strip()
        if stored_hash and stored_hash != password_hash:
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        # Then check account status
        status = (row.get("status") or "active").lower()
        if status == "suspended":
            raise HTTPException(status_code=403, detail="Your account has been suspended. Please contact support.")
        if status != "active":
            raise HTTPException(status_code=403, detail="Your account is not active. Please contact support.")
        if row.get("api_key_revoked"):
            raise HTTPException(status_code=403, detail="Your access has been revoked. Please contact support.")

        api_key = row.get("api_key")
        if not api_key:
            logger.error("Client %s is missing api_key", email)
            raise HTTPException(status_code=500, detail="Account misconfigured")

        client_id = row.get("id")
        now_iso = datetime.utcnow().isoformat()
        try:
            supabase.table("clients").update({"last_login_at": now_iso, "last_active_at": now_iso}).eq("id", client_id).execute()
        except Exception:
            pass

        user_agent = request.headers.get("user-agent")
        ip_address = request.client.host if request.client else None
        record_session(client_id, api_key, user_agent=user_agent, ip_address=ip_address)
        record_event(
            "auth",
            "login",
            actor_client_id=client_id,
            actor_email=email,
            metadata={"ip": ip_address, "user_agent": user_agent},
        )

        return AuthResponse(api_key=api_key, email=email, org_name=row.get("name"), role=row.get("role"))
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Login failed for %s", email)
        raise HTTPException(status_code=500, detail="Failed to login") from exc

