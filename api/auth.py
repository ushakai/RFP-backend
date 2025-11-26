"""
Authentication endpoints for client login and registration.
"""

from __future__ import annotations

import hashlib
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr, Field

from config.settings import get_supabase_client
from utils.logging_config import get_logger


router = APIRouter(prefix="/auth")
logger = get_logger(__name__, "auth")


def _hash_password(password: str) -> str:
    """Return a deterministic SHA-256 hash for the supplied password."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


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


@router.post("/register", response_model=AuthResponse)
def register(payload: RegisterRequest) -> AuthResponse:
    """Create a client row and return the generated API key."""
    supabase = get_supabase_client()

    org_name = payload.org_name.strip()
    email = payload.email.strip().lower()
    if not org_name:
        raise HTTPException(status_code=400, detail="Organization name is required")

    try:
        existing = (
            supabase.table("clients")
            .select("id")
            .eq("contact_email", email)
            .limit(1)
            .execute()
        )
        if existing.data:
            raise HTTPException(status_code=409, detail="Email already registered")

        api_key = str(uuid4())
        response = (
            supabase.table("clients")
            .insert(
                {
                    "name": org_name,
                    "contact_email": email,
                    "api_key": api_key,
                    "password_hash": _hash_password(payload.password),
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
def login(payload: LoginRequest) -> AuthResponse:
    """Validate credentials and return the API key."""
    supabase = get_supabase_client()
    email = payload.email.strip().lower()
    password_hash = _hash_password(payload.password)

    try:
        response = (
            supabase.table("clients")
            .select("api_key, password_hash, name")
            .eq("contact_email", email)
            .limit(1)
            .execute()
        )

        rows = response.data or []
        if not rows:
            raise HTTPException(status_code=401, detail="Invalid email or password")

        row = rows[0]
        stored_hash = (row.get("password_hash") or "").strip()
        if stored_hash and stored_hash != password_hash:
            raise HTTPException(status_code=401, detail="Invalid email or password")

        api_key = row.get("api_key")
        if not api_key:
            logger.error("Client %s is missing api_key", email)
            raise HTTPException(status_code=500, detail="Account misconfigured")

        return AuthResponse(api_key=api_key, email=email, org_name=row.get("name"))
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Login failed for %s", email)
        raise HTTPException(status_code=500, detail="Failed to login") from exc

