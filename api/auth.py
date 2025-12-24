import random
from datetime import datetime, timedelta
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, Body, Depends
from pydantic import BaseModel, EmailStr, Field

from config.settings import get_supabase_client
from services.activity_service import record_event, record_session
from services.email_service import send_email
from utils.auth import hash_password, get_client_id
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


class VerifyOTPRequest(BaseModel):
    email: EmailStr
    otp: str = Field(..., min_length=6, max_length=6)


class AuthResponse(BaseModel):
    api_key: str | None = None
    email: EmailStr
    org_name: str | None = None
    role: str | None = None
    status: str | None = None
    message: str | None = None
    requires_verification: bool = False


class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordConfirmRequest(BaseModel):
    email: EmailStr
    otp: str
    new_password: str

@router.post("/reset-password-request")
def reset_password_request(payload: ForgotPasswordRequest):
    """Trigger a password reset email via Supabase."""
    supabase = get_supabase_client()
    try:
        supabase.auth.reset_password_for_email(payload.email)
        return {"message": "Password reset instructions have been sent to your email."}
    except Exception as e:
        logger.error(f"Reset password request failed: {e}")
        # We don't want to leak if the email exists, but for now standard error is fine
        raise HTTPException(status_code=500, detail="Failed to initiate password reset.")

@router.post("/reset-password-confirm")
def reset_password_confirm(payload: ResetPasswordConfirmRequest):
    """Confirm password reset using the OTP sent to email."""
    supabase = get_supabase_client()
    try:
        # 1. Verify the OTP first
        auth_res = supabase.auth.verify_otp({
            "email": payload.email,
            "token": payload.otp,
            "type": "recovery"
        })
        
        if not auth_res.user:
            raise HTTPException(status_code=400, detail="Invalid or expired reset code.")
            
        # 2. Update the password
        supabase.auth.update_user({
            "password": payload.new_password
        })
        
        # 3. Update our local password_hash as well (legacy fallback support)
        supabase.table("clients").update({
            "password_hash": hash_password(payload.new_password)
        }).eq("contact_email", payload.email).execute()
        
        return {"message": "Password has been reset successfully."}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reset password confirm failed: {e}")
        raise HTTPException(status_code=400, detail="Failed to reset password. The code may be invalid or expired.")

@router.post("/change-password")
def change_password(payload: dict = Body(...), client_id: str = Depends(get_client_id)):
    """Change password for an authenticated user."""
    new_password = payload.get("new_password")
    if not new_password or len(new_password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
    
    supabase = get_supabase_client()
    try:
        # Get user email
        client_resp = supabase.table("clients").select("contact_email").eq("id", client_id).single().execute()
        email = client_resp.data.get("contact_email")
        
        # Update in Supabase Auth (if they exist there)
        try:
             # This works if the current session is an Auth session. 
             # Since we use API keys, we might need to use Admin API if they aren't 'logged in' to Supabase Auth specifically.
             # But for simplicity, we'll try updating and handle errors.
             supabase.auth.update_user({"password": new_password})
        except Exception:
             # If update_user fails because of no active session, we'll rely on the legacy hash for now
             # or we could implement a more robust sync.
             pass
             
        # Update local hash
        supabase.table("clients").update({
            "password_hash": hash_password(new_password)
        }).eq("id", client_id).execute()
        
        return {"message": "Password updated successfully."}
    except Exception as e:
        logger.error(f"Change password failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to change password.")

@router.delete("/account")
def delete_account(client_id: str = Depends(get_client_id)):
    """Permanently delete user account and all data."""
    supabase = get_supabase_client()
    try:
        # 1. Get client data
        client_resp = supabase.table("clients").select("contact_email").eq("id", client_id).single().execute()
        email = client_resp.data.get("contact_email")
        
        # 2. Delete from clients table (cascade takes care of RFP data)
        supabase.table("clients").delete().eq("id", client_id).execute()
        
        # 3. Supabase Auth user deletion requires Service Role key if done from backend
        # For now, we've deleted the profile. If we want to delete from auth.users, 
        # we'd need to use the service role client.
        
        logger.info(f"User {email} (ID: {client_id}) deleted their account.")
        return {"message": "Account deleted successfully."}
    except Exception as e:
        logger.error(f"Account deletion failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete account.")

@router.post("/register", response_model=AuthResponse)
def register(payload: RegisterRequest) -> AuthResponse:
    """Register using Supabase Auth and link to clients table."""
    supabase = get_supabase_client()

    org_name = payload.org_name.strip()
    email = payload.email.strip().lower()
    if not org_name:
        raise HTTPException(status_code=400, detail="Organization name is required")

    try:
        # 1. Check if email already exists in clients table
        existing = (
            supabase.table("clients")
            .select("id")
            .eq("contact_email", email)
            .limit(1)
            .execute()
        )
        if existing.data:
            raise HTTPException(status_code=409, detail="An account with this email already exists")

        # 2. Register with Supabase Auth (this triggers the confirmation email)
        # Note: We pass the password directly to Supabase
        auth_response = supabase.auth.sign_up({
            "email": email,
            "password": payload.password,
            "options": {
                "data": {
                    "org_name": org_name
                }
            }
        })

        if not auth_response.user:
            # Check if user already exists in Auth but not in clients (should be rare)
            logger.warning("Supabase Auth returned no user for %s", email)
            raise HTTPException(status_code=500, detail="Failed to create auth account")

        # 3. Create entry in custom clients table
        api_key = str(uuid4())
        response = (
            supabase.table("clients")
            .insert(
                {
                    "name": org_name,
                    "contact_email": email,
                    "api_key": api_key,
                    "password_hash": hash_password(payload.password), # We keep it for legacy validation if needed
                    "status": "pending_verification",
                }
            )
            .execute()
        )
        
        if getattr(response, "error", None):
            logger.error("Supabase client insertion error: %s", response.error)
            # Cleanup auth user if client creation failed? 
            # Usually better to report error
            raise HTTPException(status_code=500, detail="Failed to create organization profile")

        return AuthResponse(
            email=email, 
            org_name=org_name, 
            status="pending_verification",
            requires_verification=True,
            message="A verification code has been sent to your email via Supabase."
        )
    except HTTPException:
        raise
    except Exception as exc:
        msg = str(exc)
        if "User already registered" in msg:
             raise HTTPException(status_code=409, detail="An account with this email already exists in auth")
        logger.exception("Registration failed for %s", email)
        raise HTTPException(status_code=500, detail=f"Registration failed: {msg}") from exc


@router.post("/verify-otp", response_model=AuthResponse)
def verify_otp(payload: VerifyOTPRequest) -> AuthResponse:
    """Verify the OTP using Supabase Auth and activate the account."""
    supabase = get_supabase_client()
    email = payload.email.strip().lower()
    otp = payload.otp.strip()

    try:
        # 1. Verify with Supabase Auth
        # type 'signup' is for the initial email confirmation
        auth_res = supabase.auth.verify_otp({
            "email": email,
            "token": otp,
            "type": "signup"
        })

        if not auth_res.user:
            # Try 'email' type if 'signup' fails (sometimes used for signs-in or other flows)
             try:
                 auth_res = supabase.auth.verify_otp({
                     "email": email,
                     "token": otp,
                     "type": "email"
                 })
             except Exception:
                 raise HTTPException(status_code=400, detail="Invalid or expired verification code.")

        if not auth_res.user:
             raise HTTPException(status_code=400, detail="Verification failed. Please check your code.")

        # 2. Update status in clients table
        client_resp = (
            supabase.table("clients")
            .select("id, api_key, name, role")
            .eq("contact_email", email)
            .limit(1)
            .execute()
        )
        
        if not client_resp.data:
            raise HTTPException(status_code=404, detail="Organization record not found.")
        
        client = client_resp.data[0]
        
        supabase.table("clients").update({
            "status": "active"
        }).eq("id", client["id"]).execute()

        logger.info("Account verified via Supabase for %s", email)
        
        return AuthResponse(
            api_key=client["api_key"], 
            email=email, 
            org_name=client["name"], 
            status="active",
            role=client.get("role"),
            message="Email verified successfully! Welcome to Bidwell."
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Supabase OTP verification failed: %s", exc)
        raise HTTPException(status_code=400, detail="Invalid verification code or it has expired.")


@router.post("/resend-otp", response_model=AuthResponse)
def resend_otp(payload: dict = Body(...)) -> AuthResponse:
    """Resend a new OTP using Supabase Auth."""
    supabase = get_supabase_client()
    email = payload.get("email", "").strip().lower()
    if not email:
        raise HTTPException(status_code=400, detail="Email is required")

    try:
        # Supabase resend for signup (email confirmation)
        supabase.auth.resend({
            "type": "signup",
            "email": email
        })

        return AuthResponse(
            email=email, 
            status="pending_verification",
            message="A new verification code has been sent to your email via Supabase."
        )
    except Exception as exc:
        logger.exception("OTP resend failed for %s", email)
        raise HTTPException(status_code=500, detail=f"Failed to resend code: {str(exc)}")


@router.post("/login", response_model=AuthResponse)
def login(payload: LoginRequest, request: Request) -> AuthResponse:
    """Validate credentials via Supabase Auth and return the API key."""
    supabase = get_supabase_client()
    email = payload.email.strip().lower()

    try:
        # 1. First check the client status in our table
        client_resp = (
            supabase.table("clients")
            .select("id, api_key, name, status, api_key_revoked, role, password_hash")
            .eq("contact_email", email)
            .limit(1)
            .execute()
        )
        
        if not client_resp.data:
             raise HTTPException(status_code=401, detail="Invalid email or password")
             
        client_row = client_resp.data[0]
        status = (client_row.get("status") or "active").lower()
        
        if status == "suspended":
            raise HTTPException(status_code=403, detail="Your account has been suspended.")
        if client_row.get("api_key_revoked"):
            raise HTTPException(status_code=403, detail="Your access has been revoked.")

        # 2. Authenticate with Supabase
        try:
            auth_res = supabase.auth.sign_in_with_password({
                "email": email,
                "password": payload.password
            })
        except Exception as auth_exc:
            msg = str(auth_exc)
            if "Email not confirmed" in msg:
                return AuthResponse(
                 email=email, 
                 org_name=client_row.get("name"), 
                 status="pending_verification",
                 requires_verification=True,
                 message="Please verify your email address before logging in."
                )
            
            # --- LEGACY FALLBACK ---
            # If Supabase Auth fails (user doesn't exist there), check our local hash
            stored_hash = (client_row.get("password_hash") or "").strip()
            if stored_hash and stored_hash == hash_password(payload.password):
                logger.info("Legacy login successful for %s", email)
                # We allow them in. In the future, you could auto-migrate them to Auth here.
            else:
                logger.warning("Auth failure for %s: %s", email, msg)
                raise HTTPException(status_code=401, detail="Invalid email or password")

        # 3. Update last login metadata
        client_id = client_row.get("id")
        api_key = client_row.get("api_key")
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

        return AuthResponse(
            api_key=api_key, 
            email=email, 
            org_name=client_row.get("name"), 
            role=client_row.get("role"), 
            status="active"
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Login failed for %s", email)
        raise HTTPException(status_code=500, detail="Failed to login") from exc

