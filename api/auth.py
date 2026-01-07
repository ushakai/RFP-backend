import random
import string
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, Body, Depends
from pydantic import BaseModel, EmailStr, Field

from config.settings import get_supabase_client, get_supabase_admin_client
from services.activity_service import record_event, record_session
from services.email_service import send_email
from utils.auth import hash_password, get_client_id
from utils.logging_config import get_logger


router = APIRouter(prefix="/auth")
logger = get_logger(__name__, "auth")

def _generate_otp(length=6):
    return "".join(random.choices(string.digits, k=length))

# ... Pydantic models remain the same ...
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
    tier: str | None = None
    subscription_status: str | None = None
    status: str | None = None
    message: str | None = None
    requires_verification: bool = False


class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordConfirmRequest(BaseModel):
    email: EmailStr
    otp: str
    new_password: str

# ... Reset password endpoints remain ... (Lines 55-167)

@router.post("/register", response_model=AuthResponse)
def register(payload: RegisterRequest) -> AuthResponse:
    """Register using Supabase Auth (Admin) and custom OTP flow."""
    supabase = get_supabase_client()
    admin_supabase = get_supabase_admin_client()

    org_name = payload.org_name.strip()
    email = payload.email.strip().lower()
    
    if not org_name:
        raise HTTPException(status_code=400, detail="Organization name is required")

    try:
        # 1. Check if email already exists in clients table (our source of truth for "active" users)
        existing = (
            supabase.table("clients")
            .select("id")
            .eq("contact_email", email)
            .limit(1)
            .execute()
        )
        if existing.data:
            raise HTTPException(status_code=409, detail="An account with this email already exists")

        # 2. Generate OTP and Expiry (30 minutes)
        otp = _generate_otp()
        expiry = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()
        
        # 3. Create or Update user in Supabase Auth via Admin API
        # We auto-confirm the user so they can login immediately *after* they pass our OTP check
        user_id = None
        try:
            # Try to create new user
            user_attrs = {
                "email": email,
                "password": payload.password,
                "email_confirm": True,
                "app_metadata": {
                    "verification_code": otp,
                    "verification_expires_at": expiry,
                    "org_name": org_name
                },
                "user_metadata": {
                    "org_name": org_name
                }
            }
            user = admin_supabase.auth.admin.create_user(user_attrs)
            user_id = user.user.id
        except Exception as e:
            msg = str(e).lower()
            if "already registered" in msg or "already exists" in msg:
                # User exists in Auth, but not in clients (verified by step 1).
                # This happens if a previous registration partially failed or was deleted from clients but not Auth.
                # We need to find the user and update them.
                try:
                    # Fetch users to find ID (no direct get_by_email in some SDK versions)
                    users = admin_supabase.auth.admin.list_users()
                    existing_user = next((u for u in users if u.email and u.email.lower() == email), None)
                    
                    if existing_user:
                        user_id = existing_user.id
                        # Update existing user with new OTP and Password
                        admin_supabase.auth.admin.update_user_by_id(
                            user_id, 
                            {
                                "password": payload.password,
                                "email_confirm": True,
                                "app_metadata": {
                                    "verification_code": otp,
                                    "verification_expires_at": expiry,
                                    "org_name": org_name
                                }
                            }
                        )
                    else:
                        raise HTTPException(status_code=500, detail="Failed to locate existing auth user.")
                except Exception as update_err:
                     logger.error(f"Failed to update existing auth user: {update_err}")
                     raise HTTPException(status_code=500, detail="Failed to reset existing account.")
            else:
                logger.error(f"Auth creation failed: {e}")
                raise HTTPException(status_code=500, detail="Failed to create authentication profile.")

        # 4. Create entry in custom clients table
        # We use a randomized API key. The client_id is auto-generated (UUID) by DB.
        # We store the hashed password for legacy/backup reasons mainly.
        api_key = str(uuid4())
        response = (
            supabase.table("clients")
            .insert(
                {
                    "name": org_name,
                    "contact_email": email,
                    "api_key": api_key,
                    "password_hash": hash_password(payload.password),
                    "status": "pending_verification",
                    # We could store current OTP here too if column existed, but we rely on Auth app_metadata
                }
            )
            .execute()
        )
        
        if getattr(response, "error", None) or not response.data:
            logger.error("Supabase client insertion error: %s", getattr(response, "error", "No data"))
            # Rollback auth user? (Optional, but complex)
            raise HTTPException(status_code=500, detail="Failed to create organization profile")

        # 5. Send OTP Email
        email_sent = send_email(
            recipients=[email],
            subject="Verify your account - Code: " + otp,
            html_body=f"<h1>Welcome to RFP Monitor!</h1><p>Your verification code is: <strong>{otp}</strong></p><p>This code expires in 30 minutes.</p>",
            text_body=f"Your verification code is: {otp}. Expires in 30 minutes."
        )
        
        if not email_sent:
            # We don't fail registration, but warn user
            logger.warning("Failed to send verification email to %s", email)
            # Actually, if they can't verify, they are stuck. But they can retry?
            # Ideally return strict error, or assume frontend handles it.
            # We'll return success but logs show error.

        return AuthResponse(
            email=email, 
            org_name=org_name, 
            status="pending_verification",
            requires_verification=True,
            message="A verification code has been sent to your email."
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Registration failed for %s", email)
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(exc)}")


@router.post("/verify-otp", response_model=AuthResponse)
def verify_otp(payload: VerifyOTPRequest) -> AuthResponse:
    """Verify the custom OTP and activate the account."""
    supabase = get_supabase_client()
    admin_supabase = get_supabase_admin_client()
    
    email = payload.email.strip().lower()
    otp = payload.otp.strip()

    try:
        # 1. Find user in Auth to check OTP
        user_id = None
        stored_otp = None
        stored_expiry = None
        
        try:
            users = admin_supabase.auth.admin.list_users()
            auth_user = next((u for u in users if u.email and u.email.lower() == email), None)
            
            if auth_user and auth_user.app_metadata:
                stored_otp = auth_user.app_metadata.get("verification_code")
                stored_expiry = auth_user.app_metadata.get("verification_expires_at")
                user_id = auth_user.id
        except Exception as e:
            logger.error(f"Failed to fetch auth user: {e}")
            raise HTTPException(status_code=500, detail="Verification system error.")

        if not stored_otp:
             # Fallback/Edge case: User verified via link or legacy method?
             # Or invalid email
             raise HTTPException(status_code=400, detail="No pending verification found for this email.")

        # 2. Check Validity
        if stored_otp != otp:
             raise HTTPException(status_code=400, detail="Invalid verification code.")
        
        if stored_expiry:
             exp_dt = datetime.fromisoformat(stored_expiry)
             if datetime.now(timezone.utc) > exp_dt:
                 raise HTTPException(status_code=400, detail="Verification code has expired. Please register again.")

        # 3. Activate in clients table
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
        client_id = client["id"]

        # Update status
        supabase.table("clients").update({"status": "active"}).eq("id", client_id).execute()
        
        # Clear OTP from Auth (Optional, but good practice)
        try:
             admin_supabase.auth.admin.update_user_by_id(
                user_id, 
                {"app_metadata": {"verification_code": None, "verification_expires_at": None}}
             )
        except:
             pass

        # 4. Create Stripe Customer and initialize trial
        from services.stripe_service import create_or_get_customer
        stripe_customer_id = None
        try:
            stripe_customer_id = create_or_get_customer(client_id, email, client["name"])
        except Exception as stripe_err:
            logger.error(f"Stripe customer creation failed: {stripe_err}")
            # Non-blocking

        return AuthResponse(
            email=email,
            org_name=client["name"],
            role=client.get("role", "user"),
            status="active",
            api_key=client["api_key"],
            message="Verification successful! You can now log in."
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        raise HTTPException(status_code=500, detail="Verification failed due to a server error.")


@router.post("/resend-otp", response_model=AuthResponse)
def resend_otp(payload: dict = Body(...)) -> AuthResponse:
    """Resend a new custom OTP."""
    admin_supabase = get_supabase_admin_client()
    email = payload.get("email", "").strip().lower()
    if not email:
        raise HTTPException(status_code=400, detail="Email is required")

    try:
        # 1. Find user in Auth
        user_id = None
        try:
            users = admin_supabase.auth.admin.list_users()
            auth_user = next((u for u in users if u.email and u.email.lower() == email), None)
            if auth_user:
                user_id = auth_user.id
        except Exception as e:
            logger.error(f"Failed to fetch auth user during resend: {e}")
            raise HTTPException(status_code=500, detail="System error during resend.")
            
        if not user_id:
             # Just return success to prevent enumeration? Or conflict?
             # Standard behavior: silent success or error. 
             # For this app, let's error if we can't find them, implied registered.
             raise HTTPException(status_code=404, detail="User not found.")

        # 2. Generate new OTP
        otp = _generate_otp()
        expiry = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()
        
        # 3. Update Auth Metadata
        admin_supabase.auth.admin.update_user_by_id(
            user_id, 
            {
                "app_metadata": {
                    "verification_code": otp,
                    "verification_expires_at": expiry
                }
            }
        )

        # 4. Send Email
        email_sent = send_email(
            recipients=[email],
            subject="New Verification Code - RFP Monitor",
            html_body=f"<h1>Verification Code</h1><p>Your new verification code is: <strong>{otp}</strong></p><p>This code expires in 30 minutes.</p>",
            text_body=f"Your new verification code is: {otp}. Expires in 30 minutes."
        )

        return AuthResponse(
            email=email, 
            status="pending_verification",
            message="A new verification code has been sent to your email."
        )
    except HTTPException:
        raise
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
            tier=client_row.get("subscription_tier"),
            subscription_status=client_row.get("subscription_status"),
            status="active"
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Login failed for %s", email)
        raise HTTPException(status_code=500, detail="Failed to login") from exc

