"""
Configuration and Environment Setup
"""
import os
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
import google.generativeai as genai
from supabase import create_client, Client

# Load environment variables
load_dotenv()

def _clean_env(value: str | None) -> str:
    """Clean environment variable values"""
    if not value:
        return ""
    return value.strip().strip('"').strip("'")

# Environment variables
GOOGLE_API_KEY = _clean_env(os.getenv("GOOGLE_API_KEY"))
SUPABASE_URL = _clean_env(os.getenv("SUPABASE_URL"))
SUPABASE_KEY = _clean_env(os.getenv("SUPABASE_KEY"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Frontend configuration
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
ALLOWED_ORIGINS = [
    FRONTEND_ORIGIN, 
    "http://127.0.0.1:5173", 
    "http://localhost:5173",
    "https://localhost:5173",
    "https://127.0.0.1:5173",
    "https://rfp-two.vercel.app",  # Explicitly allow production frontend
]

# Add production origins if running on Render (regex will handle subdomains)
if os.getenv("RENDER") or os.getenv("VERCEL"):
    # Don't add wildcard strings - regex pattern in app.py handles subdomains
    if "https://rfp-two.vercel.app" not in ALLOWED_ORIGINS:
        ALLOWED_ORIGINS.append("https://rfp-two.vercel.app")

# Background task configuration
TENDER_INGESTION_INTERVAL_MINUTES = int(os.getenv("TENDER_INGESTION_INTERVAL_MINUTES", "360"))
DISABLE_TENDER_INGESTION_LOOP = os.getenv("DISABLE_TENDER_INGESTION_LOOP", "0") == "1"

# Feature flags
ENABLE_TENDER_INGESTION = os.getenv("ENABLE_TENDER_INGESTION", "1").lower() not in {"0", "false", "off", "no"}
ENABLE_PAYMENT = os.getenv("ENABLE_PAYMENT", "0").lower() not in {"0", "false", "off", "no"}
FILTER_UK_ONLY = os.getenv("FILTER_UK_ONLY", "1").lower() not in {"0", "false", "off", "no"}

# Scheduling
UK_TIMEZONE = ZoneInfo(os.getenv("UK_TIMEZONE", "Europe/London"))

# Admin configuration
ADMIN_CLIENT_IDS = {
    cid.strip()
    for cid in _clean_env(os.getenv("ADMIN_CLIENT_IDS", "")).split(",")
    if cid.strip()
}
_admin_emails_env = {
    email.strip().lower()
    for email in _clean_env(os.getenv("ADMIN_CLIENT_EMAILS", "")).split(",")
    if email.strip()
}
ADMIN_CLIENT_EMAILS = set(_admin_emails_env)
ADMIN_CLIENT_EMAILS.add("admin@rfp.com")

# Admin auth and security
ADMIN_ROLE_NAME = os.getenv("ADMIN_ROLE_NAME", "admin")
ADMIN_JWT_SECRET = _clean_env(os.getenv("ADMIN_JWT_SECRET") or SUPABASE_KEY)
ADMIN_JWT_EXPIRES_MINUTES = int(os.getenv("ADMIN_JWT_EXPIRES_MINUTES", "720"))
ADMIN_LOGIN_MAX_ATTEMPTS = int(os.getenv("ADMIN_LOGIN_MAX_ATTEMPTS", "5"))
ADMIN_LOGIN_WINDOW_SECONDS = int(os.getenv("ADMIN_LOGIN_WINDOW_SECONDS", "300"))
ADMIN_ACTIVITY_RETENTION_DAYS = int(os.getenv("ADMIN_ACTIVITY_RETENTION_DAYS", "90"))
ADMIN_ANALYTICS_CACHE_SECONDS = int(os.getenv("ADMIN_ANALYTICS_CACHE_SECONDS", "300"))
ADMIN_SESSION_MAX_AGE_DAYS = int(os.getenv("ADMIN_SESSION_MAX_AGE_DAYS", "60"))
SUPER_ADMIN_EMAIL = _clean_env(os.getenv("SUPER_ADMIN_EMAIL", "admin@bidwell.com")).lower()

# Protected admin emails that cannot be modified by other admins
PROTECTED_ADMIN_EMAILS = {
    "admin@bidwell.com",
    "admin@rfp.com",
    "dean@purple.ai"
}


# Email / SMTP settings
SMTP_HOST = _clean_env(os.getenv("SMTP_HOST"))
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = _clean_env(os.getenv("SMTP_USERNAME"))
SMTP_PASSWORD = _clean_env(os.getenv("SMTP_PASSWORD"))
SMTP_FROM_EMAIL = _clean_env(os.getenv("SMTP_FROM"))
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "1").lower() not in {"0", "false", "no"}

# Gmail OAuth configuration (optional)
GMAIL_CLIENT_ID = _clean_env(os.getenv("GMAIL_CLIENT_ID"))
GMAIL_CLIENT_SECRET = _clean_env(os.getenv("GMAIL_CLIENT_SECRET"))
GMAIL_REFRESH_TOKEN = _clean_env(os.getenv("GMAIL_REFRESH_TOKEN"))
GMAIL_SENDER_EMAIL = _clean_env(os.getenv("GMAIL_SENDER_EMAIL"))

# Stripe configuration
STRIPE_SECRET_KEY = _clean_env(os.getenv("STRIPE_SECRET_KEY"))
STRIPE_WEBHOOK_SECRET = _clean_env(os.getenv("STRIPE_WEBHOOK_SECRET"))
STRIPE_SUCCESS_URL = os.getenv("STRIPE_SUCCESS_URL", f"{FRONTEND_ORIGIN}/subscription/success")
STRIPE_CANCEL_URL = os.getenv("STRIPE_CANCEL_URL", f"{FRONTEND_ORIGIN}/pricing")

# Stripe Price IDs
STRIPE_PRICE_TENDERS_MONTHLY = _clean_env(os.getenv("STRIPE_PRICE_TENDERS_MONTHLY"))
STRIPE_PRICE_TENDERS_YEARLY = _clean_env(os.getenv("STRIPE_PRICE_TENDERS_YEARLY"))
STRIPE_PRICE_PROCESSING_MONTHLY = _clean_env(os.getenv("STRIPE_PRICE_PROCESSING_MONTHLY"))
STRIPE_PRICE_PROCESSING_YEARLY = _clean_env(os.getenv("STRIPE_PRICE_PROCESSING_YEARLY"))
STRIPE_PRICE_BOTH_MONTHLY = _clean_env(os.getenv("STRIPE_PRICE_BOTH_MONTHLY"))
STRIPE_PRICE_BOTH_YEARLY = _clean_env(os.getenv("STRIPE_PRICE_BOTH_YEARLY"))

# Trial Period Configuration
STRIPE_TRIAL_PERIOD_MINUTES = int(os.getenv("STRIPE_TRIAL_PERIOD_MINUTES", "10080")) # Default 7 days

# Validate Gemini configuration
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY for Gemini")

# Initialize Gemini client
genai.configure(api_key=GOOGLE_API_KEY)

# Validate Supabase configuration
if not SUPABASE_URL or not SUPABASE_URL.startswith("https://") or ".supabase.co" not in SUPABASE_URL:
    raise ValueError(f"Invalid SUPABASE_URL format: '{SUPABASE_URL}'. Expected like https://xxxxx.supabase.co")
if not SUPABASE_KEY:
    raise ValueError("SUPABASE_KEY is missing")

# Global Supabase client (simple and reliable)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
supabase_admin: Client | None = None

# Service role key (for administrative operations like deleting users from auth.users)
SUPABASE_SERVICE_ROLE_KEY = _clean_env(os.getenv("SUPABASE_SERVICE_ROLE_KEY"))

# Direct PostgreSQL connection configuration
# Use dedicated IPv4 for Supabase direct connection
# Format: postgresql://postgres:[PASSWORD]@[DEDICATED_IPV4]:5432/postgres?sslmode=require
DATABASE_URL = _clean_env(os.getenv("DATABASE_URL"))

# Individual database connection parameters (alternative to DATABASE_URL)
DB_HOST = _clean_env(os.getenv("DB_HOST"))  # Your dedicated IPv4 address
DB_PORT = _clean_env(os.getenv("DB_PORT", "5432"))
DB_USER = _clean_env(os.getenv("DB_USER", "postgres"))
DB_PASSWORD = _clean_env(os.getenv("DB_PASSWORD"))
DB_NAME = _clean_env(os.getenv("DB_NAME", "postgres"))

# Flag to use dedicated IP connection (enables SSL automatically)
USE_DEDICATED_IP = os.getenv("USE_DEDICATED_IP", "1").lower() not in {"0", "false", "off", "no"}

# =============================================================================
# Database Connection Functions
# =============================================================================

# Import the direct database module
from config.db import db, get_db, init_db


def get_supabase_client():
    """
    Get the database client for database operations.
    
    This now returns a direct PostgreSQL connection (via the db module)
    which provides a Supabase-like interface but with persistent connections.
    
    Usage remains the same:
        supabase = get_supabase_client()
        result = supabase.table("clients").select("*").eq("id", client_id).execute()
    """
    global supabase
    
    # Initialize the database if DATABASE_URL is configured
    if DATABASE_URL:
        try:
            db._ensure_initialized()
        except Exception as e:
            print(f"Warning: Failed to initialize direct database: {e}")
            # Fall back to Supabase client if direct connection fails
            if supabase is None:
                supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            return supabase
        return db
    
    # Fall back to Supabase client if DATABASE_URL is not configured
    if supabase is None:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return supabase


def get_supabase_admin_client() -> Client:
    """
    Get a Supabase client with service role privileges.
    Used for administrative tasks like Auth operations (creating users, etc.)
    
    NOTE: This still uses the Supabase HTTP client as Auth operations
    require the Supabase Auth API.
    """
    global supabase_admin
    if not SUPABASE_SERVICE_ROLE_KEY:
        raise ValueError("SUPABASE_SERVICE_ROLE_KEY is missing from environment variables")
        
    try:
        if supabase_admin is None:
            supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        return supabase_admin
    except Exception as e:
        print(f"Error getting Supabase admin client: {e}")
        try:
            supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
            return supabase_admin
        except Exception as e2:
            print(f"Failed to create Supabase admin client: {e2}")
            raise Exception("Cannot connect to database with administrative privileges.") from e2


def reinitialize_supabase():
    """
    Reinitialize database connections.
    Call this when experiencing connection issues.
    """
    global supabase, supabase_admin
    
    # Reinitialize direct database connection
    if DATABASE_URL:
        try:
            db.close()
            db.initialize(database_url=DATABASE_URL)
            print("✓ Reinitialized direct database connection successfully")
        except Exception as e:
            print(f"Warning: Failed to reinitialize direct database: {e}")
    
    # Also reinitialize Supabase clients (needed for Auth operations)
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        if SUPABASE_SERVICE_ROLE_KEY:
            supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        print("✓ Reinitialized Supabase clients successfully")
        return supabase
    except Exception as e:
        print(f"✗ Failed to reinitialize Supabase client: {e}")
        raise Exception("Cannot reconnect to database. Please try again later.") from e


def is_direct_db_configured() -> bool:
    """Check if direct database connection is properly configured."""
    return bool(DATABASE_URL)


def test_direct_db_connection() -> bool:
    """Test the direct PostgreSQL connection."""
    if not is_direct_db_configured():
        print("Direct database connection not configured")
        return False
    return db.test_connection()


def close_connection_pool():
    """Close all database connections. Call this on application shutdown."""
    db.close()

