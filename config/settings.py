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
    "https://rfp-two.vercel.app"
]

# Add production origins if running on Render
if os.getenv("RENDER"):
    ALLOWED_ORIGINS.extend([
        "https://rfp-two.vercel.app",
        "https://*.vercel.app",
        "https://*.onrender.com"
    ])

# Background task configuration
TENDER_INGESTION_INTERVAL_MINUTES = int(os.getenv("TENDER_INGESTION_INTERVAL_MINUTES", "360"))
DISABLE_TENDER_INGESTION_LOOP = os.getenv("DISABLE_TENDER_INGESTION_LOOP", "0") == "1"

# Feature flags
ENABLE_TENDER_INGESTION = os.getenv("ENABLE_TENDER_INGESTION", "1").lower() not in {"0", "false", "off", "no"}

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

# Global Supabase client (will be reinitialized as needed)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_supabase_client() -> Client:
    """Get or reinitialize Supabase client"""
    global supabase
    if supabase is None:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return supabase

def reinitialize_supabase():
    """Force reinitialize Supabase client"""
    global supabase
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return supabase

