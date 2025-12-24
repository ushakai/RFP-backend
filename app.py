"""
RFP Backend API - Modular Main Application
"""
import os
import time
import threading
import traceback
import importlib
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

# Import logging
from utils.logging_config import get_logger

# Setup logger
logger = get_logger(__name__, "app")

# Import configuration
from config.settings import (
    ALLOWED_ORIGINS,
    DISABLE_TENDER_INGESTION_LOOP,
    ENABLE_TENDER_INGESTION,
    UK_TIMEZONE,
)

# Import API routers
from api import health, rfps, qa, jobs, drive, tenders, admin, auth

# Import notification function for background tasks
from api.tenders import notify_client as _notify_client
import api.tenders as tenders_api

from services.digest_service import (
    send_daily_digest_since,
    record_ingestion,
    get_last_ingestion,
    record_digest,
    get_last_digest_date,
)

# Create FastAPI app
app = FastAPI(
    title="RFP Backend API", 
    version="1.0.0",
    description="Backend API for RFP processing and management",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
# Combine explicit origins with regex pattern for dynamic subdomains
cors_origins = ALLOWED_ORIGINS.copy() if isinstance(ALLOWED_ORIGINS, list) else list(ALLOWED_ORIGINS)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|.*\.onrender\.com|.*\.vercel\.app)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "Content-Disposition",
    ],
)

# Exception handler to ensure CORS headers are set on errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Ensure CORS headers are set even when exceptions occur"""
    from fastapi import HTTPException
    import re
    
    # Get origin from request
    origin = request.headers.get("origin", "")
    
    # Validate origin against allowed origins
    cors_headers = {}
    if origin:
        # Check if origin is in allowed list or matches regex
        origin_allowed = False
        if origin in cors_origins:
            origin_allowed = True
        else:
            # Check regex pattern
            regex_pattern = r"https?://(localhost|127\.0\.0\.1|.*\.onrender\.com|.*\.vercel\.app)(:\d+)?"
            if re.match(regex_pattern, origin):
                origin_allowed = True
        
        if origin_allowed:
            cors_headers = {
                "Access-Control-Allow-Origin": origin,
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*",
            }
    
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=cors_headers
        )
    
    # Log unexpected errors
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    
    # Provide more specific error messages for common issues
    error_msg = "Internal server error"
    exc_str = str(exc).lower()
    
    if "connection" in exc_str or "timeout" in exc_str:
        error_msg = "Database connection error. Please try again in a moment."
    elif "resource temporarily unavailable" in exc_str:
        error_msg = "Service temporarily unavailable. Please try again in a few moments."
    elif "too many connections" in exc_str or "pool" in exc_str:
        error_msg = "Server is experiencing high load. Please try again shortly."
    elif "permission denied" in exc_str or "forbidden" in exc_str:
        error_msg = "Access denied. Please check your permissions."
    
    return JSONResponse(
        status_code=500,
        content={"detail": error_msg},
        headers=cors_headers
)

# Register API routers
app.include_router(auth.router, tags=["Auth"])
app.include_router(health.router, tags=["Health"])
app.include_router(rfps.router, tags=["RFPs"])
app.include_router(qa.router, tags=["Q&A"])
app.include_router(jobs.router, tags=["Jobs"])
app.include_router(drive.router, tags=["Google Drive"])
app.include_router(tenders.router, tags=["Tenders"])
app.include_router(admin.router, tags=["Admin"])


# ============================================================================
# BACKGROUND TASKS - Scheduled Tender Ingestion & Digest
# ============================================================================

INGESTION_HOUR_UK = 7  # 07:00 UK time
DIGEST_HOUR_UK = 8     # 08:00 UK time


def _seconds_until(hour: int, minute: int = 0) -> float:
    now = datetime.now(UK_TIMEZONE)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    delta = (target - now).total_seconds()
    return max(delta, 1.0)


def _scheduled_ingestion_worker():
    tender_logger = get_logger("tender_ingestion", "tender")
    try:
        ingestion_module = importlib.import_module("tender_ingestion")
        tender_logger.info("Scheduled ingestion worker initialised.")
    except Exception as exc:
        tender_logger.error(f"Unable to import tender_ingestion: {exc}")
        traceback.print_exc()
        return

    while True:
        sleep_seconds = _seconds_until(INGESTION_HOUR_UK)
        tender_logger.debug(f"Sleeping {sleep_seconds:.0f}s until next {INGESTION_HOUR_UK}:00 UK ingestion.")
        time.sleep(sleep_seconds)

        if not ENABLE_TENDER_INGESTION:
            tender_logger.info("ENABLE_TENDER_INGESTION flag disabled; skipping scheduled ingestion cycle.")
            continue

        try:
            tender_logger.info("Starting scheduled tender ingestion cycle...")
            stored, matched, new_ids = ingestion_module.ingest_all_tenders()
            record_ingestion(datetime.now(timezone.utc))
            tender_logger.info(
                f"Ingestion complete: stored={stored}, matched={matched}, new_tenders={len(new_ids)}"
            )

            for cid in list(tenders_api._client_streams.keys()):
                _notify_client(cid, "matches-updated", {"reason": "ingestion"})
        except Exception as exc:
            tender_logger.error(f"Ingestion cycle failed: {exc}")
            traceback.print_exc()


def _scheduled_digest_worker():
    digest_logger = get_logger("tender_digest", "tender")
    digest_logger.info("Scheduled digest worker initialised.")

    while True:
        sleep_seconds = _seconds_until(DIGEST_HOUR_UK)
        digest_logger.debug(f"Sleeping {sleep_seconds:.0f}s until next {DIGEST_HOUR_UK}:00 UK digest.")
        time.sleep(sleep_seconds)

        uk_today = datetime.now(UK_TIMEZONE).date()
        if get_last_digest_date() == uk_today:
            digest_logger.debug("Digest already sent today; skipping.")
            continue

        since = get_last_ingestion()
        if since is None:
            start_of_day_uk = datetime.combine(uk_today, datetime.min.time(), tzinfo=UK_TIMEZONE)
            since = start_of_day_uk.astimezone(timezone.utc)

        digest_logger.info("Preparing daily tender digest emails...")
        summary = send_daily_digest_since(since)
        record_digest(uk_today)
        digest_logger.info(
            f"Daily digest dispatched. Attempted={summary['attempted']}, Sent={summary['sent']}"
        )


@app.on_event("startup")
def start_background_tasks():
    """
    Start background tender ingestion loop unless disabled.
    Set DISABLE_TENDER_INGESTION_LOOP=1 to disable.
    """
    logger.info("=" * 80)
    logger.info("RFP Backend API Starting")
    logger.info(f"Version: 1.0.0")
    logger.info(f"Docs: http://localhost:8000/docs")
    logger.info("=" * 80)
    
    if DISABLE_TENDER_INGESTION_LOOP:
        logger.info("Scheduled tender tasks DISABLED by environment variable")
        return
    
    if not ENABLE_TENDER_INGESTION:
        logger.info("ENABLE_TENDER_INGESTION=0; ingestion worker will remain idle until re-enabled")
    
    logger.info("Starting scheduled tender ingestion worker...")
    threading.Thread(target=_scheduled_ingestion_worker, daemon=True).start()
    logger.info("Starting scheduled tender digest worker...")
    threading.Thread(target=_scheduled_digest_worker, daemon=True).start()
    logger.info("Scheduled background workers started successfully")


# Main entry point for Render deployment
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)