from __future__ import annotations

import importlib
import threading
import traceback
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr

from config.settings import ADMIN_JWT_EXPIRES_MINUTES, ENABLE_TENDER_INGESTION, UK_TIMEZONE
from services.activity_service import (
    export_events_csv,
    fetch_events,
    get_admin_analytics,
    list_sessions,
    record_event,
    revoke_session,
    revoke_sessions_for_client,
)
from services.digest_service import (
    collect_digest_payloads,
    generate_digest_preview_pdf,
    record_digest,
    record_ingestion,
    send_daily_digest_since,
)
from utils.auth import (
    create_admin_token,
    get_client_id_from_key,
    hash_password,
    is_admin_client,
    is_super_admin_client,
    is_protected_admin,
    require_admin,
    verify_admin_credentials,
)
from utils.logging_config import get_logger
from config.settings import get_supabase_client

router = APIRouter(prefix="/admin")
logger = get_logger(__name__, "app")

_admin_job_status = {
    "ingestion": {
        "running": False,
        "started_at": None,
        "completed_at": None,
        "result": None,
        "error": None,
    },
    "digest": {
        "running": False,
        "started_at": None,
        "completed_at": None,
        "result": None,
        "error": None,
    }
}

def _run_ingestion_background(actor_email: str):
    global _admin_job_status
    status = _admin_job_status["ingestion"]
    try:
        ingestion_module = importlib.import_module("tender_ingestion")
        stored = matched = 0
        new_ids = []
        if ENABLE_TENDER_INGESTION:
            stored, matched, new_ids = ingestion_module.ingest_all_tenders(force=True)
            record_ingestion(datetime.now(timezone.utc))
            
            record_event(
                "system",
                "manual_ingestion",
                actor_email=actor_email,
                metadata={"stored": stored, "matched": matched, "new_ids": len(new_ids)},
            )
            
            status["result"] = {
                "stored": stored,
                "matched": matched,
                "new_tenders": len(new_ids),
            }
        else:
            status["error"] = "ENABLE_TENDER_INGESTION is disabled"
    except Exception as exc:
        logger.error(f"Background ingestion failed: {exc}")
        traceback.print_exc()
        status["error"] = str(exc)
    finally:
        status["running"] = False
        status["completed_at"] = datetime.now(timezone.utc).isoformat()

def _run_digest_background(actor_email: str):
    global _admin_job_status
    status = _admin_job_status["digest"]
    try:
        since_utc = datetime.now(timezone.utc) - timedelta(hours=24)
        digest_summary = send_daily_digest_since(since_utc)
        record_digest(datetime.now(UK_TIMEZONE).date())

        record_event(
            "system",
            "manual_digest",
            actor_email=actor_email,
            metadata={
                "emails_attempted": digest_summary["attempted"],
                "emails_sent": digest_summary["sent"]
            },
        )

        status["result"] = {
            "emails_attempted": digest_summary["attempted"],
            "emails_sent": digest_summary["sent"],
            "no_matches": digest_summary.get("no_matches", 0),
        }
    except Exception as exc:
        logger.error(f"Background digest failed: {exc}")
        traceback.print_exc()
        status["error"] = str(exc)
    finally:
        status["running"] = False
        status["completed_at"] = datetime.now(timezone.utc).isoformat()



class AdminLoginRequest(BaseModel):
    email: EmailStr
    password: str


class AdminAuthResponse(BaseModel):
    token: str
    expires_in_minutes: int
    admin_email: EmailStr


class UserStatusRequest(BaseModel):
    status: str


class UserRoleRequest(BaseModel):
    role: str


class UserPasswordRequest(BaseModel):
    password: str


def _ensure_not_protected_admin(client_id: str):
    """Ensure the user is not a protected admin that cannot be modified."""
    if is_protected_admin(client_id):
        raise HTTPException(status_code=403, detail="This admin account is protected and cannot be modified")


@router.post("/auth/login", response_model=AdminAuthResponse)
def admin_login(payload: AdminLoginRequest):
    admin_id = verify_admin_credentials(payload.email, payload.password)
    if not admin_id:
        raise HTTPException(status_code=401, detail="Invalid admin credentials")

    token = create_admin_token(admin_id, payload.email.lower())
    record_event(
        "auth",
        "admin_login",
        actor_client_id=admin_id,
        actor_email=payload.email.lower(),
    )
    return AdminAuthResponse(
        token=token,
        expires_in_minutes=ADMIN_JWT_EXPIRES_MINUTES,
        admin_email=payload.email.lower(),
    )


@router.get("/status")
def admin_status(
    authorization: str | None = Header(default=None),
    x_client_key: str | None = Header(default=None, alias="X-Client-Key"),
):
    if authorization and authorization.lower().startswith("bearer "):
        try:
            claims = require_admin(authorization=authorization)
            return {"is_admin": True, "admin_email": claims.get("email")}
        except HTTPException:
            return {"is_admin": False}

    if x_client_key:
        client_id = get_client_id_from_key(x_client_key)
        return {"is_admin": is_admin_client(client_id)}
    return {"is_admin": False}


@router.get("/users")
def list_users(
    q: str | None = Query(default=None, description="Email search"),
    status: str | None = Query(default=None),
    role: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
    _: dict = Depends(require_admin),
):
    supabase = get_supabase_client()
    offset = (page - 1) * page_size
    query = (
        supabase.table("clients")
        .select("id, name, contact_email, role, status, api_key_revoked, created_at, last_login_at, last_active_at", count="exact")
        .range(offset, offset + page_size - 1)
        .order("created_at", desc=True)
    )

    if q:
        query = query.ilike("contact_email", f"%{q}%")
    if status:
        query = query.eq("status", status)
    if role:
        query = query.eq("role", role)

    resp = query.execute()
    return {"items": resp.data or [], "total": resp.count or 0, "page": page, "page_size": page_size}


@router.get("/subscriptions")
def list_subscriptions(
    tier: str | None = Query(default=None),
    status: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
    _: dict = Depends(require_admin),
):
    supabase = get_supabase_client()
    offset = (page - 1) * page_size
    
    query = (
        supabase.table("clients")
        .select("id, name, contact_email, subscription_tier, subscription_status, subscription_period_end, trial_end, created_at", count="exact")
        .order("created_at", desc=True)
        .range(offset, offset + page_size - 1)
    )
    
    if tier:
        query = query.eq("subscription_tier", tier)
    if status:
        query = query.eq("subscription_status", status)
    else:
        # By default only show people who have or had a sub/trial
        query = query.not_.is_("subscription_status", "null")
        
    resp = query.execute()
    return {"items": resp.data or [], "total": resp.count or 0, "page": page, "page_size": page_size}


@router.get("/users/{client_id}")
def user_detail(client_id: str, _: dict = Depends(require_admin)):
    supabase = get_supabase_client()
    client_resp = (
        supabase.table("clients")
        .select("id, name, contact_email, role, status, api_key_revoked, created_at, last_login_at, last_active_at")
        .eq("id", client_id)
        .limit(1)
        .execute()
    )
    rows = client_resp.data or []
    if not rows:
        raise HTTPException(status_code=404, detail="Client not found")
    client = rows[0]

    rfps = supabase.table("client_rfps").select("id", count="exact").eq("client_id", client_id).execute()
    jobs = supabase.table("client_jobs").select("id", count="exact").eq("client_id", client_id).execute()

    return {
        "client": client,
        "counts": {
            "rfps": rfps.count or 0,
            "jobs": jobs.count or 0,
        },
    }


@router.post("/users/{client_id}/password")
def update_user_password(client_id: str, payload: UserPasswordRequest, admin: dict = Depends(require_admin)):
    _ensure_not_protected_admin(client_id)
    new_hash = hash_password(payload.password)
    supabase = get_supabase_client()
    resp = supabase.table("clients").update({"password_hash": new_hash}).eq("id", client_id).execute()
    rows = resp.data or []
    if not rows:
        raise HTTPException(status_code=404, detail="Client not found")

    record_event(
        "system",
        "user_password_changed",
        actor_client_id=admin.get("sub"),
        actor_email=admin.get("email"),
        subject_id=client_id,
        subject_type="client",
    )
    revoke_sessions_for_client(client_id)
    return {"client": rows[0]}


@router.delete("/users/{client_id}")
def delete_user(client_id: str, admin: dict = Depends(require_admin)):
    _ensure_not_protected_admin(client_id)
    from config.settings import get_supabase_admin_client
    supabase = get_supabase_client()
    
    # 1. Fetch user data before deletion to get email
    client_res = supabase.table("clients").select("contact_email").eq("id", client_id).execute()
    if not client_res.data:
        raise HTTPException(status_code=404, detail="Client not found")
    
    email = client_res.data[0].get("contact_email")
    
    # 2. Hard delete: remove user and all related data (cascade via foreign keys)
    try:
        # First log the deletion event before deleting (so we still have the client_id)
        record_event(
            "system",
            "user_deleted",
            actor_client_id=admin.get("sub"),
            actor_email=admin.get("email"),
            subject_id=client_id,
            subject_type="client",
            metadata={"email": email}
        )
        
        # 3. Try to delete from Supabase Auth if service role key is available
        try:
            admin_supabase = get_supabase_admin_client()
            # We need to find the user in auth.users by email
            # list_users doesn't support filtering by email directly in current sdk version 
            # easily, so we'll fetch them all or use a workaround if possible.
            # For most bidwell instances, user count is small enough to list.
            auth_users = admin_supabase.auth.admin.list_users()
            user_to_delete = next((u for u in auth_users if u.email.lower() == email.lower()), None)
            
            if user_to_delete:
                admin_supabase.auth.admin.delete_user(user_to_delete.id)
                logger.info(f"Deleted auth user {user_to_delete.id} for email {email}")
            else:
                logger.warning(f"No auth user found for email {email}")
        except Exception as auth_err:
            logger.warning(f"Failed to delete auth user for {email}: {auth_err}")
            # We continue even if auth deletion fails, as deleting from clients is primary
        
        # 4. Delete client (cascade will remove related rfps, questions, answers, jobs, sessions, etc.)
        resp = supabase.table("clients").delete().eq("id", client_id).execute()
        
        logger.info(f"Successfully deleted user {client_id} and all related data")
        return {"success": True, "message": "User and all related data deleted"}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to delete user {client_id}: {exc}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to delete user: {str(exc)}")


@router.delete("/rfps/{rfp_id}")
def delete_rfp_admin(rfp_id: str, admin: dict = Depends(require_admin)):
    """
    Admin-only: delete an entire RFP and all dependent data (questions, answers, jobs, mappings).
    Client-level deletes still exist; this provides cross-tenant admin control.
    """
    supabase = get_supabase_client()
    try:
        rfp_resp = (
            supabase.table("client_rfps")
            .select("id, client_id, name")
            .eq("id", rfp_id)
            .limit(1)
            .execute()
        )
        rows = rfp_resp.data or []
        if not rows:
            raise HTTPException(status_code=404, detail="RFP not found")

        rfp_row = rows[0]
        supabase.table("client_rfps").delete().eq("id", rfp_id).execute()

        record_event(
            "system",
            "rfp_deleted",
            actor_client_id=admin.get("sub"),
            actor_email=admin.get("email"),
            subject_id=rfp_id,
            subject_type="rfp",
            metadata={"client_id": rfp_row.get("client_id"), "name": rfp_row.get("name")},
        )
        return {"success": True, "message": "RFP and related data deleted"}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to delete RFP {rfp_id}: {exc}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to delete RFP")


@router.post("/users/{client_id}/status")
def update_user_status(client_id: str, payload: UserStatusRequest, admin: dict = Depends(require_admin)):
    new_status = payload.status.lower()
    if new_status not in {"active", "suspended"}:
        raise HTTPException(status_code=400, detail="Status must be 'active' or 'suspended'")

    _ensure_not_protected_admin(client_id)
    supabase = get_supabase_client()
    
    # Check current status - prevent reactivating deleted users
    current_resp = supabase.table("clients").select("status").eq("id", client_id).limit(1).execute()
    if not current_resp.data:
        raise HTTPException(status_code=404, detail="Client not found")
    
    current_status = (current_resp.data[0].get("status") or "").lower()
    if current_status == "deleted":
        raise HTTPException(status_code=400, detail="Cannot modify deleted users")
    
    updates = {
        "status": new_status,
        "api_key_revoked": new_status == "suspended",
    }
    resp = supabase.table("clients").update(updates).eq("id", client_id).execute()
    rows = resp.data or []
    if not rows:
        raise HTTPException(status_code=404, detail="Client not found")

    record_event(
        "system",
        "user_status_changed",
        actor_client_id=admin.get("sub"),
        actor_email=admin.get("email"),
        subject_id=client_id,
        subject_type="client",
        metadata={"status": new_status},
    )
    revoke_sessions_for_client(client_id)
    return {"client": rows[0]}


@router.post("/users/{client_id}/role")
def update_user_role(client_id: str, payload: UserRoleRequest, admin: dict = Depends(require_admin)):
    new_role = payload.role.lower()
    if not new_role:
        raise HTTPException(status_code=400, detail="Role is required")

    _ensure_not_protected_admin(client_id)
    supabase = get_supabase_client()
    resp = supabase.table("clients").update({"role": new_role}).eq("id", client_id).execute()
    rows = resp.data or []
    if not rows:
        raise HTTPException(status_code=404, detail="Client not found")

    record_event(
        "system",
        "user_role_changed",
        actor_client_id=admin.get("sub"),
        actor_email=admin.get("email"),
        subject_id=client_id,
        subject_type="client",
        metadata={"role": new_role},
    )
    return {"client": rows[0]}


@router.get("/activity")
def get_activity(
    event_type: str | None = Query(default=None),
    client_id: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=100, ge=1, le=500),
    since: str | None = Query(default=None),
    until: str | None = Query(default=None),
    _: dict = Depends(require_admin),
):
    since_dt = datetime.fromisoformat(since) if since else None
    until_dt = datetime.fromisoformat(until) if until else None
    offset = (page - 1) * page_size
    events = fetch_events(limit=page_size, offset=offset, event_type=event_type, client_id=client_id, since=since_dt, until=until_dt)
    return {"items": events["items"], "total": events["total"], "page": page, "page_size": page_size}


@router.get("/activity/export", response_class=StreamingResponse)
def export_activity(
    event_type: str | None = Query(default=None),
    _: dict = Depends(require_admin),
):
    events = fetch_events(limit=1000, offset=0, event_type=event_type)
    csv_bytes = export_events_csv(events["items"])
    return StreamingResponse(
        iter([csv_bytes]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="activity.csv"'},
    )


@router.get("/sessions")
def get_sessions(include_revoked: bool = Query(default=False), page: int = Query(default=1, ge=1), page_size: int = Query(default=50, ge=1, le=200), _: dict = Depends(require_admin)):
    offset = (page - 1) * page_size
    sessions = list_sessions(include_revoked=include_revoked, limit=page_size + offset)
    sliced = sessions[offset: offset + page_size]
    return {"items": sliced, "total": len(sessions), "page": page, "page_size": page_size}


@router.post("/sessions/{session_id}/revoke")
def revoke_user_session(session_id: str, admin: dict = Depends(require_admin)):
    try:
        session = revoke_session(session_id, revoked_by=admin.get("email"))
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found")

    record_event(
        "system",
        "session_revoked",
        actor_client_id=admin.get("sub"),
        actor_email=admin.get("email"),
        subject_id=session_id,
        subject_type="session",
    )
    return {"session": session}


@router.get("/analytics")
def analytics(hours: int = Query(default=168, ge=1, le=720), _: dict = Depends(require_admin)):
    return get_admin_analytics(hours=hours)


@router.post("/run-daily-cycle")
def run_daily_cycle(admin: dict = Depends(require_admin)):
    """Trigger both ingestion and digest sequentially (Sync)."""
    ingestion_module = importlib.import_module("tender_ingestion")
    stored = matched = 0
    new_ids = []
    if ENABLE_TENDER_INGESTION:
        stored, matched, new_ids = ingestion_module.ingest_all_tenders(force=True)
        record_ingestion(datetime.now(timezone.utc))
    
    since_utc = datetime.now(timezone.utc) - timedelta(hours=24)
    digest_summary = send_daily_digest_since(since_utc)
    record_digest(datetime.now(UK_TIMEZONE).date())

    record_event(
        "system",
        "manual_daily_cycle",
        actor_client_id=admin.get("sub"),
        actor_email=admin.get("email"),
        metadata={"stored": stored, "matched": matched, "new_ids": len(new_ids)},
    )
    return {
        "stored": stored,
        "matched": matched,
        "new_tenders": len(new_ids),
        "emails_attempted": digest_summary["attempted"],
        "emails_sent": digest_summary["sent"],
    }

@router.post("/ingest-tenders")
def trigger_ingestion(admin: dict = Depends(require_admin)):
    global _admin_job_status
    status = _admin_job_status["ingestion"]
    if status["running"]:
        return {"status": "already_running", "started_at": status["started_at"]}
    
    status.update({
        "running": True,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "result": None,
        "error": None,
    })
    
    thread = threading.Thread(target=_run_ingestion_background, args=(admin.get("email"),), daemon=True)
    thread.start()
    return {"status": "started", "started_at": status["started_at"]}

@router.post("/send-digests")
def trigger_digests(admin: dict = Depends(require_admin)):
    global _admin_job_status
    status = _admin_job_status["digest"]
    if status["running"]:
        return {"status": "already_running", "started_at": status["started_at"]}
    
    status.update({
        "running": True,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "result": None,
        "error": None,
    })
    
    thread = threading.Thread(target=_run_digest_background, args=(admin.get("email"),), daemon=True)
    thread.start()
    return {"status": "started", "started_at": status["started_at"]}

@router.get("/job-status")
def get_admin_job_status(_: dict = Depends(require_admin)):
    return _admin_job_status


def _parse_since_param(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        else:
            parsed = parsed.astimezone(timezone.utc)
        return parsed
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid 'since' timestamp. Use ISO 8601.") from exc


@router.get("/daily-digest-report", response_class=StreamingResponse)
def download_daily_digest_report(
    since: str | None = Query(default=None, description="ISO8601 UTC timestamp to filter matches from"),
    admin: dict = Depends(require_admin),
):
    """
    Generate a PDF report showing the pending daily digest emails per client.
    """
    since_dt = _parse_since_param(since)
    digest_bundle = collect_digest_payloads(since_dt)
    pdf_bytes = generate_digest_preview_pdf(
        digest_bundle["payloads"],
        digest_bundle["prepared_at"],
        digest_bundle["since_utc"],
    )

    record_event(
        "system",
        "download_digest_report",
        actor_client_id=admin.get("sub"),
        actor_email=admin.get("email"),
    )

    filename = f"digest-preview-{digest_bundle['prepared_at'].strftime('%Y%m%d-%H%M%S')}.pdf"
    return StreamingResponse(
        BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

