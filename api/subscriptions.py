from fastapi import APIRouter, Depends, HTTPException, Body
from typing import Dict, Any
from config.settings import (
    STRIPE_PRICE_TENDERS_MONTHLY,
    STRIPE_PRICE_TENDERS_YEARLY,
    STRIPE_PRICE_PROCESSING_MONTHLY,
    STRIPE_PRICE_PROCESSING_YEARLY,
    STRIPE_PRICE_BOTH_MONTHLY,
    STRIPE_PRICE_BOTH_YEARLY,
    get_supabase_client
)
from utils.auth import get_client_id
from services.stripe_service import (
    create_or_get_customer,
    create_checkout_session,
    create_portal_session
)
from services.supabase_service import execute_with_retry
from utils.logging_config import get_logger

router = APIRouter(prefix="/subscriptions")
logger = get_logger(__name__, "subscriptions")

# Mapping of price IDs to tiers
PRICE_TO_TIER = {}
if STRIPE_PRICE_TENDERS_MONTHLY: PRICE_TO_TIER[STRIPE_PRICE_TENDERS_MONTHLY] = "tenders"
if STRIPE_PRICE_TENDERS_YEARLY: PRICE_TO_TIER[STRIPE_PRICE_TENDERS_YEARLY] = "tenders"
if STRIPE_PRICE_PROCESSING_MONTHLY: PRICE_TO_TIER[STRIPE_PRICE_PROCESSING_MONTHLY] = "processing"
if STRIPE_PRICE_PROCESSING_YEARLY: PRICE_TO_TIER[STRIPE_PRICE_PROCESSING_YEARLY] = "processing"
if STRIPE_PRICE_BOTH_MONTHLY: PRICE_TO_TIER[STRIPE_PRICE_BOTH_MONTHLY] = "both"
if STRIPE_PRICE_BOTH_YEARLY: PRICE_TO_TIER[STRIPE_PRICE_BOTH_YEARLY] = "both"

@router.get("/status")
def get_status(client_id: str = Depends(get_client_id)):
    """Get current subscription status and tier."""
    try:
        def _fetch_status():
            supabase = get_supabase_client()
            return supabase.table("clients").select(
                "subscription_status", 
                "subscription_tier", 
                "subscription_interval", 
                "subscription_period_end",
                "trial_end"
            ).eq("id", client_id).single().execute()
        
        res = execute_with_retry(
            _fetch_status,
            retries=3,
            backoff_seconds=0.3,
            max_total_time=5.0
        )
        
        if not res.data:
            raise HTTPException(status_code=404, detail="Client not found")
            
        return res.data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch subscription status for client {client_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch subscription status. Please try again.")

@router.post("/create-checkout")
def create_checkout(
    payload: Dict[str, str] = Body(...),
    client_id: str = Depends(get_client_id)
):
    """Create a Stripe Checkout Session."""
    price_id = payload.get("price_id")
    if not price_id:
        raise HTTPException(status_code=400, detail="price_id is required")
        
    if price_id not in PRICE_TO_TIER:
        raise HTTPException(status_code=400, detail="Invalid price_id")
        
    try:
        def _fetch_client():
            supabase = get_supabase_client()
            return supabase.table("clients").select(
                "contact_email", 
                "name", 
                "subscription_status",
                "trial_end"
            ).eq("id", client_id).single().execute()
        
        client_res = execute_with_retry(
            _fetch_client,
            retries=3,
            backoff_seconds=0.3,
            max_total_time=5.0
        )
        
        if not client_res.data:
            raise HTTPException(status_code=404, detail="Client not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch client data for checkout: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch client data. Please try again.")
    
    client_data = client_res.data
    status = client_data.get("subscription_status")
    
    # 1. Enforce single subscription
    if status in ["active", "trialing"]:
        raise HTTPException(status_code=400, detail="You already have an active subscription. Please manage it via the billing portal.")

    # 2. Check if trial has been used (if trial_end indicates a past trial)
    has_used_trial = False
    if client_data.get("trial_end"):
        has_used_trial = True
        
    customer_id = create_or_get_customer(
        client_id, 
        client_data["contact_email"], 
        client_data["name"]
    )
    
    try:
        checkout_url = create_checkout_session(
            customer_id, 
            price_id, 
            client_id,
            has_used_trial=has_used_trial
        )
        return {"checkout_url": checkout_url}
    except Exception as e:
        logger.error(f"Failed to create checkout session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create-portal")
def create_portal(client_id: str = Depends(get_client_id)):
    """Create a Stripe Customer Portal session."""
    try:
        def _fetch_customer_id():
            supabase = get_supabase_client()
            return supabase.table("clients").select("stripe_customer_id").eq("id", client_id).single().execute()
        
        res = execute_with_retry(
            _fetch_customer_id,
            retries=3,
            backoff_seconds=0.3,
            max_total_time=5.0
        )
        
        customer_id = res.data.get("stripe_customer_id") if res.data else None
        if not customer_id:
            raise HTTPException(status_code=400, detail="No Stripe customer found for this account. Please subscribe first.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch Stripe customer ID: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch customer data. Please try again.")
        
    try:
        portal_url = create_portal_session(customer_id)
        return {"portal_url": portal_url}
    except Exception as e:
        logger.error(f"Failed to create portal session: {e}")
        raise HTTPException(status_code=500, detail=str(e))
