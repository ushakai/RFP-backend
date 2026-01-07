import stripe
from fastapi import APIRouter, Request, Header, HTTPException
from config.settings import STRIPE_WEBHOOK_SECRET, get_supabase_client
from services.stripe_service import (
    get_subscription_details,
    update_client_subscription
)
from utils.logging_config import get_logger
from typing import Optional

router = APIRouter(prefix="/webhooks")
logger = get_logger(__name__, "webhooks")

@router.post("/stripe")
async def stripe_webhook(
    request: Request,
    stripe_signature: str = Header(None)
):
    """Handle Stripe webhook events."""
    # Security: Ensure webhook secret is configured
    if not STRIPE_WEBHOOK_SECRET:
        logger.error("STRIPE_WEBHOOK_SECRET is not configured")
        raise HTTPException(status_code=500, detail="Webhook secret not configured")
    
    # Security: Require signature header
    if not stripe_signature:
        logger.error("Missing Stripe signature header")
        raise HTTPException(status_code=400, detail="Missing signature")
    
    payload = await request.body()
    
    try:
        event = stripe.Webhook.construct_event(
            payload, stripe_signature, STRIPE_WEBHOOK_SECRET
        )
    except ValueError as e:
        # Invalid payload
        logger.error(f"Invalid payload: {e}")
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError as e:
        # Invalid signature
        logger.error(f"Invalid signature: {e}")
        raise HTTPException(status_code=400, detail="Invalid signature")

    event_type = event["type"]
    data_object = event["data"]["object"]

    logger.info(f"Webhook received: {event_type}")

    if event_type.startswith("customer.subscription"):
        # Handle subscription events
        subscription_id = data_object["id"]
        customer_id = data_object["customer"]
        
        # Get client_id from metadata or lookup by customer_id
        client_id = data_object.get("metadata", {}).get("client_id")
        
        if not client_id:
            # Lookup client_id by customer_id in DB
            supabase = get_supabase_client()
            res = supabase.table("clients").select("id").eq("stripe_customer_id", customer_id).single().execute()
            if res.data:
                client_id = res.data["id"]
        
        if client_id:
            subscription = get_subscription_details(subscription_id)
            
            # Map price ID to tier if updating
            tier = None
            if "items" in subscription:
                price_id = subscription["items"]["data"][0]["price"]["id"]
                # We can't import PRICE_TO_TIER here due to circular deps if we aren't careful
                # but we can look it up from config or use a helper
                from api.subscriptions import PRICE_TO_TIER
                tier = PRICE_TO_TIER.get(price_id)
            
            update_data = {
                "status": subscription["status"],
                "current_period_end": subscription["current_period_end"],
                "trial_end": subscription.get("trial_end"),
                "items": subscription.get("items")
            }
            
            update_client_subscription(client_id, update_data)
            
            # If tier was identified, update it too
            if tier:
                supabase = get_supabase_client()
                supabase.table("clients").update({"subscription_tier": tier}).eq("id", client_id).execute()
                
            logger.info(f"Updated subscription for client {client_id} (Status: {subscription['status']})")
        else:
            logger.warning(f"Could not find client for customer {customer_id}")

    elif event_type == "checkout.session.completed":
        # Handle checkout completion
        customer_id = data_object["customer"]
        client_id = data_object.get("metadata", {}).get("client_id")
        
        # Checkout carries the client_id usually
        if client_id:
            supabase = get_supabase_client()
            supabase.table("clients").update({"stripe_customer_id": customer_id}).eq("id", client_id).execute()
            logger.info(f"Checkout completed for client {client_id}")

    return {"status": "success"}
