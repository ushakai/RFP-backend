import stripe
from typing import Optional, Dict, Any
from datetime import datetime, timedelta, timezone
from config.settings import (
    STRIPE_SECRET_KEY,
    STRIPE_SUCCESS_URL,
    STRIPE_CANCEL_URL,
    STRIPE_TRIAL_PERIOD_MINUTES,
    get_supabase_client
)
from utils.logging_config import get_logger

logger = get_logger(__name__, "stripe")

if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

def create_or_get_customer(client_id: str, email: str, name: str) -> str:
    """Create or get an existing Stripe customer ID for a client."""
    supabase = get_supabase_client()
    
    # Check if client already has a stripe_customer_id
    res = supabase.table("clients").select("stripe_customer_id").eq("id", client_id).single().execute()
    if res.data and res.data.get("stripe_customer_id"):
        return res.data["stripe_customer_id"]
    
    # Create new customer in Stripe
    customer = stripe.Customer.create(
        email=email,
        name=name,
        metadata={"client_id": client_id}
    )
    
    # Update local database
    supabase.table("clients").update({"stripe_customer_id": customer.id}).eq("id", client_id).execute()
    
    return customer.id

def create_checkout_session(customer_id: str, price_id: str, client_id: str, has_used_trial: bool = False) -> str:
    """Create a Stripe Checkout Session for a subscription."""
    subscription_data = {
        "metadata": {"client_id": client_id},
    }
    
    # Only add trial if they haven't used it before
    if not has_used_trial and STRIPE_TRIAL_PERIOD_MINUTES > 0:
        trial_end_dt = datetime.now(timezone.utc) + timedelta(minutes=STRIPE_TRIAL_PERIOD_MINUTES)
        subscription_data["trial_end"] = int(trial_end_dt.timestamp())
        
    session = stripe.checkout.Session.create(
        customer=customer_id,
        payment_method_types=["card"],
        line_items=[{
            "price": price_id,
            "quantity": 1,
        }],
        mode="subscription",
        success_url=STRIPE_SUCCESS_URL + "?session_id={CHECKOUT_SESSION_ID}",
        cancel_url=STRIPE_CANCEL_URL,
        subscription_data=subscription_data,
        metadata={"client_id": client_id}
    )
    return session.url

def create_portal_session(customer_id: str) -> str:
    """Create a Stripe Customer Portal session for managing subscriptions."""
    session = stripe.billing_portal.Session.create(
        customer=customer_id,
        return_url=STRIPE_CANCEL_URL,
    )
    return session.url

def get_subscription_details(subscription_id: str) -> Dict[str, Any]:
    """Retrieve subscription details from Stripe."""
    subscription = stripe.Subscription.retrieve(subscription_id)
    return subscription

def update_client_subscription(client_id: str, subscription_data: Dict[str, Any]):
    """Update the client's subscription status in the local database."""
    supabase = get_supabase_client()
    
    from datetime import datetime
    
    current_period_end = subscription_data.get("current_period_end")
    trial_end = subscription_data.get("trial_end")
    
    update_data = {
        "subscription_status": subscription_data.get("status"),
    }
    
    if current_period_end:
        update_data["subscription_period_end"] = datetime.fromtimestamp(current_period_end).isoformat()
    
    if trial_end:
        update_data["trial_end"] = datetime.fromtimestamp(trial_end).isoformat()
    
    # Determine tier and interval from price ID if available
    if "items" in subscription_data:
        items = subscription_data["items"]
        if isinstance(items, dict) and "data" in items and items["data"]:
            item = items["data"][0]
            price_id = item["price"]["id"]
            interval = item["price"]["recurring"]["interval"]
            
            update_data["subscription_interval"] = interval
            # Tier logic will be handled by mapping price IDs in the API or webhook
        
    supabase.table("clients").update(update_data).eq("id", client_id).execute()
