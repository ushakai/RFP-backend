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
    """Create or get an existing Stripe customer ID for a client.
    
    Validates that the customer exists in Stripe before returning it.
    If the customer doesn't exist (e.g., when switching from test to live mode),
    it will clear the invalid ID and create a new customer.
    """
    supabase = get_supabase_client()
    
    # Check if client already has a stripe_customer_id
    res = supabase.table("clients").select("stripe_customer_id").eq("id", client_id).single().execute()
    existing_customer_id = res.data.get("stripe_customer_id") if res.data else None
    
    if existing_customer_id:
        # Validate that the customer exists in Stripe (handles test->live mode switch)
        try:
            stripe.Customer.retrieve(existing_customer_id)
            return existing_customer_id
        except stripe.error.InvalidRequestError:
            # Customer doesn't exist (likely from test mode, now using live mode)
            logger.warning(f"Customer {existing_customer_id} not found in Stripe, creating new customer for client {client_id}")
            # Clear the invalid customer ID
            supabase.table("clients").update({"stripe_customer_id": None}).eq("id", client_id).execute()
        except Exception as e:
            # Other Stripe errors - log and create new customer
            logger.error(f"Error validating Stripe customer {existing_customer_id}: {e}")
            supabase.table("clients").update({"stripe_customer_id": None}).eq("id", client_id).execute()
    
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
    
    # Clear tier if subscription is canceled/unpaid/past_due
    status = subscription_data.get("status")
    if status in ["canceled", "unpaid", "past_due"]:
        update_data["subscription_tier"] = None
        logger.info(f"Clearing subscription_tier for status: {status}")
    # Determine tier and interval from price ID if available (only if status is active)
    elif "items" in subscription_data:
        items = subscription_data["items"]
        if isinstance(items, dict) and "data" in items and items["data"]:
            item = items["data"][0]
            price_id = item["price"]["id"]
            interval = item["price"]["recurring"]["interval"]
            
            update_data["subscription_interval"] = interval
            # Tier logic: Map price ID to tier
            # We import here to avoid circular dependency if possible, or define mapping
            try:
                from config.settings import (
                    STRIPE_PRICE_TENDERS_MONTHLY, STRIPE_PRICE_TENDERS_YEARLY,
                    STRIPE_PRICE_PROCESSING_MONTHLY, STRIPE_PRICE_PROCESSING_YEARLY,
                    STRIPE_PRICE_BOTH_MONTHLY, STRIPE_PRICE_BOTH_YEARLY
                )
                
                PRICE_MAP = {
                    STRIPE_PRICE_TENDERS_MONTHLY: "tenders",
                    STRIPE_PRICE_TENDERS_YEARLY: "tenders",
                    STRIPE_PRICE_PROCESSING_MONTHLY: "processing",
                    STRIPE_PRICE_PROCESSING_YEARLY: "processing",
                    STRIPE_PRICE_BOTH_MONTHLY: "both",
                    STRIPE_PRICE_BOTH_YEARLY: "both"
                }
                
                # Remove None keys
                PRICE_MAP = {k: v for k, v in PRICE_MAP.items() if k}
                
                if price_id in PRICE_MAP:
                    update_data["subscription_tier"] = PRICE_MAP[price_id]
                    logger.info(f"Identified tier {update_data['subscription_tier']} from price {price_id}")
            except Exception as e:
                logger.error(f"Error determining tier from price: {e}")
        
    supabase.table("clients").update(update_data).eq("id", client_id).execute()

def sync_subscription_by_session(session_id: str, client_id: str) -> Dict[str, Any]:
    """Manually sync subscription status from a checkout session."""
    try:
        # Get checkout session
        session = stripe.checkout.Session.retrieve(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
            
        subscription_id = session.get("subscription")
        
        if not subscription_id:
             return {"status": "no_subscription_in_session"}

        # Get subscription details
        subscription = get_subscription_details(subscription_id)
        
        # Prepare data for update_client_subscription
        # It expects specific keys like 'status', 'current_period_end', 'items'
        formatted_sub_data = {
            "status": subscription.get("status"),
            "current_period_end": subscription.get("current_period_end"),
            "trial_end": subscription.get("trial_end"),
            "items": subscription.get("items")
        }
        
        # Re-use the existing logic to update DB
        update_client_subscription(client_id, formatted_sub_data)
        
        return {"success": True, "status": subscription.get("status")}
        
    except Exception as e:
        logger.error(f"Manual sync failed for session {session_id}: {e}")
        raise

def cancel_subscription(customer_id: str, cancel_immediately: bool = False, client_id: Optional[str] = None) -> Dict[str, Any]:
    """Cancel a customer's subscription."""
    try:
        # Get active subscription - check active first, then trialing
        subscriptions = stripe.Subscription.list(customer=customer_id, status="active", limit=10)
        
        # Also check for subscriptions that might be set to cancel_at_period_end but still active
        if not subscriptions.data:
            subscriptions = stripe.Subscription.list(customer=customer_id, status="trialing", limit=10)
        
        if not subscriptions.data:
            # Try to get any subscription regardless of status for this customer
            all_subs = stripe.Subscription.list(customer=customer_id, limit=100)  # Increase limit to find any subscription
            if not all_subs.data:
                # Check if customer exists
                try:
                    customer = stripe.Customer.retrieve(customer_id)
                    logger.warning(f"Customer {customer_id} exists but has no subscriptions. Email: {customer.email}")
                    
                    # Handle gracefully: Update local status to canceled
                    status = "canceled"
                    message = "No active subscription found. Local records updated."
                    
                    if client_id:
                        try:
                            supabase = get_supabase_client()
                            update_data = {
                                "subscription_status": status,
                            }
                            supabase.table("clients").update(update_data).eq("id", client_id).execute()
                            logger.info(f"Updated local database for client {client_id} to {status} (no stripe sub found)")
                        except Exception as e:
                            logger.error(f"Failed to update local database for client {client_id}: {e}")

                    return {
                        "success": True,
                        "message": message,
                        "status": status,
                        "cancel_at_period_end": False,
                        "period_end": None
                    }

                except stripe.error.InvalidRequestError:
                    raise ValueError(f"Invalid customer ID: {customer_id}")
            
            # Log all subscription statuses for debugging
            statuses = [sub.status for sub in all_subs.data]
            logger.info(f"Customer {customer_id} has {len(all_subs.data)} subscription(s) with statuses: {statuses}")
            
            # Use the most recent subscription
            subscription = sorted(all_subs.data, key=lambda x: x.created, reverse=True)[0]
            
            # Handle different subscription statuses
            if subscription.status == "canceled":
                raise ValueError(
                    f"Subscription is already canceled. "
                    f"If you need to resubscribe, please visit the pricing page."
                )
            elif subscription.status in ["unpaid", "past_due"]:
                raise ValueError(
                    f"Subscription is {subscription.status}. "
                    f"Please update your payment method to reactivate your subscription."
                )
            elif subscription.status == "incomplete_expired":
                raise ValueError(
                    f"Subscription setup expired. "
                    f"Please create a new subscription from the pricing page."
                )
            elif subscription.status in ["incomplete", "incomplete"]:
                raise ValueError(
                    f"Subscription setup is incomplete. "
                    f"Please complete your subscription setup first."
                )
            # If we get here, subscription exists but might be in an unexpected state
            logger.warning(f"Subscription {subscription.id} has status: {subscription.status}")
        else:
            subscription = subscriptions.data[0]

        
        subscription_id = subscription.id
        
        if cancel_immediately:
            # Cancel immediately
            try:
                canceled_subscription = stripe.Subscription.delete(subscription_id)
                status = "canceled"
                period_end_str = datetime.fromtimestamp(canceled_subscription.current_period_end, tz=timezone.utc).strftime("%B %d, %Y") if hasattr(canceled_subscription, 'current_period_end') and canceled_subscription.current_period_end else "now"
                message = f"Subscription canceled immediately. Access will end now."
            except stripe.error.InvalidRequestError as e:
                logger.error(f"Failed to delete subscription: {e}")
                raise ValueError(f"Failed to cancel subscription: {str(e)}")
        else:
            # Cancel at period end
            try:
                updated_subscription = stripe.Subscription.modify(
                    subscription_id,
                    cancel_at_period_end=True
                )
                status = updated_subscription.status
                period_end = datetime.fromtimestamp(updated_subscription.current_period_end, tz=timezone.utc)
                period_end_str = period_end.strftime("%B %d, %Y")
                message = f"Subscription will be canceled at the end of the billing period ({period_end_str}). You will retain access until then."
            except stripe.error.InvalidRequestError as e:
                logger.error(f"Failed to modify subscription: {e}")
                raise ValueError(f"Failed to schedule cancellation: {str(e)}")
        
        # Update local database if client_id is provided
        if client_id:
            try:
                supabase = get_supabase_client()
                update_data = {
                    "subscription_status": status,
                }
                
                # If canceling immediately, clear subscription_tier to remove access
                if cancel_immediately and status == "canceled":
                    update_data["subscription_tier"] = None
                    logger.info(f"Clearing subscription_tier for immediate cancellation")
                
                # Get full subscription details to update period_end
                full_sub = stripe.Subscription.retrieve(subscription_id)
                if hasattr(full_sub, 'current_period_end') and full_sub.current_period_end:
                    update_data["subscription_period_end"] = datetime.fromtimestamp(full_sub.current_period_end, tz=timezone.utc).isoformat()
                
                supabase.table("clients").update(update_data).eq("id", client_id).execute()
                logger.info(f"Updated local database for client {client_id} - status: {status}")
            except Exception as e:
                logger.warning(f"Failed to update local database for client {client_id}: {e}. Webhook will handle this.")
        
        return {
            "success": True,
            "message": message,
            "status": status,
            "cancel_at_period_end": not cancel_immediately,
            "period_end": period_end_str if not cancel_immediately else None
        }
    except ValueError as e:
        # Re-raise ValueError with clearer message
        raise e
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error canceling subscription: {e}")
        raise ValueError(f"Stripe error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error canceling subscription: {e}")
        raise ValueError(f"Failed to cancel subscription: {str(e)}")

def get_customer_subscription(customer_id: str) -> Optional[Dict[str, Any]]:
    """Get the customer's active subscription."""
    subscriptions = stripe.Subscription.list(customer=customer_id, status="active", limit=1)
    
    if not subscriptions.data:
        subscriptions = stripe.Subscription.list(customer=customer_id, status="trialing", limit=1)
    
    if not subscriptions.data:
        return None
    
    subscription = subscriptions.data[0]
    # Convert Stripe object to dict
    return {
        "id": subscription.id,
        "status": subscription.status,
        "default_payment_method": subscription.default_payment_method,
        "cancel_at_period_end": subscription.cancel_at_period_end,
        "current_period_end": subscription.current_period_end,
    }