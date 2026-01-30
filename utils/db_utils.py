"""
Database utilities for connection management and retry logic
"""
import time
import traceback
from functools import wraps
from typing import Callable, Any
from config.settings import get_supabase_client, reinitialize_supabase


def retry_on_db_error(max_retries: int = 3, delay: float = 0.5, backoff: float = 2.0):
    """
    Decorator to retry database operations on connection errors.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay on each retry
    
    Common errors handled:
    - Connection errors (psycopg2, network)
    - Pool exhaustion
    - Timeouts
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_str = str(e).lower()
                    cause_str = str(e.__cause__).lower() if hasattr(e, "__cause__") and e.__cause__ else ""
                    combined_error_str = error_str + " " + cause_str
                    error_type = type(e).__name__.lower()
                    
                    # Check if it's a retryable error
                    is_retryable = any([
                        "operationalerror" in error_type,  # psycopg2 connection errors
                        "interfaceerror" in error_type,    # psycopg2 interface errors
                        "connectionerror" in error_type,
                        "connection" in combined_error_str and ("timeout" in combined_error_str or "closed" in combined_error_str),
                        "too many connections" in combined_error_str,
                        "pool" in combined_error_str and ("exhausted" in combined_error_str or "timeout" in combined_error_str),
                        "broken pipe" in combined_error_str,
                        "connection reset" in combined_error_str,
                        "connection refused" in combined_error_str,
                        "server closed" in combined_error_str,
                        "ssl" in combined_error_str and "error" in combined_error_str,
                        # Legacy HTTP-based errors (fallback mode)
                        "resource temporarily unavailable" in combined_error_str,
                        "winerror 10035" in combined_error_str,
                        "winerror 10060" in combined_error_str,
                        "winerror 10061" in combined_error_str,
                    ])
                    
                    if not is_retryable or attempt >= max_retries:
                        raise
                    
                    print(f"Database error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    print(f"Retrying in {current_delay}s...")
                    
                    time.sleep(current_delay)
                    current_delay *= backoff
                    
                    # Try to reinitialize connection on retry
                    if attempt > 0:
                        try:
                            reinitialize_supabase()
                            print("Reinitialized database connection")
                        except Exception as reinit_error:
                            print(f"Warning: Failed to reinitialize: {reinit_error}")
            
            if last_exception:
                raise last_exception
            
        return wrapper
    return decorator


def safe_db_operation(operation_name: str = "database operation"):
    """
    Decorator for safe database operations with better error messages.
    Wraps retry logic and provides user-friendly error messages.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        @retry_on_db_error(max_retries=2, delay=0.3, backoff=2.0)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_str = str(e).lower()
                error_type = type(e).__name__.lower()
                
                # Map technical errors to user-friendly messages
                if "operationalerror" in error_type or "connection" in error_str:
                    try:
                        reinitialize_supabase()
                    except:
                        pass
                    raise Exception(
                        f"Database connection error. Please try again."
                    ) from e
                elif "too many connections" in error_str or "pool" in error_str:
                    raise Exception(
                        f"Database is busy. Please try again shortly."
                    ) from e
                elif "timeout" in error_str:
                    raise Exception(
                        f"Database request timed out. Please try again."
                    ) from e
                elif "permission" in error_str or "denied" in error_str:
                    raise Exception(
                        f"Database permission error. Please contact support."
                    ) from e
                else:
                    raise Exception(
                        f"Database error during {operation_name}: {str(e)}"
                    ) from e
        
        return wrapper
    return decorator


def get_db_client_with_retry():
    """
    Get database client with automatic retry on connection issues.
    """
    for attempt in range(3):
        try:
            client = get_supabase_client()
            # Test connection with a simple query
            result = client.table("clients").select("id").limit(1).execute()
            if hasattr(result, 'error') and result.error:
                raise Exception(result.error)
            return client
        except Exception as e:
            if attempt < 2:
                print(f"Failed to get DB client (attempt {attempt + 1}/3): {e}")
                time.sleep(0.5 * (attempt + 1))
                try:
                    reinitialize_supabase()
                except Exception:
                    pass
            else:
                raise Exception(
                    "Cannot connect to database after multiple attempts. "
                    "Please try again later."
                ) from e

