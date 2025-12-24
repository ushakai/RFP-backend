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
    - Resource temporarily unavailable (connection pool exhausted)
    - Connection timeout
    - Network errors
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
                    # Also check the cause if it exists (for wrapped exceptions)
                    cause_str = str(e.__cause__).lower() if hasattr(e, "__cause__") and e.__cause__ else ""
                    
                    combined_error_str = error_str + " " + cause_str
                    error_type = type(e).__name__.lower()
                    
                    # Check if it's a retryable error
                    is_retryable = any([
                        "resource temporarily unavailable" in combined_error_str,
                        "winerror 10035" in combined_error_str,  # Windows non-blocking socket error
                        "winerror 10060" in combined_error_str,  # Connection timeout
                        "winerror 10061" in combined_error_str,  # Connection refused
                        "non-blocking socket" in combined_error_str,
                        "connection" in combined_error_str and "timeout" in combined_error_str,
                        "too many connections" in combined_error_str,
                        "connection refused" in combined_error_str,
                        "connection reset" in combined_error_str,
                        "forced" in combined_error_str and "closed" in combined_error_str,
                        "broken pipe" in combined_error_str,
                        "pool" in combined_error_str and ("exhausted" in combined_error_str or "timeout" in combined_error_str),
                        "readerror" in combined_error_str,  # httpx.ReadError
                        "connecterror" in combined_error_str,
                        "protocolerror" in combined_error_str,  # HTTP/2 protocol errors
                        "remoteprotocolerror" in combined_error_str,
                        "networkerror" in combined_error_str,
                        "keyerror" in error_type,  # HTTP/2 stream ID errors
                        "stream" in combined_error_str and ("id" in combined_error_str or "error" in combined_error_str),
                        "deque" in combined_error_str,  # deque mutation errors
                    ])
                    
                    if not is_retryable or attempt >= max_retries:
                        # Not retryable or out of retries
                        raise
                    
                    # Log retry attempt
                    print(f"Database error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    if e.__cause__ and e.__cause__ != e:
                        print(f"Cause: {e.__cause__}")
                    print(f"Retrying in {current_delay}s...")
                    
                    # Wait before retry
                    time.sleep(current_delay)
                    current_delay *= backoff
                    
                    # Try to reinitialize connection on retry
                    if attempt > 0:
                        try:
                            reinitialize_supabase()
                            print("Reinitialized Supabase client")
                        except Exception as reinit_error:
                            print(f"Warning: Failed to reinitialize client: {reinit_error}")
            
            # If we get here, all retries failed
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
                
                # Map technical errors to user-friendly messages
                if "resource temporarily unavailable" in error_str or "winerror 10035" in error_str:
                    raise Exception(
                        f"Database service is temporarily unavailable. "
                        f"This usually happens during high load. Please try again in a moment."
                    ) from e
                elif "connection" in error_str and "timeout" in error_str:
                    raise Exception(
                        f"Database connection timeout. The service might be slow. "
                        f"Please try again."
                    ) from e
                elif "too many connections" in error_str or "pool" in error_str:
                    raise Exception(
                        f"Database connection pool exhausted. The service is experiencing high load. "
                        f"Please try again shortly."
                    ) from e
                elif "permission denied" in error_str or "forbidden" in error_str:
                    raise Exception(
                        f"Database permission error. Please contact support if this persists."
                    ) from e
                elif "non-blocking socket" in error_str or "winerror 10054" in error_str or "forcibly closed" in error_str:
                    # Windows socket error or connection forcibly closed - retryable
                    # Reinitialize connection on this error
                    from config.settings import reinitialize_supabase
                    try:
                        reinitialize_supabase()
                    except:
                        pass
                    raise Exception(
                        f"Temporary network issue (connection closed). "
                        f"Please try again."
                    ) from e
                elif "keyerror" in str(type(e).__name__).lower() or "stream" in error_str or "protocol" in error_str or "deque" in error_str:
                    # HTTP/2 stream errors or protocol errors
                    from config.settings import reinitialize_supabase
                    try:
                        reinitialize_supabase()
                    except:
                        pass
                    raise Exception(
                        f"Connection protocol error. Please refresh the page and try again."
                    ) from e
                else:
                    # Re-raise with more context
                    raise Exception(
                        f"Database error during {operation_name}: {str(e)}"
                    ) from e
        
        return wrapper
    return decorator


def get_db_client_with_retry():
    """
    Get Supabase client with automatic retry on connection issues.
    """
    for attempt in range(3):
        try:
            client = get_supabase_client()
            # Test connection with a simple query
            client.table("clients").select("id").limit(1).execute()
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
                    "The service might be down or experiencing issues. "
                    "Please try again later."
                ) from e

