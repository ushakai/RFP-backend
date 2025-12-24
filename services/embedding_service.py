"""
Production-grade embedding service with batching, caching, and retry logic
Supports multiple providers (Gemini, OpenAI, custom)
"""
import time
import hashlib
from typing import List, Dict, Any, Optional
from functools import lru_cache
import google.generativeai as genai

from config.settings import GOOGLE_API_KEY
from config.rag_config import (
    EMBEDDING_PROVIDER,
    GEMINI_EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MAX_RETRIES,
    EMBEDDING_RETRY_DELAY,
    CACHE_EMBEDDINGS,
    EMBEDDING_CACHE_TTL_SECONDS,
    RAG_DEBUG_LOGGING,
    EmbeddingProvider,
)


# ==================== CONFIGURATION ====================

# Configure Gemini
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)


# ==================== CACHING ====================

# Simple in-memory cache (for production, use Redis or similar)
_embedding_cache: Dict[str, tuple[List[float], float]] = {}


def _get_cache_key(text: str, provider: str, model: str) -> str:
    """Generate cache key for text"""
    content = f"{provider}:{model}:{text}"
    return hashlib.sha256(content.encode()).hexdigest()


def _get_cached_embedding(text: str, provider: str, model: str) -> Optional[List[float]]:
    """Get cached embedding if available and not expired"""
    if not CACHE_EMBEDDINGS:
        return None
    
    cache_key = _get_cache_key(text, provider, model)
    if cache_key in _embedding_cache:
        embedding, timestamp = _embedding_cache[cache_key]
        age = time.time() - timestamp
        
        if age < EMBEDDING_CACHE_TTL_SECONDS:
            if RAG_DEBUG_LOGGING:
                print(f"   Cache hit for text (age: {age:.1f}s)")
            return embedding
        else:
            # Expired, remove from cache
            del _embedding_cache[cache_key]
    
    return None


def _cache_embedding(text: str, provider: str, model: str, embedding: List[float]) -> None:
    """Cache embedding with timestamp"""
    if not CACHE_EMBEDDINGS:
        return
    
    cache_key = _get_cache_key(text, provider, model)
    _embedding_cache[cache_key] = (embedding, time.time())
    
    # Simple cache size management (keep last 1000 entries)
    if len(_embedding_cache) > 1000:
        # Remove oldest 200 entries
        sorted_items = sorted(_embedding_cache.items(), key=lambda x: x[1][1])
        for key, _ in sorted_items[:200]:
            del _embedding_cache[key]


def clear_embedding_cache():
    """Clear all cached embeddings"""
    global _embedding_cache
    _embedding_cache.clear()
    print("✓ Embedding cache cleared")


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics"""
    if not _embedding_cache:
        return {"size": 0, "oldest_age": 0, "newest_age": 0}
    
    now = time.time()
    ages = [now - ts for _, (_, ts) in _embedding_cache.items()]
    
    return {
        "size": len(_embedding_cache),
        "oldest_age": max(ages) if ages else 0,
        "newest_age": min(ages) if ages else 0,
        "avg_age": sum(ages) / len(ages) if ages else 0,
    }


# ==================== GEMINI EMBEDDING ====================

def _get_gemini_embedding_single(text: str, model: str = GEMINI_EMBEDDING_MODEL) -> List[float]:
    """Get embedding from Gemini for a single text"""
    try:
        result = genai.embed_content(
            model=model,
            content=text,
            task_type="retrieval_document"  # Optimized for retrieval
        )
        return result['embedding']
    except Exception as e:
        print(f"Error getting Gemini embedding: {e}")
        raise


def _get_gemini_embeddings_batch(texts: List[str], model: str = GEMINI_EMBEDDING_MODEL) -> List[List[float]]:
    """Get embeddings from Gemini in batch (more efficient)"""
    try:
        # Gemini supports batch embedding
        results = []
        for text in texts:
            result = genai.embed_content(
                model=model,
                content=text,
                task_type="retrieval_document"
            )
            results.append(result['embedding'])
        return results
    except Exception as e:
        print(f"Error getting Gemini batch embeddings: {e}")
        raise


# ==================== RETRY LOGIC ====================

def _retry_with_backoff(func, *args, max_retries: int = EMBEDDING_MAX_RETRIES, delay: float = EMBEDDING_RETRY_DELAY, **kwargs):
    """Execute function with exponential backoff retry"""
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            
            if attempt < max_retries - 1:
                wait_time = delay * (2 ** attempt)  # Exponential backoff
                if RAG_DEBUG_LOGGING:
                    print(f"   Retry {attempt + 1}/{max_retries} after {wait_time:.1f}s: {e}")
                time.sleep(wait_time)
            else:
                print(f"   Failed after {max_retries} attempts: {e}")
    
    raise last_exception


# ==================== PUBLIC API ====================

def get_embedding(
    text: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    use_cache: bool = True
) -> List[float]:
    """
    Get embedding for a single text
    
    Args:
        text: Input text
        provider: Embedding provider (gemini, openai, custom)
        model: Model name (provider-specific)
        use_cache: Whether to use cache
        
    Returns:
        Embedding vector as list of floats
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    
    provider = provider or EMBEDDING_PROVIDER
    model = model or GEMINI_EMBEDDING_MODEL
    
    # Check cache first
    if use_cache:
        cached = _get_cached_embedding(text, provider, model)
        if cached is not None:
            return cached
    
    # Get embedding based on provider
    if provider == EmbeddingProvider.GEMINI:
        embedding = _retry_with_backoff(
            _get_gemini_embedding_single,
            text=text,
            model=model
        )
    elif provider == EmbeddingProvider.OPENAI:
        raise NotImplementedError("OpenAI embedding provider not yet implemented")
    elif provider == EmbeddingProvider.CUSTOM:
        raise NotImplementedError("Custom embedding provider not yet implemented")
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
    
    # Cache the result
    if use_cache:
        _cache_embedding(text, provider, model, embedding)
    
    # Validate dimension
    if len(embedding) != EMBEDDING_DIMENSION:
        print(f"⚠️  Warning: Expected {EMBEDDING_DIMENSION} dimensions, got {len(embedding)}")
    
    return embedding


def get_embeddings_batch(
    texts: List[str],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    use_cache: bool = True,
    batch_size: Optional[int] = None
) -> List[List[float]]:
    """
    Get embeddings for multiple texts (batched for efficiency)
    
    Args:
        texts: List of input texts
        provider: Embedding provider
        model: Model name
        use_cache: Whether to use cache
        batch_size: Batch size for processing (None = use default)
        
    Returns:
        List of embedding vectors
    """
    if not texts:
        return []
    
    provider = provider or EMBEDDING_PROVIDER
    model = model or GEMINI_EMBEDDING_MODEL
    batch_size = batch_size or EMBEDDING_BATCH_SIZE
    
    embeddings = []
    cache_hits = 0
    cache_misses = 0
    
    # Separate cached and uncached texts
    texts_to_embed = []
    text_indices = []
    
    for i, text in enumerate(texts):
        if not text or not text.strip():
            embeddings.append([0.0] * EMBEDDING_DIMENSION)  # Zero vector for empty text
            continue
        
        # Check cache
        if use_cache:
            cached = _get_cached_embedding(text, provider, model)
            if cached is not None:
                embeddings.append(cached)
                cache_hits += 1
                continue
        
        # Need to generate embedding
        texts_to_embed.append(text)
        text_indices.append(i)
        embeddings.append(None)  # Placeholder
        cache_misses += 1
    
    if RAG_DEBUG_LOGGING:
        print(f"   Embedding batch: {len(texts)} texts, {cache_hits} cache hits, {cache_misses} to generate")
    
    # Process uncached texts in batches
    if texts_to_embed:
        all_new_embeddings = []
        
        for batch_start in range(0, len(texts_to_embed), batch_size):
            batch_end = min(batch_start + batch_size, len(texts_to_embed))
            batch_texts = texts_to_embed[batch_start:batch_end]
            
            if RAG_DEBUG_LOGGING:
                print(f"   Processing batch {batch_start//batch_size + 1} ({len(batch_texts)} texts)")
            
            # Get embeddings for this batch
            if provider == EmbeddingProvider.GEMINI:
                batch_embeddings = _retry_with_backoff(
                    _get_gemini_embeddings_batch,
                    texts=batch_texts,
                    model=model
                )
            elif provider == EmbeddingProvider.OPENAI:
                raise NotImplementedError("OpenAI embedding provider not yet implemented")
            elif provider == EmbeddingProvider.CUSTOM:
                raise NotImplementedError("Custom embedding provider not yet implemented")
            else:
                raise ValueError(f"Unknown embedding provider: {provider}")
            
            all_new_embeddings.extend(batch_embeddings)
            
            # Small delay between batches to avoid rate limits
            if batch_end < len(texts_to_embed):
                time.sleep(0.1)
        
        # Insert new embeddings into result list and cache them
        for i, text_idx in enumerate(text_indices):
            embedding = all_new_embeddings[i]
            embeddings[text_idx] = embedding
            
            if use_cache:
                _cache_embedding(texts[text_idx], provider, model, embedding)
    
    return embeddings


def get_query_embedding(
    query: str,
    provider: Optional[str] = None,
    model: Optional[str] = None
) -> List[float]:
    """
    Get embedding optimized for query (search)
    
    Some models differentiate between document and query embeddings
    For Gemini, we use task_type="retrieval_query"
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    provider = provider or EMBEDDING_PROVIDER
    model = model or GEMINI_EMBEDDING_MODEL
    
    # Check cache (queries are often repeated)
    cached = _get_cached_embedding(query, f"{provider}:query", model)
    if cached is not None:
        return cached
    
    # Get query embedding
    if provider == EmbeddingProvider.GEMINI:
        try:
            result = _retry_with_backoff(
                genai.embed_content,
                model=model,
                content=query,
                task_type="retrieval_query"  # Optimized for queries
            )
            embedding = result['embedding']
        except Exception as e:
            print(f"Error getting Gemini query embedding: {e}")
            # Fallback to regular embedding
            embedding = get_embedding(query, provider, model)
    else:
        # For other providers, use regular embedding
        embedding = get_embedding(query, provider, model)
    
    # Cache the result
    _cache_embedding(query, f"{provider}:query", model, embedding)
    
    return embedding


def validate_embedding(embedding: List[float]) -> bool:
    """Validate embedding vector"""
    if not embedding:
        return False
    
    if not isinstance(embedding, list):
        return False
    
    if len(embedding) != EMBEDDING_DIMENSION:
        print(f"⚠️  Invalid embedding dimension: {len(embedding)}, expected {EMBEDDING_DIMENSION}")
        return False
    
    # Check for all zeros (invalid)
    if all(v == 0.0 for v in embedding):
        print("⚠️  Embedding is all zeros")
        return False
    
    # Check for NaN or inf
    if any(v != v or abs(v) == float('inf') for v in embedding):
        print("⚠️  Embedding contains NaN or inf")
        return False
    
    return True


def get_embedding_stats() -> Dict[str, Any]:
    """Get embedding service statistics"""
    cache_stats = get_cache_stats()
    
    return {
        "provider": EMBEDDING_PROVIDER,
        "model": GEMINI_EMBEDDING_MODEL,
        "dimension": EMBEDDING_DIMENSION,
        "batch_size": EMBEDDING_BATCH_SIZE,
        "max_retries": EMBEDDING_MAX_RETRIES,
        "cache_enabled": CACHE_EMBEDDINGS,
        "cache_ttl": EMBEDDING_CACHE_TTL_SECONDS,
        "cache_stats": cache_stats,
    }


# ==================== TESTING / DIAGNOSTICS ====================

def test_embedding_service():
    """Test embedding service with sample text"""
    print("🔍 Testing embedding service...")
    
    test_text = "This is a test document for the RAG system."
    
    try:
        # Test single embedding
        print(f"   Testing single embedding...")
        embedding = get_embedding(test_text)
        print(f"   ✓ Got embedding with {len(embedding)} dimensions")
        
        # Test cache
        print(f"   Testing cache...")
        embedding2 = get_embedding(test_text)
        assert embedding == embedding2, "Cache not working"
        print(f"   ✓ Cache working")
        
        # Test batch
        print(f"   Testing batch embedding...")
        test_texts = [
            "First test document",
            "Second test document",
            "Third test document"
        ]
        embeddings = get_embeddings_batch(test_texts)
        print(f"   ✓ Got {len(embeddings)} embeddings")
        
        # Test query embedding
        print(f"   Testing query embedding...")
        query_emb = get_query_embedding("test query")
        print(f"   ✓ Got query embedding with {len(query_emb)} dimensions")
        
        # Validate
        print(f"   Validating embeddings...")
        for emb in embeddings:
            assert validate_embedding(emb), "Invalid embedding"
        print(f"   ✓ All embeddings valid")
        
        # Stats
        stats = get_embedding_stats()
        print(f"\n📊 Embedding service stats:")
        print(f"   Provider: {stats['provider']}")
        print(f"   Model: {stats['model']}")
        print(f"   Dimension: {stats['dimension']}")
        print(f"   Cache size: {stats['cache_stats']['size']}")
        
        print("\n✅ All embedding tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Embedding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_embedding_service()

