"""
Production-grade RAG Configuration
Centralized settings for document ingestion, chunking, embedding, and retrieval
"""
import os
from enum import Enum
from typing import Dict, Any


class ChunkingStrategy(str, Enum):
    """Available chunking strategies"""
    SEMANTIC = "semantic"           # Best for general documents
    SENTENCE = "sentence"           # Best for Q&A, legal docs
    RECURSIVE = "recursive"         # Best for code, structured text
    SLIDING_WINDOW = "sliding"      # Best for dense technical docs


class EmbeddingProvider(str, Enum):
    """Supported embedding providers"""
    GEMINI = "gemini"
    OPENAI = "openai"
    CUSTOM = "custom"


# ==================== CHUNKING CONFIGURATION ====================

# Default chunking strategy
CHUNKING_STRATEGY = os.getenv("RAG_CHUNKING_STRATEGY", ChunkingStrategy.SEMANTIC)

# Chunk size in tokens (critical parameter)
# Smaller = more precise retrieval, more chunks, higher costs
# Larger = more context, fewer chunks, less precise
# Recommended: 256-512 for QA, 512-1024 for general docs
CHUNK_SIZE_TOKENS = int(os.getenv("RAG_CHUNK_SIZE", "250"))

# Overlap between chunks in tokens (prevents context loss at boundaries)
# Recommended: 10-20% of chunk size
CHUNK_OVERLAP_TOKENS = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))

# Minimum chunk size (discard smaller chunks to avoid noise)
MIN_CHUNK_SIZE_TOKENS = int(os.getenv("RAG_MIN_CHUNK_SIZE", "20"))  # Lowered from 50 to 20 for better success rate

# Maximum chunk size (hard limit, splits if exceeded)
MAX_CHUNK_SIZE_TOKENS = int(os.getenv("RAG_MAX_CHUNK_SIZE", "600"))

# Preserve paragraph boundaries when chunking
PRESERVE_PARAGRAPHS = os.getenv("RAG_PRESERVE_PARAGRAPHS", "1") == "1"

# Preserve sentence boundaries when chunking
PRESERVE_SENTENCES = os.getenv("RAG_PRESERVE_SENTENCES", "1") == "1"


# ==================== EMBEDDING CONFIGURATION ====================

# Embedding provider
EMBEDDING_PROVIDER = os.getenv("RAG_EMBEDDING_PROVIDER", EmbeddingProvider.GEMINI)

# Gemini embedding model
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")

# Embedding dimension (model-specific)
EMBEDDING_DIMENSION = int(os.getenv("RAG_EMBEDDING_DIM", "768"))

# Batch size for embedding generation (parallel processing)
EMBEDDING_BATCH_SIZE = int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "10"))

# Retry configuration for embedding API calls
EMBEDDING_MAX_RETRIES = int(os.getenv("RAG_EMBEDDING_RETRIES", "3"))
EMBEDDING_RETRY_DELAY = float(os.getenv("RAG_EMBEDDING_RETRY_DELAY", "0.5"))


# ==================== RETRIEVAL CONFIGURATION ====================

# Hybrid search weights (must sum to 1.0)
DENSE_SEARCH_WEIGHT = float(os.getenv("RAG_DENSE_WEIGHT", "0.7"))  # Vector similarity
SPARSE_SEARCH_WEIGHT = float(os.getenv("RAG_SPARSE_WEIGHT", "0.3"))  # BM25 keyword

# Number of documents to retrieve
TOP_K_DOCUMENTS = int(os.getenv("RAG_TOP_K_DOCS", "5"))
TOP_K_QA_PAIRS = int(os.getenv("RAG_TOP_K_QA", "3"))

# Minimum similarity threshold (0-1)
MIN_SIMILARITY_THRESHOLD = float(os.getenv("RAG_MIN_SIMILARITY", "0.3"))

# Reranking enabled (for better precision)
ENABLE_RERANKING = os.getenv("RAG_ENABLE_RERANKING", "0") == "1"

# Maximum context length for LLM (in tokens)
MAX_CONTEXT_TOKENS = int(os.getenv("RAG_MAX_CONTEXT_TOKENS", "4000"))


# ==================== TEXT EXTRACTION CONFIGURATION ====================

# Maximum file size for text extraction (in MB)
MAX_TEXT_FILE_SIZE_MB = int(os.getenv("RAG_MAX_FILE_SIZE_MB", "50"))

# Supported text extraction formats
SUPPORTED_TEXT_FORMATS = {
    "txt": True,
    "md": True,
    "markdown": True,
    "pdf": os.getenv("RAG_SUPPORT_PDF", "1") == "1",
    "docx": os.getenv("RAG_SUPPORT_DOCX", "1") == "1",
    "doc": os.getenv("RAG_SUPPORT_DOC", "0") == "1",  # Requires extra libs
}

# OCR for scanned PDFs
ENABLE_OCR = os.getenv("RAG_ENABLE_OCR", "0") == "1"


# ==================== PERFORMANCE & LOGGING ====================

# Enable detailed logging for debugging
RAG_DEBUG_LOGGING = os.getenv("RAG_DEBUG_LOGGING", "0") == "1"

# Log chunk statistics
LOG_CHUNK_STATS = os.getenv("RAG_LOG_CHUNK_STATS", "1") == "1"

# Log retrieval performance
LOG_RETRIEVAL_PERFORMANCE = os.getenv("RAG_LOG_RETRIEVAL_PERF", "1") == "1"

# Cache embeddings (for identical queries)
CACHE_EMBEDDINGS = os.getenv("RAG_CACHE_EMBEDDINGS", "1") == "1"
EMBEDDING_CACHE_TTL_SECONDS = int(os.getenv("RAG_EMBEDDING_CACHE_TTL", "3600"))


# ==================== QUALITY CONTROL ====================

# Minimum text quality score (0-1, filters out low-quality extractions)
MIN_TEXT_QUALITY_SCORE = float(os.getenv("RAG_MIN_QUALITY", "0.5"))

# Deduplicate similar chunks (prevents redundant storage)
DEDUPLICATE_CHUNKS = os.getenv("RAG_DEDUPLICATE_CHUNKS", "1") == "1"
DEDUP_SIMILARITY_THRESHOLD = float(os.getenv("RAG_DEDUP_THRESHOLD", "0.95"))


# ==================== VALIDATION ====================

def validate_rag_config():
    """Validate RAG configuration on startup"""
    errors = []
    
    # Check weights sum to 1.0
    if abs(DENSE_SEARCH_WEIGHT + SPARSE_SEARCH_WEIGHT - 1.0) > 0.01:
        errors.append(
            f"RAG_DENSE_WEIGHT ({DENSE_SEARCH_WEIGHT}) + RAG_SPARSE_WEIGHT ({SPARSE_SEARCH_WEIGHT}) must sum to 1.0"
        )
    
    # Check chunk size ranges
    if CHUNK_SIZE_TOKENS < MIN_CHUNK_SIZE_TOKENS:
        errors.append(
            f"RAG_CHUNK_SIZE ({CHUNK_SIZE_TOKENS}) must be >= RAG_MIN_CHUNK_SIZE ({MIN_CHUNK_SIZE_TOKENS})"
        )
    
    if CHUNK_SIZE_TOKENS > MAX_CHUNK_SIZE_TOKENS:
        errors.append(
            f"RAG_CHUNK_SIZE ({CHUNK_SIZE_TOKENS}) must be <= RAG_MAX_CHUNK_SIZE ({MAX_CHUNK_SIZE_TOKENS})"
        )
    
    # Check overlap is reasonable
    if CHUNK_OVERLAP_TOKENS >= CHUNK_SIZE_TOKENS:
        errors.append(
            f"RAG_CHUNK_OVERLAP ({CHUNK_OVERLAP_TOKENS}) must be < RAG_CHUNK_SIZE ({CHUNK_SIZE_TOKENS})"
        )
    
    # Check similarity thresholds
    if not 0 <= MIN_SIMILARITY_THRESHOLD <= 1:
        errors.append(
            f"RAG_MIN_SIMILARITY ({MIN_SIMILARITY_THRESHOLD}) must be between 0 and 1"
        )
    
    if errors:
        error_msg = "\n".join([f"  - {err}" for err in errors])
        raise ValueError(f"Invalid RAG configuration:\n{error_msg}")
    
    return True


def get_rag_config_summary() -> Dict[str, Any]:
    """Get current RAG configuration as dictionary"""
    return {
        "chunking": {
            "strategy": CHUNKING_STRATEGY,
            "chunk_size_tokens": CHUNK_SIZE_TOKENS,
            "overlap_tokens": CHUNK_OVERLAP_TOKENS,
            "min_chunk_tokens": MIN_CHUNK_SIZE_TOKENS,
            "max_chunk_tokens": MAX_CHUNK_SIZE_TOKENS,
            "preserve_paragraphs": PRESERVE_PARAGRAPHS,
            "preserve_sentences": PRESERVE_SENTENCES,
        },
        "embedding": {
            "provider": EMBEDDING_PROVIDER,
            "model": GEMINI_EMBEDDING_MODEL,
            "dimension": EMBEDDING_DIMENSION,
            "batch_size": EMBEDDING_BATCH_SIZE,
        },
        "retrieval": {
            "dense_weight": DENSE_SEARCH_WEIGHT,
            "sparse_weight": SPARSE_SEARCH_WEIGHT,
            "top_k_docs": TOP_K_DOCUMENTS,
            "top_k_qa": TOP_K_QA_PAIRS,
            "min_similarity": MIN_SIMILARITY_THRESHOLD,
            "reranking_enabled": ENABLE_RERANKING,
        },
        "quality": {
            "min_quality_score": MIN_TEXT_QUALITY_SCORE,
            "deduplicate": DEDUPLICATE_CHUNKS,
            "dedup_threshold": DEDUP_SIMILARITY_THRESHOLD,
        },
        "performance": {
            "cache_embeddings": CACHE_EMBEDDINGS,
            "cache_ttl": EMBEDDING_CACHE_TTL_SECONDS,
            "debug_logging": RAG_DEBUG_LOGGING,
        },
    }


# Validate on import
try:
    validate_rag_config()
except ValueError as e:
    print(f"⚠️  RAG Configuration Warning: {e}")
    print("⚠️  Using default values. Check your .env file.")

