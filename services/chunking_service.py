"""
Production-grade text chunking service for RAG
Implements semantic chunking with proper token limits
"""
import re
import tiktoken
from typing import List, Dict, Any


def get_token_count(text: str, model: str = "cl100k_base") -> int:
    """Get accurate token count for text"""
    try:
        encoding = tiktoken.get_encoding(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback: rough estimate (1 token ≈ 4 chars)
        return len(text) // 4


def split_by_sentences(text: str) -> List[str]:
    """Split text into sentences using multiple delimiters"""
    # Handle common sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text_semantic(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    min_chunk_size: int = 100,
    max_chunk_size: int = 600
) -> List[Dict[str, Any]]:
    """
    Chunk text semantically with proper token limits for production RAG.
    
    Args:
        text: Input text to chunk
        chunk_size: Target chunk size in tokens (default: 512 - optimal for most embeddings)
        overlap: Overlap size in tokens (default: 50 - ~10%)
        min_chunk_size: Minimum chunk size in tokens
        max_chunk_size: Maximum chunk size in tokens (hard limit)
    
    Returns:
        List of dicts with 'text', 'tokens', 'char_count', 'index'
    
    Why 512 tokens?
    - Gemini embedding model handles up to 2048 tokens but performs best with 256-512
    - Smaller chunks = more precise retrieval
    - Larger chunks = more context but less precise
    - 512 is sweet spot for question-answering tasks
    """
    if not text or not text.strip():
        return []
    
    # Clean text
    text = text.strip()
    text = re.sub(r'\n{3,}', '\n\n', text)  # Collapse multiple newlines
    text = re.sub(r' {2,}', ' ', text)  # Collapse multiple spaces
    
    # Split into paragraphs first (preserve document structure)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    chunk_index = 0
    
    for para in paragraphs:
        para_tokens = get_token_count(para)
        
        # If single paragraph exceeds max, split by sentences
        if para_tokens > max_chunk_size:
            sentences = split_by_sentences(para)
            for sent in sentences:
                sent_tokens = get_token_count(sent)
                
                # If adding this sentence exceeds chunk_size, save current chunk
                if current_tokens + sent_tokens > chunk_size and current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    if get_token_count(chunk_text) >= min_chunk_size:
                        chunks.append({
                            'text': chunk_text,
                            'tokens': get_token_count(chunk_text),
                            'char_count': len(chunk_text),
                            'index': chunk_index
                        })
                        chunk_index += 1
                    
                    # Keep overlap (last N tokens worth of text)
                    if overlap > 0 and current_chunk:
                        overlap_text = ' '.join(current_chunk[-2:])  # Last 2 sentences for context
                        overlap_tokens = get_token_count(overlap_text)
                        if overlap_tokens <= overlap:
                            current_chunk = current_chunk[-2:]
                            current_tokens = overlap_tokens
                        else:
                            current_chunk = current_chunk[-1:]
                            current_tokens = get_token_count(current_chunk[0])
                    else:
                        current_chunk = []
                        current_tokens = 0
                
                current_chunk.append(sent)
                current_tokens += sent_tokens
        else:
            # If adding this paragraph exceeds chunk_size, save current chunk
            if current_tokens + para_tokens > chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                if get_token_count(chunk_text) >= min_chunk_size:
                    chunks.append({
                        'text': chunk_text,
                        'tokens': get_token_count(chunk_text),
                        'char_count': len(chunk_text),
                        'index': chunk_index
                    })
                    chunk_index += 1
                
                # Keep overlap
                if overlap > 0 and current_chunk:
                    overlap_text = ' '.join(current_chunk[-2:])
                    overlap_tokens = get_token_count(overlap_text)
                    if overlap_tokens <= overlap:
                        current_chunk = current_chunk[-2:]
                        current_tokens = overlap_tokens
                    else:
                        current_chunk = current_chunk[-1:]
                        current_tokens = get_token_count(current_chunk[0])
                else:
                    current_chunk = []
                    current_tokens = 0
            
            current_chunk.append(para)
            current_tokens += para_tokens
    
    # Add remaining chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        if get_token_count(chunk_text) >= min_chunk_size:
            chunks.append({
                'text': chunk_text,
                'tokens': get_token_count(chunk_text),
                'char_count': len(chunk_text),
                'index': chunk_index
            })
    
    return chunks


def chunk_text_simple(
    text: str,
    chunk_size_chars: int = 2000,
    overlap_chars: int = 200
) -> List[str]:
    """
    Simple character-based chunking (fallback for when token counting fails)
    """
    if not text or not text.strip():
        return []
    
    text = text.strip()
    chunks = []
    start = 0
    n = len(text)
    
    while start < n:
        end = min(n, start + chunk_size_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap_chars
    
    return chunks

