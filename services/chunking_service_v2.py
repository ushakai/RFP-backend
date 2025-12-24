"""
Production-grade text chunking service for RAG
Implements multiple chunking strategies with best practices
"""
import re
import tiktoken
from typing import List, Dict, Any, Optional
from config.rag_config import (
    CHUNK_SIZE_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    MIN_CHUNK_SIZE_TOKENS,
    MAX_CHUNK_SIZE_TOKENS,
    PRESERVE_PARAGRAPHS,
    PRESERVE_SENTENCES,
    ChunkingStrategy,
    RAG_DEBUG_LOGGING,
    LOG_CHUNK_STATS,
)


class TextChunker:
    """Production-grade text chunker with multiple strategies"""
    
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE_TOKENS,
        overlap: int = CHUNK_OVERLAP_TOKENS,
        min_size: int = MIN_CHUNK_SIZE_TOKENS,
        max_size: int = MAX_CHUNK_SIZE_TOKENS,
        encoding_model: str = "cl100k_base"
    ):
        """
        Initialize chunker with configurable parameters
        
        Args:
            chunk_size: Target chunk size in tokens
            overlap: Overlap between chunks in tokens
            min_size: Minimum chunk size (discard smaller chunks)
            max_size: Maximum chunk size (hard limit)
            encoding_model: Tokenizer model (cl100k_base for GPT-4, Gemini)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_size = min_size
        self.max_size = max_size
        
        try:
            self.encoding = tiktoken.get_encoding(encoding_model)
        except Exception as e:
            print(f"Warning: Failed to load tiktoken encoding {encoding_model}: {e}")
            self.encoding = None
    
    def get_token_count(self, text: str) -> int:
        """Get accurate token count for text"""
        if self.encoding:
            try:
                return len(self.encoding.encode(text))
            except Exception:
                pass
        # Fallback: rough estimate (1 token ≈ 4 chars for English)
        return max(1, len(text) // 4)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\n{4,}', '\n\n\n', text)  # Max 3 newlines
        text = re.sub(r' {3,}', '  ', text)  # Max 2 spaces
        text = re.sub(r'\t+', ' ', text)  # Replace tabs
        
        # Fix common issues
        text = text.strip()
        
        return text
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        # Split on double newlines
        paragraphs = re.split(r'\n\n+', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text into sentences with better accuracy"""
        # Handle common abbreviations to avoid false splits
        text = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr|Inc|Ltd|Co|Corp|vs|etc|e\.g|i\.e)\.\s', r'\1<PERIOD> ', text)
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Restore periods
        sentences = [s.replace('<PERIOD>', '.').strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _create_chunk_metadata(
        self, 
        text: str, 
        index: int, 
        strategy: str,
        start_pos: Optional[int] = None,
        end_pos: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create metadata for a chunk"""
        tokens = self.get_token_count(text)
        
        metadata = {
            'text': text,
            'tokens': tokens,
            'char_count': len(text),
            'index': index,
            'strategy': strategy,
        }
        
        if start_pos is not None:
            metadata['start_pos'] = start_pos
        if end_pos is not None:
            metadata['end_pos'] = end_pos
        
        return metadata
    
    def chunk_semantic(self, text: str) -> List[Dict[str, Any]]:
        """
        Semantic chunking: Preserves natural text boundaries
        Best for: General documents, articles, reports
        
        Strategy:
        1. Split by paragraphs first
        2. Group paragraphs into chunks respecting token limits
        3. Split large paragraphs by sentences
        4. Maintain overlap for context continuity
        """
        text = self._clean_text(text)
        if not text:
            return []
        
        paragraphs = self._split_by_paragraphs(text) if PRESERVE_PARAGRAPHS else [text]
        
        chunks = []
        current_chunk_parts = []
        current_tokens = 0
        chunk_index = 0
        
        for para in paragraphs:
            para_tokens = self.get_token_count(para)
            
            # If single paragraph exceeds max, split by sentences
            if para_tokens > self.max_size:
                sentences = self._split_by_sentences(para) if PRESERVE_SENTENCES else [para]
                
                for sent in sentences:
                    sent_tokens = self.get_token_count(sent)
                    
                    # If sentence alone exceeds max, force split
                    if sent_tokens > self.max_size:
                        # Fallback: split by words
                        words = sent.split()
                        word_chunk = []
                        word_tokens = 0
                        
                        for word in words:
                            word_token = self.get_token_count(word)
                            if word_tokens + word_token > self.chunk_size and word_chunk:
                                chunk_text = ' '.join(word_chunk)
                                if self.get_token_count(chunk_text) >= self.min_size:
                                    chunks.append(self._create_chunk_metadata(
                                        chunk_text, chunk_index, "semantic-word-split"
                                    ))
                                    chunk_index += 1
                                word_chunk = [word]
                                word_tokens = word_token
                            else:
                                word_chunk.append(word)
                                word_tokens += word_token
                        
                        if word_chunk:
                            chunk_text = ' '.join(word_chunk)
                            if self.get_token_count(chunk_text) >= self.min_size:
                                chunks.append(self._create_chunk_metadata(
                                    chunk_text, chunk_index, "semantic-word-split"
                                ))
                                chunk_index += 1
                        continue
                    
                    # Check if adding sentence exceeds target chunk size
                    if current_tokens + sent_tokens > self.chunk_size and current_chunk_parts:
                        # Save current chunk
                        chunk_text = ' '.join(current_chunk_parts)
                        if self.get_token_count(chunk_text) >= self.min_size:
                            chunks.append(self._create_chunk_metadata(
                                chunk_text, chunk_index, "semantic"
                            ))
                            chunk_index += 1
                        
                        # Keep overlap (last few sentences)
                        if self.overlap > 0 and len(current_chunk_parts) > 1:
                            overlap_parts = []
                            overlap_tokens = 0
                            for part in reversed(current_chunk_parts):
                                part_tokens = self.get_token_count(part)
                                if overlap_tokens + part_tokens <= self.overlap:
                                    overlap_parts.insert(0, part)
                                    overlap_tokens += part_tokens
                                else:
                                    break
                            current_chunk_parts = overlap_parts
                            current_tokens = overlap_tokens
                        else:
                            current_chunk_parts = []
                            current_tokens = 0
                    
                    current_chunk_parts.append(sent)
                    current_tokens += sent_tokens
            else:
                # Normal paragraph processing
                if current_tokens + para_tokens > self.chunk_size and current_chunk_parts:
                    # Save current chunk
                    chunk_text = ' '.join(current_chunk_parts)
                    if self.get_token_count(chunk_text) >= self.min_size:
                        chunks.append(self._create_chunk_metadata(
                            chunk_text, chunk_index, "semantic"
                        ))
                        chunk_index += 1
                    
                    # Keep overlap
                    if self.overlap > 0 and len(current_chunk_parts) > 1:
                        overlap_parts = []
                        overlap_tokens = 0
                        for part in reversed(current_chunk_parts):
                            part_tokens = self.get_token_count(part)
                            if overlap_tokens + part_tokens <= self.overlap:
                                overlap_parts.insert(0, part)
                                overlap_tokens += part_tokens
                            else:
                                break
                        current_chunk_parts = overlap_parts
                        current_tokens = overlap_tokens
                    else:
                        current_chunk_parts = []
                        current_tokens = 0
                
                current_chunk_parts.append(para)
                current_tokens += para_tokens
        
        # Add final chunk
        if current_chunk_parts:
            chunk_text = ' '.join(current_chunk_parts)
            if self.get_token_count(chunk_text) >= self.min_size:
                chunks.append(self._create_chunk_metadata(
                    chunk_text, chunk_index, "semantic"
                ))
        
        return chunks
    
    def chunk_recursive(self, text: str, separators: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Recursive chunking: Tries multiple separators in order
        Best for: Code, structured documents, technical specs
        
        Separators tried in order: paragraphs → sentences → words → chars
        """
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]
        
        text = self._clean_text(text)
        if not text:
            return []
        
        def _recursive_split(text: str, sep_idx: int = 0) -> List[str]:
            """Recursively split text"""
            if sep_idx >= len(separators):
                return [text]
            
            separator = separators[sep_idx]
            if not separator:
                # Base case: split by characters
                return [text[i:i+self.chunk_size*4] for i in range(0, len(text), self.chunk_size*4)]
            
            splits = text.split(separator) if separator else [text]
            chunks = []
            current = ""
            
            for split in splits:
                split_tokens = self.get_token_count(split)
                current_tokens = self.get_token_count(current)
                
                if split_tokens > self.max_size:
                    # Split is too large, use next separator
                    if current:
                        chunks.append(current)
                        current = ""
                    chunks.extend(_recursive_split(split, sep_idx + 1))
                elif current_tokens + split_tokens > self.chunk_size:
                    if current:
                        chunks.append(current)
                    current = split
                else:
                    current = (current + separator + split) if current else split
            
            if current:
                chunks.append(current)
            
            return chunks
        
        raw_chunks = _recursive_split(text)
        
        # Convert to metadata format
        result = []
        for idx, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if self.get_token_count(chunk_text) >= self.min_size:
                result.append(self._create_chunk_metadata(
                    chunk_text, idx, "recursive"
                ))
        
        return result
    
    def chunk_sliding_window(self, text: str) -> List[Dict[str, Any]]:
        """
        Sliding window chunking: Fixed-size windows with overlap
        Best for: Dense technical documents, where every part matters
        
        Creates uniform chunks with consistent overlap
        """
        text = self._clean_text(text)
        if not text:
            return []
        
        # Use sentence splitting as base
        sentences = self._split_by_sentences(text) if PRESERVE_SENTENCES else text.split()
        
        chunks = []
        chunk_index = 0
        i = 0
        
        while i < len(sentences):
            current_chunk = []
            current_tokens = 0
            
            # Collect sentences until we reach target size
            while i < len(sentences) and current_tokens < self.chunk_size:
                sent = sentences[i]
                sent_tokens = self.get_token_count(sent)
                
                if current_tokens + sent_tokens > self.max_size and current_chunk:
                    break
                
                current_chunk.append(sent)
                current_tokens += sent_tokens
                i += 1
            
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                if self.get_token_count(chunk_text) >= self.min_size:
                    chunks.append(self._create_chunk_metadata(
                        chunk_text, chunk_index, "sliding-window"
                    ))
                    chunk_index += 1
            
            # Move back by overlap amount
            if i < len(sentences) and self.overlap > 0:
                overlap_count = 0
                overlap_tokens = 0
                for j in range(len(current_chunk) - 1, -1, -1):
                    part_tokens = self.get_token_count(current_chunk[j])
                    if overlap_tokens + part_tokens <= self.overlap:
                        overlap_count += 1
                        overlap_tokens += part_tokens
                    else:
                        break
                
                i -= overlap_count
        
        return chunks
    
    def chunk(
        self, 
        text: str, 
        strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC
    ) -> List[Dict[str, Any]]:
        """
        Main chunking method - dispatches to appropriate strategy
        
        Args:
            text: Input text to chunk
            strategy: Chunking strategy to use
            
        Returns:
            List of chunk dictionaries with metadata
        """
        if not text or not text.strip():
            return []
        
        # Select strategy
        if strategy == ChunkingStrategy.SEMANTIC:
            chunks = self.chunk_semantic(text)
        elif strategy == ChunkingStrategy.RECURSIVE:
            chunks = self.chunk_recursive(text)
        elif strategy == ChunkingStrategy.SLIDING_WINDOW:
            chunks = self.chunk_sliding_window(text)
        elif strategy == ChunkingStrategy.SENTENCE:
            # Sentence-based is semantic with sentence-only splitting
            chunks = self.chunk_semantic(text)
        else:
            # Default to semantic
            chunks = self.chunk_semantic(text)
        
        # Log statistics if enabled
        if LOG_CHUNK_STATS and chunks:
            total_tokens = sum(c['tokens'] for c in chunks)
            avg_tokens = total_tokens / len(chunks)
            min_tokens = min(c['tokens'] for c in chunks)
            max_tokens = max(c['tokens'] for c in chunks)
            
            print(f"📊 Chunking stats ({strategy}):")
            print(f"   Total chunks: {len(chunks)}")
            print(f"   Total tokens: {total_tokens:,}")
            print(f"   Avg tokens/chunk: {avg_tokens:.1f}")
            print(f"   Min/Max tokens: {min_tokens}/{max_tokens}")
            print(f"   Input length: {len(text):,} chars")
        
        return chunks


def assess_text_quality(text: str) -> float:
    """
    Assess text quality (0-1 score)
    
    Checks:
    - Readable characters ratio
    - Sentence structure
    - Word length distribution
    - Special character density
    
    Returns quality score 0-1
    """
    if not text or len(text) < 10:
        return 0.0
    
    score = 1.0
    
    # Check readable character ratio
    readable_chars = sum(1 for c in text if c.isalnum() or c.isspace() or c in '.,!?;:-')
    readable_ratio = readable_chars / len(text)
    if readable_ratio < 0.7:
        score *= 0.5  # Penalize low readability
    
    # Check for excessive special characters
    special_chars = sum(1 for c in text if not (c.isalnum() or c.isspace() or c in '.,!?;:-()\'"'))
    special_ratio = special_chars / len(text)
    if special_ratio > 0.15:
        score *= 0.7  # Penalize excessive special chars
    
    # Check average word length (too short = gibberish, too long = codes/hashes)
    words = text.split()
    if words:
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len < 2 or avg_word_len > 15:
            score *= 0.8
    
    # Check sentence structure (at least some punctuation)
    has_periods = '.' in text or '!' in text or '?' in text
    if not has_periods and len(text) > 100:
        score *= 0.9  # Slight penalty for no punctuation
    
    return min(1.0, max(0.0, score))


def deduplicate_chunks(chunks: List[Dict[str, Any]], similarity_threshold: float = 0.95) -> List[Dict[str, Any]]:
    """
    Remove near-duplicate chunks to reduce redundancy
    
    Uses simple text similarity (Jaccard on word sets)
    For production, could use embedding similarity
    """
    if not chunks or len(chunks) <= 1:
        return chunks
    
    def jaccard_similarity(text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0
    
    unique_chunks = [chunks[0]]  # Keep first chunk
    
    for chunk in chunks[1:]:
        is_duplicate = False
        chunk_text = chunk['text']
        
        for existing in unique_chunks:
            similarity = jaccard_similarity(chunk_text, existing['text'])
            if similarity >= similarity_threshold:
                is_duplicate = True
                if RAG_DEBUG_LOGGING:
                    print(f"   Dedup: Removed chunk {chunk['index']} (similarity: {similarity:.2f})")
                break
        
        if not is_duplicate:
            unique_chunks.append(chunk)
    
    if RAG_DEBUG_LOGGING:
        print(f"   Deduplication: {len(chunks)} → {len(unique_chunks)} chunks")
    
    return unique_chunks


# ==================== PUBLIC API ====================

def chunk_text(
    text: str,
    strategy: str = ChunkingStrategy.SEMANTIC,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
    deduplicate: bool = True,
    quality_filter: bool = True,
    min_quality: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Main entry point for text chunking
    
    Args:
        text: Input text
        strategy: Chunking strategy (semantic, recursive, sliding, sentence)
        chunk_size: Override default chunk size
        overlap: Override default overlap
        deduplicate: Remove near-duplicate chunks
        quality_filter: Filter out low-quality chunks
        min_quality: Minimum quality score (0-1)
        
    Returns:
        List of chunk dictionaries
    """
    if not text or not text.strip():
        return []
    
    # Create chunker instance
    chunker = TextChunker(
        chunk_size=chunk_size or CHUNK_SIZE_TOKENS,
        overlap=overlap or CHUNK_OVERLAP_TOKENS,
        min_size=MIN_CHUNK_SIZE_TOKENS,
        max_size=MAX_CHUNK_SIZE_TOKENS,
    )
    
    # Chunk the text
    chunks = chunker.chunk(text, strategy=strategy)
    
    if not chunks:
        return []
    
    # Quality filtering
    if quality_filter:
        quality_chunks = []
        for chunk in chunks:
            quality_score = assess_text_quality(chunk['text'])
            chunk['quality_score'] = quality_score
            
            if quality_score >= min_quality:
                quality_chunks.append(chunk)
            elif RAG_DEBUG_LOGGING:
                print(f"   Quality filter: Rejected chunk {chunk['index']} (score: {quality_score:.2f})")
        
        chunks = quality_chunks
    
    # Deduplication
    if deduplicate and len(chunks) > 1:
        from config.rag_config import DEDUPLICATE_CHUNKS, DEDUP_SIMILARITY_THRESHOLD
        if DEDUPLICATE_CHUNKS:
            chunks = deduplicate_chunks(chunks, DEDUP_SIMILARITY_THRESHOLD)
    
    # Reindex after filtering
    for i, chunk in enumerate(chunks):
        chunk['final_index'] = i
    
    return chunks

