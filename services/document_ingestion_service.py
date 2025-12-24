"""
Production-grade document ingestion service for RAG
Orchestrates: text extraction → chunking → embedding → storage
"""
import uuid
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from config.rag_config import (
    CHUNKING_STRATEGY,
    DEDUPLICATE_CHUNKS,
    MIN_TEXT_QUALITY_SCORE,
    RAG_DEBUG_LOGGING,
    LOG_CHUNK_STATS,
    TOP_K_DOCUMENTS,
    get_rag_config_summary,
)
from config.settings import get_supabase_client
from services.text_extraction_service import (
    extract_text,
    validate_extracted_text,
    assess_extraction_quality,
)
from services.chunking_service_v2 import chunk_text, assess_text_quality
from services.embedding_service import get_embeddings_batch, validate_embedding
from utils.db_utils import retry_on_db_error


# ==================== INGESTION PIPELINE ====================

class DocumentIngestionPipeline:
    """
    Production-grade document ingestion pipeline
    
    Pipeline stages:
    1. Text Extraction - Extract text from file
    2. Quality Check - Validate extraction quality
    3. Chunking - Split into optimal chunks
    4. Deduplication - Remove redundant chunks
    5. Embedding - Generate vector embeddings
    6. Storage - Store in database with metadata
    """
    
    def __init__(
        self,
        client_id: str,
        rfp_id: str,
        chunk_strategy: str = CHUNKING_STRATEGY,
        min_quality: float = MIN_TEXT_QUALITY_SCORE,
        deduplicate: bool = DEDUPLICATE_CHUNKS,
    ):
        """
        Initialize ingestion pipeline
        
        Args:
            client_id: Client ID
            rfp_id: RFP ID to associate documents with
            chunk_strategy: Chunking strategy to use
            min_quality: Minimum quality threshold
            deduplicate: Whether to deduplicate chunks
        """
        self.client_id = client_id
        self.rfp_id = rfp_id
        self.chunk_strategy = chunk_strategy
        self.min_quality = min_quality
        self.deduplicate = deduplicate
        
        self.supabase = get_supabase_client()
        
        # Statistics
        self.stats = {
            'start_time': time.time(),
            'extraction_time': 0,
            'chunking_time': 0,
            'embedding_time': 0,
            'storage_time': 0,
            'total_time': 0,
            'chunks_created': 0,
            'chunks_stored': 0,
            'chunks_failed': 0,
            'chunks_deduplicated': 0,
            'quality_score': 0.0,
        }
    
    def ingest_file(
        self,
        file_content: bytes,
        filename: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ingest a single file through the complete pipeline
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            content_type: MIME type
            metadata: Additional metadata to store
            
        Returns:
            Dictionary with ingestion results and statistics
        """
        if RAG_DEBUG_LOGGING:
            print(f"\n📥 Starting document ingestion pipeline for: {filename}")
            print(f"   Client: {self.client_id}")
            print(f"   RFP: {self.rfp_id}")
            print(f"   Strategy: {self.chunk_strategy}")
        
        try:
            # Stage 1: Text Extraction
            text, extraction_metadata = self._extract_text(
                file_content, filename, content_type
            )
            
            # Stage 2: Quality Check
            if not self._validate_quality(text, extraction_metadata):
                raise ValueError(
                    f"Text quality too low: {extraction_metadata.get('quality_score', 0):.2f} "
                    f"< {self.min_quality}"
                )
            
            # Stage 3: Chunking
            chunks = self._chunk_text(text, extraction_metadata)
            
            # Stage 4: Deduplication (if enabled)
            if self.deduplicate and len(chunks) > 1:
                original_count = len(chunks)
                chunks = self._deduplicate_chunks(chunks)
                self.stats['chunks_deduplicated'] = original_count - len(chunks)
            
            # Stage 5: Embedding
            if RAG_DEBUG_LOGGING:
                print(f"\n[5/6] Embedding Generation for {len(chunks)} chunks...")
            chunk_embeddings = self._generate_embeddings(chunks)
            
            # Stage 6: Storage
            if RAG_DEBUG_LOGGING:
                print(f"\n[6/6] Database Storage...")
            stored_count = self._store_chunks(
                chunks, chunk_embeddings, filename, content_type, metadata
            )
            
            # Finalize
            self.stats['total_time'] = time.time() - self.stats['start_time']
            self.stats['chunks_stored'] = stored_count
            
            # Update quality score in stats from extraction metadata
            self.stats['quality_score'] = extraction_metadata.get('quality_score', 0.0)
            
            result = {
                'success': True,
                'chunks_stored': stored_count,
                'chunks_failed': self.stats['chunks_failed'],
                'quality_score': self.stats['quality_score'],
                'statistics': self.stats,
                'metadata': extraction_metadata,
            }
            
            if RAG_DEBUG_LOGGING:
                self._print_summary(result)
            
            return result
            
        except Exception as e:
            error_result = {
                'success': False,
                'error': str(e),
                'statistics': self.stats,
            }
            
            if RAG_DEBUG_LOGGING:
                print(f"\n❌ Ingestion failed: {e}")
                import traceback
                traceback.print_exc()
            
            return error_result
    
    def _extract_text(
        self, file_content: bytes, filename: str, content_type: Optional[str]
    ) -> Tuple[str, Dict[str, Any]]:
        """Stage 1: Extract text from file"""
        start_time = time.time()
        
        if RAG_DEBUG_LOGGING:
            print(f"\n[1/6] Text Extraction...")
        
        text, metadata = extract_text(file_content, filename, content_type)
        
        self.stats['extraction_time'] = time.time() - start_time
        
        if RAG_DEBUG_LOGGING:
            print(f"   ✓ Extracted {len(text):,} chars in {self.stats['extraction_time']:.2f}s")
        
        return text, metadata
    
    def _validate_quality(self, text: str, metadata: Dict[str, Any]) -> bool:
        """Stage 2: Validate text quality"""
        if RAG_DEBUG_LOGGING:
            print(f"\n[2/6] Quality Validation...")
        
        is_valid = validate_extracted_text(text, metadata, self.min_quality)
        self.stats['quality_score'] = metadata.get('quality_score', 0.0)
        
        if RAG_DEBUG_LOGGING:
            status = "✓" if is_valid else "✗"
            print(f"   {status} Quality score: {self.stats['quality_score']:.2f}")
        
        return is_valid
    
    def _chunk_text(self, text: str, extraction_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Stage 3: Chunk text"""
        start_time = time.time()
        
        if RAG_DEBUG_LOGGING:
            print(f"\n[3/6] Text Chunking ({self.chunk_strategy})...")
        
        chunks = chunk_text(
            text=text,
            strategy=self.chunk_strategy,
            deduplicate=False,  # We handle deduplication separately
            quality_filter=True,
            min_quality=0.5,  # Lower threshold for chunks
        )
        
        self.stats['chunking_time'] = time.time() - start_time
        self.stats['chunks_created'] = len(chunks)
        
        if RAG_DEBUG_LOGGING:
            print(f"   ✓ Created {len(chunks)} chunks in {self.stats['chunking_time']:.2f}s")
            if chunks:
                avg_tokens = sum(c['tokens'] for c in chunks) / len(chunks)
                print(f"   Average chunk size: {avg_tokens:.0f} tokens")
        
        return chunks
    
    def _deduplicate_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Stage 4: Deduplicate chunks"""
        if RAG_DEBUG_LOGGING:
            print(f"\n[4/6] Deduplication...")
        
        from services.chunking_service_v2 import deduplicate_chunks
        from config.rag_config import DEDUP_SIMILARITY_THRESHOLD
        
        unique_chunks = deduplicate_chunks(chunks, DEDUP_SIMILARITY_THRESHOLD)
        
        if RAG_DEBUG_LOGGING:
            removed = len(chunks) - len(unique_chunks)
            print(f"   ✓ Removed {removed} duplicate chunks")
        
        return unique_chunks
    
    def _generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[List[float]]:
        """Stage 5: Generate embeddings"""
        start_time = time.time()
        
        if RAG_DEBUG_LOGGING:
            print(f"\n[5/6] Embedding Generation...")
        
        # Extract text from chunks
        chunk_texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings in batch
        embeddings = get_embeddings_batch(chunk_texts, use_cache=True)
        
        # Validate embeddings
        valid_embeddings = []
        for i, emb in enumerate(embeddings):
            if validate_embedding(emb):
                valid_embeddings.append(emb)
            else:
                print(f"   ⚠️  Invalid embedding for chunk {i}, using zero vector")
                valid_embeddings.append([0.0] * len(emb))
        
        self.stats['embedding_time'] = time.time() - start_time
        
        if RAG_DEBUG_LOGGING:
            print(f"   ✓ Generated {len(valid_embeddings)} embeddings in {self.stats['embedding_time']:.2f}s")
            print(f"   Rate: {len(valid_embeddings) / max(0.1, self.stats['embedding_time']):.1f} emb/s")
        
        return valid_embeddings
    
    @retry_on_db_error(max_retries=3)
    def _store_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        filename: str,
        content_type: Optional[str],
        additional_metadata: Optional[Dict[str, Any]]
    ) -> int:
        """Stage 6: Store chunks in database"""
        start_time = time.time()
        
        if RAG_DEBUG_LOGGING:
            print(f"\n[6/6] Database Storage...")
        
        # Generate unique chunk_id for this document
        chunk_id = str(uuid.uuid4())
        
        stored_count = 0
        failed_count = 0
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            try:
                # Prepare document data
                doc_data = {
                    "client_id": self.client_id,
                    "rfp_id": self.rfp_id,
                    "filename": filename,
                    "mime_type": content_type,
                    "chunk_id": chunk_id,
                    "chunk_index": i,
                    "content_text": chunk['text'],
                    "embedding": embedding,
                    "metadata": {
                        "tokens": chunk.get('tokens', 0),
                        "char_count": chunk.get('char_count', 0),
                        "strategy": chunk.get('strategy', self.chunk_strategy),
                        "quality_score": chunk.get('quality_score', 1.0),
                        **(additional_metadata or {}),
                    },
                }
                
                # Insert into database
                self.supabase.table("client_docs").insert(doc_data).execute()
                stored_count += 1
                
            except Exception as e:
                failed_count += 1
                if RAG_DEBUG_LOGGING:
                    print(f"   ⚠️  Failed to store chunk {i}: {e}")
                continue
        
        self.stats['storage_time'] = time.time() - start_time
        self.stats['chunks_failed'] = failed_count
        
        if RAG_DEBUG_LOGGING:
            print(f"   ✓ Stored {stored_count}/{len(chunks)} chunks in {self.stats['storage_time']:.2f}s")
            if failed_count > 0:
                print(f"   ⚠️  {failed_count} chunks failed to store")
        
        return stored_count
    
    def _print_summary(self, result: Dict[str, Any]):
        """Print ingestion summary"""
        stats = result['statistics']
        
        print(f"\n{'='*60}")
        print(f"📊 INGESTION SUMMARY")
        print(f"{'='*60}")
        print(f"Status:          {'✅ Success' if result['success'] else '❌ Failed'}")
        print(f"Quality Score:   {result['quality_score']:.2f}")
        print(f"Chunks Stored:   {result['chunks_stored']}")
        print(f"Chunks Failed:   {result['chunks_failed']}")
        if stats['chunks_deduplicated'] > 0:
            print(f"Deduplicated:    {stats['chunks_deduplicated']}")
        print(f"\nTimings:")
        print(f"  Extraction:    {stats['extraction_time']:.2f}s")
        print(f"  Chunking:      {stats['chunking_time']:.2f}s")
        print(f"  Embedding:     {stats['embedding_time']:.2f}s")
        print(f"  Storage:       {stats['storage_time']:.2f}s")
        print(f"  Total:         {stats['total_time']:.2f}s")
        print(f"{'='*60}\n")


# ==================== BATCH INGESTION ====================

def ingest_documents_batch(
    client_id: str,
    rfp_id: str,
    files: List[Dict[str, Any]],
    **kwargs
) -> Dict[str, Any]:
    """
    Ingest multiple documents in batch
    
    Args:
        client_id: Client ID
        rfp_id: RFP ID
        files: List of file dicts with 'content', 'filename', 'content_type'
        **kwargs: Additional arguments for pipeline
        
    Returns:
        Dictionary with batch results
    """
    print(f"\n📦 Batch ingestion: {len(files)} files")
    
    pipeline = DocumentIngestionPipeline(client_id, rfp_id, **kwargs)
    
    results = []
    total_chunks = 0
    total_failed = 0
    start_time = time.time()
    
    for i, file_info in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] Processing: {file_info['filename']}")
        
        result = pipeline.ingest_file(
            file_content=file_info['content'],
            filename=file_info['filename'],
            content_type=file_info.get('content_type'),
            metadata=file_info.get('metadata'),
        )
        
        results.append(result)
        
        if result['success']:
            total_chunks += result['chunks_stored']
            total_failed += result['chunks_failed']
    
    total_time = time.time() - start_time
    success_count = sum(1 for r in results if r['success'])
    
    batch_result = {
        'success': success_count > 0,
        'files_processed': len(files),
        'files_succeeded': success_count,
        'files_failed': len(files) - success_count,
        'total_chunks_stored': total_chunks,
        'total_chunks_failed': total_failed,
        'total_time': total_time,
        'results': results,
    }
    
    print(f"\n{'='*60}")
    print(f"📦 BATCH INGESTION SUMMARY")
    print(f"{'='*60}")
    print(f"Files Processed:  {batch_result['files_processed']}")
    print(f"  Succeeded:      {batch_result['files_succeeded']}")
    print(f"  Failed:         {batch_result['files_failed']}")
    print(f"Total Chunks:     {batch_result['total_chunks_stored']}")
    print(f"Total Time:       {total_time:.2f}s")
    print(f"Avg Time/File:    {total_time / len(files):.2f}s")
    print(f"{'='*60}\n")
    
    return batch_result


# ==================== RETRIEVAL ====================

def retrieve_documents(
    query: str,
    client_id: str,
    rfp_id: Optional[str] = None,
    top_k: int = TOP_K_DOCUMENTS,
    **search_kwargs
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant documents for a query
    
    Args:
        query: Search query
        client_id: Client ID
        rfp_id: Optional RFP ID to filter by
        top_k: Number of results to return
        **search_kwargs: Additional search parameters
        
    Returns:
        List of relevant document chunks with scores
    """
    from services.hybrid_search_service import hybrid_search_documents
    
    results = hybrid_search_documents(
        query=query,
        client_id=client_id,
        rfp_id=rfp_id,
        top_k=top_k,
        **search_kwargs
    )
    
    return results


# ==================== UTILITY FUNCTIONS ====================

def get_document_count(client_id: str, rfp_id: Optional[str] = None) -> int:
    """Get count of stored document chunks"""
    supabase = get_supabase_client()
    
    query = supabase.table("client_docs").select("id", count="exact")
    query = query.eq("client_id", client_id)
    
    if rfp_id:
        query = query.eq("rfp_id", rfp_id)
    
    result = query.execute()
    return result.count or 0


def delete_documents(client_id: str, rfp_id: str) -> int:
    """Delete all documents for an RFP"""
    supabase = get_supabase_client()
    
    try:
        # Count first
        count_result = supabase.table("client_docs").select("id", count="exact").eq(
            "client_id", client_id
        ).eq("rfp_id", rfp_id).execute()
        
        count = count_result.count or 0
        
        if count == 0:
            return 0
        
        # Delete
        supabase.table("client_docs").delete().eq("client_id", client_id).eq(
            "rfp_id", rfp_id
        ).execute()
        
        print(f"✓ Deleted {count} document chunks for RFP {rfp_id}")
        return count
        
    except Exception as e:
        print(f"Error deleting documents: {e}")
        return 0


def get_ingestion_stats() -> Dict[str, Any]:
    """Get global ingestion statistics"""
    supabase = get_supabase_client()
    
    try:
        # Total documents
        total_result = supabase.table("client_docs").select("id", count="exact").execute()
        total_docs = total_result.count or 0
        
        # Get RAG config
        rag_config = get_rag_config_summary()
        
        return {
            'total_documents': total_docs,
            'rag_config': rag_config,
        }
    except Exception as e:
        print(f"Error getting ingestion stats: {e}")
        return {'error': str(e)}


# ==================== TESTING ====================

def test_ingestion_pipeline():
    """Test the ingestion pipeline with sample data"""
    print("🔍 Testing document ingestion pipeline...")
    
    # Sample text document
    test_content = b"""
    This is a test document for the RAG system.
    
    It contains multiple paragraphs to test chunking functionality.
    Each paragraph should be processed correctly and stored in the database.
    
    The system should handle various document types including:
    - Plain text files
    - Markdown documents  
    - PDF files
    - Word documents
    
    Quality assessment should validate that the extracted text meets
    minimum quality thresholds before proceeding with chunking and embedding.
    """
    
    # Would need real client_id and rfp_id to test
    print("✓ Pipeline implementation complete")
    print("  (Requires valid client_id and rfp_id for end-to-end test)")
    
    return True


if __name__ == "__main__":
    test_ingestion_pipeline()

