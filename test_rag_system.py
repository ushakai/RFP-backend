"""
Quick test script for the RAG system components
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*60)
print("Testing Production-Grade RAG System")
print("="*60)

# Test 1: Config
print("\n[1/5] Testing RAG Configuration...")
try:
    from config.rag_config import get_rag_config_summary, validate_rag_config
    
    validate_rag_config()
    config = get_rag_config_summary()
    print(f"✓ Config validated")
    print(f"  - Chunking strategy: {config['chunking']['strategy']}")
    print(f"  - Chunk size: {config['chunking']['chunk_size_tokens']} tokens")
    print(f"  - Embedding provider: {config['embedding']['provider']}")
    print(f"  - Embedding dimension: {config['embedding']['dimension']}")
    print(f"  - Dense/Sparse weights: {config['retrieval']['dense_weight']}/{config['retrieval']['sparse_weight']}")
except Exception as e:
    print(f"✗ Config test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Text Extraction
print("\n[2/5] Testing Text Extraction...")
try:
    from services.text_extraction_service import extract_text, get_extraction_capabilities
    
    caps = get_extraction_capabilities()
    print(f"✓ Extraction capabilities:")
    print(f"  - PDF: {caps['pdf']}")
    print(f"  - DOCX: {caps['docx']}")
    print(f"  - Max file size: {caps['max_file_size_mb']}MB")
    
    # Test with sample text
    test_content = b"This is a test document.\n\nIt has multiple paragraphs.\n\nFor testing."
    text, metadata = extract_text(test_content, "test.txt")
    print(f"✓ Extracted {len(text)} chars from test file")
except Exception as e:
    print(f"✗ Extraction test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Chunking
print("\n[3/5] Testing Chunking Service...")
try:
    from services.chunking_service_v2 import chunk_text
    
    test_text = """
    This is a test document for the RAG system.
    It contains multiple paragraphs to test chunking.
    
    The system should handle various document types.
    Each paragraph should be processed correctly.
    
    Quality assessment should validate the text.
    """
    
    chunks = chunk_text(test_text, strategy='semantic')
    print(f"✓ Created {len(chunks)} chunks")
    if chunks:
        avg_tokens = sum(c['tokens'] for c in chunks) / len(chunks)
        print(f"  - Average tokens per chunk: {avg_tokens:.1f}")
        print(f"  - First chunk: {chunks[0]['tokens']} tokens, {chunks[0]['char_count']} chars")
except Exception as e:
    print(f"✗ Chunking test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Embedding (requires API key)
print("\n[4/5] Testing Embedding Service...")
try:
    from services.embedding_service import get_embedding, get_embedding_stats
    from config.settings import GOOGLE_API_KEY
    
    if not GOOGLE_API_KEY:
        print("⚠ Skipping embedding test (no API key)")
    else:
        test_text = "This is a test for embeddings."
        embedding = get_embedding(test_text)
        print(f"✓ Generated embedding with {len(embedding)} dimensions")
        
        # Test cache
        embedding2 = get_embedding(test_text)
        is_cached = embedding == embedding2
        print(f"✓ Cache working: {is_cached}")
        
        stats = get_embedding_stats()
        print(f"  - Cache size: {stats['cache_stats']['size']}")
except Exception as e:
    print(f"✗ Embedding test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Document Ingestion Pipeline
print("\n[5/5] Testing Document Ingestion Pipeline...")
try:
    from services.document_ingestion_service import test_ingestion_pipeline
    
    test_ingestion_pipeline()
    print("✓ Pipeline implementation complete")
except Exception as e:
    print(f"✗ Pipeline test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("RAG System Tests Complete!")
print("="*60)

