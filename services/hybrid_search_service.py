"""
Hybrid search service combining dense (vector) and sparse (BM25) retrieval
For production-grade RAG with better precision/recall than vector-only search
"""
from typing import List, Dict, Any, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from config.settings import get_supabase_client
from services.gemini_service import get_embedding


def normalize_scores(scores: List[float]) -> List[float]:
    """Normalize scores to 0-1 range"""
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0] * len(scores)
    return [(s - min_score) / (max_score - min_score) for s in scores]


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    a = np.array(vec1)
    b = np.array(vec2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def hybrid_search_documents(
    query: str,
    client_id: str,
    rfp_id: Optional[str] = None,
    top_k: int = 5,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    min_score: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Perform hybrid search combining vector similarity and BM25.
    
    Args:
        query: Search query (question text)
        client_id: Client ID to filter documents
        rfp_id: Optional RFP ID to filter documents
        top_k: Number of results to return
        dense_weight: Weight for vector similarity (0-1)
        sparse_weight: Weight for BM25 score (0-1, should sum to 1 with dense_weight)
        min_score: Minimum combined score threshold
    
    Returns:
        List of documents with scores, sorted by relevance
    
    Why hybrid search?
    - Vector search: Good for semantic similarity, handles synonyms
    - BM25: Good for exact keyword matches, term frequency
    - Combined: Best of both worlds, higher precision and recall
    
    Weights:
    - 0.7/0.3 (dense/sparse) works well for general QA
    - Increase sparse weight for keyword-heavy queries
    - Increase dense weight for conceptual queries
    """
    supabase = get_supabase_client()
    
    # Fetch all documents for this client (with RFP filter if provided)
    query_builder = supabase.table("client_docs").select(
        "id, content_text, embedding, filename, chunk_index, metadata, created_at"
    ).eq("client_id", client_id)
    
    if rfp_id:
        query_builder = query_builder.eq("rfp_id", rfp_id)
    
    docs = query_builder.execute().data or []
    
    if not docs:
        return []
    
    # Get query embedding for dense search
    query_embedding = get_embedding(query)
    
    # Dense search: Vectorized cosine similarity using numpy
    embeddings = []
    valid_indices = []
    for i, doc in enumerate(docs):
        emb = doc.get("embedding")
        if emb and len(emb) == len(query_embedding):
            embeddings.append(emb)
            valid_indices.append(i)
    
    dense_scores = [0.0] * len(docs)
    if embeddings:
        emb_matrix = np.array(embeddings)
        q_vec = np.array(query_embedding)
        
        # Calculate cosine similarity: (A . B) / (||A|| * ||B||)
        dot_product = np.dot(emb_matrix, q_vec)
        norms = np.linalg.norm(emb_matrix, axis=1) * np.linalg.norm(q_vec)
        # Avoid division by zero
        sims = np.divide(dot_product, norms, out=np.zeros_like(dot_product), where=norms!=0)
        
        for idx, score in zip(valid_indices, sims):
            dense_scores[idx] = float(score)
    
    # Normalize dense scores
    dense_scores_norm = normalize_scores(dense_scores)
    
    # Sparse search: BM25
    corpus = [doc["content_text"] for doc in docs]
    tokenized_corpus = [text.lower().split() for text in corpus]
    
    try:
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.lower().split()
        sparse_scores = bm25.get_scores(tokenized_query)
        # Normalize sparse scores
        sparse_scores_norm = normalize_scores(sparse_scores.tolist())
    except Exception as e:
        print(f"WARN: BM25 failed: {e}")
        sparse_scores_norm = [0.0] * len(docs)
    
    # Combine scores
    combined_scores = [
        (dense_weight * d + sparse_weight * s)
        for d, s in zip(dense_scores_norm, sparse_scores_norm)
    ]
    
    # Create results with scores
    results = []
    for i, doc in enumerate(docs):
        score = combined_scores[i]
        if score >= min_score:
            results.append({
                "id": doc["id"],
                "content": doc["content_text"],
                "filename": doc["filename"],
                "chunk_index": doc["chunk_index"],
                "metadata": doc.get("metadata", {}),
                "created_at": doc.get("created_at"),
                "score": score,
                "dense_score": dense_scores_norm[i],
                "sparse_score": sparse_scores_norm[i],
            })
    
    # Sort by combined score
    results.sort(key=lambda x: x["score"], reverse=True)
    
    return results[:top_k]


def hybrid_search_qa(
    query: str,
    client_id: str,
    rfp_id: Optional[str] = None,
    top_k: int = 5,
    min_score: float = 0.4
) -> List[Dict[str, Any]]:
    """
    Hybrid search for Q&A pairs (searches questions + answers)
    
    Returns matching Q&A pairs with their scores
    """
    supabase = get_supabase_client()
    
    # Fetch questions with embeddings
    q_query = supabase.table("client_questions").select(
        "id, original_text, embedding, category, rfp_id, created_at"
    ).eq("client_id", client_id)
    
    if rfp_id:
        q_query = q_query.eq("rfp_id", rfp_id)
    
    questions = q_query.execute().data or []
    
    if not questions:
        return []
    
    # Get question IDs to fetch answers
    q_ids = [q["id"] for q in questions]
    mappings = supabase.table("client_question_answer_mappings").select(
        "question_id, answer_id"
    ).in_("question_id", q_ids).execute().data or []
    
    q_to_a = {m["question_id"]: m["answer_id"] for m in mappings}
    a_ids = list({aid for aid in q_to_a.values() if aid})
    
    answers = {}
    if a_ids:
        a_rows = supabase.table("client_answers").select(
            "id, answer_text"
        ).in_("id", a_ids).execute().data or []
        answers = {a["id"]: a["answer_text"] for a in a_rows}
    
    # Get query embedding
    query_embedding = get_embedding(query)
    
    # Dense search on questions
    dense_scores = []
    for q in questions:
        emb = q.get("embedding")
        if emb:
            sim = cosine_similarity(query_embedding, emb)
            dense_scores.append(sim)
        else:
            dense_scores.append(0.0)
    
    dense_scores_norm = normalize_scores(dense_scores)
    
    # Sparse search on questions (BM25)
    corpus = [q["original_text"] for q in questions]
    tokenized_corpus = [text.lower().split() for text in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split()
    sparse_scores = bm25.get_scores(tokenized_query)
    sparse_scores_norm = normalize_scores(sparse_scores.tolist())
    
    # Combine scores (70% dense, 30% sparse)
    combined_scores = [
        (0.7 * d + 0.3 * s)
        for d, s in zip(dense_scores_norm, sparse_scores_norm)
    ]
    
    # Create results
    results = []
    for i, q in enumerate(questions):
        score = combined_scores[i]
        if score >= min_score:
            a_id = q_to_a.get(q["id"])
            results.append({
                "question_id": q["id"],
                "question": q["original_text"],
                "answer": answers.get(a_id, ""),
                "category": q.get("category"),
                "rfp_id": q.get("rfp_id"),
                "score": score,
                "dense_score": dense_scores_norm[i],
                "sparse_score": sparse_scores_norm[i],
            })
    
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def search_knowledge_base(
    query: str,
    client_id: str,
    rfp_id: Optional[str] = None,
    top_k: int = 10,
    include_docs: bool = True,
    include_qa: bool = True
) -> Dict[str, Any]:
    """
    Search entire knowledge base (both documents and Q&A pairs)
    Returns combined results from both sources
    
    This is the main entry point for RAG retrieval
    """
    results = {
        "query": query,
        "documents": [],
        "qa_pairs": [],
        "combined_context": ""
    }
    
    if include_docs:
        results["documents"] = hybrid_search_documents(
            query=query,
            client_id=client_id,
            rfp_id=rfp_id,
            top_k=top_k // 2 if include_qa else top_k
        )
    
    if include_qa:
        results["qa_pairs"] = hybrid_search_qa(
            query=query,
            client_id=client_id,
            rfp_id=rfp_id,
            top_k=top_k // 2 if include_docs else top_k
        )
    
    # Build combined context for LLM
    context_parts = []
    
    # Add Q&A context
    for i, qa in enumerate(results["qa_pairs"], 1):
        context_parts.append(
            f"[Q&A {i}] Q: {qa['question']}\nA: {qa['answer']}\n"
        )
    
    # Add document context
    for i, doc in enumerate(results["documents"], 1):
        context_parts.append(
            f"[DOC {i} - {doc['filename']}] {doc['content']}\n"
        )
    
    results["combined_context"] = "\n".join(context_parts)
    
    return results

