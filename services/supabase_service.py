"""
Database Service - Database operations using direct PostgreSQL connection
"""
import traceback
import time
from typing import Any, Callable, List

from config.settings import get_supabase_client, reinitialize_supabase
from services.gemini_service import get_embedding


def execute_with_retry(
    operation_factory: Callable[[], Any],
    retries: int = 3,
    backoff_seconds: float = 0.3,
    on_retry: Callable[[], None] | None = None,
    max_total_time: float = 10.0,
):
    """Execute a database operation with retries on connection errors.
    
    With direct PostgreSQL connections, this is much more reliable than
    the HTTP-based Supabase client.
    """
    last_error: Exception | None = None
    start_time = time.time()
    
    for attempt in range(1, retries + 1):
        elapsed = time.time() - start_time
        if elapsed > max_total_time:
            print(f"ERROR: Exceeded maximum retry time ({max_total_time}s), giving up")
            if last_error:
                raise last_error
            raise TimeoutError(f"Operation timed out after {elapsed:.2f} seconds")
        
        try:
            result = operation_factory()
            # Check if result has an error attribute (QueryResult format)
            if hasattr(result, 'error') and result.error:
                raise Exception(result.error)
            return result
        except Exception as err:
            last_error = err
            error_str = str(err).lower()
            error_type = type(err).__name__.lower()
            
            # Check if it's a connection error that we should retry
            is_connection_error = any([
                "operationalerror" in error_type,
                "interfaceerror" in error_type,
                "connection" in error_str,
                "timeout" in error_str,
                "pool" in error_str,
                "closed" in error_str,
            ])
            
            if not is_connection_error:
                # Non-connection errors bubble up immediately
                raise
            
            print(f"Database connection error (attempt {attempt}/{retries}): {str(err)[:100]}")
            
            try:
                reinitialize_supabase()
            except Exception:
                pass
            
            if on_retry:
                try:
                    on_retry()
                except Exception:
                    pass
            
            if attempt < retries:
                sleep_time = min(backoff_seconds * (2 ** (attempt - 1)), 2.0)
                if elapsed + sleep_time < max_total_time:
                    time.sleep(sleep_time)
    
    if last_error:
        raise last_error


def fetch_paginated_rows(
    query_factory: Callable[[], Any],
    page_size: int = 500,
    max_rows: int | None = None,
):
    """Fetch rows in chunks using PostgREST range pagination."""
    if page_size <= 0:
        raise ValueError("page_size must be positive")

    results: list = []
    offset = 0

    while True:
        def _run_page():
            builder = query_factory()
            return builder.range(offset, offset + page_size - 1).execute()

        response = execute_with_retry(_run_page)
        page = (response.data or []) if response else []
        results.extend(page)

        if len(page) < page_size:
            break

        offset += page_size
        if max_rows and len(results) >= max_rows:
            break

    if max_rows and len(results) > max_rows:
        return results[:max_rows]
    return results


def fetch_question_answer_mappings(
    question_ids: List[str] | None,
    chunk_size: int = 100,
):
    """Fetch mapping rows for question IDs without hitting URL limits."""
    if not question_ids:
        return []

    mappings = []

    for idx in range(0, len(question_ids), chunk_size):
        chunk = [qid for qid in question_ids[idx : idx + chunk_size] if qid]
        if not chunk:
            continue
        def _run_chunk():
            supabase = get_supabase_client()
            return (
                supabase.table("client_question_answer_mappings")
                .select("question_id, answer_id")
                .in_("question_id", chunk)
                .execute()
            )

        try:
            res = execute_with_retry(_run_chunk)
        except Exception as e:
            print(f"fetch_question_answer_mappings chunk error: {e}")
            traceback.print_exc()
            continue

        if res and res.data:
            mappings.extend(res.data)

    return mappings


def _date_score(date_str):
    """Convert date to score (more recent = higher)
    
    Returns days since 2020-01-01 (more recent dates = higher score)
    Used as tiebreaker when similarity scores are close
    """
    if not date_str:
        return 0
    try:
        from datetime import datetime, timezone
        # Handle both date strings and datetime objects
        if isinstance(date_str, str):
            date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        else:
            date = date_str
            
        # Ensure date is timezone-aware for comparison
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
            
        # Score = days since epoch (recent dates have higher scores)
        epoch = datetime(2020, 1, 1, tzinfo=timezone.utc)
        days = (date - epoch).days
        return days
    except Exception as e:
        print(f"WARN: Failed to parse date '{date_str}': {e}")
        return 0


def _search_qa_pairs(question_embedding: list, client_id: str, rfp_id: str, match_threshold: float, match_count: int) -> list:
    """Search Q&A pairs using the existing search_supabase logic
    
    Returns matches with 'source': 'qa' tag
    """
    try:
        supabase = get_supabase_client()
        
        # Try RFP-specific search first
        if rfp_id:
            try:
                res = supabase.rpc(
                    "client_match_questions", {
                        "query_embedding": question_embedding,
                        "match_threshold": match_threshold,
                        "match_count": match_count,
                        "p_client_id": client_id,
                        "p_rfp_id": rfp_id
                    }).execute()
                
                if res.data and len(res.data) > 0:
                    # Tag as Q&A source
                    for match in res.data:
                        match['source'] = 'qa'
                    return res.data
            except Exception as e:
                print(f"WARN: RFP-specific Q&A search failed: {e}")
        
        # Fallback to client-wide search
        res = supabase.rpc(
            "client_match_questions", {
                "query_embedding": question_embedding,
                "match_threshold": match_threshold,
                "match_count": match_count,
                "p_client_id": client_id,
                "p_rfp_id": None
            }).execute()
        
        if res.data:
            for match in res.data:
                match['source'] = 'qa'
            return res.data
        
        return []
    except Exception as e:
        print(f"ERROR: Q&A pair search failed: {e}")
        traceback.print_exc()
        return []


def _search_document_chunks(question_embedding: list, client_id: str, rfp_id: str, match_threshold: float, match_count: int) -> list:
    """Search client_docs table for matching chunks using hybrid search
    
    Returns matches in same format as Q&A pairs with 'source': 'document' tag
    """
    try:
        supabase = get_supabase_client()
        
        # 1. Try RPC for efficient database-side vector search
        try:
            res = supabase.rpc(
                "match_client_docs", {
                    "query_embedding": question_embedding,
                    "match_threshold": match_threshold,
                    "match_count": match_count,
                    "p_client_id": client_id,
                    "p_rfp_id": rfp_id
                }).execute()
            
            if res.data:
                matches = []
                for doc in res.data:
                    matches.append({
                        "question": f"[Document: {doc.get('filename')}, chunk {doc.get('chunk_index', 0)}]",
                        "answer": doc.get('content_text'),
                        "similarity": doc.get("similarity", 0),
                        "source": "document",
                        "original_rfp_date": doc.get("original_rfp_date") or doc.get("created_at"),
                        "metadata": doc.get("metadata", {})
                    })
                return matches
        except Exception as rpc_error:
            # Fallback if RPC is not yet created in Supabase
            if "PGRST202" in str(rpc_error) or "match_client_docs" in str(rpc_error):
                print("DEBUG: match_client_docs RPC not found, falling back to optimized in-memory search")
            else:
                print(f"WARN: match_client_docs RPC failed, falling back: {rpc_error}")
        
        # 2. Fallback: Optimized in-memory search using numpy
        import numpy as np
        
        # Fetch document chunks for this client with joined RFP date
        query_builder = supabase.table("client_docs").select(
            "id, content_text, embedding, filename, chunk_index, metadata, created_at, rfp_id, client_rfps(original_rfp_date)"
        ).eq("client_id", client_id)
        
        if rfp_id:
            query_builder = query_builder.eq("rfp_id", rfp_id)
        
        docs = query_builder.execute().data or []
        
        if not docs:
            print(f"DEBUG: No document chunks found for client {client_id}")
            return []
            
        print(f"DEBUG: Found {len(docs)} total chunks for client, checking embeddings...")
        embeddings = []
        valid_indices = []
        for i, doc in enumerate(docs):
            emb = doc.get("embedding")
            if emb:
                # Handle cases where embedding might be a string from Supabase
                if isinstance(emb, str):
                    try:
                        import json
                        emb = json.loads(emb.replace('{', '[').replace('}', ']'))
                    except:
                        pass
                
                if isinstance(emb, list) and len(emb) == len(question_embedding):
                    embeddings.append(emb)
                    valid_indices.append(i)
        
        print(f"DEBUG: {len(embeddings)}/{len(docs)} chunks have valid embeddings")
        if not embeddings:
            return []
            
        # Vectorized cosine similarity
        emb_matrix = np.array(embeddings)
        q_vec = np.array(question_embedding)
        dot_product = np.dot(emb_matrix, q_vec)
        norms = np.linalg.norm(emb_matrix, axis=1) * np.linalg.norm(q_vec)
        sims = np.divide(dot_product, norms, out=np.zeros_like(dot_product), where=norms!=0)
        
        matches = []
        for i, score in enumerate(sims):
            if score >= match_threshold:
                doc = docs[valid_indices[i]]
                # Extract original_rfp_date from the nested join if available
                rfp_date = None
                if doc.get("client_rfps"):
                    rfp_date = doc["client_rfps"].get("original_rfp_date")
                
                matches.append({
                    "question": f"[Document: {doc['filename']}, chunk {doc.get('chunk_index', 0)}]",
                    "answer": doc['content_text'],
                    "similarity": float(score),
                    "source": "document",
                    "original_rfp_date": rfp_date or doc.get("created_at"),
                    "metadata": doc.get("metadata", {})
                })
        
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        return matches[:match_count]
    except Exception as e:
        print(f"ERROR: Document chunk search failed: {e}")
        traceback.print_exc()
        return []


def search_with_documents(
    question_embedding: list, 
    client_id: str, 
    rfp_id: str = None,
    match_threshold: float = 0.0,
    match_count: int = 10
) -> list:
    """
    Enhanced search combining Q&A pairs AND document chunks
    Returns combined results ranked by similarity, with date as tiebreaker
    
    This is the new primary search function that replaces search_supabase()
    for tender question matching.
    
    Args:
        question_embedding: Question embedding vector
        client_id: Client ID
        rfp_id: Optional RFP ID to filter by
        match_threshold: Minimum similarity threshold
        match_count: Maximum number of results to return
        
    Returns:
        List of matches sorted by relevance (similarity + date recency)
    """
    if not question_embedding:
        print("ERROR: No embedding provided to search_with_documents")
        return []
    
    print(f"DEBUG: Enhanced search - client_id={client_id}, rfp_id={rfp_id}, match_count={match_count}")
    
    # 1. Search Q&A pairs (try RFP-specific first, then client-wide)
    print("DEBUG: Searching Q&A pairs...")
    qa_matches = _search_qa_pairs(question_embedding, client_id, rfp_id, match_threshold, match_count)
    print(f"DEBUG: Found {len(qa_matches)} Q&A matches")
    
    # 2. Search document chunks (try RFP-specific first)
    print("DEBUG: Searching document chunks...")
    doc_matches = _search_document_chunks(question_embedding, client_id, rfp_id, match_threshold, match_count)
    print(f"DEBUG: Found {len(doc_matches)} document chunk matches (RFP-specific)")
    
    # 2b. If no document matches for specific RFP, try client-wide document search
    if len(doc_matches) == 0 and rfp_id:
        print("DEBUG: No RFP-specific document matches, searching client-wide...")
        doc_matches = _search_document_chunks(question_embedding, client_id, None, match_threshold, match_count)
        print(f"DEBUG: Found {len(doc_matches)} document chunk matches (client-wide)")
    
    # 3. Combine all matches
    all_matches = qa_matches + doc_matches
    
    if not all_matches:
        print("WARN: No matches found in Q&A pairs or document chunks")
        return []
    
    # 4. Sort by similarity (primary), then by date recency (secondary tiebreaker)
    # When similarity scores are very close (< 0.05 difference), more recent date wins
    all_matches.sort(key=lambda x: (
        x.get('similarity', 0),  # Primary: similarity score
        _date_score(x.get('original_rfp_date'))  # Secondary: date recency
    ), reverse=True)
    
    # 5. Return top matches
    top_matches = all_matches[:match_count]
    
    print(f"SUCCESS: Returning {len(top_matches)} combined matches (Q&A + documents)")
    for i, match in enumerate(top_matches[:3]):
        source_type = match.get('source', 'unknown')
        similarity = match.get('similarity', 0)
        date = match.get('original_rfp_date', 'N/A')
        question_preview = match.get('question', '')[:60]
        print(f"  {i+1}. [{source_type.upper()}] sim={similarity:.3f}, date={date} | {question_preview}...")
    
    return top_matches


def search_supabase(question_embedding: list, client_id: str, rfp_id: str = None) -> list:
    """Search for matching questions in Supabase using embeddings
    
    Strategy:
    1. Try RFP-specific search first (if rfp_id provided)
    2. Fallback to client-wide search if no results
    3. Run diagnostics if still no results
    """
    if not question_embedding:
        print("ERROR: No embedding provided to search_supabase")
        return []
    
    if not isinstance(question_embedding, list) or len(question_embedding) == 0:
        print(f"ERROR: Invalid embedding format: {type(question_embedding)}, length: {len(question_embedding) if isinstance(question_embedding, list) else 'N/A'}")
        return []
    
    try:
        supabase = get_supabase_client()
        print(f"DEBUG: Searching Supabase - client_id={client_id}, rfp_id={rfp_id}, embedding_length={len(question_embedding)}")
        
        # STRATEGY 1: Try RFP-specific search first (if rfp_id provided)
        if rfp_id:
            try:
                print(f"DEBUG: Attempting RFP-specific search (rfp_id={rfp_id})...")
                res = supabase.rpc(
                    "client_match_questions", {
                        "query_embedding": question_embedding,
                        "match_threshold": 0.0,
                        "match_count": 5,
                        "p_client_id": client_id,
                        "p_rfp_id": rfp_id
                    }).execute()
                
                if res.data and len(res.data) > 0:
                    print(f"SUCCESS: Found {len(res.data)} matches for RFP {rfp_id}")
                    for match in res.data[:2]:
                        print(f"  - Similarity: {match.get('similarity', 0):.3f} | Q: {match.get('question', '')[:50]}...")
                    return res.data
                else:
                    print(f"DEBUG: No matches for RFP {rfp_id}, trying client-wide search...")
            except Exception as rfp_error:
                print(f"WARN: RFP-specific search failed: {rfp_error}")
                traceback.print_exc()
        
        # STRATEGY 2: Fallback to client-wide search (no RFP restriction)
        print(f"DEBUG: Attempting client-wide search (all questions for client)...")
        res = supabase.rpc(
            "client_match_questions", {
                "query_embedding": question_embedding,
                "match_threshold": 0.0,
                "match_count": 5,
                "p_client_id": client_id,
                "p_rfp_id": None  # Search across all RFPs for this client
            }).execute()
        
        if res.data and len(res.data) > 0:
            print(f"SUCCESS: Found {len(res.data)} matches (client-wide search)")
            for match in res.data[:2]:
                print(f"  - Similarity: {match.get('similarity', 0):.3f} | Q: {match.get('question', '')[:50]}...")
            return res.data
        
        # STRATEGY 3: No matches found - run diagnostics
        print(f"WARN: No matches found. Running diagnostics...")
        
        # Check if questions exist for this client
        try:
            q_check = supabase.table("client_questions").select("id, original_text, embedding, rfp_id").eq("client_id", client_id).limit(10).execute()
            if q_check.data:
                print(f"DEBUG: Found {len(q_check.data)} total questions for client")
                
                # Check embeddings
                with_emb = [q for q in q_check.data if q.get("embedding")]
                print(f"DEBUG: {len(with_emb)}/{len(q_check.data)} questions have embeddings")
                
                # Check RFP association
                if rfp_id:
                    rfp_qs = [q for q in q_check.data if q.get("rfp_id") == rfp_id]
                    print(f"DEBUG: {len(rfp_qs)} questions associated with RFP {rfp_id}")
                
                # Check answer mappings (RPC might require this)
                q_ids = [q["id"] for q in q_check.data[:5]]
                map_check = fetch_question_answer_mappings(q_ids)
                if map_check:
                    print(f"DEBUG: {len(map_check)}/{len(q_ids)} sample questions have answer mappings")
                else:
                    print(f"ERROR: Questions have NO answer mappings! This may cause RPC to return nothing.")
                    print(f"       Solution: Ensure questions in knowledge base have answers mapped to them.")
            else:
                print(f"ERROR: No questions found in database for client {client_id}")
                print(f"       Solution: Add questions to knowledge base first via /org/qa or /org/qa/extract")
        except Exception as diag_error:
            print(f"ERROR: Diagnostic check failed: {diag_error}")
            traceback.print_exc()
        
        return []
        
    except Exception as e:
        print(f"ERROR: Supabase search failed: {e}")
        traceback.print_exc()
        return []


def pick_best_match(matches: list):
    """Select the best match from a list of matches based on similarity"""
    if not matches:
        return None
    return max(matches, key=lambda m: m.get("similarity", 0))


def insert_qa_pair(client_id: str, question_text: str, answer_text: str, category: str = "Other", rfp_id: str = None) -> bool:
    """Insert a question-answer pair into the database"""
    try:
        supabase = get_supabase_client()
        q_emb = get_embedding(question_text)
        q_ins = supabase.table("client_questions").insert({
            "original_text": question_text,
            "normalized_text": (question_text or "").lower(),
            "embedding": q_emb,
            "category": category or "Other",
            "client_id": client_id,
            "rfp_id": rfp_id,
        }).execute()
        q_id = (q_ins.data or [{}])[0].get("id")

        a_ins = supabase.table("client_answers").insert({
            "answer_text": answer_text,
            "answer_type": "General",
            "character_count": len(answer_text or ""),
            "technical_level": 1,
            "client_id": client_id,
            "rfp_id": rfp_id,
        }).execute()
        a_id = (a_ins.data or [{}])[0].get("id")

        if q_id and a_id:
            supabase.table("client_question_answer_mappings").insert({
                "question_id": q_id,
                "answer_id": a_id,
                "confidence_score": 1.0,
                "context_requirements": None,
                "stakeholder_approved": False,
            }).execute()
            return True
    except Exception as e:
        print(f"_insert_qa_pair error: {e}")
        traceback.print_exc()
    return False

