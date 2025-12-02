"""
Supabase Service - Database operations
"""
import traceback
import time
from typing import Any, Callable, List

import httpcore
import httpx

from config.settings import get_supabase_client, reinitialize_supabase
from services.gemini_service import get_embedding


def execute_with_retry(
    operation_factory: Callable[[], Any],
    retries: int = 5,
    backoff_seconds: float = 0.5,
    on_retry: Callable[[], None] | None = None,
):
    """Execute a Supabase/PostgREST operation with retries on socket/network errors."""
    last_error: Exception | None = None
    
    for attempt in range(1, retries + 1):
        try:
            return operation_factory()
        except (
            httpx.ReadError,
            httpx.RemoteProtocolError,
            httpx.ConnectError,
            httpx.TimeoutException,
            httpcore.ReadError,
            httpcore.RemoteProtocolError,
            ConnectionError,
            ConnectionResetError,
            BrokenPipeError,
            RuntimeError,  # "Cannot send a request, as the client has been closed"
        ) as err:
            last_error = err
            error_msg = str(err)
            
            # Only log first and last attempt to reduce noise
            if attempt == 1 or attempt == retries:
                print(f"Supabase connection error (attempt {attempt}/{retries}): {error_msg[:100]}")
            
            # Reinitialize client to get fresh connection
            try:
                reinitialize_supabase()
            except Exception:
                pass  # Ignore reinit errors, will retry anyway
            
            if on_retry:
                try:
                    on_retry()
                except Exception:
                    pass
            
            if attempt < retries:
                # Exponential backoff
                sleep_time = backoff_seconds * (2 ** (attempt - 1))
                time.sleep(min(sleep_time, 5.0))  # Cap at 5 seconds
        except Exception as err:
            # Non-network errors bubble up immediately
            raise err
    
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

