"""
Supabase Service - Database operations
"""
import traceback
from config.settings import get_supabase_client
from services.gemini_service import get_embedding

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
                map_check = supabase.table("client_question_answer_mappings").select("question_id, answer_id").in_("question_id", q_ids).execute()
                if map_check.data:
                    print(f"DEBUG: {len(map_check.data)}/{len(q_ids)} sample questions have answer mappings")
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

