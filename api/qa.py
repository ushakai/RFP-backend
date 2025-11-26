"""
Q&A and Organization management endpoints
"""
import os
import json
import re
import traceback
import zipfile
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime
from fastapi import APIRouter, UploadFile, Header, HTTPException, Body, File
from utils.auth import get_client_id_from_key
from config.settings import get_supabase_client, GEMINI_MODEL
from services.gemini_service import get_embedding, extract_qa_pairs_from_sheet
from services.supabase_service import (
    insert_qa_pair,
    fetch_question_answer_mappings,
    fetch_paginated_rows,
)
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

router = APIRouter()

# ============================================================================
# ORGANIZATION MANAGEMENT
# ============================================================================

@router.get("/org")
def get_org(x_client_key: str | None = Header(default=None, alias="X-Client-Key")):
    """Get organization details"""
    client_id = get_client_id_from_key(x_client_key)
    try:
        supabase = get_supabase_client()
        res = supabase.table("clients").select("id, name, sector, contact_email").eq("id", client_id).single().execute()
        return res.data
    except Exception as e:
        print(f"Error getting organization data: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to get organization data")


@router.put("/org")
def update_org(
    payload: dict = Body(...), 
    x_client_key: str | None = Header(default=None, alias="X-Client-Key")
):
    """Update organization details"""
    client_id = get_client_id_from_key(x_client_key)
    updates = {k: v for k, v in payload.items() if k in ("name", "sector", "contact_email")}
    try:
        supabase = get_supabase_client()
        supabase.table("clients").update(updates).eq("id", client_id).execute()
        return {"ok": True}
    except Exception as e:
        print(f"Error updating organization data: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to update organization data")


# ============================================================================
# Q&A MANAGEMENT
# ============================================================================

@router.get("/org/qa")
def list_org_qa(
    x_client_key: str | None = Header(default=None, alias="X-Client-Key"), 
    rfp_id: str | None = Header(default=None, alias="X-RFP-ID")
):
    """List all Q&A pairs for the organization"""
    try:
        client_id = get_client_id_from_key(x_client_key)
        
        def _build_questions_query():
            query = get_supabase_client().table("client_questions").select("id, original_text, category, created_at, rfp_id").eq("client_id", client_id)
            if rfp_id:
                query = query.eq("rfp_id", rfp_id)
            return query.order("created_at", desc=True)

        def _build_answers_query():
            query = get_supabase_client().table("client_answers").select("id, answer_text, quality_score, last_updated, rfp_id").eq("client_id", client_id)
            if rfp_id:
                query = query.eq("rfp_id", rfp_id)
            return query.order("last_updated", desc=True)

        qs = fetch_paginated_rows(_build_questions_query, page_size=400)
        ans = fetch_paginated_rows(_build_answers_query, page_size=400)

        q_ids = [q.get("id") for q in qs] if qs else []
        mappings = []
        if q_ids:
            mappings = fetch_question_answer_mappings(q_ids)

        return {"questions": qs, "answers": ans, "mappings": mappings}
    except HTTPException:
        raise
    except Exception as e:
        print(f"list_org_qa error: {e}")
        traceback.print_exc()
        return {"questions": [], "answers": [], "mappings": []}


@router.post("/org/qa")
def ingest_org_qa(
    payload: dict = Body(...), 
    x_client_key: str | None = Header(default=None, alias="X-Client-Key"), 
    rfp_id: str | None = Header(default=None, alias="X-RFP-ID")
):
    """Ingest a list of question/answer pairs for the organization.
    payload: { pairs: [{question, answer, category?}] }
    """
    client_id = get_client_id_from_key(x_client_key)
    pairs = payload.get("pairs", []) or []
    created = 0
    for p in pairs:
        qtext = (p.get("question") or "").strip()
        atext = (p.get("answer") or "").strip()
        category = (p.get("category") or "Other").strip() or "Other"
        if not qtext or not atext:
            continue
        try:
            if insert_qa_pair(client_id, qtext, atext, category, rfp_id):
                created += 1
        except Exception as e:
            print(f"ingest_org_qa error: {e}")
            traceback.print_exc()
            continue
    return {"created": created}


@router.post("/org/qa/extract")
async def extract_qa_from_upload(
    file: UploadFile = File(...), 
    x_client_key: str | None = Header(default=None, alias="X-Client-Key"), 
    rfp_id: str | None = Header(default=None, alias="X-RFP-ID")
):
    """Extract Q&A pairs from uploaded Excel or ZIP file"""
    client_id = get_client_id_from_key(x_client_key)
    created = 0
    with tempfile.TemporaryDirectory() as td:
        tmp_path = os.path.join(td, file.filename)
        content = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(content)

        def _process_excel(path: str):
            nonlocal created
            try:
                xls = pd.ExcelFile(path)
                for sheet in xls.sheet_names:
                    df = pd.read_excel(path, sheet_name=sheet, header=None)
                    if df.empty:
                        continue
                    sheet_csv = df.to_csv(index=False, header=False)
                    pairs = extract_qa_pairs_from_sheet(sheet_csv) or []
                    for p in pairs:
                        q = (p.get("question") or "").strip()
                        a = (p.get("answer") or "").strip()
                        c = (p.get("category") or "Other").strip() or "Other"
                        if q and a and insert_qa_pair(client_id, q, a, c, rfp_id):
                            created += 1
            except Exception as e:
                print(f"extract_qa_from_upload excel error: {e}")
                traceback.print_exc()

        if file.filename.lower().endswith(".zip"):
            try:
                extract_dir = os.path.join(td, "unzipped")
                os.makedirs(extract_dir, exist_ok=True)
                with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                for root, _, files in os.walk(extract_dir):
                    for fn in files:
                        if fn.lower().endswith(".xlsx"):
                            _process_excel(os.path.join(root, fn))
            except Exception as e:
                print(f"extract_qa_from_upload zip error: {e}")
                traceback.print_exc()
        elif file.filename.lower().endswith(".xlsx"):
            _process_excel(tmp_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Upload .zip or .xlsx")

    return {"created": created}


# ============================================================================
# Q&A ANALYSIS AND GROUPING
# ============================================================================

@router.post("/org/qa/analyze-similarities")
def analyze_qa_similarities(
    payload: dict = Body(default={}), 
    x_client_key: str | None = Header(default=None, alias="X-Client-Key"), 
    rfp_id: str | None = Header(default=None, alias="X-RFP-ID")
):
    """Analyze QA pairs to find similar questions and create summary suggestions"""
    client_id = get_client_id_from_key(x_client_key)
    similarity_threshold = payload.get("similarity_threshold", 0.85)
    
    try:
        supabase = get_supabase_client()
        gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        
        # Get all QA pairs with embeddings
        questions_res = supabase.table("client_questions").select("id, original_text, embedding, rfp_id").eq("client_id", client_id)
        if rfp_id:
            questions_res = questions_res.eq("rfp_id", rfp_id)
        questions = questions_res.execute().data or []
        
        if len(questions) < 2:
            return {"groups": [], "message": "Not enough QA pairs to analyze"}
        
        # Get answer mappings
        q_ids = [q["id"] for q in questions]
        mappings = fetch_question_answer_mappings(q_ids)
        mapping_dict = {m["question_id"]: m["answer_id"] for m in mappings}
        
        # Get answers
        a_ids = list(set(mapping_dict.values()))
        if not a_ids:
            return {"groups": [], "message": "No answers found for questions"}
        
        answers_res = supabase.table("client_answers").select("id, answer_text").in_("id", a_ids).execute()
        answers = answers_res.data or []
        answer_dict = {a["id"]: a["answer_text"] for a in answers}
        
        # Find similar question groups using embeddings
        similar_groups = []
        processed_questions = set()
        
        for i, q1 in enumerate(questions):
            if q1["id"] in processed_questions:
                continue
                
            emb1 = q1.get("embedding")
            if not emb1 or not isinstance(emb1, list):
                continue
            
            group = [q1]
            processed_questions.add(q1["id"])
            
            # Find similar questions
            for j, q2 in enumerate(questions):
                if i >= j or q2["id"] in processed_questions:
                    continue
                
                emb2 = q2.get("embedding")
                if not emb2 or not isinstance(emb2, list):
                    continue
                
                # Calculate cosine similarity
                try:
                    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    
                    if similarity >= similarity_threshold:
                        group.append(q2)
                        processed_questions.add(q2["id"])
                except Exception as e:
                    print(f"Error calculating similarity: {e}")
                    traceback.print_exc()
                    continue
            
            # Only keep groups with 2+ questions
            if len(group) >= 2:
                # Create summary using AI
                questions_text = "\n".join([f"- {q['original_text']}" for q in group])
                answers_text = "\n".join([f"- {answer_dict.get(mapping_dict.get(q['id']), 'No answer')}" for q in group])
                
                summary_prompt = f"""
You are consolidating similar Q&A pairs into a single comprehensive Q&A pair.

Similar Questions:
{questions_text}

Their Answers:
{answers_text}

Create:
1. A consolidated question that covers all the similar questions
2. A comprehensive answer that combines all the answers

Return ONLY a JSON object with this exact format:
{{"question": "consolidated question here", "answer": "comprehensive answer here"}}
"""
                
                try:
                    response = gemini_model.generate_content(
                        summary_prompt,
                        safety_settings={
                            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                        },
                    )
                    
                    resp_text = (response.text or "").strip()
                    start, end = resp_text.find("{"), resp_text.rfind("}")
                    if start != -1 and end != -1:
                        summary_data = json.loads(resp_text[start:end+1])
                        
                        similar_groups.append({
                            "group_id": f"group_{len(similar_groups)}",
                            "questions": [
                                {
                                    "id": q["id"],
                                    "text": q["original_text"],
                                    "answer": answer_dict.get(mapping_dict.get(q["id"]), "No answer")
                                } for q in group
                            ],
                            "suggested_question": summary_data.get("question", ""),
                            "suggested_answer": summary_data.get("answer", ""),
                            "similarity_count": len(group)
                        })
                except Exception as e:
                    print(f"Error creating summary: {e}")
                    traceback.print_exc()
                    continue
        
        return {
            "groups": similar_groups,
            "total_groups": len(similar_groups),
            "total_questions_analyzed": len(questions),
            "similarity_threshold": similarity_threshold
        }
        
    except Exception as e:
        print(f"Error analyzing similarities: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/org/qa/ai-group")
def ai_group_qa(
    payload: dict = Body(default={}), 
    x_client_key: str | None = Header(default=None, alias="X-Client-Key"), 
    rfp_id: str | None = Header(default=None, alias="X-RFP-ID")
):
    """AI-driven grouping of all Q&A pairs for a client"""
    client_id = get_client_id_from_key(x_client_key)
    try:
        supabase = get_supabase_client()
        gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        
        # Fetch all questions
        q_query = supabase.table("client_questions").select("id, original_text, rfp_id").eq("client_id", client_id)
        if rfp_id:
            q_query = q_query.eq("rfp_id", rfp_id)
        questions = q_query.order("created_at", desc=True).execute().data or []

        if not questions:
            return {"groups": [], "message": "No questions found"}

        # Fetch mappings for answers
        q_ids = [q["id"] for q in questions]
        m_rows = fetch_question_answer_mappings(q_ids)
        q_to_a = {m["question_id"]: m["answer_id"] for m in m_rows}

        # Fetch answers
        a_ids = list({aid for aid in q_to_a.values() if aid})
        a_rows = []
        if a_ids:
            a_rows = supabase.table("client_answers").select("id, answer_text").in_("id", a_ids).execute().data or []
        a_map = {a["id"]: a.get("answer_text", "") for a in a_rows}

        # Build compact input for the LLM
        qa_lines = []
        for q in questions:
            qid = q["id"]
            qtext = (q.get("original_text") or "").strip()
            atext = (a_map.get(q_to_a.get(qid)) or "").strip()
            if not qtext:
                continue
            qa_lines.append({"id": qid, "q": qtext, "a": atext})

        if not qa_lines:
            return {"groups": [], "message": "No Q&A pairs available"}

        # Prompt Gemini to group and summarize
        qa_text = "\n".join([f"ID:{row['id']}\nQ:{row['q']}\nA:{row['a']}" for row in qa_lines])
        prompt = f"""
You will receive a list of Q&A pairs for a single client (and optionally a specific RFP). Group semantically similar questions together and produce a consolidated Q&A for each group.

Return ONLY strict JSON with this structure:
{{
  "groups": [
    {{
      "question_ids": ["<id>", "<id>", ...],
      "consolidated_question": "string",
      "consolidated_answer": "string"
    }}
  ]
}}

Rules:
1) "question_ids" must be the exact IDs provided.
2) Every ID used must exist in the input. Do not invent IDs.
3) Do not include any text before or after the JSON.

Q&A LIST:
{qa_text}
"""
        try:
            response = gemini_model.generate_content(
                prompt,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                },
            )
            text = (response.text or "").strip()
            s, e = text.find("{"), text.rfind("}")
            if s == -1 or e == -1:
                return {"groups": [], "message": "LLM returned no JSON"}
            data = json.loads(text[s:e+1])

            # Normalize output
            raw_groups = data.get("groups") or []
            valid_ids_local = {str(x["id"]) for x in questions}
            groups = []
            for g in raw_groups:
                qids = [str(x) for x in (g.get("question_ids") or []) if str(x) in valid_ids_local]
                if not qids:
                    continue
                groups.append({
                    "question_ids": qids,
                    "consolidated_question": (g.get("consolidated_question") or "").strip(),
                    "consolidated_answer": (g.get("consolidated_answer") or "").strip(),
                })
            return {"groups": groups}
        except Exception as e:
            print(f"ai_group_qa LLM error: {e}")
            traceback.print_exc()
            return {"groups": [], "message": "AI grouping failed"}

    except HTTPException:
        raise
    except Exception as e:
        print(f"ai_group_qa error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to group Q&A")


@router.post("/org/qa/approve-summary")
def approve_qa_summary(
    payload: dict = Body(...), 
    x_client_key: str | None = Header(default=None, alias="X-Client-Key"), 
    rfp_id: str | None = Header(default=None, alias="X-RFP-ID")
):
    """Approve a summary - adds consolidated QA plus summary records and mappings"""
    client_id = get_client_id_from_key(x_client_key)
    
    question_ids = payload.get("question_ids", [])
    consolidated_question = payload.get("consolidated_question", "").strip()
    consolidated_answer = payload.get("consolidated_answer", "").strip()
    
    if not question_ids or not consolidated_question or not consolidated_answer:
        raise HTTPException(status_code=400, detail="Missing required fields")
    
    try:
        supabase = get_supabase_client()
        
        # Create embedding for consolidated question
        q_emb = get_embedding(consolidated_question)

        # Insert consolidated question
        q_row = {
            "original_text": consolidated_question,
            "normalized_text": consolidated_question.lower(),
            "embedding": q_emb,
            "category": "Consolidated",
            "client_id": client_id,
            "rfp_id": rfp_id,
        }
        q_ins = supabase.table("client_questions").insert(q_row).execute()
        new_q_id = (q_ins.data or [{}])[0].get("id")

        # Insert consolidated answer
        a_row = {
            "answer_text": consolidated_answer,
            "answer_type": "Consolidated",
            "character_count": len(consolidated_answer),
            "technical_level": 1,
            "client_id": client_id,
            "rfp_id": rfp_id,
        }
        a_ins = supabase.table("client_answers").insert(a_row).execute()
        new_a_id = (a_ins.data or [{}])[0].get("id")

        # Create mapping for consolidated QA
        if new_q_id and new_a_id:
            supabase.table("client_question_answer_mappings").insert({
                "question_id": new_q_id,
                "answer_id": new_a_id,
                "confidence_score": 1.0,
                "context_requirements": None,
                "stakeholder_approved": True,
            }).execute()

        # Insert summary record
        s_row = {
            "summary_text": consolidated_answer,
            "summary_type": "Consolidated",
            "character_count": len(consolidated_answer),
            "quality_score": None,
            "approved": True,
            "client_id": client_id,
            "rfp_id": rfp_id,
        }
        s_ins = supabase.table("client_summaries").insert(s_row).execute()
        new_s_id = (s_ins.data or [{}])[0].get("id")

        # Map all original questions to the summary
        if new_s_id and question_ids:
            mappings = [{"question_id": qid, "summary_id": new_s_id} for qid in question_ids]
            supabase.table("client_question_summary_mappings").insert(mappings).execute()

        return {
            "success": True,
            "new_question_id": new_q_id,
            "new_answer_id": new_a_id,
            "new_summary_id": new_s_id,
            "mapped_questions": len(question_ids)
        }

    except Exception as e:
        print(f"Error approving summary: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to approve summary: {str(e)}")


# ============================================================================
# Q&A SCORING
# ============================================================================

def _build_scoring_prompt(question: str, answer: str, reference: str) -> str:
    return f"""
You are an expert RFP evaluator. Given QUESTION, ANSWER, and REFERENCE, return a strict JSON object:
{{ "score": <integer 0-100>, "notes": "<one sentence>" }}

QUESTION:
\"\"\"{question}\"\"\"

ANSWER:
\"\"\"{answer}\"\"\"

REFERENCE:
\"\"\"{reference}\"\"\"
"""


def _extract_score(resp_text: str) -> int:
    if not resp_text:
        return -1
    try:
        s, e = resp_text.find("{"), resp_text.rfind("}")
        if s != -1 and e != -1 and e > s:
            data = json.loads(resp_text[s:e+1])
            if "score" in data:
                val = int(round(float(data["score"])))
                return max(0, min(100, val))
    except Exception:
        pass
    # fallback: first integer
    m = re.search(r"(\d{1,3})(?:\.\d+)?", resp_text)
    if m:
        try:
            return max(0, min(100, int(round(float(m.group(1))))))
        except Exception:
            return -1
    return -1


@router.post("/org/qa/score")
def score_org_answers(
    payload: dict = Body(default={}), 
    x_client_key: str | None = Header(default=None, alias="X-Client-Key"), 
    rfp_id: str | None = Header(default=None, alias="X-RFP-ID")
):
    """Score organization answers using AI"""
    client_id = get_client_id_from_key(x_client_key)
    limit = payload.get("limit")
    reference_text = payload.get("reference_text") or ""

    try:
        supabase = get_supabase_client()
        gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        
        # fetch mappings and join QA
        rows = supabase.table("client_question_answer_mappings").select(
            "question_id, answer_id, client_questions(original_text, client_id, rfp_id), client_answers(answer_text, quality_score, client_id, rfp_id)"
        ).execute().data or []
    except Exception as e:
        print(f"score_org_answers fetch error: {e}")
        traceback.print_exc()
        rows = []

    # filter to this org and optionally RFP
    scoped = []
    for r in rows:
        q_data = r.get("client_questions") or {}
        a_data = r.get("client_answers") or {}
        if (q_data.get("client_id") == client_id and a_data.get("client_id") == client_id and
            (rfp_id is None or (q_data.get("rfp_id") == rfp_id and a_data.get("rfp_id") == rfp_id))):
            scoped.append(r)

    if limit:
        scoped = scoped[: int(limit)]

    updated = 0
    supabase = get_supabase_client()
    for r in scoped:
        question_text = (r.get("client_questions") or {}).get("original_text") or ""
        answer_id = r.get("answer_id")
        answer_text = (r.get("client_answers") or {}).get("answer_text") or ""
        if not answer_id or not question_text or not answer_text:
            continue

        prompt = _build_scoring_prompt(question_text, answer_text, reference_text)
        try:
            resp = gemini_model.generate_content(prompt)
            resp_text = resp.text or ""
        except Exception as e:
            print(f"score_org_answers LLM error: {e}")
            traceback.print_exc()
            resp_text = ""

        score = _extract_score(resp_text)
        if score == -1:
            continue

        try:
            supabase.table("client_answers").update({"quality_score": int(score)}).eq("id", answer_id).execute()
            updated += 1
        except Exception as e:
            print(f"score_org_answers update error: {e}")
            traceback.print_exc()

    return {"updated": updated}


# ============================================================================
# QUESTION/ANSWER CRUD OPERATIONS
# ============================================================================

@router.put("/questions/{question_id}")
def update_question(
    question_id: str, 
    payload: dict = Body(...), 
    x_client_key: str | None = Header(default=None, alias="X-Client-Key")
):
    """Update a question"""
    client_id = get_client_id_from_key(x_client_key)
    updates = {k: v for k, v in payload.items() if k in ("original_text", "category")}
    if "original_text" in updates:
        updates["normalized_text"] = updates["original_text"].lower()
    try:
        supabase = get_supabase_client()
        res = supabase.table("client_questions").update(updates).eq("id", question_id).eq("client_id", client_id).execute()
        return {"ok": True}
    except Exception as e:
        print(f"Error updating question {question_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to update question")


@router.delete("/questions/{question_id}")
def delete_question(
    question_id: str, 
    x_client_key: str | None = Header(default=None, alias="X-Client-Key")
):
    """Delete a question"""
    client_id = get_client_id_from_key(x_client_key)
    try:
        supabase = get_supabase_client()
        supabase.table("client_questions").delete().eq("id", question_id).eq("client_id", client_id).execute()
        return {"ok": True}
    except Exception as e:
        print(f"Error deleting question {question_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to delete question")


@router.put("/answers/{answer_id}")
def update_answer(
    answer_id: str, 
    payload: dict = Body(...), 
    x_client_key: str | None = Header(default=None, alias="X-Client-Key")
):
    """Update an answer"""
    client_id = get_client_id_from_key(x_client_key)
    updates = {k: v for k, v in payload.items() if k in ("answer_text", "quality_score")}
    if "answer_text" in updates:
        updates["character_count"] = len(updates["answer_text"])
    try:
        supabase = get_supabase_client()
        res = supabase.table("client_answers").update(updates).eq("id", answer_id).eq("client_id", client_id).execute()
        return {"ok": True}
    except Exception as e:
        print(f"Error updating answer {answer_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to update answer")


@router.delete("/answers/{answer_id}")
def delete_answer(
    answer_id: str, 
    x_client_key: str | None = Header(default=None, alias="X-Client-Key")
):
    """Delete an answer"""
    client_id = get_client_id_from_key(x_client_key)
    try:
        supabase = get_supabase_client()
        supabase.table("client_answers").delete().eq("id", answer_id).eq("client_id", client_id).execute()
        return {"ok": True}
    except Exception as e:
        print(f"delete_answer error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to delete answer")


# ============================================================================
# SUMMARIES MANAGEMENT
# ============================================================================

@router.get("/org/summaries")
def list_org_summaries(
    x_client_key: str | None = Header(default=None, alias="X-Client-Key"), 
    rfp_id: str | None = Header(default=None, alias="X-RFP-ID")
):
    """List summaries and their mapped questions for an organization"""
    client_id = get_client_id_from_key(x_client_key)
    try:
        supabase = get_supabase_client()
        s_query = supabase.table("client_summaries").select("id, summary_text, summary_type, character_count, quality_score, created_at, rfp_id, client_id, approved").eq("client_id", client_id).eq("approved", True)
        if rfp_id:
            s_query = s_query.eq("rfp_id", rfp_id)
        summaries = s_query.order("created_at", desc=True).execute().data or []

        if not summaries:
            return {"summaries": [], "mappings": [], "questions": []}

        s_ids = [s["id"] for s in summaries]
        m_rows = supabase.table("client_question_summary_mappings").select("question_id, summary_id").in_("summary_id", s_ids).execute().data or []

        q_ids = list({m["question_id"] for m in m_rows})
        questions = []
        if q_ids:
            q_rows = supabase.table("client_questions").select("id, original_text, category, created_at, rfp_id").in_("id", q_ids).execute().data or []
            questions = q_rows

        return {"summaries": summaries, "mappings": m_rows, "questions": questions}
    except Exception as e:
        print(f"list_org_summaries error: {e}")
        traceback.print_exc()
        return {"summaries": [], "mappings": [], "questions": []}


@router.get("/org/summaries/pending")
def list_pending_summaries(
    x_client_key: str | None = Header(default=None, alias="X-Client-Key"), 
    rfp_id: str | None = Header(default=None, alias="X-RFP-ID")
):
    """List pending (unapproved) summaries"""
    client_id = get_client_id_from_key(x_client_key)
    try:
        supabase = get_supabase_client()
        s_query = supabase.table("client_summaries").select("id, summary_text, summary_type, character_count, quality_score, created_at, rfp_id, client_id, approved").eq("client_id", client_id).eq("approved", False)
        if rfp_id:
            s_query = s_query.eq("rfp_id", rfp_id)
        summaries = s_query.order("created_at", desc=True).execute().data or []

        if not summaries:
            return {"summaries": [], "mappings": [], "questions": []}

        s_ids = [s["id"] for s in summaries]
        m_rows = supabase.table("client_question_summary_mappings").select("question_id, summary_id").in_("summary_id", s_ids).execute().data or []

        q_ids = list({m["question_id"] for m in m_rows})
        questions = []
        if q_ids:
            q_rows = supabase.table("client_questions").select("id, original_text, category, created_at, rfp_id").in_("id", q_ids).execute().data or []
            questions = q_rows

        return {"summaries": summaries, "mappings": m_rows, "questions": questions}
    except Exception as e:
        print(f"list_pending_summaries error: {e}")
        traceback.print_exc()
        return {"summaries": [], "mappings": [], "questions": []}


@router.post("/org/summaries/{summary_id}/set-approval")
def set_summary_approval(
    summary_id: str, 
    payload: dict = Body(...), 
    x_client_key: str | None = Header(default=None, alias="X-Client-Key")
):
    """Set approval status for a summary (approved true/false)"""
    client_id = get_client_id_from_key(x_client_key)
    approved = bool(payload.get("approved", False))
    try:
        supabase = get_supabase_client()
        supabase.table("client_summaries").update({"approved": approved}).eq("id", summary_id).eq("client_id", client_id).execute()
        return {"ok": True, "approved": approved}
    except Exception as e:
        print(f"set_summary_approval error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to update summary approval")

