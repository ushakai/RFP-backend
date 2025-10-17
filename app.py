import os
import io
import json
import difflib
import re
import pandas as pd
import openpyxl
from fastapi import FastAPI, UploadFile, Header, HTTPException
from fastapi import Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from supabase import create_client, Client
from openpyxl.styles import Alignment
from openpyxl.utils import get_column_letter
import google.generativeai as genai
import zipfile
import tempfile
from dotenv import load_dotenv
import asyncio
import threading
import uuid
from datetime import datetime, timedelta

# CONFIGURATION 
load_dotenv()
def _clean_env(value: str | None) -> str:
    if not value:
        return ""
    return value.strip().strip('"').strip("'")

GOOGLE_API_KEY = _clean_env(os.getenv("GOOGLE_API_KEY"))
SUPABASE_URL = _clean_env(os.getenv("SUPABASE_URL"))
SUPABASE_KEY = _clean_env(os.getenv("SUPABASE_KEY"))

# Initialize Gemini client
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY for Gemini")
genai.configure(api_key=GOOGLE_API_KEY)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
gemini = genai.GenerativeModel(GEMINI_MODEL)

# Validate and initialize Supabase client with clearer errors
if not SUPABASE_URL or not SUPABASE_URL.startswith("https://") or ".supabase.co" not in SUPABASE_URL:
    raise ValueError(f"Invalid SUPABASE_URL format: '{SUPABASE_URL}'. Expected like https://xxxxx.supabase.co")
if not SUPABASE_KEY:
    raise ValueError("SUPABASE_KEY is missing")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# FastAPI app
app = FastAPI()

# CORS for frontend dev and configurable origins
frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HELPER FUNCTIONS

def extract_questions_with_llm(sheet_text: str) -> list:
    prompt = f"""
You are analyzing an RFP Excel sheet. Identify ALL rows that represent questions or
requirements directed at the vendor. Treat both explicit questions (with '?') and
implicit requirements (like "Vendor must provide X", "Describe how your system handles Y")
as questions.

Return only a JSON array where each object has:
- "question": the extracted question/requirement text (cleaned and concise).
- "row": the row number in the sheet (1-based).
Return JSON array only:
[
  {{"question": "Question text here", "row": 5}}
]

If no questions are found, return [].

Sheet:
{sheet_text}
"""
    try:
        response = gemini.generate_content(prompt)
        text = (response.text or "").strip()
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end == -1:
            return []
        return json.loads(text[start:end+1])
    except Exception as e:
        print(f"LLM extract error: {e}")
        return []


def clean_markdown(text: str) -> str:
    """Remove common Markdown formatting such as **bold**, _italic_, `code`, headings, lists, and links.
    Keeps only readable plain text.
    """
    if not text:
        return ""
    cleaned = text
    # Remove images entirely
    cleaned = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", cleaned)
    # Convert markdown links [text](url) -> text
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", cleaned)
    # Remove inline code backticks
    cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
    # Remove bold/italic markers
    cleaned = re.sub(r"(\*\*|__)(.*?)\1", r"\2", cleaned)
    cleaned = re.sub(r"(\*|_)(.*?)\1", r"\2", cleaned)
    # Remove headings (leading #)
    cleaned = re.sub(r"^\s{0,3}#{1,6}\s*", "", cleaned, flags=re.MULTILINE)
    # Remove blockquotes
    cleaned = re.sub(r"^\s{0,3}>\s?", "", cleaned, flags=re.MULTILINE)
    # Remove list markers
    cleaned = re.sub(r"^\s*[-*+]\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*\d+\.\s+", "", cleaned, flags=re.MULTILINE)
    # Remove horizontal rules lines
    cleaned = re.sub(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", "", cleaned, flags=re.MULTILINE)
    # Collapse multiple spaces/newlines
    cleaned = re.sub(r"\s+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def get_embedding(text: str) -> list:
    if not text:
        return []
    try:
        resp = genai.embed_content(model=GEMINI_EMBEDDING_MODEL, content=text)
        return (resp.get("embedding") or [])
    except Exception as e:
        print(f"Embedding error: {e}")
        return []


def search_supabase(question_embedding: list, client_id: str, rfp_id: str = None) -> list:
    if not question_embedding:
        return []
    try:
        res = supabase.rpc(
            "client_match_questions",
            {
                "query_embedding": question_embedding,
                "match_threshold": 0.0,
                "match_count": 5,
                "p_client_id": client_id,
                "p_rfp_id": rfp_id,
            }
        ).execute()
        return res.data if res.data else []
    except Exception as e:
        print(f"Supabase error: {e}")
        return []


def pick_best_match(matches: list):
    if not matches:
        return None
    return max(matches, key=lambda m: m.get("similarity", 0))


def generate_tailored_answer(question: str, matches: list) -> str:
    context = "\n".join(
        f"- Q: {m['question']} | A: {m['answer']} (similarity {m['similarity']:.2f})"
        for m in matches
    )
    prompt = f"""
You are answering an RFP vendor question.  

New Question:
{question}

Reference Q&A Pairs:
{context}

Write a concise, tailored answer that best addresses the new question using the references.
If unclear, combine and adapt from references.
"""
    try:
        resp = gemini.generate_content(prompt)
        return (resp.text or "").strip()
    except Exception as e:
        print(f"LLM tailored answer error: {e}")
        return "AI could not generate tailored answer."


def _row_text(ws, r: int) -> str:
    return " ".join(
        str(ws.cell(row=r, column=c).value or "").lower()
        for c in range(1, ws.max_column + 1)
    )


def resolve_row(worksheet, reported_row: int, question_text: str) -> int:
    max_row = worksheet.max_row
    qnorm = (question_text or "").strip().lower()

    # Prefer a local window around the reported row if provided (±3 rows)
    if reported_row and 2 <= reported_row <= max_row:
        start_r = max(2, reported_row - 3)
        end_r = min(max_row, reported_row + 3)
        best_r, best_score = reported_row, -1.0
        for r in range(start_r, end_r + 1):
            row_text = _row_text(worksheet, r)
            s = difflib.SequenceMatcher(None, qnorm, row_text).ratio()
            if s > best_score:
                best_score, best_r = s, r
        return best_r

    # Fallback to global search when no reported row
    best_r, best_score = None, -1.0
    for r in range(2, max_row + 1):  # skip header
        row_text = _row_text(worksheet, r)
        s = difflib.SequenceMatcher(None, qnorm, row_text).ratio()
        if s > best_score:
            best_score, best_r = s, r
    return best_r or 2


def find_first_empty_column(ws):
    for col in range(1, ws.max_column + 2):
        values = [ws.cell(row=r, column=col).value for r in range(1, ws.max_row + 1)]
        if all(v is None for v in values):
            return col
    return ws.max_column + 1


# PROCESS EXCEL IN-MEMORY
def process_excel_file_obj(file_obj: UploadFile, client_id: str, rfp_id: str = None):
    wb = openpyxl.load_workbook(file_obj.file)
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        print(f"Processing sheet: {sheet_name}")

        df = pd.DataFrame(ws.values)
        if df.empty:
            continue
        sheet_text = df.to_string(index=False, header=False)

        extracted = extract_questions_with_llm(sheet_text)
        if not extracted:
            continue

        ai_col = find_first_empty_column(ws)
        ws.cell(row=1, column=ai_col, value="AI Answers")
        # Expand column width and enable wrap
        ai_col_letter = get_column_letter(ai_col)
        ws.column_dimensions[ai_col_letter].width = 60

        review_col = ai_col + 1
        ws.cell(row=1, column=review_col, value="Review Status")
        review_col_letter = get_column_letter(review_col)
        ws.column_dimensions[review_col_letter].width = 20

        for item in extracted:
            qtext = item.get("question", "")
            reported_row = item.get("row", 0)
            if not qtext:
                continue

            write_row = resolve_row(ws, reported_row, qtext)
            emb = get_embedding(qtext)
            matches = search_supabase(emb, client_id, rfp_id)

            final_answer = "Not found, needs review."
            review_status = ""

            if matches:
                best = pick_best_match(matches)
                if best and best.get("similarity", 0) >= 0.9:
                    final_answer = best["answer"]
                else:
                    final_answer = generate_tailored_answer(qtext, matches)
                    review_status = "Need review"

            # Clean formatting from AI output
            final_answer = clean_markdown(final_answer)

            ai_cell = ws.cell(row=write_row, column=ai_col, value=final_answer)
            ai_cell.alignment = Alignment(wrap_text=True, vertical="top")
            if review_status:
                review_cell = ws.cell(row=write_row, column=review_col, value=review_status)
                review_cell.alignment = Alignment(wrap_text=True, vertical="top")

    output = io.BytesIO()
    wb.save(output)
    output.seek(0)
    return output


# API ENDPOINT
def get_client_id_from_key(client_key: str | None) -> str:
    if not client_key:
        raise HTTPException(status_code=401, detail="Missing X-Client-Key")
    try:
        resp = supabase.table("clients").select("id").eq("api_key", client_key).limit(1).execute()
        rows = resp.data or []
        if not rows:
            raise HTTPException(status_code=401, detail="Invalid X-Client-Key")
        return rows[0]["id"]
    except HTTPException:
        raise
    except Exception as e:
        print(f"Client lookup error: {e}")
        raise HTTPException(status_code=500, detail="Client lookup failed")


@app.post("/process")
async def process_excel(file: UploadFile, x_client_key: str | None = Header(default=None, alias="X-Client-Key"), rfp_id: str | None = Header(default=None, alias="X-RFP-ID")):
    client_id = get_client_id_from_key(x_client_key)
    processed_file = process_excel_file_obj(file, client_id, rfp_id)
    return StreamingResponse(
        processed_file,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=processed_{file.filename}"}
    )


# --- RFP management routes ---

@app.get("/rfps")
def list_rfps(x_client_key: str | None = Header(default=None, alias="X-Client-Key")):
    client_id = get_client_id_from_key(x_client_key)
    res = supabase.table("client_rfps").select("id, name, description, created_at, updated_at").eq("client_id", client_id).order("created_at", desc=True).execute()
    return {"rfps": res.data or []}

@app.post("/rfps")
def create_rfp(payload: dict = Body(...), x_client_key: str | None = Header(default=None, alias="X-Client-Key")):
    client_id = get_client_id_from_key(x_client_key)
    rfp_data = {
        "client_id": client_id,
        "name": payload.get("name", "").strip(),
        "description": payload.get("description", "").strip(),
    }
    if not rfp_data["name"]:
        raise HTTPException(status_code=400, detail="RFP name is required")
    res = supabase.table("client_rfps").insert(rfp_data).execute()
    return {"rfp": res.data[0] if res.data else None}

@app.put("/rfps/{rfp_id}")
def update_rfp(rfp_id: str, payload: dict = Body(...), x_client_key: str | None = Header(default=None, alias="X-Client-Key")):
    client_id = get_client_id_from_key(x_client_key)
    updates = {k: v for k, v in payload.items() if k in ("name", "description")}
    if "name" in updates:
        updates["name"] = updates["name"].strip()
    if "description" in updates:
        updates["description"] = updates["description"].strip()
    res = supabase.table("client_rfps").update(updates).eq("id", rfp_id).eq("client_id", client_id).execute()
    return {"ok": True}

@app.delete("/rfps/{rfp_id}")
def delete_rfp(rfp_id: str, x_client_key: str | None = Header(default=None, alias="X-Client-Key")):
    client_id = get_client_id_from_key(x_client_key)
    supabase.table("client_rfps").delete().eq("id", rfp_id).eq("client_id", client_id).execute()
    return {"ok": True}

# --- Organization-focused routes ---

@app.get("/org")
def get_org(x_client_key: str | None = Header(default=None, alias="X-Client-Key")):
    client_id = get_client_id_from_key(x_client_key)
    res = supabase.table("clients").select("id, name, sector, contact_email").eq("id", client_id).single().execute()
    return res.data


@app.put("/org")
def update_org(payload: dict = Body(...), x_client_key: str | None = Header(default=None, alias="X-Client-Key")):
    client_id = get_client_id_from_key(x_client_key)
    updates = {k: v for k, v in payload.items() if k in ("name", "sector", "contact_email")}
    supabase.table("clients").update(updates).eq("id", client_id).execute()
    return {"ok": True}


@app.get("/org/qa")
def list_org_qa(x_client_key: str | None = Header(default=None, alias="X-Client-Key"), rfp_id: str | None = Header(default=None, alias="X-RFP-ID")):
    client_id = get_client_id_from_key(x_client_key)
    qs_query = supabase.table("client_questions").select("id, original_text, category, created_at, rfp_id").eq("client_id", client_id)
    ans_query = supabase.table("client_answers").select("id, answer_text, category, quality_score, last_updated, rfp_id").eq("client_id", client_id)
    
    if rfp_id:
        qs_query = qs_query.eq("rfp_id", rfp_id)
        ans_query = ans_query.eq("rfp_id", rfp_id)
    
    qs = qs_query.order("created_at", desc=True).execute().data or []
    ans = ans_query.order("last_updated", desc=True).execute().data or []
    mappings = supabase.table("client_question_answer_mappings").select("question_id, answer_id").in_("question_id", [q["id"] for q in qs] if qs else [""]).execute().data or []
    return {"questions": qs, "answers": ans, "mappings": mappings}


@app.post("/org/qa")
def ingest_org_qa(payload: dict = Body(...), x_client_key: str | None = Header(default=None, alias="X-Client-Key"), rfp_id: str | None = Header(default=None, alias="X-RFP-ID")):
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
            q_emb = get_embedding(qtext)
            q_row = {
                "original_text": qtext,
                "normalized_text": qtext.lower(),
                "embedding": q_emb,
                "category": category,
                "client_id": client_id,
                "rfp_id": rfp_id,
            }
            q_ins = supabase.table("client_questions").insert(q_row).execute()
            q_id = (q_ins.data or [{}])[0].get("id")

            a_row = {
                "answer_text": atext,
                "answer_type": "General",
                "character_count": len(atext),
                "technical_level": 1,
                "client_id": client_id,
                "rfp_id": rfp_id,
            }
            a_ins = supabase.table("client_answers").insert(a_row).execute()
            a_id = (a_ins.data or [{}])[0].get("id")

            if q_id and a_id:
                supabase.table("client_question_answer_mappings").insert({
                    "question_id": q_id,
                    "answer_id": a_id,
                    "confidence_score": 1.0,
                    "context_requirements": None,
                    "stakeholder_approved": False,
                }).execute()
                created += 1
        except Exception as e:
            print(f"ingest_org_qa error: {e}")
            continue
    return {"created": created}


# --- Q/A extraction from uploaded RFPs (ZIP or XLSX) ---

def _extract_qa_pairs(sheet_content_csv: str) -> list:
    prompt = f"""
Analyze the following Excel sheet data and identify question-answer pairs.

- Detect rows where a question and its corresponding answer are present.
- Return them as a JSON array of objects in this exact format:
  [
    {{
      "question": "string",
      "answer": "string",
      "row": 0,
      "category": "Wifi | Map | Other"
    }}
  ]

Rules:
1. Do not include any explanation or extra text outside the JSON array.
2. If you are unsure about a category, default to "Other".
3. If no valid pairs are found, return an empty array [].

Sheet Data:
{sheet_content_csv}
"""
    try:
        response = gemini.generate_content(prompt)
        text = (response.text or "").strip()
        s, e = text.find("["), text.rfind("]")
        if s == -1 or e == -1:
            return []
        return json.loads(text[s:e+1])
    except Exception as e:
        print(f"extract_qa_pairs error: {e}")
        return []


def _insert_qa_pair(client_id: str, question_text: str, answer_text: str, category: str = "Other", rfp_id: str = None) -> bool:
    try:
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
    return False


@app.post("/org/qa/extract")
async def extract_qa_from_upload(file: UploadFile, x_client_key: str | None = Header(default=None, alias="X-Client-Key"), rfp_id: str | None = Header(default=None, alias="X-RFP-ID")):
    client_id = get_client_id_from_key(x_client_key)
    created = 0
    # Save uploaded file to temp
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
                    pairs = _extract_qa_pairs(sheet_csv) or []
                    for p in pairs:
                        q = (p.get("question") or "").strip()
                        a = (p.get("answer") or "").strip()
                        c = (p.get("category") or "Other").strip() or "Other"
                        if q and a and _insert_qa_pair(client_id, q, a, c, rfp_id):
                            created += 1
            except Exception as e:
                print(f"extract_qa_from_upload excel error: {e}")

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
        elif file.filename.lower().endswith(".xlsx"):
            _process_excel(tmp_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Upload .zip or .xlsx")

    return {"created": created}


# --- Quality scoring trigger ---

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
                val = int(round(float(data["score"])) )
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


@app.post("/org/qa/score")
def score_org_answers(payload: dict = Body(default={}), x_client_key: str | None = Header(default=None, alias="X-Client-Key"), rfp_id: str | None = Header(default=None, alias="X-RFP-ID")):
    client_id = get_client_id_from_key(x_client_key)
    limit = payload.get("limit")
    reference_text = payload.get("reference_text") or ""

    # fetch mappings and join QA
    try:
        rows = supabase.table("client_question_answer_mappings").select(
            "question_id, answer_id, client_questions(original_text, client_id, rfp_id), client_answers(answer_text, quality_score, client_id, rfp_id)"
        ).execute().data or []
    except Exception as e:
        print(f"score_org_answers fetch error: {e}")
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
    for r in scoped:
        question_text = (r.get("client_questions") or {}).get("original_text") or ""
        answer_id = r.get("answer_id")
        answer_text = (r.get("client_answers") or {}).get("answer_text") or ""
        if not answer_id or not question_text or not answer_text:
            continue

        prompt = _build_scoring_prompt(question_text, answer_text, reference_text)
        try:
            resp = gemini.generate_content(prompt)
            resp_text = resp.text or ""
        except Exception as e:
            print(f"score_org_answers LLM error: {e}")
            resp_text = ""

        score = _extract_score(resp_text)
        if score == -1:
            continue

        try:
            supabase.table("client_answers").update({"quality_score": int(score)}).eq("id", answer_id).execute()
            updated += 1
        except Exception as e:
            print(f"score_org_answers update error: {e}")

    return {"updated": updated}


# --- Q&A Management routes ---

@app.put("/questions/{question_id}")
def update_question(question_id: str, payload: dict = Body(...), x_client_key: str | None = Header(default=None, alias="X-Client-Key")):
    client_id = get_client_id_from_key(x_client_key)
    updates = {k: v for k, v in payload.items() if k in ("original_text", "category")}
    if "original_text" in updates:
        updates["normalized_text"] = updates["original_text"].lower()
    res = supabase.table("client_questions").update(updates).eq("id", question_id).eq("client_id", client_id).execute()
    return {"ok": True}

@app.delete("/questions/{question_id}")
def delete_question(question_id: str, x_client_key: str | None = Header(default=None, alias="X-Client-Key")):
    client_id = get_client_id_from_key(x_client_key)
    supabase.table("client_questions").delete().eq("id", question_id).eq("client_id", client_id).execute()
    return {"ok": True}

@app.put("/answers/{answer_id}")
def update_answer(answer_id: str, payload: dict = Body(...), x_client_key: str | None = Header(default=None, alias="X-Client-Key")):
    client_id = get_client_id_from_key(x_client_key)
    updates = {k: v for k, v in payload.items() if k in ("answer_text", "quality_score")}
    if "answer_text" in updates:
        updates["character_count"] = len(updates["answer_text"])
    res = supabase.table("client_answers").update(updates).eq("id", answer_id).eq("client_id", client_id).execute()
    return {"ok": True}

@app.delete("/answers/{answer_id}")
def delete_answer(answer_id: str, x_client_key: str | None = Header(default=None, alias="X-Client-Key")):
    client_id = get_client_id_from_key(x_client_key)
    supabase.table("client_answers").delete().eq("id", answer_id).eq("client_id", client_id).execute()
    return {"ok": True}


# --- Job Management System ---

def estimate_processing_time(file_size: int, job_type: str) -> int:
    """Estimate processing time in minutes based on file size and job type"""
    if job_type == "process_rfp":
        # Rough estimate: 1 minute per 100KB for RFP processing
        return max(5, min(30, file_size // 102400))
    elif job_type == "extract_qa":
        # Rough estimate: 1 minute per 200KB for QA extraction
        return max(3, min(20, file_size // 204800))
    return 10

def update_job_progress(job_id: str, progress: int, current_step: str, result_data: dict = None):
    """Update job progress in database"""
    try:
        updates = {
            "progress_percent": progress,
            "current_step": current_step
        }
        if result_data:
            updates["result_data"] = result_data
        if progress == 100:
            updates["status"] = "completed"
            updates["completed_at"] = datetime.now().isoformat()
        elif progress == -1:
            updates["status"] = "failed"
            updates["completed_at"] = datetime.now().isoformat()
        
        supabase.table("client_jobs").update(updates).eq("id", job_id).execute()
    except Exception as e:
        print(f"Error updating job progress: {e}")

def process_rfp_background(job_id: str, file_content: bytes, file_name: str, client_id: str, rfp_id: str):
    """Background RFP processing function"""
    try:
        update_job_progress(job_id, 10, "Starting RFP processing...")
        
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
        
        try:
            update_job_progress(job_id, 20, "Loading Excel file...")
            wb = openpyxl.load_workbook(tmp_path)
            
            total_sheets = len(wb.sheetnames)
            processed_sheets = 0
            extracted_questions = []
            
            for sheet_name in wb.sheetnames:
                update_job_progress(job_id, 20 + (processed_sheets * 60 // total_sheets), 
                                  f"Processing sheet: {sheet_name}")
                
                ws = wb[sheet_name]
                df = pd.DataFrame(ws.values)
                if df.empty:
                    processed_sheets += 1
                    continue
                
                sheet_text = df.to_string(index=False, header=False)
                questions = extract_questions_with_llm(sheet_text)
                
                for q in questions:
                    extracted_questions.append({
                        "question": q.get("question", ""),
                        "row": q.get("row", 0),
                        "sheet": sheet_name
                    })
                
                processed_sheets += 1
            
            update_job_progress(job_id, 80, "Generating AI answers...")
            
            # Process questions and generate answers
            ai_col = find_first_empty_column(ws)
            ws.cell(row=1, column=ai_col, value="AI Answers")
            
            for i, q in enumerate(extracted_questions):
                progress = 80 + (i * 15 // len(extracted_questions))
                update_job_progress(job_id, progress, f"Generating answer for question {i+1}/{len(extracted_questions)}")
                
                qtext = q.get("question", "")
                if not qtext:
                    continue
                
                emb = get_embedding(qtext)
                matches = search_supabase(emb, client_id, rfp_id)
                
                final_answer = "Not found, needs review."
                if matches:
                    best = pick_best_match(matches)
                    if best and best.get("similarity", 0) >= 0.9:
                        final_answer = best["answer"]
                    else:
                        final_answer = generate_tailored_answer(qtext, matches)
                
                # Store the result
                q["answer"] = clean_markdown(final_answer)
            
            # Save processed file
            output = io.BytesIO()
            wb.save(output)
            output.seek(0)
            
            result_data = {
                "extracted_questions": extracted_questions,
                "processed_file": output.getvalue().hex(),  # Store as hex for JSON
                "total_questions": len(extracted_questions)
            }
            
            update_job_progress(job_id, 100, "Processing completed!", result_data)
            
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        print(f"RFP processing error: {e}")
        update_job_progress(job_id, -1, f"Processing failed: {str(e)}")

def extract_qa_background(job_id: str, file_content: bytes, file_name: str, client_id: str, rfp_id: str):
    """Background QA extraction function"""
    try:
        update_job_progress(job_id, 10, "Starting QA extraction...")
        
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
        
        try:
            update_job_progress(job_id, 20, "Loading Excel file...")
            xls = pd.ExcelFile(tmp_path)
            
            total_sheets = len(xls.sheet_names)
            processed_sheets = 0
            extracted_pairs = []
            
            for sheet in xls.sheet_names:
                progress = 20 + (processed_sheets * 70 // total_sheets)
                update_job_progress(job_id, progress, f"Extracting from sheet: {sheet}")
                
                df = pd.read_excel(tmp_path, sheet_name=sheet, header=None)
                if df.empty:
                    processed_sheets += 1
                    continue
                
                sheet_csv = df.to_csv(index=False, header=False)
                pairs = _extract_qa_pairs(sheet_csv)
                
                for p in pairs:
                    extracted_pairs.append({
                        "question": p.get("question", ""),
                        "answer": p.get("answer", ""),
                        "category": p.get("category", "Other"),
                        "sheet": sheet
                    })
                
                processed_sheets += 1
            
            update_job_progress(job_id, 90, "Saving extracted Q&A pairs...")
            
            # Save to database
            created_count = 0
            for p in extracted_pairs:
                q = p.get("question", "").strip()
                a = p.get("answer", "").strip()
                c = p.get("category", "Other").strip() or "Other"
                if q and a and _insert_qa_pair(client_id, q, a, c, rfp_id):
                    created_count += 1
            
            result_data = {
                "extracted_pairs": extracted_pairs,
                "created_count": created_count,
                "total_pairs": len(extracted_pairs)
            }
            
            update_job_progress(job_id, 100, "QA extraction completed!", result_data)
            
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        print(f"QA extraction error: {e}")
        update_job_progress(job_id, -1, f"Extraction failed: {str(e)}")

def create_rfp_from_filename(client_id: str, filename: str) -> str:
    """Auto-create RFP record from filename"""
    # Extract RFP name from filename (remove extension and clean up)
    rfp_name = filename.rsplit('.', 1)[0]  # Remove file extension
    rfp_name = rfp_name.replace('_', ' ').replace('-', ' ')  # Replace underscores and dashes with spaces
    rfp_name = ' '.join(word.capitalize() for word in rfp_name.split())  # Title case
    
    # Check if RFP with similar name already exists
    existing_rfps = supabase.table("client_rfps").select("id, name").eq("client_id", client_id).execute()
    
    # Find exact match or similar name
    rfp_id = None
    for rfp in existing_rfps.data or []:
        if rfp["name"].lower() == rfp_name.lower():
            rfp_id = rfp["id"]
            break
    
    # Create new RFP if not found
    if not rfp_id:
        rfp_data = {
            "client_id": client_id,
            "name": rfp_name,
            "description": f"Auto-created from uploaded file: {filename}",
            "created_at": datetime.now().isoformat()
        }
        rfp_result = supabase.table("client_rfps").insert(rfp_data).execute()
        rfp_id = rfp_result.data[0]["id"]
    
    return rfp_id

@app.post("/jobs/submit")
async def submit_job(file: UploadFile, job_type: str, x_client_key: str | None = Header(default=None, alias="X-Client-Key")):
    """Submit a job for background processing - auto-creates RFP from filename"""
    client_id = get_client_id_from_key(x_client_key)
    
    if job_type not in ["process_rfp", "extract_qa"]:
        raise HTTPException(status_code=400, detail="Invalid job type")
    
    # Auto-create RFP from filename
    rfp_id = create_rfp_from_filename(client_id, file.filename)
    
    # Read file content
    file_content = await file.read()
    file_size = len(file_content)
    
    # Estimate processing time
    estimated_minutes = estimate_processing_time(file_size, job_type)
    estimated_completion = datetime.now() + timedelta(minutes=estimated_minutes)
    
    # Create job record
    job_data = {
        "client_id": client_id,
        "rfp_id": rfp_id,
        "job_type": job_type,
        "status": "pending",
        "file_name": file.filename,
        "file_size": file_size,
        "progress_percent": 0,
        "current_step": "Job queued for processing",
        "estimated_completion": estimated_completion.isoformat(),
        "created_at": datetime.now().isoformat()
    }
    
    job_result = supabase.table("client_jobs").insert(job_data).execute()
    job_id = job_result.data[0]["id"]
    
    # Start background processing
    if job_type == "process_rfp":
        thread = threading.Thread(target=process_rfp_background, args=(job_id, file_content, file.filename, client_id, rfp_id))
    else:  # extract_qa
        thread = threading.Thread(target=extract_qa_background, args=(job_id, file_content, file.filename, client_id, rfp_id))
    
    thread.daemon = True
    thread.start()
    
    return {"job_id": job_id, "rfp_id": rfp_id, "estimated_minutes": estimated_minutes, "status": "submitted"}

@app.get("/jobs")
def list_jobs(x_client_key: str | None = Header(default=None, alias="X-Client-Key")):
    """List all jobs for a client"""
    client_id = get_client_id_from_key(x_client_key)
    res = supabase.table("client_jobs").select("*").eq("client_id", client_id).order("created_at", desc=True).execute()
    return {"jobs": res.data or []}

@app.get("/jobs/{job_id}")
def get_job(job_id: str, x_client_key: str | None = Header(default=None, alias="X-Client-Key")):
    """Get specific job details"""
    client_id = get_client_id_from_key(x_client_key)
    res = supabase.table("client_jobs").select("*").eq("id", job_id).eq("client_id", client_id).single().execute()
    return res.data

@app.delete("/jobs/{job_id}")
def cancel_job(job_id: str, x_client_key: str | None = Header(default=None, alias="X-Client-Key")):
    """Cancel a pending job"""
    client_id = get_client_id_from_key(x_client_key)
    supabase.table("client_jobs").update({"status": "cancelled"}).eq("id", job_id).eq("client_id", client_id).execute()
    return {"ok": True}
