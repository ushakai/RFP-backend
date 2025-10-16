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
from openai import OpenAI
import zipfile
import tempfile
from dotenv import load_dotenv

# CONFIGURATION 
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
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
        response = openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs only valid JSON when asked."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        text = (response.choices[0].message.content or "").strip()
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
        emb = openai_client.embeddings.create(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            input=text,
        )
        return emb.data[0].embedding
    except Exception as e:
        print(f"Embedding error: {e}")
        return []


def search_supabase(question_embedding: list, client_id: str) -> list:
    if not question_embedding:
        return []
    try:
        res = supabase.rpc(
            "match_wifi_questions",
            {
                "query_embedding": question_embedding,
                "match_threshold": 0.0,
                "match_count": 5,
                "p_client_id": client_id,
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
        resp = openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You are a precise RFP assistant. Keep answers concise and professional."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return (resp.choices[0].message.content or "").strip()
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
def process_excel_file_obj(file_obj: UploadFile, client_id: str):
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
            matches = search_supabase(emb, client_id)

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
async def process_excel(file: UploadFile, x_client_key: str | None = Header(default=None, alias="X-Client-Key")):
    client_id = get_client_id_from_key(x_client_key)
    processed_file = process_excel_file_obj(file, client_id)
    return StreamingResponse(
        processed_file,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=processed_{file.filename}"}
    )


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
def list_org_qa(x_client_key: str | None = Header(default=None, alias="X-Client-Key")):
    client_id = get_client_id_from_key(x_client_key)
    qs = supabase.table("wifi_questions").select("id, original_text, category, created_at").eq("client_id", client_id).order("created_at", desc=True).execute().data or []
    ans = supabase.table("wifi_answers").select("id, answer_text, category, quality_score, last_updated").eq("client_id", client_id).order("last_updated", desc=True).execute().data or []
    mappings = supabase.table("question_answer_mappings").select("question_id, answer_id").in_("question_id", [q["id"] for q in qs] if qs else [""]).execute().data or []
    return {"questions": qs, "answers": ans, "mappings": mappings}


@app.post("/org/qa")
def ingest_org_qa(payload: dict = Body(...), x_client_key: str | None = Header(default=None, alias="X-Client-Key")):
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
            }
            q_ins = supabase.table("wifi_questions").insert(q_row).execute()
            q_id = (q_ins.data or [{}])[0].get("id")

            a_row = {
                "answer_text": atext,
                "answer_type": "General",
                "character_count": len(atext),
                "technical_level": 1,
                "client_id": client_id,
            }
            a_ins = supabase.table("wifi_answers").insert(a_row).execute()
            a_id = (a_ins.data or [{}])[0].get("id")

            if q_id and a_id:
                supabase.table("question_answer_mappings").insert({
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
        response = openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "Respond with JSON arrays only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        text = (response.choices[0].message.content or "").strip()
        s, e = text.find("["), text.rfind("]")
        if s == -1 or e == -1:
            return []
        return json.loads(text[s:e+1])
    except Exception as e:
        print(f"extract_qa_pairs error: {e}")
        return []


def _insert_qa_pair(client_id: str, question_text: str, answer_text: str, category: str = "Other") -> bool:
    try:
        q_emb = get_embedding(question_text)
        q_ins = supabase.table("wifi_questions").insert({
            "original_text": question_text,
            "normalized_text": (question_text or "").lower(),
            "embedding": q_emb,
            "category": category or "Other",
            "client_id": client_id,
        }).execute()
        q_id = (q_ins.data or [{}])[0].get("id")

        a_ins = supabase.table("wifi_answers").insert({
            "answer_text": answer_text,
            "answer_type": "General",
            "character_count": len(answer_text or ""),
            "technical_level": 1,
            "client_id": client_id,
        }).execute()
        a_id = (a_ins.data or [{}])[0].get("id")

        if q_id and a_id:
            supabase.table("question_answer_mappings").insert({
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
async def extract_qa_from_upload(file: UploadFile, x_client_key: str | None = Header(default=None, alias="X-Client-Key")):
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
                        if q and a and _insert_qa_pair(client_id, q, a, c):
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
def score_org_answers(payload: dict = Body(default={}), x_client_key: str | None = Header(default=None, alias="X-Client-Key")):
    client_id = get_client_id_from_key(x_client_key)
    limit = payload.get("limit")
    reference_text = payload.get("reference_text") or ""

    # fetch mappings and join QA
    try:
        rows = supabase.table("question_answer_mappings").select(
            "question_id, answer_id, wifi_questions(original_text, client_id), wifi_answers(answer_text, quality_score, client_id)"
        ).execute().data or []
    except Exception as e:
        print(f"score_org_answers fetch error: {e}")
        rows = []

    # filter to this org
    scoped = [r for r in rows if (r.get("wifi_questions") or {}).get("client_id") == client_id and (r.get("wifi_answers") or {}).get("client_id") == client_id]

    if limit:
        scoped = scoped[: int(limit)]

    updated = 0
    for r in scoped:
        question_text = (r.get("wifi_questions") or {}).get("original_text") or ""
        answer_id = r.get("answer_id")
        answer_text = (r.get("wifi_answers") or {}).get("answer_text") or ""
        if not answer_id or not question_text or not answer_text:
            continue

        prompt = _build_scoring_prompt(question_text, answer_text, reference_text)
        try:
            resp = openai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "Return only a single JSON object with an integer score and concise notes."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            resp_text = resp.choices[0].message.content or ""
        except Exception as e:
            print(f"score_org_answers LLM error: {e}")
            resp_text = ""

        score = _extract_score(resp_text)
        if score == -1:
            continue

        try:
            supabase.table("wifi_answers").update({"quality_score": int(score)}).eq("id", answer_id).execute()
            updated += 1
        except Exception as e:
            print(f"score_org_answers update error: {e}")

    return {"updated": updated}
