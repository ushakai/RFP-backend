import os
import io
import json
import difflib
import re
import pandas as pd
import openpyxl
from fastapi import FastAPI, UploadFile, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from supabase import create_client, Client
from openpyxl.styles import Alignment
from openpyxl.utils import get_column_letter

# CONFIGURATION 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize clients
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")
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

def extract_questions_with_gemini(sheet_text: str) -> list:
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
        response = gemini_model.generate_content(
            prompt,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            },
        )
        text = response.text.strip()
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end == -1:
            return []
        return json.loads(text[start:end+1])
    except Exception as e:
        print(f"Gemini error: {e}")
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
        res = genai.embed_content(model="models/embedding-001", content=text, task_type="retrieval_query")
        return res["embedding"]
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
                "client_id": client_id,
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
        resp = gemini_model.generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        print(f"Gemini tailored answer error: {e}")
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

        extracted = extract_questions_with_gemini(sheet_text)
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
