"""
Gemini AI Service - Question Detection and Answer Generation
"""
import json
import threading
import traceback
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from config.settings import GEMINI_MODEL

def get_embedding(text: str) -> list:
    """Generate embedding for text using Google's embedding-001 model"""
    if not text:
        print("DEBUG: Empty text provided to get_embedding")
        return []
    
    # Clean and prepare text
    text = text.strip()
    if len(text) < 3:
        print(f"DEBUG: Text too short for embedding: '{text}'")
        return []
    
    try:
        res = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_query"
        )
        embedding = res["embedding"]
        
        # Validate embedding
        if not embedding or not isinstance(embedding, list):
            print(f"ERROR: Invalid embedding returned for text: {text[:100]}...")
            return []
        
        if len(embedding) == 0:
            print(f"ERROR: Empty embedding returned for text: {text[:100]}...")
            return []
        
        print(f"DEBUG: Generated embedding (length {len(embedding)}) for: '{text[:60]}...'")
        return embedding
        
    except Exception as e:
        print(f"ERROR: Embedding generation failed: {e}")
        print(f"       Text: {text[:100]}...")
        traceback.print_exc()
        return []


def detect_questions_in_batch(rows_with_numbers: list) -> list:
    """
    Detect questions in a batch of rows using the EXACT proven TypeScript approach.
    
    INPUT:  [{"rowNumber": 5, "rowData": ["cell1", "cell2", ...]}, ...]
    OUTPUT: [{"rowNumber": 5, "question": "extracted question text"}, ...]
    
    KEY INSIGHT: We pass row numbers TO the AI, and AI returns the SAME row numbers.
    This eliminates all ambiguity - we know EXACTLY which row contains each question.
    
    This is a 1:1 translation of the TypeScript detectQuestionsInBatch function.
    """
    if not rows_with_numbers or len(rows_with_numbers) == 0:
        return []
    
    # EXACT schema from TypeScript version
    response_schema = {
        "type": "ARRAY",
        "description": "A list of rows that have been identified as questions.",
        "items": {
            "type": "OBJECT",
            "properties": {
                "rowNumber": {
                    "type": "NUMBER",
                    "description": "The original 1-based row number from the input that contains the question."
                },
                "question": {
                    "type": "STRING",
                    "description": "The exact and complete question text extracted from the row."
                }
            },
            "required": ["rowNumber", "question"]
        }
    }
    
    # EXACT prompt from TypeScript version (with proper indentation)
    prompt = f"""
    You are an RFP analysis expert. Your task is to identify which of the following spreadsheet rows are questions or vendor requirements.
    The data is provided as a JSON array, where each object has a 'rowNumber' and 'rowData'.

    Analyze each row's 'rowData'. If it contains a question or a requirement (e.g., "Describe your process...", "Vendor must provide..."), extract it.
    
    Your response MUST be a JSON array of objects, strictly conforming to the provided schema.
    Each object in your response array must correspond to a row that you identified as a question.
    It must include the EXACT ORIGINAL 'rowNumber' and the extracted 'question' text.
    
    CRITICAL RULES:
    1. Return the EXACT 'rowNumber' from the input - DO NOT modify or recalculate row numbers
    2. Only include rows that are actual questions or requirements in your response
    3. Ignore headers, section titles, and empty rows
    4. If no questions are found, return an empty array []
    5. The 'rowNumber' in your response MUST match the 'rowNumber' from the input exactly

    Spreadsheet rows to analyze:
    {json.dumps(rows_with_numbers)}
    """
    
    try:
        result = [None]
        error = [None]
        
        def api_call():
            try:
                # Use gemini-2.5-flash as in TypeScript version
                model = genai.GenerativeModel("gemini-2.5-flash")
                response = model.generate_content(
                    prompt,
                    generation_config={
                        "response_mime_type": "application/json",
                        "response_schema": response_schema
                    },
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    },
                )
                result[0] = response
            except Exception as e:
                error[0] = e
        
        thread = threading.Thread(target=api_call)
        thread.daemon = True
        thread.start()
        
        # 60-second timeout exactly as in TypeScript withTimeout
        thread.join(timeout=60)
        
        if thread.is_alive():
            print("Gemini returned empty response for batch (timeout)")
            return []
        
        if error[0]:
            print(f"Error processing batch with Gemini: {error[0]}")
            traceback.print_exc()
            raise error[0]
        
        if not result[0]:
            print("Gemini returned empty response for batch")
            return []
        
        text = (result[0].text or "").strip()
        if not text:
            print("Gemini returned empty response for batch")
            return []
        
        detected_questions = json.loads(text)
        print(f"DEBUG: Detected {len(detected_questions)} questions in batch")
        return detected_questions
        
    except Exception as e:
        print(f"Error processing batch with Gemini: {e}")
        traceback.print_exc()
        return []


def generate_tailored_answer(question: str, matches: list, gemini_model = None) -> str:
    """Generate a tailored answer based on similar Q&A pairs"""
    if gemini_model is None:
        gemini_model = genai.GenerativeModel(GEMINI_MODEL)
    
    context = "\n".join(
        f"- Q: {m['question']} | A: {m['answer']} (similarity {m['similarity']:.2f})"
        for m in matches)
    prompt = f"""
You are answering an RFP vendor question.  

New Question:
{question}

Reference Q&A Pairs:
{context}

Write a concise, tailored answer that best addresses the new question using the references.
If unclear, combine and adapt from references.
Return only the answer text, without any additional conversational filler or markdown formatting.
"""
    try:
        result = [None]
        error = [None]
        
        def api_call():
            try:
                resp = gemini_model.generate_content(
                    prompt,
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    },
                )
                result[0] = resp
            except Exception as e:
                error[0] = e
        
        thread = threading.Thread(target=api_call)
        thread.daemon = True
        thread.start()
        
        thread.join(timeout=45)
        
        if thread.is_alive():
            print("Gemini tailored answer: API call timed out after 45 seconds")
            return "AI could not generate tailored answer (timeout)."
        
        if error[0]:
            raise error[0]
        
        if not result[0]:
            print("Gemini tailored answer: No response received")
            return "AI could not generate tailored answer."
        
        return result[0].text.strip()
            
    except Exception as e:
        print(f"Gemini tailored answer error: {e} for question: {question[:100]}...")
        traceback.print_exc()
        return "AI could not generate tailored answer."


def extract_questions_with_gemini(sheet_text: str, gemini_model = None) -> list:
    """Extract questions from sheet text using Gemini (legacy method)"""
    if gemini_model is None:
        gemini_model = genai.GenerativeModel(GEMINI_MODEL)
    
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
        result = [None]
        error = [None]
        
        def api_call():
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
                result[0] = response
            except Exception as e:
                error[0] = e
        
        thread = threading.Thread(target=api_call)
        thread.daemon = True
        thread.start()
        
        thread.join(timeout=60) 
        
        if thread.is_alive():
            print("Gemini extract: API call timed out after 60 seconds")
            return []
        
        if error[0]:
            raise error[0]
        
        if not result[0]:
            print("Gemini extract: No response received")
            return []
        
        text = (result[0].text or "").strip()
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end == -1:
            print(f"Gemini extract: No JSON array found in response for text: {text[:200]}...")
            return []
        return json.loads(text[start:end+1])
            
    except Exception as e:
        print(f"Gemini extract error: {e}")
        traceback.print_exc()
        return []


def extract_qa_pairs_from_sheet(sheet_content_csv: str, gemini_model = None) -> list:
    """Extract Q&A pairs from sheet content"""
    if gemini_model is None:
        gemini_model = genai.GenerativeModel(GEMINI_MODEL)
    
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
        response = gemini_model.generate_content(prompt)
        text = (response.text or "").strip()
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end == -1:
            return []
        pairs = json.loads(text[start:end+1])
        # If answer exists but question missing, synthesize a fallback question
        normalized = []
        for p in pairs:
            q = (p.get("question") or "").strip()
            a = (p.get("answer") or "").strip()
            if not a:
                continue
            if not q:
                q = "Auto-generated question from answer"
            normalized.append({
                "question": q,
                "answer": a,
                "row": p.get("row", 0),
                "category": (p.get("category") or "Other").strip() or "Other",
            })
        return normalized
    except Exception as e:
        print(f"extract_qa_pairs error: {e}")
        traceback.print_exc()
        return []


def summarize_tender_for_paid_view(tender_core: dict, full_payload: dict | str | None, gemini_model=None) -> dict:
    """
    Use Gemini to produce a structured, human-readable summary for paid tender access.
    Returns a dictionary with sections such as executive summary, scope, timeline, etc.
    """
    if gemini_model is None:
        gemini_model = genai.GenerativeModel(GEMINI_MODEL)

    payload = {
        "core_fields": tender_core,
        "raw_data": full_payload,
    }

    serialized = json.dumps(payload, ensure_ascii=False, default=str)
    if len(serialized) > 60000:
        serialized = serialized[:60000]

    response_schema = {
        "type": "OBJECT",
        "properties": {
            "executive_summary": {"type": "STRING"},
            "key_highlights": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
            },
            "buyer": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "role": {"type": "STRING"},
                    "address": {"type": "STRING"},
                    "contact": {"type": "STRING"},
                },
            },
            "scope_of_work": {"type": "STRING"},
            "deliverables": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
            },
            "timeline": {
                "type": "OBJECT",
                "properties": {
                    "submission_deadline": {"type": "STRING"},
                    "contract_start": {"type": "STRING"},
                    "contract_end": {"type": "STRING"},
                    "milestones": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    },
                },
            },
            "budget": {
                "type": "OBJECT",
                "properties": {
                    "value": {"type": "STRING"},
                    "currency": {"type": "STRING"},
                    "notes": {"type": "STRING"},
                },
            },
            "documents": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "title": {"type": "STRING"},
                        "url": {"type": "STRING"},
                        "notes": {"type": "STRING"},
                    },
                },
            },
            "compliance_requirements": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
            },
            "evaluation_criteria": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
            },
            "risks": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
            },
        },
        "required": ["executive_summary"],
    }

    prompt = f"""
You are preparing a structured briefing for a paying client who has unlocked full access to a public procurement notice.

Using the data provided, produce a concise, business-friendly JSON object that helps the client quickly understand the opportunity.
Focus on clarity, professional tone, and actionable detail. Summaries must be in English even if the source data is another language.

Input data (truncated if very large):
{serialized}
"""

    try:
        result = gemini_model.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": response_schema,
                "temperature": 0.3,
            },
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            },
        )
        text = (result.text or "").strip()
        if not text:
            return {}
        return json.loads(text)
    except Exception as e:
        print(f"summarize_tender_for_paid_view error: {e}")
        traceback.print_exc()
        return {}
