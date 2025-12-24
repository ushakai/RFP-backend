"""
Gemini AI Service - Question Detection and Answer Generation
"""
import json
import threading
import traceback
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from config.settings import GEMINI_MODEL

# Circuit breaker for AI keyword enhancement
_ai_enhancement_failures = 0
_ai_enhancement_disabled_until = None
MAX_CONSECUTIVE_FAILURES = 5
CIRCUIT_BREAKER_TIMEOUT_SECONDS = 300  # 5 minutes

def enhance_keywords_with_ai(keywords: list[str]) -> list[str]:
    """
    Use AI to enhance keywords by generating synonyms, related terms, and variations.
    This helps improve tender matching accuracy.
    
    Args:
        keywords: List of original keywords
        
    Returns:
        Enhanced list of keywords including original + AI-generated variations
    """
    global _ai_enhancement_failures, _ai_enhancement_disabled_until
    
    if not keywords or len(keywords) == 0:
        return []
    
    # Check circuit breaker - disable AI if too many failures
    if _ai_enhancement_disabled_until:
        from datetime import datetime, timezone
        if datetime.now(timezone.utc) < _ai_enhancement_disabled_until:
            # Circuit breaker is open - skip AI enhancement
            return []
        else:
            # Circuit breaker timeout expired - reset
            _ai_enhancement_disabled_until = None
            _ai_enhancement_failures = 0
    
    # If we've had too many failures, disable for a period
    if _ai_enhancement_failures >= MAX_CONSECUTIVE_FAILURES:
        from datetime import datetime, timezone, timedelta
        _ai_enhancement_disabled_until = datetime.now(timezone.utc) + timedelta(seconds=CIRCUIT_BREAKER_TIMEOUT_SECONDS)
        return []
    
    try:
        gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        
        # Limit to 10 keywords to avoid long prompts and reduce processing time
        keywords_str = ", ".join(keywords[:10])
        prompt = f"""Given these tender search keywords: {keywords_str}

Generate a comprehensive list of related terms, synonyms, and variations that would help match relevant tenders. Include:
1. The original keywords
2. Common synonyms and alternative terms
3. Related technical terms
4. Common abbreviations
5. Variations in spelling/formatting

Return ONLY a comma-separated list of keywords (no explanations, no formatting, just keywords separated by commas).
Keep the list focused and relevant - maximum 20 keywords total."""

        # Use request_options for timeout (5 seconds max)
        response = gemini_model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 300,
            },
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            },
            request_options={"timeout": 5.0}  # 5 second timeout
        )
        
        # Check if response was blocked by safety filters
        if not response.candidates or len(response.candidates) == 0:
            raise ValueError("Response was blocked by safety filters")
        
        candidate = response.candidates[0]
        if candidate.safety_ratings:
            # Check if any safety rating indicates blocking
            for rating in candidate.safety_ratings:
                if rating.probability.name in ["HIGH", "MEDIUM"]:
                    raise ValueError(f"Response blocked by safety filter: {rating.category.name}")
        
        # Check if candidate has content
        if not candidate.content or not candidate.content.parts:
            raise ValueError("Response has no content parts")
        
        enhanced_text = response.text.strip()
        if not enhanced_text:
            raise ValueError("Response text is empty")
        
        # Clean up the response - remove any markdown, extra formatting
        enhanced_text = enhanced_text.replace("*", "").replace("-", ",").replace("\n", ",")
        
        # Parse comma-separated keywords
        enhanced_keywords = []
        for kw in enhanced_text.split(","):
            kw_clean = kw.strip().lower()
            if kw_clean and len(kw_clean) > 2:  # Only keep meaningful keywords
                enhanced_keywords.append(kw_clean)
        
        # Combine original and enhanced, remove duplicates
        all_keywords = keywords + enhanced_keywords
        seen = set()
        unique_keywords = []
        for kw in all_keywords:
            kw_lower = kw.lower().strip()
            if kw_lower and kw_lower not in seen:
                seen.add(kw_lower)
                unique_keywords.append(kw_lower)
        
        # Limit to reasonable number
        # Reset failure counter on success
        _ai_enhancement_failures = 0
        return unique_keywords[:30]
        
    except ValueError as e:
        # Safety filter blocking or empty response - silently fall back to original keywords
        # Don't log these as they're expected for certain keyword combinations
        _ai_enhancement_failures += 1
        return [kw.lower().strip() for kw in keywords if kw and kw.strip()]
    except Exception as e:
        # Other errors - increment failure counter
        _ai_enhancement_failures += 1
        error_msg = str(e)
        # Only log non-safety errors and only occasionally to reduce noise
        if "safety" not in error_msg.lower() and "blocked" not in error_msg.lower() and _ai_enhancement_failures == 1:
            print(f"Warning: AI keyword enhancement failed (failures: {_ai_enhancement_failures}): {error_msg[:100]}")
        # Return original keywords if AI fails
        return [kw.lower().strip() for kw in keywords if kw and kw.strip()]


def get_embedding(text: str) -> list:
    """Generate embedding for text using the configured embedding model.
    
    IMPORTANT: This model MUST match the model used for document ingestion
    to ensure compatible embedding spaces for similarity search.
    """
    from config.rag_config import GEMINI_EMBEDDING_MODEL
    
    if not text:
        print("DEBUG: Empty text provided to get_embedding")
        return []
    
    # Clean and prepare text
    text = text.strip()
    if len(text) < 3:
        print(f"DEBUG: Text too short for embedding: '{text}'")
        return []
    
    try:
        # Use the same model as document ingestion for compatible embeddings
        res = genai.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
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
        
        print(f"DEBUG: Generated embedding (model={GEMINI_EMBEDDING_MODEL}, length {len(embedding)}) for: '{text[:60]}...'")
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
    """Generate a tailored answer based on similar Q&A pairs and document chunks
    
    Matches can now include both Q&A pairs (source='qa') and document chunks (source='document')
    """
    if gemini_model is None:
        gemini_model = genai.GenerativeModel(GEMINI_MODEL)
    
    # Format context with source type indicators and dates
    context_lines = []
    for m in matches:
        source_type = m.get('source', 'qa')
        source_label = '[DOC]' if source_type == 'document' else '[Q&A]'
        similarity = m.get('similarity', 0)
        date = m.get('original_rfp_date', 'N/A')
        question_text = m.get('question', '')
        answer_text = m.get('answer', '')
        
        context_lines.append(
            f"- {source_label} Q: {question_text} | A: {answer_text} "
            f"(similarity {similarity:.2f}, date: {date})"
        )
    
    context = "\n".join(context_lines)
    
    prompt = f"""
You are answering an RFP vendor question.  

New Question:
{question}

Reference Sources (Q&A Pairs and Document Chunks):
{context}

Write a concise, tailored answer that best addresses the new question using the references.
- [Q&A] sources are previous question-answer pairs from our knowledge base
- [DOC] sources are relevant excerpts from uploaded documents
- More recent dates indicate more current information
- Combine and adapt from references as needed

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


def summarize_tender_for_paid_view(tender_core: dict, full_payload: dict | str | None, gemini_model=None, matched_keywords: list[str] | None = None) -> dict:
    """
    Use Gemini to produce a structured, human-readable summary for paid tender access.
    Returns a dictionary with sections such as executive summary, scope, timeline, etc.
    
    Args:
        tender_core: Core tender fields (title, description, etc.)
        full_payload: Full tender data from source
        gemini_model: Optional Gemini model instance
        matched_keywords: List of keywords that matched this tender (for AI focus)
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

    # Build prompt with matched keywords context
    keywords_context = ""
    if matched_keywords and len(matched_keywords) > 0:
        keywords_str = ", ".join(matched_keywords)
        keywords_context = f"\n\nIMPORTANT: This tender matched the following keywords: {keywords_str}. Please pay special attention to these areas in your summary and highlight how the tender relates to these keywords."

    prompt = f"""
You are preparing a structured briefing for a paying client who has unlocked full access to a public procurement notice.

Using the data provided, produce a concise, business-friendly JSON object that helps the client quickly understand the opportunity.
Focus on clarity, professional tone, and actionable detail. Summaries must be in English even if the source data is another language.
{keywords_context}

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
