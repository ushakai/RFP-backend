"""
Score QA quality against Purple.ai scraped reference using AI,
and write numeric 0-100 scores to wifi_answers.quality_score in Supabase.
"""

import re
import time
import json
import math
from typing import List, Dict, Any

from openai import OpenAI
from supabase import create_client, Client
from dotenv import load_dotenv


# --- CONFIGURATION ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# configure OpenAI & supabase
openai_client = OpenAI(api_key=OPENAI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Paste the scraped Purple.ai reference (from your message) as a string.
PURPLE_REFERENCE = """
Purple.ai (Purple) — company overview and public info summary:
- Website & blog: purple.ai (guest WiFi, indoor maps & location services, analytics).
- Public pages: blog/insights, legal/privacy, support.
- Partner/Portal and Portal API / Postman references exist behind support portal (support.purple.ai).
- Relevant areas to validate: Portal API (OpenAPI/Postman), Webhooks (schemas/signatures), Data dictionary (venue, map, session, analytics), Security & compliance, SLAs, versioning, logging/observability, access control.
- Gaps: no public API docs, no public SLAs, no public whitepapers or schema.
- Use these items as validation anchors: "API specification", "OpenAPI / Postman", "webhooks schema", "data dictionary / entity models", "security & compliance", "SLA / rate limits", "auth / OAuth", "monitoring logs", "connector mappings".
(End of scraped reference)
"""

# ----------------------------
# Helper functions
# ----------------------------

def fetch_all_mappings() -> List[Dict[str, Any]]:
    """
    Fetch all question-answer mappings, returning rows with:
    mapping.question_id, mapping.answer_id, wifi_questions.original_text, wifi_answers.answer_text, wifi_answers.quality_score
    """
    try:
        # Select mappings and join corresponding question + answer fields
        # The select string uses PostgREST embedding for relationships; adjust if your schema differs.
        res = supabase.table("question_answer_mappings")\
            .select("question_id, answer_id, wifi_questions(original_text), wifi_answers(answer_text, quality_score)")\
            .execute()
        return res.data if res.data else []
    except Exception as e:
        print(f"[ERROR] Failed to fetch mappings from Supabase: {e}")
        return []

def build_scoring_prompt(question: str, answer: str, reference: str) -> str:
    """
    Build a robust scoring prompt that asks AI to evaluate answer quality from 0-100.
    The prompt instructs strict JSON output with numeric score and short rationale.
    """
    prompt = f"""
    You are an expert RFP evaluator with domain knowledge of guest WiFi, indoor mapping, analytics, privacy, and integration APIs
    (such as Purple.ai). Given the following:

    1) The RFP question (what was asked).
    2) A vendor's answer (the answer we want to evaluate).
    3) Reference information about Purple.ai (public facts, domain knowledge, and benchmarks).

    Your task:
    - Evaluate the vendor's answer quality in the context of the QUESTION and the REFERENCE.
    - Give a single numeric score between 0 and 100 (100 = perfect, fully correct, complete, and aligned to reference; 0 = totally incorrect or irrelevant).
    - Consider criteria (but do not limit to): factual correctness relative to reference, completeness (covers required aspects), use of correct domain terms (APIs/webhooks/data schema/SLAs/privacy), specificity (provides concrete endpoints, fields, numbers, policies when asked), and overall plausibility/consistency.
    - If the question asked for an API/spec/schema/privacy control and the answer does not reference one or omits required details, penalize heavily.
    - If the answer is a reasonable adaptation aligned with reference and industry expectations, reward accordingly.

    Return JSON ONLY in this exact format (no extra text, no markdown):
    {{ "score": <integer 0-100>, "notes": "<one-sentence rationale (20-200 chars)>" }}

    QUESTION:
    \"\"\"{question}\"\"\"

    VENDOR ANSWER:
    \"\"\"{answer}\"\"\"

    REFERENCE (Purple.ai public info & benchmarks):
    \"\"\" 
    Purple.ai provides guest WiFi, indoor mapping, and analytics solutions. Key technical and privacy benchmarks include:

    - **Data collection transparency**: Collects personal info (name, email, DOB, device MAC, logs, location, browsing behavior). Vendors must clearly state categories collected.  
    - **Data usage & purpose**: Used for service delivery, analytics, security, marketing (with consent). Vendors should specify exact purposes and anonymization practices.  
    - **Data storage locations**: Data stored in region-based data centers (EU, US, AU) with optional local hosting for compliance. Vendors should state storage regions and options.  
    - **Retention / deletion**: PII is deleted after 13 months of inactivity. Vendors should state retention rules and secure disposal methods.  
    - **Sharing / third parties**: Shared with venues, providers, and lawful authorities. Vendors must list third parties, roles, and legal safeguards.  
    - **User rights / compliance**: Supports GDPR rights (access, erase, object, restrict, portability). Vendors must show how user rights are supported.  
    - **Security measures**: Protocols for encryption, access control, monitoring, secure deletion. Vendors should describe security standards.  
    - **Consent / opt-out**: Marketing requires consent; opt-out available. Vendors must describe consent and opt-out management.  
    - **Cross-venue / data segregation**: WiFi registration data is often managed by venues; Purple enforces boundaries. Vendors should explain tenant-level segregation.  
    - **APIs & integrations**: Provides Portal API (v1.6–1.7), webhooks, connectors, supports CRM/analytics integration. Vendors should reference APIs, endpoints, schemas, and error handling.  

    Use this information as the factual and benchmark reference when scoring the vendor's response.
    \"\"\"

    Important:
    - Output valid JSON only. The "score" value must be an integer between 0 and 100.
    - Keep "notes" concise (one sentence).
    """

    return prompt

def extract_score_from_response(response_text: str) -> int:
    """
    Parse numeric score out of model response text. Return integer in [0,100] or -1 on failure.
    """
    if not response_text or not isinstance(response_text, str):
        return -1
    # try to load JSON if present
    try:
        # find first '{' ... '}' substring
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_part = response_text[start:end+1]
            data = json.loads(json_part)
            if "score" in data:
                sc = int(round(float(data["score"])))
                return max(0, min(100, sc))
    except Exception:
        pass

    # fallback: find first integer or decimal in text
    m = re.search(r"(\d{1,3})(?:\.\d+)?", response_text)
    if m:
        try:
            val = int(round(float(m.group(1))))
            return max(0, min(100, val))
        except Exception:
            pass

    return -1

def update_quality_score(answer_id: str, score: int) -> bool:
    """
    Update wifi_answers.quality_score in Supabase for given answer_id.
    """
    try:
        res = supabase.table("wifi_answers").update({"quality_score": score}).eq("id", answer_id).execute()
        # res.status_code or res.data can be inspected; assume success if no exception
        return True
    except Exception as e:
        print(f"[ERROR] Failed to update quality_score for answer {answer_id}: {e}")
        return False

# ----------------------------
# Main scoring loop
# ----------------------------

def score_all_answers(reference_text: str, min_delay_seconds: float = 0.5, limit: int = None):
    """
    Fetch mappings, ask AI to evaluate each answer, and update Supabase quality_score.
    If limit is provided, only process that many mappings.
    """
    mappings = fetch_all_mappings()
    if not mappings:
        print("[INFO] No mappings found to score.")
        return

    if limit:
        mappings = mappings[:limit]

    print(f"[INFO] Found {len(mappings)} mapping rows to evaluate (limit={limit}).")

    for idx, row in enumerate(mappings, start=1):
        try:
            question_id = row.get("question_id")
            answer_id = row.get("answer_id")
            # joined fields may return nested objects per PostgREST structure
            question_text = None
            answer_text = None
            # wifi_questions(original_text) comes back as a dict under key "wifi_questions"
            if "wifi_questions" in row and isinstance(row["wifi_questions"], dict):
                question_text = row["wifi_questions"].get("original_text") or row["wifi_questions"].get("question") or None
            # wifi_answers(answer_text) similarly
            if "wifi_answers" in row and isinstance(row["wifi_answers"], dict):
                answer_text = row["wifi_answers"].get("answer_text") or row["wifi_answers"].get("answer") or None

            # fallback to other possible keys
            if not question_text:
                question_text = row.get("question") or row.get("original_text") or ""
            if not answer_text:
                answer_text = row.get("answer") or row.get("answer_text") or ""

            if not answer_id:
                print(f"[WARN] mapping row #{idx} missing answer_id; skipping.")
                continue

            if not question_text or not answer_text:
                print(f"[WARN] mapping #{idx} missing QA text (qid={question_id}, aid={answer_id}); skipping.")
                continue

            print(f"[{idx}/{len(mappings)}] Scoring answer_id={answer_id} (question_id={question_id})")

            prompt = build_scoring_prompt(question_text, answer_text, reference_text)

            # call AI with safety settings; retry on rate limit
            retry = 0
            while True:
                try:
            resp = openai_client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": "Return only a single JSON object with integer score and concise notes."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                )
            resp_text = resp.choices[0].message.content if hasattr(resp, "choices") else ""
                    break
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str and retry < 3:
                        wait = 60 * (retry + 1)
                        print(f"[RATE LIMIT] sleeping {wait}s then retrying...")
                        time.sleep(wait)
                        retry += 1
                        continue
                    print(f"[ERROR] AI call failed for answer {answer_id}: {e}")
                    resp_text = ""
                    break

            if not resp_text:
                print(f"[WARN] No response text for answer {answer_id}; writing -1 (skip).")
                continue

            score = extract_score_from_response(resp_text)
            if score == -1:
                print(f"[WARN] Could not parse score from response for answer {answer_id}. Raw response snippet:\n{resp_text[:400]}")
                # fallback strategy: write 0 or skip; here we skip updating but continue
                continue

            # clamp
            score = max(0, min(100, int(score)))
            print(f" -> Parsed score: {score} for answer_id {answer_id}")

            ok = update_quality_score(answer_id, score)
            if ok:
                print(f" -> Updated quality_score={score} for answer_id {answer_id}")
            else:
                print(f" -> Failed to update quality_score for answer_id {answer_id}")

            # small delay to avoid hammering services
            time.sleep(min_delay_seconds)

        except Exception as outer_e:
            print(f"[ERROR] Unexpected error while processing row #{idx}: {outer_e}")

    print("[DONE] Completed scoring all mappings.")

# ----------------------------
# Run as script
# ----------------------------
if __name__ == "__main__":
    # If you want to use a different reference text, replace variable below.
    reference_text = PURPLE_REFERENCE
    score_all_answers(reference_text, min_delay_seconds=0.3,limit=50)
