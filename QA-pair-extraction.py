import os
import zipfile
import pandas as pd
import hashlib
import warnings
import json
import time
from supabase import create_client, Client
from openai import OpenAI

warnings.filterwarnings("ignore")

"""
ENV required:
  OPENAI_API_KEY, SUPABASE_URL, SUPABASE_KEY, OPENAI_MODEL (optional), OPENAI_EMBEDDING_MODEL (optional)
"""

# ----------------------------
# CONFIGURATION
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize OpenAI & Supabase
openai_client = OpenAI(api_key=OPENAI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Keyword lists for fallback categorization
wifi_keywords = [
    "wifi", "wi-fi", "wireless", "access point", "signal",
    "bandwidth", "router", "network", "internet", "connectivity"
]

map_keywords = [
    "map", "gps", "location", "route", "wayfinding",
    "turn-by-turn", "navigation", "coordinates", "directions"
]

# ----------------------------
# UTILITIES
# ----------------------------

def generate_id(text: str) -> str:
    """Generate deterministic hash ID."""
    return hashlib.sha256(text.strip().lower().encode('utf-8')).hexdigest()

def generate_embedding_gpt(text: str) -> list:
    """Generate embeddings using OpenAI."""
    if not text or not isinstance(text, str):
        return []
    try:
        response = openai_client.embeddings.create(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
            input=text,
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return []

def categorize_question(question_text: str) -> str:
    """Categorize the question into Wifi, Map, or Other."""
    text = question_text.lower()
    if any(k in text for k in wifi_keywords):
        return "Wifi"
    elif any(k in text for k in map_keywords):
        return "Map"
    else:
        return "Other"

# ----------------------------
# GPT: EXTRACT Q/A PAIRS
# ----------------------------

def extract_qa_pairs(sheet_content_csv: str) -> list:
    """
    Uses Gemini to extract question-answer pairs from the sheet.
    Returns a JSON array with question, answer, row, and category.
    """
    prompt = f"""
    Analyze the following Excel sheet data and identify **question-answer pairs**.

    - Detect rows where a question and its corresponding answer are present.
    - Return them as a JSON array of objects in this exact format:
      [
        {{
          "question": "string",
          "answer": "string",
          "row": number,
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
                {"role": "system", "content": "Return only valid JSON arrays when asked."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        resp_text = (response.choices[0].message.content or "").strip()
        if not resp_text:
            print("⚠ LLM returned empty response.")
            return []

        # Extract JSON
        start_index = resp_text.find('[')
        end_index = resp_text.rfind(']')
        if start_index == -1 or end_index == -1:
            print("⚠ Could not find JSON array in response.")
            return []

        json_str = resp_text[start_index:end_index + 1]
        try:
            qa_pairs = json.loads(json_str)
            return qa_pairs
        except json.JSONDecodeError:
            print(f"❌ JSON Decode Error. Raw response:\n{json_str}")
            return []

    except Exception as e:
        if "429" in str(e):
            print("⏳ Rate limit reached. Retrying in 60 seconds...")
            time.sleep(60)
            return extract_qa_pairs(sheet_content_csv)
        print(f"❌ Error extracting QA pairs: {e}")
        return []

# ----------------------------
# SUPABASE OPERATIONS
# ----------------------------

def upsert_question(question_text, category, client_id=None):
    """Insert question into Supabase if not already present."""
    q_id = generate_id(question_text)

    exists = supabase.table("wifi_questions").select("id").eq("id", q_id).execute()
    if len(exists.data) == 0:
        embedding = generate_embedding_gpt(question_text)
        supabase.table("wifi_questions").insert({
            "id": q_id,
            "original_text": question_text,
            "normalized_text": question_text.lower(),
            "embedding": embedding,
            "category": category,
            "client_id": client_id,
        }).execute()
        print(f"✅ Inserted question: {question_text[:60]}... | Category: {category}")
    return q_id

def upsert_answer(answer_text, client_id=None):
    """Insert answer into Supabase if not already present."""
    a_id = generate_id(answer_text)

    exists = supabase.table("wifi_answers").select("id").eq("id", a_id).execute()
    if len(exists.data) == 0:
        supabase.table("wifi_answers").insert({
            "id": a_id,
            "answer_text": answer_text,
            "answer_type": "General",
            "character_count": len(answer_text),
            "technical_level": 1,
            "client_id": client_id,
        }).execute()
        print(f"✅ Inserted answer: {answer_text[:60]}...")
    return a_id

def upsert_mapping(question_id, answer_id):
    """Link question and answer in Supabase."""
    existing = supabase.table("question_answer_mappings")\
        .select("id")\
        .eq("question_id", question_id)\
        .eq("answer_id", answer_id)\
        .execute()

    if len(existing.data) == 0:
        supabase.table("question_answer_mappings").insert({
            "question_id": question_id,
            "answer_id": answer_id,
            "confidence_score": 1.0,
            "context_requirements": None,
            "stakeholder_approved": False
        }).execute()
        print(f"🔗 Linked Q:{question_id} <-> A:{answer_id}")

# ----------------------------
# PROCESS EXCEL FILE
# ----------------------------

def process_excel(file_path, client_id=None):
    """Process a single Excel file and detect Q/A pairs using OpenAI."""
    print(f"\n📂 Processing file: {os.path.basename(file_path)}")
    xls = pd.ExcelFile(file_path)

    for sheet_name in xls.sheet_names:
        print(f"  ➡ Processing sheet: {sheet_name}")
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

        if df.empty:
            print("  ⚠ Sheet is empty. Skipping.")
            continue

        # Convert sheet to CSV for LLM
        sheet_csv = df.to_csv(index=False, header=False)

        # Extract Q/A pairs using LLM
        qa_pairs = extract_qa_pairs(sheet_csv)
        if not qa_pairs:
            print("  ⚠ No Q/A pairs found in this sheet.")
            continue

        print(f"  📊 Found {len(qa_pairs)} Q/A pairs")

        # Upload to Supabase
        for item in qa_pairs:
            question = item.get('question')
            answer = item.get('answer')
            category = item.get('category') or categorize_question(question)

            if not question or not answer:
                continue

            try:
                q_id = upsert_question(question, category, client_id=client_id)
                a_id = upsert_answer(answer, client_id=client_id)
                upsert_mapping(q_id, a_id)
            except Exception as e:
                print(f"❌ Error inserting Q/A pair: {e}")

# ----------------------------
# PROCESS ZIP FILES
# ----------------------------

def process_zip(zip_path, client_id=None):
    """Extract and process all Excel files inside a ZIP."""
    extract_dir = os.path.splitext(zip_path)[0]
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    for root, _, files in os.walk(extract_dir):
        for file in files:
            if file.endswith(".xlsx"):
                process_excel(os.path.join(root, file), client_id=client_id)

# ----------------------------
# MAIN EXECUTION
# ----------------------------
if __name__ == "__main__":
    print("🚀 Starting Q/A Extraction Pipeline...")
    client_id_env = os.getenv("ORG_CLIENT_ID")
    process_zip("2025-20250915T124602Z-1-001.zip", client_id=client_id_env)
    process_zip("2024-20250915T124207Z-1-001.zip", client_id=client_id_env)
    print("🎉 Extraction and upload completed successfully!")
