#!/usr/bin/env python3
"""
Test script to verify RFP processing and QA extraction are working correctly
"""

import os
import sys
import time
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def test_gemini_functions():
    """Test the Gemini functions directly"""
    print("Testing Gemini functions...")
    
    # Import the functions from app.py
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from app import extract_questions_with_gemini, generate_tailored_answer
    
    # Test question extraction
    test_sheet_text = """
    Row 1: Company Name: Test Corp
    Row 2: Question: What is your pricing model?
    Row 3: Answer: We offer tiered pricing based on usage
    Row 4: Question: How do you handle data security?
    Row 5: Answer: We use enterprise-grade encryption
    """
    
    print("Testing extract_questions_with_gemini...")
    questions = extract_questions_with_gemini(test_sheet_text)
    print(f"Extracted questions: {questions}")
    
    # Test tailored answer generation
    print("\nTesting generate_tailored_answer...")
    test_matches = [
        {"question": "What is your pricing model?", "answer": "We offer tiered pricing based on usage", "similarity": 0.95},
        {"question": "How do you handle data security?", "answer": "We use enterprise-grade encryption", "similarity": 0.90}
    ]
    answer = generate_tailored_answer("What are your rates?", test_matches)
    print(f"Generated answer: {answer}")
    
    return len(questions) > 0 and len(answer) > 0

def check_recent_jobs():
    """Check recent jobs to see if they're processing correctly"""
    print("\nChecking recent jobs...")
    
    try:
        # Get recent jobs
        res = supabase.table("client_jobs").select("*").order("created_at", desc=True).limit(5).execute()
        jobs = res.data or []
        
        print(f"Found {len(jobs)} recent jobs:")
        for job in jobs:
            print(f"  - {job['job_type']}: {job['status']} ({job['progress_percent']}%) - {job['file_name']}")
            if job.get('current_step'):
                print(f"    Step: {job['current_step']}")
        
        return jobs
        
    except Exception as e:
        print(f"Error checking jobs: {e}")
        return []

def test_qa_extraction():
    """Test QA extraction function"""
    print("\nTesting QA extraction...")
    
    try:
        from app import _extract_qa_pairs
        
        test_csv = """Company Name,Test Corp
Question,What is your pricing model?
Answer,We offer tiered pricing based on usage
Question,How do you handle data security?
Answer,We use enterprise-grade encryption"""
        
        pairs = _extract_qa_pairs(test_csv)
        print(f"Extracted Q&A pairs: {pairs}")
        
        return len(pairs) > 0
        
    except Exception as e:
        print(f"Error testing QA extraction: {e}")
        return False

if __name__ == "__main__":
    print("=== RFP Processing Test ===")
    
    # Test 1: Gemini functions
    gemini_ok = test_gemini_functions()
    print(f"Gemini functions working: {gemini_ok}")
    
    # Test 2: QA extraction
    qa_ok = test_qa_extraction()
    print(f"QA extraction working: {qa_ok}")
    
    # Test 3: Check recent jobs
    jobs = check_recent_jobs()
    
    print(f"\n=== Test Results ===")
    print(f"Gemini functions: {'✅ PASS' if gemini_ok else '❌ FAIL'}")
    print(f"QA extraction: {'✅ PASS' if qa_ok else '❌ FAIL'}")
    print(f"Recent jobs: {len(jobs)} found")
    
    if gemini_ok and qa_ok:
        print("\n🎉 All tests passed! RFP processing should work correctly now.")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")
