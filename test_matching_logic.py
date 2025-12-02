"""
Test script to verify tender matching logic
Run: python test_matching_logic.py
"""
from services.tender_service import match_tender_against_keywords

def test_matching():
    print("=" * 80)
    print("TESTING TENDER MATCHING LOGIC")
    print("=" * 80)
    print()
    
    # Test 1: Keyword + Location + Industry ALL match
    print("Test 1: All filters match")
    tender1 = {
        'title': 'IT Services Contract in London',
        'description': 'Provide IT support services',
        'location': 'London, UK',
        'sector': 'Technology',
        'category': 'IT Services',
        'metadata': {'location_name': 'London'}
    }
    keywords1 = {
        'keywords': ['IT'],
        'locations': ['London'],
        'sectors': ['Technology']
    }
    match1, score1, matched1 = match_tender_against_keywords(tender1, keywords1, enable_ai=False)
    print(f"   Result: {'✓ MATCH' if match1 else '✗ NO MATCH'} (score={score1})")
    print(f"   Expected: ✓ MATCH (keyword=IT, location=London, industry=Technology)")
    print()
    
    # Test 2: Keyword matches, Location matches, but NO Industry data
    print("Test 2: Keyword + Location match, but tender has NO industry")
    tender2 = {
        'title': 'IT Services Contract in London',
        'description': 'Provide IT support',
        'location': 'London, UK',
        'sector': None,  # NO INDUSTRY DATA
        'category': None,
        'metadata': {}
    }
    keywords2 = {
        'keywords': ['IT'],
        'locations': ['London'],
        'sectors': ['Technology']  # User requires Technology
    }
    match2, score2, matched2 = match_tender_against_keywords(tender2, keywords2, enable_ai=False)
    print(f"   Result: {'✓ MATCH' if match2 else '✗ NO MATCH'} (score={score2})")
    print(f"   Expected: ✗ NO MATCH (industry is required but missing)")
    print()
    
    # Test 3: Keyword matches, but WRONG Location
    print("Test 3: Keyword matches, but wrong location")
    tender3 = {
        'title': 'IT Services Contract',
        'description': 'Provide IT support',
        'location': 'Manchester, UK',  # WRONG LOCATION
        'sector': 'Technology',
        'metadata': {'location_name': 'Manchester'}
    }
    keywords3 = {
        'keywords': ['IT'],
        'locations': ['London'],  # User wants London
        'sectors': []
    }
    match3, score3, matched3 = match_tender_against_keywords(tender3, keywords3, enable_ai=False)
    print(f"   Result: {'✓ MATCH' if match3 else '✗ NO MATCH'} (score={score3})")
    print(f"   Expected: ✗ NO MATCH (location=Manchester, but required=London)")
    print()
    
    # Test 4: Keyword matches, Location matches, but WRONG Industry
    print("Test 4: Keyword + Location match, but wrong industry")
    tender4 = {
        'title': 'IT Services Contract in London',
        'description': 'Provide IT support',
        'location': 'London, UK',
        'sector': 'Construction',  # WRONG INDUSTRY
        'metadata': {}
    }
    keywords4 = {
        'keywords': ['IT'],
        'locations': ['London'],
        'sectors': ['Technology']  # User wants Technology
    }
    match4, score4, matched4 = match_tender_against_keywords(tender4, keywords4, enable_ai=False)
    print(f"   Result: {'✓ MATCH' if match4 else '✗ NO MATCH'} (score={score4})")
    print(f"   Expected: ✗ NO MATCH (industry=Construction, but required=Technology)")
    print()
    
    # Test 5: Only keyword filter (no location/industry)
    print("Test 5: Only keyword filter (location/industry not specified)")
    tender5 = {
        'title': 'IT Services Contract',
        'description': 'Provide IT support',
        'location': 'Any City',
        'sector': 'Any Sector',
        'metadata': {}
    }
    keywords5 = {
        'keywords': ['IT'],
        'locations': [],  # No location filter
        'sectors': []     # No industry filter
    }
    match5, score5, matched5 = match_tender_against_keywords(tender5, keywords5, enable_ai=False)
    print(f"   Result: {'✓ MATCH' if match5 else '✗ NO MATCH'} (score={score5})")
    print(f"   Expected: ✓ MATCH (only keyword required, location/industry ignored)")
    print()
    
    # Test 6: Tender with NO location when location IS required
    print("Test 6: Tender has NO location, but location IS required")
    tender6 = {
        'title': 'IT Services Contract',
        'description': 'Provide IT support',
        'location': None,  # NO LOCATION
        'sector': 'Technology',
        'metadata': {}
    }
    keywords6 = {
        'keywords': ['IT'],
        'locations': ['London'],  # User requires London
        'sectors': []
    }
    match6, score6, matched6 = match_tender_against_keywords(tender6, keywords6, enable_ai=False)
    print(f"   Result: {'✓ MATCH' if match6 else '✗ NO MATCH'} (score={score6})")
    print(f"   Expected: ✗ NO MATCH (location is required but missing)")
    print()
    
    # Test 7: Multiple locations (OR logic)
    print("Test 7: Multiple locations - tender matches one of them")
    tender7 = {
        'title': 'IT Services Contract in Manchester',
        'description': 'Provide IT support',
        'location': 'Manchester, UK',
        'sector': 'Technology',
        'metadata': {'location_name': 'Manchester'}
    }
    keywords7 = {
        'keywords': ['IT'],
        'locations': ['London', 'Manchester', 'Birmingham'],  # OR logic
        'sectors': []
    }
    match7, score7, matched7 = match_tender_against_keywords(tender7, keywords7, enable_ai=False)
    print(f"   Result: {'✓ MATCH' if match7 else '✗ NO MATCH'} (score={score7})")
    print(f"   Expected: ✓ MATCH (Manchester is one of the accepted locations)")
    print()
    
    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    results = [
        ("Test 1", match1, True),
        ("Test 2", match2, False),
        ("Test 3", match3, False),
        ("Test 4", match4, False),
        ("Test 5", match5, True),
        ("Test 6", match6, False),
        ("Test 7", match7, True),
    ]
    
    passed = sum(1 for _, actual, expected in results if actual == expected)
    total = len(results)
    
    for test_name, actual, expected in results:
        status = "✓ PASS" if actual == expected else "✗ FAIL"
        print(f"{test_name}: {status} (expected={expected}, got={actual})")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ ALL TESTS PASSED!")
    else:
        print(f"✗ {total - passed} test(s) failed")
    print()

if __name__ == "__main__":
    test_matching()

