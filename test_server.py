"""
Quick test to check if backend server is responding
"""
import requests
import sys

def test_server():
    base_url = "http://localhost:8000"
    
    print("Testing backend server...")
    print(f"Base URL: {base_url}")
    print("=" * 60)
    
    # Test 1: Health endpoint
    print("\n1. Testing /health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        print("   ✓ Health check passed")
    except requests.exceptions.ConnectionError:
        print("   ✗ ERROR: Cannot connect to backend server!")
        print("   Backend is NOT running on port 8000")
        return False
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        return False
    
    # Test 2: Check CORS headers
    print("\n2. Testing CORS headers...")
    try:
        response = requests.options(
            f"{base_url}/health",
            headers={"Origin": "http://localhost:5173"},
            timeout=5
        )
        cors_header = response.headers.get("Access-Control-Allow-Origin")
        print(f"   Access-Control-Allow-Origin: {cors_header}")
        if cors_header:
            print("   ✓ CORS is configured")
        else:
            print("   ✗ CORS header missing!")
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
    
    print("\n" + "=" * 60)
    print("✓ Backend server is running and responding")
    return True

if __name__ == "__main__":
    success = test_server()
    sys.exit(0 if success else 1)

