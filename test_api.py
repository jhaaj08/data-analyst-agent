#!/usr/bin/env python3
"""
Test script for the Data Analyst Agent API
Demonstrates the exact API specification: curl "https://app.example.com/api/" -F "@question.txt"
"""

import requests
import json
import sys

def test_api_with_question_file():
    """Test the API using the exact specification format"""
    
    # API endpoint (adjust URL as needed)
    api_url = "http://localhost:8000/api/"
    
    # Test question file
    question_file = "test_question.txt"
    
    try:
        # Open and send the question file exactly as specified
        with open(question_file, 'rb') as f:
            files = {'file': f}
            
            print(f"🚀 Sending request to: {api_url}")
            print(f"📄 Using question file: {question_file}")
            print("=" * 50)
            
            # Make the API request
            response = requests.post(api_url, files=files, timeout=180)
            
            print(f"📊 Response Status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    print("✅ Success! Response:")
                    print(json.dumps(result, indent=2))
                except json.JSONDecodeError:
                    print("📝 Response (text):")
                    print(response.text)
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"Details: {response.text}")
                
    except FileNotFoundError:
        print(f"❌ Question file '{question_file}' not found!")
        print("Create it first with your analysis questions.")
    except requests.ConnectionError:
        print("❌ Could not connect to API!")
        print("Make sure the server is running: python main.py")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def test_curl_equivalent():
    """Show the equivalent curl command"""
    print("\n" + "=" * 50)
    print("📋 Equivalent curl command:")
    print('curl "http://localhost:8000/api/" -F "@test_question.txt"')
    print("=" * 50)

if __name__ == "__main__":
    print("🤖 Data Analyst Agent API Test")
    print("Testing exact specification: POST with question file upload")
    
    test_api_with_question_file()
    test_curl_equivalent() 