#!/usr/bin/env python3
"""
Check Gemini API quota and activation status
"""

import google.generativeai as genai
import json

GEMINI_API_KEY = "AIzaSyC7PgLQv4QByfLjkT398fp9Jgs7MNiUdmk"

print("=" * 70)
print("Checking Gemini API Key and Quota Status")
print("=" * 70)
print(f"API Key: {GEMINI_API_KEY[:20]}...{GEMINI_API_KEY[-10:]}")
print()

try:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✓ API configured successfully")
    
    # List models to check if API is accessible
    print("\n1. Checking API access...")
    try:
        models = list(genai.list_models())
        print(f"   ✓ API is accessible - Found {len(models)} models")
        
        # Check for gemini-2.0-flash-exp specifically
        flash_models = [m for m in models if 'flash' in m.name.lower() or '2.0' in m.name]
        if flash_models:
            print(f"   ✓ Found {len(flash_models)} Flash models:")
            for m in flash_models[:5]:
                print(f"     - {m.name}")
    except Exception as e:
        print(f"   ✗ Error listing models: {e}")
        exit(1)
    
    # Try to generate content with a very simple request
    print("\n2. Testing content generation...")
    try:
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        print("   ✓ Model initialized: gemini-2.0-flash-exp")
        
        # Try a minimal request
        response = model.generate_content("Hi", generation_config={
            "max_output_tokens": 10,
            "temperature": 0.1
        })
        
        answer = response.text if hasattr(response, 'text') else str(response)
        print(f"   ✓ Content generation SUCCESSFUL!")
        print(f"   Response: {answer[:100]}")
        print("\n" + "=" * 70)
        print("✅ GEMINI API KEY IS WORKING WITH QUOTA!")
        print("=" * 70)
        
    except Exception as e:
        error_str = str(e)
        print(f"   ✗ Content generation failed")
        print(f"   Error: {error_str[:300]}")
        
        # Analyze the error
        if "429" in error_str:
            if "quota" in error_str.lower() and "0" in error_str:
                print("\n" + "=" * 70)
                print("⚠️  QUOTA ISSUE: Free Tier Quota is 0")
                print("=" * 70)
                print("\nPossible solutions:")
                print("1. Enable Gemini API in Google Cloud Console:")
                print("   https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com")
                print("\n2. Check if billing is enabled (required for some quotas)")
                print("   https://console.cloud.google.com/billing")
                print("\n3. Request quota increase:")
                print("   https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com/quotas")
                print("\n4. Try a different model (gemini-1.5-flash) that might have quota:")
                try:
                    model_15 = genai.GenerativeModel("gemini-1.5-flash")
                    response_15 = model_15.generate_content("Hi")
                    print(f"\n   ✓ gemini-1.5-flash WORKS! Response: {response_15.text[:50]}")
                    print("\n   SOLUTION: Use gemini-1.5-flash instead of gemini-2.0-flash-exp")
                except:
                    print("\n   ✗ gemini-1.5-flash also has quota issues")
            else:
                print("\n⚠️  Rate limit exceeded - wait a few minutes and try again")
        elif "401" in error_str or "unauthorized" in error_str.lower():
            print("\n✗ API Key is invalid or unauthorized")
        else:
            print(f"\n⚠️  Unknown error - check the error message above")
        
        print("\n" + "=" * 70)
        print("❌ QUOTA TEST FAILED")
        print("=" * 70)
        exit(1)
        
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

