#!/usr/bin/env python
"""
Simple Supabase Setup - Minimal Dependencies
"""
import os
import sys
import base64
import json
from pathlib import Path

# Extract project URL from JWT
def get_supabase_url_from_jwt(jwt_token):
    try:
        parts = jwt_token.split(".")
        if len(parts) < 2:
            return None
        payload = parts[1]
        padding = len(payload) % 4
        if padding:
            payload += "=" * (4 - padding)
        decoded = base64.urlsafe_b64decode(payload)
        data = json.loads(decoded)
        ref = data.get("ref")
        return f"https://{ref}.supabase.co" if ref else None
    except:
        return None

# Your credentials
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5nYmhuanZvanFxZXNhY25pandrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjUwMDEyODcsImV4cCI6MjA4MDU3NzI4N30.-B_hvypDk3NLOYmYJ5qrJdojUtSFi7HSKK2aV2COsuU"
SUPABASE_ACCESS_TOKEN = "sbp_126aa9500ca99a588c1a0c748aa2fdda4a4cf391"
SUPABASE_URL = get_supabase_url_from_jwt(SUPABASE_ANON_KEY)

print("=" * 70)
print("Supabase Setup for RAG-Anything")
print("=" * 70)
print(f"Project URL: {SUPABASE_URL}")
print()

# Update .env file
env_file = Path(".env")
supabase_config = f"""# Supabase Configuration (Auto-configured)
SUPABASE_URL={SUPABASE_URL}
SUPABASE_KEY={SUPABASE_ANON_KEY}
SUPABASE_ACCESS_TOKEN={SUPABASE_ACCESS_TOKEN}
SUPABASE_DB_USER=postgres
"""

if env_file.exists():
    content = env_file.read_text()
    if "SUPABASE_URL=" not in content:
        content += "\n" + supabase_config
        env_file.write_text(content)
        print("✓ Updated .env file")
    else:
        print("✓ .env file already configured")
else:
    env_file.write_text(supabase_config)
    print("✓ Created .env file")

# Check for psycopg2
try:
    import psycopg2
    print("✓ psycopg2 available")
    HAS_PSYCOPG2 = True
except ImportError:
    print("⚠️  psycopg2 not installed")
    print("   Install with: pip install psycopg2-binary")
    HAS_PSYCOPG2 = False

if HAS_PSYCOPG2:
    print("\nTo enable pgvector extension:")
    print("1. Get your database password from:")
    print(f"   https://supabase.com/dashboard/project/ngbhnjvojqqesacnijwk/settings/database")
    print()
    print("2. Run this script:")
    print("   python setup_pgvector.py")
    print()
    print("OR manually in SQL Editor:")
    print(f"   https://supabase.com/dashboard/project/ngbhnjvojqqesacnijwk/sql/new")
    print("   Run: CREATE EXTENSION IF NOT EXISTS vector;")
else:
    print("\nNext steps:")
    print("1. Install dependencies: pip install raganything[supabase]")
    print("2. Get database password from Supabase Dashboard")
    print("3. Run: python setup_pgvector.py")

print("\n" + "=" * 70)
print("Configuration saved to .env file")
print("=" * 70)

