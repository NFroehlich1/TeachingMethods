#!/usr/bin/env python
"""
Enable pgvector Extension - Direct Setup

This script will enable pgvector in your Supabase database.
You'll need your database password.
"""

import os
import sys
from pathlib import Path

try:
    import psycopg2
except ImportError:
    print("❌ psycopg2 not installed")
    print("   Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psycopg2-binary", "-q"])
    import psycopg2

from dotenv import load_dotenv
load_dotenv(dotenv_path=".env", override=False)

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://ngbhnjvojqqesacnijwk.supabase.co")
db_password = os.getenv("SUPABASE_DB_PASSWORD")

print("=" * 70)
print("Enable pgvector Extension in Supabase")
print("=" * 70)
print(f"Project: {SUPABASE_URL}")
print()

if not db_password:
    print("⚠️  Database password not found in .env file")
    print()
    print("Get your password from:")
    print("https://supabase.com/dashboard/project/ngbhnjvojqqesacnijwk/settings/database")
    print()
    db_password = input("Enter your Supabase database password: ").strip()
    
    if db_password:
        # Save to .env
        env_file = Path(".env")
        content = env_file.read_text() if env_file.exists() else ""
        if "SUPABASE_DB_PASSWORD=" not in content:
            content += f"\nSUPABASE_DB_PASSWORD={db_password}\n"
            env_file.write_text(content)
            print("✓ Password saved to .env")
    else:
        print("❌ Password required")
        sys.exit(1)

# Extract host
from urllib.parse import urlparse
parsed = urlparse(SUPABASE_URL)
host = parsed.hostname
if host and not host.startswith("db."):
    parts = host.split(".")
    if len(parts) >= 2 and parts[-2] == "supabase":
        project_ref = parts[0]
        host = f"db.{project_ref}.supabase.co"

print(f"\nConnecting to: {host}:5432")
print("Enabling pgvector extension...")

try:
    conn = psycopg2.connect(
        host=host,
        port=5432,
        user="postgres",
        password=db_password,
        database="postgres",
        connect_timeout=10,
    )
    conn.autocommit = True
    cursor = conn.cursor()
    
    # Check if exists
    cursor.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector');")
    exists = cursor.fetchone()[0]
    
    if exists:
        cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
        version = cursor.fetchone()[0]
        print(f"✓ pgvector already enabled (version {version})")
    else:
        try:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
            version = cursor.fetchone()[0]
            print(f"✓ pgvector extension enabled successfully!")
            print(f"  Version: {version}")
        except psycopg2.errors.InsufficientPrivilege:
            print("❌ Insufficient privileges via direct connection")
            print("\nPlease enable manually:")
            print("1. Go to: https://supabase.com/dashboard/project/ngbhnjvojqqesacnijwk/sql/new")
            print("2. Run: CREATE EXTENSION IF NOT EXISTS vector;")
            print("3. Click 'Run'")
            sys.exit(1)
    
    # Verify
    cursor.execute("SELECT COUNT(*) FROM pg_proc WHERE proname LIKE '%vector%';")
    func_count = cursor.fetchone()[0]
    print(f"✓ Verified: {func_count} vector functions available")
    
    cursor.close()
    conn.close()
    
    print("\n" + "=" * 70)
    print("✓ SUCCESS! pgvector is now enabled")
    print("=" * 70)
    print("\nYour Supabase database is ready for RAG-Anything!")
    print("You can now use vector storage and similarity search.")
    
except psycopg2.OperationalError as e:
    print(f"\n❌ Connection error: {e}")
    print("\nTroubleshooting:")
    print("1. Verify password is correct")
    print("2. Check IP restrictions in Supabase Dashboard")
    print("3. Try enabling manually in SQL Editor:")
    print("   https://supabase.com/dashboard/project/ngbhnjvojqqesacnijwk/sql/new")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

