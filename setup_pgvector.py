#!/usr/bin/env python
"""
Setup pgvector Extension in Supabase

This script helps you enable the pgvector extension in your Supabase database.
It provides multiple methods to set up pgvector.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import psycopg2
except ImportError:
    print("❌ psycopg2 is required. Install it with: pip install raganything[supabase]")
    sys.exit(1)

from raganything import get_supabase_url_from_jwt
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)

# Your Supabase credentials
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5nYmhuanZvanFxZXNhY25pandrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjUwMDEyODcsImV4cCI6MjA4MDU3NzI4N30.-B_hvypDk3NLOYmYJ5qrJdojUtSFi7HSKK2aV2COsuU"
SUPABASE_URL = get_supabase_url_from_jwt(SUPABASE_ANON_KEY) or os.getenv("SUPABASE_URL")

if not SUPABASE_URL:
    print("❌ Error: Could not determine Supabase URL")
    sys.exit(1)

print("=" * 60)
print("pgvector Extension Setup for Supabase")
print("=" * 60)
print(f"Project URL: {SUPABASE_URL}")
print()

# Get database password
db_password = os.getenv("SUPABASE_DB_PASSWORD")
if not db_password:
    print("⚠️  SUPABASE_DB_PASSWORD not set in environment")
    print("   Get it from: Supabase Dashboard → Settings → Database → Database Password")
    print()
    db_password = input("Enter your Supabase database password: ").strip()
    if not db_password:
        print("❌ Database password is required")
        sys.exit(1)

# Extract connection info
from urllib.parse import urlparse
parsed = urlparse(SUPABASE_URL)
host = parsed.hostname
if host and not host.startswith("db."):
    parts = host.split(".")
    if len(parts) >= 2 and parts[-2] == "supabase":
        project_ref = parts[0]
        host = f"db.{project_ref}.supabase.co"

db_host = host or "localhost"
db_port = 5432
db_user = "postgres"
db_name = "postgres"

print(f"Connecting to: {db_host}:{db_port}")
print()

try:
    # Connect to database
    print("Connecting to Supabase database...")
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        user=db_user,
        password=db_password,
        database=db_name,
        connect_timeout=10,
    )
    print("✓ Connected successfully")
    
    # Check if extension exists
    conn.autocommit = True
    cursor = conn.cursor()
    
    print("\nChecking pgvector extension...")
    cursor.execute("""
        SELECT EXISTS(
            SELECT 1 FROM pg_extension WHERE extname = 'vector'
        );
    """)
    extension_exists = cursor.fetchone()[0]
    
    if extension_exists:
        print("✓ pgvector extension already exists")
        
        # Check version
        cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
        version = cursor.fetchone()[0]
        print(f"  Version: {version}")
    else:
        print("⚠️  pgvector extension not found")
        print("\nCreating pgvector extension...")
        
        try:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            print("✓ pgvector extension created successfully!")
            
            # Get version
            cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
            version = cursor.fetchone()[0]
            print(f"  Version: {version}")
        except psycopg2.errors.InsufficientPrivilege as e:
            print("❌ Error: Insufficient privileges to create extension")
            print("\nYou need to create the extension manually in Supabase SQL Editor:")
            print("\n1. Go to: https://supabase.com/dashboard/project/ngbhnjvojqqesacnijwk/sql/new")
            print("2. Run this SQL command:")
            print("\n   CREATE EXTENSION IF NOT EXISTS vector;")
            print("\n3. Click 'Run' to execute")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error creating extension: {e}")
            print("\nPlease create it manually in Supabase SQL Editor:")
            print("  CREATE EXTENSION IF NOT EXISTS vector;")
            sys.exit(1)
    
    # Verify extension functions
    print("\nVerifying pgvector functions...")
    cursor.execute("""
        SELECT proname 
        FROM pg_proc 
        WHERE proname LIKE '%vector%' 
        LIMIT 5;
    """)
    functions = cursor.fetchall()
    
    if functions:
        print("✓ pgvector functions available:")
        for func in functions[:5]:
            print(f"  - {func[0]}")
    else:
        print("⚠️  No vector functions found (extension may not be fully loaded)")
    
    cursor.close()
    conn.close()
    
    print("\n" + "=" * 60)
    print("✓ Setup Complete!")
    print("=" * 60)
    print("\nYour Supabase database is ready for RAG-Anything!")
    print("You can now use vector storage with pgvector.")
    
except psycopg2.OperationalError as e:
    print(f"❌ Connection error: {e}")
    print("\nTroubleshooting:")
    print("1. Verify your database password is correct")
    print("2. Check if your IP is allowed in Supabase (Settings → Database → Connection Pooling)")
    print("3. Ensure you're using the correct project URL")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


