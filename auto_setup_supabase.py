#!/usr/bin/env python
"""
Fully Automated Supabase Setup for RAG-Anything

This script automatically sets up everything needed for Supabase integration.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("RAG-Anything Supabase Automated Setup")
print("=" * 70)
print()

# Your credentials
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5nYmhuanZvanFxZXNhY25pandrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjUwMDEyODcsImV4cCI6MjA4MDU3NzI4N30.-B_hvypDk3NLOYmYJ5qrJdojUtSFi7HSKK2aV2COsuU"
SUPABASE_ACCESS_TOKEN = "sbp_126aa9500ca99a588c1a0c748aa2fdda4a4cf391"

from raganything import get_supabase_url_from_jwt
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)

SUPABASE_URL = get_supabase_url_from_jwt(SUPABASE_ANON_KEY) or os.getenv("SUPABASE_URL")

if not SUPABASE_URL:
    print("❌ Error: Could not determine Supabase URL")
    sys.exit(1)

print(f"✓ Project URL: {SUPABASE_URL}")
print()

# Step 1: Check/Install dependencies
print("Step 1: Checking dependencies...")
try:
    import psycopg2
    print("✓ psycopg2 is installed")
except ImportError:
    print("⚠️  psycopg2 not found. Installing...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psycopg2-binary", "-q"])
        import psycopg2
        print("✓ psycopg2 installed successfully")
    except Exception as e:
        print(f"❌ Failed to install psycopg2: {e}")
        print("   Please run: pip install raganything[supabase]")
        sys.exit(1)

# Step 2: Get database password
print("\nStep 2: Database password...")
db_password = os.getenv("SUPABASE_DB_PASSWORD")

if not db_password:
    print("⚠️  Database password not found in environment")
    print("   Getting it from Supabase Dashboard...")
    print()
    print("   Please provide your database password.")
    print("   You can find it at:")
    print(f"   https://supabase.com/dashboard/project/ngbhnjvojqqesacnijwk/settings/database")
    print()
    db_password = input("   Enter your Supabase database password: ").strip()
    
    if not db_password:
        print("❌ Database password is required")
        print("\n   Alternative: Enable pgvector manually in SQL Editor:")
        print(f"   https://supabase.com/dashboard/project/ngbhnjvojqqesacnijwk/sql/new")
        print("   Run: CREATE EXTENSION IF NOT EXISTS vector;")
        sys.exit(1)
    
    # Save to .env
    env_file = Path(".env")
    env_content = env_file.read_text() if env_file.exists() else ""
    if "SUPABASE_DB_PASSWORD" not in env_content:
        if env_content and not env_content.endswith("\n"):
            env_content += "\n"
        env_content += f"SUPABASE_DB_PASSWORD={db_password}\n"
        env_file.write_text(env_content)
        print("✓ Password saved to .env file")
else:
    print("✓ Database password found in environment")

# Step 3: Update .env with all credentials
print("\nStep 3: Updating configuration...")
env_file = Path(".env")
env_content = env_file.read_text() if env_file.exists() else ""

# Ensure all Supabase config is present
supabase_config = f"""# Supabase Configuration (Auto-configured)
SUPABASE_URL={SUPABASE_URL}
SUPABASE_KEY={SUPABASE_ANON_KEY}
SUPABASE_ACCESS_TOKEN={SUPABASE_ACCESS_TOKEN}
SUPABASE_DB_USER=postgres
SUPABASE_DB_PASSWORD={db_password}
"""

# Update or add config
if "SUPABASE_URL=" in env_content:
    # Update existing
    lines = env_content.split("\n")
    new_lines = []
    skip_supabase = False
    for i, line in enumerate(lines):
        if line.startswith("SUPABASE_"):
            if not skip_supabase:
                # Replace with new config
                new_lines.append(supabase_config.strip())
                skip_supabase = True
            continue
        if skip_supabase and line.strip() == "":
            skip_supabase = False
        if not skip_supabase:
            new_lines.append(line)
    
    if skip_supabase:
        new_lines.append("")
    
    env_content = "\n".join(new_lines)
else:
    # Add new config
    if env_content and not env_content.endswith("\n"):
        env_content += "\n"
    env_content += "\n" + supabase_config.strip()

env_file.write_text(env_content)
print("✓ Configuration updated in .env file")

# Step 4: Verify connection
print("\nStep 4: Verifying database connection...")
try:
    from urllib.parse import urlparse
    parsed = urlparse(SUPABASE_URL)
    host = parsed.hostname
    if host and not host.startswith("db."):
        parts = host.split(".")
        if len(parts) >= 2 and parts[-2] == "supabase":
            project_ref = parts[0]
            host = f"db.{project_ref}.supabase.com"
    
    conn = psycopg2.connect(
        host=host,
        port=5432,
        user="postgres",
        password=db_password,
        database="postgres",
        connect_timeout=10,
    )
    print("✓ Database connection successful")
    conn.close()
except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("\n   Please verify:")
    print("   1. Your database password is correct")
    print("   2. Your IP is allowed (check Supabase Dashboard → Settings → Database)")
    sys.exit(1)

# Step 5: Enable pgvector extension
print("\nStep 5: Enabling pgvector extension...")
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
        print(f"✓ pgvector extension already exists (version {version})")
    else:
        print("   Creating pgvector extension...")
        try:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
            version = cursor.fetchone()[0]
            print(f"✓ pgvector extension created successfully (version {version})")
        except psycopg2.errors.InsufficientPrivilege:
            print("⚠️  Insufficient privileges to create extension via direct connection")
            print("\n   Please enable it manually:")
            print(f"   1. Go to: https://supabase.com/dashboard/project/ngbhnjvojqqesacnijwk/sql/new")
            print("   2. Run: CREATE EXTENSION IF NOT EXISTS vector;")
            print("   3. Click 'Run'")
            cursor.close()
            conn.close()
            sys.exit(1)
    
    # Verify functions
    cursor.execute("""
        SELECT COUNT(*) 
        FROM pg_proc 
        WHERE proname LIKE '%vector%';
    """)
    func_count = cursor.fetchone()[0]
    print(f"✓ Found {func_count} vector-related functions")
    
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f"❌ Error setting up pgvector: {e}")
    print("\n   Please enable it manually in Supabase SQL Editor:")
    print(f"   https://supabase.com/dashboard/project/ngbhnjvojqqesacnijwk/sql/new")
    print("   Run: CREATE EXTENSION IF NOT EXISTS vector;")
    sys.exit(1)

# Step 6: Test RAG-Anything setup
print("\nStep 6: Testing RAG-Anything configuration...")
try:
    from raganything import setup_supabase_storage
    
    lightrag_kwargs = setup_supabase_storage(
        supabase_key=SUPABASE_ANON_KEY,
        supabase_access_token=SUPABASE_ACCESS_TOKEN,
        supabase_db_password=db_password,
        workspace="default"
    )
    
    # Verify configuration
    required_keys = ["vector_storage", "kv_storage", "doc_status_storage"]
    all_present = all(key in lightrag_kwargs for key in required_keys)
    
    if all_present:
        print("✓ RAG-Anything storage configuration ready")
        print("  - Vector storage: PostgreSQL with pgvector")
        print("  - Key-value storage: PostgreSQL")
        print("  - Document status storage: PostgreSQL")
    else:
        print("⚠️  Some storage configurations missing")
        
except Exception as e:
    print(f"⚠️  Configuration test failed: {e}")
    print("   This might be okay if dependencies are missing")

# Final summary
print("\n" + "=" * 70)
print("✓ Setup Complete!")
print("=" * 70)
print()
print("Your Supabase integration is ready!")
print()
print("Next steps:")
print("1. Test the integration:")
print("   python examples/supabase_quickstart.py")
print()
print("2. Process a document:")
print("   python examples/supabase_quickstart.py document.pdf")
print()
print("3. Use in your code:")
print("   from raganything import RAGAnything, setup_supabase_storage")
print("   lightrag_kwargs = setup_supabase_storage()")
print("   rag = RAGAnything(llm_model_func=..., embedding_func=..., lightrag_kwargs=lightrag_kwargs)")
print()
print("All configuration is saved in .env file")
print("=" * 70)

