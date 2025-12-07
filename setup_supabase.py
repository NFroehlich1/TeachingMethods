#!/usr/bin/env python
"""
Automated Supabase Setup Script for RAG-Anything

This script automates the setup of Supabase integration with RAG-Anything.
It uses the provided anon key and access token to configure everything.

Usage:
    python setup_supabase.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from raganything import (
    setup_supabase_storage,
    verify_supabase_connection,
    create_pgvector_extension,
    get_supabase_url_from_jwt,
)

# Your Supabase credentials
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5nYmhuanZvanFxZXNhY25pandrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjUwMDEyODcsImV4cCI6MjA4MDU3NzI4N30.-B_hvypDk3NLOYmYJ5qrJdojUtSFi7HSKK2aV2COsuU"
SUPABASE_ACCESS_TOKEN = "sbp_126aa9500ca99a588c1a0c748aa2fdda4a4cf391"

# Extract project URL from JWT
SUPABASE_URL = get_supabase_url_from_jwt(SUPABASE_ANON_KEY)

if not SUPABASE_URL:
    print("❌ Error: Could not extract Supabase URL from JWT token")
    sys.exit(1)

print("=" * 60)
print("RAG-Anything Supabase Setup")
print("=" * 60)
print(f"✓ Project URL: {SUPABASE_URL}")
print(f"✓ Anon Key: {SUPABASE_ANON_KEY[:20]}...")
print(f"✓ Access Token: {SUPABASE_ACCESS_TOKEN[:20]}...")
print()

# Check if database password is set
db_password = os.getenv("SUPABASE_DB_PASSWORD")
if not db_password:
    print("⚠️  Warning: SUPABASE_DB_PASSWORD environment variable not set")
    print("   You need to provide your database password to connect directly to PostgreSQL.")
    print("   Get it from: Supabase Dashboard → Settings → Database → Database Password")
    print()
    db_password = input("Enter your Supabase database password (or press Enter to skip): ").strip()
    if not db_password:
        print("❌ Database password is required for PostgreSQL connections")
        print("   Setting up environment variables only...")
        db_password = None
    else:
        # Save to .env file
        env_file = Path(".env")
        if env_file.exists():
            content = env_file.read_text()
            if "SUPABASE_DB_PASSWORD" not in content:
                content += f"\nSUPABASE_DB_PASSWORD={db_password}\n"
                env_file.write_text(content)
        else:
            env_file.write_text(f"SUPABASE_DB_PASSWORD={db_password}\n")
        print("✓ Database password saved to .env file")

# Update .env file with Supabase configuration
env_file = Path(".env")
env_content = ""

if env_file.exists():
    env_content = env_file.read_text()
else:
    # Create from example if it doesn't exist
    example_file = Path("env.example")
    if example_file.exists():
        env_content = example_file.read_text()

# Update or add Supabase configuration
supabase_config = f"""# Supabase Configuration (Auto-configured)
SUPABASE_URL={SUPABASE_URL}
SUPABASE_KEY={SUPABASE_ANON_KEY}
SUPABASE_ACCESS_TOKEN={SUPABASE_ACCESS_TOKEN}
SUPABASE_DB_USER=postgres
"""

# Check if Supabase config already exists
if "SUPABASE_URL=" in env_content:
    # Update existing config
    lines = env_content.split("\n")
    new_lines = []
    skip_until_empty = False
    for line in lines:
        if skip_until_empty:
            if line.strip() == "":
                skip_until_empty = False
                new_lines.append(line)
            continue
        if line.startswith("SUPABASE_URL=") or line.startswith("SUPABASE_KEY=") or line.startswith("SUPABASE_ACCESS_TOKEN=") or line.startswith("SUPABASE_DB_USER="):
            skip_until_empty = True
            continue
        new_lines.append(line)
    
    # Add new config before the last empty line or at the end
    if new_lines and new_lines[-1].strip() == "":
        new_lines[-1] = supabase_config.strip()
    else:
        new_lines.append("")
        new_lines.append(supabase_config.strip())
    
    env_content = "\n".join(new_lines)
else:
    # Add new config
    if env_content and not env_content.endswith("\n"):
        env_content += "\n"
    env_content += "\n" + supabase_config.strip()

# Write updated .env file
env_file.write_text(env_content)
print("✓ Updated .env file with Supabase configuration")
print()

if db_password:
    # Verify connection
    print("Verifying Supabase connection...")
    if verify_supabase_connection(
        supabase_url=SUPABASE_URL,
        supabase_db_password=db_password
    ):
        print("✓ Connection verified successfully")
    else:
        print("❌ Connection verification failed")
        print("   Please check your database password")
        sys.exit(1)
    
    # Create pgvector extension
    print()
    print("Setting up pgvector extension...")
    if create_pgvector_extension(
        supabase_url=SUPABASE_URL,
        supabase_db_password=db_password
    ):
        print("✓ pgvector extension ready")
    else:
        print("⚠️  Could not create pgvector extension automatically")
        print("   Please run this SQL in Supabase SQL Editor:")
        print("   CREATE EXTENSION IF NOT EXISTS vector;")
    
    print()
    print("=" * 60)
    print("✓ Setup Complete!")
    print("=" * 60)
    print()
    print("You can now use RAG-Anything with Supabase:")
    print()
    print("  from raganything import RAGAnything, setup_supabase_storage")
    print()
    print("  lightrag_kwargs = setup_supabase_storage()")
    print("  # Configuration will be read from .env file")
    print()
    print("  rag = RAGAnything(")
    print("      llm_model_func=llm_model_func,")
    print("      embedding_func=embedding_func,")
    print("      lightrag_kwargs=lightrag_kwargs")
    print("  )")
    print()
else:
    print("=" * 60)
    print("⚠️  Partial Setup Complete")
    print("=" * 60)
    print()
    print("Environment variables have been configured, but database password")
    print("is required for PostgreSQL connections.")
    print()
    print("To complete setup:")
    print("1. Get your database password from Supabase Dashboard")
    print("2. Add it to .env file: SUPABASE_DB_PASSWORD=your_password")
    print("3. Run this script again or use the integration directly")
    print()


