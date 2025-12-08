#!/usr/bin/env python
"""
Create RAG-Anything Tables Directly in Supabase

This script connects directly to PostgreSQL and creates all necessary tables
for RAG-Anything storage, so you can see them in your Supabase dashboard.
"""

import os
import sys
import psycopg2
from urllib.parse import urlparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)

print("=" * 70)
print("Creating RAG-Anything Tables in Supabase")
print("=" * 70)
print()

# Get configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://ngbhnjvojqqesacnijwk.supabase.co")
db_password = os.getenv("SUPABASE_DB_PASSWORD", "Test_1082?!")

# Extract connection info
parsed = urlparse(SUPABASE_URL)
host = parsed.hostname
if host and not host.startswith("db."):
    parts = host.split(".")
    if len(parts) >= 2 and parts[-2] == "supabase":
        project_ref = parts[0]
        host = f"db.{project_ref}.supabase.com"

workspace = "default"  # Default workspace name

print(f"Project: {SUPABASE_URL}")
print(f"Host: {host}:5432")
print(f"Workspace: {workspace}")
print()

# Connect to database
try:
    print("Connecting to database...")
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
    print("✓ Connected")
    
    # Check pgvector
    cursor.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector');")
    if not cursor.fetchone()[0]:
        print("\n⚠️  pgvector extension not found. Creating it...")
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        print("✓ pgvector extension created")
    
    print("\nCreating tables for RAG-Anything storage...")
    
    # 1. Vector Storage Table
    # LightRAG typically uses a table like: lightrag_vector_storage_{workspace}
    vector_table = f"lightrag_vector_storage_{workspace}"
    print(f"  Creating vector storage table: {vector_table}")
    
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {vector_table} (
            id TEXT PRIMARY KEY,
            text TEXT,
            embedding vector(3072),
            metadata JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    
    # Create index for vector similarity search
    try:
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS {vector_table}_embedding_idx 
            ON {vector_table} 
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)
        print(f"    ✓ Vector index created")
    except Exception as e:
        print(f"    ⚠️  Index creation skipped: {e}")
    
    print(f"  ✓ Vector storage table created")
    
    # 2. Key-Value Storage Table
    kv_table = f"lightrag_kv_storage_{workspace}"
    print(f"  Creating key-value storage table: {kv_table}")
    
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {kv_table} (
            key TEXT PRIMARY KEY,
            value JSONB,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        );
    """)
    print(f"  ✓ Key-value storage table created")
    
    # 3. Document Status Storage Table
    doc_status_table = f"lightrag_doc_status_{workspace}"
    print(f"  Creating document status table: {doc_status_table}")
    
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {doc_status_table} (
            doc_id TEXT PRIMARY KEY,
            status TEXT,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        );
    """)
    print(f"  ✓ Document status table created")
    
    # 4. Graph Storage Table (if needed)
    graph_table = f"lightrag_graph_{workspace}"
    print(f"  Creating graph storage table: {graph_table}")
    
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {graph_table} (
            id TEXT PRIMARY KEY,
            node_type TEXT,
            properties JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    print(f"  ✓ Graph storage table created")
    
    # Verify tables were created
    print("\nVerifying tables...")
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name LIKE 'lightrag_%'
        ORDER BY table_name;
    """)
    
    tables = cursor.fetchall()
    if tables:
        print(f"✓ Found {len(tables)} RAG-Anything tables:")
        for table in tables:
            # Count rows
            cursor.execute(f'SELECT COUNT(*) FROM "{table[0]}";')
            count = cursor.fetchone()[0]
            print(f"  - {table[0]}: {count} rows")
    else:
        print("⚠️  No tables found (this shouldn't happen)")
    
    cursor.close()
    conn.close()
    
    print("\n" + "=" * 70)
    print("✅ Tables Created Successfully!")
    print("=" * 70)
    print()
    print("You should now see these tables in your Supabase dashboard:")
    for table in tables:
        print(f"  ✓ {table[0]}")
    print()
    print("Check your Supabase dashboard:")
    print(f"  https://supabase.com/dashboard/project/ngbhnjvojqqesacnijwk/editor")
    print()
    print("These tables are ready for RAG-Anything to store:")
    print("  - Vector embeddings (document chunks)")
    print("  - Key-value pairs (metadata, cache)")
    print("  - Document status (processing state)")
    print("  - Graph data (knowledge graph)")
    
except psycopg2.OperationalError as e:
    print(f"❌ Connection error: {e}")
    print("\nPlease verify:")
    print("  1. Database password is correct")
    print("  2. Your IP is allowed in Supabase settings")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

