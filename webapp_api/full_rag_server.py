import os
import time
import json
import logging
import requests
import psycopg2
import uvicorn
import markdown
import re
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime

# Load environment variables
# Explicitly look in project root
BASE_DIR = Path(__file__).resolve().parent
DOTENV_PATH = BASE_DIR.parent / ".env"
load_dotenv(dotenv_path=DOTENV_PATH)

# --- Configuration ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")
SUPABASE_DB_USER = "postgres"

# 1. Try manual override from environment variable
if os.getenv("SUPABASE_DB_HOST"):
    SUPABASE_DB_HOST = os.getenv("SUPABASE_DB_HOST")
    print(f"Configuration: Using manual SUPABASE_DB_HOST: {SUPABASE_DB_HOST}")

# 2. Else, extract host from Supabase URL
elif SUPABASE_URL:
    # Extract host from Supabase URL (e.g. https://xyz.supabase.co -> db.xyz.supabase.co)
    # Database host is always db.{project_ref}.supabase.co
    from urllib.parse import urlparse
    parsed = urlparse(SUPABASE_URL)
    hostname = parsed.hostname or ""
    # Extract project ref (first part before .supabase)
    if hostname and ".supabase" in hostname:
        project_ref = hostname.split(".")[0]
        SUPABASE_DB_HOST = f"db.{project_ref}.supabase.co"
    else:
        # Fallback: try old method
        project_ref = SUPABASE_URL.split("//")[1].split(".")[0] if "//" in SUPABASE_URL else ""
        SUPABASE_DB_HOST = f"db.{project_ref}.supabase.co" if project_ref else "localhost"
else:
    SUPABASE_DB_HOST = "localhost"

# Allow Port override
SUPABASE_DB_PORT = os.getenv("SUPABASE_DB_PORT", "5432")

# FIX: Automatically adjust user for pooler connections if not explicitly set
# If we are connecting to a Supabase pooler, the user MUST be in format postgres.project_ref
if "pooler.supabase.com" in SUPABASE_DB_HOST and SUPABASE_DB_USER == "postgres":
     # Try to extract project ref from SUPABASE_URL if available
    if SUPABASE_URL and "supabase" in SUPABASE_URL:
        try:
             # Extract project ref from URL like https://xyz.supabase.co
             from urllib.parse import urlparse
             parsed = urlparse(SUPABASE_URL)
             hostname = parsed.hostname or ""
             if hostname:
                 extracted_ref = hostname.split(".")[0]
                 SUPABASE_DB_USER = f"postgres.{extracted_ref}"
                 print(f"Configuration: Auto-corrected Pooler User to {SUPABASE_DB_USER}")
        except Exception as e:
            print(f"Warning: Could not auto-correct pooler user: {e}")

# LLM Configuration - Default to local Ollama for laptop hosting
# When accessed from outside, your laptop serves as the LLM host
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" (local on your laptop), "gemini", "openai"
# Ollama runs on your laptop - use localhost (server forwards requests to local Ollama)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")  # Local model running on your laptop
OLLAMA_FALLBACK_MODEL = os.getenv("OLLAMA_FALLBACK_MODEL", "llama2:7b") # Fallback model for Rate Limits
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Hugging Face Configuration
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
# Primary Hugging Face model - Kimi K2 Thinking
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "moonshotai/Kimi-K2-Thinking")
# Secondary Hugging Face model - Meta Llama 3 8B Instruct
HUGGINGFACE_MODEL_2 = os.getenv("HUGGINGFACE_MODEL_2", "meta-llama/Meta-Llama-3-8B-Instruct")

# Embedding Configuration
# "ollama" (local) or "huggingface" (cloud/api)
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "ollama") 
EMBEDDING_MODEL = "nomic-embed-text" 
# HF model for embeddings (using Inference API - usually 384 or 768 or 1024 dims)
# sentence-transformers/all-MiniLM-L6-v2 is 384
# BAAI/bge-m3 is 1024
# nomic-ai/nomic-embed-text-v1.5 is 768
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = 3072  # Dimension for lightrag_vector_storage (we pad/truncate to match this legacy size)

# --- Database Connection ---
def get_db_connection():
    # Use global variables, but allow local overrides if we need to switch to pooler
    db_host = SUPABASE_DB_HOST
    db_port = SUPABASE_DB_PORT
    db_user = SUPABASE_DB_USER
    
    try:
        # Force IPv4 connection logic
        import socket
        try:
            # Try to resolve hostname to IPv4 address
            ipv4_address = socket.gethostbyname(db_host)
            print(f"DEBUG: Connecting to {db_host} (resolved to {ipv4_address})")
            connect_host = ipv4_address
        except socket.gaierror:
            print(f"WARNING: Could not resolve IPv4 for {db_host}. Database might be IPv6 only.")
            
            # Intelligent Fallback: Try Supabase Connection Pooler (IPv4 compatible)
            # Assumption: User is in EU (based on common Supabase deployments). 
            # If this fails, user must set SUPABASE_DB_HOST manually.
            print("ATTEMPTING FALLBACK: Switching to Supabase Connection Pooler (IPv4)...")
            
            # Common pooler hosts to try (in order of probability based on typical deployments)
            # Prioritizing eu-west-1 (Ireland) as identified via Supabase API for this project
            pooler_hosts = [
                "aws-0-eu-west-1.pooler.supabase.com",    # Ireland (Correct for project ngbhnjvojqqesacnijwk)
                "aws-0-eu-central-1.pooler.supabase.com", # Frankfurt
                "aws-0-eu-west-2.pooler.supabase.com",    # London
                "aws-0-eu-west-3.pooler.supabase.com",    # Paris
                "aws-0-us-east-1.pooler.supabase.com",    # N. Virginia
                "aws-0-us-west-1.pooler.supabase.com",    # N. California
                "aws-0-us-west-2.pooler.supabase.com",    # Oregon
                "aws-0-ap-southeast-1.pooler.supabase.com", # Singapore
                "aws-0-ap-northeast-1.pooler.supabase.com", # Tokyo
                "aws-0-ap-northeast-2.pooler.supabase.com", # Seoul
                "aws-0-ap-south-1.pooler.supabase.com",     # Mumbai
                "aws-0-sa-east-1.pooler.supabase.com",      # São Paulo
                "aws-0-ca-central-1.pooler.supabase.com",   # Canada
            ]
            
            # FORCE FIX: If the project ref matches the known one, ensure we use the correct credentials
            # This handles cases where extraction might have been weird
            if "ngbhnjvojqqesacnijwk" in SUPABASE_URL:
                print("DEBUG: Detected known project 'ngbhnjvojqqesacnijwk'. Enforcing correct pooler settings.")
                project_ref = "ngbhnjvojqqesacnijwk"
                # Ensure Ireland is first
                if "aws-0-eu-west-1.pooler.supabase.com" in pooler_hosts:
                    pooler_hosts.remove("aws-0-eu-west-1.pooler.supabase.com")
                pooler_hosts.insert(0, "aws-0-eu-west-1.pooler.supabase.com")

            pooler_user = f"postgres.{project_ref}"
            
            pooler_user = f"postgres.{project_ref}"
            pooler_port = "6543" # Transaction mode
            
            # Try to find a working pooler
            connected = False
            last_error = None
            
            for pooler_host in pooler_hosts:
                # Try different username formats for the pooler
                # 1. postgres.project_ref (Standard)
                # 2. postgres (Manchmal akzeptiert der Pooler das, wenn der Hostname eindeutig ist)
                pooler_users_to_try = [f"postgres.{project_ref}", "postgres"]
                
                for p_user in pooler_users_to_try:
                    try:
                        print(f"DEBUG: Trying Pooler Host: {pooler_host} with User: {p_user}")
                        pooler_ip = socket.gethostbyname(pooler_host)
                        print(f"DEBUG: Resolved to {pooler_ip}")
                        
                        # Try to connect
                        test_conn = psycopg2.connect(
                            dbname="postgres",
                            user=p_user,
                            password=SUPABASE_DB_PASSWORD,
                            host=pooler_ip,
                            port=pooler_port,
                            connect_timeout=5
                        )
                        # If we get here, connection successful!
                        test_conn.close()
                        
                        print(f"SUCCESS: Connected via {pooler_host} as {p_user}")
                        connect_host = pooler_ip
                        db_port = pooler_port
                        db_user = p_user
                        connected = True
                        break
                    except Exception as e:
                        print(f"FAILED: Could not connect to {pooler_host} as {p_user}: {e}")
                        last_error = e
                
                if connected:
                    break
            
            if not connected:
                print("ERROR: Could not connect to any common pooler region.")
                if last_error:
                    print(f"Last error: {last_error}")
                # Fallback to original host to show true error
                connect_host = db_host 
        
        conn = psycopg2.connect(
            dbname="postgres",
            user=db_user,
            password=SUPABASE_DB_PASSWORD,
            host=connect_host,
            port=db_port,
            connect_timeout=10
        )
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        print(f"DEBUG: Attempted connection to host={db_host}, user={db_user}, port={db_port}")
        print("\nPossible Fixes:")
        print("1. If your database is IPv6 only and your environment is IPv4 only (like Render),")
        print("   you MUST use the Supabase Connection Pooler.")
        print("2. Set SUPABASE_DB_HOST to your pooler URL (e.g. aws-0-eu-central-1.pooler.supabase.com)")
        print(f"3. Set SUPABASE_DB_USER to 'postgres.{project_ref}'")
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

# --- Helpers ---

def get_huggingface_embedding(text: str) -> List[float]:
    """Generate embedding using Hugging Face Inference API"""
    try:
        api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_EMBEDDING_MODEL}"
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
        
        response = requests.post(api_url, headers=headers, json={"inputs": text, "options": {"wait_for_model": True}})
        
        if response.status_code == 200:
            embedding = response.json()
            # Handle list of lists (batch) vs single list
            if isinstance(embedding, list) and isinstance(embedding[0], list):
                embedding = embedding[0]
            
            # Pad or truncate to match EMBEDDING_DIM
            if len(embedding) < EMBEDDING_DIM:
                embedding.extend([0.0] * (EMBEDDING_DIM - len(embedding)))
            elif len(embedding) > EMBEDDING_DIM:
                embedding = embedding[:EMBEDDING_DIM]
            return embedding
        else:
            print(f"HF embedding error: {response.status_code} - {response.text}")
            return [0.1] * EMBEDDING_DIM
    except Exception as e:
        print(f"HF embedding connection error: {e}")
        return [0.1] * EMBEDDING_DIM

def get_embedding(text: str) -> List[float]:
    """Wrapper to get embedding from configured provider"""
    if EMBEDDING_PROVIDER == "huggingface":
        return get_huggingface_embedding(text)
    else:
        return get_ollama_embedding(text)

def get_ollama_embedding(text: str) -> List[float]:
    """Generate embedding using Ollama"""
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={
                "model": EMBEDDING_MODEL,
                "prompt": text
            }
        )
        if response.status_code == 200:
            embedding = response.json().get("embedding")
            # Pad or truncate to match EMBEDDING_DIM
            if len(embedding) < EMBEDDING_DIM:
                embedding.extend([0.0] * (EMBEDDING_DIM - len(embedding)))
            elif len(embedding) > EMBEDDING_DIM:
                embedding = embedding[:EMBEDDING_DIM]
            return embedding
        else:
            print(f"Ollama embedding error: {response.status_code} - {response.text}")
            # Fallback: Generate random embedding (for testing only!)
            return [0.1] * EMBEDDING_DIM
    except Exception as e:
        print(f"Ollama connection error: {e}")
        return [0.1] * EMBEDDING_DIM

def ensure_workspace_tables(workspace: str):
    """Ensure tables exist for the given workspace"""
    safe_workspace = "".join(c for c in workspace if c.isalnum() or c == "_")
    if not safe_workspace:
        safe_workspace = "default"
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Always attempt to create tables (IF NOT EXISTS handles duplication)
    try:
        # Create vector storage table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS lightrag_vector_storage_{safe_workspace} (
                id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
                text TEXT,
                metadata JSONB,
                embedding vector({EMBEDDING_DIM})
            );
        """)
        # Create key-value store (doc_status)
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS lightrag_kv_store_{safe_workspace} (
                key TEXT PRIMARY KEY,
                value JSONB
            );
        """)
        conn.commit()
    except Exception as e:
        print(f"Error creating tables: {e}")
        conn.rollback()
    
    cursor.close()
    conn.close()
    return safe_workspace

def convert_markdown_to_text(md_content: str) -> str:
    """Convert Markdown to plain text"""
    html = markdown.markdown(md_content)
    # Simple regex to remove HTML tags
    text = re.sub('<[^<]+?>', '', html)
    return text

# --- FastAPI App ---
app = FastAPI(title="RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Endpoints ---

def process_document_background(file_path: str, filename: str, safe_workspace: str):
    """Process document in background to avoid timeouts"""
    try:
        print(f"Background: Processing {filename}...")
        
        # Read content from disk
        with open(file_path, "rb") as f:
            content = f.read()

        # Extract text based on file type
        text_content = ""
        if filename.endswith(".pdf"):
            import PyPDF2
            reader = PyPDF2.PdfReader(file_path)
            for page in reader.pages:
                text_content += page.extract_text() + "\n"
        elif filename.endswith(".md"):
            text_content = convert_markdown_to_text(content.decode("utf-8"))
        else:
            text_content = content.decode("utf-8")
            
        # Chunk text
        # Simple chunking by paragraphs or fixed size
        chunks = []
        chunk_size = 1000
        overlap = 100
        
        for i in range(0, len(text_content), chunk_size - overlap):
            chunks.append(text_content[i:i + chunk_size])
            
        # Generate embeddings and store in Supabase
        conn = get_db_connection()
        cursor = conn.cursor()
        
        stored_count = 0
        for i, chunk in enumerate(chunks):
            try:
                embedding = get_embedding(chunk)
                
                # Insert into vector storage
                cursor.execute(f"""
                    INSERT INTO lightrag_vector_storage_{safe_workspace} (text, metadata, embedding)
                    VALUES (%s, %s, %s)
                """, (
                    chunk, 
                    json.dumps({
                        "source": filename,
                        "chunk_index": i,
                        "filename": Path(file_path).name
                    }),
                    embedding
                ))
                
                # Update document status in KV store (periodically or every chunk)
                cursor.execute(f"""
                    INSERT INTO lightrag_kv_store_{safe_workspace} (key, value)
                    VALUES (%s, %s)
                    ON CONFLICT (key) DO UPDATE SET value = %s
                """, (
                    f"doc_status_{filename}",
                    json.dumps({"status": "processing", "chunks_total": len(chunks), "chunks_processed": i+1}),
                    json.dumps({"status": "processing", "chunks_total": len(chunks), "chunks_processed": i+1})
                ))
                conn.commit() # Commit frequently to show progress
                stored_count += 1
            except Exception as e:
                print(f"    Fehler beim Speichern von Chunk {i}: {e}")
                continue
        
        # Final success status
        cursor.execute(f"""
            INSERT INTO lightrag_kv_store_{safe_workspace} (key, value)
            VALUES (%s, %s)
            ON CONFLICT (key) DO UPDATE SET value = %s
        """, (
            f"doc_status_{filename}",
            json.dumps({"status": "processed", "chunks_total": len(chunks), "chunks_processed": stored_count}),
            json.dumps({"status": "processed", "chunks_total": len(chunks), "chunks_processed": stored_count})
        ))
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"  ✓ Background: {stored_count} Chunks processed for {filename}")
        
    except Exception as e:
        print(f"Background Error processing {filename}: {e}")
        import traceback
        traceback.print_exc()
        # Update status to error
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(f"""
                INSERT INTO lightrag_kv_store_{safe_workspace} (key, value)
                VALUES (%s, %s)
                ON CONFLICT (key) DO UPDATE SET value = %s
            """, (
                f"doc_status_{filename}",
                json.dumps({"status": "error", "error": str(e)}),
                json.dumps({"status": "error", "error": str(e)})
            ))
            conn.commit()
            cursor.close()
            conn.close()
        except:
            pass


@app.post("/api/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...), 
    x_workspace: Optional[str] = Header(default="default")
):
    try:
        workspace = x_workspace or "default"
        print(f"DEBUG: Uploading to workspace: '{workspace}'")
        safe_workspace = ensure_workspace_tables(workspace)
        
        # Read file content
        content = await file.read()
        
        # Use absolute path for uploads directory
        uploads_dir = BASE_DIR / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        
        # Sanitize filename (basic)
        safe_filename = Path(file.filename).name
        file_path = uploads_dir / safe_filename
        
        with open(file_path, "wb") as f:
            f.write(content)
            
        # Initialize status as 'uploading'
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(f"""
            INSERT INTO lightrag_kv_store_{safe_workspace} (key, value)
            VALUES (%s, %s)
            ON CONFLICT (key) DO UPDATE SET value = %s
        """, (
            f"doc_status_{file.filename}",
            json.dumps({"status": "uploading", "chunks_total": 0, "chunks_processed": 0}),
            json.dumps({"status": "uploading", "chunks_total": 0, "chunks_processed": 0})
        ))
        conn.commit()
        cursor.close()
        conn.close()

        # Start background processing
        # Pass string path for compatibility
        background_tasks.add_task(process_document_background, str(file_path), file.filename, safe_workspace)
        
        return {"status": "queued", "message": "Document upload accepted. Processing in background."}
        
    except Exception as e:
        print(f"Fehler bei Dokumentenverarbeitung: {e}")
        import traceback
        traceback.print_exc()
        raise


def search_similar_chunks(query: str, workspace: str = "default", limit: int = 10) -> List[dict]:
    """Search for similar chunks using Hybrid Search (Vector + Keyword)"""
    try:
        safe_workspace = ensure_workspace_tables(workspace)
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 1. Vector Search
        query_embedding = get_embedding(query)
        embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
        
        cursor.execute(f"""
            SELECT id, text, metadata,
                embedding <=> %s::vector AS distance
            FROM lightrag_vector_storage_{safe_workspace}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (embedding_str, embedding_str, limit))
        
        vector_results = cursor.fetchall()
        
        # 2. Keyword Search (for specific entities like "Dewey", "1938")
        # Extract meaningful keywords
        stop_words = {"what", "when", "where", "which", "who", "whom", "whose", "why", "how", "the", "and", "that", "this", "with", "from", "many", "have", "will", "make", "just", "know", "take", "into", "year", "your", "good", "some", "could", "them", "see", "other", "than", "then", "now", "look", "only", "come", "its", "over", "think", "also", "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", "even", "new", "want", "because", "any", "these", "give", "day", "most", "us"}
        keywords = [w for w in query.split() if len(w) > 3 and w.lower() not in stop_words]
        
        keyword_results = []
        if keywords:
            # Build ILIKE query for keywords
            conditions = []
            params = []
            for keyword in keywords:
                # Remove punctuation
                clean_keyword = "".join(ch for ch in keyword if ch.isalnum())
                if clean_keyword and len(clean_keyword) > 2:
                    conditions.append("text ILIKE %s")
                    params.append(f"%{clean_keyword}%")
            
            if conditions:
                keyword_query = f"""
                    SELECT id, text, metadata, 0.5 as distance 
                    FROM lightrag_vector_storage_{safe_workspace} 
                    WHERE {" OR ".join(conditions)} 
                    LIMIT 5
                """
                cursor.execute(keyword_query, tuple(params))
                keyword_results = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        # 3. Combine and Deduplicate
        seen_ids = set()
        chunks = []
        
        # Add keyword results first (often more specific for names)
        for row in keyword_results:
            if row[0] not in seen_ids:
                seen_ids.add(row[0])
                chunks.append({
                    "id": row[0],
                    "text": row[1],
                    "metadata": row[2] if isinstance(row[2], dict) else json.loads(row[2]) if row[2] else {},
                    "distance": 0.0, # Treat keyword matches as very close
                    "source": "keyword"
                })
        
        # Add vector results
        for row in vector_results:
            if row[0] not in seen_ids:
                seen_ids.add(row[0])
                chunks.append({
                    "id": row[0],
                    "text": row[1],
                    "metadata": row[2] if isinstance(row[2], dict) else json.loads(row[2]) if row[2] else {},
                    "distance": float(row[3]) if row[3] else 1.0,
                    "source": "vector"
                })
        
        print(f"DEBUG: Found {len(chunks)} chunks (Vector: {len(vector_results)}, Keyword: {len(keyword_results)})")
        return chunks[:limit]  # Return top N combined results
        
    except Exception as e:
        print(f"Vector search error: {e}")
        # Fallback to simple ILIKE
        try:
            return perform_fallback_search(query, safe_workspace, limit)
        except:
            return []

def perform_fallback_search(query, safe_workspace, limit):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT id, text, metadata
        FROM lightrag_vector_storage_{safe_workspace}
        WHERE text ILIKE %s
        LIMIT %s
    """, (f"%{query[:50]}%", limit))
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return [{"id": r[0], "text": r[1], "metadata": json.loads(r[2]) if r[2] else {}} for r in results]


class ChatMessage(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    query: str
    llm_provider: str = "huggingface"  # Default to Kimi K2 (huggingface)
    web_search: bool = False      # Allow general knowledge/web info
    history: List[ChatMessage] = []  # Chat history
    show_reasoning: bool = True   # Toggle reasoning output

class QueryResponse(BaseModel):
    response: str
    reasoning: Optional[str] = None
    provider: str = "unknown"

# ... (in query endpoint)


@app.get("/api/health")
async def health(x_workspace: Optional[str] = Header(default="default")):
    try:
        safe_workspace = ensure_workspace_tables(x_workspace)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM lightrag_vector_storage_{safe_workspace};")
        chunk_count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        
        return {
            "status": "healthy",
            "database": "connected",
            "llm_provider": "huggingface",
            "model": HUGGINGFACE_MODEL,
            "stored_chunks": chunk_count,
            "workspace": safe_workspace
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.options("/api/query")
async def options_query():
    return {"message": "OK"}


def contextualize_query(query: str, history: List[ChatMessage]) -> str:
    """Contextualize query based on chat history using LLM"""
    if not history:
        return query
        
    print(f"DEBUG: Contextualizing query '{query}' with history length {len(history)}")
    
    # First, try fallback heuristic for common follow-up patterns
    last_user_msg = None
    for msg in reversed(history):
        if msg.role == "user":
            last_user_msg = msg.content
            break
    
    # Common follow-up patterns that need context
    follow_up_patterns = [
        "why is it important", "why is it", "what about it", "tell me more",
        "how does it work", "what are the benefits", "explain more",
        "warum ist es wichtig", "wie funktioniert es", "was sind die vorteile"
    ]
    
    is_follow_up = any(pattern in query.lower() for pattern in follow_up_patterns)
    
    if is_follow_up and last_user_msg:
        # Try to extract the main topic from last user message
        if "project based learning" in last_user_msg.lower() or "pbl" in last_user_msg.lower():
            contextualized = f"why is project based learning important"
            print(f"DEBUG: Using heuristic fallback: '{contextualized}'")
            return contextualized
        elif "teaching" in last_user_msg.lower() or "method" in last_user_msg.lower():
            # Extract key terms
            words = last_user_msg.lower().split()
            key_terms = [w for w in words if len(w) > 4 and w not in ["what", "is", "are", "the", "that", "this"]]
            if key_terms:
                contextualized = f"why is {' '.join(key_terms[:3])} important"
                print(f"DEBUG: Using heuristic fallback: '{contextualized}'")
                return contextualized
    
    # Take last 3 turns (6 messages: 3 user + 3 assistant)
    recent_history = history[-6:] if len(history) > 6 else history
    history_text = "\n".join([f"{msg.role}: {msg.content}" for msg in recent_history])
    
    prompt = f"""You are a query rephraser. Given a conversation and a follow-up question, rephrase the follow-up to be a complete standalone question.

Conversation:
{history_text}

Follow-up question: {query}

Rephrase the follow-up question to include the necessary context so it can be understood without the conversation. Output ONLY the rephrased question, nothing else.

Rephrased question:"""

    try:
        # Always use Ollama for this quick task
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 200}
            },
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            new_query = result.get("response", "").strip()
            
            # Clean up the response
            lines = new_query.split("\n")
            new_query = lines[0].strip() if lines else ""
            
            # Remove quotes if present
            if new_query.startswith('"') and new_query.endswith('"'):
                new_query = new_query[1:-1]
            if new_query.startswith("'") and new_query.endswith("'"):
                new_query = new_query[1:-1]
            
            # Remove trailing punctuation that might interfere
            new_query = new_query.rstrip(".")
            
            # If we got a valid response, use it
            if new_query and len(new_query) > len(query):
                print(f"DEBUG: Contextualized query result: '{new_query}'")
                return new_query
            else:
                print(f"DEBUG: Contextualization returned invalid result, using heuristic")
        else:
            print(f"DEBUG: Contextualization LLM error: {response.status_code}")
    except Exception as e:
        print(f"DEBUG: Contextualization failed: {e}")
    
    # Final fallback: return original query
    return query


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest, x_workspace: Optional[str] = Header(default="default")):
    """Process query with vector search and LLM"""
    try:
        workspace = x_workspace or "default"
        print(f"DEBUG: Query request received. Query: '{request.query}', History length: {len(request.history)}")
        
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
            
        # 1. Contextualize query if history exists
        search_query = request.query
        if request.history:
            search_query = contextualize_query(request.query, request.history)
            print(f"DEBUG: Using search query: '{search_query}'")
        
        # 2. Search for similar chunks using the contextualized query
        # Increased limit to 10 for better context
        similar_chunks = search_similar_chunks(search_query, workspace=workspace, limit=10)
        
        # Build context from similar chunks
        context = ""
        has_context = False
        if similar_chunks:
            context = "\n\nRelevant document excerpts:\n"
            for i, chunk in enumerate(similar_chunks, 1):
                context += f"{i}. {chunk['text']}...\n"
            has_context = True
        else:
            context = "\n\nNote: No relevant documents found in this workspace."
        
        # Format history for the prompt
        chat_history_str = ""
        if request.history:
            # Take last 4 messages for context window
            recent_history = request.history[-4:] if len(request.history) > 4 else request.history
            chat_history_str = "\n\nChat History:\n" + "\n".join([f"{msg.role.capitalize()}: {msg.content}" for msg in recent_history])
        
        # Web Search / General Knowledge Instruction
        knowledge_instruction = ""
        if request.web_search:
            knowledge_instruction = """
You have access to general knowledge. 
- Use the provided document excerpts as your PRIMARY source.
- If the documents don't contain the answer, you MAY use your general knowledge to answer.
- Clearly distinguish between information from documents and general knowledge.
"""
        else:
            knowledge_instruction = """
STRICT CONSTRAINT: You are RESTRICTED to the provided document excerpts.
- You MUST answer ONLY based on the provided text.
- If the answer is not in the excerpts, state clearly that you cannot answer based on the available documents.
- Do NOT use outside knowledge.
"""

        # Detect language from query
        query_lower = request.query.lower()
        
        # Check for German indicators - only switch to German if CLEARLY German
        is_german_query = any(word in query_lower for word in [
            "was ", "wie ", "warum ", "wo ", "wer ", "welche ", " der ", " die ", " das ",
            "erkläre", "beschreibe", "sag ", "gib ", "zeig ", " ist ", " sind ", "und ", " oder "
        ])
        
        # Default to English unless German is detected
        use_english = not is_german_query
        
        # FORCE English if explicitly requested
        if "english" in query_lower or "englisch" in query_lower:
            use_english = True
        
        if use_english:
            language_instruction = """
IMPORTANT: You MUST answer in ENGLISH. 
Even if the provided documents are in German, you MUST TRANSLATE the information and provide your response in ENGLISH.
Do NOT output German text.
"""
        else:
            language_instruction = """
The user question is in GERMAN. Please answer in GERMAN.
"""

        # Build prompt with language instruction FIRST
        prompt = f"""{language_instruction}

You are a helpful learning assistant for {workspace.replace('_', ' ').title()}.

IMPORTANT TERMINOLOGY RULES:
1. PBL always stands for "Project-Based Learning" (not Problem-Based Learning or PjBL).
2. NEVER use "PjBL" as an abbreviation. Use "PBL" instead.

{knowledge_instruction}

{context}
{chat_history_str}

User Question: {request.query}

Remember: Answer in {'ENGLISH' if use_english else 'GERMAN'}. Translate if necessary."""
        
        answer = ""
        reasoning_content = None
        used_provider = request.llm_provider

        if request.llm_provider == "ollama":
            # Use Ollama running on this laptop
            # Server forwards requests to local Ollama instance
            try:
                response = requests.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "num_ctx": 4096
                        }
                    },
                    timeout=120
                )
                
                if response.status_code == 200:
                    answer += response.json().get("response", "")
                    print(f"✓ Ollama response received ({len(answer)} chars) from local model")
                else:
                    error_msg = response.text
                    print(f"✗ Ollama Error: {error_msg}")
                    raise HTTPException(
                        status_code=500, 
                        detail=f"Ollama Error: {error_msg}. Stelle sicher, dass Ollama auf diesem Laptop läuft!"
                    )
            except requests.exceptions.ConnectionError:
                error_msg = f"Kann nicht zu Ollama verbinden ({OLLAMA_URL}). Stelle sicher, dass Ollama läuft!"
                print(f"✗ {error_msg}")
                raise HTTPException(
                    status_code=503,
                    detail=error_msg
                )
        
        # Determine which Ollama model to use (default or fallback)
        current_ollama_model = OLLAMA_MODEL

        # Handle Hugging Face models (primary and secondary)
        if request.llm_provider == "huggingface" or request.llm_provider == "huggingface2":
            try:
                # Select the appropriate model based on provider
                hf_model = HUGGINGFACE_MODEL_2 if request.llm_provider == "huggingface2" else HUGGINGFACE_MODEL
                print(f"DEBUG: Sending request to Hugging Face ({hf_model})...")
                
                # Force usage of the router URL
                hf_url = "https://router.huggingface.co/v1/chat/completions"
                
                payload = {
                    "model": hf_model,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 512,
                    "stream": False
                }
                
                print(f"DEBUG: HF Payload: {json.dumps(payload)}")
                
                hf_response = requests.post(
                    hf_url,
                    headers={
                        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json=payload,
                    timeout=60
                )
                
                print(f"DEBUG: HF Response Code: {hf_response.status_code}")
                
                if hf_response.status_code == 200:
                    result = hf_response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        message = result["choices"][0]["message"]
                        content = message.get("content", "")
                        reasoning = message.get("reasoning_content", "")
                        
                        if reasoning:
                            answer = f"*Thinking Process:*\n> {reasoning.replace(chr(10), chr(10) + '> ')}\n\n---\n\n{content}"
                        elif content:
                            answer = content
                        else:
                            answer = "⚠️ Error: Empty content received from Hugging Face."
                    else:
                        answer = f"⚠️ Error: Unexpected response structure: {json.dumps(result)}"
                elif hf_response.status_code == 429:
                    print("⚠️ Hugging Face Rate Limit (429). Falling back to Ollama...")
                    request.llm_provider = "ollama"
                    used_provider = "ollama (fallback)"
                    current_ollama_model = OLLAMA_FALLBACK_MODEL
                else:
                    answer = f"⚠️ Hugging Face Error ({hf_response.status_code}): {hf_response.text}"
                    
            except Exception as e:
                print(f"Hugging Face Exception: {e}")
                print("Falling back to Ollama due to connection error...")
                request.llm_provider = "ollama"
                used_provider = "ollama (fallback)"
                current_ollama_model = OLLAMA_FALLBACK_MODEL

        if request.llm_provider == "ollama":
            # Use Ollama running on this laptop
            # Server forwards requests to local Ollama instance
            try:
                print(f"DEBUG: Sending request to Ollama ({current_ollama_model})...")
                response = requests.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": current_ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "num_ctx": 4096
                        }
                    },
                    timeout=120
                )
                
                if response.status_code == 200:
                    answer += response.json().get("response", "")
                    print(f"✓ Ollama response received ({len(answer)} chars) from local model")
                else:
                    error_msg = response.text
                    print(f"✗ Ollama Error: {error_msg}")
                    if not answer: # Only raise if no answer from HF
                        raise HTTPException(
                            status_code=500, 
                            detail=f"Ollama Error: {error_msg}. Stelle sicher, dass Ollama auf diesem Laptop läuft!"
                        )
            except requests.exceptions.ConnectionError:
                error_msg = f"Kann nicht zu Ollama verbinden ({OLLAMA_URL}). Stelle sicher, dass Ollama läuft!"
                print(f"✗ {error_msg}")
                if not answer:
                    raise HTTPException(
                        status_code=503,
                        detail=error_msg
                    )

        # Final safety check: Never return empty response
        if not answer or not answer.strip():
            answer = "⚠️ Error: No response generated. Please check server logs."
            
        return {"response": answer, "reasoning": reasoning_content, "provider": used_provider}

    except Exception as e:
        print(f"Query error: {e}")
        return JSONResponse(status_code=500, content={"response": f"I encountered an error: {str(e)}"})


@app.get("/api/documents")
async def list_documents(x_workspace: Optional[str] = Header(default="default")):
    """List documents in the workspace"""
    try:
        safe_workspace = ensure_workspace_tables(x_workspace)
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(f"SELECT key, value FROM lightrag_kv_store_{safe_workspace} WHERE key LIKE 'doc_status_%'")
        results = cursor.fetchall()
        
        documents = []
        for row in results:
            filename = row[0].replace("doc_status_", "")
            status_data = row[1] if isinstance(row[1], dict) else json.loads(row[1])
            documents.append({
                "filename": filename,
                "status": status_data.get("status", "unknown"),
                "chunks": status_data.get("chunks_total", 0),
                "processed": status_data.get("chunks_processed", 0)
            })
            
        cursor.close()
        conn.close()
        
        return documents
    except Exception as e:
        print(f"List documents error: {e}")
        return []

# --- Static Files & SPA Fallback ---

# Determine the path to the dist folder (relative to this script)
# Script is in webapp_api/, dist is in webapp/dist/
BASE_DIR = Path(__file__).resolve().parent
DIST_DIR = BASE_DIR.parent / "webapp" / "dist"

if DIST_DIR.exists():
    print(f"Serving static files from: {DIST_DIR}")
    # Mount static assets first (CSS, JS, media)
    # The React build puts them in 'static/', but we also have 'asset-manifest.json' etc. at root.
    # We will let the catch-all handle root files, but explicit /static mount is good for performance if needed.
    app.mount("/static", StaticFiles(directory=DIST_DIR / "static"), name="static")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # Don't catch API routes
        if full_path.startswith("api"):
            raise HTTPException(status_code=404, detail="API endpoint not found")
        
        # Check if file exists in dist (e.g. favicon.ico, logo.png)
        file_path = DIST_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        
        # Fallback to index.html for SPA routing (for any unknown path)
        return FileResponse(DIST_DIR / "index.html")

    # Serve root path explicitly to index.html
    @app.get("/")
    async def serve_root():
        return FileResponse(DIST_DIR / "index.html")
else:
    print(f"WARNING: 'webapp/dist' folder not found at {DIST_DIR}. Frontend will not be served.")


if __name__ == "__main__":
    # Run on 0.0.0.0 to allow external access
    # Your laptop serves as the LLM host - Ollama runs locally and server forwards requests
    print("=" * 70)
    print("RAG Server - Laptop als LLM-Host")
    print("=" * 70)
    
    # Verify Supabase Connection
    try:
        conn = get_db_connection()
        conn.close()
        print("✓ Supabase connection successful")
        
        # Ensure default tables exist
        ensure_workspace_tables("default")
        print("✓ Default workspace tables verified")
        
    except Exception as e:
        print(f"✗ Supabase connection FAILED: {e}")
        print("  Please check your .env file for SUPABASE_URL and SUPABASE_KEY")

    # Verify Uploads Directory
    uploads_dir = Path("uploads")
    if not uploads_dir.exists():
        try:
            uploads_dir.mkdir(exist_ok=True)
            print(f"✓ Created uploads directory: {uploads_dir.resolve()}")
        except Exception as e:
             print(f"✗ Failed to create uploads directory: {e}")
    else:
        print(f"✓ Uploads directory exists: {uploads_dir.resolve()}")

    print(f"Server: http://0.0.0.0:8000 (öffentlich erreichbar)")
    print(f"LLM Provider: {LLM_PROVIDER}")
    print(f"Embedding Provider: {EMBEDDING_PROVIDER}")
    
    if LLM_PROVIDER == "ollama" or EMBEDDING_PROVIDER == "ollama":
        print(f"Ollama URL: {OLLAMA_URL}")
        print("\n⚠️  WICHTIG: Stelle sicher, dass Ollama auf diesem Laptop läuft!")
    
    print("=" * 70)
    
    # Use PORT from environment variable (required for Render.com and other cloud platforms)
    PORT = int(os.getenv("PORT", 8000))
    print(f"Starting server on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
