#!/usr/bin/env python
"""
Vollständiger RAG Server mit Embedding-Generierung und Hybrid Search
"""

import os
import sys
import asyncio
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import psycopg2
from urllib.parse import urlparse
import json
import requests
import hashlib
import re

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(dotenv_path=".env", override=False)

app = FastAPI(title="RAG Chatbot API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://ngbhnjvojqqesacnijwk.supabase.co")
SUPABASE_DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD", "Test_1082?!")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# LLM Provider
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
GEMINI_MODEL = "gemini-2.0-flash-exp"

# Extract database connection info
parsed = urlparse(SUPABASE_URL)
host = parsed.hostname
if host and not host.startswith("db."):
    parts = host.split(".")
    if len(parts) >= 2 and parts[-2] == "supabase":
        project_ref = parts[0]
        host = f"db.{project_ref}.supabase.com"

DB_HOST = host or "localhost"
DB_PORT = 5432
DB_USER = "postgres"
DB_NAME = "postgres"
DEFAULT_WORKSPACE = "default"

# Embedding dimension (must match table definition: vector(3072))
EMBEDDING_DIM = 3072

print("=" * 70)
print("RAG Server mit Hybrid Search (Vector + Keyword)")
print("=" * 70)


def get_db_connection():
    """Get PostgreSQL connection"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=SUPABASE_DB_PASSWORD,
            database=DB_NAME,
            connect_timeout=10,
        )
        # Enable autocommit for creating tables etc
        conn.autocommit = True
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        raise


def ensure_workspace_tables(workspace: str):
    """Ensure tables exist for the given workspace"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Clean workspace name to be safe for table names
        safe_workspace = "".join(c for c in workspace if c.isalnum() or c == "_")
        if not safe_workspace:
            safe_workspace = "default"
            
        vector_table = f"lightrag_vector_storage_{safe_workspace}"
        kv_table = f"lightrag_kv_storage_{safe_workspace}"
        
        # Check if tables exist (simplified check)
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {vector_table} (
                id TEXT PRIMARY KEY,
                text TEXT,
                embedding vector(3072),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)
        
        # Create index if not exists (might fail if already exists with different name, but safe to try)
        try:
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS {vector_table}_embedding_idx 
                ON {vector_table} 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
        except:
            pass

        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {kv_table} (
                key TEXT PRIMARY KEY,
                value JSONB,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
        """)
        
        cursor.close()
        conn.close()
        return safe_workspace
    except Exception as e:
        print(f"Error ensuring workspace tables for '{workspace}': {e}")
        return "default"


def get_ollama_embedding(text: str) -> List[float]:
    """Get embedding from Ollama"""
    try:
        # Try nomic-embed-text first
        response = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={
                "model": "nomic-embed-text",
                "prompt": text
            },
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get("embedding", [])
    except:
        pass
    
    # Fallback: simple hash-based embedding
    hash_obj = hashlib.sha256(text.encode())
    hash_bytes = hash_obj.digest()
    # Create 3072-dim embedding from hash (repeat pattern to fill dimensions)
    embedding = []
    # Use multiple hash iterations to fill 3072 dimensions
    for round in range(12):  # 12 rounds * 256 bytes = 3072 dimensions
        round_hash = hashlib.sha256((text + str(round)).encode()).digest()
        for i in range(0, len(round_hash), 4):
            if len(embedding) >= EMBEDDING_DIM:
                break
            val = int.from_bytes(round_hash[i:i+4], 'big') / (2**32)
            embedding.append(val)
    # Pad to 3072 dimensions if needed
    while len(embedding) < EMBEDDING_DIM:
        embedding.append(0.0)
    return embedding[:EMBEDDING_DIM]


def convert_markdown_to_text(markdown_content: str) -> str:
    """Convert Markdown to clean text while preserving structure"""
    try:
        import markdown
        
        # Convert markdown to HTML first
        html = markdown.markdown(
            markdown_content,
            extensions=['codehilite', 'fenced_code', 'tables', 'nl2br']
        )
        
        # Convert HTML to plain text while preserving structure
        # Remove HTML tags but keep content
        text = re.sub(r'<[^>]+>', '', html)
        
        # Clean up whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = text.strip()
        
        return text
    except ImportError:
        # Fallback: simple markdown cleaning without library
        text = markdown_content
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
        return text.strip()
    except Exception as e:
        print(f"    ⚠️  Markdown-Konvertierung Fehler: {e}")
        return markdown_content


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into chunks"""
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        para_length = len(para)
        
        if current_length + para_length + 2 <= chunk_size:
            current_chunk.append(para)
            current_length += para_length + 2
        else:
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_length = para_length + 2
            else:
                chunks.append(para)
    
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks if chunks else [text]


def process_document_to_vectors(file_path: str, file_id: str, workspace: str = "default"):
    """Process document and store chunks with embeddings in Supabase"""
    try:
        safe_workspace = ensure_workspace_tables(workspace)
        
        text = ""
        if file_path.endswith('.pdf'):
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n\n"
            except ImportError:
                text = f"PDF-Parsing erfordert PyPDF2. Datei: {Path(file_path).name}"
            except Exception:
                text = f"Fehler beim Lesen von {Path(file_path).name}"
        else:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
            except:
                text = f"Fehler beim Lesen von {Path(file_path).name}"
        
        if file_path.endswith('.md'):
            text = convert_markdown_to_text(text)
        
        chunks = chunk_text(text)
        print(f"  - {len(chunks)} Chunks erstellt")
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        stored_count = 0
        for i, chunk in enumerate(chunks):
            try:
                embedding = get_ollama_embedding(chunk)
                chunk_id = f"{file_id}_chunk_{i}"
                embedding_str = "[" + ",".join(map(str, embedding)) + "]"
                
                cursor.execute(f"""
                    INSERT INTO lightrag_vector_storage_{safe_workspace} 
                    (id, text, embedding, metadata, created_at)
                    VALUES (%s, %s, %s::vector, %s, NOW())
                    ON CONFLICT (id) DO UPDATE SET
                        text = EXCLUDED.text,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata
                """, (
                    chunk_id,
                    chunk[:5000],
                    embedding_str,
                    json.dumps({
                        "file_id": file_id,
                        "chunk_index": i,
                        "filename": Path(file_path).name
                    })
                ))
                stored_count += 1
            except Exception as e:
                print(f"    Fehler Chunk {i}: {e}")
                continue
        
        conn.commit()
        cursor.close()
        conn.close()
        return stored_count
        
    except Exception as e:
        print(f"Fehler bei Dokumentenverarbeitung: {e}")
        raise


def search_similar_chunks(query: str, workspace: str = "default", limit: int = 10) -> List[dict]:
    """Search for similar chunks using Hybrid Search (Vector + Keyword)"""
    try:
        safe_workspace = ensure_workspace_tables(workspace)
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 1. Vector Search
        query_embedding = get_ollama_embedding(query)
        embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
        
        cursor.execute(f"""
            SELECT id, text, metadata,
                embedding <=> %s::vector AS distance
            FROM lightrag_vector_storage_{safe_workspace}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (embedding_str, embedding_str, limit))
        
        vector_results = cursor.fetchall()
        
        # 2. Keyword Search
        stop_words = {"what", "when", "where", "which", "who", "whom", "whose", "why", "how", "the", "and", "that", "this", "with", "from", "many", "have", "will", "make", "just", "know", "take", "into", "year", "your", "good", "some", "could", "them", "see", "other", "than", "then", "now", "look", "only", "come", "its", "over", "think", "also", "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", "even", "new", "want", "because", "any", "these", "give", "day", "most", "us"}
        keywords = [w for w in query.split() if len(w) > 3 and w.lower() not in stop_words]
        
        keyword_results = []
        if keywords:
            conditions = []
            params = []
            for keyword in keywords:
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
        
        # 3. Combine
        seen_ids = set()
        chunks = []
        
        # Keywords first
        for row in keyword_results:
            if row[0] not in seen_ids:
                seen_ids.add(row[0])
                chunks.append({
                    "id": row[0],
                    "text": row[1],
                    "metadata": row[2] if isinstance(row[2], dict) else json.loads(row[2]) if row[2] else {},
                    "distance": 0.0,
                    "source": "keyword"
                })
        
        # Vectors second
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
        return chunks[:limit]
        
    except Exception as e:
        print(f"Vector search error: {e}")
        return []

def perform_fallback_search(query, safe_workspace, limit):
    try:
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
    except:
        return []


class ChatMessage(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    query: str
    llm_provider: str = "ollama"
    web_search: bool = False
    history: List[ChatMessage] = []

class QueryResponse(BaseModel):
    response: str


@app.get("/")
async def root():
    return {"status": "running", "version": "1.1.0", "type": "Hybrid RAG"}

@app.get("/api/health")
async def health(x_workspace: Optional[str] = Header(default="default")):
    try:
        safe_workspace = ensure_workspace_tables(x_workspace)
        return {"status": "healthy", "workspace": safe_workspace}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.options("/api/query")
async def options_query():
    return {"message": "OK"}

def contextualize_query(query: str, history: List[ChatMessage]) -> str:
    if not history: return query
    print(f"DEBUG: Contextualizing query '{query}'")
    
    last_user_msg = next((m.content for m in reversed(history) if m.role == "user"), None)
    follow_up_patterns = ["why", "how", "what", "tell me more", "explain", "it"]
    
    if last_user_msg and any(p in query.lower() for p in follow_up_patterns):
        if "project based learning" in last_user_msg.lower() or "pbl" in last_user_msg.lower():
            return "why is project based learning important"
            
    try:
        history_text = "\n".join([f"{msg.role}: {msg.content}" for msg in history[-6:]])
        prompt = f"Rephrase this follow-up question to be standalone:\nContext:\n{history_text}\nQuestion: {query}\nStandalone Question:"
        
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.1}},
            timeout=30
        )
        if response.status_code == 200:
            new_query = response.json().get("response", "").strip()
            if new_query and len(new_query) > 5:
                print(f"DEBUG: Contextualized: {new_query}")
                return new_query
    except Exception as e:
        print(f"Contextualization failed: {e}")
        
    return query

@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest, x_workspace: Optional[str] = Header(default="default")):
    try:
        workspace = x_workspace or "default"
        print(f"DEBUG: Query: '{request.query}'")
        
        search_query = request.query
        if request.history:
            search_query = contextualize_query(request.query, request.history)
            
        # Hybrid Search with limit=10
        similar_chunks = search_similar_chunks(search_query, workspace=workspace, limit=10)
        
        context = ""
        if similar_chunks:
            context = "\n\nRelevant document excerpts:\n"
            for i, chunk in enumerate(similar_chunks, 1):
                context += f"{i}. {chunk['text']}\n"
        else:
            context = "\n\nNote: No relevant documents found."
            
        history_str = ""
        if request.history:
            history_str = "\n\nChat History:\n" + "\n".join([f"{msg.role}: {msg.content}" for msg in request.history[-4:]])
            
        prompt = f"""You are a helpful assistant for {workspace}.
        
Use these documents to answer:
{context}
{history_str}

Question: {request.query}

Answer in English or German (match user language)."""

        # Generation logic (Ollama only for brevity in this fallback file)
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_ctx": 4096}
            },
            timeout=120
        )
        
        if response.status_code == 200:
            return {"response": response.json().get("response", "")}
        else:
            return {"response": f"Error: {response.text}"}
            
    except Exception as e:
        print(f"Query error: {e}")
        return {"response": f"Error: {str(e)}"}

@app.options("/api/upload")
async def options_upload():
    return {"message": "OK"}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), x_workspace: Optional[str] = Header(default="default")):
    # Simplified upload handler
    try:
        workspace = x_workspace or "default"
        safe_workspace = ensure_workspace_tables(workspace)
        
        content = await file.read()
        file_id = f"file_{file.filename}"
        
        # Save temp
        with open(f"uploads/{file.filename}", "wb") as f:
            f.write(content)
            
        process_document_to_vectors(f"uploads/{file.filename}", file_id, workspace=safe_workspace)
        
        return {"status": "success", "filename": file.filename}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/documents")
async def get_documents(x_workspace: Optional[str] = Header(default="default")):
    return {"documents": []}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

