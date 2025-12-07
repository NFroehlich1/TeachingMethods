#!/usr/bin/env python
"""
Vollständiger RAG Server mit Embedding-Generierung

Dieser Server verarbeitet Dokumente vollständig und generiert Embeddings.
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
        host = f"db.{project_ref}.supabase.co"

DB_HOST = host or "localhost"
DB_PORT = 5432
DB_USER = "postgres"
DB_NAME = "postgres"
DEFAULT_WORKSPACE = "default"

# Embedding dimension (must match table definition: vector(3072))
EMBEDDING_DIM = 3072

print("=" * 70)
print("RAG Server mit vollständiger Dokumentenverarbeitung")
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


def get_gemini_embedding(text: str) -> List[float]:
    """Get embedding from Gemini"""
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document",
            title="Embedding of single text chunk"
        )
        return result['embedding']
    except Exception as e:
        print(f"Gemini embedding error: {e}")
        return []

def get_gemini_response(prompt: str) -> str:
    """Get response from Gemini"""
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini API Error: {str(e)}"


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
        from markdown.extensions import codehilite, fenced_code, tables
        
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
        # Remove markdown syntax but keep text
        text = markdown_content
        # Remove headers
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        # Remove bold/italic
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)
        # Remove links but keep text
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        # Remove code blocks
        text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        # Remove images
        text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', r'\1', text)
        # Clean up whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        return text.strip()
    except Exception as e:
        print(f"    ⚠️  Markdown-Konvertierung Fehler: {e}")
        return markdown_content


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into chunks, respecting paragraph boundaries for better context"""
    # First, try to split by paragraphs (double newlines)
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        para_length = len(para)
        
        # If paragraph fits in current chunk, add it
        if current_length + para_length + 2 <= chunk_size:
            current_chunk.append(para)
            current_length += para_length + 2  # +2 for \n\n
        else:
            # Save current chunk if it has content
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                # Overlap: keep last paragraph(s)
                overlap_paras = current_chunk[-1:] if len(current_chunk) > 0 else []
                current_chunk = overlap_paras
                current_length = sum(len(p) + 2 for p in current_chunk)
            
            # If single paragraph is too large, split it by words
            if para_length > chunk_size:
                words = para.split()
                for word in words:
                    word_length = len(word) + 1
                    if current_length + word_length > chunk_size and current_chunk:
                        chunks.append('\n\n'.join(current_chunk))
                        current_chunk = []
                        current_length = 0
                    current_chunk.append(word)
                    current_length += word_length
            else:
                current_chunk.append(para)
                current_length = para_length + 2
    
    # Add remaining chunk
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks if chunks else [text]


def process_document_to_vectors(file_path: str, file_id: str, workspace: str = "default"):
    """Process document and store chunks with embeddings in Supabase"""
    try:
        # Ensure tables exist
        safe_workspace = ensure_workspace_tables(workspace)
        
        text = ""
        
        # Read document based on type
        if file_path.endswith('.pdf'):
            # Try PyPDF2 first
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page_num, page in enumerate(pdf_reader.pages):
                        try:
                            page_text = page.extract_text()
                            if page_text.strip():
                                text += f"\n\n--- Seite {page_num + 1} ---\n\n{page_text}"
                        except:
                            continue
            except ImportError:
                print("    ⚠️  PyPDF2 nicht installiert, verwende Fallback")
                text = f"Dokument {Path(file_path).name} wurde hochgeladen. PDF-Parsing erfordert PyPDF2."
            except Exception as e:
                print(f"    ⚠️  PDF-Parsing Fehler: {e}")
                text = f"Dokument {Path(file_path).name} wurde hochgeladen. PDF-Inhalt konnte nicht extrahiert werden."
        elif file_path.endswith('.md'):
            # Markdown file - convert to clean text
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    markdown_content = f.read()
                text = convert_markdown_to_text(markdown_content)
                print(f"    ✓ Markdown konvertiert ({len(markdown_content)} → {len(text)} Zeichen)")
            except Exception as e:
                print(f"    ⚠️  Markdown-Lesefehler: {e}")
                # Fallback to plain text reading
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                except:
                    with open(file_path, 'r', encoding='latin-1', errors='ignore') as f:
                        text = f.read()
        elif file_path.endswith('.txt'):
            # Plain text file
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
            except:
                with open(file_path, 'r', encoding='latin-1', errors='ignore') as f:
                    text = f.read()
        else:
            text = f"Dokument {Path(file_path).name} wurde hochgeladen. Dateityp wird noch nicht vollständig unterstützt."
        
        if not text.strip():
            text = f"Inhalt von {Path(file_path).name} - Dokument wurde hochgeladen."
        
        # Chunk text
        chunks = chunk_text(text)
        print(f"  - {len(chunks)} Text-Chunks erstellt")
        
        # Generate embeddings and store in Supabase
        conn = get_db_connection()
        cursor = conn.cursor()
        
        stored_count = 0
        for i, chunk in enumerate(chunks):
            try:
                # Generate embedding
                embedding = get_ollama_embedding(chunk)
                
                # Store in vector table
                chunk_id = f"{file_id}_chunk_{i}"
                # Convert embedding list to string format for pgvector
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
                    chunk[:5000],  # Limit text length
                    embedding_str,  # Vector as string
                    json.dumps({
                        "file_id": file_id,
                        "chunk_index": i,
                        "filename": Path(file_path).name
                    })
                ))
                stored_count += 1
            except Exception as e:
                print(f"    Fehler beim Speichern von Chunk {i}: {e}")
                continue
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"  ✓ {stored_count} Chunks mit Embeddings in Supabase gespeichert")
        return stored_count
        
    except Exception as e:
        print(f"Fehler bei Dokumentenverarbeitung: {e}")
        import traceback
        traceback.print_exc()
        raise


def search_similar_chunks(query: str, workspace: str = "default", limit: int = 5) -> List[dict]:
    """Search for similar chunks using vector similarity"""
    try:
        safe_workspace = ensure_workspace_tables(workspace)
        
        # Generate query embedding
        query_embedding = get_ollama_embedding(query)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Vector similarity search using pgvector
        # Convert embedding list to string format for pgvector
        embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
        
        cursor.execute(f"""
            SELECT id, text, metadata,
                embedding <=> %s::vector AS distance
            FROM lightrag_vector_storage_{safe_workspace}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (embedding_str, embedding_str, limit))
        
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        chunks = []
        for row in results:
            chunks.append({
                "id": row[0],
                "text": row[1],
                "metadata": row[2] if isinstance(row[2], dict) else json.loads(row[2]) if row[2] else {},
                "distance": float(row[3]) if row[3] else 1.0
            })
        
        return chunks
        
    except Exception as e:
        print(f"Vector search error: {e}")
        # Fallback: simple text search
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


class QueryRequest(BaseModel):
    query: str
    llm_provider: str = "ollama"  # "ollama" or "gemini"
    web_search: bool = False      # Allow general knowledge/web info


class QueryResponse(BaseModel):
    response: str


@app.get("/")
async def root():
    return {
        "message": "RAG Chatbot API (Full Processing)",
        "status": "running",
        "version": "1.0.0",
        "llm_provider": "ollama",
        "model": OLLAMA_MODEL
    }


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
            "llm_provider": "ollama",
            "model": OLLAMA_MODEL,
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


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest, x_workspace: Optional[str] = Header(default="default")):
    """Process query with vector search and LLM"""
    try:
        workspace = x_workspace or "default"
        
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Search for similar chunks in the specific workspace
        # Note: If switching LLMs, embeddings might not be compatible if they use different models.
        # For this implementation, we assume embeddings are always generated by Ollama (or compatible)
        # or that we use the selected provider for embeddings too (which would require re-indexing).
        # To keep it simple: Embeddings are always Ollama/Nomic for retrieval. Response generation switches LLM.
        
        similar_chunks = search_similar_chunks(request.query, workspace=workspace, limit=5)
        
        # Build context from similar chunks
        context = ""
        has_context = False
        if similar_chunks:
            context = "\n\nRelevant document excerpts:\n"
            for i, chunk in enumerate(similar_chunks[:3], 1):
                context += f"{i}. {chunk['text'][:300]}...\n"
            has_context = True
        else:
            context = "\n\nNote: No relevant documents found in this workspace."
        
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

User Question: {request.query}

Remember: Answer in {'ENGLISH' if use_english else 'GERMAN'}. Translate if necessary."""
        
        answer = ""
        
        if request.llm_provider == "gemini":
            # For Gemini, try to use system_instruction parameter
            try:
                import google.generativeai as genai
                genai.configure(api_key=GEMINI_API_KEY)
                
                # Combine instructions for system prompt
                system_prompt = f"""You are a helpful learning assistant.
{language_instruction}
{knowledge_instruction}

IMPORTANT TERMINOLOGY RULES:
- PBL always stands for "Project-Based Learning" (not Problem-Based Learning or PjBL).
- NEVER use "PjBL" as an abbreviation. Use "PBL" instead.

ALWAYS FOLLOW THE LANGUAGE INSTRUCTION ABOVE."""

                model = genai.GenerativeModel(
                    GEMINI_MODEL,
                    system_instruction=system_prompt
                )
                
                # User message contains context and query
                user_message = f"""Context from documents:
{context}

User Question: {request.query}

OUTPUT LANGUAGE: {'ENGLISH' if use_english else 'GERMAN'}"""

                response = model.generate_content(user_message)
                answer = response.text
                print(f"✓ Gemini response received ({len(answer)} chars)")
            except Exception as e:
                # Fallback to regular prompt or Ollama
                error_msg = str(e)
                print(f"❌ Gemini API Error: {error_msg[:200]}")
                
                # Check if it's a quota error
                if "429" in error_msg or "quota" in error_msg.lower() or "ResourceExhausted" in error_msg:
                    # Try fallback to Ollama with clear message
                    print("⚠️  Gemini quota exceeded, falling back to Ollama")
                    try:
                        response = requests.post(
                            f"{OLLAMA_URL}/api/generate",
                            json={
                                "model": OLLAMA_MODEL,
                                "prompt": prompt,
                                "stream": False,
                                "options": {
                                    "temperature": 0.7,
                                    "num_predict": 1000
                                }
                            },
                            timeout=120
                        )
                        if response.status_code == 200:
                            answer = response.json().get("response", "No response received")
                            answer = f"[Note: Gemini quota exceeded, using Ollama instead]\n\n{answer}"
                        else:
                            answer = f"Error: Both Gemini (quota exceeded) and Ollama failed. Please try again later."
                    except Exception as ollama_error:
                        answer = f"Error: Gemini quota exceeded and Ollama unavailable. Gemini error: {error_msg[:150]}"
                else:
                    # Other error - try fallback function
                    answer = get_gemini_response(prompt)
                    if "Error" in answer:
                        # If fallback also fails, try Ollama
                        print("⚠️  Gemini failed, trying Ollama fallback")
                        try:
                            response = requests.post(
                                f"{OLLAMA_URL}/api/generate",
                                json={
                                    "model": OLLAMA_MODEL,
                                    "prompt": prompt,
                                    "stream": False,
                                    "options": {
                                        "temperature": 0.7,
                                        "num_predict": 1000
                                    }
                                },
                                timeout=120
                            )
                            if response.status_code == 200:
                                answer = response.json().get("response", "No response received")
                                answer = f"[Note: Gemini unavailable, using Ollama instead]\n\n{answer}"
                        except:
                            answer = f"Error: Both Gemini and Ollama failed. Gemini error: {error_msg[:150]}"
        else:
            # Default to Ollama
            try:
                response = requests.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "num_predict": 1000
                        }
                    },
                    timeout=120
                )
                if response.status_code == 200:
                    answer = response.json().get("response", "Keine Antwort erhalten")
                else:
                    raise Exception(f"Ollama API Fehler: {response.status_code}")
            except Exception as e:
                answer = f"Sorry, problem with Ollama LLM: {str(e)[:200]}"
        
        return QueryResponse(response=answer)
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Query error: {e}")
        import traceback
        traceback.print_exc()
        return QueryResponse(
            response=f"Entschuldigung, es ist ein Fehler aufgetreten: {str(e)[:200]}"
        )


@app.options("/api/upload")
async def options_upload():
    return {"message": "OK"}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), x_workspace: Optional[str] = Header(default="default")):
    """Upload and process document with embedding generation"""
    file_path = None
    try:
        print(f"DEBUG: Raw x_workspace header value: '{x_workspace}'")
        workspace = x_workspace if x_workspace and x_workspace.strip() != "" else "default"
        print(f"DEBUG: Uploading to workspace: '{workspace}'")
        
        safe_workspace = ensure_workspace_tables(workspace)
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="Kein Dateiname angegeben")
        
        # Save uploaded file
        upload_dir = Path("./uploads")
        upload_dir.mkdir(exist_ok=True)
        
        # Sanitize filename
        safe_filename = "".join(c for c in file.filename if c.isalnum() or c in "._- ")
        file_path = upload_dir / safe_filename
        
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        file_id = f"file_{safe_filename}_{int(Path(file_path).stat().st_mtime)}"
        
        print(f"\n📄 Verarbeite Dokument: {safe_filename}")
        print("  - Extrahiere Text...")
        
        # Process document and generate embeddings
        try:
            chunk_count = process_document_to_vectors(str(file_path), file_id, workspace=safe_workspace)
            print(f"  ✓ {chunk_count} Chunks verarbeitet (Workspace: {safe_workspace})")
        except Exception as proc_error:
            print(f"  ⚠️  Verarbeitungsfehler: {proc_error}")
            chunk_count = 0
        
        # Store file metadata
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            metadata = {
                "filename": safe_filename,
                "size": len(content),
                "status": "processed" if chunk_count > 0 else "error",
                "chunks": chunk_count,
                "workspace": safe_workspace
            }
            cursor.execute(f"""
                INSERT INTO lightrag_kv_storage_{safe_workspace} (key, value)
                VALUES (%s, %s)
                ON CONFLICT (key) DO UPDATE SET value = %s, updated_at = NOW()
            """, (
                file_id,
                json.dumps(metadata),
                json.dumps(metadata)
            ))
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as db_error:
            print(f"  ⚠️  DB-Fehler (Metadata): {db_error}")
        
        # Clean up
        if file_path and file_path.exists():
            file_path.unlink()
        
        if chunk_count == 0:
            return {
                "status": "warning",
                "message": f"Dokument '{safe_filename}' wurde hochgeladen, aber keine Chunks konnten verarbeitet werden.",
                "filename": safe_filename,
                "chunks": 0
            }
        
        return {
            "status": "success",
            "message": f"Dokument '{safe_filename}' wurde vollständig verarbeitet. {chunk_count} Chunks mit Embeddings wurden in Supabase gespeichert.",
            "filename": safe_filename,
            "chunks": chunk_count
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Upload error: {e}")
        import traceback
        traceback.print_exc()
        # Clean up on error
        if file_path and file_path.exists():
            try:
                file_path.unlink()
            except:
                pass
        raise HTTPException(
            status_code=500, 
            detail=f"Fehler beim Verarbeiten: {str(e)[:200]}"
        )


@app.get("/api/documents")
async def get_documents(x_workspace: Optional[str] = Header(default="default")):
    """Get list of processed documents"""
    try:
        workspace = x_workspace or "default"
        safe_workspace = ensure_workspace_tables(workspace)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(f"""
            SELECT key, value, created_at
            FROM lightrag_kv_storage_{safe_workspace}
            WHERE key LIKE 'file_%'
            ORDER BY created_at DESC
        """)
        
        results = cursor.fetchall()
        documents = []
        
        for key, value, created_at in results:
            try:
                doc_data = json.loads(value) if isinstance(value, str) else value
                documents.append({
                    "id": key,
                    "filename": doc_data.get("filename", "unknown"),
                    "status": doc_data.get("status", "unknown"),
                    "chunks": doc_data.get("chunks", 0),
                    "uploaded_at": created_at.isoformat() if created_at else None
                })
            except:
                pass
        
        cursor.close()
        conn.close()
        
        return {"documents": documents}
    
    except Exception as e:
        return {"documents": [], "error": str(e)}


if __name__ == "__main__":
    print(f"\nServer: http://localhost:8000")
    print(f"LLM: Ollama ({OLLAMA_MODEL})")
    print(f"Embeddings: Ollama (nomic-embed-text oder Fallback)")
    print("=" * 70)
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

