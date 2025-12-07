import os
import time
import json
import logging
import requests
import psycopg2
import uvicorn
import markdown
import re
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime

# Load environment variables
load_dotenv(dotenv_path="../.env")

# --- Configuration ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")
SUPABASE_DB_USER = "postgres"
# Extract host from Supabase URL (e.g. https://xyz.supabase.co -> db.xyz.supabase.co)
project_ref = SUPABASE_URL.split("//")[1].split(".")[0]
SUPABASE_DB_HOST = f"db.{project_ref}.supabase.co"
SUPABASE_DB_PORT = "5432"

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama", "gemini", "openai"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash-exp"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Embedding Configuration
# Using Ollama for embeddings to keep it local/free by default
# "nomic-embed-text" is a good default for Ollama
EMBEDDING_MODEL = "nomic-embed-text" 
EMBEDDING_DIM = 3072  # Dimension for lightrag_vector_storage

# --- Database Connection ---
def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname="postgres",
            user=SUPABASE_DB_USER,
            password=SUPABASE_DB_PASSWORD,
            host=SUPABASE_DB_HOST,
            port=SUPABASE_DB_PORT
        )
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

# --- Helpers ---

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
    
    # Check if table exists
    cursor.execute(f"""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'lightrag_vector_storage_{safe_workspace}'
        );
    """)
    exists = cursor.fetchone()[0]
    
    if not exists:
        print(f"Creating tables for workspace: {safe_workspace}")
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

@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...), 
    x_workspace: Optional[str] = Header(default="default")
):
    try:
        workspace = x_workspace or "default"
        print(f"DEBUG: Uploading to workspace: '{workspace}'")
        safe_workspace = ensure_workspace_tables(workspace)
        
        # Read file content
        content = await file.read()
        file_path = f"uploads/{file.filename}"
        os.makedirs("uploads", exist_ok=True)
        
        with open(file_path, "wb") as f:
            f.write(content)
            
        # Extract text based on file type
        text_content = ""
        if file.filename.endswith(".pdf"):
            import PyPDF2
            reader = PyPDF2.PdfReader(file_path)
            for page in reader.pages:
                text_content += page.extract_text() + "\n"
        elif file.filename.endswith(".md"):
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
                embedding = get_ollama_embedding(chunk)
                
                # Insert into vector storage
                cursor.execute(f"""
                    INSERT INTO lightrag_vector_storage_{safe_workspace} (text, metadata, embedding)
                    VALUES (%s, %s, %s)
                """, (
                    chunk, 
                    json.dumps({
                        "source": file.filename,
                        "chunk_index": i,
                        "filename": Path(file_path).name
                    }),
                    embedding
                ))
                
                # Update document status in KV store
                cursor.execute(f"""
                    INSERT INTO lightrag_kv_store_{safe_workspace} (key, value)
                    VALUES (%s, %s)
                    ON CONFLICT (key) DO UPDATE SET value = %s
                """, (
                    f"doc_status_{file.filename}",
                    json.dumps({"status": "processing", "chunks_total": len(chunks), "chunks_processed": i+1}),
                    json.dumps({"status": "processing", "chunks_total": len(chunks), "chunks_processed": i+1})
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
    llm_provider: str = "ollama"  # "ollama" or "gemini"
    web_search: bool = False      # Allow general knowledge/web info
    history: List[ChatMessage] = []  # Chat history


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
{chat_history_str}

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
                    answer = f"⚠️ Gemini Quota Exceeded. Falling back to local model.\n\n"
                    request.llm_provider = "ollama"
                else:
                    return JSONResponse(
                        status_code=500, 
                        content={"response": f"Gemini Error: {error_msg}. Please try switching to Ollama."}
                    )

        if request.llm_provider == "ollama":
            # Use Ollama
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
            else:
                raise HTTPException(status_code=500, detail=f"Ollama Error: {response.text}")
        
        # Fallback if LM Studio or others (not implemented yet)
        
        return {"response": answer}
        
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
