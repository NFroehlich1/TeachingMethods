#!/usr/bin/env python
"""
Vereinfachter FastAPI Server - Direkt mit Supabase & Gemini

Dieser Server verwendet Supabase PostgreSQL direkt ohne RAG-Anything Abhängigkeiten.
"""

import os
import sys
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import psycopg2
from urllib.parse import urlparse
import json
import base64

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(dotenv_path=".env", override=False)

# Gemini API
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  google-generativeai nicht installiert. Installieren Sie es mit: pip install google-generativeai")

app = FastAPI(title="RAG Chatbot API", version="1.0.0")

# CORS middleware - Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://ngbhnjvojqqesacnijwk.supabase.co")
SUPABASE_DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD", "Test_1082?!")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# LLM Provider selection (can be "gemini", "openai", "lmstudio", or "ollama")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # Default to Ollama (local OSS 20B)

# Local model configuration - OSS 20B
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
# OSS 20B Model - kann verschiedene Namen haben
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "openai/gpt-oss-20b")  # Standard Name
# Alternative Namen für OSS 20B
OSS_20B_MODEL_NAMES = [
    "openai/gpt-oss-20b",
    "gpt-oss-20b",
    "oss-20b",
    "gpt-oss-20b-instruct",
    "openai/gpt-oss-20b-instruct"
]

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")  # OSS 20B Model

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
WORKSPACE = "default"

# Initialize Gemini
if GEMINI_AVAILABLE and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-2.0-flash-exp")
    print("✓ Gemini konfiguriert")
else:
    gemini_model = None
    print("⚠️  Gemini nicht verfügbar")

# Initialize OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
    if OPENAI_API_KEY:
        # Use new OpenAI client initialization
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        print("✓ OpenAI konfiguriert")
    else:
        OPENAI_AVAILABLE = False
        openai_client = None
        print("⚠️  OpenAI API Key nicht gesetzt")
except ImportError:
    OPENAI_AVAILABLE = False
    openai_client = None
    print("⚠️  openai nicht installiert. Installieren Sie es mit: pip install openai")

# Initialize LM Studio (local model - OSS 20B)
LM_STUDIO_AVAILABLE = False
lm_studio_client = None
detected_model = None

try:
    import openai as openai_lib
    import requests
    
    # Test connection
    test_response = requests.get(f"{LM_STUDIO_URL}/models", timeout=2)
    if test_response.status_code == 200:
        models_data = test_response.json()
        available_models = [m.get("id", "") for m in models_data.get("data", [])]
        
        # Find OSS 20B model
        for model_name in OSS_20B_MODEL_NAMES:
            if any(model_name.lower() in m.lower() for m in available_models):
                detected_model = next((m for m in available_models if model_name.lower() in m.lower()), None)
                if detected_model:
                    LM_STUDIO_MODEL = detected_model
                    break
        
        if not detected_model and available_models:
            # Use first available model if OSS 20B not found
            detected_model = available_models[0]
            LM_STUDIO_MODEL = detected_model
            print(f"⚠️  OSS 20B nicht gefunden, verwende: {detected_model}")
        
        lm_studio_client = openai_lib.OpenAI(
            base_url=LM_STUDIO_URL,
            api_key=LM_STUDIO_API_KEY
        )
        LM_STUDIO_AVAILABLE = True
        print(f"✓ LM Studio konfiguriert: {LM_STUDIO_URL}")
        print(f"  Verfügbare Modelle: {len(available_models)}")
        print(f"  Verwendetes Modell: {LM_STUDIO_MODEL}")
        
except requests.exceptions.ConnectionError:
    print(f"⚠️  LM Studio Server nicht erreichbar: {LM_STUDIO_URL}")
    print("   Bitte starten Sie LM Studio und aktivieren Sie den lokalen Server")
except Exception as e:
    print(f"⚠️  LM Studio Fehler: {e}")
    print("   Stellen Sie sicher, dass LM Studio läuft und der Server aktiviert ist")

# Initialize Ollama (alternative local model)
OLLAMA_AVAILABLE = False
try:
    import requests
    # Test Ollama connection
    response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
    if response.status_code == 200:
        OLLAMA_AVAILABLE = True
        print(f"✓ Ollama konfiguriert: {OLLAMA_URL}")
        print(f"  Modell: {OLLAMA_MODEL}")
    else:
        print(f"⚠️  Ollama nicht erreichbar (Status: {response.status_code})")
except Exception as e:
    print(f"⚠️  Ollama nicht verfügbar: {e}")
    print("   Installieren Sie Ollama von https://ollama.ai")


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
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        raise


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    response: str


@app.get("/")
async def root():
    return {
        "message": "RAG Chatbot API (Simplified)",
        "status": "running",
        "version": "1.0.0",
        "llm_provider": LLM_PROVIDER,
        "gemini_available": GEMINI_AVAILABLE and gemini_model is not None,
        "openai_available": OPENAI_AVAILABLE and bool(OPENAI_API_KEY)
    }


@app.get("/api/health")
async def health():
    try:
        conn = get_db_connection()
        conn.close()
        return {
            "status": "healthy",
            "database": "connected",
            "llm_provider": LLM_PROVIDER,
            "gemini": "available" if gemini_model else "unavailable",
            "openai": "available" if (OPENAI_AVAILABLE and openai_client) else "unavailable",
            "lm_studio": "available" if LM_STUDIO_AVAILABLE else "unavailable",
            "ollama": "available" if OLLAMA_AVAILABLE else "unavailable"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.options("/api/query")
async def options_query():
    """Handle OPTIONS request for CORS"""
    return {"message": "OK"}

@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process a query using selected LLM provider"""
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Check if provider is available
        if LLM_PROVIDER == "openai":
            if not OPENAI_AVAILABLE or not openai_client:
                raise HTTPException(
                    status_code=503,
                    detail="OpenAI API nicht verfügbar. Bitte installieren Sie openai und setzen Sie OPENAI_API_KEY."
                )
        elif LLM_PROVIDER == "gemini":
            if not gemini_model:
                raise HTTPException(
                    status_code=503,
                    detail="Gemini API nicht verfügbar. Bitte installieren Sie google-generativeai."
                )
        elif LLM_PROVIDER == "lmstudio":
            if not LM_STUDIO_AVAILABLE or not lm_studio_client:
                raise HTTPException(
                    status_code=503,
                    detail="LM Studio nicht verfügbar. Bitte starten Sie LM Studio und aktivieren Sie den lokalen Server."
                )
        elif LLM_PROVIDER == "ollama":
            if not OLLAMA_AVAILABLE:
                raise HTTPException(
                    status_code=503,
                    detail="Ollama nicht verfügbar. Bitte installieren Sie Ollama und starten Sie den Service."
                )
        # Get context from Supabase if available
        context = ""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Search for relevant documents/chunks
            # This is a simplified version - in production you'd use vector search
            cursor.execute(f"""
                SELECT text 
                FROM lightrag_vector_storage_{WORKSPACE}
                WHERE text ILIKE %s
                LIMIT 5
            """, (f"%{request.query[:50]}%",))
            
            results = cursor.fetchall()
            if results:
                context = "\n\nRelevante Dokumenten-Ausschnitte:\n" + "\n".join([r[0][:200] for r in results])
            
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"Database search error (ignoring): {e}")
            # Continue without context
        
        # Build prompt
        prompt = f"""Du bist ein hilfreicher Assistent für Project Based Learning, der Fragen zu Dokumenten beantwortet.

{context if context else "Aktuell sind keine Dokumente hochgeladen. Bitte laden Sie zuerst ein Dokument hoch."}

Frage des Benutzers: {request.query}

Antworte hilfreich und präzise auf Deutsch. Fokussiere dich auf Lerninhalte und erkläre Konzepte verständlich."""
        
        # Get response from selected LLM provider
        answer = ""
        try:
            if LLM_PROVIDER == "openai":
                # Use OpenAI
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Du bist ein hilfreicher Lernassistent für Project Based Learning. Antworte immer auf Deutsch."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                answer = response.choices[0].message.content
                
            elif LLM_PROVIDER == "gemini":
                # Use Gemini
                response = gemini_model.generate_content(prompt)
                answer = response.text if hasattr(response, 'text') else str(response)
                
            elif LLM_PROVIDER == "lmstudio":
                # Use LM Studio (local model)
                response = lm_studio_client.chat.completions.create(
                    model=LM_STUDIO_MODEL,
                    messages=[
                        {"role": "system", "content": "Du bist ein hilfreicher Lernassistent für Project Based Learning. Antworte immer auf Deutsch."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                answer = response.choices[0].message.content
                
            elif LLM_PROVIDER == "ollama":
                # Use Ollama (local model)
                import requests
                ollama_response = requests.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": OLLAMA_MODEL,
                        "prompt": f"Du bist ein hilfreicher Lernassistent für Project Based Learning. Antworte immer auf Deutsch.\n\n{prompt}",
                        "stream": False
                    },
                    timeout=60
                )
                if ollama_response.status_code == 200:
                    answer = ollama_response.json().get("response", "Keine Antwort erhalten")
                else:
                    raise Exception(f"Ollama API Fehler: {ollama_response.status_code}")
            else:
                answer = f"Unbekannter LLM Provider: {LLM_PROVIDER}"
                
        except Exception as llm_error:
            error_str = str(llm_error)
            print(f"{LLM_PROVIDER.upper()} API error: {error_str}")
            
            # Check for quota/rate limit errors
            if "quota" in error_str.lower() or "429" in error_str or "rate limit" in error_str.lower():
                answer = "⚠️ Die API-Quota wurde erreicht. Bitte warten Sie einen Moment (~30 Sekunden) und versuchen Sie es erneut."
            elif "401" in error_str or "unauthorized" in error_str.lower():
                answer = "⚠️ API-Schlüssel ungültig. Bitte überprüfen Sie die Konfiguration."
            else:
                answer = f"Entschuldigung, es gab ein Problem mit der {LLM_PROVIDER.upper()} API. Bitte versuchen Sie es später erneut."
        
        return QueryResponse(response=answer)
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing query: {e}")
        import traceback
        traceback.print_exc()
        return QueryResponse(response=f"Entschuldigung, es ist ein Fehler aufgetreten: {str(e)}")


@app.options("/api/upload")
async def options_upload():
    """Handle OPTIONS request for CORS"""
    return {"message": "OK"}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a document - simplified version that just stores metadata"""
    try:
        # Save file info to database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Store file metadata in KV storage
        file_id = f"file_{file.filename}_{int(os.path.getmtime('/tmp') if os.path.exists('/tmp') else 0)}"
        
        cursor.execute(f"""
            INSERT INTO lightrag_kv_storage_{WORKSPACE} (key, value)
            VALUES (%s, %s)
            ON CONFLICT (key) DO UPDATE SET value = %s, updated_at = NOW()
        """, (
            file_id,
            json.dumps({
                "filename": file.filename,
                "content_type": file.content_type,
                "size": file.size if hasattr(file, 'size') else 0,
                "status": "uploaded"
            }),
            json.dumps({
                "filename": file.filename,
                "content_type": file.content_type,
                "size": file.size if hasattr(file, 'size') else 0,
                "status": "uploaded"
            })
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        # Save file temporarily (in production, you'd process it with RAG-Anything)
        upload_dir = Path("./uploads")
        upload_dir.mkdir(exist_ok=True)
        file_path = upload_dir / file.filename
        
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        return {
            "status": "success",
            "message": f"Datei '{file.filename}' wurde hochgeladen. Hinweis: Für vollständige Verarbeitung verwenden Sie RAG-Anything.",
            "filename": file.filename,
            "file_id": file_id
        }
    
    except Exception as e:
        print(f"Error uploading file: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Fehler beim Hochladen: {str(e)}")


@app.get("/api/documents")
async def get_documents():
    """Get list of uploaded documents"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(f"""
            SELECT key, value, created_at
            FROM lightrag_kv_storage_{WORKSPACE}
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
                    "uploaded_at": created_at.isoformat() if created_at else None
                })
            except:
                pass
        
        cursor.close()
        conn.close()
        
        return {"documents": documents}
    
    except Exception as e:
        print(f"Error getting documents: {e}")
        return {"documents": []}


if __name__ == "__main__":
    print("=" * 70)
    print("RAG Chatbot API Server (Simplified)")
    print("=" * 70)
    print(f"Supabase: {SUPABASE_URL}")
    print(f"Gemini: {'✓ Verfügbar' if gemini_model else '✗ Nicht verfügbar'}")
    print(f"Server: http://localhost:8000")
    print("=" * 70)
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

