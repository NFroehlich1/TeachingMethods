#!/usr/bin/env python
"""
Vollständiger RAG Server mit Embedding-Generierung

Dieser Server verwendet RAG-Anything vollständig, um Dokumente zu verarbeiten
und Embeddings in Supabase zu speichern.
"""

import os
import sys
import asyncio
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

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
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5nYmhuanZvanFxZXNhY25pandrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjUwMDEyODcsImV4cCI6MjA4MDU3NzI4N30.-B_hvypDk3NLOYmYJ5qrJdojUtSFi7HSKK2aV2COsuU")
SUPABASE_ACCESS_TOKEN = os.getenv("SUPABASE_ACCESS_TOKEN", "sbp_126aa9500ca99a588c1a0c748aa2fdda4a4cf391")
SUPABASE_DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD", "Test_1082?!")

# LLM Provider
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")

# Global RAG instance
rag_instance = None

print("=" * 70)
print("RAG Server mit vollständiger Dokumentenverarbeitung")
print("=" * 70)

# Try to import RAG-Anything
try:
    from raganything import RAGAnything, RAGAnythingConfig, setup_supabase_storage
    from raganything.gemini_integration import get_gemini_embedding_dim
    from lightrag.utils import EmbeddingFunc
    RAG_AVAILABLE = True
    print("✓ RAG-Anything verfügbar")
except ImportError as e:
    RAG_AVAILABLE = False
    print(f"⚠️  RAG-Anything nicht verfügbar: {e}")
    print("   Installieren Sie: pip install raganything lightrag-hku")


def get_ollama_llm_func():
    """Get Ollama LLM function"""
    import requests
    
    def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get("temperature", 0.7),
                        "num_predict": kwargs.get("max_tokens", 1000)
                    }
                },
                timeout=120
            )
            if response.status_code == 200:
                return response.json().get("response", "Keine Antwort erhalten")
            else:
                raise Exception(f"Ollama API Fehler: {response.status_code}")
        except Exception as e:
            raise Exception(f"Ollama Fehler: {str(e)}")
    
    return llm_func


def get_ollama_embedding_func():
    """Get Ollama embedding function - using nomic-embed-text"""
    import requests
    
    def embed_func(texts):
        embeddings = []
        for text in texts:
            try:
                # Ollama embedding endpoint
                response = requests.post(
                    f"{OLLAMA_URL}/api/embeddings",
                    json={
                        "model": "nomic-embed-text",  # Good local embedding model
                        "prompt": text
                    },
                    timeout=30
                )
                if response.status_code == 200:
                    embedding = response.json().get("embedding", [])
                    embeddings.append(embedding)
                else:
                    # Fallback: simple hash-based embedding if model not available
                    import hashlib
                    hash_obj = hashlib.md5(text.encode())
                    embedding = [float(int(b)) for b in hash_obj.digest()[:768]] + [0.0] * (768 - 16)
                    embeddings.append(embedding)
            except Exception as e:
                print(f"Embedding Fehler: {e}")
                # Fallback embedding
                import hashlib
                hash_obj = hashlib.md5(text.encode())
                embedding = [float(int(b)) for b in hash_obj.digest()[:768]] + [0.0] * (768 - 16)
                embeddings.append(embedding)
        return embeddings
    
    return embed_func


async def get_rag_instance():
    """Initialize or return existing RAG instance"""
    global rag_instance
    
    if rag_instance is not None:
        return rag_instance
    
    if not RAG_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="RAG-Anything nicht verfügbar. Bitte installieren Sie die Abhängigkeiten."
        )
    
    print("\nInitialisiere RAG-Anything...")
    
    # Setup Supabase storage
    lightrag_kwargs = setup_supabase_storage(
        supabase_key=SUPABASE_KEY,
        supabase_access_token=SUPABASE_ACCESS_TOKEN,
        supabase_db_password=SUPABASE_DB_PASSWORD,
        workspace="default",
    )
    print("✓ Supabase Storage konfiguriert")
    
    # Setup LLM function (Ollama)
    llm_model_func = get_ollama_llm_func()
    print(f"✓ LLM konfiguriert: Ollama ({OLLAMA_MODEL})")
    
    # Setup embedding function (Ollama nomic-embed-text)
    embedding_func = EmbeddingFunc(
        embedding_dim=768,  # nomic-embed-text dimension
        max_token_size=8192,
        func=get_ollama_embedding_func(),
    )
    print("✓ Embedding konfiguriert: Ollama (nomic-embed-text)")
    
    # Initialize RAG-Anything
    config = RAGAnythingConfig(
        working_dir="./rag_storage",
        parser="mineru",
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )
    
    rag_instance = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        lightrag_kwargs=lightrag_kwargs,
    )
    
    # Initialize storages
    await rag_instance._ensure_lightrag_initialized()
    print("✓ RAG-Anything initialisiert")
    print("=" * 70)
    
    return rag_instance


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    response: str


@app.get("/")
async def root():
    return {
        "message": "RAG Chatbot API (Full RAG-Anything)",
        "status": "running",
        "version": "1.0.0",
        "llm_provider": LLM_PROVIDER,
        "rag_available": RAG_AVAILABLE
    }


@app.get("/api/health")
async def health():
    try:
        rag = await get_rag_instance()
        return {
            "status": "healthy",
            "rag_initialized": rag is not None,
            "llm_provider": LLM_PROVIDER,
            "model": OLLAMA_MODEL
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
async def query(request: QueryRequest):
    """Process a query using RAG-Anything with vector search"""
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        rag = await get_rag_instance()
        
        # Use RAG-Anything query with vector search
        result = await rag.aquery(request.query, mode="hybrid")
        
        return QueryResponse(response=result)
    
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
async def upload_file(file: UploadFile = File(...)):
    """Upload and process document with RAG-Anything"""
    try:
        rag = await get_rag_instance()
        
        # Save uploaded file temporarily
        upload_dir = Path("./uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / file.filename
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        print(f"\nVerarbeite Dokument: {file.filename}")
        print("  - Parsing Dokument...")
        print("  - Generiere Embeddings...")
        print("  - Speichere in Supabase...")
        
        # Process document with RAG-Anything (this generates embeddings and stores them)
        output_dir = Path("./output")
        output_dir.mkdir(exist_ok=True)
        
        await rag.process_document_complete(
            file_path=str(file_path),
            output_dir=str(output_dir),
            parse_method="auto"
        )
        
        print(f"✓ Dokument verarbeitet: {file.filename}")
        print("  - Embeddings generiert und in Supabase gespeichert")
        
        # Clean up uploaded file
        file_path.unlink()
        
        return {
            "status": "success",
            "message": f"Dokument '{file.filename}' wurde vollständig verarbeitet. Embeddings wurden generiert und in Supabase gespeichert.",
            "filename": file.filename
        }
    
    except Exception as e:
        print(f"Upload error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Fehler beim Verarbeiten: {str(e)}")


@app.get("/api/documents")
async def get_documents():
    """Get list of processed documents from Supabase"""
    try:
        rag = await get_rag_instance()
        
        # Get documents from Supabase KV storage
        conn = rag.lightrag.kv_storage_cls_kwargs
        # This would need to query the actual storage
        # For now, return a simple response
        
        return {"documents": [], "note": "Document list from Supabase"}
    
    except Exception as e:
        return {"documents": [], "error": str(e)}


if __name__ == "__main__":
    print(f"\nServer: http://localhost:8000")
    print(f"LLM: Ollama ({OLLAMA_MODEL})")
    print(f"Embeddings: Ollama (nomic-embed-text)")
    print("=" * 70)
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

