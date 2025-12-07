#!/usr/bin/env python
"""
FastAPI Backend Server für RAG Chatbot Web Interface

Dieser Server verbindet die React-Weboberfläche mit RAG-Anything.
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

# Import RAG-Anything
from raganything import RAGAnything, RAGAnythingConfig, setup_supabase_storage
from raganything.gemini_integration import (
    gemini_llm_func,
    gemini_embed_func,
    get_gemini_embedding_dim,
)
from lightrag.utils import EmbeddingFunc

app = FastAPI(title="RAG Chatbot API", version="1.0.0")

# CORS middleware für React-App
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG instance
rag_instance: Optional[RAGAnything] = None


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    response: str


async def get_rag_instance():
    """Initialize or return existing RAG instance"""
    global rag_instance
    
    if rag_instance is not None:
        return rag_instance
    
    print("Initializing RAG-Anything...")
    
    # Get configuration from environment
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    SUPABASE_ACCESS_TOKEN = os.getenv("SUPABASE_ACCESS_TOKEN")
    SUPABASE_DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    if not all([SUPABASE_KEY, SUPABASE_DB_PASSWORD, GEMINI_API_KEY]):
        raise ValueError("Missing required environment variables")
    
    # Setup Supabase storage
    lightrag_kwargs = setup_supabase_storage(
        supabase_key=SUPABASE_KEY,
        supabase_access_token=SUPABASE_ACCESS_TOKEN,
        supabase_db_password=SUPABASE_DB_PASSWORD,
        workspace="default",
    )
    
    # Setup Gemini LLM
    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return gemini_llm_func(
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            model="gemini-2.0-flash-exp",
            api_key=GEMINI_API_KEY,
            **kwargs,
        )
    
    # Setup Gemini embeddings
    embedding_dim = get_gemini_embedding_dim()
    embedding_func = EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=8192,
        func=lambda texts: gemini_embed_func(
            texts=texts,
            model="models/text-embedding-004",
            api_key=GEMINI_API_KEY,
        ),
    )
    
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
    
    print("✓ RAG-Anything initialized")
    return rag_instance


@app.get("/")
async def root():
    return {
        "message": "RAG Chatbot API",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/api/health")
async def health():
    return {"status": "healthy"}


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process a query and return response"""
    try:
        rag = await get_rag_instance()
        
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Process query
        response = await rag.aquery(request.query, mode="hybrid")
        
        return QueryResponse(response=response)
    
    except Exception as e:
        print(f"Error processing query: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a document"""
    try:
        rag = await get_rag_instance()
        
        # Save uploaded file temporarily
        upload_dir = Path("./uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / file.filename
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process document
        output_dir = Path("./output")
        output_dir.mkdir(exist_ok=True)
        
        await rag.process_document_complete(
            file_path=str(file_path),
            output_dir=str(output_dir),
            parse_method="auto"
        )
        
        # Clean up uploaded file
        file_path.unlink()
        
        return {
            "status": "success",
            "message": f"Document '{file.filename}' processed successfully",
            "filename": file.filename
        }
    
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/api/documents")
async def get_documents():
    """Get list of processed documents"""
    # TODO: Implement document listing from Supabase
    return {"documents": []}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

