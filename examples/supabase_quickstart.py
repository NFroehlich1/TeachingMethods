#!/usr/bin/env python
"""
Quick Start Example: RAG-Anything with Supabase

This example uses the pre-configured Supabase credentials to quickly
set up and use RAG-Anything with Supabase PostgreSQL storage.

Prerequisites:
1. Install dependencies: pip install raganything[supabase]
2. Set SUPABASE_DB_PASSWORD environment variable or provide it below
3. Ensure pgvector extension is enabled in Supabase
"""

import os
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, logger
from raganything import (
    RAGAnything,
    RAGAnythingConfig,
    setup_supabase_storage,
    get_supabase_url_from_jwt,
    create_pgvector_extension,
    verify_supabase_connection,
)
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)

# Pre-configured Supabase credentials
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5nYmhuanZvanFxZXNhY25pandrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjUwMDEyODcsImV4cCI6MjA4MDU3NzI4N30.-B_hvypDk3NLOYmYJ5qrJdojUtSFi7HSKK2aV2COsuU"
SUPABASE_ACCESS_TOKEN = "sbp_126aa9500ca99a588c1a0c748aa2fdda4a4cf391"

# Extract URL from JWT
SUPABASE_URL = get_supabase_url_from_jwt(SUPABASE_ANON_KEY) or os.getenv("SUPABASE_URL")

if not SUPABASE_URL:
    print("❌ Error: Could not determine Supabase URL")
    sys.exit(1)

print(f"✓ Using Supabase project: {SUPABASE_URL}")


async def main():
    """Main function demonstrating Supabase integration"""
    
    # Get database password
    db_password = os.getenv("SUPABASE_DB_PASSWORD")
    if not db_password:
        print("\n⚠️  SUPABASE_DB_PASSWORD not set in environment")
        print("   Get it from: Supabase Dashboard → Settings → Database → Database Password")
        db_password = input("\nEnter your Supabase database password: ").strip()
        if not db_password:
            print("❌ Database password is required")
            return
    
    # Verify connection
    print("\nVerifying Supabase connection...")
    if not verify_supabase_connection(
        supabase_url=SUPABASE_URL,
        supabase_db_password=db_password
    ):
        print("❌ Connection failed. Please check your database password.")
        return
    print("✓ Connection verified")
    
    # Ensure pgvector extension exists
    print("\nChecking pgvector extension...")
    if not create_pgvector_extension(
        supabase_url=SUPABASE_URL,
        supabase_db_password=db_password
    ):
        print("⚠️  Could not create pgvector extension automatically")
        print("   Please run in Supabase SQL Editor: CREATE EXTENSION IF NOT EXISTS vector;")
        return
    print("✓ pgvector extension ready")
    
    # Setup Supabase storage using JWT token (auto-extracts URL)
    print("\nConfiguring Supabase storage...")
    lightrag_kwargs = setup_supabase_storage(
        supabase_key=SUPABASE_ANON_KEY,
        supabase_access_token=SUPABASE_ACCESS_TOKEN,
        supabase_db_password=db_password,
        workspace="default"
    )
    print("✓ Storage configured")
    
    # Get API key for LLM
    api_key = os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n⚠️  LLM API key not found")
        print("   Set OPENAI_API_KEY or LLM_BINDING_API_KEY environment variable")
        api_key = input("Enter your OpenAI API key (or press Enter to skip LLM setup): ").strip()
        if not api_key:
            print("⚠️  Skipping LLM setup. You can still configure storage.")
            return
    
    # Define LLM function
    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
        )
    
    # Define embedding function
    embedding_func = EmbeddingFunc(
        embedding_dim=3072,
        max_token_size=8192,
        func=lambda texts: openai_embed(
            texts,
            model="text-embedding-3-large",
            api_key=api_key,
        ),
    )
    
    # Initialize RAG-Anything
    print("\nInitializing RAG-Anything with Supabase...")
    config = RAGAnythingConfig(
        working_dir="./rag_storage",
        parser="mineru",
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )
    
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        lightrag_kwargs=lightrag_kwargs,
    )
    
    print("✓ RAG-Anything initialized with Supabase storage")
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print("\nYou can now:")
    print("1. Process documents: await rag.process_document_complete('document.pdf')")
    print("2. Query documents: await rag.aquery('Your question')")
    print("\nAll data will be stored in Supabase PostgreSQL!")
    
    # Example: Process a document if provided
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if os.path.exists(file_path):
            print(f"\nProcessing document: {file_path}")
            await rag.process_document_complete(
                file_path=file_path,
                output_dir="./output"
            )
            print("✓ Document processed and stored in Supabase")
            
            # Example query
            print("\nExample query:")
            result = await rag.aquery("What is this document about?", mode="hybrid")
            print(f"Answer: {result}")
        else:
            print(f"\n⚠️  File not found: {file_path}")


if __name__ == "__main__":
    print("RAG-Anything Supabase Quick Start")
    print("=" * 60)
    asyncio.run(main())


