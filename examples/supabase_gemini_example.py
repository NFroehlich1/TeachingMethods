#!/usr/bin/env python
"""
Supabase + Gemini Integration Example

This example shows how to use RAG-Anything with Supabase storage
and Google Gemini Flash 2.0 as the LLM.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lightrag.utils import EmbeddingFunc, logger
from raganything import (
    RAGAnything,
    RAGAnythingConfig,
    setup_supabase_storage,
    get_supabase_url_from_jwt,
)
from raganything.gemini_integration import (
    gemini_llm_func,
    gemini_embed_func,
    get_gemini_embedding_dim,
)
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)

# Your Supabase credentials
SUPABASE_ANON_KEY = os.getenv(
    "SUPABASE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5nYmhuanZvanFxZXNhY25pandrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjUwMDEyODcsImV4cCI6MjA4MDU3NzI4N30.-B_hvypDk3NLOYmYJ5qrJdojUtSFi7HSKK2aV2COsuU",
)
SUPABASE_ACCESS_TOKEN = os.getenv("SUPABASE_ACCESS_TOKEN", "sbp_126aa9500ca99a588c1a0c748aa2fdda4a4cf391")

# Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBuTKegElwBTBfxdtbRwF47sYsa5vtF9_U")

# Extract URL from JWT
SUPABASE_URL = get_supabase_url_from_jwt(SUPABASE_ANON_KEY) or os.getenv("SUPABASE_URL")

if not SUPABASE_URL:
    print("❌ Error: Could not determine Supabase URL")
    sys.exit(1)

print("=" * 70)
print("RAG-Anything with Supabase + Gemini Flash 2.0")
print("=" * 70)
print(f"Supabase: {SUPABASE_URL}")
print(f"LLM: Gemini Flash 2.0")
print()


async def main():
    """Main function demonstrating Supabase + Gemini integration"""

    # Get database password
    db_password = os.getenv("SUPABASE_DB_PASSWORD")
    if not db_password:
        print("⚠️  SUPABASE_DB_PASSWORD not set in environment")
        db_password = input("Enter your Supabase database password: ").strip()
        if not db_password:
            print("❌ Database password is required")
            return

    # Setup Supabase storage
    print("Configuring Supabase storage...")
    lightrag_kwargs = setup_supabase_storage(
        supabase_key=SUPABASE_ANON_KEY,
        supabase_access_token=SUPABASE_ACCESS_TOKEN,
        supabase_db_password=db_password,
        workspace="default",
    )
    print("✓ Supabase storage configured")

    # Setup Gemini LLM function
    print("Configuring Gemini Flash 2.0...")
    
    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return gemini_llm_func(
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            model="gemini-2.0-flash-exp",
            api_key=GEMINI_API_KEY,
            **kwargs,
        )

    # Setup Gemini embedding function
    embedding_dim = get_gemini_embedding_dim()
    print(f"Using Gemini embeddings (dimension: {embedding_dim})")

    embedding_func = EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=8192,
        func=lambda texts: gemini_embed_func(
            texts=texts,
            model="models/text-embedding-004",
            api_key=GEMINI_API_KEY,
        ),
    )
    print("✓ Gemini configured")

    # Initialize RAG-Anything
    print("\nInitializing RAG-Anything...")
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

    print("✓ RAG-Anything initialized with Supabase + Gemini")
    print("\n" + "=" * 70)
    print("Ready to use!")
    print("=" * 70)

    # Process document if provided
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if os.path.exists(file_path):
            print(f"\nProcessing document: {file_path}")
            await rag.process_document_complete(file_path=file_path, output_dir="./output")
            print("✓ Document processed and stored in Supabase")

            # Example query
            print("\nExample query:")
            query = "What is this document about?"
            print(f"Query: {query}")
            result = await rag.aquery(query, mode="hybrid")
            print(f"Answer: {result}")
        else:
            print(f"\n⚠️  File not found: {file_path}")
    else:
        print("\nUsage:")
        print("  python examples/supabase_gemini_example.py document.pdf")
        print("\nOr use in your code:")
        print("  await rag.process_document_complete('document.pdf')")
        print("  result = await rag.aquery('Your question')")


if __name__ == "__main__":
    asyncio.run(main())

