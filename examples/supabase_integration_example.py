#!/usr/bin/env python
"""
Example script demonstrating Supabase integration with RAG-Anything

This example shows how to:
1. Configure RAG-Anything to use Supabase PostgreSQL for storage
2. Process documents with Supabase backend
3. Perform queries using Supabase-stored vectors

Prerequisites:
1. Create a Supabase project at https://supabase.com
2. Get your project URL and database password from Supabase dashboard
3. Enable pgvector extension in Supabase SQL editor:
   CREATE EXTENSION IF NOT EXISTS vector;
4. Set environment variables or provide them as arguments
"""

import os
import argparse
import asyncio
import logging
from pathlib import Path

# Add project root directory to Python path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, logger
from raganything import (
    RAGAnything,
    RAGAnythingConfig,
    setup_supabase_storage,
    setup_supabase_from_connection_string,
    verify_supabase_connection,
    create_pgvector_extension,
)
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)


async def process_with_supabase(
    file_path: str,
    output_dir: str,
    api_key: str,
    base_url: str = None,
    supabase_url: str = None,
    supabase_db_password: str = None,
    supabase_db_user: str = None,
    connection_string: str = None,
    workspace: str = "default",
):
    """
    Process document with RAG-Anything using Supabase storage

    Args:
        file_path: Path to the document
        output_dir: Output directory for parsed content
        api_key: OpenAI API key
        base_url: Optional base URL for API
        supabase_url: Supabase project URL
        supabase_db_password: Database password
        supabase_db_user: Database user (default: postgres)
        connection_string: Alternative: PostgreSQL connection string
        workspace: Workspace name for separating RAG instances
    """
    try:
        # Verify Supabase connection
        logger.info("Verifying Supabase connection...")
        if connection_string:
            # If using connection string, we can't verify easily without parsing
            logger.info("Using connection string, skipping connection verification")
        else:
            if verify_supabase_connection(
                supabase_url=supabase_url,
                supabase_db_password=supabase_db_password,
                supabase_db_user=supabase_db_user,
            ):
                logger.info("✓ Supabase connection verified")
            else:
                logger.error("✗ Failed to verify Supabase connection")
                logger.error(
                    "Please check your Supabase URL and database password"
                )
                return

        # Setup Supabase storage configuration
        logger.info("Configuring Supabase storage...")
        if connection_string:
            lightrag_kwargs = setup_supabase_from_connection_string(
                connection_string=connection_string, workspace=workspace
            )
        else:
            lightrag_kwargs = setup_supabase_storage(
                supabase_url=supabase_url,
                supabase_db_password=supabase_db_password,
                supabase_db_user=supabase_db_user,
                workspace=workspace,
            )
        logger.info("✓ Supabase storage configured")

        # Ensure pgvector extension exists
        logger.info("Checking pgvector extension...")
        if not connection_string:
            if create_pgvector_extension(
                supabase_url=supabase_url,
                supabase_db_password=supabase_db_password,
                supabase_db_user=supabase_db_user,
            ):
                logger.info("✓ pgvector extension ready")
            else:
                logger.warning(
                    "⚠ Could not create pgvector extension automatically"
                )
                logger.warning(
                    "Please run this SQL in Supabase SQL editor:"
                )
                logger.warning("  CREATE EXTENSION IF NOT EXISTS vector;")

        # Create RAGAnything configuration
        config = RAGAnythingConfig(
            working_dir="./rag_storage",  # Local cache directory
            parser="mineru",
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )

        # Define LLM model function
        def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return openai_complete_if_cache(
                "gpt-4o-mini",
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )

        # Define vision model function
        def vision_model_func(
            prompt,
            system_prompt=None,
            history_messages=[],
            image_data=None,
            messages=None,
            **kwargs,
        ):
            if messages:
                return openai_complete_if_cache(
                    "gpt-4o",
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=messages,
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs,
                )
            elif image_data:
                return openai_complete_if_cache(
                    "gpt-4o",
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=[
                        {"role": "system", "content": system_prompt}
                        if system_prompt
                        else None,
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_data}"
                                    },
                                },
                            ],
                        }
                        if image_data
                        else {"role": "user", "content": prompt},
                    ],
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs,
                )
            else:
                return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

        # Define embedding function
        embedding_dim = int(os.getenv("EMBEDDING_DIM", "3072"))
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

        embedding_func = EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=lambda texts: openai_embed(
                texts, model=embedding_model, api_key=api_key, base_url=base_url
            ),
        )

        # Initialize RAGAnything with Supabase storage
        logger.info("Initializing RAG-Anything with Supabase backend...")
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
            lightrag_kwargs=lightrag_kwargs,  # Supabase storage configuration
        )

        logger.info("✓ RAG-Anything initialized with Supabase storage")

        # Process document
        logger.info(f"Processing document: {file_path}")
        await rag.process_document_complete(
            file_path=file_path, output_dir=output_dir, parse_method="auto"
        )
        logger.info("✓ Document processed and stored in Supabase")

        # Example queries
        logger.info("\n" + "=" * 50)
        logger.info("Querying processed document from Supabase...")
        logger.info("=" * 50)

        # Text query
        query = "What is the main content of the document?"
        logger.info(f"\n[Query]: {query}")
        result = await rag.aquery(query, mode="hybrid")
        logger.info(f"[Answer]: {result}")

        # Another query
        query = "What are the key topics discussed?"
        logger.info(f"\n[Query]: {query}")
        result = await rag.aquery(query, mode="hybrid")
        logger.info(f"[Answer]: {result}")

        logger.info("\n✓ Queries completed successfully")
        logger.info(
            "All data (vectors, embeddings, documents) are stored in Supabase PostgreSQL"
        )

    except Exception as e:
        logger.error(f"Error processing with Supabase: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())


def main():
    """Main function to run the example"""
    parser = argparse.ArgumentParser(
        description="RAG-Anything Supabase Integration Example"
    )
    parser.add_argument(
        "file_path", help="Path to the document to process"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./output",
        help="Output directory for parsed content",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LLM_BINDING_API_KEY"),
        help="OpenAI API key (defaults to LLM_BINDING_API_KEY env var)",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BINDING_HOST"),
        help="Optional base URL for API",
    )
    parser.add_argument(
        "--supabase-url",
        default=os.getenv("SUPABASE_URL"),
        help="Supabase project URL (defaults to SUPABASE_URL env var)",
    )
    parser.add_argument(
        "--supabase-db-password",
        default=os.getenv("SUPABASE_DB_PASSWORD"),
        help="Supabase database password (defaults to SUPABASE_DB_PASSWORD env var)",
    )
    parser.add_argument(
        "--supabase-db-user",
        default=os.getenv("SUPABASE_DB_USER", "postgres"),
        help="Supabase database user (defaults to SUPABASE_DB_USER env var)",
    )
    parser.add_argument(
        "--connection-string",
        default=os.getenv("SUPABASE_CONNECTION_STRING"),
        help="PostgreSQL connection string (alternative to URL/password)",
    )
    parser.add_argument(
        "--workspace",
        default=os.getenv("POSTGRES_WORKSPACE", "default"),
        help="Workspace name for separating RAG instances",
    )

    args = parser.parse_args()

    # Check required arguments
    if not args.api_key:
        logger.error("Error: OpenAI API key is required")
        logger.error("Set LLM_BINDING_API_KEY environment variable or use --api-key option")
        return

    if not args.connection_string and (not args.supabase_url or not args.supabase_db_password):
        logger.error("Error: Supabase configuration is required")
        logger.error(
            "Either provide --connection-string or both --supabase-url and --supabase-db-password"
        )
        logger.error("Or set SUPABASE_URL and SUPABASE_DB_PASSWORD environment variables")
        return

    # Create output directory if specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    # Process with Supabase
    asyncio.run(
        process_with_supabase(
            args.file_path,
            args.output,
            args.api_key,
            args.base_url,
            args.supabase_url,
            args.supabase_db_password,
            args.supabase_db_user,
            args.connection_string,
            args.workspace,
        )
    )


if __name__ == "__main__":
    print("RAG-Anything Supabase Integration Example")
    print("=" * 50)
    print("Processing document with Supabase PostgreSQL backend")
    print("=" * 50)
    print()

    main()


