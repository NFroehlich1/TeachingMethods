# Supabase Configuration Complete ✅

Your Supabase integration has been configured with the following credentials:

## Project Information

- **Project URL**: `https://ngbhnjvojqqesacnijwk.supabase.co`
- **Anon Key**: Configured (JWT token)
- **Access Token**: Configured

## Quick Start

### 1. Install Dependencies

```bash
pip install raganything[supabase]
```

### 2. Set Database Password

You need your Supabase database password for direct PostgreSQL connections:

1. Go to [Supabase Dashboard](https://supabase.com/dashboard)
2. Select your project
3. Go to **Settings** → **Database**
4. Copy your **Database Password** (or reset it if needed)

Set it as an environment variable:

```bash
export SUPABASE_DB_PASSWORD="your_database_password"
```

Or add it to your `.env` file:

```env
SUPABASE_DB_PASSWORD=your_database_password
```

### 3. Enable pgvector Extension

Run this SQL in Supabase SQL Editor:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Or use the automated setup:

```python
from raganything import create_pgvector_extension, get_supabase_url_from_jwt

SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
SUPABASE_URL = get_supabase_url_from_jwt(SUPABASE_ANON_KEY)

create_pgvector_extension(
    supabase_url=SUPABASE_URL,
    supabase_db_password="your_password"
)
```

### 4. Use RAG-Anything with Supabase

#### Option A: Quick Start Script

```bash
python examples/supabase_quickstart.py [document.pdf]
```

#### Option B: Programmatic Setup

```python
import asyncio
from raganything import RAGAnything, setup_supabase_storage, get_supabase_url_from_jwt
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

# Your credentials
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5nYmhuanZvanFxZXNhY25pandrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjUwMDEyODcsImV4cCI6MjA4MDU3NzI4N30.-B_hvypDk3NLOYmYJ5qrJdojUtSFi7HSKK2aV2COsuU"
SUPABASE_ACCESS_TOKEN = "sbp_126aa9500ca99a588c1a0c748aa2fdda4a4cf391"

# Setup storage (URL auto-extracted from JWT)
lightrag_kwargs = setup_supabase_storage(
    supabase_key=SUPABASE_ANON_KEY,
    supabase_access_token=SUPABASE_ACCESS_TOKEN,
    supabase_db_password="your_database_password",
    workspace="default"
)

# Define LLM and embedding functions
def llm_model_func(prompt, system_prompt=None, **kwargs):
    return openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        api_key="your_openai_key"
    )

embedding_func = EmbeddingFunc(
    embedding_dim=3072,
    func=lambda texts: openai_embed(texts, model="text-embedding-3-large", api_key="your_openai_key")
)

# Initialize RAG-Anything
rag = RAGAnything(
    llm_model_func=llm_model_func,
    embedding_func=embedding_func,
    lightrag_kwargs=lightrag_kwargs
)

# Use it
async def main():
    # Process document
    await rag.process_document_complete("document.pdf", output_dir="./output")
    
    # Query
    result = await rag.aquery("What is this document about?")
    print(result)

asyncio.run(main())
```

#### Option C: Using Environment Variables

If you've set up your `.env` file with all credentials:

```python
from raganything import RAGAnything, setup_supabase_storage

# Configuration will be read from .env file automatically
lightrag_kwargs = setup_supabase_storage()

rag = RAGAnything(
    llm_model_func=llm_model_func,
    embedding_func=embedding_func,
    lightrag_kwargs=lightrag_kwargs
)
```

## Automated Setup Script

Run the automated setup script to configure everything:

```bash
python setup_supabase.py
```

This will:
- Extract project URL from JWT token
- Update `.env` file with credentials
- Verify database connection
- Set up pgvector extension

## What's Stored in Supabase?

RAG-Anything uses Supabase PostgreSQL to store:

- **Vector Embeddings**: Document chunks and their embeddings (using pgvector)
- **Key-Value Pairs**: Metadata, cache, and document information
- **Document Status**: Processing state and timestamps

All data is organized by workspace (default: "default") to support multiple RAG instances.

## Verification

Verify your setup:

```python
from raganything import verify_supabase_connection, get_supabase_url_from_jwt

SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
SUPABASE_URL = get_supabase_url_from_jwt(SUPABASE_ANON_KEY)

if verify_supabase_connection(
    supabase_url=SUPABASE_URL,
    supabase_db_password="your_password"
):
    print("✓ Connection successful!")
else:
    print("✗ Connection failed")
```

## Troubleshooting

### Connection Errors

- **"Connection refused"**: Check your database password
- **"Extension vector does not exist"**: Run `CREATE EXTENSION IF NOT EXISTS vector;` in SQL Editor
- **"Authentication failed"**: Verify your database password in Supabase Dashboard

### Performance

- For large datasets, consider creating HNSW indexes in Supabase
- Monitor connection pool usage in Supabase Dashboard
- Use workspace isolation for different RAG instances

## Next Steps

1. ✅ Credentials configured
2. ⏳ Set database password
3. ⏳ Enable pgvector extension
4. ⏳ Process your first document
5. ⏳ Start querying!

For more details, see [docs/supabase_integration.md](docs/supabase_integration.md)


