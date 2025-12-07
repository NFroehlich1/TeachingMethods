# Supabase Integration Guide

This guide explains how to integrate RAG-Anything with Supabase PostgreSQL for vector storage and document management.

## Overview

Supabase is a PostgreSQL-based backend-as-a-service that provides:
- **PostgreSQL database** with pgvector extension for vector similarity search
- **Built-in connection pooling** for better performance
- **Row Level Security (RLS)** for data protection
- **Real-time subscriptions** (optional, for future features)

RAG-Anything uses Supabase PostgreSQL to store:
- **Vector embeddings** (using pgvector extension)
- **Key-value pairs** (document metadata, cache, etc.)
- **Document status** (processing state, timestamps, etc.)

## Prerequisites

1. **Supabase Account**: Create a free account at [https://supabase.com](https://supabase.com)
2. **Supabase Project**: Create a new project in your Supabase dashboard
3. **Database Password**: Get your database password from Supabase project settings
4. **Python Dependencies**: Install RAG-Anything with Supabase support

## Installation

Install RAG-Anything with Supabase support:

```bash
pip install raganything[supabase]
```

Or install all optional dependencies:

```bash
pip install raganything[all]
```

## Setup Steps

### 1. Enable pgvector Extension

The pgvector extension is required for vector similarity search. Enable it in your Supabase database:

**Option A: Using Supabase SQL Editor**

1. Go to your Supabase project dashboard
2. Navigate to **SQL Editor**
3. Run the following SQL:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

**Option B: Using Python (Automatic)**

The integration can attempt to create the extension automatically:

```python
from raganything import create_pgvector_extension

create_pgvector_extension(
    supabase_url="https://xxxxx.supabase.co",
    supabase_db_password="your_password"
)
```

### 2. Get Supabase Connection Details

From your Supabase project dashboard:

1. Go to **Settings** → **API**
2. Copy your **Project URL** (e.g., `https://xxxxx.supabase.co`)
3. Go to **Settings** → **Database**
4. Copy your **Database Password** (or reset it if needed)

### 3. Configure Environment Variables

Create a `.env` file in your project root:

```env
# Supabase Configuration
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_DB_PASSWORD=your_database_password
SUPABASE_DB_USER=postgres

# Optional: Use connection string instead
# SUPABASE_CONNECTION_STRING=postgresql://postgres:password@db.xxxxx.supabase.co:5432/postgres

# LLM Configuration (required for RAG)
LLM_BINDING_API_KEY=your_openai_api_key
LLM_BINDING_HOST=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini

# Embedding Configuration
EMBEDDING_BINDING=openai
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIM=3072
```

## Usage

### Basic Usage

```python
import asyncio
from raganything import RAGAnything, RAGAnythingConfig, setup_supabase_storage
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

# Setup Supabase storage
lightrag_kwargs = setup_supabase_storage(
    supabase_url="https://xxxxx.supabase.co",
    supabase_db_password="your_password",
    workspace="default"  # Optional: separate different RAG instances
)

# Define LLM and embedding functions
def llm_model_func(prompt, system_prompt=None, **kwargs):
    return openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        api_key="your_api_key"
    )

embedding_func = EmbeddingFunc(
    embedding_dim=3072,
    func=lambda texts: openai_embed(texts, model="text-embedding-3-large", api_key="your_api_key")
)

# Initialize RAG-Anything with Supabase
rag = RAGAnything(
    llm_model_func=llm_model_func,
    embedding_func=embedding_func,
    lightrag_kwargs=lightrag_kwargs  # Supabase storage configuration
)

# Process document
async def main():
    await rag.process_document_complete(
        file_path="document.pdf",
        output_dir="./output"
    )
    
    # Query
    result = await rag.aquery("What is this document about?")
    print(result)

asyncio.run(main())
```

### Using Connection String

Alternatively, you can use a PostgreSQL connection string:

```python
from raganything import setup_supabase_from_connection_string

lightrag_kwargs = setup_supabase_from_connection_string(
    connection_string="postgresql://postgres:password@db.xxxxx.supabase.co:5432/postgres",
    workspace="default"
)
```

### Verify Connection

Before processing documents, verify your Supabase connection:

```python
from raganything import verify_supabase_connection

if verify_supabase_connection(
    supabase_url="https://xxxxx.supabase.co",
    supabase_db_password="your_password"
):
    print("✓ Connection successful")
else:
    print("✗ Connection failed")
```

## Example Script

See `examples/supabase_integration_example.py` for a complete working example:

```bash
python examples/supabase_integration_example.py document.pdf \
    --api-key your_openai_key \
    --supabase-url https://xxxxx.supabase.co \
    --supabase-db-password your_password
```

## Workspace Isolation

Use different workspace names to separate different RAG instances:

```python
# Production workspace
lightrag_kwargs_prod = setup_supabase_storage(
    supabase_url="https://xxxxx.supabase.co",
    supabase_db_password="your_password",
    workspace="production"
)

# Development workspace
lightrag_kwargs_dev = setup_supabase_storage(
    supabase_url="https://xxxxx.supabase.co",
    supabase_db_password="your_password",
    workspace="development"
)
```

## Database Schema

RAG-Anything automatically creates the following tables in your Supabase database:

- **Vector storage tables**: For storing document embeddings
- **Key-value storage tables**: For metadata and cache
- **Document status tables**: For tracking document processing state

All tables are prefixed with your workspace name to support multiple RAG instances.

## Performance Considerations

1. **Connection Pooling**: Supabase provides built-in connection pooling. The default `max_connections=12` is usually sufficient.

2. **Vector Index**: pgvector automatically creates indexes for efficient similarity search. For large datasets, consider creating additional indexes:

```sql
-- Create HNSW index for faster similarity search (PostgreSQL 12+)
CREATE INDEX ON vector_table USING hnsw (embedding vector_cosine_ops);
```

3. **Batch Processing**: Process multiple documents in batches to optimize database writes.

## Troubleshooting

### Connection Errors

**Error**: `Connection refused` or `Timeout`

- Verify your Supabase URL is correct
- Check that your database password is correct
- Ensure your IP is not blocked (check Supabase dashboard → Settings → Database → Connection Pooling)

**Error**: `extension "vector" does not exist`

- Run `CREATE EXTENSION IF NOT EXISTS vector;` in Supabase SQL Editor
- Or use `create_pgvector_extension()` function

### Performance Issues

- **Slow queries**: Check if pgvector indexes are created
- **Connection pool exhausted**: Increase `max_connections` parameter
- **Large datasets**: Consider using Supabase's connection pooling mode

## Security Best Practices

1. **Never commit credentials**: Always use environment variables or `.env` files (add `.env` to `.gitignore`)

2. **Use Row Level Security**: Enable RLS in Supabase for additional data protection

3. **Connection String**: Prefer using environment variables over hardcoding connection strings

4. **API Keys**: Rotate database passwords and API keys regularly

## Limitations

- **pgvector version**: Ensure your Supabase instance supports pgvector (most do by default)
- **Storage limits**: Free tier has storage limits; monitor usage in Supabase dashboard
- **Connection limits**: Free tier has connection limits; upgrade if needed

## Additional Resources

- [Supabase Documentation](https://supabase.com/docs)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [RAG-Anything Examples](../examples/)
- [LightRAG Documentation](https://github.com/HKUDS/LightRAG)

## Support

For issues or questions:
- Open an issue on [GitHub](https://github.com/HKUDS/RAG-Anything/issues)
- Check existing examples in `examples/` directory
- Review LightRAG documentation for storage configuration details


