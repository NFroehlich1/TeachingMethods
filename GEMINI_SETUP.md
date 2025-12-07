# Gemini Flash 2.0 Integration ✅

Your RAG-Anything setup is now configured to use **Google Gemini Flash 2.0** as the LLM!

## Configuration

- **LLM Model**: `gemini-2.0-flash-exp` (Gemini Flash 2.0)
- **Embedding Model**: `models/text-embedding-004` (768 dimensions)
- **API Key**: Configured in `.env` file
- **Storage**: Supabase PostgreSQL with pgvector

## Quick Start

### 1. Test the Integration

```bash
python examples/supabase_gemini_example.py
```

### 2. Process a Document

```bash
python examples/supabase_gemini_example.py document.pdf
```

## Usage in Code

```python
import asyncio
from raganything import RAGAnything, setup_supabase_storage
from raganything.gemini_integration import (
    gemini_llm_func,
    gemini_embed_func,
    get_gemini_embedding_dim,
)
from lightrag.utils import EmbeddingFunc

# Setup Supabase storage
lightrag_kwargs = setup_supabase_storage()

# Gemini LLM function
def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return gemini_llm_func(
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        model="gemini-2.0-flash-exp",
        api_key="AIzaSyBuTKegElwBTBfxdtbRwF47sYsa5vtF9_U",
        **kwargs,
    )

# Gemini embedding function
embedding_dim = get_gemini_embedding_dim()  # 768
embedding_func = EmbeddingFunc(
    embedding_dim=embedding_dim,
    func=lambda texts: gemini_embed_func(
        texts=texts,
        model="models/text-embedding-004",
        api_key="AIzaSyBuTKegElwBTBfxdtbRwF47sYsa5vtF9_U",
    ),
)

# Initialize RAG-Anything
rag = RAGAnything(
    llm_model_func=llm_model_func,
    embedding_func=embedding_func,
    lightrag_kwargs=lightrag_kwargs,
)

# Use it!
async def main():
    await rag.process_document_complete("document.pdf")
    result = await rag.aquery("What is this document about?")
    print(result)

asyncio.run(main())
```

## Embedding Dimensions

- **Gemini embeddings**: 768 dimensions
- **Table created**: `lightrag_vector_storage_gemini` (768 dimensions)
- **Default table**: `lightrag_vector_storage_default` (supports variable dimensions)

## Environment Variables

Your `.env` file now includes:

```env
GEMINI_API_KEY=AIzaSyBuTKegElwBTBfxdtbRwF47sYsa5vtF9_U
SUPABASE_URL=https://ngbhnjvojqqesacnijwk.supabase.co
SUPABASE_KEY=...
SUPABASE_DB_PASSWORD=...
```

## Features

✅ **Gemini Flash 2.0** - Fast and efficient LLM  
✅ **Gemini Embeddings** - 768-dimensional vectors  
✅ **Supabase Storage** - PostgreSQL with pgvector  
✅ **Automatic Setup** - Everything configured and ready  

## Next Steps

1. ✅ Gemini configured
2. ✅ Supabase storage ready
3. ⏳ Process your first document
4. ⏳ Start querying with Gemini!

---

**Everything is ready!** 🚀

Your RAG-Anything setup now uses Gemini Flash 2.0 for all LLM operations and stores everything in Supabase.

