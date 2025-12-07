"""
Google Gemini Integration for RAG-Anything

This module provides integration with Google Gemini API for LLM and embedding functions.
"""

import os
from typing import List, Optional, Dict, Any
import asyncio

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


def setup_gemini(api_key: Optional[str] = None, model: str = "gemini-2.0-flash-exp"):
    """
    Setup Google Gemini API
    
    Args:
        api_key: Gemini API key. If None, reads from GEMINI_API_KEY environment variable
        model: Model name (default: gemini-2.0-flash-exp)
        
    Returns:
        Configured genai client
    """
    if not GEMINI_AVAILABLE:
        raise ImportError(
            "google-generativeai is required. Install it with: pip install google-generativeai"
        )
    
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Gemini API key is required. Provide api_key parameter or set GEMINI_API_KEY environment variable."
        )
    
    genai.configure(api_key=api_key)
    return genai


def gemini_llm_func(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict] = None,
    model: str = "gemini-2.0-flash-exp",
    api_key: Optional[str] = None,
    **kwargs,
) -> str:
    """
    LLM function for Gemini that works with LightRAG
    
    Args:
        prompt: User prompt
        system_prompt: System prompt (optional)
        history_messages: Conversation history
        model: Gemini model name
        api_key: Gemini API key
        **kwargs: Additional parameters
        
    Returns:
        Model response as string
    """
    if not GEMINI_AVAILABLE:
        raise ImportError(
            "google-generativeai is required. Install it with: pip install google-generativeai"
        )
    
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Gemini API key is required")
    
    genai.configure(api_key=api_key)
    
    # Create model
    gemini_model = genai.GenerativeModel(model)
    
    # Build messages
    messages = []
    
    # Add system prompt if provided
    if system_prompt:
        # Gemini uses system instruction differently
        # We'll prepend it to the first user message
        prompt = f"{system_prompt}\n\n{prompt}"
    
    # Add history if provided
    if history_messages:
        for msg in history_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system" and system_prompt is None:
                prompt = f"{content}\n\n{prompt}"
            elif role in ["user", "assistant"]:
                messages.append({"role": role, "parts": [content]})
    
    # Add current prompt
    messages.append({"role": "user", "parts": [prompt]})
    
    # Generate response
    try:
        response = gemini_model.generate_content(
            messages if messages else prompt,
            **kwargs
        )
        return response.text
    except Exception as e:
        raise Exception(f"Gemini API error: {e}")


async def gemini_llm_func_async(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict] = None,
    model: str = "gemini-2.0-flash-exp",
    api_key: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Async version of Gemini LLM function
    
    Args:
        prompt: User prompt
        system_prompt: System prompt (optional)
        history_messages: Conversation history
        model: Gemini model name
        api_key: Gemini API key
        **kwargs: Additional parameters
        
    Returns:
        Model response as string
    """
    # Run sync version in thread pool
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: gemini_llm_func(
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            model=model,
            api_key=api_key,
            **kwargs,
        ),
    )


def gemini_embed_func(
    texts: List[str],
    model: str = "models/text-embedding-004",
    api_key: Optional[str] = None,
) -> List[List[float]]:
    """
    Embedding function for Gemini
    
    Args:
        texts: List of texts to embed
        model: Embedding model name
        api_key: Gemini API key
        
    Returns:
        List of embedding vectors
    """
    if not GEMINI_AVAILABLE:
        raise ImportError(
            "google-generativeai is required. Install it with: pip install google-generativeai"
        )
    
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Gemini API key is required")
    
    genai.configure(api_key=api_key)
    
    # Get embedding model
    embedding_model = genai.Embedding(model=model)
    
    # Generate embeddings
    embeddings = []
    for text in texts:
        try:
            result = embedding_model.embed_content(text)
            embeddings.append(result["embedding"])
        except Exception as e:
            raise Exception(f"Gemini embedding error: {e}")
    
    return embeddings


async def gemini_embed_func_async(
    texts: List[str],
    model: str = "models/text-embedding-004",
    api_key: Optional[str] = None,
) -> List[List[float]]:
    """
    Async version of Gemini embedding function
    
    Args:
        texts: List of texts to embed
        model: Embedding model name
        api_key: Gemini API key
        
    Returns:
        List of embedding vectors
    """
    # Run sync version in thread pool
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: gemini_embed_func(texts=texts, model=model, api_key=api_key),
    )


def get_gemini_embedding_dim(model: str = "models/text-embedding-004") -> int:
    """
    Get embedding dimension for Gemini embedding model
    
    Args:
        model: Embedding model name
        
    Returns:
        Embedding dimension
    """
    # text-embedding-004 has 768 dimensions
    if "text-embedding-004" in model:
        return 768
    # Default fallback
    return 768

