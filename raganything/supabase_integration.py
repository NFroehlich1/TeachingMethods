"""
Supabase Integration for RAG-Anything

This module provides integration with Supabase PostgreSQL for vector storage,
key-value storage, and document status storage in RAG-Anything.

Supabase is a PostgreSQL-based backend-as-a-service that provides:
- PostgreSQL database with pgvector extension for vector similarity search
- Built-in connection pooling
- Row Level Security (RLS) for data protection
- Real-time subscriptions

Usage:
    from raganything.supabase_integration import setup_supabase_storage
    
    # Configure Supabase storage
    lightrag_kwargs = setup_supabase_storage(
        supabase_url="https://your-project.supabase.co",
        supabase_key="your-anon-key",
        supabase_db_password="your-db-password",
        workspace="default"
    )
    
    # Initialize RAGAnything with Supabase storage
    rag = RAGAnything(
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        lightrag_kwargs=lightrag_kwargs
    )
"""

import os
import base64
import json
from typing import Dict, Any, Optional
from urllib.parse import urlparse
from lightrag.utils import get_env_value


def extract_project_ref_from_jwt(jwt_token: str) -> Optional[str]:
    """
    Extract project reference from Supabase JWT token
    
    Args:
        jwt_token: Supabase JWT token (anon key or service key)
        
    Returns:
        Project reference string or None if extraction fails
    """
    try:
        # JWT tokens have 3 parts separated by dots
        parts = jwt_token.split(".")
        if len(parts) < 2:
            return None
        
        # Decode the payload (second part)
        payload = parts[1]
        # Add padding if needed
        padding = len(payload) % 4
        if padding:
            payload += "=" * (4 - padding)
        
        decoded = base64.urlsafe_b64decode(payload)
        data = json.loads(decoded)
        return data.get("ref")
    except Exception:
        return None


def get_supabase_url_from_jwt(jwt_token: str) -> Optional[str]:
    """
    Get Supabase project URL from JWT token
    
    Args:
        jwt_token: Supabase JWT token (anon key)
        
    Returns:
        Supabase project URL or None if extraction fails
    """
    project_ref = extract_project_ref_from_jwt(jwt_token)
    if project_ref:
        return f"https://{project_ref}.supabase.co"
    return None


def extract_supabase_connection_info(supabase_url: str) -> Dict[str, str]:
    """
    Extract connection information from Supabase URL
    
    Args:
        supabase_url: Supabase project URL (e.g., https://xxxxx.supabase.co)
        
    Returns:
        Dictionary with host, port, and database name
    """
    parsed = urlparse(supabase_url)
    host = parsed.hostname
    
    # Supabase uses port 5432 for direct PostgreSQL connections
    # The hostname format is typically: db.xxxxx.supabase.com
    # But we can also use the project reference: xxxxx.supabase.co or .com
    if host and not host.startswith("db."):
        # Convert project URL to database host
        # Extract project reference from hostname
        parts = host.split(".")
        if len(parts) >= 2 and parts[-2] == "supabase":
            project_ref = parts[0]
            host = f"db.{project_ref}.supabase.com"
    
    return {
        "host": host or "localhost",
        "port": "5432",
        "database": "postgres"  # Supabase uses 'postgres' as default database
    }


def setup_supabase_storage(
    supabase_url: Optional[str] = None,
    supabase_key: Optional[str] = None,
    supabase_access_token: Optional[str] = None,
    supabase_db_password: Optional[str] = None,
    supabase_db_user: Optional[str] = None,
    workspace: str = "default",
    max_connections: int = 12,
    enable_pgvector: bool = True,
) -> Dict[str, Any]:
    """
    Setup Supabase PostgreSQL storage for LightRAG
    
    This function configures LightRAG to use Supabase PostgreSQL for:
    - Vector storage (using pgvector extension)
    - Key-value storage
    - Document status storage
    
    Args:
        supabase_url: Supabase project URL (e.g., https://xxxxx.supabase.co)
                     If None, will try to extract from supabase_key JWT token
                     or reads from SUPABASE_URL environment variable
        supabase_key: Supabase anon/service key (JWT token)
                     Can be used to automatically extract project URL
                     If None, reads from SUPABASE_KEY environment variable
        supabase_access_token: Supabase access token for Management API
                              If None, reads from SUPABASE_ACCESS_TOKEN environment variable
        supabase_db_password: Database password for direct PostgreSQL connection
                             If None, reads from SUPABASE_DB_PASSWORD environment variable
        supabase_db_user: Database user (default: 'postgres')
                         If None, reads from SUPABASE_DB_USER environment variable
        workspace: Workspace name for separating different RAG instances
        max_connections: Maximum number of database connections
        enable_pgvector: Whether to enable pgvector extension (required for vector storage)
        
    Returns:
        Dictionary of LightRAG kwargs for PostgreSQL storage configuration
        
    Example:
        # Using JWT token to auto-detect URL
        lightrag_kwargs = setup_supabase_storage(
            supabase_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
            supabase_db_password="your-db-password"
        )
        
        # Or with explicit URL
        lightrag_kwargs = setup_supabase_storage(
            supabase_url="https://xxxxx.supabase.co",
            supabase_db_password="your-db-password"
        )
        
        rag = RAGAnything(
            llm_model_func=llm_model_func,
            embedding_func=embedding_func,
            lightrag_kwargs=lightrag_kwargs
        )
    """
    # Get values from environment variables if not provided
    supabase_key = supabase_key or get_env_value("SUPABASE_KEY", None, str)
    supabase_access_token = supabase_access_token or get_env_value(
        "SUPABASE_ACCESS_TOKEN", None, str
    )
    supabase_url = supabase_url or get_env_value("SUPABASE_URL", None, str)
    supabase_db_password = supabase_db_password or get_env_value(
        "SUPABASE_DB_PASSWORD", None, str
    )
    supabase_db_user = supabase_db_user or get_env_value(
        "SUPABASE_DB_USER", "postgres", str
    )
    workspace = workspace or get_env_value("POSTGRES_WORKSPACE", "default", str)
    max_connections = max_connections or int(
        get_env_value("POSTGRES_MAX_CONNECTIONS", "12", str)
    )

    # If URL not provided but JWT token is, extract URL from token
    if not supabase_url and supabase_key:
        supabase_url = get_supabase_url_from_jwt(supabase_key)
        if supabase_url:
            print(f"✓ Extracted Supabase URL from JWT: {supabase_url}")

    if not supabase_url:
        raise ValueError(
            "Supabase URL is required. Provide supabase_url parameter, supabase_key (JWT token), "
            "or set SUPABASE_URL environment variable."
        )

    if not supabase_db_password:
        raise ValueError(
            "Supabase database password is required. Provide supabase_db_password parameter "
            "or set SUPABASE_DB_PASSWORD environment variable. "
            "Note: Database password is different from API keys and can be found in "
            "Supabase Dashboard → Settings → Database → Database Password"
        )

    # Extract connection information from Supabase URL
    conn_info = extract_supabase_connection_info(supabase_url)

    # Configure PostgreSQL connection parameters
    postgres_config = {
        "host": conn_info["host"],
        "port": int(conn_info["port"]),
        "user": supabase_db_user,
        "password": supabase_db_password,
        "database": conn_info["database"],
        "workspace": workspace,
        "max_connections": max_connections,
    }

    # Configure LightRAG to use PostgreSQL storage
    lightrag_kwargs = {
        # Use PostgreSQL for vector storage (requires pgvector extension)
        "vector_storage": "PGVectorStorage",
        "vector_db_storage_cls_kwargs": postgres_config,
        # Use PostgreSQL for key-value storage
        "kv_storage": "PGKVStorage",
        "kv_storage_cls_kwargs": postgres_config,
        # Use PostgreSQL for document status storage
        "doc_status_storage": "PGDocStatusStorage",
        "doc_status_storage_cls_kwargs": postgres_config,
    }

    return lightrag_kwargs


def setup_supabase_from_connection_string(
    connection_string: Optional[str] = None,
    workspace: str = "default",
    max_connections: int = 12,
) -> Dict[str, Any]:
    """
    Setup Supabase storage from a PostgreSQL connection string
    
    This is an alternative way to configure Supabase storage using a direct
    PostgreSQL connection string.
    
    Args:
        connection_string: PostgreSQL connection string
                          Format: postgresql://user:password@host:port/database
                          If None, reads from SUPABASE_CONNECTION_STRING environment variable
        workspace: Workspace name for separating different RAG instances
        max_connections: Maximum number of database connections
        
    Returns:
        Dictionary of LightRAG kwargs for PostgreSQL storage configuration
        
    Example:
        lightrag_kwargs = setup_supabase_from_connection_string(
            connection_string="postgresql://postgres:password@db.xxxxx.supabase.com:5432/postgres"
        )
    """
    from urllib.parse import urlparse, unquote

    connection_string = connection_string or get_env_value(
        "SUPABASE_CONNECTION_STRING", None, str
    )

    if not connection_string:
        raise ValueError(
            "Connection string is required. Provide connection_string parameter or set SUPABASE_CONNECTION_STRING environment variable."
        )

    # Parse connection string
    parsed = urlparse(connection_string)

    postgres_config = {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
        "user": unquote(parsed.username or "postgres"),
        "password": unquote(parsed.password or ""),
        "database": parsed.path.lstrip("/") or "postgres",
        "workspace": workspace,
        "max_connections": max_connections,
    }

    # Configure LightRAG to use PostgreSQL storage
    lightrag_kwargs = {
        "vector_storage": "PGVectorStorage",
        "vector_db_storage_cls_kwargs": postgres_config,
        "kv_storage": "PGKVStorage",
        "kv_storage_cls_kwargs": postgres_config,
        "doc_status_storage": "PGDocStatusStorage",
        "doc_status_storage_cls_kwargs": postgres_config,
    }

    return lightrag_kwargs


def verify_supabase_connection(
    supabase_url: Optional[str] = None,
    supabase_db_password: Optional[str] = None,
    supabase_db_user: Optional[str] = None,
) -> bool:
    """
    Verify Supabase PostgreSQL connection
    
    Args:
        supabase_url: Supabase project URL
        supabase_db_password: Database password
        supabase_db_user: Database user
        
    Returns:
        True if connection is successful, False otherwise
    """
    try:
        try:
            import psycopg2
        except ImportError:
            raise ImportError(
                "psycopg2 is required for Supabase integration. "
                "Install it with: pip install raganything[supabase]"
            )
        from urllib.parse import urlparse

        supabase_url = supabase_url or get_env_value("SUPABASE_URL", None, str)
        supabase_db_password = supabase_db_password or get_env_value(
            "SUPABASE_DB_PASSWORD", None, str
        )
        supabase_db_user = supabase_db_user or get_env_value(
            "SUPABASE_DB_USER", "postgres", str
        )

        if not supabase_url or not supabase_db_password:
            return False

        conn_info = extract_supabase_connection_info(supabase_url)

        # Try to connect
        conn = psycopg2.connect(
            host=conn_info["host"],
            port=int(conn_info["port"]),
            user=supabase_db_user,
            password=supabase_db_password,
            database=conn_info["database"],
            connect_timeout=5,
        )
        conn.close()
        return True

    except Exception as e:
        print(f"Connection verification failed: {e}")
        return False


def create_pgvector_extension(
    supabase_url: Optional[str] = None,
    supabase_db_password: Optional[str] = None,
    supabase_db_user: Optional[str] = None,
) -> bool:
    """
    Create pgvector extension in Supabase database if it doesn't exist
    
    This function should be run once to enable vector similarity search.
    Note: You may need to run this manually in Supabase SQL editor if you don't
    have superuser privileges.
    
    Args:
        supabase_url: Supabase project URL
        supabase_db_password: Database password
        supabase_db_user: Database user
        
    Returns:
        True if extension was created successfully, False otherwise
    """
    try:
        try:
            import psycopg2
        except ImportError:
            raise ImportError(
                "psycopg2 is required for Supabase integration. "
                "Install it with: pip install raganything[supabase]"
            )

        supabase_url = supabase_url or get_env_value("SUPABASE_URL", None, str)
        supabase_db_password = supabase_db_password or get_env_value(
            "SUPABASE_DB_PASSWORD", None, str
        )
        supabase_db_user = supabase_db_user or get_env_value(
            "SUPABASE_DB_USER", "postgres", str
        )

        if not supabase_url or not supabase_db_password:
            print("Error: Supabase URL and password are required")
            return False

        conn_info = extract_supabase_connection_info(supabase_url)

        # Connect to database
        conn = psycopg2.connect(
            host=conn_info["host"],
            port=int(conn_info["port"]),
            user=supabase_db_user,
            password=supabase_db_password,
            database=conn_info["database"],
        )
        conn.autocommit = True
        cursor = conn.cursor()

        # Create extension if it doesn't exist
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cursor.close()
        conn.close()

        print("pgvector extension created successfully")
        return True

    except Exception as e:
        print(f"Error creating pgvector extension: {e}")
        print(
            "You may need to create the extension manually in Supabase SQL editor:"
        )
        print("  CREATE EXTENSION IF NOT EXISTS vector;")
        return False

