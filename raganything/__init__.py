from .raganything import RAGAnything as RAGAnything
from .config import RAGAnythingConfig as RAGAnythingConfig
from .supabase_integration import (
    setup_supabase_storage,
    setup_supabase_from_connection_string,
    verify_supabase_connection,
    create_pgvector_extension,
    extract_project_ref_from_jwt,
    get_supabase_url_from_jwt,
)

__version__ = "1.2.8"
__author__ = "Zirui Guo"
__url__ = "https://github.com/HKUDS/RAG-Anything"

__all__ = [
    "RAGAnything",
    "RAGAnythingConfig",
    "setup_supabase_storage",
    "setup_supabase_from_connection_string",
    "verify_supabase_connection",
    "create_pgvector_extension",
    "extract_project_ref_from_jwt",
    "get_supabase_url_from_jwt",
]
