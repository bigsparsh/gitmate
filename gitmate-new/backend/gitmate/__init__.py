"""
GitMate: An onboarding application for newbies to github codebase
Modular architecture with reusable functions for multiple server types.
"""

from .models import CodeEntity
from .config import Config, get_config
from .parsers import (
    create_parsers,
    extract_entities_from_file,
    get_language_for_extension,
)
from .repo import (
    clone_repository,
    get_source_files,
    analyze_codebase,
)
from .llm import (
    create_embeddings,
    create_llm,
    create_streaming_llm,
    analyze_entity,
    generate_response,
    generate_streaming_response,
    clear_description_cache,
    get_cache_stats,
)
from .vectorstore import (
    build_vectorstore,
    search_similar,
    get_relevant_context,
)
from .chat import (
    answer_question,
    answer_question_streaming,
    find_entity_by_name,
    get_entity_references,
    get_call_hierarchy,
)
from .lsp_client import LSPManager, SymbolReferences, LSPReference, CallHierarchyItem

__all__ = [
    # Models
    "CodeEntity",
    # Config
    "Config",
    "get_config",
    # Parsers
    "create_parsers",
    "extract_entities_from_file",
    "get_language_for_extension",
    # Repository
    "clone_repository",
    "get_source_files",
    "analyze_codebase",
    # LLM
    "create_embeddings",
    "create_llm",
    "create_streaming_llm",
    "analyze_entity",
    "generate_response",
    "generate_streaming_response",
    "clear_description_cache",
    "get_cache_stats",
    # Vector Store
    "build_vectorstore",
    "search_similar",
    "get_relevant_context",
    # Chat
    "answer_question",
    "answer_question_streaming",
    "find_entity_by_name",
    "get_entity_references",
    "get_call_hierarchy",
    # LSP
    "LSPManager",
    "SymbolReferences",
    "LSPReference",
    "CallHierarchyItem",
]
