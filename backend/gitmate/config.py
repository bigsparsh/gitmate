"""
Configuration management for GitMate
"""

from dataclasses import dataclass, field
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class Config:
    """Configuration settings for GitMate"""
    # Embedding model (Google Gemini) - can override with GITMATE_EMBEDDING_MODEL
    embedding_model: str = field(default_factory=lambda: os.getenv("GITMATE_EMBEDDING_MODEL", "gemini-embedding-001"))
    
    # Gemini API key for embeddings (falls back to GOOGLE_API_KEY)
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")))
    
    # LLM model (Groq) - can override with GITMATE_LLM_MODEL
    llm_model: str = field(default_factory=lambda: os.getenv("GITMATE_LLM_MODEL", "llama-3.1-8b-instant"))
    llm_temperature: float = 0.0
    
    # Rate limit retry settings
    max_retries: int = 3
    retry_base_delay: float = 2.0  # seconds, will be multiplied exponentially
    
    # Vector store settings - reduced for token efficiency
    chunk_size: int = 1500
    chunk_overlap: int = 100
    max_context_length: int = 6000
    
    # Search settings - fewer results = fewer tokens
    default_search_k: int = 5
    
    # Chat history settings - reduced to save tokens
    max_chat_history: int = 10
    
    # Max code length for entity analysis (truncate large functions)
    max_code_length: int = 1000
    
    # Cache directory for entity descriptions
    cache_directory: Path = field(default_factory=lambda: Path(os.getenv("GITMATE_CACHE_DIR", "/tmp/gitmate_cache")))
    
    # Supported file extensions
    supported_extensions: list = field(default_factory=lambda: [
        '.c', '.h', '.cpp', '.hpp', '.cc', '.cxx', '.hxx', '.ts', '.tsx', '.json'
    ])
    
    # Directories to ignore
    ignore_directories: list = field(default_factory=lambda: [
        'node_modules', '.git', '__pycache__', 'dist', 'build'
    ])
    
    # Temp directory for cloning repos
    temp_directory: Path = field(default_factory=lambda: Path("/tmp/gitmate"))
    
    # Entity types that are callable (use call hierarchy)
    callable_types: frozenset = field(default_factory=lambda: frozenset({
        'function', 'arrow_function', 'method'
    }))
    
    # Entity types that are referenceable (use references)
    referenceable_types: frozenset = field(default_factory=lambda: frozenset({
        'global_variable', 'struct', 'union', 'enum', 'typedef', 
        'macro', 'interface', 'type_alias', 'class'
    }))


# Global config instance
_config: Config | None = None


def get_config() -> Config:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance"""
    global _config
    _config = config


def check_api_keys() -> dict[str, bool]:
    """Check which API keys are available"""
    return {
        "groq": bool(os.getenv("GROQ_API_KEY")),
        "google": bool(os.getenv("GOOGLE_API_KEY")),
        "openai": bool(os.getenv("OPENAI_API_KEY")),
    }
