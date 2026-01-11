"""
LLM and LangChain functions for GitMate
Provides LLM initialization and interaction functions
"""

import re
import time
import json
import hashlib
from typing import Generator, Any, Callable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from pathlib import Path

from .models import CodeEntity
from .config import get_config
from .parsers import get_language_name


# ============================================================================
# Retry with Exponential Backoff
# ============================================================================

def _retry_with_backoff(
    func: Callable,
    *args,
    max_retries: int | None = None,
    base_delay: float | None = None,
    **kwargs
) -> Any:
    """
    Execute a function with exponential backoff retry on rate limit errors.

    Args:
        func: Function to execute
        *args: Positional arguments for the function
        max_retries: Maximum retry attempts (defaults to config)
        base_delay: Base delay in seconds (defaults to config)
        **kwargs: Keyword arguments for the function

    Returns:
        Function result

    Raises:
        Last exception if all retries fail
    """
    config = get_config()
    if max_retries is None:
        max_retries = config.max_retries
    if base_delay is None:
        base_delay = config.retry_base_delay

    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            error_str = str(e).lower()

            # Check if it's a rate limit error
            if "rate_limit" in error_str or "rate limit" in error_str or "429" in error_str:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    time.sleep(delay)
                    continue

            # For non-rate-limit errors, don't retry
            raise

    raise last_exception


# ============================================================================
# Entity Description Cache
# ============================================================================

def _get_cache_path() -> Path:
    """Get the cache directory path, creating it if needed"""
    config = get_config()
    cache_dir = config.cache_directory
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _get_entity_cache_key(entity: CodeEntity) -> str:
    """Generate a unique cache key for an entity based on its code content"""
    # Hash the code content + entity info for a unique key
    content = f"{entity.name}|{entity.entity_type}|{
        entity.file_path}|{entity.code}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _load_description_cache(repo_path: str) -> dict[str, str]:
    """Load cached descriptions for a repository"""
    cache_file = _get_cache_path(
    ) / f"{hashlib.sha256(repo_path.encode()).hexdigest()[:16]}_descriptions.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save_description_cache(repo_path: str, cache: dict[str, str]) -> None:
    """Save descriptions cache for a repository"""
    cache_file = _get_cache_path(
    ) / f"{hashlib.sha256(repo_path.encode()).hexdigest()[:16]}_descriptions.json"
    try:
        with open(cache_file, "w") as f:
            json.dump(cache, f)
    except IOError:
        pass  # Silently fail on cache write errors


def create_embeddings(model: str | None = None) -> GoogleGenerativeAIEmbeddings:
    """
    Create embeddings instance using Google Gemini.

    Args:
        model: Embedding model name (defaults to config)

    Returns:
        GoogleGenerativeAIEmbeddings instance
    """
    config = get_config()
    if model is None:
        model = config.embedding_model
    
    api_key = config.gemini_api_key
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required for embeddings")
    
    return GoogleGenerativeAIEmbeddings(
        model=model,
        google_api_key=api_key
    )


def create_llm(
    model: str | None = None,
    temperature: float | None = None
) -> ChatGroq:
    """
    Create LLM instance.

    Args:
        model: LLM model name (defaults to config)
        temperature: Model temperature (defaults to config)

    Returns:
        ChatGroq instance
    """
    config = get_config()
    if model is None:
        model = config.llm_model
    if temperature is None:
        temperature = config.llm_temperature
    return ChatGroq(model=model, temperature=temperature)


def create_streaming_llm(
    model: str | None = None,
    temperature: float | None = None
) -> ChatGroq:
    """
    Create streaming LLM instance.

    Args:
        model: LLM model name (defaults to config)
        temperature: Model temperature (defaults to config)

    Returns:
        ChatGroq instance with streaming enabled
    """
    config = get_config()
    if model is None:
        model = config.llm_model
    if temperature is None:
        temperature = config.llm_temperature
    return ChatGroq(model=model, temperature=temperature, streaming=True)


def analyze_entity(entity: CodeEntity, llm: ChatGroq | None = None) -> str:
    """
    Use LLM to generate a description of a code entity.

    Args:
        entity: The code entity to analyze
        llm: Optional LLM instance

    Returns:
        Generated description string
    """
    if llm is None:
        llm = create_llm()

    ext = Path(entity.file_path).suffix.lower()
    lang_name = get_language_name(ext)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a code analysis expert. Provide concise descriptions of code entities in 1-2 sentences."),
        ("human", """Analyze this {lang_name} code briefly:

Type: {entity_type}
Name: {name}
Code:
```
{code}
```

What does this {entity_type} do?""")
    ])

    chain = prompt | llm
    response = chain.invoke({
        "lang_name": lang_name,
        "entity_type": entity.entity_type,
        "name": entity.name,
        "code": entity.code
    })
    return response.content


def _analyze_entity_batch_chunk(
    entities_chunk: list[CodeEntity],
    llm: ChatGroq
) -> list[str]:
    """
    Analyze a batch of entities in a single LLM request.

    Args:
        entities_chunk: List of entities to analyze together
        llm: LLM instance

    Returns:
        List of descriptions in the same order as entities
    """
    config = get_config()
    max_code = getattr(config, 'max_code_length', 1000)

    # Build batch prompt with numbered entities
    batch_items = []
    for i, entity in enumerate(entities_chunk, 1):
        ext = Path(entity.file_path).suffix.lower()
        lang_name = get_language_name(ext)
        # Truncate code to save tokens
        code = entity.code
        if len(code) > max_code:
            code = code[:max_code] + "\n... (truncated)"
        batch_items.append(
            f"[{i}] {lang_name} {entity.entity_type}: {entity.name}\n"
            f"```\n{code}\n```"
        )

    batch_text = "\n\n".join(batch_items)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a code analysis expert. Analyze multiple code entities and provide concise 1-2 sentence descriptions for each.

IMPORTANT: Return EXACTLY one description per entity, numbered to match the input.
Format your response EXACTLY as:
[1] Description for first entity.
[2] Description for second entity.
... and so on.

Do not add any other text, just the numbered descriptions."""),
        ("human", """Analyze these code entities briefly:

{batch_text}

Provide a 1-2 sentence description for each numbered entity.""")
    ])

    chain = prompt | llm
    response = chain.invoke({"batch_text": batch_text})

    # Parse the numbered responses
    content = response.content
    descriptions = []

    for i in range(1, len(entities_chunk) + 1):
        # Match patterns like [1], [2], etc.
        pattern = rf'\[{i}\]\s*(.+?)(?=\[{i+1}\]|$)'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            desc = match.group(1).strip()
            # Clean up any trailing newlines or extra whitespace
            desc = ' '.join(desc.split())
            descriptions.append(desc)
        else:
            # Fallback: if parsing fails, use a generic description
            descriptions.append(
                f"{entities_chunk[i-1].entity_type} named {entities_chunk[i-1].name}")

    return descriptions


def analyze_entities_batch(
    entities: list[CodeEntity],
    llm: ChatGroq | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
    batch_size: int = 15,
    repo_path: str | None = None,
    use_cache: bool = True
) -> list[CodeEntity]:
    """
    Analyze multiple entities and set their descriptions.
    Uses batched LLM requests to reduce API calls and improve rate limit efficiency.
    Caches results to avoid re-analyzing unchanged code.

    Args:
        entities: List of entities to analyze
        llm: Optional LLM instance
        on_progress: Optional callback (current, total, entity_name)
        batch_size: Number of entities to analyze per LLM request (default: 15)
        repo_path: Repository path for caching (optional)
        use_cache: Whether to use description cache (default: True)

    Returns:
        List of entities with descriptions set
    """
    if llm is None:
        llm = create_llm()

    # Load cache if available
    cache: dict[str, str] = {}
    cache_key_prefix = repo_path or "default"
    if use_cache:
        cache = _load_description_cache(cache_key_prefix)

    total = len(entities)
    processed = 0
    entities_to_analyze: list[tuple[int, CodeEntity]] = []
    cache_updated = False

    # First pass: apply cached descriptions and identify entities needing analysis
    for i, entity in enumerate(entities):
        cache_key = _get_entity_cache_key(entity)
        if use_cache and cache_key in cache:
            entity.description = cache[cache_key]
            processed += 1
            if on_progress:
                on_progress(processed, total, entity.name)
        else:
            entities_to_analyze.append((i, entity))

    # Process uncached entities in batches
    for batch_start in range(0, len(entities_to_analyze), batch_size):
        batch_end = min(batch_start + batch_size, len(entities_to_analyze))
        batch_items = entities_to_analyze[batch_start:batch_end]
        batch = [item[1] for item in batch_items]

        try:
            # Analyze the entire batch in one LLM call with retry
            descriptions = _retry_with_backoff(
                _analyze_entity_batch_chunk, batch, llm)

            # Apply descriptions to entities and update cache
            for (idx, entity), desc in zip(batch_items, descriptions):
                entity.description = desc
                if use_cache:
                    cache_key = _get_entity_cache_key(entity)
                    cache[cache_key] = desc
                    cache_updated = True
                processed += 1
                if on_progress:
                    on_progress(processed, total, entity.name)

        except Exception as e:
            # Fallback to individual analysis if batch fails
            for idx, entity in batch_items:
                try:
                    entity.description = _retry_with_backoff(
                        analyze_entity, entity, llm)
                    if use_cache:
                        cache_key = _get_entity_cache_key(entity)
                        cache[cache_key] = entity.description
                        cache_updated = True
                except Exception:
                    entity.description = f"{
                        entity.entity_type} named {entity.name}"
                processed += 1
                if on_progress:
                    on_progress(processed, total, entity.name)

    # Save updated cache
    if use_cache and cache_updated:
        _save_description_cache(cache_key_prefix, cache)

    return entities


def generate_response(
    question: str,
    context: str,
    chat_history: list[BaseMessage] | None = None,
    llm: ChatGroq | None = None
) -> str:
    """
    Generate a response to a question using context.

    Args:
        question: User's question
        context: Relevant code context
        chat_history: Optional chat history
        llm: Optional LLM instance

    Returns:
        Generated response string
    """
    if llm is None:
        llm = create_llm()
    if chat_history is None:
        chat_history = []

    history_text = _format_chat_history(chat_history)

    prompt = ChatPromptTemplate.from_messages([
        ("system", _get_system_prompt()),
        ("human", "{question}"),
    ])

    chain = prompt | llm

    def _invoke():
        return chain.invoke({
            "context": context,
            "history": history_text,
            "question": question
        })

    response = _retry_with_backoff(_invoke)
    return response.content


def generate_streaming_response(
    question: str,
    context: str,
    chat_history: list[BaseMessage] | None = None,
    llm: ChatGroq | None = None
) -> Generator[str, None, None]:
    """
    Generate a streaming response to a question.

    Args:
        question: User's question
        context: Relevant code context
        chat_history: Optional chat history
        llm: Optional streaming LLM instance

    Yields:
        Response chunks as they're generated
    """
    if llm is None:
        llm = create_streaming_llm()
    if chat_history is None:
        chat_history = []

    history_text = _format_chat_history(chat_history)

    prompt = ChatPromptTemplate.from_messages([
        ("system", _get_system_prompt()),
        ("human", "{question}"),
    ])

    chain = prompt | llm

    # For streaming, we handle rate limits on the initial connection
    # by wrapping the stream creation in a retry
    config = get_config()
    last_exception = None

    for attempt in range(config.max_retries + 1):
        try:
            for chunk in chain.stream({
                "context": context,
                "history": history_text,
                "question": question
            }):
                if hasattr(chunk, 'content'):
                    yield chunk.content
            return  # Successfully completed streaming
        except Exception as e:
            last_exception = e
            error_str = str(e).lower()
            if "rate_limit" in error_str or "rate limit" in error_str or "429" in error_str:
                if attempt < config.max_retries:
                    delay = config.retry_base_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
            raise

    if last_exception:
        raise last_exception


def check_api_connection(llm: ChatGroq | None = None) -> bool:
    """
    Test that the LLM API is accessible.

    Args:
        llm: Optional LLM instance to test

    Returns:
        True if connection successful
    """
    if llm is None:
        llm = create_llm()

    try:
        response = llm.invoke("Reply with just 'ready'")
        return bool(response.content)
    except Exception:
        return False


def _format_chat_history(chat_history: list[BaseMessage], max_messages: int = 6) -> str:
    """Format chat history into a string"""
    if not chat_history:
        return ""

    history_parts = []
    for msg in chat_history[-max_messages:]:
        if isinstance(msg, HumanMessage):
            history_parts.append(f"User: {msg.content}")
        else:
            # Truncate long assistant responses in history
            content = msg.content[:500] + \
                "..." if len(msg.content) > 500 else msg.content
            history_parts.append(f"Assistant: {content}")

    return "\n".join(history_parts)


def _get_system_prompt() -> str:
    """Get the system prompt for the chat assistant"""
    return """You are GitMate, a coding assistant for understanding codebases. Use the code context below to answer questions. Reference specific functions/files when relevant. Be concise.

Code Context:
{context}

Chat History:
{history}"""


def clear_description_cache(repo_path: str | None = None) -> bool:
    """
    Clear the description cache for a repository.

    Args:
        repo_path: Repository path to clear cache for, or None to clear all

    Returns:
        True if cache was cleared successfully
    """
    import shutil
    cache_dir = _get_cache_path()

    try:
        if repo_path:
            # Clear specific repo cache
            cache_file = cache_dir / \
                f"{hashlib.sha256(repo_path.encode()).hexdigest()[
                    :16]}_descriptions.json"
            if cache_file.exists():
                cache_file.unlink()
        else:
            # Clear all caches
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                cache_dir.mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False


def get_cache_stats(repo_path: str) -> dict:
    """
    Get cache statistics for a repository.

    Args:
        repo_path: Repository path

    Returns:
        Dict with cache statistics
    """
    cache = _load_description_cache(repo_path)
    cache_file = _get_cache_path(
    ) / f"{hashlib.sha256(repo_path.encode()).hexdigest()[:16]}_descriptions.json"

    return {
        "cached_entities": len(cache),
        "cache_file": str(cache_file),
        "cache_exists": cache_file.exists(),
        "cache_size_bytes": cache_file.stat().st_size if cache_file.exists() else 0,
    }
