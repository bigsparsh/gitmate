"""
Vector store operations for GitMate
FAISS-based semantic search functionality
"""

from typing import Any
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .models import CodeEntity, SearchResult
from .config import get_config
from .llm import create_embeddings


def build_vectorstore(
    entities: list[CodeEntity],
    embeddings: Any | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None
) -> FAISS:
    """
    Build a FAISS vector store from code entities.
    
    Args:
        entities: List of code entities to index
        embeddings: Optional embeddings instance
        chunk_size: Optional chunk size for splitting
        chunk_overlap: Optional chunk overlap
        
    Returns:
        FAISS vector store instance
    """
    config = get_config()
    
    if embeddings is None:
        embeddings = create_embeddings()
    if chunk_size is None:
        chunk_size = config.chunk_size
    if chunk_overlap is None:
        chunk_overlap = config.chunk_overlap
    
    # Convert entities to documents
    documents = [entity.to_document() for entity in entities]
    
    # Split large documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    split_docs = []
    for doc in documents:
        if len(doc.page_content) > chunk_size:
            chunks = text_splitter.split_documents([doc])
            for i, chunk in enumerate(chunks):
                chunk.metadata = doc.metadata.copy()
                chunk.metadata["chunk"] = i + 1
                chunk.metadata["total_chunks"] = len(chunks)
            split_docs.extend(chunks)
        else:
            split_docs.append(doc)
    
    return FAISS.from_documents(split_docs, embeddings)


def search_similar(
    vectorstore: FAISS,
    query: str,
    k: int | None = None
) -> list[SearchResult]:
    """
    Search for similar code entities.
    
    Args:
        vectorstore: FAISS vector store
        query: Search query
        k: Number of results (defaults to config)
        
    Returns:
        List of SearchResult objects
    """
    config = get_config()
    if k is None:
        k = config.default_search_k
    
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    search_results = []
    for doc, score in results:
        # Create a minimal entity from metadata
        entity = CodeEntity(
            name=doc.metadata.get("name", ""),
            entity_type=doc.metadata.get("entity_type", ""),
            file_path=doc.metadata.get("file_path", ""),
            start_line=doc.metadata.get("start_line", 0),
            end_line=doc.metadata.get("end_line", 0),
            code=""  # Code is in the document content
        )
        search_results.append(SearchResult(entity=entity, score=score))
    
    return search_results


def search_similar_raw(
    vectorstore: FAISS,
    query: str,
    k: int | None = None
) -> list[tuple[Document, float]]:
    """
    Search for similar documents (raw LangChain output).
    
    Args:
        vectorstore: FAISS vector store
        query: Search query
        k: Number of results
        
    Returns:
        List of (Document, score) tuples
    """
    config = get_config()
    if k is None:
        k = config.default_search_k
    
    return vectorstore.similarity_search_with_score(query, k=k)


def get_relevant_context(
    vectorstore: FAISS,
    query: str,
    k: int | None = None,
    max_length: int | None = None
) -> str:
    """
    Get relevant code context for a query.
    
    Args:
        vectorstore: FAISS vector store
        query: Search query
        k: Number of results to fetch
        max_length: Maximum context length
        
    Returns:
        Formatted context string
    """
    config = get_config()
    if k is None:
        k = config.default_search_k
    if max_length is None:
        max_length = config.max_context_length
    
    docs = vectorstore.similarity_search(query, k=k)
    
    # Deduplicate by entity name
    seen_entities = set()
    unique_parts = []
    
    for doc in docs:
        entity_key = (doc.metadata.get("name"), doc.metadata.get("file_path"))
        if entity_key not in seen_entities:
            seen_entities.add(entity_key)
            unique_parts.append(doc.page_content)
    
    # Limit to 4 unique entities to save tokens
    context = "\n\n---\n\n".join(unique_parts[:4])
    
    # Truncate if too long
    if len(context) > max_length:
        context = context[:max_length] + "\n... (truncated)"
    
    return context


def save_vectorstore(vectorstore: FAISS, path: str) -> None:
    """
    Save vector store to disk.
    
    Args:
        vectorstore: FAISS vector store
        path: Path to save to
    """
    vectorstore.save_local(path)


def load_vectorstore(
    path: str,
    embeddings: Any | None = None
) -> FAISS:
    """
    Load vector store from disk.
    
    Args:
        path: Path to load from
        embeddings: Optional embeddings instance
        
    Returns:
        FAISS vector store instance
    """
    if embeddings is None:
        embeddings = create_embeddings()
    
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
