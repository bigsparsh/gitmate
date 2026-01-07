"""
Chat and Q&A functions for GitMate
High-level chat functionality combining LLM, vector store, and entities
"""

from typing import Generator
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_community.vectorstores import FAISS

from .models import CodeEntity, SearchResult
from .config import get_config
from .vectorstore import get_relevant_context, search_similar_raw
from .llm import generate_response, generate_streaming_response, create_llm, create_streaming_llm


class ChatSession:
    """
    Manages a chat session with history and context.
    """
    
    def __init__(
        self,
        vectorstore: FAISS,
        entities: list[CodeEntity],
        max_history: int | None = None
    ):
        """
        Initialize chat session.
        
        Args:
            vectorstore: FAISS vector store for context retrieval
            entities: List of code entities
            max_history: Maximum chat history length
        """
        config = get_config()
        self.vectorstore = vectorstore
        self.entities = entities
        self.max_history = max_history or config.max_chat_history
        self.chat_history: list[BaseMessage] = []
        self.llm = create_llm()
        self.streaming_llm = create_streaming_llm()
    
    def ask(self, question: str) -> str:
        """
        Ask a question and get a response.
        
        Args:
            question: User's question
            
        Returns:
            Assistant's response
        """
        context = get_relevant_context(self.vectorstore, question)
        response = generate_response(
            question=question,
            context=context,
            chat_history=self.chat_history,
            llm=self.llm
        )
        
        self._update_history(question, response)
        return response
    
    def ask_streaming(self, question: str) -> Generator[str, None, str]:
        """
        Ask a question and get a streaming response.
        
        Args:
            question: User's question
            
        Yields:
            Response chunks
            
        Returns:
            Full response (after iteration complete)
        """
        context = get_relevant_context(self.vectorstore, question)
        full_response = ""
        
        for chunk in generate_streaming_response(
            question=question,
            context=context,
            chat_history=self.chat_history,
            llm=self.streaming_llm
        ):
            full_response += chunk
            yield chunk
        
        self._update_history(question, full_response)
        return full_response
    
    def search(self, query: str, k: int = 5) -> list[tuple]:
        """
        Search for relevant code.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of (Document, score) tuples
        """
        return search_similar_raw(self.vectorstore, query, k=k)
    
    def clear_history(self) -> None:
        """Clear chat history"""
        self.chat_history.clear()
    
    def _update_history(self, question: str, response: str) -> None:
        """Update chat history with new exchange"""
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=response))
        
        # Trim if too long
        if len(self.chat_history) > self.max_history:
            self.chat_history = self.chat_history[-self.max_history:]


def answer_question(
    question: str,
    vectorstore: FAISS,
    chat_history: list[BaseMessage] | None = None
) -> str:
    """
    Answer a question using the vector store.
    
    Args:
        question: User's question
        vectorstore: FAISS vector store
        chat_history: Optional chat history
        
    Returns:
        Generated answer
    """
    context = get_relevant_context(vectorstore, question)
    return generate_response(
        question=question,
        context=context,
        chat_history=chat_history
    )


def answer_question_streaming(
    question: str,
    vectorstore: FAISS,
    chat_history: list[BaseMessage] | None = None
) -> Generator[str, None, None]:
    """
    Answer a question with streaming output.
    
    Args:
        question: User's question
        vectorstore: FAISS vector store
        chat_history: Optional chat history
        
    Yields:
        Response chunks
    """
    context = get_relevant_context(vectorstore, question)
    yield from generate_streaming_response(
        question=question,
        context=context,
        chat_history=chat_history
    )


def find_entity_by_name(
    name: str,
    entities: list[CodeEntity]
) -> list[CodeEntity]:
    """
    Find entities by name (partial match).
    
    Args:
        name: Name to search for (case-insensitive partial match)
        entities: List of entities to search
        
    Returns:
        List of matching entities
    """
    name_lower = name.lower()
    return [e for e in entities if name_lower in e.name.lower()]


def get_entity_references(
    name: str,
    entities: list[CodeEntity],
    max_matches: int = 5
) -> list[dict]:
    """
    Get reference information for an entity.
    
    Args:
        name: Entity name to search for
        entities: List of entities
        max_matches: Maximum entities to return
        
    Returns:
        List of reference info dictionaries
    """
    config = get_config()
    matches = find_entity_by_name(name, entities)
    
    results = []
    for entity in matches[:max_matches]:
        info = {
            "name": entity.name,
            "entity_type": entity.entity_type,
            "file_path": entity.file_path,
            "line": entity.start_line,
            "column": entity.name_column,
        }
        
        if entity.entity_type in config.callable_types:
            # For functions, show callers
            info["callers"] = [
                {"name": c.name, "file_path": c.file_path, "line": c.line}
                for c in entity.incoming_calls
            ]
            info["callees"] = [
                {"name": c.name, "file_path": c.file_path, "line": c.line}
                for c in entity.outgoing_calls
            ]
        else:
            # For other types, show references
            info["references"] = [
                {"file_path": r.file_path, "line": r.line}
                for r in entity.references
            ]
        
        results.append(info)
    
    return results


def get_call_hierarchy(
    name: str,
    entities: list[CodeEntity],
    max_matches: int = 3
) -> list[dict]:
    """
    Get call hierarchy for a function.
    
    Args:
        name: Function name to search for
        entities: List of entities
        max_matches: Maximum functions to return
        
    Returns:
        List of call hierarchy dictionaries
    """
    config = get_config()
    matches = find_entity_by_name(name, entities)
    
    # Filter to only callable types
    matches = [e for e in matches if e.entity_type in config.callable_types]
    
    results = []
    for entity in matches[:max_matches]:
        results.append({
            "name": entity.name,
            "entity_type": entity.entity_type,
            "file_path": entity.file_path,
            "line": entity.start_line,
            "incoming_calls": [
                {"name": c.name, "kind": c.kind, "file_path": c.file_path, "line": c.line}
                for c in entity.incoming_calls
            ],
            "outgoing_calls": [
                {"name": c.name, "kind": c.kind, "file_path": c.file_path, "line": c.line}
                for c in entity.outgoing_calls
            ],
        })
    
    return results
