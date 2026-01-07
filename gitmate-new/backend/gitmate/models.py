"""
Data models for GitMate
"""

from dataclasses import dataclass, field
from pathlib import Path
from langchain_core.documents import Document


@dataclass
class CodeEntity:
    """Represents a code entity (function, variable, struct, etc.)"""
    name: str
    entity_type: str  # function, variable, struct, etc.
    file_path: str
    start_line: int
    end_line: int
    code: str
    # Column where the name starts (for precise LSP queries)
    name_column: int = 0
    description: str = ""
    # LSP-enhanced data
    # Where this entity is used (for variables/structs)
    references: list = field(default_factory=list)
    # Functions that call this (for functions only)
    incoming_calls: list = field(default_factory=list)
    # Functions this calls (for functions only)
    outgoing_calls: list = field(default_factory=list)

    def __str__(self):
        return f"{self.entity_type}: {self.name} ({self.file_path}:{self.start_line}-{self.end_line})"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "entity_type": self.entity_type,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "code": self.code,
            "name_column": self.name_column,
            "description": self.description,
            "num_references": len(self.references),
            "num_incoming_calls": len(self.incoming_calls),
            "num_outgoing_calls": len(self.outgoing_calls),
        }

    def to_document(self) -> Document:
        """Convert to LangChain Document for vector store"""
        # Determine language from file extension
        ext = Path(self.file_path).suffix.lower()
        lang_map = {'.c': 'c', '.h': 'c', '.ts': 'typescript', '.tsx': 'tsx'}
        lang = lang_map.get(ext, 'text')

        # Build references section
        refs_text = ""
        if self.references:
            refs_list = [
                f"  - {ref.file_path}:{ref.line}" for ref in self.references[:10]
            ]
            refs_text = f"\n\nUsed in ({len(self.references)} locations):\n" + "\n".join(refs_list)
            if len(self.references) > 10:
                refs_text += f"\n  ... and {len(self.references) - 10} more"

        # Build call hierarchy section
        calls_text = ""
        if self.incoming_calls:
            callers = [
                f"  - {c.name} ({c.file_path}:{c.line})" for c in self.incoming_calls[:5]
            ]
            calls_text += f"\n\nCalled by:\n" + "\n".join(callers)
        if self.outgoing_calls:
            callees = [
                f"  - {c.name} ({c.file_path}:{c.line})" for c in self.outgoing_calls[:5]
            ]
            calls_text += f"\n\nCalls:\n" + "\n".join(callees)

        content = f"""Type: {self.entity_type}
Name: {self.name}
File: {self.file_path} (lines {self.start_line}-{self.end_line})
Description: {self.description}{refs_text}{calls_text}

Code:
```{lang}
{self.code}
```"""
        return Document(
            page_content=content,
            metadata={
                "name": self.name,
                "entity_type": self.entity_type,
                "file_path": self.file_path,
                "start_line": self.start_line,
                "end_line": self.end_line,
                "num_references": len(self.references),
                "num_callers": len(self.incoming_calls),
                "num_callees": len(self.outgoing_calls),
            }
        )


@dataclass
class ChatMessage:
    """Represents a chat message"""
    role: str  # "user" or "assistant"
    content: str

    def to_langchain(self):
        """Convert to LangChain message format"""
        from langchain_core.messages import HumanMessage, AIMessage
        if self.role == "user":
            return HumanMessage(content=self.content)
        return AIMessage(content=self.content)


@dataclass
class SearchResult:
    """Represents a search result"""
    entity: CodeEntity
    score: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "entity": self.entity.to_dict(),
            "score": self.score,
            "relevance": 1 / (1 + self.score),
        }
