"""
GitMate: An onboarding application for newbies to github codebase
Enhanced with LangChain for RAG and streaming responses
"""

import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass, field

import git
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from tree_sitter import Language, Parser
import tree_sitter_c as tsc

# LangChain imports
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

console = Console()

# Models configuration
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5-coder:7b"


@dataclass
class CodeEntity:
    """Represents a code entity (function, variable, struct, etc.)"""
    name: str
    entity_type: str  # function, variable, struct, etc.
    file_path: str
    start_line: int
    end_line: int
    code: str
    description: str = ""
    
    def __str__(self):
        return f"{self.entity_type}: {self.name} ({self.file_path}:{self.start_line}-{self.end_line})"
    
    def to_document(self) -> Document:
        """Convert to LangChain Document"""
        content = f"""Type: {self.entity_type}
Name: {self.name}
File: {self.file_path} (lines {self.start_line}-{self.end_line})
Description: {self.description}

Code:
```c
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
            }
        )


class GitMate:
    """Main GitMate application class with LangChain integration"""
    
    def __init__(self):
        self.console = console
        self.repo_path: Path = None
        self.c_language = Language(tsc.language())
        self.parser = Parser(self.c_language)
        
        # LangChain components
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.llm = ChatOllama(model=LLM_MODEL, temperature=0)
        self.streaming_llm = ChatOllama(model=LLM_MODEL, temperature=0, streaming=True)
        self.vectorstore: FAISS = None
        self.chat_history: list = []
        self.entities: list[CodeEntity] = []
        
    def clone_repo(self, repo_url: str) -> Path:
        """Clone a repository to /tmp/"""
        repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
        clone_path = Path(tempfile.gettempdir()) / "gitmate" / repo_name
        
        # Clean up if exists
        if clone_path.exists():
            shutil.rmtree(clone_path)
        
        clone_path.parent.mkdir(parents=True, exist_ok=True)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            progress.add_task(f"Cloning {repo_url}...", total=None)
            git.Repo.clone_from(repo_url, clone_path)
        
        self.console.print(f"[green]✓ Cloned repository to {clone_path}[/green]")
        self.repo_path = clone_path
        return clone_path
    
    def _extract_entities_from_node(self, node, source_code: bytes, file_path: str) -> list[CodeEntity]:
        """Recursively extract code entities from AST nodes"""
        entities = []
        
        # Function definitions
        if node.type == "function_definition":
            declarator = node.child_by_field_name("declarator")
            name = self._get_declarator_name(declarator)
            if name:
                entities.append(CodeEntity(
                    name=name,
                    entity_type="function",
                    file_path=file_path,
                    start_line=node.start_point.row + 1,
                    end_line=node.end_point.row + 1,
                    code=source_code[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
                ))
        
        # Struct/union definitions
        elif node.type in ("struct_specifier", "union_specifier"):
            name_node = node.child_by_field_name("name")
            if name_node:
                name = source_code[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="ignore")
                entities.append(CodeEntity(
                    name=name,
                    entity_type="struct" if node.type == "struct_specifier" else "union",
                    file_path=file_path,
                    start_line=node.start_point.row + 1,
                    end_line=node.end_point.row + 1,
                    code=source_code[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
                ))
        
        # Enum definitions
        elif node.type == "enum_specifier":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = source_code[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="ignore")
                entities.append(CodeEntity(
                    name=name,
                    entity_type="enum",
                    file_path=file_path,
                    start_line=node.start_point.row + 1,
                    end_line=node.end_point.row + 1,
                    code=source_code[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
                ))
        
        # Global variable declarations
        elif node.type == "declaration" and node.parent and node.parent.type == "translation_unit":
            declarator = node.child_by_field_name("declarator")
            name = self._get_declarator_name(declarator)
            if name:
                entities.append(CodeEntity(
                    name=name,
                    entity_type="global_variable",
                    file_path=file_path,
                    start_line=node.start_point.row + 1,
                    end_line=node.end_point.row + 1,
                    code=source_code[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
                ))
        
        # Typedef
        elif node.type == "type_definition":
            declarator = node.child_by_field_name("declarator")
            name = self._get_declarator_name(declarator)
            if name:
                entities.append(CodeEntity(
                    name=name,
                    entity_type="typedef",
                    file_path=file_path,
                    start_line=node.start_point.row + 1,
                    end_line=node.end_point.row + 1,
                    code=source_code[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
                ))
        
        # Preprocessor defines
        elif node.type == "preproc_def":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = source_code[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="ignore")
                entities.append(CodeEntity(
                    name=name,
                    entity_type="macro",
                    file_path=file_path,
                    start_line=node.start_point.row + 1,
                    end_line=node.end_point.row + 1,
                    code=source_code[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
                ))
        
        # Recurse into children
        for child in node.children:
            entities.extend(self._extract_entities_from_node(child, source_code, file_path))
        
        return entities
    
    def _get_declarator_name(self, node) -> str | None:
        """Extract the name from a declarator node"""
        if node is None:
            return None
        
        if node.type == "identifier":
            return node.text.decode("utf-8", errors="ignore") if node.text else None
        
        if node.type in ("function_declarator", "pointer_declarator", "array_declarator"):
            declarator = node.child_by_field_name("declarator")
            return self._get_declarator_name(declarator)
        
        if node.type == "init_declarator":
            declarator = node.child_by_field_name("declarator")
            return self._get_declarator_name(declarator)
        
        return None
    
    def analyze_file(self, file_path: Path) -> list[CodeEntity]:
        """Parse a C file and extract all code entities"""
        try:
            with open(file_path, "rb") as f:
                source_code = f.read()
            
            tree = self.parser.parse(source_code)
            rel_path = str(file_path.relative_to(self.repo_path)) if self.repo_path else str(file_path)
            entities = self._extract_entities_from_node(tree.root_node, source_code, rel_path)
            return entities
        except Exception as e:
            self.console.print(f"[yellow]Warning: Could not parse {file_path}: {e}[/yellow]")
            return []
    
    def analyze_codebase(self) -> list[CodeEntity]:
        """Analyze all C files in the repository"""
        if not self.repo_path:
            raise ValueError("No repository loaded. Call clone_repo first.")
        
        all_entities = []
        c_files = list(self.repo_path.rglob("*.c")) + list(self.repo_path.rglob("*.h"))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Analyzing codebase...", total=len(c_files))
            
            for file_path in c_files:
                entities = self.analyze_file(file_path)
                all_entities.extend(entities)
                progress.update(task, advance=1)
        
        self.entities = all_entities
        self.console.print(f"[green]✓ Found {len(all_entities)} code entities in {len(c_files)} files[/green]")
        return all_entities
    
    def analyze_entity_with_llm(self, entity: CodeEntity) -> str:
        """Use LangChain LLM to generate a description of the code entity"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a code analysis expert. Provide concise descriptions of code entities in 1-2 sentences."),
            ("human", """Analyze this C code briefly:

Type: {entity_type}
Name: {name}
Code:
```c
{code}
```

What does this {entity_type} do?""")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({
            "entity_type": entity.entity_type,
            "name": entity.name,
            "code": entity.code
        })
        return response.content
    
    def build_index(self, entities: list[CodeEntity]):
        """Build FAISS vector store using LangChain"""
        self.console.print("\n[bold]Building vector index with LangChain...[/bold]")
        self.console.print(f"[dim]Analyzing {len(entities)} code entities...[/dim]\n")
        
        # Analyze entities with LLM - show progress for each
        for i, entity in enumerate(entities, 1):
            self.console.print(f"[cyan]({i}/{len(entities)})[/cyan] Analyzing [green]{entity.entity_type}[/green]: [yellow]{entity.name}[/yellow]")
            entity.description = self.analyze_entity_with_llm(entity)
            # Show truncated description
            desc_preview = entity.description[:80].replace('\n', ' ')
            if len(entity.description) > 80:
                desc_preview += "..."
            self.console.print(f"         [dim]→ {desc_preview}[/dim]")
        
        # Build vector store
        self.console.print(f"\n[cyan]Generating embeddings...[/cyan]")
        with self.console.status("[cyan]Building FAISS index...[/cyan]"):
            documents = [entity.to_document() for entity in entities]
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        
        self.console.print(f"[green]✓ Built vector index with {len(entities)} entities[/green]")
    
    def get_relevant_context(self, query: str, k: int = 5) -> str:
        """Get relevant code context for a query"""
        if self.vectorstore is None:
            raise ValueError("Vector store not built. Call build_index first.")
        
        docs = self.vectorstore.similarity_search(query, k=k)
        context_parts = [doc.page_content for doc in docs]
        return "\n\n---\n\n".join(context_parts)
    
    def answer_question_streaming(self, question: str):
        """Answer a question with streaming output"""
        # Get relevant context
        context = self.get_relevant_context(question)
        
        # Build the prompt with history
        history_text = ""
        if self.chat_history:
            history_parts = []
            for msg in self.chat_history[-10:]:  # Last 5 exchanges
                if isinstance(msg, HumanMessage):
                    history_parts.append(f"User: {msg.content}")
                else:
                    history_parts.append(f"Assistant: {msg.content}")
            history_text = "\n".join(history_parts)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are GitMate, an expert coding assistant helping developers understand a codebase.
You have access to the following code context from the repository. Use it to answer questions accurately.

When answering:
1. Reference specific functions, files, and line numbers when relevant
2. Explain code relationships and dependencies
3. Suggest specific locations for implementing new features
4. Be concise but thorough

Code Context:
{context}

Previous Conversation:
{history}"""),
            ("human", "{question}"),
        ])
        
        chain = prompt | self.streaming_llm
        
        # Stream the response
        full_response = ""
        self.console.print("\n[bold green]GitMate:[/bold green]")
        
        with Live("", console=self.console, refresh_per_second=10) as live:
            for chunk in chain.stream({
                "context": context,
                "history": history_text,
                "question": question
            }):
                if hasattr(chunk, 'content'):
                    full_response += chunk.content
                    live.update(Markdown(full_response))
        
        # Update chat history
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=full_response))
        
        # Keep history manageable
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-20:]
        
        self.console.print()  # New line after response
    
    def search_similar(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        """Search for similar code entities"""
        if self.vectorstore is None:
            raise ValueError("Vector store not built. Call build_index first.")
        
        return self.vectorstore.similarity_search_with_score(query, k=k)
    
    def display_entities(self, entities: list[CodeEntity]):
        """Display entities in a nice table"""
        table = Table(title="Code Entities Found")
        table.add_column("Type", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("File", style="yellow")
        table.add_column("Lines", style="magenta")
        
        for entity in entities:
            table.add_row(
                entity.entity_type,
                entity.name,
                entity.file_path,
                f"{entity.start_line}-{entity.end_line}"
            )
        
        self.console.print(table)
    
    def display_search_results(self, results: list[tuple[Document, float]]):
        """Display search results in a nice format"""
        self.console.print("\n[bold]Relevant Code Entities:[/bold]")
        for i, (doc, score) in enumerate(results, 1):
            meta = doc.metadata
            self.console.print(Panel(
                f"[cyan]{meta['entity_type']}[/cyan]: [green]{meta['name']}[/green]\n"
                f"File: [yellow]{meta['file_path']}[/yellow] (lines {meta['start_line']}-{meta['end_line']})\n"
                f"Relevance Score: {1 / (1 + score):.2%}",
                title=f"Result {i}",
                border_style="dim"
            ))
    
    def chat_mode(self):
        """Interactive chat mode with streaming responses"""
        self.console.print(Panel(
            "[bold green]GitMate Chat Mode[/bold green] (Powered by LangChain)\n"
            "Ask questions about the codebase. Responses are streamed in real-time!\n\n"
            "Commands:\n"
            "  [cyan]/search <query>[/cyan] - Search for relevant code\n"
            "  [cyan]/clear[/cyan] - Clear chat history\n"
            "  [cyan]/exit[/cyan] or [cyan]/quit[/cyan] - Exit chat mode",
            title="Welcome"
        ))
        
        while True:
            try:
                question = Prompt.ask("\n[bold cyan]You[/bold cyan]")
                
                if question.lower() in ("exit", "quit", "q", "/exit", "/quit"):
                    self.console.print("[yellow]Goodbye![/yellow]")
                    break
                
                if question.lower() == "/clear":
                    self.chat_history.clear()
                    self.console.print("[green]Chat history cleared.[/green]")
                    continue
                
                if question.lower().startswith("/search "):
                    query = question[8:].strip()
                    if query:
                        results = self.search_similar(query)
                        self.display_search_results(results)
                    continue
                
                if not question.strip():
                    continue
                
                # Stream the answer
                self.answer_question_streaming(question)
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Goodbye![/yellow]")
                break


def check_ollama_connection() -> bool:
    """Test that Ollama is running by asking the LLM a simple question"""
    console.print()
    
    try:
        with console.status("[cyan]Connecting to Ollama...[/cyan]"):
            llm = ChatOllama(model=LLM_MODEL, temperature=0)
            response = llm.invoke("Reply with just 'ready'")
            if not response.content:
                raise Exception("No response from LLM")
        
        console.print(f"[green]✓ Connected to Ollama ({LLM_MODEL})[/green]\n")
        return True
        
    except Exception as e:
        console.print(f"[red]✗ Could not connect to Ollama: {e}[/red]")
        console.print(f"\n[yellow]Make sure Ollama is running and models are pulled:[/yellow]")
        console.print(f"  [cyan]ollama serve[/cyan]")
        console.print(f"  [cyan]ollama pull {LLM_MODEL}[/cyan]")
        console.print(f"  [cyan]ollama pull {EMBEDDING_MODEL}[/cyan]")
        return False


def main():
    """Main entry point"""
    console.print(Panel(
        "[bold blue]GitMate[/bold blue]: Codebase Onboarding Assistant\n"
        "Analyze GitHub repositories and get answers about the code.\n"
        "[dim]Powered by LangChain + Ollama (Streaming Enabled)[/dim]",
        title="Welcome",
        border_style="blue"
    ))
    
    # Check Ollama connection first
    if not check_ollama_connection():
        return
    
    gitmate = GitMate()
    
    # Get repository URL
    repo_url = Prompt.ask(
        "\n[bold]Enter GitHub repository URL[/bold]",
        default="https://github.com/bigsparsh/bgdb"
    )
    
    try:
        # Clone repository
        gitmate.clone_repo(repo_url)
        
        # Analyze codebase
        entities = gitmate.analyze_codebase()
        
        # Display found entities
        gitmate.display_entities(entities)
        
        # Build index
        gitmate.build_index(entities)
        
        # Start chat mode
        gitmate.chat_mode()
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise


if __name__ == "__main__":
    main()
