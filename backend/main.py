"""
GitMate: An onboarding application for newbies to github codebase
Enhanced with LangChain for RAG and streaming responses
"""

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
import tree_sitter_json as ts_json
import tree_sitter_typescript as ts_typescript
import tree_sitter_c as tsc
from tree_sitter import Language, Parser
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.live import Live
from rich.panel import Panel
from rich.console import Console
import git
from dataclasses import dataclass, field
from pathlib import Path
import shutil
import tempfile
from dotenv import load_dotenv
from lsp_client import LSPManager, SymbolReferences, LSPReference, CallHierarchyItem
load_dotenv()


# LangChain imports

console = Console()

# Models configuration
EMBEDDING_MODEL = "nomic-embed-text"  # Local Ollama embedding model
LLM_MODEL = "llama-3.3-70b-versatile"  # Groq's fast LLM


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

    def to_document(self) -> Document:
        """Convert to LangChain Document"""
        # Determine language from file extension
        ext = Path(self.file_path).suffix.lower()
        lang_map = {'.c': 'c', '.h': 'c', '.ts': 'typescript',
                    '.tsx': 'tsx'}
        lang = lang_map.get(ext, 'text')

        # Build references section
        refs_text = ""
        if self.references:
            refs_list = [
                f"  - {ref.file_path}:{ref.line}" for ref in self.references[:10]]
            refs_text = f"\n\nUsed in ({len(
                self.references)} locations):\n" + "\n".join(refs_list)
            if len(self.references) > 10:
                refs_text += f"\n  ... and {len(self.references) - 10} more"

        # Build call hierarchy section
        calls_text = ""
        if self.incoming_calls:
            callers = [
                f"  - {c.name} ({c.file_path}:{c.line})" for c in self.incoming_calls[:5]]
            calls_text += f"\n\nCalled by:\n" + "\n".join(callers)
        if self.outgoing_calls:
            callees = [
                f"  - {c.name} ({c.file_path}:{c.line})" for c in self.outgoing_calls[:5]]
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


class GitMate:
    """Main GitMate application class with LangChain integration"""

    def __init__(self):
        self.console = console
        self.repo_path: Path = None

        # Initialize tree-sitter languages and parsers
        self.c_language = Language(tsc.language())
        self.ts_language = Language(ts_typescript.language_typescript())
        self.tsx_language = Language(ts_typescript.language_tsx())

        # Create parsers for each language
        self.parsers = {
            '.c': Parser(self.c_language),
            '.h': Parser(self.c_language),
            '.ts': Parser(self.ts_language),
            '.tsx': Parser(self.tsx_language),
        }

        # LangChain components
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.llm = ChatGroq(model=LLM_MODEL, temperature=0)
        self.streaming_llm = ChatGroq(
            model=LLM_MODEL, temperature=0, streaming=True)
        self.vectorstore: FAISS = None
        self.chat_history: list = []
        self.entities: list[CodeEntity] = []

        # LSP Manager for reference tracking
        self.lsp_manager: LSPManager = None
        self.lsp_available: bool = False

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

        self.console.print(f"[green]✓ Cloned repository to {
                           clone_path}[/green]")
        self.repo_path = clone_path
        return clone_path

    def _extract_entities_from_node(self, node, source_code: bytes, file_path: str, lang: str = 'c') -> list[CodeEntity]:
        """Recursively extract code entities from AST nodes"""
        entities = []

        if lang in ('ts', 'tsx'):
            entities.extend(self._extract_ts_entities(
                node, source_code, file_path))
        elif lang == 'json':
            entities.extend(self._extract_json_entities(
                node, source_code, file_path))
        else:
            entities.extend(self._extract_c_entities(
                node, source_code, file_path))

        return entities

    def _extract_c_entities(self, node, source_code: bytes, file_path: str) -> list[CodeEntity]:
        """Extract C language entities from AST nodes"""
        entities = []

        # Function definitions
        if node.type == "function_definition":
            declarator = node.child_by_field_name("declarator")
            name, name_col = self._get_declarator_name_and_column(declarator)
            if name:
                entities.append(CodeEntity(
                    name=name,
                    entity_type="function",
                    file_path=file_path,
                    start_line=node.start_point.row + 1,
                    end_line=node.end_point.row + 1,
                    code=source_code[node.start_byte:node.end_byte].decode(
                        "utf-8", errors="ignore"),
                    name_column=name_col
                ))

        # Struct/union definitions
        elif node.type in ("struct_specifier", "union_specifier"):
            name_node = node.child_by_field_name("name")
            if name_node:
                name = source_code[name_node.start_byte:name_node.end_byte].decode(
                    "utf-8", errors="ignore")
                entities.append(CodeEntity(
                    name=name,
                    entity_type="struct" if node.type == "struct_specifier" else "union",
                    file_path=file_path,
                    start_line=node.start_point.row + 1,
                    end_line=node.end_point.row + 1,
                    code=source_code[node.start_byte:node.end_byte].decode(
                        "utf-8", errors="ignore"),
                    name_column=name_node.start_point.column
                ))

        # Enum definitions
        elif node.type == "enum_specifier":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = source_code[name_node.start_byte:name_node.end_byte].decode(
                    "utf-8", errors="ignore")
                entities.append(CodeEntity(
                    name=name,
                    entity_type="enum",
                    file_path=file_path,
                    start_line=node.start_point.row + 1,
                    end_line=node.end_point.row + 1,
                    code=source_code[node.start_byte:node.end_byte].decode(
                        "utf-8", errors="ignore"),
                    name_column=name_node.start_point.column
                ))

        # Global variable declarations
        elif node.type == "declaration" and node.parent and node.parent.type == "translation_unit":
            declarator = node.child_by_field_name("declarator")
            name, name_col = self._get_declarator_name_and_column(declarator)
            if name:
                entities.append(CodeEntity(
                    name=name,
                    entity_type="global_variable",
                    file_path=file_path,
                    start_line=node.start_point.row + 1,
                    end_line=node.end_point.row + 1,
                    code=source_code[node.start_byte:node.end_byte].decode(
                        "utf-8", errors="ignore"),
                    name_column=name_col
                ))

        # Typedef
        elif node.type == "type_definition":
            declarator = node.child_by_field_name("declarator")
            name, name_col = self._get_declarator_name_and_column(declarator)
            if name:
                entities.append(CodeEntity(
                    name=name,
                    entity_type="typedef",
                    file_path=file_path,
                    start_line=node.start_point.row + 1,
                    end_line=node.end_point.row + 1,
                    code=source_code[node.start_byte:node.end_byte].decode(
                        "utf-8", errors="ignore"),
                    name_column=name_col
                ))

        # Preprocessor defines
        elif node.type == "preproc_def":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = source_code[name_node.start_byte:name_node.end_byte].decode(
                    "utf-8", errors="ignore")
                entities.append(CodeEntity(
                    name=name,
                    entity_type="macro",
                    file_path=file_path,
                    start_line=node.start_point.row + 1,
                    end_line=node.end_point.row + 1,
                    code=source_code[node.start_byte:node.end_byte].decode(
                        "utf-8", errors="ignore"),
                    name_column=name_node.start_point.column
                ))

        # Recurse into children
        for child in node.children:
            entities.extend(self._extract_c_entities(
                child, source_code, file_path))

        return entities

    def _extract_ts_entities(self, node, source_code: bytes, file_path: str) -> list[CodeEntity]:
        """Extract TypeScript/TSX entities from AST nodes"""
        entities = []

        # Function declarations
        if node.type == "function_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = source_code[name_node.start_byte:name_node.end_byte].decode(
                    "utf-8", errors="ignore")
                entities.append(CodeEntity(
                    name=name,
                    entity_type="function",
                    file_path=file_path,
                    start_line=node.start_point.row + 1,
                    end_line=node.end_point.row + 1,
                    code=source_code[node.start_byte:node.end_byte].decode(
                        "utf-8", errors="ignore"),
                    name_column=name_node.start_point.column
                ))

        # Arrow functions with variable declaration
        elif node.type == "lexical_declaration" or node.type == "variable_declaration":
            for child in node.children:
                if child.type == "variable_declarator":
                    name_node = child.child_by_field_name("name")
                    value_node = child.child_by_field_name("value")
                    if name_node and value_node and value_node.type == "arrow_function":
                        name = source_code[name_node.start_byte:name_node.end_byte].decode(
                            "utf-8", errors="ignore")
                        entities.append(CodeEntity(
                            name=name,
                            entity_type="arrow_function",
                            file_path=file_path,
                            start_line=node.start_point.row + 1,
                            end_line=node.end_point.row + 1,
                            code=source_code[node.start_byte:node.end_byte].decode(
                                "utf-8", errors="ignore"),
                            name_column=name_node.start_point.column
                        ))

        # Class declarations
        elif node.type == "class_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = source_code[name_node.start_byte:name_node.end_byte].decode(
                    "utf-8", errors="ignore")
                entities.append(CodeEntity(
                    name=name,
                    entity_type="class",
                    file_path=file_path,
                    start_line=node.start_point.row + 1,
                    end_line=node.end_point.row + 1,
                    code=source_code[node.start_byte:node.end_byte].decode(
                        "utf-8", errors="ignore"),
                    name_column=name_node.start_point.column
                ))

        # Interface declarations
        elif node.type == "interface_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = source_code[name_node.start_byte:name_node.end_byte].decode(
                    "utf-8", errors="ignore")
                entities.append(CodeEntity(
                    name=name,
                    entity_type="interface",
                    file_path=file_path,
                    start_line=node.start_point.row + 1,
                    end_line=node.end_point.row + 1,
                    code=source_code[node.start_byte:node.end_byte].decode(
                        "utf-8", errors="ignore"),
                    name_column=name_node.start_point.column
                ))

        # Type alias declarations
        elif node.type == "type_alias_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = source_code[name_node.start_byte:name_node.end_byte].decode(
                    "utf-8", errors="ignore")
                entities.append(CodeEntity(
                    name=name,
                    entity_type="type_alias",
                    file_path=file_path,
                    start_line=node.start_point.row + 1,
                    end_line=node.end_point.row + 1,
                    code=source_code[node.start_byte:node.end_byte].decode(
                        "utf-8", errors="ignore"),
                    name_column=name_node.start_point.column
                ))

        # Enum declarations
        elif node.type == "enum_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = source_code[name_node.start_byte:name_node.end_byte].decode(
                    "utf-8", errors="ignore")
                entities.append(CodeEntity(
                    name=name,
                    entity_type="enum",
                    file_path=file_path,
                    start_line=node.start_point.row + 1,
                    end_line=node.end_point.row + 1,
                    code=source_code[node.start_byte:node.end_byte].decode(
                        "utf-8", errors="ignore"),
                    name_column=name_node.start_point.column
                ))

        # Export statements (named exports)
        elif node.type == "export_statement":
            declaration = node.child_by_field_name("declaration")
            if declaration:
                entities.extend(self._extract_ts_entities(
                    declaration, source_code, file_path))
                return entities  # Don't recurse again

        # Recurse into children
        for child in node.children:
            entities.extend(self._extract_ts_entities(
                child, source_code, file_path))

        return entities

    def _extract_json_entities(self, node, source_code: bytes, file_path: str) -> list[CodeEntity]:
        """Extract JSON entities from AST nodes"""
        entities = []

        # For JSON, we extract top-level keys as entities
        if node.type == "document":
            for child in node.children:
                if child.type == "object":
                    entities.extend(self._extract_json_object_keys(
                        child, source_code, file_path, prefix=""))

        return entities

    def _extract_json_object_keys(self, node, source_code: bytes, file_path: str, prefix: str) -> list[CodeEntity]:
        """Extract keys from JSON objects"""
        entities = []

        for child in node.children:
            if child.type == "pair":
                key_node = child.child_by_field_name("key")
                value_node = child.child_by_field_name("value")
                if key_node:
                    key = source_code[key_node.start_byte:key_node.end_byte].decode(
                        "utf-8", errors="ignore").strip('"')
                    full_key = f"{prefix}.{key}" if prefix else key

                    entities.append(CodeEntity(
                        name=full_key,
                        entity_type="json_key",
                        file_path=file_path,
                        start_line=child.start_point.row + 1,
                        end_line=child.end_point.row + 1,
                        code=source_code[child.start_byte:child.end_byte].decode(
                            "utf-8", errors="ignore")
                    ))

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

    def _get_declarator_name_and_column(self, node) -> tuple[str | None, int]:
        """Extract the name and column from a declarator node"""
        if node is None:
            return None, 0

        if node.type == "identifier":
            name = node.text.decode(
                "utf-8", errors="ignore") if node.text else None
            return name, node.start_point.column

        if node.type in ("function_declarator", "pointer_declarator", "array_declarator"):
            declarator = node.child_by_field_name("declarator")
            return self._get_declarator_name_and_column(declarator)

        if node.type == "init_declarator":
            declarator = node.child_by_field_name("declarator")
            return self._get_declarator_name_and_column(declarator)

        return None, 0

    def analyze_file(self, file_path: Path) -> list[CodeEntity]:
        """Parse a source file and extract all code entities"""
        try:
            ext = file_path.suffix.lower()
            if ext not in self.parsers:
                return []

            with open(file_path, "rb") as f:
                source_code = f.read()

            parser = self.parsers[ext]
            tree = parser.parse(source_code)
            rel_path = str(file_path.relative_to(self.repo_path)
                           ) if self.repo_path else str(file_path)

            # Determine language type
            lang_map = {'.c': 'c', '.h': 'c', '.ts': 'ts',
                        '.tsx': 'tsx', '.json': 'json'}
            lang = lang_map.get(ext, 'c')

            entities = self._extract_entities_from_node(
                tree.root_node, source_code, rel_path, lang)
            return entities
        except Exception as e:
            self.console.print(f"[yellow]Warning: Could not parse {
                               file_path}: {e}[/yellow]")
            return []

    def analyze_codebase(self) -> list[CodeEntity]:
        """Analyze all supported files in the repository"""
        if not self.repo_path:
            raise ValueError("No repository loaded. Call clone_repo first.")

        all_entities = []
        # Collect all supported file types
        source_files = (
            list(self.repo_path.rglob("*.c")) +
            list(self.repo_path.rglob("*.h")) +
            list(self.repo_path.rglob("*.ts")) +
            list(self.repo_path.rglob("*.tsx")) +
            list(self.repo_path.rglob("*.json"))
        )

        # Filter out node_modules and other common directories to ignore
        source_files = [
            f for f in source_files if "node_modules" not in str(f)]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task(
                "Analyzing codebase...", total=len(source_files))

            for file_path in source_files:
                entities = self.analyze_file(file_path)
                all_entities.extend(entities)
                progress.update(task, advance=1)

        self.entities = all_entities
        self.console.print(f"[green]✓ Found {len(all_entities)} code entities in {
                           len(source_files)} files[/green]")
        return all_entities

    def initialize_lsp(self):
        """Initialize LSP clients for reference tracking"""
        if not self.repo_path:
            return

        self.console.print(
            "\n[bold]Initializing Language Servers for reference tracking...[/bold]")
        self.lsp_manager = LSPManager(self.repo_path)
        available = self.lsp_manager.initialize(self.console)

        if available:
            self.lsp_available = True
            self.console.print(f"[green]✓ LSP enabled for: {
                               ', '.join(available)}[/green]")
        else:
            self.console.print(
                "[yellow]No LSP servers available. Reference tracking disabled.[/yellow]")
            self.console.print(
                "[dim]Install clangd or typescript-language-server for enhanced analysis.[/dim]")

    def open_files_in_lsp(self, source_files: list[Path]):
        """Open source files in LSP servers for indexing"""
        if not self.lsp_available or not self.lsp_manager:
            return

        self.console.print("[dim]Opening files in language servers...[/dim]")
        for file_path in source_files:
            try:
                ext = file_path.suffix.lower()
                if ext in ['.c', '.h', '.ts', '.tsx']:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    rel_path = str(file_path.relative_to(self.repo_path))
                    self.lsp_manager.open_file(rel_path, content)
            except Exception:
                pass

    def enhance_entities_with_lsp(self, entities: list[CodeEntity]) -> list[CodeEntity]:
        """Enhance code entities with LSP reference and call hierarchy data"""
        if not self.lsp_available or not self.lsp_manager:
            return entities

        self.console.print(
            "\n[bold]Enhancing entities with LSP data (references & call hierarchy)...[/bold]")

        # Functions use call hierarchy, not references
        callable_types = {'function', 'arrow_function', 'method'}
        # Variables/structs use references
        referenceable_types = {'global_variable', 'struct', 'union',
                               'enum', 'typedef', 'macro', 'interface', 'type_alias', 'class'}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task(
                "Tracking references...", total=len(entities))

            for entity in entities:
                try:
                    # Use the exact column where the name is located
                    if entity.entity_type in callable_types:
                        # For functions: get call hierarchy (who calls this, what it calls)
                        refs_data = self.lsp_manager.get_symbol_references(
                            entity.file_path,
                            entity.start_line,
                            column=entity.name_column
                        )
                        entity.incoming_calls = refs_data.incoming_calls
                        entity.outgoing_calls = refs_data.outgoing_calls
                        # Don't set references for functions - use calls instead

                    elif entity.entity_type in referenceable_types:
                        # For variables/structs: get references (where they're used)
                        refs_data = self.lsp_manager.get_symbol_references(
                            entity.file_path,
                            entity.start_line,
                            column=entity.name_column
                        )
                        entity.references = refs_data.references
                        # Don't set calls for non-functions

                except Exception:
                    pass  # LSP might fail for some entities, that's ok

                progress.update(task, advance=1)

        # Summary
        total_refs = sum(len(e.references) for e in entities)
        total_callers = sum(len(e.incoming_calls) for e in entities)
        total_callees = sum(len(e.outgoing_calls) for e in entities)

        self.console.print(f"[green]✓ Found {total_refs} references, {
                           total_callers} callers, {total_callees} callees[/green]")

        return entities

    def analyze_entity_with_llm(self, entity: CodeEntity) -> str:
        """Use LangChain LLM to generate a description of the code entity"""
        # Determine language from file extension
        ext = Path(entity.file_path).suffix.lower()
        lang_names = {'.c': 'C', '.h': 'C', '.ts': 'TypeScript',
                      '.tsx': 'TSX/React', '.json': 'JSON'}
        lang_name = lang_names.get(ext, 'code')

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

        chain = prompt | self.llm
        response = chain.invoke({
            "lang_name": lang_name,
            "entity_type": entity.entity_type,
            "name": entity.name,
            "code": entity.code
        })
        return response.content

    def build_index(self, entities: list[CodeEntity]):
        """Build FAISS vector store using LangChain"""
        self.console.print(
            "\n[bold]Building vector index with LangChain...[/bold]")
        self.console.print(f"[dim]Analyzing {len(
            entities)} code entities...[/dim]\n")

        # Analyze entities with LLM - show progress for each
        for i, entity in enumerate(entities, 1):
            self.console.print(f"[cyan]({i}/{len(entities)})[/cyan] Analyzing [green]{
                               entity.entity_type}[/green]: [yellow]{entity.name}[/yellow]")
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

            # Split large documents to fit within embedding model context limit
            # nomic-embed-text has ~8k token limit, so we use 2000 chars (~500 tokens) chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )

            split_docs = []
            for doc in documents:
                # Only split if document is too large
                if len(doc.page_content) > 2000:
                    chunks = text_splitter.split_documents([doc])
                    # Preserve metadata and add chunk info
                    for i, chunk in enumerate(chunks):
                        chunk.metadata = doc.metadata.copy()
                        chunk.metadata["chunk"] = i + 1
                        chunk.metadata["total_chunks"] = len(chunks)
                    split_docs.extend(chunks)
                else:
                    split_docs.append(doc)

            self.console.print(f"[dim]Split {len(documents)} docs into {
                               len(split_docs)} chunks[/dim]")
            self.vectorstore = FAISS.from_documents(
                split_docs, self.embeddings)

        self.console.print(f"[green]✓ Built vector index with {
                           len(entities)} entities[/green]")

    def get_relevant_context(self, query: str, k: int = 8) -> str:
        """Get relevant code context for a query"""
        if self.vectorstore is None:
            raise ValueError("Vector store not built. Call build_index first.")

        # Get more results since some may be chunks of the same entity
        docs = self.vectorstore.similarity_search(query, k=k)

        # Deduplicate by entity name to avoid repeating the same code
        seen_entities = set()
        unique_parts = []
        for doc in docs:
            entity_key = (doc.metadata.get("name"),
                          doc.metadata.get("file_path"))
            if entity_key not in seen_entities:
                seen_entities.add(entity_key)
                unique_parts.append(doc.page_content)

        # Limit total context size to avoid LLM context overflow
        context = "\n\n---\n\n".join(unique_parts[:5])
        # Truncate if still too long (max ~12k chars for safety)
        if len(context) > 12000:
            context = context[:12000] + "\n... (truncated)"

        return context

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
        table.add_column("Refs", style="blue", justify="right")
        table.add_column("Callers", style="red", justify="right")

        for entity in entities:
            refs_count = str(len(entity.references)
                             ) if entity.references else "-"
            callers_count = str(len(entity.incoming_calls)
                                ) if entity.incoming_calls else "-"
            table.add_row(
                entity.entity_type,
                entity.name,
                entity.file_path,
                f"{entity.start_line}-{entity.end_line}",
                refs_count,
                callers_count
            )

        self.console.print(table)

    def display_search_results(self, results: list[tuple[Document, float]]):
        """Display search results in a nice format"""
        self.console.print("\n[bold]Relevant Code Entities:[/bold]")
        for i, (doc, score) in enumerate(results, 1):
            meta = doc.metadata
            refs_info = ""
            if meta.get('num_references', 0) > 0:
                refs_info = f"\nReferences: {meta['num_references']}"
            if meta.get('num_callers', 0) > 0:
                refs_info += f" | Callers: {meta['num_callers']}"
            if meta.get('num_callees', 0) > 0:
                refs_info += f" | Calls: {meta['num_callees']}"

            self.console.print(Panel(
                f"[cyan]{meta['entity_type']
                         }[/cyan]: [green]{meta['name']}[/green]\n"
                f"File: [yellow]{
                    meta['file_path']}[/yellow] (lines {meta['start_line']}-{meta['end_line']})\n"
                f"Relevance Score: {1 / (1 + score):.2%}{refs_info}",
                title=f"Result {i}",
                border_style="dim"
            ))

    def _find_entity_by_name(self, name: str) -> list[CodeEntity]:
        """Find entities by name (partial match)"""
        name_lower = name.lower()
        return [e for e in self.entities if name_lower in e.name.lower()]

    def _show_entity_references(self, name: str):
        """Show all references for an entity (for variables/structs, not functions)"""
        matches = self._find_entity_by_name(name)
        callable_types = {'function', 'arrow_function', 'method'}

        if not matches:
            self.console.print(
                f"[yellow]No entity found matching '{name}'[/yellow]")
            return

        for entity in matches[:5]:  # Show max 5 matches
            self.console.print(
                f"\n[bold cyan]{entity.entity_type}[/bold cyan]: [green]{entity.name}[/green]")
            self.console.print(f"[dim]Defined in {entity.file_path}:{
                               entity.start_line} (col {entity.name_column})[/dim]")

            # For functions, show callers instead of references
            if entity.entity_type in callable_types:
                self.console.print(
                    "[dim]Use /calls for functions to see call hierarchy[/dim]")
                if entity.incoming_calls:
                    self.console.print(
                        f"\n[bold]Called by ({len(entity.incoming_calls)}):[/bold]")
                    for caller in entity.incoming_calls[:10]:
                        self.console.print(
                            f"  ← {caller.name} ({caller.file_path}:{caller.line})")
                    if len(entity.incoming_calls) > 10:
                        self.console.print(f"  [dim]... and {len(
                            entity.incoming_calls) - 10} more[/dim]")
                else:
                    self.console.print("[dim]No callers found[/dim]")
            else:
                # For variables/structs/etc, show references
                if entity.references:
                    self.console.print(
                        f"\n[bold]References ({len(entity.references)}):[/bold]")
                    for ref in entity.references[:15]:
                        self.console.print(f"  • {ref.file_path}:{ref.line}")
                    if len(entity.references) > 15:
                        self.console.print(f"  [dim]... and {len(
                            entity.references) - 15} more[/dim]")
                else:
                    self.console.print(
                        "[dim]No references found (LSP may not be available)[/dim]")

    def _show_call_hierarchy(self, name: str):
        """Show call hierarchy for a function"""
        matches = self._find_entity_by_name(name)
        callable_types = {'function', 'arrow_function', 'method'}
        matches = [e for e in matches if e.entity_type in callable_types]

        if not matches:
            self.console.print(
                f"[yellow]No function found matching '{name}'[/yellow]")
            return

        for entity in matches[:3]:  # Show max 3 matches
            self.console.print(f"\n[bold magenta]Call Hierarchy for {
                               entity.name}[/bold magenta]")
            self.console.print(f"[dim]Defined in {entity.file_path}:{
                               entity.start_line}[/dim]")

            # Incoming calls (who calls this function)
            if entity.incoming_calls:
                self.console.print(f"\n[bold green]Called by ({
                                   len(entity.incoming_calls)}):[/bold green]")
                for caller in entity.incoming_calls:
                    self.console.print(
                        f"  ← [cyan]{caller.name}[/cyan] ({caller.file_path}:{caller.line})")
            else:
                self.console.print("\n[dim]No callers found[/dim]")

            # Outgoing calls (what this function calls)
            if entity.outgoing_calls:
                self.console.print(
                    f"\n[bold yellow]Calls ({len(entity.outgoing_calls)}):[/bold yellow]")
                for callee in entity.outgoing_calls:
                    self.console.print(
                        f"  → [cyan]{callee.name}[/cyan] ({callee.file_path}:{callee.line})")
            else:
                self.console.print("[dim]No outgoing calls found[/dim]")

    def chat_mode(self):
        """Interactive chat mode with streaming responses"""
        lsp_info = " + LSP" if self.lsp_available else ""
        self.console.print(Panel(
            f"[bold green]GitMate Chat Mode[/bold green] (Powered by LangChain{
                lsp_info})\n"
            "Ask questions about the codebase. Responses are streamed in real-time!\n\n"
            "Commands:\n"
            "  [cyan]/search <query>[/cyan] - Search for relevant code\n"
            "  [cyan]/refs <name>[/cyan] - Show references for a function/entity\n"
            "  [cyan]/calls <name>[/cyan] - Show call hierarchy for a function\n"
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

                if question.lower().startswith("/refs "):
                    name = question[6:].strip()
                    if name:
                        self._show_entity_references(name)
                    continue

                if question.lower().startswith("/calls "):
                    name = question[7:].strip()
                    if name:
                        self._show_call_hierarchy(name)
                    continue

                if not question.strip():
                    continue

                # Stream the answer
                self.answer_question_streaming(question)

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Goodbye![/yellow]")
                break


def check_api_connection() -> bool:
    """Test that Groq API is accessible by asking the LLM a simple question"""
    console.print()

    try:
        with console.status("[cyan]Connecting to Groq API...[/cyan]"):
            llm = ChatGroq(model=LLM_MODEL, temperature=0)
            response = llm.invoke("Reply with just 'ready'")
            if not response.content:
                raise Exception("No response from LLM")

        console.print(
            f"[green]✓ Connected to Groq API ({LLM_MODEL})[/green]\n")
        return True

    except Exception as e:
        console.print(f"[red]✗ Could not connect to Groq API: {e}[/red]")
        console.print(
            f"\n[yellow]Make sure you have set the API keys:[/yellow]")
        console.print(f"  [cyan]export GROQ_API_KEY=your_groq_api_key[/cyan]")
        console.print(
            f"  [cyan]export GOOGLE_API_KEY=your_google_api_key[/cyan]")
        return False


def main():
    """Main entry point"""
    console.print(Panel(
        "[bold blue]GitMate[/bold blue]: Codebase Onboarding Assistant\n"
        "Analyze GitHub repositories and get answers about the code.\n"
        "[dim]Powered by LangChain + Groq + Gemini (Streaming Enabled)[/dim]",
        title="Welcome",
        border_style="blue"
    ))

    # Check API connection first
    if not check_api_connection():
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

        # Initialize LSP servers for reference tracking
        gitmate.initialize_lsp()

        # Analyze codebase with tree-sitter
        entities = gitmate.analyze_codebase()

        # Open files in LSP servers for indexing
        source_files = (
            list(gitmate.repo_path.rglob("*.c")) +
            list(gitmate.repo_path.rglob("*.h")) +
            list(gitmate.repo_path.rglob("*.ts")) +
            list(gitmate.repo_path.rglob("*.tsx"))
        )
        source_files = [
            f for f in source_files if "node_modules" not in str(f)]
        gitmate.open_files_in_lsp(source_files)

        # Enhance entities with LSP data (references, call hierarchy)
        entities = gitmate.enhance_entities_with_lsp(entities)

        # Display found entities
        gitmate.display_entities(entities)

        # Build index
        gitmate.build_index(entities)

        # Start chat mode
        gitmate.chat_mode()

        # Cleanup LSP servers
        if gitmate.lsp_manager:
            gitmate.lsp_manager.shutdown_all()

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        # Cleanup on error
        if gitmate.lsp_manager:
            gitmate.lsp_manager.shutdown_all()
        raise


if __name__ == "__main__":
    main()
