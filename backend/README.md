# GitMate

An onboarding application for newbies to GitHub codebases. Analyzes repositories using Tree-sitter, LSP, and LangChain to provide intelligent code understanding and Q&A.

## Features

- 🌳 *Tree-sitter parsing*: Extracts code entities (functions, classes, structs, etc.)
- 🔗 *LSP integration*: Finds references and call hierarchies via language servers
- 🤖 *LLM-powered analysis*: Generates descriptions and answers questions
- 🔍 *Semantic search*: FAISS-based vector store for relevant code retrieval
- 💬 *Interactive chat*: Streaming responses with context-aware answers

## Project Structure


gitmate/
├── __init__.py       # Package exports
├── config.py         # Configuration settings
├── models.py         # Data models (CodeEntity, etc.)
├── parsers.py        # Tree-sitter parsing functions
├── repo.py           # Repository operations
├── llm.py            # LangChain/LLM functions
├── vectorstore.py    # FAISS vector store operations
├── chat.py           # Chat and Q&A functions
└── lsp_client.py     # LSP client implementations

main.py               # CLI entry point


## Installation

bash
# Install dependencies
uv sync

# Set API keys
export GROQ_API_KEY=your_groq_api_key


## Usage

### CLI

bash
# Run the CLI
python main.py


### As a Library

python
from gitmate import (
    clone_repository,
    analyze_codebase,
    build_vectorstore,
    ChatSession,
)
from gitmate.repo import initialize_lsp, enhance_entities_with_lsp
from gitmate.llm import analyze_entities_batch

# Clone and analyze a repository
repo_path = clone_repository("https://github.com/user/repo")
entities = analyze_codebase(repo_path)

# Optional: Enhance with LSP data
lsp_manager = initialize_lsp(repo_path)
if lsp_manager:
    entities = enhance_entities_with_lsp(entities, lsp_manager)

# Analyze with LLM and build vector store
entities = analyze_entities_batch(entities)
vectorstore = build_vectorstore(entities)

# Create chat session
chat = ChatSession(vectorstore, entities)
response = chat.ask("What does the main function do?")
print(response)


### Available Functions

#### Repository Operations
- clone_repository(url) - Clone a Git repository
- get_source_files(path) - Get all source files
- analyze_codebase(path) - Extract code entities

#### LLM Functions
- create_llm() / create_streaming_llm() - Create LLM instances
- analyze_entity(entity) - Generate entity description
- generate_response(question, context) - Generate Q&A response

#### Vector Store
- build_vectorstore(entities) - Build FAISS index
- search_similar(vectorstore, query) - Search for similar code
- get_relevant_context(vectorstore, query) - Get context string

#### Chat
- ChatSession(vectorstore, entities) - Create chat session
- answer_question(question, vectorstore) - One-shot Q&A
- find_entity_by_name(name, entities) - Find entities

## Configuration

Edit settings via gitmate.config:

python
from gitmate.config import Config, set_config

config = Config(
    embedding_model="nomic-embed-text",
    llm_model="llama-3.3-70b-versatile",
    chunk_size=2000,
)
set_config(config)


## Supported Languages

- C/C++ (via clangd LSP)
- TypeScript/TSX (via typescript-language-server)
- JSON

## Requirements

- Python 3.13+
- Ollama (for embeddings)
- Groq API key (for LLM)
- Optional: clangd, typescript-language-server (for LSP features)