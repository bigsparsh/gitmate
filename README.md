<div align="center">

# GitMate

### _Your AI-Powered Guide to Understanding Any Codebase_

[![Python 3.13+](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Made with LangChain](https://img.shields.io/badge/Made%20with-LangChain-orange.svg)](https://langchain.com/)
[![Tree-sitter Powered](https://img.shields.io/badge/Powered%20by-Tree--sitter-purple.svg)](https://tree-sitter.github.io/)

_Onboarding to a new codebase shouldn't feel like deciphering ancient hieroglyphics._

[Getting Started](#-installation) • [Features](#-features) • [Usage](#-usage) • [Contributing](#-contributing)

</div>

---

## Description

**GitMate** is an intelligent code exploration tool designed to help developers quickly understand and navigate unfamiliar codebases. Whether you're joining a new team, contributing to open source, or auditing code, GitMate acts as your personal AI assistant that has already read and understood every line of code.

### The Problem It Solves

Starting with a new codebase is overwhelming:

- **Documentation is outdated or missing** - You're left guessing what functions actually do
- **Finding where to make changes is hard** - "Where do I add this feature?"
- **Understanding dependencies is complex** - "What calls this function? What does it call?"
- **Onboarding takes weeks, not hours** - Valuable time spent deciphering instead of coding

### ✨ The GitMate Solution

GitMate combines the power of **AST parsing**, **Language Server Protocol (LSP)**, and **Large Language Models** to:

1. **Parse** every function, class, struct, and variable in your codebase
2. **Analyze** code flow using LSP for accurate reference tracking and call hierarchies
3. **Describe** what each piece of code does using AI
4. **Index** everything in a semantic vector database
5. **Answer** your questions with contextually relevant, streaming responses

---

## Features

<table>
<tr>
<td width="50%">

### Intelligent Code Parsing

- **Tree-sitter AST analysis** for accurate code understanding
- Extracts functions, variables, structs, enums, and more
- Supports **C/C++**, **TypeScript/TSX**, and **JSON**
- Preserves exact line numbers and code locations

</td>
<td width="50%">

### LSP Integration

- **Reference tracking** - Find every usage of any symbol
- **Call hierarchy** - See who calls what and what calls whom
- Works with **clangd** (C/C++) and **typescript-language-server**
- Gracefully degrades if LSP unavailable

</td>
</tr>
<tr>
<td width="50%">

### LLM-Powered Analysis

- Automatic description generation for all code entities
- Context-aware Q&A with memory
- **Streaming responses** - No waiting for complete answers
- Powered by **Groq's Llama 3.3 70B** for fast inference

</td>
<td width="50%">

### Semantic Search

- **FAISS vector store** for lightning-fast retrieval
- **Ollama embeddings** with nomic-embed-text
- Finds relevant code even with vague queries
- Returns contextually similar code snippets

</td>
</tr>
<tr>
<td colspan="2">

### Interactive Chat Interface

- Beautiful **Rich TUI** with markdown rendering
- Chat history with context preservation
- Special commands: `/refs`, `/calls`, `/code`, `/help`
- "Where should I change for X feature?" → Exact function location

</td>
</tr>
</table>

---

## Installation

### Prerequisites

- **Python 3.13+**
- **Ollama** - For local embeddings ([Install Ollama](https://ollama.ai/))
- **UV** - Fast Python package manager ([Install UV](https://github.com/astral-sh/uv))

### Step 1: Clone GitMate

```bash
git clone https://github.com/bigsparsh/gitmate.git
cd gitmate/backend
```

### Step 2: Install Dependencies

```bash
# Using UV (recommended)
uv sync

# Or using pip
pip install -e .
```

### Step 3: Configure API Keys

```bash
# Set your Groq API key (get one free at https://console.groq.com)
export GROQ_API_KEY=your_groq_api_key

# Or create a .env file
echo "GROQ_API_KEY=your_groq_api_key" > .env
```

### Step 4: Pull Embedding Model

```bash
# Pull the embedding model (one-time setup)
ollama pull nomic-embed-text
```

### Optional: LSP Setup

For enhanced reference tracking and call hierarchy features:

```bash
# For C/C++ support
sudo apt install clangd                    # Ubuntu/Debian
brew install llvm                          # macOS

# For TypeScript/TSX support
npm install -g typescript-language-server typescript
```

---

## Usage

### Quick Start

```bash
# Run GitMate
python main.py
```

You'll be prompted to enter a GitHub repository URL. GitMate will:

1.  Clone the repository to `/tmp/gitmate/`
2.  Parse all source files using Tree-sitter
3.  Initialize LSP servers (if available)
4.  Analyze each entity with AI
5.  Build the vector store
6.  Enter interactive chat mode

### Screenshots Walkthrough

#### 1️ Repository Analysis

Enter any GitHub repository URL to begin analysis:

![Repository Cloning and Parsing](backend/assets/1%20final.png)

#### 2️ Code Entity Extraction

GitMate extracts all code entities with detailed information:

![Entity Extraction](backend/assets/2%20final.png)

#### 3️ LSP Enhancement

Reference tracking and call hierarchy analysis:

![LSP Enhancement](backend/assets/3%20final.png)

#### 4️ AI-Powered Analysis

Each entity receives an intelligent description:

![AI Analysis](backend/assets/4%20final%20.png)

#### 5️ Vector Store Creation

Building the semantic search index:

![Vector Store](backend/assets/5%20final.png)

#### 6️ Interactive Chat

Ask questions and get contextual answers:

![Chat Interface](backend/assets/6%20final.png)

### Chat Commands

| Command           | Description                        |
| ----------------- | ---------------------------------- |
| `/refs <name>`    | Show all references to a symbol    |
| `/calls <name>`   | Show call hierarchy for a function |
| `/code <name>`    | Display the code for an entity     |
| `/entities`       | List all extracted entities        |
| `/help`           | Show available commands            |
| `/quit` or `exit` | Exit GitMate                       |

### Example Queries

```
You: What does the main function do?
You: Where should I add input validation?
You: Explain the print_record function
You: How does the search algorithm work?
You: /refs db_init
You: /calls main
```

---

## Architecture

![GitMate Architecture](backend/assets/WhatsApp%20Image%202025-12-30%20at%209.29.27%20PM.jpeg)

---

## Project Structure

```
gitmate/
├── backend/
│   ├── main.py              # 🎯 CLI entry point & main application
│   ├── lsp_client.py        # 🔗 LSP client implementations
│   ├── pyproject.toml       # 📦 Project dependencies
│   ├── tree-sitter-docs.md  # 📚 Tree-sitter documentation
│   └── instructions.md      # 📝 Development instructions
└── screenshots/             # 📸 Usage screenshots
```

---

## Roadmap

### Coming in v2.0 (Next Release)

<table>
<tr>
<td width="40%">

#### Beautiful Web Interface

- Modern React-based frontend
- Dark/Light theme support
- Mobile-responsive design
- Real-time streaming chat UI

</td>
<td width="50%">

#### Interactive Code Tree

- AI-generated visual diagram of file structure
- Expandable nodes showing functions per file
- Click-to-navigate to any code entity
- Relationship visualization between modules

</td>
</tr>
<tr>
<td width="50%">

#### Code Insights Dashboard

- Repository statistics & metrics
- Most referenced functions
- Complexity hotspots
- Dependency graphs

</td>
<td width="50%">

#### More Language Support

- Python (tree-sitter-python)
- Java (tree-sitter-java)
- Rust (tree-sitter-rust)
- Go (tree-sitter-go)

</td>
</tr>
</table>

### Future Vision

- [ ] **VS Code Extension** - Inline code explanations
- [ ] **GitHub Action** - Auto-generate onboarding docs
- [ ] **Multi-repo Support** - Analyze entire organizations
- [ ] **Custom LLM Support** - Local models, OpenAI, Anthropic
- [ ] **Export to Markdown** - Generate documentation files

---

## Contributing

We welcome contributions from the community! Here's how you can help:

### Getting Started

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Guidelines

- Follow **PEP 8** for Python code style
- Write **docstrings** for all functions and classes
- Add **tests** for new functionality
- Update **documentation** as needed
- Keep commits **atomic** and messages **descriptive**

### Areas We Need Help

- Bug fixes and issue resolution
- Documentation improvements
- Language support additions
- Test coverage expansion
- Feature suggestions

## Credits & Acknowledgments

### Built With

- **[Tree-sitter](https://tree-sitter.github.io/)** - Incremental parsing library
- **[LangChain](https://langchain.com/)** - LLM application framework
- **[Groq](https://groq.com/)** - Ultra-fast LLM inference
- **[Ollama](https://ollama.ai/)** - Local model running
- **[Rich](https://rich.readthedocs.io/)** - Beautiful terminal formatting
- **[FAISS](https://faiss.ai/)** - Efficient similarity search

### Inspiration

- Every developer who struggled with a new codebase
- The open-source community's commitment to accessibility
- The vision of AI-augmented development

---

## Contact

<div align="center">

**Sparsh Singh** - Creator & Maintainer

[![GitHub](https://img.shields.io/badge/GitHub-bigsparsh-181717?style=for-the-badge&logo=github)](https://github.com/bigsparsh)

---

**Found a bug? Have a feature request?**

[Open an Issue](https://github.com/bigsparsh/gitmate/issues/new) • [Start a Discussion](https://github.com/bigsparsh/gitmate/discussions)

---

<sub>Made with ❤️ for developers, by developers</sub>

</div>
