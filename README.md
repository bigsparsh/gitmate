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

**GitMate** transforms the overwhelming experience of diving into a new repository into a guided, conversational journey. By combining code parsing and AI-powered understanding. It helps in quickly grasping what any codebase does and where to make changes.

### The Problem It Solves

Ever cloned a repository and felt lost trying to understand the codebase ? It feels like being dropped into a maze blindfolded :

1. **Documentation is a myth**- READMEs are outdated, comments are scarce and you're left reverse engineering
2. **Dependency chains are invisible**- You change one function and break three others because you couldn't see what calls what
3. **Onboarding is measured in weeks**- New Team members spend their first month just trying to understand where things are and how they work
4. **Open Source feels inaccessible**- You want to contribute but don't know where to start, what's safe to touch, or how components interact

**Who struggles most ?**
- New Developers joining teams or companies.
- Open source contributors tackling their first PR.
- Anyone inheriting legacy code without the original authors.


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

### STEP 1 : CLONE THE REPOSITORY

```zsh
# Clone via HTTPS
git clone https://github.com/bigsparsh/gitmate.git

# Or clone via SSH (if you have ssh keys configured)
git clone git@github.com:bigsparsh/gitmate.git

# Go to the project directory
cd gitmate
```

### STEP 2 : INSTALL DEPENDENCIES

```zsh
# Using pip
pip install -e

# Or Using UV (recommended)
uv sync

```

### STEP 3: CONFIGURE API KEYS

```zsh
# Set your Groq API Key
export GROQ_API_KEY=your_groq_api_key

```

### STEP 4: PULL THE EMBEDDED MODEL

```zsh
# Pull the embedding model
ollama pull nomic-embed-text
```

### STEP 5: LSP SETUP (OPTIONAL)

This step is optional it is used for enhanced tracking and call hierarchy features :
```zsh
# For C/C++ support
sudo apt install clangd    #UBUNTU/DEBIAN
sudo pacman -S clang       #Arch Linux
brew install llvm          #macOS
```
---

## Usage

### Starting GitMate

```zsh
python main.py
```
![GitMate Running](backend/assets/1%20final.png)

You'll be prompted to enter Github Repo URL. After that GitMate will:

**1. Clone the repository to `/tmp/gitmate/`**

Enter any GitHub repository URL to begin analysis. For example : ( https://github.com/bigsparsh/bgdb.git )

**2. Using Tree-sitter GitMate will parse through all the source files and it will also initialize the LSP Server (if available)**

![Extracting and initializing](backend/assets/2%20final.png)

**3. AI will start analyzing and generating each entity signatures of function, variables and macros**

![Analyzing](backend/assets/3%20final.png)

**4. Result of the AI analysis**

![AI Analysis](backend/assets/4%20final%20.png)

**5. Creating of the Vector Store**

![Vector Store](backend/assets/5%20final.png)

**6. Enter the Interactive Chat**

![Interactive chat session](backend/assets/6%20final.png)

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
