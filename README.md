<div align="center">

# GitMate

### _Your AI-Powered Guide to Understanding Any Codebase_

[![Python 3.13+](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Made with LangChain](https://img.shields.io/badge/Made%20with-LangChain-orange.svg)](https://langchain.com/)
[![Tree-sitter Powered](https://img.shields.io/badge/Powered%20by-Tree--sitter-purple.svg)](https://tree-sitter.github.io/)

_Onboarding to a new codebase shouldn't feel like deciphering ancient hieroglyphics._

[Getting Started](#installation) . [Features](#features) . [Usage](#usage) . [Contributing](#contribution)

</div>
<img width="1115" height="629" alt="9 (1)" src="https://github.com/user-attachments/assets/d8496cf8-9622-4aaf-8b1f-ff2dca98e64f" />

---

## Description

**GitMate** transforms the daunting task of codebase onboarding into an intuitive, interactive experience. By fusing **deterministic static analysis** (ASTs, LSP) with **probabilistic AI reasoning** (LLMs, RAG), GitMate provides a complete semantic understanding of any repository. It allows developers to "chat" with their code, visualizes complex dependencies, and accelerates the time-to-understanding by **up to 70%**.

### The Core Problem: Cognitive Overload

Modern software repositories are complex, interconnected ecosystems. For new developers, the learning curve is steep and costly:

1.  **High Code Volume, Low Documentation:** READMEs rarely capture the intricate runtime behaviors or architectural decisions.
2.  **Invisible Dependencies:** Modifying a single function can have cascading effects that static linters miss.
3.  **Inefficient Onboarding:** Developers spend nearly **75% of their time reading code** versus writing it. The "Time-to-First-Commit" often spans weeks.
4.  **Legacy Black Boxes:** Inheriting undocumented legacy code is risky and error-prone without deep contextual understanding.

**Who struggles most?**
- **New Team Members** needing to become productive immediately.
- **Open Source Contributors** navigating massive, unfamiliar projects.
- **Maintainers** auditing legacy systems or refactoring complex modules.


### The Solution: Neuro-Symbolic Code Analysis

GitMate bridges the gap between raw code and human understanding:

1.  **Precision Parsing (The Logic)**
    *   Utilizing **Tree-sitter**, GitMate constructs a rigorous Abstract Syntax Tree (AST) of the codebase, ensuring every function, class, and variable is indexed with 100% accuracy.

2.  **Semantic Enrichment (The Knowledge)**
    *   An **LSP (Language Server Protocol)** client resolves symbol references and call hierarchies, mapping the "connectome" of the software.

3.  **AI Synthesis (The Insight)**
    *   Large Language Models (Llama 3.3 via Groq) generate human-readable explanations for every entity, stored in a **FAISS vector database** for semantic retrieval.

4.  **Interactive Exploration**
    *   A modern **Next.js 16** web dashboard allows users to query the codebase using natural language, visualize data flows, and navigate complex architectures effortlessly.

---

## Features

<table>
<tr>
<td width="50%">

### Intelligent Code Parsing

- **Tree-sitter AST analysis** for accurate, error-tolerant parsing
- Extracts functions, variables, structs, enums across files
- **Polyglot Support**: C/C++, TypeScript/TSX, JSON, and Python
- Captures exact file locations and structural context

</td>
<td width="50%">

### Architecture Awareness

- **Reference Tracking**: Instantly find symbol usage across the project
- **Call Hierarchy**: Visualize upstream callers and downstream dependencies
- **LSP Integration**: Leverages `clangd` and `typescript-language-server`
- **Smart Degradation**: Falls back to heuristic analysis if LSP is absent

</td>
</tr>
<tr>
<td width="50%">

### Neuro-Symbolic AI Core

- **Context-Aware Explanations**: Auto-generates docs for undocumented code
- **Streaming RAG**: Retrieval Augmented Generation with sub-second latency
- **High-Performance Inference**: Powered by **Groq's Llama 3.3 70B**
- **Vector Memory**: Persistent semantic index using **FAISS** & **Ollama**

</td>
<td width="50%">

### Modern Web Interface

- **Next.js 16 Dashboard**: Built with React 19 and TailwindCSS 4
- **Real-time Chat**: Streaming responses via WebSockets
- **Visualizations**: Interactive dependency graphs and file trees
- **Multi-Tenant**: Project isolation with **Postgres** & **Prisma**

</td>
</tr>
<tr>
<td colspan="2">

### Impact & Use Cases

- **"Explain This Function"**: Instant clarity on complex logic
- **"What breaks if I change this?"**: Impact analysis via dependency graphs
- **"How does Auth work?"**: Semantic search across the entire modules
- **Secure by Design**: Local embedding generation and isolated project environments

</td>
</tr>
</table>

---

## Project Gallery

Experience the power of GitMate through our modern web interface.

<div align="center">
   <img width="1199" height="675" alt="1" src="https://github.com/user-attachments/assets/b87dbc6d-0a90-4fbd-b723-1afa53f7dd57" />

  <p><em>Comprehensive Dashboard Overview</em></p>
</div>

<table>
  <tr>
    <td width="33%"><img width="1296" height="710" alt="2" src="https://github.com/user-attachments/assets/68ff0bfd-9ed5-4377-a998-19e4b4b5e51e" />
</td>
    <td width="33%"><img width="1061" height="578" alt="3" src="https://github.com/user-attachments/assets/f093f0d0-a829-4937-874b-8831e90f4483" />
</td>
    <td width="33%"><img width="1035" height="596" alt="4" src="https://github.com/user-attachments/assets/0345df2c-b5ff-4b79-9ff0-e912685d4d13" />
</td>
  </tr>
  <tr>
    <td width="33%"><img width="1128" height="623" alt="5" src="https://github.com/user-attachments/assets/6dc04919-5591-45b9-9937-67cbcc2f86f8" />
</td>
    <td width="33%"><img width="1025" height="679" alt="6" src="https://github.com/user-attachments/assets/955b2311-06fe-486d-a5c1-abcf39a0224d" />
</td>
    <td width="33%"><img width="1106" height="624" alt="7" src="https://github.com/user-attachments/assets/208790b4-6913-4f0b-a667-327590b1ac59" />
</td>
  </tr>
  <tr>
    <td width="33%"><img width="923" height="550" alt="8" src="https://github.com/user-attachments/assets/b6cfb768-b401-4b1b-9a00-675da0782a09" />
</td>
    <td width="33%"><img width="1115" height="629" alt="9 (1)" src="https://github.com/user-attachments/assets/24f3e3e3-94ee-42cc-ada6-0edf16592373" />
</td>
    <td width="33%"><img width="1199" height="675" alt="1" src="https://github.com/user-attachments/assets/f3bbc908-9a18-45a1-8b78-2a81b8f66c46" />
</td>
  </tr>
</table>

---

## INSTALLATION

### Prerequisites

- **Python 3.13+** (Backend)
- **Node.js 18+ & pnpm** (Frontend)
- **PostgreSQL** (Database)
- **Ollama** (Embeddings) - [Install Ollama](https://ollama.ai/)
- **UV** (Python Package Manager) - [Install UV](https://github.com/astral-sh/uv)

### STEP 1: CLONE THE REPOSITORY

```zsh
git clone https://github.com/bigsparsh/gitmate.git
cd gitmate
```

### STEP 2: BACKEND SETUP

```zsh
cd backend

# Install dependencies using UV
uv sync

# Configure environment
# Create .env file with GROQ_API_KEY and DATABASE_URL
```

### STEP 3: FRONTEND SETUP

```zsh
cd frontend

# Install dependencies
pnpm install

# Initialize Database
pnpm prisma generate
pnpm prisma db push
```

### STEP 4: AI & SERVICES SETUP

```zsh
# Pull the embedding model locally
ollama pull nomic-embed-text
```

### STEP 5: LSP SETUP (OPTIONAL)

For enhanced tracking and call hierarchy features:
```zsh
# For C/C++ support
sudo apt install clangd    # Ubuntu/Debian
brew install llvm          # macOS
```
---

## USAGE

### 1. Start the Backend Server

```zsh
cd backend
source .venv/bin/activate
uv run server.py
# Server runs at http://localhost:8000
```

### 2. Start the Frontend Dashboard

```zsh
cd frontend
pnpm dev
# Dashboard available at http://localhost:3000
```

### 3. Explore Your Codebase

1. Open `http://localhost:3000` in your browser.
2. Enter a GitHub Repository URL to start a new project.
3. The system will clone, parse, and analyze the repo in the background.
4. Interact with the **Chat**, **File Explorer**, or **Dependency Graph** to understand the code.

<div align="center">
  <img src="https://via.placeholder.com/800x200?text=Running+Analysis+Demo" alt="Analysis Demo"/>
</div>



---

## ARCHITECTURE

### Technical Stack & Data Flow

GitMate employs a **Hybrid Neuro-Symbolic Architecture** that combines the deterministic precision of static analysis with the probabilistic reasoning of Large Language Models.

![Architecture](https://github.com/user-attachments/assets/ff8adac3-6e70-4fd1-a5cf-cebb578b4e6d)

#### 1. The Persistence Layer (Backend)
- **FastAPI (Python 3.13):** High-performance async API server handling WebSocket streams for real-time chat.
- **Tree-sitter:** Incremental parsing library extracting precise ASTs for C++, Python, TypeScript, and Java.
- **LSP Client:** A custom Python wrapper interacting with `clangd` and `tsserver` via stdio pipes to extract Call Hierarchies and References.

#### 2. The Cognitive Layer (AI Engine)
- **Vector Store:** **FAISS** (Facebook AI Similarity Search) indexes code chunks using **Nomic Embed Text** (via Ollama) for local, privacy-focused semantic retrieval.
- **Inference:** **Groq API** running **Llama 3.3 70B** provides near-instantaneous reasoning and code explanation.
- **RAG Pipeline:** LangChain orchestrates the retrieval of semantic context + AST structure + Call Graph data to ground the LLM's responses in reality.

#### 3. The Presentation Layer (Frontend)
- **Framework:** **Next.js 16 (App Router)** & **React 19** for server-side rendering and static generation.
- **State & UI:** **TailwindCSS 4** for styling, **Mermaid.js** for rendering live dependency graphs, and **Prisma ORM** for managing user sessions and history.
- **Streaming:** Server-Sent Events (SSE) and WebSockets ensure a fluid, "typing-like" experience during AI generation.

#### 4. Data Model
- **PostgreSQL:** Stores relational data (Users, Projects, Chat History).
- **Relational Integrity:** Tracks the lineage of every analysis session and user interaction.


---

## Project Structure

```

gitmate/
├── backend/
│   ├── assets/
│   ├── instructions.md
│   ├── lsp_client.py
│   ├── main.py
│   ├── pyproject.toml
│   ├── tree-sitter-docs.md
│   └── uv.lock
└── README.md

```

---

## FUTURE VISION

<table>
<tr>
<td width="50%">

#### IDE Integration

- VS Code and JetBrains plugins
- Real-time "Copilot" style explanations in-editor
- One-click navigation from IDE to GitMate Graph View

</td>
<td width="50%">

#### Collaborative Onboarding

- Multiplayer sessions for team code reviews
- Shared annotations and "Knowledge Trails"
- Interactive "Walkthrough" recording for new hires

</td>
</tr>
<tr>
<td width="50%">

#### CI/CD Autonomous Agents

- Github Action to auto-analyze PRs
- "Risk Report" generation for every commit
- Automated architecture drift detection

</td>
<td width="50%">

#### Self-Healing Repositories

- Auto-generation of test cases for legacy code
- Automated refactoring suggestions based on AST patterns
- Vulnerability detection and patch suggestion

</td>
</tr>
</table>


---

## ROADMAP

- [x] **v0.1:** Core Tree-sitter + LSP + LLM integration (CLI)
- [x] **v0.2:** Vector Database Memory & Context Awareness
- [x] **v1.0:** Full Web Dashboard (Next.js 16) & Streaming Chat
- [ ] **v1.1:** Multi-repo support & Organization workspaces
- [ ] **v1.2:** IDE Extensions for VS Code & JetBrains
- [ ] **v2.0:** Autonomous Refactoring Agents



## CONTRIBUTION

<img width="513" height="204" alt="image" src="https://github.com/user-attachments/assets/e93bb0b1-ef95-4e6d-a1a1-6413499d8179" />

## INSPIRATION

- Every developer who struggled with a new codebase
- The open-source community's commitment to accessibility
- The vision of AI-augmented development

---

<sub>Made with ❤️ for Developers, by Developers</sub>

</div>
