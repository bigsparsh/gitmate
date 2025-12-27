# GitMate: An onboarding application for newbies to github codebase
- Clones the repo in /tmp/.
- Then analyze each file and classifies each function, variables and stuff into data.
- **NEW: Uses LSP (Language Server Protocol) to track function references and call hierarchy**
- That data is analyzed by LLM to get the info for what the function is doing.
- Data is then stored in FAISS after being embedded.
- Now if user asks for questions such as "Where should I change for this feature" the output would be exactly the function they need to change.
- When in chat mode stream the model's responses. So users doesnt need to wait for the entire response.
- **NEW: /refs and /calls commands to explore function usage and call hierarchy**

# Tech
- UV for dependancy management
- Tree Sitter for AST (Documenation is provided in tree-sitter-docs.md file)
- **LSP Clients for reference tracking:**
    - clangd for C/C++
    - typescript-language-server for TypeScript/TSX
- Langchain
- Models:
    - **embedding:** nomic-embed-text
    - **LLM:** qwen2.5-coder:7b
- Rich for good TUI
- FAISS for vector storage

# LSP Setup (Optional but recommended)
For enhanced reference tracking and call hierarchy:
```bash
# For C/C++ support
sudo apt install clangd

# For TypeScript support  
npm install -g typescript-language-server typescript
```

# For testing
- Use https://github.com/bigsparsh/bgdb
    - In this repo there is a main.c file, output should give me all the functions and other identifiers used in this and their info, and in chat mode I should be able to query it