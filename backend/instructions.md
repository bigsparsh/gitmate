# GitMate: An onboarding application for newbies to github codebase
- Clones the repo in /tmp/.
- Then analyze each file and classifies each function, variables and stuff into data.
- That data is analyzed by LLM to get the info for what the function is doing.
- Data is then stored in FAISS after being embedded.
- Now if user asks for questions such as "Where should I change for this feature" the output would be exactly the function they need to change.
- When in chat mode stream the model's responses. So users doesnt need to wait for the entire response.

# Tech
- UV for dependancy management
- Tree Sitter for AST (Documenation is provided in tree-sitter-docs.md file)
- Langchain
- Models:
    - **embedding:** nomic-embed-text
    - **LLM:** qwen2.5-coder:7b
- Rich for good TUI
- FAISS for vector storage

# For testing
- Use https://github.com/bigsparsh/bgdb
    - In this repo there is a main.c file, output should give me all the functions and other identifiers used in this and their info, and in chat mode I should be able to query it