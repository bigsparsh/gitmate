# GitMate: An onboarding application for newbies to github codebase

### Backend: (another application)
- Clones the repo in /tmp/.
- Then analyze each file and classifies each function, variables and stuff into data.
- **NEW: Uses LSP (Language Server Protocol) to track function references and call hierarchy**
- That data is analyzed by LLM to get the info for what the function is doing.
- Data is then stored in FAISS after being embedded.
- Now if user asks for questions such as "Where should I change for this feature" the output would be exactly the function they need to change.
- When in chat mode stream the model's responses. So users doesnt need to wait for the entire response.
- **NEW: /refs and /calls commands to explore function usage and call hierarchy**

### Frontend: (This application)
- User comes to a landing page (route: /)
- User then login's using github / google id, after that forwards to the dashboard (handled by nextauth)
- User can then create a new project on the dashboard and the dashboard has all their projects listed (dashboard route: /dashboard) (project route: /dashboard/project/<project-id>)
- User can then go into that project where they have the structure of the project (file tree, function names, refs, calls, etc [provided by the backend]). And two further options
    - Chart mode: File tree is shown in an interactive mermaid (or other) chart. Heirarchy is file name, then functions and variables inside then if functions have variables inside that will be handled as well. Also focusing over the function will show the backend generated (Langchain) description of that particular function or variable. (route: /dashboard/project/<project-id>/chart)
    - Chat Mode: Here the user can chat will the langchain agent (handled by backend) and ask questions about the github project. Output will be markdown and streamed (add library to handle markdown). (route: /dashboard/project/<project-id>/chat)

### Frontend Tech
- Next JS
- Prisma Postgres (Neon Tech)
- Axios
- Zod
- NextAuth
- TailwindCSS and ShadCN
- pnpm