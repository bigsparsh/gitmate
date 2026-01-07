"""
GitMate FastAPI Server
REST API and streaming endpoints for frontend integration
Multi-tenant support with user-specific project isolation
"""

import asyncio
import shutil
import json
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager
from datetime import datetime
import uuid

from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from gitmate import (
    Config, get_config,
    CodeEntity,
    clone_repository,
    get_source_files,
    analyze_codebase,
    LSPManager,
)
from gitmate.repo import (
    initialize_lsp,
    open_files_in_lsp,
    enhance_entities_with_lsp,
    get_entity_stats,
)
from gitmate.llm import (
    create_llm,
    create_streaming_llm,
    create_embeddings,
    analyze_entities_batch,
    check_api_connection,
)
from gitmate.vectorstore import (
    build_vectorstore,
    search_similar_raw,
    get_relevant_context,
)
from gitmate.chat import (
    ChatSession,
    find_entity_by_name,
    get_entity_references,
    get_call_hierarchy,
)


# ============== Data Directory ==============

DATA_DIR = Path("/tmp/gitmate/users")
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============== Project Model ==============

class ProjectData(BaseModel):
    """Project data model stored in JSON"""
    id: str
    name: str
    repo_url: str
    description: Optional[str] = None
    status: str = "pending"  # pending, analyzing, ready, error
    user_id: str
    created_at: str
    updated_at: str
    total_entities: int = 0
    lsp_available: bool = False
    stats: dict = {}


def get_user_dir(user_id: str) -> Path:
    """Get the directory for a specific user"""
    user_dir = DATA_DIR / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


def get_projects_file(user_id: str) -> Path:
    """Get the projects JSON file for a user"""
    return get_user_dir(user_id) / "projects.json"


def load_user_projects(user_id: str) -> list[ProjectData]:
    """Load all projects for a user"""
    projects_file = get_projects_file(user_id)
    if not projects_file.exists():
        return []
    try:
        with open(projects_file, "r") as f:
            data = json.load(f)
            return [ProjectData(**p) for p in data]
    except Exception:
        return []


def save_user_projects(user_id: str, projects: list[ProjectData]) -> None:
    """Save projects for a user"""
    projects_file = get_projects_file(user_id)
    with open(projects_file, "w") as f:
        json.dump([p.model_dump() for p in projects], f, indent=2)


def get_project_repo_path(user_id: str, project_id: str) -> Path:
    """Get the repository path for a specific project"""
    return get_user_dir(user_id) / "repos" / project_id


# ============== State Management ==============

class ProjectState:
    """State for a single project"""
    def __init__(self):
        self.repo_path: Optional[Path] = None
        self.entities: list[CodeEntity] = []
        self.lsp_manager: Optional[LSPManager] = None
        self.vectorstore = None
        self.chat_session: Optional[ChatSession] = None
        self.is_initialized: bool = False
        self.repo_url: Optional[str] = None
        self.project_id: Optional[str] = None

    def reset(self):
        """Reset all state"""
        if self.lsp_manager:
            self.lsp_manager.shutdown_all()
        self.__init__()


class AppState:
    """Global application state with multi-tenant support"""
    def __init__(self):
        # Map of user_id -> project_id -> ProjectState
        self.user_projects: dict[str, dict[str, ProjectState]] = {}
        # Legacy single-project state for backwards compatibility
        self.repo_path: Optional[Path] = None
        self.entities: list[CodeEntity] = []
        self.lsp_manager: Optional[LSPManager] = None
        self.vectorstore = None
        self.chat_session: Optional[ChatSession] = None
        self.is_initialized: bool = False
        self.repo_url: Optional[str] = None

    def get_project_state(self, user_id: str, project_id: str) -> Optional[ProjectState]:
        """Get state for a specific user's project"""
        return self.user_projects.get(user_id, {}).get(project_id)
    
    def create_project_state(self, user_id: str, project_id: str) -> ProjectState:
        """Create or get state for a user's project"""
        if user_id not in self.user_projects:
            self.user_projects[user_id] = {}
        if project_id not in self.user_projects[user_id]:
            self.user_projects[user_id][project_id] = ProjectState()
        return self.user_projects[user_id][project_id]

    def remove_project_state(self, user_id: str, project_id: str) -> None:
        """Remove state for a specific project"""
        if user_id in self.user_projects and project_id in self.user_projects[user_id]:
            state = self.user_projects[user_id][project_id]
            state.reset()
            del self.user_projects[user_id][project_id]

    def reset(self):
        """Reset all state"""
        for user_projects in self.user_projects.values():
            for state in user_projects.values():
                state.reset()
        self.user_projects = {}
        if self.lsp_manager:
            self.lsp_manager.shutdown_all()
        self.__init__()


app_state = AppState()


# ============== Lifespan ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown"""
    # Startup
    yield
    # Shutdown - cleanup LSP servers
    if app_state.lsp_manager:
        app_state.lsp_manager.shutdown_all()


# ============== FastAPI App ==============

app = FastAPI(
    title="GitMate API",
    description="API for analyzing GitHub repositories and chatting about code",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Request/Response Models ==============

class InitializeRequest(BaseModel):
    repo_url: str = Field(..., description="GitHub repository URL to analyze")
    skip_lsp: bool = Field(False, description="Skip LSP initialization for faster loading")
    skip_llm_analysis: bool = Field(False, description="Skip LLM entity analysis")


class InitializeResponse(BaseModel):
    success: bool
    message: str
    repo_path: str
    total_entities: int
    lsp_available: bool
    stats: dict


class CreateProjectRequest(BaseModel):
    repo_url: str = Field(..., description="GitHub repository URL to analyze")
    name: str = Field(..., description="Project name")
    description: Optional[str] = Field(None, description="Project description")


class ProjectResponse(BaseModel):
    id: str
    name: str
    repo_url: str = Field(alias="repoUrl")
    description: Optional[str] = None
    status: str
    user_id: str = Field(alias="userId")
    created_at: str = Field(alias="createdAt")
    updated_at: str = Field(alias="updatedAt")
    total_entities: int = Field(0, alias="totalEntities")
    stats: dict = {}

    class Config:
        populate_by_name = True


class ChatRequest(BaseModel):
    message: str = Field(..., description="User's question about the codebase")


class ChatResponse(BaseModel):
    response: str
    sources: list[dict] = []


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    limit: int = Field(5, ge=1, le=20, description="Number of results to return")


class EntityInfo(BaseModel):
    name: str
    entity_type: str
    file_path: str
    start_line: int
    end_line: int
    code: str
    description: str
    num_references: int
    num_incoming_calls: int
    num_outgoing_calls: int


class FileTreeNode(BaseModel):
    name: str
    path: str
    is_dir: bool
    children: list["FileTreeNode"] = []


class MermaidChartData(BaseModel):
    chart_type: str
    chart_code: str


# ============== Utility Functions ==============

def get_user_id(x_user_id: Optional[str] = Header(None)) -> str:
    """Extract user ID from header, default to 'anonymous' if not provided"""
    return x_user_id or "anonymous"


def ensure_initialized():
    """Ensure the repository is initialized"""
    if not app_state.is_initialized:
        raise HTTPException(
            status_code=400,
            detail="Repository not initialized. Call /api/initialize first."
        )


def ensure_project_initialized(user_id: str, project_id: str) -> ProjectState:
    """Ensure a specific project is initialized and return its state"""
    state = app_state.get_project_state(user_id, project_id)
    if not state or not state.is_initialized:
        raise HTTPException(
            status_code=400,
            detail=f"Project {project_id} not initialized. Call /api/projects/{project_id}/analyze first."
        )
    return state


def project_to_response(project: ProjectData) -> dict:
    """Convert ProjectData to response format with camelCase keys"""
    return {
        "id": project.id,
        "name": project.name,
        "repoUrl": project.repo_url,
        "description": project.description,
        "status": project.status,
        "userId": project.user_id,
        "createdAt": project.created_at,
        "updatedAt": project.updated_at,
        "totalEntities": project.total_entities,
        "stats": project.stats,
    }


def build_file_tree(repo_path: Path, current_path: Path = None) -> list[dict]:
    """Build a file tree structure from the repository"""
    if current_path is None:
        current_path = repo_path
    
    config = get_config()
    items = []
    
    try:
        for entry in sorted(current_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            # Skip hidden files and ignored directories
            if entry.name.startswith('.'):
                continue
            if any(ignored in str(entry) for ignored in config.ignore_directories):
                continue
            
            rel_path = str(entry.relative_to(repo_path))
            
            if entry.is_dir():
                items.append({
                    "name": entry.name,
                    "path": rel_path,
                    "is_dir": True,
                    "children": build_file_tree(repo_path, entry)
                })
            else:
                items.append({
                    "name": entry.name,
                    "path": rel_path,
                    "is_dir": False,
                    "children": []
                })
    except PermissionError:
        pass
    
    return items


def build_file_tree_with_entities(repo_path: Path, entities: list[CodeEntity], current_path: Path = None) -> list[dict]:
    """Build a file tree structure with analyzed entity information"""
    if current_path is None:
        current_path = repo_path
    
    config = get_config()
    
    # Build entity lookup by file path
    entities_by_file = {}
    for entity in entities:
        rel_path = str(Path(entity.file_path).relative_to(repo_path)) if entity.file_path.startswith(str(repo_path)) else entity.file_path
        if rel_path not in entities_by_file:
            entities_by_file[rel_path] = {"functions": [], "variables": []}
        
        entity_info = {
            "name": entity.name,
            "description": entity.description or "No description available",
            "startLine": entity.start_line,
            "endLine": entity.end_line,
            "parameters": [{"name": p.name, "type": p.type} for p in getattr(entity, 'parameters', [])] if hasattr(entity, 'parameters') else [],
            "returnType": getattr(entity, 'return_type', None),
            "references": [{"filePath": ref.file_path, "line": ref.line} for ref in entity.references[:5]] if entity.references else [],
        }
        
        if entity.entity_type in config.callable_types:
            entities_by_file[rel_path]["functions"].append(entity_info)
        else:
            entities_by_file[rel_path]["variables"].append({
                "name": entity.name,
                "description": entity.description or "No description available",
                "type": getattr(entity, 'variable_type', None),
                "line": entity.start_line,
            })
    
    items = []
    
    try:
        for entry in sorted(current_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            # Skip hidden files and ignored directories
            if entry.name.startswith('.'):
                continue
            if any(ignored in str(entry) for ignored in config.ignore_directories):
                continue
            
            rel_path = str(entry.relative_to(repo_path))
            
            if entry.is_dir():
                items.append({
                    "name": entry.name,
                    "path": rel_path,
                    "is_dir": True,
                    "children": build_file_tree_with_entities(repo_path, entities, entry)
                })
            else:
                file_entities = entities_by_file.get(rel_path, {"functions": [], "variables": []})
                items.append({
                    "name": entry.name,
                    "path": rel_path,
                    "is_dir": False,
                    "children": [],
                    "functions": file_entities["functions"],
                    "variables": file_entities["variables"],
                })
    except PermissionError:
        pass
    
    return items


def generate_mermaid_call_graph(entities: list[CodeEntity], focus_entity: str = None) -> str:
    """Generate a Mermaid flowchart showing call relationships"""
    config = get_config()
    
    # Filter to callable types only
    functions = [e for e in entities if e.entity_type in config.callable_types]
    
    if focus_entity:
        # Focus on a specific entity and its direct connections
        focus_matches = [e for e in functions if focus_entity.lower() in e.name.lower()]
        if not focus_matches:
            return "graph TD\n    A[No matching function found]"
        
        relevant = set()
        for e in focus_matches:
            relevant.add(e.name)
            for c in e.incoming_calls:
                relevant.add(c.name)
            for c in e.outgoing_calls:
                relevant.add(c.name)
        
        functions = [e for e in functions if e.name in relevant]
    
    lines = ["graph TD"]
    seen_edges = set()
    
    for entity in functions[:50]:  # Limit to prevent huge charts
        node_id = entity.name.replace(" ", "_").replace("-", "_")
        lines.append(f"    {node_id}[{entity.name}]")
        
        for callee in entity.outgoing_calls[:10]:
            callee_id = callee.name.replace(" ", "_").replace("-", "_")
            edge = f"{node_id} --> {callee_id}"
            if edge not in seen_edges:
                lines.append(f"    {edge}")
                seen_edges.add(edge)
    
    return "\n".join(lines)


def generate_mermaid_entity_diagram(entities: list[CodeEntity]) -> str:
    """Generate a Mermaid class diagram showing entity structure"""
    lines = ["classDiagram"]
    
    # Group entities by file
    by_file = {}
    for e in entities:
        file_key = Path(e.file_path).stem
        if file_key not in by_file:
            by_file[file_key] = []
        by_file[file_key].append(e)
    
    for file_name, file_entities in list(by_file.items())[:10]:  # Limit files
        class_name = file_name.replace("-", "_").replace(".", "_")
        lines.append(f"    class {class_name} {{")
        
        for entity in file_entities[:15]:  # Limit entities per file
            symbol = "+" if entity.entity_type == "function" else "-"
            type_indicator = "()" if entity.entity_type == "function" else ""
            lines.append(f"        {symbol}{entity.name}{type_indicator}")
        
        lines.append("    }")
    
    return "\n".join(lines)


def generate_mermaid_reference_graph(entity_name: str, entities: list[CodeEntity]) -> str:
    """Generate a Mermaid graph showing references for an entity"""
    matches = find_entity_by_name(entity_name, entities)
    
    if not matches:
        return "graph TD\n    A[No matching entity found]"
    
    entity = matches[0]
    lines = ["graph LR"]
    
    node_id = entity.name.replace(" ", "_").replace("-", "_")
    lines.append(f"    {node_id}[[\"{entity.name}\"]]")
    lines.append(f"    style {node_id} fill:#f9f,stroke:#333,stroke-width:2px")
    
    # Show incoming calls (callers)
    for i, caller in enumerate(entity.incoming_calls[:10]):
        caller_id = f"caller_{i}"
        lines.append(f"    {caller_id}({caller.name}) --> {node_id}")
    
    # Show outgoing calls
    for i, callee in enumerate(entity.outgoing_calls[:10]):
        callee_id = f"callee_{i}"
        lines.append(f"    {node_id} --> {callee_id}({callee.name})")
    
    # Show references (for non-functions)
    for i, ref in enumerate(entity.references[:10]):
        ref_id = f"ref_{i}"
        file_name = Path(ref.file_path).name
        lines.append(f"    {ref_id}[{file_name}:{ref.line}] -.-> {node_id}")
    
    return "\n".join(lines)


def generate_mermaid_file_tree_diagram(
    entities: list[CodeEntity], 
    repo_path: Path
) -> tuple[str, dict]:
    """
    Generate a Mermaid flowchart showing file tree structure with entities.
    Returns (mermaid_code, node_metadata) where node_metadata contains descriptions.
    """
    config = get_config()
    
    # Build structure: file -> entities
    by_file = {}
    for e in entities:
        if e.file_path not in by_file:
            by_file[e.file_path] = {"functions": [], "variables": []}
        
        if e.entity_type in config.callable_types:
            by_file[e.file_path]["functions"].append(e)
        else:
            by_file[e.file_path]["variables"].append(e)
    
    # Build directory structure
    dir_structure = {}
    for file_path in by_file.keys():
        parts = Path(file_path).parts
        current = dir_structure
        for part in parts[:-1]:  # Directories
            if part not in current:
                current[part] = {"__files__": [], "__subdirs__": {}}
            current = current[part]["__subdirs__"]
        # Add file
        if "__files__" not in current:
            current["__files__"] = []
        current["__files__"].append(parts[-1])
    
    lines = ["graph TB"]
    node_metadata = {}  # id -> {name, type, description, entityType, filePath, etc.}
    node_counter = [0]  # Use list for mutable counter in nested function
    
    def sanitize_id(name: str) -> str:
        """Create a safe node ID"""
        return f"node_{node_counter[0]}"
    
    def add_directory(name: str, path: str, parent_id: str = None):
        node_counter[0] += 1
        node_id = sanitize_id(name)
        
        # Folder node with folder emoji
        lines.append(f"    {node_id}[\"📁 {name}\"]")
        lines.append(f"    style {node_id} fill:#3b82f6,stroke:#1e40af,color:#fff")
        
        node_metadata[node_id] = {
            "id": node_id,
            "name": name,
            "type": "directory",
            "path": path,
            "description": f"Directory: {path}",
        }
        
        if parent_id:
            lines.append(f"    {parent_id} --> {node_id}")
        
        return node_id
    
    def add_file(name: str, path: str, parent_id: str, file_entities: dict):
        node_counter[0] += 1
        node_id = sanitize_id(name)
        
        func_count = len(file_entities.get("functions", []))
        var_count = len(file_entities.get("variables", []))
        
        # File node
        lines.append(f"    {node_id}[\"📄 {name}\"]")
        lines.append(f"    style {node_id} fill:#6b7280,stroke:#374151,color:#fff")
        
        # Build description from entities
        descriptions = []
        for fn in file_entities.get("functions", [])[:5]:
            descriptions.append(f"• {fn.name}(): {fn.description or 'No description'}")
        for var in file_entities.get("variables", [])[:5]:
            descriptions.append(f"• {var.name}: {var.description or 'No description'}")
        
        node_metadata[node_id] = {
            "id": node_id,
            "name": name,
            "type": "file",
            "path": path,
            "description": "\n".join(descriptions) if descriptions else f"File: {name}",
            "functionCount": func_count,
            "variableCount": var_count,
        }
        
        if parent_id:
            lines.append(f"    {parent_id} --> {node_id}")
        
        # Add function nodes
        for fn in file_entities.get("functions", [])[:8]:
            node_counter[0] += 1
            fn_id = sanitize_id(fn.name)
            
            lines.append(f"    {fn_id}([\"⚙️ {fn.name}()\"])")
            lines.append(f"    style {fn_id} fill:#a855f7,stroke:#7e22ce,color:#fff")
            lines.append(f"    {node_id} --> {fn_id}")
            
            node_metadata[fn_id] = {
                "id": fn_id,
                "name": fn.name,
                "type": "function",
                "path": path,
                "description": fn.description or "No description available",
                "startLine": fn.start_line,
                "endLine": fn.end_line,
                "references": len(fn.references),
                "calls": len(fn.outgoing_calls),
                "calledBy": len(fn.incoming_calls),
            }
        
        # Add variable nodes (limited)
        for var in file_entities.get("variables", [])[:5]:
            node_counter[0] += 1
            var_id = sanitize_id(var.name)
            
            lines.append(f"    {var_id}{{\"📊 {var.name}\"}}")
            lines.append(f"    style {var_id} fill:#22c55e,stroke:#15803d,color:#fff")
            lines.append(f"    {node_id} --> {var_id}")
            
            node_metadata[var_id] = {
                "id": var_id,
                "name": var.name,
                "type": "variable",
                "path": path,
                "description": var.description or "No description available",
                "startLine": var.start_line,
                "endLine": var.end_line,
                "references": len(var.references),
            }
        
        return node_id
    
    # Process files directly (group by top-level directory for cleaner visualization)
    files_by_dir = {}
    for file_path, file_entities in by_file.items():
        parts = Path(file_path).parts
        if len(parts) > 1:
            top_dir = parts[0]
        else:
            top_dir = "root"
        
        if top_dir not in files_by_dir:
            files_by_dir[top_dir] = {}
        files_by_dir[top_dir][file_path] = file_entities
    
    # Add root node
    node_counter[0] += 1
    root_id = "root"
    lines.append(f"    {root_id}[\"📦 Project\"]")
    lines.append(f"    style {root_id} fill:#0ea5e9,stroke:#0369a1,color:#fff")
    node_metadata[root_id] = {
        "id": root_id,
        "name": "Project",
        "type": "root",
        "description": f"Project root with {len(by_file)} analyzed files",
    }
    
    # Add directories and files
    for dir_name, dir_files in sorted(files_by_dir.items()):
        if dir_name == "root":
            parent_for_files = root_id
        else:
            dir_id = add_directory(dir_name, dir_name, root_id)
            parent_for_files = dir_id
        
        for file_path, file_entities in sorted(dir_files.items()):
            file_name = Path(file_path).name
            add_file(file_name, file_path, parent_for_files, file_entities)
    
    return "\n".join(lines), node_metadata


# ============== API Endpoints ==============

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "initialized": app_state.is_initialized,
        "repo_url": app_state.repo_url,
    }


# ============== Project Management Endpoints ==============

@app.get("/api/projects")
async def list_projects(x_user_id: Optional[str] = Header(None)):
    """List all projects for a user"""
    user_id = get_user_id(x_user_id)
    projects = load_user_projects(user_id)
    return [project_to_response(p) for p in projects]


@app.post("/api/projects")
async def create_project(request: CreateProjectRequest, x_user_id: Optional[str] = Header(None)):
    """Create a new project (clone repo but don't analyze yet)"""
    user_id = get_user_id(x_user_id)
    
    # Generate project ID
    project_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    
    # Create project data
    project = ProjectData(
        id=project_id,
        name=request.name,
        repo_url=request.repo_url,
        description=request.description,
        status="pending",
        user_id=user_id,
        created_at=now,
        updated_at=now,
    )
    
    # Save to user's projects
    projects = load_user_projects(user_id)
    projects.insert(0, project)  # Add to beginning
    save_user_projects(user_id, projects)
    
    return project_to_response(project)


@app.get("/api/projects/{project_id}")
async def get_project(project_id: str, x_user_id: Optional[str] = Header(None)):
    """Get a specific project"""
    user_id = get_user_id(x_user_id)
    projects = load_user_projects(user_id)
    
    for project in projects:
        if project.id == project_id:
            return project_to_response(project)
    
    raise HTTPException(status_code=404, detail=f"Project {project_id} not found")


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str, x_user_id: Optional[str] = Header(None)):
    """Delete a project and its data"""
    user_id = get_user_id(x_user_id)
    projects = load_user_projects(user_id)
    
    # Find and remove project
    found = False
    for i, project in enumerate(projects):
        if project.id == project_id:
            projects.pop(i)
            found = True
            break
    
    if not found:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    
    # Save updated projects list
    save_user_projects(user_id, projects)
    
    # Remove project state from memory
    app_state.remove_project_state(user_id, project_id)
    
    # Remove repo directory
    repo_path = get_project_repo_path(user_id, project_id)
    if repo_path.exists():
        shutil.rmtree(repo_path)
    
    return {"message": f"Project {project_id} deleted"}


@app.post("/api/projects/{project_id}/analyze")
async def analyze_project(
    project_id: str, 
    skip_lsp: bool = Query(False, description="Skip LSP initialization"),
    skip_llm_analysis: bool = Query(False, description="Skip LLM analysis"),
    x_user_id: Optional[str] = Header(None)
):
    """
    Analyze a project: clone repo, extract entities, build vector store.
    This initializes the project for chat and exploration.
    """
    user_id = get_user_id(x_user_id)
    projects = load_user_projects(user_id)
    
    # Find project
    project = None
    project_idx = None
    for i, p in enumerate(projects):
        if p.id == project_id:
            project = p
            project_idx = i
            break
    
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    
    try:
        # Update status to analyzing
        project.status = "analyzing"
        project.updated_at = datetime.utcnow().isoformat()
        projects[project_idx] = project
        save_user_projects(user_id, projects)
        
        # Check API connection
        if not skip_llm_analysis:
            if not check_api_connection():
                raise HTTPException(status_code=503, detail="Could not connect to LLM API")
        
        # Clone repository to user-specific path
        repo_path = get_project_repo_path(user_id, project_id)
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Clean up if exists
        if repo_path.exists():
            shutil.rmtree(repo_path)
        
        import git
        git.Repo.clone_from(project.repo_url, repo_path)
        
        # Create project state
        state = app_state.create_project_state(user_id, project_id)
        state.repo_path = repo_path
        state.repo_url = project.repo_url
        state.project_id = project_id
        
        # Initialize LSP (optional)
        lsp_available = False
        if not skip_lsp:
            state.lsp_manager = initialize_lsp(repo_path)
            lsp_available = state.lsp_manager is not None
        
        # Analyze codebase
        entities = analyze_codebase(repo_path)
        
        # Open files in LSP and enhance entities
        if state.lsp_manager:
            source_files = get_source_files(repo_path)
            open_files_in_lsp(state.lsp_manager, repo_path, source_files)
            entities = enhance_entities_with_lsp(entities, state.lsp_manager)
        
        # Analyze entities with LLM (optional)
        if not skip_llm_analysis:
            llm = create_llm()
            entities = analyze_entities_batch(entities, llm)
            
            # Build vector store
            state.vectorstore = build_vectorstore(entities)
            
            # Create chat session
            state.chat_session = ChatSession(state.vectorstore, entities)
        
        state.entities = entities
        state.is_initialized = True
        
        stats = get_entity_stats(entities)
        
        # Update project data
        project.status = "ready"
        project.total_entities = len(entities)
        project.lsp_available = lsp_available
        project.stats = stats
        project.updated_at = datetime.utcnow().isoformat()
        projects[project_idx] = project
        save_user_projects(user_id, projects)
        
        return {
            "success": True,
            "message": f"Project analyzed successfully with {len(entities)} entities",
            "project": project_to_response(project),
        }
        
    except Exception as e:
        # Update status to error
        project.status = "error"
        project.updated_at = datetime.utcnow().isoformat()
        projects[project_idx] = project
        save_user_projects(user_id, projects)
        
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/projects/{project_id}/file-tree")
async def get_project_file_tree(project_id: str, x_user_id: Optional[str] = Header(None)):
    """Get the file tree for a specific project with analyzed entity descriptions"""
    user_id = get_user_id(x_user_id)
    state = ensure_project_initialized(user_id, project_id)
    
    # Use enhanced tree with entity information
    tree = build_file_tree_with_entities(state.repo_path, state.entities)
    
    return {
        "repo_path": str(state.repo_path),
        "tree": tree,
    }


@app.get("/api/projects/{project_id}/charts/structure-diagram")
async def get_project_structure_diagram(project_id: str, x_user_id: Optional[str] = Header(None)):
    """
    Get Mermaid diagram showing file tree structure with analyzed entities.
    Returns the chart code and metadata for each node (for tooltips/hover).
    """
    user_id = get_user_id(x_user_id)
    state = ensure_project_initialized(user_id, project_id)
    
    chart_code, node_metadata = generate_mermaid_file_tree_diagram(
        state.entities, 
        state.repo_path
    )
    
    return {
        "chart_type": "flowchart",
        "chart_code": chart_code,
        "node_metadata": node_metadata,
    }


@app.get("/api/projects/{project_id}/entities")
async def get_project_entities(
    project_id: str,
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    file_path: Optional[str] = Query(None, description="Filter by file path"),
    search: Optional[str] = Query(None, description="Search by name"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    x_user_id: Optional[str] = Header(None)
):
    """Get all code entities for a project with optional filtering"""
    user_id = get_user_id(x_user_id)
    state = ensure_project_initialized(user_id, project_id)
    
    entities = state.entities
    
    # Apply filters
    if entity_type:
        entities = [e for e in entities if e.entity_type == entity_type]
    
    if file_path:
        entities = [e for e in entities if file_path in e.file_path]
    
    if search:
        entities = find_entity_by_name(search, entities)
    
    total = len(entities)
    entities = entities[offset:offset + limit]
    
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "entities": [e.to_dict() for e in entities],
    }


@app.get("/api/projects/{project_id}/stats")
async def get_project_stats(project_id: str, x_user_id: Optional[str] = Header(None)):
    """Get statistics for a project"""
    user_id = get_user_id(x_user_id)
    state = ensure_project_initialized(user_id, project_id)
    
    stats = get_entity_stats(state.entities)
    
    # Add file stats
    source_files = get_source_files(state.repo_path)
    stats["total_files"] = len(source_files)
    
    # Count by file extension
    by_extension = {}
    for f in source_files:
        ext = f.suffix.lower()
        by_extension[ext] = by_extension.get(ext, 0) + 1
    stats["files_by_extension"] = by_extension
    
    # Top entities by references
    entities_with_refs = [
        (e.name, e.entity_type, len(e.references) + len(e.incoming_calls))
        for e in state.entities
    ]
    entities_with_refs.sort(key=lambda x: x[2], reverse=True)
    stats["top_referenced"] = [
        {"name": name, "type": etype, "count": count}
        for name, etype, count in entities_with_refs[:10]
    ]
    
    return stats


@app.post("/api/projects/{project_id}/chat")
async def project_chat_stream(
    project_id: str, 
    request: ChatRequest, 
    x_user_id: Optional[str] = Header(None)
):
    """Send a chat message and get a streaming response for a project"""
    user_id = get_user_id(x_user_id)
    state = ensure_project_initialized(user_id, project_id)
    
    if not state.chat_session:
        raise HTTPException(
            status_code=400,
            detail="Chat not available. Analyze the project first."
        )
    
    async def generate():
        """Generator for SSE streaming"""
        try:
            for chunk in state.chat_session.ask_streaming(request.message):
                yield chunk
        except Exception as e:
            yield f"[ERROR] {str(e)}"
    
    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/api/projects/{project_id}/chat/history")
async def get_project_chat_history(project_id: str, x_user_id: Optional[str] = Header(None)):
    """Get chat history for a project"""
    user_id = get_user_id(x_user_id)
    state = app_state.get_project_state(user_id, project_id)
    
    if not state or not state.chat_session:
        return []
    
    # Convert chat history to response format
    history = []
    for msg in state.chat_session.chat_history:
        history.append({
            "id": str(len(history)),
            "role": msg["role"],
            "content": msg["content"],
        })
    
    return history


@app.post("/api/projects/{project_id}/chat/clear")
async def clear_project_chat_history(project_id: str, x_user_id: Optional[str] = Header(None)):
    """Clear chat history for a project"""
    user_id = get_user_id(x_user_id)
    state = app_state.get_project_state(user_id, project_id)
    
    if state and state.chat_session:
        state.chat_session.clear_history()
    
    return {"message": "Chat history cleared"}


@app.get("/api/projects/{project_id}/charts/call-graph")
async def get_project_call_graph(
    project_id: str,
    focus: Optional[str] = Query(None, description="Focus on a specific function"),
    x_user_id: Optional[str] = Header(None)
):
    """Get Mermaid call graph chart for a project"""
    user_id = get_user_id(x_user_id)
    state = ensure_project_initialized(user_id, project_id)
    
    chart_code = generate_mermaid_call_graph(state.entities, focus)
    
    return MermaidChartData(
        chart_type="flowchart",
        chart_code=chart_code,
    )


@app.get("/api/projects/{project_id}/charts/entity-diagram")
async def get_project_entity_diagram(project_id: str, x_user_id: Optional[str] = Header(None)):
    """Get Mermaid entity diagram for a project"""
    user_id = get_user_id(x_user_id)
    state = ensure_project_initialized(user_id, project_id)
    
    chart_code = generate_mermaid_entity_diagram(state.entities)
    
    return MermaidChartData(
        chart_type="classDiagram",
        chart_code=chart_code,
    )


@app.get("/api/projects/{project_id}/refs/{entity_name}")
async def get_project_entity_refs(
    project_id: str, 
    entity_name: str, 
    max_matches: int = 5,
    x_user_id: Optional[str] = Header(None)
):
    """Get references for an entity in a project"""
    user_id = get_user_id(x_user_id)
    state = ensure_project_initialized(user_id, project_id)
    
    references = get_entity_references(entity_name, state.entities, max_matches)
    
    if not references:
        raise HTTPException(status_code=404, detail=f"No entity found matching '{entity_name}'")
    
    return {"entity_name": entity_name, "results": references}


@app.get("/api/projects/{project_id}/calls/{function_name}")
async def get_project_call_hierarchy(
    project_id: str, 
    function_name: str, 
    max_matches: int = 3,
    x_user_id: Optional[str] = Header(None)
):
    """Get call hierarchy for a function in a project"""
    user_id = get_user_id(x_user_id)
    state = ensure_project_initialized(user_id, project_id)
    
    hierarchy = get_call_hierarchy(function_name, state.entities, max_matches)
    
    if not hierarchy:
        raise HTTPException(status_code=404, detail=f"No function found matching '{function_name}'")
    
    return {"function_name": function_name, "results": hierarchy}


@app.get("/api/projects/{project_id}/file/{file_path:path}")
async def get_project_file_content(
    project_id: str, 
    file_path: str, 
    x_user_id: Optional[str] = Header(None)
):
    """Get content of a file in a project"""
    user_id = get_user_id(x_user_id)
    state = ensure_project_initialized(user_id, project_id)
    
    full_path = state.repo_path / file_path
    
    if not full_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    
    try:
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read file: {e}")
    
    # Get entities in this file
    file_entities = [e for e in state.entities if e.file_path == file_path]
    
    return {
        "file_path": file_path,
        "content": content,
        "entities": [e.to_dict() for e in file_entities],
    }


# ============== Legacy API Endpoints (backwards compatibility) ==============
async def initialize_repository(request: InitializeRequest):
    """
    Initialize the server with a GitHub repository.
    Clones the repo, analyzes code, and builds the vector store.
    """
    try:
        # Reset previous state
        app_state.reset()
        
        # Check API connection
        if not request.skip_llm_analysis:
            if not check_api_connection():
                raise HTTPException(status_code=503, detail="Could not connect to LLM API")
        
        # Clone repository
        repo_path = clone_repository(request.repo_url)
        app_state.repo_path = repo_path
        app_state.repo_url = request.repo_url
        
        # Initialize LSP (optional)
        lsp_available = False
        if not request.skip_lsp:
            app_state.lsp_manager = initialize_lsp(repo_path)
            lsp_available = app_state.lsp_manager is not None
        
        # Analyze codebase
        entities = analyze_codebase(repo_path)
        
        # Open files in LSP and enhance entities
        if app_state.lsp_manager:
            source_files = get_source_files(repo_path)
            open_files_in_lsp(app_state.lsp_manager, repo_path, source_files)
            entities = enhance_entities_with_lsp(entities, app_state.lsp_manager)
        
        # Analyze entities with LLM (optional)
        if not request.skip_llm_analysis:
            llm = create_llm()
            entities = analyze_entities_batch(entities, llm)
            
            # Build vector store
            app_state.vectorstore = build_vectorstore(entities)
            
            # Create chat session
            app_state.chat_session = ChatSession(app_state.vectorstore, entities)
        
        app_state.entities = entities
        app_state.is_initialized = True
        
        stats = get_entity_stats(entities)
        
        return InitializeResponse(
            success=True,
            message=f"Repository initialized successfully with {len(entities)} entities",
            repo_path=str(repo_path),
            total_entities=len(entities),
            lsp_available=lsp_available,
            stats=stats,
        )
        
    except Exception as e:
        app_state.reset()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/file-tree")
async def get_file_tree():
    """
    Get the file tree structure of the repository.
    """
    ensure_initialized()
    
    tree = build_file_tree(app_state.repo_path)
    
    return {
        "repo_path": str(app_state.repo_path),
        "tree": tree,
    }


@app.get("/api/entities")
async def get_all_entities(
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    file_path: Optional[str] = Query(None, description="Filter by file path"),
    search: Optional[str] = Query(None, description="Search by name"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
):
    """
    Get all code entities with optional filtering.
    Returns functions, variables, structs, etc. with their metadata.
    """
    ensure_initialized()
    
    entities = app_state.entities
    
    # Apply filters
    if entity_type:
        entities = [e for e in entities if e.entity_type == entity_type]
    
    if file_path:
        entities = [e for e in entities if file_path in e.file_path]
    
    if search:
        entities = find_entity_by_name(search, entities)
    
    total = len(entities)
    entities = entities[offset:offset + limit]
    
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "entities": [e.to_dict() for e in entities],
    }


@app.get("/api/entities/{entity_name}")
async def get_entity_detail(entity_name: str):
    """
    Get detailed information about a specific entity including code,
    references, and call hierarchy.
    """
    ensure_initialized()
    
    matches = find_entity_by_name(entity_name, app_state.entities)
    
    if not matches:
        raise HTTPException(status_code=404, detail=f"Entity '{entity_name}' not found")
    
    results = []
    for entity in matches[:10]:
        result = entity.to_dict()
        result["code"] = entity.code
        
        # Add reference details
        result["references"] = [
            {"file_path": r.file_path, "line": r.line}
            for r in entity.references[:50]
        ]
        
        # Add call hierarchy
        result["incoming_calls"] = [
            {"name": c.name, "kind": c.kind, "file_path": c.file_path, "line": c.line}
            for c in entity.incoming_calls
        ]
        result["outgoing_calls"] = [
            {"name": c.name, "kind": c.kind, "file_path": c.file_path, "line": c.line}
            for c in entity.outgoing_calls
        ]
        
        results.append(result)
    
    return {"matches": results}


@app.get("/api/repo-structure")
async def get_repo_structure():
    """
    Get the complete repository structure including all entities,
    their relationships (calls, references), grouped by file.
    """
    ensure_initialized()
    
    config = get_config()
    
    # Group entities by file
    by_file = {}
    for entity in app_state.entities:
        if entity.file_path not in by_file:
            by_file[entity.file_path] = {
                "file_path": entity.file_path,
                "functions": [],
                "variables": [],
                "structs": [],
                "other": [],
            }
        
        entity_data = {
            "name": entity.name,
            "start_line": entity.start_line,
            "end_line": entity.end_line,
            "description": entity.description,
        }
        
        if entity.entity_type in config.callable_types:
            entity_data["incoming_calls"] = [
                {"name": c.name, "file_path": c.file_path, "line": c.line}
                for c in entity.incoming_calls
            ]
            entity_data["outgoing_calls"] = [
                {"name": c.name, "file_path": c.file_path, "line": c.line}
                for c in entity.outgoing_calls
            ]
            by_file[entity.file_path]["functions"].append(entity_data)
        elif entity.entity_type == "variable":
            entity_data["references"] = [
                {"file_path": r.file_path, "line": r.line}
                for r in entity.references
            ]
            by_file[entity.file_path]["variables"].append(entity_data)
        elif entity.entity_type in ["struct", "class", "interface", "type"]:
            entity_data["references"] = [
                {"file_path": r.file_path, "line": r.line}
                for r in entity.references
            ]
            by_file[entity.file_path]["structs"].append(entity_data)
        else:
            by_file[entity.file_path]["other"].append(entity_data)
    
    stats = get_entity_stats(app_state.entities)
    
    return {
        "repo_url": app_state.repo_url,
        "repo_path": str(app_state.repo_path),
        "stats": stats,
        "files": list(by_file.values()),
    }


@app.get("/api/references/{entity_name}")
async def get_references(entity_name: str, max_matches: int = 5):
    """
    Get all references for a specific entity (function, variable, etc.)
    """
    ensure_initialized()
    
    references = get_entity_references(entity_name, app_state.entities, max_matches)
    
    if not references:
        raise HTTPException(status_code=404, detail=f"No entity found matching '{entity_name}'")
    
    return {"entity_name": entity_name, "results": references}


@app.get("/api/call-hierarchy/{function_name}")
async def get_function_call_hierarchy(function_name: str, max_matches: int = 3):
    """
    Get the call hierarchy for a function (callers and callees).
    """
    ensure_initialized()
    
    hierarchy = get_call_hierarchy(function_name, app_state.entities, max_matches)
    
    if not hierarchy:
        raise HTTPException(status_code=404, detail=f"No function found matching '{function_name}'")
    
    return {"function_name": function_name, "results": hierarchy}


@app.post("/api/search")
async def search_code(request: SearchRequest):
    """
    Search for relevant code entities using semantic search.
    """
    ensure_initialized()
    
    if not app_state.vectorstore:
        raise HTTPException(
            status_code=400,
            detail="Vector store not available. Initialize with skip_llm_analysis=false"
        )
    
    results = search_similar_raw(app_state.vectorstore, request.query, k=request.limit)
    
    search_results = []
    for doc, score in results:
        search_results.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": score,
            "relevance": 1 / (1 + score),
        })
    
    return {"query": request.query, "results": search_results}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Send a message and get a response (non-streaming).
    """
    ensure_initialized()
    
    if not app_state.chat_session:
        raise HTTPException(
            status_code=400,
            detail="Chat not available. Initialize with skip_llm_analysis=false"
        )
    
    response = app_state.chat_session.ask(request.message)
    
    return ChatResponse(response=response)


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Send a message and get a streaming response.
    Returns Server-Sent Events (SSE) format.
    """
    ensure_initialized()
    
    if not app_state.chat_session:
        raise HTTPException(
            status_code=400,
            detail="Chat not available. Initialize with skip_llm_analysis=false"
        )
    
    async def generate():
        """Generator for SSE streaming"""
        try:
            for chunk in app_state.chat_session.ask_streaming(request.message):
                # Format as SSE
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/api/chat/clear")
async def clear_chat_history():
    """
    Clear the chat history.
    """
    ensure_initialized()
    
    if app_state.chat_session:
        app_state.chat_session.clear_history()
    
    return {"message": "Chat history cleared"}


@app.get("/api/charts/call-graph")
async def get_call_graph_chart(
    focus: Optional[str] = Query(None, description="Focus on a specific function")
):
    """
    Get Mermaid chart data for the function call graph.
    """
    ensure_initialized()
    
    chart_code = generate_mermaid_call_graph(app_state.entities, focus)
    
    return MermaidChartData(
        chart_type="flowchart",
        chart_code=chart_code,
    )


@app.get("/api/charts/entity-diagram")
async def get_entity_diagram_chart():
    """
    Get Mermaid class diagram showing entity structure by file.
    """
    ensure_initialized()
    
    chart_code = generate_mermaid_entity_diagram(app_state.entities)
    
    return MermaidChartData(
        chart_type="classDiagram",
        chart_code=chart_code,
    )


@app.get("/api/charts/reference-graph/{entity_name}")
async def get_reference_graph_chart(entity_name: str):
    """
    Get Mermaid chart showing references and calls for a specific entity.
    """
    ensure_initialized()
    
    chart_code = generate_mermaid_reference_graph(entity_name, app_state.entities)
    
    return MermaidChartData(
        chart_type="flowchart",
        chart_code=chart_code,
    )


@app.get("/api/stats")
async def get_statistics():
    """
    Get statistics about the analyzed repository.
    """
    ensure_initialized()
    
    stats = get_entity_stats(app_state.entities)
    
    # Add file stats
    source_files = get_source_files(app_state.repo_path)
    stats["total_files"] = len(source_files)
    
    # Count by file extension
    by_extension = {}
    for f in source_files:
        ext = f.suffix.lower()
        by_extension[ext] = by_extension.get(ext, 0) + 1
    stats["files_by_extension"] = by_extension
    
    # Top entities by references
    entities_with_refs = [
        (e.name, e.entity_type, len(e.references) + len(e.incoming_calls))
        for e in app_state.entities
    ]
    entities_with_refs.sort(key=lambda x: x[2], reverse=True)
    stats["top_referenced"] = [
        {"name": name, "type": etype, "count": count}
        for name, etype, count in entities_with_refs[:10]
    ]
    
    return stats


@app.get("/api/file/{file_path:path}")
async def get_file_content(file_path: str):
    """
    Get the content of a specific file along with its entities.
    """
    ensure_initialized()
    
    full_path = app_state.repo_path / file_path
    
    if not full_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    
    try:
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read file: {e}")
    
    # Get entities in this file
    file_entities = [e for e in app_state.entities if e.file_path == file_path]
    
    return {
        "file_path": file_path,
        "content": content,
        "entities": [e.to_dict() for e in file_entities],
    }


# ============== Main Entry Point ==============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
