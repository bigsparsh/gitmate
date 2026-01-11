"""
Repository operations for GitMate
Functions for cloning and analyzing repositories
"""

import git
import shutil
from pathlib import Path
from typing import Callable

from .models import CodeEntity
from .config import get_config
from .parsers import create_parsers, extract_entities_from_file
from .lsp_client import LSPManager


def clone_repository(
    repo_url: str,
    target_path: Path | None = None,
    on_progress: Callable[[str], None] | None = None
) -> Path:
    """
    Clone a Git repository to the specified path.

    Args:
        repo_url: URL of the Git repository
        target_path: Where to clone (defaults to temp directory)
        on_progress: Optional callback for progress updates

    Returns:
        Path to the cloned repository
    """
    config = get_config()
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")

    if target_path is None:
        target_path = config.temp_directory / repo_name

    # Clean up if exists
    if target_path.exists():
        shutil.rmtree(target_path)

    target_path.parent.mkdir(parents=True, exist_ok=True)

    if on_progress:
        on_progress(f"Cloning {repo_url}...")

    git.Repo.clone_from(repo_url, target_path)

    return target_path


def get_source_files(
    repo_path: Path,
    extensions: list[str] | None = None,
    ignore_dirs: list[str] | None = None
) -> list[Path]:
    """
    Get all source files from a repository.

    Args:
        repo_path: Path to the repository
        extensions: File extensions to include (defaults to config)
        ignore_dirs: Directories to ignore (defaults to config)

    Returns:
        List of file paths
    """
    config = get_config()

    if extensions is None:
        extensions = config.supported_extensions
    if ignore_dirs is None:
        ignore_dirs = config.ignore_directories

    all_files = []
    for ext in extensions:
        pattern = f"*{ext}"
        all_files.extend(repo_path.rglob(pattern))

    # Filter out ignored directories
    filtered = []
    for f in all_files:
        path_str = str(f)
        if not any(ignore_dir in path_str for ignore_dir in ignore_dirs):
            filtered.append(f)

    return filtered


def analyze_codebase(
    repo_path: Path,
    extensions: list[str] | None = None,
    ignore_dirs: list[str] | None = None,
    on_progress: Callable[[int, int], None] | None = None
) -> list[CodeEntity]:
    """
    Analyze all source files in a repository and extract code entities.

    Args:
        repo_path: Path to the repository
        extensions: File extensions to include
        ignore_dirs: Directories to ignore
        on_progress: Optional callback (current, total)

    Returns:
        List of extracted code entities
    """
    parsers = create_parsers()
    source_files = get_source_files(repo_path, extensions, ignore_dirs)

    all_entities = []
    total = len(source_files)

    for i, file_path in enumerate(source_files):
        entities = extract_entities_from_file(file_path, repo_path, parsers)
        all_entities.extend(entities)

        if on_progress:
            on_progress(i + 1, total)

    return all_entities


def initialize_lsp(repo_path: Path) -> LSPManager | None:
    """
    Initialize LSP clients for a repository.

    Args:
        repo_path: Path to the repository

    Returns:
        LSPManager instance or None if no servers available
    """
    lsp_manager = LSPManager(repo_path)
    available = lsp_manager.initialize()

    if available:
        return lsp_manager
    return None


def open_files_in_lsp(
    lsp_manager: LSPManager,
    repo_path: Path,
    source_files: list[Path] | None = None
) -> None:
    """
    Open source files in LSP servers for indexing.

    Args:
        lsp_manager: The LSP manager instance
        repo_path: Path to the repository
        source_files: Files to open (defaults to all source files)
    """
    import time
    
    if source_files is None:
        source_files = get_source_files(repo_path)

    opened_count = 0
    for file_path in source_files:
        try:
            ext = file_path.suffix.lower()
            if ext in ['.c', '.h', '.ts', '.tsx']:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                rel_path = str(file_path.relative_to(repo_path))
                lsp_manager.open_file(rel_path, content)
                opened_count += 1
        except Exception:
            pass
    
    # Wait for LSP servers to fully index all files
    # More files = more time needed
    if opened_count > 0:
        wait_time = min(5.0, 1.0 + (opened_count * 0.1))  # 1-5 seconds
        time.sleep(wait_time)


def enhance_entities_with_lsp(
    entities: list[CodeEntity],
    lsp_manager: LSPManager,
    on_progress: Callable[[int, int], None] | None = None
) -> list[CodeEntity]:
    """
    Enhance code entities with LSP reference and call hierarchy data.

    Args:
        entities: List of code entities to enhance
        lsp_manager: The LSP manager instance
        on_progress: Optional callback (current, total)

    Returns:
        Enhanced list of code entities
    """
    config = get_config()
    total = len(entities)

    for i, entity in enumerate(entities):
        try:
            if entity.entity_type in config.callable_types:
                # For functions: get call hierarchy
                refs_data = lsp_manager.get_symbol_references(
                    entity.file_path,
                    entity.start_line,
                    column=entity.name_column
                )
                entity.incoming_calls = refs_data.incoming_calls
                entity.outgoing_calls = refs_data.outgoing_calls

            elif entity.entity_type in config.referenceable_types:
                # For variables/structs: get references
                refs_data = lsp_manager.get_symbol_references(
                    entity.file_path,
                    entity.start_line,
                    column=entity.name_column
                )
                entity.references = refs_data.references

        except Exception:
            pass

        if on_progress:
            on_progress(i + 1, total)

    return entities


def get_entity_stats(entities: list[CodeEntity]) -> dict:
    """
    Get statistics about the extracted entities.

    Args:
        entities: List of code entities

    Returns:
        Dictionary with statistics
    """
    stats = {
        "total_entities": len(entities),
        "by_type": {},
        "total_references": 0,
        "total_callers": 0,
        "total_callees": 0,
    }

    for entity in entities:
        entity_type = entity.entity_type
        stats["by_type"][entity_type] = stats["by_type"].get(
            entity_type, 0) + 1
        stats["total_references"] += len(entity.references)
        stats["total_callers"] += len(entity.incoming_calls)
        stats["total_callees"] += len(entity.outgoing_calls)

    return stats
