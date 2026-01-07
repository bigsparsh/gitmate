"""
GitMate CLI: Command-line interface for GitMate
A thin wrapper using the modular gitmate package
"""

import re
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.live import Live
from pathlib import Path

from gitmate import (
    # Config
    Config, get_config,
    # Models
    CodeEntity,
    # Repository operations
    clone_repository,
    get_source_files,
    analyze_codebase,
    # LSP
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


console = Console()


def display_entities(entities: list[CodeEntity]) -> None:
    """Display entities in a nice table"""
    table = Table(title="Code Entities Found")
    table.add_column("Type", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("File", style="yellow")
    table.add_column("Lines", style="magenta")
    table.add_column("Refs", style="blue", justify="right")
    table.add_column("Callers", style="red", justify="right")

    for entity in entities:
        refs_count = str(len(entity.references)) if entity.references else "-"
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

    console.print(table)


def display_search_results(results: list) -> None:
    """Display search results in a nice format"""
    console.print("\n[bold]Relevant Code Entities:[/bold]")
    for i, (doc, score) in enumerate(results, 1):
        meta = doc.metadata
        refs_info = ""
        if meta.get('num_references', 0) > 0:
            refs_info = f"\nReferences: {meta['num_references']}"
        if meta.get('num_callers', 0) > 0:
            refs_info += f" | Callers: {meta['num_callers']}"
        if meta.get('num_callees', 0) > 0:
            refs_info += f" | Calls: {meta['num_callees']}"

        console.print(Panel(
            f"[cyan]{meta['entity_type']
                     }[/cyan]: [green]{meta['name']}[/green]\n"
            f"File: [yellow]{
                meta['file_path']}[/yellow] (lines {meta['start_line']}-{meta['end_line']})\n"
            f"Relevance Score: {1 / (1 + score):.2%}{refs_info}",
            title=f"Result {i}",
            border_style="dim"
        ))


def show_entity_references(name: str, entities: list[CodeEntity]) -> None:
    """Show all references for an entity"""
    config = get_config()
    matches = find_entity_by_name(name, entities)

    if not matches:
        console.print(f"[yellow]No entity found matching '{name}'[/yellow]")
        return

    for entity in matches[:5]:
        console.print(f"\n[bold cyan]{
                      entity.entity_type}[/bold cyan]: [green]{entity.name}[/green]")
        console.print(f"[dim]Defined in {entity.file_path}:{
                      entity.start_line} (col {entity.name_column})[/dim]")

        if entity.entity_type in config.callable_types:
            console.print(
                "[dim]Use /calls for functions to see call hierarchy[/dim]")
            if entity.incoming_calls:
                console.print(
                    f"\n[bold]Called by ({len(entity.incoming_calls)}):[/bold]")
                for caller in entity.incoming_calls[:10]:
                    console.print(
                        f"  ← {caller.name} ({caller.file_path}:{caller.line})")
                if len(entity.incoming_calls) > 10:
                    console.print(f"  [dim]... and {
                                  len(entity.incoming_calls) - 10} more[/dim]")
            else:
                console.print("[dim]No callers found[/dim]")
        else:
            if entity.references:
                console.print(
                    f"\n[bold]References ({len(entity.references)}):[/bold]")
                for ref in entity.references[:15]:
                    console.print(f"  • {ref.file_path}:{ref.line}")
                if len(entity.references) > 15:
                    console.print(f"  [dim]... and {
                                  len(entity.references) - 15} more[/dim]")
            else:
                console.print(
                    "[dim]No references found (LSP may not be available)[/dim]")


def show_call_hierarchy(name: str, entities: list[CodeEntity]) -> None:
    """Show call hierarchy for a function"""
    config = get_config()
    matches = find_entity_by_name(name, entities)
    matches = [e for e in matches if e.entity_type in config.callable_types]

    if not matches:
        console.print(f"[yellow]No function found matching '{name}'[/yellow]")
        return

    for entity in matches[:3]:
        console.print(f"\n[bold magenta]Call Hierarchy for {
                      entity.name}[/bold magenta]")
        console.print(f"[dim]Defined in {entity.file_path}:{
                      entity.start_line}[/dim]")

        if entity.incoming_calls:
            console.print(f"\n[bold green]Called by ({
                          len(entity.incoming_calls)}):[/bold green]")
            for caller in entity.incoming_calls:
                console.print(
                    f"  ← [cyan]{caller.name}[/cyan] ({caller.file_path}:{caller.line})")
        else:
            console.print("\n[dim]No callers found[/dim]")

        if entity.outgoing_calls:
            console.print(
                f"\n[bold yellow]Calls ({len(entity.outgoing_calls)}):[/bold yellow]")
            for callee in entity.outgoing_calls:
                console.print(
                    f"  → [cyan]{callee.name}[/cyan] ({callee.file_path}:{callee.line})")
        else:
            console.print("[dim]No outgoing calls found[/dim]")


def chat_mode(chat_session: ChatSession, entities: list[CodeEntity], lsp_available: bool) -> None:
    """Interactive chat mode with streaming responses"""
    lsp_info = " + LSP" if lsp_available else ""
    console.print(Panel(
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
                console.print("[yellow]Goodbye![/yellow]")
                break

            if question.lower() == "/clear":
                chat_session.clear_history()
                console.print("[green]Chat history cleared.[/green]")
                continue

            if question.lower().startswith("/search "):
                query = question[8:].strip()
                if query:
                    results = chat_session.search(query)
                    display_search_results(results)
                continue

            if question.lower().startswith("/refs "):
                name = question[6:].strip()
                if name:
                    show_entity_references(name, entities)
                continue

            if question.lower().startswith("/calls "):
                name = question[7:].strip()
                if name:
                    show_call_hierarchy(name, entities)
                continue

            if not question.strip():
                continue

            # Stream the answer
            console.print("\n[bold green]GitMate:[/bold green]")
            full_response = ""
            with Live("", console=console, refresh_per_second=10) as live:
                for chunk in chat_session.ask_streaming(question):
                    full_response += chunk
                    live.update(Markdown(full_response))

            console.print()

        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye![/yellow]")
            break


def main():
    """Main entry point"""
    config = get_config()

    console.print(Panel(
        "[bold blue]GitMate[/bold blue]: Codebase Onboarding Assistant\n"
        "Analyze GitHub repositories and get answers about the code.\n"
        f"[dim]Powered by LangChain + Groq (Streaming Enabled)[/dim]",
        title="Welcome",
        border_style="blue"
    ))

    # Check API connection
    console.print()
    try:
        with console.status("[cyan]Connecting to Groq API...[/cyan]"):
            if not check_api_connection():
                raise Exception("No response from LLM")
        console.print(f"[green]✓ Connected to Groq API ({
                      config.llm_model})[/green]\n")
    except Exception as e:
        console.print(f"[red]✗ Could not connect to Groq API: {e}[/red]")
        console.print(
            f"\n[yellow]Make sure you have set the API keys:[/yellow]")
        console.print(f"  [cyan]export GROQ_API_KEY=your_groq_api_key[/cyan]")
        return

    # Get repository URL
    repo_url = Prompt.ask(
        "\n[bold]Enter GitHub repository URL[/bold]",
        default="https://github.com/bigsparsh/bgdb"
    )
    
    # Sanitize URL - remove terminal escape sequences that can appear in some terminals
    # Remove ANSI escape sequences and common escape codes like ^[[O (cursor/arrow keys)
    repo_url = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]|\x1bO[A-Z]|\^?\[O', '', repo_url).strip()
    
    # If URL was completely corrupted, use default
    if not repo_url or not repo_url.startswith(('http://', 'https://', 'git@')):
        repo_url = "https://github.com/bigsparsh/bgdb"
        console.print(f"[yellow]Using default repository: {repo_url}[/yellow]")

    lsp_manager = None
    repo_path = None

    try:
        # Clone repository
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(f"Cloning {repo_url}...", total=None)
            repo_path = clone_repository(repo_url)
        console.print(f"[green]✓ Cloned repository to {repo_path}[/green]")

        # Initialize LSP servers
        console.print(
            "\n[bold]Initializing Language Servers for reference tracking...[/bold]")
        lsp_manager = initialize_lsp(repo_path)
        lsp_available = lsp_manager is not None

        if lsp_available:
            console.print("[green]✓ LSP enabled[/green]")
        else:
            console.print(
                "[yellow]No LSP servers available. Reference tracking disabled.[/yellow]")
            console.print(
                "[dim]Install clangd or typescript-language-server for enhanced analysis.[/dim]")

        # Analyze codebase
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing codebase...", total=None)
            entities = analyze_codebase(repo_path)

        console.print(f"[green]✓ Found {len(entities)} code entities[/green]")

        # Open files in LSP and enhance entities
        if lsp_manager:
            source_files = get_source_files(repo_path)
            console.print("[dim]Opening files in language servers...[/dim]")
            open_files_in_lsp(lsp_manager, repo_path, source_files)

            console.print("\n[bold]Enhancing entities with LSP data...[/bold]")
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "Tracking references...", total=len(entities))

                def update_progress(current, total):
                    progress.update(task, completed=current, total=total)

                entities = enhance_entities_with_lsp(
                    entities, lsp_manager, update_progress)

            stats = get_entity_stats(entities)
            console.print(f"[green]✓ Found {stats['total_references']} references, "
                          f"{stats['total_callers']} callers, {stats['total_callees']} callees[/green]")

        # Display found entities
        display_entities(entities)

        # Analyze entities with LLM
        console.print("\n[bold]Building vector index with LangChain...[/bold]")
        console.print(f"[dim]Analyzing {
                      len(entities)} code entities...[/dim]\n")

        llm = create_llm()

        def on_analysis_progress(current, total, name):
            console.print(
                f"[cyan]({current}/{total})[/cyan] Analyzing: [yellow]{name}[/yellow]")

        entities = analyze_entities_batch(entities, llm, on_analysis_progress)

        # Build vector store
        console.print(f"\n[cyan]Generating embeddings...[/cyan]")
        with console.status("[cyan]Building FAISS index...[/cyan]"):
            vectorstore = build_vectorstore(entities)
        console.print(f"[green]✓ Built vector index with {
                      len(entities)} entities[/green]")

        # Create chat session and start chat mode
        chat_session = ChatSession(vectorstore, entities)
        chat_mode(chat_session, entities, lsp_available)

        # Cleanup
        if lsp_manager:
            lsp_manager.shutdown_all()

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if lsp_manager:
            lsp_manager.shutdown_all()
        raise


if __name__ == "__main__":
    main()
