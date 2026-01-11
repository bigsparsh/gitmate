"""
LSP Client for GitMate
Provides function call tracing and reference finding using Language Server Protocol
"""

import subprocess
import json
import os
import threading
import queue
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import quote
import time


@dataclass
class LSPReference:
    """Represents a reference/usage of a symbol"""
    file_path: str
    line: int
    column: int
    context: str = ""  # The line of code containing the reference


@dataclass
class CallHierarchyItem:
    """Represents an item in the call hierarchy"""
    name: str
    kind: str  # function, method, etc.
    file_path: str
    line: int
    column: int


@dataclass 
class SymbolReferences:
    """All references for a symbol"""
    symbol_name: str
    definition_file: str
    definition_line: int
    references: list[LSPReference] = field(default_factory=list)
    incoming_calls: list[CallHierarchyItem] = field(default_factory=list)  # Who calls this
    outgoing_calls: list[CallHierarchyItem] = field(default_factory=list)  # What this calls


class LSPClient:
    """
    Generic LSP Client that can communicate with various language servers
    Supports: clangd (C/C++), typescript-language-server (TS/TSX)
    """
    
    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
        self.process: Optional[subprocess.Popen] = None
        self.request_id = 0
        self.pending_requests: dict = {}
        self.response_queue = queue.Queue()
        self.reader_thread: Optional[threading.Thread] = None
        self.initialized = False
        self.server_capabilities = {}
        
    def _uri_from_path(self, file_path: str) -> str:
        """Convert file path to URI"""
        abs_path = str(self.workspace_path / file_path) if not os.path.isabs(file_path) else file_path
        return f"file://{quote(abs_path, safe='/:')}"
    
    def _path_from_uri(self, uri: str) -> str:
        """Convert URI to relative file path"""
        from urllib.parse import unquote
        path = unquote(uri.replace("file://", ""))
        try:
            return str(Path(path).relative_to(self.workspace_path))
        except ValueError:
            return path
    
    def _send_message(self, message: dict):
        """Send a JSON-RPC message to the language server"""
        if not self.process or self.process.poll() is not None:
            return
            
        content = json.dumps(message)
        header = f"Content-Length: {len(content)}\r\n\r\n"
        
        try:
            self.process.stdin.write(header.encode('utf-8'))
            self.process.stdin.write(content.encode('utf-8'))
            self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
    
    def _send_request(self, method: str, params: dict = None) -> int:
        """Send a request and return the request ID"""
        self.request_id += 1
        message = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
        }
        if params is not None:
            message["params"] = params
            
        self._send_message(message)
        return self.request_id
    
    def _send_notification(self, method: str, params: dict = None):
        """Send a notification (no response expected)"""
        message = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            message["params"] = params
            
        self._send_message(message)
    
    def _read_response(self) -> Optional[dict]:
        """Read a response from the language server"""
        if not self.process or self.process.poll() is not None:
            return None
            
        try:
            # Read header
            headers = {}
            while True:
                line = self.process.stdout.readline().decode('utf-8')
                if line == '\r\n' or line == '\n' or line == '':
                    break
                if ':' in line:
                    key, value = line.split(':', 1)
                    headers[key.strip().lower()] = value.strip()
            
            # Read content
            content_length = int(headers.get('content-length', 0))
            if content_length > 0:
                content = self.process.stdout.read(content_length).decode('utf-8')
                return json.loads(content)
        except (json.JSONDecodeError, ValueError, OSError):
            pass
        
        return None
    
    def _reader_loop(self):
        """Background thread to read responses"""
        while self.process and self.process.poll() is None:
            response = self._read_response()
            if response:
                self.response_queue.put(response)
    
    def _wait_for_response(self, request_id: int, timeout: float = 30.0) -> Optional[dict]:
        """Wait for a specific response"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = self.response_queue.get(timeout=0.1)
                if response.get('id') == request_id:
                    return response
                # Put back if not our response
                self.response_queue.put(response)
            except queue.Empty:
                continue
        return None


class ClangdClient(LSPClient):
    """LSP Client for clangd (C/C++)"""
    
    def start(self) -> bool:
        """Start the clangd language server"""
        try:
            # Check if clangd is available
            result = subprocess.run(['which', 'clangd'], capture_output=True)
            if result.returncode != 0:
                return False
            
            self.process = subprocess.Popen(
                ['clangd', '--background-index', '--clang-tidy=false'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.workspace_path)
            )
            
            # Start reader thread
            self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self.reader_thread.start()
            
            # Initialize
            return self._initialize()
            
        except FileNotFoundError:
            return False
    
    def _initialize(self) -> bool:
        """Send initialize request"""
        params = {
            "processId": os.getpid(),
            "rootUri": self._uri_from_path(str(self.workspace_path)),
            "capabilities": {
                "textDocument": {
                    "references": {"dynamicRegistration": False},
                    "definition": {"dynamicRegistration": False},
                    "callHierarchy": {"dynamicRegistration": False},
                }
            },
            "workspaceFolders": [
                {"uri": self._uri_from_path(str(self.workspace_path)), "name": self.workspace_path.name}
            ]
        }
        
        request_id = self._send_request("initialize", params)
        response = self._wait_for_response(request_id, timeout=10.0)
        
        if response and 'result' in response:
            self.server_capabilities = response['result'].get('capabilities', {})
            self._send_notification("initialized")
            self.initialized = True
            return True
        
        return False
    
    def open_file(self, file_path: str, content: str):
        """Notify the server that a file is open"""
        self._send_notification("textDocument/didOpen", {
            "textDocument": {
                "uri": self._uri_from_path(file_path),
                "languageId": "c",
                "version": 1,
                "text": content
            }
        })
        # Small delay per file, batch wait happens after all files opened
        time.sleep(0.1)
    
    def find_references(self, file_path: str, line: int, column: int) -> list[LSPReference]:
        """Find all references to a symbol at the given position"""
        if not self.initialized:
            return []
        
        params = {
            "textDocument": {"uri": self._uri_from_path(file_path)},
            "position": {"line": line - 1, "character": column},  # LSP is 0-indexed
            "context": {"includeDeclaration": True}
        }
        
        request_id = self._send_request("textDocument/references", params)
        response = self._wait_for_response(request_id)
        
        references = []
        if response and 'result' in response and response['result']:
            for ref in response['result']:
                references.append(LSPReference(
                    file_path=self._path_from_uri(ref['uri']),
                    line=ref['range']['start']['line'] + 1,
                    column=ref['range']['start']['character']
                ))
        
        return references
    
    def get_incoming_calls(self, file_path: str, line: int, column: int) -> list[CallHierarchyItem]:
        """Get functions that call the function at the given position"""
        if not self.initialized:
            return []
        
        # First, prepare call hierarchy item
        params = {
            "textDocument": {"uri": self._uri_from_path(file_path)},
            "position": {"line": line - 1, "character": column}
        }
        
        request_id = self._send_request("textDocument/prepareCallHierarchy", params)
        response = self._wait_for_response(request_id)
        
        if not response or 'result' not in response or not response['result']:
            return []
        
        call_item = response['result'][0]
        
        # Now get incoming calls
        request_id = self._send_request("callHierarchy/incomingCalls", {"item": call_item})
        response = self._wait_for_response(request_id)
        
        incoming = []
        if response and 'result' in response and response['result']:
            for call in response['result']:
                from_item = call['from']
                incoming.append(CallHierarchyItem(
                    name=from_item['name'],
                    kind=self._symbol_kind_to_string(from_item.get('kind', 12)),
                    file_path=self._path_from_uri(from_item['uri']),
                    line=from_item['range']['start']['line'] + 1,
                    column=from_item['range']['start']['character']
                ))
        
        return incoming
    
    def get_outgoing_calls(self, file_path: str, line: int, column: int) -> list[CallHierarchyItem]:
        """Get functions called by the function at the given position"""
        if not self.initialized:
            return []
        
        # First, prepare call hierarchy item
        params = {
            "textDocument": {"uri": self._uri_from_path(file_path)},
            "position": {"line": line - 1, "character": column}
        }
        
        request_id = self._send_request("textDocument/prepareCallHierarchy", params)
        response = self._wait_for_response(request_id)
        
        if not response or 'result' not in response or not response['result']:
            return []
        
        call_item = response['result'][0]
        
        # Now get outgoing calls
        request_id = self._send_request("callHierarchy/outgoingCalls", {"item": call_item})
        response = self._wait_for_response(request_id)
        
        outgoing = []
        if response and 'result' in response and response['result']:
            for call in response['result']:
                to_item = call['to']
                outgoing.append(CallHierarchyItem(
                    name=to_item['name'],
                    kind=self._symbol_kind_to_string(to_item.get('kind', 12)),
                    file_path=self._path_from_uri(to_item['uri']),
                    line=to_item['range']['start']['line'] + 1,
                    column=to_item['range']['start']['character']
                ))
        
        return outgoing
    
    def _symbol_kind_to_string(self, kind: int) -> str:
        """Convert LSP SymbolKind to string"""
        kinds = {
            1: "file", 2: "module", 3: "namespace", 4: "package",
            5: "class", 6: "method", 7: "property", 8: "field",
            9: "constructor", 10: "enum", 11: "interface", 12: "function",
            13: "variable", 14: "constant", 15: "string", 16: "number",
            17: "boolean", 18: "array", 19: "object", 20: "key",
            21: "null", 22: "enum_member", 23: "struct", 24: "event",
            25: "operator", 26: "type_parameter"
        }
        return kinds.get(kind, "unknown")
    
    def shutdown(self):
        """Shutdown the language server"""
        if self.process and self.process.poll() is None:
            self._send_request("shutdown")
            time.sleep(0.5)
            self._send_notification("exit")
            self.process.terminate()
            self.process = None


class TypeScriptLSPClient(LSPClient):
    """LSP Client for TypeScript Language Server"""
    
    def start(self) -> bool:
        """Start the TypeScript language server"""
        try:
            # Check if typescript-language-server is available
            result = subprocess.run(['which', 'typescript-language-server'], capture_output=True)
            if result.returncode != 0:
                # Try npx
                result = subprocess.run(['npx', '--yes', 'typescript-language-server', '--version'], capture_output=True)
                if result.returncode != 0:
                    return False
                    
                self.process = subprocess.Popen(
                    ['npx', '--yes', 'typescript-language-server', '--stdio'],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=str(self.workspace_path)
                )
            else:
                self.process = subprocess.Popen(
                    ['typescript-language-server', '--stdio'],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=str(self.workspace_path)
                )
            
            # Start reader thread
            self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self.reader_thread.start()
            
            # Initialize
            return self._initialize()
            
        except FileNotFoundError:
            return False
    
    def _initialize(self) -> bool:
        """Send initialize request"""
        params = {
            "processId": os.getpid(),
            "rootUri": self._uri_from_path(str(self.workspace_path)),
            "capabilities": {
                "textDocument": {
                    "references": {"dynamicRegistration": False},
                    "definition": {"dynamicRegistration": False},
                    "callHierarchy": {"dynamicRegistration": False},
                }
            },
            "workspaceFolders": [
                {"uri": self._uri_from_path(str(self.workspace_path)), "name": self.workspace_path.name}
            ]
        }
        
        request_id = self._send_request("initialize", params)
        response = self._wait_for_response(request_id, timeout=15.0)
        
        if response and 'result' in response:
            self.server_capabilities = response['result'].get('capabilities', {})
            self._send_notification("initialized")
            self.initialized = True
            return True
        
        return False
    
    def open_file(self, file_path: str, content: str):
        """Notify the server that a file is open"""
        ext = Path(file_path).suffix.lower()
        lang_id = "typescriptreact" if ext == ".tsx" else "typescript"
        
        self._send_notification("textDocument/didOpen", {
            "textDocument": {
                "uri": self._uri_from_path(file_path),
                "languageId": lang_id,
                "version": 1,
                "text": content
            }
        })
        # Small delay per file, batch wait happens after all files opened
        time.sleep(0.1)
    
    def find_references(self, file_path: str, line: int, column: int) -> list[LSPReference]:
        """Find all references to a symbol at the given position"""
        if not self.initialized:
            return []
        
        params = {
            "textDocument": {"uri": self._uri_from_path(file_path)},
            "position": {"line": line - 1, "character": column},
            "context": {"includeDeclaration": True}
        }
        
        request_id = self._send_request("textDocument/references", params)
        response = self._wait_for_response(request_id)
        
        references = []
        if response and 'result' in response and response['result']:
            for ref in response['result']:
                references.append(LSPReference(
                    file_path=self._path_from_uri(ref['uri']),
                    line=ref['range']['start']['line'] + 1,
                    column=ref['range']['start']['character']
                ))
        
        return references
    
    def get_incoming_calls(self, file_path: str, line: int, column: int) -> list[CallHierarchyItem]:
        """Get functions that call the function at the given position"""
        if not self.initialized:
            return []
        
        params = {
            "textDocument": {"uri": self._uri_from_path(file_path)},
            "position": {"line": line - 1, "character": column}
        }
        
        request_id = self._send_request("textDocument/prepareCallHierarchy", params)
        response = self._wait_for_response(request_id)
        
        if not response or 'result' not in response or not response['result']:
            return []
        
        call_item = response['result'][0]
        
        request_id = self._send_request("callHierarchy/incomingCalls", {"item": call_item})
        response = self._wait_for_response(request_id)
        
        incoming = []
        if response and 'result' in response and response['result']:
            for call in response['result']:
                from_item = call['from']
                incoming.append(CallHierarchyItem(
                    name=from_item['name'],
                    kind=self._symbol_kind_to_string(from_item.get('kind', 12)),
                    file_path=self._path_from_uri(from_item['uri']),
                    line=from_item['range']['start']['line'] + 1,
                    column=from_item['range']['start']['character']
                ))
        
        return incoming
    
    def get_outgoing_calls(self, file_path: str, line: int, column: int) -> list[CallHierarchyItem]:
        """Get functions called by the function at the given position"""
        if not self.initialized:
            return []
        
        params = {
            "textDocument": {"uri": self._uri_from_path(file_path)},
            "position": {"line": line - 1, "character": column}
        }
        
        request_id = self._send_request("textDocument/prepareCallHierarchy", params)
        response = self._wait_for_response(request_id)
        
        if not response or 'result' not in response or not response['result']:
            return []
        
        call_item = response['result'][0]
        
        request_id = self._send_request("callHierarchy/outgoingCalls", {"item": call_item})
        response = self._wait_for_response(request_id)
        
        outgoing = []
        if response and 'result' in response and response['result']:
            for call in response['result']:
                to_item = call['to']
                outgoing.append(CallHierarchyItem(
                    name=to_item['name'],
                    kind=self._symbol_kind_to_string(to_item.get('kind', 12)),
                    file_path=self._path_from_uri(to_item['uri']),
                    line=to_item['range']['start']['line'] + 1,
                    column=to_item['range']['start']['character']
                ))
        
        return outgoing
    
    def _symbol_kind_to_string(self, kind: int) -> str:
        """Convert LSP SymbolKind to string"""
        kinds = {
            1: "file", 2: "module", 3: "namespace", 4: "package",
            5: "class", 6: "method", 7: "property", 8: "field",
            9: "constructor", 10: "enum", 11: "interface", 12: "function",
            13: "variable", 14: "constant", 15: "string", 16: "number",
            17: "boolean", 18: "array", 19: "object", 20: "key",
            21: "null", 22: "enum_member", 23: "struct", 24: "event",
            25: "operator", 26: "type_parameter"
        }
        return kinds.get(kind, "unknown")
    
    def shutdown(self):
        """Shutdown the language server"""
        if self.process and self.process.poll() is None:
            self._send_request("shutdown")
            time.sleep(0.5)
            self._send_notification("exit")
            self.process.terminate()
            self.process = None


class LSPManager:
    """
    Manages multiple LSP clients for different languages.
    Provides a unified interface to get references and call hierarchies.
    """
    
    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
        self.clients: dict[str, LSPClient] = {}
        self.available_languages: list[str] = []
    
    def initialize(self, console=None) -> list[str]:
        """Initialize all available LSP clients"""
        available = []
        
        # Try C/C++ (clangd)
        if console:
            console.print("[dim]Checking for clangd (C/C++)...[/dim]")
        c_client = ClangdClient(self.workspace_path)
        if c_client.start():
            self.clients['c'] = c_client
            self.clients['h'] = c_client
            available.append('C/C++ (clangd)')
            if console:
                console.print("[green]  ✓ clangd available[/green]")
        else:
            if console:
                console.print("[yellow]  ✗ clangd not found (install with: apt install clangd)[/yellow]")
        
        # Try TypeScript
        if console:
            console.print("[dim]Checking for typescript-language-server...[/dim]")
        ts_client = TypeScriptLSPClient(self.workspace_path)
        if ts_client.start():
            self.clients['ts'] = ts_client
            self.clients['tsx'] = ts_client
            available.append('TypeScript (typescript-language-server)')
            if console:
                console.print("[green]  ✓ typescript-language-server available[/green]")
        else:
            if console:
                console.print("[yellow]  ✗ typescript-language-server not found (install with: npm i -g typescript-language-server)[/yellow]")
        
        self.available_languages = available
        return available
    
    def get_client_for_file(self, file_path: str) -> Optional[LSPClient]:
        """Get the appropriate LSP client for a file"""
        ext = Path(file_path).suffix.lower().lstrip('.')
        return self.clients.get(ext)
    
    def open_file(self, file_path: str, content: str):
        """Open a file in the appropriate LSP client"""
        client = self.get_client_for_file(file_path)
        if client:
            client.open_file(file_path, content)
    
    def get_symbol_references(self, file_path: str, line: int, column: int = 0) -> SymbolReferences:
        """Get all references for a symbol, plus call hierarchy"""
        client = self.get_client_for_file(file_path)
        
        result = SymbolReferences(
            symbol_name="",
            definition_file=file_path,
            definition_line=line
        )
        
        if not client:
            return result
        
        # Get references
        result.references = client.find_references(file_path, line, column)
        
        # Get call hierarchy (if function)
        result.incoming_calls = client.get_incoming_calls(file_path, line, column)
        result.outgoing_calls = client.get_outgoing_calls(file_path, line, column)
        
        return result
    
    def shutdown_all(self):
        """Shutdown all LSP clients"""
        for client in set(self.clients.values()):
            client.shutdown()
        self.clients.clear()
