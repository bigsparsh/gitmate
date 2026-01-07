"""
Tree-sitter parsing functions for GitMate
Provides language-agnostic code entity extraction
"""

import tree_sitter_json as ts_json
import tree_sitter_typescript as ts_typescript
import tree_sitter_c as tsc
import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser
from pathlib import Path
from typing import Callable

from .models import CodeEntity
from .config import get_config


# Language registry
_c_language = Language(tsc.language())
_cpp_language = Language(tscpp.language())
_ts_language = Language(ts_typescript.language_typescript())
_tsx_language = Language(ts_typescript.language_tsx())


def create_parsers() -> dict[str, Parser]:
    """Create and return parsers for all supported languages"""
    return {
        '.c': Parser(_c_language),
        '.h': Parser(_c_language),
        '.cpp': Parser(_cpp_language),
        '.hpp': Parser(_cpp_language),
        '.cc': Parser(_cpp_language),
        '.cxx': Parser(_cpp_language),
        '.hxx': Parser(_cpp_language),
        '.ts': Parser(_ts_language),
        '.tsx': Parser(_tsx_language),
    }


def get_language_for_extension(ext: str) -> str:
    """Get the language identifier for a file extension"""
    lang_map = {
        '.c': 'c',
        '.h': 'c',
        '.cpp': 'cpp',
        '.hpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.hxx': 'cpp',
        '.ts': 'ts',
        '.tsx': 'tsx',
        '.json': 'json'
    }
    return lang_map.get(ext.lower(), 'text')


def get_language_name(ext: str) -> str:
    """Get human-readable language name for a file extension"""
    lang_names = {
        '.c': 'C',
        '.h': 'C',
        '.cpp': 'C++',
        '.hpp': 'C++',
        '.cc': 'C++',
        '.cxx': 'C++',
        '.hxx': 'C++',
        '.ts': 'TypeScript',
        '.tsx': 'TSX/React',
        '.json': 'JSON'
    }
    return lang_names.get(ext.lower(), 'code')


def extract_entities_from_file(
    file_path: Path,
    repo_path: Path | None = None,
    parsers: dict[str, Parser] | None = None
) -> list[CodeEntity]:
    """Parse a source file and extract all code entities"""
    if parsers is None:
        parsers = create_parsers()
    
    ext = file_path.suffix.lower()
    if ext not in parsers:
        return []
    
    try:
        with open(file_path, "rb") as f:
            source_code = f.read()
        
        parser = parsers[ext]
        tree = parser.parse(source_code)
        rel_path = str(file_path.relative_to(repo_path)) if repo_path else str(file_path)
        
        lang = get_language_for_extension(ext)
        return _extract_entities_from_node(tree.root_node, source_code, rel_path, lang)
    except Exception:
        return []


def _extract_entities_from_node(
    node, 
    source_code: bytes, 
    file_path: str, 
    lang: str = 'c'
) -> list[CodeEntity]:
    """Recursively extract code entities from AST nodes"""
    if lang in ('ts', 'tsx'):
        return _extract_ts_entities(node, source_code, file_path)
    elif lang == 'json':
        return _extract_json_entities(node, source_code, file_path)
    elif lang == 'cpp':
        return _extract_cpp_entities(node, source_code, file_path)
    else:
        return _extract_c_entities(node, source_code, file_path)


def _get_declarator_name_and_column(node) -> tuple[str | None, int]:
    """Extract the name and column from a declarator node"""
    if node is None:
        return None, 0

    if node.type == "identifier":
        name = node.text.decode("utf-8", errors="ignore") if node.text else None
        return name, node.start_point.column

    if node.type in ("function_declarator", "pointer_declarator", "array_declarator"):
        declarator = node.child_by_field_name("declarator")
        return _get_declarator_name_and_column(declarator)

    if node.type == "init_declarator":
        declarator = node.child_by_field_name("declarator")
        return _get_declarator_name_and_column(declarator)

    return None, 0


def _extract_c_entities(node, source_code: bytes, file_path: str) -> list[CodeEntity]:
    """Extract C language entities from AST nodes"""
    entities = []

    # Function definitions
    if node.type == "function_definition":
        declarator = node.child_by_field_name("declarator")
        name, name_col = _get_declarator_name_and_column(declarator)
        if name:
            entities.append(CodeEntity(
                name=name,
                entity_type="function",
                file_path=file_path,
                start_line=node.start_point.row + 1,
                end_line=node.end_point.row + 1,
                code=source_code[node.start_byte:node.end_byte].decode("utf-8", errors="ignore"),
                name_column=name_col
            ))

    # Struct/union definitions
    elif node.type in ("struct_specifier", "union_specifier"):
        name_node = node.child_by_field_name("name")
        if name_node:
            name = source_code[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="ignore")
            entities.append(CodeEntity(
                name=name,
                entity_type="struct" if node.type == "struct_specifier" else "union",
                file_path=file_path,
                start_line=node.start_point.row + 1,
                end_line=node.end_point.row + 1,
                code=source_code[node.start_byte:node.end_byte].decode("utf-8", errors="ignore"),
                name_column=name_node.start_point.column
            ))

    # Enum definitions
    elif node.type == "enum_specifier":
        name_node = node.child_by_field_name("name")
        if name_node:
            name = source_code[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="ignore")
            entities.append(CodeEntity(
                name=name,
                entity_type="enum",
                file_path=file_path,
                start_line=node.start_point.row + 1,
                end_line=node.end_point.row + 1,
                code=source_code[node.start_byte:node.end_byte].decode("utf-8", errors="ignore"),
                name_column=name_node.start_point.column
            ))

    # Global variable declarations
    elif node.type == "declaration" and node.parent and node.parent.type == "translation_unit":
        declarator = node.child_by_field_name("declarator")
        name, name_col = _get_declarator_name_and_column(declarator)
        if name:
            entities.append(CodeEntity(
                name=name,
                entity_type="global_variable",
                file_path=file_path,
                start_line=node.start_point.row + 1,
                end_line=node.end_point.row + 1,
                code=source_code[node.start_byte:node.end_byte].decode("utf-8", errors="ignore"),
                name_column=name_col
            ))

    # Typedef
    elif node.type == "type_definition":
        declarator = node.child_by_field_name("declarator")
        name, name_col = _get_declarator_name_and_column(declarator)
        if name:
            entities.append(CodeEntity(
                name=name,
                entity_type="typedef",
                file_path=file_path,
                start_line=node.start_point.row + 1,
                end_line=node.end_point.row + 1,
                code=source_code[node.start_byte:node.end_byte].decode("utf-8", errors="ignore"),
                name_column=name_col
            ))

    # Preprocessor defines
    elif node.type == "preproc_def":
        name_node = node.child_by_field_name("name")
        if name_node:
            name = source_code[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="ignore")
            entities.append(CodeEntity(
                name=name,
                entity_type="macro",
                file_path=file_path,
                start_line=node.start_point.row + 1,
                end_line=node.end_point.row + 1,
                code=source_code[node.start_byte:node.end_byte].decode("utf-8", errors="ignore"),
                name_column=name_node.start_point.column
            ))

    # Recurse into children
    for child in node.children:
        entities.extend(_extract_c_entities(child, source_code, file_path))

    return entities


def _extract_cpp_entities(node, source_code: bytes, file_path: str) -> list[CodeEntity]:
    """Extract C++ language entities from AST nodes"""
    entities = []

    # Function definitions
    if node.type == "function_definition":
        declarator = node.child_by_field_name("declarator")
        name, name_col = _get_declarator_name_and_column(declarator)
        if name:
            entities.append(CodeEntity(
                name=name,
                entity_type="function",
                file_path=file_path,
                start_line=node.start_point.row + 1,
                end_line=node.end_point.row + 1,
                code=source_code[node.start_byte:node.end_byte].decode("utf-8", errors="ignore"),
                name_column=name_col
            ))

    # Class definitions
    elif node.type == "class_specifier":
        name_node = node.child_by_field_name("name")
        if name_node:
            name = source_code[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="ignore")
            entities.append(CodeEntity(
                name=name,
                entity_type="class",
                file_path=file_path,
                start_line=node.start_point.row + 1,
                end_line=node.end_point.row + 1,
                code=source_code[node.start_byte:node.end_byte].decode("utf-8", errors="ignore"),
                name_column=name_node.start_point.column
            ))

    # Struct/union definitions
    elif node.type in ("struct_specifier", "union_specifier"):
        name_node = node.child_by_field_name("name")
        if name_node:
            name = source_code[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="ignore")
            entities.append(CodeEntity(
                name=name,
                entity_type="struct" if node.type == "struct_specifier" else "union",
                file_path=file_path,
                start_line=node.start_point.row + 1,
                end_line=node.end_point.row + 1,
                code=source_code[node.start_byte:node.end_byte].decode("utf-8", errors="ignore"),
                name_column=name_node.start_point.column
            ))

    # Enum definitions
    elif node.type == "enum_specifier":
        name_node = node.child_by_field_name("name")
        if name_node:
            name = source_code[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="ignore")
            entities.append(CodeEntity(
                name=name,
                entity_type="enum",
                file_path=file_path,
                start_line=node.start_point.row + 1,
                end_line=node.end_point.row + 1,
                code=source_code[node.start_byte:node.end_byte].decode("utf-8", errors="ignore"),
                name_column=name_node.start_point.column
            ))

    # Namespace definitions
    elif node.type == "namespace_definition":
        name_node = node.child_by_field_name("name")
        if name_node:
            name = source_code[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="ignore")
            entities.append(CodeEntity(
                name=name,
                entity_type="namespace",
                file_path=file_path,
                start_line=node.start_point.row + 1,
                end_line=node.end_point.row + 1,
                code=source_code[node.start_byte:node.end_byte].decode("utf-8", errors="ignore"),
                name_column=name_node.start_point.column
            ))

    # Template declarations
    elif node.type == "template_declaration":
        # Get the underlying declaration
        for child in node.children:
            if child.type in ("function_definition", "class_specifier", "struct_specifier"):
                child_entities = _extract_cpp_entities(child, source_code, file_path)
                for entity in child_entities:
                    entity.entity_type = f"template_{entity.entity_type}"
                entities.extend(child_entities)
                break

    # Type aliases (using declarations)
    elif node.type == "alias_declaration":
        name_node = node.child_by_field_name("name")
        if name_node:
            name = source_code[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="ignore")
            entities.append(CodeEntity(
                name=name,
                entity_type="type_alias",
                file_path=file_path,
                start_line=node.start_point.row + 1,
                end_line=node.end_point.row + 1,
                code=source_code[node.start_byte:node.end_byte].decode("utf-8", errors="ignore"),
                name_column=name_node.start_point.column
            ))

    # Typedef
    elif node.type == "type_definition":
        declarator = node.child_by_field_name("declarator")
        name, name_col = _get_declarator_name_and_column(declarator)
        if name:
            entities.append(CodeEntity(
                name=name,
                entity_type="typedef",
                file_path=file_path,
                start_line=node.start_point.row + 1,
                end_line=node.end_point.row + 1,
                code=source_code[node.start_byte:node.end_byte].decode("utf-8", errors="ignore"),
                name_column=name_col
            ))

    # Preprocessor defines
    elif node.type == "preproc_def":
        name_node = node.child_by_field_name("name")
        if name_node:
            name = source_code[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="ignore")
            entities.append(CodeEntity(
                name=name,
                entity_type="macro",
                file_path=file_path,
                start_line=node.start_point.row + 1,
                end_line=node.end_point.row + 1,
                code=source_code[node.start_byte:node.end_byte].decode("utf-8", errors="ignore"),
                name_column=name_node.start_point.column
            ))

    # Recurse into children
    for child in node.children:
        entities.extend(_extract_cpp_entities(child, source_code, file_path))

    return entities


def _extract_ts_entities(node, source_code: bytes, file_path: str) -> list[CodeEntity]:
    """Extract TypeScript/TSX entities from AST nodes"""
    entities = []

    # Function declarations
    if node.type == "function_declaration":
        name_node = node.child_by_field_name("name")
        if name_node:
            name = source_code[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="ignore")
            entities.append(CodeEntity(
                name=name,
                entity_type="function",
                file_path=file_path,
                start_line=node.start_point.row + 1,
                end_line=node.end_point.row + 1,
                code=source_code[node.start_byte:node.end_byte].decode("utf-8", errors="ignore"),
                name_column=name_node.start_point.column
            ))

    # Arrow functions with variable declaration
    elif node.type == "lexical_declaration" or node.type == "variable_declaration":
        for child in node.children:
            if child.type == "variable_declarator":
                name_node = child.child_by_field_name("name")
                value_node = child.child_by_field_name("value")
                if name_node and value_node and value_node.type == "arrow_function":
                    name = source_code[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="ignore")
                    entities.append(CodeEntity(
                        name=name,
                        entity_type="arrow_function",
                        file_path=file_path,
                        start_line=node.start_point.row + 1,
                        end_line=node.end_point.row + 1,
                        code=source_code[node.start_byte:node.end_byte].decode("utf-8", errors="ignore"),
                        name_column=name_node.start_point.column
                    ))

    # Class declarations
    elif node.type == "class_declaration":
        name_node = node.child_by_field_name("name")
        if name_node:
            name = source_code[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="ignore")
            entities.append(CodeEntity(
                name=name,
                entity_type="class",
                file_path=file_path,
                start_line=node.start_point.row + 1,
                end_line=node.end_point.row + 1,
                code=source_code[node.start_byte:node.end_byte].decode("utf-8", errors="ignore"),
                name_column=name_node.start_point.column
            ))

    # Interface declarations
    elif node.type == "interface_declaration":
        name_node = node.child_by_field_name("name")
        if name_node:
            name = source_code[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="ignore")
            entities.append(CodeEntity(
                name=name,
                entity_type="interface",
                file_path=file_path,
                start_line=node.start_point.row + 1,
                end_line=node.end_point.row + 1,
                code=source_code[node.start_byte:node.end_byte].decode("utf-8", errors="ignore"),
                name_column=name_node.start_point.column
            ))

    # Type alias declarations
    elif node.type == "type_alias_declaration":
        name_node = node.child_by_field_name("name")
        if name_node:
            name = source_code[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="ignore")
            entities.append(CodeEntity(
                name=name,
                entity_type="type_alias",
                file_path=file_path,
                start_line=node.start_point.row + 1,
                end_line=node.end_point.row + 1,
                code=source_code[node.start_byte:node.end_byte].decode("utf-8", errors="ignore"),
                name_column=name_node.start_point.column
            ))

    # Enum declarations
    elif node.type == "enum_declaration":
        name_node = node.child_by_field_name("name")
        if name_node:
            name = source_code[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="ignore")
            entities.append(CodeEntity(
                name=name,
                entity_type="enum",
                file_path=file_path,
                start_line=node.start_point.row + 1,
                end_line=node.end_point.row + 1,
                code=source_code[node.start_byte:node.end_byte].decode("utf-8", errors="ignore"),
                name_column=name_node.start_point.column
            ))

    # Export statements (named exports)
    elif node.type == "export_statement":
        declaration = node.child_by_field_name("declaration")
        if declaration:
            entities.extend(_extract_ts_entities(declaration, source_code, file_path))
            return entities  # Don't recurse again

    # Recurse into children
    for child in node.children:
        entities.extend(_extract_ts_entities(child, source_code, file_path))

    return entities


def _extract_json_entities(node, source_code: bytes, file_path: str) -> list[CodeEntity]:
    """Extract JSON entities from AST nodes"""
    entities = []

    # For JSON, we extract top-level keys as entities
    if node.type == "document":
        for child in node.children:
            if child.type == "object":
                entities.extend(_extract_json_object_keys(child, source_code, file_path, prefix=""))

    return entities


def _extract_json_object_keys(
    node, 
    source_code: bytes, 
    file_path: str, 
    prefix: str
) -> list[CodeEntity]:
    """Extract keys from JSON objects"""
    entities = []

    for child in node.children:
        if child.type == "pair":
            key_node = child.child_by_field_name("key")
            if key_node:
                key = source_code[key_node.start_byte:key_node.end_byte].decode("utf-8", errors="ignore").strip('"')
                full_key = f"{prefix}.{key}" if prefix else key

                entities.append(CodeEntity(
                    name=full_key,
                    entity_type="json_key",
                    file_path=file_path,
                    start_line=child.start_point.row + 1,
                    end_line=child.end_point.row + 1,
                    code=source_code[child.start_byte:child.end_byte].decode("utf-8", errors="ignore")
                ))

    return entities
