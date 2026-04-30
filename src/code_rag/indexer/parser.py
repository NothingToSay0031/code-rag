from __future__ import annotations

import ast
import importlib
import re
import warnings
from dataclasses import dataclass, field

from tree_sitter import Language, Parser, Tree

from code_rag.models import SymbolInfo


@dataclass
class LanguageConfig:
    module: str
    factory: str
    node_types: dict[str, list[str]]
    name_field: str
    doc_types: list[str] = field(default_factory=list)


LANGUAGE_CONFIGS: dict[str, LanguageConfig] = {
    "python": LanguageConfig(
        module="tree_sitter_python",
        factory="language",
        node_types={
            "function": ["function_definition"],
            "class": ["class_definition"],
        },
        name_field="name",
        doc_types=["expression_statement"],
    ),
    "javascript": LanguageConfig(
        module="tree_sitter_javascript",
        factory="language",
        node_types={
            "function": ["function_declaration"],
            "class": ["class_declaration"],
            "method": ["method_definition"],
        },
        name_field="name",
    ),
    "typescript": LanguageConfig(
        module="tree_sitter_typescript",
        factory="language_typescript",
        node_types={
            "function": ["function_declaration"],
            "class": ["class_declaration"],
            "method": ["method_definition"],
            "interface": ["interface_declaration"],
        },
        name_field="name",
    ),
    "tsx": LanguageConfig(
        module="tree_sitter_typescript",
        factory="language_tsx",
        node_types={
            "function": ["function_declaration"],
            "class": ["class_declaration"],
            "method": ["method_definition"],
            "interface": ["interface_declaration"],
        },
        name_field="name",
    ),
    "cpp": LanguageConfig(
        module="tree_sitter_cpp",
        factory="language",
        node_types={
            "function": ["function_definition"],
            "class": ["class_specifier", "struct_specifier"],
            "method": ["function_definition"],
        },
        name_field="declarator",
    ),
    "c": LanguageConfig(
        module="tree_sitter_c",
        factory="language",
        node_types={
            "function": ["function_definition"],
            "struct": ["struct_specifier"],
            "enum": ["enum_specifier"],
        },
        name_field="declarator",
    ),
    "java": LanguageConfig(
        module="tree_sitter_java",
        factory="language",
        node_types={
            "class": ["class_declaration"],
            "method": ["method_declaration"],
            "interface": ["interface_declaration"],
        },
        name_field="name",
    ),
    "csharp": LanguageConfig(
        module="tree_sitter_c_sharp",
        factory="language",
        node_types={
            "class": ["class_declaration"],
            "struct": ["struct_declaration"],
            "enum": ["enum_declaration"],
            "interface": ["interface_declaration"],
            "namespace": ["namespace_declaration"],
            "constructor": ["constructor_declaration"],
            "method": ["method_declaration"],
            "property": ["property_declaration"],
        },
        name_field="name",
    ),
    "lua": LanguageConfig(
        module="tree_sitter_lua",
        factory="language",
        node_types={
            "function": ["function_declaration", "function_definition_statement"],
        },
        name_field="name",
    ),
    "rust": LanguageConfig(
        module="tree_sitter_rust",
        factory="language",
        node_types={
            "function": ["function_item"],
            "struct": ["struct_item"],
            "impl": ["impl_item"],
            "trait": ["trait_item"],
            "enum": ["enum_item"],
        },
        name_field="name",
    ),
    "go": LanguageConfig(
        module="tree_sitter_go",
        factory="language",
        node_types={
            "function": ["function_declaration"],
            "method": ["method_declaration"],
            "type": ["type_declaration"],
        },
        name_field="name",
    ),
}


LANGUAGE_ALIASES: dict[str, str] = {
    "py": "python",
    "js": "javascript",
    "ts": "typescript",
    "cs": "csharp",
    "c#": "csharp",
    "cc": "cpp",
    "cxx": "cpp",
    "hpp": "cpp",
    "h": "c",
    # Shader languages → C/C++ grammar
    "glsl": "c",
    "hlsl": "c",
    "fx": "c",
    "vert": "c",
    "frag": "c",
    "comp": "c",
    "geom": "c",
    "tesc": "c",
    "tese": "c",
    "metal": "c",
    "usf": "cpp",
    "ush": "cpp",
}

CONTAINER_KINDS = {"class", "struct", "interface", "impl", "trait", "namespace", "type"}
IDENTIFIER_NODE_TYPES = {
    "identifier",
    "field_identifier",
    "property_identifier",
    "type_identifier",
    "namespace_identifier",
}

_parser_cache: dict[str, Parser] = {}


@dataclass
class ASTNode:
    node: object
    symbol: SymbolInfo | None
    doc_comment: str | None
    children: list[ASTNode]
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int


def _normalize_language(language: str) -> str:
    normalized = language.strip().lower().lstrip(".")
    resolved = LANGUAGE_ALIASES.get(normalized, normalized)
    if resolved not in LANGUAGE_CONFIGS:
        raise ValueError(f"Unsupported language: {language}")
    return resolved


def get_parser(language: str) -> Parser:
    resolved_language = _normalize_language(language)
    parser = _parser_cache.get(resolved_language)
    if parser is not None:
        return parser

    config = LANGUAGE_CONFIGS[resolved_language]
    module = importlib.import_module(config.module)
    language_factory = getattr(module, config.factory)
    parser = Parser()
    parser.language = Language(language_factory())
    _parser_cache[resolved_language] = parser
    return parser


def parse_file(source: bytes, language: str) -> Tree:
    return get_parser(language).parse(source)


def get_node_text(node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8")


def _iter_cursor_children(node):
    cursor = node.walk()
    if not cursor.goto_first_child():
        return
    while True:
        yield cursor.node
        if not cursor.goto_next_sibling():
            break


def _build_kind_lookup(config: LanguageConfig) -> dict[str, list[str]]:
    lookup: dict[str, list[str]] = {}
    for kind, node_types in config.node_types.items():
        for node_type in node_types:
            lookup.setdefault(node_type, []).append(kind)
    return lookup


def _resolve_symbol_kind(
    node, parent_ast: ASTNode | None, config: LanguageConfig
) -> str | None:
    kind_lookup = _build_kind_lookup(config)
    candidates = kind_lookup.get(node.type)
    if not candidates:
        return None

    container_parent = (
        parent_ast is not None
        and parent_ast.symbol is not None
        and parent_ast.symbol.kind in CONTAINER_KINDS
    )

    if container_parent and "method" in candidates:
        return "method"
    if container_parent and candidates == ["function"]:
        return "method"
    if len(candidates) == 1:
        return candidates[0]
    if "function" in candidates:
        return "function"
    return candidates[0]


def _find_leftmost_identifier(node, source: bytes) -> str | None:
    if node.type in IDENTIFIER_NODE_TYPES:
        return get_node_text(node, source)
    for child in _iter_cursor_children(node) or ():
        identifier = _find_leftmost_identifier(child, source)
        if identifier:
            return identifier
    return None


def _extract_name(node, source: bytes, config: LanguageConfig) -> str | None:
    name_node = node.child_by_field_name(config.name_field)
    if name_node is not None:
        if config.name_field == "declarator":
            return _find_leftmost_identifier(name_node, source)
        return get_node_text(name_node, source)

    for fallback_field in ("type", "name", "declarator"):
        fallback_node = node.child_by_field_name(fallback_field)
        if fallback_node is None:
            continue
        if fallback_field == "declarator":
            name = _find_leftmost_identifier(fallback_node, source)
        else:
            name = get_node_text(fallback_node, source)
        if name:
            return name

    for child in _iter_cursor_children(node) or ():
        if child.type == "type_spec":
            type_name = child.child_by_field_name("name")
            if type_name is not None:
                return get_node_text(type_name, source)
    return None


def _has_semantic_body(node) -> bool:
    for field_name in ("body", "declaration_list"):
        if node.child_by_field_name(field_name) is not None:
            return True
    return False


def _should_include_symbol(node, kind: str) -> bool:
    if kind in {"class", "struct", "enum", "interface", "trait", "namespace", "impl"}:
        return _has_semantic_body(node)
    return True


_INVALID_ESCAPE_RE = re.compile(r"\\(?![\\'\"abfnrtv0-7xNuU])")


def _normalize_python_docstring(text: str) -> str:
    stripped = text.strip()
    # Escape invalid backslash sequences (e.g. \*) so ast.literal_eval
    # can parse the string literal without SyntaxError / SyntaxWarning.
    sanitized = _INVALID_ESCAPE_RE.sub(r"\\\\", stripped)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SyntaxWarning)
            value = ast.literal_eval(sanitized)
    except (SyntaxError, ValueError):
        return stripped
    return value if isinstance(value, str) else stripped


def _comment_text(node, source: bytes, config: LanguageConfig) -> str | None:
    if node.type == "comment":
        return get_node_text(node, source).strip()
    if node.type in config.doc_types:
        first_named_child = next(
            (child for child in (_iter_cursor_children(node) or ()) if child.is_named),
            None,
        )
        if first_named_child is None or first_named_child.type != "string":
            return None
        text = get_node_text(node, source)
        return _normalize_python_docstring(text)
    return None


def _collect_previous_comments(
    node, source: bytes, config: LanguageConfig
) -> str | None:
    comments: list[str] = []
    sibling = node.prev_sibling
    while sibling is not None:
        text = _comment_text(sibling, source, config)
        if text is None:
            if sibling.is_named:
                break
            sibling = sibling.prev_sibling
            continue
        comments.append(text)
        sibling = sibling.prev_sibling
    if not comments:
        return None
    comments.reverse()
    return "\n".join(comments)


def _extract_python_inner_doc(
    node, source: bytes, config: LanguageConfig
) -> str | None:
    for field_name in ("body", "declaration_list"):
        body = node.child_by_field_name(field_name)
        if body is None:
            continue
        for child in _iter_cursor_children(body) or ():
            if not child.is_named:
                continue
            text = _comment_text(child, source, config)
            if text is not None:
                return text
            break
    return None


def _extract_doc_comment(node, source: bytes, config: LanguageConfig) -> str | None:
    previous = _collect_previous_comments(node, source, config)
    if previous is not None:
        return previous
    if config.doc_types:
        return _extract_python_inner_doc(node, source, config)
    return None


def get_ast_children(tree: Tree, source: bytes, language: str) -> list[ASTNode]:
    resolved_language = _normalize_language(language)
    config = LANGUAGE_CONFIGS[resolved_language]
    root = tree.root_node
    ast_roots: list[ASTNode] = []

    def visit(node, parent_ast: ASTNode | None) -> None:
        current_parent = parent_ast
        kind = _resolve_symbol_kind(node, parent_ast, config)
        if kind is not None and _should_include_symbol(node, kind):
            name = _extract_name(node, source, config)
            if name:
                symbol = SymbolInfo(
                    name=name,
                    kind=kind,
                    file_path="",
                    start_line=node.start_point.row + 1,
                    end_line=node.end_point.row + 1,
                    language=resolved_language,
                )
                current_parent = ASTNode(
                    node=node,
                    symbol=symbol,
                    doc_comment=_extract_doc_comment(node, source, config),
                    children=[],
                    start_byte=node.start_byte,
                    end_byte=node.end_byte,
                    start_line=node.start_point.row + 1,
                    end_line=node.end_point.row + 1,
                )
                if parent_ast is None:
                    ast_roots.append(current_parent)
                else:
                    parent_ast.children.append(current_parent)

        for child in _iter_cursor_children(node) or ():
            visit(child, current_parent)

    visit(root, None)
    return ast_roots


def extract_symbols(tree: Tree, source: bytes, language: str) -> list[SymbolInfo]:
    symbols: list[SymbolInfo] = []

    def collect(nodes: list[ASTNode]) -> None:
        for ast_node in nodes:
            if ast_node.symbol is not None:
                symbols.append(ast_node.symbol)
            collect(ast_node.children)

    collect(get_ast_children(tree, source, language))
    return symbols
