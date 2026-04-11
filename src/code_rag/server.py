from __future__ import annotations

import json
import importlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    FastMCP = importlib.import_module("fastmcp").FastMCP
except ImportError:

    class FastMCP:
        def __init__(self, name: str):
            self.name = name

        def tool(self):
            def decorator(func):
                return func

            return decorator


from code_rag.config import CodeRagConfig
from code_rag.indexer.embedder import Embedder
from code_rag.models import SearchResult, SymbolInfo
from code_rag.retriever.hybrid import HybridRetriever
from code_rag.retriever.keyword import KeywordRetriever
from code_rag.retriever.semantic import SemanticRetriever
from code_rag.storage.bm25_store import BM25Store
from code_rag.storage.browse_db import BrowseDBProvider, discover_browse_dbs
from code_rag.storage.metadata import MetadataStore
from code_rag.storage.ripgrep import find_references as _rg_find_references
from code_rag.storage.vector_store import VectorStore


mcp = FastMCP("CodeRAG")


# ---------------------------------------------------------------------------
# Per-index container
# ---------------------------------------------------------------------------


@dataclass
class _Index:
    """One indexed repository."""

    prefix: str  # relative path from root to repo (empty for single-index)
    repo_path: Path
    config: CodeRagConfig
    metadata: MetadataStore
    hybrid: HybridRetriever
    vector_store: VectorStore
    bm25_store: BM25Store
    browse_db: BrowseDBProvider | None = None


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_state: dict = {}


def _find_sub_indices(root: Path) -> list[Path]:
    """Find all ``.code-rag`` directories under *root* (non-recursive first level,
    then deeper).  Returns the repo path (parent of ``.code-rag``)."""
    found: list[Path] = []
    for entry in root.rglob(".code-rag"):
        if entry.is_dir():
            found.append(entry.parent.resolve())
    return sorted(found)


def _build_index(repo_path: Path, prefix: str, embedder: Embedder) -> _Index:
    data_dir = repo_path / ".code-rag"
    config = CodeRagConfig(repo_path=repo_path)
    vector_store = VectorStore(data_dir / "vectors.db", embedder.dimension)
    bm25_store = BM25Store(data_dir / "bm25.pkl")
    metadata = MetadataStore(data_dir / "metadata.json")
    semantic = SemanticRetriever(vector_store, embedder)
    keyword = KeywordRetriever(bm25_store)
    hybrid = HybridRetriever(semantic, keyword)

    # Try to discover MSVC IntelliSense database for accelerated symbol lookup
    browse_db: BrowseDBProvider | None = None
    browse_db_env = os.environ.get("CODE_RAG_BROWSE_DB")
    if browse_db_env:
        db_path = Path(browse_db_env)
        if db_path.is_file():
            browse_db = BrowseDBProvider(db_path)
    else:
        candidates = discover_browse_dbs(repo_path)
        if candidates:
            browse_db = BrowseDBProvider(candidates[0])

    return _Index(
        prefix=prefix,
        repo_path=repo_path,
        config=config,
        metadata=metadata,
        hybrid=hybrid,
        vector_store=vector_store,
        bm25_store=bm25_store,
        browse_db=browse_db,
    )


def _get_state() -> dict:
    if _state:
        return _state

    root = Path(os.environ.get("CODE_RAG_REPO", ".")).resolve()
    data_dir = root / ".code-rag"

    # Shared embedder (one model instance for all indices)
    config_for_model = CodeRagConfig(repo_path=root)
    embedder = Embedder(
        config_for_model.resolve_model_name(), config_for_model.resolve_device()
    )
    # Eagerly load the model so the first MCP request doesn't time out
    embedder._ensure_loaded()

    indices: list[_Index] = []

    if data_dir.exists():
        # Single-index: CWD itself is an indexed repo
        indices.append(_build_index(root, prefix="", embedder=embedder))
        print(f"[CodeRAG] Single index: {root}", file=sys.stderr)
    else:
        # Multi-index: scan subdirectories
        sub_repos = _find_sub_indices(root)
        if not sub_repos:
            raise RuntimeError(
                f"No indexed repositories found under {root}.\n"
                f"Run 'code-rag init <repo_path>' first."
            )
        for repo_path in sub_repos:
            try:
                prefix = repo_path.relative_to(root).as_posix()
            except ValueError:
                prefix = repo_path.name
            indices.append(_build_index(repo_path, prefix=prefix, embedder=embedder))
        names = [idx.prefix for idx in indices]
        print(f"[CodeRAG] Multi-index ({len(indices)} repos): {names}", file=sys.stderr)

    _state.update(
        {
            "root": root,
            "indices": indices,
            "embedder": embedder,
        }
    )
    return _state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _qualify_path(prefix: str, file_path: str) -> str:
    """Prepend the index prefix to a file path (no-op if prefix is empty)."""
    if not prefix:
        return file_path
    return f"{prefix}/{file_path}"


_EXT_LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "jsx",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c": "c",
    ".h": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".java": "java",
    ".cs": "csharp",
    ".rs": "rust",
    ".go": "go",
    ".lua": "lua",
}


def _guess_language(file_path: str) -> str:
    """Guess language from file extension (used for Browse.VC.db results)."""
    ext = Path(file_path).suffix.lower()
    return _EXT_LANGUAGE_MAP.get(ext, "")


def _resolve_file(
    indices: list[_Index], qualified_path: str
) -> tuple[_Index, str] | None:
    """Given a qualified path, find the owning index and the local file path."""
    for idx in indices:
        if not idx.prefix:
            # Single-index mode — path is already local
            return idx, qualified_path
        if qualified_path.startswith(idx.prefix + "/"):
            local = qualified_path[len(idx.prefix) + 1 :]
            return idx, local
    # Fallback: try each index directly (path might already be local)
    for idx in indices:
        abs_path = idx.repo_path / qualified_path
        if abs_path.exists():
            return idx, qualified_path
    return None


def _get_context_lines(
    repo_path: Path, file_path: str, start: int, end: int, context_lines: int = 3
) -> str | None:
    abs_path = repo_path / file_path
    if not abs_path.exists():
        return None
    try:
        lines = abs_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None
    ctx_start = max(0, start - 1 - context_lines)
    ctx_end = min(len(lines), end + context_lines)
    return "\n".join(lines[ctx_start:ctx_end])


# ---------------------------------------------------------------------------
# Index resolution helpers
# ---------------------------------------------------------------------------


def _check_vector_availability(idx: _Index) -> str | None:
    """Return a warning string if the index has a degraded vector store, else None."""
    if idx.vector_store is not None and not idx.vector_store.available:
        name = idx.prefix or "(root)"
        return (
            f"Vector search unavailable for: {name}. "
            "The ChromaDB database may be corrupt or from an incompatible version. "
            "Run 'code-rag index' to re-index."
        )
    return None


def _resolve_index(indices: list[_Index], index_name: str | None) -> _Index:
    """Find the target index by name.

    - Single-index mode: *index_name* is optional (auto-selects the only index).
    - Multi-index mode: *index_name* is required and must match an ``_Index.prefix``.
    """
    if len(indices) == 1:
        return indices[0]
    if index_name is None:
        names = [idx.prefix for idx in indices]
        raise ValueError(
            f"Multiple indices available ({names}). "
            "Call list_indices() first, then pass index_name."
        )
    for idx in indices:
        if idx.prefix == index_name:
            return idx
    available = [idx.prefix for idx in indices]
    raise ValueError(f"Index '{index_name}' not found. Available: {available}")


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def list_indices() -> str:
    """List available code indices (repositories).

    Returns:
        JSON array of index objects with name, path, and file count.
        Use the ``name`` value as ``index_name`` in other search tools.
    """
    state = _get_state()
    indices = state["indices"]
    return json.dumps(
        [
            {
                "name": idx.prefix or "(root)",
                "path": str(idx.repo_path),
                "files": idx.metadata.count_files(),
            }
            for idx in indices
        ],
        indent=2,
    )


@mcp.tool()
def search_code(
    query: str,
    top_k: int = 10,
    language: str | None = None,
    index_name: str | None = None,
    exclude_paths: list[str] | None = None,
) -> str:
    """Search code using semantic + keyword hybrid search (RRF).

    Args:
        query: Search query string.  For best results, use specific symbol names,
            class names, or distinctive identifiers rather than vague natural language.
            After an initial broad search, refine by extracting key class/function names
            from the results and searching for those directly.
        top_k: Number of results to return (default 10).
        language: Optional language filter (e.g. 'python', 'javascript').
            **Important**: When investigating cross-language mechanisms such as
            bindings, FFI, wrappers, or code generation, do NOT set this parameter.
            For example, Python bindings in a C++ engine are implemented in C++,
            so filtering by 'python' would miss all the relevant code.
            Only use this filter when you are certain the results should be in
            one specific language.
        index_name: Index to search. Required when multiple indices exist.
            Call list_indices() to see available names.
        exclude_paths: Path substrings to suppress from results (case-insensitive).
            Results whose ``file_path`` contains any of these strings are excluded.
            Default (``null``) suppresses files whose path contains ``"Test"``,
            preventing large test suites from drowning out production code.
            Pass ``[]`` to disable all path filtering.
            Examples: ``["Test", "ThirdParty"]``  or  ``["vendor", "Mock"]``.

    Returns:
        JSON array of search results with file_path, lines, score, and code snippet.
    """
    state = _get_state()
    indices = state["indices"]

    try:
        idx = _resolve_index(indices, index_name)
    except ValueError as exc:
        return json.dumps({"error": str(exc)}, indent=2)

    warning = _check_vector_availability(idx)
    results = idx.hybrid.search_code(
        query, top_k=top_k, language=language, exclude_paths=exclude_paths
    )
    output = []
    for r in results:
        qpath = _qualify_path(idx.prefix, r.chunk.file_path)
        # Truncate code to first 20 lines to avoid flooding the LLM context
        code_text = r.chunk.text
        code_lines = code_text.splitlines()
        if len(code_lines) > 20:
            code_text = (
                "\n".join(code_lines[:20])
                + f"\n... ({len(code_lines) - 20} more lines)"
            )
        entry = {
            "file_path": qpath,
            "start_line": r.chunk.start_line,
            "end_line": r.chunk.end_line,
            "language": r.chunk.language,
            "symbol_name": r.chunk.symbol_name,
            "score": round(r.score, 4),
            "code": code_text,
        }
        output.append(entry)

    if warning and not output:
        return json.dumps({"warning": warning, "results": []}, indent=2)
    resp: dict | list = output
    if warning:
        resp = {"warning": warning, "results": output}
    return json.dumps(resp, indent=2)


@mcp.tool()
def search_docs(
    query: str,
    top_k: int = 10,
    index_name: str | None = None,
    exclude_paths: list[str] | None = None,
) -> str:
    """Search project documentation (markdown, rst, text files).

    Args:
        query: Search query string.
        top_k: Number of results to return (default 10).
        index_name: Index to search. Required when multiple indices exist.
            Call list_indices() to see available names.
        exclude_paths: Path substrings to suppress from results (case-insensitive).
            Default (``null``) suppresses files whose path contains ``"Test"``.
            Pass ``[]`` to disable all path filtering.

    Returns:
        JSON array of search results with file_path, lines, score, and content.
    """
    state = _get_state()
    indices = state["indices"]

    try:
        idx = _resolve_index(indices, index_name)
    except ValueError as exc:
        return json.dumps({"error": str(exc)}, indent=2)

    warning = _check_vector_availability(idx)
    results = idx.hybrid.search_docs(query, top_k=top_k, exclude_paths=exclude_paths)
    output = [
        {
            "file_path": _qualify_path(idx.prefix, r.chunk.file_path),
            "start_line": r.chunk.start_line,
            "end_line": r.chunk.end_line,
            "score": round(r.score, 4),
            "content": r.chunk.text,
        }
        for r in results
    ]

    if warning and not output:
        return json.dumps({"warning": warning, "results": []}, indent=2)
    resp: dict | list = output
    if warning:
        resp = {"warning": warning, "results": output}
    return json.dumps(resp, indent=2)


@mcp.tool()
def get_file_context(file_path: str) -> str:
    """Get file content with symbol information.

    Args:
        file_path: Relative path to the file in the repository.
            In multi-index mode, use the qualified path returned by search
            (e.g. 'Engine/Sources/Runtime/Core/main.cpp').

    Returns:
        JSON object with file content, symbols, and metadata.
    """
    state = _get_state()
    indices = state["indices"]

    resolved = _resolve_file(indices, file_path)
    if resolved is None:
        return json.dumps({"error": f"File not found: {file_path}"})
    idx, local_path = resolved

    abs_path = idx.repo_path / local_path
    if not abs_path.exists():
        return json.dumps({"error": f"File not found: {file_path}"})

    try:
        content = abs_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = abs_path.read_text(encoding="latin-1")

    symbols = idx.metadata.get_symbols(local_path)
    file_info = idx.metadata.get_file_info(local_path)

    return json.dumps(
        {
            "file_path": file_path,
            "content": content,
            "language": file_info.language if file_info else None,
            "size": file_info.size if file_info else len(content),
            "symbols": [
                {
                    "name": s.name,
                    "kind": s.kind,
                    "start_line": s.start_line,
                    "end_line": s.end_line,
                }
                for s in symbols
            ],
        },
        indent=2,
    )


@mcp.tool()
def get_repo_structure(
    depth: int = 3,
    path: str | None = None,
    index_name: str | None = None,
) -> str:
    """Get repository directory tree.

    For large repositories, always specify ``path`` to avoid overwhelming output.
    Use search results to identify the relevant directory first, then inspect
    its structure with this tool.

    Args:
        depth: Maximum depth of directory tree (default 3).
        path: Subdirectory path to inspect (e.g. 'Engine/Sources/Runtime/Plugins/Python').
            When omitted, shows the repository root — but for large repos this will be
            truncated.  Prefer specifying a path.
        index_name: Index to inspect. Required when multiple indices exist.
            Call list_indices() to see available names.

    Returns:
        Formatted directory tree string.  Large directories are summarised
        to keep the output compact.
    """
    state = _get_state()
    indices = state["indices"]

    lines: list[str] = []

    if path is not None:
        # Path-specific mode: resolve into a single index
        if len(indices) == 1 and not indices[0].prefix:
            idx = indices[0]
            target = idx.repo_path / path
        else:
            # Multi-index: try to resolve via index_name or path prefix
            idx = None
            target = None
            if index_name is not None:
                try:
                    idx = _resolve_index(indices, index_name)
                except ValueError as exc:
                    return json.dumps({"error": str(exc)})
                target = idx.repo_path / path
            else:
                # Auto-detect: see if path starts with an index prefix
                for candidate in indices:
                    if candidate.prefix and path.startswith(candidate.prefix + "/"):
                        idx = candidate
                        local_path = path[len(candidate.prefix) + 1 :]
                        target = candidate.repo_path / local_path
                        break
                    elif candidate.prefix and path == candidate.prefix:
                        idx = candidate
                        target = candidate.repo_path
                        break
                if idx is None:
                    # Fallback: try each index directly
                    for candidate in indices:
                        t = candidate.repo_path / path
                        if t.exists():
                            idx = candidate
                            target = t
                            break
            if idx is None or target is None:
                return json.dumps(
                    {
                        "error": f"Path not found: {path}. Call list_indices() to see available repos."
                    }
                )

        if not target.exists():
            return json.dumps({"error": f"Path not found: {path}"})
        if not target.is_dir():
            return json.dumps({"error": f"Not a directory: {path}"})

        lines.append(f"{target.name}/  (subtree of {idx.prefix or idx.repo_path.name})")
        _build_tree(target, "", depth, lines, idx.metadata, idx.repo_path)
    elif len(indices) == 1 and not indices[0].prefix:
        # Single-index mode, no path specified
        idx = indices[0]
        lines.append(f"{idx.repo_path.name}/  ({idx.metadata.count_files()} files)")
        _build_tree(idx.repo_path, "", depth, lines, idx.metadata, idx.repo_path)
    else:
        # Multi-index mode, no path specified
        root = state["root"]
        lines.append(f"{root.name}/ (multi-index, {len(indices)} repos)")
        for idx in indices:
            lines.append(f"├── [{idx.prefix}]  ({idx.metadata.count_files()} files)")
            _build_tree(
                idx.repo_path, "│   ", depth, lines, idx.metadata, idx.repo_path
            )

    # Hard cap: if the tree is still too big, truncate
    if len(lines) > _MAX_TREE_LINES:
        lines = lines[:_MAX_TREE_LINES]
        lines.append(
            f"... (truncated at {_MAX_TREE_LINES} lines — specify a 'path' to see a subdirectory)"
        )
    return "\n".join(lines)


# Maximum lines in a directory tree output
_MAX_TREE_LINES = 200

# Maximum children to list per directory before summarising
_MAX_DIR_CHILDREN = 30


def _build_tree(
    path: Path,
    prefix: str,
    depth: int,
    lines: list,
    metadata: MetadataStore,
    repo_root: Path,
):
    if depth <= 0:
        return

    skip_dirs = {
        ".git",
        ".code-rag",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "build",
        "dist",
    }

    entries = sorted(path.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
    visible = [e for e in entries if not e.name.startswith(".") or e.is_file()]
    visible = [e for e in visible if not (e.is_dir() and e.name in skip_dirs)]

    # If too many children, only show directories + summary
    if len(visible) > _MAX_DIR_CHILDREN:
        dirs = [e for e in visible if e.is_dir()]
        files = [e for e in visible if not e.is_dir()]
        for i, entry in enumerate(dirs):
            is_last = i == len(dirs) - 1 and not files
            connector = "└── " if is_last else "├── "
            new_prefix = prefix + ("    " if is_last else "│   ")
            lines.append(f"{prefix}{connector}{entry.name}/")
            _build_tree(entry, new_prefix, depth - 1, lines, metadata, repo_root)
        if files:
            lines.append(f"{prefix}└── ({len(files)} files)")
        return

    for i, entry in enumerate(visible):
        is_last = i == len(visible) - 1
        connector = "└── " if is_last else "├── "
        new_prefix = prefix + ("    " if is_last else "│   ")

        if entry.is_dir():
            lines.append(f"{prefix}{connector}{entry.name}/")
            _build_tree(entry, new_prefix, depth - 1, lines, metadata, repo_root)
        else:
            try:
                rel_posix = entry.relative_to(repo_root).as_posix()
                info = metadata.get_file_info(rel_posix)
            except (ValueError, Exception):
                info = None
            desc = ""
            if info and info.language:
                desc = f"  ({info.language}, {info.chunk_count} chunks)"
            elif info:
                desc = f"  ({info.chunk_count} chunks)"
            lines.append(f"{prefix}{connector}{entry.name}{desc}")


@mcp.tool()
def get_symbol_info(
    symbol_name: str,
    mode: str = "grouped",
    max_results: int = 20,
    include_references: bool = False,
    max_refs_per_symbol: int = 30,
    index_name: str | None = None,
) -> str:
    """Find symbol definitions and references by name.

    Args:
        symbol_name: Symbol name to search for (case-insensitive substring match).
        mode: Controls how results are structured.

            - ``"grouped"`` (default): Group symbols by file and merge adjacent
              implementations into consolidated code blocks.  For example, querying
              a class name returns one entry for the header declaration and one
              merged entry for the .cpp implementation file instead of dozens of
              tiny individual method entries.
            - ``"declaration"``: Only return top-level declarations
              (class, struct, interface, enum).  Useful for getting an API overview
              without implementation details.
            - ``"all"``: Return every matching symbol individually (raw, ungrouped).

        max_results: Maximum number of result entries to return (default 20).
            In ``"grouped"`` mode this counts file groups, not individual symbols.
        include_references: Whether to search for references across the codebase.
            This is expensive on large repos and disabled by default.
        max_refs_per_symbol: Maximum references to collect per symbol (default 30).
            Only used when include_references is True.
        index_name: Index to search. When omitted, searches all indices.
            Call list_indices() to see available names.

    Returns:
        JSON array of matching symbols with definition location, file content,
        and optionally references.
    """
    state = _get_state()
    indices = state["indices"]

    # Allow searching all indices by default (symbol lookup is cheap)
    if index_name is not None:
        try:
            indices = [_resolve_index(indices, index_name)]
        except ValueError as exc:
            return json.dumps({"error": str(exc)}, indent=2)

    # Collect raw symbol matches across all target indices
    # Tier 0: Try Browse.VC.db (fastest, most accurate for C++ projects)
    # Tier 1: Fall back to MetadataStore (always available)
    raw_symbols: list[tuple[_Index, SymbolInfo]] = []
    browse_db_extras: dict | None = None  # inheritance info from Browse.VC.db

    for idx in indices:
        # ── Tier 0: Browse.VC.db ──
        if idx.browse_db and idx.browse_db.available:
            db_results = idx.browse_db.find_symbol(symbol_name)
            if db_results:
                for item in db_results:
                    # Convert absolute path to relative
                    abs_file = Path(item["file_path"])
                    try:
                        rel_path = abs_file.relative_to(idx.repo_path).as_posix()
                    except ValueError:
                        rel_path = item["file_path"]
                    raw_symbols.append(
                        (
                            idx,
                            SymbolInfo(
                                name=item["name"],
                                kind=item["kind"],
                                file_path=rel_path,
                                start_line=item["start_line"],
                                end_line=item["end_line"],
                                language=_guess_language(rel_path),
                            ),
                        )
                    )
                # Collect inheritance info for class/struct symbols
                class_items = [i for i in db_results if i["kind_id"] in (1, 2, 5)]
                if class_items:
                    bases = idx.browse_db.find_base_classes(symbol_name)
                    derived = idx.browse_db.find_derived_classes(symbol_name)
                    if bases or derived:
                        browse_db_extras = {}
                        if bases:
                            browse_db_extras["base_classes"] = bases
                        if derived:
                            browse_db_extras["derived_classes"] = [
                                d["name"] for d in derived
                            ]
                continue  # Skip MetadataStore if Browse.VC.db had results

        # ── Tier 1: MetadataStore ──
        for s in idx.metadata.find_symbol(symbol_name):
            raw_symbols.append((idx, s))

    if mode == "declaration":
        result = _symbol_info_declaration(
            raw_symbols, max_results, include_references, max_refs_per_symbol
        )
    elif mode == "grouped":
        result = _symbol_info_grouped(
            raw_symbols, max_results, include_references, max_refs_per_symbol
        )
    else:
        # "all" — legacy per-symbol behaviour
        result = _symbol_info_all(
            raw_symbols, max_results, include_references, max_refs_per_symbol
        )

    # Inject inheritance info from Browse.VC.db if available
    if browse_db_extras:
        parsed = json.loads(result)
        if isinstance(parsed, list) and parsed:
            parsed[0]["inheritance"] = browse_db_extras
        return json.dumps(parsed, indent=2)
    return result


_DECLARATION_KINDS: frozenset[str] = frozenset(
    {"class", "struct", "interface", "enum", "trait", "protocol", "type_alias"}
)

# When merging adjacent symbols in grouped mode, treat symbols whose start
# lines are within this many lines of the previous symbol's end as contiguous.
_MERGE_GAP = 5

# Maximum lines to include per merged code block (prevents dumping huge files).
_MAX_GROUP_LINES = 200


def _read_line_range(
    repo_path: Path, file_path: str, start: int, end: int
) -> str | None:
    """Read lines [start, end] (1-based inclusive) from a file."""
    abs_path = repo_path / file_path
    if not abs_path.exists():
        return None
    try:
        lines = abs_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None
    s = max(0, start - 1)
    e = min(len(lines), end)
    return "\n".join(lines[s:e])


def _symbol_info_all(
    raw_symbols: list[tuple[_Index, SymbolInfo]],
    max_results: int,
    include_references: bool,
    max_refs_per_symbol: int,
) -> str:
    """Original per-symbol output (mode='all')."""
    results = []
    for idx, s in raw_symbols:
        if len(results) >= max_results:
            break
        qpath = _qualify_path(idx.prefix, s.file_path)
        entry: dict = {
            "name": s.name,
            "kind": s.kind,
            "file_path": qpath,
            "start_line": s.start_line,
            "end_line": s.end_line,
            "language": s.language,
        }
        code = _read_line_range(idx.repo_path, s.file_path, s.start_line, s.end_line)
        if code is not None:
            entry["code"] = code
        if include_references:
            entry["references"] = _find_references(idx, s, max_refs=max_refs_per_symbol)
        results.append(entry)
    return json.dumps(results, indent=2)


def _symbol_info_declaration(
    raw_symbols: list[tuple[_Index, SymbolInfo]],
    max_results: int,
    include_references: bool,
    max_refs_per_symbol: int,
) -> str:
    """Only return top-level declarations (class/struct/enum/…)."""
    results = []
    for idx, s in raw_symbols:
        if len(results) >= max_results:
            break
        if s.kind not in _DECLARATION_KINDS:
            continue
        qpath = _qualify_path(idx.prefix, s.file_path)
        entry: dict = {
            "name": s.name,
            "kind": s.kind,
            "file_path": qpath,
            "start_line": s.start_line,
            "end_line": s.end_line,
            "language": s.language,
        }
        code = _read_line_range(idx.repo_path, s.file_path, s.start_line, s.end_line)
        if code is not None:
            entry["code"] = code
        if include_references:
            entry["references"] = _find_references(idx, s, max_refs=max_refs_per_symbol)
        results.append(entry)
    return json.dumps(results, indent=2)


def _symbol_info_grouped(
    raw_symbols: list[tuple[_Index, SymbolInfo]],
    max_results: int,
    include_references: bool,
    max_refs_per_symbol: int,
) -> str:
    """Group symbols by file and merge adjacent ranges."""
    from collections import OrderedDict

    # Group by (index prefix, file_path) preserving discovery order
    groups: OrderedDict[tuple[str, str], list[tuple[_Index, SymbolInfo]]] = (
        OrderedDict()
    )
    for idx, s in raw_symbols:
        key = (idx.prefix, s.file_path)
        groups.setdefault(key, []).append((idx, s))

    results = []
    for (_prefix, _fpath), members in groups.items():
        if len(results) >= max_results:
            break
        idx = members[0][0]  # all share the same index
        symbols = [s for _, s in members]
        qpath = _qualify_path(idx.prefix, _fpath)

        # Sort by start_line for merging
        symbols.sort(key=lambda s: s.start_line)

        # Merge adjacent/overlapping ranges
        merged_ranges: list[tuple[int, int, list[SymbolInfo]]] = []
        for sym in symbols:
            if merged_ranges:
                prev_start, prev_end, prev_syms = merged_ranges[-1]
                if sym.start_line <= prev_end + _MERGE_GAP:
                    # Extend existing range
                    merged_ranges[-1] = (
                        prev_start,
                        max(prev_end, sym.end_line),
                        prev_syms + [sym],
                    )
                    continue
            merged_ranges.append((sym.start_line, sym.end_line, [sym]))

        # Build one entry per merged range in this file
        for range_start, range_end, range_syms in merged_ranges:
            if len(results) >= max_results:
                break

            # Cap line count
            effective_end = min(range_end, range_start + _MAX_GROUP_LINES - 1)
            truncated = range_end > effective_end

            entry: dict = {
                "file_path": qpath,
                "language": symbols[0].language,
                "start_line": range_start,
                "end_line": effective_end,
                "symbols": [
                    {
                        "name": s.name,
                        "kind": s.kind,
                        "lines": f"{s.start_line}-{s.end_line}",
                    }
                    for s in range_syms
                ],
            }

            code = _read_line_range(idx.repo_path, _fpath, range_start, effective_end)
            if code is not None:
                if truncated:
                    code += f"\n... (truncated, {range_end - effective_end} more lines)"
                entry["code"] = code

            if include_references:
                # Collect references for the first (primary) symbol in the group
                entry["references"] = _find_references(
                    idx, range_syms[0], max_refs=max_refs_per_symbol
                )

            results.append(entry)

    return json.dumps(results, indent=2)


def _find_references(idx: _Index, symbol: SymbolInfo, max_refs: int = 30) -> list[dict]:
    """Find textual references to *symbol* across the index.

    Uses a tiered strategy:
    - **Tier 2**: ripgrep (``rg``) subprocess — ~100× faster than Python.
    - **Tier 3**: Pure-Python line-by-line scan — fallback when ``rg`` is
      unavailable.

    Both tiers skip files larger than 512 KB and stop after *max_refs*.
    """
    refs = _rg_find_references(
        idx.repo_path,
        symbol.name,
        max_refs=max_refs,
        exclude_file=symbol.file_path,
        exclude_lines=(symbol.start_line, symbol.end_line),
    )
    # Qualify paths with index prefix
    for ref in refs:
        ref["file_path"] = _qualify_path(idx.prefix, ref["file_path"])
    return refs
