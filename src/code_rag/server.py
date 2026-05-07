from __future__ import annotations

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
    embedder: Embedder
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


def _build_index(repo_path: Path, prefix: str) -> _Index:
    data_dir = repo_path / ".code-rag"
    config = CodeRagConfig.load(repo_path)
    embedder = Embedder(config.resolve_model_name(), config.resolve_device())
    embedder._ensure_loaded()
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
        embedder=embedder,
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

    indices: list[_Index] = []

    if data_dir.exists():
        # Single-index: CWD itself is an indexed repo
        indices.append(_build_index(root, prefix=""))
        idx = indices[0]
        print(
            f"[CodeRAG] Single index: {root}  model={idx.config.resolve_model_name()}",
            file=sys.stderr,
        )
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
            indices.append(_build_index(repo_path, prefix=prefix))
        info = [f"{idx.prefix}({idx.config.resolve_model_name()})" for idx in indices]
        print(f"[CodeRAG] Multi-index ({len(indices)} repos): {info}", file=sys.stderr)

    _state.update(
        {
            "root": root,
            "indices": indices,
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
            f"Multiple indices available. Re-call with index_name set to one of: {names}."
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
def read_code(
    file_path: str,
    start_line: int,
    end_line: int,
    context_lines: int = 0,
    index_name: str | None = None,
) -> str:
    """Read a line range from a source file. Use to expand search results or view code beyond chunk boundaries.

    Args:
        file_path: Relative path as returned by search tools. In multi-index mode, use qualified path.
        start_line: First line (1-based, inclusive).
        end_line: Last line (1-based, inclusive).
        context_lines: Extra lines before/after the range (default 0).
        index_name: Multi-index only. See list_indices().
    """
    state = _get_state()
    indices = state["indices"]

    resolved = _resolve_file(indices, file_path)
    if resolved is None:
        return f"error: File not found: {file_path}"
    idx, local_path = resolved

    abs_path = idx.repo_path / local_path
    if not abs_path.exists():
        return f"error: File not found: {file_path}"

    try:
        all_lines = abs_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        try:
            all_lines = abs_path.read_text(encoding="latin-1").splitlines()
        except Exception as exc:
            return f"error: Could not read file: {exc}"
    except Exception as exc:
        return f"error: Could not read file: {exc}"

    total = len(all_lines)
    effective_start = max(1, start_line - context_lines)
    effective_end = min(total, end_line + context_lines)

    selected = all_lines[effective_start - 1 : effective_end]
    code = "\n".join(
        f"{effective_start + i}: {line}" for i, line in enumerate(selected)
    )

    header = f"{file_path}  L{effective_start}–{effective_end}  (total: {total} lines)"
    return f"{header}\n\n{code}"


@mcp.tool()
def list_indices() -> str:
    """List available code indices (repositories). Use returned names as index_name in other tools."""
    state = _get_state()
    indices = state["indices"]
    return "\n".join(
        f"{idx.prefix or '(root)'}  {idx.repo_path}  "
        f"model={idx.config.resolve_model_name()}  "
        f"({idx.metadata.count_files()} files)"
        for idx in indices
    )


def _render_refs(refs: list[dict]) -> str:
    """Render a reference list as indented plain text."""
    if not refs:
        return "  (no references found)"
    rendered: list[str] = []
    for r in refs:
        file_path = r.get("file_path", "(unknown)")
        start_line = r.get("start_line", "?")
        end_line = r.get("end_line", start_line)
        line_label = (
            f"L{start_line}–{end_line}" if start_line != end_line else f"L{start_line}"
        )
        rendered.append(f"  {file_path}  {line_label}  {str(r.get('text', '')).strip()}")
    return "\n".join(rendered)


def _render_results(
    output: list[dict],
    offset: int = 0,
    warning: str | None = None,
) -> str:
    """Format search results (code or doc) as a human-readable text block.

    Uses actual newlines and Unicode so the output is legible without JSON
    escape sequences (no ``\\n``, no ``\\uXXXX``).
    """
    lines: list[str] = []
    if warning:
        lines.append(f"⚠  {warning}")
    if not output:
        lines.append("(no results)")
        return "\n".join(lines)
    for i, entry in enumerate(output, 1):
        rank = offset + i
        lang = entry.get("language") or ""
        sym = entry.get("symbol_name") or ""
        meta = (
            f"[{rank}]  {entry['file_path']}"
            f"  L{entry['start_line']}–{entry['end_line']}"
        )
        if lang:
            meta += f"  {lang}"
        meta += f"  score={entry['score']}"
        if sym:
            meta += f"  symbol={sym}"
        lines.append(_SEP)
        lines.append(meta)
        lines.append("")
        lines.append(entry["snippet"])
    lines.append(_SEP)
    return "\n".join(lines)


def _build_snippet(
    r: "SearchResult",
    idx: "_Index",
    top_k: int,
    rank: int,
) -> str:
    """Build a display snippet for a single search result.

    Applies adaptive line limits (see :func:`_adaptive_snippet_limits`):
    - Code chunks: symbol-aware preview for large chunks.
    - Doc chunks:  heading-line prefix + truncation hint for large chunks.
    Both chunk types receive line-number prefixes.
    """
    snippet_full_limit, snippet_sym_lines = _adaptive_snippet_limits(top_k, rank)
    code_lines = (r.chunk.text or "").splitlines()

    # chunk.text = doc_comment_prefix + context_prefix + node_source_text.
    # The first header_lines lines are NOT real file lines at start_line; they
    # are prepended metadata.  Skip them so line-number labels are accurate.
    header_lines = r.chunk.metadata.get("header_lines", 0)
    content_lines = code_lines[
        header_lines:
    ]  # Actual file lines starting at start_line
    total_lines = len(content_lines)

    # Fast path: chunk fits within the adaptive limit — show everything.
    if total_lines <= snippet_full_limit:
        return "\n".join(
            f"{r.chunk.start_line + i}: {line}" for i, line in enumerate(content_lines)
        )

    if r.chunk.chunk_type == "code":
        # Large code chunk: show per-symbol previews when metadata is available.
        file_syms = idx.metadata.get_symbols(r.chunk.file_path)
        chunk_syms = sorted(
            [
                s
                for s in file_syms
                if r.chunk.start_line <= s.start_line <= r.chunk.end_line
            ],
            key=lambda s: s.start_line,
        )
        if not chunk_syms:
            # No symbol metadata: first 30 lines + hint.
            visible = content_lines[:30]
            snippet = "\n".join(
                f"{r.chunk.start_line + i}: {line}" for i, line in enumerate(visible)
            )
            snippet += (
                f"\n... ({total_lines - 30} more lines"
                f" — use read_code({r.chunk.start_line},"
                f" {r.chunk.end_line}) for full content)"
            )
            return snippet
        sym_parts = [
            f"({total_lines} lines — showing"
            f" {snippet_sym_lines} lines per symbol;"
            f" call read_code for full content)"
        ]
        for sym in chunk_syms:
            # header_lines shifts all content: code_lines[header_lines + k]
            # corresponds to file line (start_line + k).
            rel = sym.start_line - r.chunk.start_line + header_lines
            if 0 <= rel < len(code_lines):
                sym_len = min(
                    snippet_sym_lines,
                    sym.end_line - sym.start_line + 1,
                )
                sym_lines = code_lines[rel : rel + sym_len]
                sym_code = "\n".join(
                    f"  {sym.start_line + j}: {line}"
                    for j, line in enumerate(sym_lines)
                )
                span = (
                    f"L{sym.start_line}–{sym.end_line}"
                    if sym.start_line != sym.end_line
                    else f"L{sym.start_line}"
                )
                sym_parts.append(f"▸ {sym.kind} {sym.name} ({span})\n{sym_code}")
        return "\n\n".join(sym_parts)
    else:
        # Large doc chunk: first snippet_full_limit lines + truncation hint.
        visible = content_lines[:snippet_full_limit]
        snippet = "\n".join(
            f"{r.chunk.start_line + i}: {line}" for i, line in enumerate(visible)
        )
        snippet += (
            f"\n... ({total_lines - snippet_full_limit} more lines"
            f" — use read_code({r.chunk.start_line},"
            f" {r.chunk.end_line}) for full content)"
        )
        return snippet


@mcp.tool()
def search_code(
    query: str,
    top_k: int = 8,
    offset: int = 0,
    language: str | None = None,
    index_name: str | None = None,
    exclude_paths: list[str] | None = None,
) -> str:
    """Semantic + BM25 hybrid code search. Use for concept/behavior queries when the exact symbol name is unknown. If you already know the identifier, use get_symbol_info instead.

    Expensive (two retrieval passes). Read all results before calling again; use read_code to expand truncated snippets; use offset to paginate.

    Snippet size scales with top_k: ≤3→200 ln, ≤5→150, ≤10→100, >10→80-100. Oversized chunks show per-symbol previews (▸ kind name L#). Use read_code to expand any result.

    Args:
        query: Concept/behavior description, not an exact identifier.
        top_k: Results to return (default 8). Lower top_k → more lines per result. Max useful value is 15.
        offset: Skip N results for pagination (default 0).
        language: Language filter. Omit for cross-language queries (e.g. C++ bindings, FFI, codegen).
        index_name: Multi-index only. See list_indices().
        exclude_paths: Path substrings to exclude. Default excludes "Test". Pass [] to disable.
    """
    state = _get_state()
    indices = state["indices"]

    try:
        idx = _resolve_index(indices, index_name)
    except ValueError as exc:
        return f"error: {exc}"

    warning = _check_vector_availability(idx)
    results = idx.hybrid.search_code(
        query, top_k=top_k + offset, language=language, exclude_paths=exclude_paths
    )
    results = results[offset : offset + top_k]

    output = [
        {
            "file_path": _qualify_path(idx.prefix, r.chunk.file_path),
            "start_line": r.chunk.start_line,
            "end_line": r.chunk.end_line,
            "language": r.chunk.language,
            "symbol_name": r.chunk.symbol_name,
            "score": round(r.score, 4),
            "snippet": _build_snippet(r, idx, top_k, rank),
        }
        for rank, r in enumerate(results)
    ]

    return _render_results(output, offset=offset, warning=warning)


@mcp.tool()
def search_docs(
    query: str,
    top_k: int = 8,
    offset: int = 0,
    index_name: str | None = None,
    exclude_paths: list[str] | None = None,
) -> str:
    """Semantic + BM25 search over Markdown, RST, and text documentation files.

    Snippet size scales with top_k: ≤3→200 ln, ≤5→150, ≤10→100, >10→80-100. Oversized sections are truncated with a read_code hint. Use offset to paginate.

    Args:
        query: Search query.
        top_k: Number of results (default 8). Lower top_k → more lines per result.
        offset: Skip N results for pagination (default 0).
        index_name: Multi-index only. See list_indices().
        exclude_paths: Path substrings to exclude. Default excludes "Test". Pass [] to disable.
    """
    state = _get_state()
    indices = state["indices"]

    try:
        idx = _resolve_index(indices, index_name)
    except ValueError as exc:
        return f"error: {exc}"

    warning = _check_vector_availability(idx)
    results = idx.hybrid.search_docs(
        query, top_k=top_k + offset, exclude_paths=exclude_paths
    )
    results = results[offset : offset + top_k]

    output = [
        {
            "file_path": _qualify_path(idx.prefix, r.chunk.file_path),
            "start_line": r.chunk.start_line,
            "end_line": r.chunk.end_line,
            "score": round(r.score, 4),
            "snippet": _build_snippet(r, idx, top_k, rank),
        }
        for rank, r in enumerate(results)
    ]
    return _render_results(output, offset=offset, warning=warning)


@mcp.tool()
def get_file_symbols(file_path: str) -> str:
    """Get the symbol map of a file (functions, classes, methods with line ranges). Returns metadata only, no source code. Use read_code to view actual source.

    Args:
        file_path: Relative path. In multi-index mode, use qualified path from search results.
    """
    state = _get_state()
    indices = state["indices"]

    resolved = _resolve_file(indices, file_path)
    if resolved is None:
        return f"error: File not found: {file_path}"
    idx, local_path = resolved

    abs_path = idx.repo_path / local_path
    if not abs_path.exists():
        return f"error: File not found: {file_path}"

    try:
        content = abs_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = abs_path.read_text(encoding="latin-1")

    symbols = idx.metadata.get_symbols(local_path)
    file_info = idx.metadata.get_file_info(local_path)
    lang = file_info.language if file_info else ""
    size = file_info.size if file_info else len(content.encode())

    lines = [f"{file_path}  {lang}  {size} bytes", "", f"Symbols ({len(symbols)}):"]
    for s in symbols:
        lines.append(f"  {s.kind}  {s.name}  L{s.start_line}–{s.end_line}")
    if not symbols:
        lines.append("  (none)")
    return "\n".join(lines)


@mcp.tool()
def get_repo_structure(
    depth: int = 3,
    path: str | None = None,
    index_name: str | None = None,
) -> str:
    """Get directory tree from the indexed file list. For large repos, always specify path to avoid truncation.

    Args:
        depth: Max tree depth (default 3).
        path: Subdirectory to inspect (e.g. 'Engine/Sources/Runtime'). Omit for root.
        index_name: Multi-index only. See list_indices().
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
                    return f"error: {exc}"
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
                return f"error: Path not found: {path}. Call list_indices() to see available repos."

        if not target.exists():
            return f"error: Path not found: {path}"
        if not target.is_dir():
            return f"error: Not a directory: {path}"

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

# Snippet strategy for search_code results — adaptive to top_k:
#   top_k ≤  3                  → full_limit=200, per_symbol=15
#   top_k ≤  5                  → full_limit=150, per_symbol=10
#   top_k ≤ 10                  → full_limit=100, per_symbol= 5
#   top_k > 10, rank <  10      → full_limit=100, per_symbol= 5
#   top_k > 10, rank >= 10      → full_limit= 80, per_symbol= 5
# These constants are the base values (top_k ≤ 3); see _adaptive_snippet_limits().
# Use read_code(file_path, start_line, end_line) to read the full content.
_SNIPPET_FULL_LIMIT = 200
_SNIPPET_PER_SYMBOL_LINES = 15


def _adaptive_snippet_limits(top_k: int, rank: int) -> tuple[int, int]:
    """Return *(full_limit, per_symbol_lines)* based on query breadth and result rank.

    Larger ``top_k`` → smaller snippets per result so total context stays bounded.

    ============  =========  ================  ================
    top_k         rank       full_limit (ln)   per_symbol (ln)
    ============  =========  ================  ================
    ≤ 3           any        200               15
    ≤ 5           any        150               10
    ≤ 10          any        100                5
    > 10          < 10       100                5
    > 10          ≥ 10        80                5
    ============  =========  ================  ================
    """
    if top_k <= 3:
        return 200, 15
    if top_k <= 5:
        return 150, 10
    if top_k <= 10:
        return 100, 5
    # top_k > 10: lower-ranked results get a tighter limit
    if rank < 10:
        return 100, 5
    return 80, 5


# Separator line used by _render_code_results.
_SEP = "---"

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
            except ValueError, Exception:
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
    exact_match: bool = True,
    include_code: bool = False,
    include_references: bool = False,
    max_refs_per_symbol: int = 30,
    index_name: str | None = None,
) -> str:
    """Look up symbol definitions and references from the AST index. Use when you know the exact identifier name. For concept/behavior queries, use search_code instead.

    symbol_name must be the literal identifier as it appears in source code — not a description.

    Common patterns:
      get_symbol_info("LoadResource")                                           # definition only
      get_symbol_info("LoadResource", include_code=True)                       # + source
      get_symbol_info("LoadResource", include_references=True)                 # + all call sites
      get_symbol_info("LoadResource", include_code=True, include_references=True)  # full picture

    Args:
        symbol_name: Exact identifier as in source code (e.g. "LoadResource"). Case-insensitive.
        mode: "grouped" (default) merges adjacent symbols per file; "declaration" returns only class/struct/enum/interface; "all" returns every match individually.
        max_results: Max entries (default 20). In grouped mode, counts file groups.
        exact_match: True (default) for exact name match. False for substring matching.
        include_code: Include source code of each symbol (default False). Use read_code for on-demand fetching.
        include_references: Find all call sites/usages via ripgrep (default False). Replaces a second search_code call for the same name.
        max_refs_per_symbol: Max references per symbol (default 30). Only when include_references=True.
        index_name: Omit to search all indices. See list_indices().
    """
    state = _get_state()
    indices = state["indices"]

    # Allow searching all indices by default (symbol lookup is cheap)
    if index_name is not None:
        try:
            indices = [_resolve_index(indices, index_name)]
        except ValueError as exc:
            return f"error: {exc}"

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
        for s in idx.metadata.find_symbol(symbol_name, exact_only=exact_match):
            raw_symbols.append((idx, s))

    if mode == "declaration":
        result = _symbol_info_declaration(
            raw_symbols,
            max_results,
            include_code,
            include_references,
            max_refs_per_symbol,
        )
    elif mode == "grouped":
        result = _symbol_info_grouped(
            raw_symbols,
            max_results,
            include_code,
            include_references,
            max_refs_per_symbol,
        )
    else:
        # "all" — legacy per-symbol behaviour
        result = _symbol_info_all(
            raw_symbols,
            max_results,
            include_code,
            include_references,
            max_refs_per_symbol,
        )

    # Inject inheritance info from Browse.VC.db if available
    if browse_db_extras:
        extra_lines: list[str] = []
        if "base_classes" in browse_db_extras:
            extra_lines.append(
                f"Base classes: {', '.join(browse_db_extras['base_classes'])}"
            )
        if "derived_classes" in browse_db_extras:
            extra_lines.append(
                f"Derived classes: {', '.join(browse_db_extras['derived_classes'])}"
            )
        if extra_lines:
            # Insert after the first separator line
            nl = result.find("\n")
            if nl != -1:
                result = (
                    result[: nl + 1] + "\n".join(extra_lines) + "\n" + result[nl + 1 :]
                )
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
    include_code: bool,
    include_references: bool,
    max_refs_per_symbol: int,
) -> str:
    """Original per-symbol output (mode='all')."""
    lines: list[str] = []
    count = 0
    for idx, s in raw_symbols:
        if count >= max_results:
            break
        qpath = _qualify_path(idx.prefix, s.file_path)
        meta = f"{s.kind} {s.name}  {qpath}  L{s.start_line}–{s.end_line}  {s.language}"
        lines.append(_SEP)
        lines.append(meta)
        if include_code:
            code = _read_line_range(
                idx.repo_path, s.file_path, s.start_line, s.end_line
            )
            if code is not None:
                lines.append("")
                lines.append(code)
        if include_references:
            refs = _find_references(idx, s, max_refs=max_refs_per_symbol)
            lines.append(f"\nReferences ({len(refs)}):")
            lines.append(_render_refs(refs))
        count += 1
    lines.append(_SEP)
    return "\n".join(lines) if lines else "(no results)"


def _symbol_info_declaration(
    raw_symbols: list[tuple[_Index, SymbolInfo]],
    max_results: int,
    include_code: bool,
    include_references: bool,
    max_refs_per_symbol: int,
) -> str:
    """Only return top-level declarations (class/struct/enum/…)."""
    lines: list[str] = []
    count = 0
    for idx, s in raw_symbols:
        if count >= max_results:
            break
        if s.kind not in _DECLARATION_KINDS:
            continue
        qpath = _qualify_path(idx.prefix, s.file_path)
        meta = f"{s.kind} {s.name}  {qpath}  L{s.start_line}–{s.end_line}  {s.language}"
        lines.append(_SEP)
        lines.append(meta)
        if include_code:
            code = _read_line_range(
                idx.repo_path, s.file_path, s.start_line, s.end_line
            )
            if code is not None:
                lines.append("")
                lines.append(code)
        if include_references:
            refs = _find_references(idx, s, max_refs=max_refs_per_symbol)
            lines.append(f"\nReferences ({len(refs)}):")
            lines.append(_render_refs(refs))
        count += 1
    lines.append(_SEP)
    return "\n".join(lines) if lines else "(no results)"


def _symbol_info_grouped(
    raw_symbols: list[tuple[_Index, SymbolInfo]],
    max_results: int,
    include_code: bool,
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

    out_lines: list[str] = []
    result_count = 0
    for (_prefix, _fpath), members in groups.items():
        if result_count >= max_results:
            break
        idx = members[0][0]
        symbols = [s for _, s in members]
        qpath = _qualify_path(idx.prefix, _fpath)

        symbols.sort(key=lambda s: s.start_line)

        # Merge adjacent/overlapping ranges
        merged_ranges: list[tuple[int, int, list[SymbolInfo]]] = []
        for sym in symbols:
            if merged_ranges:
                prev_start, prev_end, prev_syms = merged_ranges[-1]
                if sym.start_line <= prev_end + _MERGE_GAP:
                    merged_ranges[-1] = (
                        prev_start,
                        max(prev_end, sym.end_line),
                        prev_syms + [sym],
                    )
                    continue
            merged_ranges.append((sym.start_line, sym.end_line, [sym]))

        for range_start, range_end, range_syms in merged_ranges:
            if result_count >= max_results:
                break

            effective_end = min(range_end, range_start + _MAX_GROUP_LINES - 1)
            truncated = range_end > effective_end

            sym_list = "  ".join(
                f"{s.kind} {s.name} (L{s.start_line}–{s.end_line})" for s in range_syms
            )
            meta = f"{qpath}  L{range_start}–{effective_end}  {symbols[0].language}"
            out_lines.append(_SEP)
            out_lines.append(meta)
            out_lines.append(f"Symbols: {sym_list}")

            code = _read_line_range(idx.repo_path, _fpath, range_start, effective_end)
            if include_code and code is not None:
                if truncated:
                    hidden_syms = [
                        s for s in range_syms if s.start_line > effective_end
                    ]
                    if hidden_syms:
                        sig_list = ", ".join(
                            f"{s.kind} {s.name} (L{s.start_line})" for s in hidden_syms
                        )
                        code += f"\n... (truncated, {range_end - effective_end} more lines — {sig_list})"
                    else:
                        code += (
                            f"\n... (truncated, {range_end - effective_end} more lines)"
                        )
                out_lines.append("")
                out_lines.append(code)

            if include_references:
                refs = _find_references(
                    idx, range_syms[0], max_refs=max_refs_per_symbol
                )
                out_lines.append(f"\nReferences ({len(refs)}):")
                out_lines.append(_render_refs(refs))

            result_count += 1

    out_lines.append(_SEP)
    return "\n".join(out_lines) if out_lines else "(no results)"


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
    # Keep only source-code references by default.
    code_exts = set(_EXT_LANGUAGE_MAP.keys())
    code_refs: list[dict] = []
    for ref in refs:
        file_path = ref.get("file_path")
        if not file_path:
            continue
        if Path(file_path).suffix.lower() in code_exts:
            code_refs.append(ref)

    # Deduplicate by file/line/text to reduce noisy repeats.
    deduped: list[dict] = []
    seen: set[tuple[str, int, str]] = set()
    for ref in code_refs:
        file_path = str(ref.get("file_path", ""))
        line_no = int(ref.get("start_line", 0))
        text = str(ref.get("text", "")).strip()
        key = (file_path, line_no, text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ref)

    # Prioritize references in the same language as the symbol.
    sym_lang = (symbol.language or "").lower()

    def _sort_key(ref: dict) -> tuple[int, str, int]:
        rel_path = str(ref.get("file_path", ""))
        ref_lang = _guess_language(rel_path).lower()
        same_lang_rank = 0 if sym_lang and ref_lang == sym_lang else 1
        line_no = int(ref.get("start_line", 0))
        return (same_lang_rank, rel_path, line_no)

    deduped.sort(key=_sort_key)
    limited = deduped[:max_refs]

    # Qualify paths with index prefix.
    for ref in limited:
        file_path = ref.get("file_path")
        if file_path:
            ref["file_path"] = _qualify_path(idx.prefix, file_path)
    return limited
