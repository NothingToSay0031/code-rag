"""File discovery with filtering for code-rag indexer."""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from queue import Queue

from pathspec import PathSpec
from tqdm import tqdm

from code_rag.config import CodeRagConfig, parse_coderagfilter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_JUNK_DIRS: frozenset[str] = frozenset(
    {
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "build",
        "dist",
        ".tox",
        ".code-rag",
    }
)

_MAX_FILE_SIZE: int = 2 * 1024 * 1024  # 2 MB
_BINARY_CHECK_SIZE: int = 8192

# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

_EXTENSION_MAP: dict[str, str] = {
    # Code
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".h": "cpp",
    ".hxx": "cpp",
    ".inl": "cpp",
    ".c": "c",
    ".java": "java",
    ".cs": "csharp",
    ".lua": "lua",
    ".rs": "rust",
    ".go": "go",
    # Docs
    ".md": "markdown",
    ".rst": "rst",
    ".txt": "text",
    # Config
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
}

_SPECIAL_FILES: dict[str, str] = {
    "Makefile": "makefile",
    "Dockerfile": "dockerfile",
}

_CODE_LANGUAGES: frozenset[str] = frozenset(
    {
        "python",
        "javascript",
        "typescript",
        "tsx",
        "cpp",
        "c",
        "java",
        "csharp",
        "lua",
        "rust",
        "go",
        # Shader languages (parsed with C/C++ grammar via LANGUAGE_ALIASES)
        "glsl",
        "hlsl",
        "fx",
        "vert",
        "frag",
        "comp",
        "geom",
        "tesc",
        "tese",
        "usf",
        "ush",
        "metal",
    }
)

_DOC_LANGUAGES: frozenset[str] = frozenset({"markdown", "rst", "text"})

_CONFIG_LANGUAGES: frozenset[str] = frozenset(
    {"json", "yaml", "toml", "makefile", "dockerfile"}
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_language(
    path: Path, custom_mappings: dict[str, str] | None = None
) -> str | None:
    """Detect language from file extension or special filename.

    *custom_mappings* map extensions to the tree-sitter grammar used for
    **parsing** (e.g. ``{".glsl": "c"}``).  However the returned language
    identity is the **extension name** (e.g. ``"glsl"``), not the grammar.
    The parser resolves the grammar via ``LANGUAGE_ALIASES`` at parse time.

    Returns ``None`` for unrecognised files.
    """
    name = path.name

    # Special filenames first
    if name in _SPECIAL_FILES:
        return _SPECIAL_FILES[name]

    suffix = path.suffix.lower()

    # Custom overrides: return extension name as language identity
    # (the parser maps it to the grammar via LANGUAGE_ALIASES)
    if custom_mappings and suffix in custom_mappings:
        return suffix.lstrip(".")

    return _EXTENSION_MAP.get(suffix)


def classify_file(path: Path, language: str | None) -> str:
    """Classify a file as ``code``, ``doc``, ``config``, or ``unknown``."""
    if language in _CODE_LANGUAGES:
        return "code"
    if language in _DOC_LANGUAGES:
        return "doc"
    if language in _CONFIG_LANGUAGES:
        return "config"
    return "unknown"


def get_file_fingerprint(path: Path) -> str:
    """Return the SHA-256 hex digest of *path*'s contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def discover_files(repo_path: Path, config: CodeRagConfig) -> list[Path]:
    """Walk *repo_path* and return a sorted list of relative ``Path`` objects.

    Uses a parallel producer-consumer pattern: worker threads pull
    directories from a shared queue, list them with ``os.scandir()``,
    push subdirectories back, and filter files in parallel.  Symlinks
    are followed with cycle detection.
    """
    repo_path = repo_path.resolve()

    # --- build pathspec matchers ------------------------------------------------

    # .coderagfilter
    coderagfilter_path = repo_path / ".coderagfilter"
    filter_cfg = parse_coderagfilter(coderagfilter_path)
    if filter_cfg.exclude_patterns:
        coderag_exclude_spec = PathSpec.from_lines(
            "gitwildmatch", filter_cfg.exclude_patterns
        )
    else:
        coderag_exclude_spec = None
    if filter_cfg.include_patterns:
        coderag_include_spec = PathSpec.from_lines(
            "gitwildmatch", filter_cfg.include_patterns
        )
    else:
        coderag_include_spec = None

    # CLI overrides
    if config.include_patterns:
        cli_include_spec = PathSpec.from_lines("gitwildmatch", config.include_patterns)
    else:
        cli_include_spec = None

    if config.exclude_patterns:
        cli_exclude_spec = PathSpec.from_lines("gitwildmatch", config.exclude_patterns)
    else:
        cli_exclude_spec = None

    # --- parallel tree walk -----------------------------------------------------

    max_workers = min(8, (os.cpu_count() or 4))
    dir_queue: Queue[Path | None] = Queue()
    dir_queue.put(repo_path)

    visited: set[Path] = set()
    visited_lock = threading.Lock()

    thread_results: list[list[Path]] = [[] for _ in range(max_workers)]

    pbar = tqdm(
        desc="Scanning",
        unit=" entries",
        mininterval=0.2,
        leave=False,
    )
    pbar_lock = threading.Lock()

    def filter_file(abs_path: Path, rel: Path, rel_str: str) -> bool:
        """Apply path-based and I/O-based filters to a single file entry."""
        # .coderagfilter: exclude first, then include
        if coderag_exclude_spec and coderag_exclude_spec.match_file(rel_str):
            return False
        if coderag_include_spec and not coderag_include_spec.match_file(rel_str):
            return False

        # CLI: exclude first, then include
        if cli_exclude_spec and cli_exclude_spec.match_file(rel_str):
            return False
        if cli_include_spec and not cli_include_spec.match_file(rel_str):
            return False

        # Large file guard — code files are never skipped by size
        try:
            size = abs_path.stat().st_size
        except OSError:
            return False

        file_lang = detect_language(abs_path, config.custom_type_mappings)

        if file_lang is None:
            return False

        is_code = file_lang in _CODE_LANGUAGES

        if size > _MAX_FILE_SIZE and not is_code:
            print(f"Warning: skipping large non-code file ({size} bytes): {rel}")
            return False

        # Binary detection (empty files pass through)
        if size > 0:
            try:
                chunk = abs_path.read_bytes()[:_BINARY_CHECK_SIZE]
                if b"\x00" in chunk:
                    return False
            except OSError:
                return False

        return True

    def worker(thread_id: int) -> None:
        local_results = thread_results[thread_id]
        while True:
            dir_path = dir_queue.get()
            if dir_path is None:  # shutdown sentinel
                dir_queue.task_done()
                break

            try:
                for entry in os.scandir(dir_path):
                    with pbar_lock:
                        pbar.update(1)

                    entry_path = Path(entry.path)

                    # Symlink cycle detection
                    try:
                        resolved = entry_path.resolve()
                    except OSError:
                        continue
                    with visited_lock:
                        if resolved in visited:
                            continue
                        visited.add(resolved)

                    if entry.is_dir(follow_symlinks=False):
                        rel = entry_path.relative_to(repo_path)
                        parts = rel.parts

                        # Skip hidden directories
                        if any(p.startswith(".") and p != "." for p in parts):
                            continue
                        # Skip junk directories
                        if any(p in _JUNK_DIRS for p in parts):
                            continue

                        dir_queue.put(entry_path)

                    elif entry.is_file(follow_symlinks=False):
                        rel = entry_path.relative_to(repo_path)
                        rel_str = rel.as_posix()

                        if filter_file(entry_path, rel, rel_str):
                            local_results.append(rel)
            except OSError:
                pass
            finally:
                dir_queue.task_done()

    threads = [
        threading.Thread(target=worker, args=(i,), daemon=True)
        for i in range(max_workers)
    ]
    for t in threads:
        t.start()

    # Wait until all directory listings are complete
    dir_queue.join()

    # Tell workers to exit
    for _ in range(max_workers):
        dir_queue.put(None)

    for t in threads:
        t.join()

    pbar.close()

    # Merge per-thread results
    result: list[Path] = []
    for tr in thread_results:
        result.extend(tr)
    result.sort()
    return result
