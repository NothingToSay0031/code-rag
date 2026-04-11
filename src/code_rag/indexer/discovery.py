"""File discovery with filtering for code-rag indexer."""

from __future__ import annotations

import hashlib
from pathlib import Path

from pathspec import PathSpec

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

    Applies a chain of filters: hidden dirs, junk dirs, .gitignore,
    .coderagfilter, CLI include/exclude patterns, binary detection and a 2 MB
    size guard.  Symlinks are followed with cycle detection.
    """
    repo_path = repo_path.resolve()

    # --- build pathspec matchers ------------------------------------------------

    # .gitignore
    gitignore_path = repo_path / ".gitignore"
    if gitignore_path.is_file():
        lines = gitignore_path.read_text(encoding="utf-8").splitlines()
        gitignore_spec = PathSpec.from_lines("gitwildmatch", lines)
    else:
        gitignore_spec = None

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

    # --- walk -------------------------------------------------------------------

    visited: set[Path] = set()
    result: list[Path] = []

    for entry in repo_path.rglob("*"):
        # Only files
        if not entry.is_file():
            continue

        # Symlink cycle detection
        try:
            resolved = entry.resolve()
        except OSError:
            continue
        if resolved in visited:
            continue
        visited.add(resolved)

        # Relative path for pattern matching (use forward-slash string)
        rel = entry.relative_to(repo_path)
        parts = rel.parts

        # (a) Skip hidden directories
        if any(p.startswith(".") and p != "." for p in parts[:-1]):
            continue

        # (b) Skip junk directories
        if any(p in _JUNK_DIRS for p in parts[:-1]):
            continue

        # Use forward-slash relative string for pathspec matching
        rel_str = rel.as_posix()

        # CLI --include is the highest-priority override: if specified and matches,
        # the file is included regardless of .gitignore/.coderagfilter excludes.
        force_included = cli_include_spec and cli_include_spec.match_file(rel_str)

        if not force_included:
            # (c) .gitignore
            if gitignore_spec and gitignore_spec.match_file(rel_str):
                # Check .coderagfilter include (re-include overrides .gitignore)
                if not (
                    coderag_include_spec and coderag_include_spec.match_file(rel_str)
                ):
                    continue

            # (d) .coderagfilter excludes
            if coderag_exclude_spec and coderag_exclude_spec.match_file(rel_str):
                # Check .coderagfilter include (! rules override exclude rules)
                if not (
                    coderag_include_spec and coderag_include_spec.match_file(rel_str)
                ):
                    continue

        # (e) CLI --include: if specified, file must match
        if cli_include_spec and not cli_include_spec.match_file(rel_str):
            continue
        # CLI --exclude: always respected
        if cli_exclude_spec and cli_exclude_spec.match_file(rel_str):
            continue

        # Large file guard — code files are never skipped by size
        try:
            size = entry.stat().st_size
        except OSError:
            continue

        file_lang = detect_language(entry, config.custom_type_mappings)

        # Skip files with unrecognised extensions — they would be classified
        # as "unknown" in the pipeline and discarded anyway.  Skipping early
        # avoids unnecessary I/O (reading bytes for binary detection, hashing).
        if file_lang is None:
            continue

        is_code = file_lang in _CODE_LANGUAGES

        if size > _MAX_FILE_SIZE and not is_code:
            print(f"Warning: skipping large non-code file ({size} bytes): {rel}")
            continue

        # Binary detection (empty files pass through)
        if size > 0:
            try:
                chunk = entry.read_bytes()[:_BINARY_CHECK_SIZE]
                if b"\x00" in chunk:
                    continue
            except OSError:
                continue

        result.append(rel)

    result.sort()
    return result
